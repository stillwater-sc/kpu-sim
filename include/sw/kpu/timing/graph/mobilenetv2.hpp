// ============================================================================
// include/sw/kpu/timing/graph/mobilenetv2.hpp
// Reusable MobileNetV2 builder: constructs the network (stem + inverted-residual
// bottleneck stack + 1x1 head conv + global-average-pool + FC) as a KernelGraph
// DFG, with matching per-node weights and a composed host oracle (M3-T4, #131).
// See docs/plans/m3_mobilenet_dfg.md and docs/milestones/M3_mobilenet.md.
//
// Executed through GraphCspExecutor on the CSP value path; every operator lowers
// through the M2/M3 bridge (pointwise conv im2col->GEMM, depthwise conv via the
// pooling-window unfold, folded BN, standalone ReLU6, residual add, GAP, FC) -
// no new runners. Dims are scaled for a fast CSP demo; batch N keeps every conv
// GEMM's M = N*Hout*Wout tile-aligned (the FC's M = N, so N >= tile).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/graph_csp_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing::graph {

/// MobileNetV2 topology (scaled for the CSP demo). All channels are tile-aligned
/// and the batch keeps every conv GEMM's M = batch*Hout*Wout a multiple of the
/// tile size (the FC's M = batch, so batch >= tile). Each bottleneck stage is
/// {expansion t, out_channels c, repeats n, first-block stride s}: the first
/// block of a stage strides by s and changes channels (no residual), the rest
/// stride 1 with identity residuals. The default is a fast, CI-friendly scale
/// exercising the full structure (stem, expansion, depthwise, downsampling,
/// identity residuals, head conv, GAP, FC).
struct MobileNetV2Spec {
    struct Stage { Size t, c, n, s; };

    Size batch = 16;
    Size in_channels = 16, height = 8, width = 8;   // stem input (16ch, 8x8)
    std::vector<Stage> stages = {
        {1, 16, 1, 1},   // t=1, 16ch, s1  identity residual   (hidden = 16)
        {2, 32, 1, 2},   // t=2, 32ch, s2  downsample 8->4, no residual (hidden = 32)
        {2, 32, 1, 1},   // t=2, 32ch, s1  identity residual   (hidden = 64)
    };
    Size head_channels = 16;   ///< final 1x1 conv width (scaled from 1280)
    Size num_classes = 16;
    Size tile = 16;   ///< GEMM tile; batch/channels/num_classes must be multiples
    float eps = 1e-3f;
    std::uint64_t seed = 2000;
};

/// Built network: the graph is populated into the caller's KernelGraph; this
/// carries the per-node weights, the synthetic input, the host-oracle output,
/// and the sink node id.
struct MobileNetV2 {
    std::unordered_map<std::size_t, NodeData> node_data;
    std::vector<float> input;
    std::vector<float> oracle;   // [batch, num_classes]
    std::size_t output_node = 0;
    std::size_t num_nodes = 0;
};

namespace detail {

inline std::vector<float> mb_synth(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<float> v(n);
    std::uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<std::uint32_t>(s >> 33);       // [0, 0x7FFFFFFF]
        x = (2.0f * static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

// groups defaults to 1 (standard conv); set groups = Cin for depthwise.
inline sw::kpu::Conv2DConfig mb_conv_cfg(Size N, Size Cin, Size Cout, Size H, Size W,
                                         Size K, Size stride, Size pad, Size groups = 1) {
    sw::kpu::Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = c.kernel_width = K;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = pad;
    c.groups = groups;
    return c;
}
inline schedule::Conv2DGeometry mb_geom(const sw::kpu::Conv2DConfig& c) {
    schedule::Conv2DGeometry g;
    g.N = static_cast<Size>(c.batch_size); g.C_in = static_cast<Size>(c.in_channels);
    g.H_in = static_cast<Size>(c.input_height); g.W_in = static_cast<Size>(c.input_width);
    g.C_out = static_cast<Size>(c.out_channels); g.Kh = static_cast<Size>(c.kernel_height);
    g.Kw = static_cast<Size>(c.kernel_width); g.stride_h = static_cast<Size>(c.stride_h);
    g.stride_w = static_cast<Size>(c.stride_w); g.pad_h = static_cast<Size>(c.padding_h);
    g.pad_w = static_cast<Size>(c.padding_w);
    return g;
}

struct MbBN { std::vector<float> gamma, beta, mean, var; float eps; };
inline MbBN mb_bn(Size C, std::uint64_t seed, float eps) {
    MbBN b; b.eps = eps;
    b.gamma = mb_synth(C, seed * 4 + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;
    b.beta  = mb_synth(C, seed * 4 + 2, 0.2f);
    b.mean  = mb_synth(C, seed * 4 + 3, 0.3f);
    b.var   = mb_synth(C, seed * 4 + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;
    return b;
}
inline void mb_set_bn(NodeData& nd, const MbBN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}
inline void relu6_ip(std::vector<float>& v) {
    for (auto& x : v) x = std::min(std::max(0.0f, x), 6.0f);
}

// Standard (pointwise) conv -> BN inference -> optional ReLU6. NCHW.
inline std::vector<float> mb_pw_conv_bn(const std::vector<float>& x, const std::vector<float>& w,
                                        const sw::kpu::Conv2DConfig& cc, const MbBN& bn, bool relu6) {
    const auto g = mb_geom(cc);
    auto z = schedule::conv2d_reference(x, w, {}, g, false);
    schedule::BatchNormGeometry bg; bg.N = g.N; bg.C = g.C_out; bg.H = g.H_out(); bg.W = g.W_out();
    auto y = schedule::batchnorm_reference(z, bn.gamma, bn.beta, bn.mean, bn.var, bn.eps, bg);
    if (relu6) relu6_ip(y);
    return y;
}

// Per-channel depthwise conv -> BN inference -> optional ReLU6. NCHW.
inline std::vector<float> mb_dw_conv_bn(const std::vector<float>& x, const std::vector<float>& filter,
                                        const sw::kpu::Conv2DConfig& cc, const MbBN& bn, bool relu6) {
    const auto g = mb_geom(cc);
    const Size Hout = g.H_out(), Wout = g.W_out();
    std::vector<float> y(static_cast<std::size_t>(g.N) * g.C_out * Hout * Wout);
    for (Size n = 0; n < g.N; ++n)
        for (Size c = 0; c < g.C_out; ++c) {
            const std::size_t plane = (static_cast<std::size_t>(n) * g.C_in + c) * g.H_in * g.W_in;
            const float scale = bn.gamma[c] / std::sqrt(bn.var[c] + bn.eps);
            const float shift = bn.beta[c] - bn.mean[c] * scale;
            for (Size ho = 0; ho < Hout; ++ho)
                for (Size wo = 0; wo < Wout; ++wo) {
                    float acc = 0.0f;
                    for (Size kh = 0; kh < g.Kh; ++kh) {
                        const long ih = static_cast<long>(ho * g.stride_h + kh) - static_cast<long>(g.pad_h);
                        if (ih < 0 || ih >= static_cast<long>(g.H_in)) continue;
                        for (Size kw = 0; kw < g.Kw; ++kw) {
                            const long iw = static_cast<long>(wo * g.stride_w + kw) - static_cast<long>(g.pad_w);
                            if (iw < 0 || iw >= static_cast<long>(g.W_in)) continue;
                            acc += x[plane + static_cast<std::size_t>(ih) * g.W_in + iw] *
                                   filter[(static_cast<std::size_t>(c) * g.Kh + kh) * g.Kw + kw];
                        }
                    }
                    acc = acc * scale + shift;
                    if (relu6) acc = std::min(std::max(0.0f, acc), 6.0f);
                    y[((static_cast<std::size_t>(n) * g.C_out + c) * Hout + ho) * Wout + wo] = acc;
                }
        }
    return y;
}

} // namespace detail

/**
 * @brief Build MobileNetV2 into `g`; return weights + input + oracle.
 *
 * Topology: stem (3x3 s1 conv -> BN -> ReLU6) + a stack of inverted-residual
 * bottlenecks (1x1 expand -> BN -> ReLU6 -> 3x3 depthwise -> BN -> ReLU6 -> 1x1
 * project -> BN, with an identity residual when stride==1 and Cin==Cout) + a 1x1
 * head conv -> BN -> ReLU6 + global-average-pool + FC. Execute the returned graph
 * through GraphCspExecutor and compare its output to `oracle` (tol ~5e-3).
 */
[[nodiscard]] inline MobileNetV2 build_mobilenetv2(KernelGraph& g, const MobileNetV2Spec& sp) {
    using namespace detail;
    using sw::kpu::Kernel;
    using sw::kpu::ElementwiseOp;
    using sw::kpu::ActivationType;

    // Tile-alignment: batch (FC/conv M axis), channels (K/N), num_classes (FC N),
    // and head_channels must be nonzero multiples of the tile. Expansion hidden =
    // t*Cin is a multiple whenever Cin is. Fail fast with a clear message.
    auto aligned = [&](Size v) { return v != 0 && v % sp.tile == 0; };
    if (sp.tile == 0 || !aligned(sp.batch))
        throw std::invalid_argument("build_mobilenetv2: batch must be a nonzero multiple of tile");
    if (!aligned(sp.in_channels) || !aligned(sp.head_channels) || !aligned(sp.num_classes))
        throw std::invalid_argument("build_mobilenetv2: in/head/class channels must be multiples of tile");
    if (sp.height == 0 || sp.width == 0 || sp.stages.empty())
        throw std::invalid_argument("build_mobilenetv2: invalid geometry");
    if (!std::isfinite(sp.eps) || sp.eps <= 0.0f)
        throw std::invalid_argument("build_mobilenetv2: eps must be finite and positive");
    for (const auto& st : sp.stages)
        if (st.t == 0 || !aligned(st.c) || st.n == 0 || (st.s != 1 && st.s != 2))
            throw std::invalid_argument("build_mobilenetv2: invalid stage");

    MobileNetV2 net;
    auto& nd = net.node_data;
    const Size N = sp.batch;
    std::uint64_t seed = sp.seed;

    net.input = mb_synth(static_cast<std::size_t>(N) * sp.in_channels * sp.height * sp.width, seed, 1.0f);
    std::vector<float> ox = net.input;
    Size cur_ch = sp.in_channels, cur_H = sp.height, cur_W = sp.width;

    auto out_dim = [](Size in, Size K, Size stride, Size pad) { return (in + 2 * pad - K) / stride + 1; };

    // ----- stem: 3x3 s1 conv -> BN -> ReLU6 -----------------------------------
    std::size_t in_node;
    {
        auto cc = mb_conv_cfg(N, sp.in_channels, sp.in_channels, cur_H, cur_W, 3, 1, 1);
        auto w  = mb_synth(static_cast<std::size_t>(sp.in_channels) * sp.in_channels * 9, seed, 0.1f);
        auto bn = mb_bn(sp.in_channels, seed, sp.eps);
        auto c = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "stem_conv");
        auto b = g.add_kernel(Kernel::create_batchnorm(N, sp.in_channels, cur_H, cur_W, sp.eps), "stem_bn");
        auto r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {sp.in_channels * cur_H * cur_W}), "stem_relu6");
        g.add_edge(c, b); g.add_edge(b, r);
        nd[c].filter = w; mb_set_bn(nd[b], bn);
        ox = mb_pw_conv_bn(ox, w, cc, bn, true);
        in_node = r;
        ++seed;
    }

    // ----- inverted-residual bottleneck stack ---------------------------------
    for (const auto& st : sp.stages) {
        for (Size blk = 0; blk < st.n; ++blk) {
            const Size stride = (blk == 0) ? st.s : 1;
            const Size in_ch = cur_ch;
            const Size out_ch = st.c;
            const Size hidden = st.t * in_ch;
            const bool residual = (stride == 1 && in_ch == out_ch);
            const Size Hd = out_dim(cur_H, 3, stride, 1), Wd = out_dim(cur_W, 3, stride, 1);

            auto cce = mb_conv_cfg(N, in_ch, hidden, cur_H, cur_W, 1, 1, 0);              // expand 1x1
            auto ccd = mb_conv_cfg(N, hidden, hidden, cur_H, cur_W, 3, stride, 1, hidden); // depthwise 3x3
            auto ccp = mb_conv_cfg(N, hidden, out_ch, Hd, Wd, 1, 1, 0);                    // project 1x1
            auto we = mb_synth(static_cast<std::size_t>(hidden) * in_ch, seed, 0.12f);
            auto wd = mb_synth(static_cast<std::size_t>(hidden) * 9, seed + 1, 0.25f);
            auto wp = mb_synth(static_cast<std::size_t>(out_ch) * hidden, seed + 2, 0.12f);
            auto bne = mb_bn(hidden, seed, sp.eps), bnd = mb_bn(hidden, seed + 1, sp.eps),
                 bnp = mb_bn(out_ch, seed + 2, sp.eps);

            auto ce = g.add_kernel(Kernel::create_conv2d(cce, false, ActivationType::NONE), "expand");
            auto be = g.add_kernel(Kernel::create_batchnorm(N, hidden, cur_H, cur_W, bne.eps), "bn_e");
            auto re = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * cur_H * cur_W}), "relu6_e");
            auto cd = g.add_kernel(Kernel::create_conv2d(ccd, false, ActivationType::NONE), "depthwise");
            auto bd = g.add_kernel(Kernel::create_batchnorm(N, hidden, Hd, Wd, bnd.eps), "bn_d");
            auto rd = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * Hd * Wd}), "relu6_d");
            auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "project");
            auto bp = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, bnp.eps), "bn_p");
            g.add_edge(in_node, ce); g.add_edge(ce, be); g.add_edge(be, re); g.add_edge(re, cd);
            g.add_edge(cd, bd); g.add_edge(bd, rd); g.add_edge(rd, cp); g.add_edge(cp, bp);
            nd[ce].filter = we; mb_set_bn(nd[be], bne);
            nd[cd].filter = wd; mb_set_bn(nd[bd], bnd);
            nd[cp].filter = wp; mb_set_bn(nd[bp], bnp);

            auto e = mb_pw_conv_bn(ox, we, cce, bne, true);
            auto d = mb_dw_conv_bn(e, wd, ccd, bnd, true);
            auto p = mb_pw_conv_bn(d, wp, ccp, bnp, false);   // linear bottleneck

            std::size_t sink = bp;
            if (residual) {
                auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {out_ch * Hd * Wd}), "add");
                // Identity skip is an EXPLICIT edge from the block's input node
                // (not the external-input fallback, which resolves to the whole-
                // network input - wrong for a block deep in the stack).
                g.add_edge(bp, ad);                  // main branch (project BN output)
                g.add_edge(in_node, ad, "C", "B");   // identity skip = block input
                for (std::size_t i = 0; i < p.size(); ++i) p[i] += ox[i];
                sink = ad;
            }
            ox = std::move(p);
            in_node = sink; cur_ch = out_ch; cur_H = Hd; cur_W = Wd;
            seed += 3;
        }
    }

    // ----- head: 1x1 conv -> BN -> ReLU6 --------------------------------------
    {
        auto cc = mb_conv_cfg(N, cur_ch, sp.head_channels, cur_H, cur_W, 1, 1, 0);
        auto w  = mb_synth(static_cast<std::size_t>(sp.head_channels) * cur_ch, seed, 0.12f);
        auto bn = mb_bn(sp.head_channels, seed, sp.eps);
        auto c = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "head_conv");
        auto b = g.add_kernel(Kernel::create_batchnorm(N, sp.head_channels, cur_H, cur_W, sp.eps), "head_bn");
        auto r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {sp.head_channels * cur_H * cur_W}), "head_relu6");
        g.add_edge(in_node, c); g.add_edge(c, b); g.add_edge(b, r);
        nd[c].filter = w; mb_set_bn(nd[b], bn);
        ox = mb_pw_conv_bn(ox, w, cc, bn, true);
        in_node = r; cur_ch = sp.head_channels;
        ++seed;
    }

    // ----- head: global-average-pool -> FC ------------------------------------
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, cur_ch, cur_H, cur_W), "gap");
    auto fc = g.add_kernel(Kernel::create_matmul(N, sp.num_classes, cur_ch), "fc");
    g.add_edge(in_node, gp); g.add_edge(gp, fc);

    auto wfc = mb_synth(static_cast<std::size_t>(cur_ch) * sp.num_classes, seed, 0.1f);
    auto bfc = mb_synth(sp.num_classes, seed + 1, 0.2f);
    nd[fc].fc_weight = wfc; nd[fc].fc_bias = bfc;
    nd[fc].fc_M = N; nd[fc].fc_K = cur_ch; nd[fc].fc_N = sp.num_classes;

    // oracle head: GAP over the plane per (n,c), then FC.
    schedule::Pool2DGeometry pg; pg.N = N; pg.C = cur_ch; pg.H = cur_H; pg.W = cur_W;
    auto gap = schedule::global_avg_pool_reference(ox, pg);   // [N*cur_ch]
    net.oracle.assign(static_cast<std::size_t>(N) * sp.num_classes, 0.0f);
    for (Size m = 0; m < N; ++m)
        for (Size n = 0; n < sp.num_classes; ++n) {
            float acc = bfc[n];
            for (Size k = 0; k < cur_ch; ++k)
                acc += gap[m * cur_ch + k] * wfc[k * sp.num_classes + n];
            net.oracle[m * sp.num_classes + n] = acc;
        }

    net.output_node = fc;
    net.num_nodes = g.num_nodes();
    return net;
}

} // namespace sw::kpu::timing::graph
