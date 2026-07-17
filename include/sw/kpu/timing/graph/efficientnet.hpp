// ============================================================================
// include/sw/kpu/timing/graph/efficientnet.hpp
// Reusable EfficientNet-B0 builder: constructs the network (stem + MBConv+SE
// bottleneck stack + 1x1 head conv + global-average-pool + FC) as a KernelGraph
// DFG, with matching per-node weights and a composed host oracle (M3, #131).
// See docs/plans/m3_mobilenet_dfg.md and docs/milestones/M3_efficientnet.md.
//
// Executed through GraphCspExecutor on the CSP value path. Beyond MobileNetV2 the
// MBConv block adds a squeeze-and-excitation gate between the depthwise and the
// project: GAP -> FC_reduce -> ReLU -> FC_expand -> sigmoid -> channel-broadcast
// multiply. Depthwise convs use per-stage kernel sizes (3 or 5). Activations are
// SiLU/swish (x*sigmoid(x)) as in the real model. Dims are scaled for a fast CSP
// demo; batch N keeps every conv GEMM's M = N*Hout*Wout tile-aligned.
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

/// EfficientNet-B0 topology (scaled for the CSP demo). All channels and the SE
/// reduce dim are tile-aligned; the batch keeps every conv GEMM's
/// M = batch*Hout*Wout a multiple of the tile (the FC's M = batch, so batch >=
/// tile). Each MBConv stage is {expansion t, out_channels c, repeats n, first-
/// block stride s, depthwise kernel k, SE reduce dim se}: the first block of a
/// stage strides by s and changes channels (no residual), the rest stride 1 with
/// identity residuals.
struct EfficientNetB0Spec {
    struct Stage { Size t, c, n, s, k, se; };

    Size batch = 16;
    Size in_channels = 16, height = 8, width = 8;   // stem input
    Size stem_channels = 16;                        // stem 3x3 conv output width
    std::vector<Stage> stages = {
        {1, 16, 1, 1, 3, 16},   // MBConv1, k3, no expansion, identity residual
        {2, 32, 1, 2, 3, 16},   // MBConv2, k3, s2 downsample 8->4
        {2, 32, 1, 1, 5, 16},   // MBConv2, k5, identity residual
    };
    Size head_channels = 16;   ///< final 1x1 conv width (scaled from 1280)
    Size num_classes = 16;
    Size tile = 16;   ///< GEMM tile; batch/channels/num_classes/se must be multiples
    float eps = 1e-3f;
    std::uint64_t seed = 3000;
};

struct EfficientNetB0 {
    std::unordered_map<std::size_t, NodeData> node_data;
    std::vector<float> input;
    std::vector<float> oracle;   // [batch, num_classes]
    std::size_t output_node = 0;
    std::size_t num_nodes = 0;
};

namespace detail {

inline std::vector<float> ef_synth(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<float> v(n);
    std::uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<std::uint32_t>(s >> 33);
        x = (2.0f * static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

inline sw::kpu::Conv2DConfig ef_conv_cfg(Size N, Size Cin, Size Cout, Size H, Size W,
                                         Size K, Size stride, Size pad, Size groups = 1) {
    sw::kpu::Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = c.kernel_width = K;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = pad;
    c.groups = groups;
    return c;
}
inline schedule::Conv2DGeometry ef_geom(const sw::kpu::Conv2DConfig& c) {
    schedule::Conv2DGeometry g;
    g.N = static_cast<Size>(c.batch_size); g.C_in = static_cast<Size>(c.in_channels);
    g.H_in = static_cast<Size>(c.input_height); g.W_in = static_cast<Size>(c.input_width);
    g.C_out = static_cast<Size>(c.out_channels); g.Kh = static_cast<Size>(c.kernel_height);
    g.Kw = static_cast<Size>(c.kernel_width); g.stride_h = static_cast<Size>(c.stride_h);
    g.stride_w = static_cast<Size>(c.stride_w); g.pad_h = static_cast<Size>(c.padding_h);
    g.pad_w = static_cast<Size>(c.padding_w);
    return g;
}

struct EfBN { std::vector<float> gamma, beta, mean, var; float eps; };
inline EfBN ef_bn(Size C, std::uint64_t seed, float eps) {
    EfBN b; b.eps = eps;
    b.gamma = ef_synth(C, seed * 4 + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;
    b.beta  = ef_synth(C, seed * 4 + 2, 0.2f);
    b.mean  = ef_synth(C, seed * 4 + 3, 0.3f);
    b.var   = ef_synth(C, seed * 4 + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;
    return b;
}
inline void ef_set_bn(NodeData& nd, const EfBN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}
inline float ef_silu(float x) { return x / (1.0f + std::exp(-x)); }   // x * sigmoid(x)
inline void ef_silu_ip(std::vector<float>& v) { for (auto& x : v) x = ef_silu(x); }

// Standard (pointwise) conv -> BN inference -> optional SiLU. NCHW.
inline std::vector<float> ef_pw_conv_bn(const std::vector<float>& x, const std::vector<float>& w,
                                        const sw::kpu::Conv2DConfig& cc, const EfBN& bn, bool silu) {
    const auto g = ef_geom(cc);
    auto z = schedule::conv2d_reference(x, w, {}, g, false);
    schedule::BatchNormGeometry bg; bg.N = g.N; bg.C = g.C_out; bg.H = g.H_out(); bg.W = g.W_out();
    auto y = schedule::batchnorm_reference(z, bn.gamma, bn.beta, bn.mean, bn.var, bn.eps, bg);
    if (silu) ef_silu_ip(y);
    return y;
}

// Per-channel depthwise conv (kxk) -> BN inference -> optional SiLU. NCHW.
inline std::vector<float> ef_dw_conv_bn(const std::vector<float>& x, const std::vector<float>& filter,
                                        const sw::kpu::Conv2DConfig& cc, const EfBN& bn, bool silu) {
    const auto g = ef_geom(cc);
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
                    if (silu) acc = ef_silu(acc);
                    y[((static_cast<std::size_t>(n) * g.C_out + c) * Hout + ho) * Wout + wo] = acc;
                }
        }
    return y;
}

// Host matmul C[M,N] = A[M,K] @ W[K,N] + bias, optional ReLU.
inline std::vector<float> ef_matmul(const std::vector<float>& a, const std::vector<float>& w,
                                    const std::vector<float>& bias, Size M, Size Kd, Size Nd, bool relu) {
    std::vector<float> c(static_cast<std::size_t>(M) * Nd);
    for (Size m = 0; m < M; ++m)
        for (Size n = 0; n < Nd; ++n) {
            float acc = bias.empty() ? 0.0f : bias[n];
            for (Size k = 0; k < Kd; ++k) acc += a[m * Kd + k] * w[k * Nd + n];
            c[m * Nd + n] = relu ? std::max(0.0f, acc) : acc;
        }
    return c;
}

} // namespace detail

/**
 * @brief Build EfficientNet-B0 into `g`; return weights + input + oracle.
 *
 * Topology: stem (3x3 s1 conv -> BN -> SiLU) + a stack of MBConv+SE bottlenecks
 * (1x1 expand -> BN -> SiLU -> kxk depthwise -> BN -> SiLU -> SE gate -> 1x1
 * project -> BN, with an identity residual when stride==1 and Cin==Cout; the
 * t==1 stage omits the expansion) + a 1x1 head conv -> BN -> SiLU + global-
 * average-pool + FC. Execute the returned graph through GraphCspExecutor and
 * compare its output to `oracle` (tol ~5e-3).
 */
[[nodiscard]] inline EfficientNetB0 build_efficientnet_b0(KernelGraph& g, const EfficientNetB0Spec& sp) {
    using namespace detail;
    using sw::kpu::Kernel;
    using sw::kpu::ElementwiseOp;
    using sw::kpu::ActivationType;

    auto aligned = [&](Size v) { return v != 0 && v % sp.tile == 0; };
    if (sp.tile == 0 || !aligned(sp.batch))
        throw std::invalid_argument("build_efficientnet_b0: batch must be a nonzero multiple of tile");
    if (!aligned(sp.in_channels) || !aligned(sp.stem_channels) ||
        !aligned(sp.head_channels) || !aligned(sp.num_classes))
        throw std::invalid_argument("build_efficientnet_b0: in/stem/head/class channels must be multiples of tile");
    if (sp.height == 0 || sp.width == 0 || sp.stages.empty())
        throw std::invalid_argument("build_efficientnet_b0: invalid geometry");
    if (!std::isfinite(sp.eps) || sp.eps <= 0.0f)
        throw std::invalid_argument("build_efficientnet_b0: eps must be finite and positive");
    for (const auto& st : sp.stages)
        if (st.t == 0 || !aligned(st.c) || st.n == 0 || (st.s != 1 && st.s != 2) ||
            (st.k != 3 && st.k != 5) || !aligned(st.se))
            throw std::invalid_argument("build_efficientnet_b0: invalid stage");

    EfficientNetB0 net;
    auto& nd = net.node_data;
    const Size N = sp.batch;
    std::uint64_t seed = sp.seed;

    net.input = ef_synth(static_cast<std::size_t>(N) * sp.in_channels * sp.height * sp.width, seed, 1.0f);
    std::vector<float> ox = net.input;
    Size cur_ch = sp.in_channels, cur_H = sp.height, cur_W = sp.width;
    auto out_dim = [](Size in, Size K, Size stride, Size pad) { return (in + 2 * pad - K) / stride + 1; };

    // ----- stem: 3x3 s1 conv -> BN -> SiLU -----------------------------------
    std::size_t in_node;
    {
        auto cc = ef_conv_cfg(N, sp.in_channels, sp.stem_channels, cur_H, cur_W, 3, 1, 1);
        auto w  = ef_synth(static_cast<std::size_t>(sp.stem_channels) * sp.in_channels * 9, seed, 0.1f);
        auto bn = ef_bn(sp.stem_channels, seed, sp.eps);
        auto c = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "stem_conv");
        auto b = g.add_kernel(Kernel::create_batchnorm(N, sp.stem_channels, cur_H, cur_W, sp.eps), "stem_bn");
        auto r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SILU, {sp.stem_channels * cur_H * cur_W}), "stem_silu");
        g.add_edge(c, b); g.add_edge(b, r);
        nd[c].filter = w; ef_set_bn(nd[b], bn);
        ox = ef_pw_conv_bn(ox, w, cc, bn, true);
        in_node = r; cur_ch = sp.stem_channels;
        ++seed;
    }

    // ----- MBConv+SE bottleneck stack -----------------------------------------
    for (const auto& st : sp.stages) {
        for (Size blk = 0; blk < st.n; ++blk) {
            const Size stride = (blk == 0) ? st.s : 1;
            const Size in_ch = cur_ch, out_ch = st.c;
            const Size hidden = st.t * in_ch;
            const Size pad = st.k / 2;
            const bool expand = (st.t > 1);
            const bool residual = (stride == 1 && in_ch == out_ch);
            const Size Hd = out_dim(cur_H, st.k, stride, pad), Wd = out_dim(cur_W, st.k, stride, pad);

            auto ccd = ef_conv_cfg(N, hidden, hidden, cur_H, cur_W, st.k, stride, pad, hidden); // depthwise
            auto ccp = ef_conv_cfg(N, hidden, out_ch, Hd, Wd, 1, 1, 0);                          // project 1x1
            auto wd = ef_synth(static_cast<std::size_t>(hidden) * st.k * st.k, seed + 1, 0.25f);
            auto wp = ef_synth(static_cast<std::size_t>(out_ch) * hidden, seed + 2, 0.12f);
            auto bnd = ef_bn(hidden, seed + 1, sp.eps), bnp = ef_bn(out_ch, seed + 2, sp.eps);

            // Expansion (only t > 1): 1x1 -> BN -> SiLU.
            std::size_t dw_in_node = in_node;
            std::vector<float> dw_in = ox;
            if (expand) {
                auto cce = ef_conv_cfg(N, in_ch, hidden, cur_H, cur_W, 1, 1, 0);
                auto we  = ef_synth(static_cast<std::size_t>(hidden) * in_ch, seed, 0.12f);
                auto bne = ef_bn(hidden, seed, sp.eps);
                auto ce = g.add_kernel(Kernel::create_conv2d(cce, false, ActivationType::NONE), "expand");
                auto be = g.add_kernel(Kernel::create_batchnorm(N, hidden, cur_H, cur_W, bne.eps), "bn_e");
                auto re = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SILU, {hidden * cur_H * cur_W}), "silu_e");
                g.add_edge(in_node, ce); g.add_edge(ce, be); g.add_edge(be, re);
                nd[ce].filter = we; ef_set_bn(nd[be], bne);
                dw_in = ef_pw_conv_bn(ox, we, cce, bne, true);
                dw_in_node = re;
            }

            // Depthwise -> BN -> SiLU  => A
            auto cd = g.add_kernel(Kernel::create_conv2d(ccd, false, ActivationType::NONE), "depthwise");
            auto bd = g.add_kernel(Kernel::create_batchnorm(N, hidden, Hd, Wd, bnd.eps), "bn_d");
            auto rd = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SILU, {hidden * Hd * Wd}), "silu_d");
            g.add_edge(dw_in_node, cd); g.add_edge(cd, bd); g.add_edge(bd, rd);
            nd[cd].filter = wd; ef_set_bn(nd[bd], bnd);
            auto A = ef_dw_conv_bn(dw_in, wd, ccd, bnd, true);

            // Squeeze-and-excitation: GAP -> FC_reduce(ReLU) -> FC_expand -> sigmoid -> scale
            auto wr = ef_synth(static_cast<std::size_t>(hidden) * st.se, seed + 3, 0.15f);
            auto br = ef_synth(st.se, seed + 4, 0.1f);
            auto wx = ef_synth(static_cast<std::size_t>(st.se) * hidden, seed + 5, 0.15f);
            auto bx = ef_synth(hidden, seed + 6, 0.1f);
            auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, hidden, Hd, Wd), "se_gap");
            auto fr = g.add_kernel(Kernel::create_matmul(N, st.se, hidden), "se_reduce");
            auto fx = g.add_kernel(Kernel::create_matmul(N, hidden, st.se), "se_expand");
            auto sg = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SIGMOID, {hidden}), "se_sigmoid");
            auto sc = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::MUL, {hidden * Hd * Wd}), "se_scale");
            g.add_edge(rd, gp); g.add_edge(gp, fr); g.add_edge(fr, fx); g.add_edge(fx, sg);
            g.add_edge(rd, sc);              // scale main branch = A
            g.add_edge(sg, sc, "C", "B");    // scale gate branch
            nd[fr].fc_weight = wr; nd[fr].fc_bias = br; nd[fr].fc_M = N; nd[fr].fc_K = hidden; nd[fr].fc_N = st.se; nd[fr].fc_relu = true;
            nd[fx].fc_weight = wx; nd[fx].fc_bias = bx; nd[fx].fc_M = N; nd[fx].fc_K = st.se; nd[fx].fc_N = hidden; nd[fx].fc_relu = false;

            // SE oracle
            std::vector<float> sq(static_cast<std::size_t>(N) * hidden, 0.0f);
            for (Size n = 0; n < N; ++n)
                for (Size c = 0; c < hidden; ++c) {
                    float s = 0.0f;
                    for (Size p = 0; p < Hd * Wd; ++p)
                        s += A[(static_cast<std::size_t>(n) * hidden + c) * Hd * Wd + p];
                    sq[n * hidden + c] = s / static_cast<float>(Hd * Wd);
                }
            auto red = ef_matmul(sq, wr, br, N, hidden, st.se, true);
            auto exv = ef_matmul(red, wx, bx, N, st.se, hidden, false);
            std::vector<float> A_scaled(A.size());
            for (Size n = 0; n < N; ++n)
                for (Size c = 0; c < hidden; ++c) {
                    const float gate = 1.0f / (1.0f + std::exp(-exv[n * hidden + c]));
                    for (Size p = 0; p < Hd * Wd; ++p) {
                        const std::size_t idx = (static_cast<std::size_t>(n) * hidden + c) * Hd * Wd + p;
                        A_scaled[idx] = A[idx] * gate;
                    }
                }

            // Project 1x1 -> BN (linear bottleneck)
            auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "project");
            auto bp = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, bnp.eps), "bn_p");
            g.add_edge(sc, cp); g.add_edge(cp, bp);
            nd[cp].filter = wp; ef_set_bn(nd[bp], bnp);
            auto p = ef_pw_conv_bn(A_scaled, wp, ccp, bnp, false);

            std::size_t sink = bp;
            if (residual) {
                auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {out_ch * Hd * Wd}), "add");
                g.add_edge(bp, ad);                  // main branch
                g.add_edge(in_node, ad, "C", "B");   // explicit identity skip = block input
                for (std::size_t i = 0; i < p.size(); ++i) p[i] += ox[i];
                sink = ad;
            }
            ox = std::move(p);
            in_node = sink; cur_ch = out_ch; cur_H = Hd; cur_W = Wd;
            seed += 7;
        }
    }

    // ----- head: 1x1 conv -> BN -> SiLU --------------------------------------
    {
        auto cc = ef_conv_cfg(N, cur_ch, sp.head_channels, cur_H, cur_W, 1, 1, 0);
        auto w  = ef_synth(static_cast<std::size_t>(sp.head_channels) * cur_ch, seed, 0.12f);
        auto bn = ef_bn(sp.head_channels, seed, sp.eps);
        auto c = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "head_conv");
        auto b = g.add_kernel(Kernel::create_batchnorm(N, sp.head_channels, cur_H, cur_W, sp.eps), "head_bn");
        auto r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SILU, {sp.head_channels * cur_H * cur_W}), "head_silu");
        g.add_edge(in_node, c); g.add_edge(c, b); g.add_edge(b, r);
        nd[c].filter = w; ef_set_bn(nd[b], bn);
        ox = ef_pw_conv_bn(ox, w, cc, bn, true);
        in_node = r; cur_ch = sp.head_channels;
        ++seed;
    }

    // ----- head: global-average-pool -> FC ------------------------------------
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, cur_ch, cur_H, cur_W), "gap");
    auto fc = g.add_kernel(Kernel::create_matmul(N, sp.num_classes, cur_ch), "fc");
    g.add_edge(in_node, gp); g.add_edge(gp, fc);

    auto wfc = ef_synth(static_cast<std::size_t>(cur_ch) * sp.num_classes, seed, 0.1f);
    auto bfc = ef_synth(sp.num_classes, seed + 1, 0.2f);
    nd[fc].fc_weight = wfc; nd[fc].fc_bias = bfc;
    nd[fc].fc_M = N; nd[fc].fc_K = cur_ch; nd[fc].fc_N = sp.num_classes;

    schedule::Pool2DGeometry pg; pg.N = N; pg.C = cur_ch; pg.H = cur_H; pg.W = cur_W;
    auto gap = schedule::global_avg_pool_reference(ox, pg);
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
