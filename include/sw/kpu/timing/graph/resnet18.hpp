// ============================================================================
// include/sw/kpu/timing/graph/resnet18.hpp
// Reusable ResNet-18 builder: constructs the full network (stem + 4 stages of
// BasicBlocks + global-average-pool + FC) as a KernelGraph DFG, with matching
// per-node weights and a composed host oracle (M2-T4, #206). See
// docs/plans/m2_resnet_dfg.md and docs/milestones/M2_resnet.md.
//
// The graph is executed through GraphCspExecutor on the CSP value path; the
// oracle is the elementwise reference the CSP output is validated against. Both
// use the same deterministic synthetic weights. Dims are scaled down for a fast
// CSP demo; batch N keeps every conv GEMM's M = N*Hout*Wout tile-aligned.
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
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing::graph {

/// ResNet-18 topology (scaled for the CSP demo). Channels are tile-aligned; the
/// batch keeps every conv GEMM's M = batch*Hout*Wout a multiple of the tile size
/// (the FC requires batch itself to be a multiple, so batch >= 16).
///
/// The default is a fast, CI-friendly scale that exercises the FULL structure
/// (stem, four stages with stride-2 downsampling + 1x1 projection, GAP, FC). The
/// [2,2,2,2] block depth of the real ResNet-18 is validated separately by
/// test_m2_resnet18_tower; here blocks_per_stage defaults to 1 to keep the
/// tile-by-tile CSP simulation fast. Bump height/width/blocks_per_stage for a
/// larger run.
struct ResNet18Spec {
    Size batch = 16;
    Size in_channels = 16, height = 4, width = 4;      // stem input (16ch, 4x4)
    // Uniform channels keep the global-average-pool's N*C per-channel reductions
    // small (the GAP dominates wall-clock at low spatial); channel growth is
    // exercised by the tower test. Downsampling (spatial) + projections remain.
    std::vector<Size> stage_channels = {16, 16, 16, 16};
    Size blocks_per_stage = 1;                          // fast default; [2,2,2,2] in the tower test
    Size num_classes = 16;
    Size tile = 16;   ///< GEMM tile size; batch/channels/num_classes must be multiples
    float eps = 1e-3f;
    std::uint64_t seed = 1000;
};

/// Built network: the graph is populated into the caller's KernelGraph; this
/// carries the per-node weights, the synthetic input, the host-oracle output,
/// and the sink node id.
struct ResNet18 {
    std::unordered_map<std::size_t, NodeData> node_data;
    std::vector<float> input;
    std::vector<float> oracle;   // [batch, num_classes]
    std::size_t output_node = 0;
    std::size_t num_nodes = 0;
};

namespace detail {

inline std::vector<float> rn_synth(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<float> v(n);
    std::uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<std::uint32_t>(s >> 33);   // [0, 0x7FFFFFFF]
        // Map to [-scale, +scale] (both signs, so ReLU/fusion are exercised).
        x = (2.0f * static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

inline sw::kpu::Conv2DConfig rn_conv_cfg(Size N, Size Cin, Size Cout, Size H, Size W,
                                         Size K, Size stride, Size pad) {
    sw::kpu::Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = c.kernel_width = K;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = pad;
    return c;
}
inline schedule::Conv2DGeometry rn_geom(const sw::kpu::Conv2DConfig& c) {
    schedule::Conv2DGeometry g;
    g.N = static_cast<Size>(c.batch_size); g.C_in = static_cast<Size>(c.in_channels);
    g.H_in = static_cast<Size>(c.input_height); g.W_in = static_cast<Size>(c.input_width);
    g.C_out = static_cast<Size>(c.out_channels); g.Kh = static_cast<Size>(c.kernel_height);
    g.Kw = static_cast<Size>(c.kernel_width); g.stride_h = static_cast<Size>(c.stride_h);
    g.stride_w = static_cast<Size>(c.stride_w); g.pad_h = static_cast<Size>(c.padding_h);
    g.pad_w = static_cast<Size>(c.padding_w);
    return g;
}

struct RnBN { std::vector<float> gamma, beta, mean, var; float eps; };
inline RnBN rn_bn(Size C, std::uint64_t seed, float eps) {
    RnBN b; b.eps = eps;
    b.gamma = rn_synth(C, seed * 4 + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;
    b.beta  = rn_synth(C, seed * 4 + 2, 0.2f);
    b.mean  = rn_synth(C, seed * 4 + 3, 0.3f);
    b.var   = rn_synth(C, seed * 4 + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;
    return b;
}
inline void rn_set_bn(NodeData& nd, const RnBN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}
// conv (no bias) -> BN inference -> optional ReLU (the host oracle for a conv+BN block).
inline std::vector<float> rn_conv_bn(const std::vector<float>& x, const std::vector<float>& w,
                                     const sw::kpu::Conv2DConfig& cc, const RnBN& bn, bool relu) {
    const auto g = rn_geom(cc);
    auto z = schedule::conv2d_reference(x, w, {}, g, false);
    schedule::BatchNormGeometry bg; bg.N = g.N; bg.C = g.C_out; bg.H = g.H_out(); bg.W = g.W_out();
    auto y = schedule::batchnorm_reference(z, bn.gamma, bn.beta, bn.mean, bn.var, bn.eps, bg);
    if (relu) for (auto& v : y) v = std::max(0.0f, v);
    return y;
}

} // namespace detail

/**
 * @brief Build the full ResNet-18 into `g`; return weights + input + oracle.
 *
 * Topology: stem (3x3 conv -> BN -> ReLU) + four stages of BasicBlocks (the first
 * block of stages 2-4 stride-2 downsamples with a 1x1 projection skip) ->
 * global-average-pool -> FC. Execute the returned graph through GraphCspExecutor
 * and compare its output to `oracle` (tolerance ~5e-3 over the depth).
 */
[[nodiscard]] inline ResNet18 build_resnet18(KernelGraph& g, const ResNet18Spec& sp) {
    using namespace detail;
    using sw::kpu::Kernel;
    using sw::kpu::ElementwiseOp;
    using sw::kpu::ActivationType;

    // Every conv/FC GEMM is tiled at sp.tile; batch (the FC/conv M axis),
    // channels (K/N), and num_classes (FC N) must be nonzero multiples so the
    // dimensions are tile-aligned. Fail fast with a clear message here rather
    // than deep inside a runner.
    if (sp.tile == 0 || sp.batch == 0 || sp.batch % sp.tile != 0)
        throw std::invalid_argument("build_resnet18: batch must be a nonzero multiple of tile");
    if (sp.in_channels == 0 || sp.in_channels % sp.tile != 0)
        throw std::invalid_argument("build_resnet18: in_channels must be a nonzero multiple of tile");
    if (sp.num_classes == 0 || sp.num_classes % sp.tile != 0)
        throw std::invalid_argument("build_resnet18: num_classes must be a nonzero multiple of tile");
    for (Size ch : sp.stage_channels)
        if (ch == 0 || ch % sp.tile != 0)
            throw std::invalid_argument("build_resnet18: stage channels must be nonzero multiples of tile");
    if (sp.height == 0 || sp.width == 0 || sp.stage_channels.empty() || sp.blocks_per_stage == 0)
        throw std::invalid_argument("build_resnet18: invalid geometry");

    ResNet18 net;
    auto& nd = net.node_data;
    const Size N = sp.batch;
    std::uint64_t seed = sp.seed;

    net.input = rn_synth(static_cast<std::size_t>(N) * sp.in_channels * sp.height * sp.width,
                         seed, 1.0f);
    std::vector<float> ox = net.input;
    Size cur_ch = sp.in_channels, cur_H = sp.height, cur_W = sp.width;

    auto out_dim = [](Size in, Size K, Size stride, Size pad) {
        return (in + 2 * pad - K) / stride + 1;
    };

    // ----- stem: 3x3 s1 conv -> BN -> ReLU ------------------------------------
    std::size_t in_node;
    {
        auto cc = rn_conv_cfg(N, sp.in_channels, sp.in_channels, cur_H, cur_W, 3, 1, 1);
        auto w = rn_synth(static_cast<std::size_t>(sp.in_channels) * sp.in_channels * 9, seed, 0.1f);
        auto bn = rn_bn(sp.in_channels, seed, sp.eps);
        auto c = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "stem_conv");
        auto b = g.add_kernel(Kernel::create_batchnorm(N, sp.in_channels, cur_H, cur_W, sp.eps), "stem_bn");
        auto r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {sp.in_channels * cur_H * cur_W}), "stem_relu");
        g.add_edge(c, b); g.add_edge(b, r);
        nd[c].filter = w; rn_set_bn(nd[b], bn);
        ox = rn_conv_bn(ox, w, cc, bn, true);
        in_node = r;
        ++seed;
    }

    // ----- 4 stages of BasicBlocks --------------------------------------------
    for (std::size_t s = 0; s < sp.stage_channels.size(); ++s) {
        const Size out_ch = sp.stage_channels[s];
        for (Size blk = 0; blk < sp.blocks_per_stage; ++blk) {
            const bool downsample = (s > 0 && blk == 0);
            const Size stride = downsample ? 2 : 1;
            const Size in_ch = cur_ch;
            const Size Hd = out_dim(cur_H, 3, stride, 1), Wd = out_dim(cur_W, 3, stride, 1);

            auto cc1 = rn_conv_cfg(N, in_ch, out_ch, cur_H, cur_W, 3, stride, 1);
            auto cc2 = rn_conv_cfg(N, out_ch, out_ch, Hd, Wd, 3, 1, 1);
            auto w1 = rn_synth(static_cast<std::size_t>(out_ch) * in_ch * 9, seed, 0.1f);
            auto w2 = rn_synth(static_cast<std::size_t>(out_ch) * out_ch * 9, seed + 1, 0.1f);
            auto bn1 = rn_bn(out_ch, seed, sp.eps), bn2 = rn_bn(out_ch, seed + 1, sp.eps);

            auto c1 = g.add_kernel(Kernel::create_conv2d(cc1, false, ActivationType::NONE), "conv1");
            auto b1 = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn1");
            auto r1 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {out_ch * Hd * Wd}), "relu1");
            auto c2 = g.add_kernel(Kernel::create_conv2d(cc2, false, ActivationType::NONE), "conv2");
            auto b2 = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn2");
            auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {out_ch * Hd * Wd}), "add");
            auto r2 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {out_ch * Hd * Wd}), "relu2");
            g.add_edge(in_node, c1); g.add_edge(c1, b1); g.add_edge(b1, r1);
            g.add_edge(r1, c2); g.add_edge(c2, b2); g.add_edge(b2, ad); g.add_edge(ad, r2);
            nd[c1].filter = w1; rn_set_bn(nd[b1], bn1);
            nd[c2].filter = w2; rn_set_bn(nd[b2], bn2);

            auto o1 = rn_conv_bn(ox, w1, cc1, bn1, true);
            auto o2 = rn_conv_bn(o1, w2, cc2, bn2, false);

            std::vector<float> skip;
            if (downsample) {
                auto ccp = rn_conv_cfg(N, in_ch, out_ch, cur_H, cur_W, 1, stride, 0);
                auto wp = rn_synth(static_cast<std::size_t>(out_ch) * in_ch, seed + 2, 0.1f);
                auto bnp = rn_bn(out_ch, seed + 2, sp.eps);
                auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "conv_proj");
                auto bp = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn_proj");
                g.add_edge(in_node, cp); g.add_edge(cp, bp); g.add_edge(bp, ad, "C", "B");
                nd[cp].filter = wp; rn_set_bn(nd[bp], bnp);
                skip = rn_conv_bn(ox, wp, ccp, bnp, false);
            } else {
                g.add_edge(in_node, ad, "C", "B");
                skip = ox;
            }
            std::vector<float> ores(o2.size());
            for (std::size_t i = 0; i < o2.size(); ++i) ores[i] = std::max(0.0f, o2[i] + skip[i]);
            ox = std::move(ores);

            in_node = r2; cur_ch = out_ch; cur_H = Hd; cur_W = Wd;
            seed += 3;
        }
    }

    // ----- head: global-average-pool -> FC ------------------------------------
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, cur_ch, cur_H, cur_W), "gap");
    auto fc = g.add_kernel(Kernel::create_matmul(N, sp.num_classes, cur_ch), "fc");
    g.add_edge(in_node, gp); g.add_edge(gp, fc);

    auto wfc = rn_synth(static_cast<std::size_t>(cur_ch) * sp.num_classes, seed, 0.1f);
    auto bfc = rn_synth(sp.num_classes, seed + 1, 0.2f);
    nd[fc].fc_weight = wfc; nd[fc].fc_bias = bfc;
    nd[fc].fc_M = N; nd[fc].fc_K = cur_ch; nd[fc].fc_N = sp.num_classes;

    // oracle head
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
