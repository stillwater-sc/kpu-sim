// ============================================================================
// tests/timing/test_m2_resnet18_tower.cpp
// M2-T3 (#203), part A: the ResNet-18 residual TOWER (stem + 4 stages of stacked
// BasicBlocks with stride-2 downsampling + 1x1 projection skips) built as a
// KernelGraph DFG and executed end-to-end on the CSP value path through
// GraphCspExecutor, validated per-stage and end-to-end vs a composed host oracle.
//
// Uses only conv/BN/ReLU/residual-add (the M2-T2 bridge); the GAP+FC head is
// part B. batch N=16 keeps every conv GEMM's M = N*Hout*Wout tile-aligned; all
// channel counts are multiples of the tile size. See docs/plans/m2_resnet_dfg.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/graph_csp_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::timing::graph;
using sw::kpu::timing::schedule::Conv2DGeometry;
using sw::kpu::timing::schedule::BatchNormGeometry;
using G = sw::kpu::timing::Size;

namespace {

std::vector<float> synth(std::size_t n, uint64_t seed, float scale) {
    std::vector<float> v(n);
    uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<uint32_t>(s >> 33);
        x = (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}
void relu_ip(std::vector<float>& v) { for (auto& x : v) x = std::max(0.0f, x); }

Conv2DConfig conv_cfg(Size N, Size Cin, Size Cout, Size H, Size W, Size K, Size stride, Size pad) {
    Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = c.kernel_width = K;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = pad;
    return c;
}
Conv2DGeometry geom_of(const Conv2DConfig& c) {
    Conv2DGeometry g;
    g.N = static_cast<G>(c.batch_size); g.C_in = static_cast<G>(c.in_channels);
    g.H_in = static_cast<G>(c.input_height); g.W_in = static_cast<G>(c.input_width);
    g.C_out = static_cast<G>(c.out_channels); g.Kh = static_cast<G>(c.kernel_height);
    g.Kw = static_cast<G>(c.kernel_width); g.stride_h = static_cast<G>(c.stride_h);
    g.stride_w = static_cast<G>(c.stride_w); g.pad_h = static_cast<G>(c.padding_h);
    g.pad_w = static_cast<G>(c.padding_w);
    return g;
}

// Deterministic BatchNorm params for C channels, seeded.
struct BN { std::vector<float> gamma, beta, mean, var; float eps = 1e-3f; };
BN synth_bn(Size C, uint64_t seed) {
    BN b;
    b.gamma = synth(C, seed * 4 + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;
    b.beta  = synth(C, seed * 4 + 2, 0.2f);
    b.mean  = synth(C, seed * 4 + 3, 0.3f);
    b.var   = synth(C, seed * 4 + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;
    return b;
}

// Host conv (no bias) -> BN inference -> optional ReLU. NCHW in/out.
std::vector<float> conv_bn(const std::vector<float>& x, const std::vector<float>& w,
                           const Conv2DConfig& cc, const BN& bn, bool relu) {
    const auto g = geom_of(cc);
    auto z = sw::kpu::timing::schedule::conv2d_reference(x, w, {}, g, false);
    BatchNormGeometry bg; bg.N = g.N; bg.C = g.C_out; bg.H = g.H_out(); bg.W = g.W_out();
    auto y = sw::kpu::timing::schedule::batchnorm_reference(z, bn.gamma, bn.beta,
                                                            bn.mean, bn.var, bn.eps, bg);
    if (relu) relu_ip(y);
    return y;
}
void set_bn(NodeData& nd, const BN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}

// ResNet-18 tower shape (scaled down for the CSP demo; batch keeps M aligned).
struct Spec {
    Size N = 16, C0 = 16, H = 8, W = 8;              // stem input/output (16ch, 8x8)
    std::vector<Size> stage_ch = {16, 32, 64, 128};  // channels per stage
    Size blocks_per_stage = 2;                       // ResNet-18: [2,2,2,2]
    float eps = 1e-3f;
};

// A running spatial extent for a stride.
Size out_dim(Size in, Size K, Size stride, Size pad) {
    return (in + 2 * pad - K) / stride + 1;
}

} // namespace

TEST_CASE("M2 ResNet-18 residual tower on the CSP executor",
          "[timing][m2][resnet][tower]") {
    Spec sp;
    const Size N = sp.N;
    auto input = synth(static_cast<std::size_t>(N) * sp.C0 * sp.H * sp.W, 1, 1.0f);

    KernelGraph g;
    std::unordered_map<std::size_t, NodeData> nd;
    uint64_t seed = 1000;

    // ----- host oracle state, threaded in lockstep with the graph build -------
    std::vector<float> ox = input;      // current oracle tensor
    Size cur_ch = sp.C0, cur_H = sp.H, cur_W = sp.W;

    // ----- stem: 3x3 s1 conv -> BN -> ReLU ------------------------------------
    auto stem_cc = conv_cfg(N, sp.C0, sp.C0, cur_H, cur_W, 3, 1, 1);
    auto stem_w = synth(static_cast<std::size_t>(sp.C0) * sp.C0 * 9, seed, 0.1f);
    BN stem_bn = synth_bn(sp.C0, seed);
    std::size_t stem_c = g.add_kernel(Kernel::create_conv2d(stem_cc, false, ActivationType::NONE), "stem_conv");
    std::size_t stem_b = g.add_kernel(Kernel::create_batchnorm(N, sp.C0, cur_H, cur_W, sp.eps), "stem_bn");
    std::size_t stem_r = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {sp.C0 * cur_H * cur_W}), "stem_relu");
    g.add_edge(stem_c, stem_b); g.add_edge(stem_b, stem_r);   // stem_conv reads external input
    nd[stem_c].filter = stem_w; set_bn(nd[stem_b], stem_bn);
    ox = conv_bn(ox, stem_w, stem_cc, stem_bn, true);
    ++seed;
    std::size_t in_node = stem_r;   // the tower's running input node

    // ----- 4 stages of BasicBlocks --------------------------------------------
    for (std::size_t s = 0; s < sp.stage_ch.size(); ++s) {
        const Size out_ch = sp.stage_ch[s];
        for (Size blk = 0; blk < sp.blocks_per_stage; ++blk) {
            const bool downsample = (s > 0 && blk == 0);   // first block of stages 2-4
            const Size stride = downsample ? 2 : 1;
            const Size in_ch = cur_ch;
            const Size Hd = out_dim(cur_H, 3, stride, 1), Wd = out_dim(cur_W, 3, stride, 1);

            // main branch: conv1(in->out, stride) -> BN -> ReLU -> conv2(out->out) -> BN
            auto cc1 = conv_cfg(N, in_ch, out_ch, cur_H, cur_W, 3, stride, 1);
            auto cc2 = conv_cfg(N, out_ch, out_ch, Hd, Wd, 3, 1, 1);
            auto w1 = synth(static_cast<std::size_t>(out_ch) * in_ch * 9, seed, 0.1f);
            auto w2 = synth(static_cast<std::size_t>(out_ch) * out_ch * 9, seed + 1, 0.1f);
            BN bn1 = synth_bn(out_ch, seed), bn2 = synth_bn(out_ch, seed + 1);

            std::size_t c1 = g.add_kernel(Kernel::create_conv2d(cc1, false, ActivationType::NONE), "conv1");
            std::size_t b1 = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn1");
            std::size_t r1 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {out_ch * Hd * Wd}), "relu1");
            std::size_t c2 = g.add_kernel(Kernel::create_conv2d(cc2, false, ActivationType::NONE), "conv2");
            std::size_t b2 = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn2");
            std::size_t ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {out_ch * Hd * Wd}), "add");
            std::size_t r2 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {out_ch * Hd * Wd}), "relu2");
            g.add_edge(in_node, c1); g.add_edge(c1, b1); g.add_edge(b1, r1);
            g.add_edge(r1, c2); g.add_edge(c2, b2); g.add_edge(b2, ad); g.add_edge(ad, r2);
            nd[c1].filter = w1; set_bn(nd[b1], bn1);
            nd[c2].filter = w2; set_bn(nd[b2], bn2);

            // oracle main branch
            auto o1 = conv_bn(ox, w1, cc1, bn1, true);
            auto o2 = conv_bn(o1, w2, cc2, bn2, false);

            // skip branch: identity, or 1x1 projection when the block downsamples
            std::vector<float> skip;
            if (downsample) {
                auto ccp = conv_cfg(N, in_ch, out_ch, cur_H, cur_W, 1, stride, 0);
                auto wp = synth(static_cast<std::size_t>(out_ch) * in_ch, seed + 2, 0.1f);
                BN bnp = synth_bn(out_ch, seed + 2);
                std::size_t cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "conv_proj");
                std::size_t bp = g.add_kernel(Kernel::create_batchnorm(N, out_ch, Hd, Wd, sp.eps), "bn_proj");
                g.add_edge(in_node, cp); g.add_edge(cp, bp);
                g.add_edge(bp, ad, "C", "B");     // projected skip -> ADD operand B
                nd[cp].filter = wp; set_bn(nd[bp], bnp);
                skip = conv_bn(ox, wp, ccp, bnp, false);
            } else {
                g.add_edge(in_node, ad, "C", "B");  // identity skip edge -> ADD operand B
                skip = ox;
            }

            // oracle: add + relu
            std::vector<float> ores(o2.size());
            for (std::size_t i = 0; i < o2.size(); ++i) ores[i] = o2[i] + skip[i];
            relu_ip(ores);
            ox = std::move(ores);

            in_node = r2;
            cur_ch = out_ch; cur_H = Hd; cur_W = Wd;
            seed += 3;
        }
    }

    INFO("tower: " << g.num_nodes() << " nodes, output " << cur_ch << "x" << cur_H << "x" << cur_W);
    REQUIRE(g.get_execution_order().size() == g.num_nodes());

    // ----- execute the whole tower DFG on the CSP value path ------------------
    GraphCspExecutor exec;
    auto result = exec.run(g, input, nd, /*T*/16);

    REQUIRE(result.output.size() == ox.size());
    // Lock in the fusion contract: the 65-node graph collapses to exactly 36 CSP
    // ops - stem conv (1), and per block conv1+conv2+add+relu2 (4) plus conv_proj
    // for each downsample block (3). A regression in BN-fold or ReLU-fusion would
    // change this count. (1 + 8*4 + 3*1 = 36.)
    REQUIRE(result.stats.ops == 36);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < ox.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - ox[i]));
    INFO("end-to-end max_err=" << max_err << " ops=" << result.stats.ops
         << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 5e-3f);   // 17-conv depth: looser than the single-block 1e-3
}
