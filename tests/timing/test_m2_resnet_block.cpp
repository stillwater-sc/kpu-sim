// ============================================================================
// tests/timing/test_m2_resnet_block.cpp
// M2-T2 (#201): a ResNet BasicBlock expressed as a KernelGraph DFG, executed on
// the CSP value path through GraphCspExecutor, validated vs a composed host
// oracle. Exercises the operators (conv/BN/ReLU/residual-add) and the fusions
// (conv+BN fold, conv+ReLU epilogue) in composition. See docs/plans/m2_resnet_dfg.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/graph_csp_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::timing::graph;
using sw::kpu::timing::schedule::Conv2DGeometry;
using sw::kpu::timing::schedule::BatchNormGeometry;

namespace {

// Small deterministic weights (fixed-seed LCG), as M1 - keeps activations O(1)
// so fp32 stays comfortable through the block.
std::vector<float> synth(std::size_t n, uint64_t seed, float scale) {
    std::vector<float> v(n);
    uint64_t s = seed;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<uint32_t>(s >> 33);
        x = (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

void relu_ip(std::vector<float>& v) { for (auto& x : v) x = std::max(0.0f, x); }
std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

Conv2DConfig conv_cfg(Size N, Size Cin, Size Cout, Size H, Size W, Size stride) {
    Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = 3; c.kernel_width = 3;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = 1;
    return c;
}
Conv2DGeometry geom_of(const Conv2DConfig& c) {
    // Conv2DConfig fields are sw::kpu::Size (64-bit); Conv2DGeometry uses the
    // timing Size (32-bit) - cast to avoid MSVC C4267.
    using G = sw::kpu::timing::Size;
    Conv2DGeometry g;
    g.N = static_cast<G>(c.batch_size); g.C_in = static_cast<G>(c.in_channels);
    g.H_in = static_cast<G>(c.input_height); g.W_in = static_cast<G>(c.input_width);
    g.C_out = static_cast<G>(c.out_channels); g.Kh = static_cast<G>(c.kernel_height);
    g.Kw = static_cast<G>(c.kernel_width); g.stride_h = static_cast<G>(c.stride_h);
    g.stride_w = static_cast<G>(c.stride_w); g.pad_h = static_cast<G>(c.padding_h);
    g.pad_w = static_cast<G>(c.padding_w);
    return g;
}

// BatchNorm raw params for C channels.
struct BN { std::vector<float> gamma, beta, mean, var; float eps = 1e-3f; };
BN synth_bn(Size C, uint64_t seed) {
    BN b;
    b.gamma = synth(C, seed + 1, 0.5f);      for (auto& g : b.gamma) g += 1.0f;   // ~1
    b.beta  = synth(C, seed + 2, 0.2f);
    b.mean  = synth(C, seed + 3, 0.3f);
    b.var   = synth(C, seed + 4, 0.2f);      for (auto& v : b.var) v = std::abs(v) + 0.5f;  // >0
    return b;
}

// Host oracle: conv (no bias) -> BN inference -> optional ReLU. Returns NCHW.
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

} // namespace

TEST_CASE("M2 ResNet BasicBlock (identity skip) on the CSP executor",
          "[timing][m2][resnet]") {
    const Size N = 1, C = 16, H = 8, W = 8;  // aligned: M=64, K=144, Cout=16 (T=16)
    auto cc = conv_cfg(N, C, C, H, W, 1);
    auto x  = synth(static_cast<std::size_t>(N) * C * H * W, 1, 1.0f);
    auto w1 = synth(static_cast<std::size_t>(C) * C * 9, 2, 0.1f);
    auto w2 = synth(static_cast<std::size_t>(C) * C * 9, 3, 0.1f);
    BN bn1 = synth_bn(C, 100), bn2 = synth_bn(C, 200);

    // --- host oracle: conv1->BN1->ReLU -> conv2->BN2 -> +x -> ReLU ------------
    auto o1 = conv_bn(x, w1, cc, bn1, /*relu*/true);
    auto o2 = conv_bn(o1, w2, cc, bn2, /*relu*/false);
    auto oref = add(o2, x);
    relu_ip(oref);

    // --- DFG: build the BasicBlock KernelGraph -------------------------------
    KernelGraph g;
    auto c1 = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "conv1");
    auto b1 = g.add_kernel(Kernel::create_batchnorm(N, C, H, W, bn1.eps), "bn1");
    auto r1 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {C * H * W}), "relu1");
    auto c2 = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "conv2");
    auto b2 = g.add_kernel(Kernel::create_batchnorm(N, C, H, W, bn2.eps), "bn2");
    auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {C * H * W}), "add");
    auto r2 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {C * H * W}), "relu2");
    g.add_edge(c1, b1); g.add_edge(b1, r1); g.add_edge(r1, c2);
    g.add_edge(c2, b2); g.add_edge(b2, ad); g.add_edge(ad, r2);
    // identity skip: the ADD has one edge (bn2->add); its 2nd operand is x.

    // DFG structural checks: the fusion pass will find conv->BN pairs.
    REQUIRE(g.num_nodes() == 7);
    REQUIRE(g.get_execution_order().size() == 7);

    std::unordered_map<std::size_t, NodeData> nd;
    nd[c1].filter = w1; set_bn(nd[b1], bn1);
    nd[c2].filter = w2; set_bn(nd[b2], bn2);

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, /*T*/16);

    REQUIRE(result.output.size() == oref.size());
    // Fusion fired: 7 graph nodes collapse to 4 CSP ops - bn1/bn2 folded into
    // their convs, relu1 fused as conv1's epilogue; only conv1, conv2, add, and
    // the final relu2 execute.
    REQUIRE(result.stats.ops == 4);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < oref.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - oref[i]));
    INFO("max_err=" << max_err << " ops=" << result.stats.ops
         << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 1e-3f);
}

TEST_CASE("M2 ResNet BasicBlock (1x1 projection skip) on the CSP executor",
          "[timing][m2][resnet]") {
    // Channel-changing block: Cin=16 -> Cout=32; skip = 1x1 conv projection + BN.
    const Size N = 1, Cin = 16, Cout = 32, H = 8, W = 8;
    auto cc1 = conv_cfg(N, Cin, Cout, H, W, 1);   // 3x3
    auto cc2 = conv_cfg(N, Cout, Cout, H, W, 1);  // 3x3
    Conv2DConfig ccp = conv_cfg(N, Cin, Cout, H, W, 1);  // 1x1 projection
    ccp.kernel_height = ccp.kernel_width = 1; ccp.padding_h = ccp.padding_w = 0;

    auto x   = synth(static_cast<std::size_t>(N) * Cin * H * W, 11, 1.0f);
    auto w1  = synth(static_cast<std::size_t>(Cout) * Cin * 9, 12, 0.1f);
    auto w2  = synth(static_cast<std::size_t>(Cout) * Cout * 9, 13, 0.1f);
    auto wp  = synth(static_cast<std::size_t>(Cout) * Cin * 1, 14, 0.1f);
    BN bn1 = synth_bn(Cout, 300), bn2 = synth_bn(Cout, 400), bnp = synth_bn(Cout, 500);

    // oracle
    auto o1   = conv_bn(x, w1, cc1, bn1, true);
    auto o2   = conv_bn(o1, w2, cc2, bn2, false);
    auto skip = conv_bn(x, wp, ccp, bnp, false);   // projection branch
    auto oref = add(o2, skip);
    relu_ip(oref);

    // DFG with a real skip branch (conv_proj -> bn_proj -> add)
    KernelGraph g;
    auto c1 = g.add_kernel(Kernel::create_conv2d(cc1, false, ActivationType::NONE), "conv1");
    auto b1 = g.add_kernel(Kernel::create_batchnorm(N, Cout, H, W, bn1.eps), "bn1");
    auto r1 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {Cout * H * W}), "relu1");
    auto c2 = g.add_kernel(Kernel::create_conv2d(cc2, false, ActivationType::NONE), "conv2");
    auto b2 = g.add_kernel(Kernel::create_batchnorm(N, Cout, H, W, bn2.eps), "bn2");
    auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "conv_proj");
    auto bp = g.add_kernel(Kernel::create_batchnorm(N, Cout, H, W, bnp.eps), "bn_proj");
    auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {Cout * H * W}), "add");
    auto r2 = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU, {Cout * H * W}), "relu2");
    g.add_edge(c1, b1); g.add_edge(b1, r1); g.add_edge(r1, c2); g.add_edge(c2, b2);
    g.add_edge(cp, bp);
    g.add_edge(b2, ad); g.add_edge(bp, ad, "C", "B");  // main + projection skip
    g.add_edge(ad, r2);

    std::unordered_map<std::size_t, NodeData> nd;
    nd[c1].filter = w1; set_bn(nd[b1], bn1);
    nd[c2].filter = w2; set_bn(nd[b2], bn2);
    nd[cp].filter = wp; set_bn(nd[bp], bnp);

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, 16);

    REQUIRE(result.output.size() == oref.size());
    // 9 graph nodes -> 5 CSP ops: conv1, conv2, conv_proj (each with folded BN),
    // add, relu2; bn1/bn2/bn_proj folded and relu1 fused.
    REQUIRE(result.stats.ops == 5);
    float max_err = 0.0f;
    for (std::size_t i = 0; i < oref.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - oref[i]));
    INFO("projection max_err=" << max_err);
    REQUIRE(max_err < 1e-3f);
}
