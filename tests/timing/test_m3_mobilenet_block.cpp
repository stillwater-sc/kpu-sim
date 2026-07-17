// ============================================================================
// tests/timing/test_m3_mobilenet_block.cpp
// M3-T3 (#131): a MobileNetV2 inverted-residual block expressed as a KernelGraph
// DFG, executed on the CSP value path through GraphCspExecutor, validated vs a
// composed host oracle. Exercises the M3 bridge additions - depthwise conv
// dispatch (groups == Cin -> run_depthwise_conv) and the standalone ReLU6 op -
// in composition with the M2 pointwise-conv/BN-fold/residual-add path.
//
//   expand 1x1 (Cin -> hidden) -> BN -> ReLU6
//   3x3 depthwise (groups=hidden, stride s) -> BN -> ReLU6
//   project 1x1 (hidden -> Cout) -> BN            (linear bottleneck, no act)
//   + residual  when s == 1 && Cin == Cout        (no activation after add)
//
// See docs/plans/m3_mobilenet_dfg.md.
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
#include <sw/kpu/timing/schedule/pooling_window.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::timing::graph;
using sw::kpu::timing::schedule::Conv2DGeometry;
using sw::kpu::timing::schedule::BatchNormGeometry;
using sw::kpu::timing::schedule::Pool2DGeometry;
using G = sw::kpu::timing::Size;

namespace {

// Deterministic weights in [-scale, +scale] (both signs exercise ReLU6 clamps).
std::vector<float> synth(std::size_t n, std::uint64_t seed, float scale) {
    std::vector<float> v(n);
    std::uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<std::uint32_t>(s >> 33);
        x = (2.0f * static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

void relu6_ip(std::vector<float>& v) {
    for (auto& x : v) x = std::min(std::max(0.0f, x), 6.0f);
}

sw::kpu::Conv2DConfig conv_cfg(G N, G Cin, G Cout, G H, G W, G K, G stride, G pad, G groups) {
    sw::kpu::Conv2DConfig c;
    c.batch_size = N; c.in_channels = Cin; c.out_channels = Cout;
    c.input_height = H; c.input_width = W;
    c.kernel_height = c.kernel_width = K;
    c.stride_h = c.stride_w = stride; c.padding_h = c.padding_w = pad;
    c.groups = groups;
    return c;
}
Conv2DGeometry geom_of(const sw::kpu::Conv2DConfig& c) {
    Conv2DGeometry g;
    g.N = static_cast<G>(c.batch_size); g.C_in = static_cast<G>(c.in_channels);
    g.H_in = static_cast<G>(c.input_height); g.W_in = static_cast<G>(c.input_width);
    g.C_out = static_cast<G>(c.out_channels); g.Kh = static_cast<G>(c.kernel_height);
    g.Kw = static_cast<G>(c.kernel_width); g.stride_h = static_cast<G>(c.stride_h);
    g.stride_w = static_cast<G>(c.stride_w); g.pad_h = static_cast<G>(c.padding_h);
    g.pad_w = static_cast<G>(c.padding_w);
    return g;
}
Pool2DGeometry dw_geom(const sw::kpu::Conv2DConfig& c) {
    Pool2DGeometry g;
    g.N = static_cast<G>(c.batch_size); g.C = static_cast<G>(c.in_channels);
    g.H = static_cast<G>(c.input_height); g.W = static_cast<G>(c.input_width);
    g.Kh = static_cast<G>(c.kernel_height); g.Kw = static_cast<G>(c.kernel_width);
    g.stride_h = static_cast<G>(c.stride_h); g.stride_w = static_cast<G>(c.stride_w);
    g.pad_h = static_cast<G>(c.padding_h); g.pad_w = static_cast<G>(c.padding_w);
    return g;
}

struct BN { std::vector<float> gamma, beta, mean, var; float eps = 1e-3f; };
BN synth_bn(G C, std::uint64_t seed) {
    BN b;
    b.gamma = synth(C, seed + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;   // ~1
    b.beta  = synth(C, seed + 2, 0.2f);
    b.mean  = synth(C, seed + 3, 0.3f);
    b.var   = synth(C, seed + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;  // >0
    return b;
}
void set_bn(NodeData& nd, const BN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}

// Host oracle: standard (pointwise) conv -> BN inference -> optional ReLU6. NCHW.
std::vector<float> pw_conv_bn(const std::vector<float>& x, const std::vector<float>& w,
                              const sw::kpu::Conv2DConfig& cc, const BN& bn, bool relu6) {
    const auto g = geom_of(cc);
    auto z = sw::kpu::timing::schedule::conv2d_reference(x, w, {}, g, false);
    BatchNormGeometry bg; bg.N = g.N; bg.C = g.C_out; bg.H = g.H_out(); bg.W = g.W_out();
    auto y = sw::kpu::timing::schedule::batchnorm_reference(z, bn.gamma, bn.beta,
                                                            bn.mean, bn.var, bn.eps, bg);
    if (relu6) relu6_ip(y);
    return y;
}

// Host oracle: per-channel depthwise conv -> BN inference -> optional ReLU6. NCHW.
std::vector<float> dw_conv_bn(const std::vector<float>& x, const std::vector<float>& filter,
                              const sw::kpu::Conv2DConfig& cc, const BN& bn, bool relu6) {
    const auto g = dw_geom(cc);
    const G Hout = g.H_out(), Wout = g.W_out();
    std::vector<float> y(static_cast<std::size_t>(g.N) * g.C * Hout * Wout);
    for (G n = 0; n < g.N; ++n)
        for (G c = 0; c < g.C; ++c) {
            const std::size_t plane = (static_cast<std::size_t>(n) * g.C + c) * g.H * g.W;
            const float scale = bn.gamma[c] / std::sqrt(bn.var[c] + bn.eps);
            const float shift = bn.beta[c] - bn.mean[c] * scale;
            for (G ho = 0; ho < Hout; ++ho)
                for (G wo = 0; wo < Wout; ++wo) {
                    float acc = 0.0f;
                    for (G kh = 0; kh < g.Kh; ++kh) {
                        const long ih = static_cast<long>(ho * g.stride_h + kh) - static_cast<long>(g.pad_h);
                        if (ih < 0 || ih >= static_cast<long>(g.H)) continue;
                        for (G kw = 0; kw < g.Kw; ++kw) {
                            const long iw = static_cast<long>(wo * g.stride_w + kw) - static_cast<long>(g.pad_w);
                            if (iw < 0 || iw >= static_cast<long>(g.W)) continue;
                            acc += x[plane + static_cast<std::size_t>(ih) * g.W + iw] *
                                   filter[(static_cast<std::size_t>(c) * g.Kh + kh) * g.Kw + kw];
                        }
                    }
                    acc = acc * scale + shift;
                    if (relu6) acc = std::min(std::max(0.0f, acc), 6.0f);
                    y[((static_cast<std::size_t>(n) * g.C + c) * Hout + ho) * Wout + wo] = acc;
                }
        }
    return y;
}

// Build the inverted-residual block graph + node data + host oracle. Returns the
// expected output and the number of CSP ops the fusion should collapse to.
struct BuildResult { std::vector<float> oracle; std::size_t expected_ops; std::size_t nodes; };

BuildResult build_block(KernelGraph& g, std::unordered_map<std::size_t, NodeData>& nd,
                        const std::vector<float>& x, G N, G Cin, G Cout, G H, G W,
                        G hidden, G stride) {
    const bool residual = (stride == 1 && Cin == Cout);
    const G Hd = (H + 2 * 1 - 3) / stride + 1, Wd = (W + 2 * 1 - 3) / stride + 1;

    // configs
    auto cce = conv_cfg(N, Cin, hidden, H, W, 1, 1, 0, 1);            // expand 1x1
    auto ccd = conv_cfg(N, hidden, hidden, H, W, 3, stride, 1, hidden); // depthwise 3x3
    auto ccp = conv_cfg(N, hidden, Cout, Hd, Wd, 1, 1, 0, 1);         // project 1x1

    // weights
    auto we = synth(static_cast<std::size_t>(hidden) * Cin, 10, 0.15f);
    auto wd = synth(static_cast<std::size_t>(hidden) * 9, 20, 0.30f);  // [hidden,3,3]
    auto wp = synth(static_cast<std::size_t>(Cout) * hidden, 30, 0.15f);
    BN bne = synth_bn(hidden, 100), bnd = synth_bn(hidden, 200), bnp = synth_bn(Cout, 300);

    // --- host oracle ---------------------------------------------------------
    auto e  = pw_conv_bn(x, we, cce, bne, /*relu6*/true);
    auto d  = dw_conv_bn(e, wd, ccd, bnd, /*relu6*/true);
    auto p  = pw_conv_bn(d, wp, ccp, bnp, /*relu6*/false);
    std::vector<float> oref = p;
    if (residual) for (std::size_t i = 0; i < oref.size(); ++i) oref[i] += x[i];

    // --- DFG -----------------------------------------------------------------
    auto ce = g.add_kernel(Kernel::create_conv2d(cce, false, ActivationType::NONE), "expand");
    auto be = g.add_kernel(Kernel::create_batchnorm(N, hidden, H, W, bne.eps), "bn_e");
    auto re = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * H * W}), "relu6_e");
    auto cd = g.add_kernel(Kernel::create_conv2d(ccd, false, ActivationType::NONE), "depthwise");
    auto bd = g.add_kernel(Kernel::create_batchnorm(N, hidden, Hd, Wd, bnd.eps), "bn_d");
    auto rd = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * Hd * Wd}), "relu6_d");
    auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "project");
    auto bp = g.add_kernel(Kernel::create_batchnorm(N, Cout, Hd, Wd, bnp.eps), "bn_p");
    g.add_edge(ce, be); g.add_edge(be, re); g.add_edge(re, cd);
    g.add_edge(cd, bd); g.add_edge(bd, rd); g.add_edge(rd, cp); g.add_edge(cp, bp);

    nd[ce].filter = we; set_bn(nd[be], bne);
    nd[cd].filter = wd; set_bn(nd[bd], bnd);
    nd[cp].filter = wp; set_bn(nd[bp], bnp);

    std::size_t nodes = 8;
    // CSP op count (each fused conv/depthwise = 1 op; run_relu6 issues 2 VE ops,
    // MAX then MIN): expand(1) + ReLU6(2) + depthwise(1) + ReLU6(2) + project(1)
    // = 7; + the residual ADD (1) when present.
    std::size_t ops = 7;
    if (residual) {
        auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {Cout * Hd * Wd}), "add");
        g.add_edge(bp, ad, "C", "B");   // main branch; 2nd operand = block input (identity skip)
        ++nodes; ++ops;
    }
    return {oref, ops, nodes};
}

} // namespace

TEST_CASE("M3 MobileNetV2 inverted-residual block (identity skip) on the CSP executor",
          "[timing][m3][mobilenet]") {
    // Tile-aligned (T=16): N=16, Cin=Cout=16, hidden=32 (expansion t=2), stride 1.
    const G N = 16, C = 16, H = 8, W = 8, hidden = 32;
    auto x = synth(static_cast<std::size_t>(N) * C * H * W, 1, 1.0f);

    KernelGraph g;
    std::unordered_map<std::size_t, NodeData> nd;
    auto built = build_block(g, nd, x, N, C, C, H, W, hidden, /*stride*/1);

    REQUIRE(g.num_nodes() == built.nodes);   // 9: 3 conv + 3 BN + 2 relu6 + add
    REQUIRE(g.get_execution_order().size() == built.nodes);

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, /*T*/16);

    REQUIRE(result.output.size() == built.oracle.size());
    REQUIRE(result.stats.ops == built.expected_ops);   // 8
    float max_err = 0.0f;
    for (std::size_t i = 0; i < built.oracle.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - built.oracle[i]));
    INFO("identity max_err=" << max_err << " ops=" << result.stats.ops
         << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 1e-3f);
}

TEST_CASE("M3 MobileNetV2 inverted-residual block (stride-2 downsample, no skip)",
          "[timing][m3][mobilenet]") {
    // Downsampling block: stride 2, channel change 16 -> 32, hidden = 48 (t=3).
    // No residual (stride != 1), so the block sink is the project BN.
    const G N = 16, Cin = 16, Cout = 32, H = 8, W = 8, hidden = 48;
    auto x = synth(static_cast<std::size_t>(N) * Cin * H * W, 2, 1.0f);

    KernelGraph g;
    std::unordered_map<std::size_t, NodeData> nd;
    auto built = build_block(g, nd, x, N, Cin, Cout, H, W, hidden, /*stride*/2);

    REQUIRE(g.num_nodes() == built.nodes);   // 8: no add
    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, 16);

    REQUIRE(result.output.size() == built.oracle.size());
    REQUIRE(result.stats.ops == built.expected_ops);   // 7
    float max_err = 0.0f;
    for (std::size_t i = 0; i < built.oracle.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - built.oracle[i]));
    INFO("downsample max_err=" << max_err);
    REQUIRE(max_err < 1e-3f);
}

TEST_CASE("M3 depthwise channel multiplier (Cout != Cin) is rejected",
          "[timing][m3][mobilenet]") {
    // groups == Cin also admits a channel-multiplier depthwise (Cout = k*Cin);
    // run_depthwise_conv produces one output channel per input channel, so the
    // bridge must reject k>1 rather than silently mis-size the result.
    const G N = 16, C = 16, H = 8, W = 8;
    auto cc = conv_cfg(N, C, 2 * C, H, W, 3, 1, 1, /*groups*/C);   // Cout = 2*Cin
    auto x = synth(static_cast<std::size_t>(N) * C * H * W, 3, 1.0f);

    KernelGraph g;
    std::unordered_map<std::size_t, NodeData> nd;
    auto cd = g.add_kernel(Kernel::create_conv2d(cc, false, ActivationType::NONE), "dw_mult");
    nd[cd].filter = synth(static_cast<std::size_t>(C) * 9, 4, 0.3f);

    GraphCspExecutor exec;
    REQUIRE_THROWS_AS(exec.run(g, x, nd, /*T*/16), std::runtime_error);
}
