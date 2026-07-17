// ============================================================================
// tests/timing/test_m3_efficientnet_block.cpp
// M3 (#131): an EfficientNet MBConv block with squeeze-and-excitation, expressed
// as a KernelGraph DFG and executed on the CSP value path through
// GraphCspExecutor, validated vs a composed host oracle. Exercises the SE gate
// additions - a sigmoid runner (composed from VE ops) and the channel-broadcast
// multiply (per-channel [N,C] gate x [N,C,H,W] activation) - on top of the
// inverted-residual (expand / depthwise / project / residual) path.
//
//   expand 1x1 -> BN -> ReLU6
//   3x3 depthwise (stride s) -> BN -> ReLU6            = A
//   SE: GAP(A) -> FC_reduce -> ReLU -> FC_expand -> sigmoid = gate[N,hidden]
//       A_scaled = A * gate            (channel broadcast)
//   project 1x1 -> BN
//   + residual  when s == 1 && Cin == Cout
//
// SiLU/swish is approximated by ReLU6 for the M3 subset (per the design). See
// docs/plans/m3_mobilenet_dfg.md.
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
void relu6_ip(std::vector<float>& v) { for (auto& x : v) x = std::min(std::max(0.0f, x), 6.0f); }
float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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

struct BN { std::vector<float> gamma, beta, mean, var; float eps = 1e-3f; };
BN synth_bn(G C, std::uint64_t seed) {
    BN b;
    b.gamma = synth(C, seed + 1, 0.4f); for (auto& g : b.gamma) g += 1.0f;
    b.beta  = synth(C, seed + 2, 0.2f);
    b.mean  = synth(C, seed + 3, 0.3f);
    b.var   = synth(C, seed + 4, 0.2f); for (auto& v : b.var) v = std::abs(v) + 0.5f;
    return b;
}
void set_bn(NodeData& nd, const BN& bn) {
    nd.gamma = bn.gamma; nd.beta = bn.beta; nd.mean = bn.mean; nd.var = bn.var; nd.eps = bn.eps;
}

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
std::vector<float> dw_conv_bn(const std::vector<float>& x, const std::vector<float>& filter,
                              const sw::kpu::Conv2DConfig& cc, const BN& bn, bool relu6) {
    const auto g = geom_of(cc);
    const G Hout = g.H_out(), Wout = g.W_out();
    std::vector<float> y(static_cast<std::size_t>(g.N) * g.C_out * Hout * Wout);
    for (G n = 0; n < g.N; ++n)
        for (G c = 0; c < g.C_out; ++c) {
            const std::size_t plane = (static_cast<std::size_t>(n) * g.C_in + c) * g.H_in * g.W_in;
            const float scale = bn.gamma[c] / std::sqrt(bn.var[c] + bn.eps);
            const float shift = bn.beta[c] - bn.mean[c] * scale;
            for (G ho = 0; ho < Hout; ++ho)
                for (G wo = 0; wo < Wout; ++wo) {
                    float acc = 0.0f;
                    for (G kh = 0; kh < g.Kh; ++kh) {
                        const long ih = static_cast<long>(ho * g.stride_h + kh) - static_cast<long>(g.pad_h);
                        if (ih < 0 || ih >= static_cast<long>(g.H_in)) continue;
                        for (G kw = 0; kw < g.Kw; ++kw) {
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

// Host matmul C[M,N] = A[M,K] @ W[K,N] + bias, optional ReLU.
std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& w,
                          const std::vector<float>& bias, G M, G Kd, G Nd, bool relu) {
    std::vector<float> c(static_cast<std::size_t>(M) * Nd);
    for (G m = 0; m < M; ++m)
        for (G n = 0; n < Nd; ++n) {
            float acc = bias.empty() ? 0.0f : bias[n];
            for (G k = 0; k < Kd; ++k) acc += a[m * Kd + k] * w[k * Nd + n];
            c[m * Nd + n] = relu ? std::max(0.0f, acc) : acc;
        }
    return c;
}

float maxerr(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.0f;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

} // namespace

TEST_CASE("M3 EfficientNet MBConv+SE block on the CSP executor matches host oracle",
          "[timing][m3][efficientnet]") {
    // Tile-aligned (T=16): N=16, Cin=Cout=16, hidden=32 (t=2), SE reduce = 16.
    const G N = 16, C = 16, H = 4, W = 4, hidden = 32, se = 16;
    auto x = synth(static_cast<std::size_t>(N) * C * H * W, 1, 1.0f);

    auto cce = conv_cfg(N, C, hidden, H, W, 1, 1, 0, 1);            // expand 1x1
    auto ccd = conv_cfg(N, hidden, hidden, H, W, 3, 1, 1, hidden);  // depthwise 3x3 s1
    auto ccp = conv_cfg(N, hidden, C, H, W, 1, 1, 0, 1);            // project 1x1
    auto we = synth(static_cast<std::size_t>(hidden) * C, 10, 0.15f);
    auto wd = synth(static_cast<std::size_t>(hidden) * 9, 20, 0.30f);
    auto wp = synth(static_cast<std::size_t>(C) * hidden, 30, 0.15f);
    BN bne = synth_bn(hidden, 100), bnd = synth_bn(hidden, 200), bnp = synth_bn(C, 300);
    auto wr = synth(static_cast<std::size_t>(hidden) * se, 40, 0.15f);   // FC_reduce [hidden, se]
    auto br = synth(se, 41, 0.1f);
    auto wx = synth(static_cast<std::size_t>(se) * hidden, 42, 0.15f);   // FC_expand [se, hidden]
    auto bx = synth(hidden, 43, 0.1f);

    // --- host oracle ---------------------------------------------------------
    auto e = pw_conv_bn(x, we, cce, bne, /*relu6*/true);
    auto d = dw_conv_bn(e, wd, ccd, bnd, /*relu6*/true);            // A [N,hidden,H,W]
    // SE squeeze: mean over the H*W plane per (n,c).
    std::vector<float> sq(static_cast<std::size_t>(N) * hidden, 0.0f);
    for (G n = 0; n < N; ++n)
        for (G c = 0; c < hidden; ++c) {
            float s = 0.0f;
            for (G p = 0; p < H * W; ++p) s += d[(static_cast<std::size_t>(n) * hidden + c) * H * W + p];
            sq[n * hidden + c] = s / static_cast<float>(H * W);
        }
    auto red  = matmul(sq, wr, br, N, hidden, se, /*relu*/true);    // [N, se]
    auto ex   = matmul(red, wx, bx, N, se, hidden, /*relu*/false);  // [N, hidden]
    std::vector<float> gate(ex.size());
    for (std::size_t i = 0; i < ex.size(); ++i) gate[i] = sigmoidf(ex[i]);
    std::vector<float> d_scaled(d.size());                          // A * gate (channel broadcast)
    for (G n = 0; n < N; ++n)
        for (G c = 0; c < hidden; ++c)
            for (G p = 0; p < H * W; ++p) {
                const std::size_t idx = (static_cast<std::size_t>(n) * hidden + c) * H * W + p;
                d_scaled[idx] = d[idx] * gate[n * hidden + c];
            }
    auto p = pw_conv_bn(d_scaled, wp, ccp, bnp, /*relu6*/false);
    std::vector<float> oref = p;
    for (std::size_t i = 0; i < oref.size(); ++i) oref[i] += x[i];   // residual

    // --- DFG -----------------------------------------------------------------
    KernelGraph g;
    std::unordered_map<std::size_t, NodeData> nd;
    auto ce = g.add_kernel(Kernel::create_conv2d(cce, false, ActivationType::NONE), "expand");
    auto be = g.add_kernel(Kernel::create_batchnorm(N, hidden, H, W, bne.eps), "bn_e");
    auto re = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * H * W}), "relu6_e");
    auto cd = g.add_kernel(Kernel::create_conv2d(ccd, false, ActivationType::NONE), "depthwise");
    auto bd = g.add_kernel(Kernel::create_batchnorm(N, hidden, H, W, bnd.eps), "bn_d");
    auto rd = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::RELU6, {hidden * H * W}), "relu6_d");
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, hidden, H, W), "se_gap");
    auto fr = g.add_kernel(Kernel::create_matmul(N, se, hidden), "se_reduce");
    auto fx = g.add_kernel(Kernel::create_matmul(N, hidden, se), "se_expand");
    auto sg = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::SIGMOID, {hidden}), "se_sigmoid");
    auto sc = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::MUL, {hidden * H * W}), "se_scale");
    auto cp = g.add_kernel(Kernel::create_conv2d(ccp, false, ActivationType::NONE), "project");
    auto bp = g.add_kernel(Kernel::create_batchnorm(N, C, H, W, bnp.eps), "bn_p");
    auto ad = g.add_kernel(Kernel::create_elementwise(ElementwiseOp::ADD, {C * H * W}), "add");

    g.add_edge(ce, be); g.add_edge(be, re); g.add_edge(re, cd);
    g.add_edge(cd, bd); g.add_edge(bd, rd);
    g.add_edge(rd, gp); g.add_edge(gp, fr); g.add_edge(fr, fx); g.add_edge(fx, sg);
    g.add_edge(rd, sc);                    // SE scale main branch = depthwise activation A
    g.add_edge(sg, sc, "C", "B");          // SE scale gate branch
    g.add_edge(sc, cp); g.add_edge(cp, bp);
    // Residual: single block, so the identity skip is the block input (== the
    // external network input x) - one edge + the bridge's external-input
    // fallback supplies x as the ADD's second operand.
    g.add_edge(bp, ad);

    nd[ce].filter = we; set_bn(nd[be], bne);
    nd[cd].filter = wd; set_bn(nd[bd], bnd);
    nd[cp].filter = wp; set_bn(nd[bp], bnp);
    nd[fr].fc_weight = wr; nd[fr].fc_bias = br; nd[fr].fc_M = N; nd[fr].fc_K = hidden; nd[fr].fc_N = se; nd[fr].fc_relu = true;
    nd[fx].fc_weight = wx; nd[fx].fc_bias = bx; nd[fx].fc_M = N; nd[fx].fc_K = se; nd[fx].fc_N = hidden; nd[fx].fc_relu = false;

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, /*T*/16);

    REQUIRE(result.output.size() == oref.size());
    INFO("max_err=" << maxerr(result.output, oref) << " ops=" << result.stats.ops);
    REQUIRE(maxerr(result.output, oref) < 2e-3f);
}
