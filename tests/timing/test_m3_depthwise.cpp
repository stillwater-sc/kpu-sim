// ============================================================================
// tests/timing/test_m3_depthwise.cpp
// M3-T2 (#209): depthwise convolution on the CSP executor via the pooling-window
// unfold + a per-channel filter dot-product reduce, validated vs a host
// depthwise reference (with and without folded BN + ReLU6). Plus run_relu6.
// See docs/plans/m3_mobilenet_dfg.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/graph/csp_op_runners.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace sw::kpu::timing::graph;
using sw::kpu::timing::schedule::Pool2DGeometry;
using sw::kpu::timing::schedule::BatchNormAffine;
using sw::kpu::timing::schedule::bn_fold;
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

// Host depthwise conv reference: per-channel 2D conv, optional BN + ReLU6. NCHW.
std::vector<float> depthwise_reference(const std::vector<float>& x,
                                       const std::vector<float>& filter,
                                       const Pool2DGeometry& g,
                                       const std::vector<float>* gamma,
                                       const std::vector<float>* beta,
                                       const std::vector<float>* mean,
                                       const std::vector<float>* var, float eps,
                                       bool relu6) {
    const G Hout = g.H_out(), Wout = g.W_out();
    std::vector<float> y(static_cast<std::size_t>(g.N) * g.C * Hout * Wout);
    for (G n = 0; n < g.N; ++n)
        for (G c = 0; c < g.C; ++c) {
            const std::size_t plane = (static_cast<std::size_t>(n) * g.C + c) * g.H * g.W;
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
                    if (gamma) acc = (*gamma)[c] * (acc - (*mean)[c]) / std::sqrt((*var)[c] + eps) + (*beta)[c];
                    if (relu6) acc = std::min(std::max(0.0f, acc), 6.0f);
                    y[((static_cast<std::size_t>(n) * g.C + c) * Hout + ho) * Wout + wo] = acc;
                }
        }
    return y;
}

Pool2DGeometry geom(G N, G C, G H, G W, G K, G s, G p) {
    Pool2DGeometry g; g.N = N; g.C = C; g.H = H; g.W = W;
    g.Kh = g.Kw = K; g.stride_h = g.stride_w = s; g.pad_h = g.pad_w = p;
    return g;
}

float maxerr(const std::vector<float>& a, const std::vector<float>& b) {
    float e = 0.0f;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

} // namespace

TEST_CASE("M3 depthwise conv on the CSP executor matches host reference",
          "[timing][m3][depthwise]") {
    // batch 16 keeps M = N*Hout*Wout tile-aligned; C independent per channel.
    RunStats st;

    SECTION("3x3 stride1 pad1") {
        auto g = geom(16, 16, 4, 4, 3, 1, 1);
        auto x = synth(g.elems(), 1, 1.0f);
        auto f = synth(static_cast<std::size_t>(g.C) * g.window(), 2, 0.5f);
        auto csp = run_depthwise_conv(x, g, f, nullptr, {}, false, 16, st);
        auto ref = depthwise_reference(x, f, g, nullptr, nullptr, nullptr, nullptr, 0.0f, false);
        REQUIRE(csp.size() == ref.size());
        REQUIRE(maxerr(csp, ref) < 1e-4f);
    }
    SECTION("3x3 stride2 pad1 (downsample)") {
        auto g = geom(16, 16, 8, 8, 3, 2, 1);
        auto x = synth(g.elems(), 3, 1.0f);
        auto f = synth(static_cast<std::size_t>(g.C) * g.window(), 4, 0.5f);
        auto csp = run_depthwise_conv(x, g, f, nullptr, {}, false, 16, st);
        auto ref = depthwise_reference(x, f, g, nullptr, nullptr, nullptr, nullptr, 0.0f, false);
        REQUIRE(maxerr(csp, ref) < 1e-4f);
    }
    SECTION("C=48 exceeds the old 256 compute-result cap (#210)") {
        // 48 channels * (N*Hout*Wout / 16) output tiles = 48*16 = 768 drains,
        // past the old hardcoded 256-entry compute-result CAM that deadlocked
        // depthwise/pooling at high channel counts.
        auto g = geom(16, 48, 4, 4, 3, 1, 1);
        auto x = synth(g.elems(), 11, 1.0f);
        auto f = synth(static_cast<std::size_t>(g.C) * g.window(), 12, 0.5f);
        auto csp = run_depthwise_conv(x, g, f, nullptr, {}, false, 16, st);
        auto ref = depthwise_reference(x, f, g, nullptr, nullptr, nullptr, nullptr, 0.0f, false);
        REQUIRE(csp.size() == ref.size());
        REQUIRE(maxerr(csp, ref) < 1e-4f);
    }
    SECTION("with folded BN + ReLU6") {
        auto g = geom(16, 16, 4, 4, 3, 1, 1);
        auto x = synth(g.elems(), 5, 1.0f);
        auto f = synth(static_cast<std::size_t>(g.C) * g.window(), 6, 0.5f);
        auto gamma = synth(g.C, 7, 0.3f); for (auto& v : gamma) v += 1.0f;
        auto beta  = synth(g.C, 8, 0.3f);
        auto mean  = synth(g.C, 9, 0.3f);
        auto var   = synth(g.C, 10, 0.2f); for (auto& v : var) v = std::abs(v) + 0.5f;
        const float eps = 1e-3f;
        auto affine = bn_fold(gamma, beta, mean, var, eps);
        auto csp = run_depthwise_conv(x, g, f, &affine, {}, /*relu6*/true, 16, st);
        auto ref = depthwise_reference(x, f, g, &gamma, &beta, &mean, &var, eps, true);
        REQUIRE(maxerr(csp, ref) < 1e-3f);
    }
}

TEST_CASE("M3 run_relu6 clamps to [0,6] on the CSP executor",
          "[timing][m3][depthwise]") {
    RunStats st;
    std::vector<float> x(256);
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = -8.0f + 0.1f * static_cast<float>(i);   // spans below 0 and above 6
    auto csp = run_relu6(x, st);
    REQUIRE(csp.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        REQUIRE_THAT(csp[i], Catch::Matchers::WithinAbs(std::min(std::max(0.0f, x[i]), 6.0f), 1e-5));
}
