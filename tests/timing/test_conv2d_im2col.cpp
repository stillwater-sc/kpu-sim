// ============================================================================
// tests/timing/test_conv2d_im2col.cpp
// Unit tests for the conv2d im2col patchify helper (E6-T2, issue #120).
//
// The helper lowers conv2d to a single GEMM C_out = A_col @ B_w. These tests
// verify (1) the conv geometry (floored Hout/Wout, GEMM M/K/N), (2) that
// im2col writes explicit zeros in the padding region, (3) a hand-computed tiny
// case (guards against a convention bug shared by both helper functions), and
// (4) that A_col @ B_w reproduces the independent direct-conv reference across a
// stride/padding/non-square/batch/bias/ReLU matrix.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using namespace sw::kpu::timing::schedule;
using sw::kpu::Size;

namespace {

// C_out = A_col @ B_w, a plain reference GEMM: A_col is [M, K], B_w is [K, Ncols].
std::vector<float> gemm(const std::vector<float>& a, const std::vector<float>& b,
                        Size M, Size K, Size Ncols) {
    std::vector<float> c(static_cast<std::size_t>(M) * Ncols, 0.0f);
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k) {
            const float av = a[static_cast<std::size_t>(i) * K + k];
            for (Size j = 0; j < Ncols; ++j)
                c[static_cast<std::size_t>(i) * Ncols + j] +=
                    av * b[static_cast<std::size_t>(k) * Ncols + j];
        }
    return c;
}

// Deterministic non-trivial fill.
std::vector<float> ramp(std::size_t n, float start = 1.0f, float step = 1.0f) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = start + step * static_cast<float>(i);
    return v;
}

// Check that the im2col lowering reproduces the direct-conv reference.
void check_lowering(const Conv2DGeometry& g, const std::vector<float>& bias,
                    bool relu) {
    auto input = ramp(g.input_elems(), 1.0f, 0.5f);
    auto filter = ramp(g.filter_elems(), -0.5f, 0.25f);

    const auto a_col = im2col_nchw(input, g);
    const auto b_w = filter_to_bw_nchw(filter, g);
    REQUIRE(a_col.size() == static_cast<std::size_t>(g.M()) * g.K());
    REQUIRE(b_w.size() == static_cast<std::size_t>(g.K()) * g.C_out);

    auto c = gemm(a_col, b_w, g.M(), g.K(), g.Ncols());
    // Apply the epilogue the executor's MatMulComputeSpec would apply.
    for (Size i = 0; i < g.M(); ++i)
        for (Size co = 0; co < g.C_out; ++co) {
            float& v = c[static_cast<std::size_t>(i) * g.C_out + co];
            if (!bias.empty()) v += bias[co];
            if (relu && v < 0.0f) v = 0.0f;
        }

    const auto ref = conv2d_reference(input, filter, bias, g, relu);

    // Map GEMM [m, co] (m = (n*Hout+ho)*Wout+wo) to NCHW y[n, co, ho, wo].
    const Size Hout = g.H_out(), Wout = g.W_out();
    for (Size n = 0; n < g.N; ++n)
        for (Size co = 0; co < g.C_out; ++co)
            for (Size ho = 0; ho < Hout; ++ho)
                for (Size wo = 0; wo < Wout; ++wo) {
                    const Size m = (n * Hout + ho) * Wout + wo;
                    const std::size_t gi = static_cast<std::size_t>(m) * g.C_out + co;
                    const std::size_t yi =
                        ((static_cast<std::size_t>(n) * g.C_out + co) * Hout + ho) *
                            Wout + wo;
                    REQUIRE_THAT(c[gi],
                        Catch::Matchers::WithinAbs(ref[yi], 1e-4));
                }
}

} // namespace

TEST_CASE("conv2d geometry: floored output extents and GEMM dims", "[conv2d][im2col]") {
    // (H+2p-Kh)/s + 1 with an explicit floor.
    Conv2DGeometry g;
    g.N = 2; g.C_in = 3; g.H_in = 5; g.W_in = 7;
    g.C_out = 4; g.Kh = 3; g.Kw = 3;

    SECTION("no padding, unit stride") {
        REQUIRE(g.H_out() == 3);   // (5-3)/1+1
        REQUIRE(g.W_out() == 5);   // (7-3)/1+1
        REQUIRE(g.M() == 2 * 3 * 5);
        REQUIRE(g.K() == 3 * 3 * 3);
        REQUIRE(g.Ncols() == 4);
        REQUIRE(g.valid());
    }
    SECTION("stride 2 requires a floor") {
        g.stride_h = 2; g.stride_w = 2;
        REQUIRE(g.H_out() == 2);   // floor((5-3)/2)+1 = 2
        REQUIRE(g.W_out() == 3);   // floor((7-3)/2)+1 = 3
    }
    SECTION("same padding") {
        g.pad_h = 1; g.pad_w = 1;
        REQUIRE(g.H_out() == 5);   // (5+2-3)/1+1
        REQUIRE(g.W_out() == 7);
    }
    SECTION("kernel larger than padded input is invalid, not a wrap") {
        g.H_in = 2; g.Kh = 3; g.pad_h = 0;
        REQUIRE(g.H_out() == 0);
        REQUIRE_FALSE(g.valid());
    }
}

TEST_CASE("conv2d im2col: public guards reject malformed input", "[conv2d][im2col]") {
    Conv2DGeometry g;
    g.C_in = 2; g.H_in = 4; g.W_in = 4; g.C_out = 3; g.Kh = 3; g.Kw = 3;
    REQUIRE(g.valid());

    const auto input = ramp(g.input_elems());
    const auto filter = ramp(g.filter_elems());

    SECTION("invalid geometry throws") {
        Conv2DGeometry bad = g;
        bad.Kh = 9;  // kernel larger than padded input -> H_out() == 0
        REQUIRE_FALSE(bad.valid());
        REQUIRE_THROWS_AS(im2col_nchw(std::vector<float>(bad.input_elems()), bad),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(filter_to_bw_nchw(std::vector<float>(bad.filter_elems()), bad),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(conv2d_reference(std::vector<float>(bad.input_elems()),
                                           std::vector<float>(bad.filter_elems()), {}, bad),
                          std::invalid_argument);
    }
    SECTION("input size mismatch throws") {
        REQUIRE_THROWS_AS(im2col_nchw(std::vector<float>(input.size() + 1), g),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(conv2d_reference(std::vector<float>(input.size() + 1), filter, {}, g),
                          std::invalid_argument);
    }
    SECTION("filter size mismatch throws") {
        REQUIRE_THROWS_AS(filter_to_bw_nchw(std::vector<float>(filter.size() - 1), g),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(conv2d_reference(input, std::vector<float>(filter.size() - 1), {}, g),
                          std::invalid_argument);
    }
    SECTION("bias must be C_out or empty") {
        REQUIRE_THROWS_AS(conv2d_reference(input, filter, std::vector<float>(g.C_out + 1), g),
                          std::invalid_argument);
        REQUIRE_NOTHROW(conv2d_reference(input, filter, {}, g));
        REQUIRE_NOTHROW(conv2d_reference(input, filter, std::vector<float>(g.C_out), g));
    }
}

TEST_CASE("conv2d im2col: padding produces explicit zeros", "[conv2d][im2col]") {
    // 1 batch, 1 channel, 3x3 input, 3x3 kernel, pad 1 -> 3x3 output. The
    // top-left output position's receptive field has its top row and left
    // column in the padding region, which must be exactly 0.
    Conv2DGeometry g;
    g.C_in = 1; g.H_in = 3; g.W_in = 3; g.C_out = 1;
    g.Kh = 3; g.Kw = 3; g.pad_h = 1; g.pad_w = 1;

    auto input = ramp(g.input_elems());  // 1..9
    auto a_col = im2col_nchw(input, g);
    const Size K = g.K();  // 9

    // Row m = 0 is output (ho=0, wo=0). Column k = (kh*3 + kw).
    // Padded taps: kh==0 (top row) or kw==0 (left col).
    for (Size kh = 0; kh < 3; ++kh)
        for (Size kw = 0; kw < 3; ++kw) {
            const Size k = kh * 3 + kw;
            const float v = a_col[k];  // m=0
            if (kh == 0 || kw == 0) {
                REQUIRE(v == 0.0f);           // padding
            } else {
                // interior tap maps to input[(kh-1)*3 + (kw-1)]
                REQUIRE(v == input[(kh - 1) * 3 + (kw - 1)]);
            }
        }
    REQUIRE(a_col.size() == static_cast<std::size_t>(g.M()) * K);
}

TEST_CASE("conv2d im2col: hand-computed tiny case", "[conv2d][im2col]") {
    // 1x1x3x3 input, one 2x2 kernel, stride 1, no pad -> 1x1x2x2 output.
    // input =            filter =
    //   1 2 3              1 0
    //   4 5 6              0 1
    //   7 8 9
    // y[0,0] = 1*1 + 5*1 = 6 ; y[0,1] = 2 + 6 = 8
    // y[1,0] = 4 + 8 = 12     ; y[1,1] = 5 + 9 = 14
    Conv2DGeometry g;
    g.C_in = 1; g.H_in = 3; g.W_in = 3; g.C_out = 1; g.Kh = 2; g.Kw = 2;

    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> filter = {1, 0, 0, 1};

    auto a_col = im2col_nchw(input, g);
    auto b_w = filter_to_bw_nchw(filter, g);
    auto c = gemm(a_col, b_w, g.M(), g.K(), g.Ncols());
    REQUIRE(c.size() == 4);
    REQUIRE(c[0] == 6.0f);
    REQUIRE(c[1] == 8.0f);
    REQUIRE(c[2] == 12.0f);
    REQUIRE(c[3] == 14.0f);

    auto ref = conv2d_reference(input, filter, {}, g, false);
    REQUIRE(ref == c);  // both lowerings agree exactly on integers
}

TEST_CASE("conv2d im2col lowering matches direct conv reference", "[conv2d][im2col]") {
    auto base = [] {
        Conv2DGeometry g;
        g.N = 1; g.C_in = 3; g.H_in = 8; g.W_in = 8; g.C_out = 5;
        g.Kh = 3; g.Kw = 3;
        return g;
    };

    SECTION("3x3 stride1 pad1 (ResNet-style)") {
        auto g = base(); g.pad_h = 1; g.pad_w = 1;
        check_lowering(g, {}, false);
    }
    SECTION("1x1 pointwise") {
        auto g = base(); g.Kh = 1; g.Kw = 1;
        check_lowering(g, {}, false);
    }
    SECTION("stride 2, no pad") {
        auto g = base(); g.stride_h = 2; g.stride_w = 2;
        check_lowering(g, {}, false);
    }
    SECTION("non-square kernel 3x5, asymmetric pad") {
        auto g = base(); g.Kh = 3; g.Kw = 5; g.pad_h = 1; g.pad_w = 2;
        check_lowering(g, {}, false);
    }
    SECTION("batch > 1") {
        auto g = base(); g.N = 3; g.pad_h = 1; g.pad_w = 1;
        check_lowering(g, {}, false);
    }
    SECTION("with per-channel bias") {
        auto g = base(); g.pad_h = 1; g.pad_w = 1;
        check_lowering(g, {0.5f, -1.0f, 2.0f, -0.25f, 10.0f}, false);
    }
    SECTION("with bias and ReLU (exercises the negative clamp)") {
        auto g = base(); g.pad_h = 1; g.pad_w = 1;
        check_lowering(g, {-100.0f, -100.0f, -100.0f, -100.0f, -100.0f}, true);
    }
}
