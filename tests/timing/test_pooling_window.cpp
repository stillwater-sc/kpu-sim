// ============================================================================
// tests/timing/test_pooling_window.cpp
// Unit tests for the pooling window-patchify helper (E7-T2, #192).
//
// Verifies (1) geometry, (2) that the per-channel window unfold + reduce
// reproduces the direct pooling reference for MAX and AVG, (3) hand-computed
// tiny cases, (4) padding semantics (-inf fill for MAX, count-excludes-padding
// for AVG), (5) global average pool, and (6) the public guards.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/schedule/pooling_window.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

std::vector<float> fill(std::size_t n, int period, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = base + step * static_cast<float>(i % static_cast<std::size_t>(period));
    return v;
}

// Reconstruct the pooled output via window unfold + reduce, compare to the
// direct reference; return the max abs error.
double check_via_window(const Pool2DGeometry& g, PoolType type) {
    auto input = fill(g.elems(), 7, -1.5f, 0.5f);
    const auto ref = pool2d_reference(input, g, type);
    const Size Hout = g.H_out(), Wout = g.W_out(), K = g.window(), M = Hout * Wout;

    double max_err = 0.0;
    for (Size n = 0; n < g.N; ++n)
        for (Size c = 0; c < g.C; ++c) {
            const auto win = pool_window_channel(input, g, n, c, type);
            REQUIRE(win.rows.size() == static_cast<std::size_t>(M) * K);
            REQUIRE(win.counts.size() == M);
            for (Size m = 0; m < M; ++m) {
                const float got = reduce_window_row(
                    &win.rows[static_cast<std::size_t>(m) * K], K, win.counts[m], type);
                const float want =
                    ref[((static_cast<std::size_t>(n) * g.C + c) * Hout) * Wout + m];
                max_err = std::max(max_err, static_cast<double>(std::abs(got - want)));
            }
        }
    return max_err;
}

Pool2DGeometry base() {
    Pool2DGeometry g;
    g.N = 2; g.C = 3; g.H = 8; g.W = 8; g.Kh = 2; g.Kw = 2; g.stride_h = g.stride_w = 2;
    return g;
}

} // namespace

TEST_CASE("pooling geometry: floored output extents", "[pooling][window]") {
    Pool2DGeometry g; g.H = 7; g.W = 7; g.Kh = 3; g.Kw = 3; g.stride_h = g.stride_w = 2;
    REQUIRE(g.H_out() == 3);  // floor((7-3)/2)+1
    REQUIRE(g.W_out() == 3);
    g.pad_h = g.pad_w = 1;
    REQUIRE(g.H_out() == 4);  // floor((7+2-3)/2)+1
    REQUIRE(g.window() == 9);
    REQUIRE(g.valid());
    g.Kh = 20; g.pad_h = 0;
    REQUIRE(g.H_out() == 0);
    REQUIRE_FALSE(g.valid());
}

TEST_CASE("pooling window+reduce matches direct reference", "[pooling][window]") {
    SECTION("2x2 max, stride 2") { REQUIRE(check_via_window(base(), PoolType::MAX) < 1e-6); }
    SECTION("2x2 avg, stride 2") { REQUIRE(check_via_window(base(), PoolType::AVG) < 1e-6); }
    SECTION("3x3 max, stride 2, pad 1") {
        auto g = base(); g.Kh = g.Kw = 3; g.pad_h = g.pad_w = 1;
        REQUIRE(check_via_window(g, PoolType::MAX) < 1e-6);
    }
    SECTION("3x3 avg, stride 2, pad 1 (count excludes padding)") {
        auto g = base(); g.Kh = g.Kw = 3; g.pad_h = g.pad_w = 1;
        REQUIRE(check_via_window(g, PoolType::AVG) < 1e-6);
    }
    SECTION("non-square window 2x3, stride 1") {
        auto g = base(); g.Kh = 2; g.Kw = 3; g.stride_h = g.stride_w = 1;
        REQUIRE(check_via_window(g, PoolType::MAX) < 1e-6);
    }
}

TEST_CASE("pooling hand-computed tiny cases", "[pooling][window]") {
    // 1x1x2x2 input [[1,2],[3,4]], 2x2 window stride 2 -> single output.
    Pool2DGeometry g; g.N = 1; g.C = 1; g.H = 2; g.W = 2; g.Kh = 2; g.Kw = 2;
    g.stride_h = g.stride_w = 2;
    std::vector<float> x = {1, 2, 3, 4};

    REQUIRE(pool2d_reference(x, g, PoolType::MAX) == std::vector<float>{4.0f});
    REQUIRE(pool2d_reference(x, g, PoolType::AVG) == std::vector<float>{2.5f});

    auto win = pool_window_channel(x, g, 0, 0, PoolType::MAX);
    REQUIRE(win.counts == std::vector<Size>{4});
    REQUIRE(reduce_window_row(win.rows.data(), 4, win.counts[0], PoolType::MAX) == 4.0f);
}

TEST_CASE("pooling padding: -inf for max, excluded count for avg", "[pooling][window]") {
    // 1x1x3x3 input [1..9], 2x2 window, pad 1, stride 1. The top-left output
    // (ho=0,wo=0) sees ih,iw in {-1,0} -> only x[0,0]=1 is valid; the other 3
    // taps are padding.
    Pool2DGeometry g; g.N = 1; g.C = 1; g.H = 3; g.W = 3; g.Kh = 2; g.Kw = 2;
    g.pad_h = g.pad_w = 1; g.stride_h = g.stride_w = 1;
    std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto win = pool_window_channel(x, g, 0, 0, PoolType::MAX);
    REQUIRE(win.counts[0] == 1);  // corner sees a single valid tap
    Size neg_inf_taps = 0;
    for (Size k = 0; k < g.window(); ++k)
        if (win.rows[k] == -std::numeric_limits<float>::infinity()) ++neg_inf_taps;
    REQUIRE(neg_inf_taps == g.window() - win.counts[0]);  // 3 padded taps are -inf
    REQUIRE(reduce_window_row(win.rows.data(), g.window(), win.counts[0], PoolType::MAX)
            == 1.0f);  // max over {1, -inf, -inf, -inf} = 1

    // AVG at the corner: mean over the valid taps only (count excludes padding).
    auto wavg = pool_window_channel(x, g, 0, 0, PoolType::AVG);
    const float avg0 = reduce_window_row(wavg.rows.data(), g.window(), wavg.counts[0],
                                         PoolType::AVG);
    REQUIRE_THAT(avg0, Catch::Matchers::WithinAbs(1.0f, 1e-6));  // only x[0,0]=1 valid
    // count-includes-padding would give 1/4; excludes-padding gives 1/1.
}

TEST_CASE("global average pool matches reference", "[pooling][window]") {
    Pool2DGeometry g; g.N = 2; g.C = 4; g.H = 4; g.W = 4;
    auto input = fill(g.elems(), 5, 0.0f, 1.0f);
    auto gap = global_avg_pool_reference(input, g);
    REQUIRE(gap.size() == static_cast<std::size_t>(g.N) * g.C);
    // Independently compute channel means.
    const std::size_t plane = static_cast<std::size_t>(g.H) * g.W;
    for (Size n = 0; n < g.N; ++n)
        for (Size c = 0; c < g.C; ++c) {
            const std::size_t b = (static_cast<std::size_t>(n) * g.C + c) * plane;
            float s = 0.0f;
            for (std::size_t i = 0; i < plane; ++i) s += input[b + i];
            REQUIRE_THAT(gap[static_cast<std::size_t>(n) * g.C + c],
                Catch::Matchers::WithinAbs(s / static_cast<float>(plane), 1e-5));
        }
}

TEST_CASE("pooling public guards reject malformed input", "[pooling][window]") {
    Pool2DGeometry g = base();
    auto input = fill(g.elems(), 5, 0.0f, 1.0f);
    SECTION("invalid geometry") {
        Pool2DGeometry bad = g; bad.Kh = 99; bad.pad_h = 0;
        REQUIRE_FALSE(bad.valid());
        REQUIRE_THROWS_AS(pool2d_reference(std::vector<float>(bad.elems()), bad, PoolType::MAX),
                          std::invalid_argument);
    }
    SECTION("input size mismatch") {
        REQUIRE_THROWS_AS(pool2d_reference(std::vector<float>(input.size() + 1), g, PoolType::AVG),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(pool_window_channel(std::vector<float>(input.size() - 1), g, 0, 0, PoolType::MAX),
                          std::invalid_argument);
    }
    SECTION("channel/batch out of range") {
        REQUIRE_THROWS_AS(pool_window_channel(input, g, g.N, 0, PoolType::MAX),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(pool_window_channel(input, g, 0, g.C, PoolType::MAX),
                          std::invalid_argument);
    }
}
