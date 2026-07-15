// ============================================================================
// tests/timing/test_batchnorm_affine.cpp
// Unit tests for the BatchNorm-inference scale/shift fold helper (E9-T2, #179).
//
// BatchNorm inference folds to y = x*scale[c] + shift[c] (see
// docs/plans/e9_batchnorm_pattern.md). These tests verify (1) the fold math,
// (2) that the folded apply reproduces the independent 4-param direct reference,
// (3) a hand-computed tiny case, and (4) the public guards.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::timing::schedule;
using sw::kpu::Size;

namespace {

std::vector<float> fill(std::size_t n, int period, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = base + step * static_cast<float>(i % static_cast<std::size_t>(period));
    return v;
}

} // namespace

TEST_CASE("bn_fold: scale/shift math", "[batchnorm][affine]") {
    // Single channel, hand-checked. gamma=2, beta=1, mean=3, var=3, eps=1.
    // scale = 2/sqrt(4) = 1 ; shift = 1 - 3*1 = -2.
    auto a = bn_fold({2.0f}, {1.0f}, {3.0f}, {3.0f}, 1.0f);
    REQUIRE(a.scale.size() == 1);
    REQUIRE(a.shift.size() == 1);
    REQUIRE_THAT(a.scale[0], Catch::Matchers::WithinAbs(1.0f, 1e-6));
    REQUIRE_THAT(a.shift[0], Catch::Matchers::WithinAbs(-2.0f, 1e-6));
}

TEST_CASE("batchnorm apply == direct 4-param reference", "[batchnorm][affine]") {
    BatchNormGeometry g;
    g.N = 2; g.C = 4; g.H = 3; g.W = 5;
    const float eps = 1e-3f;

    auto input = fill(g.elems(), 7, -1.0f, 0.5f);
    auto gamma = fill(g.C, 4, 0.5f, 0.5f);
    auto beta  = fill(g.C, 3, -1.0f, 0.75f);
    auto mean  = fill(g.C, 5, 0.25f, 0.5f);
    auto var   = fill(g.C, 4, 0.5f, 0.25f);  // all > 0

    const auto affine = bn_fold(gamma, beta, mean, var, eps);
    const auto folded = batchnorm_apply(input, affine, g);
    const auto ref = batchnorm_reference(input, gamma, beta, mean, var, eps, g);

    REQUIRE(folded.size() == ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
        REQUIRE_THAT(folded[i], Catch::Matchers::WithinAbs(ref[i], 1e-5));
}

TEST_CASE("batchnorm hand-computed tiny case", "[batchnorm][affine]") {
    // 1x1x1x3 input [1,2,3]; gamma=2, beta=1, mean=1, var=3, eps=1.
    // scale=2/2=1, shift=1-1*1=0 -> y = x*1 + 0 = [1,2,3].
    BatchNormGeometry g; g.N = 1; g.C = 1; g.H = 1; g.W = 3;
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    auto a = bn_fold({2.0f}, {1.0f}, {1.0f}, {3.0f}, 1.0f);
    auto y = batchnorm_apply(x, a, g);
    REQUIRE(y == std::vector<float>{1.0f, 2.0f, 3.0f});

    auto ref = batchnorm_reference(x, {2.0f}, {1.0f}, {1.0f}, {3.0f}, 1.0f, g);
    REQUIRE(ref == y);
}

TEST_CASE("batchnorm per-channel affine is applied per channel", "[batchnorm][affine]") {
    // 2 channels, distinct scale/shift; a constant input isolates the channel map.
    BatchNormGeometry g; g.N = 1; g.C = 2; g.H = 1; g.W = 2;
    std::vector<float> x = {10.0f, 10.0f,   // channel 0
                            10.0f, 10.0f};  // channel 1
    BatchNormAffine a; a.scale = {2.0f, 3.0f}; a.shift = {1.0f, -1.0f};
    auto y = batchnorm_apply(x, a, g);
    // channel 0: 10*2+1 = 21 ; channel 1: 10*3-1 = 29
    REQUIRE(y == std::vector<float>{21.0f, 21.0f, 29.0f, 29.0f});
}

TEST_CASE("batchnorm public guards reject malformed input", "[batchnorm][affine]") {
    BatchNormGeometry g; g.N = 1; g.C = 3; g.H = 2; g.W = 2;
    auto input = fill(g.elems(), 5, 0.0f, 1.0f);
    std::vector<float> p3 = {1.0f, 1.0f, 1.0f};

    SECTION("bn_fold size mismatch") {
        REQUIRE_THROWS_AS(bn_fold({1.0f, 1.0f}, p3, p3, p3, 1e-3f), std::invalid_argument);
    }
    SECTION("bn_fold non-positive eps") {
        REQUIRE_THROWS_AS(bn_fold(p3, p3, p3, p3, 0.0f), std::invalid_argument);
    }
    SECTION("bn_fold var + eps <= 0") {
        REQUIRE_THROWS_AS(bn_fold(p3, p3, p3, {-2.0f, 0.0f, 0.0f}, 1e-3f),
                          std::invalid_argument);
    }
    SECTION("apply geometry/size mismatch") {
        BatchNormAffine a; a.scale = p3; a.shift = p3;
        REQUIRE_THROWS_AS(batchnorm_apply(std::vector<float>(input.size() + 1), a, g),
                          std::invalid_argument);
        BatchNormAffine bad; bad.scale = {1.0f}; bad.shift = {1.0f};
        REQUIRE_THROWS_AS(batchnorm_apply(input, bad, g), std::invalid_argument);
    }
    SECTION("reference param/size guards") {
        REQUIRE_THROWS_AS(
            batchnorm_reference(input, {1.0f}, p3, p3, p3, 1e-3f, g),
            std::invalid_argument);
        REQUIRE_THROWS_AS(
            batchnorm_reference(std::vector<float>(input.size() + 1), p3, p3, p3, p3, 1e-3f, g),
            std::invalid_argument);
    }
}
