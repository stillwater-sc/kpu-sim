// ============================================================================
// tests/timing/test_functional_softmax.cpp
// Value-producing online softmax on the CSP executor, verified against an
// independent host safe-softmax oracle (issue #157, epic E8).
//
// The oracle is a plain host loop, separate from the executor kernels, so a
// semantic error cannot match itself. Edge cases (all-(-inf) rows, an
// all-(-inf) prefix) are pinned per the T1 design.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_softmax_executor.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

OnlineSoftmaxScheduleGenerator::Config make_config(Size num_rows, Size reduction_elems) {
    OnlineSoftmaxScheduleGenerator::Config c;
    c.num_rows = num_rows;
    c.reduction_elems = reduction_elems;
    c.tile_elems = 256;
    c.in_base = 0x100000;
    c.stat_base = 0x200000;
    c.out_base = 0x300000;
    return c;
}

ConcurrentTimingExecutor::Config make_executor_config() {
    ConcurrentTimingExecutor::Config config;
    config.max_cycles = 2'000'000;
    return config;
}

// Independent host safe-softmax over each row of a num_rows x re tensor
std::vector<float> host_softmax(const std::vector<float>& data, Size rows, Size re) {
    std::vector<float> out(data.size());
    const float ninf = -std::numeric_limits<float>::infinity();
    for (Size r = 0; r < rows; ++r) {
        const Size base = r * re;
        double m = -std::numeric_limits<double>::infinity();
        for (Size i = 0; i < re; ++i) m = std::max(m, static_cast<double>(data[base + i]));
        double l = 0.0;
        if (m > -std::numeric_limits<double>::infinity())
            for (Size i = 0; i < re; ++i) l += std::exp(static_cast<double>(data[base + i]) - m);
        for (Size i = 0; i < re; ++i) {
            out[base + i] = (l > 0.0)
                ? static_cast<float>(std::exp(static_cast<double>(data[base + i]) - m) / l)
                : 1.0f / static_cast<float>(re);
        }
    }
    (void)ninf;
    return out;
}

void require_softmax_match(const std::vector<float>& actual,
                           const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO("element " << i);
        REQUIRE(actual[i] == Catch::Approx(expected[i]).epsilon(1e-5).margin(1e-6));
    }
}

std::vector<float> ramp(Size n, float base, float step) {
    std::vector<float> v(n);
    for (Size i = 0; i < n; ++i) v[i] = base + step * static_cast<float>(i);
    return v;
}

} // namespace

TEST_CASE("Online softmax matches the host oracle (single row)",
          "[timing][functional][softmax]") {
    const Size n = 1024;   // 4 tiles
    auto data = ramp(n, -2.0f, 0.005f);

    FunctionalSoftmaxExecutor exec(make_config(1, n), make_executor_config());
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    REQUIRE_FALSE(result.execution.livelock_detected);

    auto expected = host_softmax(data, 1, n);
    require_softmax_match(result.values, expected);

    // Softmax sums to 1 across the row
    double sum = 0.0; for (float v : result.values) sum += v;
    REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-4));
}

TEST_CASE("Online softmax is numerically stable for large logits",
          "[timing][functional][softmax]") {
    const Size n = 512;
    std::vector<float> data(n);
    for (Size i = 0; i < n; ++i) data[i] = 50.0f + static_cast<float>(i % 13);  // large

    FunctionalSoftmaxExecutor exec(make_config(1, n), make_executor_config());
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    // No inf/NaN despite exp of large logits (max-subtraction)
    for (float v : result.values) REQUIRE(std::isfinite(v));
    require_softmax_match(result.values, host_softmax(data, 1, n));
}

TEST_CASE("Online softmax per-row over a batch",
          "[timing][functional][softmax]") {
    const Size rows = 4, re = 1024;
    std::vector<float> data(rows * re);
    for (Size r = 0; r < rows; ++r)
        for (Size i = 0; i < re; ++i)
            data[r * re + i] = static_cast<float>(r) - 1.0f + 0.003f * static_cast<float>(i);

    FunctionalSoftmaxExecutor exec(make_config(rows, re), make_executor_config());
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    require_softmax_match(result.values, host_softmax(data, rows, re));
}

TEST_CASE("Online softmax non-aligned row",
          "[timing][functional][softmax]") {
    const Size n = 1000;   // 4 tiles, 232 tail
    auto data = ramp(n, -1.0f, 0.007f);
    FunctionalSoftmaxExecutor exec(make_config(1, n), make_executor_config());
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    require_softmax_match(result.values, host_softmax(data, 1, n));
}

TEST_CASE("Online softmax under partitioned credits and restreamed realization",
          "[timing][functional][softmax][partition]") {
    const Size n = 4096;   // 16 tiles
    auto data = ramp(n, 0.0f, 0.001f);

    SECTION("partitioned credits") {
        auto ec = make_executor_config();
        ec.partition_l3_credits = true; ec.partition_l2_credits = true;
        FunctionalSoftmaxExecutor exec(make_config(1, n), ec);
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        require_softmax_match(result.values, host_softmax(data, 1, n));
    }
    SECTION("restreamed realization (constrained envelope)") {
        auto config = make_config(1, n);
        config.l3_buffer_count = 16; config.l2_bank_count = 16;
        REQUIRE(config.realization() ==
                OnlineSoftmaxScheduleGenerator::Realization::RESTREAMED);
        FunctionalSoftmaxExecutor exec(config, make_executor_config());
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        require_softmax_match(result.values, host_softmax(data, 1, n));
    }
}

TEST_CASE("Online softmax edge cases: all -inf and -inf prefix",
          "[timing][functional][softmax][edge]") {
    const Size n = 512;
    const float ninf = -std::numeric_limits<float>::infinity();

    SECTION("fully masked row -> uniform") {
        std::vector<float> data(n, ninf);
        FunctionalSoftmaxExecutor exec(make_config(1, n), make_executor_config());
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        const float uniform = 1.0f / static_cast<float>(n);
        for (float v : result.values) REQUIRE(v == Catch::Approx(uniform));
    }
    SECTION("masked prefix then finite values (no NaN poisoning)") {
        std::vector<float> data(n, ninf);
        for (Size i = n / 2; i < n; ++i) data[i] = 0.01f * static_cast<float>(i);
        FunctionalSoftmaxExecutor exec(make_config(1, n), make_executor_config());
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        for (float v : result.values) REQUIRE(std::isfinite(v));
        require_softmax_match(result.values, host_softmax(data, 1, n));
        // masked positions contribute ~0 probability
        for (Size i = 0; i < n / 2; ++i) REQUIRE(result.values[i] == Catch::Approx(0.0f).margin(1e-6));
    }
}
