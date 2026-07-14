// ============================================================================
// tests/timing/test_functional_reduction.cpp
// Value-producing streaming reduction on the CSP executor, verified against
// independent host oracles (issue #107, epic E3).
//
// Oracles are plain host loops, deliberately separate from reduce_payloads,
// so a semantic error in the simulator kernel cannot match itself.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_reduction_executor.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

using Form = OnlineReductionScheduleGenerator::Form;
using ReduceOp = OnlineReductionScheduleGenerator::ReduceOp;

OnlineReductionScheduleGenerator::Config make_config(Size num_rows,
                                                     Size reduction_elems,
                                                     Form form, ReduceOp op) {
    OnlineReductionScheduleGenerator::Config c;
    c.num_rows = num_rows;
    c.reduction_elems = reduction_elems;
    c.tile_elems = 256;
    c.form = form;
    c.op = op;
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

std::vector<float> make_stream(Size count, float base, float step) {
    std::vector<float> v(count);
    for (Size i = 0; i < count; ++i) v[i] = base + step * static_cast<float>(i);
    return v;
}

// Independent host oracle over a contiguous span
double oracle(ReduceOp op, const std::vector<float>& v, Size begin, Size end) {
    double mx = -1e300, mn = 1e300, sum = 0.0, sumsq = 0.0;
    size_t n = end - begin;
    for (Size i = begin; i < end; ++i) {
        mx = std::max(mx, static_cast<double>(v[i]));
        mn = std::min(mn, static_cast<double>(v[i]));
        sum += v[i]; sumsq += static_cast<double>(v[i]) * v[i];
    }
    switch (op) {
        case ReduceOp::MAX: return mx;
        case ReduceOp::MIN: return mn;
        case ReduceOp::SUM: return sum;
        case ReduceOp::MEAN: return sum / n;
        case ReduceOp::VAR: return std::max(0.0, sumsq / n - (sum / n) * (sum / n));
    }
    return 0.0;
}

void require_close(ReduceOp op, float actual, double expected) {
    if (op == ReduceOp::MAX || op == ReduceOp::MIN) {
        REQUIRE(static_cast<double>(actual) == expected);  // exact
    } else {
        REQUIRE(actual == Catch::Approx(expected).epsilon(1e-4).margin(1e-4));
    }
}

const ReduceOp kAllOps[] = {ReduceOp::MAX, ReduceOp::MIN, ReduceOp::SUM,
                            ReduceOp::MEAN, ReduceOp::VAR};

} // namespace

TEST_CASE("FULL_REDUCE matches the host oracle for every op",
          "[timing][functional][reduction]") {
    const Size n = 4096;   // 16 tiles
    auto data = make_stream(n, -5.0f, 0.013f);

    for (ReduceOp op : kAllOps) {
        INFO("op=" << static_cast<int>(op));
        FunctionalReductionExecutor exec(
            make_config(1, n, Form::FULL_REDUCE, op), make_executor_config());
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        REQUIRE_FALSE(result.execution.livelock_detected);
        REQUIRE(result.stats.size() == 1);
        require_close(op, result.stats[0], oracle(op, data, 0, n));
    }
}

TEST_CASE("ROW_STATS produces the right stat per row",
          "[timing][functional][reduction]") {
    const Size rows = 4, re = 1024;   // 4 rows x 4 tiles
    auto data = make_stream(rows * re, 2.0f, -0.007f);

    for (ReduceOp op : {ReduceOp::SUM, ReduceOp::MEAN, ReduceOp::VAR, ReduceOp::MAX}) {
        INFO("op=" << static_cast<int>(op));
        FunctionalReductionExecutor exec(
            make_config(rows, re, Form::ROW_STATS, op), make_executor_config());
        auto result = exec.run(data);
        REQUIRE(result.execution.success);
        REQUIRE(result.stats.size() == rows);
        for (Size r = 0; r < rows; ++r) {
            INFO("row=" << r);
            require_close(op, result.stats[r], oracle(op, data, r * re, (r + 1) * re));
        }
    }
}

TEST_CASE("Reduction under partitioned credits stays correct",
          "[timing][functional][reduction][partition]") {
    const Size n = 4096;
    auto data = make_stream(n, 0.0f, 0.5f);

    auto exec_config = make_executor_config();
    exec_config.partition_l3_credits = true;
    exec_config.partition_l2_credits = true;

    FunctionalReductionExecutor exec(
        make_config(1, n, Form::FULL_REDUCE, ReduceOp::SUM), exec_config);
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    require_close(ReduceOp::SUM, result.stats[0], oracle(ReduceOp::SUM, data, 0, n));
}

TEST_CASE("Non-aligned reduction span reduces every element",
          "[timing][functional][reduction]") {
    const Size n = 1000;   // 4 tiles: 256+256+256+232
    auto data = make_stream(n, -1.0f, 0.011f);

    FunctionalReductionExecutor exec(
        make_config(1, n, Form::FULL_REDUCE, ReduceOp::SUM), make_executor_config());
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    require_close(ReduceOp::SUM, result.stats[0], oracle(ReduceOp::SUM, data, 0, n));
}

TEST_CASE("ROW_NORMALIZE is rejected by the functional reduction executor",
          "[timing][functional][reduction]") {
    REQUIRE_THROWS_AS(
        FunctionalReductionExecutor(
            make_config(1, 1024, Form::ROW_NORMALIZE, ReduceOp::SUM),
            make_executor_config()),
        std::invalid_argument);
}
