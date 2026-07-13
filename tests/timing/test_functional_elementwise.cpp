// ============================================================================
// tests/timing/test_functional_elementwise.cpp
// Value-producing elementwise execution on the CSP executor, verified
// elementwise against an independent host oracle (issue #102, epic E2).
//
// The oracle is computed with a plain host loop - a deliberately separate
// path from apply_ve_op - so a semantic error in the simulator kernel cannot
// hide by matching itself.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_elementwise_executor.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::VEOp;

namespace {

using Form = ElementwiseScheduleGenerator::Form;

ElementwiseScheduleGenerator::Config make_generator_config(Size num_elements,
                                                           Form form) {
    ElementwiseScheduleGenerator::Config config;
    config.num_elements = num_elements;
    config.tile_elems = 256;
    config.form = form;
    config.a_base = 0x100000;
    config.b_base = 0x200000;
    config.c_base = 0x300000;
    return config;
}

ConcurrentTimingExecutor::Config make_executor_config() {
    ConcurrentTimingExecutor::Config config;
    config.max_cycles = 2'000'000;
    return config;
}

// Deterministic non-trivial test vectors (no RNG in tests)
std::vector<float> make_tensor(Size count, float base, float step) {
    std::vector<float> values(count);
    for (Size i = 0; i < count; ++i) {
        values[i] = base + step * static_cast<float>(i);
    }
    return values;
}

void require_elementwise_match(const std::vector<float>& actual,
                               const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO("element " << i);
        if (std::isnan(expected[i])) {
            REQUIRE(std::isnan(actual[i]));
        } else {
            REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1e-6f));
        }
    }
}

} // namespace

TEST_CASE("Binary elementwise CSP execution matches the host oracle",
          "[timing][functional][elementwise]") {
    const Size n = 1024;   // 4 tiles
    auto a = make_tensor(n, -3.0f, 0.25f);
    auto b = make_tensor(n, 2.0f, -0.125f);

    struct Case {
        VEOp op;
        float (*oracle)(float, float);
    };
    const Case cases[] = {
        {VEOp::ADD, [](float x, float y) { return x + y; }},
        {VEOp::SUB, [](float x, float y) { return x - y; }},
        {VEOp::MUL, [](float x, float y) { return x * y; }},
        {VEOp::DIV, [](float x, float y) { return x / y; }},
        {VEOp::MAX, [](float x, float y) { return x > y ? x : y; }},
        {VEOp::MIN, [](float x, float y) { return x < y ? x : y; }},
    };

    for (const auto& c : cases) {
        INFO("op=" << static_cast<int>(c.op));
        FunctionalElementwiseExecutor executor(
            make_generator_config(n, Form::BINARY), make_executor_config());
        auto result = executor.run(c.op, a, b);
        REQUIRE(result.execution.success);
        REQUIRE_FALSE(result.execution.livelock_detected);

        std::vector<float> expected(n);
        for (Size i = 0; i < n; ++i) expected[i] = c.oracle(a[i], b[i]);
        require_elementwise_match(result.values, expected);
    }
}

TEST_CASE("Broadcast-B CSP execution matches the host oracle",
          "[timing][functional][elementwise][broadcast]") {
    const Size n = 1024;   // 4 A tiles against 1 resident B tile
    auto a = make_tensor(n, 1.0f, 0.5f);
    auto bias = make_tensor(256, -8.0f, 0.0625f);

    FunctionalElementwiseExecutor executor(
        make_generator_config(n, Form::BROADCAST_B), make_executor_config());
    auto result = executor.run(VEOp::ADD, a, bias);
    REQUIRE(result.execution.success);
    REQUIRE_FALSE(result.execution.livelock_detected);

    // Every A tile pairs against the SAME resident bias tile: a value error
    // here would also implicate the 1:1:k feed accounting, not just the op
    std::vector<float> expected(n);
    for (Size i = 0; i < n; ++i) expected[i] = a[i] + bias[i % 256];
    require_elementwise_match(result.values, expected);
}

TEST_CASE("Unary and scalar CSP execution matches the host oracle",
          "[timing][functional][elementwise]") {
    const Size n = 512;   // 2 tiles
    auto a = make_tensor(n, 0.25f, 0.03125f);   // positive domain for sqrt/log

    SECTION("unary chain ops") {
        struct Case {
            VEOp op;
            float (*oracle)(float);
        };
        const Case cases[] = {
            {VEOp::NEG,  [](float x) { return -x; }},
            {VEOp::ABS,  [](float x) { return std::fabs(x); }},
            {VEOp::SQRT, [](float x) { return std::sqrt(x); }},
            {VEOp::EXP,  [](float x) { return std::exp(x); }},
            {VEOp::LOG,  [](float x) { return std::log(x); }},
        };
        for (const auto& c : cases) {
            INFO("op=" << static_cast<int>(c.op));
            FunctionalElementwiseExecutor executor(
                make_generator_config(n, Form::UNARY), make_executor_config());
            auto result = executor.run(c.op, a);
            REQUIRE(result.execution.success);

            std::vector<float> expected(n);
            for (Size i = 0; i < n; ++i) expected[i] = c.oracle(a[i]);
            require_elementwise_match(result.values, expected);
        }
    }

    SECTION("scalar-broadcast ops") {
        const float scalar = 1.75f;
        FunctionalElementwiseExecutor executor(
            make_generator_config(n, Form::UNARY), make_executor_config());
        auto result = executor.run(VEOp::MUL_S, a, {}, scalar);
        REQUIRE(result.execution.success);

        std::vector<float> expected(n);
        for (Size i = 0; i < n; ++i) expected[i] = a[i] * scalar;
        require_elementwise_match(result.values, expected);
    }
}

TEST_CASE("Non-aligned functional elementwise clamps the trailing tile",
          "[timing][functional][elementwise]") {
    const Size n = 1000;   // 4 tiles: 256+256+256+232 (the #101 clamp)
    auto a = make_tensor(n, -1.0f, 0.01f);
    auto b = make_tensor(n, 4.0f, -0.02f);

    FunctionalElementwiseExecutor executor(
        make_generator_config(n, Form::BINARY), make_executor_config());
    auto result = executor.run(VEOp::MUL, a, b);
    REQUIRE(result.execution.success);

    std::vector<float> expected(n);
    for (Size i = 0; i < n; ++i) expected[i] = a[i] * b[i];
    require_elementwise_match(result.values, expected);
}

TEST_CASE("Functional elementwise under partitioned credits",
          "[timing][functional][elementwise][partition]") {
    const Size n = 1024;
    auto a = make_tensor(n, 0.0f, 0.5f);
    auto b = make_tensor(n, 100.0f, -0.5f);

    auto exec_config = make_executor_config();
    exec_config.partition_l3_credits = true;
    exec_config.partition_l2_credits = true;

    FunctionalElementwiseExecutor executor(
        make_generator_config(n, Form::BINARY), exec_config);
    auto result = executor.run(VEOp::MAX, a, b);
    REQUIRE(result.execution.success);
    REQUIRE_FALSE(result.execution.livelock_detected);

    std::vector<float> expected(n);
    for (Size i = 0; i < n; ++i) expected[i] = a[i] > b[i] ? a[i] : b[i];
    require_elementwise_match(result.values, expected);
}

TEST_CASE("Functional elementwise input contract is enforced",
          "[timing][functional][elementwise]") {
    const Size n = 512;
    auto a = make_tensor(n, 1.0f, 1.0f);

    SECTION("wrong A size") {
        FunctionalElementwiseExecutor executor(
            make_generator_config(n, Form::UNARY), make_executor_config());
        REQUIRE_THROWS_AS(executor.run(VEOp::NEG, std::vector<float>(n - 1)),
                          std::invalid_argument);
    }
    SECTION("binary op on unary form") {
        FunctionalElementwiseExecutor executor(
            make_generator_config(n, Form::UNARY), make_executor_config());
        REQUIRE_THROWS_AS(executor.run(VEOp::ADD, a), std::invalid_argument);
    }
    SECTION("broadcast B must be one tile") {
        FunctionalElementwiseExecutor executor(
            make_generator_config(n, Form::BROADCAST_B), make_executor_config());
        REQUIRE_THROWS_AS(executor.run(VEOp::ADD, a, std::vector<float>(n)),
                          std::invalid_argument);
    }
}
