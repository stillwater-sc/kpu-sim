// ============================================================================
// tests/timing/test_reduction_generator.cpp
// OnlineReductionScheduleGenerator structure + execution (issue #106, epic E3)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/online_reduction_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <string>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

using Form = OnlineReductionScheduleGenerator::Form;
using Realization = OnlineReductionScheduleGenerator::Realization;
using ReduceOp = OnlineReductionScheduleGenerator::ReduceOp;

OnlineReductionScheduleGenerator::Config base_config() {
    OnlineReductionScheduleGenerator::Config c;
    c.num_rows = 1;
    c.reduction_elems = 1024;   // 4 reduction tiles at tile_elems=256
    c.tile_elems = 256;
    c.op = ReduceOp::SUM;
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

} // namespace

TEST_CASE("FULL_REDUCE emits one compute over all streamed feeds",
          "[timing][schedule][reduction]") {
    auto config = base_config();
    config.form = Form::FULL_REDUCE;
    OnlineReductionScheduleGenerator gen(config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    const Size rt = config.reduction_tiles();  // 4
    REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == rt);
    REQUIRE(schedule.count_ops(ScheduleOpType::FEED) == rt);
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == 1);
    REQUIRE(schedule.count_ops(ScheduleOpType::DRAIN) == 1);
    REQUIRE(schedule.count_ops(ScheduleOpType::STORE) == 1);

    // The single compute depends on every streamed tile
    size_t compute_deps = 0;
    for (const auto& op : schedule.operations) {
        if (op.type == ScheduleOpType::COMPUTE) compute_deps = op.dependency_tiles.size();
    }
    REQUIRE(compute_deps == rt);
    REQUIRE(config.required_working_set() == 2);
}

TEST_CASE("ROW_STATS emits one compute per row",
          "[timing][schedule][reduction]") {
    auto config = base_config();
    config.form = Form::ROW_STATS;
    config.num_rows = 3;
    OnlineReductionScheduleGenerator gen(config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    const Size rt = config.reduction_tiles();
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == config.num_rows);
    REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == config.num_rows * rt);
    REQUIRE(schedule.count_ops(ScheduleOpType::STORE) == config.num_rows);  // one stat/row
}

TEST_CASE("ROW_NORMALIZE selects realization from the envelope a priori",
          "[timing][schedule][reduction][envelope]") {
    SECTION("row fits the share -> ROW_RESIDENT, delivered once") {
        auto config = base_config();
        config.form = Form::ROW_NORMALIZE;
        config.num_rows = 1;
        // share = min(l3,l2)/4; with defaults 32/64 -> 8 >= rt(4)+2
        REQUIRE(config.realization() == Realization::ROW_RESIDENT);
        OnlineReductionScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        const Size rt = config.reduction_tiles();
        // Row tiles delivered once (rt), stat operand once (1) = rt+1 loads;
        // RESTREAMED would load the row twice.
        REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == rt + 1);
        // Two compute phases: 1 stat + rt apply
        REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == 1 + rt);
    }

    SECTION("row exceeds the share -> RESTREAMED, row read twice") {
        auto config = base_config();
        config.form = Form::ROW_NORMALIZE;
        config.reduction_elems = 4096;   // rt=16
        config.l3_buffer_count = 16;     // share = 4 < rt+2
        config.l2_bank_count = 16;
        REQUIRE(config.realization() == Realization::RESTREAMED);
        OnlineReductionScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        const Size rt = config.reduction_tiles();
        // Row read twice (2*rt) + stat operand (1)
        REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == 2 * rt + 1);
    }

    SECTION("degenerate envelope is refused a priori") {
        auto config = base_config();
        config.form = Form::FULL_REDUCE;
        config.l3_buffer_count = 4;
        config.l2_bank_count = 4;   // share = 1 < working set 2
        OnlineReductionScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE_FALSE(schedule.valid);
        REQUIRE(schedule.error_message.find("working set") != std::string::npos);
    }
}

TEST_CASE("Reduction schedules execute to completion",
          "[timing][executor][regression][reduction]") {
    auto run = [](const OnlineReductionScheduleGenerator::Config& config) {
        OnlineReductionScheduleGenerator gen(config);
        auto schedule = gen.generate();
        REQUIRE(schedule.valid);

        ConcurrentTimingExecutor executor(make_executor_config());
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(schedule);

        INFO("strategy=" << schedule.metadata.strategy
             << " cycles=" << result.total_cycles
             << " error=" << result.error_message);
        REQUIRE(result.success);
        REQUIRE_FALSE(result.livelock_detected);

        auto stats = executor.get_statistics();
        REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
        REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    };

    SECTION("FULL_REDUCE") {
        auto c = base_config();
        c.form = Form::FULL_REDUCE;
        c.reduction_elems = 4096;   // 16 tiles
        run(c);
    }
    SECTION("ROW_STATS batched rows") {
        auto c = base_config();
        c.form = Form::ROW_STATS;
        c.num_rows = 4;
        run(c);
    }
    SECTION("ROW_NORMALIZE row-resident") {
        auto c = base_config();
        c.form = Form::ROW_NORMALIZE;
        c.num_rows = 2;
        REQUIRE(c.realization() == Realization::ROW_RESIDENT);
        run(c);
    }
    SECTION("ROW_NORMALIZE restreamed") {
        auto c = base_config();
        c.form = Form::ROW_NORMALIZE;
        c.reduction_elems = 4096;   // rt=16
        c.l3_buffer_count = 16;
        c.l2_bank_count = 16;
        REQUIRE(c.realization() == Realization::RESTREAMED);
        run(c);
    }
}
