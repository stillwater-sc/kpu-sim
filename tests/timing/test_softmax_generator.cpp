// ============================================================================
// tests/timing/test_softmax_generator.cpp
// OnlineSoftmaxScheduleGenerator structure + execution (issue #156, epic E8)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/online_softmax_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

using Realization = OnlineSoftmaxScheduleGenerator::Realization;

OnlineSoftmaxScheduleGenerator::Config base_config() {
    OnlineSoftmaxScheduleGenerator::Config c;
    c.num_rows = 1;
    c.reduction_elems = 1024;   // 4 tiles
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

size_t count_resident_computes(const ScheduleResult& s) {
    size_t n = 0;
    for (const auto& op : s.operations) {
        if (op.type == ScheduleOpType::COMPUTE && !op.resident_tiles.empty()) ++n;
    }
    return n;
}

} // namespace

TEST_CASE("Online softmax ROW_RESIDENT delivers the row once, apply resident",
          "[timing][schedule][softmax]") {
    auto config = base_config();
    REQUIRE(config.realization() == Realization::ROW_RESIDENT);
    OnlineSoftmaxScheduleGenerator gen(config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    const Size rt = config.reduction_tiles();  // 4
    // Row delivered once (rt loads); NO stat load (it stays resident)
    REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == rt);
    // Stats feeds (rt) + apply feeds (rt)
    REQUIRE(schedule.count_ops(ScheduleOpType::FEED) == 2 * rt);
    // 1 stats compute + rt apply computes
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) == 1 + rt);
    // Every apply compute carries a resident (m,l) dependency
    REQUIRE(count_resident_computes(schedule) == rt);
    REQUIRE(schedule.count_ops(ScheduleOpType::STORE) == rt);   // outputs only
}

TEST_CASE("Online softmax RESTREAMED re-reads the row, apply still resident",
          "[timing][schedule][softmax][envelope]") {
    auto config = base_config();
    config.reduction_elems = 4096;   // rt=16
    config.l3_buffer_count = 16;     // share = 4 < rt+2 -> RESTREAMED
    config.l2_bank_count = 16;
    REQUIRE(config.realization() == Realization::RESTREAMED);
    OnlineSoftmaxScheduleGenerator gen(config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    const Size rt = config.reduction_tiles();
    REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) == 2 * rt);   // row read twice
    REQUIRE(count_resident_computes(schedule) == rt);             // apply resident
}

TEST_CASE("Online softmax refuses a degenerate envelope",
          "[timing][schedule][softmax][envelope]") {
    auto config = base_config();
    config.reduction_elems = 4096;   // rt=16, restreamed needs share>=3
    config.l3_buffer_count = 8;      // share = 2 < 3
    config.l2_bank_count = 8;
    auto schedule = OnlineSoftmaxScheduleGenerator(config).generate();
    REQUIRE_FALSE(schedule.valid);
    REQUIRE(schedule.error_message.find("working set") != std::string::npos);
}

TEST_CASE("Online softmax schedules execute to completion",
          "[timing][executor][regression][softmax]") {
    auto run = [](const OnlineSoftmaxScheduleGenerator::Config& config) {
        OnlineSoftmaxScheduleGenerator gen(config);
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

    SECTION("row-resident, single row") {
        auto c = base_config();
        REQUIRE(c.realization() == Realization::ROW_RESIDENT);
        run(c);
    }
    SECTION("row-resident, batched rows") {
        auto c = base_config();
        c.num_rows = 3;
        run(c);
    }
    SECTION("restreamed, large row") {
        auto c = base_config();
        c.reduction_elems = 4096;
        c.l3_buffer_count = 16;
        c.l2_bank_count = 16;
        REQUIRE(c.realization() == Realization::RESTREAMED);
        run(c);
    }
    SECTION("non-aligned row") {
        auto c = base_config();
        c.reduction_elems = 1000;   // 4 tiles, 232 tail
        run(c);
    }
}
