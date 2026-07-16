// ============================================================================
// tests/timing/test_pooling_execution.cpp
// End-to-end execution regression for PoolingScheduleGenerator (E7-T3, #193).
//
// The pooling generator is executable from the start: every output tile has a
// reduce COMPUTE before its drain (so no DRAIN lacks a producer - it never has
// the #139 defect). This EXECUTES the generated schedules (timing-only) and
// asserts a COMPUTE per output tile and livelock-free completion, for windowed
// max/avg pooling and global average pooling. Value correctness is T4 (#194).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

ConcurrentTimingExecutor::Config exec_config() {
    ConcurrentTimingExecutor::Config config;
    config.l3_buffer_count = 32;
    config.l2_bank_count = 64;
    config.max_cycles = 1'000'000;
    config.enable_livelock_detection = true;
    config.livelock_threshold = 20000;
    return config;
}

PoolingScheduleGenerator::Config windowed(Size N, Size C, Size H, Size W,
                                          Size Kh, Size Kw, Size stride, Size pad,
                                          PoolType type) {
    PoolingScheduleGenerator::Config cfg;
    cfg.geom.N = N; cfg.geom.C = C; cfg.geom.H = H; cfg.geom.W = W;
    cfg.geom.Kh = Kh; cfg.geom.Kw = Kw;
    cfg.geom.stride_h = cfg.geom.stride_w = stride;
    cfg.geom.pad_h = cfg.geom.pad_w = pad;
    cfg.pool_type = type;
    cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED;
    cfg.Ti = 16;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    return cfg;
}

void run_and_check(const PoolingScheduleGenerator::Config& cfg, const char* label) {
    DYNAMIC_SECTION(label) {
        auto schedule = PoolingScheduleGenerator(cfg).generate();
        REQUIRE(schedule.valid);

        // Executable-from-the-start: a COMPUTE per output tile.
        const auto n_compute = schedule.count_ops(ScheduleOpType::COMPUTE);
        const auto n_drain   = schedule.count_ops(ScheduleOpType::DRAIN);
        const auto n_store   = schedule.count_ops(ScheduleOpType::STORE);
        REQUIRE(n_compute > 0);
        REQUIRE(n_compute == n_drain);
        REQUIRE(n_compute == n_store);

        ConcurrentTimingExecutor executor(exec_config());
        ScheduleExecutor sched_exec(executor);
        auto result = sched_exec.execute(schedule);
        INFO("cycles=" << result.total_cycles << " error=" << result.error_message);
        REQUIRE(result.success);
        REQUIRE_FALSE(result.livelock_detected);
        REQUIRE(result.total_cycles > 0);

        auto stats = executor.get_statistics();
        REQUIRE(stats.tiles_drained == n_drain);
        REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    }
}

} // namespace

TEST_CASE("Pooling generator emits an executable COMPUTE per output tile "
          "and executes livelock-free", "[timing][pooling][generator][executor]") {
    run_and_check(windowed(1, 8, 8, 8, 2, 2, 2, 0, PoolType::MAX), "max_2x2_s2");
    run_and_check(windowed(1, 8, 8, 8, 2, 2, 2, 0, PoolType::AVG), "avg_2x2_s2");
    run_and_check(windowed(1, 16, 16, 16, 3, 3, 2, 1, PoolType::MAX), "max_3x3_s2_p1");
    run_and_check(windowed(2, 8, 8, 8, 2, 2, 2, 0, PoolType::MAX), "max_batch2");
    run_and_check(windowed(1, 8, 7, 7, 3, 3, 2, 1, PoolType::MAX), "max_nonaligned");

    PoolingScheduleGenerator::Config gap;
    gap.geom.N = 1; gap.geom.C = 16; gap.geom.H = 8; gap.geom.W = 8;
    gap.mode = PoolingScheduleGenerator::Mode::GLOBAL_AVG;
    gap.Ti = 16; gap.input_base = 0x100000; gap.output_base = 0x400000;
    run_and_check(gap, "global_avg_pool");
}

TEST_CASE("Pooling envelope refusal boundary", "[timing][pooling][envelope]") {
    // working set 3; share = min(l3,l2)/4: 12 -> 3 generates, 11 -> 2 refused.
    auto cfg = windowed(1, 8, 8, 8, 2, 2, 2, 0, PoolType::MAX);
    cfg.l3_buffer_count = 12; cfg.l2_bank_count = 12;
    REQUIRE(PoolingScheduleGenerator(cfg).generate().valid);
    cfg.l3_buffer_count = 11; cfg.l2_bank_count = 11;
    auto refused = PoolingScheduleGenerator(cfg).generate();
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("working set") != std::string::npos);
}
