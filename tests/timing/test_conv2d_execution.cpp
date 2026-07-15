// ============================================================================
// tests/timing/test_conv2d_execution.cpp
// End-to-end execution regression for Conv2DScheduleGenerator (E6-T3, #121).
//
// Locks in the conv2d half of #139: before T3 the generator emitted DRAIN with
// no COMPUTE, so an executed conv2d schedule had a drain with no producer and
// would deadlock. T3 emits COMPUTE with the full A/B K-slice dependency set;
// this test EXECUTES the generated schedules (timing-only, mirroring
// test_multi_tile_execution) and asserts livelock-free completion, plus that a
// COMPUTE is present for every output tile. Functional value correctness is
// T4 (#122).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

ConcurrentTimingExecutor::Config make_executor_config() {
    ConcurrentTimingExecutor::Config config;
    config.num_memory_controllers = 1;
    config.l3_buffer_count = 32;
    config.num_block_movers = 4;
    config.l2_bank_count = 64;
    config.num_row_streamers = 2;
    config.num_col_streamers = 2;
    config.max_cycles = 1'000'000;
    config.enable_livelock_detection = true;
    config.livelock_threshold = 10000;
    return config;
}

struct ConvCase {
    const char* name;
    Size H, W, C_in, C_out, Kh, Kw, stride, pad;
};

// Tile size 16; dims chosen so M, C_out, and K = Kh*Kw*C_in are multiples of 16.
constexpr ConvCase kConvCases[] = {
    {"3x3_s1_p1",   8, 8, 16, 16, 3, 3, 1, 1},  // M=64,  K=144, N=16
    {"1x1_pointwise", 8, 8, 32, 16, 1, 1, 1, 0}, // M=64,  K=32,  N=16
    {"3x3_s2_p1",   8, 8, 16, 32, 3, 3, 2, 1},  // strided: M=16, K=144, N=32
};

struct StrategyCase {
    Conv2DScheduleGenerator::Strategy strategy;
    const char* name;
};

constexpr StrategyCase kStrategies[] = {
    {Conv2DScheduleGenerator::Strategy::IM2COL_INTERLEAVED,       "im2col_interleaved"},
    {Conv2DScheduleGenerator::Strategy::IM2COL_OUTPUT_STATIONARY, "im2col_output_stationary"},
};

Conv2DScheduleGenerator::Config make_config(const ConvCase& c,
                                            Conv2DScheduleGenerator::Strategy s) {
    Conv2DScheduleGenerator::Config config;
    config.N = 1;
    config.H_in = c.H; config.W_in = c.W; config.C_in = c.C_in;
    config.C_out = c.C_out;
    config.Kh = c.Kh; config.Kw = c.Kw;
    config.stride_h = c.stride; config.stride_w = c.stride;
    config.padding_h = c.pad; config.padding_w = c.pad;
    config.Ti = 16; config.Tj = 16; config.Tk = 16;
    config.strategy = s;
    config.input_base = 0x00001000;
    config.filter_base = 0x00100000;
    config.output_base = 0x00200000;
    return config;
}

} // namespace

TEST_CASE("Conv2D generator emits an executable COMPUTE for every output tile",
          "[timing][conv2d][generator]") {
    for (const auto& strat : kStrategies) {
        for (const auto& conv : kConvCases) {
            DYNAMIC_SECTION(strat.name << " " << conv.name) {
                auto schedule =
                    Conv2DScheduleGenerator(make_config(conv, strat.strategy)).generate();
                REQUIRE(schedule.valid);

                // #139 regression: a COMPUTE exists for each C tile, so no DRAIN
                // is left without a producer.
                const auto n_compute = schedule.count_ops(ScheduleOpType::COMPUTE);
                const auto n_drain   = schedule.count_ops(ScheduleOpType::DRAIN);
                const auto n_store   = schedule.count_ops(ScheduleOpType::STORE);
                REQUIRE(n_compute > 0);
                REQUIRE(n_compute == n_drain);
                REQUIRE(n_compute == n_store);

                // LOAD:MOVE pair 1:1 (schedule invariant).
                REQUIRE(schedule.count_ops(ScheduleOpType::LOAD) ==
                        schedule.count_ops(ScheduleOpType::MOVE));
            }
        }
    }
}

TEST_CASE("Conv2D schedules execute livelock-free through the executor",
          "[timing][conv2d][executor][regression]") {
    for (const auto& strat : kStrategies) {
        for (const auto& conv : kConvCases) {
            DYNAMIC_SECTION(strat.name << " " << conv.name) {
                auto schedule =
                    Conv2DScheduleGenerator(make_config(conv, strat.strategy)).generate();
                REQUIRE(schedule.valid);

                ConcurrentTimingExecutor executor(make_executor_config());
                ScheduleExecutor sched_exec(executor);
                auto result = sched_exec.execute(schedule);

                INFO("strategy=" << strat.name << " case=" << conv.name
                     << " cycles=" << result.total_cycles
                     << " error=" << result.error_message);

                // The #139 surface: before COMPUTE was emitted, the drain had no
                // producer and this deadlocked. Now it completes.
                REQUIRE(result.success);
                REQUIRE_FALSE(result.livelock_detected);
                REQUIRE(result.total_cycles > 0);

                auto stats = executor.get_statistics();
                REQUIRE(stats.tiles_drained ==
                        schedule.count_ops(ScheduleOpType::DRAIN));
                REQUIRE(stats.tiles_fed ==
                        schedule.count_ops(ScheduleOpType::FEED));
                REQUIRE(stats.tiles_writeback ==
                        schedule.count_ops(ScheduleOpType::WRITEBACK));
            }
        }
    }
}
