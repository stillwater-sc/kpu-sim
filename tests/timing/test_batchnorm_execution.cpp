// ============================================================================
// tests/timing/test_batchnorm_execution.cpp
// End-to-end execution regression for BatchNormScheduleGenerator (E9-T3, #180).
//
// Locks in the batchnorm half of #139: before T3 the inference generator emitted
// DRAIN with no COMPUTE, so an executed BN schedule had a drain with no producer
// and would deadlock. T3 emits an executable per-channel affine COMPUTE (input +
// resident scale/shift). This test EXECUTES the generated schedules (timing-only)
// and asserts a COMPUTE per output tile and livelock-free completion. Functional
// value correctness is T4 (#181).
//
// The all-channel-preload design makes the working set 2C+1, so each case sizes
// its envelope to C (min(l3,l2)/4 >= 2C+1).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

struct BNCase {
    const char* name;
    Size N, C, H, W, Ti;
};

constexpr BNCase kCases[] = {
    {"1x3x8x8",   1, 3,  8,  8, 16},   // 2C+1 = 7
    {"2x3x4x4",   2, 3,  4,  4, 16},   // batch > 1
    {"1x16x8x8",  1, 16, 8,  8, 16},   // larger C (needs bigger envelope)
    {"1x8x7x7",   1, 8,  7,  7, 16},   // non-tile-aligned spatial (49)
};

// Size l3/l2 so min/4 >= 2C+1 (all-channel preload residency).
Size envelope_for(Size C) {
    Size need = 2 * C + 1;
    Size credits = 4 * need + 4;  // margin above the 4x share bound
    return credits < 32 ? 32 : credits;
}

BatchNormScheduleGenerator::Config gen_config(const BNCase& c) {
    BatchNormScheduleGenerator::Config config;
    config.N = c.N; config.C = c.C; config.H = c.H; config.W = c.W;
    config.Ti = c.Ti; config.Tj = c.Ti;
    config.training = false;
    const Size credits = envelope_for(c.C);
    config.l3_buffer_count = credits;
    config.l2_bank_count = credits;
    config.input_base  = 0x00100000;
    config.output_base = 0x00400000;
    config.scale_base  = 0x00010000;
    config.shift_base  = 0x00020000;
    return config;
}

ConcurrentTimingExecutor::Config exec_config(Size C) {
    ConcurrentTimingExecutor::Config config;
    const Size credits = envelope_for(C);
    config.l3_buffer_count = credits;
    config.l2_bank_count = credits;
    config.max_cycles = 2'000'000;
    config.enable_livelock_detection = true;
    config.livelock_threshold = 20000;
    return config;
}

} // namespace

TEST_CASE("BatchNorm inference generator emits a COMPUTE per output tile",
          "[timing][batchnorm][generator]") {
    for (const auto& c : kCases) {
        DYNAMIC_SECTION(c.name) {
            auto schedule = BatchNormScheduleGenerator(gen_config(c)).generate();
            REQUIRE(schedule.valid);

            // #139 regression: a COMPUTE exists for every output tile, so no
            // DRAIN is left without a producer.
            const auto n_compute = schedule.count_ops(ScheduleOpType::COMPUTE);
            const auto n_drain   = schedule.count_ops(ScheduleOpType::DRAIN);
            const auto n_store   = schedule.count_ops(ScheduleOpType::STORE);
            REQUIRE(n_compute > 0);
            REQUIRE(n_compute == n_drain);
            REQUIRE(n_compute == n_store);

            // Every input/output tile is one (n, c, spatial-tile).
            const Size spatial_tiles = (c.H * c.W + c.Ti - 1) / c.Ti;
            REQUIRE(n_compute == c.N * c.C * spatial_tiles);
        }
    }
}

TEST_CASE("BatchNorm inference schedules execute livelock-free",
          "[timing][batchnorm][executor][regression]") {
    for (const auto& c : kCases) {
        DYNAMIC_SECTION(c.name) {
            auto schedule = BatchNormScheduleGenerator(gen_config(c)).generate();
            REQUIRE(schedule.valid);

            ConcurrentTimingExecutor executor(exec_config(c.C));
            ScheduleExecutor sched_exec(executor);
            auto result = sched_exec.execute(schedule);

            INFO("case=" << c.name << " cycles=" << result.total_cycles
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
        }
    }
}

TEST_CASE("BatchNorm inference working set is 2C+1 (folded)",
          "[timing][batchnorm][envelope]") {
    // The fold halves residency from 4C+1 to 2C+1; the generator refuses a
    // priori when the envelope share is one tile short.
    BNCase c{"boundary", 1, 4, 8, 8, 16};  // 2C+1 = 9
    auto cfg = gen_config(c);

    // share = min(l3,l2)/4 = 9 -> generates (== working set)
    cfg.l3_buffer_count = 36; cfg.l2_bank_count = 36;
    REQUIRE(BatchNormScheduleGenerator(cfg).generate().valid);

    // share = 8 -> refused (< working set)
    cfg.l3_buffer_count = 32; cfg.l2_bank_count = 32;
    auto refused = BatchNormScheduleGenerator(cfg).generate();
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("working set") != std::string::npos);
}
