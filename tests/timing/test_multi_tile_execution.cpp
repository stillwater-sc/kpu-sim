// ============================================================================
// tests/timing/test_multi_tile_execution.cpp
// End-to-end regression: execute generated multi-tile schedules through the
// ConcurrentTimingExecutor across the strategy x size matrix.
//
// Locks in the fix for issue #61 (multi-tile livelock): the livelock was
// invisible to CI because component suites test in isolation and schedule
// suites only generate/validate — no test EXECUTED a generated multi-tile
// schedule. This one does. See also issue #64.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>
#include <sw/kpu/timing/schedule/schedule_validator.hpp>

#include <string>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

// Mirrors examples/schedule/run_matmul.cpp defaults
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

MatMulScheduleGenerator::Config make_generator_config(
    Size n, MatMulScheduleGenerator::Strategy strategy) {
    MatMulScheduleGenerator::Config config;
    config.M = n;
    config.N = n;
    config.K = n;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.strategy = strategy;
    config.a_base = 0x00001000;
    config.b_base = 0x00100000;
    config.c_base = 0x00200000;
    return config;
}

struct StrategyCase {
    MatMulScheduleGenerator::Strategy strategy;
    const char* name;
};

constexpr StrategyCase kStrategies[] = {
    {MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY, "output_stationary"},
    {MatMulScheduleGenerator::Strategy::INTERLEAVED_AB,    "interleaved_ab"},
    {MatMulScheduleGenerator::Strategy::PREFETCH_NEXT,     "prefetch_next"},
    {MatMulScheduleGenerator::Strategy::BLOCKED_AB,        "blocked_ab"},
};

struct SizeCase {
    Size n;          // M = N = K (16^3 tiles -> n/16 tile grid per dim)
    Cycle ceiling;   // generous total-cycle bound; cycle counts vary across
                     // platforms (work assignment hashes with std::hash)
};

constexpr SizeCase kSizes[] = {
    {32,  10'000},    // 2x2x2 tile grid   (measured ~334 on linux/gcc)
    {64,  30'000},    // 4x4x4 tile grid   (measured ~1132)
    {128, 150'000},   // 8x8x8 tile grid   (measured ~8034)
};

} // namespace

// ============================================================================
// Strategy x size execution matrix
// ============================================================================

TEST_CASE("Multi-tile schedules execute to completion across strategies and sizes",
          "[timing][executor][regression]") {
    for (const auto& strat : kStrategies) {
        for (const auto& size : kSizes) {
            DYNAMIC_SECTION(strat.name << " " << size.n << "^3") {
                auto gen_config = make_generator_config(size.n, strat.strategy);
                MatMulScheduleGenerator generator(gen_config);
                auto schedule = generator.generate();
                REQUIRE(schedule.valid);

                ConcurrentTimingExecutor executor(make_executor_config());
                ScheduleExecutor sched_exec(executor);
                auto result = sched_exec.execute(schedule);

                INFO("strategy=" << strat.name << " n=" << size.n
                     << " cycles=" << result.total_cycles
                     << " error=" << result.error_message);

                // The #61 regression surface: completion without livelock
                REQUIRE(result.success);
                REQUIRE_FALSE(result.livelock_detected);
                REQUIRE(result.total_cycles > 0);
                REQUIRE(result.total_cycles < size.ceiling);

                // Every scheduled pipeline stage completed exactly once
                // (dedup completions count as completions)
                auto stats = executor.get_statistics();
                REQUIRE(stats.tiles_moved ==
                        schedule.count_ops(ScheduleOpType::MOVE));
                REQUIRE(stats.tiles_writeback ==
                        schedule.count_ops(ScheduleOpType::WRITEBACK));
                REQUIRE(stats.tiles_fed ==
                        schedule.count_ops(ScheduleOpType::FEED));
                REQUIRE(stats.tiles_drained ==
                        schedule.count_ops(ScheduleOpType::DRAIN));

                // Stall accounting invariant: each component counts at most
                // one stall cycle per tick, so aggregate stalls are bounded
                // by component count x total cycles. Catches per-request
                // over-counting regressions.
                const auto& config = executor.config();
                REQUIRE(stats.dma_credit_stalls <=
                        config.num_dma_engines * stats.total_cycles);
                REQUIRE(stats.bm_tag_stalls + stats.bm_credit_stalls <=
                        config.num_block_movers * stats.total_cycles);
                size_t n_streamers =
                    config.num_row_streamers + config.num_col_streamers;
                REQUIRE(stats.str_tag_stalls + stats.str_credit_stalls <=
                        n_streamers * stats.total_cycles);
            }
        }
    }
}

// ============================================================================
// Per-matrix credit partitioning (issue #89)
// ============================================================================

TEST_CASE("Multi-tile schedules execute under partitioned credits",
          "[timing][executor][regression][partition]") {
    // Defense-in-depth mode: the same strategy x size matrix must complete
    // with per-matrix (A/B/C) credit partitioning enabled
    for (const auto& strat : kStrategies) {
        for (size_t n : {Size(64), Size(128)}) {
            DYNAMIC_SECTION(strat.name << " " << n << "^3 partitioned") {
                auto gen_config = make_generator_config(static_cast<Size>(n),
                                                        strat.strategy);
                MatMulScheduleGenerator generator(gen_config);
                auto schedule = generator.generate();
                REQUIRE(schedule.valid);

                auto exec_config = make_executor_config();
                exec_config.partition_l3_credits = true;
                exec_config.partition_l2_credits = true;
                ConcurrentTimingExecutor executor(exec_config);
                ScheduleExecutor sched_exec(executor);
                auto result = sched_exec.execute(schedule);

                INFO("strategy=" << strat.name << " n=" << n
                     << " cycles=" << result.total_cycles
                     << " error=" << result.error_message);
                REQUIRE(result.success);
                REQUIRE_FALSE(result.livelock_detected);
            }
        }
    }
}

TEST_CASE("Partitioned credits prevent single-matrix buffer monopolization",
          "[timing][executor][partition]") {
    // Adversarial pattern no generator emits but a hand-written or dynamic
    // schedule could: a flood of A loads with NO downstream consumption,
    // plus one B load. With a shared pool the A flood takes every L3 credit
    // and the B tile never arrives; with per-matrix partitioning the B
    // partition is untouchable by A traffic.
    auto make_flood_tile = [](MatrixID m, Size ti) {
        TileDescriptor tile;
        tile.tile_id = {m, ti, 0, 0};
        tile.dram_address = 0x1000 + static_cast<Address>(ti) * 0x1000;
        tile.height = 16;
        tile.width = 16;
        tile.element_size = 4;
        tile.size_bytes = 1024;
        return tile;
    };

    auto run_flood = [&](bool partitioned) {
        auto config = make_executor_config();
        config.l3_buffer_count = 9;  // partitions to 3/3/3
        config.enable_livelock_detection = false;
        config.partition_l3_credits = partitioned;
        ConcurrentTimingExecutor executor(config);

        // 12 A loads against 9 buffers - demand exceeds the pool - and one
        // B load queued behind them. No moves: the A tiles never leave.
        for (Size i = 0; i < 12; ++i) {
            executor.schedule_load(make_flood_tile(MatrixID::A, i));
        }
        executor.schedule_load(make_flood_tile(MatrixID::B, 0));

        for (int i = 0; i < 2000 && !executor.is_complete(); ++i) {
            executor.step();
        }

        for (const auto& event : executor.events()) {
            if (event.type == EventType::TILE_ARRIVED_L3 &&
                event.tile_id.matrix == MatrixID::B) {
                return true;  // B tile made it into L3
            }
        }
        return false;
    };

    SECTION("shared pool: the A flood starves B") {
        REQUIRE_FALSE(run_flood(false));
    }
    SECTION("partitioned pool: B arrives despite the A flood") {
        REQUIRE(run_flood(true));
    }
}

// ============================================================================
// Envelope-aware blocked schedules under constrained buffers (issue #67)
// ============================================================================

TEST_CASE("BLOCKED_AB executes under a constrained resource envelope",
          "[timing][executor][regression][envelope]") {
    // A small envelope that the historical all-A-then-all-B ordering would
    // stress: 8 L3 buffers / 16 L2 banks for an 8-K-slice problem. The
    // generator must chunk the K loop so the working set fits.
    auto exec_config = make_executor_config();
    exec_config.l3_buffer_count = 8;
    exec_config.l2_bank_count = 16;

    auto gen_config = make_generator_config(
        128, MatMulScheduleGenerator::Strategy::BLOCKED_AB);
    gen_config.l3_buffer_count = 8;
    gen_config.l2_bank_count = 16;

    MatMulScheduleGenerator generator(gen_config);
    auto schedule = generator.generate();
    REQUIRE(schedule.valid);
    REQUIRE(is_livelock_safe(schedule, 8, 16));

    ConcurrentTimingExecutor executor(exec_config);
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("cycles=" << result.total_cycles << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
}

// ============================================================================
// COMPUTE dependency coverage (issue #63)
// ============================================================================

TEST_CASE("COMPUTE operations carry the full K-slice dependency set",
          "[timing][schedule][regression]") {
    // 64^3 at 16^3 tiles: k_tiles = 4, so every COMPUTE must depend on
    // 4 A tiles + 4 B tiles
    auto gen_config = make_generator_config(
        64, MatMulScheduleGenerator::Strategy::INTERLEAVED_AB);
    MatMulScheduleGenerator generator(gen_config);
    auto schedule = generator.generate();
    REQUIRE(schedule.valid);

    size_t compute_count = 0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::COMPUTE) continue;
        compute_count++;

        REQUIRE(op.dependency_tiles.size() == 8);

        size_t a_deps = 0;
        size_t b_deps = 0;
        for (const auto& dep : op.dependency_tiles) {
            if (dep.matrix == MatrixID::A) {
                a_deps++;
                REQUIRE(dep.ti == op.tile.tile_id.ti);
            } else if (dep.matrix == MatrixID::B) {
                b_deps++;
                REQUIRE(dep.tj == op.tile.tile_id.tj);
            }
        }
        REQUIRE(a_deps == 4);
        REQUIRE(b_deps == 4);

        // Legacy single-dependency field mirrors the last entry
        REQUIRE(op.dependency_tile == op.dependency_tiles.back());
    }
    REQUIRE(compute_count == 16);  // 4x4 C tiles
}

// ============================================================================
// K-scaled compute latency (issue #63)
// ============================================================================

TEST_CASE("Compute latency scales with the K-slice count",
          "[timing][executor][regression]") {
    // 32^3 at 16^3 tiles: k_tiles = 2, so compute latency must be
    // compute_latency + (2 - 1) * compute_cycles_per_k_slice
    auto exec_config = make_executor_config();
    ConcurrentTimingExecutor executor(exec_config);

    auto gen_config = make_generator_config(
        32, MatMulScheduleGenerator::Strategy::INTERLEAVED_AB);
    MatMulScheduleGenerator generator(gen_config);
    auto schedule = generator.generate();
    REQUIRE(schedule.valid);

    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);

    Cycle expected_latency =
        exec_config.compute_latency + exec_config.compute_cycles_per_k_slice;

    size_t complete_events = 0;
    for (const auto& event : executor.events()) {
        if (event.type == EventType::COMPUTE_COMPLETE) {
            complete_events++;
            REQUIRE(event.duration == expected_latency);
        }
    }
    REQUIRE(complete_events == 4);  // 2x2 C tiles
}
