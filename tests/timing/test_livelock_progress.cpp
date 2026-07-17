// ============================================================================
// tests/timing/test_livelock_progress.cpp
// Regression: the livelock detector must count progress across ALL pipeline
// stages - forward (load/move/feed), backward (store/writeback/drain), and
// compute - not just the forward path. A large op spends a long phase draining
// results to DRAM after every input has been fed; counting only load/move/feed
// plateaus there and false-trips the detector on a progressing schedule (the
// depthwise-with-thousands-of-output-tiles case).
//
// This drives a windowed pooling schedule with a deliberately LOW stall
// threshold so its drain-back phase exceeds the threshold quickly: before the
// fix the detector false-tripped, now it completes.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <limits>
#include <optional>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

TEST_CASE("Livelock detector counts backward-path progress (no false trip on drain)",
          "[timing][livelock][regression]") {
    // Windowed pooling over 16 channels @ 8x8: 16 * (16*8*8 / 16) = 1024 output
    // tiles, so the drain-back phase (all inputs fed, results still draining)
    // comfortably exceeds the low stall threshold below.
    PoolingScheduleGenerator::Config cfg;
    cfg.geom.N = 16; cfg.geom.C = 16; cfg.geom.H = 8; cfg.geom.W = 8;
    cfg.geom.Kh = cfg.geom.Kw = 3;
    cfg.geom.stride_h = cfg.geom.stride_w = 1;
    cfg.geom.pad_h = cfg.geom.pad_w = 1;
    cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED; cfg.Ti = 16;
    cfg.l3_buffer_count = 128; cfg.l2_bank_count = 128;
    auto sch = PoolingScheduleGenerator(cfg).generate();
    REQUIRE(sch.valid);

    ConcurrentTimingExecutor::Config ec;
    ec.enable_livelock_detection = true;   // detection ON
    ec.livelock_threshold = 500;           // low: the drain-back phase exceeds it
    ec.l3_buffer_count = 128; ec.l2_bank_count = 128;
    ec.max_cycles = 3'000'000;
    ConcurrentTimingExecutor exec(ec);

    const Size K = cfg.geom.window();
    for (const auto& op : sch.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        exec.set_tile_payload(op.tile.tile_id,
                              TilePayload{16, K, std::vector<float>(16 * K, 1.0f)});
    }

    ScheduleExecutor se(exec);
    se.set_functional_compute_binder(
        [K](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;
            spec.operation = [K](const std::vector<TilePayload>& in) {
                const auto& x = in.at(0);
                TilePayload out{x.rows, 1, std::vector<float>(x.rows)};
                for (Size r = 0; r < x.rows; ++r) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (Size k = 0; k < K; ++k) m = std::max(m, x.values[r * K + k]);
                    out.values[r] = m;
                }
                return out;
            };
            return spec;
        });

    auto result = se.execute(sch);
    // The run must actually last longer than the stall threshold, otherwise the
    // drain-back phase is never long enough to exercise the detector and the
    // regression would pass vacuously.
    REQUIRE(result.total_cycles > ec.livelock_threshold);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(exec.get_statistics().tiles_stored == sch.count_ops(ScheduleOpType::STORE));
}
