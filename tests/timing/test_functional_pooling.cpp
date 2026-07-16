// ============================================================================
// tests/timing/test_functional_pooling.cpp
// Value-producing pooling on the CSP executor vs a host oracle (E7-T4, #194).
//
// Seeds the per-channel window rows (max/avg) or plane chunks (global average)
// as tile payloads, executes the PoolingScheduleGenerator schedule through the
// ScheduleExecutor functional-compute binder applying the reduce (MAX / MEAN),
// and checks the drained outputs elementwise against pool2d_reference /
// global_avg_pool_reference. This is the pooling.functional coverage gate - the
// last M2 ResNet gate cell.
//
// Windowed cases use N=1 so a tile of output positions stays within one batch;
// avg-pool cases use pad=0 so the window count is Kh*Kw (count-excludes-padding
// with a window operand is covered host-side in test_pooling_window).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

std::vector<float> fill(std::size_t n, int period, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = base + step * static_cast<float>(i % static_cast<std::size_t>(period));
    return v;
}

PoolingScheduleGenerator::Config windowed_cfg(const Pool2DGeometry& g, PoolType type,
                                              Size Ti) {
    PoolingScheduleGenerator::Config cfg;
    cfg.geom = g; cfg.pool_type = type; cfg.Ti = Ti;
    cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    return cfg;
}

// Windowed max/avg pool: N=1. Returns max abs error vs the host oracle.
double run_windowed(const Pool2DGeometry& g, PoolType type, Size Ti) {
    REQUIRE(g.N == 1);
    REQUIRE(g.out_spatial() % Ti == 0);  // whole tiles per channel
    auto input = fill(g.elems(), 7, -1.5f, 0.5f);
    const auto ref = pool2d_reference(input, g, type);  // [1,C,Hout,Wout]
    const Size K = g.window(), M = g.out_spatial();

    // Per-channel window matrices (rows [M, K]).
    std::vector<std::vector<float>> win(g.C);
    for (Size c = 0; c < g.C; ++c)
        win[c] = pool_window_channel(input, g, 0, c, type).rows;

    auto cfg = windowed_cfg(g, type, Ti);
    auto schedule = PoolingScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg; ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    // Seed each input tile (c, ti) = the [Ti, K] window block for positions
    // [ti*Ti, ti*Ti+Ti).
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        const Size c = id.ti, ti = id.tj;
        std::vector<float> blk(static_cast<std::size_t>(Ti) * K);
        for (Size r = 0; r < Ti; ++r) {
            const Size m = ti * Ti + r;
            for (Size k = 0; k < K; ++k)
                blk[r * K + k] = win[c][static_cast<std::size_t>(m) * K + k];
        }
        exec.set_tile_payload(id, TilePayload{Ti, K, std::move(blk)});
    }

    ScheduleExecutor sched_exec(exec);
    sched_exec.set_functional_compute_binder(
        [type, K](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;  // {window tile}
            spec.operation = [type, K](const std::vector<TilePayload>& in) {
                const auto& x = in.at(0);           // [Ti, K]
                const Size rows = x.rows;
                TilePayload out{rows, 1, std::vector<float>(rows)};
                for (Size r = 0; r < rows; ++r) {
                    const float* row = &x.values[static_cast<std::size_t>(r) * K];
                    if (type == PoolType::MAX) {
                        float m = -std::numeric_limits<float>::infinity();
                        for (Size k = 0; k < K; ++k) m = std::max(m, row[k]);
                        out.values[r] = m;
                    } else {
                        float s = 0.0f;
                        for (Size k = 0; k < K; ++k) s += row[k];
                        out.values[r] = s / static_cast<float>(K);  // pad=0 -> count=K
                    }
                }
                return out;
            };
            return spec;
        });

    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const Size c = id.ti, ti = id.tj;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < Ti; ++r) {
            const Size m = ti * Ti + r;
            const double got = p.values[r];
            REQUIRE(std::isfinite(got));
            const double want = ref[(static_cast<std::size_t>(c) * M) + m];
            max_err = std::max(max_err, std::abs(got - want));
        }
    }
    return max_err;
}

// Global average pool: mean over the H*W plane per (n,c). Returns max abs error.
double run_global_avg(const Pool2DGeometry& g, Size Ti) {
    REQUIRE((g.H * g.W) % Ti == 0);
    auto input = fill(g.elems(), 5, 0.0f, 0.5f);
    const auto ref = global_avg_pool_reference(input, g);  // [N*C]
    const Size plane = g.H * g.W;

    PoolingScheduleGenerator::Config cfg;
    cfg.geom = g; cfg.mode = PoolingScheduleGenerator::Mode::GLOBAL_AVG; cfg.Ti = Ti;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    auto schedule = PoolingScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg; ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    // Seed each plane tile (ti = n*C+c, tj = chunk) with Ti plane elements.
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        const Size nc = id.ti, chunk = id.tj;
        const std::size_t base = static_cast<std::size_t>(nc) * plane + chunk * Ti;
        exec.set_tile_payload(id, TilePayload{Ti, 1,
            std::vector<float>(input.begin() + static_cast<std::ptrdiff_t>(base),
                               input.begin() + static_cast<std::ptrdiff_t>(base + Ti))});
    }

    ScheduleExecutor sched_exec(exec);
    sched_exec.set_functional_compute_binder(
        [plane](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;  // all plane chunks
            spec.operation = [plane](const std::vector<TilePayload>& in) {
                float s = 0.0f;
                for (const auto& t : in)
                    for (float v : t.values) s += v;
                return TilePayload{1, 1, {s / static_cast<float>(plane)}};
            };
            return spec;
        });

    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const Size nc = id.ti;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        const double got = p.values.at(0);
        REQUIRE(std::isfinite(got));
        max_err = std::max(max_err, std::abs(got - ref[nc]));
    }
    return max_err;
}

Pool2DGeometry geom(Size N, Size C, Size H, Size W, Size Kh, Size Kw, Size s, Size p) {
    Pool2DGeometry g; g.N = N; g.C = C; g.H = H; g.W = W;
    g.Kh = Kh; g.Kw = Kw; g.stride_h = g.stride_w = s; g.pad_h = g.pad_w = p;
    return g;
}

} // namespace

TEST_CASE("Pooling functional: max/avg/global on CSP executor match host oracle",
          "[timing][pooling][functional]") {
    SECTION("max 2x2 s2 (8x8 -> 4x4, 16 pos/channel)") {
        REQUIRE(run_windowed(geom(1, 8, 8, 8, 2, 2, 2, 0), PoolType::MAX, 16) < 1e-4);
    }
    SECTION("max 3x3 s2 pad1 (16x16 -> 8x8, 64 pos/channel, padded)") {
        REQUIRE(run_windowed(geom(1, 4, 16, 16, 3, 3, 2, 1), PoolType::MAX, 16) < 1e-4);
    }
    SECTION("avg 2x2 s2, no padding") {
        REQUIRE(run_windowed(geom(1, 8, 8, 8, 2, 2, 2, 0), PoolType::AVG, 16) < 1e-4);
    }
    SECTION("global average pool (1x16x8x8)") {
        REQUIRE(run_global_avg(geom(1, 16, 8, 8, 1, 1, 1, 0), 16) < 1e-4);
    }
    SECTION("global average pool, batch 2 (2x8x8x8)") {
        REQUIRE(run_global_avg(geom(2, 8, 8, 8, 1, 1, 1, 0), 16) < 1e-4);
    }
}
