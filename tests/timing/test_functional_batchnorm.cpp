// ============================================================================
// tests/timing/test_functional_batchnorm.cpp
// Value-producing BatchNorm inference on the CSP executor vs a host oracle
// (E9-T4, #181).
//
// Seeds the streamed input tiles and the folded per-channel scale/shift params
// (E9-T2 bn_fold) as tile payloads, executes the BatchNormScheduleGenerator
// inference schedule through the value-producing functional path (the
// ScheduleExecutor functional-compute binder applying y = x*scale + shift), and
// checks every drained output tile elementwise against batchnorm_reference (the
// independent direct 4-param formula). This is the batchnorm.functional gate.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
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

// Size l3/l2 so min/4 >= 2C+1 (all-channel preload residency).
Size envelope_for(Size C) {
    const Size credits = 4 * (2 * C + 1) + 4;
    return credits < 32 ? 32 : credits;
}

// Execute BN inference and return the max abs error vs the host oracle.
double run_and_compare(const BatchNormGeometry& g, Size Ti) {
    REQUIRE(g.spatial() % Ti == 0);  // full tiles only

    // Host operands (bounded) and the folded scale/shift.
    auto input = fill(g.elems(), 7, -1.0f, 0.5f);
    auto gamma = fill(g.C, 4, 0.5f, 0.5f);
    auto beta  = fill(g.C, 3, -1.0f, 0.75f);
    auto mean  = fill(g.C, 5, 0.25f, 0.5f);
    auto var   = fill(g.C, 4, 0.5f, 0.25f);  // all > 0
    const float eps = 1e-3f;
    const auto affine = bn_fold(gamma, beta, mean, var, eps);
    const auto ref = batchnorm_reference(input, gamma, beta, mean, var, eps, g);

    BatchNormScheduleGenerator::Config cfg;
    cfg.N = g.N; cfg.C = g.C; cfg.H = g.H; cfg.W = g.W;
    cfg.Ti = Ti; cfg.Tj = Ti; cfg.training = false;
    const Size credits = envelope_for(g.C);
    cfg.l3_buffer_count = credits; cfg.l2_bank_count = credits;
    auto schedule = BatchNormScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.l3_buffer_count = credits; ecfg.l2_bank_count = credits;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    const Size spatial = g.spatial();
    // Seed input tiles (A: [Ti x 1] spatial slices) and scale/shift params
    // (B: [1 x 1] per channel, distinguished by tj = SCALE/SHIFT ordinal).
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A) {
            const Size n = id.ti / g.C, c = id.ti % g.C, si = id.tj;
            const std::size_t base =
                (static_cast<std::size_t>(n) * g.C + c) * spatial + si * Ti;
            exec.set_tile_payload(id, TilePayload{Ti, 1,
                std::vector<float>(input.begin() + static_cast<std::ptrdiff_t>(base),
                                   input.begin() + static_cast<std::ptrdiff_t>(base + Ti))});
        } else {  // param tile: ti = channel, tj = ParamType ordinal
            // Inference emits only SCALE (ordinal 4) and SHIFT (ordinal 5); see
            // BatchNormScheduleGenerator::ParamType.
            const Size c = id.ti;
            const bool is_scale = (id.tj == 4);
            exec.set_tile_payload(id, TilePayload{1, 1,
                {is_scale ? affine.scale[c] : affine.shift[c]}});
        }
    }

    ScheduleExecutor sched_exec(exec);
    sched_exec.set_functional_compute_binder(
        [](const ScheduleOperation& compute_op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = compute_op.dependency_tiles;  // {input, scale, shift}
            spec.operation = [](const std::vector<TilePayload>& in) {
                const auto& x = in.at(0);
                const float scale = in.at(1).values.at(0);
                const float shift = in.at(2).values.at(0);
                TilePayload out{x.rows, x.cols, std::vector<float>(x.values.size())};
                for (std::size_t i = 0; i < x.values.size(); ++i)
                    out.values[i] = x.values[i] * scale + shift;
                return out;
            };
            return spec;
        });

    auto result = sched_exec.execute(schedule);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);

    // Gather output tiles and compare to the oracle.
    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const Size n = id.ti / g.C, c = id.ti % g.C, si = id.tj;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        const std::size_t base =
            (static_cast<std::size_t>(n) * g.C + c) * spatial + si * Ti;
        for (Size i = 0; i < Ti; ++i) {
            const double got = p.values[i];
            REQUIRE(std::isfinite(got));
            max_err = std::max(max_err, std::abs(got - ref[base + i]));
        }
    }
    return max_err;
}

} // namespace

TEST_CASE("BatchNorm functional: inference on CSP executor matches host oracle",
          "[timing][batchnorm][functional]") {
    SECTION("1x4x8x8 (4 spatial tiles/channel)") {
        BatchNormGeometry g; g.N = 1; g.C = 4; g.H = 8; g.W = 8;
        REQUIRE(run_and_compare(g, 16) < 1e-3);
    }
    SECTION("single spatial tile per channel") {
        BatchNormGeometry g; g.N = 1; g.C = 6; g.H = 4; g.W = 4;  // spatial 16 = Ti
        REQUIRE(run_and_compare(g, 16) < 1e-3);
    }
    SECTION("batch N=2") {
        BatchNormGeometry g; g.N = 2; g.C = 4; g.H = 4; g.W = 8;  // spatial 32
        REQUIRE(run_and_compare(g, 16) < 1e-3);
    }
    SECTION("larger channel count") {
        BatchNormGeometry g; g.N = 1; g.C = 16; g.H = 4; g.W = 4;
        REQUIRE(run_and_compare(g, 16) < 1e-3);
    }
}
