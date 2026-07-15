// ============================================================================
// tests/timing/test_batchnorm_regression.cpp
// BatchNorm inference execution regression: shape x envelope matrix with
// per-stage tile accounting, credit conservation, stall invariants, an exact
// 2C+1 refusal boundary, a functional-under-credit-pressure oracle cell, and
// performance characterization (issue #182, epic E9).
//
// The fold makes the all-channel-preload working set 2C+1, so each envelope is
// sized to the case's C (min(l3,l2)/4 >= 2C+1).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

struct BNCase {
    const char* name;
    Size N, C, H, W, Ti;
};

// Spread of shapes; the last has a non-tile-aligned spatial extent (49).
constexpr BNCase kCases[] = {
    {"1x4x8x8",    1, 4,  8,  8, 16},
    {"2x8x4x4",    2, 8,  4,  4, 16},
    {"1x16x8x8",   1, 16, 8,  8, 16},
    {"1x8x7x7",    1, 8,  7,  7, 16},  // spatial 49, partial trailing tile
};

// share = min(l3,l2)/4 must be >= 2C+1.
struct EnvelopeMode { const char* name; bool minimum; bool partitioned; };
constexpr EnvelopeMode kEnvelopes[] = {
    {"ample",       false, false},
    {"minimum",     true,  false},
    {"partitioned", false, true},
};

Size credits_for(Size C, bool minimum) {
    const Size need = 4 * (2 * C + 1);            // share == 2C+1
    return minimum ? need : need + 4 * (2 * C + 1);  // ample = 2x share
}

BatchNormScheduleGenerator::Config gen_config(const BNCase& c, const EnvelopeMode& e) {
    BatchNormScheduleGenerator::Config config;
    config.N = c.N; config.C = c.C; config.H = c.H; config.W = c.W;
    config.Ti = c.Ti; config.Tj = c.Ti; config.training = false;
    const Size credits = credits_for(c.C, e.minimum);
    config.l3_buffer_count = credits;
    config.l2_bank_count = credits;
    config.input_base  = 0x00100000;
    config.output_base = 0x00400000;
    config.scale_base  = 0x00010000;
    config.shift_base  = 0x00020000;
    return config;
}

ConcurrentTimingExecutor::Config exec_config(const BNCase& c, const EnvelopeMode& e) {
    ConcurrentTimingExecutor::Config config;
    const Size credits = credits_for(c.C, e.minimum);
    config.l3_buffer_count = credits;
    config.l2_bank_count = credits;
    config.partition_l3_credits = e.partitioned;
    config.partition_l2_credits = e.partitioned;
    config.max_cycles = 5'000'000;
    return config;
}

struct CellResult {
    std::string shape, envelope;
    std::size_t out_tiles = 0;  // count_ops is size_t (MSVC C4267)
    Cycle cycles = 0;
    double cycles_per_tile = 0.0;
    Cycle stalls = 0;
    double dma_util = 0.0, str_util = 0.0;
};

std::vector<CellResult>& characterization() {
    static std::vector<CellResult> rows;
    return rows;
}

void run_cell(const BNCase& c, const EnvelopeMode& e) {
    auto schedule = BatchNormScheduleGenerator(gen_config(c, e)).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor executor(exec_config(c, e));
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("shape=" << c.name << " envelope=" << e.name
         << " cycles=" << result.total_cycles << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());  // generation envelope == execution envelope

    const auto stats = executor.get_statistics();

    // Per-stage tile accounting (a tile reaches L3 from LOAD and from WRITEBACK).
    REQUIRE(stats.tiles_loaded == schedule.count_ops(ScheduleOpType::LOAD) +
                                  schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_moved == schedule.count_ops(ScheduleOpType::MOVE));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_writeback == schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));

    // A COMPUTE per output tile (the #139 fix): no DRAIN without a producer.
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) ==
            schedule.count_ops(ScheduleOpType::DRAIN));

    // Credit conservation: every L3/L2 credit returned (broadcast params too).
    REQUIRE(executor.l3_credits().available() == exec_config(c, e).l3_buffer_count);
    REQUIRE(executor.l2_credits().available() == exec_config(c, e).l2_bank_count);

    // Stall sanity.
    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls +
                         stats.str_credit_stalls;
    const auto& ec = executor.config();
    const Cycle stall_capable =
        static_cast<Cycle>(ec.num_dma_engines + ec.num_block_movers +
                           ec.num_row_streamers + ec.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * stall_capable);

    const std::size_t out_tiles = schedule.count_ops(ScheduleOpType::STORE);
    characterization().push_back(
        {c.name, e.name, out_tiles, result.total_cycles,
         static_cast<double>(result.total_cycles) / static_cast<double>(out_tiles),
         stalls, stats.dma_utilization(), stats.str_utilization()});
}

} // namespace

TEST_CASE("BatchNorm execution matrix: shape x envelope",
          "[timing][regression][batchnorm][matrix]") {
    for (const auto& c : kCases)
        for (const auto& e : kEnvelopes)
            DYNAMIC_SECTION(c.name << "/" << e.name) {
                run_cell(c, e);
            }
}

TEST_CASE("BatchNorm envelope refusal boundary is exact (2C+1)",
          "[timing][regression][batchnorm][envelope]") {
    BNCase c{"1x4x8x8", 1, 4, 8, 8, 16};  // 2C+1 = 9
    auto cfg = gen_config(c, {"", true, false});

    cfg.l3_buffer_count = 36; cfg.l2_bank_count = 36;  // share 9 == working set
    REQUIRE(BatchNormScheduleGenerator(cfg).generate().valid);

    cfg.l3_buffer_count = 35; cfg.l2_bank_count = 35;  // share 8 < 9
    auto refused = BatchNormScheduleGenerator(cfg).generate();
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("working set") != std::string::npos);
}

TEST_CASE("BatchNorm functional correctness under credit pressure",
          "[timing][regression][batchnorm][functional]") {
    // Values must survive the minimum partitioned envelope: every tile contends
    // for its partition. A credit bug that reorders/drops a tile shows up as a
    // value error against the direct 4-param oracle.
    BatchNormGeometry g; g.N = 1; g.C = 4; g.H = 8; g.W = 8;  // spatial 64
    const Size Ti = 16;
    BNCase c{"1x4x8x8", 1, 4, 8, 8, 16};
    EnvelopeMode e{"minimum-partitioned", true, true};

    std::vector<float> input(g.elems()), gamma(g.C), beta(g.C), mean(g.C), var(g.C);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = -1.0f + 0.5f * static_cast<float>(i % 7);
    for (Size ch = 0; ch < g.C; ++ch) {
        gamma[ch] = 0.5f + 0.5f * static_cast<float>(ch % 4);
        beta[ch]  = -1.0f + 0.75f * static_cast<float>(ch % 3);
        mean[ch]  = 0.25f + 0.5f * static_cast<float>(ch % 5);
        var[ch]   = 0.5f + 0.25f * static_cast<float>(ch % 4);
    }
    const float eps = 1e-3f;
    const auto affine = bn_fold(gamma, beta, mean, var, eps);
    const auto ref = batchnorm_reference(input, gamma, beta, mean, var, eps, g);

    auto schedule = BatchNormScheduleGenerator(gen_config(c, e)).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor exec(exec_config(c, e));
    const Size spatial = g.spatial();
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A) {
            const Size n = id.ti / g.C, ch = id.ti % g.C, si = id.tj;
            const std::size_t base =
                (static_cast<std::size_t>(n) * g.C + ch) * spatial + si * Ti;
            exec.set_tile_payload(id, TilePayload{Ti, 1,
                std::vector<float>(input.begin() + static_cast<std::ptrdiff_t>(base),
                                   input.begin() + static_cast<std::ptrdiff_t>(base + Ti))});
        } else {
            const Size ch = id.ti;
            const bool is_scale = (id.tj == 4);  // ParamType::SCALE
            exec.set_tile_payload(id, TilePayload{1, 1,
                {is_scale ? affine.scale[ch] : affine.shift[ch]}});
        }
    }

    ScheduleExecutor sched_exec(exec);
    sched_exec.set_functional_compute_binder(
        [](const ScheduleOperation& compute_op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = compute_op.dependency_tiles;
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

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const Size n = id.ti / g.C, ch = id.ti % g.C, si = id.tj;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        const std::size_t base =
            (static_cast<std::size_t>(n) * g.C + ch) * spatial + si * Ti;
        for (Size i = 0; i < Ti; ++i) {
            const double got = p.values[i];
            REQUIRE(std::isfinite(got));
            max_err = std::max(max_err, std::abs(got - ref[base + i]));
        }
    }
    REQUIRE(max_err < 1e-3);
}

// Runs last: prints the characterization table gathered by the matrix test.
TEST_CASE("BatchNorm characterization report",
          "[timing][regression][batchnorm][report]") {
    const auto& rows = characterization();
    if (rows.empty()) { SUCCEED("matrix test did not run in this invocation"); return; }
    std::printf("\n%-12s %-16s %8s %9s %10s %8s %6s %6s\n",
                "shape", "envelope", "outtiles", "cycles", "cyc/tile",
                "stalls", "dma%", "str%");
    for (const auto& r : rows) {
        std::printf("%-12s %-16s %8zu %9llu %10.1f %8llu %6.1f %6.1f\n",
                    r.shape.c_str(), r.envelope.c_str(),
                    static_cast<size_t>(r.out_tiles),
                    static_cast<unsigned long long>(r.cycles),
                    r.cycles_per_tile,
                    static_cast<unsigned long long>(r.stalls),
                    100.0 * r.dma_util, 100.0 * r.str_util);
    }
    SUCCEED("characterization recorded");
}
