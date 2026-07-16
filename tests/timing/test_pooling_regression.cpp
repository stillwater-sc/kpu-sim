// ============================================================================
// tests/timing/test_pooling_regression.cpp
// Pooling execution regression: pool-type x shape x envelope matrix with value
// checks vs the host oracle, per-stage tile accounting, credit conservation,
// stall invariants, an envelope-refusal boundary, and characterization
// (issue #195, epic E7).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
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

struct EnvelopeMode { const char* name; Size l3, l2; bool part; };
constexpr EnvelopeMode kEnvelopes[] = {
    {"default",     32, 64, false},
    {"minimum",     12, 12, false},   // share 3 == working set
    {"partitioned", 32, 64, true},
};

struct CharRow { std::string cell; std::size_t out_tiles; Cycle cycles, stalls; double dma, str; };
std::vector<CharRow>& characterization() { static std::vector<CharRow> r; return r; }

ConcurrentTimingExecutor::Config exec_cfg(const EnvelopeMode& e) {
    ConcurrentTimingExecutor::Config c;
    c.l3_buffer_count = e.l3; c.l2_bank_count = e.l2;
    c.partition_l3_credits = e.part; c.partition_l2_credits = e.part;
    c.max_cycles = 2'000'000;
    return c;
}

// Shared invariant + characterization check after a run.
void check_invariants(const ScheduleResult& schedule, ConcurrentTimingExecutor& exec,
                      const ExecutionResult& result, const EnvelopeMode& e,
                      const std::string& cell) {
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());

    const auto stats = exec.get_statistics();
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) ==
            schedule.count_ops(ScheduleOpType::DRAIN));  // #139 lock-in
    REQUIRE(exec.l3_credits().available() == e.l3);
    REQUIRE(exec.l2_credits().available() == e.l2);

    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls + stats.str_credit_stalls;
    const auto& ec = exec.config();
    const Cycle cap = static_cast<Cycle>(ec.num_dma_engines + ec.num_block_movers +
                                         ec.num_row_streamers + ec.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * cap);

    const std::size_t out_tiles = schedule.count_ops(ScheduleOpType::STORE);
    characterization().push_back({cell, out_tiles, result.total_cycles, stalls,
                                  stats.dma_utilization(), stats.str_utilization()});
}

Pool2DGeometry geom(Size N, Size C, Size H, Size W, Size Kh, Size Kw, Size s, Size p) {
    Pool2DGeometry g; g.N = N; g.C = C; g.H = H; g.W = W;
    g.Kh = Kh; g.Kw = Kw; g.stride_h = g.stride_w = s; g.pad_h = g.pad_w = p;
    return g;
}

// Windowed cell: seed window rows, reduce, value-check + invariants.
void run_windowed_cell(const Pool2DGeometry& g, PoolType type, const EnvelopeMode& e,
                       const std::string& cell) {
    const Size Ti = 16, K = g.window(), M = g.out_spatial();
    auto input = fill(g.elems(), 7, -1.5f, 0.5f);
    const auto ref = pool2d_reference(input, g, type);
    std::vector<std::vector<float>> win(g.C);
    for (Size c = 0; c < g.C; ++c) win[c] = pool_window_channel(input, g, 0, c, type).rows;

    PoolingScheduleGenerator::Config cfg;
    cfg.geom = g; cfg.pool_type = type; cfg.Ti = Ti;
    cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED;
    cfg.l3_buffer_count = e.l3; cfg.l2_bank_count = e.l2;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    auto schedule = PoolingScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor exec(exec_cfg(e));
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        std::vector<float> blk(static_cast<std::size_t>(Ti) * K);
        for (Size r = 0; r < Ti; ++r)
            for (Size k = 0; k < K; ++k)
                blk[r * K + k] = win[id.ti][static_cast<std::size_t>(id.tj * Ti + r) * K + k];
        exec.set_tile_payload(id, TilePayload{Ti, K, std::move(blk)});
    }

    ScheduleExecutor se(exec);
    se.set_functional_compute_binder(
        [type, K](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;
            spec.operation = [type, K](const std::vector<TilePayload>& in) {
                const auto& x = in.at(0);
                TilePayload out{x.rows, 1, std::vector<float>(x.rows)};
                for (Size r = 0; r < x.rows; ++r) {
                    const float* row = &x.values[static_cast<std::size_t>(r) * K];
                    if (type == PoolType::MAX) {
                        float m = -std::numeric_limits<float>::infinity();
                        for (Size k = 0; k < K; ++k) m = std::max(m, row[k]);
                        out.values[r] = m;
                    } else {
                        float s = 0.0f;
                        for (Size k = 0; k < K; ++k) s += row[k];
                        out.values[r] = s / static_cast<float>(K);
                    }
                }
                return out;
            };
            return spec;
        });
    auto result = se.execute(schedule);
    check_invariants(schedule, exec, result, e, cell);

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < Ti; ++r) {
            const double got = p.values[r];
            REQUIRE(std::isfinite(got));
            max_err = std::max(max_err,
                std::abs(got - static_cast<double>(ref[static_cast<std::size_t>(id.ti) * M + id.tj * Ti + r])));
        }
    }
    REQUIRE(max_err < 1e-4);
}

} // namespace

TEST_CASE("Pooling execution matrix: pool-type x shape x envelope",
          "[timing][regression][pooling][matrix]") {
    struct Case { const char* name; Pool2DGeometry g; PoolType type; };
    const Case cases[] = {
        {"max_2x2s2", geom(1, 8, 8, 8, 2, 2, 2, 0), PoolType::MAX},
        {"max_3x3s2p1", geom(1, 4, 16, 16, 3, 3, 2, 1), PoolType::MAX},
        {"avg_2x2s2", geom(1, 8, 8, 8, 2, 2, 2, 0), PoolType::AVG},
    };
    for (const auto& c : cases)
        for (const auto& e : kEnvelopes)
            DYNAMIC_SECTION(c.name << "/" << e.name) {
                run_windowed_cell(c.g, c.type, e, std::string(c.name) + "/" + e.name);
            }
}

TEST_CASE("Pooling envelope refusal boundary (working set 3)",
          "[timing][regression][pooling][envelope]") {
    PoolingScheduleGenerator::Config cfg;
    cfg.geom = geom(1, 8, 8, 8, 2, 2, 2, 0); cfg.Ti = 16;
    cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED;
    cfg.l3_buffer_count = 12; cfg.l2_bank_count = 12;  // share 3 == working set
    REQUIRE(PoolingScheduleGenerator(cfg).generate().valid);
    cfg.l3_buffer_count = 11; cfg.l2_bank_count = 11;  // share 2 < 3
    auto refused = PoolingScheduleGenerator(cfg).generate();
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("working set") != std::string::npos);
}

TEST_CASE("Pooling characterization report", "[timing][regression][pooling][report]") {
    const auto& rows = characterization();
    if (rows.empty()) { SUCCEED("matrix did not run"); return; }
    std::printf("\n%-22s %8s %9s %8s %6s %6s\n", "cell", "outtiles", "cycles",
                "stalls", "dma%", "str%");
    for (const auto& r : rows)
        std::printf("%-22s %8zu %9llu %8llu %6.1f %6.1f\n", r.cell.c_str(),
                    static_cast<size_t>(r.out_tiles),
                    static_cast<unsigned long long>(r.cycles),
                    static_cast<unsigned long long>(r.stalls),
                    100.0 * r.dma, 100.0 * r.str);
    SUCCEED("characterization recorded");
}
