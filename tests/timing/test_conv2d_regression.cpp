// ============================================================================
// tests/timing/test_conv2d_regression.cpp
// Conv2D execution regression: shape x strategy x envelope matrix with
// per-stage tile accounting, credit conservation, stall invariants, and
// performance characterization (issue #123, epic E6).
//
// Every cell must execute to completion with exact per-stage tile accounting
// and full credit return - a leaked credit or an unconsumed tile anywhere in
// the pipeline fails the cell, not just a livelock. A functional-under-credit-
// pressure cell additionally checks values survive the minimum partitioned
// envelope.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

using Strategy = Conv2DScheduleGenerator::Strategy;

struct ConvCase {
    const char* name;
    Size N, C_in, H, W, C_out, Kh, Kw, stride, pad;
};

// A spread of shapes: ResNet-style 3x3, pointwise 1x1, strided downsample,
// large receptive field, batched, and a non-tile-aligned case (partial tiles).
constexpr ConvCase kConv[] = {
    {"3x3_s1p1",    1, 32,  8,  8, 16, 3, 3, 1, 1},
    {"1x1_pw",      1, 64,  8,  8, 32, 1, 1, 1, 0},
    {"3x3_s2p1",    1, 16, 16, 16, 32, 3, 3, 2, 1},
    {"5x5_s1p2",    1, 16,  8,  8, 16, 5, 5, 1, 2},
    {"batch2_3x3",  2, 16,  8,  8, 16, 3, 3, 1, 1},
    {"non_aligned", 1, 10,  7,  7, 12, 3, 3, 1, 1},  // M/K/Cout not multiples of 16
};

constexpr Strategy kStrategies[] = {
    Strategy::IM2COL_INTERLEAVED,
    Strategy::IM2COL_OUTPUT_STATIONARY,
};

const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::IM2COL_INTERLEAVED:       return "interleaved";
        case Strategy::IM2COL_OUTPUT_STATIONARY: return "output_stationary";
        case Strategy::DIRECT_CONV:              return "direct";
    }
    return "?";
}

struct EnvelopeCase {
    const char* name;
    Size l3;
    Size l2;
    bool partitioned;
};

// default, the MINIMUM safe envelope (conv working set = 3, share =
// min(12,12)/4 = 3; one credit less refuses generation), and partitioned.
constexpr EnvelopeCase kEnvelopes[] = {
    {"default",         32, 64, false},
    {"constrained-min", 12, 12, false},
    {"partitioned",     32, 64, true},
};

Conv2DScheduleGenerator::Config gen_config(const ConvCase& c, Strategy s,
                                           const EnvelopeCase& env) {
    Conv2DScheduleGenerator::Config config;
    config.N = c.N; config.H_in = c.H; config.W_in = c.W; config.C_in = c.C_in;
    config.C_out = c.C_out; config.Kh = c.Kh; config.Kw = c.Kw;
    config.stride_h = c.stride; config.stride_w = c.stride;
    config.padding_h = c.pad; config.padding_w = c.pad;
    config.Ti = 16; config.Tj = 16; config.Tk = 16;
    config.strategy = s;
    config.l3_buffer_count = env.l3;
    config.l2_bank_count = env.l2;
    config.input_base = 0x100000; config.filter_base = 0x400000; config.output_base = 0x700000;
    return config;
}

ConcurrentTimingExecutor::Config exec_config(const EnvelopeCase& env) {
    ConcurrentTimingExecutor::Config config;
    config.l3_buffer_count = env.l3;
    config.l2_bank_count = env.l2;
    config.partition_l3_credits = env.partitioned;
    config.partition_l2_credits = env.partitioned;
    config.max_cycles = 5'000'000;
    return config;
}

struct CellResult {
    std::string shape, strategy, envelope;
    Size c_tiles = 0;
    Cycle cycles = 0;
    double cycles_per_ctile = 0.0;
    Cycle stalls = 0;
    double dma_util = 0.0, str_util = 0.0;
};

std::vector<CellResult>& characterization() {
    static std::vector<CellResult> rows;
    return rows;
}

void run_cell(const ConvCase& c, Strategy s, const EnvelopeCase& env) {
    auto schedule = Conv2DScheduleGenerator(gen_config(c, s, env)).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor executor(exec_config(env));
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("shape=" << c.name << " strategy=" << strategy_name(s)
         << " envelope=" << env.name << " cycles=" << result.total_cycles
         << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());  // generation envelope == execution envelope

    const auto stats = executor.get_statistics();

    // Per-stage tile accounting. A tile reaches L3 from both directions: DMA
    // loads (downstream) and BlockMover writebacks (upstream, C tiles).
    REQUIRE(stats.tiles_loaded == schedule.count_ops(ScheduleOpType::LOAD) +
                                  schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_moved == schedule.count_ops(ScheduleOpType::MOVE));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_writeback == schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));

    // A COMPUTE per output tile (the #139 fix) — DRAIN never lacks a producer.
    REQUIRE(schedule.count_ops(ScheduleOpType::COMPUTE) ==
            schedule.count_ops(ScheduleOpType::DRAIN));

    // Credit conservation: every L3/L2 credit is returned by the end.
    REQUIRE(executor.l3_credits().available() == env.l3);
    REQUIRE(executor.l2_credits().available() == env.l2);

    // Stall sanity: the pipeline cannot have stalled every cycle across all
    // stall-capable components.
    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls +
                         stats.str_credit_stalls;
    const auto& ec = executor.config();
    const Cycle stall_capable =
        static_cast<Cycle>(ec.num_dma_engines + ec.num_block_movers +
                           ec.num_row_streamers + ec.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * stall_capable);

    const Size c_tiles = schedule.count_ops(ScheduleOpType::STORE);
    characterization().push_back(
        {c.name, strategy_name(s), env.name, c_tiles, result.total_cycles,
         static_cast<double>(result.total_cycles) / static_cast<double>(c_tiles),
         stalls, stats.dma_utilization(), stats.str_utilization()});
}

} // namespace

TEST_CASE("Conv2D execution matrix: shape x strategy x envelope",
          "[timing][regression][conv2d][matrix]") {
    for (const auto& c : kConv)
        for (Strategy s : kStrategies)
            for (const auto& env : kEnvelopes)
                DYNAMIC_SECTION(c.name << "/" << strategy_name(s) << "/" << env.name) {
                    run_cell(c, s, env);
                }
}

TEST_CASE("Conv2D envelope refusal boundary is exact",
          "[timing][regression][conv2d][envelope]") {
    // conv working set = 3; share = min(l3,l2)/4: 12 -> 3 (generates),
    // 11 -> 2 (< working set, refused a priori).
    ConvCase c{"3x3_s1p1", 1, 32, 8, 8, 16, 3, 3, 1, 1};
    EnvelopeCase safe{"safe", 12, 12, false};
    EnvelopeCase unsafe{"unsafe", 11, 11, false};

    REQUIRE(Conv2DScheduleGenerator(gen_config(c, Strategy::IM2COL_INTERLEAVED, safe))
                .generate().valid);
    auto refused =
        Conv2DScheduleGenerator(gen_config(c, Strategy::IM2COL_INTERLEAVED, unsafe)).generate();
    REQUIRE_FALSE(refused.valid);
    REQUIRE(refused.error_message.find("working set") != std::string::npos);
}

TEST_CASE("Conv2D functional correctness under credit pressure",
          "[timing][regression][conv2d][functional]") {
    // Values must survive the minimum partitioned envelope: 12 credits, every
    // tile contends for its partition. A credit bug that reorders or drops a
    // tile shows up as a value error against the direct-conv oracle.
    Conv2DGeometry g;
    g.N = 1; g.C_in = 16; g.H_in = 8; g.W_in = 8; g.C_out = 16; g.Kh = 3; g.Kw = 3;
    g.pad_h = 1; g.pad_w = 1;  // M=64, K=144, Cout=16 (all multiples of 16)
    const Size T = 16;
    EnvelopeCase env{"constrained-partitioned", 12, 12, true};
    ConvCase c{"3x3_s1p1", 1, 16, 8, 8, 16, 3, 3, 1, 1};

    std::vector<float> input(g.input_elems()), filter(g.filter_elems());
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.5f + 0.5f * static_cast<float>(i % 7);
    for (std::size_t i = 0; i < filter.size(); ++i)
        filter[i] = -1.0f + 0.5f * static_cast<float>(i % 5);
    const auto a_col = im2col_nchw(input, g);
    const auto b_w = filter_to_bw_nchw(filter, g);
    const auto ref = conv2d_reference(input, filter, {}, g, false);

    auto schedule = Conv2DScheduleGenerator(
        gen_config(c, Strategy::IM2COL_INTERLEAVED, env)).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor exec(exec_config(env));
    auto block = [&](const std::vector<float>& mat, Size cols, Size br, Size bc) {
        std::vector<float> b(static_cast<std::size_t>(T) * T);
        for (Size r = 0; r < T; ++r)
            for (Size cc = 0; cc < T; ++cc)
                b[r * T + cc] = mat[(br * T + r) * cols + (bc * T + cc)];
        return b;
    };
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A)
            exec.set_tile_payload(id, TilePayload{T, T, block(a_col, g.K(), id.ti, id.tk)});
        else
            exec.set_tile_payload(id, TilePayload{T, T, block(b_w, g.C_out, id.tk, id.tj)});
    }
    for (const auto& op : schedule.operations) {
        switch (op.type) {
            case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
            case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
            case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
            case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
            case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
            case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
            case ScheduleOpType::COMPUTE: {
                ConcurrentTimingExecutor::MatMulComputeSpec spec;
                for (const auto& dep : op.dependency_tiles) {
                    if (dep.matrix == MatrixID::A) spec.a_tiles.push_back(dep);
                    else                            spec.b_tiles.push_back(dep);
                }
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles)
        exec.step();
    REQUIRE(exec.is_complete());

    const Size Hout = g.H_out(), Wout = g.W_out();
    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size cc = 0; cc < T; ++cc) {
                const Size m = id.ti * T + r, co = id.tj * T + cc;
                const Size n = m / (Hout * Wout), rem = m % (Hout * Wout);
                const Size ho = rem / Wout, wo = rem % Wout;
                const double want =
                    ref[((static_cast<std::size_t>(n) * g.C_out + co) * Hout + ho) *
                            Wout + wo];
                max_err = std::max(max_err, std::abs(p.values[r * T + cc] - want));
            }
    }
    REQUIRE(max_err < 1e-3);
}

// Runs last: prints the characterization table gathered by the matrix test.
TEST_CASE("Conv2D characterization report",
          "[timing][regression][conv2d][report]") {
    const auto& rows = characterization();
    if (rows.empty()) {
        SUCCEED("matrix test did not run in this invocation");
        return;
    }
    std::printf("\n%-12s %-18s %-16s %7s %9s %10s %8s %6s %6s\n",
                "shape", "strategy", "envelope", "c_tiles", "cycles",
                "cyc/ctile", "stalls", "dma%", "str%");
    for (const auto& r : rows) {
        std::printf("%-12s %-18s %-16s %7zu %9llu %10.1f %8llu %6.1f %6.1f\n",
                    r.shape.c_str(), r.strategy.c_str(), r.envelope.c_str(),
                    static_cast<size_t>(r.c_tiles),
                    static_cast<unsigned long long>(r.cycles),
                    r.cycles_per_ctile,
                    static_cast<unsigned long long>(r.stalls),
                    100.0 * r.dma_util, 100.0 * r.str_util);
    }
    SUCCEED("characterization recorded");
}
