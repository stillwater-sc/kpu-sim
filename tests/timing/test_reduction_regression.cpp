// ============================================================================
// tests/timing/test_reduction_regression.cpp
// Streaming-reduction execution regression: op x stream-length x envelope
// with credit/stall invariants and performance characterization
// (issue #108, epic E3).
//
// Invariants stronger than completion: exact per-stage tile accounting and
// full credit conservation - a leaked credit or an unconsumed tile fails
// the cell, not just a livelock.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_reduction_executor.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

using Form = OnlineReductionScheduleGenerator::Form;
using ReduceOp = OnlineReductionScheduleGenerator::ReduceOp;

struct EnvelopeCase { const char* name; Size l3; Size l2; bool partitioned; };

// default, the minimum safe envelope (share = min(8,8)/4 = 2 = the stats
// working set; one less refuses), and per-matrix partitioned credits.
constexpr EnvelopeCase kEnvelopes[] = {
    {"default",         32, 64, false},
    {"constrained-min",  8,  8, false},
    {"partitioned",     32, 64, true},
};

struct SizeCase { const char* name; Size reduction_elems; };
constexpr SizeCase kSizes[] = {
    {"single-tile", 256},
    {"16-tile",     4096},
    {"64-tile",     16384},
    {"non-aligned", 4000},
};

constexpr ReduceOp kOps[] = {ReduceOp::MAX, ReduceOp::SUM, ReduceOp::VAR};

const char* op_name(ReduceOp op) {
    switch (op) {
        case ReduceOp::MAX: return "MAX";
        case ReduceOp::MIN: return "MIN";
        case ReduceOp::SUM: return "SUM";
        case ReduceOp::MEAN: return "MEAN";
        case ReduceOp::VAR: return "VAR";
    }
    return "?";
}

OnlineReductionScheduleGenerator::Config make_config(ReduceOp op,
                                                     const SizeCase& size,
                                                     const EnvelopeCase& env) {
    OnlineReductionScheduleGenerator::Config c;
    c.num_rows = 1;
    c.reduction_elems = size.reduction_elems;
    c.tile_elems = 256;
    c.form = Form::FULL_REDUCE;
    c.op = op;
    c.l3_buffer_count = env.l3;
    c.l2_bank_count = env.l2;
    c.in_base = 0x100000;
    c.stat_base = 0x200000;
    c.out_base = 0x300000;
    return c;
}

ConcurrentTimingExecutor::Config make_executor_config(const EnvelopeCase& env) {
    ConcurrentTimingExecutor::Config config;
    config.l3_buffer_count = env.l3;
    config.l2_bank_count = env.l2;
    config.partition_l3_credits = env.partitioned;
    config.partition_l2_credits = env.partitioned;
    config.max_cycles = 5'000'000;
    return config;
}

struct CellResult {
    std::string op, size, envelope;
    Size tiles = 0;
    Cycle cycles = 0;
    double cycles_per_tile = 0.0;
    Cycle stalls = 0;
    double str_util = 0.0;
};
std::vector<CellResult>& characterization() {
    static std::vector<CellResult> rows;
    return rows;
}

void run_cell(ReduceOp op, const SizeCase& size, const EnvelopeCase& env) {
    auto gen_config = make_config(op, size, env);
    OnlineReductionScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor executor(make_executor_config(env));
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("op=" << op_name(op) << " size=" << size.name << " env=" << env.name
         << " cycles=" << result.total_cycles << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());

    const auto stats = executor.get_statistics();

    // Per-stage tile accounting (tiles reach L3 from LOAD and WRITEBACK)
    REQUIRE(stats.tiles_loaded == schedule.count_ops(ScheduleOpType::LOAD) +
                                  schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_moved == schedule.count_ops(ScheduleOpType::MOVE));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));

    // Credit conservation (aggregates across partitions)
    REQUIRE(executor.l3_credits().available() == env.l3);
    REQUIRE(executor.l2_credits().available() == env.l2);

    // Normalized stall bound: aggregated stalls cannot reach
    // total_cycles x stall-capable components (that would be "stalled every
    // cycle, no work ever done")
    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls +
                         stats.str_credit_stalls;
    const auto& ec = executor.config();
    const Cycle stall_capable = static_cast<Cycle>(
        ec.num_dma_engines + ec.num_block_movers +
        ec.num_row_streamers + ec.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * stall_capable);

    const Size tiles = gen_config.reduction_tiles();
    characterization().push_back(
        {op_name(op), size.name, env.name, tiles, result.total_cycles,
         static_cast<double>(result.total_cycles) / static_cast<double>(tiles),
         stalls, stats.str_utilization()});
}

} // namespace

TEST_CASE("Reduction execution matrix: op x stream-length x envelope",
          "[timing][regression][reduction][matrix]") {
    for (ReduceOp op : kOps) {
        for (const auto& size : kSizes) {
            for (const auto& env : kEnvelopes) {
                DYNAMIC_SECTION(op_name(op) << "/" << size.name << "/" << env.name) {
                    run_cell(op, size, env);
                }
            }
        }
    }
}

TEST_CASE("Reduction envelope refusal boundary is exact",
          "[timing][regression][reduction][envelope]") {
    // share = min(l3,l2)/4: 8 -> 2 (= working set, generates);
    // 4 -> 1 (< working set, refused)
    SizeCase big{"16-tile", 4096};
    auto safe = make_config(ReduceOp::SUM, big, {"boundary-safe", 8, 8, false});
    REQUIRE(OnlineReductionScheduleGenerator(safe).generate().valid);

    auto unsafe = make_config(ReduceOp::SUM, big, {"boundary-unsafe", 4, 4, false});
    auto sched = OnlineReductionScheduleGenerator(unsafe).generate();
    REQUIRE_FALSE(sched.valid);
    REQUIRE(sched.error_message.find("working set") != std::string::npos);
}

TEST_CASE("Reduction values survive the minimum envelope + partitioned credits",
          "[timing][regression][reduction][functional]") {
    const Size n = 4000;   // non-aligned, 16 tiles
    std::vector<float> data(n);
    for (Size i = 0; i < n; ++i) data[i] = -3.0f + 0.0021f * static_cast<float>(i);

    EnvelopeCase env{"constrained-partitioned", 8, 8, true};
    auto gen_config = make_config(ReduceOp::SUM, {"non-aligned", n}, env);
    FunctionalReductionExecutor exec(gen_config, make_executor_config(env));
    auto result = exec.run(data);
    REQUIRE(result.execution.success);
    REQUIRE_FALSE(result.execution.livelock_detected);

    double sum = 0.0; for (float v : data) sum += v;
    REQUIRE(result.stats[0] == Catch::Approx(sum).epsilon(1e-4).margin(1e-3));
}

TEST_CASE("Reduction characterization report",
          "[timing][regression][reduction][report]") {
    const auto& rows = characterization();
    if (rows.empty()) { SUCCEED("matrix test did not run"); return; }
    std::printf("\n%-6s %-12s %-16s %6s %9s %9s %8s %6s\n",
                "op", "size", "envelope", "tiles", "cycles", "cyc/tile",
                "stalls", "str%");
    for (const auto& r : rows) {
        std::printf("%-6s %-12s %-16s %6zu %9llu %9.1f %8llu %6.1f\n",
                    r.op.c_str(), r.size.c_str(), r.envelope.c_str(),
                    static_cast<size_t>(r.tiles),
                    static_cast<unsigned long long>(r.cycles), r.cycles_per_tile,
                    static_cast<unsigned long long>(r.stalls), 100.0 * r.str_util);
    }
    SUCCEED("characterization recorded");
}
