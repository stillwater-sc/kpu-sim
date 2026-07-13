// ============================================================================
// tests/timing/test_elementwise_regression.cpp
// Elementwise/broadcast execution regression: form x size x envelope matrix
// with credit/stall invariants and performance characterization
// (issue #103, epic E2).
//
// Every cell of the matrix must execute to completion with exact per-stage
// tile accounting and full credit return - a leaked credit or an unconsumed
// tile anywhere in the pipeline fails the cell, not just a livelock.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_elementwise_executor.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::VEOp;

namespace {

using Form = ElementwiseScheduleGenerator::Form;

struct EnvelopeCase {
    const char* name;
    Size l3;
    Size l2;
    bool partitioned;
};

// Default, the MINIMUM safe envelope (share = min(12,12)/4 = 3 = the
// binary/broadcast working set - one tile less refuses generation), and
// per-matrix partitioned credits.
constexpr EnvelopeCase kEnvelopes[] = {
    {"default",         32, 64, false},
    {"constrained-min", 12, 12, false},
    {"partitioned",     32, 64, true},
};

struct SizeCase {
    const char* name;
    Size num_elements;
};

constexpr SizeCase kSizes[] = {
    {"single-tile", 256},
    {"16-tile",     4096},
    {"64-tile",     16384},
    {"non-aligned", 4000},   // 16 tiles, 160-elem tail
};

constexpr Form kForms[] = {Form::BINARY, Form::BROADCAST_B, Form::UNARY};

const char* form_name(Form form) {
    switch (form) {
        case Form::BINARY:      return "binary";
        case Form::UNARY:       return "unary";
        case Form::BROADCAST_B: return "broadcast_b";
    }
    return "?";
}

ElementwiseScheduleGenerator::Config make_generator_config(
    Size num_elements, Form form, const EnvelopeCase& env) {
    ElementwiseScheduleGenerator::Config config;
    config.num_elements = num_elements;
    config.tile_elems = 256;
    config.form = form;
    config.l3_buffer_count = env.l3;
    config.l2_bank_count = env.l2;
    config.a_base = 0x100000;
    config.b_base = 0x200000;
    config.c_base = 0x300000;
    return config;
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
    std::string form;
    std::string size;
    std::string envelope;
    Size tiles = 0;
    Cycle cycles = 0;
    double cycles_per_tile = 0.0;
    Cycle stall_cycles = 0;
    double dma_util = 0.0;
    double str_util = 0.0;
};

std::vector<CellResult>& characterization() {
    static std::vector<CellResult> rows;
    return rows;
}

/**
 * @brief Execute one matrix cell and enforce the invariants
 *
 * Invariants (beyond completion):
 *  - per-stage tile accounting: executor throughput counters equal the
 *    schedule's op counts exactly (a duplicate or dropped tile at any
 *    stage fails here even if execution completes)
 *  - credit conservation: every L3/L2 credit is returned by the end
 *    (#inserts == #invalidates end-to-end)
 *  - stall sanity: the pipeline cannot have been stalled every cycle
 */
void run_cell(Form form, const SizeCase& size, const EnvelopeCase& env) {
    auto gen_config = make_generator_config(size.num_elements, form, env);
    ElementwiseScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor executor(make_executor_config(env));
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("form=" << form_name(form) << " size=" << size.name
         << " envelope=" << env.name << " cycles=" << result.total_cycles
         << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());   // generation envelope == execution envelope

    const auto stats = executor.get_statistics();

    // Per-stage tile accounting. tiles_loaded counts TILE_ARRIVED_L3
    // events, and a tile reaches L3 from BOTH directions: DMA loads
    // (downstream) and BlockMover writebacks (upstream, C tiles) - so the
    // invariant is LOAD + WRITEBACK, still exact.
    REQUIRE(stats.tiles_loaded == schedule.count_ops(ScheduleOpType::LOAD) +
                                  schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_moved == schedule.count_ops(ScheduleOpType::MOVE));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_writeback == schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));

    // Credit conservation (works in both flat and partition mode:
    // available() aggregates across partitions)
    REQUIRE(executor.l3_credits().available() == env.l3);
    REQUIRE(executor.l2_credits().available() == env.l2);

    // Stall sanity. Stall counters aggregate across ALL components, so
    // they can legitimately exceed total_cycles (e.g. 3707 stalls in a
    // 1678-cycle run when several engines wait concurrently); the true
    // "stalled every cycle" ceiling is total_cycles x stall-capable
    // components. Hitting it would mean no component ever did work.
    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls +
                         stats.str_credit_stalls;
    const auto& exec_config = executor.config();
    const Cycle stall_capable =
        static_cast<Cycle>(exec_config.num_dma_engines +
                           exec_config.num_block_movers +
                           exec_config.num_row_streamers +
                           exec_config.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * stall_capable);

    const Size tiles = gen_config.data_tiles();
    characterization().push_back(
        {form_name(form), size.name, env.name, tiles, result.total_cycles,
         static_cast<double>(result.total_cycles) / static_cast<double>(tiles),
         stalls, stats.dma_utilization(), stats.str_utilization()});
}

} // namespace

TEST_CASE("Elementwise execution matrix: form x size x envelope",
          "[timing][regression][elementwise][matrix]") {
    for (Form form : kForms) {
        for (const auto& size : kSizes) {
            for (const auto& env : kEnvelopes) {
                DYNAMIC_SECTION(form_name(form) << "/" << size.name << "/"
                                                << env.name) {
                    run_cell(form, size, env);
                }
            }
        }
    }
}

TEST_CASE("Envelope refusal boundary is exact",
          "[timing][regression][elementwise][envelope]") {
    // share = min(l3,l2)/4: 12 -> 3 (= working set, generates);
    // 11 -> 2 (< working set, refused a priori)
    EnvelopeCase safe{"boundary-safe", 12, 12, false};
    EnvelopeCase unsafe{"boundary-unsafe", 11, 11, false};

    auto safe_schedule = ElementwiseScheduleGenerator(
        make_generator_config(4096, Form::BINARY, safe)).generate();
    REQUIRE(safe_schedule.valid);

    auto unsafe_schedule = ElementwiseScheduleGenerator(
        make_generator_config(4096, Form::BINARY, unsafe)).generate();
    REQUIRE_FALSE(unsafe_schedule.valid);
    REQUIRE(unsafe_schedule.error_message.find("working set") != std::string::npos);
}

TEST_CASE("Functional correctness under credit pressure",
          "[timing][regression][elementwise][functional]") {
    // Values must survive the MINIMUM envelope with partitioned credits:
    // 12 credits -> 4 per matrix, every tile contends for its partition.
    // A credit bug that reorders or drops a tile shows up as a value error.
    const Size n = 4000;   // non-aligned: 16 tiles, 160-elem tail
    EnvelopeCase env{"constrained-partitioned", 12, 12, true};

    std::vector<float> a(n), b(n);
    for (Size i = 0; i < n; ++i) {
        a[i] = -2.0f + 0.001f * static_cast<float>(i);
        b[i] = 3.0f - 0.0005f * static_cast<float>(i);
    }

    FunctionalElementwiseExecutor executor(
        make_generator_config(n, Form::BINARY, env), make_executor_config(env));
    auto result = executor.run(VEOp::MUL, a, b);
    REQUIRE(result.execution.success);
    REQUIRE_FALSE(result.execution.livelock_detected);

    REQUIRE(result.values.size() == n);
    for (Size i = 0; i < n; ++i) {
        INFO("element " << i);
        REQUIRE(result.values[i] == Catch::Approx(a[i] * b[i]).margin(1e-6f));
    }
}

// Runs last (alphabetical tags do not order; Catch2 runs in declaration
// order within a file): prints the characterization table gathered by the
// matrix test for recording in the epic.
TEST_CASE("Elementwise characterization report",
          "[timing][regression][elementwise][report]") {
    const auto& rows = characterization();
    if (rows.empty()) {
        SUCCEED("matrix test did not run in this invocation");
        return;
    }
    std::printf("\n%-12s %-12s %-16s %6s %9s %9s %8s %6s %6s\n",
                "form", "size", "envelope", "tiles", "cycles", "cyc/tile",
                "stalls", "dma%", "str%");
    for (const auto& r : rows) {
        std::printf("%-12s %-12s %-16s %6zu %9llu %9.1f %8llu %6.1f %6.1f\n",
                    r.form.c_str(), r.size.c_str(), r.envelope.c_str(),
                    static_cast<size_t>(r.tiles),
                    static_cast<unsigned long long>(r.cycles),
                    r.cycles_per_tile,
                    static_cast<unsigned long long>(r.stall_cycles),
                    100.0 * r.dma_util, 100.0 * r.str_util);
    }
    SUCCEED("characterization recorded");
}
