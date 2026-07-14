// ============================================================================
// tests/timing/test_softmax_regression.cpp
// Online softmax execution regression: shape x envelope with credit/stall
// invariants, functional-under-pressure, and a single-pass-vs-multi-pass
// DRAM-traffic comparison - the online-softmax payoff (issue #158, epic E8).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/schedule/functional_softmax_executor.hpp>
#include <sw/kpu/timing/schedule/softmax_schedule_generator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

namespace {

using Realization = OnlineSoftmaxScheduleGenerator::Realization;

struct EnvelopeCase { const char* name; Size l3; Size l2; bool partitioned; };
constexpr EnvelopeCase kEnvelopes[] = {
    {"default",     32,  64, false},
    {"large",      128, 128, false},
    {"partitioned", 32,  64, true},
};

struct SizeCase { const char* name; Size reduction_elems; };
constexpr SizeCase kSizes[] = {
    {"single-tile", 256},
    {"4-tile",      1024},
    {"16-tile",     4096},
    {"non-aligned", 1000},
};

OnlineSoftmaxScheduleGenerator::Config make_config(const SizeCase& s,
                                                   const EnvelopeCase& e) {
    OnlineSoftmaxScheduleGenerator::Config c;
    c.num_rows = 1;
    c.reduction_elems = s.reduction_elems;
    c.tile_elems = 256;
    c.l3_buffer_count = e.l3;
    c.l2_bank_count = e.l2;
    c.in_base = 0x100000; c.stat_base = 0x200000; c.out_base = 0x300000;
    return c;
}

ConcurrentTimingExecutor::Config make_executor_config(const EnvelopeCase& e) {
    ConcurrentTimingExecutor::Config config;
    config.l3_buffer_count = e.l3;
    config.l2_bank_count = e.l2;
    config.partition_l3_credits = e.partitioned;
    config.partition_l2_credits = e.partitioned;
    config.max_cycles = 5'000'000;
    return config;
}

struct CellResult {
    std::string size, envelope, realization;
    Size tiles = 0; Cycle cycles = 0; Cycle stalls = 0;
};
std::vector<CellResult>& characterization() { static std::vector<CellResult> r; return r; }

void run_cell(const SizeCase& size, const EnvelopeCase& env) {
    auto gen_config = make_config(size, env);
    OnlineSoftmaxScheduleGenerator gen(gen_config);
    auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor executor(make_executor_config(env));
    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    INFO("size=" << size.name << " env=" << env.name
         << " strategy=" << schedule.metadata.strategy
         << " cycles=" << result.total_cycles << " error=" << result.error_message);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.livelock_detected);
    REQUIRE(result.warnings.empty());

    const auto stats = executor.get_statistics();
    // Per-stage tile accounting (L3 arrivals = LOAD + WRITEBACK)
    REQUIRE(stats.tiles_loaded == schedule.count_ops(ScheduleOpType::LOAD) +
                                  schedule.count_ops(ScheduleOpType::WRITEBACK));
    REQUIRE(stats.tiles_moved == schedule.count_ops(ScheduleOpType::MOVE));
    REQUIRE(stats.tiles_fed == schedule.count_ops(ScheduleOpType::FEED));
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));

    // Credit conservation (both credit modes)
    REQUIRE(executor.l3_credits().available() == env.l3);
    REQUIRE(executor.l2_credits().available() == env.l2);

    // Normalized stall bound
    const Cycle stalls = stats.dma_credit_stalls + stats.bm_tag_stalls +
                         stats.bm_credit_stalls + stats.str_tag_stalls +
                         stats.str_credit_stalls;
    const auto& ec = executor.config();
    const Cycle stall_capable = static_cast<Cycle>(
        ec.num_dma_engines + ec.num_block_movers +
        ec.num_row_streamers + ec.num_col_streamers);
    REQUIRE(result.total_cycles > 0);
    REQUIRE(stalls < result.total_cycles * stall_capable);

    characterization().push_back(
        {size.name, env.name, schedule.metadata.strategy,
         gen_config.reduction_tiles(), result.total_cycles, stalls});
}

} // namespace

TEST_CASE("Online softmax execution matrix: shape x envelope",
          "[timing][regression][softmax][matrix]") {
    for (const auto& size : kSizes) {
        for (const auto& env : kEnvelopes) {
            DYNAMIC_SECTION(size.name << "/" << env.name) {
                run_cell(size, env);
            }
        }
    }
}

TEST_CASE("Online softmax values survive the minimum envelope + partitioned",
          "[timing][regression][softmax][functional]") {
    const Size n = 4096;   // 16 tiles -> RESTREAMED at a tight envelope
    std::vector<float> data(n);
    for (Size i = 0; i < n; ++i) data[i] = -2.0f + 0.001f * static_cast<float>(i);

    EnvelopeCase env{"constrained-partitioned", 16, 16, true};
    auto config = make_config({"16-tile", n}, env);
    REQUIRE(config.realization() == Realization::RESTREAMED);

    FunctionalSoftmaxExecutor exec(config, make_executor_config(env));
    auto result = exec.run(data);
    REQUIRE(result.execution.success);

    // Independent host softmax
    double m = -std::numeric_limits<double>::infinity();
    for (float v : data) m = std::max(m, static_cast<double>(v));
    double l = 0.0; for (float v : data) l += std::exp(static_cast<double>(v) - m);
    for (Size i = 0; i < n; ++i) {
        INFO("i=" << i);
        REQUIRE(result.values[i] ==
                Catch::Approx(std::exp(static_cast<double>(data[i]) - m) / l)
                    .epsilon(1e-5).margin(1e-6));
    }
}

TEST_CASE("Online softmax reads the row fewer times than the multi-pass generator",
          "[timing][regression][softmax][payoff]") {
    // The online-softmax payoff: single-pass stats + resident (m,l) means
    // the row is read once (row-resident); the 4-pass safe-softmax re-reads
    // it every pass. Compare LOAD counts at a shape both support.
    const Size n = 1024;   // 4 tiles

    OnlineSoftmaxScheduleGenerator::Config on;
    on.num_rows = 1; on.reduction_elems = n; on.tile_elems = 256;
    auto online = OnlineSoftmaxScheduleGenerator(on).generate();
    REQUIRE(online.valid);

    SoftmaxScheduleGenerator::Config mp;
    mp.batch_size = 1; mp.reduction_dim = n; mp.Ti = 1; mp.Tj = 256;
    auto multipass = SoftmaxScheduleGenerator(mp).generate();
    REQUIRE(multipass.valid);

    const auto online_loads = online.count_ops(ScheduleOpType::LOAD);
    const auto mp_loads = multipass.count_ops(ScheduleOpType::LOAD);
    std::printf("\nDRAM-read payoff (4-tile row): online LOAD=%zu, multi-pass LOAD=%zu (%.1fx fewer)\n",
                static_cast<size_t>(online_loads), static_cast<size_t>(mp_loads),
                static_cast<double>(mp_loads) / static_cast<double>(online_loads));
    REQUIRE(online_loads < mp_loads);
    // The online row is read exactly once (row-resident realization)
    REQUIRE(online_loads == on.reduction_elems / on.tile_elems);
    // Pin the documented 4x: the 4-pass safe-softmax re-reads the row once
    // per pass (4 passes), the online form reads it once. A regression in
    // either would break the claim in CHANGELOG/session/coverage notes.
    REQUIRE(mp_loads == 4 * online_loads);
}

TEST_CASE("Online softmax characterization report",
          "[timing][regression][softmax][report]") {
    const auto& rows = characterization();
    if (rows.empty()) { SUCCEED("matrix test did not run"); return; }
    std::printf("\n%-12s %-14s %-22s %6s %9s %8s\n",
                "size", "envelope", "realization", "tiles", "cycles", "stalls");
    for (const auto& r : rows) {
        std::printf("%-12s %-14s %-22s %6zu %9llu %8llu\n",
                    r.size.c_str(), r.envelope.c_str(), r.realization.c_str(),
                    static_cast<size_t>(r.tiles),
                    static_cast<unsigned long long>(r.cycles),
                    static_cast<unsigned long long>(r.stalls));
    }
    SUCCEED("characterization recorded");
}
