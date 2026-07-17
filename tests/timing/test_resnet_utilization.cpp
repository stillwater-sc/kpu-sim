// ============================================================================
// tests/timing/test_resnet_utilization.cpp
// Regression guard for the ResNet movement-fabric utilization metric (PR #220).
//
// RunStats aggregates the executor's per-op Statistics so the m2_resnet demo can
// report DMA/BlockMover/Streamer utilization. Every RunStats-producing runner must
// contribute BOTH its busy cycles (numerator) AND its total_cycles (denominator);
// an earlier version instrumented total_cycles for some ops (elementwise,
// depthwise, global-avg-pool) without their busy cycles, understating utilization.
//
// The consistency invariant that catches that bug: because the executor derives
// busy = cycles - stalls/N (clamped >= 0) per op, the aggregate satisfies
//   busy + stalls >= total_cycles      for each mover.
// An op that adds cycles without its busy/stalls breaks it. This test asserts the
// invariant (and util in [0,1], and the bandwidth clock guard) on a full network.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/resnet18.hpp>

#include <cmath>

using namespace sw::kpu::timing::graph;
using sw::kpu::timing::Cycle;

namespace {

RunStats run_resnet(const ResNet18Spec& sp) {
    sw::kpu::KernelGraph g;
    auto net = build_resnet18(g, sp);
    GraphCspExecutor exec;
    return exec.run(g, net.input, net.node_data, /*T*/16).stats;
}

// busy + stalls >= cycles, and busy <= cycles (util in [0,1]), per mover.
void check_mover(const char* who, Cycle busy, Cycle stalls, Cycle total, double util) {
    INFO(who << ": busy=" << busy << " stalls=" << stalls << " total=" << total
             << " util=" << util);
    REQUIRE(busy <= total);                    // utilization <= 1
    REQUIRE(busy + stalls >= total);           // instrumentation-consistency invariant
    REQUIRE(util >= 0.0);
    REQUIRE(util <= 1.0);
    REQUIRE_THAT(util, Catch::Matchers::WithinAbs(
        total > 0 ? static_cast<double>(busy) / static_cast<double>(total) : 0.0, 1e-12));
}

} // namespace

TEST_CASE("ResNet RunStats utilization is instrumentation-consistent",
          "[timing][resnet][utilization]") {
    // Two shapes so the invariant is exercised across op mixes (the base and the
    // wider-batch sweep the demo publishes).
    for (const auto& sp : {ResNet18Spec{}, [] { ResNet18Spec s; s.batch = 32; return s; }()}) {
        RunStats st = run_resnet(sp);

        REQUIRE(st.ops > 0);
        REQUIRE(st.total_cycles > 0);

        check_mover("dma", st.dma_busy, st.dma_stalls, st.total_cycles, st.dma_utilization());
        check_mover("bm",  st.bm_busy,  st.bm_stalls,  st.total_cycles, st.bm_utilization());
        check_mover("str", st.str_busy, st.str_stalls, st.total_cycles, st.str_utilization());

        // Real fp32 moved through the fabric: tiles and bytes are non-zero, and
        // every op contributed to ops (so the denominator is not diluted).
        REQUIRE(st.tiles_loaded > 0);
        REQUIRE(st.tiles_moved > 0);
        REQUIRE(st.bytes_loaded > 0);
    }
}

TEST_CASE("RunStats effective bandwidth guards the clock argument",
          "[timing][resnet][utilization]") {
    RunStats st = run_resnet(ResNet18Spec{});
    REQUIRE(st.bytes_loaded > 0);

    // Non-positive clock yields 0.0, not inf/NaN.
    REQUIRE(st.effective_load_bandwidth(0.0) == 0.0);
    REQUIRE(st.effective_load_bandwidth(-1.0) == 0.0);
    REQUIRE(st.effective_store_bandwidth(0.0) == 0.0);

    // A positive clock gives a positive, finite bandwidth (bytes were moved).
    const double bw = st.effective_load_bandwidth(1.0);
    REQUIRE(bw > 0.0);
    REQUIRE(std::isfinite(bw));
}
