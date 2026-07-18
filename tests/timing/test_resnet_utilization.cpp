// ============================================================================
// tests/timing/test_resnet_utilization.cpp
// Regression guard for the ResNet movement-fabric utilization metric.
//
// RunStats aggregates the executor's per-op Statistics so the m2_resnet demo can
// report DMA/BlockMover/Streamer utilization. busy_cycles is now a DIRECTLY
// MEASURED active-cycle count (follow-on 1b): each component increments a counter
// in its tick() on every cycle a transfer occupies it, so busy excludes both
// stalled and idle cycles (the former total - stalls/N heuristic counted idle as
// busy). Utilization = busy / total_cycles is therefore a true activity fraction.
//
// Invariants asserted here (on a full network, two shapes):
//   - 0 <= busy <= total_cycles for every mover (util in [0,1]) -- guards
//     over-counting / bad per-component normalization.
//   - busy > 0 for every mover -- guards a component whose active counter was
//     never wired into its tick() (would read a flat 0).
//   - not every mover pinned at 100% -- guards an unconditional per-cycle
//     increment that ignores whether a transfer actually occupied the cycle.
//   - util == busy / total (accessor consistency), tiles/bytes > 0.
//   - effective bandwidth guards a non-positive clock.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/resnet18.hpp>

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace sw::kpu::timing::graph;
using sw::kpu::timing::Cycle;

namespace {

RunStats run_resnet(const ResNet18Spec& sp) {
    sw::kpu::KernelGraph g;
    auto net = build_resnet18(g, sp);
    GraphCspExecutor exec;
    return exec.run(g, net.input, net.node_data, /*T*/16).stats;
}

// Directly measured active cycles (fractional per-component mean): 0 < busy <=
// cycles (util in (0,1]), per mover.
void check_mover(const char* who, double busy, Cycle total, double util) {
    INFO(who << ": busy=" << busy << " total=" << total << " util=" << util);
    REQUIRE(busy > 0.0);                        // active counter wired into tick()
    REQUIRE(busy <= static_cast<double>(total)); // utilization <= 1 (no over-count)
    REQUIRE(util >= 0.0);
    REQUIRE(util <= 1.0);
    REQUIRE_THAT(util, Catch::Matchers::WithinAbs(
        total > 0 ? busy / static_cast<double>(total) : 0.0, 1e-12));
}

} // namespace

TEST_CASE("ResNet RunStats utilization is directly measured and consistent",
          "[timing][resnet][utilization]") {
    // Two shapes so the invariants are exercised across op mixes (the base and the
    // wider-batch sweep the demo publishes).
    for (const auto& sp : {ResNet18Spec{}, [] { ResNet18Spec s; s.batch = 32; return s; }()}) {
        RunStats st = run_resnet(sp);

        REQUIRE(st.ops > 0);
        REQUIRE(st.total_cycles > 0);

        check_mover("dma", st.dma_busy, st.total_cycles, st.dma_utilization());
        check_mover("bm",  st.bm_busy,  st.total_cycles, st.bm_utilization());
        check_mover("str", st.str_busy, st.total_cycles, st.str_utilization());

        // Not every mover pinned at 100%: direct measurement must exclude idle
        // cycles, so at least one stage shows headroom (on ResNet the on-chip
        // movers starve behind the DRAM-bound DMA).
        REQUIRE((st.dma_busy < st.total_cycles ||
                 st.bm_busy  < st.total_cycles ||
                 st.str_busy < st.total_cycles));

        // Real fp32 moved through the fabric.
        REQUIRE(st.tiles_loaded > 0);
        REQUIRE(st.tiles_moved > 0);
        REQUIRE(st.bytes_loaded > 0);

        // Compute FLOP efficiency (roofline). GEMM MACs are counted, FLOPs = 2*MACs,
        // and the efficiencies are bounded fractions.
        REQUIRE(st.total_macs > 0);
        REQUIRE(st.total_flops() == Catch::Approx(2.0 * static_cast<double>(st.total_macs)));
        REQUIRE(st.arithmetic_intensity() > 0.0);
        REQUIRE(st.achieved_gflops(1.0) > 0.0);
        const double peak_eff = st.compute_efficiency(/*peak_flops_per_cycle*/ 512.0);
        REQUIRE(peak_eff > 0.0);
        REQUIRE(peak_eff <= 1.0);                     // cannot exceed the array peak
        const double roof_eff = st.roofline_efficiency(512.0, 64.0, 1.0);
        REQUIRE(roof_eff > 0.0);
        REQUIRE(std::isfinite(roof_eff));
        REQUIRE(roof_eff <= 1.0 + 1e-9);              // cannot beat the roofline
    }
}

TEST_CASE("GraphCspExecutor counts GEMM MACs from the FC matmul",
          "[timing][resnet][flops]") {
    // A single matmul node: MACs must equal M*N*K exactly (the FC hook), and the
    // fused/pool/elementwise ops that carry no GEMM contribute nothing.
    const sw::kpu::Size M = 32, N = 16, K = 48;   // tile-aligned (T = 16)
    sw::kpu::KernelGraph g;
    const auto mm = g.add_kernel(sw::kpu::Kernel::create_matmul(M, N, K), "mm");

    std::unordered_map<std::size_t, NodeData> nd;
    nd[mm].fc_M = M; nd[mm].fc_N = N; nd[mm].fc_K = K;
    nd[mm].fc_weight.assign(static_cast<std::size_t>(K) * N, 0.1f);
    nd[mm].fc_bias.assign(N, 0.0f);
    const std::vector<float> input(static_cast<std::size_t>(M) * K, 0.5f);

    GraphCspExecutor exec;
    const auto result = exec.run(g, input, nd, /*T*/ 16);

    const std::uint64_t expected = static_cast<std::uint64_t>(M) * N * K;   // 24576
    REQUIRE(result.stats.total_macs == expected);
    REQUIRE(result.stats.total_flops() == Catch::Approx(2.0 * static_cast<double>(expected)));
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
