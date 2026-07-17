// ============================================================================
// tests/timing/test_m3_efficientnet.cpp
// M3 (#131): the full EfficientNet-B0 network (stem + MBConv+SE bottleneck stack
// + 1x1 head conv + global-average-pool + FC) built as a KernelGraph DFG and
// executed end-to-end on the CSP value path through GraphCspExecutor, validated
// against a composed whole-network host oracle (including the SE gate). See
// docs/plans/m3_mobilenet_dfg.md and docs/milestones/M3_efficientnet.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/efficientnet.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace sw::kpu::timing::graph;

TEST_CASE("M3 run_silu matches x*sigmoid(x) on the CSP executor",
          "[timing][m3][efficientnet]") {
    RunStats st;
    std::vector<float> x(64);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = -4.0f + 0.125f * static_cast<float>(i);
    auto y = run_silu(x, st);
    REQUIRE(y.size() == x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        REQUIRE_THAT(y[i], Catch::Matchers::WithinAbs(x[i] / (1.0f + std::exp(-x[i])), 1e-5));
}

TEST_CASE("M3 EfficientNet-B0 full network on the CSP executor matches host oracle",
          "[timing][m3][efficientnet]") {
    sw::kpu::KernelGraph g;
    EfficientNetB0Spec sp;                     // default fast scaled topology
    auto net = build_efficientnet_b0(g, sp);

    REQUIRE(g.num_nodes() == net.num_nodes);
    REQUIRE(g.get_execution_order().size() == net.num_nodes);

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, /*T*/16);

    REQUIRE(result.output.size() == net.oracle.size());
    REQUIRE(result.output.size() == static_cast<std::size_t>(sp.batch) * sp.num_classes);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < net.oracle.size(); ++i) {
        REQUIRE(std::isfinite(result.output[i]));
        REQUIRE(std::isfinite(net.oracle[i]));
        max_err = std::max(max_err, std::abs(result.output[i] - net.oracle[i]));
    }
    INFO("max_err=" << max_err << " nodes=" << net.num_nodes
         << " ops=" << result.stats.ops << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 5e-3f);
    REQUIRE(result.stats.ops > 0);
}

TEST_CASE("M3 EfficientNet-B0 builder rejects non-tile-aligned specs",
          "[timing][m3][efficientnet]") {
    sw::kpu::KernelGraph g;
    EfficientNetB0Spec sp;
    sp.stages[0].se = 20;   // SE reduce dim not a multiple of tile (16)
    REQUIRE_THROWS_AS(build_efficientnet_b0(g, sp), std::invalid_argument);
}
