// ============================================================================
// tests/timing/test_m3_mobilenet.cpp
// M3-T4 (#131): the full MobileNetV2 network (stem + inverted-residual bottleneck
// stack + 1x1 head conv + global-average-pool + FC) built as a KernelGraph DFG
// and executed end-to-end on the CSP value path through GraphCspExecutor,
// validated against a composed whole-network host oracle. See
// docs/plans/m3_mobilenet_dfg.md and docs/milestones/M3_mobilenet.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/mobilenetv2.hpp>

#include <algorithm>
#include <cmath>

using namespace sw::kpu::timing::graph;

TEST_CASE("M3 MobileNetV2 full network on the CSP executor matches host oracle",
          "[timing][m3][mobilenet]") {
    sw::kpu::KernelGraph g;
    MobileNetV2Spec sp;                       // default fast scaled topology
    auto net = build_mobilenetv2(g, sp);

    REQUIRE(g.num_nodes() == net.num_nodes);
    REQUIRE(g.get_execution_order().size() == net.num_nodes);

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, /*T*/16);

    REQUIRE(result.output.size() == net.oracle.size());
    REQUIRE(result.output.size() == static_cast<std::size_t>(sp.batch) * sp.num_classes);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < net.oracle.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - net.oracle[i]));
    INFO("max_err=" << max_err << " nodes=" << net.num_nodes
         << " ops=" << result.stats.ops << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 5e-3f);

    // Fusion collapses each conv+BN(+fused activation is not used here) so the CSP
    // op count is well below the graph node count, but every block still executes.
    REQUIRE(result.stats.ops > 0);
    REQUIRE(result.stats.ops < net.num_nodes);
}

TEST_CASE("M3 MobileNetV2 builder rejects non-tile-aligned specs",
          "[timing][m3][mobilenet]") {
    sw::kpu::KernelGraph g;
    MobileNetV2Spec sp;
    sp.num_classes = 20;   // not a multiple of tile (16)
    REQUIRE_THROWS_AS(build_mobilenetv2(g, sp), std::invalid_argument);
}
