// ============================================================================
// tests/timing/test_m2_resnet18.cpp
// M2-T4 (#206): the FULL ResNet-18 (stem + 4 stages + GAP + FC) built via the
// reusable resnet18.hpp builder, executed end-to-end as a KernelGraph DFG on the
// CSP value path through GraphCspExecutor, validated against the composed
// whole-network host oracle. The demonstrate+validate tier of milestone M2 (#130).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/resnet18.hpp>

#include <algorithm>
#include <cmath>

using namespace sw::kpu::timing::graph;

TEST_CASE("M2 full ResNet-18 (stem + stages + GAP + FC) on the CSP executor",
          "[timing][m2][resnet][network]") {
    ResNet18Spec spec;                 // scaled demo: batch 16, 16ch 8x8 input
    sw::kpu::KernelGraph g;
    auto net = build_resnet18(g, spec);

    // Structural: the whole network is one connected DAG. Default spec is
    // [1,1,1,1]: stem (3) + stage1 identity block (7) + 3 downsample blocks (9
    // each) + head gap/fc (2) = 39 nodes.
    REQUIRE(g.num_nodes() == net.num_nodes);
    REQUIRE(g.get_execution_order().size() == g.num_nodes());
    REQUIRE(g.num_nodes() == 39);

    GraphCspExecutor exec;
    auto result = exec.run(g, net.input, net.node_data, /*T*/16);

    REQUIRE(result.output.size() == net.oracle.size());   // [batch, num_classes]
    // Fusion contract: stem conv (1) + identity block (4) + 3 downsample blocks
    // (5 each) + gap + fc = 22 CSP ops (every BN folded, block ReLUs fused).
    REQUIRE(result.stats.ops == 22);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < net.oracle.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - net.oracle[i]));
    INFO("ResNet-18 end-to-end max_err=" << max_err << " ops=" << result.stats.ops
         << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 5e-3f);
}
