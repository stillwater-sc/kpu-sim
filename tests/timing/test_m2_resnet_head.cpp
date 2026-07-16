// ============================================================================
// tests/timing/test_m2_resnet_head.cpp
// M2-T3 (#203) part B: the ResNet classification HEAD - global-average-pool ->
// fully-connected - as a KernelGraph DFG executed on the CSP value path through
// GraphCspExecutor (POOL2D + MATMUL dispatch), validated vs a host oracle.
// Completes the operator set for the full ResNet-18 (tower from part A + this).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/kernel_graph.hpp>
#include <sw/kpu/timing/graph/graph_csp_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::timing::graph;
using sw::kpu::timing::schedule::Pool2DGeometry;
using sw::kpu::timing::schedule::global_avg_pool_reference;

namespace {

std::vector<float> synth(std::size_t n, uint64_t seed, float scale) {
    std::vector<float> v(n);
    uint64_t s = seed * 2654435761ULL + 12345ULL;
    for (auto& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        auto bits = static_cast<uint32_t>(s >> 33);
        x = (static_cast<float>(bits) / static_cast<float>(0x7FFFFFFF) - 1.0f) * scale;
    }
    return v;
}

} // namespace

TEST_CASE("M2 ResNet head (global-avg-pool -> FC) on the CSP executor",
          "[timing][m2][resnet][head]") {
    // Tower output shape: [N=16, C=32, H=2, W=2]; classify into 16 classes.
    const Size N = 16, C = 32, H = 2, W = 2, CLASSES = 16;

    Pool2DGeometry pg; pg.N = N; pg.C = C; pg.H = H; pg.W = W;
    auto x  = synth(pg.elems(), 1, 1.0f);
    auto wfc = synth(static_cast<std::size_t>(C) * CLASSES, 2, 0.1f);   // [C, CLASSES]
    auto bfc = synth(CLASSES, 3, 0.2f);

    // --- host oracle: GAP -> [N, C] then FC + bias ----------------------------
    auto gap = global_avg_pool_reference(x, pg);   // [N*C]
    std::vector<float> oref(static_cast<std::size_t>(N) * CLASSES, 0.0f);
    for (Size m = 0; m < N; ++m)
        for (Size n = 0; n < CLASSES; ++n) {
            float acc = bfc[n];
            for (Size k = 0; k < C; ++k)
                acc += gap[m * C + k] * wfc[k * CLASSES + n];
            oref[m * CLASSES + n] = acc;
        }

    // --- DFG: GAP node -> FC node ---------------------------------------------
    KernelGraph g;
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, C, H, W), "gap");
    auto fc = g.add_kernel(Kernel::create_matmul(N, CLASSES, C), "fc");  // M,N,K
    g.add_edge(gp, fc);  // gap reads the external input tensor

    std::unordered_map<std::size_t, NodeData> nd;
    nd[fc].fc_weight = wfc; nd[fc].fc_bias = bfc;
    nd[fc].fc_M = N; nd[fc].fc_K = C; nd[fc].fc_N = CLASSES;

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, /*T*/16);

    REQUIRE(result.output.size() == oref.size());
    REQUIRE(result.stats.ops == 2);   // GAP + FC, both on the CSP executor
    float max_err = 0.0f;
    for (std::size_t i = 0; i < oref.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - oref[i]));
    INFO("head max_err=" << max_err << " cycles=" << result.stats.total_cycles);
    REQUIRE(max_err < 1e-3f);
}

TEST_CASE("M2 ResNet head: FC ReLU activation on the CSP executor",
          "[timing][m2][resnet][head]") {
    const Size N = 16, C = 16, H = 4, W = 4, CLASSES = 16;
    Pool2DGeometry pg; pg.N = N; pg.C = C; pg.H = H; pg.W = W;
    auto x  = synth(pg.elems(), 11, 1.0f);
    auto wfc = synth(static_cast<std::size_t>(C) * CLASSES, 12, 0.1f);
    auto bfc = synth(CLASSES, 13, -2.0f);   // negative bias -> exercises the ReLU clamp

    auto gap = global_avg_pool_reference(x, pg);
    std::vector<float> oref(static_cast<std::size_t>(N) * CLASSES, 0.0f);
    for (Size m = 0; m < N; ++m)
        for (Size n = 0; n < CLASSES; ++n) {
            float acc = bfc[n];
            for (Size k = 0; k < C; ++k) acc += gap[m * C + k] * wfc[k * CLASSES + n];
            oref[m * CLASSES + n] = std::max(0.0f, acc);
        }

    KernelGraph g;
    auto gp = g.add_kernel(Kernel::create_global_avg_pool2d(N, C, H, W), "gap");
    auto fc = g.add_kernel(Kernel::create_matmul(N, CLASSES, C), "fc");
    g.add_edge(gp, fc);
    std::unordered_map<std::size_t, NodeData> nd;
    nd[fc].fc_weight = wfc; nd[fc].fc_bias = bfc;
    nd[fc].fc_M = N; nd[fc].fc_K = C; nd[fc].fc_N = CLASSES; nd[fc].fc_relu = true;

    GraphCspExecutor exec;
    auto result = exec.run(g, x, nd, 16);

    float max_err = 0.0f;
    for (std::size_t i = 0; i < oref.size(); ++i)
        max_err = std::max(max_err, std::abs(result.output[i] - oref[i]));
    INFO("head+relu max_err=" << max_err);
    REQUIRE(max_err < 1e-3f);
}
