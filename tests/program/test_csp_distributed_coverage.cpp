// ============================================================================
// tests/program/test_csp_distributed_coverage.cpp
// The coverage measures (#617; csp/distributed_coverage.hpp): what they read off a program agrees
// with what is known of it by other means. P7's own cost record (hops and L3), P5's reach (node (r, c)
// is r + c hops away, nothing forwarded), and P2's interior node (all 8 movers).
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/distributed_coverage.hpp>
#include <sw/kpu/program/csp/gen/distributed_broadcast.hpp>
#include <sw/kpu/program/csp/gen/distributed_transpose.hpp>
#include <sw/kpu/program/csp/gen/distributed_window.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;

namespace {

platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}

} // namespace

TEST_CASE("the coverage measures agree with P7's own cost record", "[program][csp][distributed][coverage]") {
    for (const auto& [file, side] : {std::pair{"kpu_dmm_2x2.json", 2u}, std::pair{"kpu_dmm_4x4.json", 4u}}) {
        const platform::DeviceSpecification d = device(file);
        gen::DistributedTransposeCost cost;
        const DistributedProgram p = gen::generate_distributed_transpose({8 * side, 8 * side}, d, &cost);
        const DistributedCoverage m = measure_distributed(p, d);
        INFO(file);
        CHECK(m.checks);
        CHECK(m.l3_slots == cost.max_l3_slots);
        // The record counts the route home to destination; the measure counts from the load at the
        // west edge, so it is the longest route from the edge: (R - 1) to the far column, then home's
        // route, at most 3 (R - 1) on the anti-diagonal corner.
        CHECK(m.hops >= cost.max_hops);
        CHECK(m.hops <= 3 * (side - 1));
        CHECK(m.transit > 0);                                  // the routes pass through nodes that only forward
    }
}

TEST_CASE("the coverage measures read P5's reach and P2's interior node", "[program][csp][distributed][coverage]") {
    const platform::DeviceSpecification d = device("kpu_dmm_4x4.json");
    const DistributedCoverage b = measure_distributed(gen::generate_distributed_broadcast({16, 16}, d), d);
    CHECK(b.hops == 6);                                        // node (3, 3) is 3 + 3 hops from the corner
    CHECK(b.transit == 0);                                     // every node holds the vector
    const DistributedCoverage w = measure_distributed(gen::generate_distributed_box3x3({32, 32}, d), d);
    CHECK(w.movers == 8);                                      // an interior node: 4 sends, a, b, x and c
    CHECK(w.sends == 4);
    CHECK(w.dma_bytes == 2u * 32 * 32 * 4);                    // X in and Y out, int32
}
