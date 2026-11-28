// ============================================================================
// tests/program/test_csp_gen_distributed_transpose.cpp
// The P7 generator (#613; docs/plans/distributed-csp-programs.md §3.4): Y = X^T on a square mesh, one
// tile per node, every tile routed along its row to the diagonal and along the diagonal's column to its
// destination. Every program passes D0-D4 and runs on L-B (D5), Y is X transposed element by element
// (D6), and the cost it reports is the routes' (hops, L3 slots).
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/distributed_behavioral.hpp>
#include <sw/kpu/program/csp/distributed_check.hpp>
#include <sw/kpu/program/csp/gen/distributed_transpose.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstring>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}

gen::DistributedTransposeCost transposes(const gen::DistributedTranspose& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    gen::DistributedTransposeCost cost;
    const DistributedProgram p = gen::generate_distributed_transpose(g, d, &cost);
    INFO(deploy << ": " << g.m << "x" << g.n << ", " << g.type);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 6);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    const TensorOperand& x = in.operand("X");                  // D6: Y[j, i] = X[i, j], as words
    const TensorOperand& y = lb.result().operand("Y");
    REQUIRE(y.rows == x.cols);
    REQUIRE(y.cols == x.rows);
    bool same = true;
    for (Dim i = 0; i < x.rows; ++i)
        for (Dim j = 0; j < x.cols; ++j)
            same &= std::memcmp(&y.values[static_cast<std::size_t>(j) * y.cols + i], &x.values[static_cast<std::size_t>(i) * x.cols + j],
                                sizeof(float)) == 0;
    CHECK(same);
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
    return cost;
}

} // namespace

TEST_CASE("P7 on a 1x1 mesh: the tile transposes where it lands", "[program][csp][distributed][gen][p7]") {
    const auto cost = transposes({8, 8}, "kpu_dmm_1x1.json");
    CHECK(cost.max_hops == 0);
}

TEST_CASE("P7 on 2x2 and 4x4 meshes: along the row to the diagonal, along its column to the destination",
          "[program][csp][distributed][gen][p7]") {
    const auto two = transposes({16, 16}, "kpu_dmm_2x2.json");
    CHECK(two.max_hops == 2);                                  // X[0,1]: west to (0,0), south to (1,0)
    transposes({16, 32, "fp32"}, "kpu_dmm_2x2.json");          // 8 x 16 tiles into 16 x 8, in fp32
    const auto four = transposes({32, 32}, "kpu_dmm_4x4.json");
    CHECK(four.max_hops == 6);                                 // X[0,3]: three west, three south
    CHECK(four.total_hops == 40);                              // the sum over the off-diagonal tiles of 2|r - c|
    // The transform's L3 cost: the busiest node gives a slot to every tile it receives (no credit
    // waits), 5 on the 2x2 and 11 of the 16 on the 4x4.
    CHECK(two.max_l3_slots == 5);
    CHECK(four.max_l3_slots == 11);
}

TEST_CASE("P7 refuses a mesh it cannot transpose onto, by name", "[program][csp][distributed][gen][p7]") {
    platform::DeviceSpecification wide = device("kpu_dmm_2x2.json");
    wide.array.cols = 4;
    platform::derive_mesh_views(wide);
    CHECK_THROWS_WITH(gen::generate_distributed_transpose({16, 16}, wide), ContainsSubstring("so the mesh is square; this one is 2x4"));
    CHECK_THROWS_WITH(gen::generate_distributed_transpose({15, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("X divides into one tile per node"));
}

TEST_CASE("transpose checks its result's shape", "[program][csp][distributed][behavioral]") {
    DistributedProgram p = gen::generate_distributed_transpose({16, 16}, device("kpu_dmm_2x2.json"));
    p.operands[1].tile_cols = 4;                               // Y tiled 8 x 4, not X's 8 x 8 exchanged
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(p, distributed_inputs(p)), ContainsSubstring("transpose's result is its operand's shape exchanged"));
}
