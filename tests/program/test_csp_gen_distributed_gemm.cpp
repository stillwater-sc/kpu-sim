// ============================================================================
// tests/program/test_csp_gen_distributed_gemm.cpp
// The P1 generator (#608; docs/plans/distributed-csp-programs.md §3.4): edge-streamed, systolic GEMM
// on 1x1, 2x2 and 4x4 meshes. Every program passes D0-D4 (the checker), D5 (L-B finishes), and D6
// (C equals the L0 kernels' C = A . B, bit for bit).
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
#include <sw/kpu/program/csp/gen/distributed_gemm.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}

// C = A . B by the L0 kernels, tile by tile, k ascending: the reference L-B's calls must reproduce.
TileProgram reference(const DistributedProgram& p, const TileProgram& in) {
    TileProgram ref = in;
    const Operand2& c = *p.operand("C");
    const Operand2& a = *p.operand("A");
    for (Dim i = 0; i < c.tile_grid_rows(); ++i)
        for (Dim j = 0; j < c.tile_grid_cols(); ++j)
            for (Dim k = 0; k < a.tile_grid_cols(); ++k) {
                TileOp op;
                op.kind = TileOpKind::MatMulAccum;
                op.inputs = {TileCoord{"A", i, k}, TileCoord{"B", k, j}};
                op.outputs = {TileCoord{"C", i, j}};
                TileKernelState st;
                apply(ref, op, st);
            }
    return ref;
}

void passes_d0_to_d6(const gen::DistributedGemm& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_gemm(g, d);
    INFO(deploy << ": " << g.m << "x" << g.n << "x" << g.k << " in " << g.tile << "x" << g.tile);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 3);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    CHECK(compare_values(lb.result(), reference(p, in)).empty());   // D6
    // The text form reads back as the same program.
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P1 on a 1x1 mesh: one node owns every C tile", "[program][csp][distributed][gen][p1]") {
    passes_d0_to_d6({32, 32, 32, 16}, "kpu_dmm_1x1.json");
}

TEST_CASE("P1 on a 2x2 mesh: A east, B south, C north, each a hop at a time", "[program][csp][distributed][gen][p1]") {
    passes_d0_to_d6({64, 64, 32, 16}, "kpu_dmm_2x2.json");     // a 2x2 block of C per node
    passes_d0_to_d6({32, 32, 48, 16}, "kpu_dmm_2x2.json");     // one C tile per node, three k steps
}

TEST_CASE("P1 on a 4x4 mesh", "[program][csp][distributed][gen][p1]") {
    passes_d0_to_d6({64, 64, 48, 16}, "kpu_dmm_4x4.json");
}

TEST_CASE("P1 refuses what it cannot distribute, by name", "[program][csp][distributed][gen][p1]") {
    CHECK_THROWS_WITH(gen::generate_distributed_gemm({48, 32, 32, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("C's 3x2 tiles do not divide over the 2x2 mesh"));
    CHECK_THROWS_WITH(gen::generate_distributed_gemm({48, 48, 16, 16}, device("kpu_dmm_1x1.json")),   // 3x3 C tiles: 15 L2 slots
                      ContainsSubstring("a compute tile needs 15 L2 slots, and it has 8"));
    platform::DeviceSpecification few = device("kpu_dmm_4x4.json");
    few.dma.engines = 4;                                       // the 4x4 has 7 input edge nodes
    CHECK_THROWS_WITH(gen::generate_distributed_gemm({64, 64, 16, 16}, few),
                      ContainsSubstring("7 edge nodes take a stream in, one DMA engine each, and the device has 4"));
}
