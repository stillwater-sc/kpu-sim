// ============================================================================
// tests/program/test_csp_gen_distributed_window.cpp
// The P2 generator (#612; docs/plans/distributed-csp-programs.md §3.4): a 3 x 3 box filter as a row pass
// and a column pass, with a halo exchange with four neighbours. Every program passes D0-D4 and runs on
// L-B (D5); Y is each element's 3 x 3 neighbourhood summed, zero past X's edges (D6). On a mesh of at
// least 3 x 3 an interior node uses all eight of its movers.
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
#include <sw/kpu/program/csp/gen/distributed_window.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <set>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}

void filters(const gen::DistributedWindow& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_box3x3(g, d);
    INFO(deploy << ": " << g.m << "x" << g.n << ", " << g.type);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 4);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    TileProgram want = in;                                     // D6: the 3 x 3 sum, zero past the edges
    const TensorOperand& x = in.operand("X");
    const auto at = [&](long r, long c) { return r < 0 || c < 0 || r >= long(x.rows) || c >= long(x.cols) ? 0.0 : x.value(Dim(r), Dim(c)); };
    TensorOperand& y = want.operand("Y");
    for (long r = 0; r < long(x.rows); ++r)
        for (long c = 0; c < long(x.cols); ++c) {
            double sum = 0;
            for (long dr = -1; dr <= 1; ++dr)
                for (long dc = -1; dc <= 1; ++dc) sum += at(r + dr, c + dc);
            y.set_value(Dim(r), Dim(c), sum);
        }
    CHECK(compare_values(lb.result(), want).empty());
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P2 on a 1x1 mesh: no halo at all", "[program][csp][distributed][gen][p2]") {
    filters({8, 8}, "kpu_dmm_1x1.json");
}

TEST_CASE("P2 on 2x2 and 4x4 meshes: a halo exchange with four neighbours", "[program][csp][distributed][gen][p2]") {
    filters({16, 16}, "kpu_dmm_2x2.json");
    filters({16, 32, "fp32"}, "kpu_dmm_2x2.json");             // 8 x 16 tiles, in fp32
    filters({32, 32}, "kpu_dmm_4x4.json");
}

TEST_CASE("P2 on a 4x4 mesh: an interior node uses all eight of its movers", "[program][csp][distributed][gen][p2]") {
    const DistributedProgram p = gen::generate_distributed_box3x3({32, 32}, device("kpu_dmm_4x4.json"));
    std::set<std::string> movers;
    for (const Process2& pr : p.processes)
        if (pr.id.role != Role::Dma && pr.id.role != Role::Compute && pr.id.node == 5) movers.insert(role_name(pr.id.role));
    CHECK(movers == std::set<std::string>{"bm.N", "bm.E", "bm.S", "bm.W", "bm.a", "bm.b", "bm.x", "bm.c"});
}

TEST_CASE("P2 refuses what it cannot tile, by name", "[program][csp][distributed][gen][p2]") {
    CHECK_THROWS_WITH(gen::generate_distributed_box3x3({15, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("X divides into one tile per node"));
}

TEST_CASE("box3 refuses a halo of another shape", "[program][csp][distributed][behavioral]") {
    DistributedProgram p = gen::generate_distributed_box3x3({16, 16}, device("kpu_dmm_2x2.json"));
    DistributedProgram wide = p;
    wide.operands[1].tile_cols = 4;                            // H tiled 8 x 4 against X's 8 x 8
    wide.operands[1].cols = 16;
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(wide, distributed_inputs(wide)), ContainsSubstring("s tiles are its result's shape: X[0,0] is 8x8, H[0,1] is 8x4"));
}
