// ============================================================================
// tests/program/test_csp_gen_distributed_broadcast.cpp
// The P5 generator (#609; docs/plans/distributed-csp-programs.md §3.4): one small operand to every
// node, east along row 0 and south down every column, one hop at a time. Every program passes D0-D4
// and runs on L-B (D5), and every node's compute tile ends holding the operand exactly as DRAM has it
// (D6).
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
#include <sw/kpu/program/csp/gen/distributed_broadcast.hpp>
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

void reaches_every_node(const gen::DistributedBroadcast& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_broadcast(g, d);
    INFO(deploy << ": " << g.length << " in tiles of " << g.tile);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 5);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    const Operand2& b = *p.operand("b");
    for (Dim t = 0; t < p.node_count(); ++t)                   // D6: every node holds DRAM's tiles
        for (std::uint32_t s = 0; s < b.tile_grid_rows(); ++s) {
            const auto held = lb.slot(SlotRef{SlotRef::Pool::L2, t, s});
            INFO("node " << t << ", tile " << s);
            REQUIRE(held.has_value());
            CHECK(held->first == TileCoord{"b", s, 0}.to_string());
            // Words, compared as bits: an integer's word may read as a NaN float.
            const auto want = BehavioralInterpreter::extract_tile(in.operand("b"), s, 0);
            REQUIRE(held->second.size() == want.size());
            CHECK(std::memcmp(held->second.data(), want.data(), want.size() * sizeof(float)) == 0);
        }
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P5 on a 1x1 mesh: the corner is the whole mesh", "[program][csp][distributed][gen][p5]") {
    reaches_every_node({32, 16}, "kpu_dmm_1x1.json");
}

TEST_CASE("P5 on 2x2 and 4x4 meshes: east along row 0, south down every column", "[program][csp][distributed][gen][p5]") {
    reaches_every_node({16, 16}, "kpu_dmm_2x2.json");
    reaches_every_node({48, 16}, "kpu_dmm_2x2.json");          // three tiles, each held everywhere
    reaches_every_node({64, 16}, "kpu_dmm_4x4.json");
}

TEST_CASE("P5 reaches node (r, c) in r + c hops, the least one hop at a time allows", "[program][csp][distributed][gen][p5]") {
    const DistributedProgram p = gen::generate_distributed_broadcast({16, 16}, device("kpu_dmm_4x4.json"));
    // Pushes: row 0 passes east (3), every node above the last row passes south (12): 15 hops, one per
    // node but the corner -- each node is reached once.
    std::size_t pushes = 0;
    for (const Action2& a : p.actions) pushes += a.kind == Action2::Kind::Push;
    CHECK(pushes == 15);
}

TEST_CASE("P5 refuses an operand the tiles cannot hold", "[program][csp][distributed][gen][p5]") {
    CHECK_THROWS_WITH(gen::generate_distributed_broadcast({160, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("10 tiles held in every compute tile's L2, which has 8 slots"));
}
