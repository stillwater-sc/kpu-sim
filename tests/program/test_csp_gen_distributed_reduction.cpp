// ============================================================================
// tests/program/test_csp_gen_distributed_reduction.cpp
// The P3 generator (#611; docs/plans/distributed-csp-programs.md §3.4): s = the sums of X's rows,
// chained west along each mesh row, stored from its west end, and broadcast back east. Every program
// passes D0-D4 and runs on L-B (D5); the stored s is the row sums (D6), and every node of a row holds
// its row's sums in its compute tile's L2.
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
#include <sw/kpu/program/csp/gen/distributed_reduction.hpp>
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

void reduces_rows(const gen::DistributedRowReduction& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_rowsum(g, d);
    INFO(deploy << ": " << g.m << "x" << g.n << " in " << g.tile << "x" << g.tile << ", " << g.type);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 2);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    TileProgram want = in;                                     // D6: each row of X summed, in full
    const TensorOperand& x = in.operand("X");
    TensorOperand& s = want.operand("s");
    for (Dim r = 0; r < x.rows; ++r) {
        double total = 0;
        for (Dim c = 0; c < x.cols; ++c) total += x.value(r, c);
        s.set_value(r, 0, total);
    }
    CHECK(compare_values(lb.result(), want).empty());
    // Every node of mesh row r holds its row's sums in L2, after the X and partial slots.
    const Dim a = (g.m / g.tile) / p.rows;
    for (Dim t = 0; t < p.node_count(); ++t)
        for (Dim ii = 0; ii < a; ++ii) {
            const Dim i = (t / p.cols) * a + ii;
            const auto held = lb.slot(SlotRef{SlotRef::Pool::L2, t, static_cast<std::uint32_t>(3 + a + ii)});
            INFO("node " << t << ", s tile " << i);
            REQUIRE(held.has_value());
            CHECK(held->first == TileCoord{"s", i, 0}.to_string());
            const auto want_tile = BehavioralInterpreter::extract_tile(want.operand("s"), i, 0);
            REQUIRE(held->second.size() == want_tile.size());
            CHECK(std::memcmp(held->second.data(), want_tile.data(), want_tile.size() * sizeof(float)) == 0);
        }
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P3 on a 1x1 mesh: the node is both ends of its row", "[program][csp][distributed][gen][p3]") {
    reduces_rows({32, 32, 16}, "kpu_dmm_1x1.json");
}

TEST_CASE("P3 on 2x2 and 4x4 meshes: partials chain west, the total comes back east", "[program][csp][distributed][gen][p3]") {
    reduces_rows({32, 32, 16}, "kpu_dmm_2x2.json");
    reduces_rows({64, 64, 16, "fp32"}, "kpu_dmm_2x2.json");    // a 2x2 block per node, in fp32
    reduces_rows({64, 128, 16}, "kpu_dmm_4x4.json");           // a 1x2 block per node
}

TEST_CASE("P3 refuses what it cannot hold, by name", "[program][csp][distributed][gen][p3]") {
    CHECK_THROWS_WITH(gen::generate_distributed_rowsum({48, 32, 16}, device("kpu_dmm_1x1.json")),
                      ContainsSubstring("a compute tile needs 9 L2 slots (a block 3 tiles tall), and it has 8"));
    CHECK_THROWS_WITH(gen::generate_distributed_rowsum({48, 32, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("X's 3x2 tiles do not divide over the 2x2 mesh"));
}

TEST_CASE("rowsum and sum check their shapes and number systems", "[program][csp][distributed][behavioral]") {
    DistributedProgram p = gen::generate_distributed_rowsum({16, 16, 16}, device("kpu_dmm_1x1.json"));
    DistributedProgram mixed = p;
    mixed.operands[1].type = "fp32";                           // s in another number system than X
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(mixed, distributed_inputs(mixed)), ContainsSubstring("rowsum's operand and result share one number system"));
    DistributedProgram tall = p;
    tall.operands[1].tile_rows = 8;                            // s tiled 8 long against X's 16-row tiles
    tall.operands[1].rows = 16;
    CHECK_THROWS_WITH(lb.run(tall, distributed_inputs(tall)), ContainsSubstring("rowsum's result is a column as tall as its operand"));
}
