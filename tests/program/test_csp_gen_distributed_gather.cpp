// ============================================================================
// tests/program/test_csp_gen_distributed_gather.cpp
// The P4 generator (#616; docs/plans/distributed-csp-programs.md §3.4, §7 decision 2): gather and
// scatter with the run-time indices at the DMA. Every program passes D0-D4 and runs on L-B (D5), and W
// is the table's rows gathered by idx, doubled on the mesh, scattered by jdx (D6). Also the language's
// gather and scatter: their text round trip, and the refusals.
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
#include <sw/kpu/program/csp/gen/distributed_gather.hpp>
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

// idx: a run-time pattern, repeats included (row r reads (5r + 3) mod V); jdx: a permutation, the
// rows reversed, so every row of W is written once.
TileProgram inputs(const DistributedProgram& p, const gen::DistributedGather& g) {
    TileProgram in = distributed_inputs(p, 4);
    for (Dim r = 0; r < g.m; ++r) {
        in.operand("idx").set_value(r, 0, static_cast<double>((5 * r + 3) % g.table_rows));
        in.operand("jdx").set_value(r, 0, static_cast<double>(g.m - 1 - r));
    }
    return in;
}

void gathers(const gen::DistributedGather& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_gather(g, d);
    INFO(deploy << ": " << g.m << " rows of " << g.cols << " from " << g.table_rows);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = inputs(p, g);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    TensorOperand want = in.operand("W");                      // D6: W[jdx[r]] = 2 E[idx[r]]
    const TensorOperand& E = in.operand("E");
    for (Dim r = 0; r < g.m; ++r) {
        const Dim from = static_cast<Dim>(in.operand("idx").value(r, 0)), to = static_cast<Dim>(in.operand("jdx").value(r, 0));
        for (Dim c = 0; c < g.cols; ++c) want.set_value(to, c, 2 * E.value(from, c));
    }
    const TensorOperand& got = lb.result().operand("W");
    CHECK(std::memcmp(got.values.data(), want.values.data(), want.values.size() * sizeof(float)) == 0);
    // The mesh carries only G and Z; only the DMA engines name E and W, and only by index.
    for (const Action2& a : p.actions) {
        CHECK(a.tile.operand != "E");
        CHECK(a.tile.operand != "W");
    }
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P4 on a 1x1 mesh: gather, compute, scatter", "[program][csp][distributed][gen][p4]") {
    gathers({16, 16, 11, 16, 8}, "kpu_dmm_1x1.json");
}

TEST_CASE("P4 on 2x2 and 4x4 meshes: gathered at the west edge, moved to the owner and back", "[program][csp][distributed][gen][p4]") {
    gathers({32, 16, 13, 32, 8}, "kpu_dmm_2x2.json");          // a tile row per node
    gathers({64, 16, 21, 64, 8}, "kpu_dmm_2x2.json");          // two per node
    gathers({128, 8, 37, 128, 8}, "kpu_dmm_4x4.json");
    gathers({24, 8, 9, 24, 8}, "kpu_dmm_4x4.json");            // fewer tile rows than nodes: the east of row 0 idle
}

TEST_CASE("a gather or scatter refuses an index outside its operand, at run time", "[program][csp][distributed][behavioral]") {
    const gen::DistributedGather g{16, 8, 5, 16, 8};
    const DistributedProgram p = gen::generate_distributed_gather(g, device("kpu_dmm_1x1.json"));
    TileProgram in = inputs(p, g);
    in.operand("idx").set_value(3, 0, 5.0);                    // E has rows 0..4
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(p, in), ContainsSubstring("idx[3] is 5, not a row of E (it has 5)"));
}

TEST_CASE("a gather's index and rows are checked against its tile's operand", "[program][csp][distributed]") {
    DistributedProgram p = gen::generate_distributed_gather({16, 16, 11, 16, 8}, device("kpu_dmm_1x1.json"));
    DistributedProgram q = p;
    for (Operand2& o : q.operands)
        if (o.name == "idx") o.type = "fp32";
    CHECK_THAT(validate_structure(q), ContainsSubstring("a gather's index idx is an int32 vector, in"));
    q = p;
    for (Operand2& o : q.operands)
        if (o.name == "E") o.cols = 8;
    CHECK_THAT(validate_structure(q), ContainsSubstring("a gather moves whole rows: E and G share their columns"));
    q = p;
    for (Action2& a : q.actions)
        if (a.kind == Action2::Kind::Load) a.rows.clear();     // G is temp: an ordinary load of it is refused
    CHECK_THAT(validate_structure(q), ContainsSubstring("G is temp, so it never leaves the nodes"));
}
