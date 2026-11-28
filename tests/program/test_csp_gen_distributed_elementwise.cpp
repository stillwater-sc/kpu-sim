// ============================================================================
// tests/program/test_csp_gen_distributed_elementwise.cpp
// The P6 generator (#610; docs/plans/distributed-csp-programs.md §3.4): Z = X + Y with both operands
// streamed aligned into every compute tile. Every program passes D0-D4, runs on L-B (D5), and gives
// Z = X + Y element by element (D6, against values decoded and added one at a time).
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
#include <sw/kpu/program/csp/gen/distributed_elementwise.hpp>
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

void adds_aligned(const gen::DistributedElementwise& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_add(g, d);
    INFO(deploy << ": " << g.m << "x" << g.n << " in " << g.tile << "x" << g.tile << ", " << g.type);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 7);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    TileProgram want = in;                                     // D6: decoded, added, encoded, one at a time
    TensorOperand& z = want.operand("Z");
    for (Dim r = 0; r < z.rows; ++r)
        for (Dim c = 0; c < z.cols; ++c) z.set_value(r, c, in.operand("X").value(r, c) + in.operand("Y").value(r, c));
    CHECK(compare_values(lb.result(), want).empty());
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P6 on a 1x1 mesh", "[program][csp][distributed][gen][p6]") {
    adds_aligned({32, 32, 16}, "kpu_dmm_1x1.json");
}

TEST_CASE("P6 on 2x2 and 4x4 meshes: each row's pairs stream in from the west, aligned", "[program][csp][distributed][gen][p6]") {
    adds_aligned({32, 32, 16}, "kpu_dmm_2x2.json");
    adds_aligned({64, 32, 16, "fp32"}, "kpu_dmm_2x2.json");    // a 2x1 block per node, in fp32
    adds_aligned({64, 64, 16}, "kpu_dmm_4x4.json");
}

TEST_CASE("P6 refuses what it cannot distribute, by name", "[program][csp][distributed][gen][p6]") {
    CHECK_THROWS_WITH(gen::generate_distributed_add({48, 32, 16}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("the 3x2 tiles do not divide over the 2x2 mesh"));
    platform::DeviceSpecification few = device("kpu_dmm_4x4.json");
    few.dma.engines = 2;
    CHECK_THROWS_WITH(gen::generate_distributed_add({64, 64, 16}, few),
                      ContainsSubstring("4 mesh rows take a stream in, one DMA engine each, and the device has 2"));
}

TEST_CASE("add is element by element in its result's number system, and refuses a mix", "[program][csp][distributed][behavioral]") {
    DistributedProgram p = gen::generate_distributed_add({16, 16, 16}, device("kpu_dmm_1x1.json"));
    p.operands[1].type = "fp32";                               // Y in another number system than Z
    DistributedBehavioral lb;
    CHECK_THROWS_WITH(lb.run(p, distributed_inputs(p)), ContainsSubstring("add's operands and result share one number system"));
}
