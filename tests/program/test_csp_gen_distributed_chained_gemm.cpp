// ============================================================================
// tests/program/test_csp_gen_distributed_chained_gemm.cpp
// The P8 generator (#614; docs/plans/distributed-csp-programs.md §3.4): E = (A . B) . D with T = A . B
// fused, resident in the compute tiles' L2, never stored. Every program passes D0-D4 and runs on L-B
// (D5), and E is (A . B) . D (D6) against a reference computed here, independently of the program.
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
#include <sw/kpu/program/csp/gen/distributed_chained_gemm.hpp>
#include <sw/kpu/program/csp/lang/distributed_text.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const char* f) {
    return platform::read_spec_file(std::string("tests/program/deploy/") + f).device(0);
}

// The inputs are small integers, so every sum is exact in fp32, in any order: the reference is the
// plain triple loops, twice.
std::vector<double> reference(const TileProgram& in) {
    const TensorOperand &A = in.operand("A"), &B = in.operand("B"), &D = in.operand("D");
    std::vector<double> T(static_cast<std::size_t>(A.rows) * B.cols, 0.0), E(static_cast<std::size_t>(A.rows) * D.cols, 0.0);
    for (Dim i = 0; i < A.rows; ++i)
        for (Dim j = 0; j < B.cols; ++j)
            for (Dim k = 0; k < A.cols; ++k) T[i * B.cols + j] += A.values[i * A.cols + k] * B.values[k * B.cols + j];
    for (Dim i = 0; i < A.rows; ++i)
        for (Dim q = 0; q < D.cols; ++q)
            for (Dim j = 0; j < B.cols; ++j) E[i * D.cols + q] += T[i * B.cols + j] * D.values[j * D.cols + q];
    return E;
}

void fuses(const gen::DistributedChainedGemm& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_chained_gemm(g, d);
    INFO(deploy << ": " << g.m << "x" << g.k << "x" << g.n << "x" << g.p);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 3);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    const std::vector<double> want = reference(in);            // D6, as words
    const TensorOperand& E = lb.result().operand("E");
    bool same = true;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const float w = static_cast<float>(want[i]);
        same &= std::memcmp(&E.values[i], &w, sizeof(float)) == 0;
    }
    CHECK(same);
    // T is fused: no action loads it or stores it, and no drain takes it out of L2.
    for (const Action2& a : p.actions)
        if (a.tile.operand == "T") CHECK(a.kind == Action2::Kind::Call);
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P8 on a 1x1 mesh: both GEMMs in one compute tile, T resident between them", "[program][csp][distributed][gen][p8]") {
    fuses({8, 16, 8, 16, 8}, "kpu_dmm_1x1.json");               // one T tile, K and P two tiles each
}

TEST_CASE("P8 on 2x2 and 4x4 meshes: D follows B down each column, E's partials chain west", "[program][csp][distributed][gen][p8]") {
    fuses({16, 16, 16, 16, 8}, "kpu_dmm_2x2.json");
    fuses({16, 24, 32, 8, 8}, "kpu_dmm_2x2.json");             // a 1 x 2 block of T per node; E one tile column
    fuses({32, 16, 32, 24, 8}, "kpu_dmm_4x4.json");
}

TEST_CASE("P8 refuses what does not fit, by name", "[program][csp][distributed][gen][p8]") {
    CHECK_THROWS_WITH(gen::generate_distributed_chained_gemm({24, 16, 16, 16, 8}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("T's 3x2 tiles do not divide over the 2x2 mesh"));
    CHECK_THROWS_WITH(gen::generate_distributed_chained_gemm({16, 16, 16, 16, 8}, device("kpu_dmm_1x1.json")),
                      ContainsSubstring("a compute tile needs 11 L2 slots (A, B and D, T resident, E's partials), and it has 8"));
}

TEST_CASE("a temp operand never leaves the nodes", "[program][csp][distributed]") {
    DistributedProgram p = gen::generate_distributed_chained_gemm({8, 16, 8, 16, 8}, device("kpu_dmm_1x1.json"));
    for (Action2& a : p.actions)
        if (a.kind == Action2::Kind::Store) {
            a.tile.operand = "T";
            break;
        }
    CHECK_THAT(validate_structure(p), ContainsSubstring("T is temp, so it never leaves the nodes"));
    CHECK_THROWS_WITH(lang::parse_distributed("dcsp 1\nprogram x machine mesh(1x1) {\n  tensor T[8,8] fp32 tile 8x8 scratch;\n}\n"),
                      ContainsSubstring("in, out, inout or temp"));
}
