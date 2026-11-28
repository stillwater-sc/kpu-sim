// ============================================================================
// tests/program/test_csp_gen_distributed_append.cpp
// The P9 generator (#615; docs/plans/distributed-csp-programs.md §3.4): decode's append to a growing
// extent and its windowed re-read. Every program passes D0-D4 and runs on L-B (D5); K is the cache
// with each step's X appended, and O each step's window summed (D6), from step 1 on re-reading
// positions the program itself appended.
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
#include <sw/kpu/program/csp/gen/distributed_append.hpp>
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

bool same_words(const TensorOperand& got, const TensorOperand& want) {
    return got.values.size() == want.values.size() &&
           std::memcmp(got.values.data(), want.values.data(), want.values.size() * sizeof(float)) == 0;
}

void appends(const gen::DistributedAppend& g, const char* deploy) {
    const platform::DeviceSpecification d = device(deploy);
    const DistributedProgram p = gen::generate_distributed_append(g, d);
    INFO(deploy << ": L " << g.prefix << ", S " << g.steps << ", W " << g.window);
    REQUIRE(validate_structure(p).empty());
    std::string v;
    for (const auto& x : check_distributed(p, d)) v += x.rule + " " + x.message + "\n";
    INFO(v);
    CHECK(v.empty());                                          // D0-D4
    const TileProgram in = distributed_inputs(p, 2);
    DistributedBehavioral lb;
    REQUIRE_NOTHROW(lb.run(p, in));                            // D5
    // D6: the reference, element by element. The inputs are small integers, so every sum is exact.
    const Dim T = g.tile;
    TensorOperand K = in.operand("K"), O = in.operand("O");
    const TensorOperand& X = in.operand("X");
    for (Dim h = 0; h < K.rows / T; ++h)
        for (Dim s = 0; s < g.steps; ++s)
            for (Dim i = 0; i < T; ++i)
                for (Dim j = 0; j < T; ++j)
                    K.set_value(h * T + i, (g.prefix + s) * T + j, X.value(h * T + i, s * T + j));
    for (Dim h = 0; h < K.rows / T; ++h)
        for (Dim s = 0; s < g.steps; ++s)
            for (Dim i = 0; i < T; ++i)
                for (Dim j = 0; j < T; ++j) {
                    double sum = 0;
                    for (Dim w = 0; w < g.window; ++w) sum += K.value(h * T + i, (g.prefix + s - w) * T + j);
                    O.set_value(h * T + i, s * T + j, sum);
                }
    CHECK(same_words(lb.result().operand("K"), K));            // the appends landed
    CHECK(same_words(lb.result().operand("O"), O));            // the windows read them back
    CHECK(lang::print_distributed(lang::parse_distributed(lang::print_distributed(p))) == lang::print_distributed(p));
}

} // namespace

TEST_CASE("P9 on a 1x1 mesh: append, then re-read the window", "[program][csp][distributed][gen][p9]") {
    appends({3, 3, 3, 8}, "kpu_dmm_1x1.json");
    appends({1, 1, 1, 8}, "kpu_dmm_1x1.json");                 // W 1: the new position alone
}

TEST_CASE("P9 on 2x2 and 4x4 meshes: a head per node, its DRAM through its column's row 0",
          "[program][csp][distributed][gen][p9]") {
    appends({2, 4, 3, 8}, "kpu_dmm_2x2.json");                 // steps 1-3 re-read positions this program appended
    appends({4, 3, 4, 8}, "kpu_dmm_4x4.json");
}

TEST_CASE("P9 refuses a window longer than the cache", "[program][csp][distributed][gen][p9]") {
    CHECK_THROWS_WITH(gen::generate_distributed_append({1, 2, 3, 8}, device("kpu_dmm_2x2.json")),
                      ContainsSubstring("the first step's window reads the 2 positions before it, and the cache has 1"));
}
