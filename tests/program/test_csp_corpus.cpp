// ============================================================================
// tests/program/test_csp_corpus.cpp
// The CSP corpus (docs/plans/kpu-run-csp-programs.md step 4a): checked-in .csp programs, run
// through the virtual platform at every level that can run them on S1, against their
// checked-in answers. Same two claims as the L0 corpus (test_l0_corpus.cpp), for the same
// reason -- Release builds use -march=native, so the last bits follow the host CPU:
//   - CROSS-MACHINE: every level's operands equal the recorded .result.l0 within tolerance;
//   - SAME-MACHINE: every level is bit-identical to the program's own L0 oracle.
// A level that cannot run a corpus program says why, and the test asserts the reason.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace lang = sw::kpu::program::csp::lang;
using sw::kpu::program::driver::ExecutionLevel;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kCorpus = "tests/program/corpus/";

std::string read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());                         // a missing corpus file is a failure, not a skip
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

struct Case {
    const char* program;                        // .csp
    const char* inputs;                         // .l0 carrying the input values
    const char* expected;                       // .result.l0
    const char* lca_refused;                    // nullptr: L-CA runs it; else its reason
};

const std::vector<Case>& cases() {
    static const std::vector<Case> c = {
        {"matmul_48x48x48_t16.csp", "matmul_48x48x48_t16.l0", "matmul_48x48x48_t16.result.l0", nullptr},
        {"linear_64_t16_relu.csp", "linear_64_t16_relu.l0", "linear_64_t16_relu.result.l0", nullptr},
        {"lu_64_t16.csp", "lu_64_t16.l0", "lu_64_t16.result.l0", "getrf (LU) at L-CA is a later increment"},
    };
    return c;
}

constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-4;

bool close_enough(const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (!(std::abs(double(got[i]) - double(want[i])) <= kAtol + kRtol * std::abs(double(want[i])))) return false;
    return true;
}

}  // namespace

TEST_CASE("CSP corpus: every program runs at every level it can, and computes its recorded answer",
          "[program][csp][corpus]") {
    platform::VirtualPlatform vp(platform::read_spec_file("tests/program/deploy/kpu_s1.json"));
    for (const Case& c : cases()) {
        CAPTURE(c.program);
        const lang::Program ast = lang::parse(read(std::string(kCorpus) + c.program));
        const TileProgram values = serialize::from_string(read(std::string(kCorpus) + c.inputs));
        const TileProgram expected = serialize::from_string(read(std::string(kCorpus) + c.expected));
        TileProgram inputs = driver::csp_inputs(ast);
        for (const auto& name : inputs.operand_order()) inputs.operand(name).values = values.operand(name).values;

        const auto h = vp.load_csp(ast, inputs);
        const auto oracle = vp.csp_reference(h);
        REQUIRE(oracle.values.has_value());
        for (ExecutionLevel level :
             {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}) {
            CAPTURE(to_string(level));
            const auto r = vp.run_csp(h, level);
            if (level == ExecutionLevel::CycleAccurate && c.lca_refused) {
                REQUIRE(r.outcome.skipped);
                CHECK_THAT(*r.outcome.skipped, ContainsSubstring(c.lca_refused));
                continue;
            }
            REQUIRE_FALSE(r.outcome.skipped);
            for (const auto& name : expected.operand_order()) {
                CAPTURE(name);
                CHECK(close_enough(r.outcome.values.operand(name).values, expected.operand(name).values));
                CHECK(r.outcome.values.operand(name).values == oracle.values->operand(name).values);
            }
        }
    }
}

TEST_CASE("CSP corpus: a program's identity is its canonical text", "[program][csp][corpus][platform]") {
    platform::VirtualPlatform vp(platform::read_spec_file("tests/program/deploy/kpu_s1.json"));
    const std::string text = read(std::string(kCorpus) + "matmul_48x48x48_t16.csp");
    const lang::Program ast = lang::parse(text);
    // The same program, spelled differently: comments, whitespace, redundant parentheses.
    std::string respelled = "// a comment\n" + text;
    for (std::size_t at = respelled.find("0..3"); at != std::string::npos; at = respelled.find("0..3", at + 6))
        respelled.replace(at, 4, "(0)..(1 + 2)");
    const lang::Program ast2 = lang::parse(respelled);
    CHECK(lang::format(ast) == lang::format(ast2));

    const TileProgram inputs = driver::csp_inputs(ast);
    const auto h1 = vp.load_csp(ast, inputs);
    const auto h2 = vp.load_csp(ast2, inputs);
    CHECK(vp.csp_text(h1) == vp.csp_text(h2));
    const auto r1 = vp.run_csp(h1, ExecutionLevel::CycleAccurate);
    const auto r2 = vp.run_csp(h2, ExecutionLevel::CycleAccurate);
    CHECK(r1.identity == r2.identity);
    CHECK(r1.outcome.makespan == r2.outcome.makespan);
    // The window is part of the identity: a different schedule is a different run.
    const auto r3 = vp.run_csp(h1, ExecutionLevel::CycleAccurate, 16);
    CHECK_FALSE(r3.identity == r1.identity);
    CHECK(r3.identity.str().find("window=16") != std::string::npos);
    // Different inputs are a different run of the same program.
    TileProgram other = inputs;
    other.operand("A").values[0] += 1.0f;
    const auto h3 = vp.load_csp(ast, other);
    CHECK(vp.run_csp(h3, ExecutionLevel::Behavioral).identity.program_digest == r1.identity.program_digest);
    CHECK_FALSE(vp.run_csp(h3, ExecutionLevel::Behavioral).identity.snapshot_digest == r1.identity.snapshot_digest);
}

TEST_CASE("CSP format: the canonical text is a fixed point and compiles to the same program",
          "[program][csp][format]") {
    for (const char* f : {"matmul_48x48x48_t16.csp", "linear_64_t16_relu.csp", "lu_64_t16.csp"}) {
        CAPTURE(f);
        const lang::Program ast = lang::parse(read(std::string(kCorpus) + f));
        const std::string once = lang::format(ast);
        CHECK(lang::format(lang::parse(once)) == once);
        const auto a = lang::compile(ast);
        const auto b = lang::compile(once);
        REQUIRE(a.actions.size() == b.actions.size());
        for (std::size_t i = 0; i < a.actions.size(); ++i) {
            CHECK(a.actions[i].kind == b.actions[i].kind);
            CHECK(a.actions[i].tile.to_string() == b.actions[i].tile.to_string());
        }
    }
    // Precedence and associativity survive: a - (b - c) keeps its parentheses, (a * b) + c drops them.
    const std::string src = R"(csp 1.0
program e machine flat(l3 = 8) {
  tensor A[256,32] tile 32x32 in;
  for i in 0..2 {
    for j in 0..2 {
      resident A[(i * 2) + j, 0];
      release A[4 - (j - i) + 1, 0];
    }
  }
}
)";
    const std::string out = lang::format(lang::parse(src));
    CHECK(out.find("resident A[i * 2 + j, 0];") != std::string::npos);
    CHECK(out.find("release A[4 - (j - i) + 1, 0];") != std::string::npos);
}
