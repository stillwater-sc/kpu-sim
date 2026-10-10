// ============================================================================
// tests/program/test_csp_inherit_retain.cpp
// Residency across operators (docs/plans/kpu-run-csp-programs.md step 4d): `inherit X;` opens a
// program with X resident and no Load; `retain X;` ends a residency with no Release, so the
// slot and the tile outlive the program.
//   - Each fixture runs at L-B, L-T1 and L-CA bit-identical to its L0 oracle; the retaining
//     operator stores nothing, the inheriting one loads nothing it inherits.
//   - A chain: the second operator's inherited C is the first's retained C, at every level, and
//     the chain computes what the two oracles composed compute.
//   - L-T1's slots: inherited ones open at cycle 0, retained ones are held to the makespan.
//   - The validator and the walker refuse the same misuses, by name.
//   - The text round-trips: format is a fixed point, and the trace prints back as inherit/retain.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string read(const std::string& name) {
    std::ifstream in("tests/program/csp/" + name, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

platform::DeviceSpecification s1() { return platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0); }

const std::vector<ExecutionLevel> kLevels = {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                             ExecutionLevel::CycleAccurate};

// Both refuse: the symbolic validator and the walker (compile), with the same reason.
void refused(const std::string& text, const std::string& why) {
    CAPTURE(text);
    CHECK_THROWS_WITH(lang::validate(text), ContainsSubstring(why));
    CHECK_THROWS_WITH(lang::compile(text), ContainsSubstring(why));
}

const char* kHead = "csp 1.0\nprogram p machine flat(l3 = 8) {\n"
                    "  tensor A[32,32] tile 16x16 inout;\n  tensor B[32,32] tile 16x16 in;\n"
                    "  tensor C[32,32] tile 16x16 out;\n";

}  // namespace

TEST_CASE("inherit/retain: each operator runs at every level, bit-identical to its oracle", "[csp][inherit]") {
    const auto dev = s1();
    for (const char* name : {"matmul_relu_retain_32_t16.csp", "bias_inherit_32_t16.csp"}) {
        CAPTURE(name);
        const lang::Program ast = lang::parse(read(name));
        const CspRunResult r = run_csp(CspRunRequest{ast, &dev, csp_inputs(ast), kLevels});
        REQUIRE(r.reference.has_value());
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            REQUIRE_FALSE(o.skipped.has_value());
            for (const auto& op : r.reference->operand_order())
                CHECK(o.values.operand(op).values == r.reference->operand(op).values);
        }
    }
}

TEST_CASE("inherit/retain: what each operator moves", "[csp][inherit]") {
    const auto dev = s1();
    {
        const lang::Program ast = lang::parse(read("matmul_relu_retain_32_t16.csp"));
        const auto v = lang::validate(ast);
        CHECK(v.retained == 4);                     // every C tile, held at the end
        CHECK(v.inherited == 0);
        CHECK(v.stores == std::optional<std::uint64_t>(0));
        const CspRunResult r = run_csp(CspRunRequest{ast, &dev, csp_inputs(ast), kLevels});
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            if (!o.has_timing) continue;
            CHECK(o.dram_stores == 0);              // the result never reaches DRAM
            CHECK(o.dram_loads == 12);              // A by row (4), B by column panel per (i, j) (8)
        }
        // L-T1: four retained slots, held to the makespan.
        const auto& lt1 = r.levels.at(1);
        std::size_t retained = 0;
        for (const auto& sl : lt1.slots)
            if (sl.retained) {
                ++retained;
                CHECK(sl.t1 == lt1.makespan);
                CHECK(sl.tile.operand == "C");
            }
        CHECK(retained == 4);
        CHECK(lt1.peak_l3 <= static_cast<std::size_t>(ast.l3));
    }
    {
        const lang::Program ast = lang::parse(read("bias_inherit_32_t16.csp"));
        const auto v = lang::validate(ast);
        CHECK(v.inherited == 4);
        CHECK(v.retained == 0);
        const CspRunResult r = run_csp(CspRunRequest{ast, &dev, csp_inputs(ast), kLevels});
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            if (!o.has_timing) continue;
            CHECK(o.dram_loads == 2);               // the bias only: C arrived resident
            CHECK(o.dram_stores == 4);              // the store the chain owes C
        }
        const auto& lt1 = r.levels.at(1);
        std::size_t inherited = 0;
        for (const auto& sl : lt1.slots)
            if (sl.inherited) {
                ++inherited;
                CHECK(sl.t0 == 0);
                CHECK_FALSE(sl.retained);
            }
        CHECK(inherited == 4);
    }
}

TEST_CASE("inherit/retain: a chain hands the retained tiles to the next operator", "[csp][inherit]") {
    const auto dev = s1();
    const lang::Program first = lang::parse(read("matmul_relu_retain_32_t16.csp"));
    const lang::Program second = lang::parse(read("bias_inherit_32_t16.csp"));

    // The composed oracle: the first's L0 on its inputs; its C is the second's.
    const TileProgram in1 = csp_inputs(first);
    const auto ref1 = csp_reference(first, in1, 1'000'000);
    REQUIRE(ref1.values.has_value());
    TileProgram in2 = csp_inputs(second);
    in2.operand("C").values = ref1.values->operand("C").values;
    const auto ref2 = csp_reference(second, in2, 1'000'000);
    REQUIRE(ref2.values.has_value());

    for (ExecutionLevel level : kLevels) {
        CAPTURE(to_string(level));
        const CspLevelOutcome a = csp_run_level(level, first, &dev, in1);
        REQUIRE_FALSE(a.skipped.has_value());
        // What the first left in its retained slots is what the second inherits.
        TileProgram handed = csp_inputs(second);
        handed.operand("C").values = a.values.operand("C").values;
        const CspLevelOutcome b = csp_run_level(level, second, &dev, handed);
        REQUIRE_FALSE(b.skipped.has_value());
        CHECK(b.values.operand("C").values == ref2.values->operand("C").values);
    }
}

TEST_CASE("inherit/retain: L-B reports the retained tiles' values", "[csp][inherit]") {
    const lang::Program ast = lang::parse(read("matmul_relu_retain_32_t16.csp"));
    lang::ActionStream s(ast);
    csp::BehavioralInterpreter lb;
    lb.begin(csp_inputs(ast));
    while (auto e = s.next()) lb.step(e->action, e->op ? &*e->op : nullptr);
    (void)lb.finish();
    const auto kept = lb.retained();
    REQUIRE(kept.size() == 4);
    CHECK(kept.count("C[1,1]") == 1);
    CHECK(lb.held(csp::Chan::L3) == 4);
}

TEST_CASE("inherit/retain: the validator and the walker refuse the same misuses", "[csp][inherit]") {
    const std::string h = kHead;
    // inherit opens the program, outside any loop
    refused(h + "  resident B[0, 0];\n  inherit A[0, 0];\n  release A[0, 0];\n  release B[0, 0];\n}\n",
            "inherit comes first");
    refused(h + "  for i in 0..2 { inherit A[i, 0]; release A[i, 0]; }\n}\n", "inherit comes first");
    // an inherited tile arrives with a value
    refused(h + "  inherit C[0, 0];\n  release C[0, 0];\n}\n", "declared out");
    // a retained tile cannot be named again
    refused(h + "  resident A[0, 0];\n  retain A[0, 0];\n  resident A[0, 0];\n  release A[0, 0];\n}\n",
            "retained");
    // retain names something resident or an accumulator
    refused(h + "  retain A[0, 0];\n}\n", "neither resident nor an accumulator");
    // a resident tile's retain moves nothing to run stages on
    refused(h + "  resident A[0, 0];\n  retain A[0, 0] via relu @ fabric;\n}\n", "stays where it is");
    // every iteration retains different tiles: proved over the loop, or found on its second iteration
    const std::string same_tile = h + "  for k in 0..2 { resident B[0, 0]; retain B[0, 0]; }\n}\n";
    CHECK_THROWS_WITH(lang::validate(same_tile), ContainsSubstring("every iteration must retain different tiles"));
    CHECK_THROWS_WITH(lang::compile(same_tile), ContainsSubstring("B[0,0] is retained"));
    // retained slots count against capacity to the end
    refused("csp 1.0\nprogram p machine flat(l3 = 4) {\n  tensor A[64,64] tile 16x16 in;\n"
            "  resident A[0, :];\n  retain A[0, :];\n  resident A[1, 0];\n  release A[1, 0];\n}\n",
            "L3 slot");
}

TEST_CASE("inherit/retain: the inherited tile may be stored without being written", "[csp][inherit]") {
    // The chain's store: the operator before retained a result it did not store.
    const std::string text = std::string(kHead) + "  inherit A[:, :];\n  store A[:, :];\n  release A[:, :];\n}\n";
    CHECK_NOTHROW(lang::validate(text));
    const auto p = lang::compile(text);
    CHECK(p.reuse().stores == 4);
    CHECK(p.reuse().loads == 0);
    // ...but a loaded one still may not.
    refused(std::string(kHead) + "  resident A[:, :];\n  store A[:, :];\n  release A[:, :];\n}\n",
            "nothing has written it");
}

TEST_CASE("inherit/retain: the text round-trips", "[csp][inherit]") {
    for (const char* name : {"matmul_relu_retain_32_t16.csp", "bias_inherit_32_t16.csp"}) {
        CAPTURE(name);
        const lang::Program ast = lang::parse(read(name));
        const std::string canon = lang::format(ast);
        CHECK(lang::format(lang::parse(canon)) == canon);
        // The trace prints back as a program with the same actions.
        const auto trace = lang::compile(ast);
        const std::string printed = lang::print(trace);
        CAPTURE(printed);
        const auto again = lang::compile(printed);
        REQUIRE(again.actions.size() == trace.actions.size());
        for (std::size_t i = 0; i < trace.actions.size(); ++i) {
            CHECK(again.actions[i].kind == trace.actions[i].kind);
            CHECK(again.actions[i].tile.to_string() == trace.actions[i].tile.to_string());
        }
    }
    CHECK_THAT(lang::format(lang::parse(read("matmul_relu_retain_32_t16.csp"))),
               ContainsSubstring("retain C[i, j] via relu @ fabric;"));
    CHECK_THAT(lang::format(lang::parse(read("bias_inherit_32_t16.csp"))), ContainsSubstring("inherit C[:, :];"));
}
