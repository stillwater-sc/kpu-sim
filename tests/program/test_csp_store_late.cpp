// ============================================================================
// tests/program/test_csp_store_late.cpp
// `store C[i, j] late;` (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 2b): an
// accumulator's store whose Drain, Writeback, Store and Release the walker emits after the NEXT
// accumulator's calls, or at the end of the program, or before a statement that names the tile
// again. The movers issue in program order, so the next chain's Loads and Moves then go ahead of
// this chain's results, and the chains pipeline on the fabric. Values never change.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>

#include <cstddef>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

// Row panels of A outer, column panels of B inner, one accumulator per C tile.
std::string matmul(bool late, int l3 = 9) {
    return std::string("csp 1.0\nprogram mm machine flat(l3 = ") + std::to_string(l3) + ") {\n"
           "  tensor A[64,64] tile 16x16 in;\n  tensor B[64,64] tile 16x16 in;\n  tensor C[64,64] tile 16x16 out;\n"
           "  for i in 0..4 {\n"
           "    resident A[i, :];\n"
           "    for j in 0..4 {\n"
           "      resident B[:, j];\n"
           "      acc C[i, j] in fabric {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
           "      }\n"
           "      store C[i, j]" + (late ? " late" : "") + ";\n"
           "      release B[:, j];\n"
           "    }\n"
           "    release A[i, :];\n"
           "  }\n"
           "}\n";
}

std::string error_of(const std::string& src, bool symbolic) {
    try {
        if (symbolic) (void)lang::validate(src);
        else (void)lang::compile(src);
    } catch (const lang::CompileError& e) {
        return e.what();
    }
    return "accepted";
}

// The index of the first action of `kind` on tile `t`.
std::size_t first(const CspProgram& p, Action::Kind kind, const std::string& t) {
    for (std::size_t i = 0; i < p.actions.size(); ++i)
        if (p.actions[i].kind == kind && p.actions[i].tile.to_string() == t) return i;
    FAIL("no " << to_string(kind) << " of " << t);
    return 0;
}
std::size_t last(const CspProgram& p, Action::Kind kind, const std::string& t) {
    std::size_t at = p.actions.size();
    for (std::size_t i = 0; i < p.actions.size(); ++i)
        if (p.actions[i].kind == kind && p.actions[i].tile.to_string() == t) at = i;
    REQUIRE(at != p.actions.size());
    return at;
}

TileProgram lb(const std::string& src) {
    CspProgram trace = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : trace.source.operand_order()) trace.source.operand(name).values = in.operand(name).values;
    BehavioralInterpreter b;
    b.run(trace);
    return b.result();
}

}  // namespace

TEST_CASE("store late: parsed, formatted, and validated in place", "[program][csp][lang][late]") {
    const lang::Program ast = lang::parse(matmul(true));
    CHECK_THAT(lang::format(ast), ContainsSubstring("store C[i, j] late;"));
    CHECK(lang::format(lang::parse(lang::format(ast))) == lang::format(ast));
    const lang::Validation late = lang::validate(matmul(true)), now = lang::validate(matmul(false));
    CHECK(late.peak_l3 == now.peak_l3);                  // the writeback's slot, at a later moment
    CHECK(late.stores == now.stores);
}

TEST_CASE("store late: the results leave after the next accumulator's calls", "[program][csp][lang][late]") {
    const CspProgram p = lang::compile(matmul(true));
    // C[0,0] drains after C[0,1]'s last call, and before C[0,2]'s first.
    CHECK(first(p, Action::Kind::Drain, "C[0,0]") > last(p, Action::Kind::Call, "C[0,1]"));
    CHECK(first(p, Action::Kind::Drain, "C[0,0]") < first(p, Action::Kind::Call, "C[0,2]"));
    // So the dma loads the next panels before it stores this result.
    CHECK(first(p, Action::Kind::Load, "B[0,1]") < first(p, Action::Kind::Store, "C[0,0]"));
    // Across the outer loop too: C[0,3] leaves after C[1,0]'s calls, behind A[1, :]'s Loads.
    CHECK(first(p, Action::Kind::Load, "A[1,0]") < first(p, Action::Kind::Store, "C[0,3]"));
    CHECK(first(p, Action::Kind::Drain, "C[0,3]") > last(p, Action::Kind::Call, "C[1,0]"));
    // The last one at the end of the program.
    CHECK(last(p, Action::Kind::Release, "C[3,3]") == p.actions.size() - 1);
    // Each store's four actions stay together, in order.
    const std::size_t d = first(p, Action::Kind::Drain, "C[2,1]");
    CHECK(p.actions[d + 1].kind == Action::Kind::Writeback);
    CHECK(p.actions[d + 2].kind == Action::Kind::Store);
    CHECK(p.actions[d + 3].kind == Action::Kind::Release);
}

TEST_CASE("store late: the actions carry the store's own line and loop indices", "[program][csp][lang][late]") {
    lang::ActionStream s(lang::parse(matmul(true)));
    bool seen = false;
    while (auto e = s.next()) {
        if (e->action.kind != Action::Kind::Store || e->action.tile.to_string() != "C[1,3]") continue;
        CHECK(e->where.line == 13);
        REQUIRE(e->where.loops.size() == 2);
        CHECK(e->where.loops[0].value == 1);             // i = 1, though it leaves during i = 2
        CHECK(e->where.loops[1].value == 3);
        seen = true;
    }
    CHECK(seen);
}

TEST_CASE("store late: the stream yields the trace's actions", "[program][csp][lang][late]") {
    const std::string src = matmul(true);
    const CspProgram trace = lang::compile(src);
    lang::ActionStream stream(lang::parse(src));
    std::size_t i = 0;
    while (auto e = stream.next()) {
        REQUIRE(i < trace.actions.size());
        CAPTURE(i);
        CHECK(e->action.kind == trace.actions[i].kind);
        CHECK(e->action.tile.to_string() == trace.actions[i].tile.to_string());
        CHECK(e->action.residency == trace.actions[i].residency);
        ++i;
    }
    CHECK(i == trace.actions.size());
}

TEST_CASE("store late: L-B computes the same values", "[program][csp][lang][late]") {
    CHECK(lb(matmul(true)).operand("C").values == lb(matmul(false)).operand("C").values);
}

TEST_CASE("store late: naming the tile again lets it out first", "[program][csp][lang][late]") {
    // The second pass loads C[0,0] back: the late store must reach DRAM before that Load.
    const std::string src =
        "csp 1.0\nprogram t machine flat(l3 = 4) {\n"
        "  tensor A[16,16] tile 16x16 in;\n  tensor B[16,16] tile 16x16 in;\n  tensor C[16,16] tile 16x16 inout;\n"
        "  resident A[0, 0], B[0, 0];\n"
        "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
        "  store C[0, 0] late;\n"
        "  release A[0, 0], B[0, 0];\n"
        "  resident C[0, 0];\n  call relu(C[0, 0]) -> C[0, 0];\n  store C[0, 0];\n  release C[0, 0];\n}\n";
    REQUIRE(lang::validate(src).peak_l3 <= 4);
    const CspProgram p = lang::compile(src);
    CHECK(first(p, Action::Kind::Store, "C[0,0]") < first(p, Action::Kind::Load, "C[0,0]"));
    CHECK(first(p, Action::Kind::Release, "A[0,0]") < first(p, Action::Kind::Drain, "C[0,0]"));
}

TEST_CASE("store late: what it refuses, by line", "[program][csp][lang][late]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 3) {\n"
                             "  tensor A[16,16] tile 16x16 inout;\n  tensor B[16,16] tile 16x16 in;\n"
                             "  tensor C[32,16] tile 16x16 out;\n  vector b[16] tile 16 in;\n";
    // A resident tile is stored where it is written.
    const std::string resident = head + "  resident A[0, 0];\n  call relu(A[0, 0]) -> A[0, 0];\n"
                                        "  store A[0, 0] late;\n  release A[0, 0];\n}\n";
    for (bool symbolic : {true, false})
        CHECK_THAT(error_of(resident, symbolic), ContainsSubstring(symbolic ? "csp line 9: store A[0, 0] late: late defers"
                                                                             : "csp line 9: store A[0,0] late: late defers"));
    // No epilogue: the vector it reads may be gone where the store is emitted.
    std::string roomy = head;
    roomy.replace(roomy.find("l3 = 3"), 6, "l3 = 4");
    const std::string via = roomy + "  resident A[0, 0], B[0, 0], b[0];\n"
                                   "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                   "  store C[0, 0] late via add(b[0]) @ fabric;\n  release A[0, 0], B[0, 0], b[0];\n}\n";
    CHECK_THAT(error_of(via, true), ContainsSubstring("csp line 9: store C[0, 0] late via ...: a late store carries no "
                                                      "epilogue"));
    // The writeback needs its slot where the store leaves, and a third resident fills the L3
    // there: the validator refuses at the next acc's open, the walker at its close, where the
    // store is emitted (with the store's own line).
    const std::string full = head + "  resident A[0, 0], B[0, 0];\n"
                                    "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                    "  store C[0, 0] late;\n  resident b[0];\n"
                                    "  acc C[1, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[1, 0]; }\n"
                                    "  store C[1, 0];\n  release A[0, 0], B[0, 0], b[0];\n}\n";
    CHECK_THAT(error_of(full, true), ContainsSubstring("csp line 11: a late store is emitted at the open of acc C[1, 0]"));
    CHECK_THAT(error_of(full, false), ContainsSubstring("csp line 9: store C[0,0]'s writeback needs an L3 slot"));
    // The peak counts the writeback where a late store can land, never less than the walker holds:
    // here a third resident and the deferred writeback, at the next acc's close.
    const std::string peak = head + "  resident A[0, 0], B[0, 0];\n"
                                    "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                    "  store C[0, 0] late;\n  resident b[0];\n"
                                    "  acc C[1, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[1, 0]; }\n"
                                    "  store C[1, 0];\n  release A[0, 0], B[0, 0], b[0];\n}\n";
    std::string four = peak;
    four.replace(four.find("l3 = 3"), 6, "l3 = 4");
    CHECK(lang::validate(four).peak_l3 == 4);
    CHECK(lang::compile(four).reuse().peak_l3 == 4);
    // Stored twice.
    const std::string twice = head + "  resident A[0, 0], B[0, 0];\n"
                                     "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                     "  store C[0, 0] late;\n  store C[0, 0];\n  release A[0, 0], B[0, 0];\n}\n";
    CHECK_THAT(error_of(twice, false), ContainsSubstring("csp line 10: store C[0,0]: its late store is already pending"));
}

TEST_CASE("store late: kpu-csp-gen's matmul writes it, and its fused linear does not", "[program][csp][gen][late]") {
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 64;
    for (const char* orient : {"col", "row"}) {
        o.orient = orient;
        CHECK_THAT(gen::generate(o), ContainsSubstring("store C[i, j] late;"));
    }
    o.algo = "linear";
    CHECK_THAT(gen::generate(o), !ContainsSubstring(" late"));
    o.place = "unfused";                                 // the plain nest, then C read back
    const std::string unfused = gen::generate(o);
    CHECK_THAT(unfused, ContainsSubstring("store C[i, j] late;"));
    const CspProgram p = lang::compile(unfused);
    CHECK(first(p, Action::Kind::Store, "C[3,3]") < first(p, Action::Kind::Load, "C[3,3]"));
}
