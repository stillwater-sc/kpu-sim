// ============================================================================
// tests/program/test_csp_language.cpp
// The CSP language (docs/plans/csp-language.md step 1; ADR 0004): written programs parse,
// validate, compile to the CSP program IR, print back exactly, and compute the L0 reference's
// values; the validator refuses each class of error by line; derived L0 is emitted as text.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/emit.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

// matmul 256^3 / 32^3, written: A resident for the whole run, B's column panel per j.
const char* kMatmulAllA = R"(csp 1.0
// matmul, A held whole: every tile loads once
program matmul machine flat(l3 = 128) {
  tensor A[256,256] tile 32x32 in;
  tensor B[256,256] tile 32x32 in;
  tensor C[256,256] tile 32x32 out;

  resident A[:, :];                       // 64 tiles, for the whole run
  for j in 0..8 {
    resident B[:, j];                     // B's column panel
    for i in 0..8 {
      acc C[i, j] in fabric {
        for k in 0..8 {
          call gemm(A[i, k], B[k, j]) +-> C[i, j];
        }
      }
      store C[i, j];
    }
    release B[:, j];
  }
  release A[:, :];
}
)";

// The same product, with A's row panel reloaded for every column panel of B.
const char* kMatmulPanels = R"(csp 1.0
program matmul machine flat(l3 = 128) {
  tensor A[256,256] tile 32x32 in;
  tensor B[256,256] tile 32x32 in;
  tensor C[256,256] tile 32x32 out;
  for j in 0..8 {
    resident B[:, j];
    for i in 0..8 {
      resident A[i, :];
      acc C[i, j] in fabric {
        for k in 0..8 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }
      }
      store C[i, j];
      release A[i, :];
    }
    release B[:, j];
  }
}
)";

// Right-looking LU, 64^2 / 16^2, the whole matrix resident.
const char* kLu = R"(csp 1.0
program lu machine flat(l3 = 16) {
  tensor A[64,64] tile 16x16 inout;
  resident A[:, :];
  for k in 0..4 {
    call getrf(A[k, k]) -> A[k, k] pivot k;
    for g in 0..k { call laswp(A[k, g]) -> A[k, g] pivot k; }
    for j in k+1..4 { call laswp(A[k, j]) -> A[k, j] pivot k; }
    for j in k+1..4 { call trsm_ll(A[k, k], A[k, j]) -> A[k, j]; }
    for i in k+1..4 { call trsm_ur(A[k, k], A[i, k]) -> A[i, k]; }
    for i in k+1..4 {
      for j in k+1..4 { call gemm(A[i, k], A[k, j]) +-> A[i, j] alpha -1; }
    }
  }
  store A[:, :];
  release A[:, :];
}
)";

driver::ProgramSpec spec(const std::string& algo, Dim size, Dim tile) {
    driver::ProgramSpec s;
    s.algo = algo;
    s.size = size;
    s.tile = tile;
    return s;
}

// The written program's values against the derived program's L0 reference.
void check_values(CspProgram p, const driver::ProgramSpec& s, const std::string& out) {
    driver::fill(p.source, s);
    TileProgram ref = driver::derive(s);
    driver::fill(ref, s);
    TileProgramReference().run(ref);
    BehavioralInterpreter lb;
    lb.run(p);
    CHECK(lb.result().operand(out).values == ref.operand(out).values);
}

std::size_t count(const CspProgram& p, Action::Kind k) {
    std::size_t n = 0;
    for (const auto& a : p.actions) n += a.kind == k;
    return n;
}

// The actions two programs perform, compared statement-level: kind, tile, process, accumulate.
bool same_actions(const CspProgram& a, const CspProgram& b) {
    if (a.actions.size() != b.actions.size()) return false;
    for (std::size_t i = 0; i < a.actions.size(); ++i) {
        const auto& x = a.actions[i];
        const auto& y = b.actions[i];
        if (x.kind != y.kind || x.tile.to_string() != y.tile.to_string() || x.process != y.process ||
            x.accumulate != y.accumulate)
            return false;
    }
    return true;
}

std::string compile_error(const std::string& body, const std::string& decls =
                              "  tensor A[64,64] tile 32x32 in;\n  tensor B[64,64] tile 32x32 in;\n"
                              "  tensor C[64,64] tile 32x32 out;\n",
                          int l3 = 4) {
    try {
        lang::compile("csp 1.0\nprogram t machine flat(l3 = " + std::to_string(l3) + ") {\n" + decls + body + "}\n");
    } catch (const lang::CompileError& e) {
        return e.what();
    } catch (const lang::SyntaxError& e) {
        return e.what();
    }
    return "compiled";
}

}  // namespace

TEST_CASE("CSP language: a written matmul computes the reference, loading exactly what it says",
          "[program][csp][lang]") {
    const CspProgram all_a = lang::compile(kMatmulAllA);
    CHECK(all_a.reuse().loads == 128);                    // 64 A + 64 B: each tile once
    CHECK(all_a.reuse().peak_l3 <= 128);
    check_values(all_a, spec("matmul", 256, 32), "C");

    // A different residency is a different program, with the loads it wrote.
    const CspProgram panels = lang::compile(kMatmulPanels);
    CHECK(panels.reuse().loads == 64 + 8 * 64);           // B once; A's row panel per column panel
    check_values(panels, spec("matmul", 256, 32), "C");
}

TEST_CASE("CSP language: a written LU computes the reference", "[program][csp][lang]") {
    const CspProgram lu = lang::compile(kLu);
    CHECK(lu.reuse().loads == 16);
    CHECK(count(lu, Action::Kind::Store) == 16);
    check_values(lu, spec("lu", 64, 16), "A");
}

TEST_CASE("CSP language: print and compile reproduce a program exactly", "[program][csp][lang]") {
    for (const char* src : {kMatmulAllA, kMatmulPanels, kLu}) {
        const CspProgram p = lang::compile(src);
        const std::string text = lang::print(p);
        CAPTURE(text.substr(0, 400));
        const CspProgram q = lang::compile(text);
        CHECK(same_actions(p, q));
        CHECK(lang::print(q) == text);                    // the printed form is a fixed point
    }
}

TEST_CASE("CSP language: derived L0 is emitted as a program that compiles and computes the reference",
          "[program][csp][lang]") {
    for (std::size_t cap : {std::size_t{128}, std::size_t{16}, std::size_t{4}}) {
        CAPTURE(cap);
        const auto s = spec("matmul", 128, 32);
        TileProgram l0 = driver::derive(s);
        const std::string text = lang::emit(l0, cap, "matmul");
        const CspProgram p = lang::compile(text);
        CHECK(p.reuse().peak_l3 <= cap);
        if (cap >= 32) CHECK(p.reuse().reloads == 0);      // all 32 input tiles fit
        check_values(p, s, "C");
    }
    for (std::size_t cap : {std::size_t{16}, std::size_t{4}}) {
        CAPTURE(cap);
        const auto s = spec("lu", 64, 16);
        const std::string text = lang::emit(driver::derive(s), cap, "lu");
        const CspProgram p = lang::compile(text);
        CHECK(p.reuse().peak_l3 <= cap);
        check_values(p, s, "A");
    }
}

TEST_CASE("CSP language: the validator refuses each error by line", "[program][csp][lang]") {
    // Residency is explicit only: a call on a tile the program did not make resident.
    CHECK_THAT(compile_error("  acc C[0, 0] in fabric {\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n"),
               ContainsSubstring("csp line 7: call gemm reads A[0,0], which is not resident"));
    // Capacity, checked statically.
    CHECK_THAT(compile_error("  resident A[:, :];\n  resident B[0, 0];\n"),
               ContainsSubstring("csp line 7: resident B[0,0] needs an L3 slot, and all 4 are held"));
    // A read after release.
    CHECK_THAT(compile_error("  resident A[0, 0], B[0, 0];\n  release A[0, 0];\n  acc C[0, 0] in fabric {\n"
                             "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n"),
               ContainsSubstring("csp line 9: call gemm reads A[0,0], which is not resident"));
    // Resident twice; release of what is not resident.
    CHECK_THAT(compile_error("  resident A[0, 0];\n  resident A[0, 0];\n"),
               ContainsSubstring("csp line 7: A[0,0] is already resident"));
    CHECK_THAT(compile_error("  release A[0, 0];\n"), ContainsSubstring("csp line 6: release A[0,0]: it is not resident"));
    // A store of a tile nothing wrote; an accumulator never stored; a tile left resident.
    CHECK_THAT(compile_error("  resident A[0, 0];\n  store A[0, 0];\n"),
               ContainsSubstring("csp line 7: store A[0,0]: nothing has written it"));
    CHECK_THAT(compile_error("  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric {\n"
                             "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n  release A[0, 0], B[0, 0];\n"),
               ContainsSubstring("the accumulator C[0,0] is never stored"));
    CHECK_THAT(compile_error("  resident A[0, 0];\n"), ContainsSubstring("A[0,0] is still resident at the end"));
    // A written, unstored tile cannot be released.
    CHECK_THAT(compile_error("  resident A[0, 0];\n  call getrf(A[0, 0]) -> A[0, 0] pivot 0;\n  release A[0, 0];\n",
                             "  tensor A[64,64] tile 32x32 inout;\n"),
               ContainsSubstring("csp line 6: release A[0,0]: a call wrote it and it is not stored"));
    // Indices, functions, and what this step does not build yet.
    CHECK_THAT(compile_error("  resident A[2, 0];\n"), ContainsSubstring("csp line 6: A: tile index 2 is outside 0..2"));
    CHECK_THAT(compile_error("  resident A[0, 0];\n  call gemv(A[0, 0]) -> A[0, 0];\n"),
               ContainsSubstring("unknown tile function 'gemv'"));
    CHECK_THAT(compile_error("  broadcast A[0, :] along row;\n"), ContainsSubstring("level 2"));
    CHECK_THAT(compile_error("  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric {\n"
                             "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n  store C[0, 0] via relu @ bm.egress;\n"),
               ContainsSubstring("tile contexts ('via') arrive with the linear operator"));
    // Syntax errors name their line too.
    CHECK_THAT(compile_error("  resident A[0, 0]\n  release A[0, 0];\n"), ContainsSubstring("csp line 6: expected ';'"));
}

TEST_CASE("CSP language: review fixes -- loop bounds from variables, empty accumulators, nested accumulators, alpha",
          "[program][csp][lang]") {
    const std::string decls = "  tensor A[64,64] tile 16x16 in;\n  tensor B[64,64] tile 16x16 in;\n"
                              "  tensor C[64,64] tile 16x16 out;\n";
    auto program = [&](const std::string& body) {
        return "csp 1.0\nprogram t machine flat(l3 = 40) {\n" + decls + body + "}\n";
    };
    // A loop whose bounds are both variables: `i..k`, `k..k+2`.
    const CspProgram tri = lang::compile(program(
        "  resident A[:, :];\n"
        "  for k in 0..4 { for i in k..4 { for j in i..k+1 { release A[i, j]; resident A[i, j]; } } }\n"
        "  release A[:, :];\n"));
    CHECK(tri.reuse().loads > 16);
    // An accumulator that receives no call is refused by line.
    CHECK_THAT(compile_error("  acc C[0, 0] in fabric {\n    for k in 0..0 { }\n  }\n  store C[0, 0];\n"),
               ContainsSubstring("csp line 6: acc C[0,0] receives no call"));

    // Two accumulators, sequential and stored later; and interleaved (register-blocked) calls.
    // Both print, round-trip exactly, and compute the reference.
    const std::string sequential = program(
        "  resident A[:, :], B[:, :];\n"
        "  acc C[0, 0] in fabric { for k in 0..4 { call gemm(A[0, k], B[k, 0]) +-> C[0, 0]; } }\n"
        "  acc C[0, 1] in fabric { for k in 0..4 { call gemm(A[0, k], B[k, 1]) +-> C[0, 1]; } }\n"
        "  store C[0, 0];\n  store C[0, 1];\n"
        "  release A[:, :], B[:, :];\n");
    const std::string blocked = program(
        "  resident A[:, :], B[:, :];\n"
        "  acc C[0, 0] in fabric {\n    acc C[0, 1] in fabric {\n"
        "      for k in 0..4 {\n"
        "        call gemm(A[0, k], B[k, 0]) +-> C[0, 0];\n        call gemm(A[0, k], B[k, 1]) +-> C[0, 1];\n"
        "      }\n    }\n  }\n"
        "  store C[0, 0];\n  store C[0, 1];\n"
        "  release A[:, :], B[:, :];\n");
    for (const std::string& src : {sequential, blocked}) {
        const CspProgram p = lang::compile(src);
        const std::string text = lang::print(p);
        CAPTURE(text);
        const CspProgram q = lang::compile(text);
        CHECK(same_actions(p, q));
        CHECK(lang::print(q) == text);
        CspProgram v = p;
        auto s = spec("matmul", 64, 16);
        driver::fill(v.source, s);
        TileProgram ref = driver::derive(s);
        driver::fill(ref, s);
        TileProgramReference().run(ref);
        BehavioralInterpreter lb;
        lb.run(v);
        const auto& got = lb.result().operand("C");
        const auto& want = ref.operand("C");
        for (Dim r = 0; r < 16; ++r)
            for (Dim c = 0; c < 32; ++c) CHECK(got.at(r, c) == want.at(r, c));   // C[0,0] and C[0,1]
    }

    // alpha reads back to the same float, exponents included.
    for (float a : {0.1f, -1.0f / 3.0f, 1e-7f, 6.02e23f}) {
        CAPTURE(a);
        const std::string src = "csp 1.0\nprogram t machine flat(l3 = 8) {\n  tensor A[16,16] tile 16x16 inout;\n"
                                "  tensor B[16,16] tile 16x16 in;\n  resident A[0, 0], B[0, 0];\n"
                                "  call gemm(B[0, 0], B[0, 0]) +-> A[0, 0] alpha " + lang::alpha_text(a) + ";\n"
                                "  store A[0, 0];\n  release A[0, 0], B[0, 0];\n}\n";
        const CspProgram p = lang::compile(src);
        float got = 0;
        for (const auto& op : p.source.ops()) if (op.kind == TileOpKind::MatMulAccum) got = op.alpha;
        CHECK(got == a);
        CHECK(lang::print(lang::compile(lang::print(p))) == lang::print(p));
    }
}
