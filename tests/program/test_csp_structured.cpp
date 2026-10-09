// ============================================================================
// tests/program/test_csp_structured.cpp
// The structured CSP program executes and validates WITHOUT unrolling (docs/plans/csp-language.md
// step 1c; ADR 0004 §4): the ActionStream yields exactly the trace's actions, one at a time, in
// bounded memory; the symbolic validator proves residency, capacity and bounds over the loop
// structure; a 1M x 1M matmul validates and streams.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string matmul_all_a(int n_tiles, int tile, int l3) {
    const std::string N = std::to_string(n_tiles * tile), T = std::to_string(tile), n = std::to_string(n_tiles);
    return "csp 1.0\nprogram matmul machine flat(l3 = " + std::to_string(l3) + ") {\n"
           "  tensor A[" + N + "," + N + "] tile " + T + "x" + T + " in;\n"
           "  tensor B[" + N + "," + N + "] tile " + T + "x" + T + " in;\n"
           "  tensor C[" + N + "," + N + "] tile " + T + "x" + T + " out;\n"
           "  resident A[:, :];\n"
           "  for j in 0.." + n + " {\n"
           "    resident B[:, j];\n"
           "    for i in 0.." + n + " {\n"
           "      acc C[i, j] in fabric { for k in 0.." + n + " { call gemm(A[i, k], B[k, j]) +-> C[i, j]; } }\n"
           "      store C[i, j];\n"
           "    }\n"
           "    release B[:, j];\n"
           "  }\n"
           "  release A[:, :];\n"
           "}\n";
}

// One tile of A and of B at a time: L3 holds three tiles whatever the problem size.
std::string matmul_streaming(long long size, int tile, int l3) {
    const std::string N = std::to_string(size), T = std::to_string(tile), n = std::to_string(size / tile);
    return "csp 1.0\nprogram matmul machine flat(l3 = " + std::to_string(l3) + ") {\n"
           "  tensor A[" + N + "," + N + "] tile " + T + "x" + T + " in;\n"
           "  tensor B[" + N + "," + N + "] tile " + T + "x" + T + " in;\n"
           "  tensor C[" + N + "," + N + "] tile " + T + "x" + T + " out;\n"
           "  for i in 0.." + n + " {\n"
           "    for j in 0.." + n + " {\n"
           "      acc C[i, j] in fabric {\n"
           "        for k in 0.." + n + " {\n"
           "          resident A[i, k], B[k, j];\n"
           "          call gemm(A[i, k], B[k, j]) +-> C[i, j];\n"
           "          release A[i, k], B[k, j];\n"
           "        }\n"
           "      }\n"
           "      store C[i, j];\n"
           "    }\n"
           "  }\n"
           "}\n";
}

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

std::string symbolic_error(const std::string& src) {
    try {
        lang::validate(src);
    } catch (const lang::CompileError& e) {
        return e.what();
    }
    return "validated";
}

}  // namespace

TEST_CASE("CSP structured: the stream yields exactly the trace's actions, one at a time",
          "[program][csp][lang][structured]") {
    for (const std::string& src : {matmul_all_a(4, 32, 64), matmul_streaming(128, 32, 3), std::string(kLu)}) {
        const CspProgram trace = lang::compile(src);
        lang::ActionStream stream(lang::parse(src));
        std::size_t i = 0, most_buffered = 0;
        while (auto e = stream.next()) {
            REQUIRE(i < trace.actions.size());
            const Action& t = trace.actions[i];
            CAPTURE(i);
            CHECK((e->action.kind == t.kind && e->action.tile.to_string() == t.tile.to_string() &&
                   e->action.process == t.process && e->action.accumulate == t.accumulate &&
                   e->action.residency == t.residency));
            most_buffered = std::max(most_buffered, stream.buffered());
            ++i;
        }
        CHECK(i == trace.actions.size());
        // One statement's actions at most, never the program: a slice statement (resident A[:, :])
        // emits up to the L3 capacity of them, a call at most 8.
        CHECK(most_buffered <= std::max<std::size_t>(stream.capacity(), 8));
    }
}

TEST_CASE("CSP structured: L-B over the stream computes what L-B over the trace does",
          "[program][csp][lang][structured]") {
    auto check = [](const std::string& src, const char* algo, Dim size, Dim tile, const char* out) {
        driver::ProgramSpec s;
        s.algo = algo;
        s.size = size;
        s.tile = tile;
        CspProgram trace = lang::compile(src);
        driver::fill(trace.source, s);
        BehavioralInterpreter by_trace;
        by_trace.run(trace);

        const lang::Program ast = lang::parse(src);
        lang::ActionStream stream(ast);
        BehavioralInterpreter by_stream;
        by_stream.begin(trace.source);                   // the same operand values
        while (auto e = stream.next()) by_stream.step(e->action, e->op ? &*e->op : nullptr);
        by_stream.finish();
        CHECK(by_stream.result().operand(out).values == by_trace.result().operand(out).values);

        TileProgram ref = driver::derive(s);
        driver::fill(ref, s);
        TileProgramReference().run(ref);
        CHECK(by_stream.result().operand(out).values == ref.operand(out).values);
    };
    check(matmul_all_a(4, 32, 64), "matmul", 128, 32, "C");
    check(matmul_streaming(128, 32, 3), "matmul", 128, 32, "C");
    check(kLu, "lu", 64, 16, "A");
}

TEST_CASE("CSP structured: symbolic validation agrees with the trace -- peak, loads, calls, stores",
          "[program][csp][lang][structured]") {
    for (const std::string& src : {matmul_all_a(8, 32, 128), matmul_streaming(256, 32, 3)}) {
        const lang::Validation v = lang::validate(src);
        const CspProgram trace = lang::compile(src);
        const ReuseReport r = trace.reuse();
        CHECK(v.peak_l3 == r.peak_l3);
        REQUIRE(v.loads);
        CHECK(*v.loads == r.loads);
        CHECK(*v.calls == r.calls);
        CHECK(*v.stores == r.stores);
    }
    // LU's triangular loops: proven valid; totals are left out (not rectangular).
    const lang::Validation lu = lang::validate(kLu);
    CHECK(lu.peak_l3 == 16);
    CHECK_FALSE(lu.loads.has_value());
}

TEST_CASE("CSP structured: a 1M x 1M matmul validates and streams without unrolling",
          "[program][csp][lang][structured]") {
    // 1,048,576^2 with 32 x 32 tiles: 32,768 tiles a side, 3.5 x 10^13 calls.
    const std::string src = matmul_streaming(1048576, 32, 3);
    const auto t0 = std::chrono::steady_clock::now();
    const lang::Validation v = lang::validate(src);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const std::uint64_t n = 32768;
    CHECK(v.peak_l3 == 2);                               // A[i,k] and B[k,j]; C's writeback comes after
    REQUIRE(v.loads);
    CHECK(*v.loads == 2 * n * n * n);
    CHECK(*v.calls == n * n * n);
    CHECK(*v.stores == n * n);
    CHECK(ms < 1000.0);
    std::printf("csp 1M x 1M matmul: validated in %.2f ms -- peak L3 %zu, %llu loads, %llu calls, %llu stores\n", ms,
                v.peak_l3, static_cast<unsigned long long>(*v.loads), static_cast<unsigned long long>(*v.calls),
                static_cast<unsigned long long>(*v.stores));

    // The stream starts at once and stays small: the first million actions, never the program.
    lang::ActionStream stream(lang::parse(src));
    std::size_t calls = 0, most_buffered = 0;
    for (std::size_t i = 0; i < 1'000'000; ++i) {
        auto e = stream.next();
        REQUIRE(e);
        calls += e->action.kind == Action::Kind::Call;
        most_buffered = std::max(most_buffered, stream.buffered());
    }
    CHECK(calls > 100'000);
    CHECK(most_buffered <= 8);
}

TEST_CASE("CSP structured: the symbolic validator refuses by line, without executing",
          "[program][csp][lang][structured]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8) {\n"
                             "  tensor A[128,128] tile 32x32 in;\n  tensor B[128,128] tile 32x32 in;\n"
                             "  tensor C[128,128] tile 32x32 out;\n";
    // An index that runs past the edge for some iteration.
    CHECK_THAT(symbolic_error(head + "  for i in 0..4 { resident A[i+1, 0]; release A[i+1, 0]; }\n}\n"),
               ContainsSubstring("csp line 6: A: tile index i + 1 may be outside 0..4"));
    // A family that can overlap a resident one.
    CHECK_THAT(symbolic_error(head + "  resident A[0, :];\n  for i in 0..4 { resident A[i, 0]; release A[i, 0]; }\n"
                                     "  release A[0, :];\n}\n"),
               ContainsSubstring("csp line 7: A[i, 0] may already be resident"));
    // An unbalanced loop: what an iteration makes resident, it must release.
    CHECK_THAT(symbolic_error(head + "  for i in 0..4 { resident A[i, 0]; }\n}\n"),
               ContainsSubstring("csp line 6: the loop over i leaves A[i, 0] resident"));
    // A call on a tile that is not provably resident.
    CHECK_THAT(symbolic_error(head + "  resident A[0, :], B[:, 0];\n  acc C[0, 0] in fabric {\n"
                                     "    for k in 0..4 { call gemm(A[k, 0], B[k, 0]) +-> C[0, 0]; }\n  }\n"
                                     "  store C[0, 0];\n  release A[0, :], B[:, 0];\n}\n"),
               ContainsSubstring("csp line 8: call gemm reads A[k, 0], which is not provably resident"));
    // Capacity, from family sizes: a 1M matmul's column panel of B is 32,768 tiles.
    CHECK_THAT(symbolic_error("csp 1.0\nprogram t machine flat(l3 = 128) {\n"
                              "  tensor B[1048576,1048576] tile 32x32 in;\n"
                              "  for j in 0..32768 { resident B[:, j]; release B[:, j]; }\n}\n"),
               ContainsSubstring("csp line 4: resident B[:, j] needs 32768 L3 slots, and 128 of 128 are free"));
    // An accumulator a loop may never feed: a loop that never runs, and one whose trip count can
    // be zero for some outer iteration (`0..j` at j = 0).
    CHECK_THAT(symbolic_error(head + "  resident A[0, :], B[:, 0];\n"
                                     "  acc C[0, 0] in fabric { for k in 0..0 { call gemm(A[0, k], B[k, 0]) +-> C[0, 0]; } }\n"
                                     "  store C[0, 0];\n  release A[0, :], B[:, 0];\n}\n"),
               ContainsSubstring("csp line 7: acc C[0, 0] receives no call"));
    CHECK_THAT(symbolic_error(head + "  resident A[0, :], B[:, 0];\n"
                                     "  for j in 0..4 {\n"
                                     "    acc C[0, j] in fabric { for k in 0..j { call gemm(A[0, k], B[k, 0]) +-> C[0, j]; } }\n"
                                     "    store C[0, j];\n  }\n  release A[0, :], B[:, 0];\n}\n"),
               ContainsSubstring("csp line 8: acc C[0, j] receives no call"));
    // A loop body checked once must not consume state it found: an accumulator closed before the
    // loop, or a tile written before it (review of 1c.1).
    CHECK_THAT(symbolic_error(head + "  resident A[0, 0], B[0, 0];\n"
                                     "  acc C[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n"
                                     "  for i in 0..4 { store C[0, 0]; }\n  release A[0, 0], B[0, 0];\n}\n"),
               ContainsSubstring("csp line 8: the loop over i stores the accumulator C[0, 0], which it did not open"));
    CHECK_THAT(symbolic_error("csp 1.0\nprogram t machine flat(l3 = 8) {\n  tensor A[128,128] tile 32x32 inout;\n"
                              "  resident A[0, 0];\n  call getrf(A[0, 0]) -> A[0, 0] pivot 0;\n"
                              "  for i in 0..4 { store A[0, 0]; }\n  release A[0, 0];\n}\n"),
               ContainsSubstring("csp line 6: the loop over i stores A[0, 0], written before it"));
}

TEST_CASE("CSP structured: cross-iteration residency runs, and is named as beyond symbolic validation",
          "[program][csp][lang][structured]") {
    // Double buffering: the next k's tile is loaded before this one's is released. The walker
    // runs it (and its values are right); the symbolic validator says why it cannot prove it.
    const std::string src = R"(csp 1.0
program prefetch machine flat(l3 = 8) {
  tensor A[128,128] tile 32x32 in;
  tensor B[128,128] tile 32x32 in;
  tensor C[128,128] tile 32x32 out;
  resident B[:, 0];
  resident A[0, 0];
  acc C[0, 0] in fabric {
    for k in 0..3 {
      resident A[0, k+1];
      call gemm(A[0, k], B[k, 0]) +-> C[0, 0];
      release A[0, k];
    }
    call gemm(A[0, 3], B[3, 0]) +-> C[0, 0];
  }
  release A[0, 3];
  store C[0, 0];
  release B[:, 0];
}
)";
    CspProgram trace = lang::compile(src);
    driver::ProgramSpec s;
    s.algo = "matmul";
    s.size = 128;
    s.tile = 32;
    driver::fill(trace.source, s);
    BehavioralInterpreter lb;
    lb.run(trace);
    TileProgram ref = driver::derive(s);
    driver::fill(ref, s);
    TileProgramReference().run(ref);
    for (Dim c = 0; c < 32; ++c) CHECK(lb.result().operand("C").at(5, c) == ref.operand("C").at(5, c));
    // The first thing it cannot prove: this iteration's A[0, k] was made resident by the last one.
    CHECK_THAT(symbolic_error(src), ContainsSubstring("csp line 11: call gemm reads A[0, k], which is not provably resident"));
}
