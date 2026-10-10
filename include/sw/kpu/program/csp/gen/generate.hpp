// ============================================================================
// include/sw/kpu/program/csp/gen/generate.hpp
// csp-gen: write CSP programs (docs/plans/kpu-run-csp-programs.md step 1).
//
// The simulator (kpu-run) takes a CSP program and executes it; producing programs is this
// library's job, and the kpu-csp-gen tool's. Each operator has ONE canonical schedule, written
// as a structured program with its loops -- never unrolled and re-rolled -- and every program
// is validated symbolically (lang::validate) before it is returned, so the generator cannot
// emit a program the language refuses.
//
//   matmul  output-stationary panels: a column panel of B and a row panel of A resident at a
//           time, an accumulator per C tile. orient = "col" (B's panel outer, j) or "row" (A's
//           panel outer, i). L3: 2 x tiles-per-panel + 1 (the store's writeback).
//   linear  matmul plus the epilogue act(C + b): the bias vector b[j] is resident with B's
//           panel, and `place` writes the context -- `via add(b[j]) @ P, act @ P` for
//           P = fabric | str.drain | bm.egress -- or "unfused": a second pass that reads C
//           back and applies add and act as calls. L3: the matmul's + 1 (fused), and 2 for the
//           unfused pass.
//   lu      right-looking tile LU, A resident whole: getrf, laswp across the row block, the
//           two trsm panels, the trailing gemm update (alpha -1) -- the L0 derivation's order,
//           so its trace computes bit-identically to derive_lu_tile_program. L3: tiles^2.
//
// Schedule options (orient, place) are the program's choices, never machine parameters; the
// machine's L3 is an input. With a target, placement is checked against its vector units.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sw::kpu::program::csp::gen {

class GenError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string algo = "matmul";            // matmul | linear | lu
    Dim size = 64;                          // square: M = N = K = size (N for LU)
    Dim tile = 16;
    std::size_t l3 = 0;                     // the machine's L3, in tiles (required)
    std::string orient = "col";             // matmul, linear: col | row
    ActivationFn act = ActivationFn::Relu;  // linear
    std::string place = "fabric";           // linear: fabric | str.drain | bm.egress | unfused
};

// Tiles along one dimension.
inline Dim tiles(const Options& o) { return o.tile ? (o.size + o.tile - 1) / o.tile : 0; }

// The L3 slots the operator's schedule needs at its peak.
inline std::size_t slots_needed(const Options& o) {
    const std::size_t nt = tiles(o);
    if (o.algo == "lu") return nt * nt;
    const std::size_t mm = 2 * nt + 1;
    if (o.algo == "linear" && o.place != "unfused") return mm + 1;
    return mm;                              // the unfused pass needs 2, which mm always covers
}

namespace detail {

inline void check(const Options& o) {
    if (o.algo != "matmul" && o.algo != "linear" && o.algo != "lu")
        throw GenError("csp-gen: unknown algo '" + o.algo + "' (matmul | linear | lu)");
    if (o.size == 0 || o.tile == 0) throw GenError("csp-gen: size and tile must be positive");
    if (o.l3 == 0) throw GenError("csp-gen: the machine's L3 capacity (tiles) is required");
    if (o.algo != "lu" && o.orient != "col" && o.orient != "row")
        throw GenError("csp-gen: orient '" + o.orient + "' (col | row)");
    if (o.algo == "linear" && o.place != "fabric" && o.place != "str.drain" && o.place != "bm.egress" &&
        o.place != "unfused")
        throw GenError("csp-gen: place '" + o.place + "' (fabric | str.drain | bm.egress | unfused)");
    const std::size_t need = slots_needed(o);
    if (need > o.l3) {
        const std::size_t nt = tiles(o);
        const std::string why =
            o.algo == "lu" ? "the whole matrix, " + std::to_string(nt) + " x " + std::to_string(nt) + " tiles"
                           : "2 panels of " + std::to_string(nt) + " tiles and the store's writeback" +
                                 (o.algo == "linear" && o.place != "unfused" ? ", and the bias tile" : "");
        throw GenError("csp-gen: the " + o.algo + " schedule at " + std::to_string(o.size) + " in " +
                       std::to_string(o.tile) + "-tiles needs " + std::to_string(need) + " L3 slots (" + why +
                       "); the L3 holds " + std::to_string(o.l3) + " -- use a larger tile or a larger L3");
    }
}

inline std::string header(const Options& o, const std::string& what) {
    std::ostringstream h;
    h << "csp " << lang::kVersion << "\n";
    h << "// generated by kpu-csp-gen: " << what << ", " << o.size << " in " << o.tile << "x" << o.tile
      << " tiles, L3 " << o.l3 << " (" << slots_needed(o) << " used at peak)\n";
    return h.str();
}

inline std::string tensor(const char* name, const Options& o, const char* io) {
    return "  tensor " + std::string(name) + "[" + std::to_string(o.size) + "," + std::to_string(o.size) +
           "] tile " + std::to_string(o.tile) + "x" + std::to_string(o.tile) + " " + io + ";\n";
}

// The matmul nest, and for a fused linear its bias (resident with B's panel) and context.
inline std::string matmul_nest(const Options& o, bool linear_fused) {
    const std::string nt = std::to_string(tiles(o));
    const std::string bias = linear_fused ? ", b[j]" : "";
    std::string via;
    if (linear_fused)
        via = std::string(" via add(b[j]) @ ") + o.place + ", " + to_string(o.act) + " @ " + o.place;
    const std::string acc = "      acc C[i, j] in fabric {\n"
                            "        for k in 0.." + nt + " { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
                            "      }\n"
                            "      store C[i, j]" + via + ";\n";
    std::string s;
    if (o.orient == "col") {
        s += "  for j in 0.." + nt + " {\n";
        s += "    resident B[:, j]" + bias + ";\n";
        s += "    for i in 0.." + nt + " {\n";
        s += "      resident A[i, :];\n";
        s += acc;
        s += "      release A[i, :];\n";
        s += "    }\n";
        s += "    release B[:, j]" + bias + ";\n";
        s += "  }\n";
    } else {
        s += "  for i in 0.." + nt + " {\n";
        s += "    resident A[i, :];\n";
        s += "    for j in 0.." + nt + " {\n";
        s += "      resident B[:, j]" + bias + ";\n";
        s += acc;
        s += "      release B[:, j]" + bias + ";\n";
        s += "    }\n";
        s += "    release A[i, :];\n";
        s += "  }\n";
    }
    return s;
}

inline std::string matmul(const Options& o) {
    std::string s = header(o, "matmul C = A . B, " + o.orient + "-panel schedule");
    s += "program matmul machine flat(l3 = " + std::to_string(o.l3) + ") {\n";
    s += tensor("A", o, "in") + tensor("B", o, "in") + tensor("C", o, "out");
    s += matmul_nest(o, false);
    return s + "}\n";
}

inline std::string linear(const Options& o) {
    const bool fused = o.place != "unfused";
    std::string s = header(o, std::string("linear C = ") + to_string(o.act) + "(A . B + b), " + o.orient +
                                  "-panel schedule, epilogue " + (fused ? "@ " + o.place : std::string("unfused")));
    s += "program linear machine flat(l3 = " + std::to_string(o.l3) + ") {\n";
    s += tensor("A", o, "in") + tensor("B", o, "in");
    s += "  vector b[" + std::to_string(o.size) + "] tile " + std::to_string(o.tile) + " in;\n";
    s += tensor("C", o, fused ? "out" : "inout");
    s += matmul_nest(o, fused);
    if (!fused) {
        // The epilogue as calls, on C read back from DRAM: a second round trip.
        const std::string nt = std::to_string(tiles(o));
        s += "  for j in 0.." + nt + " {\n";
        s += "    resident b[j];\n";
        s += "    for i in 0.." + nt + " {\n";
        s += "      resident C[i, j];\n";
        s += "      call add(C[i, j], b[j]) -> C[i, j];\n";
        s += "      call " + std::string(to_string(o.act)) + "(C[i, j]) -> C[i, j];\n";
        s += "      store C[i, j];\n";
        s += "      release C[i, j];\n";
        s += "    }\n";
        s += "    release b[j];\n";
        s += "  }\n";
    }
    return s + "}\n";
}

inline std::string lu(const Options& o) {
    const std::string nt = std::to_string(tiles(o));
    std::string s = header(o, "LU (right-looking, in place), A resident whole");
    s += "program lu machine flat(l3 = " + std::to_string(o.l3) + ") {\n";
    s += tensor("A", o, "inout");
    s += "  resident A[:, :];\n";
    s += "  for k in 0.." + nt + " {\n";
    s += "    call getrf(A[k, k]) -> A[k, k] pivot k;\n";
    s += "    for g in 0..k { call laswp(A[k, g]) -> A[k, g] pivot k; }\n";
    s += "    for j in k + 1.." + nt + " { call laswp(A[k, j]) -> A[k, j] pivot k; }\n";
    s += "    for j in k + 1.." + nt + " { call trsm_ll(A[k, k], A[k, j]) -> A[k, j]; }\n";
    s += "    for i in k + 1.." + nt + " { call trsm_ur(A[k, k], A[i, k]) -> A[i, k]; }\n";
    s += "    for i in k + 1.." + nt + " {\n";
    s += "      for j in k + 1.." + nt + " { call gemm(A[i, k], A[k, j]) +-> A[i, j] alpha -1; }\n";
    s += "    }\n";
    s += "  }\n";
    s += "  store A[:, :];\n";
    s += "  release A[:, :];\n";
    return s + "}\n";
}

}  // namespace detail

// The operator's program, as .csp text, validated (against `target`'s sites when given).
// Throws GenError for a bad option or a schedule that does not fit the L3, and the
// validator's CompileError -- by line -- for anything else it refuses.
inline std::string generate(const Options& o, const lang::Target* target = nullptr) {
    detail::check(o);
    const std::string text = o.algo == "lu" ? detail::lu(o) : o.algo == "linear" ? detail::linear(o) : detail::matmul(o);
    (void)lang::validate(text, target);
    return text;
}

}  // namespace sw::kpu::program::csp::gen
