// ============================================================================
// include/sw/kpu/program/csp/gen/generate.hpp
// csp-gen: write CSP programs (docs/plans/kpu-run-csp-programs.md step 1).
//
// The simulator (kpu-run) takes a CSP program and executes it; producing programs is this
// library's job, and the kpu-csp-gen tool's. Each (operator, dataflow) has ONE canonical
// schedule, written as a structured program with its loops -- never unrolled and re-rolled --
// and every program is validated symbolically (lang::validate) before it is returned, so the
// generator cannot emit a program the language refuses.
//
//   matmul  dataflow os (the default) -- output-stationary panels: a column panel of B and a row panel of A resident at a
//           time, an accumulator per C tile. orient = "col" (B's panel outer, j) or "row" (A's
//           panel outer, i). L3: 2 x tiles-per-panel + 1 (the store's writeback). Each store is
//           `late`: it leaves after the next accumulator's calls, so the movers bring the next
//           chain's tiles in before this chain's results go out, and the chains pipeline.
//           stage_a (the row schedule): A's row panel is staged in L2 for the j loop, so each
//           A tile leaves L3 once per row instead of once per C tile; the machine clause
//           declares `l2 = tiles-per-panel` (docs/plans/l2-reuse-and-fabric-pipeline-view.md
//           step 3).
//           dataflow ws -- weight-stationary (#430): B's tile is the stationary operand. For each
//           column panel j, C[:, j]'s partials are an L2 accumulator (`acc C[:, j] in l2`); for each
//           k, A's column panel and B[k, j] are made resident, B[k, j] is staged in L2 and held
//           there across the i loop (the reuse the program states), and every call's result is
//           merged by the Combiner into C[i, j]'s slot. So each B tile leaves L3 once, not once per
//           call. L3: tiles-per-panel + 1 (A[:, k] and B[k, j]); L2: tiles-per-panel + 1 (the
//           panel's partials and the stationary B tile).
//           dataflow is -- input-stationary (#430): the same with A's tile stationary. For each row
//           panel i, C[i, :]'s partials are the L2 accumulator; for each k, A[i, k] and B's row
//           panel are resident, A[i, k] is staged and held across the j loop. L3 and L2 as ws.
//           Neither takes orient (the dataflow fixes the loop order), stage_a or an init.
//   linear  (any dataflow; ws and is: the epilogue @ combiner on each tile's final merge, @ bm.egress
//           on ws's panel store, or unfused -- never @ fabric, which would act on one partial)
//           matmul plus the epilogue act(C + b): the bias vector b[j] is resident with B's
//           panel, and `place` writes the context -- `via add(b[j]) @ P, act @ P` for
//           P = fabric | combiner | bm.egress -- or "unfused": a second pass that reads C
//           back and applies add and act as calls. L3: the matmul's + 1 (fused), and 2 for the
//           unfused pass.
//   lu      right-looking tile LU, A resident whole: getrf, laswp across the row block, the
//           two trsm panels, the trailing gemm update (alpha -1), per diagonal step. L3: tiles^2.
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
    std::string place = "fabric";           // linear: fabric | combiner | bm.egress | unfused
    bool stage_a = false;                   // matmul, orient row: stage A's row panel in L2
    // matmul: where each accumulator starts (docs/plans/accumulator-init-stream.md). none: zero;
    // bias: C = A . B + D, D tiled as C; bias-row: C = A . B + b, b broadcast down C's rows;
    // accumulate: C += A . B, C inout.
    std::string init = "none";
    // matmul: the inputs' number system (numeric.hpp; docs/plans/typed-value-path.md). A and B are
    // in it; C, and an init's D or b, are in its registered accumulator's (int8 -> int32).
    std::string dtype = "fp32";
    // matmul: the dataflow -- which operand the schedule holds stationary (#430). os: C, in the
    // fabric; ws: B, staged in L2 across the i loop; is: A, staged across the j loop. ws and is
    // reduce across K blocks through the Combiner's merge into C's L2 slots.
    std::string dataflow = "os";
    // linear: the block format the result is converted into, after the activation, at the
    // epilogue's place (#427): none, or an OCP Microscaling format (mxint8, mxfp8).
    std::string format = "none";
};

// Tiles along one dimension.
inline Dim tiles(const Options& o) { return o.tile ? (o.size + o.tile - 1) / o.tile : 0; }

// The L3 slots the operator's schedule needs at its peak.
// The accumulator a gemm on `in` is registered with (numeric::mac_pairs); empty: none.
inline std::string accumulator_of(const std::string& in) {
    for (const auto& [i, a] : numeric::mac_pairs())
        if (i == in) return a;
    return {};
}

inline std::size_t slots_needed(const Options& o) {
    const std::size_t nt = tiles(o);
    if (o.algo == "lu") return nt * nt;
    if (o.dataflow != "os") {               // ws, is: one operand's panel and the stationary tile,
        return nt + (o.algo == "linear" && o.place != "unfused" ? 2 : 1);   // and b's tile
    }
    const std::size_t mm = 2 * nt + 1;
    if (o.algo == "linear" && o.place != "unfused") return mm + 1;
    if (o.algo == "matmul" && (o.init == "bias" || o.init == "bias-row")) return mm + 1;   // D's tile, or b's
    return mm;                              // accumulate: C's tile is handed to the fabric before its calls                              // the unfused pass needs 2, which mm always covers
}

namespace detail {

inline void check(const Options& o) {
    if (o.algo != "matmul" && o.algo != "linear" && o.algo != "lu")
        throw GenError("csp-gen: unknown algo '" + o.algo + "' (matmul | linear | lu)");
    if (o.size == 0 || o.tile == 0) throw GenError("csp-gen: size and tile must be positive");
    if (o.l3 == 0) throw GenError("csp-gen: the machine's L3 capacity (tiles) is required");
    if (o.algo != "lu" && o.orient != "col" && o.orient != "row")
        throw GenError("csp-gen: orient '" + o.orient + "' (col | row)");
    if (o.algo == "linear" && o.place != "fabric" && o.place != "combiner" && o.place != "bm.egress" &&
        o.place != "unfused")
        throw GenError("csp-gen: place '" + o.place + "' (fabric | combiner | bm.egress | unfused)");
    if (o.stage_a && o.algo != "matmul") throw GenError("csp-gen: stage_a is matmul's; --algo is '" + o.algo + "'");
    if (o.init != "none" && o.init != "bias" && o.init != "bias-row" && o.init != "accumulate")
        throw GenError("csp-gen: init '" + o.init + "' (none | bias | bias-row | accumulate)");
    if (o.init != "none" && o.algo != "matmul") throw GenError("csp-gen: init is matmul's; --algo is '" + o.algo + "'");
    if (o.dtype != "fp32" && o.algo != "matmul") throw GenError("csp-gen: dtype is matmul's; --algo is '" + o.algo + "'");
    if (o.dtype != "fp32" && accumulator_of(o.dtype).empty())
        throw GenError("csp-gen: dtype '" + o.dtype + "' has no registered gemm (" + numeric::mac_names() + ")");
    if (o.stage_a && o.orient != "row")
        throw GenError("csp-gen: stage_a stages A's row panel for the loop over B's panels, which needs orient row");
    if (o.format != "none") {
        if (o.algo != "linear") throw GenError("csp-gen: format is linear's; --algo is '" + o.algo + "'");
        if (!numeric::is_block_format(o.format))
            throw GenError("csp-gen: format '" + o.format + "' (none | " + numeric::block_format_names() + ")");
        if (o.place != "combiner" && o.place != "bm.egress")
            throw GenError("csp-gen: format " + o.format + " @ " + o.place + ": a block-format conversion rides the "
                           "result's move into memory, on the Combiner or at bm.egress");
    }
    if (o.dataflow != "os" && o.dataflow != "ws" && o.dataflow != "is")
        throw GenError("csp-gen: dataflow '" + o.dataflow + "' (os | ws | is)");
    if (o.dataflow != "os") {
        const std::string df = "csp-gen: dataflow " + o.dataflow;
        if (o.algo != "matmul" && o.algo != "linear")
            throw GenError(df + " is matmul's and linear's; --algo is '" + o.algo + "'");
        if (o.algo == "linear" && o.place == "fabric")
            throw GenError(df + ": a fabric epilogue would act on one partial; the reduction's sum is the "
                                "Combiner's, so place it @ combiner (or bm.egress, unfused)");
        if (o.algo == "linear" && o.place == "bm.egress" && o.dataflow == "is")
            throw GenError(df + ": an epilogue @ bm.egress rides the row panel's store, where b[j] varies by tile; "
                                "place it @ combiner (on each tile's final merge), or unfused");
        if (o.orient != "col")
            throw GenError(df + " fixes its own loop order (" +
                           (o.dataflow == "ws" ? std::string("B's column panels outer") : std::string("A's row panels outer")) +
                           "); orient is the output-stationary schedule's");
        if (o.stage_a) throw GenError(df + " stages its stationary operand itself; stage_a is the os schedule's");
        if (o.init != "none")
            throw GenError(df + ": its first merge overwrites C's L2 slot, so init '" + o.init +
                           "' is not built yet (the init through the Combiner is #427's)");
    }
    const std::size_t need = slots_needed(o);
    if (need > o.l3) {
        const std::size_t nt = tiles(o);
        const std::string why =
            o.dataflow != "os" ? "a panel of " + std::to_string(nt) + " tiles and the stationary tile" +
                                     std::string(o.algo == "linear" && o.place != "unfused" ? ", and the bias tile" : "")
            : o.algo == "lu" ? "the whole matrix, " + std::to_string(nt) + " x " + std::to_string(nt) + " tiles"
                           : "2 panels of " + std::to_string(nt) + " tiles and the store's writeback" +
                                 (o.algo == "linear" && o.place != "unfused" ? ", and the bias tile" : "") +
                                 (o.init == "bias" || o.init == "bias-row" ? ", and the init's tile" : "");
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

inline std::string tensor(const char* name, const Options& o, const char* io, const std::string& type = "fp32") {
    return "  tensor " + std::string(name) + "[" + std::to_string(o.size) + "," + std::to_string(o.size) + "]" +
           (type == "fp32" ? std::string() : " " + type) + " tile " + std::to_string(o.tile) + "x" +
           std::to_string(o.tile) + " " + io + ";\n";
}

// A linear's block-format conversion (#427), the epilogue's last stage, at the epilogue's place.
inline std::string format_stage(const Options& o) {
    return o.format == "none" ? std::string() : ", " + o.format + " @ " + o.place;
}

// The matmul nest, and for a fused linear its bias (resident with B's panel) and context.
inline std::string matmul_nest(const Options& o, bool linear_fused) {
    const std::string nt = std::to_string(tiles(o));
    const std::string bias = linear_fused || o.init == "bias-row" ? ", b[j]" : "";
    // A fused store carries its epilogue, which reads b[j] where it is written; a late store
    // carries none (lang/validate.hpp), so only the plain matmul nest defers its stores.
    std::string via = " late";
    if (linear_fused)
        via = std::string(" via add(b[j]) @ ") + o.place + ", " + to_string(o.act) + " @ " + o.place + format_stage(o);
    // The accumulator's init: a tile made resident for it (D's, or C's own), or b's with B's panel.
    std::string before, init, after;
    if (o.init == "bias") {
        before = "      resident D[i, j];\n";
        init = " init D[i, j]";
        after = "      release D[i, j];\n";
    } else if (o.init == "bias-row") {
        init = " init b[j]";
    } else if (o.init == "accumulate") {
        before = "      resident C[i, j];\n";   // the fabric takes the residency over: no release
        init = " init C[i, j]";
    }
    const std::string acc = before +
                            "      acc C[i, j] in fabric" + init + " {\n"
                            "        for k in 0.." + nt + " { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
                            "      }\n" + after +
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
        // Staged, the j loop runs inside A's stage block, one level deeper.
        const std::string in = o.stage_a ? "  " : "";
        std::string body = acc;
        if (o.stage_a) {
            std::string deeper;
            for (std::size_t at = 0; at < body.size();) {
                const std::size_t nl = body.find('\n', at);
                deeper += "  " + body.substr(at, nl - at + 1);
                at = nl + 1;
            }
            body = deeper;
        }
        s += "  for i in 0.." + nt + " {\n";
        s += "    resident A[i, :];\n";
        if (o.stage_a) s += "    stage A[i, :] in l2 {\n";
        s += in + "    for j in 0.." + nt + " {\n";
        s += in + "      resident B[:, j]" + bias + ";\n";
        s += body;
        s += in + "      release B[:, j]" + bias + ";\n";
        s += in + "    }\n";
        if (o.stage_a) s += "    }\n";
        s += "    release A[i, :];\n";
        s += "  }\n";
    }
    return s;
}

// The weight- and input-stationary nests (#430): the stationary operand's tile staged in L2 and
// held across the loop that reuses it, the result panel's partials an L2 accumulator that the
// Combiner merges every call into.
//
// A fused linear epilogue act(C + b) (#427): @ combiner, the k loop's last block is written out on
// its own, and its calls -- each tile's final merge -- carry the epilogue, which the Combiner runs
// on the merged value. b[j] is resident around them: for ws, with the panel's last block (one b
// tile per panel); for is, around each call (b varies along the j loop). @ bm.egress (ws only), the
// panel's store carries it.
inline std::string stationary_nest(const Options& o, bool linear_fused) {
    const std::size_t n = tiles(o);
    const std::string nt = std::to_string(n);
    const bool ws = o.dataflow == "ws";
    const bool at_combiner = linear_fused && o.place == "combiner";
    const bool at_egress = linear_fused && o.place == "bm.egress";
    const std::string epilogue =
        std::string(" via add(b[j]) @ ") + o.place + ", " + to_string(o.act) + " @ " + o.place + format_stage(o);
    // ws: j outer, B[k, j] stationary across i. is: i outer, A[i, k] stationary across j.
    const std::string outer = ws ? "j" : "i", inner = ws ? "i" : "j";
    const std::string panel = ws ? "C[:, j]" : "C[i, :]";
    auto block = [&](const std::string& k, bool last) {
        const std::string moving = ws ? "A[:, " + k + "]" : "B[" + k + ", :]";   // the streamed operand's panel
        const std::string held = ws ? "B[" + k + ", j]" : "A[i, " + k + "]";     // the stationary tile
        const std::string in = last ? "      " : "        ";
        std::string call = "call gemm(A[i, " + k + "], B[" + k + ", j]) +-> C[i, j]" + (last ? epilogue : std::string()) + ";";
        const std::string bias = last && ws ? std::string(", b[j]") : std::string();
        if (last && !ws) call = "resident b[j]; " + call + " release b[j];";
        std::string t;
        t += in + "resident " + moving + ", " + held + bias + ";\n";
        t += in + "stage " + held + " in l2 {\n";
        t += in + "  for " + inner + " in 0.." + nt + " { " + call + " }\n";
        t += in + "}\n";
        t += in + "release " + moving + ", " + held + bias + ";\n";
        return t;
    };
    std::string s;
    s += "  for " + outer + " in 0.." + nt + " {\n";
    if (at_egress) s += "    resident b[j];\n";
    s += "    acc " + panel + " in l2 {\n";
    s += "      for k in 0.." + std::to_string(at_combiner ? n - 1 : n) + " {\n";
    s += block("k", false);
    s += "      }\n";
    if (at_combiner) {
        s += "      // the last K block: each tile's final merge, with the epilogue on the Combiner\n";
        s += block(std::to_string(n - 1), true);
    }
    s += "    }\n";
    s += "    store " + panel + (at_egress ? epilogue : std::string()) + ";\n";
    if (at_egress) s += "    release b[j];\n";
    s += "  }\n";
    return s;
}

inline std::string matmul(const Options& o) {
    if (o.dataflow != "os") {
        const bool ws = o.dataflow == "ws";
        std::string s = header(o, std::string("matmul C = A . B, ") +
                                      (ws ? "weight-stationary: B's tile staged in L2 across the i loop"
                                          : "input-stationary: A's tile staged in L2 across the j loop") +
                                      ", C's partials merged by the Combiner in L2");
        s += "program matmul machine flat(l3 = " + std::to_string(o.l3) + ", l2 = " + std::to_string(tiles(o) + 1) +
             ") {\n";
        const std::string acc = o.dtype == "fp32" ? std::string("fp32") : accumulator_of(o.dtype);
        s += tensor("A", o, "in", o.dtype) + tensor("B", o, "in", o.dtype) + tensor("C", o, "out", acc);
        s += stationary_nest(o, false);
        return s + "}\n";
    }
    const std::string what = o.init == "bias"       ? "matmul C = A . B + D"
                             : o.init == "bias-row" ? "matmul C = A . B + b"
                             : o.init == "accumulate" ? "matmul C += A . B"
                                                      : "matmul C = A . B";
    std::string s = header(o, what + ", " + o.orient + "-panel schedule" +
                                  (o.stage_a ? std::string(", A's row panel staged in L2") : std::string()) +
                                  (o.init != "none" ? ", the accumulator's init streamed in" : std::string()));
    s += "program matmul machine flat(l3 = " + std::to_string(o.l3) +
         (o.stage_a ? ", l2 = " + std::to_string(tiles(o)) : std::string()) + ") {\n";
    const std::string acc = o.dtype == "fp32" ? std::string("fp32") : accumulator_of(o.dtype);
    s += tensor("A", o, "in", o.dtype) + tensor("B", o, "in", o.dtype);
    if (o.init == "bias") s += tensor("D", o, "in", acc);
    if (o.init == "bias-row")
        s += "  vector b[" + std::to_string(o.size) + "]" + (acc == "fp32" ? std::string() : " " + acc) + " tile " +
             std::to_string(o.tile) + " in;\n";
    s += tensor("C", o, o.init == "accumulate" ? "inout" : "out", acc);
    s += matmul_nest(o, false);
    return s + "}\n";
}

inline std::string linear(const Options& o) {
    const bool fused = o.place != "unfused";
    const std::string schedule = o.dataflow == "ws"   ? std::string("weight-stationary schedule")
                                 : o.dataflow == "is" ? std::string("input-stationary schedule")
                                                      : o.orient + "-panel schedule";
    std::string s = header(o, std::string("linear C = ") + to_string(o.act) + "(A . B + b), " + schedule +
                                  ", epilogue " + (fused ? "@ " + o.place : std::string("unfused")) +
                                  (o.format != "none" ? ", into " + o.format : std::string()) +
                                  (o.dataflow != "os" && o.place == "combiner" ? " on each tile's final merge"
                                                                               : std::string()));
    s += "program linear machine flat(l3 = " + std::to_string(o.l3) +
         (o.dataflow != "os" ? ", l2 = " + std::to_string(tiles(o) + 1) : std::string()) + ") {\n";
    s += tensor("A", o, "in") + tensor("B", o, "in");
    s += "  vector b[" + std::to_string(o.size) + "] tile " + std::to_string(o.tile) + " in;\n";
    s += tensor("C", o, fused ? "out" : "inout");
    s += o.dataflow != "os" ? stationary_nest(o, fused) : matmul_nest(o, fused);
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
