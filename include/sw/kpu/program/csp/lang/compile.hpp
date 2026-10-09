// ============================================================================
// include/sw/kpu/program/csp/lang/compile.hpp
// The CSP language -> the CSP program's TRACE form (docs/plans/csp-language.md §3.3).
//
// compile() runs the program's structure to completion (walk.hpp) and collects every action:
// the flat CspProgram the step-1 interpreters and the printer use. That is the trace form --
// right for small programs, the corpus, debugging and the record, and impossible for a large
// one (ADR 0004 §4). A large program is validated symbolically (validate.hpp) and executed from
// its ActionStream (walk.hpp) instead.
//
//   resident X      Load X (dma)                       X not already resident; L3 not full
//   release X       Release X (its credit returns)     X resident; not written and unstored
//   call f(..) -> y each operand: Move (bm), Feed (str); then Call (cf); an in-place result:
//                   Drain (str), Writeback (bm)        every operand resident in L3
//   acc y in fabric { ... }   y accumulates in the fabric (from zero), holds no L3 slot, and
//                   receives at least one call
//   store y         from an accumulator: Drain, Writeback, Store, Release (a slot for the
//                   writeback's moment); a resident, written tile: Store
//
// Functions: gemm(a, b) +-> y [alpha s]; getrf(x) -> x pivot p; laswp(x) -> x pivot p;
// trsm_ll(d, x) -> x; trsm_ur(d, x) -> x.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>

#include <string>

namespace sw::kpu::program::csp::lang {

// Compile a parsed program to its trace. Throws CompileError, naming the line.
inline CspProgram compile(const Program& ast) {
    CspProgram p;
    p.name = ast.name;
    p.processes = {{ProcessKind::Dma, "dma", {}}, {ProcessKind::BlockMover, "bm", {}},
                   {ProcessKind::Streamer, "str", {}}, {ProcessKind::Compute, "cf", {}}};
    TraceSink sink(p);
    Walker<TraceSink> walker(ast, sink);
    p.source = TileProgram(ast.name);   // the trace carries values: allocate them here
    for (const auto& name : walker.operands().operand_order()) {
        const TensorOperand& d = walker.operands().operand(name);
        p.source.add_operand(TensorOperand(d.name, d.rows, d.cols, d.tile_rows, d.tile_cols));
    }
    p.channels = {{Chan::Dram, "dram", 0}, {Chan::L3, "l3", walker.capacity()}, {Chan::L2, "l2", 0}, {Chan::Cf, "cf", 0}};
    walker.run();
    return p;
}

// Parse and compile .csp source.
inline CspProgram compile(const std::string& source) { return compile(parse(source)); }

}  // namespace sw::kpu::program::csp::lang
