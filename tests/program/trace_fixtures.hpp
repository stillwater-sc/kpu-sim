// ============================================================================
// tests/program/trace_fixtures.hpp
// L0 fixtures, taken the one way an L0 is made: L-B runs a CSP program and records its trace
// (docs/plans/retire-l0-derivations.md §3.1-3.2; ADR 0004 §4).
//
//   WRONG    TileProgram p = derive(ps); fill(p, ps);      an L0 written by C++
//   CORRECT  TileProgram p = trace_of(gen("matmul", 48, 16));
//
// A trace carries its inputs (csp_inputs: exact multiples of 1/8; an operand factored in place
// strictly dominant), and the oracle for its values is L-B on the same program (lb_values).
// Shapes kpu-csp-gen does not write -- rectangular tiles -- are hand-written programs in
// tests/program/csp/, read relative to the source tree (the tests run from CMAKE_SOURCE_DIR).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>

#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trace_fixtures {

using sw::kpu::program::Dim;
using sw::kpu::program::TileProgram;
namespace csp = sw::kpu::program::csp;
namespace driver = sw::kpu::program::driver;

// kpu-csp-gen's options for a square operator, with the L3 its schedule needs.
inline csp::gen::Options gen(const std::string& algo, Dim size, Dim tile) {
    csp::gen::Options o;
    o.algo = algo;
    o.size = size;
    o.tile = tile;
    o.l3 = csp::gen::slots_needed(o);
    return o;
}

inline csp::lang::Program program_of(const csp::gen::Options& o) {
    return csp::lang::parse(csp::gen::generate(o));
}

// A .csp file, relative to the source tree.
inline csp::lang::Program program_of_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("trace_fixtures: cannot read " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return csp::lang::parse(ss.str());
}

// L-B's run of the program on its synthesized inputs: the trace and the oracle's values.
inline driver::CspReference reference_of(const csp::lang::Program& ast) {
    driver::CspReference r = driver::csp_reference(ast, driver::csp_inputs(ast), std::numeric_limits<std::size_t>::max());
    if (!r.trace) throw std::runtime_error("trace_fixtures: " + ast.name + ": " + r.note);
    return r;
}

// The program's L0, on its inputs, before it runs: what `kpu-run --emit-l0` writes.
inline TileProgram trace_of(const csp::lang::Program& ast) { return *reference_of(ast).trace; }
inline TileProgram trace_of(const csp::gen::Options& o) { return trace_of(program_of(o)); }
inline TileProgram trace_of_file(const std::string& path) { return trace_of(program_of_file(path)); }

// Every operand as L-B leaves it: the oracle.
inline TileProgram lb_values(const csp::lang::Program& ast) { return *reference_of(ast).values; }
inline TileProgram lb_values(const csp::gen::Options& o) { return lb_values(program_of(o)); }

}  // namespace trace_fixtures
