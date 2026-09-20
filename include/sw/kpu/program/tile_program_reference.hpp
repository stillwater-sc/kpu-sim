// ============================================================================
// include/sw/kpu/program/tile_program_reference.hpp
// L0 functional reference — executes a TileProgram to the correct numeric
// result, with NO streams and NO timing.
//
// This is the whole point of the L0 layer (D6): processing the tile sequence
// with the operator's tile-level compute yields the correct answer, so L0 alone
// is a device-independent functional oracle. The arithmetic mirrors the CSP
// executor's ground-truth semantics (row-major, out += a[i,k]*b[k,j]) so that
// the L1/timing lowering stays value-consistent with L0.
//
// The arithmetic itself lives in tile_kernels.hpp, shared with the transactional
// tier: this class is now just the IN-ORDER driver over those kernels plus the
// run summary. A dataflow-order driver over the same kernels is bit-identical by
// construction (ADR 0001 D3.1) — which is why the kernels are not duplicated
// here.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_kernels.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstddef>
#include <vector>

namespace sw::kpu::program {

// ============================================================================
// TileProgramReference — walks the ops in order, mutating operand buffers.
// ============================================================================
class TileProgramReference {
public:
    struct RunSummary {
        std::size_t ops = 0;
        std::size_t feeds = 0;
        std::size_t drains = 0;
        std::size_t computes = 0;   // MatMulAccum (GEMM)
        std::size_t diag_factors = 0;   // LuDiagFactor (GETRF)
        std::size_t trsms = 0;          // TrsmLowerLeft + TrsmUpperRight
        std::size_t pivot_applies = 0;  // PivotApply (LASWP)
        std::size_t row_swaps = 0;      // total within-tile row swaps performed by GETRF
        // Row permutation produced by pivoting (LU): perm[i] = original row now at
        // position i. Identity when no pivoting occurred. Sized to the largest
        // square operand touched by a LuPanelFactor (0 if none).
        std::vector<Dim> permutation;
    };

    // Execute program in order. Mutates program's operand value buffers.
    RunSummary run(TileProgram& program) {
        state_.clear();   // reset up front so a throw mid-run can't leak stale state
        RunSummary sum;
        for (const auto& op : program.ops()) {
            ++sum.ops;
            switch (op.kind) {
                case TileOpKind::Feed:           ++sum.feeds; break;   // structural (L1 attaches streams)
                case TileOpKind::Drain:          ++sum.drains; break;  // structural
                case TileOpKind::MatMulAccum:    ++sum.computes; break;
                case TileOpKind::LuDiagFactor:   ++sum.diag_factors; break;
                case TileOpKind::PivotApply:     ++sum.pivot_applies; break;
                case TileOpKind::TrsmLowerLeft:
                case TileOpKind::TrsmUpperRight: ++sum.trsms; break;
            }
            apply(program, op, state_);   // the shared L0 kernels
        }
        sum.row_swaps = state_.swaps_performed;
        sum.permutation = state_.perm;
        return sum;
    }

private:
    // Transient pivot state + row permutation, carried between ops by the kernels.
    TileKernelState state_;
};

} // namespace sw::kpu::program
