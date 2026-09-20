// ============================================================================
// include/sw/kpu/program/tile_kernels.hpp
// The L0 tile kernels — the tile-level compute of each TileOpKind, and the
// transient state they carry between ops (pivot slots + row permutation).
//
// These are the SINGLE implementation of L0 arithmetic. Every driver executes a
// TileProgram by calling apply() per op, and differs only in the ORDER it
// chooses and what it measures:
//   - TileProgramReference (tile_program_reference.hpp) walks ops in program
//     order with no timing — the BEHAVIORAL tier and the value oracle.
//   - TileTransactionExecutor (docs/plans/tile-transaction-executor.md, #264)
//     fires ops in dataflow order under credits and resource contention — the
//     TRANSACTIONAL tier.
// Because both call these kernels, their values are bit-identical BY
// CONSTRUCTION rather than by testing: dependency-respecting orders cannot
// differ, since each op declares its full tile I/O (§3a of
// docs/plans/kpu-program-model.md) and WAW ordering on an output tile preserves
// accumulation order. See ADR 0001 D3.1.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sw::kpu::program {

// ============================================================================
// TileKernelState — what a kernel leaves behind for a LATER op to read.
//
// This is the data-dependent control matmul does not have: a pivot decision
// taken in one tile op flows to row swaps in trailing tiles. A driver must keep
// one instance per run and pass it to every apply() call, in an order that
// respects the program's dependencies (including the pivot-slot edges).
// ============================================================================
struct TileKernelState {
    // slot -> ordered list of (row_a, row_b) absolute row swaps recorded by
    // LuDiagFactor and replayed by PivotApply.
    std::map<int, std::vector<std::pair<Dim, Dim>>> pivots;

    // Current row permutation (LU): perm[i] = original row now at position i.
    std::vector<Dim> perm;

    // Total within-tile row swaps performed (statistic, not state).
    std::size_t swaps_performed = 0;

    void clear() {
        pivots.clear();
        perm.clear();
        swaps_performed = 0;
    }

    void ensure_perm(Dim n) {
        if (perm.size() < n) {
            perm.resize(n);
            for (Dim i = 0; i < n; ++i) perm[i] = i;
        }
    }
};

// ----------------------------------------------------------------------------
// out += alpha * (A_tile . B_tile). Handles matmul and the LU trailing
// update uniformly (they are the same operation — see kpu-program-model.md).
// ----------------------------------------------------------------------------
inline void kernel_matmul_accum(TileProgram& prog, const TileOp& op) {
    const TileCoord& ac = op.inputs.at(0);
    const TileCoord& bc = op.inputs.at(1);
    const TileCoord& cc = op.outputs.at(0);
    const TensorOperand& A = prog.operand(ac.operand);
    const TensorOperand& B = prog.operand(bc.operand);
    TensorOperand& C = prog.operand(cc.operand);

    const Dim ar0 = A.row_begin(ac.ti), ar1 = A.row_end(ac.ti);
    const Dim ak0 = A.col_begin(ac.tj), ak1 = A.col_end(ac.tj);
    const Dim bk0 = B.row_begin(bc.ti), bk1 = B.row_end(bc.ti);
    const Dim bc0 = B.col_begin(bc.tj), bc1 = B.col_end(bc.tj);
    const Dim cr0 = C.row_begin(cc.ti), cr1 = C.row_end(cc.ti);
    const Dim cc0 = C.col_begin(cc.tj), cc1 = C.col_end(cc.tj);

    const Dim m = cr1 - cr0;         // output rows
    const Dim n = cc1 - cc0;         // output cols
    const Dim kk = ak1 - ak0;        // contract length
    if ((ar1 - ar0) != m || (bc1 - bc0) != n || (bk1 - bk0) != kk)
        throw std::invalid_argument("MatMulAccum: incompatible tile shapes at " + cc.to_string());

    for (Dim li = 0; li < m; ++li)
        for (Dim lj = 0; lj < n; ++lj) {
            float acc = 0.0f;
            for (Dim lk = 0; lk < kk; ++lk)
                acc += A.at(ar0 + li, ak0 + lk) * B.at(bk0 + lk, bc0 + lj);
            C.at(cr0 + li, cc0 + lj) += op.alpha * acc;
        }
}

// ----------------------------------------------------------------------------
// GETRF: factor the diagonal tile A[k,k] in place with within-tile partial
// pivoting. Pivot search and row swaps are confined to this tile's rows, and
// applied here only to this tile's columns; the SAME swaps are replayed onto
// the rest of the row-block by explicit PivotApply (LASWP) ops — so the total
// effect is a full-row interchange and the factorization stays P.A = L.U with
// extractable L,U (P block-diagonal in the tile grid). Records the swaps
// (absolute rows) into op.pivot_slot and advances the global permutation.
//
// Pivoting is confined to the diagonal tile (a valid restricted-pivoting tile
// scheme). Cross-tile pairwise pivoting (PLASMA TSTRF/SSSSM) is the numerically
// stronger variant tracked separately — see docs/plans/plasma-tile-algorithms.md.
// ----------------------------------------------------------------------------
inline void kernel_lu_diag_factor(TileProgram& prog, const TileOp& op, TileKernelState& st) {
    TensorOperand& A = prog.operand(op.outputs.at(0).operand);
    st.ensure_perm(A.rows);
    auto& swaps = st.pivots[op.pivot_slot];
    swaps.clear();

    const Dim ti = op.outputs.at(0).ti;
    const Dim tj = op.outputs.at(0).tj;
    const Dim r0 = A.row_begin(ti), r1 = A.row_end(ti);   // diagonal tile row range
    const Dim c0 = A.col_begin(tj), c1 = A.col_end(tj);   // diagonal tile col range
    const Dim n = std::min(r1 - r0, c1 - c0);             // square block dimension

    for (Dim d = 0; d < n; ++d) {
        const Dim p = r0 + d;        // pivot row (global)
        const Dim c = c0 + d;        // pivot column (global)
        // partial pivot: largest magnitude in column c among this tile's rows
        Dim best = p;
        float bestval = std::fabs(A.at(p, c));
        for (Dim r = p + 1; r < r1; ++r) {
            const float v = std::fabs(A.at(r, c));
            if (v > bestval) { bestval = v; best = r; }
        }
        if (best != p) {
            for (Dim col = c0; col < c1; ++col)          // swap within this tile only
                std::swap(A.at(p, col), A.at(best, col));
            std::swap(st.perm[p], st.perm[best]);
            swaps.emplace_back(p, best);
            ++st.swaps_performed;
        }
        const float pv = A.at(p, c);
        if (pv == 0.0f) continue;                        // singular column; leave as-is
        for (Dim r = p + 1; r < r1; ++r) {
            const float f = A.at(r, c) / pv;
            A.at(r, c) = f;                              // store L multiplier
            for (Dim col = c + 1; col < c1; ++col)
                A.at(r, col) -= f * A.at(p, col);        // eliminate within the tile
        }
    }
}

// ----------------------------------------------------------------------------
// LASWP: replay the diagonal tile's recorded row swaps (op.pivot_slot) onto
// another tile in the same row-block (this op's output tile-column). This is
// the pivot decision from GETRF propagating explicitly to the rest of the row.
// ----------------------------------------------------------------------------
inline void kernel_pivot_apply(TileProgram& prog, const TileOp& op, const TileKernelState& st) {
    TensorOperand& A = prog.operand(op.outputs.at(0).operand);
    const Dim tj = op.outputs.at(0).tj;
    const Dim col0 = A.col_begin(tj), col1 = A.col_end(tj);
    auto it = st.pivots.find(op.pivot_slot);
    if (it == st.pivots.end()) return;        // no swaps recorded for this panel
    for (const auto& [ra, rb] : it->second)
        for (Dim col = col0; col < col1; ++col)
            std::swap(A.at(ra, col), A.at(rb, col));
}

// ----------------------------------------------------------------------------
// TRSM (left, unit-lower): X := unit-lower(A[k,k])^{-1} . X in place. Forward
// substitution; the implicit unit diagonal means no division. Produces the U
// row-panel U[k,j] = L_kk^{-1} . A[k,j].
// ----------------------------------------------------------------------------
inline void kernel_trsm_lower_left(TileProgram& prog, const TileOp& op) {
    const TileCoord& lc = op.inputs.at(0);   // diagonal block A[k,k] (holds L_kk)
    const TileCoord& xc = op.outputs.at(0);  // A[k,j] (in place)
    const TensorOperand& L = prog.operand(lc.operand);
    TensorOperand& X = prog.operand(xc.operand);

    const Dim lr0 = L.row_begin(lc.ti);
    const Dim lc0 = L.col_begin(lc.tj);
    const Dim xr0 = X.row_begin(xc.ti), xr1 = X.row_end(xc.ti);
    const Dim xc0 = X.col_begin(xc.tj), xc1 = X.col_end(xc.tj);
    const Dim m = xr1 - xr0;                 // block rows
    const Dim w = xc1 - xc0;                 // block cols

    for (Dim a = 0; a < m; ++a)
        for (Dim col = 0; col < w; ++col) {
            float v = X.at(xr0 + a, xc0 + col);
            for (Dim b = 0; b < a; ++b)      // strictly-below-diagonal L only
                v -= L.at(lr0 + a, lc0 + b) * X.at(xr0 + b, xc0 + col);
            X.at(xr0 + a, xc0 + col) = v;    // unit diagonal: no divide
        }
}

// ----------------------------------------------------------------------------
// TRSM (right, upper): X := X . upper(A[k,k])^{-1} in place. Computes the L
// column-panel L[i,k] = A[i,k] . U_kk^{-1}. U_kk has a non-unit diagonal, so
// each column solve divides by the diagonal.
// ----------------------------------------------------------------------------
inline void kernel_trsm_upper_right(TileProgram& prog, const TileOp& op) {
    const TileCoord& uc = op.inputs.at(0);   // diagonal block A[k,k] (holds U_kk)
    const TileCoord& xc = op.outputs.at(0);  // A[i,k] (in place)
    const TensorOperand& U = prog.operand(uc.operand);
    TensorOperand& X = prog.operand(xc.operand);

    const Dim ur0 = U.row_begin(uc.ti);
    const Dim uc0 = U.col_begin(uc.tj);
    const Dim xr0 = X.row_begin(xc.ti), xr1 = X.row_end(xc.ti);
    const Dim xc0 = X.col_begin(xc.tj), xc1 = X.col_end(xc.tj);
    const Dim m = xr1 - xr0;                 // rows of X
    const Dim w = xc1 - xc0;                 // cols of X (== U dimension)

    for (Dim a = 0; a < m; ++a)
        for (Dim c = 0; c < w; ++c) {
            float v = X.at(xr0 + a, xc0 + c);
            for (Dim b = 0; b < c; ++b)      // strictly-above-diagonal U only
                v -= X.at(xr0 + a, xc0 + b) * U.at(ur0 + b, uc0 + c);
            X.at(xr0 + a, xc0 + c) = v / U.at(ur0 + c, uc0 + c);
        }
}

// ============================================================================
// apply — execute ONE tile op's compute. The whole of L0 arithmetic dispatch.
//
// Feed and Drain are structural: they name a tile crossing a fabric port, and
// carry no arithmetic (L1 attaches the streams; the transactional tier attaches
// movement cost). They are no-ops here, deliberately, so a driver can call
// apply() unconditionally for every op.
//
// A driver must not reorder ops past their dependencies — tile RAW/WAR/WAW and
// the pivot-slot edges — or the state above is read out of order.
// ============================================================================
inline void apply(TileProgram& prog, const TileOp& op, TileKernelState& st) {
    switch (op.kind) {
        case TileOpKind::Feed:           break;   // structural
        case TileOpKind::Drain:          break;   // structural
        case TileOpKind::MatMulAccum:    kernel_matmul_accum(prog, op); break;
        case TileOpKind::LuDiagFactor:   kernel_lu_diag_factor(prog, op, st); break;
        case TileOpKind::PivotApply:     kernel_pivot_apply(prog, op, st); break;
        case TileOpKind::TrsmLowerLeft:  kernel_trsm_lower_left(prog, op); break;
        case TileOpKind::TrsmUpperRight: kernel_trsm_upper_right(prog, op); break;
    }
}

} // namespace sw::kpu::program
