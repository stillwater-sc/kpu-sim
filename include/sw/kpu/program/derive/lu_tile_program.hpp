// ============================================================================
// include/sw/kpu/program/derive/lu_tile_program.hpp
// Derive an L0 TileProgram for in-place LU factorization with neighbor
// (pairwise) pivoting of a square N x N matrix, block size T.
//
// This is the coverage bar beyond matmul (issue #230): LU exercises exactly what
// matmul does not, and what every dense factorization (Cholesky/QR/LU/triangular
// solve) shares — so representing it proves the L0 abstraction generalizes:
//   * cross-tile data dependencies   (panel -> TRSM -> trailing update)
//   * a shrinking trailing submatrix (the k-loop over block-columns)
//   * data-dependent control         (neighbor-pivot row swaps decided from
//                                      values in one op, flowing into trailing
//                                      tiles via PivotApply)
//
// Right-looking blocked LU, per block-column k:
//   LU_PANEL_FACTOR A[k,k]                  ; factor tall column panel k in place,
//                                             neighbor pivoting -> unit-L + U + pivots
//   for j>k: PIVOT_APPLY  A[*,j]  (slot k)  ; replay row swaps onto trailing columns
//   for j>k: TRSM_LEFT    A[k,j] <- A[k,k]  ; U[k,j] = L_kk^{-1} . A[k,j]
//   for i>k,j>k: MATMUL_ACCUM A[i,j] -= A[i,k].A[k,j]   ; trailing update (alpha=-1)
//
// The trailing update is literally a matmul (alpha=-1), reusing MatMulAccum — the
// unification called out in docs/plans/kpu-program-model.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu::program {

// Derive the LU tile program for a square N x N matrix, block T x T. Creates one
// operand A (0-initialized; fill it with the matrix before running). After the
// run, A holds the unit-lower L below the diagonal and U on/above it in place;
// the reference reports the neighbor-pivot permutation P (with P.A = L.U).
inline TileProgram derive_lu_neighbor_pivot_tile_program(
        Dim N, Dim T, const std::string& a = "A") {
    if (N == 0 || T == 0)
        throw std::invalid_argument("derive_lu_neighbor_pivot_tile_program: N and T must be > 0");

    TileProgram prog("lu-neighbor-pivot " + std::to_string(N) + "x" + std::to_string(N) +
                     " T=" + std::to_string(T));
    const TensorOperand& A = prog.add_operand(TensorOperand(a, N, N, T, T));
    const Dim nt = A.n_tile_rows();   // == n_tile_cols() (square)

    for (Dim k = 0; k < nt; ++k) {
        // Factor the tall column panel k in place, with neighbor pivoting.
        TileOp factor;
        factor.kind = TileOpKind::LuPanelFactor;
        factor.outputs = {TileCoord{a, k, k}};   // diagonal tile anchors the block-column
        factor.pivot_slot = static_cast<int>(k);
        factor.label = "factor column panel " + std::to_string(k);
        prog.push(std::move(factor));

        // Apply the panel's row swaps to the trailing block-columns.
        for (Dim j = k + 1; j < nt; ++j) {
            TileOp piv;
            piv.kind = TileOpKind::PivotApply;
            piv.outputs = {TileCoord{a, k, j}};  // tj=j selects the trailing column range
            piv.pivot_slot = static_cast<int>(k);
            prog.push(std::move(piv));
        }

        // U row-panel: U[k,j] = L_kk^{-1} . A[k,j].
        for (Dim j = k + 1; j < nt; ++j) {
            TileOp trsm;
            trsm.kind = TileOpKind::TrsmLeft;
            trsm.inputs = {TileCoord{a, k, k}};
            trsm.outputs = {TileCoord{a, k, j}};
            prog.push(std::move(trsm));
        }

        // Trailing update: A[i,j] -= L[i,k] . U[k,j]  (matmul, alpha = -1).
        for (Dim i = k + 1; i < nt; ++i) {
            for (Dim j = k + 1; j < nt; ++j) {
                TileOp upd;
                upd.kind = TileOpKind::MatMulAccum;
                upd.inputs = {TileCoord{a, i, k}, TileCoord{a, k, j}};
                upd.outputs = {TileCoord{a, i, j}};
                upd.alpha = -1.0f;
                prog.push(std::move(upd));
            }
        }
    }
    (void)A;
    return prog;
}

} // namespace sw::kpu::program
