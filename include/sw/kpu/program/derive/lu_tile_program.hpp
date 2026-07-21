// ============================================================================
// include/sw/kpu/program/derive/lu_tile_program.hpp
// Derive an L0 TileProgram for in-place LU factorization of a square N x N matrix,
// block size T, as an EXPLICIT DAG of PLASMA-style tile kernels — every op names
// the exact tiles it reads/writes (§3a of docs/plans/kpu-program-model.md), so the
// multi-tile column propagation is in the program, not hidden in a kernel.
//
// Right-looking tile LU (pivoting confined to the diagonal tile), per block-column k:
//   LU_DIAG_FACTOR A[k,k]                        ; GETRF: factor diagonal tile (within-tile pivot)
//   for g<k: PIVOT_APPLY  A[k,g] (slot k)        ; LASWP: replay swaps onto already-computed L (left)
//   for j>k: PIVOT_APPLY  A[k,j] (slot k)        ; LASWP: replay swaps onto the trailing row-block
//   for j>k: TRSM_LOWER_LEFT  A[k,j] <- A[k,k]   ; U row-panel  U[k,j] = L_kk^{-1} A[k,j]
//   for i>k: TRSM_UPPER_RIGHT A[i,k] <- A[k,k]   ; L col-panel  L[i,k] = A[i,k] U_kk^{-1}
//   for i>k,j>k: MATMUL_ACCUM A[i,j] -= A[i,k].A[k,j]   ; GEMM trailing update (alpha=-1)
//
// The trailing update is literally a matmul (alpha=-1), reusing MatMulAccum. This is
// the {GETRF, LASWP, TRSM, GEMM} tile-LU decomposition (the Cholesky-analog kernel
// set); cross-tile pairwise/neighbor pivoting (PLASMA {GETRF, GESSM, TSTRF, SSSSM})
// is the numerically stronger variant tracked in docs/plans/plasma-tile-algorithms.md.
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
// operand A (0-initialized; fill it with the matrix before running). After the run,
// A holds the unit-lower L below the diagonal and U on/above it in place; the
// reference reports the pivot permutation P (with P.A = L.U).
inline TileProgram derive_lu_tile_program(Dim N, Dim T, const std::string& a = "A") {
    if (N == 0 || T == 0)
        throw std::invalid_argument("derive_lu_tile_program: N and T must be > 0");

    TileProgram prog("lu " + std::to_string(N) + "x" + std::to_string(N) +
                     " T=" + std::to_string(T));
    const TensorOperand& A = prog.add_operand(TensorOperand(a, N, N, T, T));
    const Dim nt = A.n_tile_rows();   // == n_tile_cols() (square)

    for (Dim k = 0; k < nt; ++k) {
        const int slot = static_cast<int>(k);

        // GETRF: factor the diagonal tile in place (within-tile partial pivoting).
        TileOp getrf;
        getrf.kind = TileOpKind::LuDiagFactor;
        getrf.outputs = {TileCoord{a, k, k}};
        getrf.pivot_slot = slot;
        getrf.label = "GETRF diag " + std::to_string(k);
        prog.push(std::move(getrf));

        // LASWP: apply the diagonal tile's row swaps to the rest of the row-block —
        // the already-computed L to the left and the trailing columns to the right.
        for (Dim g = 0; g < k; ++g) {
            TileOp laswp;
            laswp.kind = TileOpKind::PivotApply;
            laswp.outputs = {TileCoord{a, k, g}};
            laswp.pivot_slot = slot;
            prog.push(std::move(laswp));
        }
        for (Dim j = k + 1; j < nt; ++j) {
            TileOp laswp;
            laswp.kind = TileOpKind::PivotApply;
            laswp.outputs = {TileCoord{a, k, j}};
            laswp.pivot_slot = slot;
            prog.push(std::move(laswp));
        }

        // TRSM: U row-panel  U[k,j] = L_kk^{-1} . A[k,j].
        for (Dim j = k + 1; j < nt; ++j) {
            TileOp trsm;
            trsm.kind = TileOpKind::TrsmLowerLeft;
            trsm.inputs = {TileCoord{a, k, k}};
            trsm.outputs = {TileCoord{a, k, j}};
            prog.push(std::move(trsm));
        }

        // TRSM: L col-panel  L[i,k] = A[i,k] . U_kk^{-1}.
        for (Dim i = k + 1; i < nt; ++i) {
            TileOp trsm;
            trsm.kind = TileOpKind::TrsmUpperRight;
            trsm.inputs = {TileCoord{a, k, k}};
            trsm.outputs = {TileCoord{a, i, k}};
            prog.push(std::move(trsm));
        }

        // GEMM trailing update: A[i,j] -= L[i,k] . U[k,j].
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
