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
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <cmath>
#include <map>
#include <utility>
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
        std::size_t computes = 0;   // MatMulAccum
        std::size_t panel_factors = 0;
        std::size_t trsms = 0;
        std::size_t pivot_applies = 0;
        std::size_t neighbor_swaps = 0;   // total adjacent row swaps performed
        // Row permutation produced by pivoting (LU): perm[i] = original row now at
        // position i. Identity when no pivoting occurred. Sized to the largest
        // square operand touched by a LuPanelFactor (0 if none).
        std::vector<Dim> permutation;
    };

    // Execute program in order. Mutates program's operand value buffers.
    RunSummary run(TileProgram& program) {
        pivots_.clear();
        perm_.clear();
        RunSummary sum;
        for (const auto& op : program.ops()) {
            ++sum.ops;
            switch (op.kind) {
                case TileOpKind::Feed:          ++sum.feeds; break;   // structural (L1 attaches streams)
                case TileOpKind::Drain:         ++sum.drains; break;  // structural
                case TileOpKind::MatMulAccum:   exec_matmul_accum(program, op); ++sum.computes; break;
                case TileOpKind::LuPanelFactor: exec_lu_panel_factor(program, op); ++sum.panel_factors; break;
                case TileOpKind::TrsmLeft:      exec_trsm_left(program, op); ++sum.trsms; break;
                case TileOpKind::PivotApply:    exec_pivot_apply(program, op); ++sum.pivot_applies; break;
            }
        }
        sum.neighbor_swaps = swaps_performed_;
        sum.permutation = perm_;
        swaps_performed_ = 0;
        return sum;
    }

private:
    // Transient pivot state: slot -> ordered list of adjacent (r, r+1) row swaps.
    std::map<int, std::vector<std::pair<Dim, Dim>>> pivots_;
    std::vector<Dim> perm_;          // current row permutation (LU)
    std::size_t swaps_performed_ = 0;

    void ensure_perm(Dim n) {
        if (perm_.size() < n) {
            perm_.resize(n);
            for (Dim i = 0; i < n; ++i) perm_[i] = i;
        }
    }

    // out += alpha * (A_tile . B_tile). Handles matmul and the LU trailing
    // update uniformly (they are the same operation — see kpu-program-model.md).
    void exec_matmul_accum(TileProgram& prog, const TileOp& op) {
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

    // Factor column panel k (the block-column at tile-col == outputs[0].tj) in
    // place, over rows [k*T, rows), with neighbor (pairwise) pivoting. Stores the
    // unit-lower L multipliers below the diagonal and U on/above it, and records
    // the adjacent row swaps into op.pivot_slot for the trailing PivotApply ops.
    //
    // Neighbor pivoting: at each pivot column only the immediately-adjacent row
    // below is considered, and rows are exchanged only if that neighbor has the
    // larger magnitude. This keeps all data movement nearest-neighbor (systolic-
    // friendly) — the deliberately dataflow-faithful pivot scheme. The recorded
    // swaps compose into a permutation P with P.A = L.U.
    void exec_lu_panel_factor(TileProgram& prog, const TileOp& op) {
        TensorOperand& A = prog.operand(op.outputs.at(0).operand);
        ensure_perm(A.rows);
        auto& swaps = pivots_[op.pivot_slot];
        swaps.clear();

        const Dim tj = op.outputs.at(0).tj;      // panel block-column
        const Dim c0 = A.col_begin(tj);          // first panel column
        const Dim c1 = A.col_end(tj);            // one past last panel column
        const Dim rN = A.rows;

        for (Dim c = c0; c < c1; ++c) {
            const Dim p = c;                     // pivot sits on the diagonal
            // neighbor pivot: compare with the immediate row below only
            if (p + 1 < rN &&
                std::fabs(A.at(p + 1, c)) > std::fabs(A.at(p, c))) {
                // exchange rows p, p+1 across columns [0, c1): left (finalized L)
                // + panel. Trailing columns [c1, cols) are swapped by PivotApply.
                for (Dim col = 0; col < c1; ++col)
                    std::swap(A.at(p, col), A.at(p + 1, col));
                std::swap(perm_[p], perm_[p + 1]);
                swaps.emplace_back(p, p + 1);
                ++swaps_performed_;
            }
            const float pv = A.at(p, c);
            if (pv == 0.0f) continue;            // singular column; leave as-is
            for (Dim r = p + 1; r < rN; ++r) {
                const float f = A.at(r, c) / pv;
                A.at(r, c) = f;                  // store L multiplier
                for (Dim col = c + 1; col < c1; ++col)
                    A.at(r, col) -= f * A.at(p, col);   // eliminate within panel columns
            }
        }
    }

    // Solve L_kk . X = B in place, with L_kk the unit-lower-triangular part of the
    // diagonal tile (inputs[0]) and B the row-panel tile (outputs[0]). Forward
    // substitution; the implicit unit diagonal means no division. Produces the U
    // row-panel U[k,j] = L_kk^{-1} . A[k,j].
    void exec_trsm_left(TileProgram& prog, const TileOp& op) {
        const TileCoord& lc = op.inputs.at(0);   // diagonal block A[k,k]
        const TileCoord& xc = op.outputs.at(0);  // A[k,j] (in place)
        const TensorOperand& L = prog.operand(lc.operand);
        TensorOperand& X = prog.operand(xc.operand);

        const Dim lr0 = L.row_begin(lc.ti);      // diagonal block row/col origin
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

    // Replay the recorded neighbor-pivot swaps (from op.pivot_slot) onto a
    // trailing tile-column: apply each row exchange to the tile's column range.
    // This is the pivot decision from LuPanelFactor flowing into trailing tiles.
    void exec_pivot_apply(TileProgram& prog, const TileOp& op) {
        TensorOperand& A = prog.operand(op.outputs.at(0).operand);
        const Dim tj = op.outputs.at(0).tj;
        const Dim col0 = A.col_begin(tj), col1 = A.col_end(tj);
        auto it = pivots_.find(op.pivot_slot);
        if (it == pivots_.end()) return;         // no swaps recorded for this panel
        for (const auto& [ra, rb] : it->second)
            for (Dim col = col0; col < col1; ++col)
                std::swap(A.at(ra, col), A.at(rb, col));
    }
};

} // namespace sw::kpu::program
