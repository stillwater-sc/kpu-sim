// ============================================================================
// include/sw/kpu/program/derive/matmul_tile_program.hpp
// Derive an L0 TileProgram for matmul C = A . B (row-major), tiled Ti x Tj x Tk.
//
// This is the trivial coverage case: embarrassingly parallel, no cross-tile
// dependency, no data-dependent control. It establishes the outer-loop tile
// sequence exactly as sketched in docs/plans/kpu-program-model.md §3:
//
//   for (ti,tj):
//     for tk: feed A[ti,tk]->West ; feed B[tk,tj]->North ; C[ti,tj] += A.B
//     drain C[ti,tj]->South
//
// Device-independent: parameterized purely by shapes + tiling (no engine ids).
// The caller fills operands A and B before running the functional reference.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu::program {

// Derive the matmul tile program. Operands created 0-initialized:
//   A: M x K tiled Ti x Tk   (fill before running)
//   B: K x N tiled Tk x Tj   (fill before running)
//   C: M x N tiled Ti x Tj   (result; accumulated during the run)
// Non-divisible tilings are supported (trailing tiles clamp).
inline TileProgram derive_matmul_tile_program(
        Dim M, Dim N, Dim K, Dim Ti, Dim Tj, Dim Tk,
        const std::string& a = "A", const std::string& b = "B", const std::string& c = "C") {
    if (Ti == 0 || Tj == 0 || Tk == 0)
        throw std::invalid_argument("derive_matmul_tile_program: tile dims must be > 0");

    TileProgram prog("matmul " + std::to_string(M) + "x" + std::to_string(N) +
                     "x" + std::to_string(K));
    const TensorOperand& A = prog.add_operand(TensorOperand(a, M, K, Ti, Tk));
    const TensorOperand& B = prog.add_operand(TensorOperand(b, K, N, Tk, Tj));
    const TensorOperand& C = prog.add_operand(TensorOperand(c, M, N, Ti, Tj));

    const Dim mt = A.n_tile_rows();   // M / Ti  (ceil)
    const Dim kt = A.n_tile_cols();   // K / Tk
    const Dim nt = B.n_tile_cols();   // N / Tj

    for (Dim ti = 0; ti < mt; ++ti) {
        for (Dim tj = 0; tj < nt; ++tj) {
            for (Dim tk = 0; tk < kt; ++tk) {
                TileOp feed_a;
                feed_a.kind = TileOpKind::Feed;
                feed_a.port_kind = PortKind::Input;
                feed_a.port = "West";
                feed_a.inputs = {TileCoord{a, ti, tk}};
                prog.push(std::move(feed_a));

                TileOp feed_b;
                feed_b.kind = TileOpKind::Feed;
                feed_b.port_kind = PortKind::Input;
                feed_b.port = "North";
                feed_b.inputs = {TileCoord{b, tk, tj}};
                prog.push(std::move(feed_b));

                TileOp mac;
                mac.kind = TileOpKind::MatMulAccum;
                mac.inputs = {TileCoord{a, ti, tk}, TileCoord{b, tk, tj}};
                mac.outputs = {TileCoord{c, ti, tj}};
                mac.alpha = 1.0f;
                prog.push(std::move(mac));
            }
            TileOp drain;
            drain.kind = TileOpKind::Drain;
            drain.port_kind = PortKind::Output;
            drain.port = "South";
            drain.outputs = {TileCoord{c, ti, tj}};
            prog.push(std::move(drain));
        }
    }
    (void)C;
    return prog;
}

} // namespace sw::kpu::program
