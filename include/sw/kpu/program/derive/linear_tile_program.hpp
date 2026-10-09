// ============================================================================
// include/sw/kpu/program/derive/linear_tile_program.hpp
// Derive an L0 TileProgram for the linear operator Y = act(X . W + b) (docs/plans/
// csp-language.md step 2): matmul with its epilogue, the fusion case.
//
//   for (ti,tj):
//     for tk: feed X[ti,tk]->West ; feed W[tk,tj]->North ; Y[ti,tj] += X.W
//     feed b[tj]->North ; Y[ti,tj] += b (broadcast down the rows) ; Y[ti,tj] := act(Y[ti,tj])
//     drain Y[ti,tj]->South
//
// The epilogue sits between the accumulation and the drain: in the fabric, where L0 has no
// notion of placement. Where it runs is the CSP program's decision (a tile context), and every
// placement must compute what this order computes.
//
// The operands keep matmul's names, A . B -> C, with the bias b, so the existing value checks
// read C for every algo that is not LU.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <stdexcept>
#include <string>

namespace sw::kpu::program {

// A: M x K tiled Ti x Tk; B: K x N tiled Tk x Tj; b: a vector of N, tiled Tj (one column);
// C: M x N tiled Ti x Tj, the result.
inline TileProgram derive_linear_tile_program(Dim M, Dim N, Dim K, Dim Ti, Dim Tj, Dim Tk,
                                              ActivationFn act = ActivationFn::Relu) {
    const TileProgram mm = derive_matmul_tile_program(M, N, K, Ti, Tj, Tk);
    TileProgram prog("linear " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + " " +
                     to_string(act));
    for (const auto& name : mm.operand_order()) prog.add_operand(mm.operand(name));
    prog.add_operand(TensorOperand("b", N, 1, Tj, 1));
    for (const TileOp& op : mm.ops()) {
        if (op.kind == TileOpKind::Drain) {
            const TileCoord y = op.outputs.at(0);
            TileOp feed;
            feed.kind = TileOpKind::Feed;
            feed.port_kind = PortKind::Input;
            feed.port = "North";
            feed.inputs = {TileCoord{"b", y.tj, 0}};
            prog.push(feed);
            TileOp bias;
            bias.kind = TileOpKind::BiasAdd;
            bias.inputs = {TileCoord{"b", y.tj, 0}};
            bias.outputs = {y};
            prog.push(bias);
            TileOp a;
            a.kind = TileOpKind::Activation;
            a.act = act;
            a.outputs = {y};
            prog.push(a);
        }
        prog.push(op);
    }
    return prog;
}

} // namespace sw::kpu::program
