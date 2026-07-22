// ============================================================================
// include/sw/kpu/program/stream/derive/matmul_streams.hpp
// Derive L1 stream signatures for an L0 matmul TileProgram.
//
// Output-stationary systolic schedule σ(i,j,k)=i+j+k (docs/plans/l1-stream-signatures.md):
//   A[i,k] streams in at the West edge  (lane = row i,  time = i+k),
//   B[k,j] streams in at the North edge (lane = col j,  time = j+k),
//   C[i,j] drains at the South edge     (lane = col j,  time = i+j+(K-1)),
//   one tile-compute has latency (R-1)+(S-1)+(K-1)+1.
//
// Increment-1 assumption: each tile fits the physical array (R×S = Ti×Tj, K = Tk).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/stream/stream_signature.hpp>

#include <string>

namespace sw::kpu::program::stream {

// Derive the L1 layer for a matmul L0 program built by derive_matmul_tile_program.
// `a`/`b`/`c` name the operands (defaults match the deriver). Values are untouched.
inline StreamProgram derive_matmul_streams(const TileProgram& l0,
                                           const std::string& a = "A",
                                           [[maybe_unused]] const std::string& b = "B",
                                           const std::string& c = "C") {
    StreamProgram sp;
    const TensorOperand& C = l0.operand(c);
    sp.array_rows = C.tile_rows;    // nominal physical array = the C tile shape
    sp.array_cols = C.tile_cols;

    const auto& ops = l0.ops();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const TileOp& op = ops[i];
        switch (op.kind) {
            case TileOpKind::Feed: {
                const TileCoord& t = op.inputs.at(0);
                const TensorOperand& operand = l0.operand(t.operand);
                const Dim rows = operand.row_end(t.ti) - operand.row_begin(t.ti);
                const Dim cols = operand.col_end(t.tj) - operand.col_begin(t.tj);
                StreamSignature s;
                s.port = op.port;
                s.rows = rows; s.cols = cols;
                s.skew_row = 1; s.skew_col = 1; s.rate = 1.0; s.is_output = false;
                if (t.operand == a) {           // A[i,k] -> West, lane = row i
                    s.edge = Edge::West; s.lane_axis = LaneAxis::Row; s.lanes = rows;
                } else {                         // B[k,j] -> North, lane = col j
                    s.edge = Edge::North; s.lane_axis = LaneAxis::Col; s.lanes = cols;
                }
                sp.streams[i] = s;
                break;
            }
            case TileOpKind::Drain: {            // C[i,j] -> South, lane = col j
                const TileCoord& t = op.outputs.at(0);
                const TensorOperand& operand = l0.operand(t.operand);
                const Dim rows = operand.row_end(t.ti) - operand.row_begin(t.ti);
                const Dim cols = operand.col_end(t.tj) - operand.col_begin(t.tj);
                StreamSignature s;
                s.port = op.port;
                s.edge = Edge::South; s.lane_axis = LaneAxis::Col; s.lanes = cols;
                s.rows = rows; s.cols = cols;
                s.skew_row = 1; s.skew_col = 1; s.rate = 1.0; s.is_output = true;
                sp.streams[i] = s;
                break;
            }
            case TileOpKind::MatMulAccum: {
                const TileCoord& ac = op.inputs.at(0);   // A tile: cols = K depth
                const TensorOperand& A = l0.operand(ac.operand);
                const TileCoord& cc = op.outputs.at(0);
                const TensorOperand& Cop = l0.operand(cc.operand);
                WavefrontTiming w;
                w.array_rows = Cop.row_end(cc.ti) - Cop.row_begin(cc.ti);
                w.array_cols = Cop.col_end(cc.tj) - Cop.col_begin(cc.tj);
                w.k_depth = A.col_end(ac.tj) - A.col_begin(ac.tj);
                sp.computes[i] = w;
                break;
            }
            default:
                break;   // matmul L0 has no other op kinds
        }
    }
    return sp;
}

} // namespace sw::kpu::program::stream
