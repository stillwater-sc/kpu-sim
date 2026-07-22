// ============================================================================
// include/sw/kpu/program/stream/derive/matmul_streams.hpp
// Derive L1 stream signatures for an L0 matmul TileProgram under a chosen
// space-time mapping. The streams depend on WHICH operand is stationary:
//
//   output-stationary  (proj=k) : C held; A→West, B→North, C evacuates North (bubble 1)
//   weight(B)-stationary (proj=i): B held; A→North, C→East (dense)
//   A-stationary       (proj=j) : A held; B→North, C→East (dense)
//   fully-streaming    (proj=[1,1,1]): all stream; hexagonal network (overlay on a mesh)
//
// The result bubble in the output-stationary case is the crux: C traverses the filled
// array to the free North edge, so successive results exit two cycles apart. The other
// mappings evacuate C at the edge where accumulation finishes → dense.
// See docs/plans/l1-stream-signatures.md.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/stream/stream_signature.hpp>

#include <string>

namespace sw::kpu::program::stream {

namespace detail {
inline StreamSignature make_sig(const std::string& var, FlowRole role, Edge edge,
                                std::array<int, 2> flow, int skew, int stride,
                                Dim rows, Dim cols) {
    StreamSignature s;
    s.var = var; s.role = role; s.rows = rows; s.cols = cols;
    if (role == FlowRole::Stationary) {
        s.edge = Edge::None; s.flow = {0, 0}; s.lanes = 0;
        s.lane_skew = 0; s.element_stride = 1;
    } else {
        s.edge = edge; s.flow = flow; s.lane_skew = skew; s.element_stride = stride;
        s.lanes = (flow[0] != 0) ? cols : rows;   // extent perpendicular to the flow
    }
    s.rate = 1.0 / static_cast<double>(s.element_stride);
    return s;
}
} // namespace detail

// Derive the L1 layer for a matmul L0 program under `map` (default output-stationary).
// Values are untouched. Array dims are read from the (square) tile shape.
inline StreamProgram derive_matmul_streams(const TileProgram& l0,
                                           SpaceTimeMap map = SpaceTimeMap::output_stationary(),
                                           const std::string& a = "A",
                                           [[maybe_unused]] const std::string& b = "B",
                                           const std::string& c = "C") {
    using detail::make_sig;
    StreamProgram sp;
    sp.map = map;

    const TensorOperand& C = l0.operand(c);
    const TensorOperand& A = l0.operand(a);
    sp.array_rows = C.tile_rows;
    sp.array_cols = C.tile_cols;
    const Dim Ti = C.tile_rows;      // M-tile
    const Dim Tj = C.tile_cols;      // N-tile
    const Dim Tk = A.tile_cols;      // K-tile
    // per-variable tile shapes: A=Ti×Tk, B=Tk×Tj, C=Ti×Tj

    const Vec3& u = map.proj;
    if (u == Vec3{0, 0, 1}) {                 // output-stationary
        sp.signatures["A"] = make_sig("A", FlowRole::StreamIn,  Edge::West,  {0, 1}, 1, 1, Ti, Tk);
        sp.signatures["B"] = make_sig("B", FlowRole::StreamIn,  Edge::North, {1, 0}, 1, 1, Tk, Tj);
        sp.signatures["C"] = make_sig("C", FlowRole::StreamOut, Edge::North, {-1, 0}, 1, 2, Ti, Tj);
        sp.network = {FabricTopology::Mesh2D, false, {{0, 1}, {1, 0}, {-1, 0}}};
    } else if (u == Vec3{1, 0, 0}) {          // weight(B)-stationary
        sp.signatures["A"] = make_sig("A", FlowRole::StreamIn,  Edge::North, {1, 0}, 1, 1, Ti, Tk);
        sp.signatures["B"] = make_sig("B", FlowRole::Stationary, Edge::None, {0, 0}, 0, 1, Tk, Tj);
        sp.signatures["C"] = make_sig("C", FlowRole::StreamOut, Edge::East,  {0, 1}, 1, 1, Ti, Tj);
        sp.network = {FabricTopology::Mesh2D, false, {{1, 0}, {0, 1}}};
    } else if (u == Vec3{0, 1, 0}) {          // A-stationary
        sp.signatures["A"] = make_sig("A", FlowRole::Stationary, Edge::None, {0, 0}, 0, 1, Ti, Tk);
        sp.signatures["B"] = make_sig("B", FlowRole::StreamIn,  Edge::North, {1, 0}, 1, 1, Tk, Tj);
        sp.signatures["C"] = make_sig("C", FlowRole::StreamOut, Edge::East,  {0, 1}, 1, 1, Ti, Tj);
        sp.network = {FabricTopology::Mesh2D, false, {{1, 0}, {0, 1}}};
    } else {                                  // fully-streaming: hexagonal (proj=[1,1,1])
        // τ ∥ proj → aligned → dense (no bubbles), perfect concurrency; three stream
        // directions at 60° require a hex network (an overlay on a 2-D mesh fabric).
        sp.signatures["A"] = make_sig("A", FlowRole::StreamIn,  Edge::None, {1, 0},  0, 1, Ti, Tk);
        sp.signatures["B"] = make_sig("B", FlowRole::StreamIn,  Edge::None, {0, 1},  0, 1, Tk, Tj);
        sp.signatures["C"] = make_sig("C", FlowRole::StreamOut, Edge::None, {-1, -1}, 0, 1, Ti, Tj);
        sp.network = {FabricTopology::Hexagonal, true, {{1, 0}, {0, 1}, {-1, -1}}};
    }

    // wavefront timing per compute op (clamped trailing tiles handled)
    const auto& ops = l0.ops();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].kind != TileOpKind::MatMulAccum) continue;
        const TileCoord& ac = ops[i].inputs.at(0);
        const TileCoord& cc = ops[i].outputs.at(0);
        const TensorOperand& Aop = l0.operand(ac.operand);
        const TensorOperand& Cop = l0.operand(cc.operand);
        WavefrontTiming w;
        w.array_rows = Cop.row_end(cc.ti) - Cop.row_begin(cc.ti);
        w.array_cols = Cop.col_end(cc.tj) - Cop.col_begin(cc.tj);
        w.k_depth = Aop.col_end(ac.tj) - Aop.col_begin(ac.tj);
        sp.computes[i] = w;
    }
    return sp;
}

} // namespace sw::kpu::program::stream
