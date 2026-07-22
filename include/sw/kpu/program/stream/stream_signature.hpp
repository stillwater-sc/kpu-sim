// ============================================================================
// include/sw/kpu/program/stream/stream_signature.hpp
// L1 stream signatures — the spatial/temporal (timing) layer over an L0 TileProgram.
//
// For each L0 Feed/Drain (a tile <-> fabric-port injection), L1 records HOW the tile's
// elements become an element stream at an array edge (lane assignment + wavefront skew
// + rate); for each compute op it records the systolic wavefront latency. L1 never
// changes values — it is derived from L0 + the array mapping. See
// docs/plans/l1-stream-signatures.md and kpu-program-model.md §3.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <map>
#include <string>

namespace sw::kpu::program::stream {

using program::Dim;
using program::TileProgram;

// Array boundary the stream injects into / extracts from.
enum class Edge { West, North, South, East };
inline const char* to_string(Edge e) {
    switch (e) {
        case Edge::West:  return "West";
        case Edge::North: return "North";
        case Edge::South: return "South";
        case Edge::East:  return "East";
    }
    return "?";
}

// Which tile axis maps to the array-edge lane index.
enum class LaneAxis { Row, Col };

// ============================================================================
// StreamSignature — how one tile's rows*cols elements stream at an array edge.
//
// Element (r,c) of the tile injects on lane = (r if Row axis else c), at relative
// time t0 + skew_row*r + skew_col*c, at `rate` elements/cycle/lane.
// ============================================================================
struct StreamSignature {
    std::string port;               // L0 logical port name (West/North/South)
    Edge edge = Edge::West;
    LaneAxis lane_axis = LaneAxis::Row;
    Dim lanes = 0;                  // array-edge cells fed in parallel
    Dim rows = 0, cols = 0;         // tile element extent
    int skew_row = 1, skew_col = 1; // affine injection-time skew
    double rate = 1.0;              // elements / cycle / lane
    bool is_output = false;         // Drain (extract) vs Feed (inject)

    Dim element_count() const { return rows * cols; }

    Dim lane_of(Dim r, Dim c) const { return lane_axis == LaneAxis::Row ? r : c; }
    int time_of(Dim r, Dim c) const {
        return skew_row * static_cast<int>(r) + skew_col * static_cast<int>(c);
    }
    // cycles from first to last element injection within the tile.
    int time_span() const {
        const int r = rows > 0 ? static_cast<int>(rows) - 1 : 0;
        const int c = cols > 0 ? static_cast<int>(cols) - 1 : 0;
        return skew_row * r + skew_col * c;
    }
};

// ============================================================================
// WavefrontTiming — systolic latency of one tile-compute on an R×S array
// reducing over k_depth, with the output-stationary schedule σ(i,j,k)=i+j+k.
// ============================================================================
struct WavefrontTiming {
    Dim array_rows = 0, array_cols = 0, k_depth = 0;

    // latency = (R-1) + (S-1) + (K-1) + 1  =  fill + reduce + drain
    Dim latency() const {
        auto m = [](Dim x) { return x > 0 ? x - 1 : 0; };
        return m(array_rows) + m(array_cols) + m(k_depth) + 1;
    }
};

// ============================================================================
// StreamProgram — the L1 layer over an L0 TileProgram, keyed by L0 op index.
// ============================================================================
struct StreamProgram {
    std::map<std::size_t, StreamSignature> streams;   // op index -> feed/drain signature
    std::map<std::size_t, WavefrontTiming> computes;  // op index -> wavefront timing
    Dim array_rows = 0, array_cols = 0;               // inferred physical array

    std::string disassemble(const TileProgram& l0) const;
};

// ---- disassembly -----------------------------------------------------------
inline std::string StreamProgram::disassemble(const TileProgram& l0) const {
    std::string s = "StreamProgram (array " + std::to_string(array_rows) + "x" +
                    std::to_string(array_cols) + ")\n";
    const auto& ops = l0.ops();
    for (std::size_t i = 0; i < ops.size(); ++i) {
        auto st = streams.find(i);
        if (st != streams.end()) {
            const auto& sg = st->second;
            const auto& op = ops[i];
            const auto tile = op.outputs.empty()
                ? (op.inputs.empty() ? std::string() : op.inputs[0].to_string())
                : op.outputs[0].to_string();
            s += "  " + std::to_string(i) + ": " + (sg.is_output ? "OUT " : "IN  ") + tile +
                 " @" + to_string(sg.edge) + " lanes=" + std::to_string(sg.lanes) +
                 " skew(" + std::to_string(sg.skew_row) + "," + std::to_string(sg.skew_col) +
                 ") span=" + std::to_string(sg.time_span()) +
                 " elems=" + std::to_string(sg.element_count()) + "\n";
        }
        auto ct = computes.find(i);
        if (ct != computes.end()) {
            const auto& w = ct->second;
            s += "  " + std::to_string(i) + ": COMPUTE " + ops[i].outputs[0].to_string() +
                 " wavefront " + std::to_string(w.array_rows) + "x" + std::to_string(w.array_cols) +
                 "x" + std::to_string(w.k_depth) + " latency=" + std::to_string(w.latency()) + "\n";
        }
    }
    return s;
}

} // namespace sw::kpu::program::stream
