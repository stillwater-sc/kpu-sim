// ============================================================================
// include/sw/kpu/program/stream/stream_signature.hpp
// L1 stream signatures — the spatial/temporal (timing) layer over an L0 TileProgram.
//
// The streams are a FUNCTION OF THE SPACE-TIME MAPPING (which operand is stationary),
// so L1 is parameterized by a SpaceTimeMap (schedule vector τ + projection u) and also
// records the NETWORK the resulting streams require (a 2-D mesh vs. a hexagonal
// overlay). L1 never changes values — it is derived from L0 + the mapping. See
// docs/plans/l1-stream-signatures.md and kpu-program-model.md §3.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace sw::kpu::program::stream {

using program::Dim;
using program::TileProgram;

// 3-D integer vector over the matmul iteration axes (i, j, k).
using Vec3 = std::array<int, 3>;

inline int dot(const Vec3& a, const Vec3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
// parallel (same or opposite direction) for the axis/[1,1,1] vectors we use.
inline bool parallel(const Vec3& a, const Vec3& b) {
    // cross product == 0
    return a[1]*b[2] - a[2]*b[1] == 0 &&
           a[2]*b[0] - a[0]*b[2] == 0 &&
           a[0]*b[1] - a[1]*b[0] == 0;
}

// ============================================================================
// SpaceTimeMap — schedule τ (time = τ·x) + projection u (collapsed/stationary axis).
// A variable with propagation direction e_V is stationary iff u ∥ e_V.
// ============================================================================
struct SpaceTimeMap {
    Vec3 tau  = {1, 1, 1};   // schedule vector
    Vec3 proj = {0, 0, 1};   // projection direction (which axis is projected out)
    std::string name = "output-stationary";

    static SpaceTimeMap output_stationary() { return {{1,1,1}, {0,0,1}, "output-stationary"}; }
    static SpaceTimeMap b_stationary()      { return {{1,1,1}, {1,0,0}, "weight(B)-stationary"}; }
    static SpaceTimeMap a_stationary()      { return {{1,1,1}, {0,1,0}, "A-stationary"}; }
    static SpaceTimeMap fully_streaming()   { return {{1,1,1}, {1,1,1}, "fully-streaming(hex)"}; }

    // schedule/projection aligned (τ ∥ u) → the contention-free hexagonal case.
    bool aligned() const { return parallel(tau, proj); }

    // The operand held stationary is the one whose propagation direction is ∥ u
    // (C for proj=k, B for proj=i, A for proj=j; "" when nothing is stationary/hex).
    std::string stationary_operand() const {
        if (proj == Vec3{1, 1, 1}) return "";
        if (proj == Vec3{0, 0, 1}) return "C";
        if (proj == Vec3{1, 0, 0}) return "B";
        if (proj == Vec3{0, 1, 0}) return "A";
        return "";
    }
};

// Matmul variable propagation directions (the axis each is invariant along).
inline Vec3 propagation_dir(const std::string& var) {
    if (var == "A") return {0, 1, 0};   // A[i,k] invariant along j
    if (var == "B") return {1, 0, 0};   // B[k,j] invariant along i
    return {0, 0, 1};                    // C[i,j] accumulates along k
}

// ---- streams & network -----------------------------------------------------
enum class Edge { West, North, South, East, None };
inline const char* to_string(Edge e) {
    switch (e) {
        case Edge::West:  return "West";
        case Edge::North: return "North";
        case Edge::South: return "South";
        case Edge::East:  return "East";
        case Edge::None:  return "None";
    }
    return "?";
}

enum class FlowRole { Stationary, StreamIn, StreamOut };
inline const char* to_string(FlowRole r) {
    switch (r) {
        case FlowRole::Stationary: return "STATIONARY";
        case FlowRole::StreamIn:   return "STREAM_IN";
        case FlowRole::StreamOut:  return "STREAM_OUT";
    }
    return "?";
}

enum class FabricTopology { Mesh2D, Hexagonal };
inline const char* to_string(FabricTopology f) {
    return f == FabricTopology::Mesh2D ? "Mesh2D" : "Hexagonal";
}

// ============================================================================
// StreamSignature — how one variable's tile streams (or stays) in the array.
// element (r,c) of a tile injects on `lane` at relative time
//   t0 + lane_skew·(lane) + element_stride·(sequence-index)   (schedule-derived).
// ============================================================================
struct StreamSignature {
    std::string var;                 // "A" / "B" / "C"
    FlowRole role = FlowRole::StreamIn;
    Edge edge = Edge::West;          // entry (StreamIn) / exit (StreamOut) edge
    std::array<int, 2> flow = {0, 0}; // array-space step per cycle {d_row, d_col}
    int lane_skew = 0;               // per-lane start offset
    int element_stride = 1;          // cycles between consecutive elements on a lane
    Dim lanes = 0;
    Dim rows = 0, cols = 0;          // tile element extent
    double rate = 1.0;               // effective elements/cycle/lane = 1/element_stride

    int bubble() const { return element_stride > 0 ? element_stride - 1 : 0; }
    Dim element_count() const { return rows * cols; }
    bool dense() const { return element_stride == 1; }
};

// Systolic wavefront latency of one tile-compute on an R×S array reducing over
// k_depth, schedule σ(i,j,k)=i+j+k: (R-1)+(S-1)+(K-1)+1 = fill + reduce + drain.
struct WavefrontTiming {
    Dim array_rows = 0, array_cols = 0, k_depth = 0;
    Dim latency() const {
        auto m = [](Dim x) { return x > 0 ? x - 1 : 0; };
        return m(array_rows) + m(array_cols) + m(k_depth) + 1;
    }
};

// The interconnect the chosen mapping's streams require.
struct NetworkOverlay {
    FabricTopology required = FabricTopology::Mesh2D;
    bool needs_overlay_on_mesh = false;                  // Hexagonal on a 2-D mesh fabric
    std::vector<std::array<int, 2>> stream_directions;   // distinct array flow directions
};

// ============================================================================
// StreamProgram — the L1 layer over an L0 matmul TileProgram for one SpaceTimeMap.
// ============================================================================
struct StreamProgram {
    SpaceTimeMap map;
    std::map<std::string, StreamSignature> signatures;   // by variable name
    NetworkOverlay network;
    std::map<std::size_t, WavefrontTiming> computes;     // by L0 op index
    Dim array_rows = 0, array_cols = 0;

    const StreamSignature* signature(const std::string& var) const {
        auto it = signatures.find(var);
        return it == signatures.end() ? nullptr : &it->second;
    }

    std::string disassemble() const;
};

// ---- disassembly -----------------------------------------------------------
inline std::string StreamProgram::disassemble() const {
    std::string s = "StreamProgram [" + map.name + "] array " +
                    std::to_string(array_rows) + "x" + std::to_string(array_cols) +
                    "  network=" + to_string(network.required) +
                    (network.needs_overlay_on_mesh ? " (overlay-on-mesh)" : "") + "\n";
    for (const auto& var : {std::string("A"), std::string("B"), std::string("C")}) {
        auto it = signatures.find(var);
        if (it == signatures.end()) continue;
        const auto& sg = it->second;
        s += "  " + var + ": " + to_string(sg.role);
        if (sg.role != FlowRole::Stationary) {
            s += " @" + std::string(to_string(sg.edge)) +
                 " flow(" + std::to_string(sg.flow[0]) + "," + std::to_string(sg.flow[1]) + ")" +
                 " skew=" + std::to_string(sg.lane_skew) +
                 " stride=" + std::to_string(sg.element_stride) +
                 " bubble=" + std::to_string(sg.bubble()) +
                 " lanes=" + std::to_string(sg.lanes);
        }
        s += "\n";
    }
    // one representative wavefront
    if (!computes.empty()) {
        const auto& w = computes.begin()->second;
        s += "  compute wavefront " + std::to_string(w.array_rows) + "x" +
             std::to_string(w.array_cols) + "x" + std::to_string(w.k_depth) +
             " latency=" + std::to_string(w.latency()) + " (x" +
             std::to_string(computes.size()) + ")\n";
    }
    return s;
}

} // namespace sw::kpu::program::stream
