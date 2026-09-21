// ============================================================================
// include/sw/kpu/program/tile_work.hpp
// The structural work of one tile op, and the cost models that turn it into
// cycles — one implementation, shared by the analysis harness and the executor.
//
// Work is STRUCTURAL: MACs for compute ops, bytes for movement ops, derived from
// the op's ACTUAL (clamped) tile extents so trailing tiles at non-divisible
// dimensions are costed correctly. Turning work into cycles is a separate,
// device-parameterized step, because the same program is costed differently on
// different devices — and, with an L1 StreamProgram, differently again per array
// dataflow.
//
// Deliberately scalar-parameterized (macs_per_cycle, bytes_per_cycle,
// element_bytes) rather than taking a DeviceDescriptor, so the portable-program
// layer does not depend on the characterization layer.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/stream/stream_signature.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sw::kpu::program {

struct TileWork {
    double macs = 0.0;      // multiply-accumulates (compute ops)
    double bytes = 0.0;     // bytes moved (movement ops)
    bool is_compute = false;
};

// Half-open tile extent for a coord, clamped to the operand — the reason a
// trailing 3-row tile is not costed as a full one.
inline void tile_extent(const TileProgram& p, const TileCoord& c, Dim& rows, Dim& cols) {
    const TensorOperand& op = p.operand(c.operand);
    rows = op.row_end(c.ti) - op.row_begin(c.ti);
    cols = op.col_end(c.tj) - op.col_begin(c.tj);
}

// ----------------------------------------------------------------------------
// Structural work of one op. Compute kinds carry MACs; movement kinds carry
// bytes. PivotApply is movement: it permutes rows, it does not compute.
// ----------------------------------------------------------------------------
inline TileWork tile_work_of(const TileProgram& p, const TileOp& op, double element_bytes) {
    TileWork w;
    Dim r = 0, c = 0, r2 = 0, c2 = 0;
    switch (op.kind) {
        case TileOpKind::MatMulAccum: {
            tile_extent(p, op.outputs[0], r, c);        // m x n
            tile_extent(p, op.inputs[0], r2, c2);       // m x k
            w.macs = double(r) * c * c2;                // m*n*k
            w.is_compute = true;
            break;
        }
        case TileOpKind::LuDiagFactor: {
            tile_extent(p, op.outputs[0], r, c);
            const double t = std::min(r, c);
            w.macs = t * t * t / 3.0;                   // ~ (1/3) t^3
            w.is_compute = true;
            break;
        }
        case TileOpKind::TrsmLowerLeft: {               // L(m x m) . X(m x w)
            tile_extent(p, op.outputs[0], r, c);        // m x w
            w.macs = double(r) * r * c / 2.0;
            w.is_compute = true;
            break;
        }
        case TileOpKind::TrsmUpperRight: {              // X(m x w) . U(w x w)
            tile_extent(p, op.outputs[0], r, c);        // m x w
            w.macs = double(r) * c * c / 2.0;
            w.is_compute = true;
            break;
        }
        case TileOpKind::PivotApply: {                  // row swaps: movement
            tile_extent(p, op.outputs[0], r, c);
            w.bytes = double(r) * c * element_bytes;
            break;
        }
        case TileOpKind::Feed:
        case TileOpKind::Drain: {
            const TileCoord& t = op.inputs.empty() ? op.outputs[0] : op.inputs[0];
            tile_extent(p, t, r, c);
            w.bytes = double(r) * c * element_bytes;
            break;
        }
    }
    return w;
}

// First-order lumped cost: work divided by throughput.
inline double lumped_duration(const TileWork& w, double macs_per_cycle, double bytes_per_cycle) {
    return w.is_compute ? w.macs / std::max(1.0, macs_per_cycle)
                        : w.bytes / std::max(1.0, bytes_per_cycle);
}

// ----------------------------------------------------------------------------
// L1-timed duration (systolic cycles), when a StreamProgram is available:
// compute takes its wavefront latency; a Drain is the C stream drained at the C
// signature's element stride, so an output-stationary drain bubble shows up as a
// longer drain; a Feed fills at one element per cycle per lane. Movement timing
// uses the op's ACTUAL (clamped) extent, not the nominal signature shape.
//
// Returns `fallback` for anything the L1 program does not describe.
// ----------------------------------------------------------------------------
inline double l1_duration(const TileProgram& prog, const TileOp& op, std::size_t idx,
                          double fallback, const stream::StreamProgram& l1) {
    if (op.kind == TileOpKind::MatMulAccum) {
        auto it = l1.computes.find(idx);
        return it != l1.computes.end() ? static_cast<double>(it->second.latency()) : fallback;
    }
    const TileCoord& tile = op.inputs.empty() ? op.outputs[0] : op.inputs[0];
    Dim rows = 0, cols = 0;
    tile_extent(prog, tile, rows, cols);
    const double elements = static_cast<double>(rows) * cols;
    if (op.kind == TileOpKind::Drain) {
        if (const auto* c = l1.signature("C")) {
            const double lanes = c->lanes > 0 ? static_cast<double>(c->lanes)
                               : (cols > 0 ? static_cast<double>(cols) : 1.0);
            return c->element_stride * (elements / lanes);   // bubble stretches the drain
        }
    } else if (op.kind == TileOpKind::Feed) {
        // A stationary operand still preloads into the array (lanes == 0), so cost it
        // over its rows rather than falling back to the byte model.
        if (const auto* s = l1.signature(tile.operand)) {
            const double lanes = s->lanes > 0 ? static_cast<double>(s->lanes)
                               : (rows > 0 ? static_cast<double>(rows) : 1.0);
            return elements / lanes;
        }
    }
    return fallback;
}

// ----------------------------------------------------------------------------
// Cycle quantization (design note §7.1, normative).
//
// `Cycle` is an unsigned integer while the cost models produce doubles, so:
// round UP, and give any op with non-zero work at least one cycle. Truncation
// would let sub-cycle work become 0 and contradict the requirement that
// aggregate compute cycles are positive whenever the run computes anything.
// Genuinely zero work — a resident-tile feed, a degenerate extent — costs 0.
// ----------------------------------------------------------------------------
inline std::uint64_t quantize_cycles(double duration, bool has_work) {
    if (!has_work) return 0;
    const double up = std::ceil(duration);
    return up < 1.0 ? 1u : static_cast<std::uint64_t>(up);
}

} // namespace sw::kpu::program
