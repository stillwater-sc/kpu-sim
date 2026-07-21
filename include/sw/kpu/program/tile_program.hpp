// ============================================================================
// include/sw/kpu/program/tile_program.hpp
// L0 tile-sequence program — the device-independent "outer loop" of a tiled
// linear-algebra operator.
//
// This is the *portable program* layer (D6, docs/plans/kpu-program-model.md):
// the ordered sequence of tiles pushed into / drained from the fabric's logical
// ports plus the tile-level compute over them. It carries NO engine/bank ids and
// NO stream/timing information — those belong to L1 (stream signatures) and the
// driver JIT (data-path config). Because processing the tile sequence with the
// operator's tile-level compute produces the correct numeric result, L0 alone is
// a *pure functional reference* (see tile_program_reference.hpp).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program {

// Logical dimension / index / tile-count. Device-independent.
using Dim = std::uint32_t;

// ============================================================================
// TensorOperand — a logical 2D tensor partitioned into a grid of tiles.
//
// Holds the full row-major value buffer used by the functional reference. The
// tiling defines the *outer loop* structure (which tiles the program sequences);
// trailing tiles that do not divide evenly are clamped to the operand extent.
// ============================================================================
struct TensorOperand {
    std::string name;                 // "A", "B", "C", "L", "U", ...
    Dim rows = 0, cols = 0;           // logical shape
    Dim tile_rows = 0, tile_cols = 0; // tile shape
    std::vector<float> values;        // rows*cols, row-major (0-initialized)

    TensorOperand() = default;
    TensorOperand(std::string n, Dim r, Dim c, Dim tr, Dim tc)
        : name(std::move(n)), rows(r), cols(c), tile_rows(tr), tile_cols(tc),
          values(static_cast<std::size_t>(r) * c, 0.0f) {}

    Dim n_tile_rows() const { return tile_rows ? (rows + tile_rows - 1) / tile_rows : 0; }
    Dim n_tile_cols() const { return tile_cols ? (cols + tile_cols - 1) / tile_cols : 0; }

    // Row-major element access.
    float& at(Dim r, Dim c) { return values[static_cast<std::size_t>(r) * cols + c]; }
    float  at(Dim r, Dim c) const { return values[static_cast<std::size_t>(r) * cols + c]; }

    // Half-open tile extents, clamped for trailing tiles.
    Dim row_begin(Dim ti) const { return ti * tile_rows; }
    Dim row_end(Dim ti)   const { return std::min(rows, (ti + 1) * tile_rows); }
    Dim col_begin(Dim tj) const { return tj * tile_cols; }
    Dim col_end(Dim tj)   const { return std::min(cols, (tj + 1) * tile_cols); }
};

// ============================================================================
// TileCoord — names a tile within an operand (operand, tile-row, tile-col).
// Deliberately string-keyed on the operand so the representation is not tied to
// a fixed A/B/C matrix enum and can express any linear-algebra operator.
// ============================================================================
struct TileCoord {
    std::string operand;
    Dim ti = 0, tj = 0;
    std::string to_string() const {
        return operand + "[" + std::to_string(ti) + "," + std::to_string(tj) + "]";
    }
};

// Logical fabric port direction (no engine/bank id — that is JIT output).
enum class PortKind { Input, Output };

// ----------------------------------------------------------------------------
// TileOpKind — the tile-level compute kinds L0 expresses. These are opaque,
// device-independent kernels; the tile *sequence* over them is the program.
// Grows as more operators are covered.
// ----------------------------------------------------------------------------
enum class TileOpKind {
    Feed,          // inject an input tile into a logical port (structural; L1 attaches streams)
    Drain,         // extract a result tile from a logical port (structural)
    MatMulAccum,   // out += alpha * (A_tile . B_tile)  — the systolic MAC; also the LU trailing update
    LuPanelFactor, // factor column panel k in place w/ neighbor (pairwise) pivoting -> unit-L + U + pivots
    TrsmLeft,      // solve L_kk . X = B in place (forward-substitution; unit-lower-triangular L on the left)
    PivotApply,    // replay a recorded neighbor-pivot row permutation onto a trailing tile-column
};

inline const char* to_string(TileOpKind k) {
    switch (k) {
        case TileOpKind::Feed:          return "FEED";
        case TileOpKind::Drain:         return "DRAIN";
        case TileOpKind::MatMulAccum:   return "MATMUL_ACCUM";
        case TileOpKind::LuPanelFactor: return "LU_PANEL_FACTOR";
        case TileOpKind::TrsmLeft:      return "TRSM_LEFT";
        case TileOpKind::PivotApply:    return "PIVOT_APPLY";
    }
    return "?";
}

// ----------------------------------------------------------------------------
// TileOp — one step of the outer loop.
// ----------------------------------------------------------------------------
struct TileOp {
    TileOpKind kind;
    std::vector<TileCoord> inputs;   // tiles read
    std::vector<TileCoord> outputs;  // tiles written (in place for the LA kernels)

    // MatMulAccum: out += alpha * (A . B). alpha = -1 gives the LU trailing update.
    float alpha = 1.0f;

    // Feed/Drain: which logical port the tile streams through.
    PortKind port_kind = PortKind::Input;
    std::string port;

    // LU pivot dataflow: LuPanelFactor *writes* this slot, PivotApply *reads* it.
    // This is the data-dependent control matmul does not have — a pivot decision
    // in one tile op flowing to row swaps in trailing tiles.
    int pivot_slot = -1;

    std::string label;               // human-readable note for disassembly
};

// ============================================================================
// TileProgram — an ordered list of TileOps over a registry of tiled operands.
// Device-independent: parameterized purely by shapes + tiling.
// ============================================================================
class TileProgram {
public:
    explicit TileProgram(std::string name = "") : name_(std::move(name)) {}

    const std::string& name() const { return name_; }

    // ---- operands -----------------------------------------------------------
    TensorOperand& add_operand(TensorOperand op) {
        const std::string key = op.name;
        auto [it, inserted] = operands_.emplace(key, std::move(op));
        if (!inserted) throw std::invalid_argument("TileProgram: duplicate operand '" + key + "'");
        order_.push_back(key);
        return it->second;
    }
    bool has_operand(const std::string& n) const { return operands_.count(n) != 0; }
    TensorOperand& operand(const std::string& n) {
        auto it = operands_.find(n);
        if (it == operands_.end()) throw std::invalid_argument("TileProgram: no operand '" + n + "'");
        return it->second;
    }
    const TensorOperand& operand(const std::string& n) const {
        auto it = operands_.find(n);
        if (it == operands_.end()) throw std::invalid_argument("TileProgram: no operand '" + n + "'");
        return it->second;
    }
    const std::vector<std::string>& operand_order() const { return order_; }

    // ---- ops ----------------------------------------------------------------
    void push(TileOp op) { ops_.push_back(std::move(op)); }
    const std::vector<TileOp>& ops() const { return ops_; }
    std::vector<TileOp>& ops() { return ops_; }

    std::size_t count(TileOpKind k) const {
        std::size_t n = 0;
        for (const auto& op : ops_) if (op.kind == k) ++n;
        return n;
    }

    // Human-readable listing of the tile sequence (foreshadows the L1/JIT
    // disassembler; useful for tests and demos).
    std::string disassemble() const;

private:
    std::string name_;
    std::map<std::string, TensorOperand> operands_;
    std::vector<std::string> order_;   // operand declaration order (stable listing)
    std::vector<TileOp> ops_;
};

// ---- disassembly -----------------------------------------------------------
inline std::string TileProgram::disassemble() const {
    std::string s = "TileProgram \"" + name_ + "\"\n";
    s += "  operands:\n";
    for (const auto& key : order_) {
        const auto& op = operands_.at(key);
        s += "    " + op.name + " : " + std::to_string(op.rows) + "x" + std::to_string(op.cols) +
             " tiled " + std::to_string(op.tile_rows) + "x" + std::to_string(op.tile_cols) +
             " (" + std::to_string(op.n_tile_rows()) + "x" + std::to_string(op.n_tile_cols()) + " tiles)\n";
    }
    s += "  ops (" + std::to_string(ops_.size()) + "):\n";
    std::size_t idx = 0;
    for (const auto& op : ops_) {
        s += "    " + std::to_string(idx++) + ": " + to_string(op.kind);
        if (op.kind == TileOpKind::Feed || op.kind == TileOpKind::Drain) {
            s += " " + (op.inputs.empty() ? (op.outputs.empty() ? std::string("?")
                                                                : op.outputs[0].to_string())
                                          : op.inputs[0].to_string());
            s += (op.port_kind == PortKind::Input ? " -> in:" : " -> out:") + op.port;
        } else {
            if (!op.outputs.empty()) s += " " + op.outputs[0].to_string();
            if (!op.inputs.empty()) {
                s += " <-";
                for (const auto& in : op.inputs) s += " " + in.to_string();
            }
            if (op.kind == TileOpKind::MatMulAccum && op.alpha != 1.0f)
                s += " (alpha=" + std::to_string(op.alpha) + ")";
            if (op.pivot_slot >= 0) s += " {pivot#" + std::to_string(op.pivot_slot) + "}";
        }
        if (!op.label.empty()) s += "  ; " + op.label;
        s += "\n";
    }
    return s;
}

} // namespace sw::kpu::program
