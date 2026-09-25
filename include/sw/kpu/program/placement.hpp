// ============================================================================
// include/sw/kpu/program/placement.hpp
// Where an L0 program's work runs — the driver JIT's output, modelled as a real
// object from the start.
//
// The spatial layout is deliberately NOT in L0 (D6 §4a): one device-independent
// TileProgram runs on a single L3/CF set, on a NEWS arrangement, or on a
// checkerboard, and only the JIT mapping differs. Until the placement pass
// exists (#230 increment 3), `Placement::single(device)` supplies the default
// mapping, so when the JIT arrives it replaces a VALUE rather than forcing every
// signature to change (decided on #269).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program {

// ----------------------------------------------------------------------------
// Placement — compute-tile assignment for each op.
//
// Two modes, both real:
//   - UNPINNED: the device's compute tiles are interchangeable and the executor
//     takes the lowest-numbered free one when an op fires. This is what a device
//     with no placement decision behaves like, and it is the default until the
//     JIT's placement pass lands.
//   - PINNED: op -> compute tile, fixed ahead of time. This is the shape the JIT
//     emits, including for checkerboard topologies where *which* tile matters
//     because inter-tile reuse becomes an on-chip move.
// ----------------------------------------------------------------------------
class Placement {
public:
    enum class Mode { Unpinned, Pinned };

    // Default for a device: unpinned over its compute tiles.
    static Placement single(Dim compute_tiles) {
        Placement p;
        p.mode_ = Mode::Unpinned;
        p.compute_tiles_ = compute_tiles ? compute_tiles : 1;
        return p;
    }

    // A JIT-style fixed assignment: op index -> compute tile id.
    static Placement pinned(std::vector<Dim> op_to_compute_tile, Dim compute_tiles) {
        Placement p;
        p.mode_ = Mode::Pinned;
        p.compute_tiles_ = compute_tiles ? compute_tiles : 1;
        p.op_to_cf_ = std::move(op_to_compute_tile);
        for (Dim t : p.op_to_cf_)
            if (t >= p.compute_tiles_)
                throw std::invalid_argument(
                    "Placement: compute tile " + std::to_string(t) +
                    " out of range for a device with " + std::to_string(p.compute_tiles_) +
                    " compute tiles");
        return p;
    }

    Mode mode() const { return mode_; }
    bool is_pinned() const { return mode_ == Mode::Pinned; }
    Dim compute_tiles() const { return compute_tiles_; }

    // The tile this op must run on, for a pinned placement. Unpinned placements
    // leave the choice to the executor, which is why this returns a flag rather
    // than inventing an assignment.
    bool compute_tile_for(std::size_t op, Dim& tile) const {
        if (mode_ != Mode::Pinned || op >= op_to_cf_.size()) return false;
        tile = op_to_cf_[op];
        return true;
    }

    // Placements are compared and reported, so runs stay attributable: a makespan
    // without its placement cannot be compared with another makespan.
    std::string label() const {
        return (mode_ == Mode::Pinned ? "pinned/cf" : "unpinned/cf") +
               std::to_string(compute_tiles_);
    }

    // The WHOLE placement, for a run identity. label() is for a human and is deliberately
    // short, which makes it ambiguous here: two DIFFERENT pinned assignments over the same
    // compute tiles share a label, and a run identity built from it would call two different
    // runs the same run. The assignment itself is the thing that changes scheduling, so it
    // is what gets recorded.
    std::string canonical_bytes() const {
        std::string out = (mode_ == Mode::Pinned ? "pinned " : "unpinned ");
        out += std::to_string(compute_tiles_);
        for (Dim t : op_to_cf_) out += " " + std::to_string(t);
        return out;
    }

private:
    Mode mode_ = Mode::Unpinned;
    Dim compute_tiles_ = 1;
    std::vector<Dim> op_to_cf_;
};

} // namespace sw::kpu::program
