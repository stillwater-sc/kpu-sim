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
// The arithmetic itself lives in tile_kernels.hpp, shared with the transactional
// tier: this class is now just the IN-ORDER driver over those kernels plus the
// run summary. A dataflow-order driver over the same kernels is bit-identical by
// construction (ADR 0001 D3.1) — which is why the kernels are not duplicated
// here.
//
// It also replays PLACEMENT (opset 1.2; ADR 0004 §5): a trace's L3 transactions run on an
// addressed L3 of tags, and a write into a live slot, a read of a slot holding another tile, or a
// release of a free one throws PlacementError, as L-B refuses them. A trace without L3 ops (opset
// 1.1 and older) replays as before.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_kernels.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program {

// A trace whose L3 transactions break placement: a write into a live slot, a read of a slot that
// holds another tile, a release of a free slot, or a program that ends with a slot neither
// released nor retained. The message names the op, the slot and the tiles.
class PlacementError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ============================================================================
// TileProgramReference — walks the ops in order, mutating operand buffers.
// ============================================================================
class TileProgramReference {
public:
    struct RunSummary {
        std::size_t ops = 0;
        std::size_t feeds = 0;
        std::size_t drains = 0;
        std::size_t computes = 0;   // MatMulAccum (GEMM)
        std::size_t diag_factors = 0;   // LuDiagFactor (GETRF)
        std::size_t trsms = 0;          // TrsmLowerLeft + TrsmUpperRight
        std::size_t pivot_applies = 0;  // PivotApply (LASWP)
        std::size_t epilogues = 0;      // BiasAdd + Activation
        std::size_t l3_ops = 0;         // the L3 transactions (opset 1.2)
        std::size_t l2_ops = 0;         // the L2 transactions (opset 1.3)
        std::size_t peak_l2 = 0;        // L2 slots staged at once, as replayed
        std::size_t peak_l3 = 0;        // L3 slots tagged at once, as replayed
        std::size_t row_swaps = 0;      // total within-tile row swaps performed by GETRF
        // Row permutation produced by pivoting (LU): perm[i] = original row now at
        // position i. Identity when no pivoting occurred. Sized to the largest
        // square operand touched by a LuPanelFactor (0 if none).
        std::vector<Dim> permutation;
    };

    // Execute program in order. Mutates program's operand value buffers.
    RunSummary run(TileProgram& program) {
        state_.clear();   // reset up front so a throw mid-run can't leak stale state
        l3_.clear();
        l2_.clear();
        retained_.clear();
        RunSummary sum;
        for (const auto& op : program.ops()) {
            if (is_l3(op.kind)) {
                place(op, sum.ops);
                ++sum.l3_ops;
                sum.peak_l3 = std::max(sum.peak_l3, l3_.size());
            }
            if (is_l2(op.kind)) {
                stage(op, sum.ops);
                ++sum.l2_ops;
                sum.peak_l2 = std::max(sum.peak_l2, l2_.size());
            }
            ++sum.ops;
            switch (op.kind) {
                case TileOpKind::Feed:           ++sum.feeds; break;   // structural (L1 attaches streams)
                case TileOpKind::Drain:          ++sum.drains; break;  // structural
                case TileOpKind::MatMulAccum:    ++sum.computes; break;
                case TileOpKind::LuDiagFactor:   ++sum.diag_factors; break;
                case TileOpKind::PivotApply:     ++sum.pivot_applies; break;
                case TileOpKind::TrsmLowerLeft:
                case TileOpKind::TrsmUpperRight: ++sum.trsms; break;
                case TileOpKind::BiasAdd:
                case TileOpKind::Activation:     ++sum.epilogues; break;
                case TileOpKind::L3Load:
                case TileOpKind::L3Store:
                case TileOpKind::L3Move:
                case TileOpKind::L3Writeback:
                case TileOpKind::L3Release:
                case TileOpKind::L3Inherit:
                case TileOpKind::L3Retain:       break;   // replayed by place()
                case TileOpKind::L2Stage:
                case TileOpKind::L2Release:      break;   // replayed by stage()
            }
            apply(program, op, state_);   // the shared L0 kernels
        }
        // A trace that records placement ends with L3 holding only what it retained.
        for (const auto& [at, slot] : l3_)
            if (!retained_.count(at))
                throw PlacementError("l0: the trace ends with " + slot.tile + " still resident at " + slot.addr +
                                     " (neither released nor retained)");
        for (const auto& [at, slot] : l2_)
            throw PlacementError("l0: the trace ends with " + slot.tile + " still staged at " + slot.addr);
        sum.row_swaps = state_.swaps_performed;
        sum.permutation = state_.perm;
        return sum;
    }

    // The tiles L3 holds now, by address (after run(): the retained ones).
    std::map<std::string, std::string> l3_contents() const {
        std::map<std::string, std::string> out;
        for (const auto& [at, slot] : l3_) out[slot.addr] = slot.tile;
        return out;
    }

private:
    // Transient pivot state + row permutation, carried between ops by the kernels.
    TileKernelState state_;

    // The addressed L3, as the trace's L3 transactions leave it: (module, slot) -> its tile.
    struct Tag {
        std::string tile;
        std::string addr;
    };
    using Key = std::pair<std::uint32_t, std::uint32_t>;
    std::map<Key, Tag> l3_;
    std::map<Key, bool> retained_;
    std::map<Key, Tag> l2_;                       // (cf, slot) -> the staged tile

    // One L2 transaction, checked as L-B executes it: a stage into a free slot, of a tile staged
    // nowhere else; a release of the slot that holds it.
    void stage(const TileOp& op, std::size_t index) {
        const std::string where = "l0 op " + std::to_string(index) + " " + to_string(op.kind);
        if (op.outputs.size() != 1) throw PlacementError(where + ": an L2 transaction names one tile");
        const std::string t = op.outputs[0].to_string();
        const std::string addr = op.l2.to_string();
        const Key at{op.l2.cf, op.l2.slot};
        auto it = l2_.find(at);
        if (op.kind == TileOpKind::L2Stage) {
            if (it != l2_.end())
                throw PlacementError(where + " " + t + ": " + addr + " holds " + it->second.tile + ", which is live");
            for (const auto& [k, slot] : l2_)
                if (slot.tile == t) throw PlacementError(where + " " + t + ": " + t + " is already staged at " + slot.addr);
            l2_[at] = Tag{t, addr};
            return;
        }
        if (it == l2_.end()) throw PlacementError(where + " " + t + ": " + addr + ": the slot is free");
        if (it->second.tile != t)
            throw PlacementError(where + " " + t + ": " + addr + " holds " + it->second.tile + ", not " + t);
        l2_.erase(it);
    }

    // One L3 transaction, checked as L-B executes it (csp/behavioral.hpp).
    void place(const TileOp& op, std::size_t index) {
        const std::string where = "l0 op " + std::to_string(index) + " " + to_string(op.kind);
        if (op.outputs.size() != 1) throw PlacementError(where + ": an L3 transaction names one tile");
        const std::string t = op.outputs[0].to_string();
        const std::string addr = op.l3.to_string();
        const Key at{op.l3.l3_tile, op.l3.slot};
        auto it = l3_.find(at);
        switch (op.kind) {
            case TileOpKind::L3Load:
            case TileOpKind::L3Inherit:
            case TileOpKind::L3Writeback:
                if (it == l3_.end()) {
                    for (const auto& [k, slot] : l3_)
                        if (slot.tile == t)
                            throw PlacementError(where + " " + t + ": " + addr + ": " + t + " is already resident at " +
                                                 slot.addr);
                    l3_[at] = Tag{t, addr};
                } else if (op.kind != TileOpKind::L3Writeback || it->second.tile != t) {
                    throw PlacementError(where + " " + t + ": " + addr + " holds " + it->second.tile +
                                         ", which is live");
                } else if (it->second.addr != addr) {   // the whole address, bank included
                    throw PlacementError(where + " " + t + ": " + addr + " is not " + t + "'s address, " +
                                         it->second.addr);
                }                                       // else: a result rewritten in its own slot
                break;
            default:                                    // reads and frees: the slot holds the tile
                if (it == l3_.end()) throw PlacementError(where + " " + t + ": " + addr + ": the slot is free");
                if (it->second.tile != t)
                    throw PlacementError(where + " " + t + ": " + addr + " holds " + it->second.tile + ", not " + t);
                if (it->second.addr != addr)            // the whole address, bank included
                    throw PlacementError(where + " " + t + ": " + addr + " is not " + t + "'s address, " +
                                         it->second.addr);
                if (op.kind == TileOpKind::L3Release) {
                    if (retained_.count(at)) throw PlacementError(where + " " + t + ": " + addr + " is retained");
                    l3_.erase(it);
                } else if (op.kind == TileOpKind::L3Retain) {
                    retained_[at] = true;
                }
                break;
        }
    }
};

} // namespace sw::kpu::program
