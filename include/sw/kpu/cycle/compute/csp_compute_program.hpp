#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/csp_compute_program.hpp
// A CSP program's compute traffic as the BlockMover's, the L2's and the Streamer's programs (#546;
// docs/plans/cycle-based-kpu-prototype.md §5.2, step 2e): the whole path L3 -> L2 -> L1 -> array ->
// Combiner -> L2 -> L3, orchestrated by the walker's actions.
//
// FROM THE WALKER'S ACTIONS (csp::CspProgram::actions, in the program's order), each with the L2 slot
// the walker placed it in (slot-credits-at-lca.md: every copy has its slot):
//   Move       -> an L2 Move of the tile's lanes from its residency's L3 slot into its L2 slot
//   Feed       -> an L2 Feed of the tile's lanes: one operand of the Call it feeds
//   Call       -> a Streamer command: A and B the Call's inputs (fed since the previous Call), K the
//                 tiles' inner dimension. The chain's first call (the one that opens its result's L2
//                 slot) marks its elements `first`, its last (the call before the chain's Drain, or
//                 before the next chain's first call) marks them `last` and names the result tile:
//                 the RTL's `last` launches the results north at once. A Drain the program defers
//                 (`store ... late`) is the same L2 Result, placed in the program's order
//   Drain      -> an L2 Result: the Combiner's rows of the result tile into its L2 slot
//   Writeback  -> an L2 Writeback of the result's rows into its residency's L3 slot
//   L2Release  -> an L2 Release
// The L3 actions (Load, Store, Release, Inherit, Retain) are the memory subsystem's (Phase 1). Here
// THE L3 STARTS WITH A AND B IN THE SLOTS THE PROGRAM PLACES THEM IN (plan §5.2: "the same state
// loaded directly"): each Load's or Inherit's tile is in its slot from the start. So a program whose
// L3 slot holds two residencies over the run is refused: that needs the Loads run in time, which
// the join does (step 3).
//
// THE LAYOUT. L3 holds each tile as the memory subsystem places it (csp_l3_program.hpp, Phase 1):
// densely, row-major at its element's width, little-endian, padded to whole 64-byte beats, from
// its slot's address (slot * slot_bytes, the largest padded tile). L2's data is pre-arranged, one
// row a lane (docs/01-architecture/memory-hierarchy-roles.md), so the BlockMover transforms each
// tile on its way into L2 (#548; block_mover.hpp's dense layouts): an A tile (h x K) into h lanes,
// its rows; a B tile (K x w) into w lanes, its columns (the transpose); and on the way back a C
// tile's h rows of w int32 into its dense tile in L3.
//
// THE ORACLE: L-B's result. `outputs` holds, for every Writeback, the bytes its L3 slot must hold
// after the run: L-B's C tile, densely packed, int32 exact -- the bytes Phase 1's plan fills it with.
//
// WHAT IS REFUSED, before any cycle is run (std::invalid_argument, naming the action): a Call other
// than an int8 x int8 -> int32 OS gemm (alpha 1), an L2 accumulator's merge, an accumulator's init
// stream, a stage (transform or epilogue) on an action, an unplaced copy, a chain that opens before the
// last one on its tile drained, a Call reading A and B from one L2 slot, a Call whose input was not fed since the previous Call, and an L3 slot
// that holds two residencies.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/numeric.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sw::kpu::cycle {

struct CspComputePlan {
    // One L3 slot's bytes: before the run (an input) or after it (a result).
    struct Image {
        sw::kpu::program::TileCoord tile;
        std::uint32_t l3_slot = 0;
        std::uint64_t address = 0;
        std::vector<std::uint8_t> bytes;   // the tile, densely packed and padded to 64-byte beats
    };
    L2Program l2;
    StreamerProgram commands;
    // The geometry the programs need: the L2's slots (the program's `l2 = N`), its rows per slot (the
    // most lanes of a tile) and its row (the most bytes of a lane, a multiple of `row_align`).
    std::uint32_t l2_slots = 0, rows_per_slot = 0, row_bytes = 0;
    std::uint32_t l3_slots = 0;
    std::uint64_t slot_bytes = 0;          // an L3 slot: the largest tile, densely packed and padded to beats
    std::uint32_t max_h = 0, max_w = 0;    // the most lanes a command uses: A's, B's
    std::vector<Image> inputs;             // L3 before the run
    std::vector<Image> outputs;            // L3 after it: each written-back tile, L-B's
    std::size_t moves = 0, feeds = 0, calls = 0, drains = 0, writebacks = 0, l2_releases = 0;
};

namespace csp_compute_detail {

using sw::kpu::program::Dim;
using sw::kpu::program::TensorOperand;
using sw::kpu::program::TileCoord;
using Action = sw::kpu::program::csp::Action;
using Kind = Action::Kind;

inline unsigned element_bytes(const TensorOperand& op) {
    const auto* ns = sw::kpu::program::numeric::find(op.type);
    if (!ns) throw std::invalid_argument("plan_compute: operand " + op.name + " has an unknown number system " + op.type);
    return (ns->bits + 7) / 8;
}

inline bool integer(const TensorOperand& op) {
    return sw::kpu::program::numeric::find(op.type)->kind == sw::kpu::program::numeric::Kind::Integer;
}

inline std::tuple<std::string, Dim, Dim> key(const TileCoord& t) { return {t.operand, t.ti, t.tj}; }

struct Shape {
    std::uint32_t rows = 0, cols = 0;
};
inline Shape shape(const TensorOperand& op, Dim ti, Dim tj) {
    return {static_cast<std::uint32_t>(op.row_end(ti) - op.row_begin(ti)),
            static_cast<std::uint32_t>(op.col_end(tj) - op.col_begin(tj))};
}

inline sw::kpu::timing::TileID tile_id(char role, const TileCoord& t) {
    const auto m = role == 'A' ? sw::kpu::isa::MatrixID::A : role == 'B' ? sw::kpu::isa::MatrixID::B
                                                                         : sw::kpu::isa::MatrixID::C;
    return sw::kpu::timing::TileID{m, static_cast<sw::kpu::timing::Size>(t.ti), static_cast<sw::kpu::timing::Size>(t.tj), 0};
}

}  // namespace csp_compute_detail

// `inputs`: every operand as it is before the run; `result`: as L-B leaves it after. `row_align`: the
// L2's row is a multiple of it (the BlockMover's bus).
inline CspComputePlan plan_compute(const sw::kpu::program::csp::CspProgram& p,
                                   const sw::kpu::program::TileProgram& inputs,
                                   const sw::kpu::program::TileProgram& result, std::uint32_t row_align = 16) {
    using namespace csp_compute_detail;
    CspComputePlan plan;
    auto at = [&](std::size_t i, const std::string& what) {
        return std::invalid_argument("plan_compute: action " + std::to_string(i) + " (" +
                                     sw::kpu::program::csp::to_string(p.actions[i].kind) + " " +
                                     p.actions[i].tile.to_string() + ") " + what);
    };
    if (row_align == 0) throw std::invalid_argument("plan_compute: the row alignment must be positive");
    plan.l2_slots = p.l2_slots;
    if (plan.l2_slots == 0) throw std::invalid_argument("plan_compute: the program declares no L2 slots (`l2 = N`)");

    // The roles, from the Calls: each operand is A (a gemm's first input), B (its second) or C (its
    // output), and only one. The geometry, from the tiles.
    std::map<std::string, char> role;
    auto assign = [&](std::size_t i, const std::string& operand, char r) {
        const auto it = role.find(operand);
        if (it != role.end() && it->second != r)
            throw at(i, ": operand " + operand + " is both " + std::string(1, it->second) + " and " + std::string(1, r));
        role[operand] = r;
    };
    std::uint32_t lane_bytes = 0;
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        if (!a.context.empty()) throw at(i, "carries a stage (a transform or an epilogue): not built in step 2e");
        if (a.merge != sw::kpu::program::csp::Merge::None)
            throw at(i, "merges into an L2 accumulator (COMPUTE_ACCUMULATE): not built");
        if (a.seeds) throw at(i, "seeds an accumulator (an init stream): not built in step 2e");
        if (a.kind != Kind::Call) continue;
        const auto& op = p.source.ops().at(a.l0_op);
        if (op.kind != sw::kpu::program::TileOpKind::MatMulAccum || op.alpha != 1.0f || op.inputs.size() != 2 ||
            op.outputs.size() != 1)
            throw at(i, "is not a gemm C += A . B");
        if (a.dataflow == sw::kpu::program::csp::Dataflow::WS || a.dataflow == sw::kpu::program::csp::Dataflow::IS)
            throw at(i, "is not output-stationary");
        assign(i, op.inputs[0].operand, 'A');
        assign(i, op.inputs[1].operand, 'B');
        assign(i, op.outputs[0].operand, 'C');
        const TensorOperand& A = inputs.operand(op.inputs[0].operand);
        const TensorOperand& B = inputs.operand(op.inputs[1].operand);
        const TensorOperand& C = result.operand(op.outputs[0].operand);
        if (element_bytes(A) != 1 || element_bytes(B) != 1 || element_bytes(C) != 4 ||
            !integer(A) || !integer(B) || !integer(C))
            throw at(i, "is not int8 x int8 -> int32 (" + A.type + ", " + B.type + " -> " + C.type + ")");
        const Shape sa = shape(A, op.inputs[0].ti, op.inputs[0].tj), sb = shape(B, op.inputs[1].ti, op.inputs[1].tj),
                    sc = shape(C, op.outputs[0].ti, op.outputs[0].tj);
        if (sa.cols != sb.rows || sc.rows != sa.rows || sc.cols != sb.cols)
            throw at(i, ": its tiles' shapes do not compose");
        const auto dense = [](std::uint64_t n) { return (n + kL3BeatBytes - 1) / kL3BeatBytes * kL3BeatBytes; };
        plan.slot_bytes = std::max({plan.slot_bytes, dense(std::uint64_t(sa.rows) * sa.cols),
                                    dense(std::uint64_t(sb.rows) * sb.cols), dense(std::uint64_t(sc.rows) * sc.cols * 4)});
        plan.max_h = std::max(plan.max_h, sa.rows);
        plan.max_w = std::max(plan.max_w, sb.cols);
        plan.rows_per_slot = std::max({plan.rows_per_slot, sa.rows, sb.cols});
        lane_bytes = std::max({lane_bytes, sa.cols, sc.cols * 4});
    }
    if (plan.rows_per_slot == 0) throw std::invalid_argument("plan_compute: the program has no Call");
    plan.row_bytes = (lane_bytes + row_align - 1) / row_align * row_align;
    // A slot also holds the largest tile whole, padded to its beats: the BlockMover's tile buffer is
    // one slot (block_mover.hpp). A tile of few lanes (a 1 x 1 gemm) needs more rows than lanes.
    plan.rows_per_slot = std::max<std::uint32_t>(plan.rows_per_slot,
                                                 static_cast<std::uint32_t>((plan.slot_bytes + plan.row_bytes - 1) / plan.row_bytes));
    plan.l3_slots = static_cast<std::uint32_t>(p.channel(sw::kpu::program::csp::Chan::L3).capacity);

    auto role_of = [&](std::size_t i, const TileCoord& t) {
        const auto it = role.find(t.operand);
        if (it == role.end()) throw at(i, ": operand " + t.operand + " is no gemm's");
        return it->second;
    };
    auto rows_of = [&](char r, const TileCoord& t) {
        const TensorOperand& op = r == 'C' ? result.operand(t.operand) : inputs.operand(t.operand);
        const Shape s = shape(op, t.ti, t.tj);
        return r == 'B' ? s.cols : s.rows;
    };
    auto l2_slot = [&](std::size_t i) {
        const Action& a = p.actions[i];
        if (!a.l2) throw at(i, "has no L2 slot (an unplaced copy)");
        if (a.l2->cf != 0) throw at(i, "is placed in compute tile " + std::to_string(a.l2->cf) + "; step 2e has one");
        if (a.l2->slot >= plan.l2_slots)
            throw at(i, "is placed in L2 slot " + std::to_string(a.l2->slot) + " of " + std::to_string(plan.l2_slots));
        return a.l2->slot;
    };
    auto l3_slot = [&](std::size_t i) {
        const Action& a = p.actions[i];
        if (!a.place) throw at(i, "has no L3 address");
        if (a.place->l3_tile != 0) throw at(i, "addresses L3 module " + std::to_string(a.place->l3_tile) + "; step 2e has one");
        return a.place->slot;
    };

    // The L3 slots each hold one residency over the run (the L3 starts with them all in place).
    std::map<std::uint32_t, std::size_t> owner;   // L3 slot -> the action that opened its residency
    // The chain in the fabric: its output tile, the slot its first call opened, its calls so far.
    struct Chain {
        TileCoord tile;
        std::uint32_t slot = 0;
        std::size_t last_call = 0;   // index into plan.commands
    };
    std::optional<Chain> chain;
    // A chain ends at its last call: the RTL's `last` launches its results north at once, and the
    // Combiner writes them into the slot its first call opened. A Drain the program places later
    // (`store ... late`, after the next chain's first call) is that same L2 Result, in its order.
    std::map<std::tuple<std::string, Dim, Dim>, std::uint32_t> undrained;   // ended chains: tile -> result slot
    auto close = [&](const Chain& ch) {
        StreamerCommand& last = plan.commands[ch.last_call];
        last.last = true;
        last.result = StreamerCommand::Result{ch.slot, tile_id('C', ch.tile), {}};
        undrained[key(ch.tile)] = ch.slot;
    };
    std::map<std::tuple<std::string, Dim, Dim>, std::uint32_t> fed;   // since the previous Call: tile -> L2 slot

    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        switch (a.kind) {
            case Kind::Load:
            case Kind::Inherit:
            case Kind::Writeback: {
                const std::uint32_t s = l3_slot(i);
                if (s >= plan.l3_slots && plan.l3_slots) throw at(i, "names L3 slot " + std::to_string(s));
                const auto [it, fresh] = owner.emplace(s, i);
                if (!fresh)
                    throw at(i, "opens a second residency in L3 slot " + std::to_string(s) + " (action " +
                                    std::to_string(it->second) +
                                    " opened the first): step 2e starts with every tile in place, so each slot "
                                    "holds one residency over the run; the Loads run in time at the join (step 3)");
                const char r = role_of(i, a.tile);
                const std::uint64_t address = std::uint64_t(s) * plan.slot_bytes;
                if (a.kind == Kind::Writeback) {
                    if (r != 'C') throw at(i, "writes back an operand the gemm reads");
                    const std::uint32_t slot = l2_slot(i);
                    const Shape sc = shape(result.operand(a.tile.operand), a.tile.ti, a.tile.tj);
                    plan.l2.push_back(l2_writeback_dense(slot, tile_id('C', a.tile), address, sc.rows, sc.cols, 4));
                    ++plan.writebacks;
                    plan.outputs.push_back(
                        {a.tile, s, address, csp_l3_detail::pack(result.operand(a.tile.operand), a.tile.ti, a.tile.tj)});
                } else {
                    if (r == 'C') throw at(i, "loads the gemm's result: an accumulate is not built in step 2e");
                    plan.inputs.push_back(
                        {a.tile, s, address, csp_l3_detail::pack(inputs.operand(a.tile.operand), a.tile.ti, a.tile.tj)});
                }
                break;
            }
            case Kind::Store:
            case Kind::Release:
            case Kind::Retain:
                break;   // the memory subsystem's
            case Kind::Move: {
                const char r = role_of(i, a.tile);
                if (r == 'C') throw at(i, "moves the gemm's result into L2: an accumulate is not built in step 2e");
                const std::uint32_t s = l3_slot(i);
                const auto it = owner.find(s);
                if (it == owner.end() || p.actions[it->second].tile.to_string() != a.tile.to_string())
                    throw at(i, "reads L3 slot " + std::to_string(s) + ", which does not hold its tile");
                const Shape st = shape(inputs.operand(a.tile.operand), a.tile.ti, a.tile.tj);
                plan.l2.push_back(l2_move_dense(l2_slot(i), tile_id(r, a.tile), std::uint64_t(s) * plan.slot_bytes,
                                                st.rows, st.cols, 1, r == 'B'));
                ++plan.moves;
                break;
            }
            case Kind::Feed: {
                const char r = role_of(i, a.tile);
                const std::uint32_t slot = l2_slot(i);
                plan.l2.push_back(l2_feed(slot, tile_id(r, a.tile), rows_of(r, a.tile)));
                fed[key(a.tile)] = slot;
                ++plan.feeds;
                break;
            }
            case Kind::L2Release: {
                plan.l2.push_back(l2_release(l2_slot(i), tile_id(role_of(i, a.tile), a.tile)));
                ++plan.l2_releases;
                break;
            }
            case Kind::Call: {
                const auto& op = p.source.ops().at(a.l0_op);
                const TileCoord &ta = op.inputs[0], &tb = op.inputs[1], &tc = op.outputs[0];
                if (a.result_l2) {
                    if (chain) close(*chain);   // the open chain's last call was the one before
                    if (a.result_l2->cf != 0 || a.result_l2->slot >= plan.l2_slots)
                        throw at(i, "opens result slot " + a.result_l2->to_string());
                    if (undrained.count(key(tc))) throw at(i, "opens a chain on " + tc.to_string() + " before its last one's Drain");
                    chain = Chain{tc, a.result_l2->slot, 0};
                } else if (!chain || chain->tile.to_string() != tc.to_string()) {
                    throw at(i, "continues a chain on " + tc.to_string() + " that no call opened");
                }
                const auto fa = fed.find(key(ta)), fb = fed.find(key(tb));
                if (fa == fed.end() || fb == fed.end())
                    throw at(i, "reads " + (fa == fed.end() ? ta : tb).to_string() + ", which was not fed since the previous call");
                if (fa->second == fb->second)
                    throw at(i, "reads A and B from one L2 slot (" + std::to_string(fa->second) +
                                    "): the Streamer reads both operands' rows in the same clocks, so each needs a "
                                    "slot of its own (the program's `l2 = N` leaves too few)");
                const TensorOperand& A = inputs.operand(ta.operand);
                StreamerCommand c{{fa->second, tile_id('A', ta), 0, rows_of('A', ta)},
                                  {fb->second, tile_id('B', tb), 0, rows_of('B', tb)},
                                  shape(A, ta.ti, ta.tj).cols,
                                  a.result_l2.has_value(),
                                  false,
                                  std::nullopt};
                chain->last_call = plan.commands.size();
                plan.commands.push_back(c);
                fed.clear();
                ++plan.calls;
                break;
            }
            case Kind::Drain: {
                if (chain && chain->tile.to_string() == a.tile.to_string()) {
                    close(*chain);
                    chain.reset();
                }
                const auto it = undrained.find(key(a.tile));
                if (it == undrained.end()) throw at(i, "drains a tile no chain computed");
                const std::uint32_t slot = l2_slot(i);
                if (slot != it->second)
                    throw at(i, "drains into L2 slot " + std::to_string(slot) + ", but its chain opened slot " +
                                    std::to_string(it->second));
                undrained.erase(it);
                plan.l2.push_back(l2_result(slot, tile_id('C', a.tile), rows_of('C', a.tile)));
                ++plan.drains;
                break;
            }
        }
    }
    if (chain) close(*chain);
    if (!undrained.empty())
        throw std::invalid_argument("plan_compute: the chain on " + std::get<0>(undrained.begin()->first) + "[" +
                                    std::to_string(std::get<1>(undrained.begin()->first)) + "," +
                                    std::to_string(std::get<2>(undrained.begin()->first)) + "] is never drained");
    if (plan.l3_slots == 0)
        for (const auto& [s, i] : owner) plan.l3_slots = std::max(plan.l3_slots, s + 1);
    validate_l2_program(plan.l2, plan.l2_slots, plan.rows_per_slot, plan.row_bytes, "plan_compute");
    return plan;
}

}  // namespace sw::kpu::cycle
