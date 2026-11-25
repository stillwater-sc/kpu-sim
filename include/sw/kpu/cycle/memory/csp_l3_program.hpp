#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/csp_l3_program.hpp
// A CSP program's L3 traffic as the DMA engine's and the L3 tile's program (#540;
// docs/plans/cycle-based-kpu-prototype.md §4.2, rung 4).
//
// FROM THE WALKER'S L3 ACTIONS (csp::CspProgram::actions, in the program's order), each with its
// residency's L3 address (ADR 0004 §5: the walker places every residency once; no level allocates
// a slot of its own):
//   Load       -> Load into its slot, from the tile's DRAM address
//   Store      -> Store from its slot, to the tile's DRAM address
//   Release    -> Release of its slot
//   Writeback  -> Fill of its slot with L-B's tile: THE HYPOTHESIZED C. Phase 1 has no compute
//                 subsystem, so what the array would write back into L3 is L-B's result, placed at
//                 the program's Writeback, under the slot's credit (l3_program.hpp). L-B is the
//                 data oracle only, not a timing reference. Only a Writeback that opens its
//                 residency is supported: one that rewrites a live slot has no Phase 1 meaning
//   Inherit    -> Fill of its slot with the tile as it is at the start (resident from before)
//   Retain     -> nothing: the tile stays in its slot past the program
// Every other action (Move, Feed, Call, Drain, L2Release) is the compute subsystem's.
//
// DRAM. The walker places no tile in DRAM, so the plan does: each operand in its own region from
// `base`, 64 KiB aligned (a layout that runs past the end of the address space is refused), its tiles in row-major tile order, each tile packed row-major at its
// number system's width (an int8 element is a byte, int32 four, little-endian) and padded to whole
// 64-byte beats. An L3 slot holds the largest padded tile.
//
// THE ORACLE. The plan follows the program's data: DRAM starts with the inputs, a Load places in
// its slot what DRAM holds for the tile then, a Store writes back what its slot holds. `placed`
// records, for every Load and Fill, the bytes its slot must hold once it is ready; `outputs`, what
// DRAM must hold at the end for every stored tile. The plan refuses a program whose Stores do not
// leave L-B's result in DRAM (a stored tile that was loaded, not written back, and that L-B
// changed): then the program and L-B disagree, before any cycle is run.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/numeric.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sw::kpu::cycle {

struct CspL3Plan {
    struct Tile {
        sw::kpu::program::TileCoord tile;
        std::uint64_t address = 0;          // in DRAM
        std::uint32_t bytes = 0;            // padded to whole beats
        std::vector<std::uint8_t> data;     // `bytes` of them: the tile, then zeros
    };
    L3Program program;
    std::uint32_t slots = 0, slot_bytes = 0, banks = 1;
    std::vector<Tile> inputs;               // DRAM before the run
    std::vector<Tile> outputs;              // DRAM after it: every stored tile, L-B's bytes
    std::map<std::size_t, std::vector<std::uint8_t>> placed;   // by Load/Fill ticket: its slot's bytes
    std::map<std::size_t, sw::kpu::program::TileCoord> what;   // by Load/Fill ticket: the tile
    std::size_t loads = 0, stores = 0, fills = 0, releases = 0;
};

namespace csp_l3_detail {

using sw::kpu::program::Dim;
using sw::kpu::program::TensorOperand;
using sw::kpu::program::TileCoord;
using sw::kpu::program::TileProgram;

inline std::uint32_t round_beats(std::uint64_t n) {
    return static_cast<std::uint32_t>((n + kL3BeatBytes - 1) / kL3BeatBytes * kL3BeatBytes);
}

inline unsigned element_bytes(const TensorOperand& op) {
    const auto* ns = sw::kpu::program::numeric::find(op.type);
    if (!ns) throw std::invalid_argument("plan_l3: operand " + op.name + " has an unknown number system " + op.type);
    return (ns->bits + 7) / 8;
}

// The tile's bytes, packed row-major at the element's width, little-endian, padded to beats.
inline std::vector<std::uint8_t> pack(const TensorOperand& op, Dim ti, Dim tj) {
    const unsigned w = element_bytes(op);
    std::vector<std::uint8_t> v;
    for (Dim r = op.row_begin(ti); r < op.row_end(ti); ++r)
        for (Dim c = op.col_begin(tj); c < op.col_end(tj); ++c) {
            const auto word = op.word(r, c);
            for (unsigned b = 0; b < w; ++b) v.push_back(static_cast<std::uint8_t>(word >> (8 * b)));
        }
    v.resize(round_beats(v.size()), 0);
    return v;
}

inline std::tuple<std::string, Dim, Dim> key(const TileCoord& t) { return {t.operand, t.ti, t.tj}; }

}  // namespace csp_l3_detail

// `inputs`: every operand as DRAM holds it before the run; `result`: as L-B leaves it after.
inline CspL3Plan plan_l3(const sw::kpu::program::csp::CspProgram& p, const sw::kpu::program::TileProgram& inputs,
                         const sw::kpu::program::TileProgram& result, std::uint64_t base = 0x100000) {
    using namespace csp_l3_detail;
    using Kind = sw::kpu::program::csp::Action::Kind;
    CspL3Plan plan;
    plan.slots = static_cast<std::uint32_t>(p.channel(sw::kpu::program::csp::Chan::L3).capacity);
    plan.banks = std::max<std::uint32_t>(1, p.l3_banks);
    if (plan.slots == 0) throw std::invalid_argument("plan_l3: the program's L3 has no slot bound");

    // DRAM: each operand's region, its tiles in row-major tile order.
    std::map<std::tuple<std::string, Dim, Dim>, CspL3Plan::Tile> dram;
    std::uint64_t at = base;
    constexpr std::uint64_t kMax = ~std::uint64_t{0};
    auto overflow = [&](const std::string& what) {
        return std::invalid_argument("plan_l3: " + what + " does not fit in the address space from base " +
                                     std::to_string(base));
    };
    for (const std::string& name : inputs.operand_order()) {
        const TensorOperand& op = inputs.operand(name);
        if (at > kMax - 0xFFFF) throw overflow("operand " + name + "'s region");
        at = (at + 0xFFFF) & ~std::uint64_t{0xFFFF};
        for (Dim ti = 0; ti < op.n_tile_rows(); ++ti)
            for (Dim tj = 0; tj < op.n_tile_cols(); ++tj) {
                CspL3Plan::Tile t;
                t.tile = {name, ti, tj};
                t.address = at;
                t.data = pack(op, ti, tj);
                t.bytes = static_cast<std::uint32_t>(t.data.size());
                if (at > kMax - t.bytes) throw overflow("tile " + t.tile.to_string());
                at += t.bytes;
                plan.slot_bytes = std::max(plan.slot_bytes, t.bytes);
                plan.inputs.push_back(t);
                dram[key(t.tile)] = t;
            }
    }
    auto tile_of = [&](const TileCoord& c) -> CspL3Plan::Tile& {
        const auto it = dram.find(key(c));
        if (it == dram.end()) throw std::invalid_argument("plan_l3: no operand tile " + c.to_string());
        return it->second;
    };

    // The actions, following the data: what each slot holds, and what DRAM holds.
    std::map<std::uint32_t, std::vector<std::uint8_t>> slot;
    std::map<std::tuple<std::string, Dim, Dim>, bool> stored;
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const auto& a = p.actions[i];
        const bool l3 = a.kind == Kind::Load || a.kind == Kind::Store || a.kind == Kind::Release ||
                        a.kind == Kind::Writeback || a.kind == Kind::Inherit;
        if (!l3) continue;
        if (!a.place) throw std::invalid_argument("plan_l3: action " + std::to_string(i) + " on " + a.tile.to_string() +
                                                  " has no L3 address");
        if (a.place->l3_tile != 0)
            throw std::invalid_argument("plan_l3: action " + std::to_string(i) + " addresses L3 module " +
                                        std::to_string(a.place->l3_tile) + "; Phase 1 has one L3 tile");
        const std::uint32_t s = a.place->slot;
        const std::size_t ticket = plan.program.size();
        switch (a.kind) {
            case Kind::Load: {
                const CspL3Plan::Tile& t = tile_of(a.tile);
                plan.program.push_back(l3_load(s, t.address, t.bytes));
                slot[s] = t.data;
                plan.placed[ticket] = t.data;
                plan.what[ticket] = a.tile;
                ++plan.loads;
                break;
            }
            case Kind::Writeback: {
                if (a.residency >= p.residencies.size() || p.residencies[a.residency].open != i)
                    throw std::invalid_argument("plan_l3: action " + std::to_string(i) + " writes " + a.tile.to_string() +
                                                " back into a live residency; Phase 1 fills a slot only once");
                std::vector<std::uint8_t> v = pack(result.operand(a.tile.operand), a.tile.ti, a.tile.tj);
                plan.program.push_back(l3_fill(s, v));
                plan.slot_bytes = std::max(plan.slot_bytes, static_cast<std::uint32_t>(v.size()));
                slot[s] = v;
                plan.placed[ticket] = std::move(v);
                plan.what[ticket] = a.tile;
                ++plan.fills;
                break;
            }
            case Kind::Inherit: {
                const CspL3Plan::Tile& t = tile_of(a.tile);
                plan.program.push_back(l3_fill(s, t.data));
                slot[s] = t.data;
                plan.placed[ticket] = t.data;
                plan.what[ticket] = a.tile;
                ++plan.fills;
                break;
            }
            case Kind::Store: {
                CspL3Plan::Tile& t = tile_of(a.tile);
                if (!slot.count(s))
                    throw std::invalid_argument("plan_l3: action " + std::to_string(i) + " stores slot " +
                                                std::to_string(s) + ", which nothing has filled");
                plan.program.push_back(l3_store(s, t.address, t.bytes));
                t.data = slot[s];
                stored[key(a.tile)] = true;
                ++plan.stores;
                break;
            }
            case Kind::Release:
                plan.program.push_back(l3_release(s));
                ++plan.releases;
                break;
            default: break;
        }
    }
    // The stored tiles, as DRAM ends: they must be L-B's.
    for (const auto& [k, t] : dram) {
        if (!stored.count(k)) continue;
        const std::vector<std::uint8_t> want = pack(result.operand(t.tile.operand), t.tile.ti, t.tile.tj);
        if (t.data != want)
            throw std::logic_error("plan_l3: the program's Stores leave " + t.tile.to_string() +
                                   " in DRAM other than L-B's result");
        plan.outputs.push_back(t);
    }
    return plan;
}

}  // namespace sw::kpu::cycle
