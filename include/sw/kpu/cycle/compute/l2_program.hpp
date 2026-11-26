#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/l2_program.hpp
// What the BlockMover and an L2 execute (#542; docs/plans/cycle-based-kpu-prototype.md §5.1, step
// 2a), and what moves between them and the Streamer (#543, step 2b).
//
// A program is a sequence of actions on L2 slots, in the program's order:
//   Move       L3 -> L2: `rows` rows of the L2's row size, the first at `l3_address` and each next
//              `l3_pitch` bytes on (strided), into slot `slot`'s rows 0..rows-1, as tile `tile`
//   Writeback  L2 -> L3: the slot's rows, back to `l3_address`, `l3_pitch` apart
//   Feed       L2 -> L1: one Streamer command's reads of `rows` of the slot's rows, as one operand
//              (streamer.hpp); a command reads each of its two operands as a Feed of its own
//   Result     the result path's rows into the slot (#545, step 2d): `rows` rows of a Streamer
//              command's result tile, written by the Combiner (combiner.hpp)
//   Release    frees the slot: its credit returns to the slot's next producer, the BlockMover for a
//              Move into it, the Streamer for a Result
// The BlockMover executes the Moves and Writebacks, in order, and the Streamer its commands; the L2
// executes the Releases, each bound to the PRODUCER before it into its slot (a Move or a Result),
// after every Writeback and Feed of that tile before it: the tile's CONSUMERS.
//
// A slot's credit is held by its next producer: the BlockMover's for a Move, the Streamer's for a
// Result (it spends it when the command that produces the tile starts: the RTL's ACQUIRE as the
// COMPUTE starts). At the start each slot's credit is its first producer's; a slot no producer
// names is the BlockMover's.
//
// THE SLOT PROTOCOL is #387's (timing/slot_pool.hpp): a Move's first row ACQUIREs its slot (FREE ->
// OWNED), its last SIGNALs it (READY), and a Result's rows do the same; a consumer's first read TAKEs it (READING), its last row's
// data DONE (consumers may overlap: the slot counts its readers); a Release frees it (READY -> FREE).
// A transition from the wrong state is a SlotFault.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/tile_descriptor.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::cycle {

using TileID = sw::kpu::timing::TileID;

struct L2Action {
    enum class Kind : std::uint8_t { Move, Writeback, Feed, Result, Release };
    Kind kind = Kind::Move;
    std::uint32_t slot = 0;
    TileID tile{};
    std::uint64_t l3_address = 0;   // Move: the source's first row; Writeback: the destination's
    std::uint32_t l3_pitch = 0;     // bytes from one row to the next in L3 (at least the row size)
    std::uint32_t rows = 0;         // Move, Writeback, Feed and Result
};
using L2Program = std::vector<L2Action>;

inline L2Action l2_move(std::uint32_t slot, TileID tile, std::uint64_t l3_address, std::uint32_t l3_pitch,
                        std::uint32_t rows) {
    return {L2Action::Kind::Move, slot, tile, l3_address, l3_pitch, rows};
}
inline L2Action l2_writeback(std::uint32_t slot, TileID tile, std::uint64_t l3_address, std::uint32_t l3_pitch,
                             std::uint32_t rows) {
    return {L2Action::Kind::Writeback, slot, tile, l3_address, l3_pitch, rows};
}
// `rows`: the command's lanes of this operand (its h A rows or w B rows).
inline L2Action l2_feed(std::uint32_t slot, TileID tile, std::uint32_t rows) {
    return {L2Action::Kind::Feed, slot, tile, 0, 0, rows};
}
// `rows`: the result tile's rows (the command's h).
inline L2Action l2_result(std::uint32_t slot, TileID tile, std::uint32_t rows) {
    return {L2Action::Kind::Result, slot, tile, 0, 0, rows};
}
inline L2Action l2_release(std::uint32_t slot, TileID tile) { return {L2Action::Kind::Release, slot, tile, 0, 0, 0}; }

inline const char* to_string(L2Action::Kind k) {
    switch (k) {
        case L2Action::Kind::Move: return "Move";
        case L2Action::Kind::Writeback: return "Writeback";
        case L2Action::Kind::Feed: return "Feed";
        case L2Action::Kind::Result: return "Result";
        case L2Action::Kind::Release: return "Release";
    }
    return "?";
}

// The program's own consistency, before anything runs: every slot exists, every transfer fits its
// slot and strides at least a row, a Move goes only into a slot its last tile was released from,
// and a Writeback, Feed or Release names the tile the slot holds (a Writeback or Feed no more rows
// than its Move wrote). Raises std::invalid_argument naming the action. (The run checks the same protocol again, state by state: a level's defect is a SlotFault.)
inline void validate_l2_program(const L2Program& p, std::uint32_t slots, std::uint32_t rows_per_slot,
                                std::uint32_t row_bytes, const std::string& who) {
    std::vector<int> held(slots, 0);           // 0 free, 1 holds a tile
    std::vector<TileID> tile(slots);
    std::vector<std::uint32_t> rows(slots, 0);   // the slot's Move's rows
    for (std::size_t i = 0; i < p.size(); ++i) {
        const L2Action& a = p[i];
        const std::string at = who + ": action " + std::to_string(i) + " (" + to_string(a.kind) + " slot " +
                               std::to_string(a.slot) + ", " + a.tile.to_string() + ")";
        if (a.slot >= slots) throw std::invalid_argument(at + ": there are " + std::to_string(slots) + " slots");
        if (a.kind == L2Action::Kind::Move || a.kind == L2Action::Kind::Writeback) {
            if (a.rows == 0 || a.rows > rows_per_slot)
                throw std::invalid_argument(at + ": " + std::to_string(a.rows) + " rows, a slot holds 1 to " +
                                            std::to_string(rows_per_slot));
            if (a.rows > 1 && a.l3_pitch < row_bytes)
                throw std::invalid_argument(at + ": pitch " + std::to_string(a.l3_pitch) + " is under the row's " +
                                            std::to_string(row_bytes) + " bytes, so its rows overlap in L3");
        } else if ((a.kind == L2Action::Kind::Feed || a.kind == L2Action::Kind::Result) &&
                   (a.rows == 0 || a.rows > rows_per_slot)) {
            throw std::invalid_argument(at + ": " + std::to_string(a.rows) + " rows, a slot holds 1 to " +
                                        std::to_string(rows_per_slot));
        }
        switch (a.kind) {
            case L2Action::Kind::Move:
            case L2Action::Kind::Result:
                if (held[a.slot])
                    throw std::invalid_argument(at + ": the slot still holds " + tile[a.slot].to_string() +
                                                " (no Release before this Move)");
                held[a.slot] = 1;
                tile[a.slot] = a.tile;
                rows[a.slot] = a.rows;
                break;
            case L2Action::Kind::Writeback:
            case L2Action::Kind::Feed:
            case L2Action::Kind::Release:
                if (!held[a.slot] || tile[a.slot] != a.tile)
                    throw std::invalid_argument(at + ": the slot holds " +
                                                (held[a.slot] ? tile[a.slot].to_string() : std::string("no tile")));
                if (a.kind != L2Action::Kind::Release && a.rows > rows[a.slot])
                    throw std::invalid_argument(at + ": " + std::to_string(a.rows) + " rows, but its Move wrote " +
                                                std::to_string(rows[a.slot]));
                if (a.kind == L2Action::Kind::Release) held[a.slot] = 0;
                break;
        }
    }
}

// A producer: it fills a slot (FREE -> OWNED -> READY).
inline bool is_producer(L2Action::Kind k) { return k == L2Action::Kind::Move || k == L2Action::Kind::Result; }

// Whether slot `slot`'s credit is the Streamer's at the start: its first producer is a Result.
inline bool streamer_holds_at_start(const L2Program& p, std::uint32_t slot) {
    for (const L2Action& a : p)
        if (a.slot == slot && is_producer(a.kind)) return a.kind == L2Action::Kind::Result;
    return false;
}

// --- what moves between the BlockMover and the L2 (a 1 GHz to 250 MHz crossing each way) ----------

// One row into an L2 slot, from a Move.
struct L2RowWrite {
    std::uint64_t id = 0;
    std::uint32_t slot = 0;
    std::uint32_t row = 0;           // within the slot
    std::uint32_t rows = 0;          // the Move's rows: the last one SIGNALs the slot
    TileID tile{};
    std::vector<std::uint8_t> bytes; // the L2's row size
};
// A request for one row of a slot, from a consumer (a Writeback, or one operand of a Streamer
// command): `first` and `last` are its first and last.
struct L2RowRead {
    std::uint64_t id = 0;
    std::uint32_t slot = 0;
    std::uint32_t row = 0;
    TileID tile{};
    bool first = false, last = false;
};
// The row's bytes, `read_latency` L2 clocks after the read was served.
struct L2RowData {
    std::uint64_t id = 0;            // the read's
    std::uint32_t slot = 0;
    std::uint32_t row = 0;
    TileID tile{};
    std::vector<std::uint8_t> bytes;
};

// The tile-ready token, L2 -> Streamer: slot `slot` is READY with `tile`, its Move's last row
// written. It admits `feeds` commands' reads of this residency, the program's Feeds of it, and no
// more: the Streamer spends one per command that reads the operand, and drops the token when none is
// left, so a later residency of the same tile is admitted only by its own token (CodeRabbit, #593).
// The Streamer fires a command only when it holds a token for each of its operands.
struct SlotReady {
    std::uint64_t id = 0;
    std::uint32_t slot = 0;
    TileID tile{};
    std::uint32_t feeds = 0;
};

// --- what moves between the BlockMover and L3 -----------------------------------------------------

// A request for `bytes` of L3 at `address`, returned as beats (cycle::Beat) of the mover's bus width,
// in order, the request's `tag` on each.
struct L3RowRead {
    std::uint64_t id = 0;
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    std::uint64_t tag = 0;
};

}  // namespace sw::kpu::cycle
