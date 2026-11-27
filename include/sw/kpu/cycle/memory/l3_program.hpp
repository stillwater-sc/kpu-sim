#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/l3_program.hpp
// A program's L3 actions, and what the L3 tile and the DMA engine exchange (#538;
// docs/plans/cycle-based-kpu-prototype.md §4.1-4.3).
//
// THE PROGRAM. One list of L3 actions in the program's order, as the walker emits them:
//   Load     DRAM [dram_address, +bytes) into an L3 slot. The DMA engine executes it.
//   Store    an L3 slot's tile to DRAM [dram_address, +bytes). The L3 tile ejects the slot's bytes
//            into the DMA engine's store buffer, and the engine writes them to DRAM: the store is
//            two pushes, never a read of L3 by the engine (credits up, data down).
//   Release  the slot's tile is no longer needed: the L3 tile frees the slot and returns its credit
//            to the DMA engine, for the next Load into it.
//   Fill     `data` into an L3 slot, with no DRAM: a tile that arrives in L3 from elsewhere. In
//            Phase 1 it stands in for what the compute subsystem writes back (a program's
//            Writeback, with L-B's result as the data) and for a tile that is resident from the
//            start (an Inherit). The DMA engine executes it as it does a Load -- under the slot's
//            credit, one beat at a time over the L3 link -- so it is timed as a Load without its
//            DRAM reads.
//   Read     the BlockMover's Move of the slot's tile into L2 (#548, the join): a CONSUMER of the
//            tile, as a Store is. The BlockMover asks for the tile's `bytes`; the L3 tile serves it
//            once the slot holds the tile, and the slot's Release waits until it is served.
//   Write    the BlockMover's Writeback of a result tile into the slot: a PRODUCER, as a Load is.
//            Its credit is the BlockMover's: a Release returns a slot's credit to its next
//            producer, the DMA engine for a Load or Fill, the BlockMover for a Write; at the start
//            each slot's credit is its first producer's.
// Both components execute the same list, each its own part (the engine: Loads, Fills and
// Stores; the tile: Stores and Releases, and the BlockMover's Reads and Writes), in the list's order. An action's index is its ticket: a Store's
// beats carry it from the tile to the engine, which matches them to its own Store.
//
// THE LINKS (the engine and the tile in one domain, over a direct link: no NoC yet):
//   engine -> tile    L3Beat: a Load's bytes into its slot, one 64-byte beat each
//   tile -> engine    L3Beat: a Store's bytes out of its slot, into the store buffer
//   tile -> engine    SlotCredit: a slot freed by a Release; the engine holds it until a Load
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::cycle {

// The size of one beat between the engine and the tile, and of one DRAM burst.
inline constexpr std::uint32_t kL3BeatBytes = 64;

struct L3Action {
    enum class Kind : std::uint8_t { Load, Store, Release, Fill, Read, Write };
    Kind kind = Kind::Load;
    std::uint32_t slot = 0;
    std::uint64_t dram_address = 0;   // Load and Store
    std::uint32_t bytes = 0;          // Load, Store and Fill: the tile's size
    std::vector<std::uint8_t> data;   // Fill: the tile's bytes, `bytes` of them
};

using L3Program = std::vector<L3Action>;

inline L3Action l3_load(std::uint32_t slot, std::uint64_t dram_address, std::uint32_t bytes) {
    return {L3Action::Kind::Load, slot, dram_address, bytes, {}};
}
inline L3Action l3_store(std::uint32_t slot, std::uint64_t dram_address, std::uint32_t bytes) {
    return {L3Action::Kind::Store, slot, dram_address, bytes, {}};
}
inline L3Action l3_release(std::uint32_t slot) { return {L3Action::Kind::Release, slot, 0, 0, {}}; }
// The BlockMover's read of the slot's tile (`bytes` of it), and its write of a tile into the slot.
inline L3Action l3_read(std::uint32_t slot, std::uint32_t bytes) { return {L3Action::Kind::Read, slot, 0, bytes, {}}; }
inline L3Action l3_write(std::uint32_t slot, std::uint32_t bytes) { return {L3Action::Kind::Write, slot, 0, bytes, {}}; }
inline L3Action l3_fill(std::uint32_t slot, std::vector<std::uint8_t> data) {
    const auto n = static_cast<std::uint32_t>(data.size());
    return {L3Action::Kind::Fill, slot, 0, n, std::move(data)};
}

inline const char* to_string(L3Action::Kind k) {
    switch (k) {
        case L3Action::Kind::Load: return "Load";
        case L3Action::Kind::Store: return "Store";
        case L3Action::Kind::Release: return "Release";
        case L3Action::Kind::Fill: return "Fill";
        case L3Action::Kind::Read: return "Read";
        case L3Action::Kind::Write: return "Write";
    }
    return "?";
}

// The program's shape, checked once by each component that executes it: every slot exists, and
// every tile is whole beats, fits its slot, starts on a burst boundary and ends inside the address
// space.
inline void validate_l3_program(const L3Program& p, std::uint32_t slots, std::uint32_t slot_bytes,
                                const std::string& who) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        const L3Action& a = p[i];
        const std::string at = who + ": action " + std::to_string(i) + " (" + to_string(a.kind) + " slot " +
                               std::to_string(a.slot) + ")";
        if (a.slot >= slots) throw std::invalid_argument(at + ": there are " + std::to_string(slots) + " slots");
        if (a.kind == L3Action::Kind::Release) continue;
        if (a.bytes == 0 || a.bytes % kL3BeatBytes != 0)
            throw std::invalid_argument(at + ": " + std::to_string(a.bytes) + " bytes, not a positive multiple of " +
                                        std::to_string(kL3BeatBytes));
        if (a.bytes > slot_bytes)
            throw std::invalid_argument(at + ": " + std::to_string(a.bytes) + " bytes, past the slot's " +
                                        std::to_string(slot_bytes));
        if (a.kind == L3Action::Kind::Read || a.kind == L3Action::Kind::Write) continue;   // no DRAM
        if (a.kind == L3Action::Kind::Fill) {
            if (a.data.size() != a.bytes)
                throw std::invalid_argument(at + ": " + std::to_string(a.data.size()) + " bytes of data for a tile of " +
                                            std::to_string(a.bytes));
            continue;   // no DRAM
        }
        if (a.dram_address % kL3BeatBytes != 0)
            throw std::invalid_argument(at + ": DRAM address " + std::to_string(a.dram_address) +
                                        " is not on a " + std::to_string(kL3BeatBytes) + "-byte burst boundary");
        if (a.dram_address > ~std::uint64_t{0} - (a.bytes - 1))
            throw std::invalid_argument(at + ": DRAM [" + std::to_string(a.dram_address) + ", +" +
                                        std::to_string(a.bytes) + ") runs past the end of the address space");
    }
}

// A producer: it fills a slot (a Load, a Fill, or the BlockMover's Write).
inline bool is_l3_producer(L3Action::Kind k) {
    return k == L3Action::Kind::Load || k == L3Action::Kind::Fill || k == L3Action::Kind::Write;
}
// Whether slot `slot`'s credit is the BlockMover's at the start: its first producer is a Write.
inline bool blockmover_holds_at_start(const L3Program& p, std::uint32_t slot) {
    for (const L3Action& a : p)
        if (a.slot == slot && is_l3_producer(a.kind)) return a.kind == L3Action::Kind::Write;
    return false;
}

struct L3Beat {
    std::uint64_t id = 0;            // unique per sender, for the trace
    std::uint64_t ticket = 0;        // the action (Load or Store) it belongs to: its index in the program
    std::uint32_t slot = 0;
    std::uint32_t offset = 0;        // into the slot
    std::uint32_t tile_bytes = 0;    // the whole tile's size, so the receiver knows when it is complete
    std::vector<std::uint8_t> bytes; // kL3BeatBytes of them
};

struct SlotCredit {
    std::uint64_t id = 0;
    std::uint32_t slot = 0;
};

}  // namespace sw::kpu::cycle
