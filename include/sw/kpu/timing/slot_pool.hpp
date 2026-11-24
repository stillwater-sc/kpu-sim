// ============================================================================
// include/sw/kpu/timing/slot_pool.hpp
// The state of every L2 slot a CSP program places (docs/plans/slot-credits-at-lca.md §3.4, #387
// step 3b).
//
// The RTL orders its processes with whole-tile slot credits, and each slot is always in exactly
// one state:
//
//   FREE     a RELEASE left it (or the run began)
//   OWNED    the producer's ACQUIRE, at its start: a Move, a partial's first merge, or the call
//            that opens a result's slot (decision 1) -- being filled
//   READY    the producer's SIGNAL, at its completion: the tile-ready token
//   READING  a consumer took the token, at its start: a Feed, or a result's Writeback
//
// A transition from the wrong state is a protocol error, raised by name with the slot, the tile
// and the transition (SlotFault): never a stall, never silent. A wait is not a fault: a producer
// whose slot is not FREE at its turn simply does not ACQUIRE yet (TagCAM::slot_ready).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing {

/// A slot transition from the wrong state: a defect in a level, never a property of the program.
class SlotFault : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

class SlotPool {
public:
    enum class State : std::uint8_t { Free, Owned, Ready, Reading };

    static const char* name(State s) {
        switch (s) {
            case State::Free:    return "FREE";
            case State::Owned:   return "OWNED";
            case State::Ready:   return "READY";
            case State::Reading: return "READING";
        }
        return "?";
    }

    SlotPool() = default;
    explicit SlotPool(std::uint32_t slots) : slots_(slots) {}

    [[nodiscard]] std::uint32_t size() const { return static_cast<std::uint32_t>(slots_.size()); }
    [[nodiscard]] bool covers(std::uint32_t slot) const { return slot < slots_.size(); }
    [[nodiscard]] State state(std::uint32_t slot) const { return at(slot).state; }
    [[nodiscard]] std::optional<TileID> owner(std::uint32_t slot) const {
        const Slot& s = at(slot);
        if (s.state == State::Free) return std::nullopt;
        return s.tile;
    }
    [[nodiscard]] std::uint32_t readers(std::uint32_t slot) const { return at(slot).readers; }

    // FREE -> OWNED: the producer takes the slot's credit at its start.
    void acquire(std::uint32_t slot, const TileID& tile) {
        Slot& s = expect(slot, tile, "ACQUIRE", {State::Free}, /*same_tile=*/false);
        s.state = State::Owned;
        s.tile = tile;
        ++acquires_;
    }
    // OWNED -> READY: the producer's tile landed (the tile-ready token).
    void signal(std::uint32_t slot, const TileID& tile) {
        expect(slot, tile, "SIGNAL", {State::Owned}).state = State::Ready;
    }
    // READY | READING -> READING: a consumer starts reading the tile. Consumers of one copy may
    // overlap (two streamers feeding one staged tile), so a slot counts its readers.
    void take(std::uint32_t slot, const TileID& tile) {
        Slot& s = expect(slot, tile, "TAKE", {State::Ready, State::Reading});
        s.state = State::Reading;
        ++s.readers;
    }
    // READING -> READING | READY: a consumer finished reading; the last returns the slot to READY.
    void done(std::uint32_t slot, const TileID& tile) {
        Slot& s = expect(slot, tile, "DONE", {State::Reading});
        if (--s.readers == 0) s.state = State::Ready;
    }
    // READY -> FREE: the program's L2Release (its credit returns upstream).
    void release(std::uint32_t slot, const TileID& tile) {
        Slot& s = expect(slot, tile, "RELEASE", {State::Ready});
        s.state = State::Free;
        ++releases_;
    }
    // OWNED -> FREE: a claim undone before its transfer started (TagCAM::unclaim_slot).
    void unacquire(std::uint32_t slot, const TileID& tile) {
        Slot& s = expect(slot, tile, "UNACQUIRE", {State::Owned});
        s.state = State::Free;
        --acquires_;
    }

    [[nodiscard]] std::uint32_t free_count() const {
        std::uint32_t n = 0;
        for (const Slot& s : slots_) n += s.state == State::Free;
        return n;
    }
    [[nodiscard]] bool all_free() const { return free_count() == slots_.size(); }
    [[nodiscard]] std::uint64_t acquires() const { return acquires_; }
    [[nodiscard]] std::uint64_t releases() const { return releases_; }

    void reset() {
        for (Slot& s : slots_) s = Slot{};
        acquires_ = releases_ = 0;
    }

private:
    struct Slot {
        State state = State::Free;
        TileID tile{};
        std::uint32_t readers = 0;
    };

    const Slot& at(std::uint32_t slot) const {
        if (slot >= slots_.size())
            throw SlotFault("L2 slot " + std::to_string(slot) + " is past the program's " + std::to_string(slots_.size()) +
                            " slots");
        return slots_[slot];
    }
    Slot& expect(std::uint32_t slot, const TileID& tile, const char* what, std::initializer_list<State> from,
                 bool same_tile = true) {
        Slot& s = const_cast<Slot&>(at(slot));
        bool ok = false;
        for (State f : from) ok = ok || s.state == f;
        if (!ok)
            throw SlotFault("L2 slot " + std::to_string(slot) + ": " + what + " by " + tile.to_string() + " from " +
                            name(s.state) + (s.state == State::Free ? "" : " (held by " + s.tile.to_string() + ")"));
        if (same_tile && !(s.tile == tile))
            throw SlotFault("L2 slot " + std::to_string(slot) + ": " + what + " by " + tile.to_string() + ", but " +
                            s.tile.to_string() + " holds it");
        return s;
    }

    std::vector<Slot> slots_;
    std::uint64_t acquires_ = 0, releases_ = 0;
};

}  // namespace sw::kpu::timing
