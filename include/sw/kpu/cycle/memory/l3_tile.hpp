#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/l3_tile.hpp
// An L3 tile on the cycle core (#538; docs/plans/cycle-based-kpu-prototype.md §4.1, §4.3):
// addressed SRAM of `slots` x `slot_bytes`, interleaved over `banks`, with one write port and one
// read port. It holds the bytes it was written.
//
// A SLOT is FREE (its credit is with the DMA engine, or on its way back), FILLING (a Load's first
// beat has landed) or READY (every beat of the Load has landed: the tile-ready token). It holds one
// tile at a time:
//   - a beat lands in a FREE slot, and starts its fill, or in a slot FILLING with the same Load.
//     A beat of another Load into a FILLING or READY slot -- a Load into a live slot -- is a fault,
//     and so are a beat outside its tile and a beat written twice;
//   - a Store ejects a READY slot's tile, one beat a cycle, into the DMA engine's store buffer,
//     while the eject link has a credit. It is bound to the Load before it into its slot, in the
//     program's order, and waits, with no fault, until the slot is READY with that Load's tile: a
//     Store never reads a slot whose tile has not arrived, nor another Load's tile;
//   - a Release, bound the same way, frees the slot (after any Store before it in the program) and
//     pushes the slot's credit back to the engine.
// The tile executes its part of the program (Stores and Releases, l3_program.hpp) in the program's
// order, one at a time.
//
// PORTS AND BANKS. Each cycle, at most one beat is read (a Store's) and one written (a Load's). A
// beat's bank is its beat index in the slot modulo `banks`; a bank serves one access a cycle, and
// the read goes first, so a write to the bank being read waits a cycle (bank_conflicts() counts
// them; the cycle is busy, since the read moved).
//
// COUNTERS. A tile is an item (counters.hpp): it arrives when its first beat lands and completes
// when its Release frees the slot, so the occupancy is the live slots and the mean residence a
// tile's time in L3. The tile is busy in a cycle either port moves; otherwise it is stalled on
// the head command's wait: no data (its slot's tile has not arrived) or no credit (the eject or
// credit link is full).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class L3Tile : public Component {
public:
    struct Config {
        std::uint32_t slots = 2;
        std::uint32_t slot_bytes = 4096;
        std::uint32_t banks = 4;
    };
    enum class SlotState : std::uint8_t { Free, Filling, Ready };
    // A tile's life in a slot, for the tests and the trace: the Load (ticket) it came from, when
    // its first beat landed, when it became READY, and when its Release freed the slot (unset
    // while it is live).
    struct Residency {
        std::uint64_t ticket = 0;
        std::uint32_t slot = 0;
        Cycle filling_from = 0;
        Cycle ready_at = 0;
        std::optional<Cycle> released_at;
    };

    L3Tile(std::string name, Domain& d, Config cfg, const L3Program& program)
        : Component(std::move(name), d), config_(cfg) {
        if (cfg.slots == 0 || cfg.banks == 0 || cfg.slot_bytes == 0 || cfg.slot_bytes % kL3BeatBytes != 0)
            throw std::invalid_argument("cycle::L3Tile " + this->name() +
                                        ": slots, banks and slot bytes must be positive, the bytes whole beats");
        validate_l3_program(program, cfg.slots, cfg.slot_bytes, "cycle::L3Tile " + this->name());
        // Each Store and Release is bound to the Load before it into its slot, in the program's
        // order: it waits until the slot holds that Load's tile, whatever else has filled it.
        std::vector<std::optional<std::size_t>> last_load(cfg.slots);
        for (std::size_t i = 0; i < program.size(); ++i) {
            const auto k = program[i].kind;
            if (k == L3Action::Kind::Load || k == L3Action::Kind::Fill) last_load[program[i].slot] = i;
            else commands_.push_back({i, program[i], last_load[program[i].slot]});
        }
        clear();
    }

    // in: the Loads' beats; eject: the Stores' beats, to the store buffer; credits: freed slots.
    void connect(Link<L3Beat>& in, Link<L3Beat>& eject, Link<SlotCredit>& credits) {
        in_ = &in;
        eject_ = &eject;
        credits_ = &credits;
    }

    void evaluate(Cycle now) override {
        if (!in_ || !eject_ || !credits_) throw std::logic_error("cycle::L3Tile " + name() + ": not connected");
        bool moved = false;
        std::optional<Stall> wait;
        // The read port: the head command.
        std::optional<std::uint32_t> read_bank;
        if (head_ < commands_.size()) {
            const Command& c = commands_[head_];
            Slot& s = slots_[c.action.slot];
            if (!holds_its_tile(c)) {
                wait = Stall::NoData;   // the tile-ready token of the slot's Load
            } else if (c.action.kind == L3Action::Kind::Store) {
                if (c.action.bytes != s.tile_bytes)
                    throw std::logic_error(where(now) + ": action " + std::to_string(c.ticket) + " stores " +
                                           std::to_string(c.action.bytes) + " bytes from slot " +
                                           std::to_string(c.action.slot) + ", which holds a tile of " +
                                           std::to_string(s.tile_bytes));
                if (eject_->can_push(now)) {
                    L3Beat b;
                    b.id = ++beat_id_;
                    b.ticket = c.ticket;
                    b.slot = c.action.slot;
                    b.offset = offset_;
                    b.tile_bytes = c.action.bytes;
                    b.bytes.assign(s.bytes.begin() + offset_, s.bytes.begin() + offset_ + kL3BeatBytes);
                    read_bank = bank(offset_);
                    eject_->push(now, std::move(b));
                    offset_ += kL3BeatBytes;
                    if (offset_ == c.action.bytes) {
                        offset_ = 0;
                        ++head_;
                    }
                    moved = true;
                } else {
                    wait = Stall::NoCredit;
                }
            } else {   // Release
                if (credits_->can_push(now)) {
                    s.state = SlotState::Free;
                    residencies_[s.residency].released_at = now;
                    credits_->push(now, SlotCredit{++credit_id_, c.action.slot});
                    counters().complete();
                    ++head_;
                    moved = true;
                } else {
                    wait = Stall::NoCredit;
                }
            }
        }
        // The write port: one landed beat, unless its bank is being read.
        if (in_->can_pop(now)) {
            if (read_bank && bank(in_->peek(now).offset) == *read_bank) {
                ++bank_conflicts_;   // the read moved, so the cycle is busy; the write waits for the bank
            } else {
                write(now, in_->pop(now));
                moved = true;
            }
        }
        if (moved) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override { return head_ == commands_.size(); }

    void check(Cycle now) const override {
        std::uint32_t live = 0;
        for (std::uint32_t i = 0; i < config_.slots; ++i) {
            const Slot& s = slots_[i];
            if (s.state == SlotState::Free) continue;
            ++live;
            const std::size_t beats = s.tile_bytes / kL3BeatBytes;
            if ((s.state == SlotState::Ready) != (s.landed == beats))
                throw std::logic_error(where(now) + ": slot " + std::to_string(i) + " has " + std::to_string(s.landed) +
                                       " of " + std::to_string(beats) + " beats in state " + to_string(s.state));
        }
        if (live != counters().counts().occupancy)
            throw std::logic_error(where(now) + ": " + std::to_string(live) + " live slots, but the counters hold " +
                                   std::to_string(counters().counts().occupancy) + " tiles");
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::string d;
        if (head_ < commands_.size()) {
            const Command& c = commands_[head_];
            d = "command " + std::to_string(head_ + 1) + " of " + std::to_string(commands_.size()) + " (" +
                cycle::to_string(c.action.kind) + " slot " + std::to_string(c.action.slot);
            if (!holds_its_tile(c))
                d += ": waiting for its tile" + (c.load ? " (Load " + std::to_string(*c.load) + ")" : std::string(", which no Load places"));
            else if (c.action.kind == L3Action::Kind::Store)
                d += ": ejected " + std::to_string(offset_) + " of " + std::to_string(c.action.bytes) + " bytes";
            d += ")";
        } else {
            d = "all " + std::to_string(commands_.size()) + " commands done";
        }
        d += "; slots:";
        for (std::uint32_t i = 0; i < config_.slots; ++i) {
            const Slot& s = slots_[i];
            d += " " + std::to_string(i) + " " + to_string(s.state);
            if (s.state != SlotState::Free) d += " (ticket " + std::to_string(s.ticket) + ")";
        }
        return d;
    }

    // Queries, for the tests and the cross-component invariants.
    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] SlotState state(std::uint32_t slot) const { return slots_.at(slot).state; }
    [[nodiscard]] bool live(std::uint32_t slot) const { return state(slot) != SlotState::Free; }
    [[nodiscard]] bool ready(std::uint32_t slot) const { return state(slot) == SlotState::Ready; }
    [[nodiscard]] std::uint32_t live_count() const {
        std::uint32_t n = 0;
        for (std::uint32_t i = 0; i < config_.slots; ++i) n += live(i) ? 1 : 0;
        return n;
    }
    // The Load whose tile the slot holds (or last held).
    [[nodiscard]] std::uint64_t ticket(std::uint32_t slot) const { return slots_.at(slot).ticket; }
    // The slot's tile: its bytes, as many as the Load placed.
    [[nodiscard]] std::vector<std::uint8_t> peek(std::uint32_t slot) const {
        const Slot& s = slots_.at(slot);
        return {s.bytes.begin(), s.bytes.begin() + s.tile_bytes};
    }
    [[nodiscard]] const std::vector<Residency>& residencies() const { return residencies_; }
    // Cycles a landed write waited because the read port held its bank.
    [[nodiscard]] std::uint64_t bank_conflicts() const { return bank_conflicts_; }

    static const char* to_string(SlotState s) {
        switch (s) {
            case SlotState::Free: return "free";
            case SlotState::Filling: return "filling";
            case SlotState::Ready: return "ready";
        }
        return "?";
    }

private:
    struct Command {
        std::size_t ticket;
        L3Action action;
        std::optional<std::size_t> load;   // the Load (or Fill) before it into its slot
    };
    // The slot holds the tile the command is bound to, all of it.
    [[nodiscard]] bool holds_its_tile(const Command& c) const {
        const Slot& s = slots_[c.action.slot];
        return c.load && s.state == SlotState::Ready && s.ticket == *c.load;
    }
    struct Slot {
        SlotState state = SlotState::Free;
        std::uint64_t ticket = 0;
        std::uint32_t tile_bytes = 0;
        std::vector<bool> landed_beats;
        std::size_t landed = 0;
        std::size_t residency = 0;
        Cycle filling_from = 0;
        std::vector<std::uint8_t> bytes;
    };

    [[nodiscard]] std::uint32_t bank(std::uint32_t offset) const { return (offset / kL3BeatBytes) % config_.banks; }
    [[nodiscard]] std::string where(Cycle now) const {
        return "cycle::L3Tile " + name() + " at cycle " + std::to_string(now);
    }

    void write(Cycle now, const L3Beat& b) {
        const std::string beat = "the beat at offset " + std::to_string(b.offset) + " of Load " + std::to_string(b.ticket) +
                                 " into slot " + std::to_string(b.slot);
        if (b.slot >= config_.slots) throw std::logic_error(where(now) + ": " + beat + ": there is no such slot");
        if (b.bytes.size() != kL3BeatBytes || b.offset % kL3BeatBytes != 0 || b.tile_bytes == 0 ||
            b.tile_bytes % kL3BeatBytes != 0 || b.tile_bytes > config_.slot_bytes || b.offset >= b.tile_bytes)
            throw std::logic_error(where(now) + ": " + beat + " (" + std::to_string(b.bytes.size()) + " bytes, tile of " +
                                   std::to_string(b.tile_bytes) + ") is outside its tile or its slot");
        Slot& s = slots_[b.slot];
        if (s.state == SlotState::Ready || (s.state == SlotState::Filling && s.ticket != b.ticket))
            throw std::logic_error(where(now) + ": " + beat + ", which holds Load " + std::to_string(s.ticket) +
                                   "'s tile (" + to_string(s.state) + "): a Load into a live slot");
        if (s.state == SlotState::Free) {
            s.state = SlotState::Filling;
            s.ticket = b.ticket;
            s.tile_bytes = b.tile_bytes;
            s.landed_beats.assign(b.tile_bytes / kL3BeatBytes, false);
            s.filling_from = now;
            s.landed = 0;
            counters().accept();
        } else if (b.tile_bytes != s.tile_bytes) {
            throw std::logic_error(where(now) + ": " + beat + " says its tile is " + std::to_string(b.tile_bytes) +
                                   " bytes, its first beat " + std::to_string(s.tile_bytes));
        }
        const std::size_t k = b.offset / kL3BeatBytes;
        if (s.landed_beats[k]) throw std::logic_error(where(now) + ": " + beat + " landed twice");
        std::copy(b.bytes.begin(), b.bytes.end(), s.bytes.begin() + b.offset);
        s.landed_beats[k] = true;
        if (++s.landed == s.landed_beats.size()) {
            s.state = SlotState::Ready;
            s.residency = residencies_.size();
            residencies_.push_back({s.ticket, b.slot, s.filling_from, now, std::nullopt});
        }
    }

    void clear() {
        slots_.assign(config_.slots, Slot{});
        for (Slot& s : slots_) s.bytes.assign(config_.slot_bytes, 0);
        head_ = 0;
        offset_ = 0;
        beat_id_ = 0;
        credit_id_ = 0;
        bank_conflicts_ = 0;
        residencies_.clear();
    }

    Config config_;
    std::vector<Command> commands_;
    std::vector<Slot> slots_;
    std::size_t head_ = 0;
    std::uint32_t offset_ = 0;   // the head Store's next beat
    std::uint64_t beat_id_ = 0, credit_id_ = 0;
    std::uint64_t bank_conflicts_ = 0;
    std::vector<Residency> residencies_;
    Link<L3Beat>* in_ = nullptr;
    Link<L3Beat>* eject_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
};

}  // namespace sw::kpu::cycle
