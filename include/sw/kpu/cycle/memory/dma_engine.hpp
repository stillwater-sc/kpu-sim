#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/dma_engine.hpp
// A DMA engine on the cycle core (#538; docs/plans/cycle-based-kpu-prototype.md §4.1, §4.3):
// it executes a program's Loads and Stores (l3_program.hpp) between one DRAM channel
// (Lpddr5Channel, on its request and response links) and one L3 tile (L3Tile, on a direct link).
//
//   Load   starts, in the program's order, once the engine holds its slot's credit, and spends
//          it. Its tile is split into 64-byte bursts, each a DRAM read; each read's bytes are
//          pushed to the L3 tile as one beat, when the L3 link has a credit. It completes when its
//          last beat is pushed. The slot's credit comes back only when the tile's Release frees it.
//   Store  starts in the program's order, with no wait. The L3 tile ejects the slot's beats into
//          the engine's STORE BUFFER (`store_buffer_beats` of them, popped only with room); each
//          staged beat becomes a DRAM write, which leaves the buffer as it is sent. It completes
//          when its last write is acknowledged.
// The engine holds every slot's credit at the start. A Load waiting for a credit holds the
// commands after it (they start in order); commands already started keep going.
//
// THE BURST WINDOW. At most `window` bursts, reads and writes, are in flight (sent, not yet
// answered). Each cycle the engine sends at most one burst: the oldest started command's next
// one (a Load's next read, or a Store's first staged beat).
//
// Each cycle, in this order: (1) every landed slot credit is taken; (2) the oldest DRAM response:
// a read's beat is pushed to L3 (or waits for the L3 link's credit, holding the responses behind
// it), a write's acknowledgment is counted; (3) one ejected beat is staged, if the store buffer
// has room; (4) one burst is sent, within the window and with the request link's credit; (5) the
// next command starts, if it can.
//
// COUNTERS. A command is an item (counters.hpp): it arrives when it starts and completes when its
// last burst does. The engine is busy in a cycle anything moves, or while bursts are in flight;
// otherwise stalled on what blocked it (no credit: a slot's, L3's, the request link's, the window
// or the store buffer). dma() adds the engine's own: bursts, the most in flight, and the cycles
// each cause held something up (several can in one cycle).
//
// CHECKS, every cycle: the window and the store buffer within their bounds, and a slot credit
// never received twice. check_slot_credits() checks the engine and its L3 tile together: a slot's
// credit is never with the engine while the slot is live.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

enum class DmaWait : std::uint8_t { SlotCredit, L3Credit, RequestCredit, Window, StoreBuffer };
inline constexpr std::size_t kDmaWaits = 5;

inline const char* to_string(DmaWait w) {
    switch (w) {
        case DmaWait::SlotCredit: return "no slot credit";
        case DmaWait::L3Credit: return "no L3 link credit";
        case DmaWait::RequestCredit: return "no DRAM request credit";
        case DmaWait::Window: return "burst window full";
        case DmaWait::StoreBuffer: return "store buffer full";
    }
    return "?";
}

struct DmaCounts {
    std::uint64_t loads = 0, stores = 0;            // completed
    std::uint64_t read_bursts = 0, write_bursts = 0; // sent
    std::uint64_t max_in_flight = 0;
    std::array<std::uint64_t, kDmaWaits> waits{};    // cycles each cause held something up
    [[nodiscard]] std::uint64_t waited(DmaWait w) const { return waits[static_cast<std::size_t>(w)]; }
};

class DmaEngine : public Component {
public:
    struct Config {
        std::uint32_t slots = 2;               // the L3 tile's: the engine starts with every credit
        std::uint32_t slot_bytes = 4096;       // the L3 tile's, to check the program
        std::size_t window = 8;                // bursts in flight
        std::size_t store_buffer_beats = 4;
    };
    // When each command started and completed, for the tests.
    struct Span {
        std::uint64_t ticket = 0;
        Cycle started = 0;
        std::optional<Cycle> completed;
    };

    DmaEngine(std::string name, Domain& d, Config cfg, const L3Program& program)
        : Component(std::move(name), d), config_(cfg) {
        if (cfg.slots == 0 || cfg.window == 0 || cfg.store_buffer_beats == 0)
            throw std::invalid_argument("cycle::DmaEngine " + this->name() +
                                        ": slots, the window and the store buffer must be positive");
        validate_l3_program(program, cfg.slots, cfg.slot_bytes, "cycle::DmaEngine " + this->name());
        for (std::size_t i = 0; i < program.size(); ++i)
            if (program[i].kind != L3Action::Kind::Release) {
                by_ticket_[i] = commands_.size();
                commands_.push_back({i, program[i]});
            }
        clear();
    }

    void connect(Link<DramRequest>& to_dram, Link<DramResponse>& from_dram, Link<L3Beat>& to_l3,
                 Link<L3Beat>& from_l3, Link<SlotCredit>& credits) {
        to_dram_ = &to_dram;
        from_dram_ = &from_dram;
        to_l3_ = &to_l3;
        from_l3_ = &from_l3;
        credits_ = &credits;
    }

    void evaluate(Cycle now) override {
        if (!to_dram_ || !from_dram_ || !to_l3_ || !from_l3_ || !credits_)
            throw std::logic_error("cycle::DmaEngine " + name() + ": not connected");
        bool moved = false;
        std::array<bool, kDmaWaits> waiting{};
        auto wait = [&](DmaWait w) { waiting[static_cast<std::size_t>(w)] = true; };

        // 1. Slot credits back from L3.
        while (credits_->can_pop(now)) {
            const SlotCredit c = credits_->pop(now);
            if (c.slot >= config_.slots || held_[c.slot])
                throw std::logic_error(where(now) + ": a credit for slot " + std::to_string(c.slot) +
                                       ", which it already holds or does not have");
            held_[c.slot] = true;
            moved = true;
        }
        // 2. The oldest DRAM response.
        if (from_dram_->can_pop(now)) {
            const auto it = in_flight_.find(from_dram_->peek(now).id);
            if (it == in_flight_.end())
                throw std::logic_error(where(now) + ": a DRAM response to request " +
                                       std::to_string(from_dram_->peek(now).id) + ", which it never sent");
            const Burst burst = it->second;
            Command& c = commands_[burst.command];
            if (c.action.kind == L3Action::Kind::Load) {
                if (to_l3_->can_push(now)) {
                    DramResponse r = from_dram_->pop(now);
                    L3Beat b;
                    b.id = ++beat_id_;
                    b.ticket = c.ticket;
                    b.slot = c.action.slot;
                    b.offset = burst.offset;
                    b.tile_bytes = c.action.bytes;
                    b.bytes = std::move(r.bytes);
                    to_l3_->push(now, std::move(b));
                    finish_burst(now, it);
                    moved = true;
                } else {
                    wait(DmaWait::L3Credit);
                }
            } else {
                from_dram_->pop(now);
                finish_burst(now, it);
                moved = true;
            }
        }
        // 3. One ejected beat into the store buffer.
        if (from_l3_->can_pop(now)) {
            if (staged_.size() < config_.store_buffer_beats) {
                L3Beat b = from_l3_->pop(now);
                const auto t = by_ticket_.find(b.ticket);
                if (t == by_ticket_.end() || commands_[t->second].action.kind != L3Action::Kind::Store ||
                    commands_[t->second].action.slot != b.slot || commands_[t->second].action.bytes != b.tile_bytes)
                    throw std::logic_error(where(now) + ": an ejected beat of action " + std::to_string(b.ticket) +
                                           " (slot " + std::to_string(b.slot) + ", " + std::to_string(b.tile_bytes) +
                                           " bytes), which is not one of its Stores");
                staged_.push_back(std::move(b));
                moved = true;
            } else {
                wait(DmaWait::StoreBuffer);
            }
        }
        // 4. One burst: the oldest started command's next.
        if (send_burst(now, wait)) moved = true;
        // 5. The next command, in the program's order.
        if (next_ < commands_.size()) {
            Command& c = commands_[next_];
            const bool load = c.action.kind == L3Action::Kind::Load;
            if (!load || held_[c.action.slot]) {
                if (load) held_[c.action.slot] = false;
                active_.push_back(next_);
                spans_.push_back({c.ticket, now, std::nullopt});
                c.span = spans_.size() - 1;
                counters().accept();
                ++next_;
                moved = true;
            } else {
                wait(DmaWait::SlotCredit);
            }
        }

        std::optional<Stall> stall;
        for (std::size_t w = 0; w < kDmaWaits; ++w)
            if (waiting[w]) {
                ++dma_.waits[w];
                if (!stall) stall = Stall::NoCredit;
            }
        if (moved || !in_flight_.empty()) counters().busy();
        else if (stall) counters().stall(*stall);
    }

    [[nodiscard]] bool idle() const override { return next_ == commands_.size() && active_.empty() && staged_.empty(); }

    void check(Cycle now) const override {
        if (in_flight_.size() > config_.window)
            throw std::logic_error(where(now) + ": " + std::to_string(in_flight_.size()) + " bursts in flight, past the window " +
                                   std::to_string(config_.window));
        if (staged_.size() > config_.store_buffer_beats)
            throw std::logic_error(where(now) + ": " + std::to_string(staged_.size()) +
                                   " beats in the store buffer, past its " + std::to_string(config_.store_buffer_beats));
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::string d;
        if (next_ < commands_.size()) {
            const Command& c = commands_[next_];
            d = "next command " + std::to_string(next_ + 1) + " of " + std::to_string(commands_.size()) + " (" +
                to_string(c.action.kind) + " slot " + std::to_string(c.action.slot) +
                (c.action.kind == L3Action::Kind::Load && !held_[c.action.slot] ? ": waiting for its credit)" : ")");
        } else {
            d = "every command started";
        }
        std::uint32_t held = 0;
        for (const bool h : held_) held += h ? 1 : 0;
        return d + "; " + std::to_string(active_.size()) + " active, " + std::to_string(in_flight_.size()) +
               " bursts in flight, store buffer " + std::to_string(staged_.size()) + " of " +
               std::to_string(config_.store_buffer_beats) + ", slot credits held " + std::to_string(held) + " of " +
               std::to_string(config_.slots);
    }

    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] bool holds(std::uint32_t slot) const { return held_.at(slot); }
    [[nodiscard]] std::uint32_t held_count() const {
        return static_cast<std::uint32_t>(std::count(held_.begin(), held_.end(), true));
    }
    [[nodiscard]] std::size_t in_flight() const { return in_flight_.size(); }
    [[nodiscard]] const DmaCounts& dma() const { return dma_; }
    [[nodiscard]] const std::vector<Span>& spans() const { return spans_; }

private:
    struct Command {
        std::size_t ticket;
        L3Action action;
        std::uint32_t sent = 0, done = 0;   // bursts
        std::size_t span = 0;
        [[nodiscard]] std::uint32_t bursts() const { return action.bytes / kL3BeatBytes; }
    };
    struct Burst {
        std::size_t command;
        std::uint32_t offset;
    };

    [[nodiscard]] std::string where(Cycle now) const {
        return "cycle::DmaEngine " + name() + " at cycle " + std::to_string(now);
    }

    template <class Wait>
    bool send_burst(Cycle now, Wait&& wait) {
        for (const std::size_t ci : active_) {
            Command& c = commands_[ci];
            std::optional<DramRequest> r;
            std::deque<L3Beat>::iterator staged = staged_.end();
            if (c.action.kind == L3Action::Kind::Load) {
                if (c.sent == c.bursts()) continue;
                r.emplace();
                r->address = c.action.dram_address + std::uint64_t(c.sent) * kL3BeatBytes;
            } else {
                staged = std::find_if(staged_.begin(), staged_.end(), [&](const L3Beat& b) { return b.ticket == c.ticket; });
                if (staged == staged_.end()) continue;
                r.emplace();
                r->write = true;
                r->address = c.action.dram_address + staged->offset;
            }
            if (in_flight_.size() >= config_.window) {
                wait(DmaWait::Window);
                return false;
            }
            if (!to_dram_->can_push(now)) {
                wait(DmaWait::RequestCredit);
                return false;
            }
            r->id = ++request_id_;
            r->size = kL3BeatBytes;
            r->tag = c.ticket;
            const std::uint32_t offset = static_cast<std::uint32_t>(r->address - c.action.dram_address);
            if (r->write) {
                r->bytes = std::move(staged->bytes);
                staged_.erase(staged);
                ++dma_.write_bursts;
            } else {
                ++dma_.read_bursts;
            }
            in_flight_.emplace(r->id, Burst{ci, offset});
            to_dram_->push(now, std::move(*r));
            ++c.sent;
            dma_.max_in_flight = std::max<std::uint64_t>(dma_.max_in_flight, in_flight_.size());
            return true;
        }
        return false;
    }

    void finish_burst(Cycle now, std::unordered_map<std::uint64_t, Burst>::iterator it) {
        Command& c = commands_[it->second.command];
        const std::size_t ci = it->second.command;
        in_flight_.erase(it);
        if (++c.done < c.bursts()) return;
        active_.erase(std::find(active_.begin(), active_.end(), ci));
        spans_[c.span].completed = now;
        counters().complete();
        if (c.action.kind == L3Action::Kind::Load) ++dma_.loads;
        else ++dma_.stores;
    }

    void clear() {
        for (Command& c : commands_) {
            c.sent = c.done = 0;
            c.span = 0;
        }
        held_.assign(config_.slots, true);
        next_ = 0;
        active_.clear();
        staged_.clear();
        in_flight_.clear();
        spans_.clear();
        dma_ = {};
        beat_id_ = request_id_ = 0;
    }

    Config config_;
    std::vector<Command> commands_;
    std::unordered_map<std::size_t, std::size_t> by_ticket_;   // a Load's or Store's action -> its command
    std::vector<bool> held_;                                   // the slot credits it holds
    std::size_t next_ = 0;                                     // the next command to start
    std::vector<std::size_t> active_;                          // started, not complete, in start order
    std::deque<L3Beat> staged_;                                // the store buffer
    std::unordered_map<std::uint64_t, Burst> in_flight_;       // by DRAM request id
    std::vector<Span> spans_;
    DmaCounts dma_;
    std::uint64_t beat_id_ = 0, request_id_ = 0;
    Link<DramRequest>* to_dram_ = nullptr;
    Link<DramResponse>* from_dram_ = nullptr;
    Link<L3Beat>* to_l3_ = nullptr;
    Link<L3Beat>* from_l3_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
};

// The engine and its L3 tile together: a slot's credit is never with the engine while the slot is
// live, which would let a Load into a live slot start. For Simulator::add_invariant.
inline void check_slot_credits(const DmaEngine& dma, const L3Tile& l3) {
    const std::uint32_t slots = l3.config().slots;
    if (dma.config().slots != slots)
        throw std::logic_error("check_slot_credits: " + dma.name() + " has " + std::to_string(dma.config().slots) +
                               " slot credits, " + l3.name() + " " + std::to_string(slots) + " slots");
    for (std::uint32_t s = 0; s < slots; ++s)
        if (dma.holds(s) && l3.live(s))
            throw std::logic_error("check_slot_credits: " + dma.name() + " holds slot " + std::to_string(s) +
                                   "'s credit while " + l3.name() + "'s slot is " + L3Tile::to_string(l3.state(s)));
}

}  // namespace sw::kpu::cycle
