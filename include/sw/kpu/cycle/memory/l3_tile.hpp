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
// THE BLOCKMOVER'S SIDE (#548, the join; connect_blockmover): the compute subsystem's BlockMover
// reads tiles for its Moves into L2 and writes its Writebacks' result tiles, on its own links, by
// address (a slot's bytes are at slot * slot_bytes), in 64-byte beats:
//   - a READ (l3_program.hpp's Read, a Move of the tile) is one request (a dense Move's whole tile)
//     or several (a strided Move's rows), each inside the slot, and done when its bytes have all
//     been served. Reads are served in the program's order, once the slot is READY with the tile
//     it is bound to (the producer before it into its
//     slot): the BlockMover never reads early, nor another residency's tile. A Release waits until
//     every Read of its tile before it in the program has been served: a Read is a consumer, as a
//     Store is;
//   - a WRITE (a Writeback) lands as a Load's beats do, into a FREE slot, under the slot's credit,
//     which is the BlockMover's: a Release pushes the slot's credit to its NEXT PRODUCER, the DMA
//     engine for a Load or Fill, the BlockMover for a Write.
// The read port serves one beat a cycle, a Store's or a Read's, taking turns when both wait; the
// write port lands one, a Load's or a Write's, taking turns the same way.
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
#include <sw/kpu/cycle/compute/l2_program.hpp>
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
            const std::uint32_t slot = program[i].slot;
            if (k == L3Action::Kind::Write) bm_writes_.push_back({i, slot, program[i].bytes});
            if (is_l3_producer(k)) last_load[slot] = i;
            else if (k == L3Action::Kind::Read) bm_reads_.push_back({i, slot, program[i].bytes, last_load[slot]});
            else commands_.push_back({i, program[i], last_load[slot], {}, false});
        }
        // A Release waits for the Reads of its tile before it; its credit goes to the slot's next producer.
        for (Command& c : commands_) {
            if (c.action.kind != L3Action::Kind::Release) continue;
            for (std::size_t r = 0; r < bm_reads_.size(); ++r)
                if (bm_reads_[r].ticket < c.ticket && bm_reads_[r].slot == c.action.slot && bm_reads_[r].producer == c.load)
                    c.reads.push_back(r);
            for (std::size_t i = c.ticket + 1; i < program.size(); ++i)
                if (program[i].slot == c.action.slot && is_l3_producer(program[i].kind)) {
                    c.to_blockmover = program[i].kind == L3Action::Kind::Write;
                    break;
                }
        }
        clear();
    }

    // in: the Loads' beats; eject: the Stores' beats, to the store buffer; credits: freed slots.
    void connect(Link<L3Beat>& in, Link<L3Beat>& eject, Link<SlotCredit>& credits) {
        in_ = &in;
        eject_ = &eject;
        credits_ = &credits;
    }
    // The BlockMover's side: its read requests, the beats back, its Writes' beats, and the credits of
    // the slots its Writes produce into.
    void connect_blockmover(Link<L3RowRead>& reads, Link<Beat>& data, Link<Beat>& writes, Link<SlotCredit>& credits) {
        bm_reads_in_ = &reads;
        bm_data_ = &data;
        bm_writes_in_ = &writes;
        bm_credits_ = &credits;
    }

    void evaluate(Cycle now) override {
        if (!in_ || !eject_ || !credits_) throw std::logic_error("cycle::L3Tile " + name() + ": not connected");
        bool moved = false;
        std::optional<Stall> wait;
        // The BlockMover's next read request, matched to the next Read in the program.
        if (bm_reads_in_ && !bm_req_ && bm_reads_in_->can_pop(now)) {
            const L3RowRead& q = bm_reads_in_->peek(now);
            if (bm_read_head_ >= bm_reads_.size())
                throw std::logic_error(where(now) + ": a read of " + std::to_string(q.bytes) + " bytes at " +
                                       std::to_string(q.address) + " from the BlockMover, past the program's Reads");
            // A Read is one request (a dense Move's whole tile) or several (a strided Move's rows), each
            // inside its slot and in whole beats; it is done when its bytes have all been served.
            const BmRead& r = bm_reads_[bm_read_head_];
            const std::uint64_t base = std::uint64_t(r.slot) * config_.slot_bytes;
            const std::string what = where(now) + ": a read of " + std::to_string(q.bytes) + " bytes at " +
                                     std::to_string(q.address) + " from the BlockMover, where action " +
                                     std::to_string(r.ticket) + " reads " + std::to_string(r.bytes) + " bytes of slot " +
                                     std::to_string(r.slot) + " (at " + std::to_string(base) + ")";
            // The offset first, then the length against what is left of the slot: no sum to wrap.
            if (q.address < base || q.address - base > config_.slot_bytes ||
                q.bytes > config_.slot_bytes - (q.address - base) || q.bytes == 0 || q.bytes % kL3BeatBytes ||
                (q.address - base) % kL3BeatBytes)
                throw std::logic_error(what + ": outside its slot, or not whole beats");
            if (bm_read_served_ + q.bytes > r.bytes)
                throw std::logic_error(what + ": " + std::to_string(bm_read_served_) + " served already, past the " +
                                       std::to_string(r.bytes) + " bytes action " + std::to_string(r.ticket) + " reads");
            bm_req_ = bm_reads_in_->pop(now);
            bm_sent_ = 0;
            moved = true;
        }
        // The read port: a Store's beat or a Read's, taking turns when both can go.
        std::optional<std::uint32_t> read_bank;
        std::optional<Stall> store_wait, read_wait;
        bool store_can = false, read_can = false;
        const Command* head = head_ < commands_.size() ? &commands_[head_] : nullptr;
        if (head) {
            if (!holds_its_tile(head->slot_of(), head->load) || !reads_done(*head)) store_wait = Stall::NoData;
            else if (head->action.kind == L3Action::Kind::Store) {
                if (eject_->can_push(now)) store_can = true;
                else store_wait = Stall::NoCredit;
            }
        }
        if (bm_req_) {
            const BmRead& r = bm_reads_[bm_read_head_];
            if (!holds_its_tile(r.slot, r.producer)) read_wait = Stall::NoData;
            else if (bm_data_->can_push(now)) read_can = true;
            else read_wait = Stall::NoCredit;
        }
        const bool serve_read = read_can && (!store_can || read_turn_);
        if (store_can && !serve_read) {
            const Command& c = *head;
            Slot& sl = slots_[c.action.slot];
            if (c.action.bytes != sl.tile_bytes)
                throw std::logic_error(where(now) + ": action " + std::to_string(c.ticket) + " stores " +
                                       std::to_string(c.action.bytes) + " bytes from slot " +
                                       std::to_string(c.action.slot) + ", which holds a tile of " +
                                       std::to_string(sl.tile_bytes));
            L3Beat b;
            b.id = ++beat_id_;
            b.ticket = c.ticket;
            b.slot = c.action.slot;
            b.offset = offset_;
            b.tile_bytes = c.action.bytes;
            b.bytes.assign(sl.bytes.begin() + offset_, sl.bytes.begin() + offset_ + kL3BeatBytes);
            read_bank = bank(offset_);
            eject_->push(now, std::move(b));
            offset_ += kL3BeatBytes;
            if (offset_ == c.action.bytes) {
                offset_ = 0;
                ++head_;
            }
            moved = true;
            read_turn_ = true;
        } else if (serve_read) {
            const BmRead& r = bm_reads_[bm_read_head_];
            const Slot& sl = slots_[r.slot];
            if (r.bytes > sl.tile_bytes)
                throw std::logic_error(where(now) + ": action " + std::to_string(r.ticket) + " reads " +
                                       std::to_string(r.bytes) + " bytes from slot " + std::to_string(r.slot) +
                                       ", which holds a tile of " + std::to_string(sl.tile_bytes));
            const std::uint32_t off =
                static_cast<std::uint32_t>(bm_req_->address - std::uint64_t(r.slot) * config_.slot_bytes) + bm_sent_;
            Beat b;
            b.id = ++beat_id_;
            b.address = bm_req_->address + bm_sent_;
            b.bytes.assign(sl.bytes.begin() + off, sl.bytes.begin() + off + kL3BeatBytes);
            b.tag = bm_req_->tag;
            b.first = bm_sent_ == 0;
            read_bank = bank(off);
            bm_sent_ += kL3BeatBytes;
            b.last = bm_sent_ >= bm_req_->bytes;
            bm_data_->push(now, std::move(b));
            if (bm_sent_ >= bm_req_->bytes) {   // the request is served; the Read when all its bytes are
                bm_read_served_ += bm_req_->bytes;
                bm_req_.reset();
                if (bm_read_served_ == r.bytes) {
                    reads_done_[bm_read_head_] = true;
                    ++bm_read_head_;
                    bm_read_served_ = 0;
                }
            }
            moved = true;
            read_turn_ = false;
        } else if (head && head->action.kind == L3Action::Kind::Release && !store_wait) {
            const Command& c = *head;
            Slot& sl = slots_[c.action.slot];
            Link<SlotCredit>* to = c.to_blockmover ? bm_credits_ : credits_;
            if (!to)
                throw std::logic_error(where(now) + ": action " + std::to_string(c.ticket) +
                                       " returns slot " + std::to_string(c.action.slot) +
                                       "'s credit to the BlockMover, which is not connected");
            if (to->can_push(now)) {
                sl.state = SlotState::Free;
                residencies_[sl.residency].released_at = now;
                to->push(now, SlotCredit{++credit_id_, c.action.slot});
                counters().complete();
                ++head_;
                moved = true;
            } else {
                store_wait = Stall::NoCredit;
            }
        }
        wait = store_wait ? store_wait : read_wait;

        // The write port: a Load's beat or a Write's, taking turns when both have landed, unless its
        // bank is being read.
        const bool load_has = in_->can_pop(now);
        const bool write_has = bm_writes_in_ && bm_writes_in_->can_pop(now);
        const bool take_write = write_has && (!load_has || write_turn_);
        if (take_write) {
            const Beat& w = bm_writes_in_->peek(now);
            const std::uint32_t off = static_cast<std::uint32_t>(w.address % config_.slot_bytes);
            if (read_bank && bank(off) == *read_bank) {
                ++bank_conflicts_;
            } else {
                write(now, blockmover_beat(now, bm_writes_in_->pop(now)));
                moved = true;
                write_turn_ = false;
            }
        } else if (load_has) {
            if (read_bank && bank(in_->peek(now).offset) == *read_bank) {
                ++bank_conflicts_;   // the read moved, so the cycle is busy; the write waits for the bank
            } else {
                write(now, in_->pop(now));
                moved = true;
                write_turn_ = true;
            }
        }
        if (moved) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override { return head_ == commands_.size() && !bm_req_; }

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
            if (!holds_its_tile(c.action.slot, c.load))
                d += ": waiting for its tile" + (c.load ? " (Load " + std::to_string(*c.load) + ")" : std::string(", which no Load places"));
            else if (!reads_done(c))
                d += ": waiting for the BlockMover's reads of its tile";
            else if (c.action.kind == L3Action::Kind::Store)
                d += ": ejected " + std::to_string(offset_) + " of " + std::to_string(c.action.bytes) + " bytes";
            d += ")";
        } else {
            d = "all " + std::to_string(commands_.size()) + " commands done";
        }
        if (!bm_reads_.empty() || !bm_writes_.empty())
            d += "; the BlockMover's reads " + std::to_string(bm_read_head_) + " of " + std::to_string(bm_reads_.size()) +
                 " served" + (bm_req_ ? " (one in progress, " + std::to_string(bm_sent_) + " bytes sent)" : std::string()) +
                 ", writes " + std::to_string(bm_write_head_) + " of " + std::to_string(bm_writes_.size()) + " landed";
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
        std::optional<std::size_t> load;   // the producer (Load, Fill or Write) before it into its slot
        std::vector<std::size_t> reads;    // a Release: the BlockMover's Reads of its tile before it
        bool to_blockmover;                // a Release: the slot's next producer is a Write
        [[nodiscard]] std::uint32_t slot_of() const { return action.slot; }
    };
    struct BmRead {
        std::size_t ticket;
        std::uint32_t slot, bytes;
        std::optional<std::size_t> producer;
    };
    struct BmWrite {
        std::size_t ticket;
        std::uint32_t slot, bytes;
    };
    // The slot holds the tile `producer` placed, all of it.
    [[nodiscard]] bool holds_its_tile(std::uint32_t slot, const std::optional<std::size_t>& producer) const {
        const Slot& s = slots_[slot];
        return producer && s.state == SlotState::Ready && s.ticket == *producer;
    }
    [[nodiscard]] bool reads_done(const Command& c) const {
        for (const std::size_t r : c.reads)
            if (!reads_done_[r]) return false;
        return true;
    }
    // A Write's beat from the BlockMover, by address, as the beat of its Write action into its slot.
    L3Beat blockmover_beat(Cycle now, Beat w) {
        const std::string beat = "the BlockMover's beat at " + std::to_string(w.address);
        if (bm_write_head_ >= bm_writes_.size())
            throw std::logic_error(where(now) + ": " + beat + ", past the program's Writes");
        const BmWrite& x = bm_writes_[bm_write_head_];
        const std::uint64_t base = std::uint64_t(x.slot) * config_.slot_bytes;
        if (w.address < base || w.address >= base + x.bytes || w.bytes.size() != kL3BeatBytes || (w.address - base) % kL3BeatBytes)
            throw std::logic_error(where(now) + ": " + beat + " (" + std::to_string(w.bytes.size()) +
                                   " bytes), where action " + std::to_string(x.ticket) + " writes " +
                                   std::to_string(x.bytes) + " bytes into slot " + std::to_string(x.slot) + " at " +
                                   std::to_string(base) + " in " + std::to_string(kL3BeatBytes) + "-byte beats");
        L3Beat b;
        b.id = w.id;
        b.ticket = x.ticket;
        b.slot = x.slot;
        b.offset = static_cast<std::uint32_t>(w.address - base);
        b.tile_bytes = x.bytes;
        b.bytes = std::move(w.bytes);
        if (++bm_write_beats_ == x.bytes / kL3BeatBytes) {
            bm_write_beats_ = 0;
            ++bm_write_head_;
        }
        return b;
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
        reads_done_.assign(bm_reads_.size(), false);
        bm_read_head_ = bm_write_head_ = 0;
        bm_write_beats_ = 0;
        bm_req_.reset();
        bm_sent_ = 0;
        bm_read_served_ = 0;
        read_turn_ = write_turn_ = false;
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
    // The BlockMover's side.
    std::vector<BmRead> bm_reads_;
    std::vector<BmWrite> bm_writes_;
    std::vector<bool> reads_done_;
    std::size_t bm_read_head_ = 0, bm_write_head_ = 0, bm_write_beats_ = 0;
    std::optional<L3RowRead> bm_req_;
    std::uint32_t bm_sent_ = 0;          // the current request's bytes sent
    std::uint32_t bm_read_served_ = 0;   // the head Read's bytes served by its finished requests
    bool read_turn_ = false, write_turn_ = false;   // the BlockMover's turn at the read / write port
    Link<L3RowRead>* bm_reads_in_ = nullptr;
    Link<Beat>* bm_data_ = nullptr;
    Link<Beat>* bm_writes_in_ = nullptr;
    Link<SlotCredit>* bm_credits_ = nullptr;
};

}  // namespace sw::kpu::cycle
