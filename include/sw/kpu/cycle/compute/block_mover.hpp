#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/block_mover.hpp
// The BlockMover on the cycle core (#542; docs/plans/cycle-based-kpu-prototype.md §5.1, step 2a):
// L3 -> L2 by rows, at its bus width per beat, strided, under the L2 slot's credit; and the
// writeback, L2 -> L3. It executes the program's Moves and Writebacks (l2_program.hpp) in order, one
// at a time.
//
// A MOVE waits for its slot's credit (the BlockMover holds every slot's at the start, and a
// Release returns one), spends it, and then, each clock:
//   - asks L3 for its next row (row k at l3_address + k * l3_pitch: the stride), while it has a row
//     buffer free and the request link a credit;
//   - takes one beat of `bus_bytes` from L3 into the row it belongs to (one beat a clock: the bus
//     width);
//   - sends its oldest complete row to the L2 slot, while the link to L2 has a credit.
// It ends when its last row has gone to L2.
//
// A WRITEBACK asks L2 for the slot's rows in order, with at most `read_window` requested and not
// yet sent on to L3 (the mover's requests in flight); takes each returning row into a row buffer
// while one is free; and writes the oldest out to L3 one beat a clock, at l3_address + k * l3_pitch.
// L2 holds its reads until the slot is READY with the tile: the BlockMover never asks early.
//
// COUNTERS. A Move or Writeback is an item: it arrives when it starts and completes when its last
// row (or beat) is sent. The mover is busy in a clock it moves anything; otherwise stalled for a
// credit (the slot's, or a link's) or for data (L3's beats, L2's rows).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

class BlockMover : public Component {
public:
    struct Config {
        std::uint32_t slots = 2;
        std::uint32_t rows_per_slot = 16;
        std::uint32_t row_bytes = 256;     // the L2's row
        std::uint32_t bus_bytes = 64;      // one beat a clock
        std::uint32_t row_buffers = 2;     // rows assembled or held at once
        std::uint32_t read_window = 8;     // a Writeback's rows requested from L2, not yet sent on to L3
    };

    BlockMover(std::string name, Domain& d, Config cfg, const L2Program& program)
        : Component(std::move(name), d), config_(cfg) {
        if (cfg.slots == 0 || cfg.rows_per_slot == 0 || cfg.row_bytes == 0 || cfg.bus_bytes == 0 ||
            cfg.row_bytes % cfg.bus_bytes != 0 || cfg.row_buffers == 0 || cfg.read_window == 0)
            throw std::invalid_argument("cycle::BlockMover " + this->name() +
                                        ": slots, rows, row buffers and the read window must be positive, and a row "
                                        "whole beats of the bus");
        validate_l2_program(program, cfg.slots, cfg.rows_per_slot, cfg.row_bytes, "cycle::BlockMover " + this->name());
        for (std::size_t i = 0; i < program.size(); ++i)
            if (program[i].kind != L2Action::Kind::Release) actions_.push_back({i, program[i]});
        clear();
    }

    void connect(Link<L3RowRead>& l3_reads, Link<Beat>& l3_data, Link<Beat>& l3_writes, Link<L2RowWrite>& l2_writes,
                 Link<L2RowRead>& l2_reads, Link<L2RowData>& l2_data, Link<SlotCredit>& credits) {
        l3_reads_ = &l3_reads;
        l3_data_ = &l3_data;
        l3_writes_ = &l3_writes;
        l2_writes_ = &l2_writes;
        l2_reads_ = &l2_reads;
        l2_data_ = &l2_data;
        credits_ = &credits;
    }

    void evaluate(Cycle now) override {
        if (!l3_reads_ || !l3_data_ || !l3_writes_ || !l2_writes_ || !l2_reads_ || !l2_data_ || !credits_)
            throw std::logic_error(where(now) + ": not connected");
        while (credits_->can_pop(now)) {
            const SlotCredit c = credits_->pop(now);
            if (c.slot >= config_.slots || held_[c.slot])
                throw std::logic_error(where(now) + ": a credit for slot " + std::to_string(c.slot) + ", which " +
                                       (c.slot >= config_.slots ? "does not exist" : "it already holds"));
            held_[c.slot] = true;
        }
        if (head_ == actions_.size()) return;
        const Action& a = actions_[head_];
        bool moved = false;
        std::optional<Stall> wait;
        const auto waiting = [&](Stall s) { if (!wait) wait = s; };

        if (!started_) {
            if (a.action.kind == L2Action::Kind::Move) {
                if (!held_[a.action.slot]) {
                    counters().stall(Stall::NoCredit);   // the slot's credit: its last tile not yet released
                    return;
                }
                held_[a.action.slot] = false;
            }
            started_ = true;
            counters().accept();
        }
        if (a.action.kind == L2Action::Kind::Move) move(now, a.action, moved, waiting);
        else writeback(now, a.action, moved, waiting);

        if (done_ == a.action.rows) {
            counters().complete();
            ++head_;
            started_ = false;
            requested_ = done_ = 0;
            rows_.clear();
            moved = true;
        }
        if (moved) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override { return head_ == actions_.size(); }

    void check(Cycle now) const override {
        if (rows_.size() > config_.row_buffers)
            throw std::logic_error(where(now) + ": " + std::to_string(rows_.size()) + " rows held in " +
                                   std::to_string(config_.row_buffers) + " row buffers");
        if (head_ < actions_.size() && actions_[head_].action.kind == L2Action::Kind::Writeback &&
            requested_ - done_ > config_.read_window)
            throw std::logic_error(where(now) + ": " + std::to_string(requested_ - done_) +
                                   " rows requested and not sent on, past the read window of " +
                                   std::to_string(config_.read_window));
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        if (head_ == actions_.size()) return "all " + std::to_string(actions_.size()) + " transfers done";
        const L2Action& a = actions_[head_].action;
        std::string d = std::string(to_string(a.kind)) + " " + std::to_string(head_ + 1) + " of " +
                        std::to_string(actions_.size()) + " (slot " + std::to_string(a.slot) + ", " + a.tile.to_string();
        if (!started_) d += ": waiting for the slot's credit";
        else d += ": " + std::to_string(done_) + " of " + std::to_string(a.rows) + " rows done, " +
                  std::to_string(requested_) + " requested";
        d += "); credits held:";
        for (std::uint32_t s = 0; s < config_.slots; ++s)
            if (held_[s]) d += " " + std::to_string(s);
        return d;
    }

    [[nodiscard]] const Config& config() const { return config_; }
    // Holds slot `slot`'s credit: the slot is free for its next Move.
    [[nodiscard]] bool holds(std::uint32_t slot) const { return held_.at(slot); }

private:
    struct Action {
        std::size_t ticket;
        L2Action action;
    };
    struct Row {
        std::uint32_t row = 0;
        std::vector<std::uint8_t> bytes;   // filled so far (a Move), or the whole row (a Writeback)
        std::uint32_t sent = 0;            // a Writeback's bytes written out to L3
    };

    [[nodiscard]] std::string where(Cycle now) const {
        return "cycle::BlockMover " + name() + " at cycle " + std::to_string(now);
    }
    [[nodiscard]] std::uint64_t l3_row(const L2Action& a, std::uint32_t row) const {
        return a.l3_address + std::uint64_t(row) * a.l3_pitch;
    }

    template <class Waiting>
    void move(Cycle now, const L2Action& a, bool& moved, Waiting&& waiting) {
        // The oldest complete row to L2.
        if (!rows_.empty() && rows_.front().bytes.size() == config_.row_bytes) {
            if (l2_writes_->can_push(now)) {
                Row r = std::move(rows_.front());
                rows_.pop_front();
                l2_writes_->push(now, L2RowWrite{++id_, a.slot, r.row, a.rows, a.tile, std::move(r.bytes)});
                ++done_;
                moved = true;
            } else {
                waiting(Stall::NoCredit);
            }
        }
        // One beat from L3, into the oldest row still filling: L3 returns them in order.
        if (l3_data_->can_pop(now)) {
            Row* filling = nullptr;
            for (Row& r : rows_)
                if (r.bytes.size() < config_.row_bytes) {
                    filling = &r;
                    break;
                }
            const Beat& b = l3_data_->peek(now);
            const std::uint64_t want = filling ? l3_row(a, filling->row) + filling->bytes.size() : 0;
            if (!filling || b.tag != filling->row || b.address != want || b.bytes.size() != config_.bus_bytes)
                throw std::logic_error(where(now) + ": a beat from L3 at " + std::to_string(b.address) + " (row " +
                                       std::to_string(b.tag) + ", " + std::to_string(b.bytes.size()) + " bytes), " +
                                       (filling ? "expected " + std::to_string(config_.bus_bytes) + " bytes at " +
                                                      std::to_string(want) + " for row " + std::to_string(filling->row)
                                                : std::string("with no row requested")));
            const Beat beat = l3_data_->pop(now);
            filling->bytes.insert(filling->bytes.end(), beat.bytes.begin(), beat.bytes.end());
            moved = true;
        } else {
            for (const Row& r : rows_)
                if (r.bytes.size() < config_.row_bytes) {
                    waiting(Stall::NoData);   // a requested row's beats have not come
                    break;
                }
        }
        // The next row's request.
        if (requested_ < a.rows && rows_.size() < config_.row_buffers) {
            if (l3_reads_->can_push(now)) {
                l3_reads_->push(now, L3RowRead{++id_, l3_row(a, requested_), config_.row_bytes, requested_});
                rows_.push_back(Row{requested_, {}, 0});
                ++requested_;
                moved = true;
            } else {
                waiting(Stall::NoCredit);
            }
        }
    }

    template <class Waiting>
    void writeback(Cycle now, const L2Action& a, bool& moved, Waiting&& waiting) {
        // One beat of the oldest row out to L3.
        if (!rows_.empty()) {
            if (l3_writes_->can_push(now)) {
                Row& r = rows_.front();
                Beat b;
                b.id = ++id_;
                b.address = l3_row(a, r.row) + r.sent;
                b.bytes.assign(r.bytes.begin() + r.sent, r.bytes.begin() + r.sent + config_.bus_bytes);
                b.tag = r.row;
                b.first = r.sent == 0;
                r.sent += config_.bus_bytes;
                b.last = r.sent == config_.row_bytes;
                l3_writes_->push(now, std::move(b));
                if (r.sent == config_.row_bytes) {
                    rows_.pop_front();
                    ++done_;
                }
                moved = true;
            } else {
                waiting(Stall::NoCredit);
            }
        }
        // A row from L2, while a row buffer is free; they return in order.
        if (l2_data_->can_pop(now) && rows_.size() < config_.row_buffers) {
            const L2RowData& d = l2_data_->peek(now);
            const std::uint32_t want = done_ + static_cast<std::uint32_t>(rows_.size());
            if (d.slot != a.slot || !(d.tile == a.tile) || d.row != want || d.bytes.size() != config_.row_bytes)
                throw std::logic_error(where(now) + ": a row from L2 (slot " + std::to_string(d.slot) + ", " +
                                       d.tile.to_string() + " row " + std::to_string(d.row) + "), expected row " +
                                       std::to_string(want) + " of " + a.tile.to_string() + " in slot " +
                                       std::to_string(a.slot));
            L2RowData data = l2_data_->pop(now);
            rows_.push_back(Row{data.row, std::move(data.bytes), 0});
            moved = true;
        } else if (rows_.empty() && requested_ > done_) {
            waiting(Stall::NoData);
        }
        // The next row's request, within the read window.
        if (requested_ < a.rows && requested_ - done_ < config_.read_window) {
            if (l2_reads_->can_push(now)) {
                l2_reads_->push(now, L2RowRead{++id_, a.slot, requested_, a.tile, requested_ == 0, requested_ + 1 == a.rows});
                ++requested_;
                moved = true;
            } else {
                waiting(Stall::NoCredit);
            }
        }
    }

    void clear() {
        held_.assign(config_.slots, true);
        head_ = 0;
        started_ = false;
        requested_ = done_ = 0;
        rows_.clear();
        id_ = 0;
    }

    Config config_;
    std::vector<Action> actions_;
    std::vector<bool> held_;     // the slots whose credit the mover holds
    std::size_t head_ = 0;
    bool started_ = false;
    std::uint32_t requested_ = 0, done_ = 0;
    std::deque<Row> rows_;
    std::uint64_t id_ = 0;
    Link<L3RowRead>* l3_reads_ = nullptr;
    Link<Beat>* l3_data_ = nullptr;
    Link<Beat>* l3_writes_ = nullptr;
    Link<L2RowWrite>* l2_writes_ = nullptr;
    Link<L2RowRead>* l2_reads_ = nullptr;
    Link<L2RowData>* l2_data_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
};

// The BlockMover and its L2 together, every cycle: the mover never holds a slot's credit while
// the L2's slot is live.
inline void check_l2_slot_credits(const BlockMover& bm, const L2& l2) {
    const std::uint32_t slots = l2.config().slots;
    if (bm.config().slots != slots)
        throw std::logic_error("check_l2_slot_credits: " + bm.name() + " has " + std::to_string(bm.config().slots) +
                               " slot credits, " + l2.name() + " " + std::to_string(slots) + " slots");
    for (std::uint32_t s = 0; s < slots; ++s)
        if (bm.holds(s) && l2.live(s))
            throw std::logic_error("check_l2_slot_credits: " + bm.name() + " holds slot " + std::to_string(s) +
                                   "'s credit while " + l2.name() + "'s slot is " +
                                   sw::kpu::timing::SlotPool::name(l2.state(s)));
}

}  // namespace sw::kpu::cycle
