#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/block_mover.hpp
// The BlockMover on the cycle core (#542; docs/plans/cycle-based-kpu-prototype.md §5.1, step 2a):
// L3 -> L2 by rows, at its bus width per beat, strided, under the L2 slot's credit; and the
// writeback, L2 -> L3. It executes the program's Moves and Writebacks (l2_program.hpp) in order, one
// at a time.
//
// A MOVE waits for its slot's credit (the BlockMover holds every slot's at the start but those whose
// first producer is a Result, the Streamer's; a Release returns one), spends it, and then, each clock:
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
// THE DENSE LAYOUTS (#548, the join; l2_program.hpp): L3 holds the tile densely, row-major, as the
// memory subsystem places it, and L2 one row a lane. The transform is the BlockMover's, on its
// write into L2 (docs/01-architecture/memory-hierarchy-roles.md), through a TILE BUFFER of one L2
// slot's bytes:
//   - a dense Move asks L3 for the whole tile in one request, and takes its beats into the tile
//     buffer, one a clock. A DenseRows lane (the tile's row m) goes to L2 as soon as its bytes are
//     in; a DenseColumns lane (column j, B's transpose) needs every row, so its lanes go only once
//     the whole tile is in: the transpose's latency, built as state;
//   - a dense Writeback asks L2 for the slot's rows, as a strided one does, packs each row's first
//     tile_cols elements into the tile buffer, and writes the buffer to L3 one beat a clock, each
//     beat as soon as its bytes are in, the last padded with zeros to the 64-byte beat.
//
// THE L3 SLOT'S CREDIT (#548, the join; connect_l3_credits): a Writeback into an L3 tile's slot is
// that slot's producer, so it waits for the slot's credit, from the L3 tile, before it starts, and
// spends it. The L3 tile returns it when its next producer is a Writeback (l3_tile.hpp).
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
        for (std::size_t i = 0; i < program.size(); ++i) {
            const L2Action& a = program[i];
            // A Writeback into an L3 tile's slot is a Write of a whole tile, its beats contiguous
            // (l3_tile.hpp): strided rows must be packed, a row apart.
            if (a.kind == L2Action::Kind::Writeback && a.l3_slot && a.layout == L2Action::Layout::Strided && a.rows > 1 &&
                a.l3_pitch != cfg.row_bytes)
                throw std::invalid_argument("cycle::BlockMover " + this->name() + ": action " + std::to_string(i) +
                                            " writes back into L3 slot " + std::to_string(*a.l3_slot) + " rows " +
                                            std::to_string(a.l3_pitch) + " bytes apart; an L3 Write is a whole tile, "
                                            "its rows " + std::to_string(cfg.row_bytes) + " bytes apart");
            if ((a.kind != L2Action::Kind::Move && a.kind != L2Action::Kind::Writeback) ||
                a.layout == L2Action::Layout::Strided)
                continue;
            if (a.dense_bytes() > std::uint64_t(cfg.rows_per_slot) * cfg.row_bytes || a.dense_bytes() % cfg.bus_bytes != 0)
                throw std::invalid_argument("cycle::BlockMover " + this->name() + ": action " + std::to_string(i) +
                                            "'s dense tile is " + std::to_string(a.dense_bytes()) +
                                            " bytes, for a tile buffer of one slot (" +
                                            std::to_string(cfg.rows_per_slot * cfg.row_bytes) + ") in " +
                                            std::to_string(cfg.bus_bytes) + "-byte beats");
        }
        program_ = program;
        for (std::size_t i = 0; i < program.size(); ++i)
            if (program[i].kind == L2Action::Kind::Move || program[i].kind == L2Action::Kind::Writeback)
                actions_.push_back({i, program[i]});   // the Feeds are the Streamer's, the Releases the L2's
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
    // The L3 tile's credits for the slots the Writebacks produce into; `held`: the slots whose credit
    // the mover holds at the start (their first producer is a Writeback).
    void connect_l3_credits(Link<SlotCredit>& credits, std::vector<bool> held) {
        l3_credits_ = &credits;
        l3_initial_ = std::move(held);
        l3_held_ = l3_initial_;
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
        while (l3_credits_ && l3_credits_->can_pop(now)) {
            const SlotCredit c = l3_credits_->pop(now);
            if (c.slot >= l3_held_.size() || l3_held_[c.slot])
                throw std::logic_error(where(now) + ": an L3 credit for slot " + std::to_string(c.slot) + ", which " +
                                       (c.slot >= l3_held_.size() ? "does not exist" : "it already holds"));
            l3_held_[c.slot] = true;
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
            if (a.action.kind == L2Action::Kind::Writeback && l3_credits_ && a.action.l3_slot) {
                const std::uint32_t s3 = *a.action.l3_slot;
                if (s3 >= l3_held_.size())
                    throw std::logic_error(where(now) + ": a Writeback into L3 slot " + std::to_string(s3) + " of " +
                                           std::to_string(l3_held_.size()));
                if (!l3_held_[s3]) {
                    counters().stall(Stall::NoCredit);   // the L3 slot's credit: its last tile not yet released
                    return;
                }
                l3_held_[s3] = false;
            }
            started_ = true;
            counters().accept();
        }
        const bool dense = a.action.layout != L2Action::Layout::Strided;
        if (a.action.kind == L2Action::Kind::Move) (dense ? move_dense(now, a.action, moved, waiting) : move(now, a.action, moved, waiting));
        else (dense ? writeback_dense(now, a.action, moved, waiting) : writeback(now, a.action, moved, waiting));

        // A dense Writeback ends with its last beat to L3; a dense Move with its last lane and its
        // whole response taken (the padding beats too: a next Move must not receive them); every
        // other transfer with its last row.
        const bool finished = !dense ? done_ == a.action.rows
                            : a.action.kind == L2Action::Kind::Writeback ? sent_ == a.action.dense_bytes()
                                                                         : done_ == a.action.rows && tile_.size() == a.action.dense_bytes();
        if (finished) {
            counters().complete();
            ++head_;
            started_ = false;
            requested_ = done_ = 0;
            sent_ = 0;
            rows_.clear();
            tile_.clear();
            moved = true;
        }
        if (moved) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override { return head_ == actions_.size(); }

    void check(Cycle now) const override {
        if (tile_.size() > std::size_t(config_.rows_per_slot) * config_.row_bytes)
            throw std::logic_error(where(now) + ": " + std::to_string(tile_.size()) + " bytes in a tile buffer of " +
                                   std::to_string(config_.rows_per_slot * config_.row_bytes));
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
    // Holds L3 slot `slot`'s credit: the slot is free for its next Writeback.
    [[nodiscard]] bool holds_l3(std::uint32_t slot) const { return slot < l3_held_.size() && l3_held_[slot]; }

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

    // A dense Move: the whole tile from L3 into the tile buffer, its lanes out to L2.
    template <class Waiting>
    void move_dense(Cycle now, const L2Action& a, bool& moved, Waiting&& waiting) {
        // Sizes checked against the row and the tile buffer when the program was (validate_l2_program).
        const auto lane = static_cast<std::uint32_t>(a.lane_bytes());
        const std::uint32_t e = a.element_bytes, whole = a.tile_rows * a.tile_cols * e;
        const bool columns = a.layout == L2Action::Layout::DenseColumns;
        // The next lane to L2, once its bytes are in: a row's own, or (a column) the whole tile's.
        if (done_ < a.rows) {
            const std::size_t need = columns ? whole : std::size_t(done_ + 1) * lane;
            if (tile_.size() >= need) {
                if (l2_writes_->can_push(now)) {
                    std::vector<std::uint8_t> row(config_.row_bytes, 0);
                    if (columns)
                        for (std::uint32_t r = 0; r < a.tile_rows; ++r)
                            for (std::uint32_t b = 0; b < e; ++b)
                                row[std::size_t(r) * e + b] = tile_[(std::size_t(r) * a.tile_cols + done_) * e + b];
                    else
                        std::copy(tile_.begin() + std::size_t(done_) * lane, tile_.begin() + std::size_t(done_ + 1) * lane,
                                  row.begin());
                    l2_writes_->push(now, L2RowWrite{++id_, a.slot, done_, a.rows, a.tile, std::move(row)});
                    ++done_;
                    moved = true;
                } else {
                    waiting(Stall::NoCredit);
                }
            }
        }
        // One beat from L3 into the tile buffer: they return in order.
        if (l3_data_->can_pop(now)) {
            const Beat& b = l3_data_->peek(now);
            const std::uint64_t want = a.l3_address + tile_.size();
            if (requested_ == 0 || b.tag != 0 || b.address != want || b.bytes.size() != config_.bus_bytes ||
                tile_.size() + b.bytes.size() > a.dense_bytes())
                throw std::logic_error(where(now) + ": a beat from L3 at " + std::to_string(b.address) + " (" +
                                       std::to_string(b.bytes.size()) + " bytes), expected " +
                                       std::to_string(config_.bus_bytes) + " bytes at " + std::to_string(want) +
                                       " of the dense tile " + a.tile.to_string());
            const Beat beat = l3_data_->pop(now);
            tile_.insert(tile_.end(), beat.bytes.begin(), beat.bytes.end());
            moved = true;
        } else if (requested_ && tile_.size() < a.dense_bytes()) {
            waiting(Stall::NoData);
        }
        // The tile's one request.
        if (requested_ == 0) {
            if (l3_reads_->can_push(now)) {
                l3_reads_->push(now, L3RowRead{++id_, a.l3_address, static_cast<std::uint32_t>(a.dense_bytes()), 0});
                requested_ = 1;
                moved = true;
            } else {
                waiting(Stall::NoCredit);
            }
        }
    }

    // A dense Writeback: the slot's rows from L2, packed into the tile buffer, out to L3 by beats.
    template <class Waiting>
    void writeback_dense(Cycle now, const L2Action& a, bool& moved, Waiting&& waiting) {
        const auto lane = static_cast<std::uint32_t>(a.lane_bytes());
        const std::uint32_t bus = config_.bus_bytes, whole = a.tile_rows * a.tile_cols * a.element_bytes;
        // The next beat out to L3, once its bytes are in (the last one padded with zeros).
        if (sent_ < a.dense_bytes()) {
            const bool all_in = done_ == a.rows;
            if (tile_.size() >= std::size_t(sent_) + bus || (all_in && sent_ < a.dense_bytes())) {
                if (l3_writes_->can_push(now)) {
                    if (all_in && tile_.size() < a.dense_bytes()) tile_.resize(a.dense_bytes(), 0);
                    Beat b;
                    b.id = ++id_;
                    b.address = a.l3_address + sent_;
                    b.bytes.assign(tile_.begin() + sent_, tile_.begin() + sent_ + bus);
                    b.tag = 0;
                    b.first = sent_ == 0;
                    sent_ += bus;
                    b.last = sent_ == a.dense_bytes();
                    l3_writes_->push(now, std::move(b));
                    moved = true;
                } else {
                    waiting(Stall::NoCredit);
                }
            }
        }
        // A row from L2: its first tile_cols elements into the tile buffer; they return in order.
        if (l2_data_->can_pop(now)) {
            const L2RowData& d = l2_data_->peek(now);
            if (d.slot != a.slot || !(d.tile == a.tile) || d.row != done_ || d.bytes.size() != config_.row_bytes)
                throw std::logic_error(where(now) + ": a row from L2 (slot " + std::to_string(d.slot) + ", " +
                                       d.tile.to_string() + " row " + std::to_string(d.row) + "), expected row " +
                                       std::to_string(done_) + " of " + a.tile.to_string() + " in slot " +
                                       std::to_string(a.slot));
            const L2RowData data = l2_data_->pop(now);
            tile_.insert(tile_.end(), data.bytes.begin(), data.bytes.begin() + lane);
            ++done_;
            if (done_ == a.rows && tile_.size() != whole)
                throw std::logic_error(where(now) + ": " + std::to_string(tile_.size()) + " bytes packed of a " +
                                       std::to_string(whole) + "-byte tile");
            moved = true;
        } else if (requested_ > done_) {
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
        for (std::uint32_t s = 0; s < config_.slots; ++s) held_[s] = !streamer_holds_at_start(program_, s);
        head_ = 0;
        started_ = false;
        requested_ = done_ = 0;
        sent_ = 0;
        rows_.clear();
        tile_.clear();
        id_ = 0;
        l3_held_ = l3_initial_;
    }

    Config config_;
    L2Program program_;
    std::vector<Action> actions_;
    std::vector<bool> held_;     // the slots whose credit the mover holds
    std::size_t head_ = 0;
    bool started_ = false;
    std::uint32_t requested_ = 0, done_ = 0;
    std::deque<Row> rows_;
    std::vector<std::uint8_t> tile_;   // a dense transfer's tile buffer
    std::uint32_t sent_ = 0;           // a dense Writeback's bytes written out to L3
    std::uint64_t id_ = 0;
    Link<L3RowRead>* l3_reads_ = nullptr;
    Link<Beat>* l3_data_ = nullptr;
    Link<Beat>* l3_writes_ = nullptr;
    Link<L2RowWrite>* l2_writes_ = nullptr;
    Link<L2RowRead>* l2_reads_ = nullptr;
    Link<L2RowData>* l2_data_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
    Link<SlotCredit>* l3_credits_ = nullptr;
    std::vector<bool> l3_initial_, l3_held_;   // the L3 slots whose credit the mover holds
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
