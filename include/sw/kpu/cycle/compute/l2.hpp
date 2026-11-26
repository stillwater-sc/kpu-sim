#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/l2.hpp
// A compute tile's L2 on the cycle core (#542; docs/plans/cycle-based-kpu-prototype.md §5.1, step
// 2a): banks of rows, read and write ports per bank, a read latency, in its own clock domain (250
// MHz by default). It holds the bytes it was written. Step 2b (#543) adds the Streamer's reads.
//
// ROWS AND BANKS. The L2 is `slots` x `rows_per_slot` rows of `row_bytes`, numbered slot-major; row
// n is in bank n mod `banks`, at index n / `banks` in it (docs/plans/banked-l2-at-lca.md §3: slot
// s's row k is in bank (s * rows_per_slot + k) mod banks). Each L2 clock, a bank serves at most
// `read_ports` reads and `write_ports` writes. Requests are served in arrival order, reads and
// writes each: one whose bank has no port left this clock waits, and so does every request behind
// it (a stall, counted on its bank).
//
// TWO WRITERS: the BlockMover's Moves and the Combiner's result rows (connect_combiner, #545). They
// share the banks' write ports, the Combiner's first: the result path cannot wait, so the
// BlockMover writes in the ports it leaves free, as in the RTL.
//
// TWO READERS: the BlockMover's Writebacks and the Streamer's commands (connect_streamer), each with
// its own request and data links, served in its own order. They share the banks' read ports: each
// clock the readers take turns at the ports, the first in turn alternating clock by clock.
//
// THE READ LATENCY is state, not a formula: a read served at clock c is in the read pipeline until
// clock c + read_latency, when its row leaves on the data link. While a row cannot leave (the link
// has no credit) it waits, and no new read is served: the pipeline never holds more than a bank's
// ports times the latency. A read is IN FLIGHT from the clock it is served to the clock its row
// leaves, and each bank counts its reads in flight every clock: their integral and their maximum
// answer #495's question (how many requests must be outstanding to keep up) by measurement. Each
// reader has its own pipeline, held only by its own data link.
//
// SLOTS follow #387's protocol (timing/slot_pool.hpp, l2_program.hpp): a Move's first row ACQUIREs
// the slot, its last SIGNALs it READY and sends the Streamer the tile-ready token (SlotReady); a
// consumer's first read (a Writeback's, or a Streamer command's of one operand) TAKEs it (READING)
// and its last row's departure is DONE; a Release frees it and pushes its credit to the slot's next
// producer: the BlockMover for a Move, the Streamer for a Result (l2_program.hpp). A Result's rows
// (the Combiner's) ACQUIRE and SIGNAL the slot as a Move's do. A row of
// another tile into a live slot, or a transition from the wrong state, is a SlotFault. A read waits,
// with no fault, until its slot holds its tile, READY: the tile-ready token. The L2 executes the
// program's Releases in order, one at a time; each waits until its slot is READY with its tile and
// every consumer (Writeback and Feed) of that tile before it in the program is done.
//
// COUNTERS. A tile is an item: it arrives when its slot is ACQUIREd and completes when its Release
// frees it, so the occupancy is the live slots. The L2 is busy in a clock it moves anything (a row
// written or read, a row leaving, a slot freed) or holds reads in its pipeline; otherwise it is
// stalled on what its head request waits for: a bank (bank busy), its tile (no data), or a credit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/timing/slot_pool.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

using SlotFault = sw::kpu::timing::SlotFault;

class L2 : public Component {
public:
    struct Config {
        std::uint32_t slots = 2;
        std::uint32_t rows_per_slot = 16;
        std::uint32_t banks = 4;
        std::uint32_t row_bytes = 256;
        std::uint32_t read_latency = 4;   // L2 clocks from a read served to its row leaving
        std::uint32_t read_ports = 2;     // per bank, per clock
        std::uint32_t write_ports = 1;
    };
    using State = sw::kpu::timing::SlotPool::State;
    // Each bank's accesses, waits and reads in flight, over the run.
    struct BankStats {
        std::uint64_t reads = 0, writes = 0;
        std::uint64_t stalls = 0;               // clocks the head request waited for this bank's port
        std::uint64_t in_flight_integral = 0;   // the sum over clocks of the reads in flight
        std::uint64_t max_in_flight = 0;
    };

    L2(std::string name, Domain& d, Config cfg, const L2Program& program)
        : Component(std::move(name), d), config_(cfg) {
        if (cfg.slots == 0 || cfg.rows_per_slot == 0 || cfg.banks == 0 || cfg.row_bytes == 0 || cfg.read_latency == 0 ||
            cfg.read_ports == 0 || cfg.write_ports == 0)
            throw std::invalid_argument("cycle::L2 " + this->name() +
                                        ": slots, rows, banks, row bytes, the read latency and the ports must be positive");
        validate_l2_program(program, cfg.slots, cfg.rows_per_slot, cfg.row_bytes, "cycle::L2 " + this->name());
        // Each Release is bound to the Move before it into its slot, and waits for that tile's
        // consumers before it in the program.
        // Each Move's Feeds are what its tile-ready token admits.
        std::vector<std::uint32_t> consumers(cfg.slots, 0);
        feeds_.assign(cfg.slots, {});
        for (std::size_t i = 0; i < program.size(); ++i) {
            const L2Action& a = program[i];
            if (is_producer(a.kind)) {
                consumers[a.slot] = 0;
                feeds_[a.slot].push_back(0);
            } else if (a.kind == L2Action::Kind::Writeback || a.kind == L2Action::Kind::Feed) {
                ++consumers[a.slot];
                if (a.kind == L2Action::Kind::Feed) ++feeds_[a.slot].back();
            } else {
                // Its credit goes to the slot's next producer: the Streamer's for a Result.
                bool to_streamer = false;
                for (std::size_t n = i + 1; n < program.size(); ++n)
                    if (program[n].slot == a.slot && is_producer(program[n].kind)) {
                        to_streamer = program[n].kind == L2Action::Kind::Result;
                        break;
                    }
                releases_.push_back({i, a.slot, a.tile, consumers[a.slot], to_streamer});
            }
        }
        clear();
    }

    // The BlockMover's side. writes: a Move's rows; reads: a Writeback's row requests; data: their
    // rows back; credits: freed slots.
    void connect(Link<L2RowWrite>& writes, Link<L2RowRead>& reads, Link<L2RowData>& data, Link<SlotCredit>& credits) {
        writes_ = &writes;
        readers_[0].reads = &reads;
        readers_[0].data = &data;
        credits_ = &credits;
    }
    // The Streamer's side (#543). reads: its commands' row requests; data: their rows back; ready:
    // the tile-ready tokens.
    // credits: the result slots' credits, for the Streamer's commands that produce results (#545).
    void connect_streamer(Link<L2RowRead>& reads, Link<L2RowData>& data, Link<SlotReady>& ready,
                          Link<SlotCredit>* credits = nullptr) {
        readers_[1].reads = &reads;
        readers_[1].data = &data;
        ready_ = &ready;
        st_credits_ = credits;
    }
    // The Combiner's result rows (#545).
    void connect_combiner(Link<L2RowWrite>& writes) {
        result_writes_ = &writes;
    }

    void evaluate(Cycle now) override {
        if (!writes_ || !readers_[0].reads || !credits_) throw std::logic_error(where(now) + ": not connected");
        std::fill(reads_now_.begin(), reads_now_.end(), 0);
        std::fill(writes_now_.begin(), writes_now_.end(), 0);
        bool moved = false;
        std::optional<Stall> wait;
        const auto waiting = [&](Stall s) { if (!wait) wait = s; };

        // Each reader's rows whose latency is over leave, in order, while its data link has a credit.
        for (Reader& rd : readers_) {
            rd.held = false;
            if (!rd.reads) continue;
            while (!rd.pipe.empty() && rd.pipe.front().ready_at <= now) {
                if (!rd.data->can_push(now)) {
                    rd.held = true;
                    waiting(Stall::NoCredit);
                    break;
                }
                InFlight r = std::move(rd.pipe.front());
                rd.pipe.pop_front();
                --in_flight_[r.bank];
                --rd.in_flight[r.bank];
                if (r.last) {
                    pool_.done(r.data.slot, r.data.tile);
                    ++consumers_done_[r.data.slot];
                }
                rd.data->push(now, std::move(r.data));
                moved = true;
            }
        }

        // Writes, each writer's in arrival order, while the head's bank has a write port this clock:
        // the Combiner's first, then the BlockMover's.
        for (Link<L2RowWrite>* writes : {result_writes_, writes_}) {
            while (writes && writes->can_pop(now)) {
                const L2RowWrite& w = writes->peek(now);
                check_row(now, w.slot, w.row, "the write of " + w.tile.to_string());
                const std::uint32_t b = bank(w.slot, w.row);
                if (writes_now_[b] == config_.write_ports) {
                    ++banks_[b].stalls;
                    waiting(Stall::BankBusy);
                    break;
                }
                write(now, writes->pop(now));
                ++writes_now_[b];
                ++banks_[b].writes;
                moved = true;
            }
        }

        // Each reader's reads, in its arrival order, while the head's slot holds its tile and its bank
        // has a read port; none while a row is held at the end of its pipeline. The readers take
        // turns at the ports, the first alternating clock by clock.
        for (std::size_t turn = 0; turn < readers_.size(); ++turn) {
            Reader& rd = readers_[(now + turn) % readers_.size()];
            if (rd.reads) serve(now, rd, moved, waiting);
        }

        // Tile-ready tokens to the Streamer, in order, while its link has a credit.
        while (!tokens_.empty() && ready_->can_push(now)) {
            ready_->push(now, tokens_.front());
            tokens_.pop_front();
            moved = true;
        }
        if (!tokens_.empty()) waiting(Stall::NoCredit);

        // The head Release.
        if (head_ < releases_.size()) {
            const Release& c = releases_[head_];
            if (pool_.state(c.slot) == State::Ready && pool_.owner(c.slot) == c.tile &&
                consumers_done_[c.slot] == c.consumers) {
                Link<SlotCredit>* to = c.to_streamer ? st_credits_ : credits_;
                if (!to)
                    throw std::logic_error(where(now) + ": slot " + std::to_string(c.slot) +
                                           "'s next producer is a Result, and no Streamer takes its credit");
                if (to->can_push(now)) {
                    pool_.release(c.slot, c.tile);
                    slots_[c.slot] = Slot{};
                    consumers_done_[c.slot] = 0;
                    to->push(now, SlotCredit{++credit_id_, c.slot});
                    counters().complete();
                    ++head_;
                    moved = true;
                } else {
                    waiting(Stall::NoCredit);
                }
            } else {
                waiting(Stall::NoData);
            }
        }

        for (std::uint32_t b = 0; b < config_.banks; ++b) {
            banks_[b].in_flight_integral += in_flight_[b];
            banks_[b].max_in_flight = std::max<std::uint64_t>(banks_[b].max_in_flight, in_flight_[b]);
        }
        // Reads waiting out their latency are work under way, not a stall.
        bool under_way = false;
        for (const Reader& rd : readers_) under_way = under_way || (!rd.held && !rd.pipe.empty());
        if (moved || under_way) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override {
        bool empty = tokens_.empty();
        for (const Reader& rd : readers_) empty = empty && rd.pipe.empty();
        return head_ == releases_.size() && empty;
    }

    // Every clock: no bank past its ports, every slot's rows agree with its state, the reads in
    // flight agree with the pipelines and stay within each reader's ports times the latency, and
    // the counters' tiles are the live slots.
    void check(Cycle now) const override {
        std::size_t in_flight = 0, piped = 0;
        for (const Reader& rd : readers_) piped += rd.pipe.size();
        for (std::uint32_t b = 0; b < config_.banks; ++b) {
            std::uint64_t by_reader = 0;
            for (std::size_t i = 0; i < readers_.size(); ++i) {
                const std::uint64_t n = readers_[i].in_flight[b];
                if (n > std::uint64_t(config_.read_ports) * config_.read_latency)
                    throw std::logic_error(where(now) + ": bank " + std::to_string(b) + " has " + std::to_string(n) +
                                           " reads of reader " + std::to_string(i) +
                                           " in flight, past its ports times the latency");
                by_reader += n;
            }
            if (by_reader != in_flight_[b])
                throw std::logic_error(where(now) + ": bank " + std::to_string(b) + " has " +
                                       std::to_string(in_flight_[b]) + " reads in flight, its readers " +
                                       std::to_string(by_reader));
            if (reads_now_[b] > config_.read_ports || writes_now_[b] > config_.write_ports)
                throw std::logic_error(where(now) + ": bank " + std::to_string(b) + " served " +
                                       std::to_string(reads_now_[b]) + " reads and " + std::to_string(writes_now_[b]) +
                                       " writes, past its " + std::to_string(config_.read_ports) + " and " +
                                       std::to_string(config_.write_ports) + " ports");
            in_flight += in_flight_[b];
        }
        if (in_flight != piped)
            throw std::logic_error(where(now) + ": " + std::to_string(in_flight) + " reads in flight by bank, " +
                                   std::to_string(piped) + " in the pipelines");
        std::uint32_t live = 0;
        for (std::uint32_t s = 0; s < config_.slots; ++s) {
            const Slot& sl = slots_[s];
            const State st = pool_.state(s);
            live += st != State::Free;
            const bool ok = st == State::Free    ? sl.written == 0
                            : st == State::Owned ? sl.written > 0 && sl.written < sl.rows
                                                 : sl.rows > 0 && sl.written == sl.rows;
            if (!ok)
                throw std::logic_error(where(now) + ": slot " + std::to_string(s) + " is " +
                                       sw::kpu::timing::SlotPool::name(st) + " with " + std::to_string(sl.written) +
                                       " of " + std::to_string(sl.rows) + " rows written");
        }
        if (live != counters().counts().occupancy)
            throw std::logic_error(where(now) + ": " + std::to_string(live) + " live slots, but the counters hold " +
                                   std::to_string(counters().counts().occupancy) + " tiles");
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::string d;
        if (head_ < releases_.size()) {
            const Release& c = releases_[head_];
            d = "Release " + std::to_string(head_ + 1) + " of " + std::to_string(releases_.size()) + " (slot " +
                std::to_string(c.slot) + ", " + c.tile.to_string() + ": " + std::to_string(consumers_done_[c.slot]) +
                " of " + std::to_string(c.consumers) + " consumers done)";
        } else {
            d = "all " + std::to_string(releases_.size()) + " Releases done";
        }
        std::size_t piped = 0;
        for (const Reader& rd : readers_) piped += rd.pipe.size();
        d += "; " + std::to_string(piped) + " reads in flight; slots:";
        for (std::uint32_t s = 0; s < config_.slots; ++s) {
            d += " " + std::to_string(s) + " " + sw::kpu::timing::SlotPool::name(pool_.state(s));
            if (pool_.state(s) != State::Free)
                d += " (" + slots_[s].tile.to_string() + ", " + std::to_string(slots_[s].written) + "/" +
                     std::to_string(slots_[s].rows) + " rows)";
        }
        return d;
    }

    // Queries, for the tests and the cross-component invariants.
    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] State state(std::uint32_t slot) const { return pool_.state(slot); }
    [[nodiscard]] bool live(std::uint32_t slot) const { return state(slot) != State::Free; }
    // The consumers reading the slot now (#387's reader count).
    [[nodiscard]] std::uint32_t readers(std::uint32_t slot) const { return pool_.readers(slot); }
    // The bank a slot's row is in, and its index there.
    [[nodiscard]] std::uint32_t bank(std::uint32_t slot, std::uint32_t row) const {
        return static_cast<std::uint32_t>(global(slot, row) % config_.banks);
    }
    [[nodiscard]] std::uint32_t index(std::uint32_t slot, std::uint32_t row) const {
        return static_cast<std::uint32_t>(global(slot, row) / config_.banks);
    }
    // Row `index` of bank `bank`, as the bank holds it.
    [[nodiscard]] std::vector<std::uint8_t> bank_row(std::uint32_t bank, std::uint32_t index) const {
        const std::size_t at = std::size_t(index) * config_.row_bytes;
        const auto& s = storage_.at(bank);
        if (at + config_.row_bytes > s.size()) throw std::out_of_range("cycle::L2 " + name() + ": no such bank row");
        return {s.begin() + at, s.begin() + at + config_.row_bytes};
    }
    [[nodiscard]] const std::vector<BankStats>& banks() const { return banks_; }
    [[nodiscard]] std::uint64_t acquires() const { return pool_.acquires(); }
    [[nodiscard]] std::uint64_t releases() const { return pool_.releases(); }

private:
    struct Release {
        std::size_t ticket;
        std::uint32_t slot;
        TileID tile;
        std::uint32_t consumers;   // the tile's Writebacks and Feeds before the Release
        bool to_streamer;          // its credit goes to the Streamer: the slot's next producer is a Result
    };
    struct Slot {
        TileID tile{};
        std::uint32_t rows = 0;        // the Move's
        std::uint32_t written = 0;
        std::vector<bool> landed;
    };
    struct InFlight {
        Cycle ready_at;
        std::uint32_t bank;
        bool last;
        L2RowData data;
    };
    // One reader's links, its served reads in order (their ready clocks rise), and its reads in
    // flight by bank.
    struct Reader {
        Link<L2RowRead>* reads = nullptr;
        Link<L2RowData>* data = nullptr;
        std::deque<InFlight> pipe;
        std::vector<std::uint64_t> in_flight;
        bool held = false;   // its pipeline's head cannot leave this clock
    };

    template <class Waiting>
    void serve(Cycle now, Reader& rd, bool& moved, Waiting&& waiting) {
        while (!rd.held && rd.reads->can_pop(now)) {
            const L2RowRead& r = rd.reads->peek(now);
            check_row(now, r.slot, r.row, "the read of " + r.tile.to_string());
            if (!holds(r.slot, r.tile)) {
                waiting(Stall::NoData);   // the tile-ready token
                return;
            }
            if (r.row >= slots_[r.slot].rows)
                throw std::logic_error(where(now) + ": the read of " + r.tile.to_string() + " row " + std::to_string(r.row) +
                                       " in slot " + std::to_string(r.slot) + ", whose Move wrote " +
                                       std::to_string(slots_[r.slot].rows) + " rows");
            const std::uint32_t b = bank(r.slot, r.row);
            if (reads_now_[b] == config_.read_ports) {
                ++banks_[b].stalls;
                waiting(Stall::BankBusy);
                return;
            }
            const L2RowRead req = rd.reads->pop(now);
            if (req.first) pool_.take(req.slot, req.tile);
            const std::size_t at = offset(req.slot, req.row);
            rd.pipe.push_back({now + config_.read_latency, b, req.last,
                               L2RowData{req.id, req.slot, req.row, req.tile,
                                         {storage_[b].begin() + at, storage_[b].begin() + at + config_.row_bytes}}});
            ++in_flight_[b];
            ++rd.in_flight[b];
            ++reads_now_[b];
            ++banks_[b].reads;
            moved = true;
        }
    }

    [[nodiscard]] std::uint64_t global(std::uint32_t slot, std::uint32_t row) const {
        return std::uint64_t(slot) * config_.rows_per_slot + row;
    }
    [[nodiscard]] std::size_t offset(std::uint32_t slot, std::uint32_t row) const {
        return std::size_t(index(slot, row)) * config_.row_bytes;
    }
    [[nodiscard]] std::string where(Cycle now) const { return "cycle::L2 " + name() + " at cycle " + std::to_string(now); }
    [[nodiscard]] bool holds(std::uint32_t slot, const TileID& tile) const {
        const State st = pool_.state(slot);
        return (st == State::Ready || st == State::Reading) && slots_[slot].tile == tile;
    }
    void check_row(Cycle now, std::uint32_t slot, std::uint32_t row, const std::string& what) const {
        if (slot >= config_.slots || row >= config_.rows_per_slot)
            throw std::logic_error(where(now) + ": " + what + " names slot " + std::to_string(slot) + " row " +
                                   std::to_string(row) + ", past the L2's " + std::to_string(config_.slots) + " x " +
                                   std::to_string(config_.rows_per_slot) + " rows");
    }

    void write(Cycle now, const L2RowWrite& w) {
        const std::string row = w.tile.to_string() + " row " + std::to_string(w.row) + " into slot " + std::to_string(w.slot);
        if (w.bytes.size() != config_.row_bytes || w.rows == 0 || w.rows > config_.rows_per_slot || w.row >= w.rows)
            throw std::logic_error(where(now) + ": " + row + " (" + std::to_string(w.bytes.size()) + " bytes, of " +
                                   std::to_string(w.rows) + " rows) is not a row of its Move");
        Slot& s = slots_[w.slot];
        const State st = pool_.state(w.slot);
        if (st == State::Free) {
            pool_.acquire(w.slot, w.tile);   // FREE -> OWNED: the Move's first row
            s = Slot{w.tile, w.rows, 0, std::vector<bool>(w.rows, false)};
            counters().accept();
        } else if (st != State::Owned || !(s.tile == w.tile)) {
            throw SlotFault(where(now) + ": " + row + ", which holds " + s.tile.to_string() + " (" +
                            sw::kpu::timing::SlotPool::name(st) + "): a Move into a live slot");
        } else if (w.rows != s.rows) {
            throw std::logic_error(where(now) + ": " + row + " says its Move is " + std::to_string(w.rows) +
                                   " rows, its first row " + std::to_string(s.rows));
        }
        if (s.landed[w.row]) throw std::logic_error(where(now) + ": " + row + " written twice");
        const std::uint32_t b = bank(w.slot, w.row);
        std::copy(w.bytes.begin(), w.bytes.end(), storage_[b].begin() + offset(w.slot, w.row));
        s.landed[w.row] = true;
        if (++s.written == s.rows) {
            pool_.signal(w.slot, w.tile);   // OWNED -> READY: the tile-ready token
            // The token for this residency's Feeds, the slot's next Move's in the program.
            const std::uint32_t feeds = feeds_[w.slot].at(moves_[w.slot]++);
            if (ready_ && feeds > 0) tokens_.push_back(SlotReady{++token_id_, w.slot, w.tile, feeds});
        }
    }

    void clear() {
        pool_ = sw::kpu::timing::SlotPool(config_.slots);
        slots_.assign(config_.slots, Slot{});
        consumers_done_.assign(config_.slots, 0);
        const std::uint64_t rows = std::uint64_t(config_.slots) * config_.rows_per_slot;
        const std::uint64_t per_bank = (rows + config_.banks - 1) / config_.banks;
        storage_.assign(config_.banks, std::vector<std::uint8_t>(per_bank * config_.row_bytes, 0));
        banks_.assign(config_.banks, BankStats{});
        in_flight_.assign(config_.banks, 0);
        reads_now_.assign(config_.banks, 0);
        writes_now_.assign(config_.banks, 0);
        for (Reader& rd : readers_) {
            rd.pipe.clear();
            rd.in_flight.assign(config_.banks, 0);
            rd.held = false;
        }
        tokens_.clear();
        moves_.assign(config_.slots, 0);
        head_ = 0;
        credit_id_ = token_id_ = 0;
    }

    Config config_;
    std::vector<Release> releases_;
    sw::kpu::timing::SlotPool pool_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> consumers_done_;
    std::vector<std::vector<std::uint8_t>> storage_;   // by bank, row-major
    std::vector<BankStats> banks_;
    std::vector<std::uint64_t> in_flight_;             // by bank
    std::vector<std::uint32_t> reads_now_, writes_now_;   // this clock's accesses, by bank
    std::vector<Reader> readers_ = std::vector<Reader>(2);   // 0: the BlockMover's, 1: the Streamer's
    std::deque<SlotReady> tokens_;                     // tile-ready tokens not yet sent
    std::vector<std::vector<std::uint32_t>> feeds_;    // by slot: each of its Moves' Feeds, in order
    std::vector<std::size_t> moves_;                   // by slot: its Moves signalled READY so far
    std::size_t head_ = 0;
    std::uint64_t credit_id_ = 0, token_id_ = 0;
    Link<L2RowWrite>* writes_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
    Link<SlotReady>* ready_ = nullptr;
    Link<SlotCredit>* st_credits_ = nullptr;
    Link<L2RowWrite>* result_writes_ = nullptr;
};

}  // namespace sw::kpu::cycle
