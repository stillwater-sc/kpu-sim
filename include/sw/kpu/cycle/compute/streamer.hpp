#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/streamer.hpp
// The Streamer on the cycle core (#543; docs/plans/cycle-based-kpu-prototype.md §5.1, step 2b;
// docs/plans/systolic-tile-at-lca.md §3.2): a command queue. An OS COMPUTE reads its two operands
// from L2, one A row and one B row per clock, and loads the returning rows into the L1 lane buffers
// (l1_lanes.hpp), which stream them into the array. It signals `completed` when its last read lands.
//
// A COMMAND names its A and B operands (a slot, its tile, the first row, and the lanes: h A rows, w
// B rows, B pre-arranged in L2 as one row per column), and K, the elements each lane streams (at
// most a row's). A reduction over K longer than a row is a chain of commands; only the chain's first
// command marks its elements `first` and only its last marks them `last` (cmd_chain MORE/CONTINUES).
//
// WHEN A COMMAND FIRES (streamer.py:408-414), at clock a:
//   - the Streamer holds the tile-ready tokens of both its operands (SlotReady, from the L2: the tag
//     CAM): it waits for its tiles, never reads early. A token admits its residency's Feeds, one per
//     command, and is dropped when they are spent: it never admits a later residency of the tile;
//   - the previous command's interval has passed: a >= a_prev + I_prev, I = max(K, h, w). This is
//     what keeps the L1 credit: lane m's next block lands in the clock its last element leaves;
//   - a command that produces a result tile (#545) holds its result slot's credit, and spends it as
//     it starts: the RTL's ACQUIRE as the COMPUTE starts. The L2 returns the credit when the slot
//     is released, and the Combiner writes the tile's rows (combiner.hpp);
//   - a command that ends a reduction (and so launches results north) keeps the RTL's DRAIN RULE
//     (streamer.py): each sets a countdown of K + 2R - 2, and the next fires only when its own K
//     covers what is left. Their results then leave the array 2R - 1 clocks apart or more, and the
//     north links never collide.
//
// ITS SCHEDULE is the RTL's, built as state: read slot m (A row m if m < h, B row m if m < w) goes to
// L2 at a + 1 + m. Each row returns through the Streamer's read pipeline, `read_delay` (D) clocks
// deep, and loads into its lane at a + 1 + D + m, whole; the lane streams from the next clock, one
// element a clock, so the lanes' stagger is the array's skew. `completed` is the clock the last row
// loads, a + D + L with L = max(h, w): the operands' slots can be released then, before the array has
// finished. The L2's latency is not fixed (its own clock, its banks' ports, the crossings): a row
// that returns early waits in the pipeline, and a row not back by its load clock is an L1Fault, as a
// row loaded into a lane without its credit is. The hardware cannot wait, so neither does this.
// What D a configuration needs is measured: max_round_trip().
//
// COUNTERS. A command is an item: it arrives when it fires and completes at `completed`. The
// Streamer is busy in a clock it has a command in flight or a lane streaming; otherwise it is
// stalled waiting for its next command's tiles (no data).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::cycle {

struct StreamerCommand {
    struct Operand {
        std::uint32_t slot = 0;
        TileID tile{};
        std::uint32_t row = 0;     // lane m reads the slot's row `row + m`
        std::uint32_t lanes = 0;   // h for A, w for B
    };
    // What the Combiner applies as it writes a result row (combiner.hpp): bias[j] added to column j
    // (none if empty), then ReLU.
    struct Epilogue {
        std::vector<std::int32_t> bias;
        bool relu = false;
    };
    // The command's result tile: h x w int32, into slot `slot` as `tile`.
    struct Result {
        std::uint32_t slot = 0;
        TileID tile{};
        Epilogue epilogue{};
    };
    Operand a, b;
    std::uint32_t k = 0;                // elements per lane
    bool first = true, last = true;     // the reduction chain's first and last command
    std::optional<Result> result;       // a reduction-ending command's tile, written by the Combiner
};
using StreamerProgram = std::vector<StreamerCommand>;

class Streamer : public Component {
public:
    struct Config {
        std::uint32_t a_lanes = 4;        // R: the array's rows
        std::uint32_t b_lanes = 4;        // C: its columns
        std::uint32_t slots = 2;          // the L2's
        std::uint32_t rows_per_slot = 16;
        std::uint32_t row_bytes = 16;     // the L2's row
        std::uint32_t element_bytes = 1;  // 1, 2 or 4
        std::uint32_t read_delay = 32;    // D: clocks from a read to its row's load into L1
        // Added to every interval, for the tests of the credit only: a negative value fires commands
        // early, and the lanes must fault.
        std::int32_t interval_adjust = 0;
        // Added to the drain rule's countdown, for the tests of the north drain only: -1 lets
        // results leave 2R - 2 clocks apart, and the array must fault.
        std::int32_t drain_adjust = 0;
    };
    // A command's clocks: when it fired and when its last row loaded.
    struct Record {
        Cycle accepted = 0;
        std::optional<Cycle> completed;
    };
    // One read to L2.
    struct Issue {
        Cycle at;
        char side;   // 'A' or 'B'
        std::uint32_t lane, command;
    };

    // `l2`: the L2's program, which says the result slots whose credit the Streamer holds at the
    // start (those whose first producer is a Result).
    Streamer(std::string name, Domain& d, Config cfg, StreamerProgram program, const L2Program& l2 = {})
        : Component(std::move(name), d),
          config_(cfg),
          program_(std::move(program)),
          a_("A", cfg.a_lanes ? cfg.a_lanes : 1, elements_per_row(cfg)),
          b_("B", cfg.b_lanes ? cfg.b_lanes : 1, elements_per_row(cfg)) {
        if (cfg.a_lanes == 0 || cfg.b_lanes == 0 || cfg.slots == 0 || cfg.rows_per_slot == 0 || cfg.read_delay == 0)
            throw std::invalid_argument("cycle::Streamer " + this->name() +
                                        ": the lanes, slots, rows and the read delay must be positive");
        validate(program_);
        initial_.assign(cfg.slots, false);
        for (std::uint32_t s = 0; s < cfg.slots; ++s) initial_[s] = streamer_holds_at_start(l2, s);
        clear();
    }

    // reads: row requests to L2; data: the rows back; ready: L2's tile-ready tokens; a_out, b_out:
    // one link per lane into the array; credits: the result slots' credits back from L2.
    void connect(Link<L2RowRead>& reads, Link<L2RowData>& data, Link<SlotReady>& ready,
                 std::vector<Link<L1Element>*> a_out, std::vector<Link<L1Element>*> b_out,
                 Link<SlotCredit>* credits = nullptr) {
        if (a_out.size() != config_.a_lanes || b_out.size() != config_.b_lanes)
            throw std::invalid_argument("cycle::Streamer " + name() + ": " + std::to_string(a_out.size()) + " A and " +
                                        std::to_string(b_out.size()) + " B lane links, for " +
                                        std::to_string(config_.a_lanes) + " and " + std::to_string(config_.b_lanes) +
                                        " lanes");
        reads_ = &reads;
        data_ = &data;
        ready_ = &ready;
        a_out_ = std::move(a_out);
        b_out_ = std::move(b_out);
        credits_ = credits;
    }

    void evaluate(Cycle now) override {
        if (!reads_ || !data_ || !ready_) throw std::logic_error(where(now) + ": not connected");
        bool moved = false;

        // The tile-ready tokens: a slot's latest tile.
        while (ready_->can_pop(now)) {
            const SlotReady t = ready_->pop(now);
            if (t.slot >= config_.slots || t.feeds == 0 || tokens_[t.slot])
                throw std::logic_error(where(now) + ": a token for slot " + std::to_string(t.slot) + " (" +
                                       t.tile.to_string() + ", " + std::to_string(t.feeds) + " Feeds)" +
                                       (tokens_.size() > t.slot && tokens_[t.slot]
                                            ? ", which still holds " + tokens_[t.slot]->tile.to_string() + "'s"
                                            : std::string()));
            tokens_[t.slot] = Token{t.tile, t.feeds};
        }
        // The result slots' credits, back from their Releases.
        while (credits_ && credits_->can_pop(now)) {
            const SlotCredit c = credits_->pop(now);
            if (c.slot >= config_.slots || held_[c.slot])
                throw std::logic_error(where(now) + ": a credit for slot " + std::to_string(c.slot) + ", which " +
                                       (c.slot >= config_.slots ? "does not exist" : "it already holds"));
            held_[c.slot] = true;
        }
        // Rows back from L2, in the order they were asked for, into the read pipeline.
        while (data_->can_pop(now)) {
            L2RowData d = data_->pop(now);
            auto it = std::find_if(awaited_.begin(), awaited_.end(), [](const Awaited& w) { return !w.bytes; });
            if (it == awaited_.end() || it->id != d.id || d.bytes.size() != config_.row_bytes)
                throw std::logic_error(where(now) + ": a row from L2 (read " + std::to_string(d.id) + ", " +
                                       d.tile.to_string() + " row " + std::to_string(d.row) + "), " +
                                       (it == awaited_.end() ? std::string("with no read awaiting one")
                                                             : "expected read " + std::to_string(it->id)));
            it->bytes = std::move(d.bytes);
            max_round_trip_ = std::max(max_round_trip_, now - it->issued);
        }
        // The lanes stream, before this clock's loads: a lane's last element leaves as its next
        // block loads.
        stream(now, a_, a_out_, moved);
        stream(now, b_, b_out_, moved);
        // The rows due now load into their lanes.
        while (!awaited_.empty() && awaited_.front().load_at <= now) {
            Awaited& w = awaited_.front();
            if (w.load_at < now) throw std::logic_error(where(now) + ": read " + std::to_string(w.id) + " never loaded");
            if (!w.bytes)
                throw L1Fault(where(now) + ": the row for " + std::string(1, w.side) + " lane " + std::to_string(w.lane) +
                              " of command " + std::to_string(w.command) + " is not back from L2 by its load, " +
                              std::to_string(config_.read_delay) + " clocks after its read (the read delay)");
            const StreamerCommand& c = program_[w.command];
            std::vector<std::uint32_t> block(c.k);
            for (std::uint32_t k = 0; k < c.k; ++k) {
                std::uint32_t v = 0;
                for (std::uint32_t i = 0; i < config_.element_bytes; ++i)
                    v |= std::uint32_t((*w.bytes)[std::size_t(k) * config_.element_bytes + i]) << (8 * i);
                block[k] = v;
            }
            (w.side == 'A' ? a_ : b_).load(now, w.lane, std::move(block), w.command, c.first, c.last);
            if (--loads_left_[w.command] == 0) {
                records_[w.command].completed = now;   // `completed`: the last read has landed
                counters().complete();
            }
            awaited_.pop_front();
            moved = true;
        }
        a_.settle();
        b_.settle();
        // The reads due now.
        while (!pending_.empty() && pending_.front().at <= now) {
            const Pending p = pending_.front();
            if (p.at < now) throw std::logic_error(where(now) + ": a read due at " + std::to_string(p.at) + " not issued");
            if (!reads_->can_push(now))
                throw std::logic_error(where(now) + ": no credit on the read link to L2 for " + std::string(1, p.side) + " lane " +
                              std::to_string(p.lane) + " of command " + std::to_string(p.command) +
                              ": the Streamer reads on its schedule and cannot wait");
            reads_->push(now, L2RowRead{++id_, p.slot, p.row, p.tile, p.first, p.last});
            awaited_.push_back({id_, p.side, p.lane, p.command, now, now + config_.read_delay, std::nullopt});
            issues_.push_back({now, p.side, p.lane, p.command});
            pending_.pop_front();
            moved = true;
        }
        // The head command, when its tiles are ready, its result slot's credit is held, the previous
        // command's interval has passed and, if it ends a reduction, the drain rule allows it.
        std::optional<Stall> wait;
        if (head_ < program_.size()) {
            const StreamerCommand& c = program_[head_];
            const bool interval = !last_ || std::int64_t(now) >= std::int64_t(last_->accepted) + last_->interval +
                                                                     config_.interval_adjust;
            const bool tiles = holds(c.a) && holds(c.b);
            const bool slot = !c.result || held_[c.result->slot];
            const bool drain = !c.last || std::int64_t(c.k) >= drain_left(now);
            if (interval && tiles && slot && drain) {
                fire(now, c);
                moved = true;
            } else if (!tiles) {
                wait = Stall::NoData;
            } else if (!slot) {
                wait = Stall::NoCredit;
            }
        }
        if (moved || !pending_.empty() || !awaited_.empty() || !a_.empty() || !b_.empty()) counters().busy();
        else if (wait) counters().stall(*wait);
    }

    [[nodiscard]] bool idle() const override {
        return head_ == program_.size() && pending_.empty() && awaited_.empty() && a_.empty() && b_.empty();
    }

    // Every clock: the lanes' credits are conserved, and nothing due has been left behind.
    void check(Cycle now) const override {
        a_.check(now);
        b_.check(now);
        if (!pending_.empty() && pending_.front().at <= now)
            throw std::logic_error(where(now) + ": a read due at " + std::to_string(pending_.front().at) + " not issued");
        if (!awaited_.empty() && awaited_.front().load_at <= now)
            throw std::logic_error(where(now) + ": a row due at " + std::to_string(awaited_.front().load_at) +
                                   " not loaded");
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::string d;
        if (head_ < program_.size()) {
            const StreamerCommand& c = program_[head_];
            d = "command " + std::to_string(head_) + " of " + std::to_string(program_.size()) + " (A " +
                c.a.tile.to_string() + " in slot " + std::to_string(c.a.slot) + ", B " + c.b.tile.to_string() +
                " in slot " + std::to_string(c.b.slot) + ")";
            if (!holds(c.a) || !holds(c.b)) d += ": waiting for its tiles";
            for (std::uint32_t s = 0; s < config_.slots; ++s)
                if (tokens_[s])
                    d += "; slot " + std::to_string(s) + " token " + tokens_[s]->tile.to_string() + " (" +
                         std::to_string(tokens_[s]->feeds) + " Feeds left)";
        } else {
            d = "all " + std::to_string(program_.size()) + " commands fired";
        }
        std::size_t back = 0;
        for (const Awaited& w : awaited_) back += w.bytes.has_value();
        d += "; " + std::to_string(pending_.size()) + " reads to issue, " + std::to_string(awaited_.size()) +
             " rows to load (" + std::to_string(back) + " back from L2)";
        return d;
    }

    // Queries, for the tests and the cross-component invariants.
    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const std::vector<Record>& records() const { return records_; }
    [[nodiscard]] const std::vector<Issue>& issues() const { return issues_; }
    [[nodiscard]] const L1LaneBuffers& a_lanes() const { return a_; }
    [[nodiscard]] const L1LaneBuffers& b_lanes() const { return b_; }
    // The longest a row took to come back from L2, in the Streamer's clocks: the least read delay
    // this run could have kept.
    [[nodiscard]] Cycle max_round_trip() const { return max_round_trip_; }
    // Holds slot `slot`'s credit: free for the result tile of a command still to fire.
    [[nodiscard]] bool holds_credit(std::uint32_t slot) const { return held_.at(slot); }

    // The commands' elements per lane, at most: one L2 row's.
    static std::uint32_t elements_per_row(const Config& c) {
        if (c.element_bytes != 1 && c.element_bytes != 2 && c.element_bytes != 4)
            throw std::invalid_argument("cycle::Streamer: an element is 1, 2 or 4 bytes");
        if (c.row_bytes == 0 || c.row_bytes % c.element_bytes != 0)
            throw std::invalid_argument("cycle::Streamer: a row is whole elements");
        return c.row_bytes / c.element_bytes;
    }

private:
    struct Pending {
        Cycle at;
        char side;
        std::uint32_t lane, command, slot, row;
        TileID tile;
        bool first, last;
    };
    struct Awaited {
        std::uint64_t id;
        char side;
        std::uint32_t lane, command;
        Cycle issued, load_at;
        std::optional<std::vector<std::uint8_t>> bytes;
    };
    struct Token {
        TileID tile;
        std::uint32_t feeds;   // the commands it still admits
    };
    struct Last {
        Cycle accepted;
        std::int64_t interval;
    };
    // The drain rule's countdown: K + 2R - 2 at the clock after a reduction-ending command fired,
    // one less each clock after.
    [[nodiscard]] std::int64_t drain_left(Cycle now) const {
        if (!drain_) return 0;
        const std::int64_t gone = std::int64_t(now) - std::int64_t(drain_->accepted) - 1;
        return std::max<std::int64_t>(0, drain_->wait - gone);
    }
    struct Drain {
        Cycle accepted;
        std::int64_t wait;
    };

    [[nodiscard]] std::string where(Cycle now) const {
        return "cycle::Streamer " + name() + " at cycle " + std::to_string(now);
    }
    // The Streamer holds the operand's tile-ready token.
    [[nodiscard]] bool holds(const StreamerCommand::Operand& o) const {
        return tokens_[o.slot].has_value() && tokens_[o.slot]->tile == o.tile;
    }

    // The program's own consistency, before anything runs. Raises std::invalid_argument naming the
    // command.
    void validate(const StreamerProgram& p) const {
        const std::uint32_t kb = elements_per_row(config_);
        bool open = false;   // within a chain
        for (std::size_t i = 0; i < p.size(); ++i) {
            const StreamerCommand& c = p[i];
            const std::string at = "cycle::Streamer " + name() + ": command " + std::to_string(i) + " (A " +
                                   c.a.tile.to_string() + ", B " + c.b.tile.to_string() + ")";
            const auto operand = [&](const StreamerCommand::Operand& o, std::uint32_t lanes, const char* side) {
                if (o.lanes == 0 || o.lanes > lanes)
                    throw std::invalid_argument(at + ": " + std::to_string(o.lanes) + " " + side + " lanes, the array has 1 to " +
                                                std::to_string(lanes));
                if (o.slot >= config_.slots)
                    throw std::invalid_argument(at + ": " + side + " in slot " + std::to_string(o.slot) + ", there are " +
                                                std::to_string(config_.slots));
                if (std::uint64_t(o.row) + o.lanes > config_.rows_per_slot)
                    throw std::invalid_argument(at + ": " + side + " rows " + std::to_string(o.row) + " to " +
                                                std::to_string(o.row + o.lanes - 1) + ", past the slot's " +
                                                std::to_string(config_.rows_per_slot));
            };
            operand(c.a, config_.a_lanes, "A");
            operand(c.b, config_.b_lanes, "B");
            if (c.a.slot == c.b.slot) throw std::invalid_argument(at + ": A and B in one slot");
            if (c.k == 0 || c.k > kb)
                throw std::invalid_argument(at + ": K = " + std::to_string(c.k) + ", a row holds 1 to " + std::to_string(kb) +
                                            " elements");
            if (c.result) {
                const StreamerCommand::Result& r = *c.result;
                if (!c.last) throw std::invalid_argument(at + ": a result tile on a command that does not end its chain");
                if (r.slot >= config_.slots)
                    throw std::invalid_argument(at + ": its result in slot " + std::to_string(r.slot) + ", there are " +
                                                std::to_string(config_.slots));
                if (r.slot == c.a.slot || r.slot == c.b.slot)
                    throw std::invalid_argument(at + ": its result in an operand's slot");
                if (!r.epilogue.bias.empty() && r.epilogue.bias.size() != c.b.lanes)
                    throw std::invalid_argument(at + ": a bias of " + std::to_string(r.epilogue.bias.size()) +
                                                " for " + std::to_string(c.b.lanes) + " columns");
            }
            if (c.first == open)
                throw std::invalid_argument(at + (open ? ": opens a chain inside the open one before it"
                                                       : ": continues a chain no command opened"));
            open = !c.last;
        }
        if (open) throw std::invalid_argument("cycle::Streamer " + name() + ": the last command leaves its chain open");
    }

    void fire(Cycle now, const StreamerCommand& c) {
        const std::uint32_t n = static_cast<std::uint32_t>(head_);
        const std::uint32_t reads = std::max(c.a.lanes, c.b.lanes);
        std::vector<Pending> due;
        for (std::uint32_t m = 0; m < reads; ++m) {
            if (m < c.a.lanes) due.push_back({now + 1 + m, 'A', m, n, c.a.slot, c.a.row + m, c.a.tile, m == 0, m + 1 == c.a.lanes});
            if (m < c.b.lanes) due.push_back({now + 1 + m, 'B', m, n, c.b.slot, c.b.row + m, c.b.tile, m == 0, m + 1 == c.b.lanes});
        }
        // In clock order, after any of an earlier command's due in the same clock.
        for (Pending& p : due) {
            auto it = std::upper_bound(pending_.begin(), pending_.end(), p.at,
                                       [](Cycle at, const Pending& q) { return at < q.at; });
            pending_.insert(it, p);
        }
        for (const StreamerCommand::Operand* o : {&c.a, &c.b})
            if (--tokens_[o->slot]->feeds == 0) tokens_[o->slot].reset();   // the residency's last Feed
        loads_left_[n] = c.a.lanes + c.b.lanes;
        records_[n].accepted = now;
        last_ = Last{now, std::int64_t(std::max({c.k, c.a.lanes, c.b.lanes}))};
        if (c.last) drain_ = Drain{now, std::int64_t(c.k) + 2 * std::int64_t(config_.a_lanes) - 2 + config_.drain_adjust};
        if (c.result) held_[c.result->slot] = false;   // the ACQUIRE, as the producing command starts
        counters().accept();
        ++head_;
    }

    void stream(Cycle now, L1LaneBuffers& lanes, std::vector<Link<L1Element>*>& out, bool& moved) {
        for (std::uint32_t m = 0; m < lanes.lanes(); ++m) {
            std::optional<L1Element> e = lanes.stream(now, m);
            if (!e) continue;
            if (!out[m]->can_push(now))
                throw L1Fault(where(now) + ": the array did not take lane " + std::to_string(m) + "'s element " +
                              std::to_string(e->k) + " of command " + std::to_string(e->command) +
                              " (it takes one every clock)");
            out[m]->push(now, *e);
            moved = true;
        }
    }

    void clear() {
        a_ = L1LaneBuffers("A", config_.a_lanes, elements_per_row(config_));
        b_ = L1LaneBuffers("B", config_.b_lanes, elements_per_row(config_));
        tokens_.assign(config_.slots, std::nullopt);
        pending_.clear();
        awaited_.clear();
        loads_left_.assign(program_.size(), 0);
        records_.assign(program_.size(), Record{});
        issues_.clear();
        last_.reset();
        drain_.reset();
        held_ = initial_;
        head_ = 0;
        id_ = 0;
        max_round_trip_ = 0;
    }
    Config config_;
    StreamerProgram program_;
    L1LaneBuffers a_, b_;
    std::vector<std::optional<Token>> tokens_;   // by slot: its residency's token, while it admits a Feed
    std::deque<Pending> pending_;         // reads not yet issued, in clock order
    std::deque<Awaited> awaited_;         // reads issued, rows not yet loaded, in issue order
    std::vector<std::uint32_t> loads_left_;
    std::vector<Record> records_;
    std::vector<Issue> issues_;
    std::optional<Last> last_;
    std::optional<Drain> drain_;
    std::vector<bool> initial_, held_;   // by slot: the result slots' credits, at the start and now
    std::size_t head_ = 0;
    std::uint64_t id_ = 0;
    Cycle max_round_trip_ = 0;
    Link<L2RowRead>* reads_ = nullptr;
    Link<L2RowData>* data_ = nullptr;
    Link<SlotReady>* ready_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
    std::vector<Link<L1Element>*> a_out_, b_out_;
};

// A Streamer program and the L2's agree, and together they can run to the end (CodeRabbit, #593):
//   1. In each slot, the commands read the tiles the L2 program's Feeds name, in the same order and
//      with the same rows. The L2 frees a slot after its Feeds, so a command past them would find
//      the slot gone.
//   2. The schedule completes. Order within each slot is not enough: a command waits for both its
//      operands, so one that needs a tile whose Move is queued behind another slot's Release, which
//      waits for this very command, never fires. The check steps the protocol without time, as the
//      components execute it: the BlockMover's Moves and Writebacks in program order (a Move waits
//      for its slot to be free), the L2's Releases in program order (each waits for its residency's
//      consumers), and the commands in order (each waits for both operands' residencies, with a
//      Feed left on each). Whatever cannot proceed when nothing else can is a deadlock.
// Raises std::invalid_argument naming the slot and the first read that differs, or where each of
// the three is stuck.
inline void check_feeds(const StreamerProgram& commands, const L2Program& l2) {
    using Read = std::pair<std::string, std::uint32_t>;   // the tile, its rows
    std::map<std::uint32_t, std::vector<Read>> feeds, reads;
    for (const L2Action& a : l2)
        if (a.kind == L2Action::Kind::Feed) feeds[a.slot].push_back({a.tile.to_string(), a.rows});
    for (const StreamerCommand& c : commands) {
        reads[c.a.slot].push_back({c.a.tile.to_string(), c.a.lanes});
        reads[c.b.slot].push_back({c.b.tile.to_string(), c.b.lanes});
    }
    std::map<std::uint32_t, bool> slots;
    for (const auto& [s, v] : feeds) slots[s] = true;
    for (const auto& [s, v] : reads) slots[s] = true;
    for (const auto& [s, unused] : slots) {
        const std::vector<Read>& f = feeds[s];
        const std::vector<Read>& r = reads[s];
        const std::string slot = "check_feeds: slot " + std::to_string(s) + ": ";
        for (std::size_t i = 0; i < std::min(f.size(), r.size()); ++i) {
            if (f[i].first != r[i].first)
                throw std::invalid_argument(slot + "read " + std::to_string(i) + " is of " + r[i].first +
                                            " by the commands, of " + f[i].first + " by the L2 program's Feeds");
            if (f[i].second != r[i].second)
                throw std::invalid_argument(slot + "read " + std::to_string(i) + " of " + r[i].first + " is " +
                                            std::to_string(r[i].second) + " rows by the commands, " +
                                            std::to_string(f[i].second) + " by its Feed");
        }
        if (f.size() != r.size())
            throw std::invalid_argument(slot + "the commands read it " + std::to_string(r.size()) +
                                        " times, the L2 program's Feeds " + std::to_string(f.size()));
    }

    // 2. Step the protocol to its end, or to where nothing can proceed.
    struct Residency {
        TileID tile;
        std::uint32_t feeds = 0, consumers = 0;   // Feeds left to admit, consumers left before the Release
    };
    std::vector<std::size_t> mover, releases;    // the program's indices, in order
    std::map<std::size_t, Residency> of_move;    // each producer's residency, from the program
    std::map<std::uint32_t, std::size_t> last_move;
    std::map<std::uint32_t, std::vector<std::size_t>> results;   // by slot: its Results, in order
    for (std::size_t i = 0; i < l2.size(); ++i) {
        const L2Action& a = l2[i];
        if (is_producer(a.kind)) {
            of_move[i] = Residency{a.tile, 0, 0};
            last_move[a.slot] = i;
        }
        if (a.kind == L2Action::Kind::Result) results[a.slot].push_back(i);
        if (a.kind == L2Action::Kind::Move || a.kind == L2Action::Kind::Writeback) mover.push_back(i);
        if (a.kind == L2Action::Kind::Release) releases.push_back(i);
        if ((a.kind == L2Action::Kind::Feed || a.kind == L2Action::Kind::Writeback) && last_move.count(a.slot)) {
            ++of_move[last_move[a.slot]].consumers;
            if (a.kind == L2Action::Kind::Feed) ++of_move[last_move[a.slot]].feeds;
        }
    }
    // Each producing command's Result: the n-th command producing into a slot is its n-th Result.
    std::vector<std::optional<std::size_t>> produces(commands.size());
    {
        std::map<std::uint32_t, std::size_t> n;
        for (std::size_t i = 0; i < commands.size(); ++i) {
            if (!commands[i].result) continue;
            const StreamerCommand::Result& res = *commands[i].result;
            const std::vector<std::size_t>& v = results[res.slot];
            const std::size_t at = n[res.slot]++;
            const std::string slot = "check_feeds: slot " + std::to_string(res.slot) + ": ";
            if (at >= v.size())
                throw std::invalid_argument(slot + "command " + std::to_string(i) + "'s result tile " +
                                            res.tile.to_string() + " has no Result in the L2 program");
            const L2Action& a = l2[v[at]];
            if (!(a.tile == res.tile) || a.rows != commands[i].a.lanes)
                throw std::invalid_argument(slot + "command " + std::to_string(i) + "'s result is " +
                                            res.tile.to_string() + ", " + std::to_string(commands[i].a.lanes) +
                                            " rows; the L2 program's Result is " + a.tile.to_string() + ", " +
                                            std::to_string(a.rows) + " rows");
            produces[i] = v[at];
        }
        for (const auto& [slot, v] : results)
            if (n[slot] != v.size())
                throw std::invalid_argument("check_feeds: slot " + std::to_string(slot) + ": the L2 program has " +
                                            std::to_string(v.size()) + " Results, the commands produce " +
                                            std::to_string(n[slot]));
    }
    std::map<std::uint32_t, Residency> live;   // by slot
    std::size_t m = 0, r = 0, c = 0;
    for (bool progress = true; progress;) {
        progress = false;
        while (m < mover.size()) {   // the BlockMover
            const L2Action& a = l2[mover[m]];
            if (a.kind == L2Action::Kind::Move) {
                if (live.count(a.slot)) break;   // waits for the slot's credit
                live[a.slot] = of_move[mover[m]];
            } else {   // a Writeback: of the residency live in its slot, once its producer has filled it
                const auto it = live.find(a.slot);
                if (it == live.end() || !(it->second.tile == a.tile)) break;
                --it->second.consumers;
            }
            ++m;
            progress = true;
        }
        while (r < releases.size()) {   // the L2's Releases
            const L2Action& a = l2[releases[r]];
            const auto it = live.find(a.slot);
            if (it == live.end() || !(it->second.tile == a.tile) || it->second.consumers != 0) break;
            live.erase(it);
            ++r;
            progress = true;
        }
        while (c < commands.size()) {   // the Streamer
            const StreamerCommand& k = commands[c];
            const auto ready = [&](const StreamerCommand::Operand& o) {
                const auto it = live.find(o.slot);
                return it != live.end() && it->second.tile == o.tile && it->second.feeds > 0;
            };
            if (!ready(k.a) || !ready(k.b)) break;
            if (k.result && live.count(k.result->slot)) break;   // waits for its result slot's credit
            for (const StreamerCommand::Operand* o : {&k.a, &k.b}) {
                --live[o->slot].feeds;
                --live[o->slot].consumers;
            }
            if (k.result) live[k.result->slot] = of_move[*produces[c]];   // its Result's residency
            ++c;
            progress = true;
        }
    }
    if (c < commands.size() || m < mover.size() || r < releases.size()) {
        const auto at = [&](const std::vector<std::size_t>& v, std::size_t i) {
            if (i == v.size()) return std::string("done");
            const L2Action& a = l2[v[i]];
            return "action " + std::to_string(v[i]) + " (" + to_string(a.kind) + " " + a.tile.to_string() +
                   " in slot " + std::to_string(a.slot) + ")";
        };
        throw std::invalid_argument(
            "check_feeds: the schedule cannot complete: the Streamer waits at " +
            (c == commands.size() ? std::string("its end")
                                  : "command " + std::to_string(c) + " (A " + commands[c].a.tile.to_string() + ", B " +
                                        commands[c].b.tile.to_string() + ")") +
            ", the BlockMover at " + at(mover, m) + ", the L2's Releases at " + at(releases, r));
    }
}

}  // namespace sw::kpu::cycle
