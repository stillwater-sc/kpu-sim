#pragma once
// ============================================================================
// include/sw/kpu/cycle/compute/combiner.hpp
// The north result FIFOs and the Combiner on the cycle core (#545; docs/plans/cycle-based-kpu-
// prototype.md §5.1, step 2d; the RTL's result path, embodied-ai-soc-generator b57c881,
// tiles/systolic/controller/result_path.py): the array's results into the result's L2 slot.
//
// THE NORTH FIFOs, one per column, take the result each column of the array drives out of its north
// edge. A ROW LEAVES when every column of its tile has its element: the FIFOs turn the array's
// columns back into rows. A padded result (a column past the tile's, pe_array.hpp) is dropped. The
// fabric has no flow control, so results never wait: a FIFO past its depth is an L1Fault, and so
// is a result for a command that produces no result tile (the RTL's RESULT_WITHOUT_COMMAND).
//
// THE COMBINER writes each row as it leaves into the result tile's L2 slot (a Result, l2_program.hpp),
// applying the tile's epilogue: bias[j] added to column j, then ReLU. A row is the tile's w int32
// results, little-endian, in the L2's row. The tile's first row ACQUIREs the slot in the L2 and its
// last SIGNALs it READY. RESULT is the clock the Combiner writes the tile's last row: the slot is
// then the BlockMover's to write back. The Combiner's write port has the L2's banks first; a row
// it cannot hand to the L2 at once is an L1Fault, as the hardware cannot wait. (The merge of
// partials, COMPUTE_ACCUMULATE, is a read-modify-write into the slot: not built here.)
//
// The result tiles are the Streamer program's (streamer.hpp): each reduction-ending command that
// names one, in command order, as the rows arrive.
//
// COUNTERS. A result tile is an item: it arrives with its first result and completes at `result`.
// The Combiner is busy in a clock a FIFO holds a result or it writes a row.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/pe_array.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/link.hpp>

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

class Combiner : public Component {
public:
    struct Config {
        std::uint32_t cols = 4;         // C: the array's columns, one FIFO each
        std::uint32_t fifo_depth = 4;   // results a FIFO holds
        std::uint32_t row_bytes = 16;   // the L2's row: at least 4 bytes a column of a tile
    };
    // A result tile's clocks: each row's write, and `result`.
    struct Record {
        std::uint32_t command = 0;
        std::vector<Cycle> rows;
        std::optional<Cycle> result;
    };

    Combiner(std::string name, Domain& d, Config cfg, const StreamerProgram& program)
        : Component(std::move(name), d), config_(cfg) {
        if (cfg.cols == 0 || cfg.fifo_depth == 0 || cfg.row_bytes == 0)
            throw std::invalid_argument("cycle::Combiner " + this->name() +
                                        ": the columns, the FIFO depth and the row must be positive");
        for (std::size_t i = 0; i < program.size(); ++i) {
            const StreamerCommand& c = program[i];
            if (!c.result) continue;
            if (c.b.lanes > cfg.cols || std::uint64_t(c.b.lanes) * 4 > cfg.row_bytes)
                throw std::invalid_argument("cycle::Combiner " + this->name() + ": command " + std::to_string(i) +
                                            "'s result tile is " + std::to_string(c.b.lanes) + " columns, for " +
                                            std::to_string(cfg.cols) + " FIFOs and a " +
                                            std::to_string(cfg.row_bytes) + "-byte row");
            tiles_.push_back({static_cast<std::uint32_t>(i), c.result->slot, c.result->tile, c.a.lanes, c.b.lanes,
                              c.result->epilogue});
        }
        clear();
    }

    // north: one link per column, from the array; writes: the result rows into the L2.
    void connect(std::vector<Link<ArrayResult>*> north, Link<L2RowWrite>& writes) {
        if (north.size() != config_.cols)
            throw std::invalid_argument("cycle::Combiner " + name() + ": " + std::to_string(north.size()) +
                                        " north links for " + std::to_string(config_.cols) + " columns");
        north_ = std::move(north);
        writes_ = &writes;
    }

    void evaluate(Cycle now) override {
        if (!writes_) throw std::logic_error(where(now) + ": not connected");
        bool moved = false;
        // Each column's result into its FIFO; a padded one is dropped.
        for (std::uint32_t j = 0; j < config_.cols; ++j)
            while (north_[j]->can_pop(now)) {
                const ArrayResult r = north_[j]->pop(now);
                moved = true;
                if (r.padded) {
                    ++dropped_;
                    continue;
                }
                const auto it = by_command_.find(r.command);
                if (it == by_command_.end() || it->second < head_)
                    throw L1Fault(where(now) + ": a result from PE(" + std::to_string(r.row) + "," +
                                  std::to_string(j) + ") for command " + std::to_string(r.command) +
                                  ", which produces no result tile still to write");
                const Tile& t = tiles_[it->second];
                if (j >= t.cols || r.row >= t.rows)
                    throw std::logic_error(where(now) + ": a result from PE(" + std::to_string(r.row) + "," +
                                           std::to_string(j) + ") for command " + std::to_string(r.command) +
                                           ", past its " + std::to_string(t.rows) + " x " + std::to_string(t.cols) +
                                           " tile");
                if (!started_[it->second]) {
                    started_[it->second] = true;
                    counters().accept();
                }
                fifos_[j].push_back(r);
                if (fifos_[j].size() > config_.fifo_depth)
                    throw L1Fault(where(now) + ": column " + std::to_string(j) + "'s north FIFO overflows its " +
                                  std::to_string(config_.fifo_depth) + " results");
            }
        // The head tile's next row leaves when every column of the tile has its element.
        if (head_ < tiles_.size()) {
            const Tile& t = tiles_[head_];
            Record& rec = records_[head_];
            const std::uint32_t row = static_cast<std::uint32_t>(rec.rows.size());
            bool whole = true;
            for (std::uint32_t j = 0; j < t.cols; ++j) whole = whole && !fifos_[j].empty();
            if (whole) {
                std::vector<std::uint8_t> bytes(config_.row_bytes, 0);
                for (std::uint32_t j = 0; j < t.cols; ++j) {
                    const ArrayResult r = fifos_[j].front();
                    if (r.command != t.command || r.row != row)
                        throw std::logic_error(where(now) + ": column " + std::to_string(j) + "'s FIFO holds PE(" +
                                               std::to_string(r.row) + ")'s result for command " +
                                               std::to_string(r.command) + ", row " + std::to_string(row) +
                                               " of command " + std::to_string(t.command) + " expected");
                    fifos_[j].pop_front();
                    std::int64_t v = r.value;
                    if (!t.epilogue.bias.empty()) v += t.epilogue.bias[j];
                    std::int32_t out = static_cast<std::int32_t>(static_cast<std::uint32_t>(v));
                    if (t.epilogue.relu) out = std::max<std::int32_t>(out, 0);
                    const std::uint32_t u = static_cast<std::uint32_t>(out);
                    for (std::uint32_t b = 0; b < 4; ++b) bytes[std::size_t(j) * 4 + b] = static_cast<std::uint8_t>(u >> (8 * b));
                }
                if (!writes_->can_push(now))
                    throw L1Fault(where(now) + ": the L2 did not take row " + std::to_string(row) + " of " +
                                  t.tile.to_string() + " (the result path cannot wait)");
                writes_->push(now, L2RowWrite{++id_, t.slot, row, t.rows, t.tile, std::move(bytes)});
                rec.rows.push_back(now);
                moved = true;
                if (rec.rows.size() == t.rows) {
                    rec.result = now;   // `result`: the tile's last row is written
                    counters().complete();
                    ++head_;
                }
            }
        }
        bool held = false;
        for (const auto& f : fifos_) held = held || !f.empty();
        if (moved || held) counters().busy();
    }

    [[nodiscard]] bool idle() const override {
        for (const auto& f : fifos_)
            if (!f.empty()) return false;
        return true;
    }

    void check(Cycle now) const override {
        for (std::uint32_t j = 0; j < config_.cols; ++j)
            if (fifos_[j].size() > config_.fifo_depth)
                throw std::logic_error(where(now) + ": column " + std::to_string(j) + "'s FIFO holds " +
                                       std::to_string(fifos_[j].size()) + " results");
    }

    void reset() override { clear(); }

    [[nodiscard]] std::string describe() const override {
        std::string d = head_ < tiles_.size()
                            ? "result tile " + std::to_string(head_) + " of " + std::to_string(tiles_.size()) + " (" +
                                  tiles_[head_].tile.to_string() + " in slot " + std::to_string(tiles_[head_].slot) +
                                  ": " + std::to_string(records_[head_].rows.size()) + " of " +
                                  std::to_string(tiles_[head_].rows) + " rows written)"
                            : "all " + std::to_string(tiles_.size()) + " result tiles written";
        d += "; FIFOs:";
        for (const auto& f : fifos_) d += " " + std::to_string(f.size());
        return d;
    }

    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const std::vector<Record>& records() const { return records_; }
    [[nodiscard]] std::uint64_t dropped() const { return dropped_; }

private:
    struct Tile {
        std::uint32_t command, slot;
        TileID tile;
        std::uint32_t rows, cols;
        StreamerCommand::Epilogue epilogue;
    };

    [[nodiscard]] std::string where(Cycle now) const { return "cycle::Combiner " + name() + " at cycle " + std::to_string(now); }

    void clear() {
        fifos_.assign(config_.cols, {});
        records_.assign(tiles_.size(), Record{});
        started_.assign(tiles_.size(), false);
        by_command_.clear();
        for (std::size_t i = 0; i < tiles_.size(); ++i) {
            records_[i].command = tiles_[i].command;
            by_command_[tiles_[i].command] = i;
        }
        head_ = 0;
        id_ = 0;
        dropped_ = 0;
    }

    Config config_;
    std::vector<Tile> tiles_;
    std::map<std::uint32_t, std::size_t> by_command_;   // a command's result tile
    std::vector<std::deque<ArrayResult>> fifos_;
    std::vector<Record> records_;
    std::vector<bool> started_;
    std::size_t head_ = 0;
    std::uint64_t id_ = 0, dropped_ = 0;
    std::vector<Link<ArrayResult>*> north_;
    Link<L2RowWrite>* writes_ = nullptr;
};

// The slots' credits, every cycle: each slot's is held by the BlockMover or the Streamer, never
// both, and by neither while the L2's slot is live.
inline void check_l2_slot_credits(const BlockMover& bm, const Streamer& st, const L2& l2) {
    check_l2_slot_credits(bm, l2);
    for (std::uint32_t s = 0; s < l2.config().slots; ++s) {
        if (!st.holds_credit(s)) continue;
        if (bm.holds(s) || l2.live(s))
            throw std::logic_error("check_l2_slot_credits: " + st.name() + " holds slot " + std::to_string(s) +
                                   "'s credit while " +
                                   (bm.holds(s) ? bm.name() + " holds it too"
                                                : l2.name() + "'s slot is " +
                                                      sw::kpu::timing::SlotPool::name(l2.state(s))));
    }
}

}  // namespace sw::kpu::cycle
