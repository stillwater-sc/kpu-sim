// ============================================================================
// include/sw/kpu/program/record/domain_flow_record.hpp
// The domain-flow record (.dflow): a CSP program's L-B run as a sequence of CSP state transitions
// (docs/plans/domain-flow-debugger.md §3.4, step 2; #363). One step is one action; the frame is
// domains (residencies) in channels, at their L3 addresses; the program position is the walker's.
//
// A .dflow bundle is a directory in the shared columnar container (record/columnar.hpp):
//
//   manifest.json   format "kpu-dflow", version, level "L-B", run_id, the program (name, text),
//                   machine {l3, banks, place}, channels (capacity, or null: unbounded at L-B),
//                   processes, kinds, operands, loop variables, totals {actions, loads, stores},
//                   the step limit and whether it cut steps.bin, the refusal (if L-B refused the
//                   program), what L-B does not model, and tables{name: {file, rows, columns}}
//   steps.bin       one row per recorded action, in program order (cut at the step limit)
//   tiles.bin       the tiles steps and residencies name (operand, ti, tj), interned
//   residency.bin   one row per L3 residency, for the WHOLE run: tile, address, open, close
//   where.bin       the program positions steps name: line and a range of loops.bin, interned
//   loops.bin       one row per (loop, iteration) a position names: variable, value, bound, line
//   iteration.bin   one row per loop iteration at every depth, for the WHOLE run (and depth 0, the
//                   program): its first and last action, its loads, stores and peak occupancy
//
// There is no time column: L-B has no time (§3.7). Indices that name nothing are kNone8/32/64.
// The record is written by the run that is the oracle (driver::csp_reference, through its
// observer), never by a separate walk of the program (§6 invariant 1).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/record/tile_flow_record.hpp>   // RecordError

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::record {

inline constexpr std::uint32_t kDflowVersion = 1;
inline constexpr std::uint8_t kNone8 = 0xFF;
inline constexpr std::uint32_t kNone32 = 0xFFFFFFFFu;
inline constexpr std::uint64_t kNone64 = ~std::uint64_t{0};

struct DomainFlowRecord {
    // Step flags.
    static constexpr std::uint8_t kOpens = 1, kCloses = 2, kRetains = 4, kHasPlace = 8, kPinned = 16;
    // Residency flags.
    static constexpr std::uint8_t kLoaded = 1, kInherited = 2, kRetained = 4, kResPinned = 8;

    struct Operand { std::string name; std::uint64_t rows = 0, cols = 0, tile_rows = 0, tile_cols = 0; };
    struct Tile { std::uint32_t operand = 0, ti = 0, tj = 0; };
    struct Step {
        std::uint64_t action = 0;
        std::uint8_t kind = 0;
        std::uint8_t process = kNone8;              // csp::ProcessKind; none for the credit actions
        std::uint8_t from = kNone8, to = kNone8;    // csp::Chan; none unless the action moves a tile
        std::uint32_t tile = 0;                     // index into tiles
        std::uint64_t residency = kNone64;
        std::uint8_t flags = 0;
        std::uint64_t where = 0;                    // index into wheres
        std::array<std::uint32_t, csp::kChannels> held{};   // tiles each channel holds AFTER the action
        std::uint32_t l3_tile = kNone32, slot = kNone32, bank = kNone32;   // valid when kHasPlace
    };
    struct Residency {
        std::uint32_t tile = 0;
        std::uint8_t chan = static_cast<std::uint8_t>(csp::Chan::L3);
        std::uint64_t open = 0;
        std::uint64_t close = kNone64;              // its Release, or its Retain (kRetained)
        std::uint32_t l3_tile = 0, slot = 0, bank = 0;
        std::uint8_t flags = 0;
    };
    struct Where { std::uint32_t line = 0, loop_first = 0; std::uint16_t loop_count = 0; };
    struct Loop { std::uint32_t var = 0; std::int64_t value = 0, hi = 0; std::uint32_t line = 0; };
    struct Iteration {
        std::uint16_t depth = 0;                    // 0: the whole program
        std::uint32_t line = 0;                     // the loop's `for` line (0 at depth 0)
        std::int64_t value = 0;                     // the loop's index in this iteration
        std::uint64_t first = 0, last = 0;          // its first and last action
        std::uint64_t actions = 0, loads = 0, stores = 0;
        std::array<std::uint32_t, csp::kChannels> peak{};
    };
    struct Refusal { std::uint64_t action = 0; std::string what; };

    std::string program_name, program_text, run_id;
    std::uint32_t l3 = 0, banks = 1;
    std::string place = "fifo";
    std::vector<Operand> operands;
    std::vector<std::string> vars;                  // loop variables, by index
    std::vector<Tile> tiles;
    std::vector<Step> steps;
    std::vector<Residency> residencies;
    std::vector<Where> wheres;
    std::vector<Loop> loops;
    std::vector<Iteration> iterations;
    std::uint64_t actions = 0, loads = 0, stores = 0;
    std::uint64_t step_limit = 0;                   // 0: none
    bool cut = false;                               // steps.bin holds fewer than `actions` rows
    std::optional<Refusal> refusal;

    std::string tile_name(std::uint32_t t) const {
        const Tile& x = tiles.at(t);
        return operands.at(x.operand).name + "[" + std::to_string(x.ti) + "," + std::to_string(x.tj) + "]";
    }
};

// Write the bundle into `dir`, creating it. Throws RecordError.
void write_dflow(const DomainFlowRecord& rec, const std::string& dir);
// Read a bundle back. Throws RecordError on a malformed or unknown-version bundle.
DomainFlowRecord read_dflow(const std::string& dir);

// ----------------------------------------------------------------------------
// The recorder: an observer of the oracle's run (driver::csp_reference). Every field is read from
// the walker's Emission or the interpreter that just executed it (§3.2).
// ----------------------------------------------------------------------------
class DomainFlowRecorder final : public driver::CspObserver {
public:
    using DFR = DomainFlowRecord;

    // `text`: the program's source, kept in the manifest; `limit`: steps.bin holds at most this many
    // actions (0: all). The iteration summary and the residencies are always complete.
    DomainFlowRecorder(const csp::lang::Program& ast, std::string text, std::uint64_t limit) {
        rec_.program_name = ast.name;
        rec_.program_text = std::move(text);
        rec_.l3 = static_cast<std::uint32_t>(ast.l3);
        rec_.banks = static_cast<std::uint32_t>(ast.banks);
        rec_.place = ast.place;
        rec_.step_limit = limit;
        const TileProgram ops = csp::lang::declared_operands(ast);
        for (const auto& name : ops.operand_order()) {
            const auto& o = ops.operand(name);
            operand_index_[name] = static_cast<std::uint32_t>(rec_.operands.size());
            rec_.operands.push_back({name, o.rows, o.cols, o.tile_rows, o.tile_cols});
        }
    }

    void step(const csp::lang::Emission& e, const csp::BehavioralInterpreter& lb) override {
        const csp::Action& a = e.action;
        const std::uint64_t i = n_++;
        std::array<std::uint32_t, csp::kChannels> held{};
        for (std::size_t c = 0; c < csp::kChannels; ++c)
            held[c] = static_cast<std::uint32_t>(lb.held(static_cast<csp::Chan>(c)));
        const bool load = a.kind == csp::Action::Kind::Load, store = a.kind == csp::Action::Kind::Store;
        rec_.loads += load;
        rec_.stores += store;

        // Residencies: the whole run, opened and ended by the actions that open and end them.
        const std::uint32_t tile = tile_index(a.tile);
        if (a.residency != csp::kNone) {
            if (e.opens) {
                if (a.residency != rec_.residencies.size())
                    throw RecordError("dflow: residency " + std::to_string(a.residency) + " opens out of order");
                DFR::Residency r;
                r.tile = tile;
                r.open = i;
                r.l3_tile = a.place->l3_tile;
                r.slot = a.place->slot;
                r.bank = a.place->bank;
                r.flags = static_cast<std::uint8_t>((e.loaded ? DFR::kLoaded : 0) |
                                                    (a.kind == csp::Action::Kind::Inherit ? DFR::kInherited : 0) |
                                                    (e.pinned ? DFR::kResPinned : 0));
                rec_.residencies.push_back(r);
            } else if (a.kind == csp::Action::Kind::Release || a.kind == csp::Action::Kind::Retain) {
                DFR::Residency& r = rec_.residencies.at(a.residency);
                r.close = i;
                if (a.kind == csp::Action::Kind::Retain) r.flags |= DFR::kRetained;
            }
        }

        // The iteration summary: every loop iteration at every depth, for the whole run.
        iterate(i, e.where, held, load, store);

        // The step, while within the limit.
        if (rec_.step_limit && i >= rec_.step_limit) {
            rec_.cut = true;
            return;
        }
        DFR::Step s;
        s.action = i;
        s.kind = static_cast<std::uint8_t>(a.kind);
        if (a.process != csp::kNone) s.process = static_cast<std::uint8_t>(a.process);
        if (moves(a.kind)) {
            s.from = static_cast<std::uint8_t>(csp::from_chan(a.kind));
            s.to = static_cast<std::uint8_t>(csp::to_chan(a.kind));
        }
        s.tile = tile;
        s.residency = a.residency == csp::kNone ? kNone64 : a.residency;
        s.flags = static_cast<std::uint8_t>((e.opens ? DFR::kOpens : 0) |
                                            (a.kind == csp::Action::Kind::Release ? DFR::kCloses : 0) |
                                            (a.kind == csp::Action::Kind::Retain ? DFR::kRetains : 0) |
                                            (a.place ? DFR::kHasPlace : 0) | (e.pinned ? DFR::kPinned : 0));
        s.where = where_index(e.where);
        s.held = held;
        if (a.place) {
            s.l3_tile = a.place->l3_tile;
            s.slot = a.place->slot;
            s.bank = a.place->bank;
        }
        rec_.steps.push_back(s);
    }

    void refused(std::size_t action, const std::string& what) override {
        rec_.refusal = DFR::Refusal{action, what};
    }
    // The action L-B refused, if it refused one.
    std::optional<std::uint64_t> refused_at() const {
        return rec_.refusal ? std::optional<std::uint64_t>(rec_.refusal->action) : std::nullopt;
    }

    // The record, once the run has ended (or been refused). Closes the open iterations.
    DomainFlowRecord finish() {
        close_from(0);
        DFR::Iteration program;
        program.first = 0;
        program.last = n_ ? n_ - 1 : 0;
        program.actions = n_;
        program.loads = rec_.loads;
        program.stores = rec_.stores;
        program.peak = peak_;
        rec_.iterations.push_back(program);
        std::stable_sort(rec_.iterations.begin(), rec_.iterations.end(), [](const auto& x, const auto& y) {
            return x.depth != y.depth ? x.depth < y.depth : x.first < y.first;
        });
        rec_.actions = n_;
        return std::move(rec_);
    }

private:
    struct Open { DFR::Iteration it; };

    static bool moves(csp::Action::Kind k) {
        using K = csp::Action::Kind;
        return k == K::Load || k == K::Store || k == K::Move || k == K::Writeback || k == K::Feed || k == K::Drain;
    }

    std::uint32_t tile_index(const TileCoord& t) {
        const auto key = t.to_string();
        auto it = tile_index_.find(key);
        if (it != tile_index_.end()) return it->second;
        const auto op = operand_index_.find(t.operand);
        if (op == operand_index_.end()) throw RecordError("dflow: tile " + key + " names no declared operand");
        const auto idx = static_cast<std::uint32_t>(rec_.tiles.size());
        rec_.tiles.push_back({op->second, static_cast<std::uint32_t>(t.ti), static_cast<std::uint32_t>(t.tj)});
        tile_index_.emplace(key, idx);
        return idx;
    }

    std::uint32_t var_index(const std::string& v) {
        auto it = var_index_.find(v);
        if (it != var_index_.end()) return it->second;
        const auto idx = static_cast<std::uint32_t>(rec_.vars.size());
        rec_.vars.push_back(v);
        var_index_.emplace(v, idx);
        return idx;
    }

    // Consecutive actions from one statement in one iteration share a position.
    std::uint64_t where_index(const csp::lang::Where& w) {
        if (last_where_ && *last_where_ == w) return rec_.wheres.size() - 1;
        DFR::Where row;
        row.line = static_cast<std::uint32_t>(w.line);
        row.loop_first = static_cast<std::uint32_t>(rec_.loops.size());
        row.loop_count = static_cast<std::uint16_t>(w.loops.size());
        for (const auto& l : w.loops)
            rec_.loops.push_back({var_index(l.var), l.value, l.hi, static_cast<std::uint32_t>(l.line)});
        rec_.wheres.push_back(row);
        last_where_ = w;
        return rec_.wheres.size() - 1;
    }

    // An iteration at depth d is one value of the first d loops; it changes when any of them does.
    void iterate(std::uint64_t i, const csp::lang::Where& w, const std::array<std::uint32_t, csp::kChannels>& held,
                 bool load, bool store) {
        std::size_t k = 0;                          // the loops this action shares with the open ones
        while (k < open_.size() && k < w.loops.size() && open_[k].it.line == static_cast<std::uint32_t>(w.loops[k].line) &&
               open_[k].it.value == w.loops[k].value)
            ++k;
        close_from(k);
        for (std::size_t d = k; d < w.loops.size(); ++d) {
            Open o;
            o.it.depth = static_cast<std::uint16_t>(d + 1);
            o.it.line = static_cast<std::uint32_t>(w.loops[d].line);
            o.it.value = w.loops[d].value;
            o.it.first = i;
            open_.push_back(o);
        }
        for (Open& o : open_) {
            o.it.last = i;
            ++o.it.actions;
            o.it.loads += load;
            o.it.stores += store;
            for (std::size_t c = 0; c < csp::kChannels; ++c) o.it.peak[c] = std::max(o.it.peak[c], held[c]);
        }
        for (std::size_t c = 0; c < csp::kChannels; ++c) peak_[c] = std::max(peak_[c], held[c]);
    }

    void close_from(std::size_t k) {
        while (open_.size() > k) {
            rec_.iterations.push_back(open_.back().it);
            open_.pop_back();
        }
    }

    DomainFlowRecord rec_;
    std::uint64_t n_ = 0;
    std::map<std::string, std::uint32_t> operand_index_, tile_index_, var_index_;
    std::optional<csp::lang::Where> last_where_;
    std::vector<Open> open_;
    std::array<std::uint32_t, csp::kChannels> peak_{};
};

} // namespace sw::kpu::program::record
