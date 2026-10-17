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
//   snapshots.bin   (version 2) one row per recorded tile value: the action after which it was
//                   taken (kNone64: the run's final value), the tile, the channel that holds it,
//                   its shape (height x width), and where its elements start in elements.bin
//   elements.bin    (version 2) the snapshots' elements, float32, row-major, back to back
//   lt1_legs.bin    (version 3) the L-T1 overlay: one row per leg of an action as L-T1 timed it
//                   -- the action (the join key), its kind, process, lane, start and finish in
//                   cycles, the leg (a Store has two), and a Load's slot wait. Empty without one.
//   lt1_tiles.bin   (version 3) every final tile's proximity to L-B under L-T1 (tile_check):
//                   max |a - o|, max relative error, and whether it is within the bound
//   lt1_chains.bin  (version 4) the fabric pipeline: one row per accumulator's gemm chain L-T1
//                   timed as a systolic stream (docs/plans/l2-reuse-and-fabric-pipeline-view.md):
//                   its tile, compute tile, the PEs its C tile occupies, its first, last and Drain
//                   actions, and in0, front, in1, last, out1, link_delay in cycles. Empty when the
//                   deployment declares no fabric (manifest lt1.fabric is then null).
//   lt1_calls.bin   (version 4) each stream call: its action, its chain, and its place in it
//   l2_staging.bin  (version 5) L2 staging (docs/plans/l2-reuse-and-fabric-pipeline-view.md
//                   step 3): one row per staged copy -- its tile, compute tile and L2 slot, the
//                   Move that staged it and the L2Release that freed it, its L3 residency, and
//                   its Feeds (the reuse count). Steps carry the L2 address they write, read or
//                   free (l2_cf, l2_slot; flag kHasL2), the staging (staging), and kStages on a
//                   staging Move. The manifest's machine.l2 is the program's `l2 = N` (0: none).
//
// THE L-T1 OVERLAY (§3.7, step 5). L-B has no time; the overlay is a second run, L-T1's, of the same
// program, joined on the action index. Its cycles are UNCALIBRATED, as L-T1 always says, and only
// its two tables carry time. Its values are checked per tile against L-B's, the oracle.
//
// VALUES (§3.6, step 4). At L-B the values are the oracle's. A run records the FINAL value of every
// operand's every tile (from L-B's result), and, for the tiles selected (kpu-run --dflow-values),
// the tile's value in the channel each action leaves it in: Load (L3), Move (L2), Feed and Call
// (the fabric; a Call's is the accumulator so far), Drain (L2), Writeback (L3), Store (DRAM) and
// Inherit (L3). A byte cap bounds them: finals first; snapshots stop at the cap, and the manifest
// says where. The elements are float32, the format the operands carry (tile_check's ElementFormat).
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
#include <sw/kpu/program/tile_check.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::record {

// Version 2: tile values (snapshots, elements). A version-1 bundle has none and is still read.
// Version 3: the L-T1 overlay (lt1_legs, lt1_tiles). Version 4: the fabric pipeline (lt1_chains,
// lt1_calls, manifest lt1.fabric and lt1.one_leg). Version 5: L2 staging (l2_staging, the steps'
// l2_cf, l2_slot and staging, machine.l2). Versions 1 to 4 are still read.
inline constexpr std::uint32_t kDflowVersion = 5;
inline constexpr std::uint8_t kNone8 = 0xFF;
inline constexpr std::uint32_t kNone32 = 0xFFFFFFFFu;
inline constexpr std::uint64_t kNone64 = ~std::uint64_t{0};

struct DomainFlowRecord {
    // Step flags.
    static constexpr std::uint8_t kOpens = 1, kCloses = 2, kRetains = 4, kHasPlace = 8, kPinned = 16, kHasL2 = 32,
                                  kStages = 64;
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
        std::uint32_t l2_cf = kNone32, l2_slot = kNone32;                 // valid when kHasL2
        std::uint64_t staging = kNone64;                                    // index into stagings
    };
    struct Staging {
        std::uint32_t tile = 0, cf = 0, slot = 0;
        std::uint64_t open = 0;                     // the staging Move
        std::uint64_t close = kNone64;              // its L2Release
        std::uint64_t residency = kNone64;          // the L3 residency it copies
        std::uint64_t feeds = 0;                    // Feeds of the copy: its reuse count
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
    struct Snapshot {
        std::uint64_t step = kNone64;               // the action after which it was taken; none: final
        std::uint64_t first = 0;                    // its first element in `elements`
        std::uint32_t tile = 0, rows = 0, cols = 0;
        std::uint8_t chan = static_cast<std::uint8_t>(csp::Chan::Dram);
    };

    std::string program_name, program_text, run_id;
    std::uint32_t l3 = 0, banks = 1;
    std::uint32_t l2 = 0;                           // the program's `l2 = N`: staging slots per compute tile
    std::vector<Staging> stagings;
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
    // Values (version 2).
    std::vector<Snapshot> snapshots;
    std::vector<float> elements;
    std::string values_selected;                    // "" (none), "all", or the selection as given
    std::uint64_t values_cap = 0;                   // bytes
    std::optional<std::uint64_t> values_cut_at;     // the first action whose snapshot the cap refused
    std::string finals_note;                        // empty: finals recorded; else why not
    // The L-T1 overlay (version 3); `lt1` is false when the run had none.
    struct Leg {
        std::uint64_t action = 0;
        std::uint64_t start = 0, finish = 0, slot_wait = 0;     // cycles, L-T1's, uncalibrated
        std::uint32_t lane = 0;
        std::uint8_t kind = 0, proc = kNone8, leg = 0;          // proc: none for a credit action
    };
    struct Proximity { std::uint32_t tile = 0; double max_abs = 0.0, max_rel = 0.0; bool pass = true; };
    bool lt1 = false;
    std::string lt1_run_id;
    std::uint64_t lt1_makespan = 0, lt1_slot_bound = 0, lt1_slot_wait_cycles = 0;
    std::vector<std::pair<std::string, std::uint32_t>> lt1_lanes;   // process -> lanes
    double lt1_atol = 0.0, lt1_rtol = 0.0;
    bool lt1_pass = true;
    std::vector<Leg> legs;
    std::vector<Proximity> proximity;
    // The fabric pipeline (version 4): present when the deployment gives the PE array.
    struct Chain {
        std::uint32_t tile = 0, lane = 0, rows = 0, cols = 0, calls = 0;
        std::uint64_t first_action = 0, last_action = 0, drain_action = kNone64;   // none: no Drain recorded
        std::uint64_t in0 = 0, front = 0, in1 = 0, last = kNone64, out1 = kNone64, link_delay = 0;   // none: calls not over
    };
    struct Call { std::uint64_t action = 0; std::uint32_t chain = 0, pos = 0; };
    bool fabric = false;
    std::uint32_t pe_rows = 0, pe_cols = 0;
    std::uint64_t link_delay_cycles = 0;
    std::vector<std::pair<std::string, std::uint64_t>> one_leg;     // calls timed as one leg, by why
    std::vector<Chain> chains;
    std::vector<Call> calls;

    std::string tile_name(std::uint32_t t) const {
        const Tile& x = tiles.at(t);
        return operands.at(x.operand).name + "[" + std::to_string(x.ti) + "," + std::to_string(x.tj) + "]";
    }
};

// Write the bundle into `dir`, creating it. Throws RecordError.
void write_dflow(const DomainFlowRecord& rec, const std::string& dir);
// Read a bundle back. Throws RecordError on a malformed or unknown-version bundle.
DomainFlowRecord read_dflow(const std::string& dir);

// The L-T1 overlay: L-T1's run of the same program, joined on the action index, and its values
// checked per tile against L-B's (`oracle`, the reference). Tiles are interned into `rec.tiles`.
inline void attach_lt1(DomainFlowRecord& rec, const driver::CspLevelOutcome& lt1, const TileProgram& oracle,
                       const std::string& run_id) {
    using DFR = DomainFlowRecord;
    using TI = csp::TransactionalInterpreter;
    std::map<std::string, std::uint32_t> tile_at, op_at;
    for (std::uint32_t i = 0; i < rec.operands.size(); ++i) op_at[rec.operands[i].name] = i;
    for (std::uint32_t t = 0; t < rec.tiles.size(); ++t) tile_at[rec.tile_name(t)] = t;
    auto intern = [&](const TileCoord& c) {
        const auto k = c.to_string();
        auto it = tile_at.find(k);
        if (it != tile_at.end()) return it->second;
        const auto op = op_at.find(c.operand);
        if (op == op_at.end()) throw RecordError("dflow: L-T1 names " + k + ", no declared operand");
        const auto idx = static_cast<std::uint32_t>(rec.tiles.size());
        rec.tiles.push_back({op->second, static_cast<std::uint32_t>(c.ti), static_cast<std::uint32_t>(c.tj)});
        tile_at.emplace(k, idx);
        return idx;
    };
    rec.lt1 = true;
    rec.lt1_run_id = run_id;
    rec.lt1_makespan = lt1.makespan;
    rec.lt1_slot_bound = lt1.slot_bound;
    rec.lt1_slot_wait_cycles = lt1.slot_wait_cycles;
    for (const auto& [p, n] : lt1.lanes) rec.lt1_lanes.emplace_back(p, static_cast<std::uint32_t>(n));
    std::map<std::size_t, std::uint8_t> legs_of;
    for (const auto& r : lt1.records) {
        DFR::Leg l;
        l.action = r.action;
        l.kind = static_cast<std::uint8_t>(r.kind);
        l.proc = TI::has_process(r.kind) ? static_cast<std::uint8_t>(r.proc) : kNone8;
        l.lane = static_cast<std::uint32_t>(r.lane);
        l.start = r.start;
        l.finish = r.finish;
        l.slot_wait = r.slot_wait;
        l.leg = legs_of[r.action]++;
        rec.legs.push_back(l);
        if (r.chain != TI::kNoChain) rec.calls.push_back({r.action, r.chain, r.chain_pos});
    }
    rec.fabric = lt1.pe_rows > 0;
    rec.pe_rows = lt1.pe_rows;
    rec.pe_cols = lt1.pe_cols;
    rec.link_delay_cycles = lt1.link_delay_cycles;
    for (const auto& [why, n] : lt1.one_leg) rec.one_leg.emplace_back(why, n);
    for (const TI::Chain& c : lt1.chains) {
        DFR::Chain x;
        x.tile = intern(c.tile);
        x.lane = static_cast<std::uint32_t>(c.lane);
        x.rows = c.rows;
        x.cols = c.cols;
        x.calls = c.calls;
        x.first_action = c.first_action;
        x.last_action = c.last_action;
        x.in0 = c.in0;
        x.front = c.front;
        x.in1 = c.in1;
        if (c.closed) {                     // its calls are over: last and out1 are known
            x.last = c.last;
            x.out1 = c.out1;
            x.link_delay = c.link_delay;
        }
        if (c.drain_action != TI::kNoAction) x.drain_action = c.drain_action;
        rec.chains.push_back(x);
    }
    const ElementFormat format{};
    const Proximity bound = proximity_for(format);
    rec.lt1_atol = bound.atol;
    rec.lt1_rtol = bound.rtol;
    const OperandsCheck c = check_operands(lt1.values, oracle, format, bound);
    rec.lt1_pass = c.pass;
    for (const TileCheck& t : c.operands)
        for (const auto& pt : t.per_tile) rec.proximity.push_back({intern(pt.tile), pt.max_abs, pt.max_rel, pt.pass});
}

// Which tiles' values to record at each action (kpu-run --dflow-values): every tile, the tiles of
// named operands, or named tiles. Empty: none (the finals are recorded regardless).
struct ValueSelection {
    bool all = false;
    std::vector<std::string> operands;              // every tile of these
    std::vector<std::string> tiles;                 // TileCoord::to_string(), e.g. "C[2,3]"
    std::string as_given;                           // for the manifest
    std::uint64_t cap_bytes = 64ull << 20;
    bool empty() const { return !all && operands.empty() && tiles.empty(); }
    bool selects(const TileCoord& t) const {
        if (all) return true;
        for (const auto& o : operands) if (o == t.operand) return true;
        const std::string k = t.to_string();
        for (const auto& x : tiles) if (x == k) return true;
        return false;
    }
};

// Parse "C[2,3],A[1,0],B" (commas inside brackets belong to the tile) or "all". Throws
// std::invalid_argument on a malformed item.
inline ValueSelection parse_value_selection(const std::string& text) {
    ValueSelection v;
    v.as_given = text;
    if (text == "all") { v.all = true; return v; }
    std::string item;
    int depth = 0;
    auto flush = [&] {
        if (item.empty()) throw std::invalid_argument("--dflow-values: an empty item in '" + text + "'");
        const auto lb = item.find('[');
        if (lb == std::string::npos) {
            v.operands.push_back(item);
        } else {
            const auto comma = item.find(',', lb), rb = item.find(']', lb);
            if (lb == 0 || comma == std::string::npos || rb != item.size() - 1 || comma > rb)
                throw std::invalid_argument("--dflow-values: '" + item + "' is not operand[ti,tj]");
            const std::string a = item.substr(lb + 1, comma - lb - 1), b = item.substr(comma + 1, rb - comma - 1);
            auto digits = [](const std::string& x) { return !x.empty() && x.find_first_not_of("0123456789") == std::string::npos; };
            if (!digits(a) || !digits(b)) throw std::invalid_argument("--dflow-values: '" + item + "' is not operand[ti,tj]");
            try {
                v.tiles.push_back(item.substr(0, lb) + "[" + std::to_string(std::stoull(a)) + "," + std::to_string(std::stoull(b)) + "]");
            } catch (const std::out_of_range&) {
                throw std::invalid_argument("--dflow-values: '" + item + "' has a coordinate out of range");
            }
        }
        item.clear();
    };
    for (char c : text) {
        if (c == ' ') continue;
        if (c == '[') ++depth;
        if (c == ']') --depth;
        if (c == ',' && depth == 0) flush();
        else item.push_back(c);
    }
    flush();
    return v;
}

// ----------------------------------------------------------------------------
// The recorder: an observer of the oracle's run (driver::csp_reference). Every field is read from
// the walker's Emission or the interpreter that just executed it (§3.2).
// ----------------------------------------------------------------------------
class DomainFlowRecorder final : public driver::CspObserver {
public:
    using DFR = DomainFlowRecord;

    // `text`: the program's source, kept in the manifest; `limit`: steps.bin holds at most this many
    // actions (0: all). The iteration summary and the residencies are always complete.
    DomainFlowRecorder(const csp::lang::Program& ast, std::string text, std::uint64_t limit,
                       ValueSelection values = {}) : values_(std::move(values)) {
        rec_.program_name = ast.name;
        rec_.program_text = std::move(text);
        rec_.l3 = static_cast<std::uint32_t>(ast.l3);
        rec_.banks = static_cast<std::uint32_t>(ast.banks);
        rec_.l2 = static_cast<std::uint32_t>(ast.l2);
        rec_.place = ast.place;
        rec_.step_limit = limit;
        shapes_ = csp::lang::declared_operands(ast);
        std::uint64_t finals = 0;
        for (const auto& name : shapes_.operand_order()) {
            const auto& o = shapes_.operand(name);
            operand_index_[name] = static_cast<std::uint32_t>(rec_.operands.size());
            rec_.operands.push_back({name, o.rows, o.cols, o.tile_rows, o.tile_cols});
            finals += static_cast<std::uint64_t>(o.rows) * o.cols * sizeof(float);
        }
        for (const auto& name : values_.operands)
            if (!operand_index_.count(name)) throw RecordError("--dflow-values: " + name + " is not a declared operand");
        for (const auto& k : values_.tiles)
            if (!operand_index_.count(k.substr(0, k.find('['))))
                throw RecordError("--dflow-values: " + k + " names no declared operand");
        rec_.values_selected = values_.empty() ? "" : values_.as_given;
        rec_.values_cap = values_.cap_bytes;
        // The finals come first in the cap; the snapshots get what is left.
        if (finals > values_.cap_bytes) {
            finals_fit_ = false;
            budget_ = values_.cap_bytes;
        } else {
            budget_ = values_.cap_bytes - finals;
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

        // Stagings: a staging Move opens one, its Feeds count, its L2Release ends it.
        if (a.staging != csp::kNone) {
            if (e.opens_staging) {
                if (a.staging != rec_.stagings.size())
                    throw RecordError("dflow: staging " + std::to_string(a.staging) + " opens out of order");
                DFR::Staging st;
                st.tile = tile;
                st.cf = a.l2->cf;
                st.slot = a.l2->slot;
                st.open = i;
                st.residency = a.residency == csp::kNone ? kNone64 : a.residency;
                rec_.stagings.push_back(st);
            } else if (a.kind == csp::Action::Kind::Feed) {
                ++rec_.stagings.at(a.staging).feeds;
            } else if (a.kind == csp::Action::Kind::L2Release) {
                rec_.stagings.at(a.staging).close = i;
            }
        }

        // The iteration summary: every loop iteration at every depth, for the whole run.
        iterate(i, e.where, held, load, store);

        // The tile's value where this action left it, if it is selected (§3.6).
        if (!values_.empty() && !rec_.values_cut_at && values_.selects(a.tile)) {
            if (const auto chan = lands_in(a.kind)) {
                if (const std::vector<float>* v = lb.tile_in(*chan, a.tile)) {
                    if (static_cast<std::uint64_t>(v->size()) * sizeof(float) > budget_) {
                        rec_.values_cut_at = i;
                    } else {
                        budget_ -= v->size() * sizeof(float);
                        snapshot(i, tile, *chan, a.tile, *v);
                    }
                }
            }
        }

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
                                            (a.place ? DFR::kHasPlace : 0) | (e.pinned ? DFR::kPinned : 0) |
                                            (a.l2 ? DFR::kHasL2 : 0) | (e.opens_staging ? DFR::kStages : 0));
        s.where = where_index(e.where);
        s.held = held;
        if (a.place) {
            s.l3_tile = a.place->l3_tile;
            s.slot = a.place->slot;
            s.bank = a.place->bank;
        }
        if (a.l2) {
            s.l2_cf = a.l2->cf;
            s.l2_slot = a.l2->slot;
            s.staging = a.staging;
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

    // The record, once the run has ended (or been refused). Closes the open iterations. `finals`:
    // the operands as L-B left them (CspReference::values), whose every tile is recorded.
    DomainFlowRecord finish(const TileProgram* finals = nullptr) {
        if (!finals) {
            rec_.finals_note = rec_.refusal ? "the run was refused, so it has no final values" : "not given to the recorder";
        } else if (!finals_fit_) {
            rec_.finals_note = "the operands exceed --dflow-values-cap (" + std::to_string(values_.cap_bytes) + " bytes)";
        } else {
            for (const auto& name : shapes_.operand_order()) {
                const TensorOperand& o = finals->operand(name);
                for (Dim ti = 0; ti < o.n_tile_rows(); ++ti)
                    for (Dim tj = 0; tj < o.n_tile_cols(); ++tj) {
                        const TileCoord t{name, ti, tj};
                        std::vector<float> v;
                        for (Dim r = o.row_begin(ti); r < o.row_end(ti); ++r)
                            for (Dim c = o.col_begin(tj); c < o.col_end(tj); ++c) v.push_back(o.at(r, c));
                        snapshot(kNone64, tile_index(t), csp::Chan::Dram, t, v);
                    }
            }
        }
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

    // The channel an action leaves its tile in (none: Release and Retain move nothing).
    static std::optional<csp::Chan> lands_in(csp::Action::Kind k) {
        using K = csp::Action::Kind;
        switch (k) {
            case K::Load: case K::Writeback: case K::Inherit: return csp::Chan::L3;
            case K::Move: case K::Drain:                      return csp::Chan::L2;
            case K::Feed: case K::Call:                       return csp::Chan::Cf;
            case K::Store:                                    return csp::Chan::Dram;
            default:                                          return std::nullopt;
        }
    }

    void snapshot(std::uint64_t step, std::uint32_t tile, csp::Chan chan, const TileCoord& t, const std::vector<float>& v) {
        const TensorOperand& o = shapes_.operand(t.operand);
        DFR::Snapshot s;
        s.step = step;
        s.tile = tile;
        s.chan = static_cast<std::uint8_t>(chan);
        s.rows = static_cast<std::uint32_t>(o.row_end(t.ti) - o.row_begin(t.ti));
        s.cols = static_cast<std::uint32_t>(o.col_end(t.tj) - o.col_begin(t.tj));
        if (static_cast<std::uint64_t>(s.rows) * s.cols != v.size())
            throw RecordError("dflow: " + t.to_string() + " holds " + std::to_string(v.size()) + " elements, not " +
                              std::to_string(s.rows) + "x" + std::to_string(s.cols));
        s.first = rec_.elements.size();
        rec_.elements.insert(rec_.elements.end(), v.begin(), v.end());
        rec_.snapshots.push_back(s);
    }

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
    ValueSelection values_;
    TileProgram shapes_;
    std::uint64_t budget_ = 0;                      // snapshot bytes left under the cap
    bool finals_fit_ = true;
    std::uint64_t n_ = 0;
    std::map<std::string, std::uint32_t> operand_index_, tile_index_, var_index_;
    std::optional<csp::lang::Where> last_where_;
    std::vector<Open> open_;
    std::array<std::uint32_t, csp::kChannels> peak_{};
};

} // namespace sw::kpu::program::record
