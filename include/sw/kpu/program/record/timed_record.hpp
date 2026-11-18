// ============================================================================
// include/sw/kpu/program/record/timed_record.hpp
// The domain-flow record of a timed level's run (#402): L-T1's or L-CA's actions as the shared
// record's events, with their cycles (docs/plans/shared-tile-flow-record.md §5, version 6.2).
//
// A timed level executes the same CSP program as L-B (ADR 0002), so its record keeps L-B's steps,
// tiles, residencies, positions and iterations, and adds what only it has: time, and the events.
// Its values are its own: the level's final tiles. It has no per-step snapshots (L-B's record has
// them) and no L-T1 overlay (it is the run the overlay summarizes).
//
// The events come from the level's legs (TransactionalInterpreter::Record; L-CA's are matched from
// its protocol events by timing/csp_records.hpp), one transaction per data step, joined to its
// action, as the RTL sample writes them (tools/trace/dflow_rtl_sample.py):
//
//   LOAD       DMA         dram -> l3   at its slot          then a TOKEN l3 -> l2 (tile ready)
//   MOVE       BLOCK_MOVE  l3 -> l2     slot -> its L2 slot  then a TOKEN l2 -> l1
//   FEED       STREAM      l2 -> l1     from its L2 slot     then, unstaged, a CREDIT l2 -> l3
//   CALL       COMPUTE     l1 -> acc    (the accumulator's tile)
//   DRAIN      DRAIN       acc -> l2    into its L2 slot     then a CREDIT acc -> l1 and a TOKEN l2 -> l3
//   WRITEBACK  BLOCK_MOVE  l2 -> l3     -> its slot          then a TOKEN l3 -> dram
//   STORE      DMA         l3 -> dram   from its slot        (its two legs, the ejection and the write, as one)
//   RELEASE    CREDIT      l3 -> dram   its slot is free
//   L2RELEASE  CREDIT      l2 -> l3     a staged copy's or a partial's L2 slot is free
//
// L2 and the accumulator are addressed by slot (6.2): an event's L2 address is the action's L2
// slot (a staged copy, or an L2 accumulator's partial), or none. No level models their rows, so
// there are no row events (BM_ROW, STR_ROW, DRAIN_ROW) and the record says why.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/timing/csp_driver.hpp>
#include <sw/kpu/timing/l2_banks.hpp>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace sw::kpu::program::record {

// The shared record's enums (§3), by index: event_kinds, layers and processes.
namespace ev {
enum Kind : std::uint8_t { Dma, BlockMove, Stream, Compute, Drain, Credit, Token, BmRow, StrRow, DrainRow };
enum Layer : std::uint8_t { Dram, L3, L2, Acc, L1 };
enum Process : std::uint8_t { PDma, PBm, PStr, PCf };
}  // namespace ev

// `lb` is L-B's record of the run (every step: not cut, not refused), `records` the level's legs,
// `values` its final values. `pe_rows` x `pe_cols` is the array's edge, when the deployment gives it.
inline DomainFlowRecord timed_record(const DomainFlowRecord& lb, const std::string& level,
                                     const std::vector<csp::TransactionalInterpreter::Record>& records,
                                     const TileProgram& values, std::uint32_t pe_rows = 0, std::uint32_t pe_cols = 0,
                                     const std::vector<timing::CspDriver::Result::L2Booking>* bookings = nullptr,
                                     const timing::L2Banks::Geometry* l2g = nullptr) {
    using DFR = DomainFlowRecord;
    using K = csp::Action::Kind;
    if (lb.cut) throw RecordError("record: a timed record needs every step; L-B's was cut at --dflow-limit");
    if (lb.refusal) throw RecordError("record: L-B refused the program, so no level ran it to a timed record");
    DFR r = lb;
    r.level = level;
    r.run_id.clear();
    r.level_notes = {
        "row progress (BM_ROW, STR_ROW, DRAIN_ROW): " + level + " moves each tile whole, at its process's rate; "
        "row events arrive with a level that models rows (the L2 banks, #388; the PE grid, #401)",
        "L2 and the accumulator by slot, not by banks and rows (version 6.2): an event's L2 address is its staged "
        "copy's or its partial's slot, or none; the accumulator's and the array edge's, none",
        "values: " + level + "'s own final tiles; no per-step snapshots (L-B's record has them)"};
    if (level == "L-T1") r.level_notes.push_back("time: L-T1's cycles, uncalibrated against L-CA");
    // No L-T1 overlay: this record is a level's own run.
    r.lt1 = false;
    r.legs.clear();
    r.proximity.clear();
    r.chains.clear();
    r.calls.clear();
    r.one_leg.clear();
    r.fabric = pe_rows > 0;
    r.pe_rows = pe_rows;
    r.pe_cols = pe_cols;
    // Values: the level's finals, in L-B's tile order.
    r.snapshots.clear();
    r.elements.clear();
    r.values_selected.clear();
    r.values_cut_at.reset();
    r.finals_note.clear();
    std::map<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>, std::uint32_t> tile_of;
    for (std::uint32_t t = 0; t < r.tiles.size(); ++t)
        tile_of[{r.tiles[t].operand, r.tiles[t].ti, r.tiles[t].tj}] = t;
    for (std::uint32_t oi = 0; oi < r.operands.size(); ++oi) {
        const TensorOperand& o = values.operand(r.operands[oi].name);
        for (Dim ti = 0; ti < o.n_tile_rows(); ++ti)
            for (Dim tj = 0; tj < o.n_tile_cols(); ++tj) {
                auto it = tile_of.find({oi, ti, tj});
                if (it == tile_of.end()) {
                    const std::uint32_t t = static_cast<std::uint32_t>(r.tiles.size());
                    r.tiles.push_back({oi, static_cast<std::uint32_t>(ti), static_cast<std::uint32_t>(tj)});
                    it = tile_of.emplace(std::make_tuple(oi, std::uint64_t(ti), std::uint64_t(tj)), t).first;
                }
                DFR::Snapshot s;
                s.step = kNone64;
                s.first = r.elements.size();
                s.tile = it->second;
                s.rows = static_cast<std::uint32_t>(o.row_end(ti) - o.row_begin(ti));
                s.cols = static_cast<std::uint32_t>(o.col_end(tj) - o.col_begin(tj));
                s.chan = static_cast<std::uint8_t>(csp::Chan::Dram);
                for (Dim rr = o.row_begin(ti); rr < o.row_end(ti); ++rr)
                    for (Dim cc = o.col_begin(tj); cc < o.col_end(tj); ++cc) {
                        const numeric::Word w = o.word(rr, cc);     // a typed element is a bit pattern
                        float f = 0.0f;
                        std::memcpy(&f, &w, sizeof w);
                        r.elements.push_back(f);
                    }
                r.snapshots.push_back(s);
            }
    }
    // Each action's legs, earliest start to latest finish: a Store's two are one transaction.
    struct Span { std::uint64_t start = ~0ull, finish = 0; std::uint32_t lane = 0; bool any = false; };
    std::vector<Span> span(r.steps.size());
    for (const auto& x : records) {
        if (x.action >= span.size()) continue;
        Span& s = span[x.action];
        if (!s.any || x.start < s.start) s.start = x.start;
        s.finish = std::max<std::uint64_t>(s.finish, x.finish);
        if (!s.any || x.proc == csp::TransactionalInterpreter::Proc::Dma) s.lane = static_cast<std::uint32_t>(x.lane);
        s.any = true;
    }
    r.events.clear();
    auto add = [&](const DFR::Step& st, std::uint8_t kind, std::uint8_t proc, std::uint8_t src, std::uint8_t dst,
                   std::uint64_t src_addr, std::uint64_t dst_addr, std::uint64_t cycle, std::uint64_t end,
                   std::uint32_t lane = 0) {
        DFR::Event e;
        e.action = st.action;
        e.where = st.where;
        e.tile = st.tile;
        e.kind = kind;
        e.process = proc;
        e.src = src;
        e.dst = dst;
        e.src_addr = src_addr;
        e.dst_addr = dst_addr;
        e.cycle = cycle;
        e.end = end;
        e.lane = lane;
        r.events.push_back(e);
    };
    const std::uint8_t none = kNone8;
    // A banked L2's bookings, by tile, mover or streamer, the clock the leg started and direction.
    const bool banked = bookings && l2g && !bookings->empty();
    auto bkey = [](const std::string& tile, bool mover, std::uint64_t at, bool into) {
        return tile + (mover ? "/bm/" : "/str/") + std::to_string(at) + (into ? "/w" : "/r");
    };
    std::map<std::string, const timing::L2Banks::Booking*> booked;
    std::uint64_t rows_used = 1;
    if (banked)
        for (const auto& [t, b] : *bookings) {
            std::string name = t.operand + "[" + std::to_string(t.ti) + "," + std::to_string(t.tj) + "]";
            booked[bkey(name, b.who == timing::L2Banks::Client::BlockMover, b.now, b.kind != timing::L2Banks::Access::Read)] = &b;
            rows_used = std::max<std::uint64_t>(rows_used, b.first_row + b.rows.size());
        }
    std::vector<std::pair<std::uint64_t, const timing::L2Banks::Booking*>> with_rows;   // action -> its booking
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint64_t> staged_row;      // (cf, L2 slot) -> first row
    // Each tile's last transaction out of a layer, in program order, and whether it was a Store: a
    // Release's credit follows its own tile's last move out of L3 (an L2Release's, out of L2).
    struct Out { std::uint64_t end = 0; bool store = false; };
    std::map<std::pair<std::uint32_t, std::uint8_t>, Out> last_out;
    std::size_t at_store = 0, at_move = 0, in_place = 0;
    auto credit_after = [&](const DFR::Step& st, std::uint8_t layer, std::uint64_t src_addr, std::uint64_t t) {
        const auto it = last_out.find({st.tile, layer});
        if (it == last_out.end()) {
            // A tile no transaction moved out of its layer (an init, a bias or a scale's vectors,
            // which the Combiner reads in place): the record has no transaction for that read, so
            // the slot's credit has none to follow, and is left out.
            ++in_place;
            return;
        }
        if (t < it->second.end) {
            // The level freed the slot before the transaction ended: a store, at its ejection into
            // the DMA engine's buffer, which the format has no layer for. Stamped at its end.
            t = it->second.end;
            ++(it->second.store ? at_store : at_move);
        }
        add(st, ev::Credit, none, layer, layer == ev::L3 ? ev::Dram : ev::L3, src_addr, kNone64, t, t);
        // The residency is over: a later one of the same tile that nothing moves out (read in place
        // by the Combiner) has no transaction for its credit to follow (#469's review).
        last_out.erase(it);
    };
    for (const DFR::Step& st : r.steps) {
        const K k = static_cast<K>(st.kind);
        const bool data = k == K::Load || k == K::Store || k == K::Move || k == K::Writeback || k == K::Feed ||
                          k == K::Call || k == K::Drain;
        const Span& s = span.at(st.action);
        if (!s.any) {
            if (data)
                throw RecordError("record: " + level + " has no leg for action " + std::to_string(st.action) + " (" +
                                  csp::to_string(k) + " " + r.tile_name(st.tile) + ")");
            continue;
        }
        const std::uint64_t slot = (st.flags & DFR::kHasPlace) ? std::uint64_t(st.slot) : kNone64;
        // A staged copy or a partial: the step's own, whatever address a banked L2 gives it.
        const bool staged = (st.flags & DFR::kHasL2) != 0;
        std::uint64_t l2 = staged ? std::uint64_t(st.l2_slot) : kNone64;
        // A banked L2 (#401): the transfer's booking gives its L2 address, the first row it took.
        const timing::L2Banks::Booking* bk = nullptr;
        if (banked) {
            const bool into = k == K::Move || k == K::Drain, out_of = k == K::Feed || k == K::Writeback;
            if (into || out_of) {
                const auto it = booked.find(bkey(r.tile_name(st.tile), k == K::Move || k == K::Writeback, s.start, into));
                if (it != booked.end()) bk = it->second;
                l2 = bk ? bk->first_row : kNone64;
                // A staged copy or a partial: its rows, for the L2Release that frees them.
                if (bk && into && staged) staged_row[{st.l2_cf, st.l2_slot}] = bk->first_row;
            } else if (k == K::L2Release) {
                const auto it = staged_row.find({st.l2_cf, st.l2_slot});
                l2 = it != staged_row.end() ? it->second : kNone64;
            }
        }
        if (bk) with_rows.push_back({st.action, bk});
        const std::uint64_t t0 = s.start, t1 = s.finish;
        const std::uint32_t lane = s.lane;          // a token or credit is its process's, on its lane
        auto out_of = [&](std::uint8_t layer, bool store = false) { last_out[{st.tile, layer}] = {t1, store}; };
        switch (k) {
            case K::Load:
                add(st, ev::Dma, ev::PDma, ev::Dram, ev::L3, kNone64, slot, t0, t1, lane);
                add(st, ev::Token, ev::PDma, ev::L3, ev::L2, kNone64, kNone64, t1, t1, lane);
                break;
            case K::Move:
                add(st, ev::BlockMove, ev::PBm, ev::L3, ev::L2, slot, l2, t0, t1, lane);
                add(st, ev::Token, ev::PBm, ev::L2, ev::L1, l2, kNone64, t1, t1, lane);
                out_of(ev::L3);
                break;
            case K::Feed:
                add(st, ev::Stream, ev::PStr, ev::L2, ev::L1, l2, kNone64, t0, t1, lane);
                if (!staged) add(st, ev::Credit, ev::PStr, ev::L2, ev::L3, l2, kNone64, t1, t1, lane);
                out_of(ev::L2);
                break;
            case K::Call:
                add(st, ev::Compute, ev::PCf, ev::L1, ev::Acc, kNone64, kNone64, t0, t1, lane);
                break;
            case K::Drain:
                add(st, ev::Drain, ev::PStr, ev::Acc, ev::L2, kNone64, l2, t0, t1, lane);
                add(st, ev::Credit, ev::PStr, ev::Acc, ev::L1, kNone64, kNone64, t1, t1, lane);
                add(st, ev::Token, ev::PStr, ev::L2, ev::L3, l2, kNone64, t1, t1, lane);
                break;
            case K::Writeback:
                add(st, ev::BlockMove, ev::PBm, ev::L2, ev::L3, l2, slot, t0, t1, lane);
                add(st, ev::Token, ev::PBm, ev::L3, ev::Dram, slot, kNone64, t1, t1, lane);
                out_of(ev::L2);
                break;
            case K::Store:
                add(st, ev::Dma, ev::PDma, ev::L3, ev::Dram, slot, kNone64, t0, t1, lane);
                out_of(ev::L3, true);
                break;
            case K::Release:
                credit_after(st, ev::L3, slot, t1);
                break;
            case K::L2Release:
                credit_after(st, ev::L2, l2, t1);
                break;
            default:
                break;                      // Inherit, Retain: no data moves, no credit returns
        }
    }
    if (at_store)
        r.level_notes.push_back(std::to_string(at_store) + " L3 credits stamped at their store's end: " + level +
                                " frees a slot when its tile is ejected into the DMA engine's buffer, which the "
                                "record has no layer for, so a store is one transaction, ejection to DRAM write");
    if (at_move)
        r.level_notes.push_back(std::to_string(at_move) + " credits stamped at the end of their tile's last move out "
                                "of the layer: " + level + " returned them earlier");
    if (in_place)
        r.level_notes.push_back(std::to_string(in_place) + " credits left out: their tiles were read in place "
                                "(the Combiner's init, bias and scale vectors, read from L3), which the record has "
                                "no transaction for");
    // A banked L2 (#401): the geometry in rows, and each transfer's rows as row events under it.
    std::size_t too_tall = 0;
    if (banked) {
        DFR::L2Rows g;
        g.banks = l2g->banks;
        g.row_bytes = l2g->row_bytes;
        const std::uint64_t total = l2g->rows_per_bank ? std::uint64_t(l2g->rows_per_bank) * l2g->banks
                                                       : (rows_used + l2g->banks - 1) / l2g->banks * l2g->banks;
        g.rows = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, total));
        r.l2_rows = g;
        for (const auto& [action, bk] : with_rows) {
            const DFR::Step& st = r.steps.at(action);
            const K k = static_cast<K>(st.kind);
            const std::uint8_t kind = k == K::Feed ? ev::StrRow : k == K::Drain ? ev::DrainRow : ev::BmRow;
            const std::uint8_t proc = k == K::Move || k == K::Writeback ? ev::PBm : ev::PStr;
            const auto& tl = r.tiles.at(st.tile);
            const auto& op = r.operands.at(tl.operand);
            const std::uint64_t tile_rows = std::min<std::uint64_t>(op.tile_rows, op.rows - tl.ti * op.tile_rows);
            if (bk->rows.size() > tile_rows) { ++too_tall; continue; }      // E3: a row inside the tile
            const std::uint8_t src = k == K::Move ? ev::L3 : k == K::Drain ? ev::Acc : ev::L2;
            const std::uint8_t dst = k == K::Move || k == K::Drain ? ev::L2 : k == K::Feed ? ev::L1 : ev::L3;
            const std::uint64_t slot = (st.flags & DFR::kHasPlace) ? std::uint64_t(st.slot) : kNone64;
            for (std::size_t q = 0; q < bk->rows.size(); ++q) {
                DFR::Event e;
                e.action = st.action;
                e.where = st.where;
                e.tile = st.tile;
                e.kind = kind;
                e.process = proc;
                e.src = src;
                e.dst = dst;
                const std::uint64_t row = l2g->row(bk->first_row, q);
                e.src_addr = src == ev::L2 ? row : src == ev::L3 ? slot : kNone64;
                e.dst_addr = dst == ev::L2 ? row : dst == ev::L3 ? slot : kNone64;
                e.row = static_cast<std::uint32_t>(q);
                e.beat = 0;
                e.cycle = bk->rows[q];
                e.end = bk->rows[q] + bk->per_row;
                e.parent = action;                  // the action, until the transaction's row is known
                r.events.push_back(e);
            }
        }
        r.level_notes[0] = "row progress: each L2 transfer's rows as " + level + " booked them on the banks' ports "
                           "(BM_ROW, STR_ROW, DRAIN_ROW: row r of the transfer, beat 0, [its clock, + its row's clocks))";
        r.level_notes[1] = "L2 by banks and rows (row-mod-banks): an event's L2 address is its transfer's first row, "
                           "which is the producer's own placement (§5.3); the accumulator's and the array edge's, none";
        if (too_tall)
            r.level_notes.push_back(std::to_string(too_tall) + " transfers have more L2 rows than their tile has rows; "
                                    "their row events are left out (a row event is a row of its tile, E3)");
    }
    // Row order is event order: by cycle, then as emitted (a transaction before its token, a
    // transaction before its rows).
    std::stable_sort(r.events.begin(), r.events.end(), [](const DFR::Event& a, const DFR::Event& b) {
        const bool ra = a.kind >= ev::BmRow, rb = b.kind >= ev::BmRow;
        return a.cycle != b.cycle ? a.cycle < b.cycle : (!ra && rb);
    });
    // A row event's parent: its action's transaction, by its final row.
    if (banked) {
        std::map<std::uint64_t, std::uint64_t> tx_of;
        for (std::size_t i = 0; i < r.events.size(); ++i)
            if (r.events[i].kind <= ev::Drain) tx_of.emplace(r.events[i].action, i);
        for (auto& e : r.events)
            if (e.kind >= ev::BmRow) {
                e.parent = tx_of.at(e.parent);
                e.lane = r.events[e.parent].lane;      // its transaction's process instance
            }
    }
    r.cycles = 0;
    for (const auto& e : r.events) r.cycles = std::max(r.cycles, e.end);
    return r;
}

}  // namespace sw::kpu::program::record
