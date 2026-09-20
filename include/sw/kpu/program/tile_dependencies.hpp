// ============================================================================
// include/sw/kpu/program/tile_dependencies.hpp
// The dependency model of an L0 TileProgram — the single recovery of "what must
// be ordered", shared by every consumer.
//
// Because every TileOp declares its COMPLETE tile I/O (§3a of
// docs/plans/kpu-program-model.md), the dependency DAG is recoverable from the
// program alone:
//   - tile hazards: RAW, WAR, WAW over (operand, ti, tj);
//   - feed availability: a consumer waits for the Feed that made its input tile
//     available;
//   - pivot-slot hazards: the data-dependent control LU has and GEMM does not —
//     LuDiagFactor records row swaps into a slot and PivotApply replays them.
//
// One recovery, many consumers: TileDag list-schedules it and draws it, and the
// transactional executor fires against it (docs/plans/tile-transaction-executor.md
// §4). Keeping it in one place is what stops an analysis and an execution from
// disagreeing about what is legal.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace sw::kpu::program {

// ----------------------------------------------------------------------------
// Why one op must wait for another. The kind matters to consumers: true
// dataflow (RAW, feed) is what the algorithm means, anti-dependencies (WAR,
// WAW) are artefacts of reusing storage, and the pivot kinds are control rather
// than data.
// ----------------------------------------------------------------------------
enum class TileDepKind {
    TileRaw,        // read-after-write on a tile: the real dataflow edge
    TileWar,        // write-after-read: the writer must wait for earlier readers
    TileWaw,        // write-after-write: writers to one tile stay ordered
    FeedAvailable,  // a consumer waits for the Feed that made its input available
    PivotRaw,       // PivotApply replays the swaps LuDiagFactor recorded in its slot
    PivotWar,       // a later slot writer must wait for earlier slot readers
    PivotWaw,       // slot writers stay ordered
};

inline const char* to_string(TileDepKind k) {
    switch (k) {
        case TileDepKind::TileRaw:       return "RAW";
        case TileDepKind::TileWar:       return "WAR";
        case TileDepKind::TileWaw:       return "WAW";
        case TileDepKind::FeedAvailable: return "FEED";
        case TileDepKind::PivotRaw:      return "PIVOT_RAW";
        case TileDepKind::PivotWar:      return "PIVOT_WAR";
        case TileDepKind::PivotWaw:      return "PIVOT_WAW";
    }
    return "?";
}

// True dataflow and control edges express the algorithm; anti-dependencies only
// protect storage reuse. Consumers that want to show or relax the latter need to
// tell them apart.
inline bool is_anti_dependency(TileDepKind k) {
    return k == TileDepKind::TileWar || k == TileDepKind::TileWaw ||
           k == TileDepKind::PivotWar || k == TileDepKind::PivotWaw;
}
inline bool is_pivot_dependency(TileDepKind k) {
    return k == TileDepKind::PivotRaw || k == TileDepKind::PivotWar ||
           k == TileDepKind::PivotWaw;
}

struct TileDepEdge {
    std::size_t from = 0;      // producer op index — must complete first
    std::size_t to = 0;        // consumer op index
    TileDepKind kind{};
    std::string subject;       // the tile ("A[1,0]") or slot ("pivot#0") responsible
};

// ----------------------------------------------------------------------------
// TileDependencies — preds/succs per op plus the typed edge list.
//
// preds/succs are DEDUPLICATED by op pair (one pair can be ordered for several
// reasons); `edges` keeps every reason, in deterministic order, for diagnosis
// and for drawing.
// ----------------------------------------------------------------------------
struct TileDependencies {
    std::vector<std::vector<std::size_t>> preds;
    std::vector<std::vector<std::size_t>> succs;
    std::vector<TileDepEdge> edges;

    // Which ops wrote each pivot slot, in program order.
    std::map<int, std::vector<std::size_t>> slot_writers;

    std::size_t op_count() const { return preds.size(); }

    // Slots written more than once. Reuse is CORRECT here — PivotWar/PivotWaw
    // edges order it — but it serializes panels that would otherwise be
    // independent, so a generator doing it unintentionally wants to know.
    std::vector<int> reused_pivot_slots() const {
        std::vector<int> out;
        for (const auto& [slot, writers] : slot_writers)
            if (writers.size() > 1) out.push_back(slot);
        return out;
    }
    bool pivot_slots_single_assignment() const { return reused_pivot_slots().empty(); }

    // ---- diagnosis ---------------------------------------------------------
    // The edges that still hold `op` back, given which ops have completed. This
    // is what turns a wedged run into a diagnosis instead of a hang (see
    // docs/plans/tile-transaction-executor.md §4, "Stall and refusal").
    std::vector<TileDepEdge> blocking_edges(std::size_t op,
                                            const std::vector<bool>& completed) const {
        std::vector<TileDepEdge> out;
        for (const TileDepEdge& e : edges)
            if (e.to == op && e.from < completed.size() && !completed[e.from])
                out.push_back(e);
        return out;
    }

    std::string explain_blocked(const TileProgram& prog, std::size_t op,
                                const std::vector<bool>& completed) const {
        const auto blockers = blocking_edges(op, completed);
        std::string out = "op " + std::to_string(op) + " (" +
                          to_string(prog.ops().at(op).kind) + ")";
        if (blockers.empty()) return out + " is not blocked by dependencies";
        out += " waits on:";
        for (const TileDepEdge& e : blockers)
            out += "\n  op " + std::to_string(e.from) + " (" +
                   to_string(prog.ops().at(e.from).kind) + ") via " +
                   to_string(e.kind) + " on " + e.subject;
        return out;
    }
};

// ----------------------------------------------------------------------------
// build_tile_dependencies — recover the DAG from declared tile I/O.
//
// Deterministic: ops are visited in program order and readers are kept in
// insertion order, so the edge list is reproducible run to run.
// ----------------------------------------------------------------------------
inline TileDependencies build_tile_dependencies(const TileProgram& prog) {
    const auto& ops = prog.ops();
    TileDependencies d;
    d.preds.resize(ops.size());
    d.succs.resize(ops.size());

    auto key = [](const TileCoord& c) {
        return c.operand + "#" + std::to_string(c.ti) + "#" + std::to_string(c.tj);
    };

    std::map<std::string, long> last_writer;                    // tile -> op
    std::map<std::string, std::vector<std::size_t>> readers;     // tile -> readers since last write
    std::map<std::string, long> last_feed;                       // tile -> Feed that made it available
    std::map<int, long> slot_writer;                             // pivot slot -> op
    std::map<int, std::vector<std::size_t>> slot_readers;        // pivot slot -> readers since last write

    auto add = [&](std::size_t consumer, long producer, TileDepKind kind,
                   const std::string& subject) {
        if (producer < 0 || static_cast<std::size_t>(producer) == consumer) return;
        const auto p = static_cast<std::size_t>(producer);
        auto& pv = d.preds[consumer];
        if (std::find(pv.begin(), pv.end(), p) == pv.end()) {
            pv.push_back(p);
            d.succs[p].push_back(consumer);
        }
        d.edges.push_back(TileDepEdge{p, consumer, kind, subject});   // every reason kept
    };

    for (std::size_t i = 0; i < ops.size(); ++i) {
        const TileOp& op = ops[i];

        std::vector<const TileCoord*> reads, writes;
        switch (op.kind) {
            case TileOpKind::Feed:
                // A Feed is a movement SOURCE: it makes an input tile available.
                // Consumers depend on it, but it is NOT a destructive write —
                // repeated feeds of a shared input tile must stay independent, or
                // independent output-tile computations would serialize through it.
                if (!op.inputs.empty()) last_feed[key(op.inputs[0])] = static_cast<long>(i);
                break;
            case TileOpKind::Drain:
                if (!op.outputs.empty()) reads.push_back(&op.outputs[0]);
                break;
            case TileOpKind::MatMulAccum:
                reads.push_back(&op.inputs[0]);
                reads.push_back(&op.inputs[1]);
                reads.push_back(&op.outputs[0]);
                writes.push_back(&op.outputs[0]);       // read-modify-write accumulate
                break;
            case TileOpKind::LuDiagFactor:
                reads.push_back(&op.outputs[0]);
                writes.push_back(&op.outputs[0]);
                break;
            case TileOpKind::PivotApply:
                reads.push_back(&op.outputs[0]);
                writes.push_back(&op.outputs[0]);
                break;
            case TileOpKind::TrsmLowerLeft:
            case TileOpKind::TrsmUpperRight:
                reads.push_back(&op.inputs[0]);
                reads.push_back(&op.outputs[0]);
                writes.push_back(&op.outputs[0]);
                break;
        }

        // ---- pivot-slot hazards, modelled exactly like tiles ----------------
        // PivotApply READS the slot; LuDiagFactor WRITES it (clearing whatever was
        // there). Without the anti-dependencies below, a later LuDiagFactor could
        // clear a slot that earlier PivotApply ops still have to read — safe only
        // by luck today, since the derivation happens to use a unique slot per
        // panel. Modelling them makes slot reuse correct rather than lucky.
        if (op.pivot_slot >= 0) {
            const std::string subject = "pivot#" + std::to_string(op.pivot_slot);
            if (op.kind == TileOpKind::PivotApply) {
                auto it = slot_writer.find(op.pivot_slot);
                if (it != slot_writer.end())
                    add(i, it->second, TileDepKind::PivotRaw, subject);
                slot_readers[op.pivot_slot].push_back(i);
            } else if (op.kind == TileOpKind::LuDiagFactor) {
                auto it = slot_writer.find(op.pivot_slot);
                if (it != slot_writer.end())
                    add(i, it->second, TileDepKind::PivotWaw, subject);      // writer after writer
                for (std::size_t rdr : slot_readers[op.pivot_slot])
                    add(i, static_cast<long>(rdr), TileDepKind::PivotWar, subject);  // writer after readers
                slot_writer[op.pivot_slot] = static_cast<long>(i);
                slot_readers[op.pivot_slot].clear();
                d.slot_writers[op.pivot_slot].push_back(i);
            }
        }

        // ---- tile hazards ---------------------------------------------------
        for (const TileCoord* t : reads) {
            const std::string k = key(*t);
            const std::string subject = t->to_string();
            auto it = last_writer.find(k);
            if (it != last_writer.end()) add(i, it->second, TileDepKind::TileRaw, subject);
            auto fit = last_feed.find(k);
            if (fit != last_feed.end()) add(i, fit->second, TileDepKind::FeedAvailable, subject);
            readers[k].push_back(i);
        }
        for (const TileCoord* t : writes) {
            const std::string k = key(*t);
            const std::string subject = t->to_string();
            auto it = last_writer.find(k);
            if (it != last_writer.end()) add(i, it->second, TileDepKind::TileWaw, subject);
            for (std::size_t rdr : readers[k])
                add(i, static_cast<long>(rdr), TileDepKind::TileWar, subject);
            last_writer[k] = static_cast<long>(i);
            readers[k].clear();
        }
    }

    return d;
}

} // namespace sw::kpu::program
