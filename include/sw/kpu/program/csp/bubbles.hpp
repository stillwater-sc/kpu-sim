// ============================================================================
// include/sw/kpu/program/csp/bubbles.hpp
// The reuse and bubble report of an L-T1 run (#425): where the compute fabric waited for data,
// why, and what reuse the program stated.
//
// Every pipelined tiling depends on reuse, and reuse must be deliberate: the program states it,
// as L2 stagings and stationary operands. Where it states too little, the movers cannot keep the
// array fed and the array has bubbles. This report reads them off L-T1's records (one leg per
// action, each with its ready cycle -- when its inputs and its credit were there -- and its start).
//
//   A BUBBLE is a compute lane's idle time before a call: the call's start minus the lane's
//   previous finish. Its CAUSE is found by walking back from the call along program order. At
//   each step, an action that started later than it was ready either waited for a lane of its
//   process -- every lane busy over (90% of) its wait: the streamer, the BlockMover or the DMA
//   was saturated -- or was held behind the action before it in its process's program order (a
//   head-of-line block: a process issues in order), and the walk follows that one.
//   An action that started when it was ready waited for its input:
//     - a call: the Feed that finished last among its operands, or the previous call on its
//       accumulator;
//     - a Feed: its tile's Move (before it); a Move: its L2 slot (a staging), or its tile's Load
//       (a Writeback: a result's, reported as such);
//     - a Load: its L3 slot, or the DRAM transfer itself;
//     - a result's way out, when a move waits behind it in its process's order: a Drain, its call
//       (or a merge's L2 slot); a Writeback, its Drain; a Store, its Writeback.
//   The walk is by tile and program order -- the language's lowering puts a call's Moves and
//   Feeds right before it -- so it needs no causal links in the interpreter.
//
//   REUSE per operand: its Loads (DRAM reads), Moves (L3 reads) and Feeds (into the fabric).
//   Feeds per Load is the reuse the program stated across DRAM; Feeds per Move, across L3 (an
//   L2 staging, or a stationary operand).
//
//   LOAD per process: busy lane-cycles against lanes x makespan.
//
// Every number is L-T1's: uncalibrated, as L-T1 always says.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/transactional.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

struct BubbleReport {
    using Cycle = TransactionalInterpreter::Cycle;
    using Record = TransactionalInterpreter::Record;

    // Why a compute lane sat idle before a call.
    enum class Cause : std::uint8_t {
        Issue,          // the call was ready; the fabric's in-order issue held it
        Accumulator,    // it waited for the previous call on its accumulator
        StreamerBusy,   // its operand's Feed waited for a streamer lane
        L2Slot,         // the Feed's Move (a staging) waited for its L2 slot's credit
        MoverBusy,      // the Move waited for a BlockMover lane
        L3Slot,         // the Move's Load waited for its L3 slot's credit
        DmaBusy,        // the Load waited for a DMA lane
        Dram,           // the Load itself: the DRAM transfer (and what it waited on, a Store or a Release)
        Result,         // the Move's tile is a result: its Writeback, and what it waited for
        Unknown,
    };
    static const char* to_string(Cause c) {
        switch (c) {
            case Cause::Issue:        return "in-order issue";
            case Cause::Accumulator:  return "the previous call on its accumulator";
            case Cause::StreamerBusy: return "streamer lanes busy";
            case Cause::L2Slot:       return "L2 slot credit";
            case Cause::MoverBusy:    return "BlockMover lanes busy";
            case Cause::L3Slot:       return "L3 slot credit";
            case Cause::DmaBusy:      return "DMA lanes busy";
            case Cause::Dram:         return "DRAM transfer";
            case Cause::Result:       return "a result's writeback";
            case Cause::Unknown:      return "unknown";
        }
        return "?";
    }

    struct Bubble {
        std::size_t call = 0;           // the call's action index
        TileCoord tile;                 // its result tile
        std::size_t lane = 0;           // the compute lane
        Cycle idle = 0;                 // the lane's idle cycles before it
        Cause cause = Cause::Unknown;
        TileCoord operand;              // the operand it waited for (when one did)
        std::vector<std::size_t> chain; // the actions walked back, the call first
    };

    struct Reuse {
        std::string operand;
        std::size_t loads = 0, moves = 0, feeds = 0;
        double feeds_per_load() const { return loads ? double(feeds) / double(loads) : 0.0; }
        double feeds_per_move() const { return moves ? double(feeds) / double(moves) : 0.0; }
    };

    struct Load {
        std::string process;
        std::size_t lanes = 0;
        Cycle busy = 0;
        double utilization = 0.0;       // busy / (lanes x makespan)
    };

    Cycle makespan = 0;
    std::size_t calls = 0;
    Cycle cf_busy = 0;                  // the calls' cycles, over every compute lane
    Cycle cf_idle = 0;                  // the bubbles' cycles, over every compute lane
    std::vector<Bubble> bubbles;        // every call with idle > 0, in program order
    std::map<Cause, Cycle> idle_by_cause;
    std::vector<Reuse> reuse;           // by operand, in name order
    std::vector<Load> loads;            // by process

    // The share of the compute lanes' span (first call's start to last call's finish) spent idle.
    double bubble_fraction() const { return cf_busy + cf_idle ? double(cf_idle) / double(cf_busy + cf_idle) : 0.0; }

    // The report, as text: a summary, the causes, the reuse, the load, and the `show` largest bubbles.
    std::string text(std::size_t show = 8) const {
        std::ostringstream o;
        o << std::fixed << std::setprecision(1);
        o << "bubbles  " << calls << " calls; compute busy " << cf_busy << ", idle " << cf_idle << " ("
          << 100.0 * bubble_fraction() << "% of the compute lanes' span), makespan " << makespan
          << " (L-T1, uncalibrated)\n";
        for (const auto& [c, n] : idle_by_cause)
            o << "  idle " << std::setw(8) << n << "  " << to_string(c) << "\n";
        o << "reuse    (feeds per load: across DRAM; feeds per move: across L3)\n";
        for (const Reuse& r : reuse)
            o << "  " << std::setw(6) << r.operand << "  loads " << r.loads << ", moves " << r.moves << ", feeds "
              << r.feeds << "  feeds/load " << std::setprecision(2) << r.feeds_per_load() << ", feeds/move "
              << r.feeds_per_move() << std::setprecision(1) << "\n";
        o << "load     (busy lane-cycles over lanes x makespan)\n";
        for (const Load& l : loads)
            o << "  " << std::setw(6) << l.process << "  " << l.lanes << " lane(s), busy " << l.busy << ", "
              << 100.0 * l.utilization << "%\n";
        std::vector<const Bubble*> top;
        for (const Bubble& b : bubbles) top.push_back(&b);
        std::stable_sort(top.begin(), top.end(), [](const Bubble* a, const Bubble* b) { return a->idle > b->idle; });
        if (!top.empty()) o << "largest  (action: call -> what it waited for)\n";
        for (std::size_t i = 0; i < top.size() && i < show; ++i) {
            const Bubble& b = *top[i];
            o << "  " << std::setw(6) << b.idle << " idle before action " << b.call << " (" << b.tile.to_string()
              << ", lane " << b.lane << "): " << to_string(b.cause);
            if (!b.operand.operand.empty()) o << ", operand " << b.operand.to_string();
            o << "  [";
            for (std::size_t k = 0; k < b.chain.size(); ++k) o << (k ? " <- " : "") << b.chain[k];
            o << "]\n";
        }
        return o.str();
    }
};

namespace detail {

// The record of an action, by action index (a Store has two: its first).
inline const TransactionalInterpreter::Record* record_of(
    const std::map<std::size_t, const TransactionalInterpreter::Record*>& by_action, std::size_t a) {
    auto it = by_action.find(a);
    return it == by_action.end() ? nullptr : it->second;
}

}  // namespace detail

// The report of an L-T1 run: its records, makespan, and per-process busy cycles and lanes
// (TransactionalInterpreter::Stats, or driver::CspLevelOutcome).
inline BubbleReport bubble_report(const std::vector<TransactionalInterpreter::Record>& records,
                                  TransactionalInterpreter::Cycle makespan,
                                  const std::map<std::string, TransactionalInterpreter::Cycle>& busy,
                                  const std::map<std::string, std::size_t>& lanes) {
    using Rec = TransactionalInterpreter::Record;
    using Proc = TransactionalInterpreter::Proc;
    using Cause = BubbleReport::Cause;
    BubbleReport rep;
    rep.makespan = makespan;

    // Records in program order, the first leg of each action.
    std::vector<const Rec*> order;
    std::map<std::size_t, const Rec*> by_action;
    for (const Rec& r : records)
        if (by_action.emplace(r.action, &r).second) order.push_back(&r);
    std::sort(order.begin(), order.end(), [](const Rec* a, const Rec* b) { return a->action < b->action; });

    // The last action of a kind on a tile before action `before`, in program order.
    std::map<std::string, std::vector<const Rec*>> by_tile;
    for (const Rec* r : order) by_tile[r->tile.to_string()].push_back(r);
    auto last_before = [&](const TileCoord& t, std::size_t before, std::initializer_list<Action::Kind> kinds) -> const Rec* {
        auto it = by_tile.find(t.to_string());
        if (it == by_tile.end()) return nullptr;
        const Rec* found = nullptr;
        for (const Rec* r : it->second) {
            if (r->action >= before) break;
            if (std::find(kinds.begin(), kinds.end(), r->kind) != kinds.end()) found = r;
        }
        return found;
    };

    // Reuse, by operand.
    std::map<std::string, BubbleReport::Reuse> reuse;
    for (const Rec* r : order) {
        auto& u = reuse[r->tile.operand];
        u.operand = r->tile.operand;
        if (r->kind == Action::Kind::Load) ++u.loads;
        if (r->kind == Action::Kind::Move) ++u.moves;
        if (r->kind == Action::Kind::Feed) ++u.feeds;
    }
    for (auto& [k, u] : reuse)
        if (u.loads || u.moves || u.feeds) rep.reuse.push_back(u);

    // Process load.
    for (const auto& [name, n] : lanes) {
        BubbleReport::Load l;
        l.process = name;
        l.lanes = n;
        auto b = busy.find(name);
        l.busy = b == busy.end() ? 0 : b->second;
        l.utilization = makespan && n ? double(l.busy) / (double(n) * double(makespan)) : 0.0;
        rep.loads.push_back(l);
    }

    // The previous record of the same process, in program order (its in-order issue).
    std::map<const Rec*, const Rec*> prev_on_proc;
    {
        std::map<Proc, const Rec*> last;
        for (const Rec* r : order) {
            if (!TransactionalInterpreter::has_process(r->kind)) continue;
            auto it = last.find(r->proc);
            if (it != last.end()) prev_on_proc[r] = it->second;
            last[r->proc] = r;
        }
    }
    // The Feeds that delivered each call's operands: the Feeds between the previous call and this
    // one, in program order.
    std::map<const Rec*, std::vector<const Rec*>> call_feeds;
    {
        std::vector<const Rec*> feeds;
        for (const Rec* r : order) {
            if (r->kind == Action::Kind::Feed) feeds.push_back(r);
            else if (r->kind == Action::Kind::Call) {
                call_feeds[r] = feeds;
                feeds.clear();
            }
        }
    }
    auto busy_of = [](Proc p) {
        switch (p) {
            case Proc::Str: return Cause::StreamerBusy;
            case Proc::Bm:  return Cause::MoverBusy;
            case Proc::Dma: return Cause::DmaBusy;
            case Proc::Cf:  return Cause::Issue;
        }
        return Cause::Unknown;
    };
    // Was process `p` saturated over [from, to): every lane busy for at least 90% of it? Each
    // process's legs, swept for the cycles its lane count is reached.
    std::map<Proc, std::vector<std::pair<BubbleReport::Cycle, int>>> sweep;
    for (const Rec& r : records)
        if (TransactionalInterpreter::has_process(r.kind) && r.finish > r.start) {
            sweep[r.proc].push_back({r.start, +1});
            sweep[r.proc].push_back({r.finish, -1});
        }
    for (auto& [p, ev] : sweep) std::sort(ev.begin(), ev.end());
    auto saturated = [&](Proc p, BubbleReport::Cycle from, BubbleReport::Cycle to) {
        auto ln = lanes.find(TransactionalInterpreter::name(p));
        const int n = static_cast<int>(ln == lanes.end() ? 1 : std::max<std::size_t>(1, ln->second));
        auto it = sweep.find(p);
        if (it == sweep.end() || to <= from) return false;
        int live = 0;
        BubbleReport::Cycle full = 0, at = from;
        for (const auto& [t, d] : it->second) {
            const BubbleReport::Cycle clipped = std::min(std::max(t, from), to);
            if (live >= n && clipped > at) full += clipped - at;
            at = std::max(at, clipped);
            live += d;
            if (t >= to) break;
        }
        if (live >= n && to > at) full += to - at;
        return 10 * full >= 9 * (to - from);
    };
    // Why `r` started when it did. Later than it was ready: its process was saturated over the
    // wait (its lanes were busy), or it was held behind the action before it in program order,
    // which the walk follows (in-order issue: a head-of-line block). When it was ready: what it
    // was waiting for, by kind.
    BubbleReport::Bubble* into = nullptr;
    std::function<Cause(const Rec*, int)> explain = [&](const Rec* r, int depth) -> Cause {
        if (depth > 256) return Cause::Unknown;
        into->chain.push_back(r->action);
        if (r->start > r->ready) {
            if (saturated(r->proc, r->ready, r->start)) return busy_of(r->proc);
            auto p = prev_on_proc.find(r);
            if (p != prev_on_proc.end()) return explain(p->second, depth + 1);
            return busy_of(r->proc);
        }
        switch (r->kind) {
            case Action::Kind::Call: {
                const Rec* bind = nullptr;
                for (const Rec* f : call_feeds[r])
                    if (!bind || f->finish > bind->finish) bind = f;
                const Rec* prev = last_before(r->tile, r->action, {Action::Kind::Call});
                if (prev && (!bind || prev->finish >= bind->finish)) {
                    into->chain.push_back(prev->action);
                    return Cause::Accumulator;
                }
                if (!bind) return Cause::Unknown;
                if (into->operand.operand.empty()) into->operand = bind->tile;
                return explain(bind, depth + 1);
            }
            case Action::Kind::Feed: {
                const Rec* mv = last_before(r->tile, r->action, {Action::Kind::Move});
                return mv ? explain(mv, depth + 1) : Cause::Unknown;
            }
            case Action::Kind::Move: {
                if (r->l2_wait > 0) return Cause::L2Slot;
                const Rec* src =
                    last_before(r->tile, r->action, {Action::Kind::Load, Action::Kind::Writeback, Action::Kind::Inherit});
                if (!src) return Cause::Unknown;
                if (src->kind != Action::Kind::Load) {
                    into->chain.push_back(src->action);
                    return Cause::Result;
                }
                return explain(src, depth + 1);
            }
            case Action::Kind::Load:
                return r->slot_wait > 0 ? Cause::L3Slot : Cause::Dram;
            // A result's way out, met when an operand's move waits behind it in its process's order:
            // a Drain waits for its call (or a merge's L2 slot), a Writeback for its Drain, a Store
            // for its Writeback.
            case Action::Kind::Drain: {
                if (r->l2_wait > 0) return Cause::L2Slot;
                const Rec* src = last_before(r->tile, r->action, {Action::Kind::Call});
                return src ? explain(src, depth + 1) : Cause::Unknown;
            }
            case Action::Kind::Writeback: {
                const Rec* src = last_before(r->tile, r->action, {Action::Kind::Drain});
                return src ? explain(src, depth + 1) : Cause::Unknown;
            }
            case Action::Kind::Store: {
                const Rec* src = last_before(r->tile, r->action, {Action::Kind::Writeback, Action::Kind::Load});
                return src ? explain(src, depth + 1) : Cause::Unknown;
            }
            default:
                return Cause::Unknown;
        }
    };

    // The calls, by compute lane: the idle time before each, and why.
    std::map<std::size_t, BubbleReport::Cycle> lane_free;
    std::map<std::size_t, bool> lane_used;
    for (const Rec* r : order) {
        if (r->kind != Action::Kind::Call || r->proc != Proc::Cf) continue;
        ++rep.calls;
        rep.cf_busy += r->finish - r->start;
        const std::size_t lane = r->lane;
        const BubbleReport::Cycle idle = lane_used[lane] && r->start > lane_free[lane] ? r->start - lane_free[lane] : 0;
        lane_used[lane] = true;
        lane_free[lane] = std::max(lane_free[lane], r->finish);
        if (idle == 0) continue;
        BubbleReport::Bubble b;
        b.call = r->action;
        b.tile = r->tile;
        b.lane = lane;
        b.idle = idle;
        into = &b;
        b.cause = explain(r, 0);
        into = nullptr;
        rep.cf_idle += idle;
        rep.idle_by_cause[b.cause] += idle;
        rep.bubbles.push_back(std::move(b));
    }
    return rep;
}

}  // namespace sw::kpu::program::csp
