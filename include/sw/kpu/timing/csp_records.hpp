// ============================================================================
// include/sw/kpu/timing/csp_records.hpp
// An L-CA run as L-T1's records (#425): one leg per action, with its process, lane, start,
// finish and ready cycle, so the reuse and bubble report (csp/bubbles.hpp) reads either level
// with the same walk.
//
// L-CA keeps no record per action; it logs protocol events (CspDriver::keep_events). They are
// matched back to the program's actions:
//   - each action as the driver handed it over (CspDriver::Result::handovers), in program order;
//   - a transfer's START, to the next action of its kind on its tile that has not started (a
//     process takes one tile's actions of one kind in program order); its COMPLETE, placed at
//     the cycle it ends, to the oldest started one;
//   - its READY cycle, as L-T1 means it: when its inputs had finished (as the bubble walk pairs
//     them) and its credit was free (the cycle after its process's last credit stall on its tile).
//     A process sees a completion on its next tick, so a start one cycle after that is no wait.
//     The first credit stall (an L3 slot for a Load, an L2 slot for a Move or a Drain) gives its
//     slot wait.
// A Store is two legs, as at L-T1: the BlockMover's ejection, then the DMA engine's write.
//
// The lanes are the machine's: its DMA engines (a Load's lane is the engine holding its
// request, whose credit event names it), BlockMovers, streamers and compute tiles.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/timing/csp_driver.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::timing {

struct CspRecords {
    std::vector<program::csp::TransactionalInterpreter::Record> records;
    std::map<std::string, std::uint64_t> busy;   // lane-cycles per process
    std::map<std::string, std::size_t> lanes;    // the machine's lanes per process
    std::size_t unmatched = 0;                   // transfer events no action took: 0 in a sound run
};

namespace detail {

using program::csp::Action;
using Proc = program::csp::TransactionalInterpreter::Proc;

// A transfer's leg: the action kind it belongs to (a Store's two legs apart), its process, and
// whether the event begins (+1) or ends (-1) it.
struct LegOf {
    Action::Kind kind;
    int leg;                                     // a Store: 0 the ejection, 1 the DRAM write
    Proc proc;
    int edge;
};
inline std::optional<LegOf> leg_of(EventType t) {
    switch (t) {
        case EventType::DMA_LOAD_START:        return LegOf{Action::Kind::Load, 0, Proc::Dma, +1};
        case EventType::DMA_LOAD_COMPLETE:     return LegOf{Action::Kind::Load, 0, Proc::Dma, -1};
        case EventType::BM_MOVE_START:         return LegOf{Action::Kind::Move, 0, Proc::Bm, +1};
        case EventType::BM_MOVE_COMPLETE:      return LegOf{Action::Kind::Move, 0, Proc::Bm, -1};
        case EventType::BM_WRITEBACK_START:    return LegOf{Action::Kind::Writeback, 0, Proc::Bm, +1};
        case EventType::BM_WRITEBACK_COMPLETE: return LegOf{Action::Kind::Writeback, 0, Proc::Bm, -1};
        case EventType::BM_EJECT_START:        return LegOf{Action::Kind::Store, 0, Proc::Bm, +1};
        case EventType::BM_EJECT_COMPLETE:     return LegOf{Action::Kind::Store, 0, Proc::Bm, -1};
        case EventType::DMA_STORE_START:       return LegOf{Action::Kind::Store, 1, Proc::Dma, +1};
        case EventType::DMA_STORE_COMPLETE:    return LegOf{Action::Kind::Store, 1, Proc::Dma, -1};
        case EventType::STR_FEED_START:        return LegOf{Action::Kind::Feed, 0, Proc::Str, +1};
        case EventType::STR_FEED_COMPLETE:     return LegOf{Action::Kind::Feed, 0, Proc::Str, -1};
        case EventType::STR_DRAIN_START:       return LegOf{Action::Kind::Drain, 0, Proc::Str, +1};
        case EventType::STR_DRAIN_COMPLETE:    return LegOf{Action::Kind::Drain, 0, Proc::Str, -1};
        case EventType::COMPUTE_START:         return LegOf{Action::Kind::Call, 0, Proc::Cf, +1};
        case EventType::COMPUTE_COMPLETE:      return LegOf{Action::Kind::Call, 0, Proc::Cf, -1};
        default:                               return std::nullopt;
    }
}
// A stall's process, and whether it waits for a credit (a slot) rather than a tile.
inline std::optional<std::pair<Proc, bool>> stall_of(EventType t) {
    switch (t) {
        case EventType::DMA_STALL_CREDIT:  return std::pair{Proc::Dma, true};
        case EventType::DMA_STALL_TAG:     return std::pair{Proc::Dma, false};
        case EventType::BM_STALL_CREDIT:   return std::pair{Proc::Bm, true};
        case EventType::BM_STALL_TAG:      return std::pair{Proc::Bm, false};
        case EventType::STR_STALL_CREDIT:  return std::pair{Proc::Str, true};
        case EventType::STR_STALL_TAG:
        case EventType::STR_STALL_COMPUTE: return std::pair{Proc::Str, false};
        default:                           return std::nullopt;
    }
}
inline bool has_process(Action::Kind k) {
    return program::csp::TransactionalInterpreter::has_process(k);
}
inline Proc proc_of(Action::Kind k) {
    switch (k) {
        case Action::Kind::Load:      return Proc::Dma;
        case Action::Kind::Move:
        case Action::Kind::Writeback:
        case Action::Kind::Store:     return Proc::Bm;
        case Action::Kind::Feed:
        case Action::Kind::Drain:     return Proc::Str;
        default:                      return Proc::Cf;
    }
}

}  // namespace detail

// The records of an L-CA run, from its handovers and protocol events (both kept by
// CspDriver::keep_events), on a machine of `dma` engines, `bm` BlockMovers and `cf` compute tiles.
// The streamers' lanes are the streamers the events name.
inline CspRecords csp_records(const std::vector<CspDriver::Result::Handover>& handovers,
                              const std::vector<CspDriver::ProtocolEvent>& events, std::size_t dma, std::size_t bm,
                              std::size_t cf) {
    using namespace detail;
    using Rec = program::csp::TransactionalInterpreter::Record;
    using program::csp::TransactionalInterpreter;
    CspRecords out;

    // One record per leg, and the legs of each (kind, leg, tile) in program order.
    struct Leg {
        Rec rec;
        int leg = 0;
        std::uint64_t at = 0;                       // its handover
        bool started = false, finished = false;
    };
    std::vector<Leg> legs;
    legs.reserve(handovers.size() + handovers.size() / 4);
    for (const auto& h : handovers) {
        for (int leg = 0; leg < (h.kind == Action::Kind::Store ? 2 : 1); ++leg) {
            Leg l;
            l.rec.action = h.action;
            l.rec.kind = h.kind;
            l.rec.tile = h.tile;
            l.rec.proc = leg == 1 ? Proc::Dma : proc_of(h.kind);
            l.rec.start = l.rec.finish = l.rec.ready = h.at;
            l.leg = leg;
            l.at = h.at;
            legs.push_back(l);
        }
    }
    std::map<std::string, std::vector<std::size_t>> queue;    // "kind/leg/tile" -> legs, program order
    auto key = [](Action::Kind k, int leg, const program::TileCoord& t) {
        return std::to_string(static_cast<int>(k)) + "/" + std::to_string(leg) + "/" + t.to_string();
    };
    for (std::size_t i = 0; i < legs.size(); ++i)
        if (has_process(legs[i].rec.kind)) queue[key(legs[i].rec.kind, legs[i].leg, legs[i].rec.tile)].push_back(i);

    // The DMA engine that holds a tile's request: the last engine whose event named the tile.
    std::map<std::string, std::uint32_t> engine;
    std::map<std::string, std::size_t> str_lane;            // a streamer's name -> its lane
    std::map<std::uint32_t, std::size_t> bm_lane;           // a BlockMover's id -> its lane, from 0
    // Stalls, by process and tile: (cycle, waits for a credit).
    std::map<std::string, std::vector<std::pair<std::uint64_t, bool>>> stalls;
    auto pkey = [](Proc p, const program::TileCoord& t) {
        return std::string(TransactionalInterpreter::name(p)) + "/" + t.to_string();
    };
    for (const auto& e : events) {
        if (!e.tile) continue;
        const std::string tk = e.tile->to_string();
        if (e.component.rfind("DMA", 0) == 0) engine[tk] = e.component_id;
        if (const auto st = stall_of(e.type)) {
            stalls[pkey(st->first, *e.tile)].push_back({e.cycle, st->second});
            continue;
        }
        const auto lg = leg_of(e.type);
        if (!lg) continue;
        auto q = queue.find(key(lg->kind, lg->leg, *e.tile));
        if (q == queue.end()) {
            ++out.unmatched;
            continue;
        }
        bool took = false;
        for (std::size_t i : q->second) {
            Leg& l = legs[i];
            if (lg->edge > 0 ? l.started : (!l.started || l.finished)) continue;
            if (lg->edge > 0) {
                l.started = true;
                l.rec.start = e.cycle;
                // A DMA leg's lane is its engine (the memory controller serves the transfer).
                if (lg->proc == Proc::Dma) {
                    auto en = engine.find(tk);
                    l.rec.lane = en != engine.end() ? en->second : 0;
                } else if (lg->proc == Proc::Str) {
                    l.rec.lane = str_lane.emplace(e.component, str_lane.size()).first->second;
                } else if (lg->proc == Proc::Bm) {
                    l.rec.lane = bm_lane.emplace(e.component_id, bm_lane.size()).first->second;
                } else {
                    l.rec.lane = e.component_id;
                }
            } else {
                l.finished = true;
                l.rec.finish = std::max(e.cycle, l.rec.start);
            }
            took = true;
            break;
        }
        // A second START of a transfer already under way (the engine and the memory controller
        // both announce a DRAM transfer), or its second COMPLETE: the same leg, already taken.
        (void)took;
    }

    // READY, as L-T1 means it: when the leg's inputs were there and its credit was free.
    //   - its inputs' finishes, in program order on its tile, as the bubble walk pairs them: a
    //     Feed's Move; a Move's Load, Writeback or Inherit; a call's Feeds since the previous call,
    //     and the previous call on its accumulator; a Drain's call; a Writeback's Drain; a Store's
    //     Writeback or Load, and its DRAM leg its ejection;
    //   - its credit: the cycle after its process's last credit stall on its tile, since the
    //     handover. The first credit stall says how long the slot held it (a Load's L3 slot, a
    //     Move's or a Drain's L2 slot).
    // A leg that started later than that waited for a lane or behind its process's order.
    for (Leg& l : legs) {
        if (!has_process(l.rec.kind)) continue;
        if (!l.started) l.rec.start = l.rec.finish = l.at;
        else if (!l.finished) l.rec.finish = l.rec.start;
    }
    std::vector<std::size_t> order(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t x, std::size_t y) { return legs[x].rec.action < legs[y].rec.action; });
    std::map<std::string, std::map<Action::Kind, std::uint64_t>> last_finish;   // tile -> kind -> finish
    std::vector<std::uint64_t> feeds_since_call;                                // since the last call
    for (std::size_t i : order) {
        Leg& l = legs[i];
        Rec& r = l.rec;
        const std::string tk = r.tile.to_string();
        auto& lf = last_finish[tk];
        auto fin = [&](std::initializer_list<Action::Kind> kinds) {
            std::uint64_t f = 0;
            for (Action::Kind k : kinds)
                if (auto it = lf.find(k); it != lf.end()) f = std::max(f, it->second);
            return f;
        };
        if (has_process(r.kind)) {
            std::uint64_t in = 0;
            switch (r.kind) {
                case Action::Kind::Feed:      in = fin({Action::Kind::Move}); break;
                case Action::Kind::Move:      in = fin({Action::Kind::Load, Action::Kind::Writeback, Action::Kind::Inherit}); break;
                case Action::Kind::Call:
                    in = fin({Action::Kind::Call});
                    for (std::uint64_t f : feeds_since_call) in = std::max(in, f);
                    break;
                case Action::Kind::Drain:     in = fin({Action::Kind::Call}); break;
                case Action::Kind::Writeback: in = fin({Action::Kind::Drain}); break;
                case Action::Kind::Store:
                    in = l.leg == 1 ? r.start : fin({Action::Kind::Writeback, Action::Kind::Load});
                    break;
                default: break;
            }
            if (r.kind == Action::Kind::Store && l.leg == 1)    // the DMA leg: after its ejection
                for (std::size_t j = 0; j < legs.size(); ++j)
                    if (legs[j].rec.action == r.action && legs[j].leg == 0) in = legs[j].rec.finish;
            std::optional<std::uint64_t> last_credit, first_credit;
            if (auto st = stalls.find(pkey(r.proc, r.tile)); st != stalls.end())
                for (const auto& [c, credit] : st->second) {
                    if (!credit || c < l.at || c >= r.start) continue;
                    last_credit = c;
                    if (!first_credit) first_credit = c;
                }
            std::uint64_t ready = std::max<std::uint64_t>(l.at, in);
            if (last_credit) ready = std::max<std::uint64_t>(ready, *last_credit + 1);
            // A process sees a completion on its next tick: starting one cycle after its inputs
            // is starting when they were there, not a wait.
            if (r.start <= ready + 1) ready = r.start;
            r.ready = std::min(r.start, ready);
            if (first_credit) {
                const std::uint64_t wait = r.start - *first_credit;
                if (r.kind == Action::Kind::Load) r.slot_wait = wait;
                else if (r.kind == Action::Kind::Move || r.kind == Action::Kind::Drain) r.l2_wait = wait;
            }
        }
        // What later legs read: each kind's last finish on the tile (a Store's: its ejection), the
        // Feeds since the last call.
        if (!(r.kind == Action::Kind::Store && l.leg == 1)) lf[r.kind] = std::max(lf[r.kind], r.finish);
        if (r.kind == Action::Kind::Feed) feeds_since_call.push_back(r.finish);
        if (r.kind == Action::Kind::Call) feeds_since_call.clear();
    }

    for (const Leg& l : legs) {
        if (has_process(l.rec.kind)) out.busy[TransactionalInterpreter::name(l.rec.proc)] += l.rec.finish - l.rec.start;
        out.records.push_back(l.rec);
    }
    out.lanes["dma"] = std::max<std::size_t>(dma, 1);
    out.lanes["bm"] = std::max<std::size_t>(bm, 1);
    out.lanes["str"] = std::max<std::size_t>(str_lane.size(), 1);
    // Compute tiles: the machine's, or (unbounded, 0) those the calls ran on.
    std::size_t cf_seen = 0;
    for (const Leg& l : legs)
        if (l.rec.kind == Action::Kind::Call) cf_seen = std::max(cf_seen, l.rec.lane + 1);
    out.lanes["cf"] = std::max<std::size_t>(cf ? cf : cf_seen, 1);
    return out;
}

}  // namespace sw::kpu::timing
