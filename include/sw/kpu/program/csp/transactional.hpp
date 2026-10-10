// ============================================================================
// include/sw/kpu/program/csp/transactional.hpp
// L-T1 from the CSP program (docs/plans/kpu-run-csp-programs.md step 3; the sequencing plan's
// step 3).
//
// L-T1 decomposes a CSP transaction into one tile move under L3 credits -- and the credits
// are THE PROGRAM'S: a Load takes a slot, the program's Release returns it, a Writeback into
// an open residency writes in place. Nothing is inferred. (The L0-driven TileTransaction-
// Executor infers residency -- a credit at first touch, returned after the last reader --
// which describes an execution order, not a program.)
//
// THE SCHEDULE IS ONE PASS IN PROGRAM ORDER. Every dependency of a CSP action points backwards
// in program order -- its tile was put where it reads it by an earlier action, its credit was
// returned by an earlier Release -- so each action's start is known when it is reached:
//
//   start = max( the data it reads is ready,            (scoreboard per tile and channel)
//                a lane of its process is free,          (lanes x bytes/cycle, per process)
//                the process's previous action started,  (each process issues in order)
//                an L3 credit is free )                  (Load, and a Writeback that opens)
//
//   Load       dma   DRAM -> L3       after the tile's earlier Store retires (RAW via DRAM)
//                                     and a credit is free
//   Move       bm    L3 -> L2         after the L3 copy is current (its Load / Writeback)
//   Feed       str   L2 -> L1         after its Move
//   Call       cf    the tile function's MACs on a compute tile, after its operands' Feeds and
//                                     the previous call on its accumulator
//   Drain      str   L1 -> L2         after the call that produced it
//   Writeback  bm    L2 -> L3         after its Drain; in place, after the residency's readers
//   Store      bm    L3 -> DMA buffer, then dma buffer -> DRAM (the push-only store's two legs)
//   Release    --    zero time, after every reader of the residency issued before it; its
//                                     credit returns
//
// The credit pool is the PROGRAM's L3 (machine flat(l3 = N)): the program's declared machine.
// Values are the behavioral interpreter's, stepped in the same order, so L-T1 is bit-identical
// to L-B by construction. Timing is per process, per lane, from the device descriptor, and
// UNCALIBRATED -- as L-T1 always says.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/tile_work.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

class TransactionalInterpreter {
public:
    using Cycle = std::uint64_t;

    // One process of the program, as L-T1 times it.
    enum class Proc : std::uint8_t { Dma, Bm, Str, Cf };
    static const char* name(Proc p) {
        switch (p) {
            case Proc::Dma: return "dma";
            case Proc::Bm:  return "bm";
            case Proc::Str: return "str";
            case Proc::Cf:  return "cf";
        }
        return "?";
    }

    struct Record {                     // one leg of one action (Release: zero length; Store: two)
        std::size_t action = 0;         // the action's index in program order
        Action::Kind kind = Action::Kind::Call;
        TileCoord tile;
        Proc proc = Proc::Cf;
        std::size_t lane = 0;
        Cycle start = 0, finish = 0;
    };

    // One residency's L3 slot: held from its credit to its Release.
    struct Slot {
        TileCoord tile;
        Cycle t0 = 0, t1 = 0;
    };

    struct Stats {
        Cycle makespan = 0;
        std::size_t actions = 0;
        std::uint64_t dram_loads = 0, dram_stores = 0, dram_bytes = 0;
        std::size_t peak_l3 = 0;                    // slots held at once, in time
        std::size_t credit_stalls = 0;              // Loads that waited for a credit
        std::map<std::string, Cycle> busy;          // per process: lane-cycles occupied
        std::map<std::string, std::size_t> lanes;   // per process
    };

    // `l3` is the program's L3 capacity (its declared machine); `dev` gives the lanes and rates.
    TransactionalInterpreter(const characterize::DeviceDescriptor& dev, std::size_t l3) : dev_(dev), l3_(l3) {}

    void begin(const TileProgram& operands) {
        values_.begin(operands);
        shapes_ = operands;
        st_ = Stats{};
        records_.clear();
        lane_free_.clear();
        last_start_.clear();
        for (Proc p : {Proc::Dma, Proc::Bm, Proc::Str, Proc::Cf}) {
            const std::size_t n = lanes(p);
            lane_free_[p].assign(n, 0);
            last_start_[p] = 0;
            st_.lanes[name(p)] = n;
            st_.busy[name(p)] = 0;
        }
        credits_ = {};
        for (std::size_t i = 0; i < l3_; ++i) credits_.push(0);
        residency_.clear();
        dram_ready_.clear();
        moved_.clear();
        drained_.clear();
        fed_.clear();
        acc_.clear();
        intervals_.clear();
        slots_.clear();
        stage_reads_.clear();
        released_.clear();
    }

    void step(const Action& a, const TileOp* call_op) {
        values_.step(a, call_op);                   // the values: L-B's, in the same order
        const std::string k = a.tile.to_string();
        const Cycle bytes = tile_bytes(a.tile);
        switch (a.kind) {
            case Action::Kind::Load: {
                // After the tile's earlier Store retires (RAW via DRAM), and after its previous
                // residency's Release: a tile holds one L3 slot at a time (L-CA's DMA waits out
                // a copy still held the same way).
                Cycle ready = std::max(get(dram_ready_, k), get(released_, k));
                const Cycle credit = take_credit();
                if (credit > ready) {
                    ready = credit;
                    ++st_.credit_stalls;                    // the data was ready; the slot was not
                }
                const Cycle f = run(Proc::Dma, ready, transfer(Proc::Dma, bytes), a);
                Residency& r = residency_[k];
                r = Residency{};
                r.current = f;
                r.readers = f;
                r.slot = true;
                r.slot_since = ready;
                ++st_.dram_loads;
                st_.dram_bytes += bytes;
                break;
            }
            case Action::Kind::Move: {
                Residency& r = residency_[k];
                const Cycle f = run(Proc::Bm, r.current, transfer(Proc::Bm, bytes), a);
                r.readers = std::max(r.readers, f);
                moved_[k] = f;
                break;
            }
            case Action::Kind::Feed:
                fed_[k] = run(Proc::Str, get(moved_, k), transfer(Proc::Str, bytes), a);
                break;
            case Action::Kind::Call: {
                Cycle ready = get(acc_, k);                         // the chain's previous call
                for (const TileCoord& t : call_op->inputs) ready = std::max(ready, get(fed_, t.to_string()));
                for (const TileCoord& t : call_op->outputs)        // an in-place result is fed too
                    ready = std::max(ready, get(fed_, t.to_string()));
                const TileWork w = tile_work_of(shapes_, *call_op, dev_.element_bytes);
                const Cycle d = quantize_cycles(w.macs / std::max(1.0, dev_.fabric_macs_per_cycle), w.macs > 0.0);
                acc_[k] = run(Proc::Cf, ready, d, a);
                break;
            }
            case Action::Kind::Drain:
                drained_[k] = run(Proc::Str, get(acc_, k), transfer(Proc::Str, bytes), a);
                break;
            case Action::Kind::Writeback: {
                Cycle ready = get(drained_, k);
                auto it = residency_.find(k);
                const bool in_place = it != residency_.end() && it->second.slot;
                if (in_place) {
                    ready = std::max(ready, it->second.readers);   // WAR: earlier reads of the slot
                } else {
                    ready = std::max({ready, take_credit(), get(released_, k)});   // opens a residency
                }
                const Cycle f = run(Proc::Bm, ready, transfer(Proc::Bm, bytes), a);
                Residency& r = residency_[k];
                if (!in_place) {
                    r = Residency{};
                    r.slot = true;
                    r.slot_since = ready;
                }
                r.current = f;
                r.readers = std::max(r.readers, f);
                break;
            }
            case Action::Kind::Store: {
                Residency& r = residency_[k];
                // The push-only store: the BlockMover ejects into a DMA buffer, the DMA writes it.
                const Cycle e = run(Proc::Bm, r.current, transfer(Proc::Bm, bytes), a);
                const Cycle f = run(Proc::Dma, e, transfer(Proc::Dma, bytes), a);
                r.readers = std::max(r.readers, e);
                dram_ready_[k] = f;
                ++st_.dram_stores;
                st_.dram_bytes += bytes;
                break;
            }
            case Action::Kind::Release: {
                Residency& r = residency_[k];
                Cycle t = r.readers;
                // A stage's operand (add's bias) is read by the drains and writebacks that carry it.
                auto s = stage_reads_.find(k);
                if (s != stage_reads_.end()) t = std::max(t, s->second);
                credits_.push(t);
                released_[k] = std::max(get(released_, k), t);
                intervals_.push_back({r.slot_since, t});
                slots_.push_back(Slot{a.tile, r.slot_since, t});
                r.slot = false;
                Record rec;
                rec.action = st_.actions;
                rec.kind = a.kind;
                rec.tile = a.tile;
                rec.start = rec.finish = t;
                records_.push_back(rec);
                st_.makespan = std::max(st_.makespan, t);
                break;
            }
        }
        // A context's stages read their operand from L3 when the move that carries them runs.
        for (const Stage& stg : a.context)
            if (stg.op == VeOp::Add) {
                const std::string b = stg.arg.to_string();
                stage_reads_[b] = std::max(stage_reads_[b], records_.back().finish);
            }
        ++st_.actions;
    }

    Stats finish() {
        (void)values_.finish();
        // Peak in time: the most slots held at once, a slot held from its credit to its Release.
        std::vector<std::pair<Cycle, int>> ev;
        for (const auto& [b, e] : intervals_) {
            ev.push_back({b, +1});
            ev.push_back({e, -1});
        }
        std::sort(ev.begin(), ev.end());                // a release at t frees before a take at t
        int live = 0;
        for (const auto& [t, d] : ev) {
            live += d;
            st_.peak_l3 = std::max(st_.peak_l3, static_cast<std::size_t>(std::max(live, 0)));
        }
        return st_;
    }

    const TileProgram& result() const { return values_.result(); }
    const std::vector<Record>& records() const { return records_; }
    const std::vector<Slot>& slots() const { return slots_; }
    std::size_t lanes_of(Proc p) const { return lanes(p); }

private:
    struct Residency {
        Cycle current = 0;          // when the L3 copy became current
        Cycle readers = 0;          // the latest finish of anything that read it (or wrote it)
        bool slot = false;          // holds a credit
        Cycle slot_since = 0;       // when its credit was taken
    };

    characterize::DeviceDescriptor dev_;
    std::size_t l3_;
    BehavioralInterpreter values_;
    TileProgram shapes_;
    Stats st_;
    std::vector<Record> records_;
    std::map<Proc, std::vector<Cycle>> lane_free_;
    std::map<Proc, Cycle> last_start_;
    std::priority_queue<Cycle, std::vector<Cycle>, std::greater<Cycle>> credits_;   // when each free slot frees
    std::map<std::string, Residency> residency_;
    std::map<std::string, Cycle> dram_ready_, moved_, drained_, fed_, acc_, stage_reads_, released_;
    std::vector<std::pair<Cycle, Cycle>> intervals_;     // each residency's slot: credit -> Release
    std::vector<Slot> slots_;

    static Cycle get(const std::map<std::string, Cycle>& m, const std::string& k) {
        auto it = m.find(k);
        return it == m.end() ? 0 : it->second;
    }

    std::size_t lanes(Proc p) const {
        switch (p) {
            case Proc::Dma: return std::max<std::size_t>(1, dev_.dma_engines);
            case Proc::Bm:  return std::max<std::size_t>(1, dev_.block_movers);
            case Proc::Str: return std::max<std::size_t>(1, dev_.streamers);
            case Proc::Cf:  return std::max<std::size_t>(1, dev_.compute_tiles);
        }
        return 1;
    }

    Cycle tile_bytes(const TileCoord& t) const {
        const TensorOperand& o = shapes_.operand(t.operand);
        const double elems = double(o.row_end(t.ti) - o.row_begin(t.ti)) * double(o.col_end(t.tj) - o.col_begin(t.tj));
        return static_cast<Cycle>(elems * dev_.element_bytes);
    }

    Cycle transfer(Proc p, Cycle bytes) const {
        const double rate = p == Proc::Dma ? dev_.dma_bytes_per_cycle
                          : p == Proc::Bm  ? dev_.bm_bytes_per_cycle
                                           : dev_.str_bytes_per_cycle;
        return quantize_cycles(double(bytes) / std::max(1.0, rate), bytes > 0);
    }

    // The earliest a credit is free (an L3 slot), held from then on. With the program's
    // capacity, the pool never runs dry: the program validated against it, and its Releases
    // come earlier in program order.
    Cycle take_credit() {
        const Cycle t = credits_.empty() ? 0 : credits_.top();
        if (!credits_.empty()) credits_.pop();
        return t;
    }

    // Run one leg of `a` on process `p`: start at the latest of `ready`, a free lane, and the
    // process's previous start (in-order issue). Returns the finish.
    Cycle run(Proc p, Cycle ready, Cycle duration, const Action& a) {
        auto& lanes_free = lane_free_[p];
        const auto lane = static_cast<std::size_t>(std::min_element(lanes_free.begin(), lanes_free.end()) -
                                                   lanes_free.begin());
        const Cycle start = std::max({ready, lanes_free[lane], last_start_[p]});
        const Cycle finish = start + duration;
        lanes_free[lane] = finish;
        last_start_[p] = start;
        st_.busy[name(p)] += duration;
        st_.makespan = std::max(st_.makespan, finish);
        Record rec;
        rec.action = st_.actions;
        rec.kind = a.kind;
        rec.tile = a.tile;
        rec.proc = p;
        rec.lane = lane;
        rec.start = start;
        rec.finish = finish;
        records_.push_back(rec);
        return finish;
    }
};

}  // namespace sw::kpu::program::csp
