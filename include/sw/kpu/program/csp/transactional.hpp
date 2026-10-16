// ============================================================================
// include/sw/kpu/program/csp/transactional.hpp
// L-T1 from the CSP program (docs/plans/kpu-run-csp-programs.md step 3; the sequencing plan's
// step 3).
//
// L-T1 decomposes a CSP transaction into one tile move under L3 credits -- and the credits
// are THE PROGRAM'S: a Load takes a slot, the program's Release returns it, a Writeback into
// an open residency writes in place. Nothing is inferred. (The L0-driven TileTransaction-
// Executor inferred residency -- a credit at first touch, returned after the last reader --
// which described an execution order, not a program. It retired in step 4e.)
//
// THE SCHEDULE IS ONE PASS IN PROGRAM ORDER. Every dependency of a CSP action points backwards
// in program order -- its tile was put where it reads it by an earlier action, its credit was
// returned by an earlier Release -- so each action's start is known when it is reached:
//
//   start = max( the data it reads is ready,            (scoreboard per tile and channel)
//                a lane of its process is free,          (lanes x bytes/cycle, per process)
//                the process's previous action started,  (each process issues in order)
//                ITS L3 slot is free )                   (Load, Inherit, a Writeback that opens)
//
// A CREDIT IS A SLOT (ADR 0004 §5; docs/plans/l3-placement-at-lb.md step 5). Every action that
// opens a residency carries the L3 address the walker placed it at, and waits for THAT slot's
// Release -- not for whichever slot frees first. A program, or a placement policy, that reuses a
// slot early serializes on it, as a register's write-after-read does. Retained slots never free.
//
//   Load       dma   DRAM -> L3       after the tile's earlier Store retires (RAW via DRAM)
//                                     and its slot is free
//   Move       bm    L3 -> L2         after the L3 copy is current (its Load / Writeback)
//   Feed       str   L2 -> L1         after its Move
//   Call       cf    the tile function's MACs on a compute tile, after its operands' Feeds and
//                                     the previous call on its accumulator
//   Drain      str   L1 -> L2         after the call that produced it (a stream chain: after its
//                                     last result leaves the array). With a declared fabric the
//                                     results leave through each compute tile's drain port, and
//                                     the drain leg is issued and timed apart from the feeds, on
//                                     its own lanes: str lanes [n, 2n) after the n feed lanes
//   Writeback  bm    L2 -> L3         after its Drain; in place, after the residency's readers
//   Store      bm    L3 -> DMA buffer, then dma buffer -> DRAM (the push-only store's two legs)
//   Release    --    zero time, after every reader of the residency issued before it; its
//                                     credit returns
//   Inherit    --    zero time, at cycle 0: the residency is open from the start, its credit
//                                     held, with no DMA (the operator before left the tile)
//   Retain     --    zero time, after every reader of the residency; its credit is NOT returned:
//                                     the slot is held to the end of the program (step 4d)
//
// THE FABRIC IS A SYSTOLIC ARRAY when the deployment declares one (fabric.pe_rows x pe_cols;
// docs/plans/l2-reuse-and-fabric-pipeline-view.md §3.1). An accumulator's gemm calls -- `acc C in
// fabric { for k ... call gemm(...) +-> C }` -- are then ONE CHAIN, timed output-stationary: C
// stays in the PEs, A and B stream through, and the chain owns one compute tile from its first
// call to its Drain. With the C tile on r x c PEs (its extent, at most the array's) and
// skew = (r-1) + (c-1):
//
//   in0     the first call starts: its operands reach PE(0,0)
//   front   = in0 + skew        the wavefront reaches the far PE
//   in1     the last call ends at PE(0,0): each call is K = macs / (r*c) cycles there (its
//           reduction depth), so a tile smaller than the array leaves the other PEs idle
//   last    = in1 + skew        the far PE finishes
//   out1    = last + r          the far PE's result has travelled r hops north, out of the array
//
// Results drain north on their own links, so a drain never holds up the next chain: its operands
// enter right behind (from in1). The one limit is each column's top link, which carries the
// chain's r results; a chain whose results would overtake the previous chain's waits for them,
// in the grid's registers, so out1 >= the previous chain's out1 + r (link_delay). The phases
// overlap: fill [in0, front), tail [in1, last), and steady [front, in1) is empty when K <= skew.
// Other calls (getrf, trsm, an in-place gemm whose C is preloaded, a C tile larger than the
// array, or any call when no fabric is declared) keep one leg at macs / macs_per_cycle.
//
// The slots are the PROGRAM's L3 (machine flat(l3 = N, banks = B)): the program's declared machine.
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
#include <optional>
#include <string>
#include <utility>
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

    // Release, Inherit and Retain take no process: zero-length records whose `proc` means nothing.
    static bool has_process(Action::Kind k) {
        return k != Action::Kind::Release && k != Action::Kind::Inherit && k != Action::Kind::Retain;
    }

    struct Record {                     // one leg of one action (Release, Inherit, Retain: zero length; Store: two)
        std::size_t action = 0;         // the action's index in program order
        Action::Kind kind = Action::Kind::Call;
        TileCoord tile;
        std::optional<L3Address> place; // the L3 slot the action writes, reads or frees
        Proc proc = Proc::Cf;
        std::size_t lane = 0;
        Cycle start = 0, finish = 0;
        Cycle slot_wait = 0;            // a Load whose slot decided its start: what the wait cost
        std::uint32_t chain = kNoChain; // a Call in a stream chain: its index in chains()
        std::uint32_t chain_pos = 0;    //   and its place in it, from 1
    };
    static constexpr std::uint32_t kNoChain = 0xFFFFFFFFu;

    // One accumulator's gemm calls on the systolic array, from its first call to its Drain (the
    // header's THE FABRIC). Every cycle is L-T1's, uncalibrated.
    struct Chain {
        TileCoord tile;                 // the accumulator
        std::size_t lane = 0;           // the compute tile it owns
        std::uint32_t rows = 0, cols = 0;               // the PEs its C tile occupies
        Cycle in0 = 0, front = 0, in1 = 0, last = 0, out1 = 0, link_delay = 0;
        std::size_t first_action = 0, last_action = 0;
        std::size_t drain_action = kNoAction;           // its Drain (L1 -> L2), when the program ran it
        std::uint32_t calls = 0;
        // Its calls are over, so last and out1 are known: at its Drain, or earlier, when the next
        // accumulator's first call took its compute tile. Its results leave the array as they
        // emerge, into L1's drain port; the Drain action only moves them on to L2.
        bool closed = false;
    };
    static constexpr std::size_t kNoAction = ~std::size_t{0};

    // One residency's L3 slot: held from its credit to its Release (a retained one: to the end).
    struct Slot {
        TileCoord tile;
        L3Address place;                // where: the walker's address, executed
        Cycle t0 = 0, t1 = 0;
        bool inherited = false;         // opened by an Inherit: filled before the program
        bool retained = false;          // closed by a Retain: held past the program's end
    };

    struct Stats {
        Cycle makespan = 0;
        std::size_t actions = 0;
        std::uint64_t dram_loads = 0, dram_stores = 0, dram_bytes = 0;
        std::size_t peak_l3 = 0;                    // slots held at once, in time
        std::size_t credit_stalls = 0;              // Loads whose data was ready before their slot was free
        // Loads whose START their slot decided: the slot freed after the data was ready, a DMA lane
        // was free and the previous DMA action had started. The slot was the binding constraint,
        // and slot_wait_cycles is what it cost them (a reuse's write-after-read, in time).
        std::size_t slot_bound = 0;
        Cycle slot_wait_cycles = 0;
        std::size_t stream_chains = 0;              // chains timed as systolic streams
        Cycle link_delay_cycles = 0;                // what their results waited for a top link
        std::map<std::string, std::size_t> one_leg; // calls timed as one leg, by why
        std::map<std::string, Cycle> busy;          // per process: lane-cycles occupied
        std::map<std::string, std::size_t> lanes;   // per process
    };

    // `l3` and `banks` are the program's L3 (its declared machine; banks 0: unchecked); `dev` gives
    // the lanes and rates.
    TransactionalInterpreter(const characterize::DeviceDescriptor& dev, std::size_t l3, std::uint32_t banks = 0)
        : dev_(dev), l3_(l3), banks_(banks) {}

    void begin(const TileProgram& operands) {
        values_.begin(operands, false, {static_cast<std::uint32_t>(l3_), banks_});
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
        slot_free_.clear();
        residency_.clear();
        dram_ready_.clear();
        moved_.clear();
        drained_.clear();
        fed_.clear();
        acc_.clear();
        chains_.clear();
        open_chain_.clear();
        lane_chain_.assign(lanes(Proc::Cf), kNoChain);
        lane_out1_.assign(lanes(Proc::Cf), 0);
        // The drain path (a declared fabric): one drain lane per streamer, in order among drains,
        // never behind a feed. The streamer then counts its feed lanes and its drain lanes.
        drain_free_.assign(fabric() ? lanes(Proc::Str) : 0, 0);
        drain_last_start_ = 0;
        if (fabric()) st_.lanes[name(Proc::Str)] += lanes(Proc::Str);
        intervals_.clear();
        slots_.clear();
        stage_reads_.clear();
        released_.clear();
        retained_.clear();
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
                const Cycle credit = take_slot(a);
                Cycle slot_wait = 0;
                if (credit > ready) {
                    // Every other bound on the start: a lane free, and in-order issue.
                    const auto& lanes_free = lane_free_[Proc::Dma];
                    const Cycle other = std::max({ready, *std::min_element(lanes_free.begin(), lanes_free.end()),
                                                  last_start_[Proc::Dma]});
                    if (credit > other) {
                        ++st_.slot_bound;
                        st_.slot_wait_cycles += credit - other;
                        slot_wait = credit - other;
                    }
                    ready = credit;
                    ++st_.credit_stalls;                    // the data was ready; the slot was not
                }
                const Cycle f = run(Proc::Dma, ready, transfer(Proc::Dma, bytes), a);
                records_.back().slot_wait = slot_wait;
                Residency& r = residency_[k];
                r = Residency{};
                r.current = f;
                r.readers = f;
                r.slot = true;
                r.slot_since = ready;
                r.place = *a.place;
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
                const std::uint32_t c = stream_chain(a, *call_op);
                if (c == kNoChain) {
                    const Cycle d = quantize_cycles(w.macs / std::max(1.0, dev_.fabric_macs_per_cycle), w.macs > 0.0);
                    acc_[k] = run(Proc::Cf, ready, d, a);
                    break;
                }
                // K = the call's reduction depth: macs over the PEs its C tile occupies.
                Chain& ch = chains_[c];
                const Cycle d = quantize_cycles(w.macs / double(ch.rows * ch.cols), w.macs > 0.0);
                const Cycle f = run_on(Proc::Cf, ch.lane, ready, d, a);
                Record& rec = records_.back();
                rec.chain = c;
                rec.chain_pos = ++ch.calls;
                if (ch.calls == 1) {
                    ch.in0 = rec.start;
                    ch.front = ch.in0 + skew(ch);
                    ch.first_action = rec.action;
                }
                ch.in1 = f;
                ch.last_action = rec.action;
                acc_[k] = f;
                break;
            }
            case Action::Kind::Drain: {
                Cycle ready = get(acc_, k);
                auto oc = open_chain_.find(k);
                if (oc != open_chain_.end()) {                  // its last result is out of the array
                    if (!chains_[oc->second].closed) close_chain(oc->second);
                    ready = std::max(ready, chains_[oc->second].out1);
                }
                drained_[k] = fabric() ? run_drain(ready, transfer(Proc::Str, bytes), a)
                                       : run(Proc::Str, ready, transfer(Proc::Str, bytes), a);
                if (oc != open_chain_.end()) {
                    chains_[oc->second].drain_action = records_.back().action;
                    open_chain_.erase(oc);
                }
                break;
            }
            case Action::Kind::Writeback: {
                Cycle ready = get(drained_, k);
                auto it = residency_.find(k);
                const bool in_place = it != residency_.end() && it->second.slot;
                if (in_place) {
                    ready = std::max(ready, it->second.readers);   // WAR: earlier reads of the slot
                } else {
                    ready = std::max({ready, take_slot(a), get(released_, k)});   // opens a residency
                }
                const Cycle f = run(Proc::Bm, ready, transfer(Proc::Bm, bytes), a);
                Residency& r = residency_[k];
                if (!in_place) {
                    r = Residency{};
                    r.slot = true;
                    r.slot_since = ready;
                    r.place = *a.place;
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
            case Action::Kind::Inherit: {
                const Cycle credit = take_slot(a);
                Residency& r = residency_[k];
                r = Residency{};
                r.slot = true;
                r.slot_since = credit;
                r.place = *a.place;
                r.current = credit;
                r.readers = credit;
                r.inherited = true;
                zero_length(a, credit);
                break;
            }
            case Action::Kind::Retain: {
                Residency& r = residency_[k];
                Cycle t = r.readers;
                auto s = stage_reads_.find(k);
                if (s != stage_reads_.end()) t = std::max(t, s->second);
                retained_.push_back(Slot{a.tile, r.place, r.slot_since, 0, r.inherited, true});   // closed in finish()
                r.slot = false;
                zero_length(a, t);
                break;
            }
            case Action::Kind::Release: {
                Residency& r = residency_[k];
                Cycle t = r.readers;
                // A stage's operand (add's bias) is read by the drains and writebacks that carry it.
                auto s = stage_reads_.find(k);
                if (s != stage_reads_.end()) t = std::max(t, s->second);
                slot_free_[key_of(r.place)] = t;           // THIS slot is free from t
                released_[k] = std::max(get(released_, k), t);
                intervals_.push_back({r.slot_since, t});
                slots_.push_back(Slot{a.tile, r.place, r.slot_since, t, r.inherited, false});
                r.slot = false;
                zero_length(a, t);
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
        // A retained slot is held to the end of the program.
        for (Slot sl : retained_) {
            sl.t1 = st_.makespan;
            intervals_.push_back({sl.t0, sl.t1});
            slots_.push_back(sl);
        }
        retained_.clear();
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
    const std::vector<Chain>& chains() const { return chains_; }
    std::size_t lanes_of(Proc p) const { return lanes(p); }

private:
    struct Residency {
        Cycle current = 0;          // when the L3 copy became current
        Cycle readers = 0;          // the latest finish of anything that read it (or wrote it)
        bool slot = false;          // holds its slot
        Cycle slot_since = 0;       // when it took its slot
        bool inherited = false;     // opened by an Inherit
        L3Address place;            // its slot
    };

    characterize::DeviceDescriptor dev_;
    std::size_t l3_;
    std::uint32_t banks_ = 0;
    BehavioralInterpreter values_;
    TileProgram shapes_;
    Stats st_;
    std::vector<Record> records_;
    std::map<Proc, std::vector<Cycle>> lane_free_;
    std::map<Proc, Cycle> last_start_;
    std::map<std::pair<std::uint32_t, std::uint32_t>, Cycle> slot_free_;   // (module, slot) -> when its Release freed it
    std::map<std::string, Residency> residency_;
    std::map<std::string, Cycle> dram_ready_, moved_, drained_, fed_, acc_, stage_reads_, released_;
    std::vector<Chain> chains_;
    std::map<std::string, std::uint32_t> open_chain_;     // accumulator -> its open chain
    std::vector<std::uint32_t> lane_chain_;              // compute tile -> the open chain it holds
    std::vector<Cycle> lane_out1_;                       // compute tile -> its last chain's out1
    std::vector<Cycle> drain_free_;                      // the streamers' drain lanes (a declared fabric)
    Cycle drain_last_start_ = 0;
    std::vector<std::pair<Cycle, Cycle>> intervals_;     // each residency's slot: credit -> Release
    std::vector<Slot> slots_;
    std::vector<Slot> retained_;                         // open to the end of the program

    void zero_length(const Action& a, Cycle t) {
        Record rec;
        rec.action = st_.actions;
        rec.kind = a.kind;
        rec.tile = a.tile;
        rec.place = a.place;
        rec.start = rec.finish = t;
        records_.push_back(rec);
        st_.makespan = std::max(st_.makespan, t);
    }

    static std::pair<std::uint32_t, std::uint32_t> key_of(const L3Address& s) { return {s.l3_tile, s.slot}; }

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

    // When the action's own slot is free: its previous occupant's Release (0 if it has had none).
    // L-B, stepped first, has already refused an action without an address or into a live slot.
    Cycle take_slot(const Action& a) const {
        auto it = slot_free_.find(key_of(*a.place));
        return it == slot_free_.end() ? 0 : it->second;
    }

    static Cycle skew(const Chain& ch) { return Cycle(ch.rows - 1) + Cycle(ch.cols - 1); }
    bool fabric() const { return dev_.pe_rows > 0 && dev_.pe_cols > 0; }

    // A Drain leg on the drain path: a free drain lane, in order among drains only. Recorded as
    // the streamer's, on lane n + i (after its n feed lanes).
    Cycle run_drain(Cycle ready, Cycle duration, const Action& a) {
        const auto i = static_cast<std::size_t>(std::min_element(drain_free_.begin(), drain_free_.end()) - drain_free_.begin());
        const Cycle start = std::max({ready, drain_free_[i], drain_last_start_});
        const Cycle finish = start + duration;
        drain_free_[i] = finish;
        drain_last_start_ = start;
        st_.busy[name(Proc::Str)] += duration;
        st_.makespan = std::max(st_.makespan, finish);
        Record rec;
        rec.action = st_.actions;
        rec.kind = a.kind;
        rec.tile = a.tile;
        rec.place = a.place;
        rec.proc = Proc::Str;
        rec.lane = lanes(Proc::Str) + i;
        rec.start = start;
        rec.finish = finish;
        records_.push_back(rec);
        return finish;
    }

    // The stream chain a Call belongs to, opening one at an accumulator's first gemm; kNoChain
    // for a call timed as one leg (counted in Stats::one_leg, by why).
    std::uint32_t stream_chain(const Action& a, const TileOp& op) {
        const std::string k = a.tile.to_string();
        auto one_leg = [&](const char* why) { ++st_.one_leg[why]; return kNoChain; };
        auto open = open_chain_.find(k);
        if (open != open_chain_.end())
            // A chain whose compute tile the next accumulator took: its calls were over, and this
            // one comes back to it -- two accumulators interleaved on one array.
            return chains_[open->second].closed ? one_leg("accumulators interleaved on one array") : open->second;
        if (dev_.pe_rows == 0 || dev_.pe_cols == 0) return one_leg("no fabric declared");
        if (op.kind != TileOpKind::MatMulAccum || !a.accumulate)
            return one_leg(op.kind == TileOpKind::MatMulAccum ? "in-place gemm (C preloaded)" : "not a gemm");
        const TensorOperand& o = shapes_.operand(a.tile.operand);
        const Dim rows = o.row_end(a.tile.ti) - o.row_begin(a.tile.ti);
        const Dim cols = o.col_end(a.tile.tj) - o.col_begin(a.tile.tj);
        if (rows > dev_.pe_rows || cols > dev_.pe_cols) return one_leg("C tile larger than the array");
        // A compute tile with no chain on it; else the one whose chain's calls ended first, which
        // this accumulator's first call closes: C occupies the array only while its calls run.
        std::size_t lane = lane_chain_.size();
        const auto& free = lane_free_[Proc::Cf];
        for (std::size_t l = 0; l < lane_chain_.size(); ++l)
            if (lane_chain_[l] == kNoChain && (lane == lane_chain_.size() || free[l] < free[lane])) lane = l;
        if (lane == lane_chain_.size()) {
            lane = 0;
            for (std::size_t l = 1; l < lane_chain_.size(); ++l)
                if (chains_[lane_chain_[l]].in1 < chains_[lane_chain_[lane]].in1) lane = l;
            close_chain(lane_chain_[lane]);
        }
        Chain ch;
        ch.tile = a.tile;
        ch.lane = lane;
        ch.rows = static_cast<std::uint32_t>(std::max<Dim>(1, rows));
        ch.cols = static_cast<std::uint32_t>(std::max<Dim>(1, cols));
        const auto c = static_cast<std::uint32_t>(chains_.size());
        chains_.push_back(ch);
        open_chain_[k] = c;
        lane_chain_[lane] = c;
        ++st_.stream_chains;
        return c;
    }

    // The chain's calls are over: its last result leaves the array (out1), no earlier than the
    // previous chain's results allow on each column's top link, and its compute tile is free.
    Cycle close_chain(std::uint32_t c) {
        Chain& ch = chains_[c];
        ch.last = ch.in1 + skew(ch);
        const Cycle out = ch.last + ch.rows;
        const Cycle link = lane_out1_[ch.lane] ? lane_out1_[ch.lane] + ch.rows : 0;
        ch.out1 = std::max(out, link);
        ch.link_delay = ch.out1 - out;
        st_.link_delay_cycles += ch.link_delay;
        ch.closed = true;
        lane_out1_[ch.lane] = ch.out1;
        lane_chain_[ch.lane] = kNoChain;
        st_.makespan = std::max(st_.makespan, ch.out1);
        return ch.out1;
    }

    // Run one leg of `a` on process `p`: start at the latest of `ready`, a free lane, and the
    // process's previous start (in-order issue). Returns the finish.
    Cycle run(Proc p, Cycle ready, Cycle duration, const Action& a) {
        auto& lanes_free = lane_free_[p];
        const auto lane = static_cast<std::size_t>(std::min_element(lanes_free.begin(), lanes_free.end()) -
                                                   lanes_free.begin());
        return run_on(p, lane, ready, duration, a);
    }
    // The same, on a given lane (a stream chain's compute tile).
    Cycle run_on(Proc p, std::size_t lane, Cycle ready, Cycle duration, const Action& a) {
        auto& lanes_free = lane_free_[p];
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
        rec.place = a.place;
        rec.proc = p;
        rec.lane = lane;
        rec.start = start;
        rec.finish = finish;
        records_.push_back(rec);
        return finish;
    }
};

}  // namespace sw::kpu::program::csp
