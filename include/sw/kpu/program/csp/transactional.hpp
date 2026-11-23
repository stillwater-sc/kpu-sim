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
//                ITS L3 slot is free,                    (Load, Inherit, a Writeback that opens)
//                ITS L2 slot is free )                   (a stage Move)
//
// A CREDIT IS A SLOT (ADR 0004 §5; docs/plans/l3-placement-at-lb.md step 5). Every action that
// opens a residency carries the L3 address the walker placed it at, and waits for THAT slot's
// Release -- not for whichever slot frees first. A program, or a placement policy, that reuses a
// slot early serializes on it, as a register's write-after-read does. Retained slots never free.
//
//   Load       dma   DRAM -> L3       after the tile's earlier Store retires (RAW via DRAM)
//                                     and its slot is free
//   Move       bm    L3 -> L2         after the L3 copy is current (its Load / Writeback)
//   Feed       str   L2 -> L1         after its Move. An accumulator's init (`acc Y init X`) is
//                                     one more edge stream: its Feed is timed as any operand's,
//                                     and Y's first call waits for it as for A's and B's
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
//   L2Release  --    zero time, after the staged copy's last Feed; its L2 slot's credit returns
//
// L2 STAGING HOLDS AN L2 CREDIT PER SLOT (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 4).
// A stage Move (`stage T in l2 { }`) carries the L2 address the walker placed the copy at and
// waits for THAT slot's L2Release, as a Load waits for its L3 slot. The staged copy is held from
// that credit to its L2Release, and every Feed in between reads it (the reuse count). A transient
// Move -- an unstaged operand, fed once -- holds no L2 credit at L-T1: its copy streams straight on.
//
// AN L2 ACCUMULATOR'S PARTIAL HOLDS ONE TOO (#429, #430). The first merge (a Drain, Overwrite)
// waits for its slot's credit, as a stage Move does, and the slot is held to the partial's
// L2Release, which follows its store's Writeback. Every later merge (Accumulate) is a
// read-modify-write of the slot: it starts after the merge before it into the same slot has
// finished. Each merge's gemm is a stream chain of one call (its partial starts from zero in the
// array), timed output-stationary like any chain; the merge itself rides the Drain at the
// streamer's rate. Stats count the merges and their L2 bytes (an accumulate reads the partial and
// writes it back; an overwrite only writes). L2's banks are untimed (#417).
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
// WEIGHT- AND INPUT-STATIONARY (#395): the PEs hold B (ws) or Aᵀ (is), which a PRELOAD shifts down
// every column, and the other operand streams through; each call's partial sums leave the array
// SOUTH, merged in L2 by the Combiner (an `acc ... in l2 dataflow ws|is`, so every call is a chain
// of one). Its dataflow is the accumulator's, else the deployment's (fabric.dataflow). Following
// the RTL's schedule rules (systars docs/architecture/systolic-tile.md), with R = pe_rows, the
// stationary tile K x W on the PEs (ws: K x N of B; is: K x M of Aᵀ) and S rows streamed (ws: M
// rows of A; is: N columns of B):
//
//   in0     PRELOAD starts: the stationary tile enters the top of the columns
//   front   = in0 + R           PRELOAD done: R clocks whatever K is, so K < R pads with zeros
//   in1     = front + S         the last streamed row has entered the array
//   last    = in1 + (W - 1)     it has crossed the stationary tile's W columns
//   out1    = last + R          its partial sum has crossed the R rows, out of the south edge
//
// The compute tile is busy for R + max(S, R): the RTL's command interval, and the next PRELOAD
// waits for the previous stream to leave. Results leave south, so the north-link rule does not
// apply. A stationary tile larger than the array keeps one leg.
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
#include <cmath>
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

    // Release, Inherit, Retain and L2Release take no process: zero-length records whose `proc`
    // means nothing.
    static bool has_process(Action::Kind k) {
        return k != Action::Kind::Release && k != Action::Kind::Inherit && k != Action::Kind::Retain &&
               k != Action::Kind::L2Release;
    }

    struct Record {                     // one leg of one action (Release, Inherit, Retain: zero length; Store: two)
        std::size_t action = 0;         // the action's index in program order
        Action::Kind kind = Action::Kind::Call;
        TileCoord tile;
        std::optional<L3Address> place; // the L3 slot the action writes, reads or frees
        Proc proc = Proc::Cf;
        std::size_t lane = 0;
        Cycle start = 0, finish = 0;
        // When its inputs (and the credit it takes) were ready: start - ready is what a busy lane
        // or the process's in-order issue cost it (#425's bubble report). Zero-length: start.
        Cycle ready = 0;
        Cycle slot_wait = 0;            // a Load whose slot decided its start: what the wait cost
        Cycle l2_wait = 0;              // a stage Move or a first merge whose L2 slot decided its start: the same
        Cycle bank_wait = 0;            // L-CA, banked (#401): the clocks its L2 rows waited for their banks' ports
        Cycle bank_delay = 0;           //   and the clocks those waits added to its finish (<= bank_wait)
        // What set `ready`: its inputs (the data it reads), a credit (an L3 or L2 slot's), or its
        // handover (L-CA's driver window: the program's issue order). The bubble walk reads it
        // rather than inferring it from the clocks (#401).
        enum class ReadyBy : std::uint8_t { Input, Credit, Handover };
        ReadyBy ready_by = ReadyBy::Input;
        std::uint32_t chain = kNoChain; // a Call in a stream chain: its index in chains()
        std::uint32_t chain_pos = 0;    //   and its place in it, from 1
    };
    static constexpr std::uint32_t kNoChain = 0xFFFFFFFFu;

    // One accumulator's gemm calls on the systolic array, from its first call to its Drain (the
    // header's THE FABRIC). Every cycle is L-T1's, uncalibrated.
    struct Chain {
        TileCoord tile;                 // the accumulator
        std::size_t lane = 0;           // the compute tile it owns
        std::uint32_t rows = 0, cols = 0;               // the PEs its C tile occupies (ws/is: the stationary tile)
        Dataflow dataflow = Dataflow::OS;               // os, ws or is: resolved, never Default (#395)
        std::uint32_t preload = 0, streamed = 0;        // ws/is: the PRELOAD's R clocks, and S rows streamed
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

    // One L2 slot's hold, from its credit to its L2Release: a staged copy (its stage Move), an L2
    // accumulator's partial (its first merge), or a copy the walker places (slice 2): an unstaged
    // operand's (its Move, one Feed) or a result's (its opening call; Drain, Writeback).
    struct L2Slot {
        TileCoord tile;
        L2Address place;                // the walker's L2 address, executed
        Cycle t0 = 0, t1 = 0;
        std::uint32_t feeds = 0;        // the Feeds that read it: its reuse count
        bool partial = false;           // an L2 accumulator's partial (#429): merged into, not fed
        std::uint32_t merges = 0;       // a partial: the Drains the Combiner merged into it
        bool copy = false;              // a copy the walker places, not a staging the program states
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
        std::size_t peak_l2 = 0;                    // staged copies held at once, in time
        std::size_t peak_l2_held = 0;               // every L2 slot held at once, the walker's copies too
        std::size_t l2_copies = 0;                  // holds of a copy the walker places (slice 2's step 2)
        std::size_t l2_bound = 0;                   // stage Moves whose start their L2 slot decided
        Cycle l2_wait_cycles = 0;                   //   and what those waits cost
        std::size_t merges = 0;                     // Drains the Combiner merged into an L2 partial
        std::size_t merge_accumulates = 0;          //   of them, read-modify-writes (all but each first)
        std::uint64_t merge_l2_read_bytes = 0;      // the partials those read back from L2
        std::uint64_t merge_l2_write_bytes = 0;     // and what every merge wrote into L2
        std::size_t stream_chains = 0;              // chains timed as systolic streams
        std::map<std::string, std::size_t> chains_by_dataflow;   // os, ws, is (#395)
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
        partial_read_.clear();
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
        l2_free_.clear();
        l2_open_.clear();
        l2_slots_.clear();
    }

    void step(const Action& placed, const TileOp* call_op) {
        values_.step(placed, call_op);              // the values: L-B's, in the same order
        // The timing executes every L2 slot the walker placed (slice 2's step 2): a copy's Move
        // takes its slot's credit and its L2Release returns it, as a staging's do, and a result's
        // slot is taken at the call that opens it (Action::result_l2). On a device with more than
        // one compute tile it can't: a level-1 program places every copy in compute tile 0's L2
        // (L2Address::cf), and L-T1 spreads the calls over the tiles, each with its own L2. There
        // the copies run unplaced, as before, and a copy's L2Release has nothing to do.
        if (!copies_placed() && placed.copy && placed.kind == Action::Kind::L2Release) {
            ++st_.actions;
            return;
        }
        const Action a = copies_placed() ? placed : unplaced(placed);
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
                Cycle ready = r.current;
                Cycle l2_wait = 0;
                bool credit_set = false;                    // the L2 slot's credit set `ready`
                if (a.l2) {                                 // a stage: after ITS L2 slot's L2Release
                    const Cycle credit = get_l2(*a.l2);
                    if (credit > ready) {
                        const auto& lanes_free = lane_free_[Proc::Bm];
                        const Cycle other = std::max({ready, *std::min_element(lanes_free.begin(), lanes_free.end()),
                                                      last_start_[Proc::Bm]});
                        if (credit > other) {
                            ++st_.l2_bound;
                            st_.l2_wait_cycles += credit - other;
                            l2_wait = credit - other;
                        }
                        ready = credit;
                        credit_set = true;
                    }
                }
                const Cycle f = run(Proc::Bm, ready, transfer(Proc::Bm, bytes), a);
                records_.back().l2_wait = l2_wait;
                if (credit_set) records_.back().ready_by = Record::ReadyBy::Credit;
                r.readers = std::max(r.readers, f);
                moved_[k] = f;
                if (a.l2) {
                    l2_open_[k] = l2_slots_.size();
                    l2_slots_.push_back(L2Slot{a.tile, *a.l2, ready, 0, 0, false, 0, a.copy});
                }
                break;
            }
            case Action::Kind::Feed:
                fed_[k] = run(Proc::Str, get(moved_, k), transfer(Proc::Str, bytes), a);
                if (a.seeds) fed_[a.seeds->to_string()] = fed_[k];   // the accumulator's first call waits for it
                if (a.l2) {
                    auto it = l2_open_.find(k);
                    if (it != l2_open_.end()) ++l2_slots_[it->second].feeds;
                }
                break;
            case Action::Kind::Call: {
                Cycle ready = get(acc_, k);                         // the chain's previous call
                for (const TileCoord& t : call_op->inputs) ready = std::max(ready, get(fed_, t.to_string()));
                for (const TileCoord& t : call_op->outputs)        // an in-place result is fed too
                    ready = std::max(ready, get(fed_, t.to_string()));
                // A call that opens its result's L2 slot starts after that slot's credit (decision 1
                // of slot-credits-at-lca.md): the result has a home before any compute.
                Cycle l2_wait = 0;
                bool credit_set = false;
                if (a.result_l2) {
                    const Cycle credit = get_l2(*a.result_l2);
                    if (credit > ready) {
                        const auto& lanes_free = lane_free_[Proc::Cf];
                        const Cycle other = std::max({ready, *std::min_element(lanes_free.begin(), lanes_free.end()),
                                                      last_start_[Proc::Cf]});
                        if (credit > other) {
                            ++st_.l2_bound;
                            st_.l2_wait_cycles += credit - other;
                            l2_wait = credit - other;
                        }
                        ready = credit;
                        credit_set = true;
                    }
                }
                // After the call's record is made: its wait, and the result's slot opened at its start.
                auto took_result_slot = [&]() {
                    Record& rec = records_.back();
                    rec.l2_wait = l2_wait;
                    if (credit_set) rec.ready_by = Record::ReadyBy::Credit;
                    if (a.result_l2) {
                        l2_open_[k] = l2_slots_.size();
                        l2_slots_.push_back(L2Slot{a.tile, *a.result_l2, rec.start, 0, 0, false, 0, true});
                    }
                };
                const TileWork w = tile_work_of(shapes_, *call_op, dev_.element_bytes);
                const std::uint32_t c = stream_chain(a, *call_op);
                if (c == kNoChain) {
                    const Cycle d = quantize_cycles(w.macs / std::max(1.0, dev_.fabric_macs_per_cycle), w.macs > 0.0);
                    acc_[k] = run(Proc::Cf, ready, d, a);
                    took_result_slot();
                    break;
                }
                Chain& ch = chains_[c];
                if (ch.dataflow != Dataflow::OS) {
                    // PRELOAD R clocks, then S rows streamed; the tile is busy R + max(S, R).
                    const Cycle R = ch.preload, S = ch.streamed;
                    const Cycle f = run_on(Proc::Cf, ch.lane, ready, R + std::max(S, R), a);
                    Record& rec = records_.back();
                    rec.chain = c;
                    rec.chain_pos = ++ch.calls;
                    ch.in0 = rec.start;
                    ch.front = ch.in0 + R;
                    ch.in1 = ch.front + S;
                    ch.first_action = ch.last_action = rec.action;
                    acc_[k] = f;
                    took_result_slot();
                    break;
                }
                // K = the call's reduction depth: macs over the PEs its C tile occupies.
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
                took_result_slot();
                break;
            }
            case Action::Kind::Drain: {
                Cycle ready = get(acc_, k);
                Cycle l2_wait = 0;
                auto oc = open_chain_.find(k);
                if (oc != open_chain_.end()) {                  // its last result is out of the array
                    if (!chains_[oc->second].closed) close_chain(oc->second);
                    ready = std::max(ready, chains_[oc->second].out1);
                }
                // Its stages' vectors are in L3 (#446's review): applied before the L2 credit, so a
                // credit that an init's Load already outlasts is not counted as an L2 stall.
                ready = std::max(ready, stage_inputs(a));
                if (a.merge == Merge::Overwrite) {              // the partial's slot: after its credit
                    const Cycle credit = get_l2(*a.l2);
                    if (credit > ready) {
                        // The credit binds only past every other bound on the drain's start: a
                        // free drain lane and the drains' issue order (run_drain, run).
                        const auto& lanes_free = fabric() ? drain_free_ : lane_free_[Proc::Str];
                        const Cycle issue = fabric() ? drain_last_start_ : last_start_[Proc::Str];
                        const Cycle other = std::max({ready, *std::min_element(lanes_free.begin(), lanes_free.end()), issue});
                        if (credit > other) {
                            ++st_.l2_bound;
                            st_.l2_wait_cycles += credit - other;
                            l2_wait = credit - other;
                        }
                        ready = credit;
                    }
                    l2_open_[k] = l2_slots_.size();
                    l2_slots_.push_back(L2Slot{a.tile, *a.l2, ready, 0, 0, true, 0});
                } else if (a.merge == Merge::Accumulate) {     // read-modify-write: after the merge before it
                    ready = std::max(ready, get(drained_, k));
                    ++st_.merge_accumulates;
                    st_.merge_l2_read_bytes += bytes;
                }
                if (a.merge != Merge::None) {
                    ++st_.merges;
                    st_.merge_l2_write_bytes += bytes;
                    auto it = l2_open_.find(k);
                    if (it != l2_open_.end()) ++l2_slots_[it->second].merges;
                }
                drained_[k] = fabric() ? run_drain(ready, transfer(Proc::Str, bytes), a)
                                       : run(Proc::Str, ready, transfer(Proc::Str, bytes), a);
                records_.back().l2_wait = l2_wait;
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
                    // ...and its stages' reads of it: an add's bias, a scale's vectors, an init (#446's review)
                    if (auto sr = stage_reads_.find(k); sr != stage_reads_.end()) ready = std::max(ready, sr->second);
                } else {
                    ready = std::max({ready, take_slot(a), get(released_, k)});   // opens a residency
                }
                ready = std::max(ready, stage_inputs(a));       // its bm.egress stages' vectors (#446's review)
                const Cycle f = run(Proc::Bm, ready, transfer(Proc::Bm, bytes), a);
                if (a.l2) partial_read_[k] = f;             // an L2 partial, read out of its slot
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
            case Action::Kind::L2Release: {
                // A stage block's end: its slot is free once the staged copy's last Feed is out,
                // and the next stage into it may start from then (its credit returns upstream).
                const Cycle t = std::max({get(moved_, k), get(fed_, k), get(partial_read_, k)});
                partial_read_.erase(k);                     // this slot's reads are done
                if (a.l2) l2_free_[key_of(*a.l2)] = t;
                auto it = l2_open_.find(k);
                if (it != l2_open_.end()) {
                    l2_slots_[it->second].t1 = t;
                    l2_open_.erase(it);
                }
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
        for (const Stage& stg : a.context) {
            std::vector<const TileCoord*> reads;                 // an add's vector; a per-channel scale's two
            if (stg.op == VeOp::Add || stg.op == VeOp::Init) reads.push_back(&stg.arg);
            if (stg.op == VeOp::Scale && stg.channels) reads = {&stg.arg, &stg.arg2};
            for (const TileCoord* t : reads) {
                const std::string b = t->to_string();
                stage_reads_[b] = std::max(stage_reads_[b], records_.back().finish);
            }
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
        // The same for L2 (L-B, stepped first, refused a program ending with a slot held): the
        // staged copies and partials the program states, and every slot held, the walker's copies too.
        auto peak = [&](bool with_copies) {
            ev.clear();
            for (const L2Slot& sl : l2_slots_) {
                if (sl.copy && !with_copies) continue;
                ev.push_back({sl.t0, +1});
                ev.push_back({sl.t1, -1});
            }
            std::sort(ev.begin(), ev.end());
            int held = 0;
            std::size_t most = 0;
            for (const auto& [t, d] : ev) {
                held += d;
                most = std::max(most, static_cast<std::size_t>(std::max(held, 0)));
            }
            return most;
        };
        st_.peak_l2 = peak(false);
        st_.peak_l2_held = peak(true);
        st_.l2_copies = static_cast<std::size_t>(
            std::count_if(l2_slots_.begin(), l2_slots_.end(), [](const L2Slot& sl) { return sl.copy; }));
        return st_;
    }

    const TileProgram& result() const { return values_.result(); }
    const std::vector<Record>& records() const { return records_; }
    const std::vector<Slot>& slots() const { return slots_; }
    const std::vector<L2Slot>& l2_slots() const { return l2_slots_; }     // every hold; L2Slot::copy tells them apart
    // Whether L-T1 executes the walker's L2 copies in their slots: on one compute tile, whose L2
    // the program's placement names.
    bool copies_placed() const { return lanes(Proc::Cf) == 1; }
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
    std::map<std::string, Cycle> dram_ready_, moved_, drained_, fed_, acc_, stage_reads_, released_, partial_read_;
    std::vector<Chain> chains_;
    std::map<std::string, std::uint32_t> open_chain_;     // accumulator -> its open chain
    std::vector<std::uint32_t> lane_chain_;              // compute tile -> the open chain it holds
    std::vector<Cycle> lane_out1_;                       // compute tile -> its last chain's out1
    std::vector<Cycle> drain_free_;                      // the streamers' drain lanes (a declared fabric)
    Cycle drain_last_start_ = 0;
    std::vector<std::pair<Cycle, Cycle>> intervals_;     // each residency's slot: credit -> Release
    std::vector<Slot> slots_;
    std::vector<Slot> retained_;                         // open to the end of the program
    std::map<std::pair<std::uint32_t, std::uint32_t>, Cycle> l2_free_;    // (cf, slot) -> when its L2Release freed it
    std::map<std::string, std::size_t> l2_open_;         // a staged tile -> its open L2Slot
    std::vector<L2Slot> l2_slots_;

    void zero_length(const Action& a, Cycle t) {
        Record rec;
        rec.action = st_.actions;
        rec.kind = a.kind;
        rec.tile = a.tile;
        rec.place = a.place;
        rec.start = rec.finish = rec.ready = t;
        records_.push_back(rec);
        st_.makespan = std::max(st_.makespan, t);
    }

    static std::pair<std::uint32_t, std::uint32_t> key_of(const L3Address& s) { return {s.l3_tile, s.slot}; }
    static std::pair<std::uint32_t, std::uint32_t> key_of(const L2Address& s) { return {s.cf, s.slot}; }
    Cycle get_l2(const L2Address& s) const {
        auto it = l2_free_.find(key_of(s));
        return it == l2_free_.end() ? 0 : it->second;
    }

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
    // When the vectors an action's stages read are in L3: an add's bias, a per-channel scale's two, an
    // L2 accumulator's init. A stage reads them as its move runs, so the move cannot start before.
    Cycle stage_inputs(const Action& a) const {
        Cycle t = 0;
        for (const Stage& stg : a.context) {
            std::vector<const TileCoord*> reads;
            if (stg.op == VeOp::Add || stg.op == VeOp::Init) reads.push_back(&stg.arg);
            if (stg.op == VeOp::Scale && stg.channels) reads = {&stg.arg, &stg.arg2};
            for (const TileCoord* c : reads)
                if (auto it = residency_.find(c->to_string()); it != residency_.end()) t = std::max(t, it->second.current);
        }
        return t;
    }

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
        rec.ready = ready;
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
        const bool gemm = op.kind == TileOpKind::MatMulAccum || op.kind == TileOpKind::MatMulMerge;
        if (!gemm || !a.accumulate)
            return one_leg(gemm ? "in-place gemm (C preloaded)" : "not a gemm");
        const TensorOperand& o = shapes_.operand(a.tile.operand);
        const Dim rows = o.row_end(a.tile.ti) - o.row_begin(a.tile.ti);
        const Dim cols = o.col_end(a.tile.tj) - o.col_begin(a.tile.tj);
        // The reduction's dataflow (#395): the accumulator's, else the deployment's.
        Dataflow df = a.dataflow;
        if (df == Dataflow::Default && !parse_dataflow(dev_.dataflow, df)) df = Dataflow::OS;
        // ws holds B's K x N tile and streams M rows of A; is holds Aᵀ's K x M and streams N.
        Dim srows = rows, scols = cols, streamed = 0;
        if (df != Dataflow::OS) {
            const Dim m = std::max<Dim>(1, rows), n = std::max<Dim>(1, cols);
            const Dim k = std::max<Dim>(1, static_cast<Dim>(std::llround(
                tile_work_of(shapes_, op, dev_.element_bytes).macs / double(m * n))));
            srows = k;
            scols = df == Dataflow::WS ? n : m;
            streamed = df == Dataflow::WS ? m : n;
            if (srows > dev_.pe_rows || scols > dev_.pe_cols) return one_leg("stationary tile larger than the array");
        } else if (rows > dev_.pe_rows || cols > dev_.pe_cols) {
            return one_leg("C tile larger than the array");
        }
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
        ch.rows = static_cast<std::uint32_t>(std::max<Dim>(1, srows));
        ch.cols = static_cast<std::uint32_t>(std::max<Dim>(1, scols));
        ch.dataflow = df;
        if (df != Dataflow::OS) {
            ch.preload = static_cast<std::uint32_t>(dev_.pe_rows);
            ch.streamed = static_cast<std::uint32_t>(streamed);
        }
        ++st_.chains_by_dataflow[to_string(df)];
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
        if (ch.dataflow != Dataflow::OS) {
            // The last streamed row crosses W columns, and its partial sum the R rows, out south.
            ch.last = ch.in1 + (ch.cols - 1);
            ch.out1 = ch.last + ch.preload;
            ch.link_delay = 0;
            ch.closed = true;
            lane_chain_[ch.lane] = kNoChain;
            st_.makespan = std::max(st_.makespan, ch.out1);
            return ch.out1;
        }
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
        rec.ready = ready;
        rec.finish = finish;
        records_.push_back(rec);
        return finish;
    }
};

}  // namespace sw::kpu::program::csp
