#pragma once
// ============================================================================
// tests/cycle/streamer_rig.hpp
// L3 -> BlockMover -> L2 -> Streamer -> L1 lanes, for step 2b's tests (#543): step 2a's rig
// (l2_bm_rig.hpp: the stand-in L3, the BlockMover in the KPU's domain, the L2 in its own) with the
// Streamer in the KPU's domain, a crossing each way between it and the L2 and one for the L2's
// tile-ready tokens, and a stand-in for the array: a sink that takes every lane's element every
// clock, as the array does, and records it.
//
// The rig registers check_l2_slot_credits as an invariant, as step 2a's does; the Streamer checks
// its lanes' credits every clock itself.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l2_bm_rig.hpp"

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::cycle::test {

// Takes every lane's element every clock, and records it with its clock.
class LaneSink : public Component {
public:
    struct Got {
        Cycle at;
        char side;
        L1Element e;
    };
    LaneSink(std::string name, Domain& d) : Component(std::move(name), d) {}
    void connect(std::vector<Link<L1Element>*> a, std::vector<Link<L1Element>*> b) {
        a_ = std::move(a);
        b_ = std::move(b);
    }
    void evaluate(Cycle now) override {
        for (Link<L1Element>* l : a_)
            while (l->can_pop(now)) got_.push_back({now, 'A', l->pop(now)});
        for (Link<L1Element>* l : b_)
            while (l->can_pop(now)) got_.push_back({now, 'B', l->pop(now)});
    }
    [[nodiscard]] bool idle() const override { return true; }
    [[nodiscard]] const std::vector<Got>& got() const { return got_; }
    // One lane's elements, in order.
    [[nodiscard]] std::vector<Got> lane(char side, std::uint32_t m) const {
        std::vector<Got> v;
        for (const Got& g : got_)
            if (g.side == side && g.e.lane == m) v.push_back(g);
        return v;
    }

private:
    std::vector<Link<L1Element>*> a_, b_;
    std::vector<Got> got_;
};

struct StreamerRig {
    struct Options {
        L2::Config l2{2, 16, 4, 16, 4, 2, 1};   // 16-byte rows: K up to 16 int8 elements
        std::uint32_t lanes = 4;                 // R = C
        std::uint32_t read_delay = 32;
        std::int32_t interval_adjust = 0;
        std::uint32_t bus_bytes = 16;
        double kpu_ghz = 1.0;
        double l2_ghz = 0.25;
        std::size_t streamer_link = 32;          // the Streamer's crossings' entries
        Tracer* tracer = nullptr;
    };

    Simulator sim;
    Domain& kpu;
    Domain& l2d;
    L3Standin& l3;
    BlockMover& bm;
    L2& l2;
    Streamer& st;
    LaneSink& sink;

    StreamerRig(const L2Program& program, const StreamerProgram& commands, const Options& o)
        : kpu(traced(sim, o.tracer).add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          l3(sim.add<L3Standin>("l3", kpu, o.bus_bytes)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes, 2, 8},
                                 program)),
          l2(sim.add<L2>("l2", l2d, o.l2, program)),
          st(sim.add<Streamer>("st", kpu,
                               Streamer::Config{o.lanes, o.lanes, o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, 1,
                                                o.read_delay, o.interval_adjust},
                               commands)),
          sink(sim.add<LaneSink>("array", kpu)) {
        check_feeds(commands, program);
        auto& l3_reads = sim.link<L3RowRead>("bm->l3 reads", bm, l3, 2);
        auto& l3_data = sim.link<Beat>("l3->bm", l3, bm, 2);
        auto& l3_writes = sim.link<Beat>("bm->l3 writes", bm, l3, 2);
        auto& l2_writes = sim.crossing<L2RowWrite>("bm->l2 writes", bm, l2, 4);
        auto& l2_reads = sim.crossing<L2RowRead>("bm->l2 reads", bm, l2, 4);
        auto& l2_data = sim.crossing<L2RowData>("l2->bm", l2, bm, 4);
        auto& credits = sim.crossing<SlotCredit>("l2 credits", l2, bm, o.l2.slots, 2, 2);
        auto& st_reads = sim.crossing<L2RowRead>("st->l2 reads", st, l2, o.streamer_link);
        auto& st_data = sim.crossing<L2RowData>("l2->st", l2, st, o.streamer_link);
        auto& ready = sim.crossing<SlotReady>("l2 tile-ready", l2, st, 4);
        std::vector<Link<L1Element>*> a, b;
        for (std::uint32_t m = 0; m < o.lanes; ++m) {
            a.push_back(&sim.link<L1Element>("A lane " + std::to_string(m), st, sink, 2));
            b.push_back(&sim.link<L1Element>("B lane " + std::to_string(m), st, sink, 2));
        }
        l3.connect(l3_reads, l3_data, l3_writes);
        bm.connect(l3_reads, l3_data, l3_writes, l2_writes, l2_reads, l2_data, credits);
        l2.connect(l2_writes, l2_reads, l2_data, credits);
        l2.connect_streamer(st_reads, st_data, ready);
        st.connect(st_reads, st_data, ready, a, b);
        sink.connect(a, b);
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, l2); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

}  // namespace sw::kpu::cycle::test
