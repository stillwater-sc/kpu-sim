#pragma once
// ============================================================================
// tests/cycle/array_rig.hpp
// L3 -> BlockMover -> L2 -> Streamer -> L1 lanes -> PE array, for step 2c's tests (#544): step 2b's
// path (streamer_rig.hpp) with the array in the stand-in sink's place, and a stand-in for the north
// result buffers (step 2d): a sink that takes each column's result every clock and records it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l2_bm_rig.hpp"

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/pe_array.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/component.hpp>
#include <sw/kpu/cycle/link.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::cycle::test {

// Takes each column's result every clock, and records it with its clock.
class ResultSink : public Component {
public:
    struct Got {
        Cycle at;
        ArrayResult r;
    };
    ResultSink(std::string name, Domain& d) : Component(std::move(name), d) {}
    void connect(std::vector<Link<ArrayResult>*> in) { in_ = std::move(in); }
    void evaluate(Cycle now) override {
        for (Link<ArrayResult>* l : in_)
            while (l->can_pop(now)) got_.push_back({now, l->pop(now)});
    }
    [[nodiscard]] bool idle() const override { return true; }
    [[nodiscard]] const std::vector<Got>& got() const { return got_; }

private:
    std::vector<Link<ArrayResult>*> in_;
    std::vector<Got> got_;
};

struct ArrayRig {
    struct Options {
        L2::Config l2{2, 16, 4, 16, 4, 2, 1};   // 16-byte rows: K up to 16 int8 elements a command
        std::uint32_t lanes = 4;                 // R = C
        std::uint32_t read_delay = 32;
        std::uint32_t bus_bytes = 16;
        double kpu_ghz = 1.0;
        double l2_ghz = 0.25;
        std::size_t streamer_link = 32;
        Tracer* tracer = nullptr;
    };

    Simulator sim;
    Domain& kpu;
    Domain& l2d;
    L3Standin& l3;
    BlockMover& bm;
    L2& l2;
    Streamer& st;
    PEArray& array;
    ResultSink& north;

    ArrayRig(const L2Program& program, const StreamerProgram& commands, const Options& o)
        : kpu(traced(sim, o.tracer).add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          l3(sim.add<L3Standin>("l3", kpu, o.bus_bytes)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes, 2, 8},
                                 program)),
          l2(sim.add<L2>("l2", l2d, o.l2, program)),
          st(sim.add<Streamer>("st", kpu,
                               Streamer::Config{o.lanes, o.lanes, o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, 1,
                                                o.read_delay, 0},
                               commands)),
          array(sim.add<PEArray>("array", kpu, PEArray::Config{o.lanes, o.lanes, 1, true})),
          north(sim.add<ResultSink>("north", kpu)) {
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
        std::vector<Link<ArrayResult>*> c;
        for (std::uint32_t m = 0; m < o.lanes; ++m) {
            a.push_back(&sim.link<L1Element>("A lane " + std::to_string(m), st, array, 2));
            b.push_back(&sim.link<L1Element>("B lane " + std::to_string(m), st, array, 2));
            c.push_back(&sim.link<ArrayResult>("north " + std::to_string(m), array, north, 2));
        }
        l3.connect(l3_reads, l3_data, l3_writes);
        bm.connect(l3_reads, l3_data, l3_writes, l2_writes, l2_reads, l2_data, credits);
        l2.connect(l2_writes, l2_reads, l2_data, credits);
        l2.connect_streamer(st_reads, st_data, ready);
        st.connect(st_reads, st_data, ready, a, b);
        array.connect(a, b, c);
        north.connect(c);
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, l2); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

}  // namespace sw::kpu::cycle::test
