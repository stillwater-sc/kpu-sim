#pragma once
// ============================================================================
// tests/cycle/compute_rig.hpp
// The compute subsystem from L3 to L3, for step 2d's tests (#545): L3 -> BlockMover -> L2 ->
// Streamer -> L1 lanes -> PE array -> north FIFOs -> Combiner -> L2 -> BlockMover -> L3. Step 2c's
// path (array_rig.hpp) with the Combiner in the result sink's place, writing into the L2, and the
// Streamer holding the result slots' credits.
//
// The array is R x C (`lanes` x `cols`), the Streamer's A lanes R and its B lanes C, one north
// FIFO a column (step 2e, #546: kpu_s1's 128 x 64).
//
// The rig registers the three-way check_l2_slot_credits as an invariant: each slot's credit is the
// BlockMover's or the Streamer's, never both, and neither's while the slot is live.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l2_bm_rig.hpp"

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/combiner.hpp>
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

struct ComputeRig {
    struct Options {
        L2::Config l2{6, 16, 4, 16, 4, 2, 1};   // 16-byte rows: K up to 16 int8 a command, 4 int32 a result row
        std::uint32_t lanes = 4;                 // R: the array's rows
        std::uint32_t cols = 0;                  // C: its columns (0: R)
        std::uint32_t fifo_depth = 0;            // the north FIFOs' (0: C)
        bool log = true;                         // the array's MAC and launch log
        std::uint32_t read_delay = 32;
        std::int32_t drain_adjust = 0;
        std::uint32_t bus_bytes = 16;
        double kpu_ghz = 1.0;
        double l2_ghz = 0.25;
        std::size_t streamer_link = 32;
        std::size_t result_link = 16;           // the Combiner -> L2 crossing's entries
        Tracer* tracer = nullptr;
    };

    std::uint32_t cols;
    Simulator sim;
    Domain& kpu;
    Domain& l2d;
    L3Standin& l3;
    BlockMover& bm;
    L2& l2;
    Streamer& st;
    PEArray& array;
    Combiner& combiner;

    ComputeRig(const L2Program& program, const StreamerProgram& commands, const Options& o)
        : cols(o.cols ? o.cols : o.lanes),
          kpu(traced(sim, o.tracer).add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          l2d(sim.add_domain(Domain::from_ghz("l2", o.l2_ghz))),
          l3(sim.add<L3Standin>("l3", kpu, o.bus_bytes)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes, 2, 8},
                                 program)),
          l2(sim.add<L2>("l2", l2d, o.l2, program)),
          st(sim.add<Streamer>("st", kpu,
                               Streamer::Config{o.lanes, cols, o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, 1,
                                                o.read_delay, 0, o.drain_adjust},
                               commands, program)),
          array(sim.add<PEArray>("array", kpu, PEArray::Config{o.lanes, cols, 1, o.log})),
          combiner(sim.add<Combiner>("combiner", kpu, Combiner::Config{cols, o.fifo_depth ? o.fifo_depth : cols, o.l2.row_bytes}, commands)) {
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
        auto& st_credits = sim.crossing<SlotCredit>("l2 result credits", l2, st, o.l2.slots, 2, 2);
        auto& results = sim.crossing<L2RowWrite>("combiner->l2", combiner, l2, o.result_link);
        std::vector<Link<L1Element>*> a, b;
        std::vector<Link<ArrayResult>*> c;
        for (std::uint32_t m = 0; m < o.lanes; ++m)
            a.push_back(&sim.link<L1Element>("A lane " + std::to_string(m), st, array, 2));
        for (std::uint32_t m = 0; m < cols; ++m) {
            b.push_back(&sim.link<L1Element>("B lane " + std::to_string(m), st, array, 2));
            c.push_back(&sim.link<ArrayResult>("north " + std::to_string(m), array, combiner, 2));
        }
        l3.connect(l3_reads, l3_data, l3_writes);
        bm.connect(l3_reads, l3_data, l3_writes, l2_writes, l2_reads, l2_data, credits);
        l2.connect(l2_writes, l2_reads, l2_data, credits);
        l2.connect_streamer(st_reads, st_data, ready, &st_credits);
        l2.connect_combiner(results);
        st.connect(st_reads, st_data, ready, a, b, &st_credits);
        array.connect(a, b, c);
        combiner.connect(c, results);
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, st, l2); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

}  // namespace sw::kpu::cycle::test
