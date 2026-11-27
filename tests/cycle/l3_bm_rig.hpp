#pragma once
// ============================================================================
// tests/cycle/l3_bm_rig.hpp
// DRAM -> DMA -> L3 tile <-> BlockMover <-> L2, for step 3b's tests (#548, the join): the memory
// subsystem's L3 tile with the compute subsystem's BlockMover on its other side, in place of the
// stand-in L3. The DMA engine loads and stores; the BlockMover reads tiles into L2 and writes
// result tiles back, on its own links to the tile, under the L3 slots' credits it holds.
//
// The rig registers both credit checks as invariants: the L3 slots' three-way (the engine, the
// mover, the tile) and the L2 slots' (the mover, the L2).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <lpddr5_configs.hpp>

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <vector>

namespace sw::kpu::cycle::test {

struct L3BmRig {
    struct Options {
        std::uint32_t l3_slots = 4, l3_slot_bytes = 1024, l3_banks = 4;
        L2::Config l2{4, 16, 4, 64, 4, 2, 1};
        std::uint32_t bus_bytes = 64;   // the mover's beat: the L3 tile's
    };

    Simulator sim;
    Domain& dram;
    Domain& kpu;
    Domain& l2d;
    Lpddr5Channel& channel;
    DmaEngine& dma;
    L3Tile& l3;
    BlockMover& bm;
    L2& l2;

    L3BmRig(const L3Program& l3p, const L2Program& l2p, const Options& o)
        : dram(sim.add_domain(Domain::from_ghz("dram", 3.2))),
          kpu(sim.add_domain(Domain::from_ghz("kpu", 1.0))),
          l2d(sim.add_domain(Domain::from_ghz("l2", 0.25))),
          channel(sim.add<Lpddr5Channel>("lpddr5", dram, patterns::lpddr5::single_channel_config())),
          dma(sim.add<DmaEngine>("dma", kpu, DmaEngine::Config{o.l3_slots, o.l3_slot_bytes, 8, 4}, l3p)),
          l3(sim.add<L3Tile>("l3", kpu, L3Tile::Config{o.l3_slots, o.l3_slot_bytes, o.l3_banks}, l3p)),
          bm(sim.add<BlockMover>("bm", kpu,
                                 BlockMover::Config{o.l2.slots, o.l2.rows_per_slot, o.l2.row_bytes, o.bus_bytes, 2, 8}, l2p)),
          l2(sim.add<L2>("l2", l2d, o.l2, l2p)) {
        auto& requests = sim.crossing<DramRequest>("dma->lpddr5", dma, channel, 4);
        auto& responses = sim.crossing<DramResponse>("lpddr5->dma", channel, dma, 4);
        auto& loads = sim.link<L3Beat>("dma->l3", dma, l3, 2);
        auto& ejects = sim.link<L3Beat>("l3->dma", l3, dma, 2);
        auto& credits = sim.link<SlotCredit>("l3 credits", l3, dma, o.l3_slots);
        auto& bm_reads = sim.link<L3RowRead>("bm->l3 reads", bm, l3, 2);
        auto& bm_data = sim.link<Beat>("l3->bm", l3, bm, 2);
        auto& bm_writes = sim.link<Beat>("bm->l3 writes", bm, l3, 2);
        auto& bm_credits = sim.link<SlotCredit>("l3 credits to bm", l3, bm, o.l3_slots);
        auto& l2_writes = sim.crossing<L2RowWrite>("bm->l2 writes", bm, l2, 4);
        auto& l2_reads = sim.crossing<L2RowRead>("bm->l2 reads", bm, l2, 4);
        auto& l2_data = sim.crossing<L2RowData>("l2->bm", l2, bm, 4);
        auto& l2_credits = sim.crossing<SlotCredit>("l2 credits", l2, bm, o.l2.slots, 2, 2);
        channel.connect(requests, responses);
        dma.connect(requests, responses, loads, ejects, credits);
        l3.connect(loads, ejects, credits);
        l3.connect_blockmover(bm_reads, bm_data, bm_writes, bm_credits);
        bm.connect(bm_reads, bm_data, bm_writes, l2_writes, l2_reads, l2_data, l2_credits);
        std::vector<bool> held(o.l3_slots);
        for (std::uint32_t s = 0; s < o.l3_slots; ++s) held[s] = blockmover_holds_at_start(l3p, s);
        bm.connect_l3_credits(bm_credits, held);
        l2.connect(l2_writes, l2_reads, l2_data, l2_credits);
        sim.add_invariant("l3 slot credits", [this] { check_slot_credits(dma, bm, l3); });
        sim.add_invariant("l2 slot credits", [this] { check_l2_slot_credits(bm, l2); });
    }
};

}  // namespace sw::kpu::cycle::test
