#pragma once
// ============================================================================
// tests/cycle/l3_dma_rig.hpp
// DRAM -> DMA engine -> L3 tile and back, for step 1b's tests and demo (#538): one LPDDR5 channel
// in its own 3.2 GHz domain; the engine and the tile in the KPU's domain (1 GHz by default), over a
// direct link; two crossings between the engine and the channel. The rig registers
// check_slot_credits as an invariant, so it holds every cycle of every run.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <lpddr5_configs.hpp>

#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sw::kpu::cycle::test {

// A tile's bytes, distinct per seed.
inline std::vector<std::uint8_t> tile_bytes(std::uint32_t seed, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(seed * 37 + i * 7 + (i >> 8));
    return v;
}

struct L3DmaRig {
    struct Options {
        std::uint32_t slots = 2;
        std::uint32_t slot_bytes = 4096;
        std::uint32_t banks = 4;
        std::size_t window = 8;
        std::size_t store_buffer_beats = 4;
        std::size_t l3_link = 2;      // the engine -> tile and tile -> engine data links' credits
        std::size_t dram_link = 4;    // the crossings' entries
        double kpu_ghz = 1.0;
        Tracer* tracer = nullptr;     // attached before anything is added
    };

    Simulator sim;
    Domain& dram;
    Domain& kpu;
    Lpddr5Channel& channel;
    DmaEngine& dma;
    L3Tile& l3;
    Link<DramRequest>& requests;
    Link<DramResponse>& responses;
    Link<L3Beat>& loads;
    Link<L3Beat>& ejects;
    Link<SlotCredit>& credits;

    L3DmaRig(const L3Program& program, const Options& o)
        : dram(traced(sim, o.tracer).add_domain(Domain::from_ghz("dram", 3.2))),
          kpu(sim.add_domain(Domain::from_ghz("kpu", o.kpu_ghz))),
          channel(sim.add<Lpddr5Channel>("lpddr5", dram, patterns::lpddr5::single_channel_config())),
          dma(sim.add<DmaEngine>("dma", kpu, DmaEngine::Config{o.slots, o.slot_bytes, o.window, o.store_buffer_beats},
                                 program)),
          l3(sim.add<L3Tile>("l3", kpu, L3Tile::Config{o.slots, o.slot_bytes, o.banks}, program)),
          requests(sim.crossing<DramRequest>("dma->lpddr5", dma, channel, o.dram_link)),
          responses(sim.crossing<DramResponse>("lpddr5->dma", channel, dma, o.dram_link)),
          loads(sim.link<L3Beat>("dma->l3", dma, l3, o.l3_link)),
          ejects(sim.link<L3Beat>("l3->dma", l3, dma, o.l3_link)),
          credits(sim.link<SlotCredit>("l3 credits", l3, dma, o.slots)) {
        channel.connect(requests, responses);
        dma.connect(requests, responses, loads, ejects, credits);
        l3.connect(loads, ejects, credits);
        sim.add_invariant("slot credits", [this] { check_slot_credits(dma, l3); });
    }

private:
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

}  // namespace sw::kpu::cycle::test
