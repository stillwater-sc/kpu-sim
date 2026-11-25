// ============================================================================
// tests/cycle/cycle_l3_dma_demo.cpp
// DRAM -> DMA engine -> L3 tile and back on the cycle core (#538): its traces and counters, for
// the validators.
//
//   cycle_l3_dma_demo <dram-trace.json> <core-trace.json> <counters.json>
//
// Rung 2 of the memory subsystem's ladder: eight 4 KiB tiles through an L3 tile's two slots, each
// loaded from DRAM, stored back, and released for the next. The data is checked: every tile in
// its slot at the cycle it is ready, and every stored tile in DRAM. Writes:
//   - the LPDDR5 controller's trace, for patterns/memory/lpddr5/common/trace_validator.py;
//   - the cycle core's trace and counters, for check_trace.py (open either in ui.perfetto.dev).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l3_dma_rig.hpp"

#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstdio>
#include <exception>
#include <map>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::fprintf(stderr, "usage: cycle_l3_dma_demo <dram-trace.json> <core-trace.json> <counters.json>\n");
        return 2;
    }
    constexpr std::uint32_t kTiles = 8, kBytes = 4096, kSlots = 2;
    L3Program p;
    for (std::uint32_t t = 0; t < kTiles; ++t) {
        p.push_back(l3_load(t % kSlots, 0x10000 + std::uint64_t(t) * kBytes, kBytes));
        p.push_back(l3_store(t % kSlots, 0x400000 + std::uint64_t(t) * kBytes, kBytes));
        p.push_back(l3_release(t % kSlots));
    }
    Tracer tracer;
    L3DmaRig::Options o;
    o.slots = kSlots;
    o.slot_bytes = kBytes;
    o.tracer = &tracer;
    L3DmaRig r(p, o);
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i].kind == L3Action::Kind::Load)
            r.channel.preload(p[i].dram_address, tile_bytes(static_cast<std::uint32_t>(i), kBytes));
    std::map<std::uint64_t, bool> good;   // by Load ticket: its tile was right when ready
    r.sim.add_invariant("watch", [&] {
        for (std::uint32_t s = 0; s < kSlots; ++s)
            if (r.l3.ready(s) && !good.count(r.l3.ticket(s)))
                good[r.l3.ticket(s)] = r.l3.peek(s) == tile_bytes(static_cast<std::uint32_t>(r.l3.ticket(s)), kBytes);
    });
    r.channel.enable_dram_trace(true);

    if (!r.sim.run_until_idle(1'000'000'000)) {
        std::fprintf(stderr, "cycle_l3_dma_demo: the run did not finish\n");
        return 1;
    }
    for (std::uint32_t t = 0; t < kTiles; ++t) {
        const std::uint64_t load = 3 * t;
        if (!good.count(load) || !good[load]) {
            std::fprintf(stderr, "cycle_l3_dma_demo: tile %u was not in its slot, bit for bit\n", t);
            return 1;
        }
        if (r.channel.peek(p[3 * t + 1].dram_address, kBytes) != tile_bytes(static_cast<std::uint32_t>(load), kBytes)) {
            std::fprintf(stderr, "cycle_l3_dma_demo: tile %u was not stored back, bit for bit\n", t);
            return 1;
        }
    }
    if (!r.channel.write_dram_trace(argv[1])) {
        std::fprintf(stderr, "cycle_l3_dma_demo: cannot write %s\n", argv[1]);
        return 1;
    }
    tracer.write(argv[2]);
    write_counters_json(r.sim, argv[3]);
    const DmaCounts& d = r.dma.dma();
    std::printf("cycle_l3_dma_demo: %u tiles of %u bytes through %u slots, bit for bit; %llu read and %llu write "
                "bursts, at most %llu in flight; %.1f%% of the data bus\n",
                kTiles, kBytes, kSlots, static_cast<unsigned long long>(d.read_bursts),
                static_cast<unsigned long long>(d.write_bursts), static_cast<unsigned long long>(d.max_in_flight),
                100.0 * r.channel.dram().data_bus_utilization());
    std::printf("%s", format_report(report(r.sim)).c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cycle_l3_dma_demo: %s\n", e.what());
    return 1;
}
