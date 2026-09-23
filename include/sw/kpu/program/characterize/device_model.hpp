// ============================================================================
// include/sw/kpu/program/characterize/device_model.hpp
// A first-order device + cost model for characterizing L0 TilePrograms.
//
// L0 is timing-free; until the L1 stream layer and the driver-JIT placement pass
// exist (see docs/plans/kpu-program-model.md §4a), performance and energy are
// *modeled* from structural tile work using explicit, parameterized coefficients.
// The point is RELATIVE comparison across (algorithm, size, shape, HW config) — the
// design-of-experiments the harness drives — not absolute cycle counts. Every
// coefficient is a knob so experiments can sweep the hardware, not just the program.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <string>

namespace sw::kpu::program::characterize {

// Spatial arrangement of L3 memory tiles and compute-fabric (CF) tiles. This is
// coarse until the placement pass (§4a) lands; today it primarily sets the compute-
// tile count and a movement-efficiency factor.
enum class Topology { Single, NEWS, Checkerboard };

inline const char* to_string(Topology t) {
    switch (t) {
        case Topology::Single:       return "single";
        case Topology::NEWS:         return "news";
        case Topology::Checkerboard: return "checkerboard";
    }
    return "?";
}

// ============================================================================
// DeviceDescriptor — the hardware configuration an experiment targets.
// ============================================================================
struct DeviceDescriptor {
    Topology topology = Topology::Single;

    // Concurrency / capacity ---------------------------------------------------
    Dim compute_tiles = 1;      // # CF tiles that can run tile-compute ops concurrently
    // Aggregate movement concurrency for the ANALYTICAL harness only (lumped_duration,
    // TileDag). The executor does not use it: it schedules per process, above. Kept
    // because the first-order analytical model is a different tier with a different job,
    // not because movement is ever one pool.
    Dim move_lanes    = 1;
    // L3 capacity, counted IN TILES. 0 = unbounded, which is what a design-space sweep
    // wants and what preserves pre-capacity behaviour. Enforced dynamically by
    // TileTransactionExecutor as well as checked statically by the harness.
    // L2/L1 capacity is per COMPUTE TILE, so it waits for the placement pass to bind
    // tiles to compute tiles (#264 increment 5) — no field here until it is enforced.
    Dim l3_tiles      = 0;

    // Throughput ---------------------------------------------------------------
    double fabric_macs_per_cycle = 256.0;   // MAC throughput of ONE CF tile
    double bytes_per_cycle       = 64.0;    // analytical harness only (see move_lanes)
    double element_bytes         = 4.0;     // fp32

    // Movement, per CSP process (#264 increment 5, design note §6) ---------------
    // A tile does not cross the machine in one step, and IT CANNOT CROSS IT IN TWO
    // EITHER. Each leg is governed by its own CSP process over its own physical pathway:
    //
    //   DMA         DRAM <-> L3
    //   BlockMover  L3 <-> L2, and L3 -> L3 across the NoC (reuse)
    //   Streamer    L2 <-> L1
    //
    // The fabric reads only L1, and the L1 stream buffers push elements into it — that is
    // not a mover, so it owns no lanes and no bandwidth.
    //
    // THERE IS NO COLLAPSED MODE. L3->L2 and L2->L1 are distinct processes over distinct
    // pathways, so one stage standing for both describes a machine that cannot be built.
    // A span always contains all of its hops; residency changes only where a chain
    // STARTS (a tile already in L3 begins at the BlockMover), never which hops it has.
    // An earlier revision offered a two-pool descriptor with a collapsed single-pool
    // fallback. It was REMOVED rather than deprecated: a mode that models an unbuildable
    // machine has no valid use, so keeping it for compatibility would only preserve wrong
    // answers.
    //
    // Lanes belong to the process: inbound and outbound legs share one pool, because
    // there is one set of BlockMovers, not one per direction.
    Dim    dma_engines            = 1;
    double dma_bytes_per_cycle    = 64.0;    // per engine, DRAM <-> L3
    Dim    block_movers           = 1;
    double bm_bytes_per_cycle     = 128.0;   // per mover, L3 <-> L2
    Dim    streamers              = 1;
    double str_bytes_per_cycle    = 256.0;   // per streamer, L2 <-> L1
    Dim    noc_links              = 0;       // 0 = topology declares no L3<->L3 path
    double noc_bytes_per_cycle    = 128.0;   // per link, L3 -> L3 (reuse)

    // Per §6.3, every *_bytes_per_cycle above is PER LANE, never aggregate: a process's
    // peak throughput is lanes x bytes_per_cycle, one transfer occupies exactly one lane
    // for its whole duration, and lanes give concurrency, never speed-up. Left ambiguous,
    // the same descriptor would yield different makespans in different implementations and
    // calibration would mean nothing.


    // Energy (pJ), illustrative — movement >> compute is the headline principle ---
    double pj_per_mac                 = 1.0;
    double pj_per_byte                = 20.0;   // moving a byte costs ~20x a MAC
    double static_pj_per_tile_per_cyc = 5.0;    // leakage per active resource per cycle

    // Presets ------------------------------------------------------------------
    static DeviceDescriptor single() { return DeviceDescriptor{}; }

    static DeviceDescriptor news() {
        DeviceDescriptor d;
        d.topology = Topology::NEWS;
        d.compute_tiles = 1;
        d.move_lanes = 4;                       // four surrounding L3 tiles feed the CF
        return d;
    }

    static DeviceDescriptor checkerboard(Dim n) {
        DeviceDescriptor d;
        d.topology = Topology::Checkerboard;
        d.compute_tiles = n;
        d.move_lanes = n;                       // one mover per CF tile, roughly
        return d;
    }

    std::string label() const {
        return std::string(to_string(topology)) + "/cf" + std::to_string(compute_tiles) +
               "/dma" + std::to_string(std::max<Dim>(dma_engines, 1)) +
               "/bm" + std::to_string(std::max<Dim>(block_movers, 1)) +
               "/str" + std::to_string(std::max<Dim>(streamers, 1));
    }
};

} // namespace sw::kpu::program::characterize
