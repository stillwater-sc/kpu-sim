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
    Dim move_lanes    = 1;      // COLLAPSED movement: one pool for every hop (see hops())
    // L3 capacity, counted IN TILES. 0 = unbounded, which is what a design-space sweep
    // wants and what preserves pre-capacity behaviour. Enforced dynamically by
    // TileTransactionExecutor as well as checked statically by the harness.
    // L2/L1 capacity is per COMPUTE TILE, so it waits for the placement pass to bind
    // tiles to compute tiles (#264 increment 5) — no field here until it is enforced.
    Dim l3_tiles      = 0;

    // Throughput ---------------------------------------------------------------
    double fabric_macs_per_cycle = 256.0;   // MAC throughput of ONE CF tile
    double bytes_per_cycle       = 64.0;    // ONE collapsed movement lane's bandwidth
    double element_bytes         = 4.0;     // fp32

    // Movement, per hop (#264 increment 5, design note §6) -----------------------
    // A tile does not cross the machine in one step: DRAM->L3 is realized by the DMA
    // against DRAM bandwidth, and L3->CF by the on-chip movers. Modelling them as one
    // aggregate pool cannot express the bottleneck that usually decides the makespan,
    // which is DRAM rather than on-chip movement.
    //
    // THE COLLAPSE IS A DESCRIPTOR SETTING, NOT A HARDCODED ASSUMPTION (§6). Leaving
    // `dram_lanes` and `onchip_lanes` at 0 yields ONE collapsed hop over `move_lanes`
    // and `bytes_per_cycle` — identical to pre-increment-5 behaviour, which is what
    // existing harness sweeps depend on. Setting either splits the chain in two.
    Dim    dram_lanes            = 0;       // 0 = collapsed (use move_lanes)
    double dram_bytes_per_cycle  = 64.0;    // per lane, DRAM<->L3
    Dim    onchip_lanes          = 0;       // 0 = collapsed (use move_lanes)
    double onchip_bytes_per_cycle = 256.0;  // per lane, L3<->CF; on-chip is the faster hop

    // Per §6.1, every *_bytes_per_cycle above is PER LANE, never aggregate: a hop's peak
    // throughput is lanes x bytes_per_cycle, one transfer occupies exactly one lane for
    // its whole duration, and lanes give concurrency, never speed-up. Left ambiguous, the
    // same descriptor would yield different makespans in different implementations and
    // calibration would mean nothing.
    bool per_hop_movement() const { return dram_lanes > 0 || onchip_lanes > 0; }

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
        if (per_hop_movement())
            return std::string(to_string(topology)) + "/cf" + std::to_string(compute_tiles) +
                   "/dram" + std::to_string(std::max<Dim>(dram_lanes, 1)) +
                   "/onchip" + std::to_string(std::max<Dim>(onchip_lanes, 1));
        return std::string(to_string(topology)) + "/cf" + std::to_string(compute_tiles) +
               "/ml" + std::to_string(move_lanes);
    }
};

} // namespace sw::kpu::program::characterize
