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
    Dim move_lanes    = 1;      // # concurrent movement channels (DMA/BM/Streamer aggregate)
    Dim l3_tiles      = 0;      // L3 capacity, in tiles (0 = unbounded → skip feasibility)

    // Throughput ---------------------------------------------------------------
    double fabric_macs_per_cycle = 256.0;   // MAC throughput of ONE CF tile
    double bytes_per_cycle       = 64.0;    // ONE movement lane's bandwidth
    double element_bytes         = 4.0;     // fp32

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
               "/ml" + std::to_string(move_lanes);
    }
};

} // namespace sw::kpu::program::characterize
