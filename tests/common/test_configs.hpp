// ============================================================================
// tests/common/test_configs.hpp
// Right-sized KPUSimulator::Config factories for the regression suite.
//
// Why this exists
// ---------------
// Every test fixture used to inline its own Config with hand-picked sizes,
// almost always over-provisioned by 100x-1000x (e.g. 64-128 MB memory banks
// for tests that transfer at most ~128 KB). Multiplied across the parallel
// test cap (4) and a developer workstation that already has the IDE,
// browser, VS Code, and Node.js consuming 10-20 GB, the cumulative
// allocation reliably tripped std::bad_alloc in TEST_CASE_METHOD fixture
// constructors on Windows. Issue: #21.
//
// Memory budget
// -------------
// The goal: under a 4-way parallel test schedule on a 32 GB workstation
// with ~8-12 GB actually free, each per-test allocation should fit
// comfortably under ~100 MB and the total live allocation across the four
// parallel slots should never approach the contiguous-allocation ceiling
// the OS can satisfy.
//
//   minimal_config()  ~ 1 MB total   - metadata/wiring tests
//   small_config()    ~ 9 MB total   - typical data-moving tests (<=128 KB transfers)
//   medium_config()   ~ 36 MB total  - bigger tile / fabric tests
//
// All three return by value; copy the returned Config and override
// specific fields if a test legitimately needs a non-default detail
// (e.g. more DMA engines, a specific systolic-array topology).
//
// SPDX-License-Identifier: MIT
// ============================================================================

#pragma once

#include <sw/kpu/kpu_simulator.hpp>

namespace sw::kpu::test_utils {

// Smallest viable simulator.
// Use for tests that only verify wiring/metadata or transfer <=1 MB.
// Total external memory: ~1 MB. Total fixture allocation: ~1.1 MB.
inline KPUSimulator::Config minimal_config() {
    KPUSimulator::Config c;
    c.memory_bank_count         = 1;
    c.memory_bank_capacity_mb   = 1;
    c.memory_bandwidth_gbps     = 8;
    c.l3_layer.num_tiles        = 1;
    c.l3_layer.capacity_kb      = 64;
    c.l2_layer.num_banks        = 1;
    c.l2_layer.capacity_kb      = 32;
    c.l1_layer.num_buffers      = 1;
    c.l1_layer.capacity_kb      = 32;
    c.compute_tile_count        = 1;
    c.dma_engine_count          = 1;
    return c;
}

// Typical data-moving config.
// Use for DMA / streamer / block-mover unit tests with transfers up to ~128 KB.
// Total external memory: ~8 MB. Total fixture allocation: ~9 MB.
inline KPUSimulator::Config small_config() {
    KPUSimulator::Config c;
    c.memory_bank_count         = 2;
    c.memory_bank_capacity_mb   = 4;
    c.memory_bandwidth_gbps     = 8;
    c.l3_layer.num_tiles        = 4;
    c.l3_layer.capacity_kb      = 256;
    c.l2_layer.num_banks        = 4;
    c.l2_layer.capacity_kb      = 128;
    c.l1_layer.num_buffers      = 2;
    c.l1_layer.capacity_kb      = 64;
    c.compute_tile_count        = 2;
    c.dma_engine_count          = 4;
    return c;
}

// Larger working-set config.
// Use for fabric / systolic / multi-tile tests that need bigger tiles or
// multi-MB transfers. Prefer small_config() and override individual fields
// before reaching for this.
// Total external memory: ~32 MB. Total fixture allocation: ~36 MB.
inline KPUSimulator::Config medium_config() {
    KPUSimulator::Config c;
    c.memory_bank_count         = 2;
    c.memory_bank_capacity_mb   = 16;
    c.memory_bandwidth_gbps     = 16;
    c.l3_layer.num_tiles        = 4;
    c.l3_layer.capacity_kb      = 1024;
    c.l2_layer.num_banks        = 4;
    c.l2_layer.capacity_kb      = 512;
    c.l1_layer.num_buffers      = 4;
    c.l1_layer.capacity_kb      = 256;
    c.compute_tile_count        = 4;
    c.dma_engine_count          = 8;
    return c;
}

} // namespace sw::kpu::test_utils
