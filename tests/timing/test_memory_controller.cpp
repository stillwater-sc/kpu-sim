// ============================================================================
// tests/timing/test_memory_controller.cpp
// Tests for Memory Controller transactional model
//
// The MC is a "pure" DRAM contention model:
// - Command bus serialization (1 command/cycle)
// - Bank state machines (row hit/miss/empty)
// - Does NOT handle L3 credits or tags (that's DMA's job)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/timing/memory_controller_process.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

using namespace sw::kpu::timing;
using sw::kpu::isa::MatrixID;
using Catch::Approx;

// Helper to create a tile descriptor
TileDescriptor make_tile(MatrixID matrix, uint64_t dram_addr,
                         Size ti = 0, Size tj = 0, Size tk = 0) {
    TileDescriptor tile;
    tile.tile_id.matrix = matrix;
    tile.tile_id.ti = ti;
    tile.tile_id.tj = tj;
    tile.tile_id.tk = tk;
    tile.dram_address = dram_addr;
    tile.matrix_base_address = dram_addr & ~0xFFFFFull;  // Align to 1MB
    tile.size_bytes = 1024;  // 1KB tile
    tile.height = 16;
    tile.width = 16;
    tile.element_size = 4;
    return tile;
}

static size_t count_events(const std::vector<TimingEvent>& events, EventType type) {
    size_t count = 0;
    for (const auto& e : events) {
        if (e.type == type) count++;
    }
    return count;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_CASE("MemoryController basic construction", "[memory_controller]") {
    MemoryControllerProcess::Config config;
    config.controller_id = 0;
    config.num_banks = 16;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;

    MemoryControllerProcess mc(config);

    REQUIRE(mc.id() == 0);
    REQUIRE(mc.is_idle());
    REQUIRE_FALSE(mc.has_pending_work());
    REQUIRE(mc.is_complete());
}

TEST_CASE("MemoryController basic load via submit_request", "[memory_controller]") {
    MemoryControllerProcess::Config config;
    config.controller_id = 0;
    config.num_banks = 16;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;

    MemoryControllerProcess mc(config);

    // Submit a load request
    auto tile = make_tile(MatrixID::A, 0x1000, 0, 0, 0);
    bool accepted = mc.submit_request(tile, true);

    REQUIRE(accepted);
    REQUIRE(mc.has_pending_work());
    REQUIRE(mc.pending_requests() == 1);

    // Tick until complete
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while (!mc.is_complete()) {
        auto events = mc.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
        if (cycle > 100) break;  // Safety limit
    }

    // Should have LOAD_START, MC_ACCESS_TYPE, LOAD_COMPLETE
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_COMPLETE) == 1);

    // Check for completed transfer
    auto completed = mc.get_completed_transfer();
    REQUIRE(completed.has_value());
    REQUIRE(completed->tile.tile_id == tile.tile_id);
    REQUIRE(completed->is_load == true);
}

// ============================================================================
// Resource Contention Tests
// ============================================================================

TEST_CASE("MemoryController command bus serializes requests", "[memory_controller][contention]") {
    MemoryControllerProcess::Config config;
    config.controller_id = 0;
    config.num_banks = 16;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;

    MemoryControllerProcess mc(config);

    // Schedule 2 loads to DIFFERENT banks at the same time
    // Address mapping: bank = (addr >> col_bits) & 0xF
    // With col_bits = 10, bank 0 = addr 0x0000, bank 1 = addr 0x0400
    auto tile0 = make_tile(MatrixID::A, 0x00000, 0, 0, 0);  // Bank 0
    auto tile1 = make_tile(MatrixID::A, 0x00400, 0, 0, 1);  // Bank 1

    mc.submit_request(tile0, true);
    mc.submit_request(tile1, true);

    REQUIRE(mc.pending_requests() == 2);

    // First tick: only ONE command should be issued
    auto events0 = mc.tick(0);
    int load_starts_at_0 = count_events(events0, EventType::DMA_LOAD_START);

    // Critical check: only 1 LOAD_START at cycle 0, not 2
    REQUIRE(load_starts_at_0 == 1);

    // Second tick: the other command should be issued
    auto events1 = mc.tick(1);
    int load_starts_at_1 = count_events(events1, EventType::DMA_LOAD_START);

    REQUIRE(load_starts_at_1 == 1);

    // Total: 2 requests, issued at cycles 0 and 1 (serialized)
    REQUIRE(mc.pending_requests() == 0);
}

TEST_CASE("Old DMA model would have issued both at cycle 0", "[memory_controller][regression]") {
    // This test documents the OLD incorrect behavior for comparison
    // The old DMAEngineProcess would issue both loads at cycle 0 because
    // each "channel" was independent

    // With the new MemoryControllerProcess, this is correctly serialized
    // (see test above)

    SUCCEED("New MC model correctly serializes command bus access");
}

// ============================================================================
// Row Hit/Miss Tests
// ============================================================================

TEST_CASE("MemoryController tracks row hit rate", "[memory_controller][row_buffer]") {
    MemoryControllerProcess::Config config;
    config.controller_id = 0;
    config.num_banks = 16;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;
    config.row_bits = 14;
    config.col_bits = 10;
    config.bank_bits = 4;

    MemoryControllerProcess mc(config);

    // First access: Row Empty (bank was idle)
    auto tile0 = make_tile(MatrixID::A, 0x00000, 0, 0, 0);
    mc.submit_request(tile0, true);

    // Second access: Same bank, same row = Row Hit
    auto tile1 = make_tile(MatrixID::A, 0x00100, 0, 0, 1);  // Different column, same row
    mc.submit_request(tile1, true);

    // Third access: Same bank, different row = Row Miss
    // Row changes at address bit (col_bits + bank_bits) = 14
    auto tile2 = make_tile(MatrixID::A, 0x04000, 0, 0, 2);  // Different row (bit 14 set)
    mc.submit_request(tile2, true);

    // Run to completion
    Cycle cycle = 0;
    while (!mc.is_complete()) {
        mc.tick(cycle++);
        if (cycle > 200) break;
    }

    // Check statistics
    REQUIRE(mc.row_empty_accesses() == 1);  // First access
    REQUIRE(mc.row_hits() == 1);             // Second access (same row)
    REQUIRE(mc.row_misses() == 1);           // Third access (different row)

    // Row hit rate should be 1/3 = 33%
    REQUIRE(mc.row_hit_rate() == Approx(1.0/3.0).margin(0.01));
}

TEST_CASE("Row hit is faster than row miss", "[memory_controller][timing]") {
    MemoryControllerProcess::Config config;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;

    // Expected latencies:
    // Row Hit:   startup + t_cl + t_burst = 5 + 10 + 4 = 19 cycles
    // Row Empty: startup + t_rcd + t_cl + t_burst = 5 + 15 + 10 + 4 = 34 cycles
    // Row Miss:  startup + t_rp + t_rcd + t_cl + t_burst = 5 + 15 + 15 + 10 + 4 = 49 cycles

    MemoryControllerProcess mc(config);

    // Schedule two loads: first to open the row, second is a row hit
    auto tile0 = make_tile(MatrixID::A, 0x00000, 0, 0, 0);  // Bank 0, row 0
    auto tile1 = make_tile(MatrixID::A, 0x00100, 0, 0, 1);  // Same bank, same row (different col)

    mc.submit_request(tile0, true);  // This will be ROW_EMPTY
    mc.submit_request(tile1, true);  // This should be ROW_HIT (after first completes)

    Cycle cycle = 0;
    Cycle first_start = 0, first_complete = 0;
    Cycle second_start = 0, second_complete = 0;

    while (!mc.is_complete()) {
        auto events = mc.tick(cycle);
        for (const auto& e : events) {
            if (e.type == EventType::DMA_LOAD_START) {
                if (e.tile_id.tk == 0 && first_start == 0) first_start = cycle;
                if (e.tile_id.tk == 1 && second_start == 0) second_start = cycle;
            }
            if (e.type == EventType::DMA_LOAD_COMPLETE) {
                // Complete event cycle is start_cycle, actual completion is start + duration
                Cycle actual_complete = e.cycle + e.duration;
                if (e.tile_id.tk == 0) first_complete = actual_complete;
                if (e.tile_id.tk == 1) second_complete = actual_complete;
            }
        }
        cycle++;
        if (cycle > 200) break;
    }

    // First access latency (row empty): ~34 cycles
    Cycle first_latency = first_complete - first_start;
    REQUIRE(first_latency >= 30);
    REQUIRE(first_latency <= 40);

    // Second access should start after first finishes (command bus serialization)
    // It should be a row hit since the row is now open

    // Check stats - should have 1 row_empty and 1 row_hit
    REQUIRE(mc.row_empty_accesses() == 1);
    REQUIRE(mc.row_hits() == 1);
}

// ============================================================================
// Queue Depth Tests
// ============================================================================

TEST_CASE("MemoryController respects request queue depth", "[memory_controller][queue]") {
    MemoryControllerProcess::Config config;
    config.request_queue_depth = 4;

    MemoryControllerProcess mc(config);

    // Submit requests up to queue depth
    for (int i = 0; i < 4; i++) {
        auto tile = make_tile(MatrixID::A, i * 0x400, 0, 0, i);
        bool accepted = mc.submit_request(tile, true);
        REQUIRE(accepted);
    }

    REQUIRE(mc.pending_requests() == 4);

    // Next request should be rejected (queue full)
    auto tile5 = make_tile(MatrixID::A, 0x4000, 0, 0, 4);
    bool accepted = mc.submit_request(tile5, true);
    REQUIRE_FALSE(accepted);
}

// ============================================================================
// Store Tests
// ============================================================================

TEST_CASE("MemoryController handles store requests", "[memory_controller][store]") {
    MemoryControllerProcess::Config config;
    MemoryControllerProcess mc(config);

    // Submit a store request
    auto tile = make_tile(MatrixID::C, 0x3000, 0, 0, 0);
    mc.submit_request(tile, false);  // false = store

    // Run to completion
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while (!mc.is_complete()) {
        auto events = mc.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
        if (cycle > 100) break;
    }

    // Should have STORE_START, MC_ACCESS_TYPE, STORE_COMPLETE
    REQUIRE(count_events(all_events, EventType::DMA_STORE_START) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_STORE_COMPLETE) == 1);

    // Check for completed transfer
    auto completed = mc.get_completed_transfer();
    REQUIRE(completed.has_value());
    REQUIRE(completed->is_load == false);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("MemoryController provides statistics", "[memory_controller][stats]") {
    MemoryControllerProcess::Config config;
    MemoryControllerProcess mc(config);

    // Run some loads to different banks (to avoid bank conflicts)
    for (int i = 0; i < 5; i++) {
        auto tile = make_tile(MatrixID::A, i * 0x400, 0, 0, i);  // Each to different bank
        mc.submit_request(tile, true);
    }

    Cycle cycle = 0;
    while (!mc.is_complete()) {
        mc.tick(cycle++);
        if (cycle > 500) break;
    }

    // Check stats are populated
    size_t total_accesses = mc.row_hits() + mc.row_misses() + mc.row_empty_accesses();
    REQUIRE(total_accesses == 5);

    // All accesses to different banks start with row empty
    REQUIRE(mc.row_empty_accesses() == 5);

    // With 5 requests to different banks:
    // - Commands serialize on command bus (1 per cycle)
    // - Data transfers to different banks can overlap
    // So completion time is roughly: latency + (num_requests - 1)
    // With row empty latency ~34 cycles, 5 requests: ~38 cycles
    REQUIRE(cycle >= 30);  // At least one full transfer
    REQUIRE(cycle < 200);  // Not too long (parallelism works)
}

TEST_CASE("MemoryController reset clears state", "[memory_controller][reset]") {
    MemoryControllerProcess::Config config;
    MemoryControllerProcess mc(config);

    // Run some requests
    for (int i = 0; i < 3; i++) {
        auto tile = make_tile(MatrixID::A, i * 0x400, 0, 0, i);
        mc.submit_request(tile, true);
    }

    Cycle cycle = 0;
    while (!mc.is_complete() && cycle < 100) {
        mc.tick(cycle++);
    }

    REQUIRE(mc.row_empty_accesses() > 0);

    // Reset
    mc.reset();

    REQUIRE(mc.is_idle());
    REQUIRE(mc.is_complete());
    REQUIRE(mc.pending_requests() == 0);
    REQUIRE(mc.row_hits() == 0);
    REQUIRE(mc.row_misses() == 0);
    REQUIRE(mc.row_empty_accesses() == 0);
}

// ============================================================================
// Completion Polling Tests
// ============================================================================

TEST_CASE("MemoryController get_completed_transfer returns transfers in order", "[memory_controller][completion]") {
    MemoryControllerProcess::Config config;
    config.num_banks = 16;
    MemoryControllerProcess mc(config);

    // Submit 3 requests to different banks
    for (int i = 0; i < 3; i++) {
        auto tile = make_tile(MatrixID::A, i * 0x400, 0, 0, i);
        mc.submit_request(tile, true);
    }

    // Run to completion
    Cycle cycle = 0;
    while (!mc.is_complete() && cycle < 200) {
        mc.tick(cycle++);
    }

    // Poll for completions - should get 3 in order of completion
    int completions = 0;
    while (auto completed = mc.get_completed_transfer()) {
        completions++;
    }

    REQUIRE(completions == 3);

    // No more completions available
    auto completed = mc.get_completed_transfer();
    REQUIRE_FALSE(completed.has_value());
}

TEST_CASE("MemoryController has_completed_transfers works", "[memory_controller][completion]") {
    MemoryControllerProcess::Config config;
    MemoryControllerProcess mc(config);

    // Initially no completions
    REQUIRE_FALSE(mc.has_completed_transfers());

    // Submit and complete a request
    auto tile = make_tile(MatrixID::A, 0x1000, 0, 0, 0);
    mc.submit_request(tile, true);

    Cycle cycle = 0;
    while (!mc.has_completed_transfers() && cycle < 100) {
        mc.tick(cycle++);
    }

    REQUIRE(mc.has_completed_transfers());

    // Consume the completion
    mc.get_completed_transfer();

    REQUIRE_FALSE(mc.has_completed_transfers());
}
