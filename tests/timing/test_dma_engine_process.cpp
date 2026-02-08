// ============================================================================
// tests/timing/test_dma_engine_process.cpp
// Unit tests for DMAEngineProcess with Memory Controller integration
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/dma_engine_process.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileDescriptor make_load_tile(MatrixID matrix, Size ti, Size tj, Size tk = 0,
                                      Size size_bytes = 1024) {
    TileDescriptor desc;
    desc.tile_id = {matrix, ti, tj, tk};
    desc.dram_address = 0x10000 + ti * 0x1000 + tj * 0x100;
    desc.size_bytes = size_bytes;
    return desc;
}

static MemoryControllerProcess::Config default_mc_config(uint32_t id = 0) {
    MemoryControllerProcess::Config config;
    config.controller_id = id;
    config.num_banks = 16;
    config.t_cl = 10;
    config.t_rcd = 15;
    config.t_rp = 15;
    config.t_burst = 4;
    config.startup_latency = 5;
    config.request_queue_depth = 32;
    config.name = config.display_name();
    return config;
}

static DMAEngineProcess::Config default_dma_config(uint32_t id = 0) {
    DMAEngineProcess::Config config;
    config.engine_id = id;
    config.queue_depth = 32;
    config.name = config.display_name();
    return config;
}

// Helper to count events of a specific type
static size_t count_events(const std::vector<TimingEvent>& events, EventType type) {
    size_t count = 0;
    for (const auto& e : events) {
        if (e.type == type) count++;
    }
    return count;
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("DMAEngineProcess construction", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    REQUIRE(dma.id() == 0);
    REQUIRE(dma.name() == "DMA0");
    REQUIRE(dma.is_idle());
    REQUIRE_FALSE(dma.has_pending_work());
    REQUIRE(dma.is_complete());
}

// ============================================================================
// Load Tests
// ============================================================================

TEST_CASE("DMAEngineProcess single load via MC", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule a load
    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 1024);
    dma.schedule_load(tile);

    REQUIRE(dma.has_pending_work());
    REQUIRE(dma.is_idle());  // Not yet submitted to MC

    // Tick DMA - should acquire credit and submit to MC
    auto dma_events = dma.tick(0);
    REQUIRE(count_events(dma_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(l3_credits.available() == 7);  // Credit acquired
    REQUIRE(mc.has_pending_work());  // Request submitted to MC

    // Tick both MC and DMA until complete
    Cycle cycle = 1;
    std::vector<TimingEvent> all_events;
    while ((!mc.is_complete() || !dma.is_complete()) && cycle < 100) {
        auto mc_events = mc.tick(cycle);
        all_events.insert(all_events.end(), mc_events.begin(), mc_events.end());
        dma_events = dma.tick(cycle);
        all_events.insert(all_events.end(), dma_events.begin(), dma_events.end());
        cycle++;
    }

    // Should have completed
    REQUIRE(dma.is_complete());
    REQUIRE(l3_tag_cam.lookup(tile.tile_id));  // Tile is in L3
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);
}

TEST_CASE("DMAEngineProcess load stalls without credit", "[timing][dma_process]") {
    CreditPool l3_credits(1);  // Only 1 credit
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule two loads
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));

    // First tick - first load acquires credit
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(l3_credits.available() == 0);
    REQUIRE(dma.has_pending_work());  // Second load still queued

    // Second tick - second load should stall
    events = dma.tick(1);
    REQUIRE(count_events(events, EventType::DMA_STALL_CREDIT) == 1);
    REQUIRE(dma.has_pending_work());  // Still queued
}

TEST_CASE("DMAEngineProcess multiple loads through MC", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule 4 loads
    for (Size i = 0; i < 4; ++i) {
        dma.schedule_load(make_load_tile(MatrixID::A, 0, i));
    }

    // First tick - should submit all to MC (all have credits)
    auto dma_events = dma.tick(0);
    REQUIRE(count_events(dma_events, EventType::CREDIT_ACQUIRED) == 4);
    REQUIRE(l3_credits.available() == 4);  // 4 credits acquired
    REQUIRE(mc.pending_requests() == 4);  // All 4 in MC queue
}

// ============================================================================
// Store Tests
// ============================================================================

TEST_CASE("DMAEngineProcess store requires tile in L3", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule a store without tile in L3
    auto tile = make_load_tile(MatrixID::C, 0, 0);
    dma.schedule_store(tile);

    // Tick - should stall (tile not in L3)
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_STALL_TAG) == 1);
    REQUIRE(dma.has_pending_work());  // Still queued
    REQUIRE(dma.is_idle());  // Nothing submitted to MC

    // Add tile to L3 TagCAM (simulating it arrived)
    l3_credits.acquire();  // Simulate credit used
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    // Now tick should submit to MC
    events = dma.tick(1);
    REQUIRE(mc.has_pending_work());  // Request submitted to MC
}

TEST_CASE("DMAEngineProcess store releases credit on completion", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Pre-populate L3 with a tile
    auto tile = make_load_tile(MatrixID::C, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);
    REQUIRE(l3_credits.available() == 7);

    // Schedule store
    dma.schedule_store(tile);

    // Run until complete
    Cycle cycle = 0;
    while ((!mc.is_complete() || !dma.is_complete()) && cycle < 100) {
        mc.tick(cycle);
        dma.tick(cycle);
        cycle++;
    }

    REQUIRE(dma.is_complete());
    REQUIRE(l3_credits.available() == 8);  // Credit released
    REQUIRE_FALSE(l3_tag_cam.lookup(tile.tile_id));  // Tile removed from L3
}

// ============================================================================
// Load/Store Mix Tests
// ============================================================================

TEST_CASE("DMAEngineProcess interleaved loads and stores", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule loads for A tiles
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));

    // Pre-populate C tile for store
    auto c_tile = make_load_tile(MatrixID::C, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(c_tile.tile_id, 0, 0);
    dma.schedule_store(c_tile);

    // First tick - should submit 2 loads and 1 store to MC
    auto dma_events = dma.tick(0);
    REQUIRE(count_events(dma_events, EventType::CREDIT_ACQUIRED) == 2);  // For loads
    REQUIRE(mc.pending_requests() == 3);  // 2 loads + 1 store
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("DMAEngineProcess reset clears state", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule and submit loads
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));
    dma.tick(0);

    REQUIRE(dma.pending_count() > 0);

    // Reset
    dma.reset();

    REQUIRE(dma.is_idle());
    REQUIRE(dma.is_complete());
    REQUIRE(dma.pending_count() == 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("DMAEngineProcess tracks statistics", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Complete a load
    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 2048);
    dma.schedule_load(tile);

    // Run until complete
    Cycle cycle = 0;
    while ((!mc.is_complete() || !dma.is_complete()) && cycle < 200) {
        mc.tick(cycle);
        dma.tick(cycle);
        cycle++;
    }

    REQUIRE(dma.total_bytes_loaded() == 2048);
    REQUIRE(dma.total_bytes_stored() == 0);
}

// ============================================================================
// Event Generation Tests
// ============================================================================

TEST_CASE("DMAEngineProcess generates correct events", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 512);
    dma.schedule_load(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while ((!mc.is_complete() || !dma.is_complete()) && cycle < 200) {
        auto mc_events = mc.tick(cycle);
        all_events.insert(all_events.end(), mc_events.begin(), mc_events.end());
        auto dma_events = dma.tick(cycle);
        all_events.insert(all_events.end(), dma_events.begin(), dma_events.end());
        cycle++;
    }

    // Should have: CREDIT_ACQUIRED, DMA_LOAD_START (from MC), DMA_LOAD_COMPLETE (from MC), TILE_ARRIVED_L3 (from DMA)
    REQUIRE(count_events(all_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);

    // Verify LOAD_COMPLETE has duration
    for (const auto& e : all_events) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) {
            REQUIRE(e.duration > 0);
        }
    }
}

// ============================================================================
// MC Command Bus Serialization Tests
// ============================================================================

TEST_CASE("DMAEngineProcess sees MC command bus serialization", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Schedule 2 loads to different banks
    auto tile0 = make_load_tile(MatrixID::A, 0, 0, 0, 1024);
    auto tile1 = make_load_tile(MatrixID::A, 0, 1, 0, 1024);
    tile1.dram_address = 0x00400;  // Different bank

    dma.schedule_load(tile0);
    dma.schedule_load(tile1);

    // Submit to MC
    dma.tick(0);
    REQUIRE(mc.pending_requests() == 2);

    // First MC tick - only ONE command issued
    auto mc_events = mc.tick(0);
    int load_starts_at_0 = count_events(mc_events, EventType::DMA_LOAD_START);
    REQUIRE(load_starts_at_0 == 1);

    // Second MC tick - other command issued
    mc_events = mc.tick(1);
    int load_starts_at_1 = count_events(mc_events, EventType::DMA_LOAD_START);
    REQUIRE(load_starts_at_1 == 1);

    // Total: 2 commands serialized over 2 cycles
    REQUIRE(mc.pending_requests() == 0);
}

// ============================================================================
// Tile Reuse Tests
// ============================================================================

TEST_CASE("DMAEngineProcess handles tile already in L3", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma(default_dma_config(0), mc, l3_credits, l3_tag_cam);

    // Pre-populate tile in L3
    auto tile = make_load_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    // Schedule load for same tile
    dma.schedule_load(tile);

    // Tick - should immediately complete (tile already present)
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(dma.is_complete());  // Completed immediately
    REQUIRE_FALSE(mc.has_pending_work());  // Not submitted to MC
}

// ============================================================================
// Multiple DMA Engines Sharing MC
// ============================================================================

TEST_CASE("Multiple DMA engines share MC", "[timing][dma_process]") {
    CreditPool l3_credits(16);
    TagCAM l3_tag_cam(16);
    MemoryControllerProcess mc(default_mc_config());

    DMAEngineProcess dma0(default_dma_config(0), mc, l3_credits, l3_tag_cam);
    DMAEngineProcess dma1(default_dma_config(1), mc, l3_credits, l3_tag_cam);

    // Each DMA schedules a load
    dma0.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma1.schedule_load(make_load_tile(MatrixID::B, 0, 0));

    // Both submit to same MC
    dma0.tick(0);
    dma1.tick(0);

    REQUIRE(mc.pending_requests() == 2);

    // Run until both complete
    Cycle cycle = 1;
    while ((!mc.is_complete() || !dma0.is_complete() || !dma1.is_complete()) && cycle < 100) {
        mc.tick(cycle);
        dma0.tick(cycle);
        dma1.tick(cycle);
        cycle++;
    }

    REQUIRE(dma0.is_complete());
    REQUIRE(dma1.is_complete());
    REQUIRE(dma0.total_bytes_loaded() == 1024);
    REQUIRE(dma1.total_bytes_loaded() == 1024);
}
