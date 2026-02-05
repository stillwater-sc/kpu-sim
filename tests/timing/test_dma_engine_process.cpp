// ============================================================================
// tests/timing/test_dma_engine_process.cpp
// Unit tests for DMAEngineProcess
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/dma_engine_process.hpp>

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

static DMAEngineProcess::Config default_config(uint32_t id = 0) {
    DMAEngineProcess::Config config;
    config.engine_id = id;
    config.bus_width_bytes = 64;
    config.startup_latency = 10;
    config.bandwidth_gbps = 25.6;
    config.clock_ghz = 1.0;
    config.queue_depth = 4;
    config.name = "DMA";
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

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    REQUIRE(dma.id() == 0);
    REQUIRE(dma.name() == "DMA_0");
    REQUIRE(dma.is_idle());
    REQUIRE_FALSE(dma.has_pending_work());
    REQUIRE(dma.is_complete());
}

// ============================================================================
// Load Tests
// ============================================================================

TEST_CASE("DMAEngineProcess single load", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    auto config = default_config(0);

    DMAEngineProcess dma(config, l3_credits, l3_tag_cam);

    // Schedule a load
    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 1024);
    dma.schedule_load(tile);

    REQUIRE(dma.has_pending_work());
    REQUIRE(dma.is_idle());  // Not yet started

    // First tick should start the load
    auto events = dma.tick(0);

    REQUIRE(count_events(events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE_FALSE(dma.is_idle());  // Now busy
    REQUIRE_FALSE(dma.has_pending_work());  // Removed from queue
    REQUIRE(l3_credits.available() == 7);  // Credit acquired

    // Tick until complete (startup + transfer time)
    // 1024 bytes at 25.6 GB/s @ 1GHz = ~40 cycles + 10 startup = ~50 cycles
    Cycle cycle = 1;
    while (!dma.is_idle() && cycle < 100) {
        events = dma.tick(cycle++);
    }

    // Should have completed
    REQUIRE(dma.is_idle());
    REQUIRE(dma.is_complete());
    REQUIRE(l3_tag_cam.lookup(tile.tile_id));  // Tile is in L3
}

TEST_CASE("DMAEngineProcess load stalls without credit", "[timing][dma_process]") {
    CreditPool l3_credits(1);  // Only 1 credit
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    // Schedule two loads
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));

    // First tick - first load starts, acquires only credit
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(l3_credits.available() == 0);
    REQUIRE(dma.has_pending_work());  // Second load still queued

    // Second tick - second load should stall
    events = dma.tick(1);
    REQUIRE(count_events(events, EventType::DMA_STALL_CREDIT) == 1);
    REQUIRE(dma.has_pending_work());  // Still queued
}

TEST_CASE("DMAEngineProcess multiple concurrent loads", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    auto config = default_config(0);
    config.queue_depth = 4;  // Allow 4 in-flight

    DMAEngineProcess dma(config, l3_credits, l3_tag_cam);

    // Schedule 4 loads
    for (Size i = 0; i < 4; ++i) {
        dma.schedule_load(make_load_tile(MatrixID::A, 0, i));
    }

    // First tick should start all 4
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_LOAD_START) == 4);
    REQUIRE(dma.in_flight_count() == 4);
    REQUIRE(l3_credits.available() == 4);  // 4 credits acquired
}

TEST_CASE("DMAEngineProcess respects queue depth", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    auto config = default_config(0);
    config.queue_depth = 2;  // Only 2 in-flight allowed

    DMAEngineProcess dma(config, l3_credits, l3_tag_cam);

    // Schedule 4 loads
    for (Size i = 0; i < 4; ++i) {
        dma.schedule_load(make_load_tile(MatrixID::A, 0, i));
    }

    // First tick should start only 2
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_LOAD_START) == 2);
    REQUIRE(dma.in_flight_count() == 2);
    REQUIRE(dma.has_pending_work());  // 2 more queued
}

// ============================================================================
// Store Tests
// ============================================================================

TEST_CASE("DMAEngineProcess store requires tile in L3", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    // Schedule a store without tile in L3
    auto tile = make_load_tile(MatrixID::C, 0, 0);
    dma.schedule_store(tile);

    // Tick - should stall (tile not in L3)
    auto events = dma.tick(0);
    REQUIRE(dma.has_pending_work());  // Still queued
    REQUIRE(dma.is_idle());  // Nothing in flight

    // Add tile to L3 TagCAM (simulating it arrived)
    l3_credits.acquire();  // Simulate credit used
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    // Now tick should start the store
    events = dma.tick(1);
    REQUIRE(count_events(events, EventType::DMA_STORE_START) == 1);
    REQUIRE_FALSE(l3_tag_cam.lookup(tile.tile_id));  // Tile removed from L3
    REQUIRE(l3_credits.available() == 8);  // Credit released
}

TEST_CASE("DMAEngineProcess store releases credit", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    // Pre-populate L3 with a tile
    auto tile = make_load_tile(MatrixID::C, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);
    REQUIRE(l3_credits.available() == 7);

    // Schedule store
    dma.schedule_store(tile);

    // Start store - should release credit
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_STORE_START) == 1);
    REQUIRE(count_events(events, EventType::CREDIT_RELEASED) == 1);
    REQUIRE(l3_credits.available() == 8);  // Credit released immediately
}

// ============================================================================
// Load/Store Mix Tests
// ============================================================================

TEST_CASE("DMAEngineProcess interleaved loads and stores", "[timing][dma_process]") {
    CreditPool l3_credits(4);
    TagCAM l3_tag_cam(4);
    auto config = default_config(0);
    config.queue_depth = 4;

    DMAEngineProcess dma(config, l3_credits, l3_tag_cam);

    // Schedule loads for A tiles
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));

    // Pre-populate C tiles for store
    auto c_tile = make_load_tile(MatrixID::C, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(c_tile.tile_id, 0, 0);
    dma.schedule_store(c_tile);

    // First tick - should start 2 loads and 1 store
    auto events = dma.tick(0);
    REQUIRE(count_events(events, EventType::DMA_LOAD_START) == 2);
    REQUIRE(count_events(events, EventType::DMA_STORE_START) == 1);
    REQUIRE(dma.in_flight_count() == 3);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("DMAEngineProcess reset clears state", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    // Schedule and start some loads
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 0));
    dma.schedule_load(make_load_tile(MatrixID::A, 0, 1));
    dma.tick(0);

    REQUIRE_FALSE(dma.is_idle());
    REQUIRE(dma.in_flight_count() > 0);

    // Reset
    dma.reset();

    REQUIRE(dma.is_idle());
    REQUIRE(dma.is_complete());
    REQUIRE(dma.in_flight_count() == 0);
    REQUIRE(dma.load_queue_depth() == 0);
    REQUIRE(dma.store_queue_depth() == 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("DMAEngineProcess tracks statistics", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    // Complete a load
    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 2048);
    dma.schedule_load(tile);

    // Run until complete
    Cycle cycle = 0;
    while (!dma.is_complete() && cycle < 200) {
        dma.tick(cycle++);
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

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    auto tile = make_load_tile(MatrixID::A, 0, 0, 0, 512);
    dma.schedule_load(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while (!dma.is_complete() && cycle < 200) {
        auto events = dma.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }

    // Should have: LOAD_START, CREDIT_ACQUIRED, LOAD_COMPLETE, TILE_ARRIVED_L3
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);

    // Verify LOAD_COMPLETE has duration
    for (const auto& e : all_events) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) {
            REQUIRE(e.duration > 0);
        }
    }
}

TEST_CASE("DMAEngineProcess events have correct tile IDs", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    auto tile = make_load_tile(MatrixID::B, 2, 3, 1);
    dma.schedule_load(tile);

    auto events = dma.tick(0);

    // Find the LOAD_START event and check tile ID
    for (const auto& e : events) {
        if (e.type == EventType::DMA_LOAD_START) {
            REQUIRE(e.tile_id == tile.tile_id);
            REQUIRE(e.component_id == 0);
        }
    }
}

// ============================================================================
// Timing Tests
// ============================================================================

TEST_CASE("DMAEngineProcess transfer time scales with size", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    auto config = default_config(0);

    DMAEngineProcess dma(config, l3_credits, l3_tag_cam);

    // Small tile
    auto small_tile = make_load_tile(MatrixID::A, 0, 0, 0, 256);
    dma.schedule_load(small_tile);

    Cycle cycle = 0;
    while (!dma.is_complete() && cycle < 500) {
        dma.tick(cycle++);
    }
    Cycle small_time = cycle;

    dma.reset();
    l3_tag_cam.reset();

    // Large tile (4x size)
    auto large_tile = make_load_tile(MatrixID::A, 0, 1, 0, 1024);
    dma.schedule_load(large_tile);

    cycle = 0;
    while (!dma.is_complete() && cycle < 500) {
        dma.tick(cycle++);
    }
    Cycle large_time = cycle;

    // Large should take longer (not exactly 4x due to startup overhead)
    REQUIRE(large_time > small_time);
}

// ============================================================================
// Chrome Trace Tests
// ============================================================================

TEST_CASE("DMAEngineProcess events generate Chrome trace JSON", "[timing][dma_process]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);

    DMAEngineProcess dma(default_config(0), l3_credits, l3_tag_cam);

    auto tile = make_load_tile(MatrixID::A, 0, 0);
    dma.schedule_load(tile);

    auto events = dma.tick(0);

    REQUIRE_FALSE(events.empty());
    std::string json = events[0].to_chrome_trace_json();
    REQUIRE(json.find("DMA_LOAD_START") != std::string::npos);
    REQUIRE(json.find("\"pid\":1") != std::string::npos);
    REQUIRE(json.find("\"tid\":0") != std::string::npos);
}
