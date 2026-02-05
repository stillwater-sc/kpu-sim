// ============================================================================
// tests/timing/test_component_integration.cpp
// Integration tests for component processes working together
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/dma_engine_process.hpp>
#include <sw/kpu/timing/block_mover_process.hpp>
#include <sw/kpu/timing/streamer_process.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileDescriptor make_tile(MatrixID matrix, Size ti, Size tj, Size tk = 0,
                                Size size_bytes = 1024) {
    TileDescriptor desc;
    desc.tile_id = {matrix, ti, tj, tk};
    desc.dram_address = 0x10000 + ti * 0x1000 + tj * 0x100;
    desc.size_bytes = size_bytes;
    return desc;
}

static DMAEngineProcess::Config dma_config(uint32_t id = 0) {
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

static BlockMoverProcess::Config bm_config(uint32_t id = 0) {
    BlockMoverProcess::Config config;
    config.mover_id = id;
    config.bus_width_bytes = 64;
    config.startup_latency = 4;
    config.bandwidth_gbps = 51.2;
    config.clock_ghz = 1.0;
    config.supports_transpose = true;
    config.name = "BM";
    return config;
}

static StreamerProcess::Config str_config(uint32_t id = 0,
                                          StreamerType type = StreamerType::ROW_STREAMER) {
    StreamerProcess::Config config;
    config.streamer_id = id;
    config.type = type;
    config.bus_width_bytes = 64;
    config.startup_latency = 2;
    config.l1_depth = 4;
    config.bandwidth_gbps = 102.4;
    config.clock_ghz = 1.0;
    config.name = "STR";
    return config;
}

static size_t count_events(const std::vector<TimingEvent>& events, EventType type) {
    size_t count = 0;
    for (const auto& e : events) {
        if (e.type == type) count++;
    }
    return count;
}

// ============================================================================
// DMA → BlockMover Pipeline Tests
// ============================================================================

TEST_CASE("DMA to BlockMover pipeline: single tile", "[timing][integration]") {
    // Shared resources
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    // Create components
    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule: DMA loads tile, BM moves it to L2
    auto tile = make_tile(MatrixID::A, 0, 0);
    dma.schedule_load(tile);
    bm.schedule_move(tile);

    // Initially: DMA starts, BM stalls (tile not in L3 yet)
    REQUIRE(l3_credits.available() == 8);

    // Run until both complete
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while ((!dma.is_complete() || !bm.is_complete()) && cycle < 500) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        all_events.insert(all_events.end(), dma_events.begin(), dma_events.end());
        all_events.insert(all_events.end(), bm_events.begin(), bm_events.end());
        cycle++;
    }

    // Verify pipeline completed
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(count_events(all_events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(all_events, EventType::BM_MOVE_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L2) == 1);

    // Final state: tile in L2, credits returned properly
    REQUIRE(l2_tag_cam.lookup(tile.tile_id));
    REQUIRE_FALSE(l3_tag_cam.lookup(tile.tile_id));  // Moved out of L3
    REQUIRE(l3_credits.available() == 8);  // Credit returned
    REQUIRE(l2_credits.available() == 15); // One credit held by tile in L2
}

TEST_CASE("DMA to BlockMover pipeline: multiple tiles", "[timing][integration]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule 4 tiles
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        tiles.push_back(make_tile(MatrixID::A, 0, i));
        dma.schedule_load(tiles.back());
        bm.schedule_move(tiles.back());
    }

    // Run until complete
    Cycle cycle = 0;
    size_t tiles_arrived_l2 = 0;
    while (tiles_arrived_l2 < 4 && cycle < 1000) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        tiles_arrived_l2 += count_events(bm_events, EventType::TILE_ARRIVED_L2);
        cycle++;
    }

    // All tiles should be in L2
    REQUIRE(tiles_arrived_l2 == 4);
    for (const auto& tile : tiles) {
        REQUIRE(l2_tag_cam.lookup(tile.tile_id));
    }
}

// ============================================================================
// BlockMover → Streamer Pipeline Tests
// ============================================================================

TEST_CASE("BlockMover to Streamer pipeline: single tile", "[timing][integration]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);
    StreamerProcess str(str_config(0), l2_tag_cam, l2_credits);

    // Pre-populate L3 with tile (simulating DMA completed)
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    // Schedule: BM moves to L2, Streamer feeds to compute
    bm.schedule_move(tile);
    str.schedule_feed(tile);

    // Run until complete
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while ((!bm.is_complete() || !str.is_complete()) && cycle < 500) {
        auto bm_events = bm.tick(cycle);
        auto str_events = str.tick(cycle);
        all_events.insert(all_events.end(), bm_events.begin(), bm_events.end());
        all_events.insert(all_events.end(), str_events.begin(), str_events.end());
        cycle++;
    }

    // Verify pipeline completed
    REQUIRE(count_events(all_events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L2) == 1);
    REQUIRE(count_events(all_events, EventType::STR_FEED_START) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_FED_TO_COMPUTE) == 1);

    // Final state: tile consumed, all credits returned
    REQUIRE_FALSE(l2_tag_cam.lookup(tile.tile_id));  // Consumed by streamer
    REQUIRE(l3_credits.available() == 8);  // Credit returned
    REQUIRE(l2_credits.available() == 16); // Credit returned
}

// ============================================================================
// Full Pipeline Tests: DMA → BlockMover → Streamer
// ============================================================================

TEST_CASE("Full pipeline: DMA → BlockMover → Streamer", "[timing][integration]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);
    StreamerProcess str(str_config(0), l2_tag_cam, l2_credits);

    // Schedule full pipeline for a tile
    auto tile = make_tile(MatrixID::A, 0, 0);
    dma.schedule_load(tile);
    bm.schedule_move(tile);
    str.schedule_feed(tile);

    // Run until complete
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while ((!dma.is_complete() || !bm.is_complete() || !str.is_complete()) && cycle < 500) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        auto str_events = str.tick(cycle);
        all_events.insert(all_events.end(), dma_events.begin(), dma_events.end());
        all_events.insert(all_events.end(), bm_events.begin(), bm_events.end());
        all_events.insert(all_events.end(), str_events.begin(), str_events.end());
        cycle++;
    }

    // Verify full pipeline completed
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_START) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(count_events(all_events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L2) == 1);
    REQUIRE(count_events(all_events, EventType::STR_FEED_START) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_FED_TO_COMPUTE) == 1);

    // Final state: all credits returned
    REQUIRE(l3_credits.available() == 8);
    REQUIRE(l2_credits.available() == 16);
}

TEST_CASE("Full pipeline: multiple tiles with pipelining", "[timing][integration]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);
    StreamerProcess str(str_config(0), l2_tag_cam, l2_credits);

    // Schedule 4 tiles through the full pipeline
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        tiles.push_back(make_tile(MatrixID::A, 0, i));
        dma.schedule_load(tiles.back());
        bm.schedule_move(tiles.back());
        str.schedule_feed(tiles.back());
    }

    // Run until complete
    Cycle cycle = 0;
    size_t tiles_fed = 0;
    while (tiles_fed < 4 && cycle < 2000) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        auto str_events = str.tick(cycle);
        tiles_fed += count_events(str_events, EventType::TILE_FED_TO_COMPUTE);
        cycle++;
    }

    // All tiles should complete
    REQUIRE(tiles_fed == 4);
    REQUIRE(l3_credits.available() == 8);
    REQUIRE(l2_credits.available() == 16);
}

// ============================================================================
// Drain Pipeline Tests
// ============================================================================

TEST_CASE("Full round-trip: load → compute → drain → store", "[timing][integration]") {
    CreditPool l3_credits(8);
    TagCAM l3_tag_cam(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);
    StreamerProcess str(str_config(0), l2_tag_cam, l2_credits);

    // Load A tile
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    dma.schedule_load(a_tile);
    bm.schedule_move(a_tile);
    str.schedule_feed(a_tile);

    // Result C tile (drain from compute → store to DRAM)
    auto c_tile = make_tile(MatrixID::C, 0, 0);
    str.schedule_drain(c_tile);      // Drain result to L2
    bm.schedule_writeback(c_tile);   // Move from L2 to L3
    dma.schedule_store(c_tile);      // Store from L3 to DRAM

    // Run until complete
    Cycle cycle = 0;
    std::vector<TimingEvent> all_events;
    while (cycle < 2000) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        auto str_events = str.tick(cycle);
        all_events.insert(all_events.end(), dma_events.begin(), dma_events.end());
        all_events.insert(all_events.end(), bm_events.begin(), bm_events.end());
        all_events.insert(all_events.end(), str_events.begin(), str_events.end());

        // Check if fully complete
        if (dma.is_complete() && bm.is_complete() && str.is_complete()) {
            break;
        }
        cycle++;
    }

    // Verify load path completed
    REQUIRE(count_events(all_events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::BM_MOVE_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_FED_TO_COMPUTE) == 1);

    // Verify drain path completed
    REQUIRE(count_events(all_events, EventType::TILE_DRAINED) == 1);
    REQUIRE(count_events(all_events, EventType::BM_WRITEBACK_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::DMA_STORE_COMPLETE) == 1);
}

// ============================================================================
// Credit Flow Tests
// ============================================================================

TEST_CASE("Credit flow: credits released properly through pipeline", "[timing][integration]") {
    CreditPool l3_credits(2);  // Very limited L3
    TagCAM l3_tag_cam(2);
    CreditPool l2_credits(2);  // Very limited L2
    TagCAM l2_tag_cam(2);

    DMAEngineProcess dma(dma_config(0), l3_credits, l3_tag_cam);
    BlockMoverProcess bm(bm_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);
    StreamerProcess str(str_config(0), l2_tag_cam, l2_credits);

    // Schedule more tiles than buffer capacity
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        dma.schedule_load(tile);
        bm.schedule_move(tile);
        str.schedule_feed(tile);
    }

    // Run - should complete despite limited credits due to proper flow
    Cycle cycle = 0;
    size_t tiles_fed = 0;
    while (tiles_fed < 4 && cycle < 5000) {
        auto dma_events = dma.tick(cycle);
        auto bm_events = bm.tick(cycle);
        auto str_events = str.tick(cycle);
        tiles_fed += count_events(str_events, EventType::TILE_FED_TO_COMPUTE);
        cycle++;
    }

    // All tiles should complete (credits flow back)
    REQUIRE(tiles_fed == 4);
}

// ============================================================================
// Multi-Component Tests
// ============================================================================

TEST_CASE("Multiple DMA engines working in parallel", "[timing][integration]") {
    CreditPool l3_credits(16);
    TagCAM l3_tag_cam(16);

    // Create 4 DMA engines
    std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines;
    for (uint32_t i = 0; i < 4; ++i) {
        dma_engines.push_back(std::make_unique<DMAEngineProcess>(
            dma_config(i), l3_credits, l3_tag_cam));
    }

    // Distribute tiles across engines
    for (Size i = 0; i < 16; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        dma_engines[i % 4]->schedule_load(tile);
    }

    // Run until complete
    Cycle cycle = 0;
    size_t tiles_loaded = 0;
    while (tiles_loaded < 16 && cycle < 1000) {
        for (auto& engine : dma_engines) {
            auto events = engine->tick(cycle);
            tiles_loaded += count_events(events, EventType::TILE_ARRIVED_L3);
        }
        cycle++;
    }

    REQUIRE(tiles_loaded == 16);
    REQUIRE(l3_tag_cam.size() == 16);
}

TEST_CASE("Multiple BlockMovers working in parallel", "[timing][integration]") {
    CreditPool l3_credits(16);
    TagCAM l3_tag_cam(16);
    CreditPool l2_credits(32);
    TagCAM l2_tag_cam(32);

    // Pre-populate L3 with tiles
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        l3_credits.acquire();
        l3_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 0);
    }

    // Create 2 BlockMovers
    std::vector<std::unique_ptr<BlockMoverProcess>> block_movers;
    for (uint32_t i = 0; i < 2; ++i) {
        block_movers.push_back(std::make_unique<BlockMoverProcess>(
            bm_config(i), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam));
    }

    // Distribute tiles across movers
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        block_movers[i % 2]->schedule_move(tile);
    }

    // Run until complete
    Cycle cycle = 0;
    size_t tiles_moved = 0;
    while (tiles_moved < 8 && cycle < 500) {
        for (auto& mover : block_movers) {
            auto events = mover->tick(cycle);
            tiles_moved += count_events(events, EventType::TILE_ARRIVED_L2);
        }
        cycle++;
    }

    REQUIRE(tiles_moved == 8);
    REQUIRE(l2_tag_cam.size() == 8);
}
