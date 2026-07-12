// ============================================================================
// tests/timing/test_block_mover_process.cpp
// Unit tests for BlockMoverProcess
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/block_mover_process.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileDescriptor make_tile(MatrixID matrix, Size ti, Size tj, Size tk = 0,
                                 Size size_bytes = 1024) {
    TileDescriptor desc;
    desc.tile_id = {matrix, ti, tj, tk};
    desc.size_bytes = size_bytes;
    return desc;
}

static BlockMoverProcess::Config default_config(uint32_t id = 0) {
    BlockMoverProcess::Config config;
    config.mover_id = id;
    config.bus_width_bytes = 64;
    config.startup_latency = 4;
    config.bandwidth_gbps = 51.2;
    config.clock_ghz = 1.0;
    config.supports_transpose = true;
    config.name = config.display_name();
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
// Construction Tests
// ============================================================================

TEST_CASE("BlockMoverProcess construction", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    REQUIRE(bm.id() == 0);
    REQUIRE(bm.name() == "L3(0,0):BM");
    REQUIRE(bm.is_idle());
    REQUIRE_FALSE(bm.has_pending_work());
    REQUIRE(bm.is_complete());
}

// ============================================================================
// Move Tests (L3 → L2)
// ============================================================================

TEST_CASE("BlockMoverProcess move requires L3 tile", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    auto tile = make_tile(MatrixID::A, 0, 0);
    bm.schedule_move(tile);

    // Tick without tile in L3 - should stall
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_STALL_TAG) == 1);
    REQUIRE(bm.has_pending_work());
    REQUIRE(bm.is_idle());
}

TEST_CASE("BlockMoverProcess move requires L2 credit", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(1);  // Only 1 credit
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L3
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    // Use up the only L2 credit
    l2_credits.acquire();

    bm.schedule_move(tile);

    // Tick - should stall on credit
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_STALL_CREDIT) == 1);
    REQUIRE(bm.has_pending_work());
}

TEST_CASE("BlockMoverProcess move succeeds with L3 tile and L2 credit", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L3
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    bm.schedule_move(tile);

    // First tick - should start
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE_FALSE(bm.is_idle());
    REQUIRE(l2_credits.available() == 15);  // L2 credit acquired
    REQUIRE(l3_tag_cam.lookup(tile.tile_id));  // Still in L3 until complete

    // Run until complete
    Cycle cycle = 1;
    while (!bm.is_idle() && cycle < 100) {
        events = bm.tick(cycle++);
    }

    REQUIRE(bm.is_idle());
    REQUIRE_FALSE(l3_tag_cam.lookup(tile.tile_id));  // Removed from L3
    REQUIRE(l2_tag_cam.lookup(tile.tile_id));  // Now in L2
    REQUIRE(l3_credits.available() == 8);  // L3 credit released
}

TEST_CASE("BlockMoverProcess move is single-transfer", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L3 with 2 tiles
    auto tile1 = make_tile(MatrixID::A, 0, 0);
    auto tile2 = make_tile(MatrixID::A, 0, 1);
    l3_credits.acquire();
    l3_credits.acquire();
    l3_tag_cam.insert(tile1.tile_id, 0, 0);
    l3_tag_cam.insert(tile2.tile_id, 1, 0);

    bm.schedule_move(tile1);
    bm.schedule_move(tile2);

    // First tick - should start only one
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);
    REQUIRE(bm.has_pending_work());  // Second still queued
}

// ============================================================================
// Writeback Tests (L2 → L3)
// ============================================================================

TEST_CASE("BlockMoverProcess writeback requires L2 tile", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    auto tile = make_tile(MatrixID::C, 0, 0);
    bm.schedule_writeback(tile);

    // Tick without tile in L2 - should stall
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_STALL_TAG) == 1);
}

TEST_CASE("BlockMoverProcess writeback requires L3 credit", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(1);  // Only 1 credit
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L2
    auto tile = make_tile(MatrixID::C, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);

    // Use up the only L3 credit
    l3_credits.acquire();

    bm.schedule_writeback(tile);

    // Tick - should stall on credit
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_STALL_CREDIT) == 1);
}

TEST_CASE("BlockMoverProcess writeback succeeds", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L2
    auto tile = make_tile(MatrixID::C, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);

    bm.schedule_writeback(tile);

    // First tick - should start
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_WRITEBACK_START) == 1);
    REQUIRE(count_events(events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(l3_credits.available() == 7);  // L3 credit acquired

    // Run until complete
    Cycle cycle = 1;
    while (!bm.is_idle() && cycle < 100) {
        events = bm.tick(cycle++);
    }

    REQUIRE(bm.is_idle());
    REQUIRE_FALSE(l2_tag_cam.lookup(tile.tile_id));  // Removed from L2
    REQUIRE(l3_tag_cam.lookup(tile.tile_id));  // Now in L3
    REQUIRE(l2_credits.available() == 16);  // L2 credit released
}

// ============================================================================
// Priority Tests
// ============================================================================

TEST_CASE("BlockMoverProcess prioritizes moves over writebacks", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate both L3 and L2
    auto move_tile = make_tile(MatrixID::A, 0, 0);
    auto wb_tile = make_tile(MatrixID::C, 0, 0);

    l3_credits.acquire();
    l3_tag_cam.insert(move_tile.tile_id, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(wb_tile.tile_id, 0, 0);

    // Schedule writeback first, then move
    bm.schedule_writeback(wb_tile);
    bm.schedule_move(move_tile);

    // Tick - should start move (higher priority)
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(events, EventType::BM_WRITEBACK_START) == 0);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("BlockMoverProcess reset clears state", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate and start a move
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);
    bm.schedule_move(tile);
    bm.tick(0);

    REQUIRE_FALSE(bm.is_idle());

    // Reset
    bm.reset();

    REQUIRE(bm.is_idle());
    REQUIRE(bm.is_complete());
    REQUIRE(bm.move_queue_depth() == 0);
    REQUIRE(bm.writeback_queue_depth() == 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("BlockMoverProcess tracks statistics", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate and move
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);
    bm.schedule_move(tile);

    // Run until complete
    Cycle cycle = 0;
    while (!bm.is_complete() && cycle < 100) {
        bm.tick(cycle++);
    }

    REQUIRE(bm.total_tiles_moved() == 1);
    REQUIRE(bm.total_tiles_writeback() == 0);
}

// ============================================================================
// Event Generation Tests
// ============================================================================

TEST_CASE("BlockMoverProcess generates correct move events", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate and move
    auto tile = make_tile(MatrixID::A, 0, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 0);
    bm.schedule_move(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while (!bm.is_complete() && cycle < 100) {
        auto events = bm.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }

    // Should have: MOVE_START, CREDIT_ACQUIRED, MOVE_COMPLETE, TILE_ARRIVED_L2, CREDIT_RELEASED
    REQUIRE(count_events(all_events, EventType::BM_MOVE_START) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(count_events(all_events, EventType::BM_MOVE_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L2) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_RELEASED) == 1);
}

TEST_CASE("BlockMoverProcess generates correct writeback events", "[timing][block_mover_process]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Pre-populate L2 and writeback
    auto tile = make_tile(MatrixID::C, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    bm.schedule_writeback(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while (!bm.is_complete() && cycle < 100) {
        auto events = bm.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }

    // Should have: WRITEBACK_START, CREDIT_ACQUIRED, WRITEBACK_COMPLETE, TILE_ARRIVED_L3, CREDIT_RELEASED
    REQUIRE(count_events(all_events, EventType::BM_WRITEBACK_START) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(count_events(all_events, EventType::BM_WRITEBACK_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_RELEASED) == 1);
}

// ============================================================================
// Integration-style Tests
// ============================================================================

TEST_CASE("BlockMoverProcess DMA->BM pipeline", "[timing][block_mover_process]") {
    // Simulate: DMA loads tile to L3, then BM moves it to L2
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    auto tile = make_tile(MatrixID::A, 0, 0);
    bm.schedule_move(tile);

    // Initially stalls (no tile in L3)
    auto events = bm.tick(0);
    REQUIRE(count_events(events, EventType::BM_STALL_TAG) == 1);

    // Simulate DMA completing (tile arrives at L3)
    l3_credits.acquire();
    l3_tag_cam.insert(tile.tile_id, 0, 10);

    // Now BM should start
    events = bm.tick(11);
    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);

    // Run until complete
    Cycle cycle = 12;
    while (!bm.is_complete() && cycle < 100) {
        bm.tick(cycle++);
    }

    REQUIRE(l2_tag_cam.lookup(tile.tile_id));
}

// ============================================================================
// Work-Conserving Scan Tests
// ============================================================================

TEST_CASE("BlockMoverProcess work-conserving: processes ready tile not at head",
          "[timing][block_mover_process][work_conserving]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule tiles in order: [0,0], [0,1], [0,2]
    auto tile0 = make_tile(MatrixID::A, 0, 0);
    auto tile1 = make_tile(MatrixID::A, 0, 1);
    auto tile2 = make_tile(MatrixID::A, 0, 2);
    bm.schedule_move(tile0);
    bm.schedule_move(tile1);
    bm.schedule_move(tile2);

    // But tiles arrive OUT OF ORDER: [0,2] arrives first!
    l3_credits.acquire();
    l3_tag_cam.insert(tile2.tile_id, 2, 0);

    // Work-conserving scan should find and process tile2, not stall on tile0
    auto events = bm.tick(1);

    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);

    // Verify it's tile2 that started, not tile0
    for (const auto& e : events) {
        if (e.type == EventType::BM_MOVE_START) {
            REQUIRE(e.tile_id == tile2.tile_id);
        }
    }

    // tile0 and tile1 should still be queued
    REQUIRE(bm.move_queue_depth() == 2);
}

TEST_CASE("BlockMoverProcess work-conserving: skips blocked head, processes ready tile",
          "[timing][block_mover_process][work_conserving]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule tiles: [0,0], [0,1], [0,2], [0,3]
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        tiles.push_back(make_tile(MatrixID::A, 0, i));
        bm.schedule_move(tiles.back());
    }

    // Only tiles [0,2] and [0,3] arrive - [0,0] and [0,1] are still in DRAM
    l3_credits.acquire();
    l3_tag_cam.insert(tiles[2].tile_id, 2, 0);
    l3_credits.acquire();
    l3_tag_cam.insert(tiles[3].tile_id, 3, 0);

    // First tick: should skip blocked [0,0] and [0,1], process [0,2]
    auto events = bm.tick(1);
    REQUIRE(count_events(events, EventType::BM_MOVE_START) == 1);

    TileID first_processed;
    for (const auto& e : events) {
        if (e.type == EventType::BM_MOVE_START) {
            first_processed = e.tile_id;
        }
    }

    // Should process [0,2] - the first ready tile in queue order
    REQUIRE(first_processed == tiles[2].tile_id);

    // [0,0] and [0,1] still queued (blocked), [0,3] still queued (waiting)
    REQUIRE(bm.move_queue_depth() == 3);
}

TEST_CASE("BlockMoverProcess work-conserving writeback: processes ready tile not at head",
          "[timing][block_mover_process][work_conserving]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule writebacks: [0,0], [0,1], [0,2]
    auto tile0 = make_tile(MatrixID::C, 0, 0);
    auto tile1 = make_tile(MatrixID::C, 0, 1);
    auto tile2 = make_tile(MatrixID::C, 0, 2);
    bm.schedule_writeback(tile0);
    bm.schedule_writeback(tile1);
    bm.schedule_writeback(tile2);

    // Only tile1 is ready in L2
    l2_credits.acquire();
    l2_tag_cam.insert(tile1.tile_id, 1, 0);

    // Work-conserving scan should find and process tile1
    auto events = bm.tick(1);

    REQUIRE(count_events(events, EventType::BM_WRITEBACK_START) == 1);

    // Verify it's tile1 that started
    for (const auto& e : events) {
        if (e.type == EventType::BM_WRITEBACK_START) {
            REQUIRE(e.tile_id == tile1.tile_id);
        }
    }
}

// ============================================================================
// Priority Aging Tests
// ============================================================================

TEST_CASE("BlockMoverProcess priority aging: oldest ready tile processed first",
          "[timing][block_mover_process][priority_aging]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    // Enable priority aging
    auto config = default_config(0);
    config.priority_aging = true;
    BlockMoverProcess bm(config, l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Schedule tiles at different times (simulating enqueue_cycle)
    auto tile0 = make_tile(MatrixID::A, 0, 0);
    auto tile1 = make_tile(MatrixID::A, 0, 1);
    auto tile2 = make_tile(MatrixID::A, 0, 2);

    // Tile0 enqueued at cycle 0
    bm.schedule_move(tile0);
    bm.tick(0);  // Stalls, but sets enqueue_cycle

    // Tile1 enqueued at cycle 10
    bm.tick(10);
    bm.schedule_move(tile1);

    // Tile2 enqueued at cycle 20
    bm.tick(20);
    bm.schedule_move(tile2);

    // All tiles arrive in L3 at the same time (cycle 25)
    // But tile2 is in queue position before tile1 (tile2 has higher index)
    // Without priority aging: first ready tile in queue order (tile0)
    // With priority aging: oldest tile (tile0, enqueue_cycle=0)
    l3_credits.acquire();
    l3_tag_cam.insert(tile0.tile_id, 0, 25);
    l3_credits.acquire();
    l3_tag_cam.insert(tile1.tile_id, 1, 25);
    l3_credits.acquire();
    l3_tag_cam.insert(tile2.tile_id, 2, 25);

    // Process - with priority aging, should pick tile0 (oldest)
    auto events = bm.tick(25);

    TileID first_processed;
    for (const auto& e : events) {
        if (e.type == EventType::BM_MOVE_START) {
            first_processed = e.tile_id;
        }
    }

    // With priority aging, oldest tile (tile0) should be processed first
    REQUIRE(first_processed == tile0.tile_id);
}

TEST_CASE("BlockMoverProcess priority aging: prevents starvation of old requests",
          "[timing][block_mover_process][priority_aging]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);

    // Enable priority aging
    auto config = default_config(0);
    config.priority_aging = true;
    BlockMoverProcess bm(config, l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // Old tile scheduled first but blocked
    auto old_tile = make_tile(MatrixID::A, 0, 0);
    bm.schedule_move(old_tile);
    bm.tick(0);  // Stalls, enqueue_cycle = 0

    // Newer tiles scheduled later
    bm.tick(100);
    for (Size i = 1; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        bm.schedule_move(tile);
    }

    // All tiles become ready at cycle 200
    l3_credits.acquire();
    l3_tag_cam.insert(old_tile.tile_id, 0, 200);
    for (Size i = 1; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        l3_credits.acquire();
        l3_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 200);
    }

    // First process should be old_tile (enqueue_cycle = 0)
    auto events = bm.tick(200);

    TileID first_processed;
    for (const auto& e : events) {
        if (e.type == EventType::BM_MOVE_START) {
            first_processed = e.tile_id;
        }
    }

    // Old tile should not be starved - it should be processed first
    REQUIRE(first_processed == old_tile.tile_id);
}

// ============================================================================
// Broadcast consumer-count seeding (issue #100, epic E2)
// ============================================================================

TEST_CASE("BlockMoverProcess seeds L2 ref count with the tile consumer count", "[timing][block_mover_process][broadcast]") {
    TagCAM l3_tag_cam(8);
    CreditPool l3_credits(8);
    CreditPool l2_credits(8);
    TagCAM l2_tag_cam(8);
    BlockMoverProcess bm(default_config(0), l3_tag_cam, l3_credits, l2_credits, l2_tag_cam);

    // A broadcast tile: one MOVE, three downstream feeds
    auto tile = make_tile(MatrixID::B, 0, 0);
    tile.consumer_count = 3;
    REQUIRE(l3_credits.acquire());
    l3_tag_cam.insert(tile.tile_id, 0, 0);

    bm.schedule_move(tile);
    Cycle cycle = 0;
    while (!bm.is_idle() || bm.has_pending_work()) {
        bm.tick(cycle++);
        REQUIRE(cycle < 1000);
    }

    // One MOVE consumed exactly one L2 credit...
    REQUIRE(l2_credits.outstanding() == 1);

    // ...and seeded three references: the credit-release signal fires on
    // the THIRD invalidate (feed), not the first
    REQUIRE_FALSE(l2_tag_cam.invalidate(tile.tile_id));
    REQUIRE_FALSE(l2_tag_cam.invalidate(tile.tile_id));
    REQUIRE(l2_tag_cam.invalidate(tile.tile_id));
}
