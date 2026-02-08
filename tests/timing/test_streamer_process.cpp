// ============================================================================
// tests/timing/test_streamer_process.cpp
// Unit tests for StreamerProcess
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

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

static StreamerProcess::Config default_config(uint32_t id = 0,
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

TEST_CASE("StreamerProcess construction", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    SECTION("Row streamer") {
        StreamerProcess streamer(default_config(0, StreamerType::ROW_STREAMER),
                                 l2_tag_cam, l2_credits, compute_result_tag_cam);

        REQUIRE(streamer.id() == 0);
        REQUIRE(streamer.name() == "STR_ROW_0");
        REQUIRE(streamer.is_idle());
        REQUIRE_FALSE(streamer.has_pending_work());
        REQUIRE(streamer.streamer_type() == StreamerType::ROW_STREAMER);
    }

    SECTION("Column streamer") {
        StreamerProcess streamer(default_config(1, StreamerType::COL_STREAMER),
                                 l2_tag_cam, l2_credits, compute_result_tag_cam);

        REQUIRE(streamer.id() == 1);
        REQUIRE(streamer.name() == "STR_COL_1");
        REQUIRE(streamer.streamer_type() == StreamerType::COL_STREAMER);
    }
}

// ============================================================================
// Feed Tests
// ============================================================================

TEST_CASE("StreamerProcess single feed", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate L2 with tile (simulating it arrived from BlockMover)
    auto tile = make_tile(MatrixID::A, 0, 0, 0, 1024);
    l2_credits.acquire();  // Credit used when tile arrived
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    REQUIRE(l2_credits.available() == 15);

    // Schedule a feed
    streamer.schedule_feed(tile);

    REQUIRE(streamer.has_pending_work());
    REQUIRE(streamer.is_idle());  // Not yet started

    // First tick should start the feed
    auto events = streamer.tick(0);

    REQUIRE(count_events(events, EventType::STR_FEED_START) == 1);
    REQUIRE_FALSE(streamer.is_idle());  // Now busy
    REQUIRE_FALSE(streamer.has_pending_work());  // Removed from queue

    // Tick until complete
    Cycle cycle = 1;
    while (!streamer.is_idle() && cycle < 100) {
        events = streamer.tick(cycle++);
    }

    // Should have completed
    REQUIRE(streamer.is_idle());
    REQUIRE_FALSE(l2_tag_cam.lookup(tile.tile_id));  // Tile consumed from L2
    REQUIRE(l2_credits.available() == 16);  // Credit released
    REQUIRE(streamer.total_tiles_fed() == 1);
}

TEST_CASE("StreamerProcess feed stalls without tile", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Schedule a feed WITHOUT tile in L2
    auto tile = make_tile(MatrixID::A, 0, 0);
    streamer.schedule_feed(tile);

    // Tick - should stall (tile not in L2)
    auto events = streamer.tick(0);

    REQUIRE(count_events(events, EventType::STR_STALL_TAG) == 1);
    REQUIRE(streamer.has_pending_work());  // Still queued
    REQUIRE(streamer.is_idle());  // Nothing in flight
    REQUIRE(streamer.stall_cycles_tag() == 1);

    // Add tile to L2 (simulating arrival from BlockMover)
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 1);

    // Now tick should start the feed
    events = streamer.tick(1);
    REQUIRE(count_events(events, EventType::STR_FEED_START) == 1);
}

TEST_CASE("StreamerProcess multiple feeds", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate L2 with multiple tiles
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        l2_credits.acquire();
        l2_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 0);
        streamer.schedule_feed(tile);
    }

    REQUIRE(l2_credits.available() == 12);  // 4 credits used
    REQUIRE(streamer.feed_queue_depth() == 4);

    // Feed one at a time
    Cycle cycle = 0;
    size_t feeds_completed = 0;
    while (feeds_completed < 4 && cycle < 500) {
        auto events = streamer.tick(cycle++);
        feeds_completed += count_events(events, EventType::STR_FEED_COMPLETE);
    }

    REQUIRE(feeds_completed == 4);
    REQUIRE(l2_credits.available() == 16);  // All credits released
    REQUIRE(streamer.total_tiles_fed() == 4);
}

// ============================================================================
// Drain Tests
// ============================================================================

TEST_CASE("StreamerProcess single drain", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate compute_result_tag_cam with result tile (simulating compute complete)
    auto tile = make_tile(MatrixID::C, 0, 0, 0, 1024);
    compute_result_tag_cam.insert(tile.tile_id, 0, 0);

    // Schedule a drain (result from compute going to L2)
    streamer.schedule_drain(tile);

    REQUIRE(streamer.has_pending_work());
    REQUIRE(l2_credits.available() == 16);  // No credit used yet

    // First tick should start the drain (acquire credit)
    auto events = streamer.tick(0);

    REQUIRE(count_events(events, EventType::STR_DRAIN_START) == 1);
    REQUIRE(count_events(events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE_FALSE(streamer.is_idle());
    REQUIRE(l2_credits.available() == 15);  // Credit acquired

    // Tick until complete
    Cycle cycle = 1;
    while (!streamer.is_idle() && cycle < 100) {
        events = streamer.tick(cycle++);
    }

    // Should have completed
    REQUIRE(streamer.is_idle());
    REQUIRE(l2_tag_cam.lookup(tile.tile_id));  // Tile now in L2
    REQUIRE(l2_credits.available() == 15);  // Credit still held by tile
    REQUIRE(streamer.total_tiles_drained() == 1);
}

TEST_CASE("StreamerProcess drain stalls without credit", "[timing][streamer_process]") {
    CreditPool l2_credits(1);  // Only 1 credit
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Use up the only credit
    l2_credits.acquire();
    REQUIRE(l2_credits.available() == 0);

    // Pre-populate compute result
    auto tile = make_tile(MatrixID::C, 0, 0);
    compute_result_tag_cam.insert(tile.tile_id, 0, 0);

    // Schedule a drain
    streamer.schedule_drain(tile);

    // Tick - should stall (no credit)
    auto events = streamer.tick(0);

    REQUIRE(count_events(events, EventType::STR_STALL_CREDIT) == 1);
    REQUIRE(streamer.has_pending_work());
    REQUIRE(streamer.is_idle());
    REQUIRE(streamer.stall_cycles_credit() == 1);

    // Release the credit
    l2_credits.release();

    // Now tick should start the drain
    events = streamer.tick(1);
    REQUIRE(count_events(events, EventType::STR_DRAIN_START) == 1);
}

TEST_CASE("StreamerProcess multiple drains", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate compute results and schedule drains
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::C, 0, i);
        compute_result_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 0);
        streamer.schedule_drain(tile);
    }

    REQUIRE(streamer.drain_queue_depth() == 4);

    // Drain one at a time
    Cycle cycle = 0;
    size_t drains_completed = 0;
    while (drains_completed < 4 && cycle < 500) {
        auto events = streamer.tick(cycle++);
        drains_completed += count_events(events, EventType::STR_DRAIN_COMPLETE);
    }

    REQUIRE(drains_completed == 4);
    REQUIRE(l2_credits.available() == 12);  // 4 credits held by tiles
    REQUIRE(streamer.total_tiles_drained() == 4);
}

// ============================================================================
// Feed/Drain Mix Tests
// ============================================================================

TEST_CASE("StreamerProcess feeds have priority over drains", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate L2 with A tile
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(a_tile.tile_id, 0, 0);

    // Pre-populate compute result for drain
    auto c_tile = make_tile(MatrixID::C, 0, 0);
    compute_result_tag_cam.insert(c_tile.tile_id, 0, 0);

    // Schedule drain first, then feed
    streamer.schedule_drain(c_tile);
    streamer.schedule_feed(a_tile);

    // First tick - feed should start (has priority)
    auto events = streamer.tick(0);

    REQUIRE(count_events(events, EventType::STR_FEED_START) == 1);
    REQUIRE(count_events(events, EventType::STR_DRAIN_START) == 0);
}

TEST_CASE("StreamerProcess interleaved feeds and drains", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate with A tile
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(a_tile.tile_id, 0, 0);
    streamer.schedule_feed(a_tile);

    // Pre-populate compute result and schedule drain
    auto c_tile = make_tile(MatrixID::C, 0, 0);
    compute_result_tag_cam.insert(c_tile.tile_id, 0, 0);
    streamer.schedule_drain(c_tile);

    // Run until both complete
    Cycle cycle = 0;
    size_t feeds = 0, drains = 0;
    while ((feeds < 1 || drains < 1) && cycle < 200) {
        auto events = streamer.tick(cycle++);
        feeds += count_events(events, EventType::STR_FEED_COMPLETE);
        drains += count_events(events, EventType::STR_DRAIN_COMPLETE);
    }

    REQUIRE(feeds == 1);
    REQUIRE(drains == 1);
    REQUIRE(streamer.total_tiles_fed() == 1);
    REQUIRE(streamer.total_tiles_drained() == 1);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("StreamerProcess reset clears state", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate and schedule
    auto tile = make_tile(MatrixID::A, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    streamer.schedule_feed(tile);
    streamer.tick(0);

    REQUIRE_FALSE(streamer.is_idle());

    // Reset
    streamer.reset();

    REQUIRE(streamer.is_idle());
    REQUIRE_FALSE(streamer.has_pending_work());
    REQUIRE(streamer.feed_queue_depth() == 0);
    REQUIRE(streamer.drain_queue_depth() == 0);
    REQUIRE(streamer.stall_cycles_tag() == 0);
    REQUIRE(streamer.stall_cycles_credit() == 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("StreamerProcess tracks statistics", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Complete some feeds
    for (Size i = 0; i < 3; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        l2_credits.acquire();
        l2_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 0);
        streamer.schedule_feed(tile);
    }

    // Complete some drains
    for (Size i = 0; i < 2; ++i) {
        auto tile = make_tile(MatrixID::C, 0, i);
        compute_result_tag_cam.insert(tile.tile_id, static_cast<uint32_t>(i), 0);
        streamer.schedule_drain(tile);
    }

    // Run until complete
    Cycle cycle = 0;
    while (streamer.has_pending_work() || !streamer.is_idle()) {
        if (cycle > 1000) break;
        streamer.tick(cycle++);
    }

    REQUIRE(streamer.total_tiles_fed() == 3);
    REQUIRE(streamer.total_tiles_drained() == 2);
}

// ============================================================================
// Event Generation Tests
// ============================================================================

TEST_CASE("StreamerProcess generates correct events", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Pre-populate with tile
    auto tile = make_tile(MatrixID::A, 0, 0, 0, 512);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    streamer.schedule_feed(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while ((streamer.has_pending_work() || !streamer.is_idle()) && cycle < 200) {
        auto events = streamer.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }

    // Should have: FEED_START, FEED_COMPLETE, TILE_FED_TO_COMPUTE, CREDIT_RELEASED
    REQUIRE(count_events(all_events, EventType::STR_FEED_START) == 1);
    REQUIRE(count_events(all_events, EventType::STR_FEED_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_FED_TO_COMPUTE) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_RELEASED) == 1);

    // Verify FEED_COMPLETE has duration
    for (const auto& e : all_events) {
        if (e.type == EventType::STR_FEED_COMPLETE) {
            REQUIRE(e.duration > 0);
        }
    }
}

TEST_CASE("StreamerProcess drain events are correct", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    auto tile = make_tile(MatrixID::C, 0, 0, 0, 512);
    compute_result_tag_cam.insert(tile.tile_id, 0, 0);
    streamer.schedule_drain(tile);

    // Collect all events
    std::vector<TimingEvent> all_events;
    Cycle cycle = 0;
    while ((streamer.has_pending_work() || !streamer.is_idle()) && cycle < 200) {
        auto events = streamer.tick(cycle++);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }

    // Should have: DRAIN_START, CREDIT_ACQUIRED, DRAIN_COMPLETE, TILE_DRAINED
    REQUIRE(count_events(all_events, EventType::STR_DRAIN_START) == 1);
    REQUIRE(count_events(all_events, EventType::CREDIT_ACQUIRED) == 1);
    REQUIRE(count_events(all_events, EventType::STR_DRAIN_COMPLETE) == 1);
    REQUIRE(count_events(all_events, EventType::TILE_DRAINED) == 1);
}

TEST_CASE("StreamerProcess events have correct tile IDs", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0, StreamerType::COL_STREAMER),
                             l2_tag_cam, l2_credits, compute_result_tag_cam);

    auto tile = make_tile(MatrixID::B, 2, 3, 1);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    streamer.schedule_feed(tile);

    auto events = streamer.tick(0);

    // Find the FEED_START event and check tile ID
    for (const auto& e : events) {
        if (e.type == EventType::STR_FEED_START) {
            REQUIRE(e.tile_id == tile.tile_id);
            REQUIRE(e.component_id == 0);
        }
    }
}

// ============================================================================
// Timing Tests
// ============================================================================

TEST_CASE("StreamerProcess transfer time scales with size", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Small tile
    auto small_tile = make_tile(MatrixID::A, 0, 0, 0, 256);
    l2_credits.acquire();
    l2_tag_cam.insert(small_tile.tile_id, 0, 0);
    streamer.schedule_feed(small_tile);

    Cycle cycle = 0;
    while ((streamer.has_pending_work() || !streamer.is_idle()) && cycle < 500) {
        streamer.tick(cycle++);
    }
    Cycle small_time = cycle;

    streamer.reset();
    l2_tag_cam.reset();

    // Large tile (4x size)
    auto large_tile = make_tile(MatrixID::A, 0, 1, 0, 1024);
    l2_credits.acquire();
    l2_tag_cam.insert(large_tile.tile_id, 0, 0);
    streamer.schedule_feed(large_tile);

    cycle = 0;
    while ((streamer.has_pending_work() || !streamer.is_idle()) && cycle < 500) {
        streamer.tick(cycle++);
    }
    Cycle large_time = cycle;

    // Large should take longer (not exactly 4x due to startup overhead)
    REQUIRE(large_time > small_time);
}

// ============================================================================
// Chrome Trace Tests
// ============================================================================

TEST_CASE("StreamerProcess events generate Chrome trace JSON", "[timing][streamer_process]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    auto tile = make_tile(MatrixID::A, 0, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tile.tile_id, 0, 0);
    streamer.schedule_feed(tile);

    auto events = streamer.tick(0);

    REQUIRE_FALSE(events.empty());
    std::string json = events[0].to_chrome_trace_json();
    REQUIRE(json.find("STR_FEED_START") != std::string::npos);
    REQUIRE(json.find("\"pid\":1") != std::string::npos);
    REQUIRE(json.find("\"tid\":0") != std::string::npos);
}

// ============================================================================
// Work-Conserving Scan Tests
// ============================================================================

TEST_CASE("StreamerProcess work-conserving: processes ready tile not at head",
          "[timing][streamer_process][work_conserving]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Schedule feeds in order: [0,0], [0,1], [0,2]
    auto tile0 = make_tile(MatrixID::A, 0, 0);
    auto tile1 = make_tile(MatrixID::A, 0, 1);
    auto tile2 = make_tile(MatrixID::A, 0, 2);
    streamer.schedule_feed(tile0);
    streamer.schedule_feed(tile1);
    streamer.schedule_feed(tile2);

    // Only tile2 is ready in L2 (tiles arrive out of order from BlockMover)
    l2_credits.acquire();
    l2_tag_cam.insert(tile2.tile_id, 2, 0);

    // Work-conserving scan should find and process tile2, not stall on tile0
    auto events = streamer.tick(1);

    REQUIRE(count_events(events, EventType::STR_FEED_START) == 1);

    // Verify it's tile2 that started
    for (const auto& e : events) {
        if (e.type == EventType::STR_FEED_START) {
            REQUIRE(e.tile_id == tile2.tile_id);
        }
    }

    // tile0 and tile1 should still be queued
    REQUIRE(streamer.feed_queue_depth() == 2);
}

TEST_CASE("StreamerProcess work-conserving: skips blocked head, processes ready tile",
          "[timing][streamer_process][work_conserving]") {
    CreditPool l2_credits(16);
    TagCAM l2_tag_cam(16);
    TagCAM compute_result_tag_cam(16);

    StreamerProcess streamer(default_config(0), l2_tag_cam, l2_credits, compute_result_tag_cam);

    // Schedule tiles: [0,0], [0,1], [0,2], [0,3]
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        tiles.push_back(make_tile(MatrixID::A, 0, i));
        streamer.schedule_feed(tiles.back());
    }

    // Only tiles [0,2] and [0,3] arrive - [0,0] and [0,1] are still blocked
    l2_credits.acquire();
    l2_tag_cam.insert(tiles[2].tile_id, 2, 0);
    l2_credits.acquire();
    l2_tag_cam.insert(tiles[3].tile_id, 3, 0);

    // First tick: should skip blocked [0,0] and [0,1], process [0,2]
    auto events = streamer.tick(1);
    REQUIRE(count_events(events, EventType::STR_FEED_START) == 1);

    TileID first_processed;
    for (const auto& e : events) {
        if (e.type == EventType::STR_FEED_START) {
            first_processed = e.tile_id;
        }
    }

    // Should process [0,2] - the first ready tile in queue order
    REQUIRE(first_processed == tiles[2].tile_id);

    // [0,0], [0,1], and [0,3] still queued
    REQUIRE(streamer.feed_queue_depth() == 3);
}
