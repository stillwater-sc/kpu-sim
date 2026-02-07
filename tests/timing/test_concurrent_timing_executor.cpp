// ============================================================================
// tests/timing/test_concurrent_timing_executor.cpp
// Unit tests for ConcurrentTimingExecutor
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>

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

static ConcurrentTimingExecutor::Config default_config() {
    ConcurrentTimingExecutor::Config config;
    config.num_dma_engines = 2;
    config.dma_queue_depth = 4;
    config.dma_bandwidth_gbps = 25.6;
    config.l3_buffer_count = 8;
    config.num_block_movers = 2;
    config.l2_bank_count = 16;
    config.num_row_streamers = 1;
    config.num_col_streamers = 1;
    config.clock_ghz = 1.0;
    config.max_cycles = 10000;
    config.enable_livelock_detection = false;  // Disable for basic tests
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

TEST_CASE("ConcurrentTimingExecutor construction", "[timing][executor]") {
    auto config = default_config();
    ConcurrentTimingExecutor executor(config);

    REQUIRE(executor.num_dma_engines() == 2);
    REQUIRE(executor.num_block_movers() == 2);
    REQUIRE(executor.num_row_streamers() == 1);
    REQUIRE(executor.num_col_streamers() == 1);
    REQUIRE(executor.current_cycle() == 0);
    REQUIRE(executor.is_complete());  // No work scheduled
}

TEST_CASE("ConcurrentTimingExecutor configuration", "[timing][executor]") {
    ConcurrentTimingExecutor::Config config;
    config.num_dma_engines = 4;
    config.num_block_movers = 4;
    config.num_row_streamers = 2;
    config.num_col_streamers = 2;
    config.l3_buffer_count = 32;
    config.l2_bank_count = 64;

    ConcurrentTimingExecutor executor(config);

    REQUIRE(executor.num_dma_engines() == 4);
    REQUIRE(executor.num_block_movers() == 4);
    REQUIRE(executor.num_row_streamers() == 2);
    REQUIRE(executor.num_col_streamers() == 2);
    REQUIRE(executor.l3_credits().available() == 32);
    REQUIRE(executor.l2_credits().available() == 64);
}

// ============================================================================
// Single Tile Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor single tile load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);

    REQUIRE_FALSE(executor.is_complete());

    bool completed = executor.run();

    REQUIRE(completed);
    REQUIRE(executor.l3_tag_cam().lookup(tile.tile_id));

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 1);
}

TEST_CASE("ConcurrentTimingExecutor single tile pipeline: load → move → feed", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.schedule_move(tile);
    executor.schedule_feed(tile);

    bool completed = executor.run();

    REQUIRE(completed);

    // Verify events
    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::DMA_LOAD_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_ARRIVED_L3) == 1);
    REQUIRE(count_events(events, EventType::BM_MOVE_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_ARRIVED_L2) == 1);
    REQUIRE(count_events(events, EventType::STR_FEED_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 1);

    // All credits should be returned
    REQUIRE(executor.l3_credits().available() == executor.config().l3_buffer_count);
    REQUIRE(executor.l2_credits().available() == executor.config().l2_bank_count);
}

// ============================================================================
// Multiple Tile Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor multiple tiles load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Schedule 4 loads
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 4);
}

TEST_CASE("ConcurrentTimingExecutor multiple tiles full pipeline", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Schedule 4 tiles through full pipeline
    std::vector<TileDescriptor> tiles;
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        tiles.push_back(tile);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 4);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 4);
    REQUIRE(stats.tiles_moved == 4);
    REQUIRE(stats.tiles_fed == 4);
}

// ============================================================================
// Store and Writeback Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor store after load", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);

    // Load tile to L3
    executor.schedule_load(tile);
    executor.run();
    REQUIRE(executor.l3_tag_cam().lookup(tile.tile_id));

    // Store tile back to DRAM
    executor.schedule_store(tile);
    bool completed = executor.run();

    REQUIRE(completed);
    REQUIRE_FALSE(executor.l3_tag_cam().lookup(tile.tile_id));

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_stored == 1);
}

TEST_CASE("ConcurrentTimingExecutor drain and writeback", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // First, load and feed an A tile to "compute"
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(a_tile);
    executor.schedule_move(a_tile);
    executor.schedule_feed(a_tile);

    // Signal compute complete (depends on A tile being fed)
    auto c_tile = make_tile(MatrixID::C, 0, 0);
    executor.schedule_compute(c_tile, a_tile.tile_id);

    // Result tile drains from compute to L2, then writes back to L3
    executor.schedule_drain(c_tile);
    executor.schedule_writeback(c_tile);
    executor.schedule_store(c_tile);

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_DRAINED) == 1);
    REQUIRE(count_events(events, EventType::BM_WRITEBACK_COMPLETE) == 1);
    REQUIRE(count_events(events, EventType::DMA_STORE_COMPLETE) == 1);
}

// ============================================================================
// Work Distribution Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor distributes work across DMA engines", "[timing][executor]") {
    auto config = default_config();
    config.num_dma_engines = 4;
    ConcurrentTimingExecutor executor(config);

    // Schedule 8 loads - should distribute across 4 engines
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_loaded == 8);
}

TEST_CASE("ConcurrentTimingExecutor distributes work across BlockMovers", "[timing][executor]") {
    auto config = default_config();
    config.num_block_movers = 4;
    ConcurrentTimingExecutor executor(config);

    // Pre-load tiles to L3
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
    }
    executor.run();

    // Schedule moves - should distribute across movers
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_move(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    REQUIRE(stats.tiles_moved == 8);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor reset", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Run some work
    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.run();

    REQUIRE(executor.current_cycle() > 0);
    REQUIRE_FALSE(executor.events().empty());

    // Reset
    executor.reset();

    REQUIRE(executor.current_cycle() == 0);
    REQUIRE(executor.events().empty());
    REQUIRE(executor.is_complete());
    REQUIRE(executor.l3_credits().available() == executor.config().l3_buffer_count);
    REQUIRE(executor.l2_credits().available() == executor.config().l2_bank_count);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor statistics", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Full pipeline for 4 tiles
    for (Size i = 0; i < 4; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    executor.run();

    auto stats = executor.get_statistics();

    REQUIRE(stats.total_cycles > 0);
    REQUIRE(stats.tiles_loaded == 4);
    REQUIRE(stats.tiles_moved == 4);
    REQUIRE(stats.tiles_fed == 4);
    REQUIRE(stats.bytes_loaded == 4 * 1024);  // 4 tiles × 1024 bytes
}

TEST_CASE("ConcurrentTimingExecutor utilization", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    executor.run();

    auto stats = executor.get_statistics();

    // Utilization should be between 0 and 1
    REQUIRE(stats.dma_utilization() >= 0.0);
    REQUIRE(stats.dma_utilization() <= 1.0);
    REQUIRE(stats.bm_utilization() >= 0.0);
    REQUIRE(stats.bm_utilization() <= 1.0);
    REQUIRE(stats.str_utilization() >= 0.0);
    REQUIRE(stats.str_utilization() <= 1.0);
}

// ============================================================================
// Step-by-Step Execution Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor step execution", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);

    // Step through manually
    REQUIRE_FALSE(executor.is_complete());

    Cycle cycle = 0;
    while (!executor.is_complete() && cycle < 1000) {
        executor.step();
        cycle++;
    }

    REQUIRE(executor.is_complete());
    REQUIRE(executor.current_cycle() == cycle);
}

// ============================================================================
// Export Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor Chrome trace export", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.schedule_move(tile);
    executor.run();

    // Export to temp file
    std::string temp_file = "/tmp/test_trace.json";
    executor.export_chrome_trace(temp_file);

    // Verify file exists and has content
    std::ifstream file(temp_file);
    REQUIRE(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    REQUIRE(content.find("traceEvents") != std::string::npos);
    REQUIRE(content.find("DMA_LOAD_START") != std::string::npos);
}

TEST_CASE("ConcurrentTimingExecutor CSV export", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    auto tile = make_tile(MatrixID::A, 0, 0);
    executor.schedule_load(tile);
    executor.run();

    // Export to temp file
    std::string temp_file = "/tmp/test_trace.csv";
    executor.export_csv(temp_file);

    // Verify file exists and has content
    std::ifstream file(temp_file);
    REQUIRE(file.is_open());

    std::string line;
    std::getline(file, line);  // Header
    REQUIRE(line.find("cycle") != std::string::npos);
    REQUIRE(line.find("type") != std::string::npos);
}

// ============================================================================
// Credit Flow Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor limited credits", "[timing][executor]") {
    auto config = default_config();
    config.l3_buffer_count = 2;  // Very limited L3
    config.l2_bank_count = 2;    // Very limited L2
    ConcurrentTimingExecutor executor(config);

    // Schedule more tiles than buffer capacity
    for (Size i = 0; i < 8; ++i) {
        auto tile = make_tile(MatrixID::A, 0, i);
        executor.schedule_load(tile);
        executor.schedule_move(tile);
        executor.schedule_feed(tile);
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 8);
}

// ============================================================================
// A and B Matrix Tests
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor A and B matrices", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Load and feed both A (row) and B (column) tiles
    auto a_tile = make_tile(MatrixID::A, 0, 0);
    auto b_tile = make_tile(MatrixID::B, 0, 0);

    executor.schedule_load(a_tile);
    executor.schedule_load(b_tile);
    executor.schedule_move(a_tile);
    executor.schedule_move(b_tile);
    executor.schedule_feed(a_tile);  // Goes to row streamer
    executor.schedule_feed(b_tile);  // Goes to column streamer

    bool completed = executor.run();

    REQUIRE(completed);

    auto& events = executor.events();
    REQUIRE(count_events(events, EventType::TILE_FED_TO_COMPUTE) == 2);
}

// ============================================================================
// Matmul-like Pattern Test
// ============================================================================

TEST_CASE("ConcurrentTimingExecutor matmul pattern", "[timing][executor]") {
    ConcurrentTimingExecutor executor(default_config());

    // Simulate a 2x2x2 tiled matmul
    // C[ti][tj] += A[ti][tk] * B[tk][tj]

    constexpr Size Ti = 2, Tj = 2, Tk = 2;

    // Schedule all loads first
    for (Size ti = 0; ti < Ti; ++ti) {
        for (Size tk = 0; tk < Tk; ++tk) {
            auto a_tile = make_tile(MatrixID::A, ti, 0, tk);
            executor.schedule_load(a_tile);
            executor.schedule_move(a_tile);
            executor.schedule_feed(a_tile);
        }
    }

    for (Size tk = 0; tk < Tk; ++tk) {
        for (Size tj = 0; tj < Tj; ++tj) {
            auto b_tile = make_tile(MatrixID::B, 0, tj, tk);
            executor.schedule_load(b_tile);
            executor.schedule_move(b_tile);
            executor.schedule_feed(b_tile);
        }
    }

    // Schedule result drains (with COMPUTE dependency)
    for (Size ti = 0; ti < Ti; ++ti) {
        for (Size tj = 0; tj < Tj; ++tj) {
            auto c_tile = make_tile(MatrixID::C, ti, tj);
            // COMPUTE depends on last B tile for this column: B[0,tj,Tk-1]
            TileID last_b;
            last_b.matrix = MatrixID::B;
            last_b.ti = 0;
            last_b.tj = tj;
            last_b.tk = Tk - 1;
            executor.schedule_compute(c_tile, last_b);
            executor.schedule_drain(c_tile);
            executor.schedule_writeback(c_tile);
            executor.schedule_store(c_tile);
        }
    }

    bool completed = executor.run();

    REQUIRE(completed);

    auto stats = executor.get_statistics();
    // tiles_loaded counts TILE_ARRIVED_L3 events, which includes:
    // - DMA loads (A + B tiles)
    // - BlockMover writebacks (C tiles arriving at L3 from L2)
    REQUIRE(stats.tiles_loaded == Ti * Tk + Tk * Tj + Ti * Tj);  // A + B + C writebacks
    REQUIRE(stats.tiles_fed == Ti * Tk + Tk * Tj);     // All A/B tiles fed
    REQUIRE(stats.tiles_drained == Ti * Tj);           // C tiles drained
    REQUIRE(stats.tiles_stored == Ti * Tj);            // C tiles stored
}
