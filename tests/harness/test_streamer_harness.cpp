// ============================================================================
// tests/harness/test_streamer_harness.cpp
// Unit tests for Streamer test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/harness/streamer_harness.hpp>

using namespace sw::kpu::harness;
using Catch::Approx;

TEST_CASE("Streamer Harness construction", "[harness][streamer]") {
    SECTION("Default configuration") {
        StreamerHarnessConfig config;
        StreamerHarness harness(config);

        REQUIRE(harness.current_cycle() == 0);
        REQUIRE(harness.streamer_stats().rows_streamed == 0);
        REQUIRE_FALSE(harness.has_pending());
    }

    SECTION("Custom configuration") {
        StreamerHarnessConfig config;
        config.num_streamers = 4;
        config.l2_banks = 32;
        config.l1_depth = 8;
        config.systolic_size = 32;

        StreamerHarness harness(config);
        REQUIRE(harness.l1_buffers().depth() == 8);
    }
}

TEST_CASE("Streamer Harness L2 preloading", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};

    SECTION("Preload with raw data") {
        std::vector<uint8_t> data(512, 0xCD);
        harness.preload_l2(0, tile, data.data(), data.size());

        REQUIRE(harness.l2_banks().is_occupied(0));
    }

    SECTION("Preload with pattern") {
        harness.preload_l2_pattern(0, tile, 16, 16, 2);
        REQUIRE(harness.l2_banks().is_occupied(0));
    }
}

TEST_CASE("Streamer Harness row streaming", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);

    SECTION("Basic row stream") {
        bool accepted = harness.stream_rows(tile, 0);
        REQUIRE(accepted);

        bool completed = harness.run();
        REQUIRE(completed);
        REQUIRE(harness.streamer_stats().rows_streamed == 1);
    }

    SECTION("Stream from invalid bank fails") {
        bool accepted = harness.stream_rows(tile, 99);  // Non-existent bank
        REQUIRE_FALSE(accepted);
    }
}

TEST_CASE("Streamer Harness column streaming", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);

    bool accepted = harness.stream_cols(tile, 0);
    REQUIRE(accepted);

    harness.run();
    REQUIRE(harness.streamer_stats().cols_streamed == 1);
}

TEST_CASE("Streamer Harness broadcast", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);

    bool accepted = harness.broadcast(tile, 0);
    REQUIRE(accepted);

    harness.run();
    REQUIRE(harness.streamer_stats().broadcasts == 1);
}

TEST_CASE("Streamer Harness drain", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};

    // First do some streaming to populate accumulator
    harness.preload_l2_pattern(0, tile, 16, 16, 2);
    harness.stream_rows(tile, 0);
    harness.run();

    // Now drain
    TileID output_tile{1, 0, 0};
    bool accepted = harness.drain(output_tile, 1);
    REQUIRE(accepted);

    harness.run();
    REQUIRE(harness.streamer_stats().drains == 1);
}

TEST_CASE("Streamer Harness multiple operations", "[harness][streamer]") {
    StreamerHarnessConfig config;
    config.l2_banks = 16;
    StreamerHarness harness(config);

    // Preload several tiles
    for (uint32_t i = 0; i < 4; ++i) {
        TileID tile{0, 0, i};
        harness.preload_l2_pattern(i, tile, 16, 16, 2);
    }

    // Stream alternating rows and columns
    for (uint32_t i = 0; i < 4; ++i) {
        TileID tile{0, 0, i};
        if (i % 2 == 0) {
            harness.stream_rows(tile, i);
        } else {
            harness.stream_cols(tile, i);
        }
    }

    harness.run();

    REQUIRE(harness.streamer_stats().rows_streamed == 2);
    REQUIRE(harness.streamer_stats().cols_streamed == 2);
}

TEST_CASE("Streamer Harness journey tracking", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);
    harness.stream_rows(tile, 0);
    harness.run();

    auto journey_opt = harness.journey_tracker().get_journey(tile);
    REQUIRE(journey_opt.has_value());

    const auto& journey = journey_opt->get();
    REQUIRE(journey.tile_id == tile);
    REQUIRE(journey.l1_arrival > 0);
}

TEST_CASE("Streamer Harness statistics", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);
    harness.stream_rows(tile, 0);
    harness.run();

    const auto& stats = harness.streamer_stats();

    REQUIRE(stats.rows_streamed == 1);
    REQUIRE(stats.bytes_l2_to_l1 == 16 * 16 * 2);
    REQUIRE(stats.compute_operations > 0);
}

TEST_CASE("Streamer Harness reset", "[harness][streamer]") {
    StreamerHarnessConfig config;
    StreamerHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l2_pattern(0, tile, 16, 16, 2);
    harness.stream_rows(tile, 0);
    harness.run();

    REQUIRE(harness.streamer_stats().rows_streamed == 1);

    harness.reset();

    REQUIRE(harness.current_cycle() == 0);
    REQUIRE(harness.streamer_stats().rows_streamed == 0);
    REQUIRE_FALSE(harness.has_pending());
}
