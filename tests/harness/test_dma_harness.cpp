// ============================================================================
// tests/harness/test_dma_harness.cpp
// Unit tests for DMA engine test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/harness/dma_harness.hpp>

using namespace sw::kpu::harness;
using Catch::Approx;

TEST_CASE("DMA Harness construction", "[harness][dma]") {
    SECTION("Default configuration") {
        DMAHarnessConfig config;
        DMAHarness harness(config);

        REQUIRE(harness.current_cycle() == 0);
        REQUIRE(harness.tiles_transferred() == 0);
        REQUIRE(harness.credits_available() == config.l3_buffer_count);
        REQUIRE_FALSE(harness.has_pending());
    }

    SECTION("Custom configuration") {
        DMAHarnessConfig config;
        config.num_dma_engines = 4;
        config.l3_buffer_count = 16;
        config.tile_size = 1024;

        DMAHarness harness(config);
        REQUIRE(harness.credits_available() == 16);
    }
}

TEST_CASE("DMA Harness single tile load", "[harness][dma]") {
    DMAHarnessConfig config;
    config.l3_buffer_count = 8;
    DMAHarness harness(config);

    TileID tile{0, 0, 0};
    uint64_t dram_addr = 0x1000;

    SECTION("Request accepted") {
        bool accepted = harness.request_tile_load(tile, dram_addr);
        REQUIRE(accepted);
        REQUIRE(harness.has_pending());
    }

    SECTION("Transfer completes") {
        harness.request_tile_load(tile, dram_addr);
        bool completed = harness.run();

        REQUIRE(completed);
        REQUIRE_FALSE(harness.has_pending());
        REQUIRE(harness.dma_stats().tiles_loaded == 1);
        REQUIRE(harness.dma_stats().bytes_transferred == config.tile_size);
    }
}

TEST_CASE("DMA Harness multiple tile loads", "[harness][dma]") {
    DMAHarnessConfig config;
    config.l3_buffer_count = 8;
    config.num_dma_engines = 2;
    DMAHarness harness(config);

    SECTION("Load up to buffer limit") {
        // Request tiles up to L3 buffer limit
        for (uint32_t i = 0; i < config.l3_buffer_count; ++i) {
            TileID tile{0, i / 4, i % 4};
            harness.request_tile_load(tile, i * config.tile_size);
        }

        bool completed = harness.run();
        REQUIRE(completed);
        REQUIRE(harness.dma_stats().tiles_loaded == config.l3_buffer_count);
    }

    SECTION("Credit stalls when buffers exhausted") {
        // Request more tiles than buffers
        for (uint32_t i = 0; i < config.l3_buffer_count + 4; ++i) {
            TileID tile{0, i / 4, i % 4};
            harness.request_tile_load(tile, i * config.tile_size);
        }

        // Run but don't expect all to complete (no buffer releases)
        harness.run();

        // Should have loaded at most l3_buffer_count tiles
        REQUIRE(harness.dma_stats().tiles_loaded <= config.l3_buffer_count);
    }
}

TEST_CASE("DMA Harness tile journey tracking", "[harness][dma]") {
    DMAHarnessConfig config;
    DMAHarness harness(config);

    TileID tile{0, 1, 2};
    uint64_t dram_addr = 0x2000;

    harness.request_tile_load(tile, dram_addr);
    harness.run();

    SECTION("Journey recorded") {
        auto journey_opt = harness.journey_tracker().get_journey(tile);
        REQUIRE(journey_opt.has_value());

        const auto& journey = journey_opt->get();
        REQUIRE(journey.tile_id == tile);
        REQUIRE(journey.dram_request_cycle == 0);
        REQUIRE(journey.l3_arrival > 0);
    }
}

TEST_CASE("DMA Harness statistics", "[harness][dma]") {
    DMAHarnessConfig config;
    config.tile_size = 512;
    DMAHarness harness(config);

    // Load several tiles
    for (uint32_t i = 0; i < 4; ++i) {
        TileID tile{0, 0, i};
        harness.request_tile_load(tile, i * config.tile_size);
    }
    harness.run();

    const auto& stats = harness.dma_stats();

    SECTION("Transfer counts correct") {
        REQUIRE(stats.tiles_loaded == 4);
        REQUIRE(stats.tiles_stored == 0);
        REQUIRE(stats.transfers_completed == 4);
    }

    SECTION("Byte counts correct") {
        REQUIRE(stats.bytes_loaded == 4 * config.tile_size);
        REQUIRE(stats.bytes_transferred == 4 * config.tile_size);
    }
}

TEST_CASE("DMA Harness validation", "[harness][dma]") {
    DMAHarnessConfig config;
    DMAHarness harness(config);

    SECTION("Empty harness validates") {
        auto result = harness.validate();
        REQUIRE(result.passed);
    }

    SECTION("Completed transfers validate") {
        TileID tile{0, 0, 0};
        harness.request_tile_load(tile, 0x1000);
        harness.run();

        auto result = harness.validate();
        REQUIRE(result.passed);
    }

    SECTION("verify_stats helper works") {
        TileID tile{0, 0, 0};
        harness.request_tile_load(tile, 0x1000);
        harness.run();

        REQUIRE(harness.verify_stats(1, 0));  // 1 load, 0 stores
        REQUIRE_FALSE(harness.verify_stats(2, 0));  // Wrong count
    }
}

TEST_CASE("DMA Harness reset", "[harness][dma]") {
    DMAHarnessConfig config;
    DMAHarness harness(config);

    // Do some work
    TileID tile{0, 0, 0};
    harness.request_tile_load(tile, 0x1000);
    harness.run();

    REQUIRE(harness.dma_stats().tiles_loaded == 1);
    REQUIRE(harness.current_cycle() > 0);

    // Reset
    harness.reset();

    REQUIRE(harness.current_cycle() == 0);
    REQUIRE(harness.dma_stats().tiles_loaded == 0);
    REQUIRE(harness.credits_available() == config.l3_buffer_count);
    REQUIRE_FALSE(harness.has_pending());
}
