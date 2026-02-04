// ============================================================================
// tests/harness/test_block_mover_harness.cpp
// Unit tests for BlockMover test harness
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/harness/block_mover_harness.hpp>

using namespace sw::kpu::harness;
using Catch::Approx;

TEST_CASE("BlockMover Harness construction", "[harness][block_mover]") {
    SECTION("Default configuration") {
        BlockMoverHarnessConfig config;
        BlockMoverHarness harness(config);

        REQUIRE(harness.current_cycle() == 0);
        REQUIRE(harness.block_mover_stats().tiles_moved == 0);
        REQUIRE_FALSE(harness.has_pending());
    }

    SECTION("Custom configuration") {
        BlockMoverHarnessConfig config;
        config.num_block_movers = 4;
        config.l3_buffers = 16;
        config.l2_banks = 32;

        BlockMoverHarness harness(config);
        REQUIRE(harness.credits_available() == 32);  // L2 banks
    }
}

TEST_CASE("BlockMover Harness L3 preloading", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    BlockMoverHarness harness(config);

    TileID tile{0, 0, 0};

    SECTION("Preload with raw data") {
        std::vector<uint8_t> data(256, 0xAB);
        harness.preload_l3(0, tile, data.data(), data.size());

        // Verify data is accessible (indirectly through transfer)
        harness.move_l3_to_l2(tile, 0);
        harness.run();
        REQUIRE(harness.block_mover_stats().tiles_moved == 1);
    }

    SECTION("Preload with pattern") {
        harness.preload_l3_pattern(0, tile, 16, 16, 2);  // 16x16 FP16

        harness.move_l3_to_l2(tile, 0);
        harness.run();
        REQUIRE(harness.block_mover_stats().tiles_moved == 1);
    }
}

TEST_CASE("BlockMover Harness L3 to L2 transfer", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    BlockMoverHarness harness(config);

    TileID tile{0, 1, 2};
    harness.preload_l3_pattern(0, tile, 16, 16, 2);

    SECTION("Basic transfer") {
        bool accepted = harness.move_l3_to_l2(tile, 0);
        REQUIRE(accepted);

        bool completed = harness.run();
        REQUIRE(completed);
        REQUIRE(harness.block_mover_stats().tiles_moved == 1);
        REQUIRE(harness.block_mover_stats().l3_to_l2_transfers == 1);
    }

    SECTION("Transfer with specific L2 bank") {
        harness.move_l3_to_l2(tile, 0, 5);  // Use L2 bank 5
        harness.run();

        REQUIRE(harness.tile_resident_l2(tile));
        REQUIRE(harness.l2_banks().is_occupied(5));
    }

    SECTION("Transfer with auto-allocated L2 bank") {
        harness.move_l3_to_l2(tile, 0, UINT32_MAX);  // Auto-allocate
        harness.run();

        REQUIRE(harness.tile_resident_l2(tile));
    }
}

TEST_CASE("BlockMover Harness transform operations", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    config.enable_transforms = true;
    BlockMoverHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l3_pattern(0, tile, 16, 16, 2);

    SECTION("Transpose transform") {
        harness.move_l3_to_l2(tile, 0, UINT32_MAX, Transform::TRANSPOSE);
        harness.run();

        REQUIRE(harness.block_mover_stats().tiles_moved == 1);
        REQUIRE(harness.block_mover_stats().get_transform_count(Transform::TRANSPOSE) == 1);
    }
}

TEST_CASE("BlockMover Harness multiple transfers", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    config.l3_buffers = 8;
    config.l2_banks = 16;
    BlockMoverHarness harness(config);

    SECTION("Sequential transfers") {
        for (uint32_t i = 0; i < 4; ++i) {
            TileID tile{0, 0, i};
            harness.preload_l3_pattern(i, tile, 16, 16, 2);
            harness.move_l3_to_l2(tile, i);
        }

        harness.run();
        REQUIRE(harness.block_mover_stats().tiles_moved == 4);
    }
}

TEST_CASE("BlockMover Harness journey tracking", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    BlockMoverHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l3_pattern(0, tile, 16, 16, 2);
    harness.move_l3_to_l2(tile, 0);
    harness.run();

    auto journey_opt = harness.journey_tracker().get_journey(tile);
    REQUIRE(journey_opt.has_value());

    const auto& journey = journey_opt->get();
    REQUIRE(journey.tile_id == tile);
    REQUIRE(journey.l2_arrival > 0);
}

TEST_CASE("BlockMover Harness statistics", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    BlockMoverHarness harness(config);

    // Move several tiles
    for (uint32_t i = 0; i < 4; ++i) {
        TileID tile{0, 0, i};
        harness.preload_l3_pattern(i, tile, 16, 16, 2);
        harness.move_l3_to_l2(tile, i);
    }
    harness.run();

    const auto& stats = harness.block_mover_stats();

    REQUIRE(stats.tiles_moved == 4);
    REQUIRE(stats.l3_to_l2_transfers == 4);
    REQUIRE(stats.bytes_moved == 4 * 16 * 16 * 2);
}

TEST_CASE("BlockMover Harness reset", "[harness][block_mover]") {
    BlockMoverHarnessConfig config;
    BlockMoverHarness harness(config);

    TileID tile{0, 0, 0};
    harness.preload_l3_pattern(0, tile, 16, 16, 2);
    harness.move_l3_to_l2(tile, 0);
    harness.run();

    REQUIRE(harness.block_mover_stats().tiles_moved == 1);

    harness.reset();

    REQUIRE(harness.current_cycle() == 0);
    REQUIRE(harness.block_mover_stats().tiles_moved == 0);
    REQUIRE_FALSE(harness.has_pending());
}
