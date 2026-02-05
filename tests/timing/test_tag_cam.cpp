// ============================================================================
// tests/timing/test_tag_cam.cpp
// Unit tests for TagCAM
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/tag_cam.hpp>

using namespace sw::kpu::timing;
using namespace sw::kpu::isa;

// ============================================================================
// Helper functions
// ============================================================================

static TileID make_tile(MatrixID matrix, Size ti, Size tj, Size tk = 0) {
    return TileID{matrix, ti, tj, tk};
}

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST_CASE("TagCAM construction", "[timing][tag_cam]") {
    TagCAM cam(8);

    REQUIRE(cam.capacity() == 8);
    REQUIRE(cam.size() == 0);
    REQUIRE(cam.empty());
    REQUIRE_FALSE(cam.full());
}

TEST_CASE("TagCAM insert", "[timing][tag_cam]") {
    SECTION("Insert adds entry") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        REQUIRE(cam.insert(tile, 2, 100));
        REQUIRE(cam.size() == 1);
        REQUIRE_FALSE(cam.empty());
    }

    SECTION("Insert rejects duplicate") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        REQUIRE(cam.insert(tile, 2, 100));
        REQUIRE_FALSE(cam.insert(tile, 3, 200));  // Duplicate
        REQUIRE(cam.size() == 1);
    }

    SECTION("Insert rejects when full") {
        TagCAM cam(2);

        REQUIRE(cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100));
        REQUIRE(cam.insert(make_tile(MatrixID::A, 0, 1), 1, 101));
        REQUIRE(cam.full());

        REQUIRE_FALSE(cam.insert(make_tile(MatrixID::A, 1, 0), 0, 102));
    }
}

TEST_CASE("TagCAM lookup", "[timing][tag_cam]") {
    SECTION("Lookup finds inserted tile") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::B, 1, 2);

        cam.insert(tile, 3, 500);
        REQUIRE(cam.lookup(tile));
    }

    SECTION("Lookup returns false for missing") {
        TagCAM cam(4);
        TileID present = make_tile(MatrixID::A, 0, 0);
        TileID missing = make_tile(MatrixID::A, 0, 1);

        cam.insert(present, 0, 100);
        REQUIRE(cam.lookup(present));
        REQUIRE_FALSE(cam.lookup(missing));
    }
}

// ============================================================================
// Match Tests
// ============================================================================

TEST_CASE("TagCAM match", "[timing][tag_cam]") {
    SECTION("Match returns full entry") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::C, 2, 3, 1);

        cam.insert(tile, 5, 750);

        auto entry = cam.match(tile);
        REQUIRE(entry.has_value());
        REQUIRE(entry->tile_id == tile);
        REQUIRE(entry->slot_id == 5);
        REQUIRE(entry->arrival_cycle == 750);
        REQUIRE(entry->valid);
    }

    SECTION("Match returns nullopt for missing") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        REQUIRE_FALSE(cam.match(tile).has_value());
    }
}

TEST_CASE("TagCAM get_slot returns slot id", "[timing][tag_cam]") {
    TagCAM cam(4);
    TileID tile = make_tile(MatrixID::B, 0, 0);

    cam.insert(tile, 7, 100);
    auto slot = cam.get_slot(tile);
    REQUIRE(slot.has_value());
    REQUIRE(*slot == 7);
}

TEST_CASE("TagCAM get_arrival_cycle returns arrival", "[timing][tag_cam]") {
    TagCAM cam(4);
    TileID tile = make_tile(MatrixID::A, 1, 1);

    cam.insert(tile, 0, 12345);
    auto cycle = cam.get_arrival_cycle(tile);
    REQUIRE(cycle.has_value());
    REQUIRE(*cycle == 12345);
}

// ============================================================================
// Invalidate Tests
// ============================================================================

TEST_CASE("TagCAM invalidate", "[timing][tag_cam]") {
    SECTION("Invalidate removes entry") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        cam.insert(tile, 0, 100);
        REQUIRE(cam.lookup(tile));

        REQUIRE(cam.invalidate(tile));
        REQUIRE_FALSE(cam.lookup(tile));
        REQUIRE(cam.size() == 0);
    }

    SECTION("Invalidate returns false for missing") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        REQUIRE_FALSE(cam.invalidate(tile));
    }

    SECTION("Invalidate allows reinsert") {
        TagCAM cam(4);
        TileID tile = make_tile(MatrixID::A, 0, 0);

        cam.insert(tile, 0, 100);
        cam.invalidate(tile);

        // Should be able to insert again
        REQUIRE(cam.insert(tile, 1, 200));
        auto entry = cam.match(tile);
        REQUIRE(entry.has_value());
        REQUIRE(entry->slot_id == 1);
        REQUIRE(entry->arrival_cycle == 200);
    }
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_CASE("TagCAM reset", "[timing][tag_cam]") {
    TagCAM cam(4);

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100);
    cam.insert(make_tile(MatrixID::B, 0, 0), 1, 101);
    cam.insert(make_tile(MatrixID::C, 0, 0), 2, 102);
    REQUIRE(cam.size() == 3);

    cam.reset();

    REQUIRE(cam.size() == 0);
    REQUIRE(cam.empty());
    REQUIRE_FALSE(cam.lookup(make_tile(MatrixID::A, 0, 0)));
}

// ============================================================================
// Work-Conserving Helper Tests
// ============================================================================

TEST_CASE("TagCAM find_any_ready", "[timing][tag_cam]") {
    TagCAM cam(4);

    REQUIRE_FALSE(cam.find_any_ready().has_value());

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100);

    auto entry = cam.find_any_ready();
    REQUIRE(entry.has_value());
    REQUIRE(entry->tile_id.matrix == MatrixID::A);
}

TEST_CASE("TagCAM find_oldest", "[timing][tag_cam]") {
    TagCAM cam(8);

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 300);
    cam.insert(make_tile(MatrixID::B, 0, 0), 1, 100);  // Oldest
    cam.insert(make_tile(MatrixID::C, 0, 0), 2, 200);

    auto oldest = cam.find_oldest();
    REQUIRE(oldest.has_value());
    REQUIRE(oldest->tile_id.matrix == MatrixID::B);
    REQUIRE(oldest->arrival_cycle == 100);
}

TEST_CASE("TagCAM find_by_matrix", "[timing][tag_cam]") {
    TagCAM cam(8);

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100);
    cam.insert(make_tile(MatrixID::A, 0, 1), 1, 101);
    cam.insert(make_tile(MatrixID::B, 0, 0), 2, 102);
    cam.insert(make_tile(MatrixID::A, 1, 0), 3, 103);
    cam.insert(make_tile(MatrixID::C, 0, 0), 4, 104);

    auto a_tiles = cam.find_by_matrix(MatrixID::A);
    REQUIRE(a_tiles.size() == 3);

    auto b_tiles = cam.find_by_matrix(MatrixID::B);
    REQUIRE(b_tiles.size() == 1);

    auto c_tiles = cam.find_by_matrix(MatrixID::C);
    REQUIRE(c_tiles.size() == 1);
}

TEST_CASE("TagCAM all_entries", "[timing][tag_cam]") {
    TagCAM cam(4);

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100);
    cam.insert(make_tile(MatrixID::B, 0, 0), 1, 101);

    auto entries = cam.all_entries();
    REQUIRE(entries.size() == 2);
}

// ============================================================================
// TileID Differentiation Tests
// ============================================================================

TEST_CASE("TagCAM differentiates by matrix", "[timing][tag_cam]") {
    TagCAM cam(8);

    TileID a = {MatrixID::A, 0, 0, 0};
    TileID b = {MatrixID::B, 0, 0, 0};
    TileID c = {MatrixID::C, 0, 0, 0};

    cam.insert(a, 0, 100);
    cam.insert(b, 1, 101);
    cam.insert(c, 2, 102);

    REQUIRE(cam.lookup(a));
    REQUIRE(cam.lookup(b));
    REQUIRE(cam.lookup(c));

    REQUIRE(cam.get_slot(a) == 0);
    REQUIRE(cam.get_slot(b) == 1);
    REQUIRE(cam.get_slot(c) == 2);
}

TEST_CASE("TagCAM differentiates by tile coords", "[timing][tag_cam]") {
    TagCAM cam(16);

    // Insert tiles at different coordinates
    for (Size i = 0; i < 4; ++i) {
        for (Size j = 0; j < 4; ++j) {
            TileID tile = {MatrixID::A, i, j, 0};
            uint32_t slot = i * 4 + j;
            cam.insert(tile, slot, 100 + slot);
        }
    }

    REQUIRE(cam.size() == 16);
    REQUIRE(cam.full());

    // Verify each can be found
    for (Size i = 0; i < 4; ++i) {
        for (Size j = 0; j < 4; ++j) {
            TileID tile = {MatrixID::A, i, j, 0};
            uint32_t expected_slot = i * 4 + j;

            REQUIRE(cam.lookup(tile));
            REQUIRE(cam.get_slot(tile) == expected_slot);
        }
    }
}

TEST_CASE("TagCAM differentiates by K dimension", "[timing][tag_cam]") {
    TagCAM cam(8);

    // Same ti, tj but different tk (accumulation dimension)
    for (Size k = 0; k < 8; ++k) {
        TileID tile = {MatrixID::A, 0, 0, k};
        cam.insert(tile, k, 100 + k);
    }

    REQUIRE(cam.size() == 8);

    // Each k value should be distinct
    for (Size k = 0; k < 8; ++k) {
        TileID tile = {MatrixID::A, 0, 0, k};
        REQUIRE(cam.lookup(tile));
        REQUIRE(cam.get_slot(tile) == k);
        REQUIRE(cam.get_arrival_cycle(tile) == 100 + k);
    }
}

// ============================================================================
// Dataflow Scenario Tests
// ============================================================================

TEST_CASE("TagCAM DMA to BlockMover scenario", "[timing][tag_cam]") {
    // Simulate: DMA loads tiles to L3, BlockMover moves them to L2

    TagCAM l3_cam(4);  // L3 has 4 buffers

    // DMA loads 3 tiles
    l3_cam.insert(make_tile(MatrixID::A, 0, 0), 0, 100);
    l3_cam.insert(make_tile(MatrixID::A, 0, 1), 1, 150);
    l3_cam.insert(make_tile(MatrixID::B, 0, 0), 2, 200);

    // BlockMover checks for tile A[0,0] - should be ready
    TileID needed = make_tile(MatrixID::A, 0, 0);
    REQUIRE(l3_cam.lookup(needed));

    // BlockMover gets arrival time for timing calculation
    auto arrival = l3_cam.get_arrival_cycle(needed);
    REQUIRE(*arrival == 100);

    // BlockMover moves tile - invalidate and free buffer
    l3_cam.invalidate(needed);
    REQUIRE_FALSE(l3_cam.lookup(needed));

    // Now DMA can reuse buffer 0
    REQUIRE(l3_cam.insert(make_tile(MatrixID::A, 1, 0), 0, 300));
}

TEST_CASE("TagCAM work-conserving scan", "[timing][tag_cam]") {
    // BlockMover can process any ready tile, not just a specific one

    TagCAM cam(4);

    cam.insert(make_tile(MatrixID::A, 0, 0), 0, 500);
    cam.insert(make_tile(MatrixID::A, 1, 0), 1, 100);  // Oldest
    cam.insert(make_tile(MatrixID::B, 0, 0), 2, 300);

    // Work-conserving: find any ready tile
    auto any = cam.find_any_ready();
    REQUIRE(any.has_value());

    // Priority aging: process oldest first
    auto oldest = cam.find_oldest();
    REQUIRE(oldest.has_value());
    REQUIRE(oldest->tile_id.ti == 1);
    REQUIRE(oldest->arrival_cycle == 100);

    // Process the oldest tile
    cam.invalidate(oldest->tile_id);

    // Next oldest
    oldest = cam.find_oldest();
    REQUIRE(oldest.has_value());
    REQUIRE(oldest->tile_id.matrix == MatrixID::B);
    REQUIRE(oldest->arrival_cycle == 300);
}
