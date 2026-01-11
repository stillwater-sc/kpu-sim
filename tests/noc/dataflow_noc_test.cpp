// ============================================================================
// tests/noc/dataflow_noc_test.cpp
// Unit tests for the Unified Dataflow NoC
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/models/temporal/noc/dataflow_noc.hpp>

using namespace sw::kpu;
using namespace sw::kpu::noc;

// ============================================================================
// TileTag Tests
// ============================================================================

TEST_CASE("TileTag equality and hashing", "[dataflow_noc][tiletag]") {
    TileTag tag1(TensorId::A, 1, 2, 3);
    TileTag tag2(TensorId::A, 1, 2, 3);
    TileTag tag3(TensorId::B, 1, 2, 3);
    TileTag tag4(TensorId::A, 1, 2, 4);

    SECTION("Equal tags") {
        REQUIRE(tag1 == tag2);
        REQUIRE_FALSE(tag1 != tag2);
    }

    SECTION("Different tensor") {
        REQUIRE(tag1 != tag3);
    }

    SECTION("Different k_tile") {
        REQUIRE(tag1 != tag4);
    }

    SECTION("Hash consistency") {
        TileTagHash hasher;
        REQUIRE(hasher(tag1) == hasher(tag2));
    }

    SECTION("to_string") {
        REQUIRE(tag1.to_string() == "A[1,2,3]");
        REQUIRE(tag3.to_string() == "B[1,2,3]");
    }
}

TEST_CASE("TileTag from TileDescriptor", "[dataflow_noc][tiletag]") {
    TileDescriptor desc;
    desc.tensor = TensorId::C;
    desc.m_tile = 5;
    desc.n_tile = 6;
    desc.k_tile = 0;
    desc.size = 4096;

    TileTag tag(desc);

    REQUIRE(tag.tensor == TensorId::C);
    REQUIRE(tag.m_tile == 5);
    REQUIRE(tag.n_tile == 6);
    REQUIRE(tag.k_tile == 0);
}

// ============================================================================
// DataflowFlit Tests
// ============================================================================

TEST_CASE("DataflowFlit structure", "[dataflow_noc][flit]") {
    DataflowFlit flit;
    flit.tag = TileTag(TensorId::A, 0, 0, 0);
    flit.source = 0;
    flit.destination = 15;
    flit.flit_index = 0;
    flit.total_flits = 64;
    flit.tile_size = 4096;

    SECTION("First flit") {
        flit.flit_index = 0;
        REQUIRE(flit.is_first());
        REQUIRE_FALSE(flit.is_last());
        REQUIRE_FALSE(flit.is_single());
    }

    SECTION("Last flit") {
        flit.flit_index = 63;
        REQUIRE_FALSE(flit.is_first());
        REQUIRE(flit.is_last());
        REQUIRE_FALSE(flit.is_single());
    }

    SECTION("Single flit") {
        flit.total_flits = 1;
        flit.flit_index = 0;
        REQUIRE(flit.is_first());
        REQUIRE(flit.is_last());
        REQUIRE(flit.is_single());
    }

    SECTION("to_string") {
        std::string s = flit.to_string();
        REQUIRE(s.find("A[0,0,0]") != std::string::npos);
        REQUIRE(s.find("L3[15]") != std::string::npos);
    }
}

TEST_CASE("Flit size calculations", "[dataflow_noc][flit]") {
    SECTION("Small tile - single flit") {
        REQUIRE(flits_for_tile(1) == 1);
        REQUIRE(flits_for_tile(DataflowFlit::PAYLOAD_SIZE) == 1);
    }

    SECTION("Tile requiring multiple flits") {
        REQUIRE(flits_for_tile(DataflowFlit::PAYLOAD_SIZE + 1) == 2);
        REQUIRE(flits_for_tile(DataflowFlit::PAYLOAD_SIZE * 10) == 10);
    }

    SECTION("4KB tile") {
        uint16_t flits = flits_for_tile(4096);
        // 4096 / PAYLOAD_SIZE rounded up
        uint16_t expected = (4096 + DataflowFlit::PAYLOAD_SIZE - 1) / DataflowFlit::PAYLOAD_SIZE;
        REQUIRE(flits == expected);
    }
}

// ============================================================================
// DataflowFlitBuffer Tests
// ============================================================================

TEST_CASE("DataflowFlitBuffer operations", "[dataflow_noc][buffer]") {
    DataflowFlitBuffer buffer(4);

    SECTION("Empty buffer") {
        REQUIRE(buffer.empty());
        REQUIRE(buffer.can_accept());
        REQUIRE(buffer.size() == 0);
        REQUIRE(buffer.capacity() == 4);
    }

    SECTION("Push and pop") {
        DataflowFlit flit;
        flit.flit_index = 42;

        REQUIRE(buffer.push(flit));
        REQUIRE_FALSE(buffer.empty());
        REQUIRE(buffer.size() == 1);

        DataflowFlit popped = buffer.pop();
        REQUIRE(popped.flit_index == 42);
        REQUIRE(buffer.empty());
    }

    SECTION("Back-pressure when full") {
        DataflowFlit flit;
        for (int i = 0; i < 4; ++i) {
            REQUIRE(buffer.can_accept());
            REQUIRE(buffer.push(flit));
        }

        REQUIRE_FALSE(buffer.can_accept());
        REQUIRE_FALSE(buffer.push(flit));  // Should fail
        REQUIRE(buffer.size() == 4);
    }
}

// ============================================================================
// PartialTile Tests
// ============================================================================

TEST_CASE("PartialTile reassembly", "[dataflow_noc][reassembly]") {
    PartialTile partial;

    SECTION("Initially empty") {
        REQUIRE(partial.empty());
        REQUIRE_FALSE(partial.complete());
    }

    SECTION("Single flit tile") {
        partial.init(1, 32);

        DataflowFlit flit;
        flit.flit_index = 0;
        flit.total_flits = 1;

        REQUIRE(partial.add_flit(flit));
        REQUIRE(partial.complete());
    }

    SECTION("Multi-flit tile") {
        partial.init(4, 256);

        for (uint16_t i = 0; i < 4; ++i) {
            DataflowFlit flit;
            flit.flit_index = i;
            flit.total_flits = 4;

            REQUIRE_FALSE(partial.complete());
            REQUIRE(partial.add_flit(flit));
        }

        REQUIRE(partial.complete());
        REQUIRE(partial.flits().size() == 4);
    }

    SECTION("Out-of-order flits") {
        partial.init(3, 128);

        DataflowFlit flit;
        flit.total_flits = 3;

        // Add in reverse order
        flit.flit_index = 2;
        REQUIRE(partial.add_flit(flit));
        REQUIRE_FALSE(partial.complete());

        flit.flit_index = 0;
        REQUIRE(partial.add_flit(flit));
        REQUIRE_FALSE(partial.complete());

        flit.flit_index = 1;
        REQUIRE(partial.add_flit(flit));
        REQUIRE(partial.complete());
    }

    SECTION("Duplicate flit rejected") {
        partial.init(2, 64);

        DataflowFlit flit;
        flit.flit_index = 0;
        flit.total_flits = 2;

        REQUIRE(partial.add_flit(flit));
        REQUIRE_FALSE(partial.add_flit(flit));  // Duplicate
    }
}

// ============================================================================
// DataflowRouter Tests
// ============================================================================

TEST_CASE("DataflowRouter configuration", "[dataflow_noc][router]") {
    DataflowRouter::Config config;
    config.id = 5;
    config.row = 1;
    config.col = 1;
    config.mesh_rows = 4;
    config.mesh_cols = 4;

    DataflowRouter router(config);

    REQUIRE(router.id() == 5);
    REQUIRE(router.row() == 1);
    REQUIRE(router.col() == 1);
}

TEST_CASE("DataflowRouter local delivery", "[dataflow_noc][router]") {
    DataflowRouter::Config config;
    config.id = 5;
    config.row = 1;
    config.col = 1;
    config.mesh_rows = 4;
    config.mesh_cols = 4;

    DataflowRouter router(config);

    // Create a single-flit tile destined for this router
    DataflowFlit flit;
    flit.tag = TileTag(TensorId::A, 0, 0, 0);
    flit.destination = 5;  // This router
    flit.flit_index = 0;
    flit.total_flits = 1;
    flit.tile_size = 32;

    // Inject
    REQUIRE(router.inject(Direction::LOCAL, flit));

    // Process
    router.step(0);

    // Should be delivered
    REQUIRE(router.has_complete_tile());

    TileTag delivered_tag;
    auto flits = router.pop_complete_tile(&delivered_tag);
    REQUIRE(flits.size() == 1);
    REQUIRE(delivered_tag == flit.tag);
}

TEST_CASE("DataflowRouter forwarding", "[dataflow_noc][router]") {
    // Create a 2x2 mesh: routers 0,1,2,3
    //   [0] [1]
    //   [2] [3]

    DataflowRouter::Config config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;

    std::vector<DataflowRouter> routers;
    for (uint8_t i = 0; i < 4; ++i) {
        config.id = i;
        config.row = i / 2;
        config.col = i % 2;
        routers.emplace_back(config);
    }

    // Connect neighbors
    routers[0].set_neighbor(Direction::EAST, &routers[1]);
    routers[0].set_neighbor(Direction::SOUTH, &routers[2]);
    routers[1].set_neighbor(Direction::WEST, &routers[0]);
    routers[1].set_neighbor(Direction::SOUTH, &routers[3]);
    routers[2].set_neighbor(Direction::NORTH, &routers[0]);
    routers[2].set_neighbor(Direction::EAST, &routers[3]);
    routers[3].set_neighbor(Direction::NORTH, &routers[1]);
    routers[3].set_neighbor(Direction::WEST, &routers[2]);

    SECTION("Forward from 0 to 1 (EAST)") {
        DataflowFlit flit;
        flit.tag = TileTag(TensorId::A, 0, 0, 0);
        flit.destination = 1;
        flit.flit_index = 0;
        flit.total_flits = 1;
        flit.tile_size = 32;

        // Inject at router 0
        REQUIRE(routers[0].inject(Direction::LOCAL, flit));

        // Step router 0 - should forward to router 1
        routers[0].step(0);

        // Router 1 should have received it
        routers[1].step(1);

        REQUIRE(routers[1].has_complete_tile());
        REQUIRE_FALSE(routers[0].has_complete_tile());
    }

    SECTION("Forward from 0 to 3 (EAST then SOUTH)") {
        DataflowFlit flit;
        flit.tag = TileTag(TensorId::B, 1, 1, 1);
        flit.destination = 3;
        flit.flit_index = 0;
        flit.total_flits = 1;
        flit.tile_size = 32;

        // Inject at router 0
        REQUIRE(routers[0].inject(Direction::LOCAL, flit));

        // Step 0 - forwards EAST to 1
        routers[0].step(0);

        // Step 1 - forwards SOUTH to 3
        routers[1].step(1);

        // Step 3 - delivers locally
        routers[3].step(2);

        REQUIRE(routers[3].has_complete_tile());
    }
}

// ============================================================================
// DataflowNoC Integration Tests
// ============================================================================

TEST_CASE("DataflowNoC construction", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    DataflowNoC noc(config);

    REQUIRE(noc.rows() == 4);
    REQUIRE(noc.cols() == 4);
    REQUIRE(noc.num_routers() == 16);
}

TEST_CASE("DataflowNoC same-router transfer", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 2;
    config.cols = 2;

    DataflowNoC noc(config);

    bool delivered = false;
    noc.set_delivery_callback([&](uint8_t dst, const TileTag& tag,
                                   uint64_t inject, uint64_t complete) {
        delivered = true;
        REQUIRE(dst == 0);
        REQUIRE(tag.tensor == TensorId::A);
        REQUIRE(complete >= inject);
    });

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.size = 64;

    uint64_t cycle = 0;
    REQUIRE(noc.inject_tile(0, 0, tile, cycle));

    // Run until delivered
    noc.drain(cycle);

    REQUIRE(delivered);
    REQUIRE(noc.stats().total_tiles_delivered == 1);
}

TEST_CASE("DataflowNoC single-hop transfer", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 2;
    config.cols = 2;

    DataflowNoC noc(config);

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.size = 256;

    uint8_t delivered_at = 255;
    noc.set_delivery_callback([&](uint8_t dst, const TileTag&,
                                   uint64_t, uint64_t) {
        delivered_at = dst;
    });

    // Router 0 to Router 1 (EAST)
    uint64_t cycle = 0;
    REQUIRE(noc.inject_tile(0, 1, tile, cycle));

    noc.drain(cycle);

    REQUIRE(delivered_at == 1);
    REQUIRE(noc.stats().total_tiles_delivered == 1);
}

TEST_CASE("DataflowNoC multi-hop transfer", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    DataflowNoC noc(config);

    // Transfer from corner (0,0) to corner (3,3)
    // That's 6 hops: 3 EAST + 3 SOUTH

    TileDescriptor tile;
    tile.tensor = TensorId::C;
    tile.m_tile = 3;
    tile.n_tile = 3;
    tile.k_tile = 0;
    tile.size = 1024;

    uint8_t src = noc.router_id(0, 0);  // 0
    uint8_t dst = noc.router_id(3, 3);  // 15

    uint64_t delivered_cycle = 0;
    noc.set_delivery_callback([&](uint8_t, const TileTag&,
                                   uint64_t inject, uint64_t complete) {
        delivered_cycle = complete;
        // Latency should be at least the number of hops
        uint64_t latency = complete - inject;
        REQUIRE(latency >= 6);  // At least 6 cycles for 6 hops
    });

    uint64_t cycle = 0;
    REQUIRE(noc.inject_tile(src, dst, tile, cycle));

    noc.drain(cycle);

    REQUIRE(noc.stats().total_tiles_delivered == 1);
    REQUIRE(delivered_cycle > 0);
}

TEST_CASE("DataflowNoC multiple concurrent transfers", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    DataflowNoC noc(config);

    size_t deliveries = 0;
    noc.set_delivery_callback([&](uint8_t, const TileTag&,
                                   uint64_t, uint64_t) {
        deliveries++;
    });

    // Inject tiles from different sources to different destinations
    std::vector<std::pair<uint8_t, uint8_t>> transfers = {
        {0, 15},   // Corner to corner
        {3, 12},   // Top-right to bottom-left
        {5, 10},   // Interior transfers
        {7, 8},
    };

    uint64_t cycle = 0;
    for (size_t i = 0; i < transfers.size(); ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(i);
        tile.n_tile = 0;
        tile.k_tile = 0;
        tile.size = 512;

        REQUIRE(noc.inject_tile(transfers[i].first, transfers[i].second, tile, cycle));
    }

    noc.drain(cycle);

    REQUIRE(deliveries == transfers.size());
    REQUIRE(noc.stats().total_tiles_delivered == transfers.size());
}

TEST_CASE("DataflowNoC statistics", "[dataflow_noc][noc]") {
    DataflowNoC::Config config;
    config.rows = 2;
    config.cols = 2;

    DataflowNoC noc(config);

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.size = 256;  // Multiple flits

    uint64_t cycle = 0;

    // Inject a few tiles
    for (int i = 0; i < 3; ++i) {
        tile.m_tile = static_cast<uint16_t>(i);
        noc.inject_tile(0, 3, tile, cycle);
    }

    noc.drain(cycle);

    const auto& stats = noc.stats();
    REQUIRE(stats.total_tiles_injected == 3);
    REQUIRE(stats.total_tiles_delivered == 3);
    REQUIRE(stats.total_flits_transferred > 0);
    REQUIRE(stats.max_latency_cycles > 0);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_CASE("Manhattan distance calculation", "[dataflow_noc][util]") {
    const uint8_t cols = 4;

    SECTION("Same router") {
        REQUIRE(manhattan_distance(0, 0, cols) == 0);
        REQUIRE(manhattan_distance(5, 5, cols) == 0);
    }

    SECTION("Adjacent routers") {
        // 0 is at (0,0), 1 is at (0,1) - distance 1
        REQUIRE(manhattan_distance(0, 1, cols) == 1);
        // 0 is at (0,0), 4 is at (1,0) - distance 1
        REQUIRE(manhattan_distance(0, 4, cols) == 1);
    }

    SECTION("Corner to corner") {
        // 0 is at (0,0), 15 is at (3,3) - distance 6
        REQUIRE(manhattan_distance(0, 15, cols) == 6);
    }

    SECTION("Symmetry") {
        REQUIRE(manhattan_distance(0, 15, cols) == manhattan_distance(15, 0, cols));
        REQUIRE(manhattan_distance(3, 12, cols) == manhattan_distance(12, 3, cols));
    }
}

TEST_CASE("Direction utilities", "[dataflow_noc][util]") {
    REQUIRE(opposite(Direction::NORTH) == Direction::SOUTH);
    REQUIRE(opposite(Direction::SOUTH) == Direction::NORTH);
    REQUIRE(opposite(Direction::EAST) == Direction::WEST);
    REQUIRE(opposite(Direction::WEST) == Direction::EAST);
    REQUIRE(opposite(Direction::LOCAL) == Direction::LOCAL);

    REQUIRE(std::string(to_string(Direction::NORTH)) == "NORTH");
    REQUIRE(std::string(to_string(Direction::SOUTH)) == "SOUTH");
    REQUIRE(std::string(to_string(Direction::EAST)) == "EAST");
    REQUIRE(std::string(to_string(Direction::WEST)) == "WEST");
    REQUIRE(std::string(to_string(Direction::LOCAL)) == "LOCAL");
}
