// ============================================================================
// tests/noc/noc_test.cpp
// NoC Unit Tests
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

#include "sw/kpu/noc/noc.hpp"

using namespace sw::kpu;
using namespace sw::kpu::noc;

// ============================================================================
// Basic NoC Tests
// ============================================================================

TEST_CASE("NoC Construction", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;
    config.link_latency = 1;
    config.router_latency = 1;

    NoC noc(config);

    REQUIRE(noc.config().rows == 4);
    REQUIRE(noc.config().cols == 4);
    REQUIRE(noc.config().num_routers() == 16);
    REQUIRE(noc.is_idle());
    REQUIRE(noc.packets_in_flight() == 0);
}

TEST_CASE("NoC Topology", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;

    NoC noc(config);
    noc.print_topology();

    // Test neighbor relationships
    REQUIRE(config.has_neighbor(0, PortDirection::EAST));
    REQUIRE(config.has_neighbor(0, PortDirection::SOUTH));
    REQUIRE_FALSE(config.has_neighbor(0, PortDirection::NORTH));  // Top edge
    REQUIRE_FALSE(config.has_neighbor(0, PortDirection::WEST));   // Left edge

    REQUIRE(config.get_neighbor_id(0, PortDirection::EAST) == 1);
    REQUIRE(config.get_neighbor_id(0, PortDirection::SOUTH) == 4);

    // Router 5 (center-ish)
    REQUIRE(config.has_neighbor(5, PortDirection::NORTH));
    REQUIRE(config.has_neighbor(5, PortDirection::SOUTH));
    REQUIRE(config.has_neighbor(5, PortDirection::EAST));
    REQUIRE(config.has_neighbor(5, PortDirection::WEST));
}

TEST_CASE("NoC Single Hop Transfer", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;
    config.link_latency = 1;
    config.router_latency = 1;
    config.flit_size_bytes = 64;

    NoC noc(config);

    // Track delivered packets
    std::vector<NoCPacket> delivered;
    noc.set_l3_delivery_callback(1, [&delivered](const NoCPacket& pkt, uint64_t) {
        delivered.push_back(pkt);
    });

    // Create a tile descriptor
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.k_tile = 0;
    tile.size = 256;  // 256 bytes = 4 flits

    // Inject transfer from router 0 to router 1 (single hop East)
    auto result = noc.inject_l3_to_l3(0, 1, tile, 0);
    REQUIRE(result == TransferResult::SUCCESS);
    REQUIRE_FALSE(noc.is_idle());

    // Step until delivered
    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 100) {
        noc.step(cycle++);
    }

    std::cout << "Single hop transfer completed in " << cycle << " cycles\n";
    std::cout << "Packets delivered: " << delivered.size() << "\n";

    REQUIRE(delivered.size() == 1);
    REQUIRE(delivered[0].dst_router == 1);
    // hops_taken includes routing at source and destination routers
    REQUIRE(delivered[0].hops_taken >= 1);
}

TEST_CASE("NoC Multi-Hop Transfer", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;
    config.link_latency = 1;
    config.router_latency = 1;

    NoC noc(config);

    std::vector<NoCPacket> delivered;
    noc.set_l3_delivery_callback(15, [&delivered](const NoCPacket& pkt, uint64_t) {
        delivered.push_back(pkt);
    });

    TileDescriptor tile;
    tile.tensor = TensorId::C;
    tile.m_tile = 3;
    tile.n_tile = 3;
    tile.size = 1024;

    // Transfer from router 0 to router 15 (corner to corner, 6 hops)
    auto result = noc.inject_l3_to_l3(0, 15, tile, 0);
    REQUIRE(result == TransferResult::SUCCESS);

    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 200) {
        noc.step(cycle++);
    }

    std::cout << "Multi-hop transfer (0→15) completed in " << cycle << " cycles\n";
    std::cout << "Packets delivered: " << delivered.size() << "\n";

    REQUIRE(delivered.size() == 1);
    // 3 East + 3 South = 6 network hops + routing overhead
    REQUIRE(delivered[0].hops_taken >= 6);
}

TEST_CASE("NoC Concurrent Transfers Without Contention", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;

    NoC noc(config);

    // Multiple transfers that don't share links
    // Row 0: 0→1
    // Row 1: 4→5
    // Row 2: 8→9
    // Row 3: 12→13

    int delivered_count = 0;
    for (uint8_t i : {1, 5, 9, 13}) {
        noc.set_l3_delivery_callback(i, [&delivered_count](const NoCPacket&, uint64_t) {
            delivered_count++;
        });
    }

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.size = 256;

    // Inject all 4 transfers at cycle 0
    REQUIRE(noc.inject_l3_to_l3(0, 1, tile, 0) == TransferResult::SUCCESS);
    REQUIRE(noc.inject_l3_to_l3(4, 5, tile, 0) == TransferResult::SUCCESS);
    REQUIRE(noc.inject_l3_to_l3(8, 9, tile, 0) == TransferResult::SUCCESS);
    REQUIRE(noc.inject_l3_to_l3(12, 13, tile, 0) == TransferResult::SUCCESS);

    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 100) {
        noc.step(cycle++);
    }

    std::cout << "4 parallel transfers (no contention) completed in " << cycle << " cycles\n";
    REQUIRE(delivered_count == 4);
}

TEST_CASE("NoC Concurrent Transfers With Contention", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;

    NoC noc(config);

    // Multiple transfers that DO share links (both going through router 0's East port)
    int delivered_count = 0;
    noc.set_l3_delivery_callback(1, [&delivered_count](const NoCPacket&, uint64_t) {
        delivered_count++;
    });
    noc.set_l3_delivery_callback(2, [&delivered_count](const NoCPacket&, uint64_t) {
        delivered_count++;
    });

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.size = 256;

    // Inject two transfers from router 0
    // Both want to use the East output port
    REQUIRE(noc.inject_l3_to_l3(0, 1, tile, 0) == TransferResult::SUCCESS);
    REQUIRE(noc.inject_l3_to_l3(0, 2, tile, 0) == TransferResult::SUCCESS);

    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 100) {
        noc.step(cycle++);
    }

    std::cout << "2 transfers with contention completed in " << cycle << " cycles\n";
    REQUIRE(delivered_count == 2);
}

TEST_CASE("NoC Latency Estimate", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;
    config.link_latency = 1;
    config.router_latency = 1;

    NoC noc(config);

    // Estimate latency for corner-to-corner transfer
    uint64_t est = noc.estimate_latency(0, 15, 1024);
    std::cout << "Estimated latency 0→15 (1KB): " << est << " cycles\n";

    // Should be: 1 (inject) + 6 * (1 + 1) (hops) + 16 (transfer) = ~30 cycles
    REQUIRE(est > 10);
    REQUIRE(est < 100);
}

TEST_CASE("NoC Statistics", "[noc]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;
    config.link_bandwidth = 64;
    config.input_buffer_flits = 16;  // Larger buffer to accept more packets

    NoC noc(config);

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.size = 256;

    // Do several transfers, checking if injection succeeds
    int injected = 0;
    for (int i = 0; i < 10; i++) {
        if (noc.inject_l3_to_l3(0, 1, tile, 0) == TransferResult::SUCCESS) {
            injected++;
        }
    }

    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 1000) {
        noc.step(cycle++);
    }

    auto& stats = noc.stats();
    std::cout << "NoC Statistics:\n";
    std::cout << "  Injected: " << injected << "\n";
    std::cout << "  Total packets: " << stats.total_packets << "\n";
    std::cout << "  Total bytes: " << stats.total_bytes << "\n";
    std::cout << "  Avg latency: " << stats.avg_latency() << " cycles\n";
    std::cout << "  Avg hops: " << stats.avg_hops() << "\n";

    // Should have injected all 10 packets with larger buffer
    REQUIRE(injected == 10);
    REQUIRE(stats.total_packets == 10);
    REQUIRE(stats.total_bytes == 2560);
}
