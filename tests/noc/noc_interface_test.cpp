// ============================================================================
// tests/noc/noc_interface_test.cpp
// Tests for the INoC interface and adapters
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sw/kpu/models/interfaces/noc_interface.hpp>
#include <sw/kpu/models/temporal/noc/noc_adapters.hpp>

using namespace sw::kpu;
using namespace sw::kpu::noc;

// ============================================================================
// Factory Tests
// ============================================================================

TEST_CASE("NoC factory creates correct types", "[noc_interface][factory]") {
    NoCConfig config;
    config.rows = 4;
    config.cols = 4;

    SECTION("Create WormholeNoC") {
        auto noc = create_noc(NoCType::WORMHOLE, config);
        REQUIRE(noc != nullptr);
        REQUIRE(noc->type() == NoCType::WORMHOLE);
        REQUIRE(noc->type_name() == "WORMHOLE");
        REQUIRE(noc->num_routers() == 16);
    }

    SECTION("Create DataflowNoC") {
        auto noc = create_noc(NoCType::DATAFLOW, config);
        REQUIRE(noc != nullptr);
        REQUIRE(noc->type() == NoCType::DATAFLOW);
        REQUIRE(noc->type_name() == "DATAFLOW");
        REQUIRE(noc->num_routers() == 16);
    }

    SECTION("Create with default config") {
        auto wh = create_noc(NoCType::WORMHOLE);
        auto df = create_noc(NoCType::DATAFLOW);
        REQUIRE(wh != nullptr);
        REQUIRE(df != nullptr);
        REQUIRE(wh->num_routers() == 16);  // Default 4x4
        REQUIRE(df->num_routers() == 16);
    }
}

// ============================================================================
// Common Interface Tests (Parameterized by NoC Type)
// ============================================================================

TEST_CASE("INoC common interface", "[noc_interface]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);
    REQUIRE(noc != nullptr);

    SECTION("Initial state") {
        REQUIRE(noc->is_idle());
        REQUIRE(noc->num_routers() == 4);

        auto stats = noc->stats();
        REQUIRE(stats.tiles_injected == 0);
        REQUIRE(stats.tiles_delivered == 0);
    }

    SECTION("Config access") {
        const auto& cfg = noc->config();
        REQUIRE(cfg.rows == 2);
        REQUIRE(cfg.cols == 2);
    }

    SECTION("Can inject at valid routers") {
        for (uint8_t i = 0; i < 4; ++i) {
            REQUIRE(noc->can_inject(i));
        }
    }
}

// ============================================================================
// Same-Router Transfer Tests
// ============================================================================

TEST_CASE("INoC same-router transfer", "[noc_interface][transfer]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.size = 64;

    bool delivered = false;
    uint8_t delivered_dst = 255;

    noc->set_delivery_callback(0, [&](const TileDescriptor& t, uint8_t src, uint8_t dst,
                                       uint64_t inject, uint64_t complete) {
        delivered = true;
        delivered_dst = dst;
        REQUIRE(t.tensor == TensorId::A);
        REQUIRE(dst == 0);
    });

    auto result = noc->inject_tile(0, 0, tile, 0);
    REQUIRE(result == InjectResult::SUCCESS);

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(delivered);
    REQUIRE(delivered_dst == 0);

    auto stats = noc->stats();
    REQUIRE(stats.tiles_injected == 1);
    REQUIRE(stats.tiles_delivered == 1);
}

// ============================================================================
// Neighbor Transfer Tests
// ============================================================================

TEST_CASE("INoC neighbor transfer", "[noc_interface][transfer]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);

    TileDescriptor tile;
    tile.tensor = TensorId::B;
    tile.m_tile = 1;
    tile.n_tile = 2;
    tile.k_tile = 0;
    tile.size = 256;

    bool delivered = false;
    uint8_t delivered_src = 255;
    uint8_t delivered_dst = 255;

    noc->set_delivery_callback(1, [&](const TileDescriptor& t, uint8_t src, uint8_t dst,
                                       uint64_t inject, uint64_t complete) {
        delivered = true;
        delivered_src = src;
        delivered_dst = dst;
        REQUIRE(t.tensor == TensorId::B);
        REQUIRE(t.m_tile == 1);
        REQUIRE(t.n_tile == 2);
    });

    // Router 0 to Router 1 (EAST)
    auto result = noc->inject_tile(0, 1, tile, 0);
    REQUIRE(result == InjectResult::SUCCESS);

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(delivered);
    REQUIRE(delivered_src == 0);
    REQUIRE(delivered_dst == 1);
}

// ============================================================================
// Multi-Hop Transfer Tests
// ============================================================================

TEST_CASE("INoC multi-hop transfer (DataflowNoC only)", "[noc_interface][transfer][dataflow]") {
    // Note: WormholeNoC only supports single-hop nearest-neighbor routing
    // Multi-hop tests only run with DataflowNoC

    NoCConfig config;
    config.rows = 4;
    config.cols = 4;

    auto noc = create_noc(NoCType::DATAFLOW, config);

    TileDescriptor tile;
    tile.tensor = TensorId::C;
    tile.m_tile = 3;
    tile.n_tile = 3;
    tile.k_tile = 0;
    tile.size = 1024;

    bool delivered = false;
    uint64_t delivery_cycle = 0;

    // Corner to corner: router 0 (0,0) to router 15 (3,3)
    noc->set_delivery_callback(15, [&](const TileDescriptor& t, uint8_t src, uint8_t dst,
                                        uint64_t inject, uint64_t complete) {
        delivered = true;
        delivery_cycle = complete;
        REQUIRE(dst == 15);
    });

    auto result = noc->inject_tile(0, 15, tile, 0);
    REQUIRE(result == InjectResult::SUCCESS);

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(delivered);
    REQUIRE(delivery_cycle > 0);  // Should take some time

    auto stats = noc->stats();
    REQUIRE(stats.tiles_delivered == 1);
    REQUIRE(stats.max_latency_cycles > 0);
}

// ============================================================================
// Global Callback Tests
// ============================================================================

TEST_CASE("INoC global delivery callback (DataflowNoC)", "[noc_interface][callback][dataflow]") {
    // Test global callback with DataflowNoC which supports multi-hop

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(NoCType::DATAFLOW, config);

    size_t delivery_count = 0;

    noc->set_global_delivery_callback([&](const TileDescriptor& t, uint8_t src, uint8_t dst,
                                           uint64_t inject, uint64_t complete) {
        delivery_count++;
    });

    // Inject tiles to different destinations
    for (uint8_t i = 0; i < 4; ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = i;
        tile.n_tile = 0;
        tile.k_tile = 0;
        tile.size = 64;

        noc->inject_tile(0, i, tile, 0);
    }

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(delivery_count == 4);
}

TEST_CASE("INoC global delivery callback (WormholeNoC)", "[noc_interface][callback][wormhole]") {
    // WormholeNoC only supports single-hop, test single transfer
    // Note: WormholeNoC has limited concurrent injection capability

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(NoCType::WORMHOLE, config);

    size_t delivery_count = 0;

    noc->set_global_delivery_callback([&](const TileDescriptor& t, uint8_t src, uint8_t dst,
                                           uint64_t inject, uint64_t complete) {
        delivery_count++;
    });

    // Test single transfer
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.size = 64;

    noc->inject_tile(0, 1, tile, 0);

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(delivery_count == 1);
}

// ============================================================================
// Multiple Concurrent Transfers Tests
// ============================================================================

TEST_CASE("INoC concurrent transfers (DataflowNoC)", "[noc_interface][transfer][dataflow]") {
    // DataflowNoC supports multi-hop, so test with various source-dest pairs

    NoCConfig config;
    config.rows = 4;
    config.cols = 4;

    auto noc = create_noc(NoCType::DATAFLOW, config);

    size_t deliveries = 0;

    noc->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                           uint64_t, uint64_t) {
        deliveries++;
    });

    // Multiple concurrent transfers (includes multi-hop)
    std::vector<std::pair<uint8_t, uint8_t>> transfers = {
        {0, 15},   // Corner to corner
        {3, 12},   // Top-right to bottom-left corner
        {5, 10},   // Interior
        {1, 2},    // Adjacent
    };

    for (size_t i = 0; i < transfers.size(); ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(i);
        tile.n_tile = 0;
        tile.k_tile = 0;
        tile.size = 512;

        auto result = noc->inject_tile(transfers[i].first, transfers[i].second, tile, 0);
        REQUIRE(result == InjectResult::SUCCESS);
    }

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(deliveries == transfers.size());

    auto stats = noc->stats();
    REQUIRE(stats.tiles_injected == transfers.size());
    REQUIRE(stats.tiles_delivered == transfers.size());
}

TEST_CASE("INoC concurrent transfers (WormholeNoC)", "[noc_interface][transfer][wormhole]") {
    // WormholeNoC only supports single-hop, test with adjacent routers only
    // Note: WormholeNoC can only accept one injection at a time per router,
    // so we inject from different source routers for concurrent transfers

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(NoCType::WORMHOLE, config);

    size_t deliveries = 0;

    noc->set_global_delivery_callback([&](const TileDescriptor&, uint8_t, uint8_t,
                                           uint64_t, uint64_t) {
        deliveries++;
    });

    // Single-hop transfers from different source routers
    // In a 2x2 mesh: router layout is:
    //   0 - 1
    //   |   |
    //   2 - 3
    std::vector<std::pair<uint8_t, uint8_t>> transfers = {
        {0, 1},    // 0 → 1 (East)
        {2, 3},    // 2 → 3 (East)
        {1, 3},    // 1 → 3 (South)
    };

    for (size_t i = 0; i < transfers.size(); ++i) {
        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.m_tile = static_cast<uint16_t>(i);
        tile.n_tile = 0;
        tile.k_tile = 0;
        tile.size = 512;

        auto result = noc->inject_tile(transfers[i].first, transfers[i].second, tile, 0);
        REQUIRE(result == InjectResult::SUCCESS);
    }

    uint64_t cycle = 0;
    noc->drain(cycle);

    REQUIRE(deliveries == transfers.size());

    auto stats = noc->stats();
    REQUIRE(stats.tiles_injected == transfers.size());
    REQUIRE(stats.tiles_delivered == transfers.size());
}

// ============================================================================
// Step-by-Step Simulation Tests
// ============================================================================

TEST_CASE("INoC step simulation", "[noc_interface][simulation]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);

    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.n_tile = 0;
    tile.k_tile = 0;
    tile.size = 64;

    bool delivered = false;

    noc->set_delivery_callback(1, [&](const TileDescriptor&, uint8_t, uint8_t,
                                       uint64_t, uint64_t) {
        delivered = true;
    });

    noc->inject_tile(0, 1, tile, 0);

    REQUIRE_FALSE(noc->is_idle());

    // Step until delivered
    uint64_t cycle = 0;
    while (!noc->is_idle() && cycle < 1000) {
        noc->step(cycle);
        cycle++;
    }

    REQUIRE(delivered);
    REQUIRE(noc->is_idle());
}

// ============================================================================
// Adapter-Specific Tests
// ============================================================================

TEST_CASE("WormholeNoCAdapter underlying access", "[noc_interface][wormhole]") {
    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    WormholeNoCAdapter adapter(config);

    REQUIRE(adapter.type() == NoCType::WORMHOLE);

    // Access underlying WormholeNoC
    WormholeNoC& wh = adapter.underlying();
    REQUIRE(wh.num_routers() == 4);
}

TEST_CASE("DataflowNoCAdapter underlying access", "[noc_interface][dataflow]") {
    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    DataflowNoCAdapter adapter(config);

    REQUIRE(adapter.type() == NoCType::DATAFLOW);

    // Access underlying DataflowNoC
    DataflowNoC& df = adapter.underlying();
    REQUIRE(df.num_routers() == 4);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("INoC statistics tracking", "[noc_interface][stats]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);

    // Initial stats
    auto stats = noc->stats();
    REQUIRE(stats.tiles_injected == 0);
    REQUIRE(stats.tiles_delivered == 0);
    REQUIRE(stats.average_latency() == 0.0);

    // Inject and deliver a tile
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.size = 256;

    noc->inject_tile(0, 3, tile, 0);

    uint64_t cycle = 0;
    noc->drain(cycle);

    // Check stats after delivery
    stats = noc->stats();
    REQUIRE(stats.tiles_injected == 1);
    REQUIRE(stats.tiles_delivered == 1);
    REQUIRE(stats.total_flits > 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("INoC edge cases", "[noc_interface][edge]") {
    auto noc_type = GENERATE(NoCType::WORMHOLE, NoCType::DATAFLOW);

    INFO("Testing with NoC type: " << to_string(noc_type));

    NoCConfig config;
    config.rows = 2;
    config.cols = 2;

    auto noc = create_noc(noc_type, config);

    SECTION("Zero-size tile") {
        TileDescriptor tile;
        tile.size = 0;

        // Should still work (even if size is 0)
        auto result = noc->inject_tile(0, 0, tile, 0);
        // Behavior may vary - at least shouldn't crash
        (void)result;
    }

    SECTION("Same source and destination") {
        TileDescriptor tile;
        tile.size = 64;

        auto result = noc->inject_tile(0, 0, tile, 0);
        REQUIRE(result == InjectResult::SUCCESS);

        uint64_t cycle = 0;
        noc->drain(cycle);
    }

    SECTION("Drain on idle NoC") {
        uint64_t cycle = 0;
        noc->drain(cycle);  // Should not hang
        REQUIRE(noc->is_idle());
    }
}
