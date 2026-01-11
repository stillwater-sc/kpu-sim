// ============================================================================
// tests/noc/wormhole_router_test.cpp
// Tests for Wormhole Router with Correct Bandwidth Modeling
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <iostream>

#include <sw/kpu/models/temporal/noc/wormhole_router.hpp>

using namespace sw::kpu::noc;
using namespace sw::kpu;

TEST_CASE("Wormhole NoC basic creation", "[wormhole][noc]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;
    config.dma_on_north_edge = true;

    WormholeNoC noc(config);

    REQUIRE(noc.num_routers() == 16);
    REQUIRE(noc.is_idle());
}

TEST_CASE("Single tile injection and delivery", "[wormhole][noc]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);
    WormholeTracer tracer;
    noc.set_tracer(&tracer);

    // Create a 4KB tile (64 flits)
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.k_tile = 0;
    tile.size = 4 * 1024;  // 4KB

    uint16_t expected_flits = 64;  // 4KB / 64B

    bool delivered = false;
    TileDescriptor delivered_tile;
    uint64_t delivery_cycle = 0;

    // Set delivery callback
    noc.set_delivery_callback(1, [&](const TileDescriptor& t, uint8_t src, uint64_t cycle) {
        delivered = true;
        delivered_tile = t;
        delivery_cycle = cycle;
    });

    // Inject tile from R[0,0] to R[0,1] (single hop East)
    uint64_t inject_cycle = 0;
    auto result = noc.inject_tile(0, 1, tile, inject_cycle);
    REQUIRE(result == WormholeNoC::InjectResult::SUCCESS);

    // Run simulation until idle
    uint64_t cycle = 0;
    uint64_t max_cycles = 1000;

    while (!noc.is_idle() && cycle < max_cycles) {
        cycle++;
        noc.step(cycle);
    }

    REQUIRE(delivered);
    REQUIRE(delivered_tile.tensor == TensorId::A);
    REQUIRE(delivered_tile.m_tile == 0);
    REQUIRE(delivered_tile.size == 4 * 1024);

    // Verify timing: 4KB = 64 flits, should take ~64-65 cycles
    // (injection + 1 hop + delivery)
    std::cout << "\nSingle 4KB tile transfer:\n";
    std::cout << "  Inject cycle: " << inject_cycle << "\n";
    std::cout << "  Delivery cycle: " << delivery_cycle << "\n";
    std::cout << "  Total latency: " << (delivery_cycle - inject_cycle) << " cycles\n";
    std::cout << "  Expected flits: " << expected_flits << "\n";

    // Latency should be approximately equal to number of flits
    // (injection serializes at 1 flit/cycle)
    REQUIRE(delivery_cycle >= expected_flits);
    REQUIRE(delivery_cycle <= expected_flits + 10);  // Small overhead allowance

    // Check statistics
    const auto& stats = noc.stats();
    REQUIRE(stats.tiles_injected == 1);
    REQUIRE(stats.tiles_delivered == 1);
    REQUIRE(stats.total_flits == expected_flits);
}

TEST_CASE("Sequential injection serialization", "[wormhole][noc][bandwidth]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);

    // Create two 4KB tiles (64 flits each)
    TileDescriptor tile1, tile2;
    tile1.tensor = TensorId::A;
    tile1.m_tile = 0;
    tile1.k_tile = 0;
    tile1.size = 4 * 1024;

    tile2.tensor = TensorId::A;
    tile2.m_tile = 0;
    tile2.k_tile = 1;
    tile2.size = 4 * 1024;

    uint64_t tile1_delivery = 0;
    uint64_t tile2_delivery = 0;

    noc.set_delivery_callback(1, [&](const TileDescriptor& t, uint8_t src, uint64_t cycle) {
        if (t.k_tile == 0) {
            tile1_delivery = cycle;
        } else {
            tile2_delivery = cycle;
        }
    });

    // Try to inject both tiles at cycle 0
    auto r1 = noc.inject_tile(0, 1, tile1, 0);
    REQUIRE(r1 == WormholeNoC::InjectResult::SUCCESS);

    // Second injection should be BUSY (first one still injecting)
    auto r2 = noc.inject_tile(0, 1, tile2, 0);
    REQUIRE(r2 == WormholeNoC::InjectResult::BUSY);

    // Run until first tile is fully injected
    uint64_t cycle = 0;
    while (cycle < 100) {
        cycle++;
        noc.step(cycle);

        // Try to inject second tile
        if (!noc.can_inject(0)) continue;

        r2 = noc.inject_tile(0, 1, tile2, cycle);
        if (r2 == WormholeNoC::InjectResult::SUCCESS) {
            std::cout << "\nSecond tile injection started at cycle: " << cycle << "\n";
            break;
        }
    }

    REQUIRE(r2 == WormholeNoC::InjectResult::SUCCESS);

    // Second injection should start after first completes (~64 cycles)
    REQUIRE(cycle >= 64);

    // Run to completion
    uint64_t max_cycles = 200;
    while (!noc.is_idle() && cycle < max_cycles) {
        cycle++;
        noc.step(cycle);
    }

    std::cout << "Tile 1 delivered at cycle: " << tile1_delivery << "\n";
    std::cout << "Tile 2 delivered at cycle: " << tile2_delivery << "\n";
    std::cout << "Total simulation cycles: " << cycle << "\n";

    // Both tiles should be delivered
    REQUIRE(tile1_delivery > 0);
    REQUIRE(tile2_delivery > 0);

    // Tile 2 should be delivered ~64 cycles after tile 1
    REQUIRE(tile2_delivery > tile1_delivery);
    REQUIRE(tile2_delivery >= tile1_delivery + 60);  // At least 60 cycles apart

    // Total time should be ~128 cycles (2 × 64 flits)
    const auto& stats = noc.stats();
    REQUIRE(stats.tiles_delivered == 2);
    REQUIRE(stats.total_flits == 128);
}

TEST_CASE("Concurrent East and South transfers", "[wormhole][noc][concurrent]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);

    // Create two 4KB tiles
    TileDescriptor tile_east, tile_south;
    tile_east.tensor = TensorId::A;
    tile_east.m_tile = 0;
    tile_east.size = 4 * 1024;

    tile_south.tensor = TensorId::B;
    tile_south.n_tile = 0;
    tile_south.size = 4 * 1024;

    uint64_t east_delivery = 0;
    uint64_t south_delivery = 0;

    // R[0,0] → R[0,1] (East)
    noc.set_delivery_callback(1, [&](const TileDescriptor& t, uint8_t src, uint64_t cycle) {
        east_delivery = cycle;
    });

    // R[0,1] → R[1,1] (South) - callback on router 5
    noc.set_delivery_callback(5, [&](const TileDescriptor& t, uint8_t src, uint64_t cycle) {
        south_delivery = cycle;
    });

    // Inject East tile from R[0,0]
    auto r1 = noc.inject_tile(0, 1, tile_east, 0);
    REQUIRE(r1 == WormholeNoC::InjectResult::SUCCESS);

    // East injection is using LOCAL port, so South must wait for LOCAL port
    // But we could inject South from a DIFFERENT router

    // Inject South from R[0,1] (different source)
    auto r2 = noc.inject_tile(1, 5, tile_south, 0);  // R[0,1] → R[1,1]
    REQUIRE(r2 == WormholeNoC::InjectResult::SUCCESS);

    // Run to completion
    uint64_t cycle = 0;
    uint64_t max_cycles = 200;
    while (!noc.is_idle() && cycle < max_cycles) {
        cycle++;
        noc.step(cycle);
    }

    std::cout << "\nConcurrent transfers from different sources:\n";
    std::cout << "  R[0,0]→R[0,1] (East) delivered at: " << east_delivery << "\n";
    std::cout << "  R[0,1]→R[1,1] (South) delivered at: " << south_delivery << "\n";

    // Both should complete around the same time (~64 cycles)
    // since they use different source routers
    REQUIRE(east_delivery > 0);
    REQUIRE(south_delivery > 0);
    REQUIRE(east_delivery <= 70);  // ~64 cycles + small overhead
    REQUIRE(south_delivery <= 70);

    const auto& stats = noc.stats();
    REQUIRE(stats.tiles_delivered == 2);
}

TEST_CASE("Variable tile sizes", "[wormhole][noc]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);

    struct TestCase {
        uint32_t size;
        uint16_t expected_flits;
        uint64_t min_latency;
        uint64_t max_latency;
    };

    std::vector<TestCase> test_cases = {
        {1024, 16, 16, 20},         // 1KB
        {4096, 64, 64, 70},         // 4KB
        {8192, 128, 128, 135},      // 8KB
        {65536, 1024, 1024, 1035},  // 64KB
    };

    std::cout << "\nVariable tile size testing:\n";
    std::cout << "  Size (KB)  Flits   Latency   Expected\n";
    std::cout << "  --------   -----   -------   --------\n";

    for (const auto& tc : test_cases) {
        noc.reset_stats();

        TileDescriptor tile;
        tile.tensor = TensorId::A;
        tile.size = tc.size;

        uint64_t delivery_cycle = 0;
        noc.set_delivery_callback(1, [&](const TileDescriptor& t, uint8_t src, uint64_t cycle) {
            delivery_cycle = cycle;
        });

        auto result = noc.inject_tile(0, 1, tile, 0);
        REQUIRE(result == WormholeNoC::InjectResult::SUCCESS);

        uint64_t cycle = 0;
        while (!noc.is_idle() && cycle < tc.max_latency + 100) {
            cycle++;
            noc.step(cycle);
        }

        std::cout << "  " << std::setw(8) << (tc.size / 1024)
                  << "   " << std::setw(5) << tc.expected_flits
                  << "   " << std::setw(7) << delivery_cycle
                  << "   " << tc.min_latency << "-" << tc.max_latency << "\n";

        REQUIRE(noc.stats().total_flits == tc.expected_flits);
        REQUIRE(delivery_cycle >= tc.min_latency);
        REQUIRE(delivery_cycle <= tc.max_latency);
    }
}

TEST_CASE("Wormhole trace generation", "[wormhole][trace]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);
    WormholeTracer tracer;
    noc.set_tracer(&tracer);

    // Inject a 1KB tile (16 flits)
    TileDescriptor tile;
    tile.tensor = TensorId::A;
    tile.m_tile = 0;
    tile.k_tile = 0;
    tile.size = 1024;

    noc.inject_tile(0, 1, tile, 0);

    // Run to completion
    uint64_t cycle = 0;
    while (!noc.is_idle() && cycle < 100) {
        cycle++;
        noc.step(cycle);
    }

    // Check trace events
    const auto& events = tracer.events();
    std::cout << "\nTrace events for 1KB tile:\n";
    std::cout << "  Total events: " << events.size() << "\n";

    size_t inject_start = 0, inject_complete = 0;
    size_t deliver_start = 0, deliver_complete = 0;
    size_t flit_hops = 0;

    for (const auto& e : events) {
        switch (e.type) {
            case WormholeEventType::INJECT_START: inject_start++; break;
            case WormholeEventType::INJECT_COMPLETE: inject_complete++; break;
            case WormholeEventType::DELIVER_START: deliver_start++; break;
            case WormholeEventType::DELIVER_COMPLETE: deliver_complete++; break;
            case WormholeEventType::FLIT_HOP: flit_hops++; break;
            default: break;
        }
    }

    std::cout << "  INJECT_START: " << inject_start << "\n";
    std::cout << "  INJECT_COMPLETE: " << inject_complete << "\n";
    std::cout << "  DELIVER_START: " << deliver_start << "\n";
    std::cout << "  DELIVER_COMPLETE: " << deliver_complete << "\n";
    std::cout << "  FLIT_HOP: " << flit_hops << "\n";

    REQUIRE(inject_start == 1);
    REQUIRE(inject_complete == 1);
    REQUIRE(deliver_start == 1);
    REQUIRE(deliver_complete == 1);
    REQUIRE(flit_hops == 16);  // One hop per flit

    // Export CSV
    auto temp_dir = std::filesystem::temp_directory_path();
    auto csv_path = (temp_dir / "wormhole_trace.csv").string();
    REQUIRE(tracer.export_csv(csv_path));
    std::cout << "  Trace exported to: " << csv_path << "\n";

    // Export Chrome trace
    auto chrome_path = (temp_dir / "wormhole_trace_chrome.json").string();
    REQUIRE(tracer.export_chrome_trace(chrome_path, 4, 4));
    std::cout << "  Chrome trace exported to: " << chrome_path << "\n";
}

TEST_CASE("Contention on same output port", "[wormhole][noc][contention]") {
    WormholeNoC::Config config;
    config.rows = 4;
    config.cols = 4;

    WormholeNoC noc(config);

    // Two tiles both want to go East from different inputs to same router
    // R[0,0] LOCAL → R[0,1] (via EAST output)
    // Can't easily test this with single-hop since we can only inject from LOCAL

    // Instead, test that second injection waits for first
    TileDescriptor tile1, tile2;
    tile1.tensor = TensorId::A;
    tile1.m_tile = 0;
    tile1.size = 2 * 1024;  // 2KB = 32 flits

    tile2.tensor = TensorId::A;
    tile2.m_tile = 1;
    tile2.size = 2 * 1024;

    uint64_t tile1_inject_start = 0;
    uint64_t tile2_inject_start = 0;

    // First injection
    auto r1 = noc.inject_tile(0, 1, tile1, 0);
    REQUIRE(r1 == WormholeNoC::InjectResult::SUCCESS);
    tile1_inject_start = 0;

    // Second should be blocked
    REQUIRE(!noc.can_inject(0));

    // Run until can inject again
    uint64_t cycle = 0;
    while (!noc.can_inject(0) && cycle < 100) {
        cycle++;
        noc.step(cycle);
    }

    tile2_inject_start = cycle;
    auto r2 = noc.inject_tile(0, 1, tile2, cycle);
    REQUIRE(r2 == WormholeNoC::InjectResult::SUCCESS);

    std::cout << "\nContention test:\n";
    std::cout << "  Tile 1 inject started: " << tile1_inject_start << "\n";
    std::cout << "  Tile 2 inject started: " << tile2_inject_start << "\n";
    std::cout << "  Serialization delay: " << (tile2_inject_start - tile1_inject_start) << " cycles\n";

    // Second injection should start ~32 cycles after first (32 flits)
    REQUIRE(tile2_inject_start >= 32);
    REQUIRE(tile2_inject_start <= 35);
}
