#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Debug/diagnostic tests for DMA Engine
// Uses L3 tiles as the on-chip cache (DMA supports HOST, EXTERNAL, L3)

TEST_CASE("DMA Debug - Component Status", "[dma][debug]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 64;
    config.l3_tile_count = 2;
    config.l3_tile_capacity_kb = 256;
    config.dma_engine_count = 2;

    KPUSimulator sim(config);

    // Check DMA engine count
    REQUIRE(sim.get_dma_engine_count() == 2);

    // Check L3 tile count and capacity
    REQUIRE(sim.get_l3_tile_count() == 2);
    REQUIRE(sim.get_l3_tile_capacity(0) == 256 * 1024);
    REQUIRE(sim.get_l3_tile_capacity(1) == 256 * 1024);

    // Check memory bank count
    REQUIRE(sim.get_memory_bank_count() == 2);
}

TEST_CASE("DMA Debug - Transfer Verification", "[dma][debug]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 64;
    config.l3_tile_count = 1;
    config.l3_tile_capacity_kb = 256;
    config.dma_engine_count = 1;

    KPUSimulator sim(config);

    // Write test pattern to memory
    const size_t transfer_size = 256;
    std::vector<uint8_t> test_data(transfer_size, 0xAB);
    sim.write_memory_bank(0, 0x1000, test_data.data(), transfer_size);

    // Transfer to L3 tile
    Address src = sim.get_external_bank_base(0) + 0x1000;
    Address dst = sim.get_l3_tile_base(0);

    bool complete = false;
    sim.dma_external_to_l3(0, src, dst, transfer_size,
        [&complete]() { complete = true; });

    while (!complete) {
        sim.step();
    }

    // Verify transfer
    std::vector<uint8_t> result(transfer_size);
    sim.read_l3_tile(0, 0, result.data(), transfer_size);

    REQUIRE(std::equal(test_data.begin(), test_data.end(), result.begin()));
}
