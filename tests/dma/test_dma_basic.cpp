#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Test fixture for DMA tests
// DMA supports transfers between: HOST_MEMORY, KPU_MEMORY (external), and L3_TILE
class DMATestFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    DMATestFixture() {
        // Standard test configuration
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.memory_bandwidth_gbps = 8;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 256;
        config.compute_tile_count = 1;
        config.dma_engine_count = 4;

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_pattern(size_t size, uint8_t start_value = 0) {
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), start_value);
        return data;
    }

    // Helper to verify data in memory bank
    bool verify_memory_bank_data(const std::vector<uint8_t>& expected,
                                 Address addr, size_t size, size_t bank_id) {
        std::vector<uint8_t> actual(size);
        sim->read_memory_bank(bank_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    // Helper to verify data in L3 tile
    bool verify_l3_tile_data(const std::vector<uint8_t>& expected,
                             Address addr, size_t size, size_t tile_id) {
        std::vector<uint8_t> actual(size);
        sim->read_l3_tile(tile_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }
};

TEST_CASE_METHOD(DMATestFixture, "DMA Basic Transfer - External to L3 Tile", "[dma][basic]") {
    const size_t transfer_size = 1024;
    const Address src_addr = 0x1000;
    const Address dst_addr = 0x0;

    // Generate and write test data to external memory bank
    auto test_data = generate_test_pattern(transfer_size, 0xAA);
    sim->write_memory_bank(0, src_addr, test_data.data(), transfer_size);

    // Start DMA transfer (External[0] -> L3 Tile[0])
    Address global_src_addr = sim->get_external_bank_base(0) + src_addr;
    Address global_dst_addr = sim->get_l3_tile_base(0) + dst_addr;

    bool transfer_complete = false;
    sim->dma_external_to_l3(0, global_src_addr, global_dst_addr, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify data integrity in L3 tile
    REQUIRE(verify_l3_tile_data(test_data, dst_addr, transfer_size, 0));
    REQUIRE_FALSE(sim->is_dma_busy(0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Basic Transfer - L3 Tile to External", "[dma][basic]") {
    const size_t transfer_size = 2048;
    const Address src_addr = 0x0;
    const Address dst_addr = 0x2000;

    // Generate and write test data to L3 tile
    auto test_data = generate_test_pattern(transfer_size, 0x55);
    sim->write_l3_tile(0, src_addr, test_data.data(), transfer_size);

    // Start DMA transfer (L3 Tile[0] -> External[0])
    Address global_src_addr = sim->get_l3_tile_base(0) + src_addr;
    Address global_dst_addr = sim->get_external_bank_base(0) + dst_addr;

    bool transfer_complete = false;
    sim->dma_l3_to_external(0, global_src_addr, global_dst_addr, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify data integrity in external memory
    REQUIRE(verify_memory_bank_data(test_data, dst_addr, transfer_size, 0));
    REQUIRE_FALSE(sim->is_dma_busy(0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Large Transfer", "[dma][large]") {
    // Test with L3 tile capacity
    const size_t transfer_size = config.l3_tile_capacity_kb * 1024 / 2; // Half tile capacity
    const Address src_addr = 0x0;
    const Address dst_addr = 0x0;

    // Generate random test data
    std::vector<uint8_t> test_data(transfer_size);
    std::mt19937 rng(42);
    std::generate(test_data.begin(), test_data.end(), [&rng]() { return rng() & 0xFF; });

    // Write to external memory
    sim->write_memory_bank(0, src_addr, test_data.data(), transfer_size);

    // Transfer to L3 tile
    Address global_src_addr = sim->get_external_bank_base(0) + src_addr;
    Address global_dst_addr = sim->get_l3_tile_base(0) + dst_addr;

    bool transfer_complete = false;
    sim->dma_external_to_l3(0, global_src_addr, global_dst_addr, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    while (!transfer_complete) {
        sim->step();
    }

    // Verify data
    REQUIRE(verify_l3_tile_data(test_data, dst_addr, transfer_size, 0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Concurrent Transfers", "[dma][concurrent]") {
    const size_t transfer_size = 512;

    // Setup transfers to different L3 tiles using different DMA engines
    std::vector<std::vector<uint8_t>> test_data_sets;
    for (size_t i = 0; i < config.l3_tile_count && i < config.dma_engine_count; ++i) {
        auto data = generate_test_pattern(transfer_size, static_cast<uint8_t>(i * 0x10));
        test_data_sets.push_back(data);

        Address src_addr = i * 0x1000;
        sim->write_memory_bank(0, src_addr, data.data(), transfer_size);
    }

    // Start concurrent transfers
    std::vector<bool> transfers_complete(test_data_sets.size(), false);
    for (size_t i = 0; i < test_data_sets.size(); ++i) {
        Address src_addr = i * 0x1000;
        Address global_src = sim->get_external_bank_base(0) + src_addr;
        Address global_dst = sim->get_l3_tile_base(i);

        sim->dma_external_to_l3(i, global_src, global_dst, transfer_size,
            [&transfers_complete, i]() { transfers_complete[i] = true; });
    }

    // Wait for all transfers to complete
    while (!std::all_of(transfers_complete.begin(), transfers_complete.end(),
                        [](bool c) { return c; })) {
        sim->step();
    }

    // Verify all transfers
    for (size_t i = 0; i < test_data_sets.size(); ++i) {
        REQUIRE(verify_l3_tile_data(test_data_sets[i], 0, transfer_size, i));
    }
}

TEST_CASE_METHOD(DMATestFixture, "DMA Status Queries", "[dma][status]") {
    const size_t transfer_size = 4096;
    const Address src_addr = 0x0;

    // Generate test data
    auto test_data = generate_test_pattern(transfer_size, 0x77);
    sim->write_memory_bank(0, src_addr, test_data.data(), transfer_size);

    // Initially DMA should not be busy
    REQUIRE_FALSE(sim->is_dma_busy(0));

    // Start transfer
    Address global_src = sim->get_external_bank_base(0) + src_addr;
    Address global_dst = sim->get_l3_tile_base(0);

    bool transfer_complete = false;
    sim->dma_external_to_l3(0, global_src, global_dst, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    // DMA should be busy during transfer
    REQUIRE(sim->is_dma_busy(0));

    // Complete the transfer
    while (!transfer_complete) {
        sim->step();
    }

    // DMA should no longer be busy
    REQUIRE_FALSE(sim->is_dma_busy(0));
}
