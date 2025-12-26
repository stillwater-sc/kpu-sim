#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Performance tests for DMA Engine
// Tests bandwidth and latency characteristics for external memory <-> L3 tile transfers

class DMAPerformanceFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    DMAPerformanceFixture() {
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 128;
        config.memory_bandwidth_gbps = 100;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 512;
        config.dma_engine_count = 4;

        sim = std::make_unique<KPUSimulator>(config);
    }
};

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Single Transfer Throughput", "[dma][performance]") {
    // Test with progressively larger transfers
    std::vector<size_t> sizes = {1024, 4096, 16384, 65536};

    for (size_t size : sizes) {
        // Generate random test data
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), 0);

        sim->write_memory_bank(0, 0, data.data(), size);

        Address src = sim->get_external_bank_base(0);
        Address dst = sim->get_l3_tile_base(0);

        bool complete = false;
        size_t cycles = 0;

        sim->dma_external_to_l3(0, src, dst, size,
            [&complete]() { complete = true; });

        while (!complete) {
            sim->step();
            cycles++;
        }

        // Transfer should complete
        REQUIRE(complete);

        // Verify data
        std::vector<uint8_t> result(size);
        sim->read_l3_tile(0, 0, result.data(), size);
        REQUIRE(std::equal(data.begin(), data.end(), result.begin()));
    }
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Concurrent Transfers", "[dma][performance]") {
    const size_t transfer_size = 4096;
    const size_t num_transfers = std::min(config.l3_tile_count, config.dma_engine_count);

    // Setup data for each transfer
    std::vector<std::vector<uint8_t>> data_sets;
    for (size_t i = 0; i < num_transfers; ++i) {
        std::vector<uint8_t> data(transfer_size, static_cast<uint8_t>(i + 1));
        data_sets.push_back(data);
        sim->write_memory_bank(0, i * 0x10000, data.data(), transfer_size);
    }

    // Start all transfers
    std::vector<bool> complete(num_transfers, false);
    for (size_t i = 0; i < num_transfers; ++i) {
        Address src = sim->get_external_bank_base(0) + i * 0x10000;
        Address dst = sim->get_l3_tile_base(i);

        sim->dma_external_to_l3(i, src, dst, transfer_size,
            [&complete, i]() { complete[i] = true; });
    }

    // Run until all complete
    while (!std::all_of(complete.begin(), complete.end(), [](bool c) { return c; })) {
        sim->step();
    }

    // Verify all transfers
    for (size_t i = 0; i < num_transfers; ++i) {
        std::vector<uint8_t> result(transfer_size);
        sim->read_l3_tile(i, 0, result.data(), transfer_size);
        REQUIRE(std::equal(data_sets[i].begin(), data_sets[i].end(), result.begin()));
    }
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Bidirectional Transfers", "[dma][performance]") {
    const size_t transfer_size = 8192;

    // First: External -> L3
    std::vector<uint8_t> data1(transfer_size, 0xAA);
    sim->write_memory_bank(0, 0, data1.data(), transfer_size);

    bool complete1 = false;
    Address src1 = sim->get_external_bank_base(0);
    Address dst1 = sim->get_l3_tile_base(0);
    sim->dma_external_to_l3(0, src1, dst1, transfer_size,
        [&complete1]() { complete1 = true; });

    while (!complete1) { sim->step(); }

    // Modify data in L3
    std::vector<uint8_t> modified(transfer_size, 0xBB);
    sim->write_l3_tile(0, 0, modified.data(), transfer_size);

    // Then: L3 -> External (different location)
    bool complete2 = false;
    Address src2 = sim->get_l3_tile_base(0);
    Address dst2 = sim->get_external_bank_base(0) + 0x10000;
    sim->dma_l3_to_external(0, src2, dst2, transfer_size,
        [&complete2]() { complete2 = true; });

    while (!complete2) { sim->step(); }

    // Verify data was written back
    std::vector<uint8_t> result(transfer_size);
    sim->read_memory_bank(0, 0x10000, result.data(), transfer_size);
    REQUIRE(std::equal(modified.begin(), modified.end(), result.begin()));
}
