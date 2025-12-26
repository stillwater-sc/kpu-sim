#include <vector>
#include <numeric>
#include <cstring>
#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Tensor movement tests for DMA Engine
// Tests loading/storing tensor data between external memory and L3 tiles

class TensorMovementFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    TensorMovementFixture() {
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 128;
        config.memory_bandwidth_gbps = 100;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 512;
        config.dma_engine_count = 4;

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to create row-major tensor data
    std::vector<float> create_tensor(size_t rows, size_t cols) {
        std::vector<float> data(rows * cols);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i);
        }
        return data;
    }
};

TEST_CASE_METHOD(TensorMovementFixture, "Tensor Movement - Load Matrix to L3", "[dma][tensor]") {
    const size_t rows = 64;
    const size_t cols = 64;
    const size_t size = rows * cols * sizeof(float);

    // Create tensor data
    auto tensor = create_tensor(rows, cols);

    // Store in external memory
    sim->write_memory_bank(0, 0, tensor.data(), size);

    // DMA transfer to L3 tile
    Address src = sim->get_external_bank_base(0);
    Address dst = sim->get_l3_tile_base(0);

    bool complete = false;
    sim->dma_external_to_l3(0, src, dst, size,
        [&complete]() { complete = true; });

    while (!complete) { sim->step(); }

    // Verify tensor in L3
    std::vector<float> result(rows * cols);
    sim->read_l3_tile(0, 0, result.data(), size);

    REQUIRE(std::equal(tensor.begin(), tensor.end(), result.begin()));
}

TEST_CASE_METHOD(TensorMovementFixture, "Tensor Movement - Store Matrix from L3", "[dma][tensor]") {
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t size = rows * cols * sizeof(float);

    // Create tensor and write to L3
    auto tensor = create_tensor(rows, cols);
    sim->write_l3_tile(0, 0, tensor.data(), size);

    // DMA transfer back to external memory
    Address src = sim->get_l3_tile_base(0);
    Address dst = sim->get_external_bank_base(0) + 0x10000;

    bool complete = false;
    sim->dma_l3_to_external(0, src, dst, size,
        [&complete]() { complete = true; });

    while (!complete) { sim->step(); }

    // Verify tensor in external memory
    std::vector<float> result(rows * cols);
    sim->read_memory_bank(0, 0x10000, result.data(), size);

    REQUIRE(std::equal(tensor.begin(), tensor.end(), result.begin()));
}

TEST_CASE_METHOD(TensorMovementFixture, "Tensor Movement - Multiple Tiles", "[dma][tensor]") {
    const size_t tile_rows = 16;
    const size_t tile_cols = 16;
    const size_t tile_size = tile_rows * tile_cols * sizeof(float);
    const size_t num_tiles = std::min(config.l3_tile_count, config.dma_engine_count);

    // Create different tensors for each tile
    std::vector<std::vector<float>> tiles;
    for (size_t i = 0; i < num_tiles; ++i) {
        auto tensor = create_tensor(tile_rows, tile_cols);
        for (auto& val : tensor) val += static_cast<float>(i * 1000);
        tiles.push_back(tensor);
        sim->write_memory_bank(0, i * 0x10000, tensor.data(), tile_size);
    }

    // Start concurrent DMA transfers to different L3 tiles
    std::vector<bool> complete(num_tiles, false);
    for (size_t i = 0; i < num_tiles; ++i) {
        Address src = sim->get_external_bank_base(0) + i * 0x10000;
        Address dst = sim->get_l3_tile_base(i);

        sim->dma_external_to_l3(i, src, dst, tile_size,
            [&complete, i]() { complete[i] = true; });
    }

    // Wait for all transfers
    while (!std::all_of(complete.begin(), complete.end(), [](bool c) { return c; })) {
        sim->step();
    }

    // Verify all tiles
    for (size_t i = 0; i < num_tiles; ++i) {
        std::vector<float> result(tile_rows * tile_cols);
        sim->read_l3_tile(i, 0, result.data(), tile_size);
        REQUIRE(std::equal(tiles[i].begin(), tiles[i].end(), result.begin()));
    }
}

TEST_CASE_METHOD(TensorMovementFixture, "Tensor Movement - Large Tensor", "[dma][tensor][large]") {
    // Test with larger tensor that fits in L3 tile
    const size_t rows = 128;
    const size_t cols = 128;
    const size_t size = rows * cols * sizeof(float);

    REQUIRE(size <= config.l3_tile_capacity_kb * 1024);

    auto tensor = create_tensor(rows, cols);
    sim->write_memory_bank(0, 0, tensor.data(), size);

    Address src = sim->get_external_bank_base(0);
    Address dst = sim->get_l3_tile_base(0);

    bool complete = false;
    sim->dma_external_to_l3(0, src, dst, size,
        [&complete]() { complete = true; });

    while (!complete) { sim->step(); }

    std::vector<float> result(rows * cols);
    sim->read_l3_tile(0, 0, result.data(), size);

    REQUIRE(std::equal(tensor.begin(), tensor.end(), result.begin()));
}
