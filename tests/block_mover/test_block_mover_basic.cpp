#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Test fixture for BlockMover tests
class BlockMoverTestFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    BlockMoverTestFixture() {
        // Configuration with L3 tiles, L2 banks, and BlockMovers
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.memory_bandwidth_gbps = 8;
        config.l1_buffer_count = 2;
        config.l1_buffer_capacity_kb = 256;
        config.compute_tile_count = 1;
        config.dma_engine_count = 4;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.block_mover_count = 4;

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to generate 2D test block data
    std::vector<uint8_t> generate_test_block(size_t height, size_t width, size_t element_size, uint8_t start_value = 0) {
        size_t total_size = height * width * element_size;
        std::vector<uint8_t> data(total_size);
        std::iota(data.begin(), data.end(), start_value);
        return data;
    }

    // Helper to generate matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }

    // Helper to verify L2 bank data
    bool verify_l2_data(const std::vector<uint8_t>& expected, Address addr, size_t size, size_t l2_bank_id) {
        std::vector<uint8_t> actual(size);
        sim->read_l2_bank(l2_bank_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    // Helper to verify L3 tile data
    bool verify_l3_data(const std::vector<uint8_t>& expected, Address addr, size_t size, size_t l3_tile_id) {
        std::vector<uint8_t> actual(size);
        sim->read_l3_tile(l3_tile_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    // Helper to verify transposed matrix data
    bool verify_transposed_matrix(const std::vector<float>& original, size_t orig_rows, size_t orig_cols,
                                  Address addr, size_t l2_bank_id) {
        size_t total_size = orig_rows * orig_cols * sizeof(float);
        std::vector<float> transposed_read(orig_rows * orig_cols);
        sim->read_l2_bank(l2_bank_id, addr, transposed_read.data(), total_size);

        // Verify transpose: original[i][j] == transposed[j][i]
        for (size_t i = 0; i < orig_rows; ++i) {
            for (size_t j = 0; j < orig_cols; ++j) {
                size_t orig_idx = i * orig_cols + j;
                size_t trans_idx = j * orig_rows + i;
                if (std::abs(original[orig_idx] - transposed_read[trans_idx]) > 1e-6f) {
                    return false;
                }
            }
        }
        return true;
    }
};

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Basic Transfer - Identity Copy", "[block_mover][basic]") {
    const size_t block_height = 4;
    const size_t block_width = 4;
    const size_t element_size = sizeof(float);
    const Address src_addr = 0x0;
    const Address dst_addr = 0x0;

    // Generate test block data
    auto test_data = generate_test_block(block_height, block_width, element_size, 0x10);

    // Write test data to L3 tile
    sim->write_l3_tile(0, src_addr, test_data.data(), test_data.size());

    // Start block transfer (L3[0] -> L2[0]) with identity transformation
    bool transfer_complete = false;
    sim->start_block_transfer(0, 0, src_addr, 0, dst_addr,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY,
                             [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify data integrity
    REQUIRE(verify_l2_data(test_data, dst_addr, test_data.size(), 0));
    REQUIRE_FALSE(sim->is_block_mover_busy(0));
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Matrix Transpose", "[block_mover][transpose]") {
    const size_t matrix_rows = 4;
    const size_t matrix_cols = 4;
    const size_t element_size = sizeof(float);
    const Address src_addr = 0x0;
    const Address dst_addr = 0x0;

    // Generate test matrix
    auto matrix_data = generate_matrix(matrix_rows, matrix_cols, 1.0f);

    // Write matrix to L3 tile
    sim->write_l3_tile(0, src_addr, matrix_data.data(), matrix_data.size() * sizeof(float));

    // Start block transfer with transpose transformation
    bool transfer_complete = false;
    sim->start_block_transfer(0, 0, src_addr, 0, dst_addr,
                             matrix_rows, matrix_cols, element_size,
                             BlockMover::TransformType::TRANSPOSE,
                             [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify transpose was applied correctly
    REQUIRE(verify_transposed_matrix(matrix_data, matrix_rows, matrix_cols, dst_addr, 0));
    REQUIRE_FALSE(sim->is_block_mover_busy(0));
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Queue Management - Multiple Transfers", "[block_mover][queue]") {
    const size_t block_height = 2;
    const size_t block_width = 2;
    const size_t element_size = sizeof(float);
    const size_t block_size = block_height * block_width * element_size;

    // Prepare multiple test blocks
    auto block1 = generate_test_block(block_height, block_width, element_size, 0x11);
    auto block2 = generate_test_block(block_height, block_width, element_size, 0x22);
    auto block3 = generate_test_block(block_height, block_width, element_size, 0x33);

    // Write test blocks to L3 tile at different offsets
    sim->write_l3_tile(0, 0x0, block1.data(), block_size);
    sim->write_l3_tile(0, block_size, block2.data(), block_size);
    sim->write_l3_tile(0, 2 * block_size, block3.data(), block_size);

    // Queue multiple transfers
    int completions = 0;
    auto completion_callback = [&completions]() { completions++; };

    sim->start_block_transfer(0, 0, 0x0, 0, 0x0,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY, completion_callback);
    sim->start_block_transfer(0, 0, block_size, 0, block_size,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY, completion_callback);
    sim->start_block_transfer(0, 0, 2 * block_size, 0, 2 * block_size,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY, completion_callback);

    REQUIRE(sim->is_block_mover_busy(0));

    // Process until all transfers complete
    while (completions < 3) {
        sim->step();
    }

    // Verify all blocks transferred correctly (FIFO order)
    REQUIRE(verify_l2_data(block1, 0x0, block_size, 0));
    REQUIRE(verify_l2_data(block2, block_size, block_size, 0));
    REQUIRE(verify_l2_data(block3, 2 * block_size, block_size, 0));
    REQUIRE_FALSE(sim->is_block_mover_busy(0));
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Data Integrity - Various Block Sizes", "[block_mover][integrity]") {
    const size_t element_size = sizeof(float);
    std::vector<std::pair<size_t, size_t>> test_dimensions = {
        {1, 1}, {2, 2}, {4, 4}, {8, 8}, {16, 16}
    };

    for (auto [height, width] : test_dimensions) {
        SECTION("Block size: " + std::to_string(height) + "x" + std::to_string(width)) {
            size_t block_size = height * width * element_size;

            if (block_size > sim->get_l3_tile_capacity(0) || block_size > sim->get_l2_bank_capacity(0)) {
                SKIP("Block size exceeds memory capacity");
            }

            auto test_data = generate_test_block(height, width, element_size, static_cast<uint8_t>(height + width));
            sim->write_l3_tile(0, 0, test_data.data(), block_size);

            bool complete = false;
            sim->start_block_transfer(0, 0, 0, 0, 0,
                                     height, width, element_size,
                                     BlockMover::TransformType::IDENTITY,
                                     [&complete]() { complete = true; });

            while (!complete) {
                sim->step();
            }

            REQUIRE(verify_l2_data(test_data, 0, block_size, 0));
        }
    }
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Error Handling - Invalid IDs", "[block_mover][error]") {
    SECTION("Invalid BlockMover ID") {
        REQUIRE_THROWS_AS(
            sim->start_block_transfer(99, 0, 0, 0, 0, 4, 4, sizeof(float)),
            std::out_of_range
        );

        REQUIRE_THROWS_AS(
            sim->is_block_mover_busy(99),
            std::out_of_range
        );
    }

    SECTION("Invalid L3 Tile ID") {
        REQUIRE_THROWS_AS(
            sim->start_block_transfer(0, 99, 0, 0, 0, 4, 4, sizeof(float)),
            std::out_of_range
        );
    }

    SECTION("Invalid L2 Bank ID") {
        REQUIRE_THROWS_AS(
            sim->start_block_transfer(0, 0, 0, 99, 0, 4, 4, sizeof(float)),
            std::out_of_range
        );
    }
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Reset Functionality", "[block_mover][reset]") {
    const size_t block_size = 4 * 4 * sizeof(float);
    auto test_data = generate_test_block(4, 4, sizeof(float));

    // Queue a transfer
    sim->write_l3_tile(0, 0, test_data.data(), block_size);
    sim->start_block_transfer(0, 0, 0, 0, 0, 4, 4, sizeof(float));

    REQUIRE(sim->is_block_mover_busy(0));

    // Reset the simulator
    sim->reset();

    // BlockMover should no longer be busy
    REQUIRE_FALSE(sim->is_block_mover_busy(0));
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Concurrent Operations", "[block_mover][concurrent]") {
    const size_t block_height = 4;
    const size_t block_width = 4;
    const size_t element_size = sizeof(float);
    const size_t block_size = block_height * block_width * element_size;

    // Prepare test data for different BlockMovers
    auto block1 = generate_test_block(block_height, block_width, element_size, 0xAA);
    auto block2 = generate_test_block(block_height, block_width, element_size, 0xBB);

    // Write to different L3 tiles
    sim->write_l3_tile(0, 0, block1.data(), block_size);
    sim->write_l3_tile(1, 0, block2.data(), block_size);

    // Start concurrent transfers using different BlockMovers
    bool transfer1_complete = false;
    bool transfer2_complete = false;

    // BlockMover 0: L3[0] -> L2[0]
    sim->start_block_transfer(0, 0, 0, 0, 0,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY,
                             [&transfer1_complete]() { transfer1_complete = true; });

    // BlockMover 1: L3[1] -> L2[1]
    sim->start_block_transfer(1, 1, 0, 1, 0,
                             block_height, block_width, element_size,
                             BlockMover::TransformType::IDENTITY,
                             [&transfer2_complete]() { transfer2_complete = true; });

    REQUIRE(sim->is_block_mover_busy(0));
    REQUIRE(sim->is_block_mover_busy(1));

    // Process until both complete
    while (!transfer1_complete || !transfer2_complete) {
        sim->step();
    }

    // Verify both transfers completed correctly
    REQUIRE(verify_l2_data(block1, 0, block_size, 0));
    REQUIRE(verify_l2_data(block2, 0, block_size, 1));
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Large Matrix Operations", "[block_mover][large]") {
    // Test with larger matrices that stress the system
    const size_t matrix_rows = 16;
    const size_t matrix_cols = 16;
    const size_t element_size = sizeof(float);
    const size_t matrix_size = matrix_rows * matrix_cols * element_size;

    if (matrix_size > sim->get_l3_tile_capacity(0) || matrix_size > sim->get_l2_bank_capacity(0)) {
        SKIP("Matrix too large for memory capacity");
    }

    // Generate matrix data
    auto matrix_data = generate_matrix(matrix_rows, matrix_cols, 0.5f);

    // Write matrix to L3 tile
    sim->write_l3_tile(0, 0, matrix_data.data(), matrix_size);

    SECTION("Large Identity Transfer") {
        bool transfer_complete = false;
        sim->start_block_transfer(0, 0, 0, 0, 0,
                                 matrix_rows, matrix_cols, element_size,
                                 BlockMover::TransformType::IDENTITY,
                                 [&transfer_complete]() { transfer_complete = true; });

        while (!transfer_complete) {
            sim->step();
        }

        // Verify data integrity
        std::vector<float> read_matrix(matrix_rows * matrix_cols);
        sim->read_l2_bank(0, 0, read_matrix.data(), matrix_size);
        REQUIRE(std::equal(matrix_data.begin(), matrix_data.end(), read_matrix.begin()));
    }

    SECTION("Large Transpose Transfer") {
        bool transfer_complete = false;
        sim->start_block_transfer(0, 0, 0, 0, 0,
                                 matrix_rows, matrix_cols, element_size,
                                 BlockMover::TransformType::TRANSPOSE,
                                 [&transfer_complete]() { transfer_complete = true; });

        while (!transfer_complete) {
            sim->step();
        }

        // Verify transpose was applied correctly
        REQUIRE(verify_transposed_matrix(matrix_data, matrix_rows, matrix_cols, 0, 0));
    }
}

TEST_CASE_METHOD(BlockMoverTestFixture, "BlockMover Status and Configuration", "[block_mover][status]") {
    // Verify initial configuration
    REQUIRE(sim->get_l3_tile_count() == 4);
    REQUIRE(sim->get_l2_bank_count() == 8);
    REQUIRE(sim->get_block_mover_count() == 4);

    // Verify capacities
    REQUIRE(sim->get_l3_tile_capacity(0) == 128 * 1024);
    REQUIRE(sim->get_l2_bank_capacity(0) == 64 * 1024);

    // Verify initial ready states
    REQUIRE(sim->is_l3_tile_ready(0));
    REQUIRE(sim->is_l2_bank_ready(0));
    REQUIRE_FALSE(sim->is_block_mover_busy(0));

    // Test status during operation
    auto test_data = generate_test_block(4, 4, sizeof(float));
    sim->write_l3_tile(0, 0, test_data.data(), test_data.size());
    sim->start_block_transfer(0, 0, 0, 0, 0, 4, 4, sizeof(float));

    REQUIRE(sim->is_block_mover_busy(0));

    // Complete the operation
    sim->run_until_idle();
    REQUIRE_FALSE(sim->is_block_mover_busy(0));
}
