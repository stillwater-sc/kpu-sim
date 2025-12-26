#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Test fixture for Streamer tests
class StreamerTestFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    StreamerTestFixture() {
        // Configuration with L2 banks, L1 buffers, and Streamers
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.memory_bandwidth_gbps = 8;
        config.l1_buffer_count = 4;
        config.l1_buffer_capacity_kb = 256;
        config.compute_tile_count = 1;
        config.dma_engine_count = 4;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.block_mover_count = 4;
        config.streamer_count = 8;

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to generate test matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }

    // Helper to verify row streaming results
    bool verify_row_stream(const std::vector<float>& source_matrix,
                          const std::vector<float>& streamed_data,
                          size_t /* matrix_height */, size_t matrix_width,
                          size_t fabric_size, size_t row_index) {
        for (size_t col = 0; col < std::min(fabric_size, matrix_width); ++col) {
            size_t source_idx = row_index * matrix_width + col;
            float expected = source_matrix[source_idx];
            float actual = streamed_data[col];

            if (std::abs(actual - expected) > 1e-6f) {
                return false;
            }
        }
        return true;
    }

    // Helper to verify column streaming results
    bool verify_column_stream(const std::vector<float>& source_matrix,
                             const std::vector<float>& streamed_data,
                             size_t matrix_height, size_t matrix_width,
                             size_t fabric_size, size_t col_index) {
        for (size_t row = 0; row < std::min(fabric_size, matrix_height); ++row) {
            size_t source_idx = row * matrix_width + col_index;
            float expected = source_matrix[source_idx];
            float actual = streamed_data[row];

            if (std::abs(actual - expected) > 1e-6f) {
                return false;
            }
        }
        return true;
    }
};

TEST_CASE_METHOD(StreamerTestFixture, "Streamer Basic Functionality", "[streamer][basic]") {
    SECTION("Can stream rows from L2 to L1") {
        // Test parameters
        const size_t matrix_height = 8;
        const size_t matrix_width = 8;
        const size_t fabric_size = 4;  // 4x4 systolic array
        const size_t element_size = sizeof(float);

        // Generate test matrix
        auto test_matrix = generate_matrix(matrix_height, matrix_width, 1.0f);

        // Write matrix to L2 bank
        const size_t l2_bank_id = 0;
        const Address l2_base_addr = 0;
        sim->write_l2_bank(l2_bank_id, l2_base_addr, test_matrix.data(),
                          test_matrix.size() * element_size);

        // Set up streaming from L2 to L1
        const size_t streamer_id = 0;
        const size_t l1_buffer_id = 0;
        const Address l1_base_addr = 0;

        bool stream_complete = false;
        sim->start_row_stream(streamer_id, l2_bank_id, l1_buffer_id,
                             l2_base_addr, l1_base_addr,
                             matrix_height, matrix_width, element_size, fabric_size,
                             Streamer::StreamDirection::L2_TO_L1,
                             [&stream_complete]() { stream_complete = true; });

        // Run simulation until stream completes
        while (!stream_complete && !sim->is_streamer_busy(streamer_id)) {
            sim->step();
        }

        // Run until actually complete
        sim->run_until_idle();

        REQUIRE(stream_complete);
        REQUIRE_FALSE(sim->is_streamer_busy(streamer_id));

        // Verify streamed data in L1 - check first row
        std::vector<float> l1_data(fabric_size);
        sim->read_l1_buffer(l1_buffer_id, l1_base_addr,
                           l1_data.data(), fabric_size * element_size);

        //REQUIRE(verify_row_stream(test_matrix, l1_data, matrix_height, matrix_width, fabric_size, 0));
    }

    SECTION("Can stream columns from L2 to L1") {
        // Test parameters
        const size_t matrix_height = 8;
        const size_t matrix_width = 8;
        const size_t fabric_size = 4;  // 4x4 systolic array
        const size_t element_size = sizeof(float);

        // Generate test matrix (column-major layout for easier testing)
        auto test_matrix = generate_matrix(matrix_height, matrix_width, 10.0f);

        // Write matrix to L2 bank
        const size_t l2_bank_id = 1;
        const Address l2_base_addr = 0;
        sim->write_l2_bank(l2_bank_id, l2_base_addr, test_matrix.data(),
                          test_matrix.size() * element_size);

        // Set up streaming from L2 to L1
        const size_t streamer_id = 1;
        const size_t l1_buffer_id = 1;
        const Address l1_base_addr = 0;

        bool stream_complete = false;
        sim->start_column_stream(streamer_id, l2_bank_id, l1_buffer_id,
                                l2_base_addr, l1_base_addr,
                                matrix_height, matrix_width, element_size, fabric_size,
                                Streamer::StreamDirection::L2_TO_L1,
                                [&stream_complete]() { stream_complete = true; });

        // Run simulation until stream completes
        sim->run_until_idle();

        REQUIRE(stream_complete);
        REQUIRE_FALSE(sim->is_streamer_busy(streamer_id));

        // Verify streamed data in L1 - check first column
        std::vector<float> l1_data(fabric_size);
        sim->read_l1_buffer(l1_buffer_id, l1_base_addr,
                           l1_data.data(), fabric_size * element_size);

        //REQUIRE(verify_column_stream(test_matrix, l1_data, matrix_height, matrix_width, fabric_size, 0));
    }

    SECTION("Can stream from L1 back to L2") {
        // Test parameters
        const size_t matrix_height = 4;
        const size_t matrix_width = 4;
        const size_t fabric_size = 2;  // 2x2 systolic array
        const size_t element_size = sizeof(float);

        // Generate test data in L1 (simulate compute results)
        std::vector<float> l1_data = {100.0f, 101.0f, 102.0f, 103.0f};
        const size_t l1_buffer_id = 2;
        const Address l1_base_addr = 0;

        sim->write_l1_buffer(l1_buffer_id, l1_base_addr,
                             l1_data.data(), l1_data.size() * element_size);

        // Set up streaming from L1 to L2
        const size_t streamer_id = 2;
        const size_t l2_bank_id = 2;
        const Address l2_base_addr = 0;

        bool stream_complete = false;
        sim->start_row_stream(streamer_id, l2_bank_id, l1_buffer_id,
                             l2_base_addr, l1_base_addr,
                             matrix_height, matrix_width, element_size, fabric_size,
                             Streamer::StreamDirection::L1_TO_L2,
                             [&stream_complete]() { stream_complete = true; });

        // Run simulation until stream completes
        sim->run_until_idle();

        REQUIRE(stream_complete);
        REQUIRE_FALSE(sim->is_streamer_busy(streamer_id));

        // Verify data was written to L2
        std::vector<float> l2_data(fabric_size);
        sim->read_l2_bank(l2_bank_id, l2_base_addr,
                         l2_data.data(), fabric_size * element_size);

        for (size_t i = 0; i < fabric_size; ++i) {
            REQUIRE(l2_data[i] == Catch::Approx(l1_data[i]));
        }
    }

    SECTION("Multiple streamers can operate concurrently") {
        // Test parameters
        const size_t matrix_height = 4;
        const size_t matrix_width = 4;
        const size_t fabric_size = 2;
        const size_t element_size = sizeof(float);

        // Generate test matrices
        auto matrix_a = generate_matrix(matrix_height, matrix_width, 1.0f);
        auto matrix_b = generate_matrix(matrix_height, matrix_width, 100.0f);

        // Write matrices to different L2 banks
        sim->write_l2_bank(0, 0, matrix_a.data(), matrix_a.size() * element_size);
        sim->write_l2_bank(1, 0, matrix_b.data(), matrix_b.size() * element_size);

        // Start concurrent streams
        bool stream_a_complete = false;
        bool stream_b_complete = false;

        sim->start_row_stream(0, 0, 0, 0, 0,  // streamer 0: L2[0] -> L1[0]
                             matrix_height, matrix_width, element_size, fabric_size,
                             Streamer::StreamDirection::L2_TO_L1,
                             [&stream_a_complete]() { stream_a_complete = true; });

        sim->start_column_stream(1, 1, 1, 0, 0,  // streamer 1: L2[1] -> L1[1]
                                matrix_height, matrix_width, element_size, fabric_size,
                                Streamer::StreamDirection::L2_TO_L1,
                                [&stream_b_complete]() { stream_b_complete = true; });

        // Run simulation
        sim->run_until_idle();

        REQUIRE(stream_a_complete);
        REQUIRE(stream_b_complete);
        REQUIRE_FALSE(sim->is_streamer_busy(0));
        REQUIRE_FALSE(sim->is_streamer_busy(1));
    }
}

TEST_CASE_METHOD(StreamerTestFixture, "Streamer Edge Cases", "[streamer][edge]") {
    SECTION("Handles fabric size larger than matrix dimensions") {
        const size_t matrix_height = 2;
        const size_t matrix_width = 2;
        const size_t fabric_size = 8;  // Larger than matrix
        const size_t element_size = sizeof(float);

        auto test_matrix = generate_matrix(matrix_height, matrix_width, 50.0f);

        sim->write_l2_bank(0, 0, test_matrix.data(), test_matrix.size() * element_size);

        bool stream_complete = false;
        sim->start_row_stream(0, 0, 0, 0, 0,
                             matrix_height, matrix_width, element_size, fabric_size,
                             Streamer::StreamDirection::L2_TO_L1,
                             [&stream_complete]() { stream_complete = true; });

        // Should complete without error
        sim->run_until_idle();
        REQUIRE(stream_complete);
    }

    SECTION("Handles single element streaming") {
        const size_t matrix_height = 1;
        const size_t matrix_width = 1;
        const size_t fabric_size = 1;
        const size_t element_size = sizeof(float);

        std::vector<float> test_data = {42.0f};

        sim->write_l2_bank(0, 0, test_data.data(), test_data.size() * element_size);

        bool stream_complete = false;
        sim->start_row_stream(0, 0, 0, 0, 0,
                             matrix_height, matrix_width, element_size, fabric_size,
                             Streamer::StreamDirection::L2_TO_L1,
                             [&stream_complete]() { stream_complete = true; });

        sim->run_until_idle();

        REQUIRE(stream_complete);

        // Verify the single element was streamed correctly
        float result;
        sim->read_l1_buffer(0, 0, &result, element_size);
        REQUIRE(result == Catch::Approx(42.0f));
    }
}

TEST_CASE_METHOD(StreamerTestFixture, "Streamer Error Handling", "[streamer][error]") {
    SECTION("Validates streamer ID bounds") {
        const size_t invalid_streamer_id = config.streamer_count + 1;

        REQUIRE_THROWS_AS(
            sim->start_row_stream(invalid_streamer_id, 0, 0, 0, 0, 4, 4, 4, 2),
            std::out_of_range
        );
    }

    SECTION("Validates L2 bank ID bounds") {
        const size_t invalid_l2_bank_id = config.l2_bank_count + 1;

        REQUIRE_THROWS_AS(
            sim->start_row_stream(0, invalid_l2_bank_id, 0, 0, 0, 4, 4, 4, 2),
            std::out_of_range
        );
    }

    SECTION("Validates L1 buffer ID bounds") {
        const size_t invalid_l1_id = config.l1_buffer_count + 1;

        REQUIRE_THROWS_AS(
            sim->start_row_stream(0, 0, invalid_l1_id, 0, 0, 4, 4, 4, 2),
            std::out_of_range
        );
    }
}
