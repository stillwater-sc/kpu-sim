#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/models/temporal/memory/storage_scheduler.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>

using namespace sw::kpu;

// Test fixture for StorageScheduler tests
class StorageSchedulerTestFixture {
public:
    static constexpr size_t NUM_BANKS = 4;
    static constexpr size_t BANK_SIZE_KB = 64;

    StorageScheduler::BankConfig default_config{
        .bank_size_kb = BANK_SIZE_KB,
        .cache_line_size = 64,
        .num_ports = 2,
        .access_pattern = StorageScheduler::AccessPattern::SEQUENTIAL,
        .enable_prefetch = true
    };

    std::unique_ptr<StorageScheduler> scheduler;

    StorageSchedulerTestFixture() {
        scheduler = std::make_unique<StorageScheduler>(0, NUM_BANKS, default_config);
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_data(size_t size, uint8_t pattern = 0xAA) {
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>((pattern + i) & 0xFF);
        }
        return data;
    }

    // Helper to verify data integrity
    bool verify_data(const std::vector<uint8_t>& expected,
                     const std::vector<uint8_t>& actual) {
        return expected.size() == actual.size() &&
               std::equal(expected.begin(), expected.end(), actual.begin());
    }
};

TEST_CASE_METHOD(StorageSchedulerTestFixture, "StorageScheduler Basic Functionality", "[memory_scheduler][basic]") {

    SECTION("StorageScheduler initialization") {
        REQUIRE(scheduler->get_num_banks() == NUM_BANKS);
        REQUIRE(scheduler->get_scheduler_id() == 0);

        for (size_t i = 0; i < NUM_BANKS; ++i) {
            REQUIRE(scheduler->get_bank_capacity(i) == BANK_SIZE_KB * 1024);
            REQUIRE(scheduler->get_bank_occupancy(i) == 0);
            REQUIRE(scheduler->is_ready(i));
            REQUIRE_FALSE(scheduler->is_bank_busy(i));
            REQUIRE(scheduler->get_bank_operation(i) == StorageScheduler::StorageOperation::BARRIER);
        }
    }

    SECTION("Basic read/write operations") {
        constexpr size_t TEST_SIZE = 1024;
        auto test_data = generate_test_data(TEST_SIZE, 0x42);
        std::vector<uint8_t> read_buffer(TEST_SIZE);

        // Write data to bank 0
        REQUIRE_NOTHROW(scheduler->direct_write(0, 0, test_data.data(), TEST_SIZE));
        REQUIRE(scheduler->get_bank_occupancy(0) == TEST_SIZE);

        // Read data back from bank 0
        REQUIRE_NOTHROW(scheduler->direct_read(0, 0, read_buffer.data(), TEST_SIZE));
        REQUIRE(verify_data(test_data, read_buffer));
    }

    SECTION("Multi-bank operations") {
        constexpr size_t TEST_SIZE = 512;

        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            auto test_data = generate_test_data(TEST_SIZE, static_cast<uint8_t>(bank_id));

            REQUIRE_NOTHROW(scheduler->direct_write(bank_id, 0, test_data.data(), TEST_SIZE));
            REQUIRE(scheduler->get_bank_occupancy(bank_id) == TEST_SIZE);
        }

        // Verify each bank contains correct data
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            auto expected_data = generate_test_data(TEST_SIZE, static_cast<uint8_t>(bank_id));
            std::vector<uint8_t> read_buffer(TEST_SIZE);

            REQUIRE_NOTHROW(scheduler->direct_read(bank_id, 0, read_buffer.data(), TEST_SIZE));
            REQUIRE(verify_data(expected_data, read_buffer));
        }
    }

    SECTION("Bank configuration") {
        StorageScheduler::BankConfig custom_config{
            .bank_size_kb = 128,
            .cache_line_size = 128,
            .num_ports = 4,
            .access_pattern = StorageScheduler::AccessPattern::RANDOM,
            .enable_prefetch = false
        };

        REQUIRE_NOTHROW(scheduler->configure_bank(0, custom_config));
        REQUIRE(scheduler->get_bank_capacity(0) == 128 * 1024);
    }

    SECTION("Error handling") {
        constexpr size_t INVALID_BANK = 999;
        constexpr size_t TEST_SIZE = 100;
        auto test_data = generate_test_data(TEST_SIZE);

        // Invalid bank ID
        REQUIRE_THROWS(scheduler->direct_write(INVALID_BANK, 0, test_data.data(), TEST_SIZE));
        REQUIRE_THROWS(scheduler->direct_read(INVALID_BANK, 0, test_data.data(), TEST_SIZE));
        REQUIRE_FALSE(scheduler->is_ready(INVALID_BANK));

        // Address out of bounds
        Size bank_capacity = scheduler->get_bank_capacity(0);
        REQUIRE_THROWS(scheduler->direct_write(0, bank_capacity, test_data.data(), TEST_SIZE));
        REQUIRE_THROWS(scheduler->direct_read(0, bank_capacity, test_data.data(), TEST_SIZE));
    }
}

TEST_CASE_METHOD(StorageSchedulerTestFixture, "StorageScheduler Reset and Cleanup", "[memory_scheduler][cleanup]") {

    SECTION("Reset functionality") {
        constexpr size_t TEST_SIZE = 512;
        auto test_data = generate_test_data(TEST_SIZE);

        // Write data to all banks
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            scheduler->direct_write(bank_id, 0, test_data.data(), TEST_SIZE);
        }

        // Verify data is written
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(scheduler->get_bank_occupancy(bank_id) == TEST_SIZE);
        }

        // Reset scheduler
        scheduler->reset();

        // Verify all banks are reset
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(scheduler->get_bank_occupancy(bank_id) == 0);
            REQUIRE(scheduler->is_ready(bank_id));
            REQUIRE(scheduler->get_bank_operation(bank_id) == StorageScheduler::StorageOperation::BARRIER);
        }
    }

    SECTION("Flush all banks") {
        constexpr size_t TEST_SIZE = 256;
        auto test_data = generate_test_data(TEST_SIZE);

        // Write data to banks
        scheduler->direct_write(0, 0, test_data.data(), TEST_SIZE);
        scheduler->direct_write(1, 100, test_data.data(), TEST_SIZE);

        REQUIRE(scheduler->get_bank_occupancy(0) == TEST_SIZE);
        REQUIRE(scheduler->get_bank_occupancy(1) == 356); // 100 + 256

        // Flush all banks
        scheduler->flush_all_banks();

        // Verify banks are flushed
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(scheduler->get_bank_occupancy(bank_id) == 0);
        }
    }
}

TEST_CASE_METHOD(StorageSchedulerTestFixture, "Performance Metrics", "[memory_scheduler][metrics]") {

    SECTION("Basic performance tracking") {
        constexpr size_t TEST_SIZE = 512;
        constexpr size_t NUM_OPERATIONS = 10;
        auto test_data = generate_test_data(TEST_SIZE);
        std::vector<uint8_t> read_buffer(TEST_SIZE);

        // Perform multiple read/write operations
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            scheduler->direct_write(0, i * 64, test_data.data(), TEST_SIZE);
            scheduler->direct_read(0, i * 64, read_buffer.data(), TEST_SIZE);
        }

        auto metrics = scheduler->get_performance_metrics();

        REQUIRE(metrics.total_write_accesses >= NUM_OPERATIONS);
        REQUIRE(metrics.total_read_accesses >= NUM_OPERATIONS);
        REQUIRE(metrics.average_bank_utilization >= 0.0);
        REQUIRE(metrics.average_bank_utilization <= 1.0);
    }
}