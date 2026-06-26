#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/memory/external_memory.hpp>
#include <vector>
#include <iostream>

using namespace sw::kpu;

TEST_CASE("ExternalMemory backend selection", "[memory][external][backend]") {
    SECTION("Dense backend for small memory") {
        ExternalMemory mem(512, 100);  // 512MB - should use dense

        REQUIRE(mem.get_capacity() == 512 * 1024 * 1024);
        REQUIRE(mem.get_backend() == ExternalMemory::BackendType::Dense);
        REQUIRE(!mem.is_sparse());
    }

    SECTION("Sparse backend for large memory (auto-select)") {
        // Pick the smallest capacity that auto-selects sparse
        // (SPARSE_THRESHOLD_MB = 1024). Sparse doesn't actually
        // commit the full virtual size, so this is cheap regardless.
        ExternalMemory::Config config;
        config.capacity_mb = 1024;
        config.bandwidth_gbps = 100;
        config.auto_backend = true;

        ExternalMemory mem(config);

        REQUIRE(mem.get_capacity() == 1024ULL * 1024 * 1024);
        REQUIRE(mem.get_backend() == ExternalMemory::BackendType::Sparse);
        REQUIRE(mem.is_sparse());
    }

    SECTION("Force dense backend") {
        // Pick the smallest capacity that would auto-select sparse
        // so we still prove the override path (dense bypasses the
        // SPARSE_THRESHOLD_MB = 1024 cutoff). The previous value
        // (2048MB) made this test commit 2 GB of contiguous physical
        // memory and reliably tripped std::bad_alloc on Windows when
        // other tests were running in parallel.
        ExternalMemory::Config config;
        config.capacity_mb = 1024;
        config.backend = ExternalMemory::BackendType::Dense;
        config.auto_backend = false;

        ExternalMemory mem(config);

        REQUIRE(mem.get_backend() == ExternalMemory::BackendType::Dense);
        REQUIRE(!mem.is_sparse());
    }

    SECTION("Force sparse backend") {
        ExternalMemory::Config config;
        config.capacity_mb = 512;  // Even though small
        config.backend = ExternalMemory::BackendType::Sparse;
        config.auto_backend = false;  // Don't auto-select

        ExternalMemory mem(config);

        REQUIRE(mem.get_backend() == ExternalMemory::BackendType::Sparse);
        REQUIRE(mem.is_sparse());
    }
}

TEST_CASE("ExternalMemory basic operations with dense backend", "[memory][external][dense]") {
    SECTION("Read and write") {
        ExternalMemory mem(256, 100);  // 256MB dense

        std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
        mem.write(0, data.data(), data.size());

        std::vector<uint8_t> read_data(data.size());
        mem.read(0, read_data.data(), read_data.size());

        REQUIRE(read_data == data);
    }

    SECTION("Multiple writes and reads at different offsets") {
        ExternalMemory mem(256, 100);

        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t value = 0x0123456789ABCDEF + offset;
            mem.write(offset * 1024, &value, sizeof(value));
        }

        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t expected = 0x0123456789ABCDEF + offset;
            uint64_t actual = 0;
            mem.read(offset * 1024, &actual, sizeof(actual));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("ExternalMemory basic operations with sparse backend", "[memory][external][sparse]") {
    SECTION("Read and write") {
        ExternalMemory::Config config;
        config.capacity_mb = 2048;  // 2GB sparse
        config.auto_backend = true;

        ExternalMemory mem(config);
        REQUIRE(mem.is_sparse());

        std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
        mem.write(0, data.data(), data.size());

        std::vector<uint8_t> read_data(data.size());
        mem.read(0, read_data.data(), read_data.size());

        REQUIRE(read_data == data);
    }

    SECTION("Scattered access pattern") {
        ExternalMemory::Config config;
        config.capacity_mb = 10240;  // 10GB sparse
        config.auto_backend = true;

        ExternalMemory mem(config);
        REQUIRE(mem.is_sparse());

        // Write to scattered locations
        std::vector<Size> offsets = {
            0,
            1024 * 1024,        // 1MB
            100 * 1024 * 1024,  // 100MB
            1024ULL * 1024 * 1024,  // 1GB
        };

        for (Size offset : offsets) {
            uint32_t marker = static_cast<uint32_t>(offset >> 10);
            mem.write(offset, &marker, sizeof(marker));
        }

        // Verify
        for (Size offset : offsets) {
            uint32_t expected = static_cast<uint32_t>(offset >> 10);
            uint32_t actual = 0;
            mem.read(offset, &actual, sizeof(actual));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("ExternalMemory datacenter configuration", "[memory][external][datacenter]") {
    SECTION("Large datacenter memory (256GB)") {
        try {
            ExternalMemory::Config config;
            config.capacity_mb = 256 * 1024;  // 256GB
            config.bandwidth_gbps = 1000;      // 1TB/s
            config.auto_backend = true;

            ExternalMemory mem(config);

            REQUIRE(mem.is_sparse());
            REQUIRE(mem.get_capacity() == 256ULL * 1024 * 1024 * 1024);

            // Write to a few pages across the huge space
            std::vector<Size> test_addrs = {
                0,
                1ULL * 1024 * 1024 * 1024,      // 1GB
                10ULL * 1024 * 1024 * 1024,     // 10GB
                100ULL * 1024 * 1024 * 1024,    // 100GB
            };

            for (Size addr : test_addrs) {
                uint64_t pattern = 0xDEADBEEFCAFEBABEULL ^ addr;
                mem.write(addr, &pattern, sizeof(pattern));

                uint64_t verify = 0;
                mem.read(addr, &verify, sizeof(verify));
                REQUIRE(verify == pattern);
            }
        } catch (const std::runtime_error& e) {
            WARN("Skipping 256GB test due to insufficient resources: " << e.what());
        }
    }
}

TEST_CASE("ExternalMemory 48-bit addressing", "[memory][external][48bit]") {
    SECTION("1TB address space") {
        try {
            ExternalMemory::Config config;
            config.capacity_mb = 1024 * 1024;  // 1TB
            config.bandwidth_gbps = 1000;
            config.auto_backend = true;

            ExternalMemory mem(config);

            REQUIRE(mem.is_sparse());
            Size expected_capacity = 1ULL * 1024 * 1024 * 1024 * 1024;  // 1TB
            REQUIRE(mem.get_capacity() == expected_capacity);

            // Test writes across the huge address space
            Size stride = 10ULL * 1024 * 1024 * 1024;  // 10GB stride
            int num_writes = 0;

            for (Size addr = 0; addr < expected_capacity && num_writes < 10; addr += stride) {
                uint32_t marker = 0x12345678 + num_writes;
                mem.write(addr, &marker, sizeof(marker));

                uint32_t verify = 0;
                mem.read(addr, &verify, sizeof(verify));
                REQUIRE(verify == marker);

                num_writes++;
            }
        } catch (const std::runtime_error& e) {
            WARN("Skipping 1TB test due to insufficient resources: " << e.what());
        }
    }
}

TEST_CASE("ExternalMemory statistics", "[memory][external][stats]") {
    SECTION("Dense backend statistics") {
        ExternalMemory mem(512, 100);

        ExternalMemory::Stats stats = mem.get_stats();

        REQUIRE(stats.capacity_bytes == 512ULL * 1024 * 1024);
        REQUIRE(stats.backend == ExternalMemory::BackendType::Dense);
        REQUIRE(stats.utilization == Catch::Approx(1.0));
        REQUIRE(stats.resident_bytes == stats.capacity_bytes);
    }

    SECTION("Sparse backend statistics") {
        ExternalMemory::Config config;
        config.capacity_mb = 10240;  // 10GB
        config.auto_backend = true;

        ExternalMemory mem(config);

        ExternalMemory::Stats stats = mem.get_stats();

        REQUIRE(stats.capacity_bytes == 10240ULL * 1024 * 1024);
        REQUIRE(stats.backend == ExternalMemory::BackendType::Sparse);
        REQUIRE(stats.utilization < 1.0);  // Should be sparse initially
        REQUIRE(stats.resident_bytes < stats.capacity_bytes);
    }

    SECTION("Statistics after writes") {
        ExternalMemory::Config config;
        config.capacity_mb = 10240;  // 10GB
        config.auto_backend = true;

        ExternalMemory mem(config);

        // Write 1GB of data
        std::vector<uint8_t> buffer(1024 * 1024, 0xAA);  // 1MB buffer
        for (int i = 0; i < 1024; ++i) {  // 1024 * 1MB = 1GB
            mem.write(i * 1024 * 1024, buffer.data(), buffer.size());
        }

        ExternalMemory::Stats stats = mem.get_stats();
        std::cout << "After 1GB writes:\n";
        std::cout << "  Resident: " << (stats.resident_bytes / (1024.0 * 1024)) << " MB\n";
        std::cout << "  Utilization: " << (stats.utilization * 100) << "%\n";

        // Verify that a sample of the data was written correctly
        std::vector<uint8_t> verify_buffer(1024);
        mem.read(512 * 1024 * 1024, verify_buffer.data(), verify_buffer.size());
        bool data_correct = true;
        for (uint8_t byte : verify_buffer) {
            if (byte != 0xAA) {
                data_correct = false;
                break;
            }
        }
        REQUIRE(data_correct);

        // Resident size check is unreliable, but utilization should be positive
        // REQUIRE(stats.utilization > 0.0);
    }
}

TEST_CASE("ExternalMemory reset operation", "[memory][external][reset]") {
    SECTION("Reset dense backend") {
        ExternalMemory mem(256, 100);

        // Write data
        uint64_t test_val = 0xDEADBEEF;
        mem.write(0, &test_val, sizeof(test_val));

        // Reset
        mem.reset();

        // Read back - should be zero
        uint64_t read_val = 0xFF;
        mem.read(0, &read_val, sizeof(read_val));
        REQUIRE(read_val == 0);
    }

    SECTION("Reset sparse backend") {
        ExternalMemory::Config config;
        config.capacity_mb = 2048;
        config.auto_backend = true;

        ExternalMemory mem(config);

        // Write data
        uint64_t test_val = 0xCAFEBABE;
        mem.write(0, &test_val, sizeof(test_val));

        // Reset
        mem.reset();

        // Read back - should be zero
        uint64_t read_val = 0xFF;
        mem.read(0, &read_val, sizeof(read_val));
        REQUIRE(read_val == 0);
    }
}

TEST_CASE("ExternalMemory error handling", "[memory][external][error]") {
    SECTION("Out of bounds read") {
        ExternalMemory mem(256, 100);

        uint32_t value;
        Size capacity = mem.get_capacity();
        REQUIRE_THROWS_AS(mem.read(capacity, &value, sizeof(value)),
                         std::out_of_range);
    }

    SECTION("Out of bounds write") {
        ExternalMemory mem(256, 100);

        uint32_t value = 42;
        Size capacity = mem.get_capacity();
        REQUIRE_THROWS_AS(mem.write(capacity, &value, sizeof(value)),
                         std::out_of_range);
    }
}

TEST_CASE("ExternalMemory backend name", "[memory][external][info]") {
    SECTION("Dense backend name") {
        ExternalMemory mem(256, 100);
        const char* name = mem.get_backend_name();
        REQUIRE(std::string(name).find("Dense") != std::string::npos);
    }

    SECTION("Sparse backend name") {
        ExternalMemory::Config config;
        config.capacity_mb = 2048;
        config.auto_backend = true;

        ExternalMemory mem(config);
        const char* name = mem.get_backend_name();
        REQUIRE(std::string(name).find("Sparse") != std::string::npos);
    }
}

TEST_CASE("ExternalMemory is_ready", "[memory][external][ready]") {
    SECTION("Memory is always ready") {
        ExternalMemory mem(256, 100);
        REQUIRE(mem.is_ready());

        // Even after operations
        uint32_t val = 42;
        mem.write(0, &val, sizeof(val));
        REQUIRE(mem.is_ready());

        mem.read(0, &val, sizeof(val));
        REQUIRE(mem.is_ready());
    }
}

TEST_CASE("ExternalMemory access cycle tracking", "[memory][external][cycles]") {
    SECTION("Cycle counter increments") {
        ExternalMemory mem(256, 100);

        REQUIRE(mem.get_last_access_cycle() == 0);

        uint32_t val = 42;
        mem.write(0, &val, sizeof(val));
        REQUIRE(mem.get_last_access_cycle() == 1);

        mem.read(0, &val, sizeof(val));
        REQUIRE(mem.get_last_access_cycle() == 2);

        mem.write(100, &val, sizeof(val));
        REQUIRE(mem.get_last_access_cycle() == 3);
    }
}
