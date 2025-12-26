// Test suite for KPU Runtime
// Tests memory management, kernel launching, streams, and events

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/runtime/runtime.hpp>
#include <sw/kpu/kernel.hpp>

#include <vector>
#include <cstring>

using namespace sw::runtime;
using namespace sw::kpu;

// ============================================================================
// Test Fixture
// ============================================================================

class RuntimeTestFixture {
protected:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> simulator;
    std::unique_ptr<KPURuntime> runtime;

    RuntimeTestFixture() {
        // Create a small simulator configuration for testing
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.page_buffer_count = 2;
        config.page_buffer_capacity_kb = 64;
        config.l1_buffer_count = 4;
        config.l1_buffer_capacity_kb = 64;
        config.dma_engine_count = 2;
        config.block_mover_count = 4;
        config.streamer_count = 8;
        config.processor_array_rows = 16;
        config.processor_array_cols = 16;
        config.use_systolic_array_mode = true;

        simulator = std::make_unique<KPUSimulator>(config);
        runtime = std::make_unique<KPURuntime>(simulator.get());
    }
};

// ============================================================================
// KPURuntime Construction Tests
// ============================================================================

TEST_CASE("KPURuntime construction", "[runtime]") {
    SECTION("Basic construction") {
        KPUSimulator::Config config;
        KPUSimulator sim(config);
        KPURuntime runtime(&sim);

        REQUIRE(runtime.simulator() == &sim);
        REQUIRE(runtime.resource_manager() != nullptr);
    }

    SECTION("Construction with config") {
        KPUSimulator::Config sim_config;
        KPUSimulator sim(sim_config);

        KPURuntime::Config rt_config;
        rt_config.verbose = true;
        rt_config.clock_ghz = 2.0;

        KPURuntime runtime(&sim, rt_config);

        REQUIRE(runtime.config().verbose == true);
        REQUIRE(runtime.config().clock_ghz == Catch::Approx(2.0));
    }

    SECTION("Null simulator throws") {
        REQUIRE_THROWS_AS(KPURuntime(nullptr), std::invalid_argument);
    }
}

// ============================================================================
// Memory Management Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime memory management", "[runtime][memory]") {
    SECTION("malloc and free") {
        Address ptr = runtime->malloc(1024);
        REQUIRE(ptr != 0);

        // Free should not throw
        REQUIRE_NOTHROW(runtime->free(ptr));
    }

    SECTION("Multiple allocations") {
        Address ptr1 = runtime->malloc(1024);
        Address ptr2 = runtime->malloc(2048);
        Address ptr3 = runtime->malloc(512);

        REQUIRE(ptr1 != 0);
        REQUIRE(ptr2 != 0);
        REQUIRE(ptr3 != 0);

        // All addresses should be different
        REQUIRE(ptr1 != ptr2);
        REQUIRE(ptr2 != ptr3);
        REQUIRE(ptr1 != ptr3);

        runtime->free(ptr1);
        runtime->free(ptr2);
        runtime->free(ptr3);
    }

    SECTION("malloc with alignment") {
        Address ptr = runtime->malloc(1024, 128);
        REQUIRE(ptr != 0);
        REQUIRE(ptr % 128 == 0);  // Check alignment

        runtime->free(ptr);
    }

    SECTION("free of null is safe") {
        REQUIRE_NOTHROW(runtime->free(0));
    }

    SECTION("Memory info") {
        Size total_before = runtime->get_total_memory();
        Size free_before = runtime->get_free_memory();

        REQUIRE(total_before > 0);
        REQUIRE(free_before > 0);
        REQUIRE(free_before <= total_before);

        // Allocate some memory
        Address ptr = runtime->malloc(1024 * 1024);  // 1 MB
        REQUIRE(ptr != 0);

        Size free_after = runtime->get_free_memory();
        REQUIRE(free_after < free_before);

        runtime->free(ptr);
    }
}

// ============================================================================
// Memory Copy Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime memory copy", "[runtime][memory]") {
    SECTION("memcpy_h2d and memcpy_d2h") {
        const Size size = 1024;
        std::vector<float> host_src(size / sizeof(float), 3.14f);
        std::vector<float> host_dst(size / sizeof(float), 0.0f);

        Address device_ptr = runtime->malloc(size);
        REQUIRE(device_ptr != 0);

        // Copy host to device
        REQUIRE_NOTHROW(runtime->memcpy_h2d(device_ptr, host_src.data(), size));

        // Copy device to host
        REQUIRE_NOTHROW(runtime->memcpy_d2h(host_dst.data(), device_ptr, size));

        // Verify data
        for (size_t i = 0; i < host_dst.size(); ++i) {
            REQUIRE(host_dst[i] == Catch::Approx(3.14f));
        }

        runtime->free(device_ptr);
    }

    SECTION("memcpy_d2d") {
        const Size size = 1024;
        std::vector<float> host_data(size / sizeof(float), 2.71f);
        std::vector<float> host_result(size / sizeof(float), 0.0f);

        Address src = runtime->malloc(size);
        Address dst = runtime->malloc(size);
        REQUIRE(src != 0);
        REQUIRE(dst != 0);

        // Initialize source
        runtime->memcpy_h2d(src, host_data.data(), size);

        // Copy device to device
        REQUIRE_NOTHROW(runtime->memcpy_d2d(dst, src, size));

        // Verify by copying back
        runtime->memcpy_d2h(host_result.data(), dst, size);

        for (size_t i = 0; i < host_result.size(); ++i) {
            REQUIRE(host_result[i] == Catch::Approx(2.71f));
        }

        runtime->free(src);
        runtime->free(dst);
    }

    SECTION("memset") {
        const Size size = 1024;
        Address ptr = runtime->malloc(size);
        REQUIRE(ptr != 0);

        // Set memory to a value
        REQUIRE_NOTHROW(runtime->memset(ptr, 0xFF, size));

        // Verify by reading back
        std::vector<uint8_t> buffer(size);
        runtime->memcpy_d2h(buffer.data(), ptr, size);

        for (size_t i = 0; i < buffer.size(); ++i) {
            REQUIRE(buffer[i] == 0xFF);
        }

        runtime->free(ptr);
    }
}

// ============================================================================
// Kernel Launch Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime kernel launch", "[runtime][kernel]") {
    SECTION("Launch matmul kernel") {
        const Size M = 64, N = 64, K = 64;
        const Size elem_size = sizeof(float);

        // Create kernel
        Kernel kernel = Kernel::create_matmul(M, N, K);
        REQUIRE(kernel.is_valid());

        // Allocate memory
        Address A = runtime->malloc(M * K * elem_size);
        Address B = runtime->malloc(K * N * elem_size);
        Address C = runtime->malloc(M * N * elem_size);
        REQUIRE(A != 0);
        REQUIRE(B != 0);
        REQUIRE(C != 0);

        // Launch kernel
        auto result = runtime->launch(kernel, {A, B, C});

        REQUIRE(result.success);
        REQUIRE(result.cycles > 0);
        REQUIRE(result.error.empty());

        // Cleanup
        runtime->free(A);
        runtime->free(B);
        runtime->free(C);
    }

    SECTION("Launch with wrong argument count fails") {
        Kernel kernel = Kernel::create_matmul(64, 64, 64);

        Address A = runtime->malloc(64 * 64 * sizeof(float));
        Address B = runtime->malloc(64 * 64 * sizeof(float));

        // Only 2 args, but kernel expects 3
        auto result = runtime->launch(kernel, {A, B});

        REQUIRE_FALSE(result.success);
        REQUIRE(result.error.find("mismatch") != std::string::npos);

        runtime->free(A);
        runtime->free(B);
    }

    SECTION("Launch with null address fails") {
        Kernel kernel = Kernel::create_matmul(64, 64, 64);

        Address A = runtime->malloc(64 * 64 * sizeof(float));
        Address B = runtime->malloc(64 * 64 * sizeof(float));

        // C is null
        auto result = runtime->launch(kernel, {A, B, 0});

        REQUIRE_FALSE(result.success);
        REQUIRE(result.error.find("Null") != std::string::npos);

        runtime->free(A);
        runtime->free(B);
    }

    SECTION("Launch count tracking") {
        Kernel kernel = Kernel::create_matmul(32, 32, 32);

        Address A = runtime->malloc(32 * 32 * sizeof(float));
        Address B = runtime->malloc(32 * 32 * sizeof(float));
        Address C = runtime->malloc(32 * 32 * sizeof(float));

        size_t initial_count = runtime->get_launch_count();
        Cycle initial_cycles = runtime->get_total_cycles();

        runtime->launch(kernel, {A, B, C});
        runtime->launch(kernel, {A, B, C});

        REQUIRE(runtime->get_launch_count() == initial_count + 2);
        REQUIRE(runtime->get_total_cycles() > initial_cycles);

        runtime->free(A);
        runtime->free(B);
        runtime->free(C);
    }
}

// ============================================================================
// Stream Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime streams", "[runtime][stream]") {
    SECTION("Default stream") {
        Stream s = runtime->default_stream();
        REQUIRE(s.valid == true);
        REQUIRE(s.id == 0);
    }

    SECTION("Create and destroy stream") {
        Stream s = runtime->create_stream();
        REQUIRE(s.valid == true);
        REQUIRE(s.id > 0);

        REQUIRE_NOTHROW(runtime->destroy_stream(s));
    }

    SECTION("Multiple streams") {
        Stream s1 = runtime->create_stream();
        Stream s2 = runtime->create_stream();
        Stream s3 = runtime->create_stream();

        REQUIRE(s1.id != s2.id);
        REQUIRE(s2.id != s3.id);

        runtime->destroy_stream(s1);
        runtime->destroy_stream(s2);
        runtime->destroy_stream(s3);
    }

    SECTION("Stream synchronize") {
        Stream s = runtime->create_stream();

        // Should not throw even with nothing queued
        REQUIRE_NOTHROW(runtime->stream_synchronize(s));

        runtime->destroy_stream(s);
    }
}

// ============================================================================
// Event Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime events", "[runtime][event]") {
    SECTION("Create and destroy event") {
        Event e = runtime->create_event();
        REQUIRE(e.valid == true);
        REQUIRE(e.id > 0);

        REQUIRE_NOTHROW(runtime->destroy_event(e));
    }

    SECTION("Record and wait event") {
        Event e = runtime->create_event();
        Stream s = runtime->default_stream();

        REQUIRE_NOTHROW(runtime->record_event(e, s));
        REQUIRE_NOTHROW(runtime->wait_event(e));

        runtime->destroy_event(e);
    }

    SECTION("Elapsed time between events") {
        // Launch some work to get non-zero cycles
        Kernel kernel = Kernel::create_matmul(64, 64, 64);
        Address A = runtime->malloc(64 * 64 * sizeof(float));
        Address B = runtime->malloc(64 * 64 * sizeof(float));
        Address C = runtime->malloc(64 * 64 * sizeof(float));

        Event start = runtime->create_event();
        Event end = runtime->create_event();
        Stream s = runtime->default_stream();

        runtime->record_event(start, s);
        runtime->launch(kernel, {A, B, C});
        runtime->record_event(end, s);

        float elapsed = runtime->elapsed_time(start, end);
        REQUIRE(elapsed >= 0.0f);

        runtime->destroy_event(start);
        runtime->destroy_event(end);
        runtime->free(A);
        runtime->free(B);
        runtime->free(C);
    }
}

// ============================================================================
// Synchronization Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime synchronization", "[runtime][sync]") {
    SECTION("Global synchronize") {
        // Should not throw
        REQUIRE_NOTHROW(runtime->synchronize());
    }

    SECTION("Synchronize after launches") {
        Kernel kernel = Kernel::create_matmul(32, 32, 32);
        Address A = runtime->malloc(32 * 32 * sizeof(float));
        Address B = runtime->malloc(32 * 32 * sizeof(float));
        Address C = runtime->malloc(32 * 32 * sizeof(float));

        runtime->launch(kernel, {A, B, C});
        runtime->launch(kernel, {A, B, C});

        REQUIRE_NOTHROW(runtime->synchronize());

        runtime->free(A);
        runtime->free(B);
        runtime->free(C);
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE_METHOD(RuntimeTestFixture, "KPURuntime statistics", "[runtime][stats]") {
    SECTION("Print stats") {
        // Should not throw
        REQUIRE_NOTHROW(runtime->print_stats());
    }

    SECTION("Stats after work") {
        Kernel kernel = Kernel::create_matmul(64, 64, 64);
        Address A = runtime->malloc(64 * 64 * sizeof(float));
        Address B = runtime->malloc(64 * 64 * sizeof(float));
        Address C = runtime->malloc(64 * 64 * sizeof(float));

        runtime->launch(kernel, {A, B, C});

        REQUIRE(runtime->get_launch_count() >= 1);
        REQUIRE(runtime->get_total_cycles() > 0);

        runtime->free(A);
        runtime->free(B);
        runtime->free(C);
    }
}
