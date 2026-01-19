/**
 * @file test_xue.cpp
 * @brief Unit tests for XUE (eXecution Usage Events) framework
 * @version 0.3.3
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/xue/event_hierarchy.hpp>
#include <sw/xue/event_counter.hpp>
#include <sw/xue/event_collector.hpp>
#include <sw/xue/operational_analysis.hpp>

using namespace sw::xue;
using Catch::Approx;

TEST_CASE("XUE Event Hierarchy (v0.3.3)", "[xue][hierarchy][v033]") {

    SECTION("Event category classification") {
        // System events
        REQUIRE(get_category(EventType::SYSTEM_START) == EventCategory::SYSTEM);
        REQUIRE(get_category(EventType::KERNEL_END) == EventCategory::SYSTEM);

        // Compute events
        REQUIRE(get_category(EventType::MATMUL_16x16) == EventCategory::COMPUTE);
        REQUIRE(get_category(EventType::ELEM_RELU) == EventCategory::COMPUTE);
        REQUIRE(get_category(EventType::REDUCE_SOFTMAX) == EventCategory::COMPUTE);

        // Memory events
        REQUIRE(get_category(EventType::DRAM_READ) == EventCategory::MEMORY);
        REQUIRE(get_category(EventType::L3_TILE_PUSH) == EventCategory::MEMORY);
        REQUIRE(get_category(EventType::L2_CREDIT_RETURN) == EventCategory::MEMORY);

        // Data movement events
        REQUIRE(get_category(EventType::DMA_TRANSFER_START) == EventCategory::DATA_MOVEMENT);
        REQUIRE(get_category(EventType::BM_PUSH_L3_L2) == EventCategory::DATA_MOVEMENT);
        REQUIRE(get_category(EventType::STR_FEED_L2_L1) == EventCategory::DATA_MOVEMENT);

        // Synchronization events
        REQUIRE(get_category(EventType::SYNC_CREDIT_STALL) == EventCategory::SYNCHRONIZATION);
    }

    SECTION("Compute subcategory classification") {
        REQUIRE(get_compute_subcategory(EventType::MATMUL_16x16) == ComputeSubcategory::MATMUL);
        REQUIRE(get_compute_subcategory(EventType::MATMUL_ACCUMULATE) == ComputeSubcategory::MATMUL);
        REQUIRE(get_compute_subcategory(EventType::ELEM_ADD) == ComputeSubcategory::ELEMENTWISE);
        REQUIRE(get_compute_subcategory(EventType::ELEM_RELU) == ComputeSubcategory::ELEMENTWISE);
        REQUIRE(get_compute_subcategory(EventType::REDUCE_SUM) == ComputeSubcategory::REDUCTION);
        REQUIRE(get_compute_subcategory(EventType::REDUCE_SOFTMAX) == ComputeSubcategory::REDUCTION);
        REQUIRE(get_compute_subcategory(EventType::CONV_IM2COL) == ComputeSubcategory::SPECIAL);
    }

    SECTION("Memory subcategory classification") {
        REQUIRE(get_memory_subcategory(EventType::DRAM_READ) == MemorySubcategory::EXTERNAL);
        REQUIRE(get_memory_subcategory(EventType::DRAM_WRITE) == MemorySubcategory::EXTERNAL);
        REQUIRE(get_memory_subcategory(EventType::L3_TILE_PUSH) == MemorySubcategory::L3);
        REQUIRE(get_memory_subcategory(EventType::L2_TILE_POP) == MemorySubcategory::L2);
        REQUIRE(get_memory_subcategory(EventType::L1_STREAM_FEED) == MemorySubcategory::L1);
    }

    SECTION("Event type to string conversion") {
        REQUIRE(to_string(EventType::MATMUL_16x16) == "MATMUL_16x16");
        REQUIRE(to_string(EventType::DRAM_READ) == "DRAM_READ");
        REQUIRE(to_string(EventType::SYNC_CREDIT_STALL) == "SYNC_CREDIT_STALL");
    }

    SECTION("Category to string conversion") {
        REQUIRE(to_string(EventCategory::COMPUTE) == "COMPUTE");
        REQUIRE(to_string(EventCategory::MEMORY) == "MEMORY");
        REQUIRE(to_string(EventCategory::DATA_MOVEMENT) == "DATA_MOVEMENT");
    }
}

TEST_CASE("XUE Event Counter (v0.3.3)", "[xue][counter][v033]") {

    SECTION("Basic event counting") {
        EventCounter counter;

        // Record some events
        counter.record(EventType::MATMUL_16x16, EventMetadata::compute(512));
        counter.record(EventType::MATMUL_16x16, EventMetadata::compute(512));
        counter.record(EventType::MATMUL_16x16, EventMetadata::compute(512));

        auto stats = counter.get_stats(EventType::MATMUL_16x16);
        REQUIRE(stats.count.load() == 3);
        REQUIRE(stats.total_flops.load() == 1536);
    }

    SECTION("Memory event counting with bytes") {
        EventCounter counter;

        counter.record_memory(EventType::DRAM_READ, 4096);
        counter.record_memory(EventType::DRAM_READ, 4096);
        counter.record_memory(EventType::DRAM_WRITE, 2048);

        auto read_stats = counter.get_stats(EventType::DRAM_READ);
        REQUIRE(read_stats.count.load() == 2);
        REQUIRE(read_stats.total_bytes.load() == 8192);

        auto write_stats = counter.get_stats(EventType::DRAM_WRITE);
        REQUIRE(write_stats.count.load() == 1);
        REQUIRE(write_stats.total_bytes.load() == 2048);

        REQUIRE(counter.dram_bytes() == 10240);  // 8192 + 2048
    }

    SECTION("Category aggregation") {
        EventCounter counter;

        counter.record_compute(EventType::MATMUL_16x16, 512);
        counter.record_compute(EventType::ELEM_RELU, 100);
        counter.record_compute(EventType::REDUCE_SUM, 50);

        auto compute_stats = counter.get_category_stats(EventCategory::COMPUTE);
        REQUIRE(compute_stats.total_events == 3);
        REQUIRE(compute_stats.total_flops == 662);
    }

    SECTION("Arithmetic intensity calculation") {
        EventCounter counter;

        // 2*M*N*K FLOPs for M=N=K=64
        uint64_t flops = 2 * 64 * 64 * 64;  // 524288
        counter.record_compute(EventType::MATMUL_16x16, flops);

        // A, B, C matrices: 3 * 64 * 64 * 4 bytes = 49152 bytes
        uint64_t bytes = 3 * 64 * 64 * 4;
        counter.record_memory(EventType::DRAM_READ, 2 * 64 * 64 * 4);  // A and B
        counter.record_memory(EventType::DRAM_WRITE, 64 * 64 * 4);     // C

        double ai = counter.arithmetic_intensity();
        REQUIRE(ai == Approx(flops / static_cast<double>(bytes)).epsilon(0.01));
    }

    SECTION("Counter reset") {
        EventCounter counter;

        counter.record(EventType::MATMUL_16x16, EventMetadata::compute(512));
        REQUIRE(counter.get_stats(EventType::MATMUL_16x16).count.load() == 1);

        counter.reset();
        REQUIRE(counter.get_stats(EventType::MATMUL_16x16).count.load() == 0);
    }

    SECTION("JSON output") {
        EventCounter counter;

        counter.record_compute(EventType::MATMUL_16x16, 512);
        counter.record_memory(EventType::DRAM_READ, 4096);

        std::string json = counter.to_json();
        REQUIRE(json.find("\"version\": \"0.3.3\"") != std::string::npos);
        REQUIRE(json.find("\"total_flops\": 512") != std::string::npos);
        REQUIRE(json.find("\"MATMUL_16x16\"") != std::string::npos);
    }
}

TEST_CASE("XUE Event Collector (v0.3.3)", "[xue][collector][v033]") {

    // Reset the global collector before each test
    EventCollector::instance().reset();

    SECTION("Singleton access") {
        auto& xue1 = EventCollector::instance();
        auto& xue2 = EventCollector::instance();
        REQUIRE(&xue1 == &xue2);
    }

    SECTION("Enable/disable collection") {
        auto& xue = EventCollector::instance();

        xue.set_enabled(true);
        xue.record_matmul(16, 16, 16);
        REQUIRE(xue.counters().total_flops() > 0);

        xue.reset();
        xue.set_enabled(false);
        xue.record_matmul(16, 16, 16);
        REQUIRE(xue.counters().total_flops() == 0);

        xue.set_enabled(true);  // Re-enable for other tests
    }

    SECTION("Matmul event recording") {
        auto& xue = EventCollector::instance();
        xue.reset();

        // 64x64x64 matmul
        xue.record_matmul(64, 64, 64);

        // Should record tile events
        auto stats = xue.counters().get_stats(EventType::MATMUL_16x16);
        REQUIRE(stats.count.load() > 0);

        // Total FLOPs should be 2*64*64*64 = 524288
        // But recorded as tile FLOPs
        uint64_t total_flops = xue.counters().total_flops();
        REQUIRE(total_flops > 0);
    }

    SECTION("Memory event recording") {
        auto& xue = EventCollector::instance();
        xue.reset();

        xue.record_dram_read(4096);
        xue.record_dram_write(2048);
        xue.record_l3_push(1024);

        REQUIRE(xue.counters().dram_bytes() == 6144);
    }

    SECTION("Kernel scope RAII") {
        auto& xue = EventCollector::instance();
        xue.reset();

        {
            XUE_KERNEL_SCOPE("test_kernel");
            xue.record_matmul(16, 16, 16);
        }

        auto start_stats = xue.counters().get_stats(EventType::KERNEL_START);
        auto end_stats = xue.counters().get_stats(EventType::KERNEL_END);

        REQUIRE(start_stats.count.load() == 1);
        REQUIRE(end_stats.count.load() == 1);
    }

    SECTION("JSON output") {
        auto& xue = EventCollector::instance();
        xue.reset();

        xue.record_matmul(32, 32, 32);
        xue.record_dram_read(4096);

        std::string json = xue.to_json();
        REQUIRE(json.find("\"version\": \"0.3.3\"") != std::string::npos);
    }
}

TEST_CASE("XUE Operational Analysis (v0.3.3)", "[xue][analysis][v033]") {

    SECTION("Hardware model ridge points") {
        HardwareModel hw;
        hw.peak_gflops = 1024.0;
        hw.dram_bandwidth_gbs = 64.0;
        hw.l3_bandwidth_gbs = 128.0;
        hw.l2_bandwidth_gbs = 256.0;

        REQUIRE(hw.ridge_point_dram() == Approx(16.0));   // 1024/64
        REQUIRE(hw.ridge_point_l3() == Approx(8.0));      // 1024/128
        REQUIRE(hw.ridge_point_l2() == Approx(4.0));      // 1024/256
    }

    SECTION("Roofline prediction - compute bound") {
        HardwareModel hw;
        hw.peak_gflops = 1024.0;
        hw.dram_bandwidth_gbs = 64.0;

        // AI = 32 > ridge point (16) -> compute bound
        double ai = 32.0;
        double predicted = hw.predict_gflops(ai);

        REQUIRE(predicted == Approx(1024.0));
        REQUIRE(hw.bottleneck(ai) == "compute");
    }

    SECTION("Roofline prediction - memory bound") {
        HardwareModel hw;
        hw.peak_gflops = 1024.0;
        hw.dram_bandwidth_gbs = 64.0;

        // AI = 8 < ridge point (16) -> memory bound
        double ai = 8.0;
        double predicted = hw.predict_gflops(ai);

        REQUIRE(predicted == Approx(512.0));  // 8 * 64
        REQUIRE(hw.bottleneck(ai) == "dram");
    }

    SECTION("Full operational analysis") {
        EventCounter events;

        // Simulate 1024x1024x1024 matmul
        uint64_t M = 1024, N = 1024, K = 1024;
        uint64_t flops = 2 * M * N * K;

        events.record_compute(EventType::MATMUL_16x16, flops);

        // Memory traffic (simplified)
        uint64_t bytes = 3 * M * N * 4;  // A, B, C in float
        events.record_memory(EventType::DRAM_READ, 2 * M * K * 4);
        events.record_memory(EventType::DRAM_WRITE, M * N * 4);

        OperationalAnalyzer analyzer;
        auto result = analyzer.analyze(events);

        REQUIRE(result.total_flops == flops);
        REQUIRE(result.arithmetic_intensity > 0);
        REQUIRE(result.predicted_gflops > 0);
        REQUIRE(!result.predicted_bottleneck.empty());
    }

    SECTION("Validation against simulation") {
        EventCounter events;

        uint64_t flops = 2 * 512 * 512 * 512;
        events.record_compute(EventType::MATMUL_16x16, flops);
        events.record_memory(EventType::DRAM_READ, 512 * 512 * 4 * 2);
        events.record_memory(EventType::DRAM_WRITE, 512 * 512 * 4);

        OperationalAnalyzer analyzer;
        auto result = analyzer.analyze(events);

        // Simulate "actual" results close to prediction
        double actual_gflops = result.predicted_gflops * 0.95;  // 5% under
        uint64_t actual_cycles = static_cast<uint64_t>(result.predicted_cycles * 1.05);

        auto validation = analyzer.validate(events, actual_gflops, actual_cycles);

        REQUIRE(validation.within_10_percent);
        REQUIRE(validation.gflops_error_percent < 10.0);
    }
}

TEST_CASE("XUE I/O Complexity Analysis (v0.3.3)", "[xue][io_complexity][v033]") {

    SECTION("Hong-Kung I/O lower bound") {
        uint64_t M = 1024, N = 1024, K = 1024;
        uint64_t L3_bytes = 256ULL * 1024;  // 256 KB

        uint64_t lower_bound = IOComplexityAnalyzer::matmul_io_lower_bound(
            M, N, K, L3_bytes, 4);

        // Q >= MNK / sqrt(M_fast)
        // M_fast = 256KB / 4 = 64K elements
        // sqrt(64K) = 256
        // Q >= 1024^3 / 256 = 4M elements
        REQUIRE(lower_bound > 0);
        REQUIRE(lower_bound > M * N);  // Must be more than output matrix
    }

    SECTION("Optimal tile size calculation") {
        uint64_t L3_bytes = 256ULL * 1024;  // 256 KB

        uint64_t tile_size = IOComplexityAnalyzer::optimal_tile_size(L3_bytes, 4);

        // sqrt(64K / 3) ≈ 146
        REQUIRE(tile_size > 100);
        REQUIRE(tile_size < 200);

        // 3 tiles should fit in L3
        REQUIRE(3 * tile_size * tile_size * 4 <= L3_bytes);
    }

    SECTION("Reuse factor calculation") {
        uint64_t flops = 2ULL * 1024 * 1024 * 1024;  // 2 GFLOPs
        uint64_t io_bytes = 3ULL * 1024 * 1024 * 4;  // 12 MB

        double reuse = IOComplexityAnalyzer::reuse_factor(flops, io_bytes, 4);

        // reuse = flops / io_elements = 2G / 3M = 666
        REQUIRE(reuse > 500);
    }

    SECTION("I/O efficiency vs optimal") {
        uint64_t flops = 2ULL * 1024 * 1024 * 1024;
        uint64_t io_bytes = 3ULL * 1024 * 1024 * 4;
        uint64_t L3_bytes = 256ULL * 1024;

        double efficiency = IOComplexityAnalyzer::io_efficiency(
            flops, io_bytes, L3_bytes, 4);

        // Should be > 1 if achieving better than sqrt(M) reuse
        REQUIRE(efficiency > 0);
    }
}

TEST_CASE("XUE KPU Constants", "[xue][constants][v033]") {

    SECTION("Systolic array constants") {
        REQUIRE(KPUConstants::SYSTOLIC_SIZE == 16);
        REQUIRE(KPUConstants::TILE_FLOPS == 512);  // 2*16*16*16/16 = 512
    }

    SECTION("Buffer size constants") {
        REQUIRE(KPUConstants::L1_BUFFER_SIZE == 8192);
        REQUIRE(KPUConstants::L2_TILE_SIZE == 65536);
        REQUIRE(KPUConstants::L3_TILE_SIZE == 262144);
    }
}
