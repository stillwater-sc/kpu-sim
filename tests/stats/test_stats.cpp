/**
 * @file test_stats.cpp
 * @brief Unit tests for Statistics Collection Framework
 * @version 0.3.4
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/stats/cycle_breakdown.hpp>
#include <sw/kpu/stats/memory_traffic.hpp>
#include <sw/kpu/stats/utilization.hpp>
#include <sw/kpu/stats/stats_collector.hpp>

using namespace sw::kpu::stats;
using Catch::Approx;

// ============================================================================
// Cycle Breakdown Tests
// ============================================================================

TEST_CASE("Cycle Breakdown (v0.3.4)", "[stats][cycles][v034]") {

    SECTION("Basic cycle recording") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 1000);
        breakdown.record(CycleCategory::MEMORY_ACCESS, 200);
        breakdown.record(CycleCategory::MEMORY_STALL, 100);
        breakdown.record(CycleCategory::IDLE, 300);

        REQUIRE(breakdown.get(CycleCategory::COMPUTE) == 1000);
        REQUIRE(breakdown.get(CycleCategory::MEMORY_ACCESS) == 200);
        REQUIRE(breakdown.get(CycleCategory::MEMORY_STALL) == 100);
        REQUIRE(breakdown.get(CycleCategory::IDLE) == 300);
        REQUIRE(breakdown.total_cycles() == 1600);
    }

    SECTION("Compute cycles aggregation") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 500);
        breakdown.record(CycleCategory::COMPUTE, 500);

        REQUIRE(breakdown.compute_cycles() == 1000);
    }

    SECTION("Stall cycles aggregation") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::MEMORY_STALL, 100);
        breakdown.record(CycleCategory::CREDIT_STALL, 50);
        breakdown.record(CycleCategory::DATA_STALL, 75);
        breakdown.record(CycleCategory::SYNC_STALL, 25);

        REQUIRE(breakdown.stall_cycles() == 250);
    }

    SECTION("Memory cycles (access + stall)") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::MEMORY_ACCESS, 200);
        breakdown.record(CycleCategory::MEMORY_STALL, 100);

        REQUIRE(breakdown.memory_cycles() == 300);
    }

    SECTION("Percentage calculation") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 500);
        breakdown.record(CycleCategory::IDLE, 500);

        REQUIRE(breakdown.percentage(CycleCategory::COMPUTE) == Approx(50.0));
        REQUIRE(breakdown.percentage(CycleCategory::IDLE) == Approx(50.0));
    }

    SECTION("Compute efficiency") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 800);
        breakdown.record(CycleCategory::MEMORY_STALL, 200);

        // Efficiency = compute / (compute + stall) = 800 / 1000 = 0.8
        REQUIRE(breakdown.compute_efficiency() == Approx(0.8));
    }

    SECTION("Utilization (non-idle)") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 700);
        breakdown.record(CycleCategory::MEMORY_ACCESS, 100);
        breakdown.record(CycleCategory::IDLE, 200);

        // Utilization = (1000 - 200) / 1000 = 0.8
        REQUIRE(breakdown.utilization() == Approx(0.8));
    }

    SECTION("Reset clears all counters") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 1000);
        breakdown.record(CycleCategory::IDLE, 500);

        breakdown.reset();

        REQUIRE(breakdown.total_cycles() == 0);
        REQUIRE(breakdown.compute_cycles() == 0);
    }

    SECTION("Merge combines breakdowns") {
        CycleBreakdown a, b;

        a.record(CycleCategory::COMPUTE, 500);
        a.record(CycleCategory::MEMORY_STALL, 100);

        b.record(CycleCategory::COMPUTE, 300);
        b.record(CycleCategory::IDLE, 200);

        a.merge(b);

        REQUIRE(a.get(CycleCategory::COMPUTE) == 800);
        REQUIRE(a.get(CycleCategory::MEMORY_STALL) == 100);
        REQUIRE(a.get(CycleCategory::IDLE) == 200);
    }

    SECTION("JSON output") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 1000);

        std::string json = breakdown.to_json();
        REQUIRE(json.find("\"total_cycles\": 1000") != std::string::npos);
        REQUIRE(json.find("\"compute_efficiency\"") != std::string::npos);
    }

    SECTION("Summary output") {
        CycleBreakdown breakdown;

        breakdown.record(CycleCategory::COMPUTE, 1000);
        breakdown.record(CycleCategory::IDLE, 500);

        std::string summary = breakdown.summary();
        REQUIRE(summary.find("Cycle Breakdown") != std::string::npos);
        REQUIRE(summary.find("compute") != std::string::npos);
    }

    SECTION("Category to_string") {
        REQUIRE(std::string(to_string(CycleCategory::COMPUTE)) == "compute");
        REQUIRE(std::string(to_string(CycleCategory::MEMORY_STALL)) == "memory_stall");
        REQUIRE(std::string(to_string(CycleCategory::CREDIT_STALL)) == "credit_stall");
        REQUIRE(std::string(to_string(CycleCategory::IDLE)) == "idle");
    }
}

// ============================================================================
// Memory Traffic Tests
// ============================================================================

TEST_CASE("Memory Traffic (v0.3.4)", "[stats][memory][v034]") {

    SECTION("Basic traffic recording") {
        MemoryTraffic traffic;

        traffic.record_read(MemoryLevel::EXTERNAL, 4096);
        traffic.record_write(MemoryLevel::EXTERNAL, 2048);

        REQUIRE(traffic.total_bytes(MemoryLevel::EXTERNAL) == 6144);
        REQUIRE(traffic.external_bytes() == 6144);
    }

    SECTION("Multi-level traffic") {
        MemoryTraffic traffic;

        traffic.record_read(MemoryLevel::EXTERNAL, 1024);
        traffic.record_read(MemoryLevel::L3, 2048);
        traffic.record_read(MemoryLevel::L2, 4096);
        traffic.record_read(MemoryLevel::L1, 8192);

        REQUIRE(traffic.total_bytes(MemoryLevel::EXTERNAL) == 1024);
        REQUIRE(traffic.total_bytes(MemoryLevel::L3) == 2048);
        REQUIRE(traffic.total_bytes(MemoryLevel::L2) == 4096);
        REQUIRE(traffic.total_bytes(MemoryLevel::L1) == 8192);
        REQUIRE(traffic.total_traffic() == 15360);
    }

    SECTION("Traffic amplification") {
        MemoryTraffic traffic;

        // 1 KB external, 10 KB internal -> 9x amplification
        traffic.record_read(MemoryLevel::EXTERNAL, 1024);
        traffic.record_read(MemoryLevel::L3, 4096);
        traffic.record_read(MemoryLevel::L2, 4096);
        traffic.record_read(MemoryLevel::L1, 1024);

        // Amplification = internal / external = 9216 / 1024 = 9
        REQUIRE(traffic.traffic_amplification() == Approx(9.0));
    }

    SECTION("Level traffic details") {
        MemoryTraffic traffic;

        traffic.record_read(MemoryLevel::EXTERNAL, 4096, 10);
        traffic.record_read(MemoryLevel::EXTERNAL, 4096, 10);
        traffic.record_write(MemoryLevel::EXTERNAL, 2048, 5);

        const auto& level = traffic.get_level(MemoryLevel::EXTERNAL);
        REQUIRE(level.read_bytes.load() == 8192);
        REQUIRE(level.write_bytes.load() == 2048);
        REQUIRE(level.read_count.load() == 2);
        REQUIRE(level.write_count.load() == 1);
        REQUIRE(level.avg_read_size() == Approx(4096.0));
    }

    SECTION("Bandwidth calculations") {
        MemoryTraffic traffic;

        // 64 GB of traffic in 1 billion cycles at 1 GHz (use 1e9 for GB, not GiB)
        traffic.record_read(MemoryLevel::EXTERNAL, static_cast<uint64_t>(64e9));

        double achieved = traffic.achieved_bandwidth(MemoryLevel::EXTERNAL, 1.0, 1000000000);
        REQUIRE(achieved == Approx(64.0));  // 64 GB/s
    }

    SECTION("Bandwidth utilization") {
        MemoryTraffic traffic;
        traffic.set_peak_bandwidth(MemoryLevel::EXTERNAL, 100.0);  // 100 GB/s peak

        // 50 GB in 1B cycles = 50 GB/s = 50% utilization (use 1e9 for GB, not GiB)
        traffic.record_read(MemoryLevel::EXTERNAL, static_cast<uint64_t>(50e9));

        double util = traffic.bandwidth_utilization(MemoryLevel::EXTERNAL, 1.0, 1000000000);
        REQUIRE(util == Approx(0.5));
    }

    SECTION("Reset clears all levels") {
        MemoryTraffic traffic;

        traffic.record_read(MemoryLevel::EXTERNAL, 4096);
        traffic.record_read(MemoryLevel::L3, 2048);

        traffic.reset();

        REQUIRE(traffic.total_traffic() == 0);
    }

    SECTION("Merge combines traffic") {
        MemoryTraffic a, b;

        a.record_read(MemoryLevel::EXTERNAL, 1024);
        b.record_read(MemoryLevel::EXTERNAL, 1024);

        a.merge(b);

        REQUIRE(a.external_bytes() == 2048);
    }

    SECTION("JSON output") {
        MemoryTraffic traffic;

        traffic.record_read(MemoryLevel::EXTERNAL, 4096);

        std::string json = traffic.to_json();
        REQUIRE(json.find("\"total_traffic_bytes\"") != std::string::npos);
        REQUIRE(json.find("\"external\"") != std::string::npos);
    }

    SECTION("Memory level to_string") {
        REQUIRE(std::string(to_string(MemoryLevel::EXTERNAL)) == "external");
        REQUIRE(std::string(to_string(MemoryLevel::L3)) == "l3");
        REQUIRE(std::string(to_string(MemoryLevel::L2)) == "l2");
        REQUIRE(std::string(to_string(MemoryLevel::L1)) == "l1");
    }
}

// ============================================================================
// Resource Utilization Tests
// ============================================================================

TEST_CASE("Resource Utilization (v0.3.4)", "[stats][utilization][v034]") {

    SECTION("Basic utilization recording") {
        ResourceUtilization util;

        util.record_busy(800);
        util.record_idle(100);
        util.record_stall(100);

        REQUIRE(util.busy_cycles.load() == 800);
        REQUIRE(util.idle_cycles.load() == 100);
        REQUIRE(util.stall_cycles.load() == 100);
        REQUIRE(util.total_cycles() == 1000);
    }

    SECTION("Utilization calculation") {
        ResourceUtilization util;

        util.record_busy(700);
        util.record_idle(200);
        util.record_stall(100);

        // U = busy / total = 700 / 1000 = 0.7
        REQUIRE(util.utilization() == Approx(0.7));
    }

    SECTION("Efficiency calculation") {
        ResourceUtilization util;

        util.record_busy(800);
        util.record_stall(200);

        // E = busy / (busy + stall) = 800 / 1000 = 0.8
        REQUIRE(util.efficiency() == Approx(0.8));
    }

    SECTION("Throughput calculation") {
        ResourceUtilization util;

        util.record_operation(100);  // 100 work units
        util.record_busy(1000);

        // X = operations / total_cycles = 1 / 1000 = 0.001
        REQUIRE(util.throughput() == Approx(0.001));
    }

    SECTION("Operation recording with work units") {
        ResourceUtilization util;

        util.record_operation(512);  // 512 FLOPs
        util.record_operation(512);
        util.record_operation(512);

        REQUIRE(util.operations.load() == 3);
        REQUIRE(util.work_units.load() == 1536);
    }

    SECTION("Utilization tracker - multi-resource") {
        UtilizationTracker tracker;

        tracker.register_resource(ResourceType::SYSTOLIC_ARRAY, 0);
        tracker.register_resource(ResourceType::MEMORY_CONTROLLER, 0);
        tracker.register_resource(ResourceType::MEMORY_CONTROLLER, 1);

        tracker.record_busy(ResourceType::SYSTOLIC_ARRAY, 0, 900);
        tracker.record_idle(ResourceType::SYSTOLIC_ARRAY, 0, 100);

        tracker.record_busy(ResourceType::MEMORY_CONTROLLER, 0, 700);
        tracker.record_idle(ResourceType::MEMORY_CONTROLLER, 0, 300);

        tracker.record_busy(ResourceType::MEMORY_CONTROLLER, 1, 500);
        tracker.record_idle(ResourceType::MEMORY_CONTROLLER, 1, 500);

        // Systolic: 0.9 utilization
        REQUIRE(tracker.average_utilization(ResourceType::SYSTOLIC_ARRAY) == Approx(0.9));

        // MC: (0.7 + 0.5) / 2 = 0.6
        REQUIRE(tracker.average_utilization(ResourceType::MEMORY_CONTROLLER) == Approx(0.6));
    }

    SECTION("Aggregate utilization") {
        UtilizationTracker tracker;

        tracker.record_busy(ResourceType::MEMORY_CONTROLLER, 0, 700);
        tracker.record_idle(ResourceType::MEMORY_CONTROLLER, 0, 300);
        tracker.record_busy(ResourceType::MEMORY_CONTROLLER, 1, 500);
        tracker.record_idle(ResourceType::MEMORY_CONTROLLER, 1, 500);

        auto agg = tracker.get_aggregate(ResourceType::MEMORY_CONTROLLER);
        REQUIRE(agg.busy_cycles.load() == 1200);
        REQUIRE(agg.idle_cycles.load() == 800);
    }

    SECTION("Resource type to_string") {
        REQUIRE(std::string(to_string(ResourceType::SYSTOLIC_ARRAY)) == "systolic_array");
        REQUIRE(std::string(to_string(ResourceType::VECTOR_ENGINE)) == "vector_engine");
        REQUIRE(std::string(to_string(ResourceType::MEMORY_CONTROLLER)) == "memory_controller");
        REQUIRE(std::string(to_string(ResourceType::DMA_ENGINE)) == "dma_engine");
    }

    SECTION("Resource ID to_string") {
        ResourceId id{ResourceType::SYSTOLIC_ARRAY, 0};
        REQUIRE(id.to_string() == "systolic_array[0]");

        ResourceId id2{ResourceType::NOC_LINK, 5};
        REQUIRE(id2.to_string() == "noc_link[5]");
    }
}

// ============================================================================
// Stats Collector Tests
// ============================================================================

TEST_CASE("Stats Collector (v0.3.4)", "[stats][collector][v034]") {

    SECTION("Default configuration") {
        StatsCollector collector;

        auto& config = collector.config();
        REQUIRE(config.clock_frequency_ghz == 1.0);
        REQUIRE(config.peak_gflops == 1000.0);
        REQUIRE(config.external_bw_gbs == 64.0);
    }

    SECTION("Custom configuration") {
        StatsConfig cfg;
        cfg.clock_frequency_ghz = 2.0;
        cfg.peak_gflops = 2000.0;
        cfg.external_bw_gbs = 128.0;

        StatsCollector collector(cfg);

        REQUIRE(collector.config().clock_frequency_ghz == 2.0);
        REQUIRE(collector.config().peak_gflops == 2000.0);
    }

    SECTION("Cycle recording") {
        StatsCollector collector;

        collector.record_compute_cycles(1000);
        collector.record_memory_stall_cycles(200);
        collector.record_idle_cycles(300);

        REQUIRE(collector.cycles().get(CycleCategory::COMPUTE) == 1000);
        REQUIRE(collector.cycles().get(CycleCategory::MEMORY_STALL) == 200);
        REQUIRE(collector.cycles().get(CycleCategory::IDLE) == 300);
    }

    SECTION("Memory recording") {
        StatsCollector collector;

        collector.record_external_read(4096);
        collector.record_external_write(2048);
        collector.record_l3_read(8192);

        REQUIRE(collector.memory().external_bytes() == 6144);
        REQUIRE(collector.memory().total_bytes(MemoryLevel::L3) == 8192);
    }

    SECTION("Utilization recording") {
        StatsCollector collector;

        collector.record_systolic_busy(900);
        collector.record_systolic_idle(100);
        collector.record_systolic_operation(8192);  // 8K FLOPs

        REQUIRE(collector.total_flops() == 8192);
    }

    SECTION("Summary generation") {
        StatsCollector collector;

        collector.record_compute_cycles(800);
        collector.record_memory_stall_cycles(100);
        collector.record_idle_cycles(100);

        collector.record_external_read(4096);
        collector.record_systolic_operation(2048);

        auto summary = collector.summarize();

        REQUIRE(summary.total_cycles == 1000);
        REQUIRE(summary.compute_cycles == 800);
        REQUIRE(summary.stall_cycles == 100);
        REQUIRE(summary.total_flops == 2048);
    }

    SECTION("Arithmetic intensity") {
        StatsCollector collector;

        collector.record_systolic_operation(1024 * 1024);  // 1M FLOPs
        collector.record_external_read(64 * 1024);         // 64 KB

        // AI = 1M / 64K = 16 FLOP/byte
        REQUIRE(collector.arithmetic_intensity() == Approx(16.0));
    }

    SECTION("Roofline analysis - memory bound") {
        StatsConfig cfg;
        cfg.peak_gflops = 1000.0;
        cfg.external_bw_gbs = 64.0;  // Ridge point = 1000/64 = 15.625

        StatsCollector collector(cfg);

        collector.record_systolic_operation(512 * 1024);   // 512K FLOPs
        collector.record_external_read(64 * 1024);         // 64 KB
        // AI = 8 < 15.625 -> memory bound

        REQUIRE(collector.is_memory_bound());
        REQUIRE(!collector.is_compute_bound());
    }

    SECTION("Roofline analysis - compute bound") {
        StatsConfig cfg;
        cfg.peak_gflops = 1000.0;
        cfg.external_bw_gbs = 64.0;  // Ridge point = 15.625

        StatsCollector collector(cfg);

        collector.record_systolic_operation(2048 * 1024);  // 2M FLOPs
        collector.record_external_read(64 * 1024);         // 64 KB
        // AI = 32 > 15.625 -> compute bound

        REQUIRE(!collector.is_memory_bound());
        REQUIRE(collector.is_compute_bound());
    }

    SECTION("Reset clears all statistics") {
        StatsCollector collector;

        collector.record_compute_cycles(1000);
        collector.record_external_read(4096);
        collector.record_systolic_operation(512);

        collector.reset();

        REQUIRE(collector.cycles().total_cycles() == 0);
        REQUIRE(collector.memory().total_traffic() == 0);
        REQUIRE(collector.total_flops() == 0);
    }

    SECTION("JSON output") {
        StatsCollector collector;

        collector.record_compute_cycles(1000);
        collector.record_systolic_operation(512);

        std::string json = collector.to_json();
        REQUIRE(json.find("\"version\": \"0.3.4\"") != std::string::npos);
        REQUIRE(json.find("\"summary\"") != std::string::npos);
        REQUIRE(json.find("\"cycles\"") != std::string::npos);
        REQUIRE(json.find("\"memory\"") != std::string::npos);
        REQUIRE(json.find("\"utilization\"") != std::string::npos);
    }

    SECTION("Summary output") {
        StatsCollector collector;

        collector.record_compute_cycles(1000);
        collector.record_external_read(4096);

        std::string summary = collector.summary();
        REQUIRE(summary.find("KPU Simulation Statistics") != std::string::npos);
        REQUIRE(summary.find("v0.3.4") != std::string::npos);
    }

    SECTION("CSV output") {
        StatsCollector collector;

        collector.record_compute_cycles(1000);

        std::string header = StatsCollector::csv_header();
        std::string row = collector.to_csv_row();

        REQUIRE(header.find("total_cycles") != std::string::npos);
        REQUIRE(row.find("1000") != std::string::npos);
    }
}

// ============================================================================
// Global Stats Tests
// ============================================================================

TEST_CASE("Global Stats Singleton (v0.3.4)", "[stats][global][v034]") {

    // Reset before test
    GlobalStats::reset();

    SECTION("Singleton access") {
        auto& stats1 = GlobalStats::instance();
        auto& stats2 = GlobalStats::instance();

        REQUIRE(&stats1 == &stats2);
    }

    SECTION("Configure global stats") {
        StatsConfig cfg;
        cfg.peak_gflops = 2000.0;

        GlobalStats::configure(cfg);

        REQUIRE(GlobalStats::instance().config().peak_gflops == 2000.0);

        // Restore default
        GlobalStats::configure(StatsConfig{});
    }

    SECTION("Reset global stats") {
        GlobalStats::instance().record_compute_cycles(1000);

        GlobalStats::reset();

        REQUIRE(GlobalStats::instance().cycles().total_cycles() == 0);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_CASE("Statistics Integration (v0.3.4)", "[stats][integration][v034]") {

    SECTION("Simulated matmul statistics") {
        StatsConfig cfg;
        cfg.clock_frequency_ghz = 1.0;
        cfg.peak_gflops = 1024.0;
        cfg.external_bw_gbs = 64.0;

        StatsCollector collector(cfg);

        // Simulate 1024x1024x1024 matmul
        uint64_t M = 1024, N = 1024, K = 1024;
        uint64_t flops = 2ULL * M * N * K;

        // Memory: load A (M*K), load B (K*N), store C (M*N)
        uint64_t a_bytes = M * K * 4;  // float
        uint64_t b_bytes = K * N * 4;
        uint64_t c_bytes = M * N * 4;

        // Simulate execution
        collector.record_external_read(a_bytes);
        collector.record_external_read(b_bytes);
        collector.record_systolic_operation(flops);
        collector.record_external_write(c_bytes);

        // Cycle breakdown (example)
        collector.record_compute_cycles(2048);
        collector.record_memory_access_cycles(500);
        collector.record_memory_stall_cycles(200);
        collector.record_idle_cycles(100);

        collector.record_systolic_busy(2048);
        collector.record_systolic_idle(100);

        auto summary = collector.summarize();

        // Validate statistics
        REQUIRE(summary.total_flops == flops);
        REQUIRE(summary.external_bytes == a_bytes + b_bytes + c_bytes);

        // AI = 2*M*N*K / 3*M*N*4 = 2K/12 = 170.67
        double expected_ai = static_cast<double>(flops) / static_cast<double>(a_bytes + b_bytes + c_bytes);
        REQUIRE(summary.arithmetic_intensity == Approx(expected_ai).epsilon(0.01));
    }

    SECTION("Multi-layer DNN statistics") {
        StatsCollector collector;

        // Simulate 3-layer network
        for (int layer = 0; layer < 3; ++layer) {
            // Each layer: 256x256 matmul
            uint64_t flops = 2ULL * 256 * 256 * 256;
            uint64_t bytes = 3 * 256 * 256 * 4;

            collector.record_external_read(bytes * 2 / 3);  // A, B
            collector.record_external_write(bytes / 3);     // C
            collector.record_systolic_operation(flops);

            collector.record_compute_cycles(512);
            collector.record_memory_stall_cycles(100);
        }

        auto summary = collector.summarize();

        REQUIRE(summary.total_flops == 3 * 2ULL * 256 * 256 * 256);
        REQUIRE(summary.total_cycles == 3 * 612);
    }

    SECTION("Memory hierarchy traffic tracking") {
        StatsCollector collector;

        // Data flows: DRAM -> L3 -> L2 -> L1
        uint64_t tile_size = 4096;

        // External read
        collector.record_external_read(tile_size);

        // L3 traffic (2x for read/write within L3)
        collector.record_l3_read(tile_size);
        collector.record_l3_write(tile_size);

        // L2 traffic (3x for tiling)
        collector.record_l2_read(tile_size * 3);

        // L1 traffic (feeds to compute)
        collector.record_l1_read(tile_size * 3);

        // Amplification = internal / external = (2+3+3) * 4096 / 4096 = 8
        double amp = collector.memory().traffic_amplification();
        REQUIRE(amp == Approx(8.0));
    }
}
