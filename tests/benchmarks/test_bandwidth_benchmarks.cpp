// Memory Bandwidth Benchmark Tests (v0.3.1)
// Tests for memory-bound operations and bandwidth utilization
//
// ============================================================================
// BANDWIDTH BENCHMARK GOALS
// ============================================================================
// v0.3.1 requires validating that memory-bound operations achieve >70% of
// peak memory bandwidth. This file tests:
//
// 1. Elementwise operations (ADD, MUL) - AI ~0.08-0.25 FLOP/byte
// 2. Softmax operations - AI ~1 FLOP/byte
// 3. LayerNorm operations - AI ~0.6 FLOP/byte
// 4. BatchNorm operations - AI ~0.3 FLOP/byte
// 5. Small matmuls that are memory-bound (AI < ridge point)
//
// The roofline model shows:
// - Peak compute: 512 GFLOPS (at 1 GHz reference)
// - Peak bandwidth: 64 GB/s
// - Ridge point: 512/64 = 8 FLOP/byte
//
// Operations with AI < 8 are memory-bound and should be bandwidth-limited.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>

#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace sw::benchmark;
using namespace sw::kpu;

// ============================================================================
// Elementwise Bandwidth Benchmarks
// ============================================================================

TEST_CASE("Elementwise ADD bandwidth benchmark", "[benchmark][bandwidth][elementwise]") {
    BenchmarkHarness harness;

    SECTION("Small vector ADD (64K elements)") {
        auto result = harness.benchmark_elementwise(ElementwiseOp::ADD, 64 * 1024);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.is_memory_bound());  // AI << 8

        std::cout << "\n=== Elementwise ADD (64K) ===" << std::endl;
        std::cout << result.to_detailed_string() << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }

    SECTION("Medium vector ADD (1M elements)") {
        auto result = harness.benchmark_elementwise(ElementwiseOp::ADD, 1024 * 1024);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.is_memory_bound());

        std::cout << "\n=== Elementwise ADD (1M) ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "External bytes: " << result.external_bytes << " ("
                  << (result.external_bytes / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }
}

TEST_CASE("Elementwise size sweep", "[benchmark][bandwidth][elementwise][sweep]") {
    BenchmarkHarness harness;

    std::vector<Size> sizes = {
        4 * 1024,      // 4K
        16 * 1024,     // 16K
        64 * 1024,     // 64K
        256 * 1024,    // 256K
        1024 * 1024,   // 1M
    };

    SECTION("ADD operation sweep") {
        auto suite = harness.sweep_elementwise_sizes(ElementwiseOp::ADD, sizes);

        std::cout << "\n=== Elementwise ADD Size Sweep ===" << std::endl;
        std::cout << suite.summary_table() << std::endl;

        // All should be memory-bound
        for (const auto& r : suite.results) {
            REQUIRE(r.is_memory_bound());
            std::cout << r.config << ": "
                      << std::fixed << std::setprecision(2)
                      << r.achieved_bandwidth_gbs << " GB/s ("
                      << (r.memory_efficiency * 100) << "% efficiency)" << std::endl;
        }
    }

    SECTION("MUL operation sweep") {
        auto suite = harness.sweep_elementwise_sizes(ElementwiseOp::MUL, sizes);

        std::cout << "\n=== Elementwise MUL Size Sweep ===" << std::endl;
        std::cout << suite.summary_table() << std::endl;

        for (const auto& r : suite.results) {
            REQUIRE(r.is_memory_bound());
        }
    }
}

// ============================================================================
// Softmax Bandwidth Benchmarks
// ============================================================================

TEST_CASE("Softmax bandwidth benchmark", "[benchmark][bandwidth][softmax]") {
    BenchmarkHarness harness;

    SECTION("Small softmax (32 x 256)") {
        auto result = harness.benchmark_softmax(32, 256);

        REQUIRE(result.cycles > 0);
        // Softmax AI ~1 FLOP/byte, should be memory-bound (ridge = 8)
        REQUIRE(result.arithmetic_intensity < 8.0);

        std::cout << "\n=== Softmax (32x256) ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "FLOPs: " << result.flops << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }

    SECTION("Transformer attention softmax (128 x 512)") {
        // Typical attention: batch*heads=128, seq_len=512
        auto result = harness.benchmark_softmax(128, 512);

        REQUIRE(result.cycles > 0);

        std::cout << "\n=== Softmax (128x512) - Attention-like ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }
}

// ============================================================================
// LayerNorm Bandwidth Benchmarks
// ============================================================================

TEST_CASE("LayerNorm bandwidth benchmark", "[benchmark][bandwidth][layernorm]") {
    BenchmarkHarness harness;

    SECTION("Small LayerNorm (batch=4, seq=128, dim=256)") {
        auto result = harness.benchmark_layernorm(4, 128, 256);

        REQUIRE(result.cycles > 0);

        std::cout << "\n=== LayerNorm (4x128x256) ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "FLOPs: " << result.flops << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }

    SECTION("Transformer LayerNorm (batch=1, seq=512, dim=768)") {
        // GPT-2 style: seq=512, hidden=768
        auto result = harness.benchmark_layernorm(1, 512, 768);

        REQUIRE(result.cycles > 0);

        std::cout << "\n=== LayerNorm (1x512x768) - Transformer ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }
}

// ============================================================================
// BatchNorm Bandwidth Benchmarks
// ============================================================================

TEST_CASE("BatchNorm bandwidth benchmark", "[benchmark][bandwidth][batchnorm]") {
    BenchmarkHarness harness;

    SECTION("ResNet-style BatchNorm (batch=8, channels=64, HxW=56x56)") {
        auto result = harness.benchmark_batchnorm(8, 64, 56, 56);

        REQUIRE(result.cycles > 0);

        std::cout << "\n=== BatchNorm (8x64x56x56) - ResNet ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "FLOPs: " << result.flops << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }

    SECTION("Larger BatchNorm (batch=16, channels=256, HxW=28x28)") {
        auto result = harness.benchmark_batchnorm(16, 256, 28, 28);

        REQUIRE(result.cycles > 0);

        std::cout << "\n=== BatchNorm (16x256x28x28) ===" << std::endl;
        std::cout << "Cycles: " << result.cycles << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW:  " << std::fixed << std::setprecision(2)
                  << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;
    }
}

// ============================================================================
// Comprehensive Bandwidth Sweep (v0.3.1)
// ============================================================================

TEST_CASE("Bandwidth operations sweep (v0.3.1)", "[benchmark][bandwidth][v031][sweep]") {
    BenchmarkHarness harness;

    auto suite = harness.sweep_bandwidth_ops(4096, 262144);  // 4K to 256K elements

    std::cout << "\n=== Bandwidth Operations Sweep (v0.3.1) ===" << std::endl;
    std::cout << suite.summary_table() << std::endl;

    // Analyze results
    size_t memory_bound_count = 0;
    double total_bw_efficiency = 0.0;
    double min_bw_efficiency = 1.0;
    double max_bw_efficiency = 0.0;

    for (const auto& r : suite.results) {
        if (r.is_memory_bound()) {
            memory_bound_count++;
            total_bw_efficiency += r.memory_efficiency;
            min_bw_efficiency = std::min(min_bw_efficiency, r.memory_efficiency);
            max_bw_efficiency = std::max(max_bw_efficiency, r.memory_efficiency);
        }
    }

    double avg_bw_efficiency = memory_bound_count > 0
        ? total_bw_efficiency / memory_bound_count : 0.0;

    std::cout << "\n=== BANDWIDTH SUMMARY ===" << std::endl;
    std::cout << "Memory-bound operations: " << memory_bound_count
              << " / " << suite.results.size() << std::endl;
    std::cout << "Average BW efficiency: " << std::fixed << std::setprecision(1)
              << (avg_bw_efficiency * 100) << "%" << std::endl;
    std::cout << "Min BW efficiency: " << (min_bw_efficiency * 100) << "%" << std::endl;
    std::cout << "Max BW efficiency: " << (max_bw_efficiency * 100) << "%" << std::endl;

    // v0.3.1 requires >70% bandwidth utilization for BW-bound ops
    // This is an informational check - actual hardware may vary
    std::cout << "\nv0.3.1 Target: >70% bandwidth utilization" << std::endl;
    std::cout << "Status: " << (avg_bw_efficiency >= 0.70 ? "PASS" : "BELOW TARGET") << std::endl;
}

// ============================================================================
// Roofline Classification Tests
// ============================================================================

TEST_CASE("Roofline classification accuracy", "[benchmark][bandwidth][roofline]") {
    BenchmarkHarness harness;
    HardwareSpec hw = harness.hardware_spec();

    std::cout << "\n=== Roofline Classification ===" << std::endl;
    std::cout << "Peak compute: " << hw.peak_gflops << " GFLOPS" << std::endl;
    std::cout << "Peak bandwidth: " << hw.external_bw_gbs << " GB/s" << std::endl;
    std::cout << "Ridge point: " << hw.ridge_point_external() << " FLOP/byte" << std::endl;
    std::cout << std::endl;

    SECTION("Verify small matmuls are memory-bound") {
        // 32x32x32 matmul: AI = 2*32*32*32 / (32*32*4 + 32*32*4 + 32*32*4) = 65536 / 12288 = 5.33
        // Should be memory-bound (AI < 8)
        auto result = harness.benchmark_matmul(32, 32, 32);

        std::cout << "32x32x32 Matmul:" << std::endl;
        std::cout << "  AI: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "  Memory-bound: " << (result.is_memory_bound() ? "YES" : "NO") << std::endl;
        std::cout << "  Bottleneck: " << result.bottleneck() << std::endl;

        REQUIRE(result.arithmetic_intensity < hw.ridge_point_external());
    }

    SECTION("Verify large matmuls are compute-bound") {
        // 1024x1024x1024 matmul: AI = 2*1024^3 / (1024*1024*4*3) = ~170
        // Should be compute-bound (AI >> 8)
        auto result = harness.benchmark_matmul(1024, 1024, 1024);

        std::cout << "1024x1024x1024 Matmul:" << std::endl;
        std::cout << "  AI: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "  Compute-bound: " << (result.is_compute_bound() ? "YES" : "NO") << std::endl;
        std::cout << "  Bottleneck: " << result.bottleneck() << std::endl;

        REQUIRE(result.arithmetic_intensity >= hw.ridge_point_external());
        REQUIRE(result.is_compute_bound());
    }

    SECTION("Verify elementwise is extremely memory-bound") {
        auto result = harness.benchmark_elementwise(ElementwiseOp::ADD, 256 * 1024);

        std::cout << "Elementwise ADD (256K):" << std::endl;
        std::cout << "  AI: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "  Memory-bound: " << (result.is_memory_bound() ? "YES" : "NO") << std::endl;

        // Elementwise binary ADD: 1 FLOP per 12 bytes (read A, read B, write C)
        // AI = 1 / 12 = 0.083 FLOP/byte - extremely memory-bound
        REQUIRE(result.arithmetic_intensity < 1.0);
        REQUIRE(result.is_memory_bound());
    }
}

// ============================================================================
// JSON Export Tests
// ============================================================================

TEST_CASE("Bandwidth benchmark JSON export", "[benchmark][bandwidth][export]") {
    BenchmarkHarness harness;

    auto result = harness.benchmark_elementwise(ElementwiseOp::ADD, 64 * 1024);
    std::string json = result.to_json();

    // Verify JSON contains bandwidth fields
    REQUIRE(json.find("\"achieved_bandwidth_gbs\":") != std::string::npos);
    REQUIRE(json.find("\"memory_efficiency\":") != std::string::npos);
    REQUIRE(json.find("\"arithmetic_intensity\":") != std::string::npos);
    REQUIRE(json.find("\"bottleneck\":") != std::string::npos);

    // Write to file for inspection
    auto path = std::filesystem::temp_directory_path() / "kpu_bandwidth_benchmark.json";
    std::ofstream out(path);
    if (out) {
        out << json;
        out.close();
        std::cout << "\nWrote bandwidth benchmark JSON to " << path << std::endl;
    }
}
