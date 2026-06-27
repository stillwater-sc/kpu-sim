// Analytical Validation Tests (v0.3.5)
// Tests with known analytical solutions for compute and memory-bound operators
//
// ============================================================================
// PURPOSE
// ============================================================================
// These tests validate that the KPU simulator produces results that match
// theoretical predictions based on known analytical solutions.
//
// For compute-bound operations:
// - FLOP count must exactly match 2×M×N×K for matmul
// - Minimum cycles = FLOPs / MACs_per_cycle (512 for 16x16 systolic)
// - Arithmetic intensity = FLOPs / external_bytes
//
// For memory-bound operations:
// - Byte count must match read + write sizes
// - Minimum cycles = bytes / bandwidth
// - Achieved bandwidth should be > 70% of peak for BW-bound ops
//
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <sw/compiler/kernel_compiler.hpp>

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace sw::benchmark;
using namespace sw::kpu;
using namespace sw::kpu::isa;
using namespace sw::kpu::compiler;

// ============================================================================
// Analytical Constants
// ============================================================================

namespace analytical {

// Hardware configuration (must match ResourceConfig defaults)
constexpr Size SYSTOLIC_SIZE = 16;                    // 16x16 PE array
constexpr Size MACS_PER_CYCLE = SYSTOLIC_SIZE * SYSTOLIC_SIZE;  // 256 MACs
constexpr Size FLOPS_PER_CYCLE = MACS_PER_CYCLE * 2;  // 512 FLOPs (FMA = 2 ops)

// Memory bandwidth (GB/s at 1 GHz = bytes/cycle)
constexpr double EXTERNAL_BW_GBS = 64.0;
constexpr double L3_BW_GBS = 128.0;
constexpr double L2_BW_GBS = 256.0;

// Peak performance
constexpr double PEAK_GFLOPS = 512.0;  // At 1 GHz reference

// Ridge points (FLOP/byte)
constexpr double RIDGE_EXTERNAL = PEAK_GFLOPS / EXTERNAL_BW_GBS;  // 8.0
[[maybe_unused]] constexpr double RIDGE_L3 = PEAK_GFLOPS / L3_BW_GBS;  // 4.0
[[maybe_unused]] constexpr double RIDGE_L2 = PEAK_GFLOPS / L2_BW_GBS;  // 2.0

// Element sizes
constexpr Size FLOAT32_BYTES = 4;

/**
 * @brief Calculate analytical FLOP count for matmul C = A @ B
 */
constexpr uint64_t matmul_flops(Size M, Size N, Size K) {
    return 2ULL * M * N * K;
}

/**
 * @brief Calculate analytical external memory bytes for matmul
 *
 * Minimum bytes = A + B + C (no reuse from external memory)
 */
constexpr Size matmul_external_bytes(Size M, Size N, Size K) {
    return (M * K + K * N + M * N) * FLOAT32_BYTES;
}

/**
 * @brief Calculate analytical arithmetic intensity for matmul
 */
inline double matmul_arithmetic_intensity(Size M, Size N, Size K) {
    return static_cast<double>(matmul_flops(M, N, K)) /
           static_cast<double>(matmul_external_bytes(M, N, K));
}

/**
 * @brief Calculate minimum compute cycles for matmul (100% utilization)
 */
constexpr Cycle matmul_min_compute_cycles(Size M, Size N, Size K) {
    return (matmul_flops(M, N, K) + FLOPS_PER_CYCLE - 1) / FLOPS_PER_CYCLE;
}

/**
 * @brief Calculate minimum memory cycles (if memory-bound)
 */
inline Cycle matmul_min_memory_cycles(Size M, Size N, Size K) {
    return static_cast<Cycle>(static_cast<double>(matmul_external_bytes(M, N, K)) / EXTERNAL_BW_GBS);
}

/**
 * @brief Calculate roofline-predicted GFLOPS
 */
inline double roofline_gflops(double ai) {
    return std::min(ai * EXTERNAL_BW_GBS, PEAK_GFLOPS);
}

/**
 * @brief Check if operation is compute-bound based on AI
 */
inline bool is_compute_bound(double ai) {
    return ai >= RIDGE_EXTERNAL;
}

/**
 * @brief Calculate analytical bytes for elementwise binary op (A op B -> C)
 */
constexpr Size elementwise_bytes(Size num_elements) {
    return num_elements * FLOAT32_BYTES * 3;  // Read A, Read B, Write C
}

/**
 * @brief Calculate analytical FLOPs for elementwise binary op
 */
constexpr uint64_t elementwise_flops(Size num_elements) {
    return num_elements;  // 1 FLOP per element
}

/**
 * @brief Calculate arithmetic intensity for elementwise binary op
 */
inline double elementwise_arithmetic_intensity(Size num_elements) {
    return static_cast<double>(elementwise_flops(num_elements)) /
           static_cast<double>(elementwise_bytes(num_elements));
}

}  // namespace analytical

// ============================================================================
// Matmul Analytical Validation Tests
// ============================================================================

TEST_CASE("Matmul FLOP count validation", "[analytical][matmul][flops]") {
    // Test that we calculate the correct FLOP count
    SECTION("16x16x16 single tile") {
        Size M = 16, N = 16, K = 16;
        uint64_t expected_flops = analytical::matmul_flops(M, N, K);
        REQUIRE(expected_flops == 2ULL * 16 * 16 * 16);
        REQUIRE(expected_flops == 8192);
    }

    SECTION("64x64x64") {
        Size M = 64, N = 64, K = 64;
        uint64_t expected_flops = analytical::matmul_flops(M, N, K);
        REQUIRE(expected_flops == 2ULL * 64 * 64 * 64);
        REQUIRE(expected_flops == 524288);
    }

    SECTION("1024x1024x1024") {
        Size M = 1024, N = 1024, K = 1024;
        uint64_t expected_flops = analytical::matmul_flops(M, N, K);
        REQUIRE(expected_flops == 2ULL * 1024 * 1024 * 1024);
        REQUIRE(expected_flops == 2147483648ULL);
    }
}

TEST_CASE("Matmul arithmetic intensity validation", "[analytical][matmul][ai]") {
    SECTION("Square matmul AI formula") {
        // For square matmul NxNxN:
        // FLOPs = 2N³
        // Bytes = 3N² × 4 = 12N²
        // AI = 2N³ / 12N² = N/6
        Size N = 1024;
        double expected_ai = static_cast<double>(N) / 6.0;  // 170.67
        double actual_ai = analytical::matmul_arithmetic_intensity(N, N, N);

        REQUIRE_THAT(actual_ai, Catch::Matchers::WithinRel(expected_ai, 0.01));

        std::cout << "1024x1024x1024 matmul AI: " << actual_ai
                  << " (expected N/6 = " << expected_ai << ")" << std::endl;
    }

    SECTION("AI increases with problem size") {
        double ai_64 = analytical::matmul_arithmetic_intensity(64, 64, 64);
        double ai_128 = analytical::matmul_arithmetic_intensity(128, 128, 128);
        double ai_256 = analytical::matmul_arithmetic_intensity(256, 256, 256);

        REQUIRE(ai_128 > ai_64);
        REQUIRE(ai_256 > ai_128);

        std::cout << "AI scaling: 64³=" << std::fixed << std::setprecision(2) << ai_64
                  << ", 128³=" << ai_128 << ", 256³=" << ai_256 << std::endl;
    }

    SECTION("Ridge point classification") {
        // Operations below ridge point (8.0) are memory-bound
        // AI = N/6, so N < 48 is memory-bound, N >= 48 is compute-bound
        double ai_32 = analytical::matmul_arithmetic_intensity(32, 32, 32);
        double ai_64 = analytical::matmul_arithmetic_intensity(64, 64, 64);
        double ai_256 = analytical::matmul_arithmetic_intensity(256, 256, 256);

        std::cout << "Classification test:" << std::endl;
        std::cout << "  32³: AI=" << ai_32 << " -> "
                  << (analytical::is_compute_bound(ai_32) ? "compute" : "memory") << std::endl;
        std::cout << "  64³: AI=" << ai_64 << " -> "
                  << (analytical::is_compute_bound(ai_64) ? "compute" : "memory") << std::endl;
        std::cout << "  256³: AI=" << ai_256 << " -> "
                  << (analytical::is_compute_bound(ai_256) ? "compute" : "memory") << std::endl;

        REQUIRE(ai_32 < analytical::RIDGE_EXTERNAL);  // Memory-bound
        REQUIRE(ai_64 > analytical::RIDGE_EXTERNAL);  // Compute-bound
        REQUIRE(ai_256 > analytical::RIDGE_EXTERNAL); // Compute-bound
    }
}

TEST_CASE("Matmul minimum cycle validation", "[analytical][matmul][cycles]") {
    SECTION("16x16x16 single tile minimum cycles") {
        Size M = 16, N = 16, K = 16;
        Cycle min_cycles = analytical::matmul_min_compute_cycles(M, N, K);

        // 8192 FLOPs / 512 FLOPs per cycle = 16 cycles minimum
        REQUIRE(min_cycles == 16);
    }

    SECTION("64x64x64 minimum cycles") {
        Size M = 64, N = 64, K = 64;
        Cycle min_cycles = analytical::matmul_min_compute_cycles(M, N, K);

        // 524288 FLOPs / 512 FLOPs per cycle = 1024 cycles minimum
        REQUIRE(min_cycles == 1024);
    }

    SECTION("1024x1024x1024 minimum cycles") {
        Size M = 1024, N = 1024, K = 1024;
        Cycle min_cycles = analytical::matmul_min_compute_cycles(M, N, K);

        // 2^31 FLOPs / 512 FLOPs per cycle = 2^22 = 4194304 cycles minimum
        REQUIRE(min_cycles == 4194304);
    }
}

TEST_CASE("Matmul roofline prediction", "[analytical][matmul][roofline]") {
    SECTION("Memory-bound prediction (32x32x32)") {
        Size M = 32, N = 32, K = 32;
        double ai = analytical::matmul_arithmetic_intensity(M, N, K);
        double predicted = analytical::roofline_gflops(ai);

        // Memory-bound: predicted = AI × bandwidth
        double expected = ai * analytical::EXTERNAL_BW_GBS;

        std::cout << "32x32x32: AI=" << ai << ", predicted=" << predicted
                  << " GFLOPS" << std::endl;

        REQUIRE_THAT(predicted, Catch::Matchers::WithinRel(expected, 0.01));
        REQUIRE(predicted < analytical::PEAK_GFLOPS);
    }

    SECTION("Compute-bound prediction (1024x1024x1024)") {
        Size M = 1024, N = 1024, K = 1024;
        double ai = analytical::matmul_arithmetic_intensity(M, N, K);
        double predicted = analytical::roofline_gflops(ai);

        // Compute-bound: predicted = peak
        std::cout << "1024x1024x1024: AI=" << ai << ", predicted=" << predicted
                  << " GFLOPS (peak=" << analytical::PEAK_GFLOPS << ")" << std::endl;

        REQUIRE_THAT(predicted, Catch::Matchers::WithinAbs(analytical::PEAK_GFLOPS, 0.1));
    }
}

// ============================================================================
// Simulator vs Analytical Validation
// ============================================================================

TEST_CASE("Simulator FLOP count matches analytical", "[analytical][simulator][flops]") {
    KernelCompiler compiler;

    SECTION("64x64x64 kernel FLOPs") {
        Size M = 64, N = 64, K = 64;
        Kernel kernel = compiler.compile_matmul(M, N, K);

        uint64_t kernel_flops = kernel.total_flops();
        uint64_t expected_flops = analytical::matmul_flops(M, N, K);

        std::cout << "64x64x64 kernel FLOPs: " << kernel_flops
                  << " (expected: " << expected_flops << ")" << std::endl;

        REQUIRE(kernel_flops == expected_flops);
    }

    SECTION("256x256x256 kernel FLOPs") {
        Size M = 256, N = 256, K = 256;
        Kernel kernel = compiler.compile_matmul(M, N, K);

        REQUIRE(kernel.total_flops() == analytical::matmul_flops(M, N, K));
    }
}

TEST_CASE("Simulator AI matches analytical", "[analytical][simulator][ai]") {
    KernelCompiler compiler;

    std::vector<Size> sizes = {64, 128, 256, 512};

    for (Size N : sizes) {
        DYNAMIC_SECTION("Size " << N << "x" << N << "x" << N) {
            Kernel kernel = compiler.compile_matmul(N, N, N);

            double kernel_ai = kernel.arithmetic_intensity();
            double expected_ai = analytical::matmul_arithmetic_intensity(N, N, N);

            // Allow 20% tolerance for tiling effects
            double tolerance = expected_ai * 0.20;

            std::cout << N << "x" << N << "x" << N << " AI: kernel=" << kernel_ai
                      << ", analytical=" << expected_ai << std::endl;

            REQUIRE_THAT(kernel_ai, Catch::Matchers::WithinAbs(expected_ai, tolerance));
        }
    }
}

TEST_CASE("Simulator efficiency vs analytical prediction", "[analytical][simulator][efficiency]") {
    BenchmarkHarness harness;

    SECTION("Large compute-bound matmul achieves >80% of peak") {
        // 1024x1024x1024 should be highly compute-bound
        auto result = harness.benchmark_matmul(1024, 1024, 1024);

        double analytical_ai = analytical::matmul_arithmetic_intensity(1024, 1024, 1024);
        double predicted_gflops = analytical::roofline_gflops(analytical_ai);

        std::cout << "\n=== 1024x1024x1024 Analytical Validation ===" << std::endl;
        std::cout << "Analytical AI: " << analytical_ai << " FLOP/byte" << std::endl;
        std::cout << "Ridge point: " << analytical::RIDGE_EXTERNAL << " FLOP/byte" << std::endl;
        std::cout << "Predicted GFLOPS: " << predicted_gflops << std::endl;
        std::cout << "Achieved GFLOPS: " << result.gflops << std::endl;
        std::cout << "Efficiency: " << (result.compute_efficiency * 100) << "%" << std::endl;

        // v0.3 target: >80% efficiency for large compute-bound ops
        REQUIRE(result.is_compute_bound());
        REQUIRE(result.compute_efficiency >= 0.80);
    }

    SECTION("Medium compute-bound matmul achieves >50% of peak") {
        // 256x256x256
        auto result = harness.benchmark_matmul(256, 256, 256);

        std::cout << "\n=== 256x256x256 Analytical Validation ===" << std::endl;
        std::cout << "Arithmetic Intensity: " << result.arithmetic_intensity << std::endl;
        std::cout << "Efficiency: " << (result.compute_efficiency * 100) << "%" << std::endl;

        REQUIRE(result.is_compute_bound());
        REQUIRE(result.compute_efficiency >= 0.50);
    }
}

// ============================================================================
// Memory-Bound Analytical Validation
// ============================================================================

TEST_CASE("Elementwise analytical values", "[analytical][elementwise]") {
    SECTION("Arithmetic intensity is constant") {
        // For elementwise binary: AI = 1 FLOP / 12 bytes = 0.0833
        Size n1 = 1024;
        Size n2 = 1024 * 1024;

        double ai1 = analytical::elementwise_arithmetic_intensity(n1);
        double ai2 = analytical::elementwise_arithmetic_intensity(n2);

        double expected_ai = 1.0 / 12.0;  // ~0.0833

        REQUIRE_THAT(ai1, Catch::Matchers::WithinRel(expected_ai, 0.01));
        REQUIRE_THAT(ai2, Catch::Matchers::WithinRel(expected_ai, 0.01));
        REQUIRE_THAT(ai1, Catch::Matchers::WithinAbs(ai2, 0.001));

        std::cout << "Elementwise AI: " << ai1 << " (expected " << expected_ai << ")" << std::endl;
    }

    SECTION("Elementwise is always memory-bound") {
        double ai = analytical::elementwise_arithmetic_intensity(1000000);
        REQUIRE(ai < analytical::RIDGE_EXTERNAL);
        REQUIRE(!analytical::is_compute_bound(ai));
    }
}

TEST_CASE("Memory-bound bandwidth validation", "[analytical][bandwidth]") {
    BenchmarkHarness harness;

    SECTION("Elementwise ADD achieves >70% bandwidth") {
        auto result = harness.benchmark_elementwise(ElementwiseOp::ADD, 256 * 1024);

        double expected_ai = analytical::elementwise_arithmetic_intensity(256 * 1024);

        std::cout << "\n=== Elementwise ADD (256K) Analytical Validation ===" << std::endl;
        std::cout << "Expected AI: " << expected_ai << " FLOP/byte" << std::endl;
        std::cout << "Actual AI: " << result.arithmetic_intensity << " FLOP/byte" << std::endl;
        std::cout << "Achieved BW: " << result.achieved_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "BW Efficiency: " << (result.memory_efficiency * 100) << "%" << std::endl;

        REQUIRE(result.is_memory_bound());
        // v0.3.1 target: >70% bandwidth utilization
        REQUIRE(result.memory_efficiency >= 0.70);
    }
}

// ============================================================================
// Roofline Model Validation
// ============================================================================

TEST_CASE("Roofline model correctness", "[analytical][roofline]") {
    HardwareSpec hw;  // Uses defaults matching our analytical constants

    SECTION("Hardware spec matches analytical constants") {
        REQUIRE_THAT(hw.peak_gflops, Catch::Matchers::WithinAbs(analytical::PEAK_GFLOPS, 0.1));
        REQUIRE_THAT(hw.external_bw_gbs, Catch::Matchers::WithinAbs(analytical::EXTERNAL_BW_GBS, 0.1));
        REQUIRE_THAT(hw.ridge_point_external(), Catch::Matchers::WithinAbs(analytical::RIDGE_EXTERNAL, 0.1));
    }

    SECTION("Roofline prediction matches analytical") {
        std::vector<double> test_ais = {0.1, 1.0, 4.0, 8.0, 16.0, 100.0};

        for (double ai : test_ais) {
            double hw_predicted = hw.roofline_gflops(ai);
            double analytical_predicted = analytical::roofline_gflops(ai);

            REQUIRE_THAT(hw_predicted, Catch::Matchers::WithinAbs(analytical_predicted, 0.1));
        }
    }

    SECTION("Ridge point correctly separates regions") {
        // Just below ridge point
        double ai_below = analytical::RIDGE_EXTERNAL - 0.1;
        double pred_below = analytical::roofline_gflops(ai_below);
        REQUIRE(pred_below < analytical::PEAK_GFLOPS);

        // At ridge point
        double ai_at = analytical::RIDGE_EXTERNAL;
        double pred_at = analytical::roofline_gflops(ai_at);
        REQUIRE_THAT(pred_at, Catch::Matchers::WithinAbs(analytical::PEAK_GFLOPS, 0.1));

        // Above ridge point
        double ai_above = analytical::RIDGE_EXTERNAL + 10.0;
        double pred_above = analytical::roofline_gflops(ai_above);
        REQUIRE_THAT(pred_above, Catch::Matchers::WithinAbs(analytical::PEAK_GFLOPS, 0.1));
    }
}

// ============================================================================
// Cycle Count Validation
// ============================================================================

TEST_CASE("Simulator cycles vs analytical minimum", "[analytical][cycles]") {
    KernelCompiler compiler;
    ResourceConfig config;

    SECTION("Cycles never below analytical minimum") {
        std::vector<Size> sizes = {64, 128, 256};

        for (Size N : sizes) {
            Kernel kernel = compiler.compile_matmul(N, N, N);
            ConcurrentExecutor executor(config);
            Cycle actual_cycles = executor.execute(kernel.program());

            Cycle min_compute_cycles = analytical::matmul_min_compute_cycles(N, N, N);

            std::cout << N << "x" << N << "x" << N << ": actual=" << actual_cycles
                      << " cycles, min_compute=" << min_compute_cycles << std::endl;

            // Actual cycles should never be less than minimum compute cycles
            // (would violate physics)
            REQUIRE(actual_cycles >= min_compute_cycles);
        }
    }

    SECTION("Large matmul approaches minimum cycles") {
        Size N = 1024;
        Kernel kernel = compiler.compile_matmul(N, N, N);
        ConcurrentExecutor executor(config);
        Cycle actual_cycles = executor.execute(kernel.program());

        Cycle min_cycles = analytical::matmul_min_compute_cycles(N, N, N);

        // For large compute-bound ops, overhead should be small
        double overhead = static_cast<double>(actual_cycles - min_cycles) / static_cast<double>(min_cycles);

        std::cout << "1024x1024x1024: actual=" << actual_cycles
                  << ", min=" << min_cycles
                  << ", overhead=" << (overhead * 100) << "%" << std::endl;

        // Overhead should be < 5% for large problems
        REQUIRE(overhead < 0.05);
    }
}

// ============================================================================
// Summary Test
// ============================================================================

TEST_CASE("v0.3.5 Analytical Validation Summary", "[analytical][summary]") {
    BenchmarkHarness harness;

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "v0.3.5 ANALYTICAL VALIDATION SUMMARY\n";
    std::cout << "================================================================\n\n";

    std::cout << "Analytical Constants:\n";
    std::cout << "  Systolic array:      " << analytical::SYSTOLIC_SIZE << "x"
              << analytical::SYSTOLIC_SIZE << "\n";
    std::cout << "  FLOPs per cycle:     " << analytical::FLOPS_PER_CYCLE << "\n";
    std::cout << "  Peak GFLOPS:         " << analytical::PEAK_GFLOPS << "\n";
    std::cout << "  External BW:         " << analytical::EXTERNAL_BW_GBS << " GB/s\n";
    std::cout << "  Ridge point:         " << analytical::RIDGE_EXTERNAL << " FLOP/byte\n\n";

    // Test compute-bound
    auto compute_result = harness.benchmark_matmul(1024, 1024, 1024);
    double compute_analytical_ai = analytical::matmul_arithmetic_intensity(1024, 1024, 1024);

    std::cout << "Compute-bound validation (1024x1024x1024):\n";
    std::cout << "  Analytical AI:       " << compute_analytical_ai << " FLOP/byte\n";
    std::cout << "  Simulator AI:        " << compute_result.arithmetic_intensity << " FLOP/byte\n";
    std::cout << "  Efficiency:          " << (compute_result.compute_efficiency * 100) << "%\n";
    std::cout << "  Target (>80%):       "
              << (compute_result.compute_efficiency >= 0.80 ? "PASS" : "FAIL") << "\n\n";

    // Test memory-bound
    auto memory_result = harness.benchmark_elementwise(ElementwiseOp::ADD, 256 * 1024);
    double memory_analytical_ai = analytical::elementwise_arithmetic_intensity(256 * 1024);

    std::cout << "Memory-bound validation (elementwise 256K):\n";
    std::cout << "  Analytical AI:       " << memory_analytical_ai << " FLOP/byte\n";
    std::cout << "  Simulator AI:        " << memory_result.arithmetic_intensity << " FLOP/byte\n";
    std::cout << "  BW efficiency:       " << (memory_result.memory_efficiency * 100) << "%\n";
    std::cout << "  Target (>70%):       "
              << (memory_result.memory_efficiency >= 0.70 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "================================================================\n";

    // Ensure this test always passes (it's just a summary)
    REQUIRE(true);
}
