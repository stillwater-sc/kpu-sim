// MLP and Activation Benchmark Tests
// Tests for MLP kernels with various activations

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>

#include <iostream>

using namespace sw::benchmark;
using namespace sw::kpu;

TEST_CASE("MLP kernel benchmark", "[benchmark][mlp]") {
    BenchmarkHarness harness;

    SECTION("MLP with ReLU") {
        auto result = harness.benchmark_mlp(256, 512, 256, ActivationType::RELU, true);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.gflops > 0);

        std::cout << result.to_detailed_string() << std::endl;
    }

    SECTION("MLP with GELU") {
        auto result = harness.benchmark_mlp(256, 512, 256, ActivationType::GELU, true);

        REQUIRE(result.cycles > 0);
        std::cout << result.to_string() << std::endl;
    }

    SECTION("MLP without bias") {
        auto result = harness.benchmark_mlp(256, 512, 256, ActivationType::RELU, false);

        REQUIRE(result.cycles > 0);
        std::cout << result.to_string() << std::endl;
    }
}

TEST_CASE("Activation function comparison", "[benchmark][mlp][activation]") {
    BenchmarkHarness harness;

    Size M = 512, N = 1024, K = 512;
    auto suite = harness.sweep_activations(M, N, K);

    REQUIRE(suite.results.size() >= 7);  // baseline + 6 activations

    std::cout << "\n=== Activation Function Comparison ===" << std::endl;
    std::cout << "Problem size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << suite.summary_table() << std::endl;

    // Find baseline (matmul only)
    const BenchmarkResult* baseline = nullptr;
    for (const auto& r : suite.results) {
        if (r.name == "matmul_baseline") {
            baseline = &r;
            break;
        }
    }

    REQUIRE(baseline != nullptr);

    // Calculate overhead for each activation
    std::cout << "Activation Overhead (vs baseline):" << std::endl;
    for (const auto& r : suite.results) {
        if (&r != baseline) {
            double overhead = (static_cast<double>(r.cycles) / baseline->cycles - 1.0) * 100;
            std::cout << "  " << r.name << ": +" << std::fixed << std::setprecision(1)
                      << overhead << "% cycles" << std::endl;
        }
    }
}

TEST_CASE("Transformer FFN benchmark", "[benchmark][mlp][transformer]") {
    BenchmarkHarness harness;

    // GPT-2 small style: hidden=768, intermediate=3072
    Size batch_seq = 32 * 512;  // batch=32, seq=512
    Size hidden = 768;
    Size intermediate = 3072;

    SECTION("Up-projection with GELU") {
        auto result = harness.benchmark_mlp(batch_seq, intermediate, hidden,
                                             ActivationType::GELU, true);

        std::cout << "FFN up-projection (GELU):" << std::endl;
        std::cout << result.to_detailed_string() << std::endl;

        REQUIRE(result.cycles > 0);
    }

    SECTION("Down-projection (no activation)") {
        auto result = harness.benchmark_mlp(batch_seq, hidden, intermediate,
                                             ActivationType::NONE, true);

        std::cout << "FFN down-projection:" << std::endl;
        std::cout << result.to_detailed_string() << std::endl;

        REQUIRE(result.cycles > 0);
    }
}

TEST_CASE("MLP size sweep", "[benchmark][mlp][sweep]") {
    BenchmarkHarness harness;

    std::vector<std::tuple<Size, Size, Size>> sizes = {
        {64, 256, 64},
        {128, 512, 128},
        {256, 1024, 256},
        {512, 2048, 512},
        {1024, 4096, 1024},
    };

    std::cout << "\n=== MLP Size Sweep (with GELU) ===" << std::endl;

    BenchmarkSuite suite;
    suite.name = "mlp_size_sweep";

    for (const auto& [M, N, K] : sizes) {
        auto result = harness.benchmark_mlp(M, N, K, ActivationType::GELU, true);
        suite.add(result);
    }

    std::cout << suite.summary_table() << std::endl;

    // Verify scaling
    REQUIRE(suite.results.size() == sizes.size());
    for (size_t i = 1; i < suite.results.size(); i++) {
        REQUIRE(suite.results[i].flops > suite.results[i-1].flops);
    }
}

TEST_CASE("SiLU/Swish activation", "[benchmark][mlp][silu]") {
    BenchmarkHarness harness;

    // SiLU is used in many modern architectures (LLaMA, etc.)
    Size M = 512, N = 1024, K = 512;

    auto result = harness.benchmark_mlp(M, N, K, ActivationType::SILU, true);

    std::cout << "MLP with SiLU:" << std::endl;
    std::cout << result.to_detailed_string() << std::endl;

    REQUIRE(result.cycles > 0);
    REQUIRE(result.gflops > 0);
}
