// Matmul Benchmark Tests
// Tests for matrix multiplication performance across sizes and configurations

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>

#include <iostream>
#include <fstream>

using namespace sw::benchmark;
using namespace sw::kpu;

TEST_CASE("Single matmul benchmark", "[benchmark][matmul]") {
    BenchmarkHarness harness;

    SECTION("Small matmul 64x64x64") {
        auto result = harness.benchmark_matmul(64, 64, 64);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.flops == 2 * 64 * 64 * 64);
        REQUIRE(result.gflops > 0);
        REQUIRE(result.arithmetic_intensity > 0);

        std::cout << result.to_detailed_string() << std::endl;
    }

    SECTION("Medium matmul 256x256x256") {
        auto result = harness.benchmark_matmul(256, 256, 256);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.flops == 2 * 256 * 256 * 256);
        REQUIRE(result.gflops > 0);

        std::cout << result.to_detailed_string() << std::endl;
    }

    SECTION("Large matmul 1024x1024x1024") {
        auto result = harness.benchmark_matmul(1024, 1024, 1024);

        REQUIRE(result.cycles > 0);
        REQUIRE(result.flops == 2ULL * 1024 * 1024 * 1024);
        REQUIRE(result.gflops > 0);

        // Large problems should have higher arithmetic intensity
        REQUIRE(result.arithmetic_intensity > 10.0);

        std::cout << result.to_detailed_string() << std::endl;
    }
}

TEST_CASE("Matmul size sweep", "[benchmark][matmul][sweep]") {
    BenchmarkHarness harness;

    SECTION("Powers of 2 sweep") {
        auto suite = harness.sweep_matmul_square(64, 2048, 2);

        REQUIRE(suite.results.size() >= 6);  // 64, 128, 256, 512, 1024, 2048

        std::cout << suite.summary_table() << std::endl;

        // Verify performance increases with problem size (in absolute terms)
        for (size_t i = 1; i < suite.results.size(); i++) {
            REQUIRE(suite.results[i].flops > suite.results[i-1].flops);
        }

        // Find best result
        auto best = suite.best_by_gflops();
        REQUIRE(best != nullptr);
        std::cout << "Best by GFLOPS: " << best->config
                  << " at " << best->gflops << " GFLOPS" << std::endl;
    }

    SECTION("Custom sizes") {
        std::vector<std::tuple<Size, Size, Size>> sizes = {
            {128, 128, 128},
            {256, 512, 256},
            {512, 256, 512},
            {768, 768, 768},  // Transformer-like
            {1024, 4096, 1024}, // MLP-like
        };

        auto suite = harness.sweep_matmul_sizes(sizes);
        REQUIRE(suite.results.size() == sizes.size());

        std::cout << suite.summary_table() << std::endl;
    }
}

TEST_CASE("Tile size sensitivity", "[benchmark][matmul][tiles]") {
    BenchmarkHarness harness;

    Size M = 512, N = 512, K = 512;

    std::vector<std::tuple<Size, Size, Size>> tile_sizes = {
        {16, 16, 16},
        {32, 32, 32},
        {32, 32, 64},
        {64, 64, 64},
        {64, 64, 128},
        {128, 128, 128},
    };

    auto suite = harness.sweep_tile_sizes(M, N, K, tile_sizes);

    REQUIRE(suite.results.size() == tile_sizes.size());

    std::cout << "Tile Size Sensitivity for " << M << "x" << N << "x" << K << std::endl;
    std::cout << suite.summary_table() << std::endl;

    // Find best tile configuration
    auto best = suite.best_by_efficiency();
    REQUIRE(best != nullptr);
    std::cout << "Best tiles: " << best->Ti << "x" << best->Tj << "x" << best->Tk
              << " at " << (best->compute_efficiency * 100) << "% efficiency" << std::endl;
}

TEST_CASE("Non-square matmul benchmark", "[benchmark][matmul]") {
    BenchmarkHarness harness;

    SECTION("Tall matrix (M >> N)") {
        auto result = harness.benchmark_matmul(2048, 256, 512);
        REQUIRE(result.cycles > 0);
        std::cout << "Tall: " << result.to_string() << std::endl;
    }

    SECTION("Wide matrix (N >> M)") {
        auto result = harness.benchmark_matmul(256, 2048, 512);
        REQUIRE(result.cycles > 0);
        std::cout << "Wide: " << result.to_string() << std::endl;
    }

    SECTION("Deep matrix (K >> M, N)") {
        auto result = harness.benchmark_matmul(256, 256, 2048);
        REQUIRE(result.cycles > 0);
        std::cout << "Deep: " << result.to_string() << std::endl;
    }
}

TEST_CASE("Transformer-like dimensions", "[benchmark][matmul][transformer]") {
    BenchmarkHarness harness;

    // GPT-2 style dimensions
    SECTION("GPT-2 FFN up-projection") {
        // batch=32, seq=512, hidden=768, intermediate=3072
        auto result = harness.benchmark_matmul(32 * 512, 3072, 768);
        std::cout << "FFN up: " << result.to_string() << std::endl;
        REQUIRE(result.cycles > 0);
    }

    SECTION("GPT-2 FFN down-projection") {
        auto result = harness.benchmark_matmul(32 * 512, 768, 3072);
        std::cout << "FFN down: " << result.to_string() << std::endl;
        REQUIRE(result.cycles > 0);
    }

    SECTION("Attention QKV projection") {
        // batch=32, seq=512, hidden=768
        auto result = harness.benchmark_matmul(32 * 512, 768 * 3, 768);
        std::cout << "QKV: " << result.to_string() << std::endl;
        REQUIRE(result.cycles > 0);
    }
}

TEST_CASE("Roofline analysis", "[benchmark][roofline]") {
    BenchmarkHarness harness;

    auto suite = harness.sweep_matmul_square(64, 1024, 2);

    std::cout << "\n=== Roofline Data ===" << std::endl;
    std::cout << harness.generate_roofline_data(suite);

    // Verify all results fall under the roofline
    HardwareSpec hw = harness.hardware_spec();
    for (const auto& r : suite.results) {
        double predicted = hw.roofline_gflops(r.arithmetic_intensity);
        REQUIRE(r.gflops <= predicted * 1.1);  // Allow 10% tolerance for timing variance
    }
}

TEST_CASE("CSV export", "[benchmark][export]") {
    BenchmarkHarness harness;

    auto suite = harness.sweep_matmul_square(64, 512, 2);

    std::string csv = suite.to_csv();

    // Verify CSV format
    REQUIRE(!csv.empty());
    REQUIRE(csv.find("name,config,cycles") != std::string::npos);
    REQUIRE(csv.find("matmul") != std::string::npos);

    // Write to file for inspection
    std::ofstream out("/tmp/kpu_benchmark_results.csv");
    if (out) {
        out << csv;
        out.close();
        std::cout << "Wrote results to /tmp/kpu_benchmark_results.csv" << std::endl;
    }
}
