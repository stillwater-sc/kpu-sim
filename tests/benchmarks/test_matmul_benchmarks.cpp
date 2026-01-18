// Matmul Benchmark Tests
// Tests for matrix multiplication performance across sizes and configurations

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/benchmark/benchmark.hpp>

#include <filesystem>
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

    SECTION("Powers of 2 sweep (standard)") {
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

    SECTION("Extended sweep to 16K (v0.3.1)") {
        // Extended sweep for v0.3.1 - includes sizes up to 16384
        auto suite = harness.sweep_matmul_square(64, 16384, 2);

        // Should have 9 sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
        REQUIRE(suite.results.size() >= 9);

        std::cout << "\n=== Extended Matmul Sweep (v0.3.1) ===" << std::endl;
        std::cout << suite.summary_table() << std::endl;

        // Verify large problems achieve higher efficiency (compute-bound)
        auto& largest = suite.results.back();
        REQUIRE(largest.is_compute_bound());  // 16K should be compute-bound

        // Large problems should approach peak efficiency (asymptotic ~50%)
        REQUIRE(largest.compute_efficiency >= 0.49);  // At least 49% of peak

        std::cout << "Largest problem (" << largest.config << "):\n";
        std::cout << "  Efficiency: " << (largest.compute_efficiency * 100) << "%\n";
        std::cout << "  GFLOPS: " << largest.gflops << "\n";
        std::cout << "  Arithmetic Intensity: " << largest.arithmetic_intensity << " FLOP/byte\n";
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

TEST_CASE("Extended tile sensitivity (v0.3.1)", "[benchmark][matmul][tiles][v031]") {
    BenchmarkHarness harness;

    // Comprehensive tile configurations including asymmetric tiles
    std::vector<std::tuple<Size, Size, Size>> tile_configs = {
        // Symmetric small tiles
        {16, 16, 16},
        {32, 32, 32},
        {64, 64, 64},
        {128, 128, 128},
        // Asymmetric tiles - K-dimension emphasis (reduces K-loop overhead)
        {32, 32, 64},
        {32, 32, 128},
        {64, 64, 128},
        {64, 64, 256},
        // Asymmetric tiles - M/N emphasis (better for tall/wide matrices)
        {64, 32, 32},
        {32, 64, 32},
        {128, 64, 64},
        {64, 128, 64},
        // Large symmetric (for big problems)
        {256, 256, 256},
    };

    SECTION("Small problem (256x256x256) - memory-bound regime") {
        Size M = 256, N = 256, K = 256;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);

        std::cout << "\n=== Tile Sensitivity: 256x256x256 (Small/Memory-bound) ===\n";
        std::cout << suite.summary_table() << std::endl;

        auto best = suite.best_by_efficiency();
        REQUIRE(best != nullptr);
        std::cout << "Best: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " → " << (best->compute_efficiency * 100) << "% efficiency\n";

        // For small problems, smaller tiles often work better
        REQUIRE(best->Ti <= 128);
    }

    SECTION("Medium problem (1024x1024x1024) - compute-bound regime") {
        Size M = 1024, N = 1024, K = 1024;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);

        std::cout << "\n=== Tile Sensitivity: 1024x1024x1024 (Medium/Compute-bound) ===\n";
        std::cout << suite.summary_table() << std::endl;

        auto best = suite.best_by_efficiency();
        REQUIRE(best != nullptr);
        std::cout << "Best: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " → " << (best->compute_efficiency * 100) << "% efficiency\n";

        // Medium problems should achieve reasonable efficiency
        REQUIRE(best->compute_efficiency > 0.3);
    }

    SECTION("Large problem (4096x4096x4096) - deeply compute-bound") {
        Size M = 4096, N = 4096, K = 4096;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);

        std::cout << "\n=== Tile Sensitivity: 4096x4096x4096 (Large/Compute-bound) ===\n";
        std::cout << suite.summary_table() << std::endl;

        auto best = suite.best_by_efficiency();
        REQUIRE(best != nullptr);
        std::cout << "Best: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " → " << (best->compute_efficiency * 100) << "% efficiency\n";

        // Large problems should achieve high efficiency with optimal tiles
        REQUIRE(best->compute_efficiency > 0.4);
    }

    SECTION("Tall matrix tile sensitivity (2048x256x512)") {
        Size M = 2048, N = 256, K = 512;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);

        std::cout << "\n=== Tile Sensitivity: 2048x256x512 (Tall) ===\n";
        std::cout << suite.summary_table() << std::endl;

        auto best = suite.best_by_efficiency();
        REQUIRE(best != nullptr);
        std::cout << "Best: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " → " << (best->compute_efficiency * 100) << "% efficiency\n";
    }

    SECTION("Transformer FFN tile sensitivity (16384x3072x768)") {
        // GPT-2 style: batch*seq=16384, intermediate=3072, hidden=768
        Size M = 16384, N = 3072, K = 768;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);

        std::cout << "\n=== Tile Sensitivity: 16384x3072x768 (Transformer FFN) ===\n";
        std::cout << suite.summary_table() << std::endl;

        auto best = suite.best_by_efficiency();
        REQUIRE(best != nullptr);
        std::cout << "Best: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " → " << (best->compute_efficiency * 100) << "% efficiency\n";

        // Transformer workloads should benefit from larger tiles
        REQUIRE(best->compute_efficiency > 0.3);
    }

    SECTION("JSON export of tile analysis") {
        Size M = 512, N = 512, K = 512;
        auto suite = harness.sweep_tile_sizes(M, N, K, tile_configs);
        suite.name = "tile_sensitivity_512";
        suite.description = "Tile sensitivity analysis for 512x512x512 matmul";

        std::string json = suite.to_json();
        REQUIRE(!json.empty());

        // Write to file for analysis
        auto path = std::filesystem::temp_directory_path() / "kpu_tile_sensitivity.json";
        std::ofstream out(path);
        if (out) {
            out << json;
            out.close();
            std::cout << "Wrote tile sensitivity results to " << path << std::endl;
        }
    }
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

TEST_CASE("Multi-level roofline analysis (v0.3.2)", "[benchmark][roofline][v032]") {
    BenchmarkHarness harness;

    SECTION("Ridge points and bottleneck classification") {
        HardwareSpec hw = harness.hardware_spec();

        // Verify ridge points are calculated correctly
        REQUIRE(hw.ridge_point_external() == hw.peak_gflops / hw.external_bw_gbs);
        REQUIRE(hw.ridge_point_l3() == hw.peak_gflops / hw.l3_bw_gbs);
        REQUIRE(hw.ridge_point_l2() == hw.peak_gflops / hw.l2_bw_gbs);

        // Ridge points should decrease as bandwidth increases
        REQUIRE(hw.ridge_point_external() > hw.ridge_point_l3());
        REQUIRE(hw.ridge_point_l3() > hw.ridge_point_l2());

        // Test bottleneck classification
        REQUIRE(hw.bottleneck_level(hw.ridge_point_external() + 1) == "compute");
        REQUIRE(hw.bottleneck_level(hw.ridge_point_l3() + 1) == "external");
        REQUIRE(hw.bottleneck_level(hw.ridge_point_l2() + 1) == "l3");
        REQUIRE(hw.bottleneck_level(hw.ridge_point_l2() - 1) == "l2");

        std::cout << "\nMulti-level Ridge Points:\n";
        std::cout << "  External: " << hw.ridge_point_external() << " FLOP/byte\n";
        std::cout << "  L3:       " << hw.ridge_point_l3() << " FLOP/byte\n";
        std::cout << "  L2:       " << hw.ridge_point_l2() << " FLOP/byte\n";
    }

    SECTION("Roofline JSON output") {
        auto suite = harness.sweep_matmul_square(64, 512, 2);
        std::string json = harness.generate_roofline_json(suite);

        // Verify JSON structure
        REQUIRE(!json.empty());
        REQUIRE(json.find("\"version\": \"0.3.2\"") != std::string::npos);
        REQUIRE(json.find("\"hardware\":") != std::string::npos);
        REQUIRE(json.find("\"ridge_point_external\":") != std::string::npos);
        REQUIRE(json.find("\"ridge_point_l3\":") != std::string::npos);
        REQUIRE(json.find("\"ridge_point_l2\":") != std::string::npos);
        REQUIRE(json.find("\"analysis\":") != std::string::npos);
        REQUIRE(json.find("\"compute_bound_count\":") != std::string::npos);
        REQUIRE(json.find("\"bottleneck\":") != std::string::npos);

        std::cout << "\nRoofline JSON (first 500 chars):\n";
        std::cout << json.substr(0, 500) << "...\n";
    }

    SECTION("Roofline efficiency analysis") {
        auto suite = harness.sweep_matmul_square(64, 2048, 2);
        HardwareSpec hw = harness.hardware_spec();

        size_t compute_bound = 0;
        size_t memory_bound = 0;

        for (const auto& r : suite.results) {
            if (r.arithmetic_intensity >= hw.ridge_point_external()) {
                compute_bound++;
                // Compute-bound should achieve reasonable efficiency
                REQUIRE(r.compute_efficiency > 0.3);
            } else {
                memory_bound++;
            }
        }

        std::cout << "\nEfficiency Analysis:\n";
        std::cout << "  Compute-bound points: " << compute_bound << "\n";
        std::cout << "  Memory-bound points:  " << memory_bound << "\n";

        // Most large problems should be compute-bound
        REQUIRE(compute_bound > memory_bound);
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
    auto benchmark_path = std::filesystem::temp_directory_path() / "kpu_benchmark_results.csv";
    std::ofstream out(benchmark_path);
    if (out) {
        out << csv;
        out.close();
        std::cout << "Wrote results to " << benchmark_path << std::endl;
    }
}

TEST_CASE("JSON export (v0.3.1)", "[benchmark][export][json]") {
    BenchmarkHarness harness;

    SECTION("Single result JSON") {
        auto result = harness.benchmark_matmul(256, 256, 256);
        std::string json = result.to_json();

        // Verify JSON structure
        REQUIRE(!json.empty());
        REQUIRE(json.find("\"name\":") != std::string::npos);
        REQUIRE(json.find("\"config\":") != std::string::npos);
        REQUIRE(json.find("\"timing\":") != std::string::npos);
        REQUIRE(json.find("\"compute\":") != std::string::npos);
        REQUIRE(json.find("\"memory\":") != std::string::npos);
        REQUIRE(json.find("\"utilization\":") != std::string::npos);

        std::cout << "Single result JSON:\n" << json << std::endl;
    }

    SECTION("Suite JSON") {
        auto suite = harness.sweep_matmul_square(64, 512, 2);
        suite.name = "matmul_sweep_v031";
        suite.description = "Extended matmul sweep for v0.3.1 benchmarking";

        std::string json = suite.to_json();

        // Verify JSON structure
        REQUIRE(!json.empty());
        REQUIRE(json.find("\"name\": \"matmul_sweep_v031\"") != std::string::npos);
        REQUIRE(json.find("\"version\": \"0.3.1\"") != std::string::npos);
        REQUIRE(json.find("\"results\":") != std::string::npos);

        // Write to file for inspection
        auto benchmark_path = std::filesystem::temp_directory_path() / "kpu_benchmark_results.json";
        std::ofstream out(benchmark_path);
        if (out) {
            out << json;
            out.close();
            std::cout << "Wrote JSON results to " << benchmark_path << std::endl;
        }
    }
}
