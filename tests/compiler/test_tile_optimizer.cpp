#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <iostream>
#include <iomanip>

using namespace sw::kpu::compiler;

// Helper function to print tile configuration
void print_config(const char* label, const TileOptimizer::TileConfig& config) {
    std::cout << "\n" << label << ":\n";
    std::cout << "  Tiles: Ti=" << config.Ti << " Tj=" << config.Tj << " Tk=" << config.Tk << "\n";
    std::cout << "  L1_Ki: " << config.L1_Ki << "\n";
    std::cout << "  Reuse: A=" << config.reuse_A << " B=" << config.reuse_B << " C=" << config.reuse_C << "\n";
    std::cout << "  DRAM accesses: " << config.dram_accesses << " bytes\n";
    std::cout << "  L2 footprint: " << config.l2_footprint << " bytes\n";
    std::cout << "  Arithmetic intensity: " << std::fixed << std::setprecision(2)
              << config.arithmetic_intensity << " FLOPs/byte\n";
    std::cout << "  Valid: " << (config.valid ? "YES" : "NO");
    if (!config.valid) {
        std::cout << " (" << config.reason << ")";
    }
    std::cout << "\n";
}

TEST_CASE("TileOptimizer - Default Memory Hierarchy", "[tile_optimizer][unit]") {
    TileOptimizer optimizer;
    const auto& mem = optimizer.memory_hierarchy();

    SECTION("Default memory hierarchy has expected values") {
        REQUIRE(mem.L1_size == 32 * 1024);           // 32 KB
        REQUIRE(mem.L2_size == 64 * 1024);           // 64 KB
        REQUIRE(mem.L3_size == 128 * 1024);          // 128 KB
        REQUIRE(mem.systolic_rows == 16);
        REQUIRE(mem.systolic_cols == 16);
        REQUIRE(mem.element_size == 4);              // float32
        REQUIRE(mem.L1_buffer_count == 4);
        REQUIRE(mem.L2_bank_count == 8);
        REQUIRE(mem.L3_tile_count == 4);
    }
}

TEST_CASE("TileOptimizer - Small Square Matrix", "[tile_optimizer][analytical]") {
    TileOptimizer optimizer;

    SECTION("128x128x128 matrix") {
        Size M = 128, N = 128, K = 128;

        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

        INFO("Configuration: Ti=" << config.Ti << " Tj=" << config.Tj << " Tk=" << config.Tk);

        REQUIRE(config.valid);
        REQUIRE(config.Ti >= 16);
        REQUIRE(config.Tj >= 16);
        REQUIRE(config.Tk >= 16);
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
        REQUIRE(config.Tk % 16 == 0);
        REQUIRE(config.Ti <= M);
        REQUIRE(config.Tj <= N);
        REQUIRE(config.Tk <= K);

        // Should fit in L2 cache
        REQUIRE(config.l2_footprint <= 64 * 1024);

        // Reuse factors should be calculated
        REQUIRE(config.reuse_A > 0);
        REQUIRE(config.reuse_B > 0);
        REQUIRE(config.reuse_C > 0);

        print_config("128x128x128 Analytical", config);
    }
}

TEST_CASE("TileOptimizer - Medium Square Matrix", "[tile_optimizer][analytical]") {
    TileOptimizer optimizer;

    SECTION("512x512x512 matrix") {
        Size M = 512, N = 512, K = 512;

        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

        REQUIRE(config.valid);
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
        REQUIRE(config.Tk % 16 == 0);
        REQUIRE(config.l2_footprint <= 64 * 1024);

        // For 512x512x512, tiles should be smaller than matrix
        REQUIRE(config.Ti < M);
        REQUIRE(config.Tj < N);

        // Should have significant reuse
        REQUIRE(config.reuse_A >= 2);
        REQUIRE(config.reuse_B >= 2);

        print_config("512x512x512 Analytical", config);
    }
}

TEST_CASE("TileOptimizer - Large Square Matrix (1024x1024x1024)", "[tile_optimizer][analytical]") {
    TileOptimizer optimizer;

    Size M = 1024, N = 1024, K = 1024;

    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    REQUIRE(config.valid);
    REQUIRE(config.Ti % 16 == 0);
    REQUIRE(config.Tj % 16 == 0);
    REQUIRE(config.Tk % 16 == 0);
    REQUIRE(config.l2_footprint <= 64 * 1024);

    // Check that we get good reuse
    REQUIRE(config.reuse_A >= 4);  // Each A tile used multiple times
    REQUIRE(config.reuse_B >= 4);  // Each B tile used multiple times
    REQUIRE(config.reuse_C >= 4);  // Accumulation across K

    // Arithmetic intensity should be high for large matrices
    REQUIRE(config.arithmetic_intensity > 1.0);

    print_config("1024x1024x1024 Analytical", config);

    // Calculate theoretical improvement over naive
    Size naive_dram = (M * K + K * N + M * N) * 4;  // 4 bytes per float
    double improvement = static_cast<double>(naive_dram) / static_cast<double>(config.dram_accesses);
    std::cout << "  DRAM access improvement: " << std::fixed << std::setprecision(1)
              << improvement << "x over naive\n";

    REQUIRE(improvement > 2.0);  // Should have at least 2x improvement
}

TEST_CASE("TileOptimizer - Rectangular Matrices", "[tile_optimizer][analytical]") {
    TileOptimizer optimizer;

    SECTION("Tall skinny: 1024x64x512") {
        Size M = 1024, N = 64, K = 512;
        auto config = optimizer.optimize(M, N, K);

        REQUIRE(config.valid);
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
        REQUIRE(config.Tk % 16 == 0);

        print_config("1024x64x512 (tall skinny)", config);
    }

    SECTION("Short wide: 64x1024x512") {
        Size M = 64, N = 1024, K = 512;
        auto config = optimizer.optimize(M, N, K);

        REQUIRE(config.valid);
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
        REQUIRE(config.Tk % 16 == 0);

        print_config("64x1024x512 (short wide)", config);
    }

    SECTION("Large K: 256x256x2048") {
        Size M = 256, N = 256, K = 2048;
        auto config = optimizer.optimize(M, N, K);

        REQUIRE(config.valid);
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
        REQUIRE(config.Tk % 16 == 0);

        // Large K should result in high C accumulation factor
        REQUIRE(config.reuse_C >= 4);  // Depends on Tk chosen

        print_config("256x256x2048 (large K)", config);
    }
}

TEST_CASE("TileOptimizer - Bounded Search vs Analytical", "[tile_optimizer][search]") {
    TileOptimizer optimizer;

    SECTION("Compare strategies for 512x512x512") {
        Size M = 512, N = 512, K = 512;

        auto analytical = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        auto searched = optimizer.optimize(M, N, K, TileOptimizer::Strategy::BOUNDED_SEARCH);

        REQUIRE(analytical.valid);
        REQUIRE(searched.valid);

        print_config("512x512x512 Analytical", analytical);
        print_config("512x512x512 Bounded Search", searched);

        // Bounded search should be at least as good as analytical
        REQUIRE(static_cast<double>(searched.dram_accesses) <= static_cast<double>(analytical.dram_accesses) * 1.1);  // Within 10%

        std::cout << "\n  Search found improvement: "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(analytical.dram_accesses - searched.dram_accesses) / static_cast<double>(analytical.dram_accesses))
                  << "%\n";
    }
}

TEST_CASE("TileOptimizer - Heuristic Refinement", "[tile_optimizer][heuristic]") {
    TileOptimizer optimizer;

    Size M = 512, N = 512, K = 512;

    auto analytical = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
    auto heuristic = optimizer.optimize(M, N, K, TileOptimizer::Strategy::HEURISTIC_HYBRID);

    REQUIRE(analytical.valid);
    REQUIRE(heuristic.valid);

    print_config("512x512x512 Analytical", analytical);
    print_config("512x512x512 Heuristic", heuristic);

    // Heuristic should be at least as good as analytical
    REQUIRE(static_cast<double>(heuristic.dram_accesses) <= static_cast<double>(analytical.dram_accesses) * 1.05);  // Within 5%
}

TEST_CASE("TileOptimizer - Reuse Factor Calculations", "[tile_optimizer][reuse]") {
    TileOptimizer optimizer;

    SECTION("Perfect tiling: 256x256x256 with 64x64x64 tiles") {
        Size M = 256, N = 256, K = 256;
        Size Ti = 64, Tj = 64, Tk = 64;

        TileOptimizer::TileConfig config;
        config.Ti = Ti;
        config.Tj = Tj;
        config.Tk = Tk;

        optimizer.calculate_reuse_factors(M, N, K, config);

        // M_tiles = 256/64 = 4
        // N_tiles = 256/64 = 4
        // K_tiles = 256/64 = 4

        REQUIRE(config.reuse_A == 4);  // ceil(N/Tj) = ceil(256/64) = 4
        REQUIRE(config.reuse_B == 4);  // ceil(M/Ti) = ceil(256/64) = 4
        REQUIRE(config.reuse_C == 4);  // ceil(K/Tk) = ceil(256/64) = 4
    }

    SECTION("Non-perfect tiling: 1000x1000x1000 with 64x64x64 tiles") {
        Size M = 1000, N = 1000, K = 1000;
        Size Ti = 64, Tj = 64, Tk = 64;

        TileOptimizer::TileConfig config;
        config.Ti = Ti;
        config.Tj = Tj;
        config.Tk = Tk;

        optimizer.calculate_reuse_factors(M, N, K, config);

        // ceil(1000/64) = 16
        REQUIRE(config.reuse_A == 16);
        REQUIRE(config.reuse_B == 16);
        REQUIRE(config.reuse_C == 16);
    }
}

TEST_CASE("TileOptimizer - Cache Constraint Validation", "[tile_optimizer][validation]") {
    TileOptimizer optimizer;

    SECTION("Valid configuration within L2 bounds") {
        TileOptimizer::TileConfig config;
        config.Ti = 64;
        config.Tj = 64;
        config.Tk = 64;
        config.L1_Ki = 16;

        bool valid = optimizer.validate(config);

        REQUIRE(valid);
        REQUIRE(config.valid);
        REQUIRE(config.l2_footprint <= 64 * 1024);
    }

    SECTION("Invalid configuration exceeding L2 bounds") {
        TileOptimizer::TileConfig config;
        config.Ti = 256;
        config.Tj = 256;
        config.Tk = 256;  // Way too large for L2
        config.L1_Ki = 16;

        bool valid = optimizer.validate(config);

        REQUIRE_FALSE(valid);
        REQUIRE_FALSE(config.valid);
        REQUIRE(config.reason.find("L2") != std::string::npos);
    }

    SECTION("Invalid tile not aligned to systolic dimensions") {
        TileOptimizer::TileConfig config;
        config.Ti = 63;  // Not multiple of 16
        config.Tj = 64;
        config.Tk = 64;
        config.L1_Ki = 16;

        bool valid = optimizer.validate(config);

        REQUIRE_FALSE(valid);
        REQUIRE_FALSE(config.valid);
        REQUIRE(config.reason.find("aligned") != std::string::npos);
    }
}

TEST_CASE("TileOptimizer - Search Space Bounds", "[tile_optimizer][bounds]") {
    TileOptimizer optimizer;

    SECTION("Bounds for 1024x1024x1024 matrix") {
        Size M = 1024, N = 1024, K = 1024;
        Size cache_size = 64 * 1024;  // L2 size

        auto bounds = optimizer.calculate_bounds(M, N, K, cache_size);

        REQUIRE(bounds.Ti_min == 16);  // At least one systolic tile
        REQUIRE(bounds.Tj_min == 16);
        REQUIRE(bounds.Tk_min == 16);
        REQUIRE(bounds.step == 16);

        REQUIRE(bounds.Ti_max >= bounds.Ti_min);
        REQUIRE(bounds.Tj_max >= bounds.Tj_min);
        REQUIRE(bounds.Tk_max >= bounds.Tk_min);

        REQUIRE(bounds.Ti_max % 16 == 0);
        REQUIRE(bounds.Tj_max % 16 == 0);
        REQUIRE(bounds.Tk_max % 16 == 0);

        std::cout << "\nSearch space for 1024x1024x1024:\n";
        std::cout << "  Ti: [" << bounds.Ti_min << ", " << bounds.Ti_max << "]\n";
        std::cout << "  Tj: [" << bounds.Tj_min << ", " << bounds.Tj_max << "]\n";
        std::cout << "  Tk: [" << bounds.Tk_min << ", " << bounds.Tk_max << "]\n";

        Size search_space_size = ((bounds.Ti_max - bounds.Ti_min) / bounds.step + 1) *
                                ((bounds.Tj_max - bounds.Tj_min) / bounds.step + 1) *
                                ((bounds.Tk_max - bounds.Tk_min) / bounds.step + 1);
        std::cout << "  Total configurations to search: " << search_space_size << "\n";

        // Should have dramatically reduced search space
        Size naive_space = (M / 16) * (N / 16) * (K / 16);
        std::cout << "  Space reduction: " << (naive_space / search_space_size) << "x\n";

        REQUIRE(search_space_size < naive_space / 100);  // At least 100x reduction
    }
}

TEST_CASE("TileOptimizer - Arithmetic Intensity", "[tile_optimizer][performance]") {
    TileOptimizer optimizer;

    SECTION("AI increases with better tiling") {
        Size M = 512, N = 512, K = 512;

        auto config = optimizer.optimize(M, N, K);

        // Calculate naive AI (no tiling)
        Size naive_dram = (M * K + K * N + M * N) * 4;
        Size total_flops = 2 * M * N * K;
        double naive_ai = static_cast<double>(total_flops) / static_cast<double>(naive_dram);

        std::cout << "\nArithmetic Intensity for 512x512x512:\n";
        std::cout << "  Naive (no tiling): " << std::fixed << std::setprecision(2)
                  << naive_ai << " FLOPs/byte\n";
        std::cout << "  Tiled: " << config.arithmetic_intensity << " FLOPs/byte\n";
        std::cout << "  Improvement: " << (config.arithmetic_intensity / naive_ai) << "x\n";

        // Tiled should have better AI (2x is already quite good)
        REQUIRE(config.arithmetic_intensity > naive_ai * 2.0);
    }
}

TEST_CASE("TileOptimizer - Weight-Stationary Basic Functionality", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("WS optimization for 512x512x512") {
        Size M = 512, N = 512, K = 512;

        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(ws_config.valid);
        REQUIRE(ws_config.Ti >= 16);
        REQUIRE(ws_config.Tj >= 16);
        REQUIRE(ws_config.Tk >= 16);
        REQUIRE(ws_config.Ti % 16 == 0);
        REQUIRE(ws_config.Tj % 16 == 0);
        REQUIRE(ws_config.Tk % 16 == 0);

        // PE register capacity check (16×16 × 32 registers × 4 bytes = 32KB)
        Size PE_capacity = 16 * 16 * 32 * 4;
        Size B_footprint = ws_config.Tk * ws_config.Tj * 4;
        REQUIRE(B_footprint <= PE_capacity);

        // L2 holds A + C (not A + B like OS)
        Size L2_footprint = (ws_config.Ti * ws_config.Tk + ws_config.Ti * ws_config.Tj) * 4;
        REQUIRE(L2_footprint <= 64 * 1024);

        // B reuse should be high for WS
        Size M_tiles = (M + ws_config.Ti - 1) / ws_config.Ti;
        Size K_tiles = (K + ws_config.Tk - 1) / ws_config.Tk;
        Size expected_B_reuse = M_tiles * K_tiles;
        REQUIRE(ws_config.reuse_B == expected_B_reuse);

        print_config("512x512x512 Weight-Stationary", ws_config);
    }
}

TEST_CASE("TileOptimizer - WS vs OS Comparison", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("Compare WS vs OS for batch workload (large M)") {
        Size M = 1024, N = 256, K = 256;  // Large batch, small weights

        auto os_config = optimizer.optimize(M, N, K);
        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(ws_config.valid);

        print_config("1024x256x256 Output-Stationary", os_config);
        print_config("1024x256x256 Weight-Stationary", ws_config);

        // WS should have MUCH higher B reuse for large M workloads
        REQUIRE(ws_config.reuse_B > os_config.reuse_B);

        // WS should have different tile sizes
        bool tiles_differ = (ws_config.Ti != os_config.Ti) ||
                           (ws_config.Tj != os_config.Tj) ||
                           (ws_config.Tk != os_config.Tk);
        REQUIRE(tiles_differ);

        std::cout << "\n  B reuse comparison:\n";
        std::cout << "    OS: " << os_config.reuse_B << "×\n";
        std::cout << "    WS: " << ws_config.reuse_B << "×\n";
        std::cout << "    WS improvement: "
                  << (static_cast<double>(ws_config.reuse_B) / static_cast<double>(os_config.reuse_B)) << "×\n";
    }

    SECTION("Compare WS vs OS for accumulation workload (large K)") {
        Size M = 128, N = 128, K = 1024;  // Large K, small batch

        auto os_config = optimizer.optimize(M, N, K);
        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(ws_config.valid);

        print_config("128x128x1024 Output-Stationary", os_config);
        print_config("128x128x1024 Weight-Stationary", ws_config);

        // Both should accumulate across K, but OS is more efficient
        // OS accumulates in PEs (free), WS accumulates in L2 (cost)
        REQUIRE(os_config.reuse_C > 0);
        REQUIRE(ws_config.reuse_C > 0);

        // OS should have lower DRAM accesses for large K workloads
        REQUIRE(os_config.dram_accesses < ws_config.dram_accesses);

        std::cout << "\n  Comparison for large K:\n";
        std::cout << "    C reuse - OS: " << os_config.reuse_C << "× (in PEs)\n";
        std::cout << "    C reuse - WS: " << ws_config.reuse_C << "× (in L2)\n";
        std::cout << "    DRAM - OS: " << os_config.dram_accesses << " bytes\n";
        std::cout << "    DRAM - WS: " << ws_config.dram_accesses << " bytes\n";
        double os_advantage = 100.0 * static_cast<double>(ws_config.dram_accesses - os_config.dram_accesses)
                            / static_cast<double>(ws_config.dram_accesses);
        std::cout << "    OS advantage: " << std::fixed << std::setprecision(1)
                  << os_advantage << "%\n";
    }
}

TEST_CASE("TileOptimizer - WS PE Capacity Constraint", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("Small weight matrix fits in PEs") {
        Size M = 256, N = 64, K = 64;  // Small weight matrix

        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(ws_config.valid);

        // B should fit in PE registers
        Size PE_capacity = 16 * 16 * 32 * 4;  // 32KB
        Size B_size = ws_config.Tk * ws_config.Tj * 4;
        REQUIRE(B_size <= PE_capacity);

        print_config("256x64x64 WS (small weights)", ws_config);
    }

    SECTION("Large weight matrix constrained by PE capacity") {
        Size M = 256, N = 512, K = 512;  // Larger weight matrix

        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(ws_config.valid);

        // B tiles should be constrained by PE capacity
        Size PE_capacity = 16 * 16 * 32 * 4;
        Size B_size = ws_config.Tk * ws_config.Tj * 4;
        REQUIRE(B_size <= PE_capacity);

        // Tiles should be smaller than full matrix due to PE constraint
        REQUIRE(ws_config.Tk < K);
        REQUIRE(ws_config.Tj < N);

        print_config("256x512x512 WS (large weights)", ws_config);

        std::cout << "  PE capacity: " << PE_capacity << " bytes\n";
        std::cout << "  B tile size: " << B_size << " bytes ("
                  << (100.0 * static_cast<double>(B_size) / static_cast<double>(PE_capacity)) << "% of capacity)\n";
    }
}

TEST_CASE("TileOptimizer - WS L2 Allocation", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("L2 holds A + C tiles, not A + B") {
        Size M = 512, N = 512, K = 512;

        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(ws_config.valid);

        // Calculate expected L2 footprint: A[Ti,Tk] + C[Ti,Tj]
        Size expected_footprint = (ws_config.Ti * ws_config.Tk +
                                  ws_config.Ti * ws_config.Tj) * 4;

        REQUIRE(ws_config.l2_footprint == expected_footprint);
        REQUIRE(ws_config.l2_footprint <= 64 * 1024);

        // Compare with OS which holds A + B
        Size os_footprint_if_ws_tiles = (ws_config.Ti * ws_config.Tk +
                                        ws_config.Tk * ws_config.Tj) * 4;

        std::cout << "\nL2 Footprint comparison for WS tiles:\n";
        std::cout << "  WS allocation (A + C): " << ws_config.l2_footprint << " bytes\n";
        std::cout << "  OS allocation (A + B): " << os_footprint_if_ws_tiles << " bytes\n";
        std::cout << "  Difference: " << (int)(ws_config.l2_footprint - os_footprint_if_ws_tiles)
                  << " bytes\n";
    }
}

TEST_CASE("TileOptimizer - WS Reuse Pattern Verification", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("WS reuse factors for batch processing") {
        Size M = 512, N = 128, K = 128;  // Batch workload

        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(ws_config.valid);

        Size M_tiles = (M + ws_config.Ti - 1) / ws_config.Ti;
        Size N_tiles = (N + ws_config.Tj - 1) / ws_config.Tj;
        Size K_tiles = (K + ws_config.Tk - 1) / ws_config.Tk;

        // A reuse: Minimal (flows through PEs)
        Size expected_A_reuse = std::max(Size(1), ws_config.Tj / 16);
        REQUIRE(ws_config.reuse_A == expected_A_reuse);

        // B reuse: Maximal (stays in PEs)
        Size expected_B_reuse = M_tiles * K_tiles;
        REQUIRE(ws_config.reuse_B == expected_B_reuse);

        // C reuse: Accumulated in L2
        Size expected_C_reuse = K_tiles;
        REQUIRE(ws_config.reuse_C == expected_C_reuse);

        print_config("512x128x128 WS Reuse Analysis", ws_config);

        std::cout << "\n  Tile counts:\n";
        std::cout << "    M_tiles: " << M_tiles << "\n";
        std::cout << "    N_tiles: " << N_tiles << "\n";
        std::cout << "    K_tiles: " << K_tiles << "\n";
        std::cout << "\n  Reuse verification:\n";
        std::cout << "    A reuse: " << ws_config.reuse_A << "× (minimal)\n";
        std::cout << "    B reuse: " << ws_config.reuse_B << "× (maximal!)\n";
        std::cout << "    C reuse: " << ws_config.reuse_C << "× (K accumulation)\n";
    }
}

TEST_CASE("TileOptimizer - WS Energy Implications", "[tile_optimizer][weight_stationary]") {
    TileOptimizer optimizer;

    SECTION("WS should have different DRAM access pattern than OS") {
        Size M = 512, N = 256, K = 256;

        auto os_config = optimizer.optimize(M, N, K);
        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(ws_config.valid);

        print_config("512x256x256 OS Energy", os_config);
        print_config("512x256x256 WS Energy", ws_config);

        // DRAM accesses should differ due to different reuse patterns
        bool dram_differs = (os_config.dram_accesses != ws_config.dram_accesses);
        REQUIRE(dram_differs);

        std::cout << "\n  DRAM access comparison:\n";
        std::cout << "    OS: " << os_config.dram_accesses << " bytes\n";
        std::cout << "    WS: " << ws_config.dram_accesses << " bytes\n";

        if (ws_config.dram_accesses < os_config.dram_accesses) {
            double savings = 100.0 * static_cast<double>(os_config.dram_accesses - ws_config.dram_accesses)
                           / static_cast<double>(os_config.dram_accesses);
            std::cout << "    WS savings: " << std::fixed << std::setprecision(1)
                      << savings << "%\n";
        } else {
            double overhead = 100.0 * static_cast<double>(ws_config.dram_accesses - os_config.dram_accesses)
                            / static_cast<double>(os_config.dram_accesses);
            std::cout << "    WS overhead: " << std::fixed << std::setprecision(1)
                      << overhead << "%\n";
        }
    }
}

TEST_CASE("TileOptimizer - Input-Stationary Basic Functionality", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("IS optimization for 512x512x512") {
        Size M = 512, N = 512, K = 512;

        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(is_config.valid);
        REQUIRE(is_config.Ti >= 16);
        REQUIRE(is_config.Tj >= 16);
        REQUIRE(is_config.Tk >= 16);
        REQUIRE(is_config.Ti % 16 == 0);
        REQUIRE(is_config.Tj % 16 == 0);
        REQUIRE(is_config.Tk % 16 == 0);

        // PE register capacity check (16×16 × 32 registers × 4 bytes = 32KB)
        Size PE_capacity = 16 * 16 * 32 * 4;
        Size A_footprint = is_config.Ti * is_config.Tk * 4;
        REQUIRE(A_footprint <= PE_capacity);

        // L2 holds B + C (not A + B like OS)
        Size L2_footprint = (is_config.Tk * is_config.Tj + is_config.Ti * is_config.Tj) * 4;
        REQUIRE(L2_footprint <= 64 * 1024);

        // A reuse should be high for IS
        Size N_tiles = (N + is_config.Tj - 1) / is_config.Tj;
        Size K_tiles = (K + is_config.Tk - 1) / is_config.Tk;
        Size expected_A_reuse = N_tiles * K_tiles;
        REQUIRE(is_config.reuse_A == expected_A_reuse);

        print_config("512x512x512 Input-Stationary", is_config);
    }
}

TEST_CASE("TileOptimizer - IS vs OS Comparison", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("Compare IS vs OS for wide output workload (large N)") {
        Size M = 256, N = 1024, K = 256;  // Large output features, small batch

        auto os_config = optimizer.optimize(M, N, K);
        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(is_config.valid);

        print_config("256x1024x256 Output-Stationary", os_config);
        print_config("256x1024x256 Input-Stationary", is_config);

        // IS should have MUCH higher A reuse for large N workloads
        REQUIRE(is_config.reuse_A > os_config.reuse_A);

        // IS should have different tile sizes
        bool tiles_differ = (is_config.Ti != os_config.Ti) ||
                           (is_config.Tj != os_config.Tj) ||
                           (is_config.Tk != os_config.Tk);
        REQUIRE(tiles_differ);

        std::cout << "\n  A reuse comparison:\n";
        std::cout << "    OS: " << os_config.reuse_A << "×\n";
        std::cout << "    IS: " << is_config.reuse_A << "×\n";
        std::cout << "    IS improvement: "
                  << (static_cast<double>(is_config.reuse_A) / static_cast<double>(os_config.reuse_A)) << "×\n";
    }

    SECTION("Compare IS vs OS for accumulation workload (large K)") {
        Size M = 128, N = 128, K = 1024;  // Large K, small batch

        auto os_config = optimizer.optimize(M, N, K);
        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(is_config.valid);

        print_config("128x128x1024 Output-Stationary", os_config);
        print_config("128x128x1024 Input-Stationary", is_config);

        // Both should accumulate across K, but OS is more efficient
        // OS accumulates in PEs (free), IS accumulates in L2 (cost)
        REQUIRE(os_config.reuse_C > 0);
        REQUIRE(is_config.reuse_C > 0);

        // OS should have lower DRAM accesses for large K workloads
        REQUIRE(os_config.dram_accesses < is_config.dram_accesses);

        std::cout << "\n  Comparison for large K:\n";
        std::cout << "    C reuse - OS: " << os_config.reuse_C << "× (in PEs)\n";
        std::cout << "    C reuse - IS: " << is_config.reuse_C << "× (in L2)\n";
        std::cout << "    DRAM - OS: " << os_config.dram_accesses << " bytes\n";
        std::cout << "    DRAM - IS: " << is_config.dram_accesses << " bytes\n";
        double os_advantage = 100.0 * static_cast<double>(is_config.dram_accesses - os_config.dram_accesses)
                            / static_cast<double>(is_config.dram_accesses);
        std::cout << "    OS advantage: " << std::fixed << std::setprecision(1)
                  << os_advantage << "%\n";
    }
}

TEST_CASE("TileOptimizer - IS PE Capacity Constraint", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("Small input matrix fits in PEs") {
        Size M = 64, N = 256, K = 64;  // Small input matrix

        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(is_config.valid);

        // A should fit in PE registers
        Size PE_capacity = 16 * 16 * 32 * 4;  // 32KB
        Size A_size = is_config.Ti * is_config.Tk * 4;
        REQUIRE(A_size <= PE_capacity);

        print_config("64x256x64 IS (small inputs)", is_config);
    }

    SECTION("Large input matrix constrained by PE capacity") {
        Size M = 512, N = 256, K = 512;  // Larger input matrix

        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(is_config.valid);

        // A tiles must respect PE capacity
        Size PE_capacity = 16 * 16 * 32 * 4;
        Size A_size = is_config.Ti * is_config.Tk * 4;
        REQUIRE(A_size <= PE_capacity);

        // Tiles should be constrained by PE capacity
        REQUIRE(is_config.Ti * is_config.Tk <= PE_capacity / 4);

        print_config("512x256x512 IS (large inputs)", is_config);
    }
}

TEST_CASE("TileOptimizer - IS L2 Allocation", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("L2 holds B + C tiles (not A)") {
        Size M = 256, N = 256, K = 256;

        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(is_config.valid);

        // L2 footprint should be B + C tiles
        Size expected_l2 = (is_config.Tk * is_config.Tj + is_config.Ti * is_config.Tj) * 4;
        REQUIRE(is_config.l2_footprint == expected_l2);

        // Should NOT include A tiles (they're in PEs)
        Size A_tile = is_config.Ti * is_config.Tk * 4;
        REQUIRE(is_config.l2_footprint < (expected_l2 + A_tile));

        print_config("256x256x256 IS L2 Allocation", is_config);

        std::cout << "\n  L2 allocation:\n";
        std::cout << "    B tile: " << (is_config.Tk * is_config.Tj * 4) << " bytes\n";
        std::cout << "    C tile: " << (is_config.Ti * is_config.Tj * 4) << " bytes\n";
        std::cout << "    Total:  " << is_config.l2_footprint << " bytes\n";
        std::cout << "    A tile (in PEs): " << A_tile << " bytes\n";
    }
}

TEST_CASE("TileOptimizer - IS Reuse Pattern", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("IS maximizes A reuse") {
        Size M = 128, N = 512, K = 256;

        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(is_config.valid);

        Size N_tiles = (N + is_config.Tj - 1) / is_config.Tj;
        Size K_tiles = (K + is_config.Tk - 1) / is_config.Tk;

        // A reuse should be N_tiles × K_tiles (maximal!)
        Size expected_A_reuse = N_tiles * K_tiles;
        REQUIRE(is_config.reuse_A == expected_A_reuse);

        // B reuse should be minimal (flows through)
        REQUIRE(is_config.reuse_B >= 1);
        REQUIRE(is_config.reuse_B < is_config.reuse_A);

        // C accumulates K_tiles times
        REQUIRE(is_config.reuse_C == K_tiles);

        print_config("128x512x256 IS Reuse", is_config);

        std::cout << "\n  Reuse factors:\n";
        std::cout << "    A: " << is_config.reuse_A << "× (MAXIMAL - stays in PEs)\n";
        std::cout << "    B: " << is_config.reuse_B << "× (minimal - flows through)\n";
        std::cout << "    C: " << is_config.reuse_C << "× (accumulated in L2)\n";
    }
}

TEST_CASE("TileOptimizer - IS Energy Implications", "[tile_optimizer][input_stationary]") {
    TileOptimizer optimizer;

    SECTION("IS vs OS energy for wide output workload") {
        Size M = 256, N = 512, K = 256;  // Favorable for IS (large N)

        auto os_config = optimizer.optimize(M, N, K);
        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(is_config.valid);

        print_config("256x512x256 OS Energy", os_config);
        print_config("256x512x256 IS Energy", is_config);

        // DRAM accesses should differ due to different reuse patterns
        bool dram_differs = (os_config.dram_accesses != is_config.dram_accesses);
        REQUIRE(dram_differs);

        std::cout << "\n  DRAM access comparison:\n";
        std::cout << "    OS: " << os_config.dram_accesses << " bytes\n";
        std::cout << "    IS: " << is_config.dram_accesses << " bytes\n";

        if (is_config.dram_accesses < os_config.dram_accesses) {
            double savings = 100.0 * static_cast<double>(os_config.dram_accesses - is_config.dram_accesses)
                           / static_cast<double>(os_config.dram_accesses);
            std::cout << "    IS savings: " << std::fixed << std::setprecision(1)
                      << savings << "%\n";
        } else {
            double overhead = 100.0 * static_cast<double>(is_config.dram_accesses - os_config.dram_accesses)
                            / static_cast<double>(os_config.dram_accesses);
            std::cout << "    IS overhead: " << std::fixed << std::setprecision(1)
                      << overhead << "%\n";
        }
    }
}

TEST_CASE("TileOptimizer - Three-Way Strategy Comparison", "[tile_optimizer][three_way]") {
    TileOptimizer optimizer;

    SECTION("Compare OS vs WS vs IS for balanced workload") {
        Size M = 256, N = 256, K = 256;

        auto os_config = optimizer.optimize(M, N, K);
        auto ws_config = optimizer.optimize_weight_stationary(M, N, K);
        auto is_config = optimizer.optimize_input_stationary(M, N, K);

        REQUIRE(os_config.valid);
        REQUIRE(ws_config.valid);
        REQUIRE(is_config.valid);

        print_config("256x256x256 Output-Stationary", os_config);
        print_config("256x256x256 Weight-Stationary", ws_config);
        print_config("256x256x256 Input-Stationary", is_config);

        std::cout << "\n  Three-way comparison:\n";
        std::cout << "    Reuse A - OS: " << os_config.reuse_A << "×, WS: "
                  << ws_config.reuse_A << "×, IS: " << is_config.reuse_A << "×\n";
        std::cout << "    Reuse B - OS: " << os_config.reuse_B << "×, WS: "
                  << ws_config.reuse_B << "×, IS: " << is_config.reuse_B << "×\n";
        std::cout << "    Reuse C - OS: " << os_config.reuse_C << "×, WS: "
                  << ws_config.reuse_C << "×, IS: " << is_config.reuse_C << "×\n";
        std::cout << "    DRAM    - OS: " << os_config.dram_accesses
                  << ", WS: " << ws_config.dram_accesses
                  << ", IS: " << is_config.dram_accesses << " bytes\n";

        // Verify each strategy has its characteristic reuse pattern
        REQUIRE(ws_config.reuse_B > os_config.reuse_B);  // WS maximizes B reuse
        REQUIRE(is_config.reuse_A > os_config.reuse_A);  // IS maximizes A reuse
    }
}
