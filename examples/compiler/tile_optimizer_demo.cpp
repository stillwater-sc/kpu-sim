/**
 * @file tile_optimizer_demo.cpp
 * @brief Demonstration of TileOptimizer usage
 */

#include <sw/compiler/tile_optimizer.hpp>
#include <iostream>
#include <iomanip>

using namespace sw::kpu::compiler;

void print_config(const char* label, const TileOptimizer::TileConfig& config,
                 Size M, Size N, Size K) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << label << "\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Matrix: C[" << M << "," << N << "] = A[" << M << "," << K
              << "] × B[" << K << "," << N << "]\n\n";

    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti (M-tiles): " << config.Ti << " ("
              << (M / config.Ti) << " tiles)\n";
    std::cout << "  Tj (N-tiles): " << config.Tj << " ("
              << (N / config.Tj) << " tiles)\n";
    std::cout << "  Tk (K-tiles): " << config.Tk << " ("
              << (K / config.Tk) << " tiles)\n";
    std::cout << "  L1 K-chunk:   " << config.L1_Ki << "\n\n";

    std::cout << "Reuse Factors:\n";
    std::cout << "  A tiles:      " << config.reuse_A << "x reuse\n";
    std::cout << "  B tiles:      " << config.reuse_B << "x reuse\n";
    std::cout << "  C accumulate: " << config.reuse_C << "x partial sums\n\n";

    std::cout << "Memory Traffic:\n";
    std::cout << "  DRAM:         " << std::fixed << std::setprecision(2)
              << (static_cast<double>(config.dram_accesses) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  L3:           " << (static_cast<double>(config.l3_accesses) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  L2:           " << (static_cast<double>(config.l2_accesses) / (1024.0 * 1024.0)) << " MB\n\n";

    std::cout << "Cache Footprint:\n";
    std::cout << "  L2 footprint: " << (static_cast<double>(config.l2_footprint) / 1024.0) << " KB\n";
    std::cout << "  L3 footprint: " << (static_cast<double>(config.l3_footprint) / 1024.0) << " KB\n\n";

    std::cout << "Performance Metrics:\n";
    std::cout << "  Arithmetic Intensity: " << config.arithmetic_intensity
              << " FLOPs/byte\n";

    // Calculate naive DRAM traffic
    Size naive_dram = (M * K + K * N + M * N) * 4;  // 4 bytes per float32
    double improvement = static_cast<double>(naive_dram) / static_cast<double>(config.dram_accesses);
    std::cout << "  DRAM Reduction:       " << improvement << "x vs naive\n\n";

    std::cout << "Validation: " << (config.valid ? "✓ VALID" : "✗ INVALID");
    if (!config.valid) {
        std::cout << " - " << config.reason;
    }
    std::cout << "\n";
}

int main() {
    std::cout << "TileOptimizer Demonstration\n";
    std::cout << "===========================\n\n";

    // Create optimizer with default KPU memory hierarchy
    TileOptimizer optimizer;

    std::cout << "Memory Hierarchy Configuration:\n";
    const auto& mem = optimizer.memory_hierarchy();
    std::cout << "  L1 buffers:   " << (mem.L1_size / 1024) << " KB × "
              << mem.L1_buffer_count << "\n";
    std::cout << "  L2 banks:     " << (mem.L2_size / 1024) << " KB × "
              << mem.L2_bank_count << "\n";
    std::cout << "  L3 tiles:     " << (mem.L3_size / 1024) << " KB × "
              << mem.L3_tile_count << "\n";
    std::cout << "  Systolic:     " << mem.systolic_rows << "×"
              << mem.systolic_cols << " PEs\n";

    // Example 1: Small square matrix
    {
        Size M = 256, N = 256, K = 256;
        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        print_config("Example 1: Small Square Matrix (256×256×256)", config, M, N, K);
    }

    // Example 2: Large square matrix
    {
        Size M = 1024, N = 1024, K = 1024;
        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        print_config("Example 2: Large Square Matrix (1024×1024×1024)", config, M, N, K);
    }

    // Example 3: Tall skinny matrix
    {
        Size M = 2048, N = 128, K = 512;
        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        print_config("Example 3: Tall Skinny Matrix (2048×128×512)", config, M, N, K);
    }

    // Example 4: Short wide matrix
    {
        Size M = 128, N = 2048, K = 512;
        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        print_config("Example 4: Short Wide Matrix (128×2048×512)", config, M, N, K);
    }

    // Example 5: Large K dimension
    {
        Size M = 512, N = 512, K = 4096;
        auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        print_config("Example 5: Large K Dimension (512×512×4096)", config, M, N, K);
    }

    // Example 6: Compare strategies
    {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Example 6: Strategy Comparison (512×512×512)\n";
        std::cout << std::string(60, '=') << "\n";

        Size M = 512, N = 512, K = 512;

        auto analytical = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);
        auto searched = optimizer.optimize(M, N, K, TileOptimizer::Strategy::BOUNDED_SEARCH);
        auto heuristic = optimizer.optimize(M, N, K, TileOptimizer::Strategy::HEURISTIC_HYBRID);

        std::cout << "\nStrategy           | Tiles        | DRAM (MB) | AI (FLOPs/byte)\n";
        std::cout << std::string(70, '-') << "\n";

        auto print_row = [](const char* name, const TileOptimizer::TileConfig& cfg) {
            std::cout << std::setw(18) << std::left << name << " | "
                      << std::setw(3) << cfg.Ti << "×"
                      << std::setw(3) << cfg.Tj << "×"
                      << std::setw(3) << cfg.Tk << " | "
                      << std::setw(9) << std::fixed << std::setprecision(2)
                      << (static_cast<double>(cfg.dram_accesses) / (1024.0 * 1024.0)) << " | "
                      << std::setw(16) << cfg.arithmetic_intensity << "\n";
        };

        print_row("Analytical", analytical);
        print_row("Bounded Search", searched);
        print_row("Heuristic", heuristic);

        std::cout << "\nSearch improvement: "
                  << std::setprecision(1)
                  << (100.0 * static_cast<double>(analytical.dram_accesses - searched.dram_accesses)
                      / static_cast<double>(analytical.dram_accesses))
                  << "% reduction in DRAM traffic\n";
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Demonstration Complete!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
