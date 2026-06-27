/**
 * @file l3_overfetch_analyzer.cpp
 * @brief L3 Cache and DRAM Overfetch Analysis Tool
 *
 * This tool demonstrates how L3 cache size affects DRAM traffic through overfetching.
 *
 * Key Analysis:
 * 1. For various tensor workloads, show how much DRAM traffic is generated
 * 2. Sweep L3 cache size to find the "knee" in the overfetch curve
 * 3. Quantify the benefit of larger L3 in reducing DRAM traffic
 *
 * The Fundamental Question:
 * "How much L3 cache is needed to minimize DRAM overfetching for typical ML workloads?"
 */

#include <sw/compiler/l3_scheduler.hpp>
#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <sw/compiler/schedule_characterizer.hpp>
#include <sw/concepts.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>

using namespace sw::kpu::compiler;
using sw::kpu::Cycle;
using sw::kpu::Size;

/**
 * @brief Print L3 schedule analysis
 */
void print_l3_analysis(const L3Schedule& schedule, const TensorShape& shape) {
    std::cout << "\nL3 Schedule Analysis for " << shape.to_string() << ":\n";
    std::cout << "─────────────────────────────────────────────────────────\n";

    std::cout << "L3 Configuration:\n";
    std::cout << "  L3 capacity: " << (schedule.config.l3_capacity / (1024 * 1024)) << " MB\n";
    std::cout << "  L3 tiles: " << schedule.config.l3_tile_count << "\n";
    std::cout << "  Total L3: " << (schedule.config.total_l3_capacity() / (1024 * 1024)) << " MB\n\n";

    std::cout << "Ideal Tensor Sizes:\n";
    std::cout << "  Tensor A: " << std::setw(10) << (schedule.tensor_a_size / 1024) << " KB\n";
    std::cout << "  Tensor B: " << std::setw(10) << (schedule.tensor_b_size / 1024) << " KB\n";
    std::cout << "  Tensor C: " << std::setw(10) << (schedule.tensor_c_size / 1024) << " KB\n";
    std::cout << "  Total:    " << std::setw(10)
              << ((schedule.tensor_a_size + schedule.tensor_b_size + schedule.tensor_c_size) / 1024)
              << " KB\n\n";

    std::cout << "Actual DRAM Fetches (with L3=" << (schedule.config.l3_capacity / (1024 * 1024)) << "MB):\n";
    std::cout << "  Tensor A: " << std::setw(10) << (schedule.tensor_a_fetched / 1024) << " KB "
              << "(overfetch: " << std::fixed << std::setprecision(2)
              << schedule.overfetch_factor_a << "×)\n";
    std::cout << "  Tensor B: " << std::setw(10) << (schedule.tensor_b_fetched / 1024) << " KB "
              << "(overfetch: " << std::fixed << std::setprecision(2)
              << schedule.overfetch_factor_b << "×)\n";
    std::cout << "  Tensor C: " << std::setw(10) << (schedule.tensor_c_fetched / 1024) << " KB "
              << "(overfetch: " << std::fixed << std::setprecision(2)
              << schedule.overfetch_factor_c << "×)\n";
    std::cout << "  Total:    " << std::setw(10)
              << ((schedule.tensor_a_fetched + schedule.tensor_b_fetched + schedule.tensor_c_fetched) / 1024)
              << " KB "
              << "(overfetch: " << std::fixed << std::setprecision(2)
              << schedule.overfetch_factor_total << "×)\n\n";

    std::cout << "DRAM Traffic:\n";
    std::cout << "  DRAM→L3:  " << std::setw(10) << (schedule.dram_to_l3_bytes / 1024) << " KB\n";
    std::cout << "  L3→L2:    " << std::setw(10) << (schedule.l3_to_l2_bytes / 1024) << " KB\n";
    std::cout << "  DRAM Reads:  " << std::setw(10) << (schedule.total_dram_reads / 1024) << " KB\n";
    std::cout << "  DRAM Writes: " << std::setw(10) << (schedule.total_dram_writes / 1024) << " KB\n\n";

    std::cout << "L3 Cache Statistics:\n";
    std::cout << "  Peak L3 usage: " << (schedule.l3_capacity_used_peak / 1024) << " KB "
              << "(" << std::fixed << std::setprecision(1) << (schedule.l3_utilization * 100.0) << "%)\n";
    std::cout << "  L2 tile loads: " << schedule.l2_tile_loads_total << "\n";
    std::cout << "  L3 hits:       " << schedule.l2_tile_loads_hit_l3
              << " (" << std::fixed << std::setprecision(1) << (schedule.l3_hit_rate * 100.0) << "%)\n";
    std::cout << "  L3 misses:     " << schedule.l2_tile_loads_miss_l3 << "\n";
    std::cout << "  L3 evictions:  " << schedule.num_l3_evictions << "\n";
    std::cout << std::endl;
}

/**
 * @brief Export L3 sweep results to CSV
 */
void export_l3_sweep_csv(
    const TensorShape& shape,
    const std::map<Size, L3Schedule>& sweep_results,
    const std::string& filename)
{
    (void)shape;  // Reserved for future use (e.g., adding M,N,K columns to CSV)
    std::ofstream file(filename);

    file << "L3_Size_MB,Overfetch_Total,Overfetch_A,Overfetch_B,Overfetch_C,"
         << "DRAM_Reads_KB,DRAM_Writes_KB,L3_Hit_Rate,L3_Utilization,Num_Evictions,"
         << "Peak_L3_KB,Tensor_A_KB,Tensor_B_KB,Tensor_C_KB\n";

    for (const auto& [l3_size, schedule] : sweep_results) {
        file << (l3_size / (1024 * 1024)) << ","
             << schedule.overfetch_factor_total << ","
             << schedule.overfetch_factor_a << ","
             << schedule.overfetch_factor_b << ","
             << schedule.overfetch_factor_c << ","
             << (schedule.total_dram_reads / 1024) << ","
             << (schedule.total_dram_writes / 1024) << ","
             << schedule.l3_hit_rate << ","
             << schedule.l3_utilization << ","
             << schedule.num_l3_evictions << ","
             << (schedule.l3_capacity_used_peak / 1024) << ","
             << (schedule.tensor_a_size / 1024) << ","
             << (schedule.tensor_b_size / 1024) << ","
             << (schedule.tensor_c_size / 1024) << "\n";
    }

    file.close();
    std::cout << "Exported L3 sweep to " << filename << "\n";
}

/**
 * @brief Print L3 sweep summary table
 */
void print_l3_sweep_summary(const std::map<Size, L3Schedule>& sweep_results) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    L3 SIZE SWEEP SUMMARY                           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "┌──────────┬──────────────┬──────────────┬──────────────┬──────────┬──────────┐\n";
    std::cout << "│ L3 Size  │  Overfetch   │  DRAM Reads  │ DRAM Writes  │ L3 Hit   │  L3 Util │\n";
    std::cout << "│   (MB)   │    (total)   │     (KB)     │     (KB)     │  Rate    │   (%)    │\n";
    std::cout << "├──────────┼──────────────┼──────────────┼──────────────┼──────────┼──────────┤\n";

    for (const auto& [l3_size, schedule] : sweep_results) {
        std::cout << "│ " << std::setw(8) << (l3_size / (1024 * 1024))
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(2)
                  << schedule.overfetch_factor_total
                  << " │ " << std::setw(12) << (schedule.total_dram_reads / 1024)
                  << " │ " << std::setw(12) << (schedule.total_dram_writes / 1024)
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1)
                  << (schedule.l3_hit_rate * 100.0)
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1)
                  << (schedule.l3_utilization * 100.0)
                  << " │\n";
    }

    std::cout << "└──────────┴──────────────┴──────────────┴──────────────┴──────────┴──────────┘\n\n";

    // Find knee in the curve (where improvement < 10%)
    double prev_overfetch = 0;
    Size knee_l3_size = 0;
    for (const auto& [l3_size, schedule] : sweep_results) {
        if (prev_overfetch > 0) {
            double improvement = (prev_overfetch - schedule.overfetch_factor_total) / prev_overfetch;
            if (improvement < 0.10 && knee_l3_size == 0) {
                knee_l3_size = l3_size;
                std::cout << "Knee of curve: " << (l3_size / (1024 * 1024))
                          << " MB L3 (diminishing returns beyond this point)\n";
                break;
            }
        }
        prev_overfetch = schedule.overfetch_factor_total;
    }

    std::cout << "\n";
}

int main() {
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█                L3 CACHE AND DRAM OVERFETCH ANALYSIS                      █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  Analyzes how L3 cache size affects DRAM traffic through overfetching   █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n\n";

    // Demo 1: Single workload analysis with default L3
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         DEMO 1: Single Workload L3 Analysis                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    TensorShape shape1(512, 512, 512);
    std::cout << "Analyzing workload: " << shape1.to_string() << "\n";

    // Create L2 schedule
    TileOptimizer::MemoryHierarchy mem;
    TileOptimizer optimizer(mem);
    auto tile_config = optimizer.optimize(shape1.M, shape1.N, shape1.K);

    L2TileScheduler l2_scheduler;
    auto l2_schedule = l2_scheduler.generate_schedule(shape1.M, shape1.N, shape1.K, tile_config);

    // Create L3 schedule with default 16MB L3
    L3Config l3_config;
    l3_config.l3_capacity = 16 * 1024 * 1024; // 16MB
    L3Scheduler l3_scheduler(l3_config);

    auto l3_schedule = l3_scheduler.schedule_l3(shape1, l2_schedule);
    print_l3_analysis(l3_schedule, shape1);

    // Demo 2: L3 size sweep for this workload
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         DEMO 2: L3 Size Sweep (Overfetch Analysis)                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Sweeping L3 size from 1MB to 128MB...\n";

    std::vector<Size> l3_sizes = {
        1 * 1024 * 1024,   // 1MB
        2 * 1024 * 1024,   // 2MB
        4 * 1024 * 1024,   // 4MB
        8 * 1024 * 1024,   // 8MB
        16 * 1024 * 1024,  // 16MB
        32 * 1024 * 1024,  // 32MB
        64 * 1024 * 1024,  // 64MB
        128 * 1024 * 1024  // 128MB
    };

    auto sweep_results = l3_scheduler.sweep_l3_size(shape1, l2_schedule, l3_sizes);
    print_l3_sweep_summary(sweep_results);

    export_l3_sweep_csv(shape1, sweep_results, "l3_overfetch_512x512x512.csv");

    // Demo 3: Multiple workload sizes
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         DEMO 3: Multiple Workload Sizes                           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::vector<TensorShape> workloads = {
        TensorShape(128, 128, 128),    // Small
        TensorShape(256, 256, 256),    // Medium
        TensorShape(512, 512, 512),    // Large
        TensorShape(1024, 1024, 1024)  // Very large
    };

    L3Config default_l3;
    default_l3.l3_capacity = 16 * 1024 * 1024; // 16MB
    L3Scheduler default_scheduler(default_l3);

    std::cout << "Comparing overfetch for different workload sizes (L3=16MB):\n\n";
    std::cout << "┌──────────────────┬──────────────┬──────────────┬──────────┐\n";
    std::cout << "│      Shape       │  Total Size  │  Overfetch   │ DRAM/Ideal│\n";
    std::cout << "│                  │     (KB)     │   Factor     │   Ratio   │\n";
    std::cout << "├──────────────────┼──────────────┼──────────────┼──────────┤\n";

    for (const auto& workload : workloads) {
        auto wl_tile_config = optimizer.optimize(workload.M, workload.N, workload.K);
        auto wl_l2_schedule = l2_scheduler.generate_schedule(workload.M, workload.N, workload.K, wl_tile_config);
        auto wl_l3_schedule = default_scheduler.schedule_l3(workload, wl_l2_schedule);

        Size total_size = wl_l3_schedule.tensor_a_size + wl_l3_schedule.tensor_b_size +
                         wl_l3_schedule.tensor_c_size;
        Size ideal_dram = L3Scheduler::calculate_ideal_dram_traffic(workload);
        double dram_ratio = static_cast<double>(wl_l3_schedule.total_dram_reads) / static_cast<double>(ideal_dram);

        std::cout << "│ " << std::setw(16) << workload.to_string()
                  << " │ " << std::setw(12) << (total_size / 1024)
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(2)
                  << wl_l3_schedule.overfetch_factor_total
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2)
                  << dram_ratio
                  << " │\n";
    }

    std::cout << "└──────────────────┴──────────────┴──────────────┴──────────┘\n\n";

    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS COMPLETE                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Key Findings:\n";
    std::cout << "  1. Larger L3 reduces DRAM overfetching (fewer tile reloads)\n";
    std::cout << "  2. Overfetch factor shows multiplicative DRAM traffic penalty\n";
    std::cout << "  3. Sweet spot: L3 size where overfetch < 1.5× and hit rate > 80%\n";
    std::cout << "  4. Beyond the knee: diminishing returns for larger L3\n\n";

    std::cout << "Next steps:\n";
    std::cout << "  1. python ../tools/compiler/visualize_l3_overfetch.py l3_overfetch_512x512x512.csv\n";
    std::cout << "  2. Use overfetch data for resource allocation optimization\n";
    std::cout << "  3. Consider L3 size vs compute tile count tradeoff\n\n";

    return 0;
}
