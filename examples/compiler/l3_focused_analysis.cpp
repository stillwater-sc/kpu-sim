/**
 * @file l3_focused_analysis.cpp
 * @brief Focused L3 Overfetch Analysis (Fast, Practical)
 *
 * A more practical analysis that completes quickly while still showing
 * the key insights about how dataflow strategies and tensor shapes
 * interact with L3 cache requirements.
 *
 * Focus: Representative workloads × All strategies × Key L3 sizes
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
#include <sstream>

using namespace sw::kpu::compiler;
using sw::kpu::Cycle;
using sw::kpu::Size;

void analyze_and_export() {
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█              FOCUSED L3 OVERFETCH ANALYSIS                               █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  Representative workloads × Dataflow strategies × L3 sizes              █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n\n";

    // Focused workload set: diverse but manageable
    struct Workload {
        std::string name;
        std::string category;
        std::string aspect;
        Size M, N, K;
    };

    std::vector<Workload> workloads = {
        // Square
        {"Small Square", "Baseline", "Square", 512, 512, 512},
        {"Medium Square", "Baseline", "Square", 2048, 2048, 2048},

        // Tall (batch dimension large)
        {"Tall: Long Context", "Attention", "Tall", 16384, 512, 512},
        {"Tall: Batch Inference", "Projection", "Tall", 8192, 1024, 1024},

        // Wide (vocab dimension large)
        {"Wide: Vocab Projection", "Projection", "Wide", 512, 16384, 2048},
        {"Wide: Large Vocab", "Projection", "Wide", 256, 32768, 1536},

        // Deep (hidden dimension large)
        {"Deep: MLP Up", "MLP", "Deep", 1024, 1024, 8192},
        {"Deep: Extreme MLP", "MLP", "Deep", 512, 512, 16384},

        // Realistic transformers
        {"BERT Q×K^T", "Attention", "Balanced", 2048, 2048, 768},
        {"GPT-2 MLP", "MLP", "Deep", 2048, 1536, 6144},
        {"LLaMA-2 7B Attn", "Attention", "Square", 4096, 4096, 4096},
        {"User Example 32k×7k", "Projection", "Tall-Wide", 32768, 7168, 7168},
    };

    // Strategies
    std::vector<std::pair<DataflowStrategy, std::string>> strategies = {
        {DataflowStrategy::WEIGHT_STATIONARY, "WS"},
        {DataflowStrategy::INPUT_STATIONARY, "IS"},
        {DataflowStrategy::OUTPUT_STATIONARY, "OS"}
    };

    // L3 sizes (including small distributed L3 tiles)
    std::vector<Size> l3_sizes = {
        1 * 1024 * 1024,    // 1MB  (small distributed L3)
        2 * 1024 * 1024,    // 2MB  (small distributed L3)
        16 * 1024 * 1024,   // 16MB
        64 * 1024 * 1024,   // 64MB
        256 * 1024 * 1024   // 256MB
    };

    Size total_configs = workloads.size() * strategies.size() * l3_sizes.size();
    std::cout << "Workloads: " << workloads.size() << "\n";
    std::cout << "Strategies: " << strategies.size() << " (WS, IS, OS)\n";
    std::cout << "L3 sizes: " << l3_sizes.size() << " (1MB, 2MB, 16MB, 64MB, 256MB)\n";
    std::cout << "Total configurations: " << total_configs << "\n\n";

    // Setup
    TileOptimizer::MemoryHierarchy mem;
    TileOptimizer optimizer(mem);
    L2TileScheduler l2_scheduler;

    // Open CSV file
    std::ofstream csv("l3_focused_analysis.csv");
    csv << "Workload_Name,Category,Aspect,M,N,K,Strategy,"
        << "L3_Size_MB,Overfetch_Total,Overfetch_A,Overfetch_B,Overfetch_C,"
        << "L3_Hit_Rate,L3_Utilization,DRAM_Reads_MB,Num_Evictions,"
        << "Tensor_A_MB,Tensor_B_MB,Tensor_C_MB\n";

    std::cout << "Starting analysis...\n\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    Size config_count = 0;
    for (const auto& wl : workloads) {
        std::cout << "Analyzing: " << wl.name << " (" << wl.M << "×" << wl.N << "×" << wl.K << ")\n";

        // Get tile config
        auto tile_config = optimizer.optimize(wl.M, wl.N, wl.K);

        for (const auto& [strategy, strategy_name] : strategies) {
            // Map DataflowStrategy to L2TileScheduler::SchedulingStrategy.
            // Initialize to silence gcc -Wmaybe-uninitialized (the switch
            // below covers all three enum values but the optimizer can't tell).
            L2TileScheduler::SchedulingStrategy l2_strategy =
                L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY;
            switch (strategy) {
                case DataflowStrategy::WEIGHT_STATIONARY:
                    l2_strategy = L2TileScheduler::SchedulingStrategy::WEIGHT_STATIONARY;
                    break;
                case DataflowStrategy::INPUT_STATIONARY:
                    l2_strategy = L2TileScheduler::SchedulingStrategy::INPUT_STATIONARY;
                    break;
                case DataflowStrategy::OUTPUT_STATIONARY:
                    l2_strategy = L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY;
                    break;
            }

            // Generate L2 schedule with the specific strategy
            auto l2_schedule = l2_scheduler.generate_schedule(wl.M, wl.N, wl.K, tile_config,
                                                              L2TileScheduler::ReplacementPolicy::LRU,
                                                              l2_strategy);

            for (Size l3_size : l3_sizes) {
                L3Config l3_config;
                l3_config.l3_capacity = l3_size;
                L3Scheduler l3_scheduler(l3_config);

                TensorShape shape(wl.M, wl.N, wl.K);
                auto l3_sched = l3_scheduler.schedule_l3(shape, l2_schedule);

                // Write to CSV
                csv << wl.name << ","
                    << wl.category << ","
                    << wl.aspect << ","
                    << wl.M << "," << wl.N << "," << wl.K << ","
                    << strategy_name << ","
                    << (l3_size / (1024 * 1024)) << ","
                    << l3_sched.overfetch_factor_total << ","
                    << l3_sched.overfetch_factor_a << ","
                    << l3_sched.overfetch_factor_b << ","
                    << l3_sched.overfetch_factor_c << ","
                    << l3_sched.l3_hit_rate << ","
                    << l3_sched.l3_utilization << ","
                    << (l3_sched.total_dram_reads / (1024 * 1024)) << ","
                    << l3_sched.num_l3_evictions << ","
                    << (l3_sched.tensor_a_size / (1024 * 1024)) << ","
                    << (l3_sched.tensor_b_size / (1024 * 1024)) << ","
                    << (l3_sched.tensor_c_size / (1024 * 1024)) << "\n";

                config_count++;
            }
        }

        std::cout << "  Progress: " << config_count << "/" << total_configs << " complete\n";
    }

    csv.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS COMPLETE                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Completed in " << duration.count() << " seconds\n";
    std::cout << "Analyzed " << config_count << " configurations\n";
    std::cout << "Results: l3_focused_analysis.csv\n\n";

    std::cout << "Key insights to explore in the CSV:\n";
    std::cout << "  1. Group by Aspect → Compare overfetch for Square vs Tall vs Wide vs Deep\n";
    std::cout << "  2. Group by Strategy → Which strategy (WS/IS/OS) is best for each aspect?\n";
    std::cout << "  3. Group by L3_Size → Where's the knee for each workload type?\n";
    std::cout << "  4. Look at per-tensor overfetch (A, B, C) → Which tensor dominates?\n\n";

    std::cout << "Quick analysis commands:\n";
    std::cout << "  # See overfetch by strategy\n";
    std::cout << "  awk -F',' 'NR>1 {sum[$7]+=$9; count[$7]++} END {for(s in sum) print s, sum[s]/count[s]}' l3_focused_analysis.csv\n\n";
    std::cout << "  # See overfetch by aspect ratio\n";
    std::cout << "  awk -F',' 'NR>1 {sum[$3]+=$9; count[$3]++} END {for(a in sum) print a, sum[a]/count[a]}' l3_focused_analysis.csv\n\n";
}

int main() {
    analyze_and_export();
    return 0;
}
