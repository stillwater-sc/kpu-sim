/**
 * @file l3_comprehensive_analysis.cpp
 * @brief Comprehensive L3 Overfetch Analysis
 *
 * This tool performs a thorough exploration of L3 cache behavior across:
 * 1. Non-square matrices (realistic transformer shapes)
 * 2. All dataflow strategies (WS, IS, OS)
 * 3. Various L3 cache sizes
 * 4. Different tensor aspect ratios
 *
 * The goal: Understand how dataflow strategy and tensor shape affect
 * L3 requirements and DRAM overfetch.
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
#include <map>

using namespace sw::kpu::compiler;
using sw::kpu::Cycle;
using sw::kpu::Size;

/**
 * @brief Comprehensive workload specification
 */
struct ComprehensiveWorkload {
    std::string name;
    TensorShape shape;
    std::string category;  // e.g., "Attention", "MLP", "Projection"
    std::string aspect;    // e.g., "Square", "Tall", "Wide", "Extreme"

    ComprehensiveWorkload() = default;

    ComprehensiveWorkload(const std::string& n, Size m, Size n_dim, Size k,
                         const std::string& cat, const std::string& asp)
        : name(n), shape(m, n_dim, k), category(cat), aspect(asp) {}
};

/**
 * @brief Results for one configuration
 */
struct AnalysisResult {
    ComprehensiveWorkload workload;
    DataflowStrategy strategy;
    Size l3_size_mb;

    // L3 metrics
    double overfetch_total;
    double overfetch_a;
    double overfetch_b;
    double overfetch_c;
    double l3_hit_rate;
    double l3_utilization;
    Size dram_reads_mb;
    Size num_evictions;

    // Tensor sizes
    Size tensor_a_mb;
    Size tensor_b_mb;
    Size tensor_c_mb;
};

/**
 * @brief Generate comprehensive non-square workload suite
 */
std::vector<ComprehensiveWorkload> generate_comprehensive_workloads() {
    std::vector<ComprehensiveWorkload> workloads;

    // ========================================================================
    // 1. SQUARE MATRICES (baseline)
    // ========================================================================
    workloads.emplace_back("Small Square", 512, 512, 512, "Baseline", "Square");
    workloads.emplace_back("Medium Square", 2048, 2048, 2048, "Baseline", "Square");
    workloads.emplace_back("Large Square", 8192, 8192, 8192, "Baseline", "Square");

    // ========================================================================
    // 2. TALL MATRICES (M >> N, M >> K) - Common in batched inference
    // ========================================================================
    workloads.emplace_back("Tall: Long Context Q×K^T", 32768, 128, 128, "Attention", "Tall");
    workloads.emplace_back("Tall: Batch Projection", 16384, 512, 512, "Projection", "Tall");
    workloads.emplace_back("Tall: Extreme Batch", 65536, 256, 256, "Projection", "Tall");

    // ========================================================================
    // 3. WIDE MATRICES (N >> M, K >> M) - Common in vocabulary projections
    // ========================================================================
    workloads.emplace_back("Wide: Vocab Projection", 128, 32768, 4096, "Projection", "Wide");
    workloads.emplace_back("Wide: Hidden→Vocab", 256, 50257, 768, "Projection", "Wide");
    workloads.emplace_back("Wide: Embedding Lookup", 512, 100000, 1024, "Embedding", "Wide");

    // ========================================================================
    // 4. DEEP MATRICES (K >> M, K >> N) - Inner dimension large
    // ========================================================================
    workloads.emplace_back("Deep: MLP Up-projection", 1024, 1024, 16384, "MLP", "Deep");
    workloads.emplace_back("Deep: Extreme MLP", 512, 512, 32768, "MLP", "Deep");

    // ========================================================================
    // 5. REALISTIC TRANSFORMER LAYERS
    // ========================================================================

    // BERT-style (moderate size, balanced)
    workloads.emplace_back("BERT Q×K^T (seq=512)", 512, 512, 768, "Attention", "Balanced");
    workloads.emplace_back("BERT Q×K^T (seq=4096)", 4096, 4096, 768, "Attention", "Tall-ish");
    workloads.emplace_back("BERT MLP", 512, 768, 3072, "MLP", "Deep-ish");

    // GPT-style (larger hidden dim)
    workloads.emplace_back("GPT-2 Q×K^T", 2048, 2048, 1536, "Attention", "Square-ish");
    workloads.emplace_back("GPT-2 MLP", 2048, 1536, 6144, "MLP", "Deep");
    workloads.emplace_back("GPT-3 Q×K^T", 2048, 2048, 12288, "Attention", "Deep-ish");
    workloads.emplace_back("GPT-3 MLP", 2048, 12288, 49152, "MLP", "Very Deep");

    // LLaMA-style (7B model)
    workloads.emplace_back("LLaMA-2 7B Q×K^T", 4096, 4096, 4096, "Attention", "Square");
    workloads.emplace_back("LLaMA-2 7B MLP", 4096, 4096, 11008, "MLP", "Deep");
    workloads.emplace_back("LLaMA-2 7B Long Ctx", 16384, 16384, 4096, "Attention", "Tall-ish");

    // LLaMA-style (70B model)
    workloads.emplace_back("LLaMA-2 70B Q×K^T", 8192, 8192, 8192, "Attention", "Square");
    workloads.emplace_back("LLaMA-2 70B MLP", 8192, 8192, 28672, "MLP", "Very Deep");
    workloads.emplace_back("LLaMA-2 70B Long Ctx", 32768, 32768, 8192, "Attention", "Tall-ish");

    // ========================================================================
    // 6. EXTREME ASPECT RATIOS (stress test)
    // ========================================================================
    workloads.emplace_back("Extreme Wide: 100k vocab", 64, 100000, 512, "Projection", "Extreme Wide");
    workloads.emplace_back("Extreme Tall: 64k batch", 65536, 64, 768, "Projection", "Extreme Tall");
    workloads.emplace_back("Extreme Deep: 64k hidden", 256, 256, 65536, "MLP", "Extreme Deep");

    return workloads;
}

/**
 * @brief Analyze one workload with all strategies and L3 sizes
 */
std::vector<AnalysisResult> analyze_workload(
    const ComprehensiveWorkload& workload,
    const std::vector<Size>& l3_sizes,
    TileOptimizer& optimizer,
    L2TileScheduler& l2_scheduler)
{
    std::vector<AnalysisResult> results;

    // Strategies to test
    std::vector<DataflowStrategy> strategies = {
        DataflowStrategy::WEIGHT_STATIONARY,
        DataflowStrategy::INPUT_STATIONARY,
        DataflowStrategy::OUTPUT_STATIONARY
    };

    for (auto strategy : strategies) {
        // Get tile config
        auto tile_config = optimizer.optimize(workload.shape.M, workload.shape.N, workload.shape.K);

        // Map DataflowStrategy to L2TileScheduler::SchedulingStrategy.
        // The switch covers all three enum values but gcc -O3 can't tell;
        // initialize explicitly to silence -Wmaybe-uninitialized.
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
        auto l2_schedule = l2_scheduler.generate_schedule(
            workload.shape.M, workload.shape.N, workload.shape.K, tile_config,
            L2TileScheduler::ReplacementPolicy::LRU,
            l2_strategy);

        for (Size l3_size : l3_sizes) {
            L3Config l3_config;
            l3_config.l3_capacity = l3_size;
            L3Scheduler l3_scheduler(l3_config);

            auto l3_schedule = l3_scheduler.schedule_l3(workload.shape, l2_schedule);

            AnalysisResult result;
            result.workload = workload;
            result.strategy = strategy;
            result.l3_size_mb = l3_size / (1024 * 1024);
            result.overfetch_total = l3_schedule.overfetch_factor_total;
            result.overfetch_a = l3_schedule.overfetch_factor_a;
            result.overfetch_b = l3_schedule.overfetch_factor_b;
            result.overfetch_c = l3_schedule.overfetch_factor_c;
            result.l3_hit_rate = l3_schedule.l3_hit_rate;
            result.l3_utilization = l3_schedule.l3_utilization;
            result.dram_reads_mb = l3_schedule.total_dram_reads / (1024 * 1024);
            result.num_evictions = l3_schedule.num_l3_evictions;
            result.tensor_a_mb = l3_schedule.tensor_a_size / (1024 * 1024);
            result.tensor_b_mb = l3_schedule.tensor_b_size / (1024 * 1024);
            result.tensor_c_mb = l3_schedule.tensor_c_size / (1024 * 1024);

            results.push_back(result);
        }
    }

    return results;
}

/**
 * @brief Export results to CSV
 */
void export_comprehensive_csv(
    const std::vector<AnalysisResult>& results,
    const std::string& filename)
{
    std::ofstream file(filename);

    file << "Workload_Name,Category,Aspect,M,N,K,Strategy,"
         << "L3_Size_MB,Overfetch_Total,Overfetch_A,Overfetch_B,Overfetch_C,"
         << "L3_Hit_Rate,L3_Utilization,DRAM_Reads_MB,Num_Evictions,"
         << "Tensor_A_MB,Tensor_B_MB,Tensor_C_MB,Total_Tensor_MB\n";

    for (const auto& r : results) {
        std::string strategy_name;
        switch (r.strategy) {
            case DataflowStrategy::WEIGHT_STATIONARY: strategy_name = "WS"; break;
            case DataflowStrategy::INPUT_STATIONARY: strategy_name = "IS"; break;
            case DataflowStrategy::OUTPUT_STATIONARY: strategy_name = "OS"; break;
        }

        file << r.workload.name << ","
             << r.workload.category << ","
             << r.workload.aspect << ","
             << r.workload.shape.M << ","
             << r.workload.shape.N << ","
             << r.workload.shape.K << ","
             << strategy_name << ","
             << r.l3_size_mb << ","
             << r.overfetch_total << ","
             << r.overfetch_a << ","
             << r.overfetch_b << ","
             << r.overfetch_c << ","
             << r.l3_hit_rate << ","
             << r.l3_utilization << ","
             << r.dram_reads_mb << ","
             << r.num_evictions << ","
             << r.tensor_a_mb << ","
             << r.tensor_b_mb << ","
             << r.tensor_c_mb << ","
             << (r.tensor_a_mb + r.tensor_b_mb + r.tensor_c_mb) << "\n";
    }

    file.close();
    std::cout << "Exported " << results.size() << " results to " << filename << "\n";
}

/**
 * @brief Print summary statistics
 */
void print_summary_stats(const std::vector<AnalysisResult>& results) {
    // Group by aspect ratio
    std::map<std::string, std::vector<AnalysisResult>> by_aspect;
    for (const auto& r : results) {
        by_aspect[r.workload.aspect].push_back(r);
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              SUMMARY BY TENSOR ASPECT RATIO                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    for (const auto& [aspect, aspect_results] : by_aspect) {
        double avg_overfetch = 0;
        double max_overfetch = 0;
        double min_overfetch = 1e9;

        for (const auto& r : aspect_results) {
            avg_overfetch += r.overfetch_total;
            max_overfetch = std::max(max_overfetch, r.overfetch_total);
            min_overfetch = std::min(min_overfetch, r.overfetch_total);
        }
        avg_overfetch /= static_cast<double>(aspect_results.size());

        std::cout << aspect << ":\n";
        std::cout << "  Workloads: " << (aspect_results.size() / 3 / 5) << "\n";  // Divide by strategies and L3 sizes
        std::cout << "  Overfetch - Min: " << std::fixed << std::setprecision(2) << min_overfetch
                  << "×, Max: " << max_overfetch << "×, Avg: " << avg_overfetch << "×\n\n";
    }

    // Group by strategy
    std::map<std::string, std::vector<AnalysisResult>> by_strategy;
    for (const auto& r : results) {
        std::string strategy_name;
        switch (r.strategy) {
            case DataflowStrategy::WEIGHT_STATIONARY: strategy_name = "WS"; break;
            case DataflowStrategy::INPUT_STATIONARY: strategy_name = "IS"; break;
            case DataflowStrategy::OUTPUT_STATIONARY: strategy_name = "OS"; break;
        }
        by_strategy[strategy_name].push_back(r);
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              SUMMARY BY DATAFLOW STRATEGY                          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    for (const auto& [strategy, strategy_results] : by_strategy) {
        double avg_overfetch = 0;
        for (const auto& r : strategy_results) {
            avg_overfetch += r.overfetch_total;
        }
        avg_overfetch /= static_cast<double>(strategy_results.size());

        std::cout << strategy << ": Average overfetch = " << std::fixed << std::setprecision(2)
                  << avg_overfetch << "× across all workloads\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█            COMPREHENSIVE L3 OVERFETCH ANALYSIS                           █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  Explores: Non-square matrices × Dataflow strategies × L3 sizes         █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n\n";

    // Generate comprehensive workload suite
    auto workloads = generate_comprehensive_workloads();
    std::cout << "Generated " << workloads.size() << " diverse workloads\n";
    std::cout << "  - Square, Tall, Wide, Deep, and Extreme aspect ratios\n";
    std::cout << "  - Realistic transformer layers (BERT, GPT, LLaMA)\n";
    std::cout << "  - Size range: 64×64 to 100k×100k\n\n";

    // L3 sizes to test (including small distributed L3 tiles)
    std::vector<Size> l3_sizes = {
        1 * 1024 * 1024,     // 1MB (small distributed L3)
        2 * 1024 * 1024,     // 2MB (small distributed L3)
        4 * 1024 * 1024,     // 4MB
        16 * 1024 * 1024,    // 16MB
        64 * 1024 * 1024,    // 64MB
        256 * 1024 * 1024,   // 256MB
        1024 * 1024 * 1024   // 1GB
    };

    std::cout << "Testing " << l3_sizes.size() << " L3 cache sizes: 1MB, 2MB, 4MB, 16MB, 64MB, 256MB, 1GB\n";
    std::cout << "Testing 3 dataflow strategies: WS, IS, OS\n\n";

    Size total_configs = workloads.size() * l3_sizes.size() * 3;
    std::cout << "Total configurations to analyze: " << total_configs << "\n\n";

    // Setup
    TileOptimizer::MemoryHierarchy mem;
    TileOptimizer optimizer(mem);
    L2TileScheduler l2_scheduler;

    // Analyze all workloads
    std::vector<AnalysisResult> all_results;
    all_results.reserve(total_configs);

    std::cout << "Starting analysis...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < workloads.size(); ++i) {
        const auto& workload = workloads[i];

        if (i % 5 == 0) {
            std::cout << "  Progress: " << i << "/" << workloads.size()
                      << " workloads completed...\n";
        }

        auto workload_results = analyze_workload(workload, l3_sizes, optimizer, l2_scheduler);
        all_results.insert(all_results.end(), workload_results.begin(), workload_results.end());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "\nAnalysis complete in " << duration.count() << " seconds\n";
    std::cout << "Analyzed " << all_results.size() << " configurations\n\n";

    // Print summary statistics
    print_summary_stats(all_results);

    // Export to CSV
    export_comprehensive_csv(all_results, "l3_comprehensive_analysis.csv");

    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS COMPLETE                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Next steps:\n";
    std::cout << "  1. Analyze CSV with pandas/Excel to explore multi-dimensional relationships\n";
    std::cout << "  2. Plot overfetch vs aspect ratio for each strategy\n";
    std::cout << "  3. Identify which strategies work best for which tensor shapes\n";
    std::cout << "  4. Find optimal L3 size for different workload categories\n\n";

    return 0;
}
