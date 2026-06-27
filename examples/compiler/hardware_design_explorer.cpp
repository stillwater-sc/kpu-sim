/**
 * @file hardware_design_explorer.cpp
 * @brief Hardware Design Space Exploration Tool
 *
 * This tool explores the hardware design space by varying L2 cache size and
 * measuring the impact on energy and latency across representative workloads.
 *
 * Key Question: How much L2 cache should we provision in the hardware?
 *
 * Approach:
 * 1. Sweep L2 cache size from small (16KB) to large (256KB)
 * 2. For each L2 size, characterize performance across workloads
 * 3. Aggregate results: total energy, average latency, workload coverage
 * 4. Identify optimal L2 size based on energy-performance-cost tradeoffs
 *
 * This reveals the FUNDAMENTAL HARDWARE DESIGN TRADEOFF:
 * - Larger L2 → Lower energy (better reuse) but higher hardware cost
 * - Smaller L2 → Higher energy (poor reuse) but lower hardware cost
 */

#include <sw/compiler/schedule_characterizer.hpp>
#include <sw/concepts.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <fstream>

using namespace sw::kpu::compiler;
using sw::kpu::Cycle;
using sw::kpu::Size;

/**
 * @brief Hardware configuration for design space exploration
 */
struct HardwareConfig {
    Size l2_size_per_bank;      ///< L2 cache size per bank (bytes)
    Size l2_banks;              ///< Number of L2 banks
    Size pe_array_rows;         ///< Systolic array rows
    Size pe_array_cols;         ///< Systolic array columns
    Size pe_register_capacity;  ///< PE register file capacity (bytes)

    Size total_l2_size() const {
        return l2_size_per_bank * l2_banks;
    }

    std::string to_string() const {
        return "L2=" + std::to_string(l2_size_per_bank / 1024) + "KB×" +
               std::to_string(l2_banks) + " PE=" +
               std::to_string(pe_array_rows) + "×" + std::to_string(pe_array_cols);
    }
};

/**
 * @brief Aggregate results for a hardware configuration across workloads
 */
struct HardwareConfigResults {
    HardwareConfig config;

    // Aggregate metrics across all workloads
    double total_energy;        ///< Sum of energy across all workloads (pJ)
    Cycle total_latency;        ///< Sum of latency across all workloads (cycles)
    double avg_energy;          ///< Average energy per workload (pJ)
    Cycle avg_latency;          ///< Average latency per workload (cycles)
    double avg_throughput;      ///< Average throughput (GFLOP/s)

    // Efficiency metrics
    double avg_energy_slowdown;    ///< Average energy vs ideal
    double avg_latency_slowdown;   ///< Average latency vs ideal
    double avg_utilization;        ///< Average PE utilization

    // Coverage metrics
    Size num_workloads_evaluated;   ///< Number of workloads evaluated
    Size num_workloads_feasible;    ///< Number of workloads that fit in L2

    // Cost model (normalized, L2=64KB → cost=1.0)
    double hardware_cost;           ///< Relative hardware cost

    HardwareConfigResults()
        : total_energy(0), total_latency(0)
        , avg_energy(0), avg_latency(0), avg_throughput(0)
        , avg_energy_slowdown(0), avg_latency_slowdown(0), avg_utilization(0)
        , num_workloads_evaluated(0), num_workloads_feasible(0)
        , hardware_cost(1.0) {}
};

/**
 * @brief Simple cost model for hardware configuration
 *
 * Assumes cost is dominated by on-chip SRAM (L2 cache)
 * Normalized to L2=64KB → cost=1.0
 */
double estimate_hardware_cost(const HardwareConfig& config) {
    // Cost scales with total L2 size (approximately linear for SRAM)
    const double baseline_l2 = 64.0 * 1024.0;  // 64KB baseline
    return static_cast<double>(config.total_l2_size()) / baseline_l2;
}

/**
 * @brief Create memory hierarchy from hardware config
 */
TileOptimizer::MemoryHierarchy create_memory_hierarchy(const HardwareConfig& config) {
    TileOptimizer::MemoryHierarchy mem;

    // Map our config to TileOptimizer's MemoryHierarchy
    mem.L2_size = config.l2_size_per_bank;
    mem.L2_bank_count = config.l2_banks;
    mem.systolic_rows = config.pe_array_rows;
    mem.systolic_cols = config.pe_array_cols;
    // PE register capacity is implicit in the systolic array (not in MemoryHierarchy)

    return mem;
}

/**
 * @brief Evaluate a hardware configuration across workloads
 */
HardwareConfigResults evaluate_hardware_config(
    const HardwareConfig& config,
    const std::vector<TensorShape>& workloads,
    const EnergyModel& energy_model,
    const LatencyModel& latency_model)
{
    HardwareConfigResults results;
    results.config = config;
    results.num_workloads_evaluated = workloads.size();
    results.hardware_cost = estimate_hardware_cost(config);

    // Create characterizer with this hardware config
    auto mem = create_memory_hierarchy(config);
    ScheduleCharacterizer characterizer(mem, energy_model, latency_model);

    // Evaluate all workloads with all strategies
    std::vector<ScheduleEvaluation> all_evals;

    for (const auto& shape : workloads) {
        auto evals = characterizer.evaluate_all_strategies(shape);

        // Check if any strategy can fit in this L2 size
        bool any_feasible = false;
        for (const auto& eval : evals) {
            if (eval.tile_config.l2_footprint <= config.l2_size_per_bank) {
                any_feasible = true;
                break;
            }
        }

        if (any_feasible) {
            results.num_workloads_feasible++;
        }

        all_evals.insert(all_evals.end(), evals.begin(), evals.end());
    }

    // Aggregate metrics
    double sum_energy = 0;
    Cycle sum_latency = 0;
    double sum_throughput = 0;
    double sum_energy_slowdown = 0;
    double sum_latency_slowdown = 0;
    double sum_utilization = 0;

    for (const auto& eval : all_evals) {
        sum_energy += eval.metrics.total_energy;
        sum_latency += eval.metrics.total_cycles;
        sum_throughput += eval.metrics.throughput_gflops;
        sum_energy_slowdown += eval.energy_slowdown;
        sum_latency_slowdown += eval.latency_slowdown;
        sum_utilization += eval.metrics.utilization;
    }

    results.total_energy = sum_energy;
    results.total_latency = sum_latency;
    results.avg_energy = sum_energy / static_cast<double>(all_evals.size());
    results.avg_latency = sum_latency / all_evals.size();
    results.avg_throughput = sum_throughput / static_cast<double>(all_evals.size());
    results.avg_energy_slowdown = sum_energy_slowdown / static_cast<double>(all_evals.size());
    results.avg_latency_slowdown = sum_latency_slowdown / static_cast<double>(all_evals.size());
    results.avg_utilization = sum_utilization / static_cast<double>(all_evals.size());

    return results;
}

/**
 * @brief Generate L2 size sweep configurations
 */
std::vector<HardwareConfig> generate_l2_sweep() {
    std::vector<HardwareConfig> configs;

    // Baseline config (systolic array size, PE capacity fixed)
    const Size pe_rows = 8;
    const Size pe_cols = 8;
    const Size pe_capacity = 32 * 1024;  // 32KB
    const Size l2_banks = 1;

    // Sweep L2 size from 16KB to 256KB
    std::vector<Size> l2_sizes = {
        16 * 1024,   // 16KB - very small
        32 * 1024,   // 32KB - small
        64 * 1024,   // 64KB - baseline
        96 * 1024,   // 96KB - medium
        128 * 1024,  // 128KB - large
        192 * 1024,  // 192KB - very large
        256 * 1024   // 256KB - huge
    };

    for (Size l2_size : l2_sizes) {
        HardwareConfig config;
        config.l2_size_per_bank = l2_size;
        config.l2_banks = l2_banks;
        config.pe_array_rows = pe_rows;
        config.pe_array_cols = pe_cols;
        config.pe_register_capacity = pe_capacity;
        configs.push_back(config);
    }

    return configs;
}

/**
 * @brief Export hardware design space results to CSV
 */
void export_design_space_csv(
    const std::vector<HardwareConfigResults>& results,
    const std::string& filename)
{
    std::ofstream file(filename);

    file << "L2_Size_KB,L2_Banks,Total_L2_KB,PE_Rows,PE_Cols,"
         << "Total_Energy_pJ,Avg_Energy_pJ,Total_Latency_cycles,Avg_Latency_cycles,"
         << "Avg_Throughput_GFLOPS,Avg_Energy_Slowdown,Avg_Latency_Slowdown,"
         << "Avg_Utilization,Num_Workloads,Num_Feasible,Coverage_Pct,"
         << "Hardware_Cost,Energy_Per_Cost\n";

    for (const auto& result : results) {
        double coverage_pct = 100.0 * static_cast<double>(result.num_workloads_feasible) / static_cast<double>(result.num_workloads_evaluated);
        double energy_per_cost = result.avg_energy / result.hardware_cost;

        file << (result.config.l2_size_per_bank / 1024) << ","
             << result.config.l2_banks << ","
             << (result.config.total_l2_size() / 1024) << ","
             << result.config.pe_array_rows << ","
             << result.config.pe_array_cols << ","
             << result.total_energy << ","
             << result.avg_energy << ","
             << result.total_latency << ","
             << result.avg_latency << ","
             << result.avg_throughput << ","
             << result.avg_energy_slowdown << ","
             << result.avg_latency_slowdown << ","
             << result.avg_utilization << ","
             << result.num_workloads_evaluated << ","
             << result.num_workloads_feasible << ","
             << coverage_pct << ","
             << result.hardware_cost << ","
             << energy_per_cost << "\n";
    }

    file.close();
    std::cout << "Exported design space to " << filename << "\n";
}

/**
 * @brief Print hardware design space summary
 */
void print_design_space_summary(const std::vector<HardwareConfigResults>& results) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           HARDWARE DESIGN SPACE EXPLORATION RESULTS            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Evaluated " << results.size() << " hardware configurations\n";
    std::cout << "Workloads per configuration: " << results[0].num_workloads_evaluated << "\n\n";

    // Table header
    std::cout << "┌──────────┬──────────┬──────────────┬──────────────┬──────────┬──────────┬──────────┐\n";
    std::cout << "│ L2 Size  │ HW Cost  │  Avg Energy  │ Avg Latency  │   Util   │ Coverage │ Energy/$ │\n";
    std::cout << "│   (KB)   │ (norm)   │     (pJ)     │   (cycles)   │   (%)    │   (%)    │ (norm)   │\n";
    std::cout << "├──────────┼──────────┼──────────────┼──────────────┼──────────┼──────────┼──────────┤\n";

    for (const auto& result : results) {
        double coverage_pct = 100.0 * static_cast<double>(result.num_workloads_feasible) / static_cast<double>(result.num_workloads_evaluated);
        double energy_per_cost = result.avg_energy / result.hardware_cost;

        std::cout << "│ " << std::setw(8) << (result.config.l2_size_per_bank / 1024)
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << result.hardware_cost
                  << " │ " << std::setw(12) << std::scientific << std::setprecision(2) << result.avg_energy
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(0) << result.avg_latency
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1) << (result.avg_utilization * 100.0)
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1) << coverage_pct
                  << " │ " << std::setw(8) << std::scientific << std::setprecision(2) << energy_per_cost
                  << " │\n";
    }

    std::cout << "└──────────┴──────────┴──────────────┴──────────────┴──────────┴──────────┴──────────┘\n\n";

    // Find optimal configurations
    auto min_energy_it = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.avg_energy < b.avg_energy; });

    auto min_latency_it = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.avg_latency < b.avg_latency; });

    auto best_energy_per_cost_it = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return (a.avg_energy / a.hardware_cost) < (b.avg_energy / b.hardware_cost);
        });

    std::cout << "Optimal Configurations:\n";
    std::cout << "  Minimum Energy:         "
              << (min_energy_it->config.l2_size_per_bank / 1024) << " KB L2 "
              << "(" << std::scientific << min_energy_it->avg_energy << " pJ)\n";
    std::cout << "  Minimum Latency:        "
              << (min_latency_it->config.l2_size_per_bank / 1024) << " KB L2 "
              << "(" << min_latency_it->avg_latency << " cycles)\n";
    std::cout << "  Best Energy/Cost:       "
              << (best_energy_per_cost_it->config.l2_size_per_bank / 1024) << " KB L2 "
              << "(ratio: " << std::scientific
              << (best_energy_per_cost_it->avg_energy / best_energy_per_cost_it->hardware_cost) << ")\n";

    std::cout << "\n";
}

int main() {
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█                 HARDWARE DESIGN SPACE EXPLORATION                        █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  Explores L2 cache size impact on energy and latency across workloads   █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n\n";

    // Generate workloads for evaluation
    std::cout << "Generating representative workloads...\n";
    auto workloads = WorkloadGenerator::generate_ml_workloads(50);
    std::cout << "Generated " << workloads.size() << " workloads\n\n";

    // Generate hardware configurations to explore
    std::cout << "Generating hardware configurations...\n";
    auto configs = generate_l2_sweep();
    std::cout << "Generated " << configs.size() << " L2 cache size configurations\n";
    std::cout << "L2 range: " << (configs.front().l2_size_per_bank / 1024) << " KB to "
              << (configs.back().l2_size_per_bank / 1024) << " KB\n\n";

    // Energy and latency models (same for all configs)
    EnergyModel energy_model;
    LatencyModel latency_model;

    // Evaluate each hardware configuration
    std::vector<HardwareConfigResults> results;
    results.reserve(configs.size());

    std::cout << "Evaluating hardware configurations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "  [" << (i + 1) << "/" << configs.size() << "] "
                  << "L2 = " << (configs[i].l2_size_per_bank / 1024) << " KB...";
        std::cout.flush();

        auto config_results = evaluate_hardware_config(
            configs[i], workloads, energy_model, latency_model);

        results.push_back(config_results);

        std::cout << " Done (Coverage: " << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(config_results.num_workloads_feasible) / static_cast<double>(config_results.num_workloads_evaluated))
                  << "%)\n";
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nExploration time: " << duration.count() << " ms\n";

    // Print summary
    print_design_space_summary(results);

    // Export results
    export_design_space_csv(results, "hardware_design_space.csv");

    std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    EXPLORATION COMPLETE                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Next steps:\n";
    std::cout << "  1. Visualize: python ../tools/compiler/visualize_hardware_design_space.py\n";
    std::cout << "  2. Analyze tradeoffs between L2 size, energy, and hardware cost\n";
    std::cout << "  3. Choose L2 size based on application requirements\n\n";

    return 0;
}
