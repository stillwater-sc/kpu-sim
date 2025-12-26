#pragma once
// Benchmark Harness for KPU Simulator
// Provides infrastructure for measuring and reporting performance

#include <sw/kpu/kernel.hpp>
#include <sw/kpu/kernel_graph.hpp>
#include <sw/compiler/kernel_compiler.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>

#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <map>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace sw::benchmark {

using namespace sw::kpu;
using namespace sw::kpu::isa;
using namespace sw::kpu::compiler;

// ============================================================================
// Benchmark Result Types
// ============================================================================

/**
 * @brief Result from a single benchmark run
 */
struct BenchmarkResult {
    std::string name;                    ///< Benchmark name
    std::string config;                  ///< Configuration string (e.g., "1024x1024x1024")

    // Timing
    Cycle cycles = 0;                    ///< Simulated cycles
    double compile_time_us = 0.0;        ///< Compilation time in microseconds
    double wall_time_us = 0.0;           ///< Wall clock time for simulation

    // Compute metrics
    uint64_t flops = 0;                  ///< Total floating point operations
    double gflops = 0.0;                 ///< Achieved GFLOPS (at 1 GHz)
    double gflops_at_freq = 0.0;         ///< Achieved GFLOPS at configured frequency
    double compute_efficiency = 0.0;     ///< Fraction of peak compute

    // Memory metrics
    Size external_bytes = 0;             ///< External memory traffic
    Size l3_bytes = 0;                   ///< L3 traffic
    Size l2_bytes = 0;                   ///< L2 traffic
    double arithmetic_intensity = 0.0;   ///< FLOP per byte from external memory
    double memory_efficiency = 0.0;      ///< Fraction of peak memory bandwidth

    // Instruction counts
    size_t total_instructions = 0;
    size_t dma_ops = 0;
    size_t block_mover_ops = 0;
    size_t streamer_ops = 0;

    // Tile information
    Size Ti = 0, Tj = 0, Tk = 0;
    Size num_tiles = 0;

    // Utilization (from executor)
    double dma_utilization = 0.0;
    double block_mover_utilization = 0.0;
    double streamer_utilization = 0.0;
    double compute_utilization = 0.0;

    /**
     * @brief Check if this result represents compute-bound execution
     */
    bool is_compute_bound() const {
        return arithmetic_intensity > 16.0;  // FLOP/byte threshold
    }

    /**
     * @brief Check if this result represents memory-bound execution
     */
    bool is_memory_bound() const {
        return arithmetic_intensity <= 16.0;
    }

    /**
     * @brief Get the bottleneck description
     */
    std::string bottleneck() const {
        if (is_compute_bound()) {
            return "compute-bound";
        } else if (arithmetic_intensity < 8.0) {
            return "external-memory-bound";
        } else {
            return "L3-memory-bound";
        }
    }

    /**
     * @brief Format result as a single-line summary
     */
    std::string to_string() const;

    /**
     * @brief Format result as detailed multi-line report
     */
    std::string to_detailed_string() const;
};

/**
 * @brief Collection of benchmark results for a sweep
 */
struct BenchmarkSuite {
    std::string name;
    std::string description;
    std::vector<BenchmarkResult> results;

    void add(const BenchmarkResult& result) {
        results.push_back(result);
    }

    /**
     * @brief Generate summary table
     */
    std::string summary_table() const;

    /**
     * @brief Generate CSV output
     */
    std::string to_csv() const;

    /**
     * @brief Find best result by GFLOPS
     */
    const BenchmarkResult* best_by_gflops() const;

    /**
     * @brief Find best result by efficiency
     */
    const BenchmarkResult* best_by_efficiency() const;
};

// ============================================================================
// Hardware Configuration
// ============================================================================

/**
 * @brief Hardware specification for roofline analysis
 */
struct HardwareSpec {
    // Compute
    double peak_gflops = 1024.0;         ///< Peak compute (16x16 @ 2GHz, 2 ops/cycle)
    double clock_ghz = 1.0;              ///< Reference clock for cycle conversion

    // Memory bandwidth (GB/s)
    double external_bw_gbs = 64.0;       ///< External memory bandwidth
    double l3_bw_gbs = 128.0;            ///< L3↔L2 bandwidth
    double l2_bw_gbs = 256.0;            ///< L2↔L1 bandwidth

    // Derived
    double ridge_point_external() const {
        return peak_gflops / external_bw_gbs;  // FLOP/byte
    }

    double ridge_point_l3() const {
        return peak_gflops / l3_bw_gbs;
    }

    /**
     * @brief Calculate roofline-predicted performance
     * @param arithmetic_intensity FLOP per byte
     * @return Predicted GFLOPS
     */
    double roofline_gflops(double arithmetic_intensity) const {
        double memory_limited = arithmetic_intensity * external_bw_gbs;
        return std::min(memory_limited, peak_gflops);
    }

    /**
     * @brief Calculate efficiency given achieved performance
     */
    double efficiency(double achieved_gflops, double arithmetic_intensity) const {
        double predicted = roofline_gflops(arithmetic_intensity);
        return achieved_gflops / predicted;
    }

    static HardwareSpec default_kpu() {
        return HardwareSpec{};
    }
};

// ============================================================================
// Benchmark Harness
// ============================================================================

/**
 * @brief Benchmark harness for running and measuring kernel performance
 */
class BenchmarkHarness {
public:
    explicit BenchmarkHarness(const HardwareSpec& hw = HardwareSpec::default_kpu())
        : hw_spec_(hw) {}

    /**
     * @brief Set hardware specification
     */
    void set_hardware_spec(const HardwareSpec& hw) { hw_spec_ = hw; }
    const HardwareSpec& hardware_spec() const { return hw_spec_; }

    /**
     * @brief Set executor resource configuration
     */
    void set_resource_config(const ResourceConfig& config) { res_config_ = config; }
    const ResourceConfig& resource_config() const { return res_config_; }

    /**
     * @brief Benchmark a single kernel
     * @param kernel The kernel to benchmark
     * @param name Benchmark name
     * @param config Configuration description
     * @return Benchmark result
     */
    BenchmarkResult run(const Kernel& kernel,
                        const std::string& name,
                        const std::string& config);

    /**
     * @brief Benchmark a matmul with given dimensions
     */
    BenchmarkResult benchmark_matmul(Size M, Size N, Size K,
                                      DataType dtype = DataType::FLOAT32);

    /**
     * @brief Benchmark a matmul with specific tile sizes
     */
    BenchmarkResult benchmark_matmul_tiled(Size M, Size N, Size K,
                                            Size Ti, Size Tj, Size Tk,
                                            DataType dtype = DataType::FLOAT32);

    /**
     * @brief Benchmark an MLP layer
     */
    BenchmarkResult benchmark_mlp(Size M, Size N, Size K,
                                   ActivationType activation,
                                   bool has_bias = true,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Benchmark a kernel graph
     */
    BenchmarkResult benchmark_graph(const KernelGraph& graph,
                                     const std::string& name);

    // ========================================================================
    // Sweep Benchmarks
    // ========================================================================

    /**
     * @brief Run matmul sweep across problem sizes
     * @param sizes List of (M, N, K) tuples
     * @return Suite of results
     */
    BenchmarkSuite sweep_matmul_sizes(
        const std::vector<std::tuple<Size, Size, Size>>& sizes);

    /**
     * @brief Run matmul sweep for square matrices
     * @param min_size Minimum dimension
     * @param max_size Maximum dimension
     * @param step Multiplicative step (e.g., 2 for powers of 2)
     */
    BenchmarkSuite sweep_matmul_square(Size min_size, Size max_size, Size step = 2);

    /**
     * @brief Run tile size sensitivity analysis
     * @param M, N, K Problem dimensions
     * @param tile_sizes List of (Ti, Tj, Tk) to test
     */
    BenchmarkSuite sweep_tile_sizes(
        Size M, Size N, Size K,
        const std::vector<std::tuple<Size, Size, Size>>& tile_sizes);

    /**
     * @brief Run activation function comparison
     */
    BenchmarkSuite sweep_activations(Size M, Size N, Size K);

    // ========================================================================
    // Roofline Analysis
    // ========================================================================

    /**
     * @brief Generate roofline data points
     * @param results Benchmark results to plot
     * @return Formatted roofline data (for gnuplot or Python)
     */
    std::string generate_roofline_data(const BenchmarkSuite& results) const;

    /**
     * @brief Generate gnuplot script for roofline
     */
    std::string generate_roofline_gnuplot(const BenchmarkSuite& results) const;

private:
    HardwareSpec hw_spec_;
    ResourceConfig res_config_;
    KernelCompiler compiler_;

    void populate_result(BenchmarkResult& result,
                         const Kernel& kernel,
                         const CompilationStats& stats,
                         Cycle cycles,
                         const ConcurrentExecutor::UtilizationStats& util);
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Run quick matmul benchmark
 */
inline BenchmarkResult quick_benchmark_matmul(Size M, Size N, Size K) {
    BenchmarkHarness harness;
    return harness.benchmark_matmul(M, N, K);
}

/**
 * @brief Run standard matmul sweep (powers of 2 from 64 to 4096)
 */
inline BenchmarkSuite standard_matmul_sweep() {
    BenchmarkHarness harness;
    return harness.sweep_matmul_square(64, 4096, 2);
}

// ============================================================================
// Implementation
// ============================================================================

inline std::string BenchmarkResult::to_string() const {
    std::ostringstream ss;
    ss << std::setw(20) << name << " "
       << std::setw(15) << config << " "
       << std::setw(12) << cycles << " cyc "
       << std::setw(8) << std::fixed << std::setprecision(2) << gflops << " GFLOPS "
       << std::setw(6) << std::fixed << std::setprecision(1) << (compute_efficiency * 100) << "% "
       << std::setw(6) << std::fixed << std::setprecision(2) << arithmetic_intensity << " AI "
       << "[" << bottleneck() << "]";
    return ss.str();
}

inline std::string BenchmarkResult::to_detailed_string() const {
    std::ostringstream ss;
    ss << "Benchmark: " << name << " (" << config << ")\n";
    ss << std::string(60, '-') << "\n";

    ss << "Timing:\n";
    ss << "  Cycles:           " << cycles << "\n";
    ss << "  Compile time:     " << std::fixed << std::setprecision(1)
       << compile_time_us << " us\n";
    ss << "  Wall time:        " << std::fixed << std::setprecision(1)
       << wall_time_us << " us\n\n";

    ss << "Compute:\n";
    ss << "  FLOPs:            " << flops << "\n";
    ss << "  GFLOPS:           " << std::fixed << std::setprecision(2) << gflops << "\n";
    ss << "  Efficiency:       " << std::fixed << std::setprecision(1)
       << (compute_efficiency * 100) << "%\n\n";

    ss << "Memory:\n";
    ss << "  External bytes:   " << external_bytes << " ("
       << (external_bytes / 1024.0) << " KB)\n";
    ss << "  Arith. Intensity: " << std::fixed << std::setprecision(2)
       << arithmetic_intensity << " FLOP/byte\n";
    ss << "  Bottleneck:       " << bottleneck() << "\n\n";

    ss << "Tiling:\n";
    ss << "  Tile sizes:       " << Ti << "x" << Tj << "x" << Tk << "\n";
    ss << "  Total tiles:      " << num_tiles << "\n\n";

    ss << "Instructions:\n";
    ss << "  Total:            " << total_instructions << "\n";
    ss << "  DMA ops:          " << dma_ops << "\n";
    ss << "  Block mover ops:  " << block_mover_ops << "\n";
    ss << "  Streamer ops:     " << streamer_ops << "\n\n";

    ss << "Utilization:\n";
    ss << "  DMA:              " << std::fixed << std::setprecision(1)
       << (dma_utilization * 100) << "%\n";
    ss << "  Block Mover:      " << std::fixed << std::setprecision(1)
       << (block_mover_utilization * 100) << "%\n";
    ss << "  Streamer:         " << std::fixed << std::setprecision(1)
       << (streamer_utilization * 100) << "%\n";
    ss << "  Compute:          " << std::fixed << std::setprecision(1)
       << (compute_utilization * 100) << "%\n";

    return ss.str();
}

inline std::string BenchmarkSuite::summary_table() const {
    std::ostringstream ss;
    ss << "Benchmark Suite: " << name << "\n";
    if (!description.empty()) {
        ss << description << "\n";
    }
    ss << std::string(100, '=') << "\n";
    ss << std::setw(20) << "Name" << " "
       << std::setw(15) << "Config" << " "
       << std::setw(12) << "Cycles" << " "
       << std::setw(10) << "GFLOPS" << " "
       << std::setw(8) << "Eff%" << " "
       << std::setw(8) << "AI" << " "
       << "Bottleneck\n";
    ss << std::string(100, '-') << "\n";

    for (const auto& r : results) {
        ss << std::setw(20) << r.name << " "
           << std::setw(15) << r.config << " "
           << std::setw(12) << r.cycles << " "
           << std::setw(10) << std::fixed << std::setprecision(2) << r.gflops << " "
           << std::setw(7) << std::fixed << std::setprecision(1) << (r.compute_efficiency * 100) << "% "
           << std::setw(8) << std::fixed << std::setprecision(2) << r.arithmetic_intensity << " "
           << r.bottleneck() << "\n";
    }
    ss << std::string(100, '=') << "\n";
    return ss.str();
}

inline std::string BenchmarkSuite::to_csv() const {
    std::ostringstream ss;
    // Header
    ss << "name,config,cycles,flops,gflops,efficiency,external_bytes,arithmetic_intensity,"
       << "Ti,Tj,Tk,num_tiles,dma_ops,bm_ops,str_ops,dma_util,bm_util,str_util,compute_util\n";

    for (const auto& r : results) {
        ss << r.name << ","
           << r.config << ","
           << r.cycles << ","
           << r.flops << ","
           << r.gflops << ","
           << r.compute_efficiency << ","
           << r.external_bytes << ","
           << r.arithmetic_intensity << ","
           << r.Ti << "," << r.Tj << "," << r.Tk << ","
           << r.num_tiles << ","
           << r.dma_ops << ","
           << r.block_mover_ops << ","
           << r.streamer_ops << ","
           << r.dma_utilization << ","
           << r.block_mover_utilization << ","
           << r.streamer_utilization << ","
           << r.compute_utilization << "\n";
    }
    return ss.str();
}

inline const BenchmarkResult* BenchmarkSuite::best_by_gflops() const {
    if (results.empty()) return nullptr;
    const BenchmarkResult* best = &results[0];
    for (const auto& r : results) {
        if (r.gflops > best->gflops) {
            best = &r;
        }
    }
    return best;
}

inline const BenchmarkResult* BenchmarkSuite::best_by_efficiency() const {
    if (results.empty()) return nullptr;
    const BenchmarkResult* best = &results[0];
    for (const auto& r : results) {
        if (r.compute_efficiency > best->compute_efficiency) {
            best = &r;
        }
    }
    return best;
}

} // namespace sw::benchmark
