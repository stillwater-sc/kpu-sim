// ============================================================================
// include/sw/benchmark/noc_benchmark.hpp
// Network-on-Chip Benchmarking Framework
//
// Provides comprehensive benchmarking infrastructure to characterize NoC
// performance for DNN data movement patterns.
// ============================================================================

#pragma once

#include <sw/kpu/models/interfaces/noc_interface.hpp>
#include <sw/kpu/models/temporal/noc/noc_adapters.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sw::benchmark {

using namespace sw::kpu;
using namespace sw::kpu::noc;

// ============================================================================
// Benchmark Result Structures
// ============================================================================

/// Statistics for a benchmark run
struct BenchmarkStats {
    uint64_t min_cycles = 0;
    uint64_t max_cycles = 0;
    uint64_t total_cycles = 0;
    uint64_t samples = 0;

    double mean() const {
        return samples > 0 ? static_cast<double>(total_cycles) / samples : 0.0;
    }

    double variance() const;
    double stddev() const;

    void add_sample(uint64_t cycles) {
        if (samples == 0 || cycles < min_cycles) min_cycles = cycles;
        if (samples == 0 || cycles > max_cycles) max_cycles = cycles;
        total_cycles += cycles;
        samples++;
    }
};

/// Result of a latency benchmark
struct LatencyResult {
    std::string name;
    uint8_t src_router = 0;
    uint8_t dst_router = 0;
    uint8_t hop_count = 0;
    uint32_t tile_size = 0;

    BenchmarkStats stats;

    // Derived metrics
    double cycles_per_hop() const {
        return hop_count > 0 ? stats.mean() / hop_count : 0.0;
    }

    double bytes_per_cycle() const {
        return stats.mean() > 0 ? static_cast<double>(tile_size) / stats.mean() : 0.0;
    }
};

/// Result of a throughput benchmark
struct ThroughputResult {
    std::string name;
    std::string config;

    uint64_t total_bytes = 0;
    uint64_t total_cycles = 0;
    uint64_t total_tiles = 0;

    // Derived metrics
    double bytes_per_cycle() const {
        return total_cycles > 0 ? static_cast<double>(total_bytes) / total_cycles : 0.0;
    }

    double tiles_per_cycle() const {
        return total_cycles > 0 ? static_cast<double>(total_tiles) / total_cycles : 0.0;
    }

    double bandwidth_utilization(double peak_bw) const {
        return peak_bw > 0 ? bytes_per_cycle() / peak_bw : 0.0;
    }
};

/// Result of a contention benchmark
struct ContentionResult {
    std::string name;
    std::string config;

    uint32_t num_sources = 0;
    uint32_t num_destinations = 0;

    uint64_t total_cycles = 0;
    uint64_t backpressure_cycles = 0;

    // Per-source fairness
    std::vector<double> source_bandwidths;

    double contention_slowdown(double baseline_bw) const {
        double avg_bw = 0;
        for (auto bw : source_bandwidths) avg_bw += bw;
        avg_bw /= source_bandwidths.size();
        return baseline_bw > 0 ? avg_bw / baseline_bw : 0.0;
    }

    double fairness_jain() const;  // Jain's fairness index
};

/// Result of a pattern benchmark
struct PatternResult {
    std::string pattern_name;
    std::string config;

    uint64_t total_cycles = 0;
    uint64_t total_bytes = 0;
    uint32_t num_transfers = 0;

    // Pattern-specific metrics
    double completion_time_cycles() const { return total_cycles; }
    double aggregate_bandwidth() const {
        return total_cycles > 0 ? static_cast<double>(total_bytes) / total_cycles : 0.0;
    }
};

/// Result of an operator benchmark
struct OperatorResult {
    std::string operator_name;
    std::string config;

    // Timing breakdown
    uint64_t total_cycles = 0;
    uint64_t compute_cycles = 0;
    uint64_t noc_wait_cycles = 0;
    uint64_t dma_cycles = 0;

    // Data movement
    uint64_t noc_bytes_transferred = 0;
    uint32_t noc_tiles_transferred = 0;
    uint64_t external_bytes = 0;

    // Compute
    uint64_t total_flops = 0;

    // Derived metrics
    double noc_utilization(double peak_bw) const {
        return total_cycles > 0 && peak_bw > 0 ?
            static_cast<double>(noc_bytes_transferred) / (total_cycles * peak_bw) : 0.0;
    }

    double compute_utilization(double peak_flops) const {
        return total_cycles > 0 && peak_flops > 0 ?
            static_cast<double>(total_flops) / (total_cycles * peak_flops) : 0.0;
    }

    double compute_to_noc_ratio() const {
        return noc_wait_cycles > 0 ?
            static_cast<double>(compute_cycles) / noc_wait_cycles : 0.0;
    }

    bool is_noc_bound() const {
        return noc_wait_cycles > compute_cycles;
    }
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for the benchmark harness
struct NoCBenchmarkConfig {
    // NoC configuration
    NoCType noc_type = NoCType::DATAFLOW;
    NoCConfig noc_config;

    // Benchmark parameters
    uint32_t warmup_iterations = 5;
    uint32_t measurement_iterations = 10;
    uint32_t warmup_cycles = 100;

    // Problem sizes for sweeps
    std::vector<uint32_t> tile_sizes = {64, 256, 1024, 4096, 16384, 65536};
    std::vector<uint32_t> matmul_sizes = {256, 512, 1024, 2048, 4096};

    // Attention parameters for operator benchmarks
    std::vector<uint32_t> seq_lengths = {128, 256, 512, 1024};
    uint32_t d_model = 768;
    uint32_t num_heads = 12;
};

// ============================================================================
// Microbenchmark Suite
// ============================================================================

/// Latency and throughput microbenchmarks
class NoCMicrobenchmarks {
public:
    explicit NoCMicrobenchmarks(std::unique_ptr<INoC> noc);

    // Latency benchmarks
    LatencyResult measure_single_flit_latency(uint8_t src, uint8_t dst);
    LatencyResult measure_tile_latency(uint8_t src, uint8_t dst, uint32_t size);
    std::vector<LatencyResult> sweep_latency_by_distance();
    std::vector<LatencyResult> sweep_latency_by_size(uint8_t src, uint8_t dst,
                                                      const std::vector<uint32_t>& sizes);

    // Throughput benchmarks
    ThroughputResult measure_point_to_point_bandwidth(uint8_t src, uint8_t dst,
                                                       uint32_t tile_size,
                                                       uint32_t num_tiles);
    ThroughputResult measure_aggregate_bandwidth(
        const std::vector<std::pair<uint8_t, uint8_t>>& transfers,
        uint32_t tile_size, uint32_t num_tiles);
    ThroughputResult measure_bisection_bandwidth(uint32_t tile_size);

    // Contention benchmarks
    ContentionResult measure_hot_spot(const std::vector<uint8_t>& sources,
                                       uint8_t dst, uint32_t tile_size);
    ContentionResult measure_center_congestion(uint32_t tile_size);

    // Configuration
    void set_iterations(uint32_t warmup, uint32_t measure) {
        warmup_iterations_ = warmup;
        measure_iterations_ = measure;
    }

private:
    std::unique_ptr<INoC> noc_;
    uint32_t warmup_iterations_ = 5;
    uint32_t measure_iterations_ = 10;

    uint8_t manhattan_distance(uint8_t src, uint8_t dst) const;
};

// ============================================================================
// Pattern Benchmark Suite
// ============================================================================

/// Communication pattern benchmarks
class NoCPatternBenchmarks {
public:
    explicit NoCPatternBenchmarks(std::unique_ptr<INoC> noc);

    // Broadcast patterns
    PatternResult measure_broadcast_all(uint8_t src, uint32_t tile_size);
    PatternResult measure_broadcast_row(uint8_t src, uint8_t row, uint32_t tile_size);
    PatternResult measure_broadcast_col(uint8_t src, uint8_t col, uint32_t tile_size);

    // Reduce patterns
    PatternResult measure_reduce_all(uint8_t dst, uint32_t tile_size);
    PatternResult measure_reduce_tree(uint8_t dst, uint32_t tile_size);
    PatternResult measure_allreduce(uint32_t tile_size);

    // Systolic flow patterns
    PatternResult measure_east_flow(uint8_t start_col, uint32_t tile_size);
    PatternResult measure_south_flow(uint8_t start_row, uint32_t tile_size);
    PatternResult measure_systolic_flow(uint32_t a_tile_size, uint32_t b_tile_size);

    // All-to-all pattern
    PatternResult measure_all_to_all(uint32_t tile_size);
    PatternResult measure_permutation(const std::vector<uint8_t>& mapping,
                                       uint32_t tile_size);

private:
    std::unique_ptr<INoC> noc_;
};

// ============================================================================
// Operator Benchmark Suite
// ============================================================================

/// DNN operator benchmarks
class NoCOperatorBenchmarks {
public:
    explicit NoCOperatorBenchmarks(std::unique_ptr<INoC> noc);

    // Matrix multiplication
    OperatorResult benchmark_matmul(uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t Ti, uint32_t Tj, uint32_t Tk);

    // Attention mechanism (returns per-phase breakdown)
    struct AttentionResult {
        OperatorResult qkv_projection;
        OperatorResult qk_matmul;
        OperatorResult softmax;
        OperatorResult attn_v_matmul;
        OperatorResult output_projection;
        OperatorResult total;
    };
    AttentionResult benchmark_attention(uint32_t batch, uint32_t seq_len,
                                         uint32_t d_model, uint32_t num_heads);

    // Convolution
    OperatorResult benchmark_conv2d(uint32_t batch, uint32_t in_channels,
                                     uint32_t out_channels, uint32_t H, uint32_t W,
                                     uint32_t kH, uint32_t kW);

    // Reduction operations
    OperatorResult benchmark_softmax(uint32_t rows, uint32_t cols);
    OperatorResult benchmark_batchnorm(uint32_t batch, uint32_t channels, uint32_t HW);
    OperatorResult benchmark_layernorm(uint32_t batch, uint32_t features);
    OperatorResult benchmark_pooling(uint32_t H, uint32_t W, uint32_t pool_size);

private:
    std::unique_ptr<INoC> noc_;
    NoCConfig config_;
};

// ============================================================================
// Benchmark Harness (Main Entry Point)
// ============================================================================

/// Main benchmark harness that orchestrates all benchmarks
class NoCBenchmarkHarness {
public:
    explicit NoCBenchmarkHarness(const NoCBenchmarkConfig& config);

    // Run benchmark suites
    struct MicrobenchmarkSuite {
        std::vector<LatencyResult> latency_by_distance;
        std::vector<LatencyResult> latency_by_size;
        std::vector<ThroughputResult> throughput;
        std::vector<ContentionResult> contention;
    };
    MicrobenchmarkSuite run_microbenchmarks();

    struct PatternSuite {
        std::vector<PatternResult> broadcast;
        std::vector<PatternResult> reduce;
        std::vector<PatternResult> systolic;
        std::vector<PatternResult> all_to_all;
    };
    PatternSuite run_pattern_benchmarks();

    struct OperatorSuite {
        std::vector<OperatorResult> matmul;
        std::vector<NoCOperatorBenchmarks::AttentionResult> attention;
        std::vector<OperatorResult> reduction_ops;
    };
    OperatorSuite run_operator_benchmarks();

    // Run all benchmarks
    struct FullResults {
        MicrobenchmarkSuite micro;
        PatternSuite pattern;
        OperatorSuite operators;
        NoCType noc_type;
        NoCConfig noc_config;
    };
    FullResults run_all();

    // Comparison between NoC types
    struct ComparisonReport {
        NoCType baseline_type;
        NoCType candidate_type;

        struct Comparison {
            std::string benchmark_name;
            std::string metric;
            double baseline_value;
            double candidate_value;
            double speedup() const {
                return baseline_value > 0 ? candidate_value / baseline_value : 0.0;
            }
        };
        std::vector<Comparison> comparisons;

        std::string to_markdown() const;
        std::string to_csv() const;
        std::string to_json() const;
    };

    static ComparisonReport compare(const FullResults& baseline,
                                     const FullResults& candidate);

private:
    NoCBenchmarkConfig config_;
};

// ============================================================================
// Report Generation
// ============================================================================

/// Generate reports from benchmark results
class NoCBenchmarkReporter {
public:
    // Text reports
    static std::string generate_summary(const NoCBenchmarkHarness::FullResults& results);
    static std::string generate_markdown_report(const NoCBenchmarkHarness::FullResults& results);
    static std::string generate_csv(const NoCBenchmarkHarness::FullResults& results);
    static std::string generate_json(const NoCBenchmarkHarness::FullResults& results);

    // Data files for plotting
    static std::string generate_latency_heatmap_data(
        const std::vector<LatencyResult>& results, uint8_t mesh_size);
    static std::string generate_gnuplot_script(
        const NoCBenchmarkHarness::FullResults& results,
        const std::string& output_prefix);
};

} // namespace sw::benchmark
