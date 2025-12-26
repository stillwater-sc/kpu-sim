// Benchmark Harness Implementation
// Provides infrastructure for measuring and reporting performance

#include <sw/benchmark/benchmark.hpp>

#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace sw::benchmark {

// ============================================================================
// BenchmarkHarness Implementation
// ============================================================================

BenchmarkResult BenchmarkHarness::run(const Kernel& kernel,
                                       const std::string& name,
                                       const std::string& config) {
    BenchmarkResult result;
    result.name = name;
    result.config = config;

    // Create executor
    ConcurrentExecutor executor(res_config_);

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();
    Cycle cycles = executor.execute(kernel.program());
    auto end = std::chrono::high_resolution_clock::now();

    result.wall_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.cycles = cycles;

    // Get utilization stats
    auto util = executor.get_utilization();

    // Populate result from kernel metadata
    result.flops = kernel.total_flops();
    result.Ti = kernel.Ti();
    result.Tj = kernel.Tj();
    result.Tk = kernel.Tk();

    // Calculate GFLOPS (at reference 1 GHz)
    if (cycles > 0) {
        result.gflops = static_cast<double>(result.flops) / static_cast<double>(cycles);
        result.gflops_at_freq = result.gflops * hw_spec_.clock_ghz;
    }

    // Get memory estimates from kernel
    result.arithmetic_intensity = kernel.arithmetic_intensity();

    // Instruction counts from kernel
    result.total_instructions = kernel.instruction_count();

    // Estimate external bytes from arithmetic intensity
    if (result.arithmetic_intensity > 0) {
        result.external_bytes = static_cast<Size>(result.flops / result.arithmetic_intensity);
    }

    // Calculate efficiency
    double predicted_gflops = hw_spec_.roofline_gflops(result.arithmetic_intensity);
    if (predicted_gflops > 0) {
        result.compute_efficiency = result.gflops / predicted_gflops;
    }

    // Memory efficiency (for memory-bound cases)
    if (result.is_memory_bound() && cycles > 0) {
        double achieved_bw = static_cast<double>(result.external_bytes) / static_cast<double>(cycles);
        result.memory_efficiency = achieved_bw / hw_spec_.external_bw_gbs;
    }

    // Utilization from executor
    result.dma_utilization = util.dma_utilization;
    result.block_mover_utilization = util.block_mover_utilization;
    result.streamer_utilization = util.streamer_utilization;
    result.compute_utilization = util.compute_utilization;

    return result;
}

void BenchmarkHarness::populate_result(BenchmarkResult& result,
                                        const Kernel& kernel,
                                        const CompilationStats& stats,
                                        Cycle cycles,
                                        const ConcurrentExecutor::UtilizationStats& util) {
    result.cycles = cycles;
    result.compile_time_us = stats.compile_time_us;

    // From kernel
    result.flops = kernel.total_flops();
    result.Ti = stats.selected_Ti;
    result.Tj = stats.selected_Tj;
    result.Tk = stats.selected_Tk;
    result.num_tiles = stats.total_tiles;

    // From compilation stats
    result.total_instructions = stats.instruction_count;
    result.dma_ops = stats.dma_ops;
    result.block_mover_ops = stats.block_mover_ops;
    result.streamer_ops = stats.streamer_ops;
    result.external_bytes = stats.estimated_external_bytes;
    result.arithmetic_intensity = stats.estimated_arithmetic_intensity;

    // Calculate GFLOPS
    if (cycles > 0) {
        result.gflops = static_cast<double>(result.flops) / static_cast<double>(cycles);
        result.gflops_at_freq = result.gflops * hw_spec_.clock_ghz;
    }

    // Calculate efficiency
    double predicted_gflops = hw_spec_.roofline_gflops(result.arithmetic_intensity);
    if (predicted_gflops > 0) {
        result.compute_efficiency = result.gflops / predicted_gflops;
    }

    // Utilization
    result.dma_utilization = util.dma_utilization;
    result.block_mover_utilization = util.block_mover_utilization;
    result.streamer_utilization = util.streamer_utilization;
    result.compute_utilization = util.compute_utilization;
}

BenchmarkResult BenchmarkHarness::benchmark_matmul(Size M, Size N, Size K,
                                                    DataType dtype) {
    // Compile kernel with auto-optimization
    auto start = std::chrono::high_resolution_clock::now();
    Kernel kernel = compiler_.compile_matmul(M, N, K);
    auto compile_end = std::chrono::high_resolution_clock::now();

    CompilationStats stats = compiler_.last_stats();

    // Execute
    ConcurrentExecutor executor(res_config_);
    Cycle cycles = executor.execute(kernel.program());
    auto end = std::chrono::high_resolution_clock::now();

    auto util = executor.get_utilization();

    // Build result
    BenchmarkResult result;
    result.name = "matmul";
    result.config = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    result.wall_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.compile_time_us = std::chrono::duration<double, std::micro>(compile_end - start).count();

    populate_result(result, kernel, stats, cycles, util);

    return result;
}

BenchmarkResult BenchmarkHarness::benchmark_matmul_tiled(Size M, Size N, Size K,
                                                          Size Ti, Size Tj, Size Tk,
                                                          DataType dtype) {
    // Compile with explicit tile sizes
    auto start = std::chrono::high_resolution_clock::now();
    Kernel kernel = compiler_.compile_matmul(M, N, K, Ti, Tj, Tk);
    auto compile_end = std::chrono::high_resolution_clock::now();

    CompilationStats stats = compiler_.last_stats();

    // Execute
    ConcurrentExecutor executor(res_config_);
    Cycle cycles = executor.execute(kernel.program());
    auto end = std::chrono::high_resolution_clock::now();

    auto util = executor.get_utilization();

    // Build result
    BenchmarkResult result;
    result.name = "matmul_tiled";
    result.config = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) +
                    "_" + std::to_string(Ti) + "x" + std::to_string(Tj) + "x" + std::to_string(Tk);
    result.wall_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.compile_time_us = std::chrono::duration<double, std::micro>(compile_end - start).count();

    populate_result(result, kernel, stats, cycles, util);

    return result;
}

BenchmarkResult BenchmarkHarness::benchmark_mlp(Size M, Size N, Size K,
                                                 ActivationType activation,
                                                 bool has_bias,
                                                 DataType dtype) {
    // Compile MLP kernel
    auto start = std::chrono::high_resolution_clock::now();
    Kernel kernel = compiler_.compile_mlp(M, N, K, activation, has_bias, dtype);
    auto compile_end = std::chrono::high_resolution_clock::now();

    CompilationStats stats = compiler_.last_stats();

    // Execute
    ConcurrentExecutor executor(res_config_);
    Cycle cycles = executor.execute(kernel.program());
    auto end = std::chrono::high_resolution_clock::now();

    auto util = executor.get_utilization();

    // Build result
    BenchmarkResult result;
    result.name = std::string("mlp_") + activation_type_name(activation);
    result.config = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    result.wall_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.compile_time_us = std::chrono::duration<double, std::micro>(compile_end - start).count();

    populate_result(result, kernel, stats, cycles, util);

    return result;
}

BenchmarkResult BenchmarkHarness::benchmark_graph(const KernelGraph& graph,
                                                   const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();

    // Compile graph
    auto compile_result = graph.compile();
    auto compile_end = std::chrono::high_resolution_clock::now();

    if (!compile_result.success) {
        BenchmarkResult result;
        result.name = name;
        result.config = "FAILED: " + compile_result.error_message;
        return result;
    }

    // Execute the compiled program
    ConcurrentExecutor executor(res_config_);
    Cycle cycles = executor.execute(compile_result.program);
    auto end = std::chrono::high_resolution_clock::now();

    auto util = executor.get_utilization();

    // Compute graph statistics
    auto stats = graph.compute_stats();

    // Build result
    BenchmarkResult result;
    result.name = name;
    result.config = std::to_string(stats.num_nodes) + "_kernels";
    result.cycles = cycles;
    result.wall_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    result.compile_time_us = std::chrono::duration<double, std::micro>(compile_end - start).count();

    result.flops = stats.total_flops;
    result.total_instructions = stats.total_instructions;
    result.external_bytes = stats.total_input_bytes + stats.total_output_bytes;
    result.arithmetic_intensity = stats.avg_arithmetic_intensity;

    if (cycles > 0) {
        result.gflops = static_cast<double>(result.flops) / static_cast<double>(cycles);
        result.gflops_at_freq = result.gflops * hw_spec_.clock_ghz;
    }

    double predicted_gflops = hw_spec_.roofline_gflops(result.arithmetic_intensity);
    if (predicted_gflops > 0) {
        result.compute_efficiency = result.gflops / predicted_gflops;
    }

    result.dma_utilization = util.dma_utilization;
    result.block_mover_utilization = util.block_mover_utilization;
    result.streamer_utilization = util.streamer_utilization;
    result.compute_utilization = util.compute_utilization;

    return result;
}

// ============================================================================
// Sweep Benchmarks
// ============================================================================

BenchmarkSuite BenchmarkHarness::sweep_matmul_sizes(
    const std::vector<std::tuple<Size, Size, Size>>& sizes) {

    BenchmarkSuite suite;
    suite.name = "matmul_size_sweep";
    suite.description = "Matrix multiplication across different problem sizes";

    for (const auto& [M, N, K] : sizes) {
        auto result = benchmark_matmul(M, N, K);
        suite.add(result);
    }

    return suite;
}

BenchmarkSuite BenchmarkHarness::sweep_matmul_square(Size min_size, Size max_size, Size step) {
    std::vector<std::tuple<Size, Size, Size>> sizes;

    for (Size size = min_size; size <= max_size; size *= step) {
        sizes.push_back({size, size, size});
    }

    auto suite = sweep_matmul_sizes(sizes);
    suite.name = "matmul_square_sweep";
    suite.description = "Square matrix multiplication: " + std::to_string(min_size) +
                        " to " + std::to_string(max_size);
    return suite;
}

BenchmarkSuite BenchmarkHarness::sweep_tile_sizes(
    Size M, Size N, Size K,
    const std::vector<std::tuple<Size, Size, Size>>& tile_sizes) {

    BenchmarkSuite suite;
    suite.name = "tile_sensitivity";
    suite.description = "Tile size sensitivity for " + std::to_string(M) + "x" +
                        std::to_string(N) + "x" + std::to_string(K);

    for (const auto& [Ti, Tj, Tk] : tile_sizes) {
        auto result = benchmark_matmul_tiled(M, N, K, Ti, Tj, Tk);
        suite.add(result);
    }

    return suite;
}

BenchmarkSuite BenchmarkHarness::sweep_activations(Size M, Size N, Size K) {
    BenchmarkSuite suite;
    suite.name = "activation_comparison";
    suite.description = "Activation function overhead comparison";

    // First, baseline matmul without activation
    auto baseline = benchmark_matmul(M, N, K);
    baseline.name = "matmul_baseline";
    suite.add(baseline);

    // Then MLP with various activations
    std::vector<ActivationType> activations = {
        ActivationType::NONE,
        ActivationType::RELU,
        ActivationType::GELU,
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::SILU,
    };

    for (auto act : activations) {
        auto result = benchmark_mlp(M, N, K, act, true);
        suite.add(result);
    }

    return suite;
}

// ============================================================================
// Roofline Analysis
// ============================================================================

std::string BenchmarkHarness::generate_roofline_data(const BenchmarkSuite& results) const {
    std::ostringstream ss;

    ss << "# Roofline Data\n";
    ss << "# arithmetic_intensity gflops efficiency name\n";

    for (const auto& r : results.results) {
        ss << std::fixed << std::setprecision(4)
           << r.arithmetic_intensity << " "
           << r.gflops << " "
           << r.compute_efficiency << " "
           << r.name << "_" << r.config << "\n";
    }

    return ss.str();
}

std::string BenchmarkHarness::generate_roofline_gnuplot(const BenchmarkSuite& results) const {
    std::ostringstream ss;

    ss << "# Roofline Plot for " << results.name << "\n";
    ss << "set terminal pngcairo size 800,600\n";
    ss << "set output 'roofline.png'\n";
    ss << "set logscale x 2\n";
    ss << "set logscale y 2\n";
    ss << "set xlabel 'Arithmetic Intensity (FLOP/byte)'\n";
    ss << "set ylabel 'Performance (GFLOPS)'\n";
    ss << "set title 'Roofline Model - " << results.name << "'\n";
    ss << "set grid\n";
    ss << "\n";

    // Roofline ceiling lines
    double peak = hw_spec_.peak_gflops;
    double ridge = hw_spec_.ridge_point_external();

    ss << "# Peak compute: " << peak << " GFLOPS\n";
    ss << "# Ridge point: " << ridge << " FLOP/byte\n";
    ss << "\n";

    ss << "# Memory-bound region: y = x * " << hw_spec_.external_bw_gbs << "\n";
    ss << "# Compute-bound region: y = " << peak << "\n";
    ss << "\n";

    ss << "set arrow from 0.1," << (0.1 * hw_spec_.external_bw_gbs)
       << " to " << ridge << "," << peak << " nohead lw 2 lc rgb 'blue'\n";
    ss << "set arrow from " << ridge << "," << peak
       << " to 1000," << peak << " nohead lw 2 lc rgb 'blue'\n";
    ss << "\n";

    ss << "plot '-' using 1:2 with points pt 7 ps 1.5 title 'Benchmarks'\n";
    for (const auto& r : results.results) {
        ss << r.arithmetic_intensity << " " << r.gflops << "\n";
    }
    ss << "e\n";

    return ss.str();
}

} // namespace sw::benchmark
