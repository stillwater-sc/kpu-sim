/**
 * @file kpu_runner.cpp
 * @brief KPU Model Runner - Command-line tool for running KPU simulations
 *
 * Usage:
 *   kpu-runner [options] <config-file>
 *
 * Options:
 *   -h, --help              Show help message
 *   -v, --verbose           Verbose output
 *   -t, --test <type>       Test type: matmul, mlp, benchmark
 *   -m, --matrix <MxNxK>    Matrix dimensions for matmul (e.g., 128x128x128)
 *   -o, --output <file>     Output file for results
 *   --validate              Validate config and exit
 *   --show-config           Show parsed configuration
 *   --factory <name>        Use factory config: minimal, edge_ai, datacenter
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/kpu_config_loader.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/runtime/runtime.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>

using namespace sw::kpu;
using namespace sw::runtime;

// =========================================
// Command Line Parsing
// =========================================

struct Options {
    std::string config_file;
    std::string factory_config;  // minimal, edge_ai, datacenter
    std::string test_type = "matmul";
    std::string output_file;
    Size m = 64, n = 64, k = 64;
    bool verbose = false;
    bool validate_only = false;
    bool show_config = false;
    bool help = false;
};

void print_help(const char* program_name) {
    std::cout << "KPU Model Runner - Command-line tool for KPU simulations\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " [options] [config-file]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  -t, --test <type>       Test type: matmul, mlp, benchmark (default: matmul)\n";
    std::cout << "  -m, --matrix <MxNxK>    Matrix dimensions (e.g., 128x128x128)\n";
    std::cout << "  -o, --output <file>     Output file for results (JSON)\n";
    std::cout << "  --validate              Validate config and exit\n";
    std::cout << "  --show-config           Show parsed configuration\n";
    std::cout << "  --factory <name>        Use factory config: minimal, edge_ai, datacenter\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " configs/kpu/minimal.yaml\n";
    std::cout << "  " << program_name << " --factory minimal -m 256x256x256\n";
    std::cout << "  " << program_name << " -t benchmark configs/kpu/datacenter.json\n";
}

bool parse_matrix_dims(const std::string& dims, Size& m, Size& n, Size& k) {
    size_t first_x = dims.find('x');
    size_t last_x = dims.rfind('x');

    if (first_x == std::string::npos || first_x == last_x) {
        return false;
    }

    try {
        m = std::stoull(dims.substr(0, first_x));
        n = std::stoull(dims.substr(first_x + 1, last_x - first_x - 1));
        k = std::stoull(dims.substr(last_x + 1));
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_options(int argc, char* argv[], Options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--validate") {
            opts.validate_only = true;
        } else if (arg == "--show-config") {
            opts.show_config = true;
        } else if ((arg == "-t" || arg == "--test") && i + 1 < argc) {
            opts.test_type = argv[++i];
        } else if ((arg == "-m" || arg == "--matrix") && i + 1 < argc) {
            if (!parse_matrix_dims(argv[++i], opts.m, opts.n, opts.k)) {
                std::cerr << "Invalid matrix dimensions: " << argv[i] << "\n";
                return false;
            }
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else if (arg == "--factory" && i + 1 < argc) {
            opts.factory_config = argv[++i];
        } else if (arg[0] != '-') {
            opts.config_file = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }

    return true;
}

// =========================================
// Configuration Display
// =========================================

void print_config(const KPUSimulator::Config& config) {
    std::cout << "\n=== KPU Configuration ===\n\n";

    std::cout << "Host Memory:\n";
    std::cout << "  Regions:       " << config.host_memory_region_count << "\n";
    std::cout << "  Capacity:      " << config.host_memory_region_capacity_mb << " MB/region\n";
    std::cout << "  Bandwidth:     " << config.host_memory_bandwidth_gbps << " GB/s\n\n";

    std::cout << "External Memory:\n";
    std::cout << "  Banks:         " << config.memory_bank_count << "\n";
    std::cout << "  Capacity:      " << config.memory_bank_capacity_mb << " MB/bank\n";
    std::cout << "  Bandwidth:     " << config.memory_bandwidth_gbps << " GB/s\n\n";

    std::cout << "On-Chip Memory:\n";
    std::cout << "  L3 Tiles:      " << config.l3_tile_count << " x " << config.l3_tile_capacity_kb << " KB\n";
    std::cout << "  L2 Banks:      " << config.l2_bank_count << " x " << config.l2_bank_capacity_kb << " KB\n";
    std::cout << "  L1 Buffers:    " << config.l1_buffer_count << " x " << config.l1_buffer_capacity_kb << " KB\n";
    std::cout << "  Page Buffers:  " << config.page_buffer_count << " x " << config.page_buffer_capacity_kb << " KB\n\n";

    std::cout << "Data Movement:\n";
    std::cout << "  DMA Engines:   " << config.dma_engine_count << "\n";
    std::cout << "  Block Movers:  " << config.block_mover_count << "\n";
    std::cout << "  Streamers:     " << config.streamer_count << "\n\n";

    std::cout << "Compute:\n";
    std::cout << "  Tiles:         " << config.compute_tile_count << "\n";
    std::cout << "  Array:         " << config.processor_array_rows << " x " << config.processor_array_cols << "\n";
    std::cout << "  Systolic:      " << (config.use_systolic_array_mode ? "Yes" : "No") << "\n\n";
}

// =========================================
// Test Runners
// =========================================

struct TestResult {
    bool success = false;
    Cycle cycles = 0;
    double elapsed_ms = 0;
    double gflops = 0;
    std::string error;
};

TestResult run_matmul_test(KPUSimulator& sim, const Options& opts) {
    TestResult result;

    if (opts.verbose) {
        std::cout << "Running MatMul test: " << opts.m << " x " << opts.n << " x " << opts.k << "\n";
    }

    // Create runtime
    KPURuntime runtime(&sim);

    // Create kernel
    Kernel kernel = Kernel::create_matmul(opts.m, opts.n, opts.k);
    if (!kernel.is_valid()) {
        result.error = "Failed to create matmul kernel";
        return result;
    }

    // Allocate memory
    Size a_size = opts.m * opts.k * sizeof(float);
    Size b_size = opts.k * opts.n * sizeof(float);
    Size c_size = opts.m * opts.n * sizeof(float);

    Address A = runtime.malloc(a_size);
    Address B = runtime.malloc(b_size);
    Address C = runtime.malloc(c_size);

    if (A == 0 || B == 0 || C == 0) {
        result.error = "Failed to allocate memory";
        return result;
    }

    // Initialize matrices with random data
    std::vector<float> h_A(opts.m * opts.k);
    std::vector<float> h_B(opts.k * opts.n);
    std::vector<float> h_C(opts.m * opts.n, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    // Copy to device
    runtime.memcpy_h2d(A, h_A.data(), a_size);
    runtime.memcpy_h2d(B, h_B.data(), b_size);

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    auto launch_result = runtime.launch(kernel, {A, B, C});

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!launch_result.success) {
        result.error = "Kernel launch failed: " + launch_result.error;
        runtime.free(A);
        runtime.free(B);
        runtime.free(C);
        return result;
    }

    result.cycles = launch_result.cycles;
    result.success = true;

    // Calculate GFLOPS (2 * M * N * K flops for matmul)
    double flops = 2.0 * opts.m * opts.n * opts.k;
    result.gflops = (flops / 1e9) / (result.elapsed_ms / 1000.0);

    // Clean up
    runtime.free(A);
    runtime.free(B);
    runtime.free(C);

    return result;
}

TestResult run_mlp_test(KPUSimulator& sim, const Options& opts) {
    TestResult result;

    if (opts.verbose) {
        std::cout << "Running MLP test: " << opts.m << " x " << opts.n << " x " << opts.k << " with GELU\n";
    }

    // Create runtime
    KPURuntime runtime(&sim);

    // Create MLP kernel (matmul + bias + activation)
    Kernel kernel = Kernel::create_mlp(opts.m, opts.n, opts.k, ActivationType::GELU, true);
    if (!kernel.is_valid()) {
        result.error = "Failed to create MLP kernel";
        return result;
    }

    // Allocate memory
    Size a_size = opts.m * opts.k * sizeof(float);
    Size b_size = opts.k * opts.n * sizeof(float);
    Size bias_size = opts.n * sizeof(float);
    Size c_size = opts.m * opts.n * sizeof(float);

    Address A = runtime.malloc(a_size);
    Address B = runtime.malloc(b_size);
    Address bias = runtime.malloc(bias_size);
    Address C = runtime.malloc(c_size);

    if (A == 0 || B == 0 || bias == 0 || C == 0) {
        result.error = "Failed to allocate memory";
        return result;
    }

    // Initialize with random data
    std::vector<float> h_A(opts.m * opts.k);
    std::vector<float> h_B(opts.k * opts.n);
    std::vector<float> h_bias(opts.n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);
    for (auto& v : h_bias) v = dist(rng);

    runtime.memcpy_h2d(A, h_A.data(), a_size);
    runtime.memcpy_h2d(B, h_B.data(), b_size);
    runtime.memcpy_h2d(bias, h_bias.data(), bias_size);

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();

    auto launch_result = runtime.launch(kernel, {A, B, bias, C});

    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!launch_result.success) {
        result.error = "Kernel launch failed: " + launch_result.error;
        runtime.free(A);
        runtime.free(B);
        runtime.free(bias);
        runtime.free(C);
        return result;
    }

    result.cycles = launch_result.cycles;
    result.success = true;

    // Calculate GFLOPS
    double flops = 2.0 * opts.m * opts.n * opts.k + opts.m * opts.n * 10; // matmul + approx activation
    result.gflops = (flops / 1e9) / (result.elapsed_ms / 1000.0);

    // Clean up
    runtime.free(A);
    runtime.free(B);
    runtime.free(bias);
    runtime.free(C);

    return result;
}

TestResult run_benchmark(KPUSimulator& sim, const Options& opts) {
    TestResult aggregate;
    aggregate.success = true;

    std::cout << "\n=== Running Benchmark Suite ===\n\n";

    std::vector<std::tuple<Size, Size, Size, std::string>> sizes = {
        {64, 64, 64, "Small"},
        {128, 128, 128, "Medium"},
        {256, 256, 256, "Large"},
        {512, 512, 512, "XLarge"},
    };

    std::cout << std::setw(10) << "Size" << std::setw(12) << "Cycles"
              << std::setw(12) << "Time (ms)" << std::setw(12) << "GFLOPS" << "\n";
    std::cout << std::string(46, '-') << "\n";

    double total_gflops = 0;
    int count = 0;

    for (const auto& [m, n, k, name] : sizes) {
        Options test_opts = opts;
        test_opts.m = m;
        test_opts.n = n;
        test_opts.k = k;
        test_opts.verbose = false;

        TestResult result = run_matmul_test(sim, test_opts);

        if (result.success) {
            std::cout << std::setw(10) << (std::to_string(m) + "x" + std::to_string(n))
                      << std::setw(12) << result.cycles
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.elapsed_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops << "\n";

            aggregate.cycles += result.cycles;
            aggregate.elapsed_ms += result.elapsed_ms;
            total_gflops += result.gflops;
            count++;
        } else {
            std::cout << std::setw(10) << (std::to_string(m) + "x" + std::to_string(n))
                      << "  FAILED: " << result.error << "\n";
            aggregate.success = false;
        }
    }

    if (count > 0) {
        aggregate.gflops = total_gflops / count;  // Average GFLOPS
    }

    std::cout << "\n";

    return aggregate;
}

// =========================================
// Main
// =========================================

int main(int argc, char* argv[]) {
    Options opts;

    if (!parse_options(argc, argv, opts)) {
        print_help(argv[0]);
        return 1;
    }

    if (opts.help) {
        print_help(argv[0]);
        return 0;
    }

    // Load or create configuration
    KPUSimulator::Config config;

    try {
        if (!opts.factory_config.empty()) {
            // Use factory configuration
            if (opts.factory_config == "minimal") {
                config = KPUConfigLoader::create_minimal();
                if (opts.verbose) std::cout << "Using factory config: minimal\n";
            } else if (opts.factory_config == "edge_ai") {
                config = KPUConfigLoader::create_edge_ai();
                if (opts.verbose) std::cout << "Using factory config: edge_ai\n";
            } else if (opts.factory_config == "datacenter") {
                config = KPUConfigLoader::create_datacenter();
                if (opts.verbose) std::cout << "Using factory config: datacenter\n";
            } else {
                std::cerr << "Unknown factory config: " << opts.factory_config << "\n";
                return 1;
            }
        } else if (!opts.config_file.empty()) {
            // Load from file
            if (opts.verbose) {
                std::cout << "Loading configuration from: " << opts.config_file << "\n";
            }
            config = KPUConfigLoader::load(opts.config_file);
        } else {
            // No config specified, use minimal
            std::cout << "No configuration specified, using minimal factory config\n";
            config = KPUConfigLoader::create_minimal();
        }

    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << "\n";
        return 1;
    }

    // Validate if requested
    if (opts.validate_only) {
        auto result = KPUConfigLoader::validate(config);
        if (result.valid) {
            std::cout << "Configuration is valid.\n";
            for (const auto& warning : result.warnings) {
                std::cout << "Warning: " << warning << "\n";
            }
            return 0;
        } else {
            std::cerr << "Configuration is invalid:\n";
            for (const auto& error : result.errors) {
                std::cerr << "  Error: " << error << "\n";
            }
            for (const auto& warning : result.warnings) {
                std::cerr << "  Warning: " << warning << "\n";
            }
            return 1;
        }
    }

    // Show config if requested
    if (opts.show_config) {
        print_config(config);
        if (opts.validate_only || opts.test_type.empty()) {
            return 0;
        }
    }

    // Create simulator
    KPUSimulator sim(config);

    if (opts.verbose) {
        std::cout << "\nKPU Simulator initialized.\n";
        std::cout << "  Memory banks: " << sim.get_memory_bank_count() << "\n";
        std::cout << "  L3 tiles:     " << sim.get_l3_tile_count() << "\n";
        std::cout << "  L2 banks:     " << sim.get_l2_bank_count() << "\n";
        std::cout << "  L1 buffers:   " << sim.get_l1_buffer_count() << "\n";
        std::cout << "  Compute tiles:" << sim.get_compute_tile_count() << "\n";
    }

    // Run test
    TestResult result;

    if (opts.test_type == "matmul") {
        result = run_matmul_test(sim, opts);
    } else if (opts.test_type == "mlp") {
        result = run_mlp_test(sim, opts);
    } else if (opts.test_type == "benchmark") {
        result = run_benchmark(sim, opts);
    } else {
        std::cerr << "Unknown test type: " << opts.test_type << "\n";
        return 1;
    }

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << "Status:      " << (result.success ? "SUCCESS" : "FAILED") << "\n";
    if (result.success) {
        std::cout << "Cycles:      " << result.cycles << "\n";
        std::cout << "Time:        " << std::fixed << std::setprecision(3) << result.elapsed_ms << " ms\n";
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << result.gflops << " GFLOPS\n";
    } else {
        std::cout << "Error:       " << result.error << "\n";
    }

    // Write output file if requested
    if (!opts.output_file.empty()) {
        std::ofstream out(opts.output_file);
        if (out.is_open()) {
            out << "{\n";
            out << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
            out << "  \"cycles\": " << result.cycles << ",\n";
            out << "  \"elapsed_ms\": " << result.elapsed_ms << ",\n";
            out << "  \"gflops\": " << result.gflops;
            if (!result.error.empty()) {
                out << ",\n  \"error\": \"" << result.error << "\"";
            }
            out << "\n}\n";
            out.close();
            if (opts.verbose) {
                std::cout << "\nResults written to: " << opts.output_file << "\n";
            }
        }
    }

    return result.success ? 0 : 1;
}
