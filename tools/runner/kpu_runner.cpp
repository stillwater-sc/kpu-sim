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
 *   --preset <name>         Use preset: fast, balanced, accurate, mixed,
 *                           minimal, edge_ai, embodied_ai, datacenter
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/config/simulator_config_parser.hpp>
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
using namespace sw::kpu::config;
using namespace sw::runtime;

// =========================================
// Configuration Conversion
// =========================================

/**
 * @brief Convert SimulatorConfig to KPUSimulator::Config
 *
 * Maps the new multi-fidelity configuration format to the
 * legacy KPUSimulator::Config format for compatibility.
 */
KPUSimulator::Config convert_config(const SimulatorConfig& sim_config) {
    KPUSimulator::Config config;

    // Host memory (defaults for simulation)
    config.host_memory_region_count = 1;
    config.host_memory_region_capacity_mb = 1024;  // 1GB
    config.host_memory_bandwidth_gbps = 100;

    // External memory from config
    if (sim_config.memory_controller) {
        config.memory_bank_count = sim_config.memory_controller->num_channels;
        config.memory_bank_capacity_mb = 1024;  // Default 1GB per bank
        config.memory_bandwidth_gbps = 100;     // Default
    } else {
        config.memory_bank_count = sim_config.num_memory_controllers;
        config.memory_bank_capacity_mb = 1024;
        config.memory_bandwidth_gbps = 100;
    }

    // Memory controllers
    config.memory_controller_count = sim_config.num_memory_controllers;
    config.page_buffer_count = 4;
    config.page_buffer_capacity_kb = 32;

    // On-chip memory hierarchy
    config.l3_layer.num_tiles = sim_config.num_l3_tiles;
    if (sim_config.l3_tile) {
        config.l3_layer.capacity_kb = sim_config.l3_tile->capacity_kb;
    } else {
        config.l3_layer.capacity_kb = 256;  // Default
    }

    config.l2_bank_count = sim_config.num_l2_banks;
    config.l2_bank_capacity_kb = 64;  // Default

    // L1 buffers - derived from compute fabric
    config.l1_buffer_count = 0;  // Auto-compute
    config.l1_buffer_capacity_kb = 32;

    // Data movement
    config.dma_engine_count = sim_config.num_dma_engines;
    if (sim_config.dma_engine) {
        config.l3_layer.block_mover_count = sim_config.dma_engine->num_channels;
    } else {
        config.l3_layer.block_mover_count = sim_config.num_l3_tiles;
    }
    config.streamer_count = sim_config.num_l3_tiles * 4;

    // Compute
    config.compute_tile_count = sim_config.num_compute_tiles;
    if (sim_config.compute_fabric) {
        config.processor_array_rows = sim_config.compute_fabric->array_rows;
        config.processor_array_cols = sim_config.compute_fabric->array_cols;
    } else {
        config.processor_array_rows = 16;
        config.processor_array_cols = 16;
    }
    config.processor_array_topology = ProcessorArrayTopology::RECTANGULAR;
    config.use_systolic_array_mode = true;

    // Address map defaults
    config.host_memory_base = 0;
    config.external_memory_base = 0;
    config.l3_tile_base = 0;
    config.l2_bank_base = 0;
    config.l1_buffer_base = 0;
    config.page_buffer_base = 0;

    return config;
}

/**
 * @brief Get a preset SimulatorConfig by name
 */
SimulatorConfig get_preset_config(const std::string& preset_name) {
    SimulatorConfig config;

    if (preset_name == "fast" || preset_name == "behavioral") {
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.set_fidelity(SimulationFidelity::BEHAVIORAL);
    }
    else if (preset_name == "balanced" || preset_name == "transactional") {
        config.default_fidelity = SimulationFidelity::TRANSACTIONAL;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::TRANSACTIONAL;
        mc.technology = MemoryTechnology::LPDDR5;
        config.memory_controller = mc;

        DMAEngineConfig dma;
        dma.fidelity = SimulationFidelity::TRANSACTIONAL;
        config.dma_engine = dma;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        config.compute_fabric = cf;
    }
    else if (preset_name == "accurate" || preset_name == "cycle_accurate") {
        config.default_fidelity = SimulationFidelity::CYCLE_ACCURATE;
        config.default_verification = VerificationLevel::INVARIANTS;
        config.set_fidelity(SimulationFidelity::CYCLE_ACCURATE);

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        mc.technology = MemoryTechnology::LPDDR5;
        mc.verification = VerificationLevel::INVARIANTS;
        config.memory_controller = mc;
    }
    else if (preset_name == "mixed") {
        config.default_fidelity = SimulationFidelity::TRANSACTIONAL;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        mc.technology = MemoryTechnology::LPDDR5;
        config.memory_controller = mc;

        DMAEngineConfig dma;
        dma.fidelity = SimulationFidelity::TRANSACTIONAL;
        config.dma_engine = dma;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        config.compute_fabric = cf;

        NoCConfig noc;
        noc.fidelity = SimulationFidelity::CYCLE_ACCURATE;
        noc.technology = InterconnectTechnology::MESH_2D;
        config.noc = noc;
    }
    else if (preset_name == "minimal") {
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.num_memory_controllers = 1;
        config.num_dma_engines = 1;
        config.num_l3_tiles = 1;
        config.num_l2_banks = 4;
        config.num_compute_tiles = 1;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::BEHAVIORAL;
        mc.technology = MemoryTechnology::LPDDR5;
        mc.num_channels = 1;
        config.memory_controller = mc;

        L3TileConfig l3;
        l3.fidelity = SimulationFidelity::BEHAVIORAL;
        l3.capacity_kb = 64;
        config.l3_tile = l3;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        cf.array_rows = 8;
        cf.array_cols = 8;
        cf.macs_per_cycle = 64;
        config.compute_fabric = cf;
    }
    else if (preset_name == "edge_ai") {
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.num_memory_controllers = 2;
        config.num_dma_engines = 2;
        config.num_l3_tiles = 2;
        config.num_l2_banks = 16;
        config.num_compute_tiles = 2;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::BEHAVIORAL;
        mc.technology = MemoryTechnology::LPDDR5;
        mc.num_channels = 2;
        config.memory_controller = mc;

        L3TileConfig l3;
        l3.fidelity = SimulationFidelity::BEHAVIORAL;
        l3.capacity_kb = 128;
        config.l3_tile = l3;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        cf.array_rows = 16;
        cf.array_cols = 16;
        cf.macs_per_cycle = 256;
        config.compute_fabric = cf;
    }
    else if (preset_name == "embodied_ai" || preset_name == "humanoid_ai") {
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.num_memory_controllers = 4;
        config.num_dma_engines = 8;
        config.num_l3_tiles = 64;
        config.num_l2_banks = 1024;
        config.num_compute_tiles = 64;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::BEHAVIORAL;
        mc.technology = MemoryTechnology::LPDDR5;
        mc.num_channels = 8;
        config.memory_controller = mc;

        L3TileConfig l3;
        l3.fidelity = SimulationFidelity::BEHAVIORAL;
        l3.capacity_kb = 256;
        config.l3_tile = l3;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        cf.array_rows = 24;
        cf.array_cols = 24;
        cf.macs_per_cycle = 576;
        config.compute_fabric = cf;
    }
    else if (preset_name == "datacenter") {
        config.default_fidelity = SimulationFidelity::BEHAVIORAL;
        config.num_memory_controllers = 6;
        config.num_dma_engines = 32;
        config.num_l3_tiles = 256;
        config.num_l2_banks = 4096;
        config.num_compute_tiles = 256;

        MemoryControllerConfig mc;
        mc.fidelity = SimulationFidelity::BEHAVIORAL;
        mc.technology = MemoryTechnology::HBM3;
        mc.num_channels = 6;
        config.memory_controller = mc;

        L3TileConfig l3;
        l3.fidelity = SimulationFidelity::BEHAVIORAL;
        l3.capacity_kb = 512;
        config.l3_tile = l3;

        ComputeFabricConfig cf;
        cf.fidelity = SimulationFidelity::BEHAVIORAL;
        cf.technology = ComputeTechnology::MIXED_PRECISION;
        cf.array_rows = 32;
        cf.array_cols = 32;
        cf.macs_per_cycle = 1024;
        config.compute_fabric = cf;
    }
    else {
        throw ConfigParseError("Unknown preset: " + preset_name);
    }

    return config;
}

// =========================================
// Command Line Parsing
// =========================================

struct Options {
    std::string config_file;
    std::string preset;
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
    std::cout << "  --preset <name>         Use preset configuration:\n";
    std::cout << "                            fast       - All BEHAVIORAL (fastest)\n";
    std::cout << "                            balanced   - Memory TRANSACTIONAL\n";
    std::cout << "                            accurate   - All CYCLE_ACCURATE\n";
    std::cout << "                            mixed      - Memory/NoC accurate, compute fast\n";
    std::cout << "                            minimal    - 1 tile, 8x8 array\n";
    std::cout << "                            edge_ai    - 2 tiles, 16x16 arrays\n";
    std::cout << "                            embodied_ai - 64 tiles, 24x24 arrays\n";
    std::cout << "                            datacenter - 256 tiles, 32x32 arrays\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " configs/components/kpu/minimal.json\n";
    std::cout << "  " << program_name << " --preset minimal -m 256x256x256\n";
    std::cout << "  " << program_name << " -t benchmark configs/components/kpu/datacenter.json\n";
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
        } else if (arg == "--preset" && i + 1 < argc) {
            opts.preset = argv[++i];
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

void print_config(const SimulatorConfig& config) {
    std::cout << "\n=== KPU Configuration ===\n\n";

    std::cout << "Simulation Settings:\n";
    std::cout << "  Fidelity:      " << to_string(config.default_fidelity) << "\n";
    std::cout << "  Verification:  " << to_string(config.default_verification) << "\n";
    std::cout << "  Tracing:       " << (config.default_tracing ? "enabled" : "disabled") << "\n\n";

    std::cout << "Component Counts:\n";
    std::cout << "  Memory Controllers:  " << config.num_memory_controllers << "\n";
    std::cout << "  DMA Engines:         " << config.num_dma_engines << "\n";
    std::cout << "  L3 Tiles:            " << config.num_l3_tiles << "\n";
    std::cout << "  L2 Banks:            " << config.num_l2_banks << "\n";
    std::cout << "  Compute Tiles:       " << config.num_compute_tiles << "\n\n";

    if (config.memory_controller) {
        std::cout << "Memory Controller:\n";
        std::cout << "  Fidelity:    " << to_string(config.memory_controller->fidelity) << "\n";
        std::cout << "  Technology:  " << to_string(config.memory_controller->technology) << "\n";
        std::cout << "  Channels:    " << static_cast<int>(config.memory_controller->num_channels) << "\n\n";
    }

    if (config.compute_fabric) {
        std::cout << "Compute Fabric:\n";
        std::cout << "  Fidelity:    " << to_string(config.compute_fabric->fidelity) << "\n";
        std::cout << "  Array Size:  " << config.compute_fabric->array_rows
                  << " x " << config.compute_fabric->array_cols << "\n";
        std::cout << "  MACs/cycle:  " << config.compute_fabric->macs_per_cycle << "\n\n";
    }
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
    double flops = 2.0 * static_cast<double>(opts.m) * static_cast<double>(opts.n) * static_cast<double>(opts.k);
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
    double flops = 2.0 * static_cast<double>(opts.m) * static_cast<double>(opts.n) * static_cast<double>(opts.k) + static_cast<double>(opts.m) * static_cast<double>(opts.n) * 10; // matmul + approx activation
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
    SimulatorConfig sim_config;

    try {
        if (!opts.preset.empty()) {
            // Use preset configuration
            sim_config = get_preset_config(opts.preset);
            if (opts.verbose) {
                std::cout << "Using preset: " << opts.preset << "\n";
            }
        } else if (!opts.config_file.empty()) {
            // Load from file
            if (opts.verbose) {
                std::cout << "Loading configuration from: " << opts.config_file << "\n";
            }
            sim_config = SimulatorConfigParser::parse_file(opts.config_file);
        } else {
            // No config specified, use minimal
            std::cout << "No configuration specified, using minimal preset\n";
            sim_config = get_preset_config("minimal");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error loading configuration: " << e.what() << "\n";
        return 1;
    }

    // Validate if requested
    if (opts.validate_only) {
        std::cout << "Configuration is valid.\n";
        std::cout << "\nConfiguration Summary:\n";
        std::cout << "  Default fidelity: " << to_string(sim_config.default_fidelity) << "\n";
        std::cout << "  Compute tiles:    " << sim_config.num_compute_tiles << "\n";
        std::cout << "  L3 tiles:         " << sim_config.num_l3_tiles << "\n";
        std::cout << "  DMA engines:      " << sim_config.num_dma_engines << "\n";
        return 0;
    }

    // Show config if requested
    if (opts.show_config) {
        print_config(sim_config);
        if (opts.validate_only || opts.test_type.empty()) {
            return 0;
        }
    }

    // Convert to KPUSimulator::Config
    KPUSimulator::Config config = convert_config(sim_config);

    // Create simulator
    KPUSimulator sim(config);

    if (opts.verbose) {
        std::cout << "\nKPU Simulator initialized.\n";
        std::cout << "  Fidelity:     " << to_string(sim_config.default_fidelity) << "\n";
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
            out << "  \"fidelity\": \"" << to_string(sim_config.default_fidelity) << "\",\n";
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
