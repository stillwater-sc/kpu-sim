// ============================================================================
// tools/benchmark/kpu-noc-bench/main.cpp
// Network-on-Chip Benchmark CLI Tool
// ============================================================================

#include <sw/benchmark/noc_benchmark.hpp>
#include <sw/benchmark/noc_benchmark_tracer.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace sw::benchmark;
using namespace sw::kpu::noc;

// ============================================================================
// Command Line Parsing
// ============================================================================

struct Options {
    // NoC configuration
    NoCType noc_type = NoCType::DATAFLOW;
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;
    size_t buffer_depth = 8;

    // Benchmark selection
    bool run_all = false;
    bool run_micro = false;
    bool run_patterns = false;
    bool run_operators = false;
    bool run_compare = false;

    // Operator-specific options
    std::string operator_name;
    uint32_t M = 1024, N = 1024, K = 1024;
    uint32_t seq_len = 512;
    uint32_t d_model = 768;
    uint32_t num_heads = 12;

    // Output options
    std::string output_file;
    std::string output_format = "text";  // text, markdown, csv, json

    // Trace options
    std::string trace_file;
    bool trace_packets = true;
    bool trace_hops = true;
    bool trace_counters = true;

    // Iteration counts
    uint32_t warmup = 5;
    uint32_t measure = 10;

    bool verbose = false;
    bool help = false;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "NoC Benchmark Tool - Characterize Network-on-Chip performance\n\n";

    std::cout << "NoC Configuration:\n";
    std::cout << "  --noc <type>       NoC type: wormhole, dataflow (default: dataflow)\n";
    std::cout << "  --rows <n>         Mesh rows (default: 4)\n";
    std::cout << "  --cols <n>         Mesh columns (default: 4)\n";
    std::cout << "  --buffer <n>       Buffer depth per router (default: 8)\n\n";

    std::cout << "Benchmark Selection:\n";
    std::cout << "  --all              Run all benchmarks\n";
    std::cout << "  --micro            Run microbenchmarks (latency, throughput, contention)\n";
    std::cout << "  --patterns         Run pattern benchmarks (broadcast, reduce, systolic)\n";
    std::cout << "  --operators        Run operator benchmarks (matmul, attention, etc.)\n";
    std::cout << "  --compare          Compare wormhole vs dataflow NoC\n\n";

    std::cout << "Operator-Specific Options:\n";
    std::cout << "  --operator <name>  Run specific operator: matmul, attention, conv2d, softmax\n";
    std::cout << "  --M <n>            Matrix M dimension (default: 1024)\n";
    std::cout << "  --N <n>            Matrix N dimension (default: 1024)\n";
    std::cout << "  --K <n>            Matrix K dimension (default: 1024)\n";
    std::cout << "  --seq <n>          Sequence length for attention (default: 512)\n";
    std::cout << "  --d_model <n>      Model dimension for attention (default: 768)\n";
    std::cout << "  --heads <n>        Number of attention heads (default: 12)\n\n";

    std::cout << "Output Options:\n";
    std::cout << "  -o, --output <file>  Write results to file\n";
    std::cout << "  --format <fmt>       Output format: text, markdown, csv, json (default: text)\n\n";

    std::cout << "Trace Options:\n";
    std::cout << "  --trace <file>       Export Chrome Trace to file for Perfetto visualization\n";
    std::cout << "  --no-trace-packets   Disable packet flow tracing\n";
    std::cout << "  --no-trace-hops      Disable per-hop routing events\n";
    std::cout << "  --no-trace-counters  Disable bandwidth/backpressure counters\n\n";

    std::cout << "Iteration Control:\n";
    std::cout << "  --warmup <n>       Warmup iterations (default: 5)\n";
    std::cout << "  --measure <n>      Measurement iterations (default: 10)\n\n";

    std::cout << "Other:\n";
    std::cout << "  -v, --verbose      Verbose output\n";
    std::cout << "  -h, --help         Show this help message\n\n";

    std::cout << "Examples:\n";
    std::cout << "  " << program << " --all --noc dataflow\n";
    std::cout << "  " << program << " --compare --format markdown -o comparison.md\n";
    std::cout << "  " << program << " --operator matmul --M 2048 --N 2048 --K 2048\n";
    std::cout << "  " << program << " --operator attention --seq 1024 --heads 16\n";
    std::cout << "  " << program << " --compare --trace noc_trace.json  # Export Chrome Trace\n";
    std::cout << "  " << program << " --patterns --trace trace.json --no-trace-hops\n";
}

NoCType parse_noc_type(const std::string& s) {
    if (s == "wormhole" || s == "WORMHOLE") return NoCType::WORMHOLE;
    if (s == "dataflow" || s == "DATAFLOW") return NoCType::DATAFLOW;
    std::cerr << "Unknown NoC type: " << s << ", using DATAFLOW\n";
    return NoCType::DATAFLOW;
}

Options parse_args(int argc, char* argv[]) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--all") {
            opts.run_all = true;
        } else if (arg == "--micro") {
            opts.run_micro = true;
        } else if (arg == "--patterns") {
            opts.run_patterns = true;
        } else if (arg == "--operators") {
            opts.run_operators = true;
        } else if (arg == "--compare") {
            opts.run_compare = true;
        } else if (arg == "--noc" && i + 1 < argc) {
            opts.noc_type = parse_noc_type(argv[++i]);
        } else if (arg == "--rows" && i + 1 < argc) {
            opts.mesh_rows = static_cast<uint8_t>(std::stoi(argv[++i]));
        } else if (arg == "--cols" && i + 1 < argc) {
            opts.mesh_cols = static_cast<uint8_t>(std::stoi(argv[++i]));
        } else if (arg == "--buffer" && i + 1 < argc) {
            opts.buffer_depth = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--operator" && i + 1 < argc) {
            opts.operator_name = argv[++i];
        } else if (arg == "--M" && i + 1 < argc) {
            opts.M = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--N" && i + 1 < argc) {
            opts.N = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--K" && i + 1 < argc) {
            opts.K = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--seq" && i + 1 < argc) {
            opts.seq_len = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--d_model" && i + 1 < argc) {
            opts.d_model = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--heads" && i + 1 < argc) {
            opts.num_heads = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else if (arg == "--format" && i + 1 < argc) {
            opts.output_format = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--measure" && i + 1 < argc) {
            opts.measure = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--trace" && i + 1 < argc) {
            opts.trace_file = argv[++i];
        } else if (arg == "--no-trace-packets") {
            opts.trace_packets = false;
        } else if (arg == "--no-trace-hops") {
            opts.trace_hops = false;
        } else if (arg == "--no-trace-counters") {
            opts.trace_counters = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
        }
    }

    // Default to running all if nothing specified
    if (!opts.run_all && !opts.run_micro && !opts.run_patterns &&
        !opts.run_operators && !opts.run_compare && opts.operator_name.empty()) {
        opts.run_all = true;
    }

    return opts;
}

// ============================================================================
// Tracer Helper
// ============================================================================

std::shared_ptr<NoCBenchmarkTracer> create_tracer(const Options& opts) {
    if (opts.trace_file.empty()) {
        return nullptr;
    }

    TracerConfig config;
    config.enabled = true;
    config.trace_packets = opts.trace_packets;
    config.trace_hops = opts.trace_hops;
    config.trace_counters = opts.trace_counters;
    config.trace_router_state = true;
    config.trace_link_usage = true;

    return std::make_shared<NoCBenchmarkTracer>(config);
}

void export_trace_if_needed(const std::shared_ptr<NoCBenchmarkTracer>& tracer,
                            const Options& opts) {
    if (tracer && !opts.trace_file.empty()) {
        if (tracer->export_chrome_trace(opts.trace_file, opts.mesh_rows, opts.mesh_cols)) {
            std::cout << "\nChrome Trace exported to: " << opts.trace_file << "\n";
            std::cout << "View in: https://ui.perfetto.dev or chrome://tracing\n";
            std::cout << "Recorded: " << tracer->num_packets() << " packets, "
                      << tracer->num_hops() << " hops, "
                      << tracer->num_link_events() << " link events\n";
        } else {
            std::cerr << "Failed to export trace to: " << opts.trace_file << "\n";
        }
    }
}

// ============================================================================
// Benchmark Runners
// ============================================================================

void run_single_operator(const Options& opts) {
    NoCConfig noc_config;
    noc_config.rows = opts.mesh_rows;
    noc_config.cols = opts.mesh_cols;
    noc_config.buffer_depth = opts.buffer_depth;

    auto noc = create_noc(opts.noc_type, noc_config);
    NoCOperatorBenchmarks benchmarks(std::move(noc));

    std::cout << "Running " << opts.operator_name << " benchmark...\n";
    std::cout << "NoC: " << to_string(opts.noc_type)
              << " " << static_cast<int>(opts.mesh_rows) << "x"
              << static_cast<int>(opts.mesh_cols) << "\n\n";

    if (opts.operator_name == "matmul") {
        auto result = benchmarks.benchmark_matmul(opts.M, opts.N, opts.K, 64, 64, 64);

        std::cout << "MatMul " << opts.M << "x" << opts.N << "x" << opts.K << ":\n";
        std::cout << "  Total cycles:    " << result.total_cycles << "\n";
        std::cout << "  Compute cycles:  " << result.compute_cycles << "\n";
        std::cout << "  NoC wait cycles: " << result.noc_wait_cycles << "\n";
        std::cout << "  NoC bytes:       " << result.noc_bytes_transferred << "\n";
        std::cout << "  Total FLOPs:     " << result.total_flops << "\n";

        double gflops = result.total_cycles > 0 ?
            static_cast<double>(result.total_flops) / result.total_cycles : 0.0;
        std::cout << "  GFLOPS:          " << gflops << "\n";
        std::cout << "  NoC bound:       " << (result.is_noc_bound() ? "Yes" : "No") << "\n";

    } else if (opts.operator_name == "attention") {
        auto result = benchmarks.benchmark_attention(1, opts.seq_len, opts.d_model, opts.num_heads);

        std::cout << "Attention (batch=1, seq=" << opts.seq_len
                  << ", d_model=" << opts.d_model
                  << ", heads=" << opts.num_heads << "):\n\n";

        auto print_phase = [](const std::string& name, const OperatorResult& r) {
            double gflops = r.total_cycles > 0 ?
                static_cast<double>(r.total_flops) / r.total_cycles : 0.0;
            std::cout << "  " << name << ":\n";
            std::cout << "    Cycles: " << r.total_cycles
                      << ", FLOPs: " << r.total_flops
                      << ", GFLOPS: " << gflops << "\n";
        };

        print_phase("QKV Projection", result.qkv_projection);
        print_phase("Q @ K^T", result.qk_matmul);
        print_phase("Softmax", result.softmax);
        print_phase("Attention @ V", result.attn_v_matmul);
        print_phase("Output Projection", result.output_projection);

        std::cout << "\n  Total:\n";
        std::cout << "    Cycles: " << result.total.total_cycles << "\n";
        std::cout << "    FLOPs:  " << result.total.total_flops << "\n";
        double total_gflops = result.total.total_cycles > 0 ?
            static_cast<double>(result.total.total_flops) / result.total.total_cycles : 0.0;
        std::cout << "    GFLOPS: " << total_gflops << "\n";

    } else if (opts.operator_name == "softmax") {
        auto result = benchmarks.benchmark_softmax(opts.M, opts.N);

        std::cout << "Softmax " << opts.M << "x" << opts.N << ":\n";
        std::cout << "  Total cycles: " << result.total_cycles << "\n";
        std::cout << "  NoC bytes:    " << result.noc_bytes_transferred << "\n";

    } else if (opts.operator_name == "conv2d") {
        auto result = benchmarks.benchmark_conv2d(1, 64, 128, opts.M, opts.N, 3, 3);

        std::cout << "Conv2D " << opts.M << "x" << opts.N << " (64->128, 3x3):\n";
        std::cout << "  Total cycles: " << result.total_cycles << "\n";
        std::cout << "  Total FLOPs:  " << result.total_flops << "\n";

    } else {
        std::cerr << "Unknown operator: " << opts.operator_name << "\n";
        std::cerr << "Available: matmul, attention, softmax, conv2d\n";
    }
}

// Run a traced benchmark to capture NoC events for visualization
void run_traced_benchmark(const Options& opts, std::shared_ptr<NoCBenchmarkTracer>& tracer) {
    if (!tracer) return;

    NoCConfig noc_config;
    noc_config.rows = opts.mesh_rows;
    noc_config.cols = opts.mesh_cols;
    noc_config.buffer_depth = opts.buffer_depth;

    auto noc = create_noc(opts.noc_type, noc_config);

    std::cout << "Running traced benchmark for visualization...\n";

    // Run a systolic-style data movement pattern and trace it
    const uint32_t tile_size = 4096;  // 4KB tiles
    const uint8_t num_tiles = 4;      // 4 tiles in each direction

    uint64_t cycle = 0;
    std::vector<std::pair<uint64_t, uint8_t>> pending_packets;  // packet_id -> dst

    // Track bandwidth and backpressure
    uint32_t total_bytes = 0;
    uint64_t last_counter_cycle = 0;

    // Simulate A tiles flowing East (row broadcast)
    for (uint8_t row = 0; row < opts.mesh_rows && row < num_tiles; row++) {
        for (uint8_t k = 0; k < num_tiles; k++) {
            // Inject A[row, k] at column 0, flows East
            uint8_t src = row * opts.mesh_cols;
            for (uint8_t col = 1; col < opts.mesh_cols; col++) {
                uint8_t dst = row * opts.mesh_cols + col;

                TileDescriptor tile;
                tile.tensor = TensorId::A;
                tile.m_tile = row;
                tile.k_tile = k;
                tile.size = tile_size;

                uint64_t packet_id = tracer->record_inject(src, dst, tile, cycle);

                // Record hop from src to dst (simulated)
                for (uint8_t hop_col = col - 1; hop_col < col; hop_col++) {
                    uint8_t hop_router = row * opts.mesh_cols + hop_col;
                    tracer->record_hop(packet_id, hop_router,
                                       hop_col == 0 ? sw::kpu::noc::PortDir::LOCAL : sw::kpu::noc::PortDir::WEST,
                                       sw::kpu::noc::PortDir::EAST,
                                       cycle + hop_col, cycle + hop_col + 1);
                }

                // Simulate delivery
                uint64_t latency = tile_size / 64 + (col - 1);  // flits + hops
                tracer->record_deliver(packet_id, cycle + latency);

                // Record link transfer
                tracer->record_link_transfer(row * opts.mesh_cols + col - 1,
                                             dst, sw::kpu::noc::PortDir::EAST,
                                             cycle, cycle + latency, tile_size);

                total_bytes += tile_size;
                pending_packets.push_back({packet_id, dst});
            }
            cycle += tile_size / 64;  // Advance time for each tile
        }
    }

    // Simulate B tiles flowing South (column broadcast)
    for (uint8_t col = 0; col < opts.mesh_cols && col < num_tiles; col++) {
        for (uint8_t k = 0; k < num_tiles; k++) {
            // Inject B[k, col] at row 0, flows South
            uint8_t src = col;
            for (uint8_t row = 1; row < opts.mesh_rows; row++) {
                uint8_t dst = row * opts.mesh_cols + col;

                TileDescriptor tile;
                tile.tensor = TensorId::B;
                tile.k_tile = k;
                tile.n_tile = col;
                tile.size = tile_size;

                uint64_t packet_id = tracer->record_inject(src, dst, tile, cycle);

                // Record hop from src to dst
                for (uint8_t hop_row = row - 1; hop_row < row; hop_row++) {
                    uint8_t hop_router = hop_row * opts.mesh_cols + col;
                    tracer->record_hop(packet_id, hop_router,
                                       hop_row == 0 ? sw::kpu::noc::PortDir::LOCAL : sw::kpu::noc::PortDir::NORTH,
                                       sw::kpu::noc::PortDir::SOUTH,
                                       cycle + hop_row, cycle + hop_row + 1);
                }

                // Simulate delivery
                uint64_t latency = tile_size / 64 + (row - 1);
                tracer->record_deliver(packet_id, cycle + latency);

                // Record link transfer
                tracer->record_link_transfer((row - 1) * opts.mesh_cols + col,
                                             dst, sw::kpu::noc::PortDir::SOUTH,
                                             cycle, cycle + latency, tile_size);

                total_bytes += tile_size;
            }
            cycle += tile_size / 64;
        }
    }

    // Record counter samples throughout the simulation
    for (uint64_t c = 0; c <= cycle; c += 10) {
        double bw = static_cast<double>(total_bytes) / (cycle > 0 ? cycle : 1);
        uint32_t active = (c < cycle / 2) ?
            static_cast<uint32_t>(opts.mesh_rows * opts.mesh_cols / 2) :
            static_cast<uint32_t>(opts.mesh_rows * opts.mesh_cols / 4);
        tracer->record_counters(c, bw, 0, active);
    }

    std::cout << "Traced " << tracer->num_packets() << " packets over "
              << cycle << " cycles\n";
}

void run_comparison(const Options& opts) {
    std::cout << "Comparing NoC implementations...\n\n";

    // Create tracer if requested
    auto tracer = create_tracer(opts);

    NoCBenchmarkConfig config;
    config.noc_config.rows = opts.mesh_rows;
    config.noc_config.cols = opts.mesh_cols;
    config.noc_config.buffer_depth = opts.buffer_depth;
    config.warmup_iterations = opts.warmup;
    config.measurement_iterations = opts.measure;

    // Run benchmarks for both NoC types
    std::cout << "Running WORMHOLE benchmarks...\n";
    config.noc_type = NoCType::WORMHOLE;
    NoCBenchmarkHarness wormhole_harness(config);
    auto wormhole_results = wormhole_harness.run_all();

    std::cout << "Running DATAFLOW benchmarks...\n";
    config.noc_type = NoCType::DATAFLOW;
    NoCBenchmarkHarness dataflow_harness(config);
    auto dataflow_results = dataflow_harness.run_all();

    // Run traced benchmark for visualization (uses dataflow by default)
    if (tracer) {
        run_traced_benchmark(opts, tracer);
        export_trace_if_needed(tracer, opts);
    }

    // Generate comparison report
    auto report = NoCBenchmarkHarness::compare(wormhole_results, dataflow_results);

    std::string output;
    if (opts.output_format == "markdown") {
        output = report.to_markdown();
    } else if (opts.output_format == "csv") {
        output = report.to_csv();
    } else if (opts.output_format == "json") {
        output = report.to_json();
    } else {
        output = report.to_markdown();  // Default to markdown
    }

    if (!opts.output_file.empty()) {
        std::ofstream file(opts.output_file);
        file << output;
        std::cout << "Comparison report written to: " << opts.output_file << "\n";
    } else {
        std::cout << "\n" << output;
    }
}

void run_full_benchmarks(const Options& opts) {
    // Create tracer if requested
    auto tracer = create_tracer(opts);

    NoCBenchmarkConfig config;
    config.noc_type = opts.noc_type;
    config.noc_config.rows = opts.mesh_rows;
    config.noc_config.cols = opts.mesh_cols;
    config.noc_config.buffer_depth = opts.buffer_depth;
    config.warmup_iterations = opts.warmup;
    config.measurement_iterations = opts.measure;

    std::cout << "Running NoC benchmarks...\n";
    std::cout << "NoC: " << to_string(opts.noc_type)
              << " " << static_cast<int>(opts.mesh_rows) << "x"
              << static_cast<int>(opts.mesh_cols) << "\n\n";

    NoCBenchmarkHarness harness(config);

    NoCBenchmarkHarness::FullResults results;
    results.noc_type = opts.noc_type;
    results.noc_config = config.noc_config;

    if (opts.run_all || opts.run_micro) {
        std::cout << "Running microbenchmarks...\n";
        results.micro = harness.run_microbenchmarks();
    }

    if (opts.run_all || opts.run_patterns) {
        std::cout << "Running pattern benchmarks...\n";
        results.pattern = harness.run_pattern_benchmarks();
    }

    if (opts.run_all || opts.run_operators) {
        std::cout << "Running operator benchmarks...\n";
        results.operators = harness.run_operator_benchmarks();
    }

    // Run traced benchmark for visualization
    if (tracer) {
        run_traced_benchmark(opts, tracer);
        export_trace_if_needed(tracer, opts);
    }

    // Generate output
    std::string output;
    if (opts.output_format == "markdown") {
        output = NoCBenchmarkReporter::generate_markdown_report(results);
    } else if (opts.output_format == "csv") {
        output = NoCBenchmarkReporter::generate_csv(results);
    } else if (opts.output_format == "json") {
        output = NoCBenchmarkReporter::generate_json(results);
    } else {
        output = NoCBenchmarkReporter::generate_summary(results);
    }

    if (!opts.output_file.empty()) {
        std::ofstream file(opts.output_file);
        file << output;
        std::cout << "\nResults written to: " << opts.output_file << "\n";
    } else {
        std::cout << "\n" << output;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Options opts = parse_args(argc, argv);

    if (opts.help) {
        print_usage(argv[0]);
        return 0;
    }

    try {
        if (!opts.operator_name.empty()) {
            run_single_operator(opts);
        } else if (opts.run_compare) {
            run_comparison(opts);
        } else {
            run_full_benchmarks(opts);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
