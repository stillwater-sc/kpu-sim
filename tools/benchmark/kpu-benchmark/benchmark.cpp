/**
 * @file benchmark.cpp
 * @brief KPU Benchmark Suite - Standalone CLI tool for v0.3.1
 *
 * This tool provides comprehensive benchmarking capabilities for the KPU simulator.
 * It supports:
 *   - Matmul sweeps (standard, extended, quick)
 *   - Tile sensitivity analysis
 *   - Roofline analysis
 *   - JSON/CSV output for integration with analysis tools
 *
 * Usage:
 *   kpu-benchmark [options] <command>
 *
 * Commands:
 *   sweep      - Run matmul size sweep
 *   tiles      - Run tile sensitivity analysis
 *   roofline   - Run roofline analysis
 *   single     - Run single matmul benchmark
 *   all        - Run all benchmarks
 *
 * @version 0.3.1
 */

#include <sw/benchmark/benchmark.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>

using namespace sw::benchmark;
using namespace sw::kpu;

namespace {

// ANSI color codes for terminal output
constexpr const char* RESET = "\033[0m";
constexpr const char* BOLD = "\033[1m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* CYAN = "\033[36m";
constexpr const char* YELLOW = "\033[33m";

struct Options {
    std::string command = "help";
    std::string output_format = "table";  // table, csv, json
    std::string output_file;
    Size min_size = 64;
    Size max_size = 4096;
    Size matmul_m = 1024;
    Size matmul_n = 1024;
    Size matmul_k = 1024;
    bool extended = false;
    bool quick = false;
    bool verbose = false;
    bool color = true;
};

void print_usage(const char* program) {
    std::cout << BOLD << "KPU Benchmark Suite v0.3.1" << RESET << "\n\n";
    std::cout << "Usage: " << program << " [options] <command>\n\n";
    std::cout << BOLD << "Commands:" << RESET << "\n";
    std::cout << "  sweep      Run matmul size sweep (64 to max-size)\n";
    std::cout << "  tiles      Run tile sensitivity analysis\n";
    std::cout << "  roofline   Run roofline analysis\n";
    std::cout << "  single     Run single matmul benchmark\n";
    std::cout << "  all        Run all benchmarks\n";
    std::cout << "  help       Show this help message\n";
    std::cout << "\n";
    std::cout << BOLD << "Options:" << RESET << "\n";
    std::cout << "  -o, --output <file>     Output file (default: stdout)\n";
    std::cout << "  -f, --format <format>   Output format: table, csv, json (default: table)\n";
    std::cout << "  --min-size <n>          Minimum size for sweep (default: 64)\n";
    std::cout << "  --max-size <n>          Maximum size for sweep (default: 4096)\n";
    std::cout << "  -m <n>                  M dimension for single benchmark (default: 1024)\n";
    std::cout << "  -n <n>                  N dimension for single benchmark (default: 1024)\n";
    std::cout << "  -k <n>                  K dimension for single benchmark (default: 1024)\n";
    std::cout << "  --extended              Use extended sweep (64 to 16384)\n";
    std::cout << "  --quick                 Use quick sweep (64 to 1024)\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  --no-color              Disable color output\n";
    std::cout << "\n";
    std::cout << BOLD << "Examples:" << RESET << "\n";
    std::cout << "  " << program << " sweep --extended          # Extended matmul sweep\n";
    std::cout << "  " << program << " tiles -f json -o tiles.json # Tile analysis to JSON\n";
    std::cout << "  " << program << " single -m 2048 -n 2048 -k 2048\n";
    std::cout << "  " << program << " all -f csv -o results.csv  # All benchmarks to CSV\n";
}

Options parse_args(int argc, char* argv[]) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            opts.command = "help";
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) opts.output_file = argv[++i];
        } else if (arg == "-f" || arg == "--format") {
            if (i + 1 < argc) opts.output_format = argv[++i];
        } else if (arg == "--min-size") {
            if (i + 1 < argc) opts.min_size = std::stoul(argv[++i]);
        } else if (arg == "--max-size") {
            if (i + 1 < argc) opts.max_size = std::stoul(argv[++i]);
        } else if (arg == "-m") {
            if (i + 1 < argc) opts.matmul_m = std::stoul(argv[++i]);
        } else if (arg == "-n") {
            if (i + 1 < argc) opts.matmul_n = std::stoul(argv[++i]);
        } else if (arg == "-k") {
            if (i + 1 < argc) opts.matmul_k = std::stoul(argv[++i]);
        } else if (arg == "--extended") {
            opts.extended = true;
        } else if (arg == "--quick") {
            opts.quick = true;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--no-color") {
            opts.color = false;
        } else if (arg[0] != '-') {
            opts.command = arg;
        }
    }

    return opts;
}

void output_suite(const BenchmarkSuite& suite, const Options& opts,
                  std::ostream& out) {
    if (opts.output_format == "json") {
        out << suite.to_json() << std::endl;
    } else if (opts.output_format == "csv") {
        out << suite.to_csv();
    } else {
        out << suite.summary_table() << std::endl;
    }
}

void output_result(const BenchmarkResult& result, const Options& opts,
                   std::ostream& out) {
    if (opts.output_format == "json") {
        out << result.to_json() << std::endl;
    } else if (opts.output_format == "csv") {
        out << "name,config,cycles,flops,gflops,efficiency,arithmetic_intensity,bottleneck\n";
        out << result.name << "," << result.config << "," << result.cycles << ","
            << result.flops << "," << result.gflops << "," << result.compute_efficiency << ","
            << result.arithmetic_intensity << "," << result.bottleneck() << "\n";
    } else {
        out << result.to_detailed_string() << std::endl;
    }
}

int run_sweep(const Options& opts, std::ostream& out) {
    BenchmarkHarness harness;

    Size min_size = opts.min_size;
    Size max_size = opts.max_size;

    if (opts.extended) {
        max_size = 16384;
    } else if (opts.quick) {
        max_size = 1024;
    }

    if (opts.verbose) {
        std::cerr << "Running matmul sweep from " << min_size << " to " << max_size << "...\n";
    }

    auto suite = harness.sweep_matmul_square(min_size, max_size, 2);
    suite.name = "matmul_sweep";
    suite.description = "Square matrix multiplication sweep from " +
                        std::to_string(min_size) + " to " + std::to_string(max_size);

    output_suite(suite, opts, out);

    auto best = suite.best_by_gflops();
    if (best && opts.verbose) {
        std::cerr << "Best performance: " << best->config << " at "
                  << best->gflops << " GFLOPS (" << (best->compute_efficiency * 100)
                  << "% efficiency)\n";
    }

    return 0;
}

int run_tiles(const Options& opts, std::ostream& out) {
    BenchmarkHarness harness;

    Size M = opts.matmul_m;
    Size N = opts.matmul_n;
    Size K = opts.matmul_k;

    if (opts.verbose) {
        std::cerr << "Running tile sensitivity analysis for " << M << "x" << N << "x" << K << "...\n";
    }

    std::vector<std::tuple<Size, Size, Size>> tile_sizes = {
        {16, 16, 16},
        {32, 32, 32},
        {32, 32, 64},
        {64, 64, 64},
        {64, 64, 128},
        {128, 128, 128},
        {32, 64, 32},
        {64, 128, 64},
    };

    auto suite = harness.sweep_tile_sizes(M, N, K, tile_sizes);
    suite.name = "tile_sensitivity";
    suite.description = "Tile sensitivity analysis for " + std::to_string(M) + "x" +
                        std::to_string(N) + "x" + std::to_string(K);

    output_suite(suite, opts, out);

    auto best = suite.best_by_efficiency();
    if (best && opts.verbose) {
        std::cerr << "Best tile config: " << best->Ti << "x" << best->Tj << "x" << best->Tk
                  << " at " << (best->compute_efficiency * 100) << "% efficiency\n";
    }

    return 0;
}

int run_roofline(const Options& opts, std::ostream& out) {
    BenchmarkHarness harness;

    if (opts.verbose) {
        std::cerr << "Running roofline analysis...\n";
    }

    auto suite = harness.sweep_matmul_square(64, opts.extended ? 8192 : 2048, 2);
    suite.name = "roofline_analysis";
    suite.description = "Roofline data for KPU simulator";

    if (opts.output_format == "json" || opts.output_format == "csv") {
        output_suite(suite, opts, out);
    } else {
        out << harness.generate_roofline_data(suite);
    }

    return 0;
}

int run_single(const Options& opts, std::ostream& out) {
    BenchmarkHarness harness;

    Size M = opts.matmul_m;
    Size N = opts.matmul_n;
    Size K = opts.matmul_k;

    if (opts.verbose) {
        std::cerr << "Benchmarking matmul " << M << "x" << N << "x" << K << "...\n";
    }

    auto result = harness.benchmark_matmul(M, N, K);

    output_result(result, opts, out);

    return 0;
}

int run_all(const Options& opts, std::ostream& out) {
    if (opts.verbose) {
        std::cerr << "Running all benchmarks...\n";
    }

    // For JSON output, wrap in a top-level object
    if (opts.output_format == "json") {
        out << "{\n";
        out << "  \"version\": \"0.3.1\",\n";
        out << "  \"benchmarks\": {\n";
    }

    // Run sweep
    if (opts.verbose) std::cerr << "\n=== Matmul Sweep ===\n";
    BenchmarkHarness harness;
    auto sweep = harness.sweep_matmul_square(64, opts.quick ? 1024 : 4096, 2);
    sweep.name = "matmul_sweep";

    if (opts.output_format == "json") {
        out << "    \"sweep\": ";
        // Indent the JSON
        std::string json = sweep.to_json();
        for (char c : json) {
            out << c;
            if (c == '\n') out << "    ";
        }
        out << ",\n";
    } else if (opts.output_format == "csv") {
        out << "# Matmul Sweep\n";
        out << sweep.to_csv() << "\n";
    } else {
        out << "=== Matmul Sweep ===\n";
        out << sweep.summary_table() << "\n";
    }

    // Run tile sensitivity
    if (opts.verbose) std::cerr << "\n=== Tile Sensitivity ===\n";
    std::vector<std::tuple<Size, Size, Size>> tile_sizes = {
        {32, 32, 32}, {64, 64, 64}, {128, 128, 128}, {32, 64, 32}
    };
    auto tiles = harness.sweep_tile_sizes(512, 512, 512, tile_sizes);
    tiles.name = "tile_sensitivity";

    if (opts.output_format == "json") {
        out << "    \"tiles\": ";
        std::string json = tiles.to_json();
        for (char c : json) {
            out << c;
            if (c == '\n') out << "    ";
        }
        out << "\n";
    } else if (opts.output_format == "csv") {
        out << "# Tile Sensitivity\n";
        out << tiles.to_csv() << "\n";
    } else {
        out << "=== Tile Sensitivity (512x512x512) ===\n";
        out << tiles.summary_table() << "\n";
    }

    if (opts.output_format == "json") {
        out << "  }\n";
        out << "}\n";
    }

    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    auto opts = parse_args(argc, argv);

    // Disable color if not a terminal or explicitly disabled
    if (!isatty(fileno(stdout)) || !opts.color) {
        // Color codes become empty strings - we just won't use them
    }

    // Set up output stream
    std::ofstream file_out;
    std::ostream* out = &std::cout;

    if (!opts.output_file.empty()) {
        file_out.open(opts.output_file);
        if (!file_out) {
            std::cerr << "Error: Could not open output file: " << opts.output_file << "\n";
            return 1;
        }
        out = &file_out;
    }

    // Dispatch to command
    if (opts.command == "help") {
        print_usage(argv[0]);
        return 0;
    } else if (opts.command == "sweep") {
        return run_sweep(opts, *out);
    } else if (opts.command == "tiles") {
        return run_tiles(opts, *out);
    } else if (opts.command == "roofline") {
        return run_roofline(opts, *out);
    } else if (opts.command == "single") {
        return run_single(opts, *out);
    } else if (opts.command == "all") {
        return run_all(opts, *out);
    } else {
        std::cerr << "Unknown command: " << opts.command << "\n";
        std::cerr << "Run with --help for usage information.\n";
        return 1;
    }
}
