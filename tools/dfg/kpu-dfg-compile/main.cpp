// kpu-dfg-compile - BlockMover Compiler
// Compile a scheduled DFG to BlockMover programs

#include <iostream>
#include <string>
#include <cstdlib>

#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "sw/kpu/dataflow/block_mover_compiler.hpp"
#include "common/dfg_json.hpp"
#include "common/schedule_json.hpp"
#include "common/compiled_json.hpp"

using namespace sw::kpu::dataflow;
using namespace kpu::dfg::json;

struct Args {
    std::string input_file;
    std::string output_file;
    bool emit_waits = true;
    bool sync_sends = false;
    bool emit_barriers = true;
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;
    bool help = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i scheduled.json -o programs.json [options]\n\n"
              << "Compile a scheduled DFG to BlockMover programs.\n\n"
              << "Options:\n"
              << "  -i, --input FILE       Input scheduled JSON file (required)\n"
              << "  -o, --output FILE      Output programs JSON file (required)\n"
              << "  --no-waits             Don't emit WAIT_UNTIL_CYCLE commands\n"
              << "  --sync-sends           Add WAIT_DELIVERY after sends\n"
              << "  --no-barriers          Don't emit BARRIER commands\n"
              << "  --mesh ROWSxCOLS       Mesh dimensions (default: from input)\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog << " -i scheduled.json -o programs.json\n"
              << "  " << prog << " -i scheduled.json -o programs.json --sync-sends\n";
}

bool parse_mesh(const std::string& s, uint8_t& rows, uint8_t& cols) {
    size_t pos = s.find('x');
    if (pos == std::string::npos) return false;
    try {
        rows = static_cast<uint8_t>(std::stoul(s.substr(0, pos)));
        cols = static_cast<uint8_t>(std::stoul(s.substr(pos + 1)));
        return true;
    } catch (...) {
        return false;
    }
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--no-waits") {
            args.emit_waits = false;
        } else if (arg == "--sync-sends") {
            args.sync_sends = true;
        } else if (arg == "--no-barriers") {
            args.emit_barriers = false;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) args.input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.output_file = argv[++i];
        } else if (arg == "--mesh") {
            if (i + 1 < argc) {
                if (!parse_mesh(argv[++i], args.mesh_rows, args.mesh_cols)) {
                    std::cerr << "Error: Invalid mesh format. Use ROWSxCOLS\n";
                    std::exit(1);
                }
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.input_file.empty()) {
        std::cerr << "Error: Input file is required. Use -i or --input.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    if (args.output_file.empty()) {
        std::cerr << "Error: Output file is required. Use -o or --output.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    // Read input (scheduled DFG)
    ScheduleWithDFG input;
    try {
        input = read_schedule_json(args.input_file);
        if (!input.has_embedded_dfg) {
            std::cerr << "Error: Input file must contain embedded DFG.\n";
            return 1;
        }
        if (args.verbose) {
            std::cout << "Read scheduled DFG from: " << args.input_file << "\n"
                      << "  DFG nodes: " << input.dfg.num_nodes() << "\n"
                      << "  Scheduled nodes: " << input.schedule.nodes.size() << "\n"
                      << "  Makespan: " << input.schedule.makespan << " cycles\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading input: " << e.what() << "\n";
        return 1;
    }

    // Get mesh dimensions from DFG timing model
    const auto& timing = input.dfg.timing_model();
    uint8_t mesh_rows = args.mesh_rows;
    uint8_t mesh_cols = args.mesh_cols;
    if (timing.mesh_rows > 0 && timing.mesh_cols > 0) {
        mesh_rows = timing.mesh_rows;
        mesh_cols = timing.mesh_cols;
    }

    // Configure compiler
    BlockMoverCompiler::Config config;
    config.mesh_rows = mesh_rows;
    config.mesh_cols = mesh_cols;
    config.emit_cycle_waits = args.emit_waits;
    config.sync_sends = args.sync_sends;
    config.emit_barriers = args.emit_barriers;

    if (args.verbose) {
        std::cout << "Compiling with:\n"
                  << "  Mesh: " << static_cast<int>(mesh_rows) << "x"
                  << static_cast<int>(mesh_cols) << "\n"
                  << "  Emit waits: " << (args.emit_waits ? "yes" : "no") << "\n"
                  << "  Sync sends: " << (args.sync_sends ? "yes" : "no") << "\n"
                  << "  Emit barriers: " << (args.emit_barriers ? "yes" : "no") << "\n";
    }

    // Compile
    BlockMoverCompiler compiler(config);
    CompiledSchedule compiled = compiler.compile(input.dfg, input.schedule);

    if (args.verbose) {
        std::cout << "Compilation complete:\n"
                  << "  Total commands: " << compiled.stats.total_commands << "\n"
                  << "  L3 transfers: " << compiled.stats.total_l3_transfers << "\n"
                  << "  L2 transfers: " << compiled.stats.total_l2_transfers << "\n"
                  << "  Compute ops: " << compiled.stats.total_compute_ops << "\n"
                  << "  Estimated cycles: " << compiled.estimated_cycles << "\n";

        // Print per-L3 command counts
        std::cout << "  Per-L3 programs:\n";
        for (uint8_t l3 = 0; l3 < 16; ++l3) {
            if (!compiled.program(l3).empty()) {
                std::cout << "    L3[" << static_cast<int>(l3) << "]: "
                          << compiled.program(l3).size() << " commands\n";
            }
        }
    }

    // Write output
    try {
        write_compiled_json(compiled, args.output_file);
        if (args.verbose) {
            std::cout << "Wrote programs to: " << args.output_file << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing output: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
