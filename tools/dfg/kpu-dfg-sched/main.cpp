// kpu-dfg-sched - DFG Scheduler
// Schedule nodes in a DFG using various algorithms

#include <iostream>
#include <string>
#include <cstdlib>

#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "common/dfg_json.hpp"
#include "common/schedule_json.hpp"

using namespace sw::kpu::dataflow;
using namespace kpu::dfg::json;

struct Args {
    std::string input_file;
    std::string output_file;
    std::string algorithm = "ASAP";
    uint8_t l3_concurrency = 2;
    uint8_t dma_channels = 8;
    bool validate = false;
    bool help = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i input.json -o output.json [options]\n\n"
              << "Schedule nodes in a Data Flow Graph.\n\n"
              << "Options:\n"
              << "  -i, --input FILE       Input DFG JSON file (required)\n"
              << "  -o, --output FILE      Output scheduled JSON file (required)\n"
              << "  --algorithm ALG        Scheduling algorithm: ASAP (default), ALAP, LIST\n"
              << "  --l3-concurrency N     Max concurrent ops per L3 (default: 2)\n"
              << "  --dma-channels N       Number of DMA channels (default: 8)\n"
              << "  --validate             Validate schedule after generation\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog << " -i dfg.json -o scheduled.json --algorithm ASAP\n"
              << "  " << prog << " -i dfg.json -o scheduled.json --algorithm LIST --validate\n";
}

DFGScheduler::Algorithm parse_algorithm(const std::string& s) {
    if (s == "ASAP") return DFGScheduler::Algorithm::ASAP;
    if (s == "ALAP") return DFGScheduler::Algorithm::ALAP;
    if (s == "LIST" || s == "LIST_SCHEDULING") return DFGScheduler::Algorithm::LIST_SCHEDULING;
    if (s == "CRITICAL_PATH") return DFGScheduler::Algorithm::CRITICAL_PATH;
    std::cerr << "Unknown algorithm: " << s << ". Using ASAP.\n";
    return DFGScheduler::Algorithm::ASAP;
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--validate") {
            args.validate = true;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) args.input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.output_file = argv[++i];
        } else if (arg == "--algorithm") {
            if (i + 1 < argc) args.algorithm = argv[++i];
        } else if (arg == "--l3-concurrency") {
            if (i + 1 < argc) args.l3_concurrency = static_cast<uint8_t>(std::stoul(argv[++i]));
        } else if (arg == "--dma-channels") {
            if (i + 1 < argc) args.dma_channels = static_cast<uint8_t>(std::stoul(argv[++i]));
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

    // Read input DFG
    TileDataFlowGraph dfg;
    try {
        dfg = read_dfg_json(args.input_file);
        if (args.verbose) {
            std::cout << "Read DFG from: " << args.input_file << "\n"
                      << "  Nodes: " << dfg.num_nodes() << "\n"
                      << "  Edges: " << dfg.num_edges() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading input: " << e.what() << "\n";
        return 1;
    }

    // Configure scheduler
    DFGScheduler::Config config;
    config.algorithm = parse_algorithm(args.algorithm);
    config.num_l3_tiles = dfg.timing_model().mesh_rows * dfg.timing_model().mesh_cols;
    config.max_concurrent_dma = args.dma_channels;
    config.max_concurrent_l3_transfers = args.l3_concurrency;

    if (args.verbose) {
        std::cout << "Scheduling with:\n"
                  << "  Algorithm: " << args.algorithm << "\n"
                  << "  L3 tiles: " << static_cast<int>(config.num_l3_tiles) << "\n"
                  << "  L3 concurrency: " << static_cast<int>(args.l3_concurrency) << "\n"
                  << "  DMA channels: " << static_cast<int>(args.dma_channels) << "\n";
    }

    // Schedule
    DFGScheduler scheduler(config);
    DFGSchedule schedule = scheduler.schedule(dfg);

    if (args.verbose) {
        std::cout << "Schedule generated:\n"
                  << "  Makespan: " << schedule.makespan << " cycles\n"
                  << "  Scheduled nodes: " << schedule.nodes.size() << "\n";
    }

    // Validate if requested
    if (args.validate) {
        bool valid = schedule.validate(dfg);
        if (valid) {
            if (args.verbose) {
                std::cout << "Schedule validation: PASSED\n";
            }
        } else {
            std::cerr << "Schedule validation: FAILED\n";
            return 1;
        }
    }

    // Write output
    try {
        write_schedule_json(schedule, dfg, args.output_file, true);
        if (args.verbose) {
            std::cout << "Wrote schedule to: " << args.output_file << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing output: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
