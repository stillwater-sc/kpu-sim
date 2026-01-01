// kpu-dfg-analyze - Analysis Tool
// Analyze and validate DFG, schedule, or compiled programs

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cstdlib>

#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "common/dfg_json.hpp"
#include "common/schedule_json.hpp"
#include "common/compiled_json.hpp"

using namespace sw::kpu::dataflow;
using namespace kpu::dfg::json;

struct Args {
    std::string input_file;
    bool show_stats = false;
    bool show_critical_path = false;
    bool show_utilization = false;
    bool validate = false;
    bool check_order = false;
    bool help = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i input.json [options]\n\n"
              << "Analyze and validate DFG, schedule, or compiled programs.\n\n"
              << "Options:\n"
              << "  -i, --input FILE       Input JSON file (required)\n"
              << "  --stats                Show statistics\n"
              << "  --critical-path        Show critical path analysis\n"
              << "  --utilization          Show per-L3 utilization\n"
              << "  --validate             Validate schedule/programs\n"
              << "  --check-order          Check systolic ordering\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog << " -i dfg.json --stats\n"
              << "  " << prog << " -i scheduled.json --validate --critical-path\n"
              << "  " << prog << " -i programs.json --stats --check-order\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--stats") {
            args.show_stats = true;
        } else if (arg == "--critical-path") {
            args.show_critical_path = true;
        } else if (arg == "--utilization") {
            args.show_utilization = true;
        } else if (arg == "--validate") {
            args.validate = true;
        } else if (arg == "--check-order") {
            args.check_order = true;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) args.input_file = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }

    // If no specific analysis requested, show stats
    if (!args.show_stats && !args.show_critical_path &&
        !args.show_utilization && !args.validate && !args.check_order) {
        args.show_stats = true;
    }

    return args;
}

void analyze_dfg_stats(const TileDataFlowGraph& dfg) {
    std::cout << "\n=== DFG Statistics ===\n\n";

    std::cout << "Graph Structure:\n";
    std::cout << "  Total nodes: " << dfg.num_nodes() << "\n";
    std::cout << "  Total edges: " << dfg.num_edges() << "\n";
    std::cout << "  Is acyclic: " << (dfg.is_acyclic() ? "yes" : "no") << "\n";

    // Count by type
    std::map<DFNodeType, size_t> type_counts;
    int64_t total_duration = 0;
    for (const auto& node : dfg.nodes()) {
        type_counts[node.type]++;
        total_duration += node.duration;
    }

    std::cout << "\nNode Types:\n";
    for (const auto& [type, count] : type_counts) {
        std::cout << "  " << std::setw(15) << std::left << to_string(type)
                  << ": " << count << "\n";
    }

    std::cout << "\nTiming:\n";
    std::cout << "  Critical path: " << dfg.critical_path_length() << " cycles\n";
    std::cout << "  Total work: " << total_duration << " cycles\n";

    // Parallelism estimate
    double parallelism = static_cast<double>(total_duration) /
                         static_cast<double>(dfg.critical_path_length());
    std::cout << "  Avg parallelism: " << std::fixed << std::setprecision(1)
              << parallelism << "x\n";
}

void analyze_schedule_stats(const DFGSchedule& schedule, const TileDataFlowGraph& dfg) {
    std::cout << "\n=== Schedule Statistics ===\n\n";

    std::cout << "Schedule:\n";
    std::cout << "  Makespan: " << schedule.makespan << " cycles\n";
    std::cout << "  Scheduled nodes: " << schedule.nodes.size() << "\n";

    // Per-L3 statistics
    std::map<uint8_t, int64_t> l3_busy_cycles;
    for (const auto& sn : schedule.nodes) {
        int64_t duration = sn.end_cycle - sn.start_cycle;
        l3_busy_cycles[sn.resource_id] += duration;
    }

    std::cout << "\nPer-L3 Utilization:\n";
    for (const auto& [l3_id, busy] : l3_busy_cycles) {
        double util = 100.0 * static_cast<double>(busy) /
                      static_cast<double>(schedule.makespan);
        std::cout << "  L3[" << std::setw(2) << static_cast<int>(l3_id) << "]: "
                  << std::setw(5) << busy << " cycles ("
                  << std::fixed << std::setprecision(1) << util << "%)\n";
    }
}

void analyze_compiled_stats(const CompiledSchedule& compiled) {
    std::cout << "\n=== Compiled Program Statistics ===\n\n";

    std::cout << "Overall:\n";
    std::cout << "  Total commands: " << compiled.stats.total_commands << "\n";
    std::cout << "  DMA operations: " << compiled.stats.total_dma_ops << "\n";
    std::cout << "  L3 transfers: " << compiled.stats.total_l3_transfers << "\n";
    std::cout << "  L2 transfers: " << compiled.stats.total_l2_transfers << "\n";
    std::cout << "  Compute ops: " << compiled.stats.total_compute_ops << "\n";
    std::cout << "  Triggers: " << compiled.stats.total_triggers << "\n";
    std::cout << "  Barriers: " << compiled.stats.total_barriers << "\n";

    std::cout << "\nTiming Estimates:\n";
    std::cout << "  Total cycles: " << compiled.estimated_cycles << "\n";
    std::cout << "  Compute cycles: " << compiled.compute_cycles << "\n";
    std::cout << "  Data movement: " << compiled.data_movement_cycles << "\n";

    std::cout << "\nPer-L3 Programs:\n";
    for (uint8_t l3 = 0; l3 < 16; ++l3) {
        const auto& prog = compiled.program(l3);
        if (!prog.empty()) {
            // Count command types
            std::map<std::string, size_t> cmd_counts;
            for (const auto& cmd : prog.commands) {
                cmd_counts[sw::kpu::to_string(cmd.op)]++;
            }
            std::cout << "  L3[" << std::setw(2) << static_cast<int>(l3) << "]: "
                      << prog.size() << " commands\n";
        }
    }
}

void show_critical_path(const TileDataFlowGraph& dfg) {
    std::cout << "\n=== Critical Path Analysis ===\n\n";

    // Compute topological order and find critical path
    auto topo = dfg.topological_order();

    // Find longest path to each node
    std::map<size_t, int64_t> dist;
    std::map<size_t, size_t> pred;

    for (size_t id : topo) {
        dist[id] = dfg.node(id).duration;
        pred[id] = id;  // Self if no predecessor

        for (size_t p : dfg.node(id).predecessors) {
            if (dist[p] + dfg.node(id).duration > dist[id]) {
                dist[id] = dist[p] + dfg.node(id).duration;
                pred[id] = p;
            }
        }
    }

    // Find the sink with maximum distance
    size_t end_node = 0;
    int64_t max_dist = 0;
    for (const auto& [id, d] : dist) {
        if (d > max_dist) {
            max_dist = d;
            end_node = id;
        }
    }

    // Trace back the critical path
    std::vector<size_t> path;
    size_t curr = end_node;
    while (pred[curr] != curr) {
        path.push_back(curr);
        curr = pred[curr];
    }
    path.push_back(curr);
    std::reverse(path.begin(), path.end());

    std::cout << "Critical path length: " << max_dist << " cycles\n";
    std::cout << "Critical path nodes: " << path.size() << "\n\n";

    std::cout << "Path:\n";
    int64_t cumulative = 0;
    for (size_t id : path) {
        const auto& node = dfg.node(id);
        cumulative += node.duration;
        std::cout << "  [" << std::setw(4) << cumulative << "] "
                  << node.name << " (" << node.duration << " cycles)\n";
    }
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

    try {
        // Try to determine input type and read appropriately
        // First try compiled, then scheduled, then plain DFG

        // Try compiled first
        try {
            CompiledSchedule compiled = read_compiled_json(args.input_file);
            if (args.verbose) {
                std::cout << "Read compiled programs from: " << args.input_file << "\n";
            }
            if (args.show_stats) {
                analyze_compiled_stats(compiled);
            }
            if (args.validate) {
                std::cout << "\nValidation: Compiled programs validation not yet implemented.\n";
            }
            return 0;
        } catch (...) {
            // Not a compiled file, try next
        }

        // Try scheduled
        try {
            ScheduleWithDFG scheduled = read_schedule_json(args.input_file);
            if (args.verbose) {
                std::cout << "Read scheduled DFG from: " << args.input_file << "\n";
            }
            if (args.show_stats) {
                if (scheduled.has_embedded_dfg) {
                    analyze_dfg_stats(scheduled.dfg);
                }
                analyze_schedule_stats(scheduled.schedule, scheduled.dfg);
            }
            if (args.show_critical_path && scheduled.has_embedded_dfg) {
                show_critical_path(scheduled.dfg);
            }
            if (args.validate && scheduled.has_embedded_dfg) {
                bool valid = scheduled.schedule.validate(scheduled.dfg);
                std::cout << "\nSchedule validation: " << (valid ? "PASSED" : "FAILED") << "\n";
                if (!valid) return 1;
            }
            return 0;
        } catch (...) {
            // Not a scheduled file, try plain DFG
        }

        // Try plain DFG
        TileDataFlowGraph dfg = read_dfg_json(args.input_file);
        if (args.verbose) {
            std::cout << "Read DFG from: " << args.input_file << "\n";
        }
        if (args.show_stats) {
            analyze_dfg_stats(dfg);
        }
        if (args.show_critical_path) {
            show_critical_path(dfg);
        }
        if (args.validate) {
            bool acyclic = dfg.is_acyclic();
            std::cout << "\nDFG validation:\n";
            std::cout << "  Acyclic: " << (acyclic ? "PASSED" : "FAILED") << "\n";
            if (!acyclic) return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
