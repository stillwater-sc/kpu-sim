// kpu-dfg-viz - Visualization Export
// Export DFG/schedule to visualization formats (DOT, Chrome Trace)

#include <iostream>
#include <fstream>
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
    std::string format = "dot";
    bool help = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " -i input.json -o output [options]\n\n"
              << "Export DFG or schedule to visualization formats.\n\n"
              << "Options:\n"
              << "  -i, --input FILE       Input JSON file (DFG or schedule)\n"
              << "  -o, --output FILE      Output file\n"
              << "  --format FORMAT        Output format: dot (default), chrome-trace, mermaid\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nFormats:\n"
              << "  dot           GraphViz DOT format for graph visualization\n"
              << "  chrome-trace  Chrome Trace JSON for timeline visualization\n"
              << "  mermaid       Mermaid diagram syntax\n"
              << "\nExamples:\n"
              << "  " << prog << " -i dfg.json -o graph.dot --format dot\n"
              << "  " << prog << " -i scheduled.json -o timeline.json --format chrome-trace\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) args.input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.output_file = argv[++i];
        } else if (arg == "--format") {
            if (i + 1 < argc) args.format = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::exit(1);
        }
    }

    return args;
}

void export_dot(const TileDataFlowGraph& dfg, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    out << "digraph DFG {\n";
    out << "  rankdir=TB;\n";
    out << "  node [shape=box];\n\n";

    // Define node styles by type
    out << "  // Node styles\n";
    out << "  node [style=filled];\n\n";

    // Output nodes grouped by L3
    const auto& timing = dfg.timing_model();
    for (uint8_t l3 = 0; l3 < timing.mesh_rows * timing.mesh_cols; ++l3) {
        auto l3_nodes = dfg.get_nodes_by_l3(l3);
        if (l3_nodes.empty()) continue;

        out << "  subgraph cluster_L3_" << static_cast<int>(l3) << " {\n";
        out << "    label=\"L3[" << static_cast<int>(l3) << "]\";\n";
        out << "    style=dashed;\n";

        for (size_t node_id : l3_nodes) {
            const auto& node = dfg.node(node_id);
            std::string color;
            switch (node.type) {
                case DFNodeType::DMA_LOAD:
                case DFNodeType::DMA_STORE:
                    color = "lightblue";
                    break;
                case DFNodeType::L3_TRANSFER:
                    color = "lightyellow";
                    break;
                case DFNodeType::MATMUL:
                    color = "lightgreen";
                    break;
                default:
                    color = "white";
            }
            out << "    n" << node_id << " [label=\"" << node.name
                << "\\n" << node.duration << " cycles\""
                << " fillcolor=\"" << color << "\"];\n";
        }
        out << "  }\n\n";
    }

    // Output edges
    out << "  // Edges\n";
    for (const auto& edge : dfg.edges()) {
        std::string style;
        switch (edge.type) {
            case DFEdgeType::DATA:
                style = "solid";
                break;
            case DFEdgeType::CONTROL:
                style = "dashed";
                break;
            default:
                style = "dotted";
        }
        out << "  n" << edge.from_node << " -> n" << edge.to_node
            << " [style=" << style << "];\n";
    }

    out << "}\n";
}

void export_chrome_trace(const TileDataFlowGraph& dfg, const DFGSchedule& schedule,
                          const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    out << "{\n";
    out << "  \"traceEvents\": [\n";

    bool first = true;
    for (const auto& sn : schedule.nodes) {
        if (!first) out << ",\n";
        first = false;

        const auto& node = dfg.node(sn.node_id);
        std::string cat;
        switch (node.type) {
            case DFNodeType::DMA_LOAD:
            case DFNodeType::DMA_STORE:
                cat = "DMA";
                break;
            case DFNodeType::L3_TRANSFER:
                cat = "L3_Transfer";
                break;
            case DFNodeType::MATMUL:
                cat = "Compute";
                break;
            default:
                cat = "Other";
        }

        // Chrome trace format uses microseconds, we use cycles
        // Assume 1 cycle = 1 microsecond for visualization
        int64_t duration = sn.end_cycle - sn.start_cycle;
        if (duration <= 0) duration = 1;

        out << "    {"
            << "\"name\": \"" << node.name << "\", "
            << "\"cat\": \"" << cat << "\", "
            << "\"ph\": \"X\", "  // Complete event
            << "\"ts\": " << sn.start_cycle << ", "
            << "\"dur\": " << duration << ", "
            << "\"pid\": 1, "
            << "\"tid\": " << static_cast<int>(node.l3_id)
            << "}";
    }

    out << "\n  ],\n";
    out << "  \"displayTimeUnit\": \"ns\"\n";
    out << "}\n";
}

void export_mermaid(const TileDataFlowGraph& dfg, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    out << "graph TD\n";

    // Output nodes
    for (const auto& node : dfg.nodes()) {
        std::string shape_start, shape_end;
        switch (node.type) {
            case DFNodeType::DMA_LOAD:
            case DFNodeType::DMA_STORE:
                shape_start = "[(";
                shape_end = ")]";
                break;
            case DFNodeType::MATMUL:
                shape_start = "{{";
                shape_end = "}}";
                break;
            default:
                shape_start = "[";
                shape_end = "]";
        }
        out << "    n" << node.id << shape_start << node.name << shape_end << "\n";
    }

    out << "\n";

    // Output edges
    for (const auto& edge : dfg.edges()) {
        std::string arrow;
        switch (edge.type) {
            case DFEdgeType::DATA:
                arrow = "-->";
                break;
            case DFEdgeType::CONTROL:
                arrow = "-.->";
                break;
            default:
                arrow = "-.->";
        }
        out << "    n" << edge.from_node << " " << arrow << " n" << edge.to_node << "\n";
    }
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.input_file.empty() || args.output_file.empty()) {
        std::cerr << "Error: Input and output files are required.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    try {
        // Try to read as scheduled JSON first (contains both DFG and schedule)
        ScheduleWithDFG scheduled;
        bool has_schedule = false;

        try {
            scheduled = read_schedule_json(args.input_file);
            has_schedule = true;
            if (args.verbose) {
                std::cout << "Read scheduled DFG from: " << args.input_file << "\n";
            }
        } catch (...) {
            // Fall back to plain DFG
            scheduled.dfg = read_dfg_json(args.input_file);
            has_schedule = false;
            if (args.verbose) {
                std::cout << "Read DFG from: " << args.input_file << "\n";
            }
        }

        if (args.format == "dot") {
            export_dot(scheduled.dfg, args.output_file);
        } else if (args.format == "chrome-trace") {
            if (!has_schedule) {
                std::cerr << "Error: Chrome trace format requires scheduled DFG.\n";
                return 1;
            }
            export_chrome_trace(scheduled.dfg, scheduled.schedule, args.output_file);
        } else if (args.format == "mermaid") {
            export_mermaid(scheduled.dfg, args.output_file);
        } else {
            std::cerr << "Error: Unknown format: " << args.format << "\n";
            return 1;
        }

        if (args.verbose) {
            std::cout << "Wrote " << args.format << " to: " << args.output_file << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
