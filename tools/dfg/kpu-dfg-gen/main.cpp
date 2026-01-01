// kpu-dfg-gen - DFG Generator
// Generate Data Flow Graphs from templates or specifications

#include <iostream>
#include <string>
#include <cstdlib>

#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "common/dfg_json.hpp"

using namespace sw::kpu::dataflow;
using namespace kpu::dfg::json;

struct Args {
    std::string output_file;
    std::string template_name = "matmul";
    std::string dataflow = "output-stationary";
    uint32_t M = 1024;
    uint32_t N = 1024;
    uint32_t K = 1024;
    uint32_t m_tiles = 4;
    uint32_t n_tiles = 4;
    uint32_t k_tiles = 4;
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;
    bool help = false;
    bool verbose = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] -o output.json\n\n"
              << "Generate a Data Flow Graph from a template.\n\n"
              << "Options:\n"
              << "  -o, --output FILE      Output JSON file (required)\n"
              << "  --template NAME        Template to use: matmul (default)\n"
              << "  --dataflow PATTERN     Dataflow pattern: output-stationary (default),\n"
              << "                         weight-stationary, input-stationary\n"
              << "  -M SIZE                Matrix M dimension (default: 1024)\n"
              << "  -N SIZE                Matrix N dimension (default: 1024)\n"
              << "  -K SIZE                Matrix K dimension (default: 1024)\n"
              << "  --tiles MxNxK          Tiling in format MxNxK (default: 4x4x4)\n"
              << "  --mesh ROWSxCOLS       Mesh dimensions (default: 4x4)\n"
              << "  -v, --verbose          Verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nExamples:\n"
              << "  " << prog << " --template matmul -M 1024 -N 1024 -K 1024 --tiles 4x4x4 -o dfg.json\n"
              << "  " << prog << " -M 512 -N 512 -K 512 --dataflow weight-stationary -o ws_dfg.json\n";
}

bool parse_tiles(const std::string& s, uint32_t& m, uint32_t& n, uint32_t& k) {
    size_t pos1 = s.find('x');
    if (pos1 == std::string::npos) return false;
    size_t pos2 = s.find('x', pos1 + 1);
    if (pos2 == std::string::npos) return false;

    try {
        m = static_cast<uint32_t>(std::stoul(s.substr(0, pos1)));
        n = static_cast<uint32_t>(std::stoul(s.substr(pos1 + 1, pos2 - pos1 - 1)));
        k = static_cast<uint32_t>(std::stoul(s.substr(pos2 + 1)));
        return true;
    } catch (...) {
        return false;
    }
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
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) args.output_file = argv[++i];
        } else if (arg == "--template") {
            if (i + 1 < argc) args.template_name = argv[++i];
        } else if (arg == "--dataflow") {
            if (i + 1 < argc) args.dataflow = argv[++i];
        } else if (arg == "-M") {
            if (i + 1 < argc) args.M = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-N") {
            if (i + 1 < argc) args.N = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-K") {
            if (i + 1 < argc) args.K = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--tiles") {
            if (i + 1 < argc) {
                if (!parse_tiles(argv[++i], args.m_tiles, args.n_tiles, args.k_tiles)) {
                    std::cerr << "Error: Invalid tiles format. Use MxNxK (e.g., 4x4x4)\n";
                    std::exit(1);
                }
            }
        } else if (arg == "--mesh") {
            if (i + 1 < argc) {
                if (!parse_mesh(argv[++i], args.mesh_rows, args.mesh_cols)) {
                    std::cerr << "Error: Invalid mesh format. Use ROWSxCOLS (e.g., 4x4)\n";
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

TileDataFlowGraph build_systolic_matmul_dfg(const Args& args) {
    TileDataFlowGraph dfg;

    // Configure timing model
    TimingModel timing;
    timing.mesh_rows = args.mesh_rows;
    timing.mesh_cols = args.mesh_cols;
    timing.dma_bandwidth_bytes_per_cycle = 64;
    timing.l3_bandwidth_bytes_per_cycle = 64;
    timing.store_and_forward = false;  // Flit-pipelined for lower latency
    dfg.set_timing_model(timing);

    const uint32_t tile_M = args.M / args.m_tiles;
    const uint32_t tile_N = args.N / args.n_tiles;
    const uint32_t tile_K = args.K / args.k_tiles;

    // Track node IDs for dependency linking
    std::map<std::tuple<int, int, int>, size_t> dma_load_a;  // (m, k) -> node_id
    std::map<std::tuple<int, int, int>, size_t> dma_load_b;  // (k, n) -> node_id
    std::map<std::tuple<int, int, int>, size_t> l3_xfer_a;   // (m, n, k) -> node_id
    std::map<std::tuple<int, int, int>, size_t> l3_xfer_b;   // (m, n, k) -> node_id
    std::map<std::tuple<int, int, int>, size_t> matmul_ops;  // (m, n, k) -> node_id

    // Phase 1: DMA loads for A tiles (into column 0)
    for (uint32_t m = 0; m < args.m_tiles; ++m) {
        for (uint32_t k = 0; k < args.k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = m;
            tile.k_tile = k;
            tile.size = tile_M * tile_K * sizeof(float);

            uint8_t dest_l3 = m * args.mesh_cols;  // Column 0
            size_t node_id = dfg.add_dma_load(tile, dest_l3);
            dma_load_a[{m, k, 0}] = node_id;

            // K-step ordering: k>0 depends on k-1
            if (k > 0) {
                dfg.add_edge(dma_load_a[{m, k-1, 0}], node_id, DFEdgeType::CONTROL);
            }
        }
    }

    // Phase 1: DMA loads for B tiles (into row 0)
    for (uint32_t k = 0; k < args.k_tiles; ++k) {
        for (uint32_t n = 0; n < args.n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.k_tile = k;
            tile.n_tile = n;
            tile.size = tile_K * tile_N * sizeof(float);

            uint8_t dest_l3 = n;  // Row 0
            size_t node_id = dfg.add_dma_load(tile, dest_l3);
            dma_load_b[{k, n, 0}] = node_id;

            // K-step ordering
            if (k > 0) {
                dfg.add_edge(dma_load_b[{k-1, n, 0}], node_id, DFEdgeType::CONTROL);
            }
        }
    }

    // Phase 2: L3 transfers and compute
    for (uint32_t k = 0; k < args.k_tiles; ++k) {
        // A tiles flow East
        for (uint32_t m = 0; m < args.m_tiles; ++m) {
            for (uint32_t n = 0; n < args.n_tiles; ++n) {
                TileDescriptor a_tile;
                a_tile.tensor = TensorId::A;
                a_tile.m_tile = m;
                a_tile.k_tile = k;
                a_tile.size = tile_M * tile_K * sizeof(float);

                uint8_t src_l3 = m * args.mesh_cols + (n > 0 ? n - 1 : 0);
                uint8_t dst_l3 = m * args.mesh_cols + n;

                if (n == 0) {
                    // First column: DMA load is the source
                    l3_xfer_a[{m, n, k}] = dma_load_a[{m, k, 0}];
                } else {
                    // Transfer from previous column
                    size_t xfer_id = dfg.add_l3_transfer(a_tile, src_l3, dst_l3);
                    l3_xfer_a[{m, n, k}] = xfer_id;
                    dfg.add_edge(l3_xfer_a[{m, n-1, k}], xfer_id, DFEdgeType::DATA);
                }
            }
        }

        // B tiles flow South
        for (uint32_t n = 0; n < args.n_tiles; ++n) {
            for (uint32_t m = 0; m < args.m_tiles; ++m) {
                TileDescriptor b_tile;
                b_tile.tensor = TensorId::B;
                b_tile.k_tile = k;
                b_tile.n_tile = n;
                b_tile.size = tile_K * tile_N * sizeof(float);

                uint8_t src_l3 = (m > 0 ? m - 1 : 0) * args.mesh_cols + n;
                uint8_t dst_l3 = m * args.mesh_cols + n;

                if (m == 0) {
                    // First row: DMA load is the source
                    l3_xfer_b[{m, n, k}] = dma_load_b[{k, n, 0}];
                } else {
                    // Transfer from previous row
                    size_t xfer_id = dfg.add_l3_transfer(b_tile, src_l3, dst_l3);
                    l3_xfer_b[{m, n, k}] = xfer_id;
                    dfg.add_edge(l3_xfer_b[{m-1, n, k}], xfer_id, DFEdgeType::DATA);
                }
            }
        }

        // Matmul operations
        for (uint32_t m = 0; m < args.m_tiles; ++m) {
            for (uint32_t n = 0; n < args.n_tiles; ++n) {
                TileDescriptor a_tile, b_tile, c_tile;
                a_tile.tensor = TensorId::A;
                a_tile.m_tile = m;
                a_tile.k_tile = k;
                b_tile.tensor = TensorId::B;
                b_tile.k_tile = k;
                b_tile.n_tile = n;
                c_tile.tensor = TensorId::C;
                c_tile.m_tile = m;
                c_tile.n_tile = n;

                uint8_t l3_id = m * args.mesh_cols + n;
                size_t matmul_id = dfg.add_matmul(l3_id, tile_M, tile_N, tile_K, a_tile, b_tile, c_tile);
                matmul_ops[{m, n, k}] = matmul_id;

                // Dependencies on A and B arrival
                dfg.add_edge(l3_xfer_a[{m, n, k}], matmul_id, DFEdgeType::DATA);
                dfg.add_edge(l3_xfer_b[{m, n, k}], matmul_id, DFEdgeType::DATA);

                // K-step ordering for accumulation
                if (k > 0) {
                    dfg.add_edge(matmul_ops[{m, n, k-1}], matmul_id, DFEdgeType::DATA);
                }
            }
        }
    }

    // Phase 3: DMA stores for C tiles
    for (uint32_t m = 0; m < args.m_tiles; ++m) {
        for (uint32_t n = 0; n < args.n_tiles; ++n) {
            TileDescriptor c_tile;
            c_tile.tensor = TensorId::C;
            c_tile.m_tile = m;
            c_tile.n_tile = n;
            c_tile.size = tile_M * tile_N * sizeof(float);

            uint8_t l3_id = m * args.mesh_cols + n;
            size_t store_id = dfg.add_dma_store(c_tile, l3_id);

            // Depends on final K-step matmul
            dfg.add_edge(matmul_ops[{m, n, args.k_tiles - 1}], store_id, DFEdgeType::DATA);
        }
    }

    return dfg;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.output_file.empty()) {
        std::cerr << "Error: Output file is required. Use -o or --output.\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    if (args.verbose) {
        std::cout << "Generating DFG:\n"
                  << "  Template: " << args.template_name << "\n"
                  << "  Dataflow: " << args.dataflow << "\n"
                  << "  Matrix: " << args.M << "x" << args.K << " * "
                  << args.K << "x" << args.N << "\n"
                  << "  Tiles: " << args.m_tiles << "x" << args.n_tiles
                  << "x" << args.k_tiles << "\n"
                  << "  Mesh: " << static_cast<int>(args.mesh_rows) << "x"
                  << static_cast<int>(args.mesh_cols) << "\n";
    }

    TileDataFlowGraph dfg;

    if (args.template_name == "matmul") {
        dfg = build_systolic_matmul_dfg(args);
    } else {
        std::cerr << "Error: Unknown template: " << args.template_name << "\n";
        return 1;
    }

    // Compute earliest starts for analysis
    dfg.compute_earliest_starts();

    if (args.verbose) {
        std::cout << "Generated DFG:\n"
                  << "  Nodes: " << dfg.num_nodes() << "\n"
                  << "  Edges: " << dfg.num_edges() << "\n"
                  << "  Critical path: " << dfg.critical_path_length() << " cycles\n";
    }

    try {
        write_dfg_json(dfg, args.output_file);
        if (args.verbose) {
            std::cout << "Wrote DFG to: " << args.output_file << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing output: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
