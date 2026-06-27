// ============================================================================
// examples/blas/block_systolic_matmul.cpp
// Block Systolic Matrix Multiply using Stateful BlockMovers
//
// Demonstrates a scalable tiled matmul where:
// - C tiles are distributed across L3 tiles (output stationary)
// - A tiles flow East through the mesh (broadcast along rows)
// - B tiles flow South through the mesh (broadcast along columns)
// - BlockMovers orchestrate all data movement
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include <sw/kpu/models/temporal/datamovement/l3_interconnect.hpp>
#include <sw/kpu/models/temporal/datamovement/stateful_block_mover.hpp>
#include "sw/kpu/dataflow/tile_dataflow_graph.hpp"
#include "sw/kpu/dataflow/block_mover_compiler.hpp"
#include "sw/kpu/dataflow/tile_flow_tracer.hpp"
#include <sw/kpu/models/temporal/noc/wormhole_router.hpp>

using namespace sw::kpu;
using namespace sw::kpu::dataflow;

// ============================================================================
// Configuration
// ============================================================================

struct MatmulConfig {
    // Matrix dimensions
    uint32_t M = 1024;      // Rows of A and C
    uint32_t N = 1024;      // Columns of B and C
    uint32_t K = 1024;      // Columns of A, Rows of B

    // Tiling
    uint32_t m_tiles = 4;   // Number of tiles along M
    uint32_t n_tiles = 4;   // Number of tiles along N
    uint32_t k_tiles = 4;   // Number of tiles along K

    // Hardware
    uint8_t num_l3_tiles = 16;
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;

    // Systolic array size (per compute tile)
    uint32_t systolic_size = 24;

    // Timing estimates (cycles)
    uint32_t dma_bandwidth_bytes_per_cycle = 64;    // Per-channel DMA bandwidth
    uint32_t l3_bandwidth_bytes_per_cycle = 64;     // L3 interconnect bandwidth
    uint32_t l2_bandwidth_bytes_per_cycle = 64;     // L3→L2 bandwidth

    // DMA channel architecture:
    // - 4 A-row channels: one per mesh row, feeding column 0
    // - 4 B-col channels: one per mesh column, feeding row 0
    // - Total: 8 parallel DMA channels
    uint8_t num_a_dma_channels() const { return mesh_rows; }
    uint8_t num_b_dma_channels() const { return mesh_cols; }
    uint8_t total_dma_channels() const { return mesh_rows + mesh_cols; }

    // DMA timing per channel
    uint32_t dma_cycles_per_tile() const {
        return a_tile_bytes() / dma_bandwidth_bytes_per_cycle;
    }

    // Aggregate DMA bandwidth
    uint32_t total_dma_bandwidth() const {
        return total_dma_channels() * dma_bandwidth_bytes_per_cycle;
    }

    // Derived values
    uint32_t tile_m() const { return M / m_tiles; }
    uint32_t tile_n() const { return N / n_tiles; }
    uint32_t tile_k() const { return K / k_tiles; }

    uint32_t a_tile_bytes() const { return tile_m() * tile_k() * sizeof(float); }
    uint32_t b_tile_bytes() const { return tile_k() * tile_n() * sizeof(float); }
    uint32_t c_tile_bytes() const { return tile_m() * tile_n() * sizeof(float); }

    uint64_t total_flops() const { return 2ULL * M * N * K; }

    void print() const {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              BLOCK SYSTOLIC MATMUL CONFIGURATION                 ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "Matrix Dimensions:\n";
        std::cout << "  C[" << M << " × " << N << "] = A[" << M << " × " << K
                  << "] × B[" << K << " × " << N << "]\n";
        std::cout << "  Total FLOPs: " << std::fixed << std::setprecision(2)
                  << (static_cast<double>(total_flops()) / 1e9) << " GFLOPs\n\n";

        std::cout << "Tiling:\n";
        std::cout << "  M tiles: " << m_tiles << " (tile size: " << tile_m() << ")\n";
        std::cout << "  N tiles: " << n_tiles << " (tile size: " << tile_n() << ")\n";
        std::cout << "  K tiles: " << k_tiles << " (tile size: " << tile_k() << ")\n";
        std::cout << "  Total C tiles: " << (m_tiles * n_tiles) << "\n\n";

        std::cout << "Tile Memory:\n";
        std::cout << "  A tile: " << (a_tile_bytes() / 1024) << " KB\n";
        std::cout << "  B tile: " << (b_tile_bytes() / 1024) << " KB\n";
        std::cout << "  C tile: " << (c_tile_bytes() / 1024) << " KB\n";
        std::cout << "  Per L3 (A+B+C): " << ((a_tile_bytes() + b_tile_bytes() + c_tile_bytes()) / 1024) << " KB\n\n";

        std::cout << "Hardware:\n";
        std::cout << "  L3 tiles: " << static_cast<int>(num_l3_tiles) << " ("
                  << static_cast<int>(mesh_rows) << "×" << static_cast<int>(mesh_cols) << " mesh)\n";
        std::cout << "  Systolic array: " << systolic_size << "×" << systolic_size << "\n";
        std::cout << "  Peak throughput: " << (systolic_size * systolic_size) << " MACs/cycle\n\n";

        std::cout << "DMA Architecture:\n";
        std::cout << "  A-row channels: " << static_cast<int>(num_a_dma_channels())
                  << " (feeding column 0)\n";
        std::cout << "  B-col channels: " << static_cast<int>(num_b_dma_channels())
                  << " (feeding row 0)\n";
        std::cout << "  Total DMA channels: " << static_cast<int>(total_dma_channels()) << "\n";
        std::cout << "  Per-channel bandwidth: " << dma_bandwidth_bytes_per_cycle << " B/cycle\n";
        std::cout << "  Aggregate DMA bandwidth: " << total_dma_bandwidth() << " B/cycle\n\n";

        std::cout << "Bandwidth Analysis:\n";
        uint32_t noc_transfer_cycles = a_tile_bytes() / l3_bandwidth_bytes_per_cycle;
        uint32_t tiles_per_k_step = mesh_rows + mesh_cols;  // A tiles + B tiles
        uint64_t bytes_per_k_step = static_cast<uint64_t>(mesh_rows) * a_tile_bytes() +
                                    static_cast<uint64_t>(mesh_cols) * b_tile_bytes();
        std::cout << "  NoC transfer time: " << noc_transfer_cycles << " cycles/tile\n";
        std::cout << "  DMA load time: " << dma_cycles_per_tile() << " cycles/tile\n";
        std::cout << "  Ingress per K-step: " << tiles_per_k_step << " tiles ("
                  << (bytes_per_k_step / 1024) << " KB)\n";
        std::cout << "  Required ingress BW: " << (bytes_per_k_step / noc_transfer_cycles)
                  << " B/cycle\n";
        std::cout << "  Available DMA BW: " << total_dma_bandwidth() << " B/cycle ";
        if (total_dma_bandwidth() >= bytes_per_k_step / noc_transfer_cycles) {
            std::cout << "(sufficient)\n";
        } else {
            std::cout << "(BOTTLENECK)\n";
        }
    }
};

// ============================================================================
// Systolic Dataflow Patterns
// ============================================================================

/// Build a DFG for output-stationary systolic matmul with staggered wavefront
/// C tiles stay in place, A flows East, B flows South
///
/// The staggered wavefront creates a diagonal wave pattern:
/// - A[m,k] enters one row at a time (staggered by m)
/// - B[k,n] enters one column at a time (staggered by n)
/// - At each L3[m,n], A and B meet at time proportional to (m + n)
/// - K iterations are pipelined: K+1 starts loading while K is still computing
TileDataFlowGraph build_systolic_matmul_dfg(const MatmulConfig& config) {
    TileDataFlowGraph dfg;

    // Configure timing model for accurate cycle-accurate scheduling
    // Use wormhole (pipelined) timing to match actual NoC behavior
    TimingModel timing;
    timing.mesh_rows = config.mesh_rows;
    timing.mesh_cols = config.mesh_cols;
    timing.store_and_forward = false;  // Wormhole NoC uses flit-pipelined routing
    timing.router_latency_cycles = 1;
    timing.link_latency_cycles = 0;    // Single-cycle hop
    dfg.set_timing_model(timing);

    uint32_t m_tiles = config.m_tiles;
    uint32_t n_tiles = config.n_tiles;
    uint32_t k_tiles = config.k_tiles;
    uint8_t mesh_cols = config.mesh_cols;
    uint8_t mesh_rows = config.mesh_rows;

    // Track node IDs
    // load_a[m][k] - DMA load of A[m,k]
    // load_b[k][n] - DMA load of B[k,n]
    // transfer_a[m][k][col] - A tile at column col
    // transfer_b[k][n][row] - B tile at row
    // compute[m][n][k] - matmul for C[m,n] step k
    // drain[m][n] - DMA store of C[m,n]

    std::vector<std::vector<size_t>> load_a(m_tiles, std::vector<size_t>(k_tiles));
    std::vector<std::vector<size_t>> load_b(k_tiles, std::vector<size_t>(n_tiles));
    std::vector<std::vector<std::vector<size_t>>> compute(
        m_tiles, std::vector<std::vector<size_t>>(n_tiles, std::vector<size_t>(k_tiles)));
    std::vector<std::vector<size_t>> drain(m_tiles, std::vector<size_t>(n_tiles));

    // A tile transfer tracking: [m][k][col] where col is the column in the mesh
    std::vector<std::vector<std::vector<size_t>>> a_at_col(
        m_tiles, std::vector<std::vector<size_t>>(k_tiles, std::vector<size_t>(mesh_cols)));

    // B tile transfer tracking: [k][n][row]
    std::vector<std::vector<std::vector<size_t>>> b_at_row(
        k_tiles, std::vector<std::vector<size_t>>(n_tiles, std::vector<size_t>(mesh_rows)));

    // ========================================================================
    // Phase 1: DMA loads with staggered wavefront
    // ========================================================================

    // A tiles loaded to column 0 of mesh (L3 tiles 0, 4, 8, 12 for 4x4 mesh)
    // Stagger: A[m,k] waits for A[m-1,k] to create diagonal wavefront
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = static_cast<uint16_t>(m);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = config.a_tile_bytes();
            tile.height = static_cast<uint16_t>(config.tile_m());
            tile.width = static_cast<uint16_t>(config.tile_k());

            // Load to L3 in column 0, row = m % mesh_rows
            uint8_t l3_id = static_cast<uint8_t>((m % mesh_rows) * mesh_cols);
            load_a[m][k] = dfg.add_dma_load(tile, l3_id);
            a_at_col[m][k][0] = load_a[m][k];
        }
    }

    // B tiles loaded to row 0 of mesh (L3 tiles 0, 1, 2, 3 for 4x4 mesh)
    // Stagger: B[k,n] waits for B[k,n-1] to create diagonal wavefront
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = static_cast<uint16_t>(n);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = config.b_tile_bytes();
            tile.height = static_cast<uint16_t>(config.tile_k());
            tile.width = static_cast<uint16_t>(config.tile_n());

            // Load to L3 in row 0, col = n % mesh_cols
            uint8_t l3_id = static_cast<uint8_t>(n % mesh_cols);
            load_b[k][n] = dfg.add_dma_load(tile, l3_id);
            b_at_row[k][n][0] = load_b[k][n];
        }
    }

    // ========================================================================
    // Phase 1b: DMA channel K-step ordering (parallel channels)
    // ========================================================================
    //
    // DMA Architecture:
    // - 4 A-row DMA channels: one per mesh row, feeding column 0
    //   Each channel loads A[row, k=0], A[row, k=1], ... sequentially
    // - 4 B-col DMA channels: one per mesh column, feeding row 0
    //   Each channel loads B[k=0, col], B[k=1, col], ... sequentially
    // - Channels operate in parallel (no cross-row/column dependencies)
    //
    // This models 8 parallel DMA engines, each with its own address
    // generation logic for sequential tile loads.

    // A-row DMA channels: K-step ordering within each row (m)
    // A[m,k] waits for A[m,k-1] on the same channel
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 1; k < k_tiles; ++k) {
            dfg.add_edge(load_a[m][k-1], load_a[m][k], DFEdgeType::CONTROL);
        }
    }

    // B-col DMA channels: K-step ordering within each column (n)
    // B[k,n] waits for B[k-1,n] on the same channel
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint32_t k = 1; k < k_tiles; ++k) {
            dfg.add_edge(load_b[k-1][n], load_b[k][n], DFEdgeType::CONTROL);
        }
    }

    // ========================================================================
    // Phase 2: A tiles flow East, B tiles flow South
    // ========================================================================

    // A flows East: for each A tile, create transfers across columns
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t k = 0; k < k_tiles; ++k) {
            TileDescriptor tile;
            tile.tensor = TensorId::A;
            tile.m_tile = static_cast<uint16_t>(m);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = config.a_tile_bytes();

            uint8_t row = static_cast<uint8_t>(m % mesh_rows);

            for (uint8_t col = 1; col < mesh_cols; ++col) {
                uint8_t src_l3 = row * mesh_cols + (col - 1);
                uint8_t dst_l3 = row * mesh_cols + col;

                size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
                dfg.add_edge(a_at_col[m][k][col - 1], transfer);
                a_at_col[m][k][col] = transfer;
            }
        }
    }

    // B flows South: for each B tile, create transfers down rows
    for (uint32_t k = 0; k < k_tiles; ++k) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            TileDescriptor tile;
            tile.tensor = TensorId::B;
            tile.n_tile = static_cast<uint16_t>(n);
            tile.k_tile = static_cast<uint16_t>(k);
            tile.size = config.b_tile_bytes();

            uint8_t col = static_cast<uint8_t>(n % mesh_cols);

            for (uint8_t row = 1; row < mesh_rows; ++row) {
                uint8_t src_l3 = (row - 1) * mesh_cols + col;
                uint8_t dst_l3 = row * mesh_cols + col;

                size_t transfer = dfg.add_l3_transfer(tile, src_l3, dst_l3);
                dfg.add_edge(b_at_row[k][n][row - 1], transfer);
                b_at_row[k][n][row] = transfer;
            }
        }
    }

    // ========================================================================
    // Phase 2b: K-step ordering for proper pipelining
    // ========================================================================

    // For proper systolic pipelining:
    // - Tiles for K=1 should flow through AFTER tiles for K=0
    // - This allows: receive A[0,0] → forward → receive A[0,1] → forward → ...
    // - Without this, all K tiles are sent simultaneously, no pipelining

    // K-step dependencies on A transfers: A[m,k] at col c waits for A[m,k-1] at col c
    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint8_t col = 1; col < mesh_cols; ++col) {
            for (uint32_t k = 1; k < k_tiles; ++k) {
                // A[m,k] at column col waits for A[m,k-1] at same column
                dfg.add_edge(a_at_col[m][k-1][col], a_at_col[m][k][col], DFEdgeType::CONTROL);
            }
        }
    }

    // K-step dependencies on B transfers: B[k,n] at row r waits for B[k-1,n] at row r
    for (uint32_t n = 0; n < n_tiles; ++n) {
        for (uint8_t row = 1; row < mesh_rows; ++row) {
            for (uint32_t k = 1; k < k_tiles; ++k) {
                // B[k,n] at row waits for B[k-1,n] at same row
                dfg.add_edge(b_at_row[k-1][n][row], b_at_row[k][n][row], DFEdgeType::CONTROL);
            }
        }
    }

    // ========================================================================
    // Phase 2c: Wavefront pattern via independent row/column operation
    // ========================================================================
    //
    // In a true systolic array:
    // - Each row operates independently (A tiles flow East on their row)
    // - Each column operates independently (B tiles flow South on their column)
    // - The diagonal wavefront emerges from:
    //   1. Parallel DMA channels starting simultaneously
    //   2. WAIT_DELIVERY ensuring K-step sequencing within each path
    //   3. start_delay at edge L3s creating the initial offset
    //
    // NO cross-row or cross-column dependencies are needed for transfers.
    // This allows maximum NoC concurrency.

    // ========================================================================
    // Note on K-step synchronization:
    // ========================================================================
    //
    // In a true systolic design, barriers are NOT needed. The spatial and
    // temporal structure provides implicit synchronization:
    //
    // 1. DMA channel K-step ordering (Phase 1b): A[m,k] waits for A[m,k-1]
    // 2. Transfer K-step ordering (Phase 2b): A[m,k] at col c waits for A[m,k-1] at col c
    //
    // With accurate timing model (store-and-forward delays), the scheduler
    // computes when each K-step can start without contention. The WAIT_UNTIL_CYCLE
    // command enforces this timing at runtime.
    //
    // No barriers needed - tile boundaries provide natural synchronization.

    // ========================================================================
    // Phase 3: Compute nodes
    // ========================================================================

    for (uint32_t m = 0; m < m_tiles; ++m) {
        for (uint32_t n = 0; n < n_tiles; ++n) {
            // C[m,n] is computed at L3[row, col] where row = m % mesh_rows, col = n % mesh_cols
            uint8_t row = static_cast<uint8_t>(m % mesh_rows);
            uint8_t col = static_cast<uint8_t>(n % mesh_cols);
            uint8_t l3_id = row * mesh_cols + col;

            for (uint32_t k = 0; k < k_tiles; ++k) {
                TileDescriptor c_tile;
                c_tile.tensor = TensorId::C;
                c_tile.m_tile = static_cast<uint16_t>(m);
                c_tile.n_tile = static_cast<uint16_t>(n);
                c_tile.k_tile = static_cast<uint16_t>(k);
                c_tile.size = config.c_tile_bytes();

                TileDescriptor a_tile;
                a_tile.tensor = TensorId::A;
                a_tile.m_tile = static_cast<uint16_t>(m);
                a_tile.k_tile = static_cast<uint16_t>(k);

                TileDescriptor b_tile;
                b_tile.tensor = TensorId::B;
                b_tile.n_tile = static_cast<uint16_t>(n);
                b_tile.k_tile = static_cast<uint16_t>(k);

                compute[m][n][k] = dfg.add_matmul(l3_id,
                    config.tile_m(), config.tile_n(), config.tile_k(),
                    a_tile, b_tile, c_tile);

                // Dependencies: need A at this column and B at this row
                dfg.add_edge(a_at_col[m][k][col], compute[m][n][k]);
                dfg.add_edge(b_at_row[k][n][row], compute[m][n][k]);

                // K-step dependency (accumulation)
                if (k > 0) {
                    dfg.add_edge(compute[m][n][k-1], compute[m][n][k]);
                }
            }

            // ================================================================
            // Phase 4: Drain C tiles
            // ================================================================
            TileDescriptor c_final;
            c_final.tensor = TensorId::C;
            c_final.m_tile = static_cast<uint16_t>(m);
            c_final.n_tile = static_cast<uint16_t>(n);
            c_final.size = config.c_tile_bytes();

            drain[m][n] = dfg.add_dma_store(c_final, l3_id);
            dfg.add_edge(compute[m][n][k_tiles - 1], drain[m][n]);
        }
    }

    return dfg;
}

// ============================================================================
// Analysis and Reporting
// ============================================================================

void analyze_dfg(const TileDataFlowGraph& dfg, const MatmulConfig& config) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    DATA FLOW GRAPH ANALYSIS                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Graph Structure:\n";
    std::cout << "  Total nodes: " << dfg.num_nodes() << "\n";
    std::cout << "  Total edges: " << dfg.num_edges() << "\n";
    std::cout << "  Is acyclic: " << (dfg.is_acyclic() ? "yes" : "NO!") << "\n\n";

    // Count by type
    auto dma_loads = dfg.get_nodes_by_type(DFNodeType::DMA_LOAD);
    auto dma_stores = dfg.get_nodes_by_type(DFNodeType::DMA_STORE);
    auto l3_transfers = dfg.get_nodes_by_type(DFNodeType::L3_TRANSFER);
    auto matmuls = dfg.get_nodes_by_type(DFNodeType::MATMUL);

    std::cout << "Node Types:\n";
    std::cout << "  DMA loads:     " << dma_loads.size() << "\n";
    std::cout << "  DMA stores:    " << dma_stores.size() << "\n";
    std::cout << "  L3 transfers:  " << l3_transfers.size() << "\n";
    std::cout << "  Matmul ops:    " << matmuls.size() << "\n\n";

    // Compute critical path
    int64_t critical_path = dfg.critical_path_length();
    std::cout << "Critical Path:\n";
    std::cout << "  Length: " << critical_path << " cycles\n";

    // Estimate throughput
    double systolic_throughput = config.systolic_size * config.systolic_size;  // MACs/cycle
    double theoretical_compute = static_cast<double>(config.total_flops()) / systolic_throughput;
    double efficiency = theoretical_compute / static_cast<double>(critical_path) * 100.0;

    std::cout << "  Theoretical compute: " << static_cast<int64_t>(theoretical_compute) << " cycles\n";
    std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%\n\n";

    // Per-L3 distribution
    std::cout << "Per-L3 Distribution:\n";
    for (uint8_t l3 = 0; l3 < config.num_l3_tiles; ++l3) {
        auto nodes = dfg.get_nodes_by_l3(l3);
        if (!nodes.empty()) {
            std::cout << "  L3[" << std::setw(2) << static_cast<int>(l3) << "]: "
                      << nodes.size() << " nodes\n";
        }
    }
}

void analyze_schedule(const DFGSchedule& schedule, const MatmulConfig& config) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                      SCHEDULE ANALYSIS                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Schedule Metrics:\n";
    std::cout << "  Makespan: " << schedule.makespan << " cycles\n";

    // Estimate wall clock time at 1 GHz
    double time_ms = static_cast<double>(schedule.makespan) / 1e6;
    std::cout << "  Time @ 1 GHz: " << std::fixed << std::setprecision(3) << time_ms << " ms\n";

    // Throughput
    double gflops = static_cast<double>(config.total_flops()) / 1e9;
    double throughput = gflops / (time_ms / 1000.0);
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << throughput << " GFLOPS\n\n";

    // Peak comparison
    double peak_throughput = config.num_l3_tiles * config.systolic_size * config.systolic_size * 2.0;
    // 2.0 for MAC = 2 ops
    double utilization = throughput / peak_throughput * 100.0;
    std::cout << "Utilization:\n";
    std::cout << "  Peak throughput: " << std::fixed << std::setprecision(1)
              << peak_throughput << " GFLOPS @ 1 GHz\n";
    std::cout << "  Achieved: " << std::fixed << std::setprecision(1) << utilization << "%\n";
}

void analyze_compiled_schedule(const CompiledSchedule& schedule) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                 BLOCKMOVER PROGRAM ANALYSIS                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Overall Statistics:\n";
    std::cout << "  Total commands:     " << schedule.stats.total_commands << "\n";
    std::cout << "  DMA operations:     " << schedule.stats.total_dma_ops << "\n";
    std::cout << "  L3-L3 transfers:    " << schedule.stats.total_l3_transfers << "\n";
    std::cout << "  L3-L2 transfers:    " << schedule.stats.total_l2_transfers << "\n";
    std::cout << "  Compute operations: " << schedule.stats.total_compute_ops << "\n";
    std::cout << "  Triggers:           " << schedule.stats.total_triggers << "\n";
    std::cout << "  Barriers:           " << schedule.stats.total_barriers << "\n\n";

    std::cout << "Timing Estimates:\n";
    std::cout << "  Total cycles:       " << schedule.estimated_cycles << "\n";
    std::cout << "  Compute cycles:     " << schedule.compute_cycles << " ("
              << (100.0 * static_cast<double>(schedule.compute_cycles) / static_cast<double>(schedule.estimated_cycles)) << "%)\n";
    std::cout << "  Data movement:      " << schedule.data_movement_cycles << " ("
              << (100.0 * static_cast<double>(schedule.data_movement_cycles) / static_cast<double>(schedule.estimated_cycles)) << "%)\n\n";

    std::cout << "Per-L3 Programs:\n";
    for (uint8_t l3 = 0; l3 < 16; ++l3) {
        const auto& prog = schedule.program(l3);
        if (!prog.empty()) {
            std::cout << "  L3[" << std::setw(2) << static_cast<int>(l3) << "]: "
                      << std::setw(4) << prog.size() << " commands\n";
        }
    }
}

// ============================================================================
// Simulation
// ============================================================================

void simulate_execution(const CompiledSchedule& compiled, const MatmulConfig& config,
                        bool enable_tracing = true) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    SIMULATION EXECUTION                           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Create tracers
    TileFlowTracer tracer;
    noc::WormholeTracer noc_tracer;

    // Create BlockMoverArray
    BlockMoverArray::Config array_config;
    array_config.rows = config.mesh_rows;
    array_config.cols = config.mesh_cols;
    BlockMoverArray array(array_config);

    // Enable tracing
    if (enable_tracing) {
        array.set_tracer(&tracer);
        array.set_noc_tracer(&noc_tracer);
    }

    // Load programs - the schedule's WAIT_UNTIL_CYCLE commands handle all timing
    // No wavefront start delays needed; the scheduler respects dependencies
    // which naturally create the diagonal wavefront pattern
    std::vector<BlockMoverProgram> programs;
    for (uint8_t l3 = 0; l3 < config.num_l3_tiles; ++l3) {
        if (!compiled.program(l3).empty()) {
            programs.push_back(compiled.program(l3));
        }
    }
    array.load_programs(programs);

    uint64_t transfer_time = config.a_tile_bytes() / config.l3_bandwidth_bytes_per_cycle;
    std::cout << "Transfer time: " << transfer_time << " cycles/tile\n" << std::flush;

    std::cout << "Starting simulation with " << programs.size() << " active L3 tiles...\n" << std::flush;
    if (enable_tracing) {
        std::cout << "  Tracing enabled\n" << std::flush;
    }

    // Execute
    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t cycle = 0;
    uint64_t max_cycles = 500000;  // Safety limit

    std::cout << "  Estimated makespan: " << compiled.estimated_cycles << " cycles\n" << std::flush;
    std::cout << "  Note: Compute time not simulated (data movement only)\n\n" << std::flush;

    while (!array.all_idle() && cycle < max_cycles) {
        array.step(cycle);
        cycle++;

        // Progress report every 10K cycles (for long simulations)
        if (cycle % 10000 == 0) {
            std::cout << "  Cycle " << (cycle / 1000) << "K: "
                      << array.num_running() << " running, "
                      << array.num_waiting() << " waiting\n" << std::flush;
        }

        // Debug: Print stuck mover status after 1000 cycles
        if (cycle == 1000 && array.num_waiting() > 0) {
            std::cout << "\n*** DEBUG: Movers stuck at cycle 1000 ***\n";
            for (uint8_t id = 0; id < 16; ++id) {
                auto& mover = array[id];
                if (!mover.is_idle()) {
                    uint8_t row = id / 4;
                    uint8_t col = id % 4;
                    std::cout << "  L3[" << static_cast<int>(row) << "," << static_cast<int>(col)
                              << "] (id=" << static_cast<int>(id) << "): "
                              << mover.to_string();
                    if (mover.state() == BlockMoverState::WAITING_RECEIVE) {
                        std::cout << " " << mover.debug_receive_state();
                    }
                    std::cout << "\n";
                }
            }
            std::cout << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto sim_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nSimulation complete:\n";
    std::cout << "  Simulated cycles: " << cycle << "\n";
    std::cout << "  Simulation time:  " << sim_time.count() << " ms\n";
    std::cout << "  Simulation rate:  " << std::fixed << std::setprecision(1)
              << (static_cast<double>(cycle) / 1000.0 / static_cast<double>(sim_time.count())) << " M cycles/sec\n";

    // Get aggregate stats
    auto stats = array.get_aggregate_stats();
    std::cout << "\nAggregate Statistics:\n";
    std::cout << "  Total commands executed: " << stats.total_commands << "\n";
    std::cout << "  L3-L3 transfers:         " << stats.total_l3_l3_transfers << "\n";
    std::cout << "  L3-L2 transfers:         " << stats.total_l3_l2_transfers << "\n";
    std::cout << "  Total bytes moved:       " << (stats.total_bytes_moved / 1024) << " KB\n";
    std::cout << "  Average utilization:     " << std::fixed << std::setprecision(1)
              << (stats.avg_utilization * 100) << "%\n";

    // Export trace data
    if (enable_tracing && tracer.num_events() > 0) {
        std::cout << "\nTrace Statistics:\n";
        auto trace_stats = tracer.compute_stats(config.num_l3_tiles);
        std::cout << "  Total events: " << trace_stats.total_events << "\n";
        std::cout << "  L3-L3 transfers: " << trace_stats.l3_transfers << "\n";
        std::cout << "  L3-L2 transfers: " << trace_stats.l2_transfers << "\n";
        std::cout << "  Barriers: " << trace_stats.barriers << "\n";

        // Export to files
        std::string csv_file = "tile_flow_trace.csv";
        std::string json_file = "tile_flow_trace.json";
        std::string chrome_file = "tile_flow_trace_chrome.json";

        if (tracer.export_csv(csv_file)) {
            std::cout << "\n  Trace exported to: " << csv_file << "\n";
        }
        if (tracer.export_json(json_file)) {
            std::cout << "  Trace exported to: " << json_file << "\n";
        }
        if (tracer.export_chrome_trace(chrome_file)) {
            std::cout << "  Chrome Trace exported to: " << chrome_file << "\n";
            std::cout << "  Open in: chrome://tracing or https://ui.perfetto.dev\n";
        }
        std::cout << "\n  Visualize with: python3 tools/visualization/tile_flow_gantt.py " << csv_file << "\n";
    }

    // Export NoC trace data
    if (enable_tracing && noc_tracer.num_events() > 0) {
        std::cout << "\nNoC Statistics (Wormhole Router):\n";
        const auto& noc_stats = array.noc().stats();
        std::cout << "  Tiles injected:  " << noc_stats.tiles_injected << "\n";
        std::cout << "  Tiles delivered: " << noc_stats.tiles_delivered << "\n";
        std::cout << "  Total flits:     " << noc_stats.total_flits << "\n";
        std::cout << "  Total bytes:     " << (noc_stats.total_bytes / 1024) << " KB\n";
        if (noc_stats.tiles_delivered > 0) {
            std::cout << "  Avg latency:     " << std::fixed << std::setprecision(1)
                      << (static_cast<double>(noc_stats.total_latency_cycles) / static_cast<double>(noc_stats.tiles_delivered))
                      << " cycles\n";
            std::cout << "  Max latency:     " << noc_stats.max_latency_cycles << " cycles\n";
        }

        // Export NoC trace
        std::string noc_trace_file = "/tmp/noc_trace.csv";
        if (noc_tracer.export_csv(noc_trace_file)) {
            std::cout << "\n  NoC trace exported to: " << noc_trace_file << "\n";
            std::cout << "  Visualize with: python3 tools/visualization/generate_noc_animation.py "
                      << noc_trace_file << " /tmp/matmul_noc.html\n";
        }

        // Export Chrome trace
        std::string chrome_trace_file = "/tmp/noc_trace_chrome.json";
        if (noc_tracer.export_chrome_trace(chrome_trace_file, config.mesh_rows, config.mesh_cols)) {
            std::cout << "  Chrome Trace exported to: " << chrome_trace_file << "\n";
            std::cout << "  Open in: chrome://tracing or https://ui.perfetto.dev\n";
        }
    }

    // Print systolic order check results
    std::cout << "\n";
    array.order_checker().print_violations(std::cout);
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m SIZE    Matrix size M=N=K (default: 1024)\n";
    std::cout << "  -t TILES   Tile count m_tiles=n_tiles (default: 4)\n";
    std::cout << "  -k KTILES  K tile count (default: same as -t)\n";
    std::cout << "  --small    Use small 64x64 matrix with 4 tiles (for debugging)\n";
    std::cout << "  --tiny     Use tiny 32x32 matrix with 4 tiles (1KB tiles)\n";
    std::cout << "  --help     Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << "              # Default 1024x1024 with 4x4 tiles (256KB each)\n";
    std::cout << "  " << prog << " --small      # 64x64 with 4x4 tiles (1KB each)\n";
    std::cout << "  " << prog << " --tiny       # 32x32 with 4x4 tiles (256B each)\n";
    std::cout << "  " << prog << " -m 256 -t 4  # 256x256 with 4x4 tiles (16KB each)\n";
}

int main(int argc, char* argv[]) {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            BLOCK SYSTOLIC MATRIX MULTIPLY DEMONSTRATION                      ║
║                                                                              ║
║  This example demonstrates scalable tiled matrix multiplication using        ║
║  stateful BlockMovers to implement a systolic data flow pattern:             ║
║                                                                              ║
║    - C tiles distributed across L3 tiles (output stationary)                 ║
║    - A tiles flow East through the 4x4 mesh                                  ║
║    - B tiles flow South through the mesh                                     ║
║    - Synchronization via trigger network                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
)";

    // Parse command line
    MatmulConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--small") {
            // Small test case: 64x64 matrix, 4x4 tiles = 16x16 per tile = 1KB
            config.M = config.N = config.K = 64;
            config.m_tiles = config.n_tiles = config.k_tiles = 4;
        } else if (arg == "--tiny") {
            // Tiny test case: 32x32 matrix, 4x4 tiles = 8x8 per tile = 256B
            config.M = config.N = config.K = 32;
            config.m_tiles = config.n_tiles = config.k_tiles = 4;
        } else if (arg == "-m" && i + 1 < argc) {
            config.M = config.N = config.K = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-t" && i + 1 < argc) {
            config.m_tiles = config.n_tiles = config.k_tiles = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "-k" && i + 1 < argc) {
            config.k_tiles = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg[0] != '-') {
            // Legacy positional args
            config.M = config.N = config.K = static_cast<uint32_t>(std::stoul(arg));
            if (i + 1 < argc && argv[i+1][0] != '-') {
                config.m_tiles = config.n_tiles = static_cast<uint32_t>(std::stoul(argv[++i]));
            }
            if (i + 1 < argc && argv[i+1][0] != '-') {
                config.k_tiles = static_cast<uint32_t>(std::stoul(argv[++i]));
            }
        }
    }

    // Validate configuration
    if (config.M % config.m_tiles != 0 ||
        config.N % config.n_tiles != 0 ||
        config.K % config.k_tiles != 0) {
        std::cerr << "Error: Matrix dimensions must be divisible by tile counts\n";
        return 1;
    }

    config.print();

    // Step 1: Build the Data Flow Graph
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Step 1: Building Data Flow Graph\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    auto dfg_start = std::chrono::high_resolution_clock::now();
    TileDataFlowGraph dfg = build_systolic_matmul_dfg(config);
    auto dfg_end = std::chrono::high_resolution_clock::now();

    auto dfg_time = std::chrono::duration_cast<std::chrono::microseconds>(dfg_end - dfg_start);
    std::cout << "DFG built in " << dfg_time.count() << " μs\n";

    analyze_dfg(dfg, config);

    // Step 2: Schedule the DFG
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Step 2: Scheduling Data Flow Graph\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::flush;

    std::cout << "  Creating scheduler..." << std::flush;
    auto sched_start = std::chrono::high_resolution_clock::now();
    DFGScheduler::Config sched_config;
    sched_config.algorithm = DFGScheduler::Algorithm::ASAP;  // Don't serialize L3 resources
    DFGScheduler scheduler(sched_config);
    std::cout << " scheduling..." << std::flush;
    DFGSchedule schedule = scheduler.schedule(dfg);
    std::cout << " done.\n" << std::flush;
    auto sched_end = std::chrono::high_resolution_clock::now();

    auto sched_time = std::chrono::duration_cast<std::chrono::microseconds>(sched_end - sched_start);
    std::cout << "Schedule generated in " << sched_time.count() << " μs\n";

    analyze_schedule(schedule, config);

    // Step 3: Compile to BlockMover programs
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Step 3: Compiling to BlockMover Programs\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::flush;

    std::cout << "  Creating compiler..." << std::flush;
    auto compile_start = std::chrono::high_resolution_clock::now();

    // Configure compiler for concurrent A/B DMA
    // Disable sync_sends to allow A and B tiles to be sent in parallel
    // K-step ordering is enforced by DFG dependencies
    BlockMoverCompiler::Config compiler_config;
    compiler_config.mesh_rows = config.mesh_rows;
    compiler_config.mesh_cols = config.mesh_cols;
    compiler_config.sync_sends = false;  // Allow parallel A/B sends
    BlockMoverCompiler compiler(compiler_config);
    std::cout << " compiling..." << std::flush;
    CompiledSchedule compiled = compiler.compile(dfg, schedule);
    std::cout << " done.\n" << std::flush;
    auto compile_end = std::chrono::high_resolution_clock::now();

    auto compile_time = std::chrono::duration_cast<std::chrono::microseconds>(compile_end - compile_start);
    std::cout << "Compiled in " << compile_time.count() << " μs\n" << std::flush;

    std::cout << "Analyzing compiled schedule..." << std::flush;
    analyze_compiled_schedule(compiled);
    std::cout << " done.\n" << std::flush;

    // Step 4: Simulate execution
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Step 4: Simulating Execution\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    simulate_execution(compiled, config);

    // Summary
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                           SUMMARY                                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Matrix: " << config.M << "×" << config.K << " × "
              << config.K << "×" << config.N << "\n";
    std::cout << "Tiling: " << config.m_tiles << "×" << config.n_tiles
              << "×" << config.k_tiles << "\n";
    std::cout << "DFG nodes: " << dfg.num_nodes() << ", edges: " << dfg.num_edges() << "\n";
    std::cout << "Scheduled makespan: " << schedule.makespan << " cycles\n";
    std::cout << "BlockMover commands: " << compiled.stats.total_commands << "\n";

    return 0;
}
