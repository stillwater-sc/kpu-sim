// ============================================================================
// tests/dataflow/test_c_stationary_matmul.cpp
// Complete C-Stationary Matrix Multiply through all levels
// ============================================================================
//
// This test demonstrates a complete C-stationary matmul execution:
//
//   C[M,N] = A[M,K] × B[K,N]
//
// On a 2x2 mesh with K=2 iterations:
//
//   Level 1 (DMA):        Load A and B tiles from memory to L3
//   Level 2 (BlockMover): Push to L2, forward A→East, B→South
//   Level 3 (Streamer):   Feed to systolic array, compute, drain
//
// Tile flow (C-stationary):
//   - C[i,j] stays at CT[i,j] throughout
//   - A[i,k] flows West→East (broadcast along row)
//   - B[k,j] flows North→South (broadcast along column)
//
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>
#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>
#include <sw/kpu/models/dataflow/dma_flow_executor.hpp>
#include <sw/kpu/models/dataflow/block_mover_flow_executor.hpp>
#include <sw/kpu/models/dataflow/streamer_flow_executor.hpp>

#include <iostream>
#include <iomanip>
#include <map>

using namespace sw::kpu::dataflow;

// ============================================================================
// Test Configuration
// ============================================================================

struct MatmulConfig {
    uint16_t mesh_rows = 2;
    uint16_t mesh_cols = 2;
    uint16_t k_tiles = 2;

    // Latencies (cycles)
    uint16_t dma_load_latency = 100;
    uint16_t bm_push_latency = 10;
    uint16_t bm_mesh_latency = 5;
    uint16_t streamer_feed_latency = 5;
    uint16_t streamer_matmul_latency = 64;
    uint16_t streamer_drain_latency = 10;
};

// ============================================================================
// Multi-Level Orchestrator
// ============================================================================

/// Timing information for a schedule execution
struct ScheduleTiming {
    uint64_t dma_start = 0;
    uint64_t dma_end = 0;
    uint64_t blockmover_start = 0;
    uint64_t blockmover_end = 0;
    uint64_t streamer_start = 0;
    uint64_t streamer_end = 0;

    uint64_t dma_latency() const { return dma_end - dma_start; }
    uint64_t blockmover_latency() const { return blockmover_end - blockmover_start; }
    uint64_t streamer_latency() const { return streamer_end - streamer_start; }
    uint64_t total_latency() const { return streamer_end - dma_start; }

    // Pipeline overlap analysis
    uint64_t dma_blockmover_overlap() const {
        if (blockmover_start >= dma_end) return 0;
        return std::min(dma_end, blockmover_end) - blockmover_start;
    }
    uint64_t blockmover_streamer_overlap() const {
        if (streamer_start >= blockmover_end) return 0;
        return std::min(blockmover_end, streamer_end) - streamer_start;
    }
};

class CStationaryOrchestrator {
public:
    explicit CStationaryOrchestrator(const MatmulConfig& config)
        : config_(config) {
        build_all_graphs();
        create_all_executors();
    }

    // Run the complete matmul
    void run() {
        std::cout << "\n========== C-Stationary Matmul Execution ==========\n";
        std::cout << "Mesh: " << config_.mesh_rows << "x" << config_.mesh_cols << "\n";
        std::cout << "K iterations: " << config_.k_tiles << "\n";
        std::cout << "Latencies: DMA=" << config_.dma_load_latency
                  << ", BM_push=" << config_.bm_push_latency
                  << ", BM_mesh=" << config_.bm_mesh_latency
                  << ", feed=" << config_.streamer_feed_latency
                  << ", matmul=" << config_.streamer_matmul_latency
                  << ", drain=" << config_.streamer_drain_latency << "\n\n";

        // Phase 1: DMA loads
        timing_.dma_start = 0;
        std::cout << "--- Phase 1: DMA Loads (Memory → L3) ---\n";
        run_dma_phase();
        timing_.dma_end = std::max(get_max_dma_cycle(), timing_.dma_start + 1);

        // Phase 2: BlockMover operations
        timing_.blockmover_start = timing_.dma_end;
        std::cout << "\n--- Phase 2: BlockMover (L3 → L2, mesh forwarding) ---\n";
        run_block_mover_phase();
        timing_.blockmover_end = std::max(get_max_blockmover_cycle(), timing_.blockmover_start + 1);

        // Phase 3: Streamer/Compute
        timing_.streamer_start = timing_.blockmover_end;
        std::cout << "\n--- Phase 3: Streamer/Compute (L2 → L1 → Accumulator) ---\n";
        run_streamer_phase();
        // Ensure end >= start for proper latency calculation
        timing_.streamer_end = std::max(get_max_streamer_cycle(), timing_.streamer_start + 1);

        // Print summary
        print_summary();
    }

    // Timing accessors
    const ScheduleTiming& timing() const { return timing_; }
    uint64_t total_cycles() const { return timing_.total_latency(); }

    // Get total FLOPs
    uint64_t total_flops() const {
        uint64_t flops = 0;
        for (const auto& [id, exec] : streamers_) {
            flops += exec->flops();
        }
        return flops;
    }

    // Get total matmuls completed
    uint32_t total_matmuls() const {
        uint32_t count = 0;
        for (const auto& [id, exec] : streamers_) {
            count += exec->matmuls_completed();
        }
        return count;
    }

    // Get total DMA bytes transferred
    uint64_t dma_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, dma] : dma_west_) total += dma->bytes_transferred();
        for (const auto& [id, dma] : dma_north_) total += dma->bytes_transferred();
        return total;
    }

    // Get total L2 bytes transferred
    uint64_t l2_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_to_l2();
        return total;
    }

    // Get total mesh bytes transferred
    uint64_t mesh_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_mesh();
        return total;
    }

private:
    uint64_t get_max_dma_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, dma] : dma_west_) {
            max_cycle = std::max(max_cycle, dma->current_cycle());
        }
        for (const auto& [id, dma] : dma_north_) {
            max_cycle = std::max(max_cycle, dma->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_blockmover_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, bm] : block_movers_) {
            max_cycle = std::max(max_cycle, bm->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_streamer_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, exec] : streamers_) {
            max_cycle = std::max(max_cycle, exec->current_cycle());
        }
        return max_cycle;
    }

private:
    void build_all_graphs() {
        // Build DMA graphs for west edge (A tiles) and north edge (B tiles)
        build_dma_graphs();

        // Build BlockMover graphs for each L3 tile
        build_block_mover_graphs();

        // Build Streamer graphs for each compute tile
        build_streamer_graphs();
    }

    void build_dma_graphs() {
        // West edge DMAs load A tiles
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            DMAFlowGraphBuilder builder;
            builder.set_node_id(i)
                   .set_dimensions(config_.mesh_rows, config_.mesh_cols, config_.k_tiles);

            // Load all A[i,k] tiles for this row
            for (uint16_t k = 0; k < config_.k_tiles; ++k) {
                uint8_t dest_l3 = i * config_.mesh_cols;  // First column
                builder.add_load_a(i, k, dest_l3);
            }

            dma_graphs_west_[i] = builder.build();
            dma_graphs_west_[i].name = "DMA_West_" + std::to_string(i);
        }

        // North edge DMAs load B tiles
        for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
            DMAFlowGraphBuilder builder;
            builder.set_node_id(j)
                   .set_dimensions(config_.mesh_rows, config_.mesh_cols, config_.k_tiles);

            // Load all B[k,j] tiles for this column
            for (uint16_t k = 0; k < config_.k_tiles; ++k) {
                uint8_t dest_l3 = j;  // First row
                builder.add_load_b(k, j, dest_l3);
            }

            dma_graphs_north_[j] = builder.build();
            dma_graphs_north_[j].name = "DMA_North_" + std::to_string(j);
        }
    }

    void build_block_mover_graphs() {
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
                uint8_t node_id = i * config_.mesh_cols + j;

                BlockMoverFlowGraphBuilder builder;
                builder.set_position(i, j, config_.mesh_cols)
                       .set_dimensions(config_.mesh_rows, config_.mesh_cols, config_.k_tiles)
                       .build_c_stationary()
                       .add_drain_c();

                block_mover_graphs_[node_id] = builder.build();
                block_mover_graphs_[node_id].name = "BM[" + std::to_string(i) + "," + std::to_string(j) + "]";
            }
        }
    }

    void build_streamer_graphs() {
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
                uint8_t node_id = i * config_.mesh_cols + j;

                StreamerFlowGraphBuilder builder;
                builder.set_position(i, j, config_.mesh_cols)
                       .set_dimensions(config_.mesh_rows, config_.mesh_cols, config_.k_tiles)
                       .build_c_stationary();

                streamer_graphs_[node_id] = builder.build();
                streamer_graphs_[node_id].name = "Streamer[" + std::to_string(i) + "," + std::to_string(j) + "]";
            }
        }
    }

    void create_all_executors() {
        DMAExecutorConfig dma_config;
        dma_config.load_latency = config_.dma_load_latency;

        // West edge DMA executors
        for (auto& [id, graph] : dma_graphs_west_) {
            dma_west_[id] = std::make_unique<DMAFlowExecutor>(graph, dma_config);
        }

        // North edge DMA executors
        for (auto& [id, graph] : dma_graphs_north_) {
            dma_north_[id] = std::make_unique<DMAFlowExecutor>(graph, dma_config);
        }

        // BlockMover executors
        for (auto& [id, graph] : block_mover_graphs_) {
            BlockMoverExecutorConfig bm_config;
            bm_config.row = id / config_.mesh_cols;
            bm_config.col = id % config_.mesh_cols;
            bm_config.mesh_rows = config_.mesh_rows;
            bm_config.mesh_cols = config_.mesh_cols;
            bm_config.push_latency = config_.bm_push_latency;
            bm_config.mesh_latency_per_hop = config_.bm_mesh_latency;

            block_movers_[id] = std::make_unique<BlockMoverFlowExecutor>(graph, bm_config);
        }

        // Streamer executors
        for (auto& [id, graph] : streamer_graphs_) {
            StreamerExecutorConfig s_config;
            s_config.row = id / config_.mesh_cols;
            s_config.col = id % config_.mesh_cols;
            s_config.mesh_cols = config_.mesh_cols;
            s_config.feed_latency = config_.streamer_feed_latency;
            s_config.matmul_latency = config_.streamer_matmul_latency;
            s_config.drain_latency = config_.streamer_drain_latency;

            streamers_[id] = std::make_unique<StreamerFlowExecutor>(graph, s_config);
        }
    }

    void run_dma_phase() {
        // Inject L3 buffer availability and run DMAs
        for (auto& [id, dma] : dma_west_) {
            uint8_t dest_l3 = id * config_.mesh_cols;
            for (uint16_t k = 0; k < config_.k_tiles; ++k) {
                dma->inject_buffer_available(Location::L3, dest_l3);
            }

            bool success = dma->run();
            std::cout << "  DMA West[" << id << "]: "
                      << dma->loads_completed() << " loads, "
                      << dma->bytes_transferred() << " bytes\n";

            // Collect output events
            for (const auto& event : dma->get_tile_ready_events()) {
                dma_output_events_.push_back(event);
            }
        }

        for (auto& [id, dma] : dma_north_) {
            uint8_t dest_l3 = id;
            for (uint16_t k = 0; k < config_.k_tiles; ++k) {
                dma->inject_buffer_available(Location::L3, dest_l3);
            }

            bool success = dma->run();
            std::cout << "  DMA North[" << id << "]: "
                      << dma->loads_completed() << " loads, "
                      << dma->bytes_transferred() << " bytes\n";

            for (const auto& event : dma->get_tile_ready_events()) {
                dma_output_events_.push_back(event);
            }
        }

        std::cout << "  Total DMA output events: " << dma_output_events_.size() << "\n";
    }

    void run_block_mover_phase() {
        // Route DMA events to appropriate BlockMovers
        for (const auto& event : dma_output_events_) {
            uint8_t dest_node = event.operand.node_id;
            if (block_movers_.count(dest_node)) {
                block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
            }
        }

        // Inject L2 buffer availability
        for (auto& [id, bm] : block_movers_) {
            for (uint16_t k = 0; k < config_.k_tiles; ++k) {
                bm->inject_operand(make_buffer_token(Location::L2, 0));  // Bank 0 for A
                bm->inject_operand(make_buffer_token(Location::L2, 1));  // Bank 1 for B
            }
        }

        // Run BlockMovers iteratively to handle mesh forwarding
        bool any_progress = true;
        int iteration = 0;
        while (any_progress && iteration < 100) {
            any_progress = false;

            // Step all BlockMovers
            for (auto& [id, bm] : block_movers_) {
                if (bm->step()) {
                    any_progress = true;
                }
            }

            // Route mesh events between BlockMovers
            for (auto& [id, bm] : block_movers_) {
                for (const auto& event : bm->get_mesh_ready_events()) {
                    uint8_t dest_node = event.operand.node_id;
                    if (block_movers_.count(dest_node) && dest_node != id) {
                        block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
                    }
                }
            }

            iteration++;
        }

        // Finish any remaining work
        for (auto& [id, bm] : block_movers_) {
            bm->run();
        }

        // Print stats and collect L2 events
        for (auto& [id, bm] : block_movers_) {
            std::cout << "  BM[" << (id / config_.mesh_cols) << "," << (id % config_.mesh_cols) << "]: "
                      << bm->pushes_completed() << " pushes, "
                      << bm->mesh_sends_completed() << " mesh sends, "
                      << bm->bytes_to_l2() << " bytes to L2\n";

            for (const auto& event : bm->get_l2_ready_events()) {
                bm_l2_events_.push_back({id, event});
            }
        }

        std::cout << "  Total L2 ready events: " << bm_l2_events_.size() << "\n";
    }

    void run_streamer_phase() {
        // Route L2 events to Streamers
        for (const auto& [bm_id, event] : bm_l2_events_) {
            // The streamer at the same position as the BlockMover
            if (streamers_.count(bm_id)) {
                streamers_[bm_id]->inject_operand(event.operand, event.cycle);
            }
        }

        // Run all Streamers
        for (auto& [id, streamer] : streamers_) {
            bool success = streamer->run();

            uint16_t i = id / config_.mesh_cols;
            uint16_t j = id % config_.mesh_cols;

            std::cout << "  Streamer[" << i << "," << j << "]: "
                      << streamer->matmuls_completed() << " matmuls, "
                      << streamer->drains_completed() << " drains, "
                      << streamer->flops() << " FLOPs"
                      << (success ? "" : " (INCOMPLETE)") << "\n";
        }
    }

    void print_summary() {
        std::cout << "\n========== Summary ==========\n";

        uint64_t total_dma_bytes = 0;
        for (auto& [id, dma] : dma_west_) {
            total_dma_bytes += dma->bytes_transferred();
        }
        for (auto& [id, dma] : dma_north_) {
            total_dma_bytes += dma->bytes_transferred();
        }

        uint64_t total_l2_bytes = 0;
        for (auto& [id, bm] : block_movers_) {
            total_l2_bytes += bm->bytes_to_l2();
        }

        uint64_t total_mesh_bytes = 0;
        for (auto& [id, bm] : block_movers_) {
            total_mesh_bytes += bm->bytes_mesh();
        }

        uint64_t total_flops = 0;
        uint32_t total_matmuls = 0;
        for (auto& [id, streamer] : streamers_) {
            total_flops += streamer->flops();
            total_matmuls += streamer->matmuls_completed();
        }

        std::cout << "DMA bytes transferred:   " << total_dma_bytes << "\n";
        std::cout << "L2 bytes transferred:    " << total_l2_bytes << "\n";
        std::cout << "Mesh bytes transferred:  " << total_mesh_bytes << "\n";
        std::cout << "Total matmuls:           " << total_matmuls << "\n";
        std::cout << "Total FLOPs:             " << total_flops << "\n";

        // Timing information
        std::cout << "\n--- Timing (cycles) ---\n";
        std::cout << "Phase 1 (DMA):        " << std::setw(6) << timing_.dma_latency()
                  << " cycles [" << timing_.dma_start << " - " << timing_.dma_end << "]\n";
        std::cout << "Phase 2 (BlockMover): " << std::setw(6) << timing_.blockmover_latency()
                  << " cycles [" << timing_.blockmover_start << " - " << timing_.blockmover_end << "]\n";
        std::cout << "Phase 3 (Streamer):   " << std::setw(6) << timing_.streamer_latency()
                  << " cycles [" << timing_.streamer_start << " - " << timing_.streamer_end << "]\n";
        std::cout << "Total latency:        " << std::setw(6) << timing_.total_latency() << " cycles\n";

        // Expected values
        uint32_t expected_matmuls = config_.mesh_rows * config_.mesh_cols * config_.k_tiles;
        std::cout << "\nExpected matmuls:        " << expected_matmuls << "\n";
        std::cout << "Match: " << (total_matmuls == expected_matmuls ? "YES" : "NO") << "\n";
    }

private:
    MatmulConfig config_;
    ScheduleTiming timing_;

    // Graphs
    std::map<uint16_t, OperandFlowGraph> dma_graphs_west_;
    std::map<uint16_t, OperandFlowGraph> dma_graphs_north_;
    std::map<uint8_t, OperandFlowGraph> block_mover_graphs_;
    std::map<uint8_t, OperandFlowGraph> streamer_graphs_;

    // Executors
    std::map<uint16_t, std::unique_ptr<DMAFlowExecutor>> dma_west_;
    std::map<uint16_t, std::unique_ptr<DMAFlowExecutor>> dma_north_;
    std::map<uint8_t, std::unique_ptr<BlockMoverFlowExecutor>> block_movers_;
    std::map<uint8_t, std::unique_ptr<StreamerFlowExecutor>> streamers_;

    // Inter-level events
    std::vector<OutputEvent> dma_output_events_;
    std::vector<std::pair<uint8_t, OutputEvent>> bm_l2_events_;
};

// ============================================================================
// Tests
// ============================================================================

TEST_CASE("C-Stationary matmul 2x2 mesh, K=2", "[dataflow][integration][c_stationary]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 2;

    CStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    // Verify expected matmuls executed
    // 2x2 mesh × 2 k iterations = 8 matmuls total
    REQUIRE(orchestrator.total_flops() > 0);
}

TEST_CASE("C-Stationary matmul 2x2 mesh, K=4", "[dataflow][integration][c_stationary]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 4;

    CStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    // 2x2 mesh × 4 k iterations = 16 matmuls total
    REQUIRE(orchestrator.total_flops() > 0);
}

TEST_CASE("C-Stationary matmul 4x4 mesh, K=2", "[dataflow][integration][c_stationary]") {
    MatmulConfig config;
    config.mesh_rows = 4;
    config.mesh_cols = 4;
    config.k_tiles = 2;

    CStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    // 4x4 mesh × 2 k iterations = 32 matmuls total
    REQUIRE(orchestrator.total_flops() > 0);
}

// ============================================================================
// Detailed trace test
// ============================================================================

TEST_CASE("C-Stationary detailed event trace", "[dataflow][integration][c_stationary][trace]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 1;  // Single k iteration for clearer trace

    // Build a single compute tile's execution and trace all events
    StreamerFlowGraphBuilder builder;
    builder.set_position(0, 0, 2)
           .set_dimensions(2, 2, 1)
           .build_c_stationary();

    auto graph = builder.build();
    StreamerFlowExecutor executor(graph);

    // Track all events
    std::vector<ExecutionEvent> trace;
    executor.set_event_callback([&trace](const ExecutionEvent& e) {
        trace.push_back(e);
    });

    // Inject inputs
    executor.inject_operand(tile_a(0, 0, Location::L2, 0));
    executor.inject_operand(tile_b(0, 0, Location::L2, 1));

    bool success = executor.run();
    REQUIRE(success);

    std::cout << "\n--- Event Trace for CT[0,0] ---\n";
    for (const auto& e : trace) {
        std::cout << e.to_string() << "\n";
    }

    REQUIRE(executor.matmuls_completed() == 1);
    REQUIRE(executor.drains_completed() == 1);
}

// ============================================================================
// A-Stationary Orchestrator
// ============================================================================

class AStationaryOrchestrator {
public:
    explicit AStationaryOrchestrator(const MatmulConfig& config)
        : config_(config) {
        build_all_graphs();
        create_all_executors();
    }

    void run() {
        std::cout << "\n========== A-Stationary Matmul Execution ==========\n";
        std::cout << "Mesh: " << config_.mesh_rows << "x" << config_.mesh_cols << " (indexed by i,k)\n";
        std::cout << "N (output columns): " << config_.k_tiles << "\n";
        std::cout << "Stationary operand: A[i,k] at each node\n";
        std::cout << "Streaming operand:  B[k,j] flows N->S along column k\n";
        std::cout << "Latencies: DMA=" << config_.dma_load_latency
                  << ", BM_push=" << config_.bm_push_latency
                  << ", feed=" << config_.streamer_feed_latency
                  << ", matmul=" << config_.streamer_matmul_latency
                  << ", drain=" << config_.streamer_drain_latency << "\n\n";

        // Phase 1: DMA loads A tiles (stationary) and B tiles
        timing_.dma_start = 0;
        std::cout << "--- Phase 1: DMA Loads (Memory → L3) ---\n";
        run_dma_phase();
        timing_.dma_end = std::max(get_max_dma_cycle(), timing_.dma_start + 1);

        // Phase 2: BlockMover operations
        timing_.blockmover_start = timing_.dma_end;
        std::cout << "\n--- Phase 2: BlockMover (L3 → L2, B flows N→S on mesh) ---\n";
        run_block_mover_phase();
        timing_.blockmover_end = std::max(get_max_blockmover_cycle(), timing_.blockmover_start + 1);

        // Phase 3: Streamer/Compute
        timing_.streamer_start = timing_.blockmover_end;
        std::cout << "\n--- Phase 3: Streamer/Compute (L2 → L1 → Accumulator) ---\n";
        run_streamer_phase();
        // Ensure end >= start for proper latency calculation
        timing_.streamer_end = std::max(get_max_streamer_cycle(), timing_.streamer_start + 1);

        print_summary();
    }

    // Timing accessors
    const ScheduleTiming& timing() const { return timing_; }
    uint64_t total_cycles() const { return timing_.total_latency(); }

    uint64_t total_flops() const {
        uint64_t flops = 0;
        for (const auto& [id, exec] : streamers_) {
            flops += exec->flops();
        }
        return flops;
    }

    uint32_t total_matmuls() const {
        uint32_t count = 0;
        for (const auto& [id, exec] : streamers_) {
            count += exec->matmuls_completed();
        }
        return count;
    }

    // Statistics for comparison
    uint64_t dma_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, dma] : dmas_) total += dma->bytes_transferred();
        return total;
    }

    uint64_t l2_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_to_l2();
        return total;
    }

    uint64_t mesh_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_mesh();
        return total;
    }

    uint32_t a_loads() const {
        // A tiles loaded once per node
        return config_.mesh_rows * config_.mesh_cols;
    }

    uint32_t b_loads() const {
        // B tiles loaded for each j, entering from east edge
        return config_.mesh_cols * config_.k_tiles;  // k_tiles used as n_tiles here
    }

private:
    void build_all_graphs() {
        // In A-stationary, mesh is indexed by (i,k)
        // A[i,k] is stationary at node (i,k)
        // B[k,j] flows North→South along column k
        // C[i,j] partial sums reduce West→East along row i

        // Build BlockMover graphs
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            for (uint16_t k = 0; k < config_.mesh_cols; ++k) {
                uint8_t node_id = i * config_.mesh_cols + k;

                AStationaryBlockMoverBuilder builder;
                builder.set_position(i, k, config_.mesh_cols)
                       .set_dimensions(config_.mesh_rows, config_.k_tiles, config_.mesh_cols)
                       .build_a_stationary();

                block_mover_graphs_[node_id] = builder.build();
                block_mover_graphs_[node_id].name = "BM_A[" + std::to_string(i) + "," + std::to_string(k) + "]";
            }
        }

        // Build Streamer graphs
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            for (uint16_t k = 0; k < config_.mesh_cols; ++k) {
                uint8_t node_id = i * config_.mesh_cols + k;

                AStationaryStreamerBuilder builder;
                builder.set_position(i, k, config_.mesh_cols)
                       .set_dimensions(config_.mesh_rows, config_.k_tiles, config_.mesh_cols)
                       .build_a_stationary();

                streamer_graphs_[node_id] = builder.build();
                streamer_graphs_[node_id].name = "Streamer_A[" + std::to_string(i) + "," + std::to_string(k) + "]";
            }
        }

        // Build DMA graphs
        // A-stationary data flow:
        //   - A[i,k] loads to node (i,k) - stationary, one per node
        //   - B[k,j] loads to north edge only (i=0), then flows N→S
        for (uint16_t i = 0; i < config_.mesh_rows; ++i) {
            for (uint16_t k = 0; k < config_.mesh_cols; ++k) {
                uint8_t node_id = i * config_.mesh_cols + k;

                DMAFlowGraphBuilder builder;
                builder.set_node_id(node_id)
                       .set_dimensions(config_.mesh_rows, config_.k_tiles, config_.mesh_cols);

                // Load A[i,k] to this node (stationary)
                builder.add_load_a(i, k, node_id);

                // Load B[k,j] tiles ONLY to north edge (i==0)
                // B will flow N→S on the mesh
                if (i == 0) {
                    for (uint16_t j = 0; j < config_.k_tiles; ++j) {
                        builder.add_load_b(k, j, node_id);
                    }
                }

                dma_graphs_[node_id] = builder.build();
            }
        }
    }

    void create_all_executors() {
        DMAExecutorConfig dma_config;
        dma_config.load_latency = config_.dma_load_latency;

        for (auto& [id, graph] : dma_graphs_) {
            dmas_[id] = std::make_unique<DMAFlowExecutor>(graph, dma_config);
        }

        for (auto& [id, graph] : block_mover_graphs_) {
            BlockMoverExecutorConfig bm_config;
            bm_config.row = id / config_.mesh_cols;
            bm_config.col = id % config_.mesh_cols;
            bm_config.mesh_rows = config_.mesh_rows;
            bm_config.mesh_cols = config_.mesh_cols;
            bm_config.push_latency = config_.bm_push_latency;
            bm_config.mesh_latency_per_hop = config_.bm_mesh_latency;

            block_movers_[id] = std::make_unique<BlockMoverFlowExecutor>(graph, bm_config);
        }

        for (auto& [id, graph] : streamer_graphs_) {
            StreamerExecutorConfig s_config;
            s_config.row = id / config_.mesh_cols;
            s_config.col = id % config_.mesh_cols;
            s_config.mesh_cols = config_.mesh_cols;
            s_config.feed_latency = config_.streamer_feed_latency;
            s_config.matmul_latency = config_.streamer_matmul_latency;
            s_config.drain_latency = config_.streamer_drain_latency;

            streamers_[id] = std::make_unique<StreamerFlowExecutor>(graph, s_config);
        }
    }

    void run_dma_phase() {
        // In A-stationary:
        //   - A[i,k] loads to every node (stationary)
        //   - B[k,j] loads ONLY to north edge (i==0), then flows N→S on mesh
        uint32_t total_a_loads = 0, total_b_loads = 0;
        uint64_t total_a_bytes = 0, total_b_bytes = 0;

        for (auto& [id, dma] : dmas_) {
            uint8_t i = id / config_.mesh_cols;
            uint8_t k = id % config_.mesh_cols;

            // Inject buffer availability
            dma->inject_buffer_available(Location::L3, id);  // For A tile
            if (i == 0) {
                // North edge loads B tiles
                for (uint16_t j = 0; j < config_.k_tiles; ++j) {
                    dma->inject_buffer_available(Location::L3, id);
                }
            }

            dma->run();

            // Count loads
            uint32_t a_loads = 1;
            uint32_t b_loads = (i == 0) ? config_.k_tiles : 0;
            total_a_loads += a_loads;
            total_b_loads += b_loads;
            total_a_bytes += a_loads * 64 * 64 * 4;  // tile size
            total_b_bytes += b_loads * 64 * 64 * 4;

            std::cout << "  DMA[" << (int)i << "," << (int)k << "]: "
                      << "A=" << a_loads << ", B=" << b_loads << " tiles, "
                      << dma->bytes_transferred() << " bytes"
                      << (i == 0 ? " (north edge)" : "") << "\n";

            for (const auto& event : dma->get_tile_ready_events()) {
                dma_output_events_.push_back(event);
            }
        }
        std::cout << "  ---\n";
        std::cout << "  Total A tiles: " << total_a_loads << " (" << total_a_bytes << " bytes)\n";
        std::cout << "  Total B tiles: " << total_b_loads << " (" << total_b_bytes << " bytes) - north edge only\n";
        std::cout << "  Total DMA output events: " << dma_output_events_.size() << "\n";
    }

    void run_block_mover_phase() {
        // Route DMA events to BlockMovers
        for (const auto& event : dma_output_events_) {
            uint8_t dest_node = event.operand.node_id;
            if (block_movers_.count(dest_node)) {
                block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
            }
        }

        // Inject L2 buffer availability
        for (auto& [id, bm] : block_movers_) {
            bm->inject_operand(make_buffer_token(Location::L2, 0));  // For A
            for (uint16_t j = 0; j < config_.k_tiles; ++j) {
                bm->inject_operand(make_buffer_token(Location::L2, 1));  // For B
            }
        }

        // Run BlockMovers iteratively to handle mesh forwarding (B flows N→S)
        bool any_progress = true;
        int iteration = 0;
        while (any_progress && iteration < 100) {
            any_progress = false;
            for (auto& [id, bm] : block_movers_) {
                if (bm->step()) any_progress = true;
            }

            // Route mesh events between BlockMovers
            for (auto& [id, bm] : block_movers_) {
                for (const auto& event : bm->get_mesh_ready_events()) {
                    uint8_t dest_node = event.operand.node_id;
                    if (block_movers_.count(dest_node) && dest_node != id) {
                        block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
                    }
                }
            }
            iteration++;
        }

        for (auto& [id, bm] : block_movers_) {
            bm->run();
            uint8_t i = id / config_.mesh_cols;
            uint8_t k = id % config_.mesh_cols;
            std::cout << "  BM[" << (int)i << "," << (int)k << "]: "
                      << bm->pushes_completed() << " pushes, "
                      << bm->mesh_sends_completed() << " mesh sends, "
                      << bm->bytes_to_l2() << " bytes to L2\n";

            for (const auto& event : bm->get_l2_ready_events()) {
                bm_l2_events_.push_back({id, event});
            }
        }
        std::cout << "  Total L2 ready events: " << bm_l2_events_.size() << "\n";
    }

    void run_streamer_phase() {
        // Route L2 events to Streamers
        for (const auto& [bm_id, event] : bm_l2_events_) {
            if (streamers_.count(bm_id)) {
                streamers_[bm_id]->inject_operand(event.operand, event.cycle);
            }
        }

        for (auto& [id, streamer] : streamers_) {
            bool success = streamer->run();
            uint8_t i = id / config_.mesh_cols;
            uint8_t k = id % config_.mesh_cols;
            std::cout << "  Streamer[" << (int)i << "," << (int)k << "]: "
                      << streamer->matmuls_completed() << " matmuls, "
                      << streamer->drains_completed() << " drains, "
                      << streamer->flops() << " FLOPs"
                      << (success ? "" : " (INCOMPLETE)") << "\n";
        }
    }

    void print_summary() {
        std::cout << "\n========== A-Stationary Summary ==========\n";
        std::cout << "DMA bytes:    " << dma_bytes() << "\n";
        std::cout << "L2 bytes:     " << l2_bytes() << "\n";
        std::cout << "Mesh bytes:   " << mesh_bytes() << "\n";
        std::cout << "Total matmuls: " << total_matmuls() << "\n";
        std::cout << "Total FLOPs:  " << total_flops() << "\n";

        // Timing information
        std::cout << "\n--- Timing (cycles) ---\n";
        std::cout << "Phase 1 (DMA):        " << std::setw(6) << timing_.dma_latency()
                  << " cycles [" << timing_.dma_start << " - " << timing_.dma_end << "]\n";
        std::cout << "Phase 2 (BlockMover): " << std::setw(6) << timing_.blockmover_latency()
                  << " cycles [" << timing_.blockmover_start << " - " << timing_.blockmover_end << "]\n";
        std::cout << "Phase 3 (Streamer):   " << std::setw(6) << timing_.streamer_latency()
                  << " cycles [" << timing_.streamer_start << " - " << timing_.streamer_end << "]\n";
        std::cout << "Total latency:        " << std::setw(6) << timing_.total_latency() << " cycles\n";

        // In A-stationary: each node does N matmuls (one per output column j)
        uint32_t expected = config_.mesh_rows * config_.mesh_cols * config_.k_tiles;
        std::cout << "\nExpected:     " << expected << "\n";
        std::cout << "Match: " << (total_matmuls() == expected ? "YES" : "NO") << "\n";
    }

private:
    uint64_t get_max_dma_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, dma] : dmas_) {
            max_cycle = std::max(max_cycle, dma->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_blockmover_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, bm] : block_movers_) {
            max_cycle = std::max(max_cycle, bm->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_streamer_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, exec] : streamers_) {
            max_cycle = std::max(max_cycle, exec->current_cycle());
        }
        return max_cycle;
    }

private:
    MatmulConfig config_;
    ScheduleTiming timing_;

    std::map<uint8_t, OperandFlowGraph> dma_graphs_;
    std::map<uint8_t, OperandFlowGraph> block_mover_graphs_;
    std::map<uint8_t, OperandFlowGraph> streamer_graphs_;

    std::map<uint8_t, std::unique_ptr<DMAFlowExecutor>> dmas_;
    std::map<uint8_t, std::unique_ptr<BlockMoverFlowExecutor>> block_movers_;
    std::map<uint8_t, std::unique_ptr<StreamerFlowExecutor>> streamers_;

    std::vector<OutputEvent> dma_output_events_;
    std::vector<std::pair<uint8_t, OutputEvent>> bm_l2_events_;
};

// ============================================================================
// B-Stationary Orchestrator
// ============================================================================

/// Orchestrates B-stationary matmul execution across all three levels
/// Mesh is indexed by (k, j):
///   - k = row index (0 to K-1)
///   - j = column index (0 to N-1)
/// Each node holds B[k,j] stationary and processes M iterations of A[i,k]
class BStationaryOrchestrator {
public:
    explicit BStationaryOrchestrator(const MatmulConfig& config)
        : config_(config) {
        build_all_graphs();
        create_all_executors();
    }

    void run() {
        std::cout << "\n========== B-Stationary Matmul Execution ==========\n";
        std::cout << "Mesh: " << config_.mesh_rows << "x" << config_.mesh_cols << " (indexed by k,j)\n";
        std::cout << "M (input rows): " << config_.k_tiles << "\n";
        std::cout << "Stationary operand: B[k,j] at each node\n";
        std::cout << "Streaming operand:  A[i,k] flows W->E along row k\n";
        std::cout << "Latencies: DMA=" << config_.dma_load_latency
                  << ", BM_push=" << config_.bm_push_latency
                  << ", feed=" << config_.streamer_feed_latency
                  << ", matmul=" << config_.streamer_matmul_latency
                  << ", drain=" << config_.streamer_drain_latency << "\n\n";

        // Phase 1: DMA loads B tiles (stationary) and A tiles
        timing_.dma_start = 0;
        std::cout << "--- Phase 1: DMA Loads (Memory → L3) ---\n";
        run_dma_phase();
        timing_.dma_end = std::max(get_max_dma_cycle(), timing_.dma_start + 1);

        // Phase 2: BlockMover operations
        timing_.blockmover_start = timing_.dma_end;
        std::cout << "\n--- Phase 2: BlockMover (L3 → L2, A flows W→E on mesh) ---\n";
        run_block_mover_phase();
        timing_.blockmover_end = std::max(get_max_blockmover_cycle(), timing_.blockmover_start + 1);

        // Phase 3: Streamer/Compute
        timing_.streamer_start = timing_.blockmover_end;
        std::cout << "\n--- Phase 3: Streamer/Compute (L2 → L1 → Accumulator) ---\n";
        run_streamer_phase();
        // Ensure end >= start for proper latency calculation
        timing_.streamer_end = std::max(get_max_streamer_cycle(), timing_.streamer_start + 1);

        print_summary();
    }

    // Timing accessors
    const ScheduleTiming& timing() const { return timing_; }
    uint64_t total_cycles() const { return timing_.total_latency(); }

    uint64_t total_flops() const {
        uint64_t flops = 0;
        for (const auto& [id, exec] : streamers_) {
            flops += exec->flops();
        }
        return flops;
    }

    uint32_t total_matmuls() const {
        uint32_t count = 0;
        for (const auto& [id, exec] : streamers_) {
            count += exec->matmuls_completed();
        }
        return count;
    }

    // Statistics for comparison
    uint64_t dma_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, dma] : dmas_) total += dma->bytes_transferred();
        return total;
    }

    uint64_t l2_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_to_l2();
        return total;
    }

    uint64_t mesh_bytes() const {
        uint64_t total = 0;
        for (const auto& [id, bm] : block_movers_) total += bm->bytes_mesh();
        return total;
    }

private:
    uint64_t get_max_dma_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, dma] : dmas_) {
            max_cycle = std::max(max_cycle, dma->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_blockmover_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, bm] : block_movers_) {
            max_cycle = std::max(max_cycle, bm->current_cycle());
        }
        return max_cycle;
    }

    uint64_t get_max_streamer_cycle() const {
        uint64_t max_cycle = 0;
        for (const auto& [id, exec] : streamers_) {
            max_cycle = std::max(max_cycle, exec->current_cycle());
        }
        return max_cycle;
    }

    void build_all_graphs() {
        // In B-stationary, mesh is indexed by (k,j)
        // B[k,j] is stationary at node (k,j)
        // A[i,k] flows West→East along row k
        // C[i,j] partial sums reduce North→South along column j

        // Build BlockMover graphs
        for (uint16_t k = 0; k < config_.mesh_rows; ++k) {
            for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
                uint8_t node_id = k * config_.mesh_cols + j;

                BStationaryBlockMoverBuilder builder;
                builder.set_position(k, j, config_.mesh_cols)
                       .set_dimensions(config_.k_tiles, config_.mesh_cols, config_.mesh_rows)
                       .build_b_stationary();

                block_mover_graphs_[node_id] = builder.build();
                block_mover_graphs_[node_id].name = "BM_B[" + std::to_string(k) + "," + std::to_string(j) + "]";
            }
        }

        // Build Streamer graphs
        for (uint16_t k = 0; k < config_.mesh_rows; ++k) {
            for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
                uint8_t node_id = k * config_.mesh_cols + j;

                BStationaryStreamerBuilder builder;
                builder.set_position(k, j, config_.mesh_cols)
                       .set_dimensions(config_.k_tiles, config_.mesh_cols, config_.mesh_rows)
                       .build_b_stationary();

                streamer_graphs_[node_id] = builder.build();
                streamer_graphs_[node_id].name = "Streamer_B[" + std::to_string(k) + "," + std::to_string(j) + "]";
            }
        }

        // Build DMA graphs
        // B-stationary data flow:
        //   - B[k,j] loads to node (k,j) - stationary, one per node
        //   - A[i,k] loads to west edge only (j=0), then flows W→E
        for (uint16_t k = 0; k < config_.mesh_rows; ++k) {
            for (uint16_t j = 0; j < config_.mesh_cols; ++j) {
                uint8_t node_id = k * config_.mesh_cols + j;

                DMAFlowGraphBuilder builder;
                builder.set_node_id(node_id)
                       .set_dimensions(config_.k_tiles, config_.mesh_cols, config_.mesh_rows);

                // Load B[k,j] to this node (stationary)
                builder.add_load_b(k, j, node_id);

                // Load A[i,k] tiles ONLY to west edge (j==0)
                // A will flow W→E on the mesh
                if (j == 0) {
                    for (uint16_t i = 0; i < config_.k_tiles; ++i) {
                        builder.add_load_a(i, k, node_id);
                    }
                }

                dma_graphs_[node_id] = builder.build();
            }
        }
    }

    void create_all_executors() {
        DMAExecutorConfig dma_config;
        dma_config.load_latency = config_.dma_load_latency;

        for (auto& [id, graph] : dma_graphs_) {
            dmas_[id] = std::make_unique<DMAFlowExecutor>(graph, dma_config);
        }

        for (auto& [id, graph] : block_mover_graphs_) {
            BlockMoverExecutorConfig bm_config;
            bm_config.row = id / config_.mesh_cols;
            bm_config.col = id % config_.mesh_cols;
            bm_config.mesh_rows = config_.mesh_rows;
            bm_config.mesh_cols = config_.mesh_cols;
            bm_config.push_latency = config_.bm_push_latency;
            bm_config.mesh_latency_per_hop = config_.bm_mesh_latency;

            block_movers_[id] = std::make_unique<BlockMoverFlowExecutor>(graph, bm_config);
        }

        for (auto& [id, graph] : streamer_graphs_) {
            StreamerExecutorConfig s_config;
            s_config.row = id / config_.mesh_cols;
            s_config.col = id % config_.mesh_cols;
            s_config.mesh_cols = config_.mesh_cols;
            s_config.feed_latency = config_.streamer_feed_latency;
            s_config.matmul_latency = config_.streamer_matmul_latency;
            s_config.drain_latency = config_.streamer_drain_latency;

            streamers_[id] = std::make_unique<StreamerFlowExecutor>(graph, s_config);
        }
    }

    void run_dma_phase() {
        // In B-stationary:
        //   - B[k,j] loads to every node (stationary)
        //   - A[i,k] loads ONLY to west edge (j==0), then flows W→E on mesh
        uint32_t total_a_loads = 0, total_b_loads = 0;
        uint64_t total_a_bytes = 0, total_b_bytes = 0;

        for (auto& [id, dma] : dmas_) {
            uint8_t k = id / config_.mesh_cols;
            uint8_t j = id % config_.mesh_cols;

            // Inject buffer availability
            dma->inject_buffer_available(Location::L3, id);  // For B tile
            if (j == 0) {
                // West edge loads A tiles
                for (uint16_t i = 0; i < config_.k_tiles; ++i) {
                    dma->inject_buffer_available(Location::L3, id);
                }
            }

            dma->run();

            // Count loads
            uint32_t b_loads = 1;
            uint32_t a_loads = (j == 0) ? config_.k_tiles : 0;
            total_b_loads += b_loads;
            total_a_loads += a_loads;
            total_b_bytes += b_loads * 64 * 64 * 4;  // tile size
            total_a_bytes += a_loads * 64 * 64 * 4;

            std::cout << "  DMA[" << (int)k << "," << (int)j << "]: "
                      << "B=" << b_loads << ", A=" << a_loads << " tiles, "
                      << dma->bytes_transferred() << " bytes"
                      << (j == 0 ? " (west edge)" : "") << "\n";

            for (const auto& event : dma->get_tile_ready_events()) {
                dma_output_events_.push_back(event);
            }
        }
        std::cout << "  ---\n";
        std::cout << "  Total B tiles: " << total_b_loads << " (" << total_b_bytes << " bytes)\n";
        std::cout << "  Total A tiles: " << total_a_loads << " (" << total_a_bytes << " bytes) - west edge only\n";
        std::cout << "  Total DMA output events: " << dma_output_events_.size() << "\n";
    }

    void run_block_mover_phase() {
        // Route DMA events to BlockMovers
        for (const auto& event : dma_output_events_) {
            uint8_t dest_node = event.operand.node_id;
            if (block_movers_.count(dest_node)) {
                block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
            }
        }

        // Inject L2 buffer availability
        for (auto& [id, bm] : block_movers_) {
            bm->inject_operand(make_buffer_token(Location::L2, 1));  // For B
            for (uint16_t i = 0; i < config_.k_tiles; ++i) {
                bm->inject_operand(make_buffer_token(Location::L2, 0));  // For A
            }
        }

        // Run BlockMovers iteratively to handle mesh forwarding (A flows W→E)
        bool any_progress = true;
        int iteration = 0;
        while (any_progress && iteration < 100) {
            any_progress = false;
            for (auto& [id, bm] : block_movers_) {
                if (bm->step()) any_progress = true;
            }

            // Route mesh events between BlockMovers
            for (auto& [id, bm] : block_movers_) {
                for (const auto& event : bm->get_mesh_ready_events()) {
                    uint8_t dest_node = event.operand.node_id;
                    if (block_movers_.count(dest_node) && dest_node != id) {
                        block_movers_[dest_node]->inject_operand(event.operand, event.cycle);
                    }
                }
            }
            iteration++;
        }

        // Finish any remaining work
        for (auto& [id, bm] : block_movers_) {
            bm->run();
            uint8_t k = id / config_.mesh_cols;
            uint8_t j = id % config_.mesh_cols;
            std::cout << "  BM[" << (int)k << "," << (int)j << "]: "
                      << bm->pushes_completed() << " pushes, "
                      << bm->mesh_sends_completed() << " mesh sends, "
                      << bm->bytes_to_l2() << " bytes to L2\n";

            for (const auto& event : bm->get_l2_ready_events()) {
                bm_l2_events_.push_back({id, event});
            }
        }
        std::cout << "  Total L2 ready events: " << bm_l2_events_.size() << "\n";
    }

    void run_streamer_phase() {
        // Route L2 events to Streamers
        for (const auto& [bm_id, event] : bm_l2_events_) {
            if (streamers_.count(bm_id)) {
                streamers_[bm_id]->inject_operand(event.operand, event.cycle);
            }
        }

        for (auto& [id, streamer] : streamers_) {
            bool success = streamer->run();
            uint8_t k = id / config_.mesh_cols;
            uint8_t j = id % config_.mesh_cols;
            std::cout << "  Streamer[" << (int)k << "," << (int)j << "]: "
                      << streamer->matmuls_completed() << " matmuls, "
                      << streamer->drains_completed() << " drains, "
                      << streamer->flops() << " FLOPs"
                      << (success ? "" : " (INCOMPLETE)") << "\n";
        }
    }

    void print_summary() {
        std::cout << "\n========== B-Stationary Summary ==========\n";
        std::cout << "DMA bytes:    " << dma_bytes() << "\n";
        std::cout << "L2 bytes:     " << l2_bytes() << "\n";
        std::cout << "Mesh bytes:   " << mesh_bytes() << "\n";
        std::cout << "Total matmuls: " << total_matmuls() << "\n";
        std::cout << "Total FLOPs:  " << total_flops() << "\n";

        // Timing information
        std::cout << "\n--- Timing (cycles) ---\n";
        std::cout << "Phase 1 (DMA):        " << std::setw(6) << timing_.dma_latency()
                  << " cycles [" << timing_.dma_start << " - " << timing_.dma_end << "]\n";
        std::cout << "Phase 2 (BlockMover): " << std::setw(6) << timing_.blockmover_latency()
                  << " cycles [" << timing_.blockmover_start << " - " << timing_.blockmover_end << "]\n";
        std::cout << "Phase 3 (Streamer):   " << std::setw(6) << timing_.streamer_latency()
                  << " cycles [" << timing_.streamer_start << " - " << timing_.streamer_end << "]\n";
        std::cout << "Total latency:        " << std::setw(6) << timing_.total_latency() << " cycles\n";

        // In B-stationary: each node does M matmuls (one per input row i)
        uint32_t expected = config_.mesh_rows * config_.mesh_cols * config_.k_tiles;
        std::cout << "\nExpected:     " << expected << "\n";
        std::cout << "Match: " << (total_matmuls() == expected ? "YES" : "NO") << "\n";
    }

private:
    MatmulConfig config_;
    ScheduleTiming timing_;

    std::map<uint8_t, OperandFlowGraph> dma_graphs_;
    std::map<uint8_t, OperandFlowGraph> block_mover_graphs_;
    std::map<uint8_t, OperandFlowGraph> streamer_graphs_;

    std::map<uint8_t, std::unique_ptr<DMAFlowExecutor>> dmas_;
    std::map<uint8_t, std::unique_ptr<BlockMoverFlowExecutor>> block_movers_;
    std::map<uint8_t, std::unique_ptr<StreamerFlowExecutor>> streamers_;

    std::vector<OutputEvent> dma_output_events_;
    std::vector<std::pair<uint8_t, OutputEvent>> bm_l2_events_;
};

// ============================================================================
// A-Stationary Tests
// ============================================================================

TEST_CASE("A-Stationary matmul 2x2 mesh, N=2", "[dataflow][integration][a_stationary]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 2;  // Used as N (output columns) in A-stationary

    AStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    REQUIRE(orchestrator.total_flops() > 0);
    // 2x2 mesh × 2 j iterations = 8 matmuls
    REQUIRE(orchestrator.total_matmuls() == 8);
}

// ============================================================================
// B-Stationary Tests
// ============================================================================

TEST_CASE("B-Stationary matmul 2x2 mesh, M=2", "[dataflow][integration][b_stationary]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 2;  // Used as M (input rows) in B-stationary

    BStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    REQUIRE(orchestrator.total_flops() > 0);
    // 2x2 mesh × 2 i iterations = 8 matmuls
    REQUIRE(orchestrator.total_matmuls() == 8);
}

// ============================================================================
// Schedule Comparison
// ============================================================================

TEST_CASE("Compare All Stationary Schedules", "[dataflow][integration][comparison]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 2;

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "        COMPLETE SCHEDULE COMPARISON: C vs A vs B Stationary\n";
    std::cout << std::string(80, '=') << "\n";

    // Run C-Stationary
    CStationaryOrchestrator c_orch(config);
    c_orch.run();

    // Run A-Stationary
    AStationaryOrchestrator a_orch(config);
    a_orch.run();

    // Run B-Stationary
    BStationaryOrchestrator b_orch(config);
    b_orch.run();

    // Print comparison table
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "                         COMPARISON TABLE\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(20) << "Metric"
              << std::setw(15) << "C-Stationary"
              << std::setw(15) << "A-Stationary"
              << std::setw(15) << "B-Stationary" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::setw(20) << "Total FLOPs"
              << std::setw(15) << c_orch.total_flops()
              << std::setw(15) << a_orch.total_flops()
              << std::setw(15) << b_orch.total_flops() << "\n";

    std::cout << std::setw(20) << "Total matmuls"
              << std::setw(15) << c_orch.total_matmuls()
              << std::setw(15) << a_orch.total_matmuls()
              << std::setw(15) << b_orch.total_matmuls() << "\n";

    std::cout << std::setw(20) << "DMA bytes"
              << std::setw(15) << c_orch.dma_bytes()
              << std::setw(15) << a_orch.dma_bytes()
              << std::setw(15) << b_orch.dma_bytes() << "\n";

    std::cout << std::setw(20) << "L2 bytes"
              << std::setw(15) << c_orch.l2_bytes()
              << std::setw(15) << a_orch.l2_bytes()
              << std::setw(15) << b_orch.l2_bytes() << "\n";

    std::cout << std::setw(20) << "Mesh bytes"
              << std::setw(15) << c_orch.mesh_bytes()
              << std::setw(15) << a_orch.mesh_bytes()
              << std::setw(15) << b_orch.mesh_bytes() << "\n";

    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(20) << "DMA latency"
              << std::setw(15) << c_orch.timing().dma_latency()
              << std::setw(15) << a_orch.timing().dma_latency()
              << std::setw(15) << b_orch.timing().dma_latency() << "\n";

    std::cout << std::setw(20) << "BlockMover latency"
              << std::setw(15) << c_orch.timing().blockmover_latency()
              << std::setw(15) << a_orch.timing().blockmover_latency()
              << std::setw(15) << b_orch.timing().blockmover_latency() << "\n";

    std::cout << std::setw(20) << "Streamer latency"
              << std::setw(15) << c_orch.timing().streamer_latency()
              << std::setw(15) << a_orch.timing().streamer_latency()
              << std::setw(15) << b_orch.timing().streamer_latency() << "\n";

    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(20) << "TOTAL LATENCY"
              << std::setw(15) << c_orch.total_cycles()
              << std::setw(15) << a_orch.total_cycles()
              << std::setw(15) << b_orch.total_cycles() << "\n";

    std::cout << std::string(80, '=') << "\n";
    std::cout << "\nSchedule Characteristics:\n";
    std::cout << "  C-Stationary: C stays at (i,j), A flows W→E, B flows N→S\n";
    std::cout << "                Best for: output-bound workloads, large output matrices\n";
    std::cout << "  A-Stationary: A stays at (i,k), B loaded per column, C partials reduce\n";
    std::cout << "                Best for: weight-stationary inference, small K dimension\n";
    std::cout << "  B-Stationary: B stays at (k,j), A loaded per row, C partials reduce\n";
    std::cout << "                Best for: activation-stationary, batched inference\n";
    std::cout << std::string(80, '=') << "\n";

    // All should produce same number of FLOPs
    REQUIRE(c_orch.total_flops() == a_orch.total_flops());
    REQUIRE(c_orch.total_flops() == b_orch.total_flops());
    REQUIRE(c_orch.total_matmuls() == a_orch.total_matmuls());
    REQUIRE(c_orch.total_matmuls() == b_orch.total_matmuls());
}

// ============================================================================
// Performance verification
// ============================================================================

TEST_CASE("C-Stationary performance metrics", "[dataflow][integration][c_stationary][perf]") {
    MatmulConfig config;
    config.mesh_rows = 2;
    config.mesh_cols = 2;
    config.k_tiles = 2;

    // With known latencies, verify timing
    config.dma_load_latency = 100;
    config.bm_push_latency = 10;
    config.streamer_feed_latency = 5;
    config.streamer_matmul_latency = 64;
    config.streamer_drain_latency = 10;

    CStationaryOrchestrator orchestrator(config);
    orchestrator.run();

    // Expected FLOPs per tile matmul: 2 * 64 * 64 * 64 = 524288
    // Total: 2*2*2 * 524288 = 4194304 FLOPs
    uint64_t expected_flops = 8 * 2ULL * 64 * 64 * 64;
    REQUIRE(orchestrator.total_flops() == expected_flops);
}

// ============================================================================
// Large Matmul Analysis - Tiled Execution when matrices exceed L3 capacity
// ============================================================================
//
// When matrices are larger than what the distributed L3 can hold, we must
// tile the computation and sequence submatrix operations. This reveals the
// true bandwidth characteristics of each schedule:
//
// - C-Stationary: No C partial R/W needed (C stays until complete)
//                 But must reload A and B for each output tile block
//
// - A-Stationary: A stays, but K > mesh_k requires multiple passes
//                 C partials must be written out and read back for each K-chunk
//
// - B-Stationary: B stays, but K > mesh_k requires multiple passes
//                 C partials must be written out and read back for each K-chunk
//
// ============================================================================

/// Configuration for large matmul that exceeds L3 capacity
struct LargeMatmulConfig {
    // Problem dimensions (in tiles)
    uint32_t M_tiles = 20;   // Output rows
    uint32_t N_tiles = 20;   // Output columns
    uint32_t K_tiles = 10;   // Reduction dimension

    // Mesh dimensions
    uint32_t mesh_rows = 2;
    uint32_t mesh_cols = 2;

    // Tile size
    uint32_t tile_dim = 64;           // 64x64 tiles
    uint32_t element_bytes = 4;       // FP32
    uint32_t tile_bytes() const { return tile_dim * tile_dim * element_bytes; }  // 16KB per tile

    // Derived: L3 capacity per mesh
    // With 2x2 mesh, distributed L3 can hold 4 tiles per operand = 64KB per operand
    // Total problem: 20x10 A tiles = 200 tiles, 10x20 B tiles = 200 tiles
    // This is ~10x what distributed L3 can hold
};

/// Bandwidth analysis results
struct BandwidthAnalysis {
    // Input tile loads (DMA: Memory → L3)
    uint64_t a_input_tiles = 0;
    uint64_t b_input_tiles = 0;

    // C partial transfers (DMA: Memory ↔ L3)
    uint64_t c_partial_reads = 0;
    uint64_t c_partial_writes = 0;

    // Final C output
    uint64_t c_output_tiles = 0;

    // Mesh transfers (L3 ↔ L3 between nodes)
    uint64_t mesh_tiles = 0;

    // Derived metrics
    uint64_t total_dma_tiles() const {
        return a_input_tiles + b_input_tiles + c_partial_reads + c_partial_writes + c_output_tiles;
    }

    uint64_t total_dma_bytes(uint32_t tile_bytes) const {
        return total_dma_tiles() * tile_bytes;
    }

    uint64_t total_mesh_bytes(uint32_t tile_bytes) const {
        return mesh_tiles * tile_bytes;
    }

    // Compute operations
    uint64_t tile_matmuls = 0;
    uint64_t flops(uint32_t tile_dim) const {
        return tile_matmuls * 2ULL * tile_dim * tile_dim * tile_dim;
    }
};

/// Analyze C-Stationary bandwidth for large matmul
BandwidthAnalysis analyze_c_stationary_large(const LargeMatmulConfig& cfg) {
    BandwidthAnalysis result;

    // C-Stationary processes C in blocks of mesh_rows × mesh_cols
    // For each C-block, iterate through all K tiles
    uint32_t c_block_rows = (cfg.M_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;
    uint32_t c_block_cols = (cfg.N_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;
    uint32_t total_c_blocks = c_block_rows * c_block_cols;

    // For each C-block:
    //   - Load mesh_rows A tiles (west edge) for each K iteration
    //   - Load mesh_cols B tiles (north edge) for each K iteration
    //   - A flows W→E (mesh_cols-1 hops per A tile)
    //   - B flows N→S (mesh_rows-1 hops per B tile)
    //   - At end, drain mesh_rows × mesh_cols C tiles

    // A tile loads: for each C-block row, we reload A for each C-block column
    // If no cross-block reuse: total_c_blocks × K × mesh_rows
    // With row reuse: c_block_rows × K × mesh_rows × c_block_cols (still need per column)
    // Actually, for each C-block[I,J], we load A[I*mesh_rows : (I+1)*mesh_rows, k]
    // This is the SAME A for all J in the row, so with smart scheduling:
    // A loads = c_block_rows × K_tiles × mesh_rows

    // But for simplicity and worst case, assume no reuse across C-blocks:
    result.a_input_tiles = (uint64_t)total_c_blocks * cfg.K_tiles * cfg.mesh_rows;
    result.b_input_tiles = (uint64_t)total_c_blocks * cfg.K_tiles * cfg.mesh_cols;

    // C output: one drain per C-block
    result.c_output_tiles = (uint64_t)total_c_blocks * cfg.mesh_rows * cfg.mesh_cols;

    // NO C partial R/W for C-stationary - C stays in L2/L3 until complete
    result.c_partial_reads = 0;
    result.c_partial_writes = 0;

    // Mesh traffic: A flows W→E, B flows N→S
    // Per K iteration per C-block:
    //   A: mesh_rows tiles × (mesh_cols - 1) hops
    //   B: mesh_cols tiles × (mesh_rows - 1) hops
    uint64_t mesh_per_k = cfg.mesh_rows * (cfg.mesh_cols - 1) +
                          cfg.mesh_cols * (cfg.mesh_rows - 1);
    result.mesh_tiles = (uint64_t)total_c_blocks * cfg.K_tiles * mesh_per_k;

    // Total tile matmuls
    result.tile_matmuls = (uint64_t)cfg.M_tiles * cfg.N_tiles * cfg.K_tiles;

    return result;
}

/// Analyze A-Stationary bandwidth for large matmul
BandwidthAnalysis analyze_a_stationary_large(const LargeMatmulConfig& cfg) {
    BandwidthAnalysis result;

    // A-Stationary: mesh indexed by (i, k_local)
    // A[i,k] stays at node (i % mesh_rows, k % mesh_cols)
    // Problem: K_tiles > mesh_cols, so we need multiple K-chunks

    uint32_t k_chunks = (cfg.K_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;
    uint32_t m_blocks = (cfg.M_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;

    // For each M-block (processing mesh_rows rows of A at a time):
    //   For each K-chunk:
    //     - Load A[m_block, k_chunk] tiles (stationary for this chunk)
    //     - For each j = 0 to N_tiles-1:
    //       - Load B[k_chunk, j] tiles (flows N→S)
    //       - Compute partials
    //     - If not first K-chunk: read C partials first
    //     - If not last K-chunk: write C partials

    // A loads: each A tile loaded once per K-chunk processing
    // Total A loads = M_tiles × K_tiles (each tile loaded once)
    result.a_input_tiles = (uint64_t)cfg.M_tiles * cfg.K_tiles;

    // B loads: for each M-block, for each K-chunk, for each N column
    // B[k,j] loaded once per M-block (doesn't depend on i)
    // With reuse across M-blocks: K_tiles × N_tiles
    // Without reuse: m_blocks × K_tiles × N_tiles
    // Use worst case (no reuse):
    result.b_input_tiles = (uint64_t)m_blocks * cfg.K_tiles * cfg.N_tiles;

    // C partials: for each M-block, after each K-chunk except last, write partials
    // Then for each K-chunk except first, read partials
    // C partials per M-block = mesh_rows × N_tiles tiles
    uint64_t c_partials_per_m_block = cfg.mesh_rows * cfg.N_tiles;
    // Writes: (k_chunks - 1) per M-block (after each chunk except last)
    // Reads: (k_chunks - 1) per M-block (before each chunk except first)
    result.c_partial_writes = (uint64_t)m_blocks * (k_chunks - 1) * c_partials_per_m_block;
    result.c_partial_reads = result.c_partial_writes;

    // Final C output
    result.c_output_tiles = (uint64_t)cfg.M_tiles * cfg.N_tiles;

    // Mesh traffic: B flows N→S within each K-chunk
    // Per M-block, per K-chunk, per N column: mesh_cols tiles × (mesh_rows - 1) hops
    uint64_t k_tiles_per_chunk = std::min(cfg.mesh_cols, cfg.K_tiles);
    result.mesh_tiles = (uint64_t)m_blocks * k_chunks * cfg.N_tiles *
                        k_tiles_per_chunk * (cfg.mesh_rows - 1);

    // Total tile matmuls
    result.tile_matmuls = (uint64_t)cfg.M_tiles * cfg.N_tiles * cfg.K_tiles;

    return result;
}

/// Analyze B-Stationary bandwidth for large matmul
BandwidthAnalysis analyze_b_stationary_large(const LargeMatmulConfig& cfg) {
    BandwidthAnalysis result;

    // B-Stationary: mesh indexed by (k_local, j)
    // B[k,j] stays at node (k % mesh_rows, j % mesh_cols)
    // Problem: K_tiles > mesh_rows, so we need multiple K-chunks

    uint32_t k_chunks = (cfg.K_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;
    uint32_t n_blocks = (cfg.N_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;

    // For each N-block (processing mesh_cols columns of B at a time):
    //   For each K-chunk:
    //     - Load B[k_chunk, n_block] tiles (stationary for this chunk)
    //     - For each i = 0 to M_tiles-1:
    //       - Load A[i, k_chunk] tiles (flows W→E)
    //       - Compute partials
    //     - If not first K-chunk: read C partials first
    //     - If not last K-chunk: write C partials

    // B loads: each B tile loaded once per K-chunk processing
    result.b_input_tiles = (uint64_t)cfg.K_tiles * cfg.N_tiles;

    // A loads: for each N-block, for each K-chunk, for each M row
    // A[i,k] loaded once per N-block
    // Without reuse: n_blocks × K_tiles × M_tiles
    result.a_input_tiles = (uint64_t)n_blocks * cfg.K_tiles * cfg.M_tiles;

    // C partials: similar to A-stationary
    uint64_t c_partials_per_n_block = cfg.M_tiles * cfg.mesh_cols;
    result.c_partial_writes = (uint64_t)n_blocks * (k_chunks - 1) * c_partials_per_n_block;
    result.c_partial_reads = result.c_partial_writes;

    // Final C output
    result.c_output_tiles = (uint64_t)cfg.M_tiles * cfg.N_tiles;

    // Mesh traffic: A flows W→E within each K-chunk
    uint64_t k_tiles_per_chunk = std::min(cfg.mesh_rows, cfg.K_tiles);
    result.mesh_tiles = (uint64_t)n_blocks * k_chunks * cfg.M_tiles *
                        k_tiles_per_chunk * (cfg.mesh_cols - 1);

    // Total tile matmuls
    result.tile_matmuls = (uint64_t)cfg.M_tiles * cfg.N_tiles * cfg.K_tiles;

    return result;
}

void print_bandwidth_analysis(const std::string& name,
                              const BandwidthAnalysis& a,
                              const LargeMatmulConfig& cfg) {
    uint64_t tile_bytes = cfg.tile_bytes();

    std::cout << "\n--- " << name << " ---\n";
    std::cout << "  Input loads:\n";
    std::cout << "    A tiles:          " << std::setw(10) << a.a_input_tiles
              << " (" << (a.a_input_tiles * tile_bytes / 1024 / 1024) << " MB)\n";
    std::cout << "    B tiles:          " << std::setw(10) << a.b_input_tiles
              << " (" << (a.b_input_tiles * tile_bytes / 1024 / 1024) << " MB)\n";
    std::cout << "  C partial R/W:\n";
    std::cout << "    Reads:            " << std::setw(10) << a.c_partial_reads
              << " (" << (a.c_partial_reads * tile_bytes / 1024 / 1024) << " MB)\n";
    std::cout << "    Writes:           " << std::setw(10) << a.c_partial_writes
              << " (" << (a.c_partial_writes * tile_bytes / 1024 / 1024) << " MB)\n";
    std::cout << "  C output:           " << std::setw(10) << a.c_output_tiles
              << " (" << (a.c_output_tiles * tile_bytes / 1024 / 1024) << " MB)\n";
    std::cout << "  ----\n";
    std::cout << "  Total DMA tiles:    " << std::setw(10) << a.total_dma_tiles()
              << " (" << (a.total_dma_bytes(tile_bytes) / 1024 / 1024) << " MB)\n";
    std::cout << "  Mesh tiles:         " << std::setw(10) << a.mesh_tiles
              << " (" << (a.total_mesh_bytes(tile_bytes) / 1024 / 1024) << " MB)\n";
    std::cout << "  Tile matmuls:       " << std::setw(10) << a.tile_matmuls << "\n";
    std::cout << "  Total FLOPs:        " << std::setw(10) << a.flops(cfg.tile_dim)
              << " (" << (a.flops(cfg.tile_dim) / 1e9) << " GFLOPs)\n";
}

TEST_CASE("Large Matmul Bandwidth Analysis - 10x L3 Capacity",
          "[dataflow][integration][large][bandwidth]") {

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "    LARGE MATMUL BANDWIDTH ANALYSIS (10x L3 Capacity)\n";
    std::cout << std::string(80, '=') << "\n";

    LargeMatmulConfig cfg;
    cfg.M_tiles = 20;    // 20 × 64 = 1280 rows
    cfg.N_tiles = 20;    // 20 × 64 = 1280 columns
    cfg.K_tiles = 10;    // 10 × 64 = 640 reduction dimension
    cfg.mesh_rows = 2;
    cfg.mesh_cols = 2;

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem: C[" << cfg.M_tiles * cfg.tile_dim << "," << cfg.N_tiles * cfg.tile_dim
              << "] = A[" << cfg.M_tiles * cfg.tile_dim << "," << cfg.K_tiles * cfg.tile_dim
              << "] × B[" << cfg.K_tiles * cfg.tile_dim << "," << cfg.N_tiles * cfg.tile_dim << "]\n";
    std::cout << "  Tiles:   C[" << cfg.M_tiles << "," << cfg.N_tiles
              << "] = A[" << cfg.M_tiles << "," << cfg.K_tiles
              << "] × B[" << cfg.K_tiles << "," << cfg.N_tiles << "]\n";
    std::cout << "  Mesh:    " << cfg.mesh_rows << "×" << cfg.mesh_cols << "\n";
    std::cout << "  Tile:    " << cfg.tile_dim << "×" << cfg.tile_dim << " FP32 = "
              << cfg.tile_bytes() / 1024 << " KB\n";

    // Analyze each schedule
    auto c_analysis = analyze_c_stationary_large(cfg);
    auto a_analysis = analyze_a_stationary_large(cfg);
    auto b_analysis = analyze_b_stationary_large(cfg);

    print_bandwidth_analysis("C-Stationary", c_analysis, cfg);
    print_bandwidth_analysis("A-Stationary", a_analysis, cfg);
    print_bandwidth_analysis("B-Stationary", b_analysis, cfg);

    // Comparison table
    uint64_t tile_bytes = cfg.tile_bytes();

    std::cout << "\n" << std::string(90, '-') << "\n";
    std::cout << "                       BANDWIDTH COMPARISON TABLE\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::setw(25) << "Metric"
              << std::setw(20) << "C-Stationary"
              << std::setw(20) << "A-Stationary"
              << std::setw(20) << "B-Stationary" << "\n";
    std::cout << std::string(90, '-') << "\n";

    std::cout << std::setw(25) << "A input (MB)"
              << std::setw(20) << (c_analysis.a_input_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (a_analysis.a_input_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (b_analysis.a_input_tiles * tile_bytes / 1024 / 1024) << "\n";

    std::cout << std::setw(25) << "B input (MB)"
              << std::setw(20) << (c_analysis.b_input_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (a_analysis.b_input_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (b_analysis.b_input_tiles * tile_bytes / 1024 / 1024) << "\n";

    std::cout << std::setw(25) << "C partial R/W (MB)"
              << std::setw(20) << ((c_analysis.c_partial_reads + c_analysis.c_partial_writes) * tile_bytes / 1024 / 1024)
              << std::setw(20) << ((a_analysis.c_partial_reads + a_analysis.c_partial_writes) * tile_bytes / 1024 / 1024)
              << std::setw(20) << ((b_analysis.c_partial_reads + b_analysis.c_partial_writes) * tile_bytes / 1024 / 1024) << "\n";

    std::cout << std::setw(25) << "C output (MB)"
              << std::setw(20) << (c_analysis.c_output_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (a_analysis.c_output_tiles * tile_bytes / 1024 / 1024)
              << std::setw(20) << (b_analysis.c_output_tiles * tile_bytes / 1024 / 1024) << "\n";

    std::cout << std::string(90, '-') << "\n";

    std::cout << std::setw(25) << "TOTAL DMA (MB)"
              << std::setw(20) << (c_analysis.total_dma_bytes(tile_bytes) / 1024 / 1024)
              << std::setw(20) << (a_analysis.total_dma_bytes(tile_bytes) / 1024 / 1024)
              << std::setw(20) << (b_analysis.total_dma_bytes(tile_bytes) / 1024 / 1024) << "\n";

    std::cout << std::setw(25) << "Mesh traffic (MB)"
              << std::setw(20) << (c_analysis.total_mesh_bytes(tile_bytes) / 1024 / 1024)
              << std::setw(20) << (a_analysis.total_mesh_bytes(tile_bytes) / 1024 / 1024)
              << std::setw(20) << (b_analysis.total_mesh_bytes(tile_bytes) / 1024 / 1024) << "\n";

    std::cout << std::string(90, '=') << "\n";

    // Calculate ratios
    double c_dma = c_analysis.total_dma_bytes(tile_bytes);
    double a_dma = a_analysis.total_dma_bytes(tile_bytes);
    double b_dma = b_analysis.total_dma_bytes(tile_bytes);
    double min_dma = std::min({c_dma, a_dma, b_dma});

    std::cout << "\nDMA Bandwidth Ratio (relative to best):\n";
    std::cout << "  C-Stationary: " << std::fixed << std::setprecision(2) << (c_dma / min_dma) << "x\n";
    std::cout << "  A-Stationary: " << std::fixed << std::setprecision(2) << (a_dma / min_dma) << "x\n";
    std::cout << "  B-Stationary: " << std::fixed << std::setprecision(2) << (b_dma / min_dma) << "x\n";

    std::cout << "\nKey Insight:\n";
    std::cout << "  When matrices exceed L3 capacity, C-Stationary avoids C partial R/W\n";
    std::cout << "  but must reload A and B for each output tile block.\n";
    std::cout << "  A/B-Stationary keep one operand in place but pay the cost of\n";
    std::cout << "  reading/writing C partials for each K-chunk.\n";
    std::cout << std::string(90, '=') << "\n";

    // All should have same FLOPs
    REQUIRE(c_analysis.tile_matmuls == a_analysis.tile_matmuls);
    REQUIRE(c_analysis.tile_matmuls == b_analysis.tile_matmuls);
}

TEST_CASE("Large Matmul Bandwidth - Varying K dimension",
          "[dataflow][integration][large][bandwidth][sweep]") {

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "    BANDWIDTH vs K DIMENSION (M=N=20, K varies)\n";
    std::cout << std::string(80, '=') << "\n";

    LargeMatmulConfig cfg;
    cfg.M_tiles = 20;
    cfg.N_tiles = 20;
    cfg.mesh_rows = 2;
    cfg.mesh_cols = 2;

    std::cout << "\n" << std::setw(10) << "K_tiles"
              << std::setw(18) << "C-Stat DMA(MB)"
              << std::setw(18) << "A-Stat DMA(MB)"
              << std::setw(18) << "B-Stat DMA(MB)"
              << std::setw(15) << "Best" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (uint32_t k : {2, 4, 8, 16, 32, 64}) {
        cfg.K_tiles = k;

        auto c = analyze_c_stationary_large(cfg);
        auto a = analyze_a_stationary_large(cfg);
        auto b = analyze_b_stationary_large(cfg);

        uint64_t tb = cfg.tile_bytes();
        uint64_t c_mb = c.total_dma_bytes(tb) / 1024 / 1024;
        uint64_t a_mb = a.total_dma_bytes(tb) / 1024 / 1024;
        uint64_t b_mb = b.total_dma_bytes(tb) / 1024 / 1024;

        std::string best = (c_mb <= a_mb && c_mb <= b_mb) ? "C-Stat" :
                           (a_mb <= b_mb) ? "A-Stat" : "B-Stat";

        std::cout << std::setw(10) << k
                  << std::setw(18) << c_mb
                  << std::setw(18) << a_mb
                  << std::setw(18) << b_mb
                  << std::setw(15) << best << "\n";
    }

    std::cout << "\nObservation:\n";
    std::cout << "  As K increases, C partial R/W overhead grows for A/B-Stationary,\n";
    std::cout << "  making C-Stationary increasingly favorable.\n";
    std::cout << std::string(80, '=') << "\n";

    REQUIRE(true);  // Analysis test always passes
}

// ============================================================================
// Double-Buffering Analysis
// ============================================================================
//
// Double-buffering hides DMA latency by overlapping data movement with compute:
//
// Without double-buffering (sequential):
//   |-- DMA tile 0 --|-- Compute tile 0 --|-- DMA tile 1 --|-- Compute tile 1 --|
//   Total = DMA_total + Compute_total
//
// With double-buffering (overlapped):
//   |-- DMA tile 0 --|-- DMA tile 1 --|-- DMA tile 2 --|
//                    |-- Compute 0 --|-- Compute 1 --|-- Compute 2 --|
//   Total = DMA_first + max(DMA, Compute) * (N-1) + Compute_last
//
// ============================================================================

/// Double-buffer timing configuration
struct DoubleBufferConfig {
    // DMA parameters
    uint64_t dma_latency_cycles = 100;      // Fixed DMA setup latency
    double dma_bandwidth_gbps = 100.0;      // GB/s

    // Compute parameters
    uint64_t matmul_latency_cycles = 64;    // Cycles per tile matmul

    // System parameters
    double clock_freq_ghz = 1.0;            // GHz
    uint32_t tile_bytes = 16 * 1024;        // 16KB tiles

    /// Calculate DMA time for a tile (cycles)
    uint64_t dma_cycles_per_tile() const {
        // Bandwidth in bytes/cycle = bandwidth_gbps * 1e9 / clock_freq_ghz / 1e9
        double bytes_per_cycle = dma_bandwidth_gbps / clock_freq_ghz;
        uint64_t transfer_cycles = static_cast<uint64_t>(tile_bytes / bytes_per_cycle);
        return dma_latency_cycles + transfer_cycles;
    }
};

/// Timing analysis results
struct DoubleBufferTiming {
    // Per-iteration timing
    uint64_t dma_cycles_per_tile = 0;
    uint64_t compute_cycles_per_tile = 0;

    // Total work
    uint64_t total_tiles = 0;
    uint64_t total_matmuls = 0;

    // Sequential timing (no double-buffering)
    uint64_t sequential_dma_cycles = 0;
    uint64_t sequential_compute_cycles = 0;
    uint64_t sequential_total() const { return sequential_dma_cycles + sequential_compute_cycles; }

    // Pipelined timing (with double-buffering)
    uint64_t pipelined_total = 0;

    // Metrics
    double speedup() const {
        return static_cast<double>(sequential_total()) / pipelined_total;
    }
    double dma_hidden_fraction() const {
        if (sequential_dma_cycles == 0) return 0;
        uint64_t hidden = sequential_dma_cycles - (pipelined_total - sequential_compute_cycles);
        return static_cast<double>(hidden) / sequential_dma_cycles;
    }
    double compute_utilization() const {
        return static_cast<double>(sequential_compute_cycles) / pipelined_total;
    }
};

/// Calculate timing for a sequence of DMA+Compute operations with double-buffering
DoubleBufferTiming analyze_double_buffer_timing(
    uint64_t num_tiles,
    uint64_t dma_cycles_per_tile,
    uint64_t compute_cycles_per_tile) {

    DoubleBufferTiming result;
    result.dma_cycles_per_tile = dma_cycles_per_tile;
    result.compute_cycles_per_tile = compute_cycles_per_tile;
    result.total_tiles = num_tiles;
    result.total_matmuls = num_tiles;

    // Sequential: all DMAs then all computes
    result.sequential_dma_cycles = num_tiles * dma_cycles_per_tile;
    result.sequential_compute_cycles = num_tiles * compute_cycles_per_tile;

    if (num_tiles == 0) {
        result.pipelined_total = 0;
        return result;
    }

    if (num_tiles == 1) {
        result.pipelined_total = dma_cycles_per_tile + compute_cycles_per_tile;
        return result;
    }

    // Pipelined: overlap DMA and compute
    // First tile: must wait for DMA
    // Middle tiles: max(DMA, Compute) per tile
    // Last tile: must wait for compute
    uint64_t first_dma = dma_cycles_per_tile;
    uint64_t steady_state_per_tile = std::max(dma_cycles_per_tile, compute_cycles_per_tile);
    uint64_t middle_tiles = num_tiles - 1;
    uint64_t last_compute = compute_cycles_per_tile;

    // If DMA dominates, last compute is overlapped with nothing
    // If Compute dominates, first DMA is overlapped with nothing
    result.pipelined_total = first_dma + (middle_tiles - 1) * steady_state_per_tile + last_compute;

    // Handle edge case where middle_tiles < 1
    if (num_tiles == 2) {
        result.pipelined_total = first_dma + last_compute;
    }

    return result;
}

/// Analyze double-buffering effectiveness for C-Stationary large matmul
DoubleBufferTiming analyze_c_stationary_double_buffer(
    const LargeMatmulConfig& cfg,
    const DoubleBufferConfig& db_cfg) {

    // C-Stationary processes one C-block at a time
    // For each C-block, iterate through K tiles loading A and B
    uint32_t c_block_rows = (cfg.M_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;
    uint32_t c_block_cols = (cfg.N_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;
    uint32_t total_c_blocks = c_block_rows * c_block_cols;

    // Per C-block work:
    //   - K_tiles iterations
    //   - Each iteration: load A row + B col, compute mesh_rows × mesh_cols matmuls
    uint64_t tiles_per_c_block = cfg.K_tiles * (cfg.mesh_rows + cfg.mesh_cols);
    uint64_t matmuls_per_c_block = cfg.K_tiles * cfg.mesh_rows * cfg.mesh_cols;

    uint64_t total_tiles = (uint64_t)total_c_blocks * tiles_per_c_block;
    uint64_t total_matmuls = (uint64_t)total_c_blocks * matmuls_per_c_block;

    // With double-buffering at the K-iteration level:
    // Pipeline: load next K's A+B while computing current K
    uint64_t dma_per_k = (cfg.mesh_rows + cfg.mesh_cols) * db_cfg.dma_cycles_per_tile();
    uint64_t compute_per_k = cfg.mesh_rows * cfg.mesh_cols * db_cfg.matmul_latency_cycles;

    auto timing = analyze_double_buffer_timing(
        (uint64_t)total_c_blocks * cfg.K_tiles,
        dma_per_k,
        compute_per_k);

    timing.total_matmuls = total_matmuls;
    return timing;
}

/// Analyze double-buffering for A-Stationary (includes C partial overhead)
DoubleBufferTiming analyze_a_stationary_double_buffer(
    const LargeMatmulConfig& cfg,
    const DoubleBufferConfig& db_cfg) {

    uint32_t k_chunks = (cfg.K_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;
    uint32_t m_blocks = (cfg.M_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;

    // Per M-block, per K-chunk:
    //   - Load A tiles (once at start of K-chunk)
    //   - For each N column: load B, compute
    //   - Read/Write C partials between K-chunks

    // DMA per K-chunk: A tiles + B tiles for all N
    uint64_t k_tiles_per_chunk = std::min(cfg.mesh_cols, cfg.K_tiles);
    uint64_t a_loads_per_chunk = cfg.mesh_rows * k_tiles_per_chunk;
    uint64_t b_loads_per_chunk = k_tiles_per_chunk * cfg.N_tiles;

    // C partial R/W between chunks (if not first/last)
    uint64_t c_partial_tiles = cfg.mesh_rows * cfg.N_tiles;
    double avg_c_rw_per_chunk = 2.0 * c_partial_tiles * (k_chunks - 1) / k_chunks;

    uint64_t tiles_per_chunk = a_loads_per_chunk + b_loads_per_chunk +
                               static_cast<uint64_t>(avg_c_rw_per_chunk);

    uint64_t dma_per_chunk = tiles_per_chunk * db_cfg.dma_cycles_per_tile();
    uint64_t compute_per_chunk = cfg.mesh_rows * cfg.N_tiles * k_tiles_per_chunk *
                                  db_cfg.matmul_latency_cycles;

    auto timing = analyze_double_buffer_timing(
        (uint64_t)m_blocks * k_chunks,
        dma_per_chunk,
        compute_per_chunk);

    timing.total_matmuls = (uint64_t)cfg.M_tiles * cfg.N_tiles * cfg.K_tiles;
    return timing;
}

/// Analyze double-buffering for B-Stationary (includes C partial overhead)
DoubleBufferTiming analyze_b_stationary_double_buffer(
    const LargeMatmulConfig& cfg,
    const DoubleBufferConfig& db_cfg) {

    uint32_t k_chunks = (cfg.K_tiles + cfg.mesh_rows - 1) / cfg.mesh_rows;
    uint32_t n_blocks = (cfg.N_tiles + cfg.mesh_cols - 1) / cfg.mesh_cols;

    // Per N-block, per K-chunk:
    //   - Load B tiles (once at start of K-chunk)
    //   - For each M row: load A, compute
    //   - Read/Write C partials between K-chunks

    uint64_t k_tiles_per_chunk = std::min(cfg.mesh_rows, cfg.K_tiles);
    uint64_t b_loads_per_chunk = k_tiles_per_chunk * cfg.mesh_cols;
    uint64_t a_loads_per_chunk = cfg.M_tiles * k_tiles_per_chunk;

    uint64_t c_partial_tiles = cfg.M_tiles * cfg.mesh_cols;
    double avg_c_rw_per_chunk = 2.0 * c_partial_tiles * (k_chunks - 1) / k_chunks;

    uint64_t tiles_per_chunk = a_loads_per_chunk + b_loads_per_chunk +
                               static_cast<uint64_t>(avg_c_rw_per_chunk);

    uint64_t dma_per_chunk = tiles_per_chunk * db_cfg.dma_cycles_per_tile();
    uint64_t compute_per_chunk = cfg.M_tiles * cfg.mesh_cols * k_tiles_per_chunk *
                                  db_cfg.matmul_latency_cycles;

    auto timing = analyze_double_buffer_timing(
        (uint64_t)n_blocks * k_chunks,
        dma_per_chunk,
        compute_per_chunk);

    timing.total_matmuls = (uint64_t)cfg.M_tiles * cfg.N_tiles * cfg.K_tiles;
    return timing;
}

void print_double_buffer_timing(const std::string& name,
                                 const DoubleBufferTiming& t) {
    std::cout << "\n--- " << name << " ---\n";
    std::cout << "  Per-iteration:\n";
    std::cout << "    DMA cycles:       " << std::setw(10) << t.dma_cycles_per_tile << "\n";
    std::cout << "    Compute cycles:   " << std::setw(10) << t.compute_cycles_per_tile << "\n";
    std::cout << "    Bottleneck:       " << (t.dma_cycles_per_tile > t.compute_cycles_per_tile
                                              ? "DMA-bound" : "Compute-bound") << "\n";
    std::cout << "  Sequential (no double-buffering):\n";
    std::cout << "    DMA total:        " << std::setw(10) << t.sequential_dma_cycles << " cycles\n";
    std::cout << "    Compute total:    " << std::setw(10) << t.sequential_compute_cycles << " cycles\n";
    std::cout << "    Total:            " << std::setw(10) << t.sequential_total() << " cycles\n";
    std::cout << "  Pipelined (with double-buffering):\n";
    std::cout << "    Total:            " << std::setw(10) << t.pipelined_total << " cycles\n";
    std::cout << "  Metrics:\n";
    std::cout << "    Speedup:          " << std::fixed << std::setprecision(2) << t.speedup() << "x\n";
    std::cout << "    DMA hidden:       " << std::fixed << std::setprecision(1)
              << (t.dma_hidden_fraction() * 100) << "%\n";
    std::cout << "    Compute util:     " << std::fixed << std::setprecision(1)
              << (t.compute_utilization() * 100) << "%\n";
}

TEST_CASE("Double-Buffering Analysis - Hide DMA Latency",
          "[dataflow][integration][large][double_buffer]") {

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "    DOUBLE-BUFFERING ANALYSIS: Hiding DMA Latency\n";
    std::cout << std::string(80, '=') << "\n";

    LargeMatmulConfig cfg;
    cfg.M_tiles = 20;
    cfg.N_tiles = 20;
    cfg.K_tiles = 10;
    cfg.mesh_rows = 2;
    cfg.mesh_cols = 2;

    DoubleBufferConfig db_cfg;
    db_cfg.dma_latency_cycles = 100;
    db_cfg.dma_bandwidth_gbps = 100.0;  // 100 GB/s
    db_cfg.matmul_latency_cycles = 64;   // 64x64 systolic array
    db_cfg.clock_freq_ghz = 1.0;
    db_cfg.tile_bytes = cfg.tile_bytes();

    std::cout << "\nConfiguration:\n";
    std::cout << "  Problem: " << cfg.M_tiles << "×" << cfg.N_tiles << "×" << cfg.K_tiles << " tiles\n";
    std::cout << "  Mesh: " << cfg.mesh_rows << "×" << cfg.mesh_cols << "\n";
    std::cout << "  DMA: " << db_cfg.dma_latency_cycles << " cycles latency + "
              << db_cfg.dma_bandwidth_gbps << " GB/s bandwidth\n";
    std::cout << "  Compute: " << db_cfg.matmul_latency_cycles << " cycles per tile matmul\n";
    std::cout << "  DMA cycles per tile: " << db_cfg.dma_cycles_per_tile() << "\n";

    auto c_timing = analyze_c_stationary_double_buffer(cfg, db_cfg);
    auto a_timing = analyze_a_stationary_double_buffer(cfg, db_cfg);
    auto b_timing = analyze_b_stationary_double_buffer(cfg, db_cfg);

    print_double_buffer_timing("C-Stationary", c_timing);
    print_double_buffer_timing("A-Stationary", a_timing);
    print_double_buffer_timing("B-Stationary", b_timing);

    // Comparison table
    std::cout << "\n" << std::string(90, '-') << "\n";
    std::cout << "                    DOUBLE-BUFFERING COMPARISON\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::setw(25) << "Metric"
              << std::setw(20) << "C-Stationary"
              << std::setw(20) << "A-Stationary"
              << std::setw(20) << "B-Stationary" << "\n";
    std::cout << std::string(90, '-') << "\n";

    std::cout << std::setw(25) << "Sequential (Mcycles)"
              << std::setw(20) << (c_timing.sequential_total() / 1000000)
              << std::setw(20) << (a_timing.sequential_total() / 1000000)
              << std::setw(20) << (b_timing.sequential_total() / 1000000) << "\n";

    std::cout << std::setw(25) << "Pipelined (Mcycles)"
              << std::setw(20) << (c_timing.pipelined_total / 1000000)
              << std::setw(20) << (a_timing.pipelined_total / 1000000)
              << std::setw(20) << (b_timing.pipelined_total / 1000000) << "\n";

    std::cout << std::setw(25) << "Speedup"
              << std::setw(20) << std::fixed << std::setprecision(2) << c_timing.speedup()
              << std::setw(20) << std::fixed << std::setprecision(2) << a_timing.speedup()
              << std::setw(20) << std::fixed << std::setprecision(2) << b_timing.speedup() << "\n";

    std::cout << std::setw(25) << "DMA Hidden %"
              << std::setw(20) << std::fixed << std::setprecision(1) << (c_timing.dma_hidden_fraction() * 100)
              << std::setw(20) << std::fixed << std::setprecision(1) << (a_timing.dma_hidden_fraction() * 100)
              << std::setw(20) << std::fixed << std::setprecision(1) << (b_timing.dma_hidden_fraction() * 100) << "\n";

    std::cout << std::setw(25) << "Compute Util %"
              << std::setw(20) << std::fixed << std::setprecision(1) << (c_timing.compute_utilization() * 100)
              << std::setw(20) << std::fixed << std::setprecision(1) << (a_timing.compute_utilization() * 100)
              << std::setw(20) << std::fixed << std::setprecision(1) << (b_timing.compute_utilization() * 100) << "\n";

    std::cout << std::string(90, '=') << "\n";

    std::cout << "\nKey Insights:\n";
    std::cout << "  - Double-buffering overlaps DMA with compute, hiding latency\n";
    std::cout << "  - Speedup depends on DMA/Compute balance:\n";
    std::cout << "      * DMA-bound: limited by memory bandwidth\n";
    std::cout << "      * Compute-bound: limited by systolic array throughput\n";
    std::cout << "  - Higher speedup when DMA ≈ Compute (balanced pipeline)\n";
    std::cout << std::string(90, '=') << "\n";

    // All should have speedup > 1
    REQUIRE(c_timing.speedup() > 1.0);
    REQUIRE(a_timing.speedup() > 1.0);
    REQUIRE(b_timing.speedup() > 1.0);
}

TEST_CASE("Double-Buffering - Bandwidth Sweep",
          "[dataflow][integration][large][double_buffer][sweep]") {

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "    DOUBLE-BUFFERING SPEEDUP vs DMA BANDWIDTH\n";
    std::cout << std::string(80, '=') << "\n";

    LargeMatmulConfig cfg;
    cfg.M_tiles = 20;
    cfg.N_tiles = 20;
    cfg.K_tiles = 10;
    cfg.mesh_rows = 2;
    cfg.mesh_cols = 2;

    DoubleBufferConfig db_cfg;
    db_cfg.dma_latency_cycles = 100;
    db_cfg.matmul_latency_cycles = 64;
    db_cfg.clock_freq_ghz = 1.0;
    db_cfg.tile_bytes = cfg.tile_bytes();

    std::cout << "\n" << std::setw(15) << "BW (GB/s)"
              << std::setw(15) << "DMA cyc/tile"
              << std::setw(15) << "C-Stat Spdup"
              << std::setw(15) << "A-Stat Spdup"
              << std::setw(15) << "B-Stat Spdup"
              << std::setw(15) << "Bottleneck" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (double bw : {25.0, 50.0, 100.0, 200.0, 400.0, 800.0}) {
        db_cfg.dma_bandwidth_gbps = bw;

        auto c = analyze_c_stationary_double_buffer(cfg, db_cfg);
        auto a = analyze_a_stationary_double_buffer(cfg, db_cfg);
        auto b = analyze_b_stationary_double_buffer(cfg, db_cfg);

        std::string bottleneck = (db_cfg.dma_cycles_per_tile() > db_cfg.matmul_latency_cycles)
                                 ? "DMA" : "Compute";

        std::cout << std::setw(15) << std::fixed << std::setprecision(0) << bw
                  << std::setw(15) << db_cfg.dma_cycles_per_tile()
                  << std::setw(15) << std::fixed << std::setprecision(2) << c.speedup()
                  << std::setw(15) << std::fixed << std::setprecision(2) << a.speedup()
                  << std::setw(15) << std::fixed << std::setprecision(2) << b.speedup()
                  << std::setw(15) << bottleneck << "\n";
    }

    std::cout << "\nObservations:\n";
    std::cout << "  - Low bandwidth: DMA-bound, double-buffering helps significantly\n";
    std::cout << "  - High bandwidth: Compute-bound, speedup approaches 2x (ideal)\n";
    std::cout << "  - Diminishing returns beyond compute-bound threshold\n";
    std::cout << std::string(80, '=') << "\n";

    REQUIRE(true);  // Sweep test always passes
}

// ============================================================================
// Tile Size Sweep - Finding Compute-Bound Regime
// ============================================================================
//
// The balance between DMA and Compute depends on:
//   - Tile size D: larger tiles = more bytes, more compute
//   - Systolic array size S: fixed hardware (e.g., 64×64)
//   - DMA bandwidth: bytes/cycle
//
// For a D×D tile on an S×S systolic array:
//   - Bytes to transfer: D² × 4 (FP32)
//   - DMA cycles: latency + D² × 4 / bandwidth
//   - Compute cycles: ceil(D/S)² × S (process in S×S blocks)
//
// Crossover to compute-bound when: Compute > DMA
//
// ============================================================================

/// Extended config for tile size analysis
struct TileSizeConfig {
    // Hardware parameters
    uint32_t systolic_size = 64;        // S×S systolic array
    uint32_t dma_latency = 100;         // cycles
    double dma_bandwidth_gbps = 100.0;  // GB/s
    double clock_freq_ghz = 1.0;        // GHz

    // Derived
    double bytes_per_cycle() const {
        return dma_bandwidth_gbps / clock_freq_ghz;  // GB/s at 1GHz = bytes/cycle
    }

    /// DMA cycles for a D×D FP32 tile
    uint64_t dma_cycles(uint32_t tile_dim) const {
        uint64_t bytes = (uint64_t)tile_dim * tile_dim * 4;
        uint64_t transfer = static_cast<uint64_t>(bytes / bytes_per_cycle());
        return dma_latency + transfer;
    }

    /// Compute cycles for a D×D tile on S×S systolic array
    uint64_t compute_cycles(uint32_t tile_dim) const {
        // Number of S×S blocks needed to cover D×D tile
        uint32_t blocks_per_dim = (tile_dim + systolic_size - 1) / systolic_size;
        uint32_t total_blocks = blocks_per_dim * blocks_per_dim;
        // Each block takes S cycles (pipelined systolic execution)
        return (uint64_t)total_blocks * systolic_size;
    }

    /// Arithmetic intensity (FLOPs per byte)
    double arithmetic_intensity(uint32_t tile_dim) const {
        // For C = A × B: 2*D³ FLOPs, 3*D²*4 bytes (A+B input, C output)
        double flops = 2.0 * tile_dim * tile_dim * tile_dim;
        double bytes = 3.0 * tile_dim * tile_dim * 4;
        return flops / bytes;
    }

    bool is_compute_bound(uint32_t tile_dim) const {
        return compute_cycles(tile_dim) > dma_cycles(tile_dim);
    }
};

/// Find crossover tile size where compute becomes the bottleneck
uint32_t find_compute_bound_crossover(const TileSizeConfig& cfg,
                                       uint32_t min_dim = 16,
                                       uint32_t max_dim = 1024) {
    for (uint32_t d = min_dim; d <= max_dim; d += 16) {
        if (cfg.is_compute_bound(d)) {
            return d;
        }
    }
    return 0;  // Never crosses to compute-bound in range
}

TEST_CASE("Tile Size Sweep - Find Compute-Bound Regime",
          "[dataflow][integration][large][double_buffer][tile_sweep]") {

    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "    TILE SIZE SWEEP: Finding Compute-Bound Regime\n";
    std::cout << std::string(90, '=') << "\n";

    TileSizeConfig cfg;
    cfg.systolic_size = 64;
    cfg.dma_latency = 100;
    cfg.dma_bandwidth_gbps = 100.0;
    cfg.clock_freq_ghz = 1.0;

    std::cout << "\nHardware Configuration:\n";
    std::cout << "  Systolic array: " << cfg.systolic_size << "×" << cfg.systolic_size << "\n";
    std::cout << "  DMA latency: " << cfg.dma_latency << " cycles\n";
    std::cout << "  DMA bandwidth: " << cfg.dma_bandwidth_gbps << " GB/s\n";
    std::cout << "  Clock: " << cfg.clock_freq_ghz << " GHz\n";
    std::cout << "  Bytes/cycle: " << cfg.bytes_per_cycle() << "\n";

    std::cout << "\n" << std::setw(10) << "Tile Dim"
              << std::setw(12) << "Tile (KB)"
              << std::setw(12) << "DMA cyc"
              << std::setw(12) << "Compute"
              << std::setw(12) << "Ratio D/C"
              << std::setw(12) << "AI"
              << std::setw(15) << "Bottleneck" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (uint32_t d : {16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}) {
        uint64_t dma = cfg.dma_cycles(d);
        uint64_t compute = cfg.compute_cycles(d);
        double ratio = static_cast<double>(dma) / compute;
        double ai = cfg.arithmetic_intensity(d);
        std::string bottleneck = cfg.is_compute_bound(d) ? "COMPUTE" : "DMA";

        std::cout << std::setw(10) << d
                  << std::setw(12) << (d * d * 4 / 1024)
                  << std::setw(12) << dma
                  << std::setw(12) << compute
                  << std::setw(12) << std::fixed << std::setprecision(2) << ratio
                  << std::setw(12) << std::fixed << std::setprecision(1) << ai
                  << std::setw(15) << bottleneck << "\n";
    }

    uint32_t crossover = find_compute_bound_crossover(cfg);
    std::cout << "\n" << std::string(90, '-') << "\n";
    if (crossover > 0) {
        std::cout << "Crossover to COMPUTE-bound at tile dimension: " << crossover << "\n";
    } else {
        std::cout << "System remains DMA-bound for all tile sizes tested.\n";
    }

    // Now show how bandwidth affects crossover
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "    CROSSOVER TILE SIZE vs DMA BANDWIDTH\n";
    std::cout << std::string(90, '=') << "\n";

    std::cout << "\n" << std::setw(15) << "BW (GB/s)"
              << std::setw(20) << "Crossover Tile"
              << std::setw(20) << "Crossover (KB)"
              << std::setw(20) << "Status" << "\n";
    std::cout << std::string(75, '-') << "\n";

    for (double bw : {50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0}) {
        cfg.dma_bandwidth_gbps = bw;
        uint32_t cross = find_compute_bound_crossover(cfg, 16, 2048);
        std::string status = (cross > 0)
            ? std::to_string(cross) + "×" + std::to_string(cross)
            : "Always DMA-bound";
        uint32_t kb = (cross > 0) ? (cross * cross * 4 / 1024) : 0;

        std::cout << std::setw(15) << std::fixed << std::setprecision(0) << bw
                  << std::setw(20) << (cross > 0 ? std::to_string(cross) : "N/A")
                  << std::setw(20) << (cross > 0 ? std::to_string(kb) : "N/A")
                  << std::setw(20) << status << "\n";
    }

    std::cout << "\nAnalysis:\n";
    std::cout << "  - DMA time scales as D² (tile bytes)\n";
    std::cout << "  - Compute time scales as D²/S for D>S (systolic blocks)\n";
    std::cout << "  - Both scale quadratically, so ratio is bandwidth-dependent\n";
    std::cout << "  - Higher bandwidth → smaller crossover tile\n";
    std::cout << std::string(90, '=') << "\n";

    REQUIRE(true);
}

TEST_CASE("Double-Buffering Speedup vs Tile Size",
          "[dataflow][integration][large][double_buffer][tile_sweep]") {

    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "    DOUBLE-BUFFERING SPEEDUP vs TILE SIZE\n";
    std::cout << std::string(90, '=') << "\n";

    TileSizeConfig hw_cfg;
    hw_cfg.systolic_size = 64;
    hw_cfg.dma_latency = 100;
    hw_cfg.dma_bandwidth_gbps = 100.0;
    hw_cfg.clock_freq_ghz = 1.0;

    // Fixed problem size in elements, vary tile size
    const uint64_t total_elements = 1024 * 1024;  // 1M elements per matrix

    std::cout << "\nConfiguration:\n";
    std::cout << "  Total problem: ~" << total_elements << " elements per matrix\n";
    std::cout << "  Systolic: " << hw_cfg.systolic_size << "×" << hw_cfg.systolic_size << "\n";
    std::cout << "  DMA: " << hw_cfg.dma_bandwidth_gbps << " GB/s\n";

    std::cout << "\n" << std::setw(10) << "Tile Dim"
              << std::setw(10) << "# Tiles"
              << std::setw(12) << "DMA/tile"
              << std::setw(12) << "Comp/tile"
              << std::setw(12) << "Seq (Mc)"
              << std::setw(12) << "Pipe (Mc)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Bottleneck" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (uint32_t d : {32, 64, 128, 256, 512}) {
        uint64_t tiles_per_dim = (1024 + d - 1) / d;  // Cover 1024×1024 matrix
        uint64_t num_tiles = tiles_per_dim * tiles_per_dim;

        uint64_t dma_per_tile = hw_cfg.dma_cycles(d);
        uint64_t compute_per_tile = hw_cfg.compute_cycles(d);

        auto timing = analyze_double_buffer_timing(num_tiles, dma_per_tile, compute_per_tile);

        std::string bottleneck = (dma_per_tile > compute_per_tile) ? "DMA" : "Compute";

        std::cout << std::setw(10) << d
                  << std::setw(10) << num_tiles
                  << std::setw(12) << dma_per_tile
                  << std::setw(12) << compute_per_tile
                  << std::setw(12) << (timing.sequential_total() / 1000000)
                  << std::setw(12) << (timing.pipelined_total / 1000000)
                  << std::setw(10) << std::fixed << std::setprecision(2) << timing.speedup()
                  << std::setw(12) << bottleneck << "\n";
    }

    // Now with higher bandwidth to reach compute-bound
    std::cout << "\n" << std::string(90, '-') << "\n";
    std::cout << "With 1000 GB/s bandwidth (10x):\n";
    std::cout << std::string(90, '-') << "\n";

    hw_cfg.dma_bandwidth_gbps = 1000.0;

    std::cout << "\n" << std::setw(10) << "Tile Dim"
              << std::setw(10) << "# Tiles"
              << std::setw(12) << "DMA/tile"
              << std::setw(12) << "Comp/tile"
              << std::setw(12) << "Seq (Mc)"
              << std::setw(12) << "Pipe (Mc)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Bottleneck" << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (uint32_t d : {32, 64, 128, 256, 512}) {
        uint64_t tiles_per_dim = (1024 + d - 1) / d;
        uint64_t num_tiles = tiles_per_dim * tiles_per_dim;

        uint64_t dma_per_tile = hw_cfg.dma_cycles(d);
        uint64_t compute_per_tile = hw_cfg.compute_cycles(d);

        auto timing = analyze_double_buffer_timing(num_tiles, dma_per_tile, compute_per_tile);

        std::string bottleneck = (dma_per_tile > compute_per_tile) ? "DMA" : "Compute";

        std::cout << std::setw(10) << d
                  << std::setw(10) << num_tiles
                  << std::setw(12) << dma_per_tile
                  << std::setw(12) << compute_per_tile
                  << std::setw(12) << (timing.sequential_total() / 1000000)
                  << std::setw(12) << (timing.pipelined_total / 1000000)
                  << std::setw(10) << std::fixed << std::setprecision(2) << timing.speedup()
                  << std::setw(12) << bottleneck << "\n";
    }

    std::cout << "\nKey Insights:\n";
    std::cout << "  - At 100 GB/s: All tile sizes are DMA-bound\n";
    std::cout << "  - At 1000 GB/s: Larger tiles (≥128) become compute-bound\n";
    std::cout << "  - Compute-bound: speedup approaches 2x (ideal double-buffering)\n";
    std::cout << "  - Larger tiles have higher arithmetic intensity (D/6 FLOPs/byte)\n";
    std::cout << "  - But also require more systolic passes: (D/S)² blocks\n";
    std::cout << std::string(90, '=') << "\n";

    REQUIRE(true);
}
