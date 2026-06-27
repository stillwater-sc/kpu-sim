// ============================================================================
// include/sw/kpu/models/dataflow/dma_flow_executor.hpp
// DMA Flow Executor - Executes Memory ↔ L3 Operand Flow Graphs
// ============================================================================
//
// The DMA executor handles the first level of the memory hierarchy:
// - LOAD: Transfer tiles from external memory to L3 tiles
// - STORE: Transfer tiles from L3 tiles back to external memory
//
// Input events:
// - BUFFER_AVAILABLE(L3[node]) - L3 tile has space for incoming data
//
// Output events:
// - TILE_READY(T @ L3[node]) - Tile T is now available in L3
//
// ============================================================================

#pragma once

#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>

#include <unordered_map>
#include <cstdint>

namespace sw::kpu::dataflow {

// ============================================================================
// DMA Configuration
// ============================================================================

struct DMAExecutorConfig {
    // Latency for LOAD operation (cycles)
    uint16_t load_latency = 100;

    // Latency for STORE operation (cycles)
    uint16_t store_latency = 100;

    // Bandwidth in bytes per cycle
    uint32_t bandwidth_bytes_per_cycle = 64;

    // Number of L3 tiles this DMA serves
    uint8_t num_l3_tiles = 16;

    // Maximum outstanding transfers
    uint32_t max_outstanding = 8;
};

// ============================================================================
// DMA Transfer Tracking
// ============================================================================

struct DMATransferState {
    Operand operand;
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;
    uint32_t bytes = 0;
    bool is_load = true;  // true = LOAD, false = STORE
};

// ============================================================================
// DMA Flow Executor
// ============================================================================

class DMAFlowExecutor : public FlowGraphExecutor {
public:
    explicit DMAFlowExecutor(const OperandFlowGraph& graph,
                             const DMAExecutorConfig& config = {})
        : FlowGraphExecutor(graph), config_(config) {
        if (graph.level != ExecutionLevel::DMA) {
            // Warning: graph is not a DMA-level graph
        }
    }

    // ========== Configuration ==========

    const DMAExecutorConfig& config() const { return config_; }

    void set_config(const DMAExecutorConfig& config) {
        config_ = config;
    }

    // ========== Transfer Tracking ==========

    /// Get all active transfers
    const std::vector<DMATransferState>& active_transfers() const {
        return active_transfers_;
    }

    /// Get total bytes transferred
    uint64_t bytes_transferred() const { return bytes_transferred_; }

    /// Get number of loads completed
    uint32_t loads_completed() const { return loads_completed_; }

    /// Get number of stores completed
    uint32_t stores_completed() const { return stores_completed_; }

    // ========== Output Event Collection ==========

    /// Get the output event collector
    OutputEventCollector& output_events() { return output_events_; }

    /// Get tile ready events for the next level (BlockMover)
    std::vector<OutputEvent> get_tile_ready_events() const {
        return output_events_.get_ready_events();
    }

    // ========== Reset ==========

    void reset() override {
        FlowGraphExecutor::reset();
        active_transfers_.clear();
        bytes_transferred_ = 0;
        loads_completed_ = 0;
        stores_completed_ = 0;
        output_events_.clear();
    }

protected:
    // ========== Operation Execution ==========

    void execute_operation(uint32_t node_id, const FlowNode& node) override {
        switch (node.operation) {
            case Operation::LOAD:
                execute_load(node_id, node);
                break;
            case Operation::STORE:
                execute_store(node_id, node);
                break;
            case Operation::JOIN:
            case Operation::FORK:
            case Operation::NOP:
                // Control operations - no actual transfer
                break;
            default:
                // Unknown operation for DMA level
                break;
        }
    }

    uint16_t get_operation_latency(const FlowNode& node) const override {
        switch (node.operation) {
            case Operation::LOAD:
                return config_.load_latency;
            case Operation::STORE:
                return config_.store_latency;
            default:
                return 1;
        }
    }

    bool can_fire([[maybe_unused]] uint32_t node_id, const FlowNode& node) const override {
        // Check if we have capacity for more outstanding transfers
        if (node.operation == Operation::LOAD || node.operation == Operation::STORE) {
            if (active_transfers_.size() >= config_.max_outstanding) {
                return false;
            }
        }
        return true;
    }

private:
    void execute_load([[maybe_unused]] uint32_t node_id, const FlowNode& node) {
        // LOAD: Memory -> L3
        // Inputs: BUFFER_AVAILABLE(L3[node])
        // Outputs: TILE_READY(T @ L3[node])

        for (const auto& output : node.outputs) {
            // Track the transfer
            DMATransferState transfer;
            transfer.operand = output;
            transfer.start_cycle = current_cycle_;
            transfer.end_cycle = current_cycle_ + config_.load_latency;
            transfer.bytes = output.size_bytes > 0 ? output.size_bytes : default_tile_size_;
            transfer.is_load = true;
            active_transfers_.push_back(transfer);

            // Update statistics
            bytes_transferred_ += transfer.bytes;
            loads_completed_++;

            // Generate output event for next level
            output_events_.add_ready(output, transfer.end_cycle);
        }
    }

    void execute_store([[maybe_unused]] uint32_t node_id, const FlowNode& node) {
        // STORE: L3 -> Memory
        // Inputs: TILE_READY(T @ L3[node])
        // Outputs: (completion signal, buffer becomes available)

        for (const auto& input : node.inputs) {
            if (input.type != OperandType::BUFFER_TOKEN) {
                DMATransferState transfer;
                transfer.operand = input;
                transfer.start_cycle = current_cycle_;
                transfer.end_cycle = current_cycle_ + config_.store_latency;
                transfer.bytes = input.size_bytes > 0 ? input.size_bytes : default_tile_size_;
                transfer.is_load = false;
                active_transfers_.push_back(transfer);

                bytes_transferred_ += transfer.bytes;
                stores_completed_++;

                // After store completes, buffer becomes available again
                auto buffer_token = make_buffer_token(Location::L3, input.node_id);
                output_events_.add_buffer_available(buffer_token, transfer.end_cycle);
            }
        }
    }

private:
    DMAExecutorConfig config_;

    // Transfer tracking
    std::vector<DMATransferState> active_transfers_;
    uint64_t bytes_transferred_ = 0;
    uint32_t loads_completed_ = 0;
    uint32_t stores_completed_ = 0;

    // Default tile size in bytes (used when operand doesn't specify)
    static constexpr uint32_t default_tile_size_ = 64 * 64 * 4;  // 64x64 float32

    // Output events for inter-level communication
    OutputEventCollector output_events_;
};

// ============================================================================
// DMA Flow Graph Builder
// ============================================================================

/// Helper class to build DMA-level Operand Flow Graphs
class DMAFlowGraphBuilder {
public:
    DMAFlowGraphBuilder() {
        graph_.level = ExecutionLevel::DMA;
    }

    /// Set the DMA node ID
    DMAFlowGraphBuilder& set_node_id(uint8_t id) {
        graph_.node_id = id;
        return *this;
    }

    /// Set problem dimensions
    DMAFlowGraphBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k;
        return *this;
    }

    /// Add a load operation for A tiles (row i, all k)
    DMAFlowGraphBuilder& load_a_row(uint16_t i, uint8_t dest_l3) {
        for (uint16_t k = 0; k < graph_.k_tiles; ++k) {
            add_load_a(i, k, dest_l3);
        }
        return *this;
    }

    /// Add a load operation for B tiles (all k, column j)
    DMAFlowGraphBuilder& load_b_column(uint16_t j, uint8_t dest_l3) {
        for (uint16_t k = 0; k < graph_.k_tiles; ++k) {
            add_load_b(k, j, dest_l3);
        }
        return *this;
    }

    /// Add a single A tile load
    DMAFlowGraphBuilder& add_load_a(uint16_t i, uint16_t k, uint8_t dest_l3) {
        auto buffer_avail = make_buffer_token(Location::L3, dest_l3);
        auto tile_out = tile_a(i, k, Location::L3, dest_l3);

        auto wait_id = graph_.add_wait({buffer_avail}, "wait_L3[" + std::to_string(dest_l3) + "]");
        auto load_id = graph_.add_fire(Operation::LOAD, {buffer_avail}, {tile_out},
                                       "load_A[" + std::to_string(i) + "," + std::to_string(k) + "]");
        graph_.add_edge(wait_id, load_id);

        return *this;
    }

    /// Add a single B tile load
    DMAFlowGraphBuilder& add_load_b(uint16_t k, uint16_t j, uint8_t dest_l3) {
        auto buffer_avail = make_buffer_token(Location::L3, dest_l3);
        auto tile_out = tile_b(k, j, Location::L3, dest_l3);

        auto wait_id = graph_.add_wait({buffer_avail}, "wait_L3[" + std::to_string(dest_l3) + "]");
        auto load_id = graph_.add_fire(Operation::LOAD, {buffer_avail}, {tile_out},
                                       "load_B[" + std::to_string(k) + "," + std::to_string(j) + "]");
        graph_.add_edge(wait_id, load_id);

        return *this;
    }

    /// Add a store operation for C tile
    DMAFlowGraphBuilder& add_store_c(uint16_t i, uint16_t j, uint8_t src_l3) {
        auto tile_in = tile_c(i, j, Location::L3, src_l3);

        auto wait_id = graph_.add_wait({tile_in},
                                       "wait_C[" + std::to_string(i) + "," + std::to_string(j) + "]");
        auto store_id = graph_.add_fire(Operation::STORE, {tile_in}, {},
                                        "store_C[" + std::to_string(i) + "," + std::to_string(j) + "]");
        graph_.add_edge(wait_id, store_id);

        return *this;
    }

    /// Build and return the graph
    OperandFlowGraph build() {
        return std::move(graph_);
    }

private:
    OperandFlowGraph graph_;
};

} // namespace sw::kpu::dataflow
