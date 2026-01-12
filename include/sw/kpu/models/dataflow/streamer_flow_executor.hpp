// ============================================================================
// include/sw/kpu/models/dataflow/streamer_flow_executor.hpp
// Streamer Flow Executor - Executes L2 ↔ L1 and Compute Operand Flow Graphs
// ============================================================================
//
// The Streamer executor handles the third level of the memory hierarchy:
// - FEED_WEST: Feed A tiles to systolic array west edge
// - FEED_NORTH: Feed B tiles to systolic array north edge
// - DRAIN: Drain C result from accumulator to L2
// - MATMUL: Trigger matrix multiplication C += A × B
// - APPLY_BIAS: Add bias vector
// - ACTIVATE: Apply activation function
//
// Input events:
// - TILE_READY(A @ L2) - A tile ready in L2 bank
// - TILE_READY(B @ L2) - B tile ready in L2 bank
//
// Output events:
// - TILE_READY(C @ L2) - Result tile ready in L2
// - BUFFER_AVAILABLE(L2[bank]) - L2 bank freed after consumption
// - COMPUTE_DONE - Compute tile finished processing
//
// ============================================================================

#pragma once

#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>

namespace sw::kpu::dataflow {

// ============================================================================
// Streamer Configuration
// ============================================================================

struct StreamerExecutorConfig {
    // This compute tile's position
    uint8_t row = 0;
    uint8_t col = 0;
    uint8_t mesh_cols = 4;

    // Systolic array dimensions
    uint16_t array_rows = 64;
    uint16_t array_cols = 64;

    // Latencies
    uint16_t feed_latency = 5;        // L2 -> L1 feed
    uint16_t drain_latency = 10;      // Accumulator -> L2 drain
    uint16_t matmul_latency = 64;     // Full tile matmul (array_rows cycles)
    uint16_t bias_latency = 4;        // Bias addition
    uint16_t activation_latency = 4;  // Activation function

    // Number of L2 banks accessible
    uint8_t num_l2_banks = 4;

    // Accumulator depth (number of partial sums that can be held)
    uint8_t accumulator_depth = 1;

    uint8_t node_id() const {
        return row * mesh_cols + col;
    }
};

// ============================================================================
// Accumulator State
// ============================================================================

struct AccumulatorState {
    bool has_partial = false;
    uint16_t i = 0;
    uint16_t j = 0;
    uint16_t k_iterations = 0;  // How many partial products accumulated
    uint64_t last_update_cycle = 0;
};

// ============================================================================
// Streamer Flow Executor
// ============================================================================

class StreamerFlowExecutor : public FlowGraphExecutor {
public:
    explicit StreamerFlowExecutor(const OperandFlowGraph& graph,
                                  const StreamerExecutorConfig& config = {})
        : FlowGraphExecutor(graph), config_(config) {
        if (graph.level != ExecutionLevel::STREAMER &&
            graph.level != ExecutionLevel::COMPUTE) {
            // Warning: graph is not a Streamer/Compute-level graph
        }
    }

    // ========== Configuration ==========

    const StreamerExecutorConfig& config() const { return config_; }

    void set_config(const StreamerExecutorConfig& config) {
        config_ = config;
    }

    // ========== Accumulator State ==========

    const AccumulatorState& accumulator() const { return accumulator_; }

    bool accumulator_has_result() const { return accumulator_.has_partial; }

    // ========== Statistics ==========

    uint32_t feeds_completed() const { return feeds_completed_; }
    uint32_t matmuls_completed() const { return matmuls_completed_; }
    uint32_t drains_completed() const { return drains_completed_; }
    uint64_t flops() const { return flops_; }

    // ========== Output Events ==========

    OutputEventCollector& output_events() { return output_events_; }

    /// Get C tile ready events (for BlockMover drain)
    std::vector<OutputEvent> get_result_ready_events() const {
        std::vector<OutputEvent> result;
        for (const auto& e : output_events_.events()) {
            if (e.is_ready && e.operand.type == OperandType::TILE_C) {
                result.push_back(e);
            }
        }
        return result;
    }

    /// Get buffer available events (L2 banks freed)
    std::vector<OutputEvent> get_buffer_available_events() const {
        return output_events_.get_buffer_events();
    }

    // ========== Reset ==========

    void reset() override {
        FlowGraphExecutor::reset();
        accumulator_ = AccumulatorState{};
        feeds_completed_ = 0;
        matmuls_completed_ = 0;
        drains_completed_ = 0;
        flops_ = 0;
        output_events_.clear();
    }

protected:
    void execute_operation(uint32_t node_id, const FlowNode& node) override {
        switch (node.operation) {
            case Operation::FEED_WEST:
                execute_feed(node_id, node, true);  // A operand
                break;
            case Operation::FEED_NORTH:
                execute_feed(node_id, node, false); // B operand
                break;
            case Operation::DRAIN:
                execute_drain(node_id, node);
                break;
            case Operation::MATMUL:
                execute_matmul(node_id, node);
                break;
            case Operation::ACCUMULATE:
                execute_accumulate(node_id, node);
                break;
            case Operation::APPLY_BIAS:
                execute_bias(node_id, node);
                break;
            case Operation::ACTIVATE:
                execute_activate(node_id, node);
                break;
            case Operation::JOIN:
            case Operation::FORK:
            case Operation::NOP:
                break;
            default:
                break;
        }
    }

    uint16_t get_operation_latency(const FlowNode& node) const override {
        switch (node.operation) {
            case Operation::FEED_WEST:
            case Operation::FEED_NORTH:
                return config_.feed_latency;
            case Operation::DRAIN:
                return config_.drain_latency;
            case Operation::MATMUL:
            case Operation::ACCUMULATE:
                return config_.matmul_latency;
            case Operation::APPLY_BIAS:
                return config_.bias_latency;
            case Operation::ACTIVATE:
                return config_.activation_latency;
            default:
                return 1;
        }
    }

    bool can_fire(uint32_t node_id, const FlowNode& node) const override {
        // For MATMUL, check that both A and B tiles are ready
        if (node.operation == Operation::MATMUL ||
            node.operation == Operation::ACCUMULATE) {
            bool has_a = false, has_b = false;
            for (const auto& input : node.inputs) {
                if (input.type == OperandType::TILE_A) has_a = true;
                if (input.type == OperandType::TILE_B) has_b = true;
            }
            if (!has_a || !has_b) return false;
        }

        // For DRAIN, check accumulator has result
        if (node.operation == Operation::DRAIN) {
            // We'll allow it if the graph structure says inputs are ready
            // The actual check is done by the flow graph's input operands
        }

        return true;
    }

private:
    void execute_feed(uint32_t node_id, const FlowNode& node, bool is_a_operand) {
        // Feed tile from L2 to L1 edge registers
        for (const auto& input : node.inputs) {
            if (input.location == Location::L2 && input.type != OperandType::BUFFER_TOKEN) {
                feeds_completed_++;

                // After feed, L2 bank can accept new tile (if configured for single-buffer)
                auto buffer_token = make_buffer_token(Location::L2, input.node_id);
                output_events_.add_buffer_available(buffer_token, current_cycle_ + config_.feed_latency);
            }
        }

        // Produce L1 ready event
        for (const auto& output : node.outputs) {
            if (output.location == Location::L1) {
                output_events_.add_ready(output, current_cycle_ + config_.feed_latency);
            }
        }
    }

    void execute_matmul(uint32_t node_id, const FlowNode& node) {
        // Get tile coordinates from inputs
        uint16_t i = 0, j = 0, k = 0;
        for (const auto& input : node.inputs) {
            if (input.type == OperandType::TILE_A) {
                i = input.coord.i;
                k = input.coord.k;
            } else if (input.type == OperandType::TILE_B) {
                j = input.coord.j;
            }
        }

        // Update accumulator state
        if (!accumulator_.has_partial || accumulator_.i != i || accumulator_.j != j) {
            // Start new accumulation
            accumulator_.has_partial = true;
            accumulator_.i = i;
            accumulator_.j = j;
            accumulator_.k_iterations = 1;
        } else {
            // Continue accumulation
            accumulator_.k_iterations++;
        }
        accumulator_.last_update_cycle = current_cycle_;

        matmuls_completed_++;

        // Calculate FLOPs: 2 * M * N * K for matrix multiply
        // For tile: 2 * array_rows * array_cols * tile_k
        uint64_t tile_flops = 2ULL * config_.array_rows * config_.array_cols * config_.array_rows;
        flops_ += tile_flops;

        // Produce accumulator ready event (partial result)
        auto c_acc = tile_c(i, j, Location::ACCUMULATOR, config_.node_id());
        output_events_.add_ready(c_acc, current_cycle_ + config_.matmul_latency);
    }

    void execute_accumulate(uint32_t node_id, const FlowNode& node) {
        // Same as matmul but for explicit accumulation
        execute_matmul(node_id, node);
    }

    void execute_drain(uint32_t node_id, const FlowNode& node) {
        // Drain result from accumulator to L2
        uint16_t i = accumulator_.i;
        uint16_t j = accumulator_.j;

        drains_completed_++;

        // Clear accumulator
        accumulator_.has_partial = false;
        accumulator_.k_iterations = 0;

        // Produce C tile ready in L2
        for (const auto& output : node.outputs) {
            if (output.location == Location::L2 && output.type == OperandType::TILE_C) {
                output_events_.add_ready(output, current_cycle_ + config_.drain_latency);
            }
        }

        // If no outputs specified, create one
        if (node.outputs.empty()) {
            auto c_l2 = tile_c(i, j, Location::L2, 2);  // Default to bank 2
            output_events_.add_ready(c_l2, current_cycle_ + config_.drain_latency);
        }
    }

    void execute_bias(uint32_t node_id, const FlowNode& node) {
        // Bias addition happens in-place in accumulator
        // Outputs the same C tile location as input
        for (const auto& output : node.outputs) {
            output_events_.add_ready(output, current_cycle_ + config_.bias_latency);
        }
    }

    void execute_activate(uint32_t node_id, const FlowNode& node) {
        // Activation happens in-place
        for (const auto& output : node.outputs) {
            output_events_.add_ready(output, current_cycle_ + config_.activation_latency);
        }
    }

private:
    StreamerExecutorConfig config_;
    AccumulatorState accumulator_;

    // Statistics
    uint32_t feeds_completed_ = 0;
    uint32_t matmuls_completed_ = 0;
    uint32_t drains_completed_ = 0;
    uint64_t flops_ = 0;

    OutputEventCollector output_events_;
};

// ============================================================================
// Streamer Flow Graph Builder
// ============================================================================

/// Helper to build Streamer-level flow graphs for C-stationary matmul
class StreamerFlowGraphBuilder {
public:
    StreamerFlowGraphBuilder() {
        graph_.level = ExecutionLevel::STREAMER;
    }

    /// Set compute tile position
    StreamerFlowGraphBuilder& set_position(uint8_t row, uint8_t col, uint8_t mesh_cols = 4) {
        graph_.mesh_row = row;
        graph_.mesh_col = col;
        graph_.node_id = row * mesh_cols + col;
        return *this;
    }

    /// Set problem dimensions
    StreamerFlowGraphBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k;
        return *this;
    }

    /// Add C-stationary compute for one k iteration
    StreamerFlowGraphBuilder& add_compute_iteration(uint16_t k) {
        uint16_t i = graph_.mesh_row;
        uint16_t j = graph_.mesh_col;

        // Wait for A tile in L2
        auto a_l2 = tile_a(i, k, Location::L2, 0);
        auto wait_a = graph_.add_wait({a_l2}, "wait_A@L2");

        // Wait for B tile in L2
        auto b_l2 = tile_b(k, j, Location::L2, 1);
        auto wait_b = graph_.add_wait({b_l2}, "wait_B@L2");

        // Join A and B ready
        auto join_ab = graph_.add_join({a_l2, b_l2}, "join_AB");
        graph_.add_edge(wait_a, join_ab);
        graph_.add_edge(wait_b, join_ab);

        // Feed A to west edge
        auto a_l1 = tile_a(i, k, Location::L1, 0);
        auto feed_a = graph_.add_fire(Operation::FEED_WEST, {a_l2}, {a_l1}, "feed_A");
        graph_.add_edge(join_ab, feed_a);

        // Feed B to north edge
        auto b_l1 = tile_b(k, j, Location::L1, 0);
        auto feed_b = graph_.add_fire(Operation::FEED_NORTH, {b_l2}, {b_l1}, "feed_B");
        graph_.add_edge(join_ab, feed_b);

        // Execute matmul
        auto c_acc = tile_c(i, j, Location::ACCUMULATOR, graph_.node_id);
        auto matmul = graph_.add_fire(Operation::MATMUL, {a_l1, b_l1}, {c_acc}, "matmul_k" + std::to_string(k));
        graph_.add_edge(feed_a, matmul);
        graph_.add_edge(feed_b, matmul);

        // Store last matmul node for chaining
        last_matmul_node_ = matmul;

        return *this;
    }

    /// Add drain operation after all k iterations
    StreamerFlowGraphBuilder& add_drain() {
        uint16_t i = graph_.mesh_row;
        uint16_t j = graph_.mesh_col;

        // Wait for accumulator (after all matmuls)
        auto c_acc = tile_c(i, j, Location::ACCUMULATOR, graph_.node_id);
        auto wait_acc = graph_.add_wait({c_acc}, "wait_acc");

        if (last_matmul_node_ != UINT32_MAX) {
            graph_.add_edge(last_matmul_node_, wait_acc);
        }

        // Drain to L2
        auto c_l2 = tile_c(i, j, Location::L2, 2);
        auto drain = graph_.add_fire(Operation::DRAIN, {c_acc}, {c_l2}, "drain_C");
        graph_.add_edge(wait_acc, drain);

        return *this;
    }

    /// Add bias and activation after drain
    StreamerFlowGraphBuilder& add_bias_activation() {
        uint16_t i = graph_.mesh_row;
        uint16_t j = graph_.mesh_col;

        // These would be added after drain
        // For now, just a placeholder

        return *this;
    }

    /// Build complete C-stationary compute flow
    StreamerFlowGraphBuilder& build_c_stationary() {
        for (uint16_t k = 0; k < graph_.k_tiles; ++k) {
            add_compute_iteration(k);
        }
        add_drain();
        return *this;
    }

    OperandFlowGraph build() {
        return std::move(graph_);
    }

private:
    OperandFlowGraph graph_;
    uint32_t last_matmul_node_ = UINT32_MAX;
};

// ============================================================================
// A-Stationary Streamer Flow Graph Builder
// ============================================================================

/// Helper to build A-stationary Streamer flow graphs
/// In A-stationary:
///   - A[i,k] is loaded once and reused for all j iterations
///   - B[k,j] streams through for each j
///   - Each node computes partial C[i,j] = A[i,k] * B[k,j]
///   - (Full reduction across k would require additional accumulation)
///
/// This simplified version has each node independently compute and drain
/// its contribution. For a 2x2 mesh with N=2:
///   - Each node does N matmuls (one per output column j)
///   - Total matmuls = mesh_rows * mesh_cols * N = 2*2*2 = 8
class AStationaryStreamerBuilder {
public:
    AStationaryStreamerBuilder() {
        graph_.level = ExecutionLevel::STREAMER;
    }

    /// Set compute tile position - (i,k) indexing for A-stationary
    AStationaryStreamerBuilder& set_position(uint8_t i, uint8_t k, uint8_t mesh_cols = 4) {
        i_ = i;
        k_ = k;
        graph_.mesh_row = i;
        graph_.mesh_col = k;
        graph_.node_id = i * mesh_cols + k;
        mesh_cols_ = mesh_cols;
        return *this;
    }

    /// Set problem dimensions
    AStationaryStreamerBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k_tiles) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k_tiles;
        n_tiles_ = n;
        return *this;
    }

    /// Feed stationary A tile to L1 (done once)
    AStationaryStreamerBuilder& feed_stationary_a() {
        auto a_l2 = tile_a(i_, k_, Location::L2, 0);
        auto a_l1 = tile_a(i_, k_, Location::L1, 0);

        auto wait_a = graph_.add_wait({a_l2}, "wait_A_stationary@L2");
        a_feed_node_ = graph_.add_fire(Operation::FEED_WEST, {a_l2}, {a_l1}, "feed_A_stationary");
        graph_.add_edge(wait_a, a_feed_node_);

        return *this;
    }

    /// Add compute iteration for one j
    /// Computes partial C[i,j] += A[i,k] * B[k,j] and drains
    AStationaryStreamerBuilder& add_compute_for_j(uint16_t j) {
        auto a_l1 = tile_a(i_, k_, Location::L1, 0);
        auto b_l2 = tile_b(k_, j, Location::L2, 1);
        auto b_l1 = tile_b(k_, j, Location::L1, 0);

        // Wait for B tile
        auto wait_b = graph_.add_wait({b_l2}, "wait_B[" + std::to_string(k_) + "," + std::to_string(j) + "]@L2");

        // Feed B to north edge
        auto feed_b = graph_.add_fire(Operation::FEED_NORTH, {b_l2}, {b_l1}, "feed_B");
        graph_.add_edge(wait_b, feed_b);

        // Compute: C = A × B (partial contribution)
        auto c_acc = tile_c(i_, j, Location::ACCUMULATOR, graph_.node_id);
        auto compute = graph_.add_fire(Operation::MATMUL, {a_l1, b_l1}, {c_acc},
                                       "matmul[" + std::to_string(i_) + "," + std::to_string(j) + "]_k" + std::to_string(k_));

        // Connect dependencies
        if (a_feed_node_ != UINT32_MAX) {
            graph_.add_edge(a_feed_node_, compute);
        }
        graph_.add_edge(feed_b, compute);

        // Drain C to L2
        auto c_l2 = tile_c(i_, j, Location::L2, 2);
        auto drain = graph_.add_fire(Operation::DRAIN, {c_acc}, {c_l2},
                                     "drain_C[" + std::to_string(i_) + "," + std::to_string(j) + "]");
        graph_.add_edge(compute, drain);

        return *this;
    }

    /// Build complete A-stationary compute flow
    AStationaryStreamerBuilder& build_a_stationary() {
        feed_stationary_a();
        for (uint16_t j = 0; j < n_tiles_; ++j) {
            add_compute_for_j(j);
        }
        return *this;
    }

    OperandFlowGraph build() {
        return std::move(graph_);
    }

private:
    OperandFlowGraph graph_;
    uint8_t mesh_cols_ = 4;
    uint8_t i_ = 0;
    uint8_t k_ = 0;
    uint16_t n_tiles_ = 4;
    uint32_t a_feed_node_ = UINT32_MAX;
};

// ============================================================================
// B-Stationary Streamer Flow Graph Builder
// ============================================================================

/// Helper to build B-stationary Streamer flow graphs
/// In B-stationary:
///   - B[k,j] is loaded once and reused for all i iterations
///   - A[i,k] streams through for each i
///   - Each node computes partial C[i,j] = A[i,k] * B[k,j]
///
/// For a 2x2 mesh with M=2:
///   - Each node does M matmuls (one per input row i)
///   - Total matmuls = K * N * M = 2*2*2 = 8
class BStationaryStreamerBuilder {
public:
    BStationaryStreamerBuilder() {
        graph_.level = ExecutionLevel::STREAMER;
    }

    /// Set compute tile position - (k,j) indexing for B-stationary
    BStationaryStreamerBuilder& set_position(uint8_t k, uint8_t j, uint8_t mesh_cols = 4) {
        k_ = k;
        j_ = j;
        graph_.mesh_row = k;
        graph_.mesh_col = j;
        graph_.node_id = k * mesh_cols + j;
        mesh_cols_ = mesh_cols;
        return *this;
    }

    /// Set problem dimensions
    BStationaryStreamerBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k_tiles) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k_tiles;
        m_tiles_ = m;
        return *this;
    }

    /// Feed stationary B tile to L1 (done once)
    BStationaryStreamerBuilder& feed_stationary_b() {
        auto b_l2 = tile_b(k_, j_, Location::L2, 1);
        auto b_l1 = tile_b(k_, j_, Location::L1, 0);

        auto wait_b = graph_.add_wait({b_l2}, "wait_B_stationary@L2");
        b_feed_node_ = graph_.add_fire(Operation::FEED_NORTH, {b_l2}, {b_l1}, "feed_B_stationary");
        graph_.add_edge(wait_b, b_feed_node_);

        return *this;
    }

    /// Add compute iteration for one i
    /// Computes partial C[i,j] += A[i,k] * B[k,j] and drains
    BStationaryStreamerBuilder& add_compute_for_i(uint16_t i) {
        auto a_l2 = tile_a(i, k_, Location::L2, 0);
        auto a_l1 = tile_a(i, k_, Location::L1, 0);
        auto b_l1 = tile_b(k_, j_, Location::L1, 0);

        // Wait for A tile
        auto wait_a = graph_.add_wait({a_l2}, "wait_A[" + std::to_string(i) + "," + std::to_string(k_) + "]@L2");

        // Feed A to west edge
        auto feed_a = graph_.add_fire(Operation::FEED_WEST, {a_l2}, {a_l1}, "feed_A");
        graph_.add_edge(wait_a, feed_a);

        // Compute: C = A × B (partial contribution)
        auto c_acc = tile_c(i, j_, Location::ACCUMULATOR, graph_.node_id);
        auto compute = graph_.add_fire(Operation::MATMUL, {a_l1, b_l1}, {c_acc},
                                       "matmul[" + std::to_string(i) + "," + std::to_string(j_) + "]_k" + std::to_string(k_));

        // Connect dependencies
        if (b_feed_node_ != UINT32_MAX) {
            graph_.add_edge(b_feed_node_, compute);
        }
        graph_.add_edge(feed_a, compute);

        // Drain C to L2
        auto c_l2 = tile_c(i, j_, Location::L2, 2);
        auto drain = graph_.add_fire(Operation::DRAIN, {c_acc}, {c_l2},
                                     "drain_C[" + std::to_string(i) + "," + std::to_string(j_) + "]");
        graph_.add_edge(compute, drain);

        return *this;
    }

    /// Build complete B-stationary compute flow
    BStationaryStreamerBuilder& build_b_stationary() {
        feed_stationary_b();
        for (uint16_t i = 0; i < m_tiles_; ++i) {
            add_compute_for_i(i);
        }
        return *this;
    }

    OperandFlowGraph build() {
        return std::move(graph_);
    }

private:
    OperandFlowGraph graph_;
    uint8_t mesh_cols_ = 4;
    uint8_t k_ = 0;
    uint8_t j_ = 0;
    uint16_t m_tiles_ = 4;
    uint32_t b_feed_node_ = UINT32_MAX;
};

} // namespace sw::kpu::dataflow
