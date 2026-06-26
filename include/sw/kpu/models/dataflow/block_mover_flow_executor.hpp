// ============================================================================
// include/sw/kpu/models/dataflow/block_mover_flow_executor.hpp
// BlockMover Flow Executor - Executes L3 ↔ L2 and L3 ↔ L3 Operand Flow Graphs
// ============================================================================
//
// The BlockMover executor handles the second level of the memory hierarchy:
// - PUSH_TO_L2: Transfer tiles from L3 to L2 banks
// - PULL_FROM_L2: Transfer tiles from L2 banks back to L3
// - SEND_EAST/WEST/NORTH/SOUTH: Transfer tiles between L3 tiles in the mesh
// - RECEIVE: Accept incoming tiles from neighbors
//
// Input events:
// - TILE_READY(T @ L3[node]) - Tile from DMA or neighbor
// - BUFFER_AVAILABLE(L2[bank]) - L2 bank has space
//
// Output events:
// - TILE_READY(T @ L2[bank]) - Tile ready for Streamer
// - TILE_READY(T @ L3[neighbor]) - Tile forwarded to neighbor
// - BUFFER_AVAILABLE(L3[node]) - L3 buffer freed for DMA
//
// ============================================================================

#pragma once

#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>

#include <array>

namespace sw::kpu::dataflow {

// ============================================================================
// BlockMover Configuration
// ============================================================================

struct BlockMoverExecutorConfig {
    // Mesh dimensions
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;

    // This BlockMover's position
    uint8_t row = 0;
    uint8_t col = 0;

    // Latency for L3 -> L2 push (cycles)
    uint16_t push_latency = 10;

    // Latency for L2 -> L3 pull (cycles)
    uint16_t pull_latency = 10;

    // Latency for L3 -> L3 mesh transfer (cycles per hop)
    uint16_t mesh_latency_per_hop = 5;

    // Number of L2 banks
    uint8_t num_l2_banks = 4;

    // L2 bank buffer depth (tiles)
    uint8_t l2_buffer_depth = 2;

    // Compute node ID from position
    uint8_t node_id() const {
        return row * mesh_cols + col;
    }

    // Get neighbor node IDs
    std::optional<uint8_t> east_neighbor() const {
        if (col + 1 < mesh_cols) return row * mesh_cols + (col + 1);
        return std::nullopt;
    }

    std::optional<uint8_t> west_neighbor() const {
        if (col > 0) return row * mesh_cols + (col - 1);
        return std::nullopt;
    }

    std::optional<uint8_t> north_neighbor() const {
        if (row > 0) return (row - 1) * mesh_cols + col;
        return std::nullopt;
    }

    std::optional<uint8_t> south_neighbor() const {
        if (row + 1 < mesh_rows) return (row + 1) * mesh_cols + col;
        return std::nullopt;
    }
};

// ============================================================================
// L2 Bank State
// ============================================================================

struct L2BankState {
    uint8_t bank_id = 0;
    uint8_t tiles_buffered = 0;
    uint8_t max_tiles = 2;

    bool has_space() const { return tiles_buffered < max_tiles; }
    bool is_empty() const { return tiles_buffered == 0; }

    void add_tile() { if (tiles_buffered < max_tiles) tiles_buffered++; }
    void remove_tile() { if (tiles_buffered > 0) tiles_buffered--; }
};

// ============================================================================
// BlockMover Flow Executor
// ============================================================================

class BlockMoverFlowExecutor : public FlowGraphExecutor {
public:
    explicit BlockMoverFlowExecutor(const OperandFlowGraph& graph,
                                    const BlockMoverExecutorConfig& config = {})
        : FlowGraphExecutor(graph), config_(config) {
        if (graph.level != ExecutionLevel::BLOCK_MOVER) {
            // Warning: graph is not a BlockMover-level graph
        }
        init_l2_banks();
    }

    // ========== Configuration ==========

    const BlockMoverExecutorConfig& config() const { return config_; }

    void set_config(const BlockMoverExecutorConfig& config) {
        config_ = config;
        init_l2_banks();
    }

    // ========== L2 Bank State ==========

    const L2BankState& get_l2_bank(uint8_t bank_id) const {
        if (bank_id < l2_banks_.size()) {
            return l2_banks_[bank_id];
        }
        static L2BankState empty;
        return empty;
    }

    bool l2_bank_available(uint8_t bank_id) const {
        if (bank_id < l2_banks_.size()) {
            return l2_banks_[bank_id].has_space();
        }
        return false;
    }

    // ========== Transfer Statistics ==========

    uint32_t pushes_completed() const { return pushes_completed_; }
    uint32_t pulls_completed() const { return pulls_completed_; }
    uint32_t mesh_sends_completed() const { return mesh_sends_completed_; }
    uint64_t bytes_to_l2() const { return bytes_to_l2_; }
    uint64_t bytes_from_l2() const { return bytes_from_l2_; }
    uint64_t bytes_mesh() const { return bytes_mesh_; }

    // ========== Output Events ==========

    OutputEventCollector& output_events() { return output_events_; }

    /// Get tile ready events for Streamer level (tiles in L2)
    std::vector<OutputEvent> get_l2_ready_events() const {
        std::vector<OutputEvent> result;
        for (const auto& e : output_events_.events()) {
            if (e.is_ready && e.operand.location == Location::L2) {
                result.push_back(e);
            }
        }
        return result;
    }

    /// Get tile ready events for neighboring BlockMovers (mesh transfers)
    std::vector<OutputEvent> get_mesh_ready_events() const {
        std::vector<OutputEvent> result;
        for (const auto& e : output_events_.events()) {
            if (e.is_ready && e.operand.location == Location::L3 &&
                e.operand.node_id != config_.node_id()) {
                result.push_back(e);
            }
        }
        return result;
    }

    // ========== Reset ==========

    void reset() override {
        FlowGraphExecutor::reset();
        init_l2_banks();
        pushes_completed_ = 0;
        pulls_completed_ = 0;
        mesh_sends_completed_ = 0;
        bytes_to_l2_ = 0;
        bytes_from_l2_ = 0;
        bytes_mesh_ = 0;
        output_events_.clear();
    }

protected:
    void execute_operation(uint32_t node_id, const FlowNode& node) override {
        switch (node.operation) {
            case Operation::PUSH_TO_L2:
                execute_push_to_l2(node_id, node);
                break;
            case Operation::PULL_FROM_L2:
                execute_pull_from_l2(node_id, node);
                break;
            case Operation::SEND_EAST:
                execute_send(node_id, node, config_.east_neighbor());
                break;
            case Operation::SEND_WEST:
                execute_send(node_id, node, config_.west_neighbor());
                break;
            case Operation::SEND_NORTH:
                execute_send(node_id, node, config_.north_neighbor());
                break;
            case Operation::SEND_SOUTH:
                execute_send(node_id, node, config_.south_neighbor());
                break;
            case Operation::SEND_TO:
                // Use the dest node from the first output operand
                if (!node.outputs.empty()) {
                    execute_send(node_id, node, node.outputs[0].node_id);
                }
                break;
            case Operation::RECEIVE:
                execute_receive(node_id, node);
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
            case Operation::PUSH_TO_L2:
                return config_.push_latency;
            case Operation::PULL_FROM_L2:
                return config_.pull_latency;
            case Operation::SEND_EAST:
            case Operation::SEND_WEST:
            case Operation::SEND_NORTH:
            case Operation::SEND_SOUTH:
            case Operation::SEND_TO:
                return config_.mesh_latency_per_hop;
            case Operation::RECEIVE:
                return 1;
            default:
                return 1;
        }
    }

    bool can_fire(uint32_t node_id, const FlowNode& node) const override {
        // For PUSH_TO_L2, check L2 bank availability
        if (node.operation == Operation::PUSH_TO_L2) {
            for (const auto& output : node.outputs) {
                if (output.location == Location::L2) {
                    if (!l2_bank_available(output.node_id)) {
                        return false;
                    }
                }
            }
        }

        // For SEND operations, check neighbor exists
        switch (node.operation) {
            case Operation::SEND_EAST:
                return config_.east_neighbor().has_value();
            case Operation::SEND_WEST:
                return config_.west_neighbor().has_value();
            case Operation::SEND_NORTH:
                return config_.north_neighbor().has_value();
            case Operation::SEND_SOUTH:
                return config_.south_neighbor().has_value();
            default:
                break;
        }

        return true;
    }

private:
    void init_l2_banks() {
        l2_banks_.resize(config_.num_l2_banks);
        for (uint8_t i = 0; i < config_.num_l2_banks; ++i) {
            l2_banks_[i].bank_id = i;
            l2_banks_[i].tiles_buffered = 0;
            l2_banks_[i].max_tiles = config_.l2_buffer_depth;
        }
    }

    void execute_push_to_l2(uint32_t node_id, const FlowNode& node) {
        for (const auto& output : node.outputs) {
            if (output.location == Location::L2) {
                uint8_t bank_id = output.node_id;
                if (bank_id < l2_banks_.size()) {
                    l2_banks_[bank_id].add_tile();
                }

                uint32_t bytes = output.size_bytes > 0 ? output.size_bytes : default_tile_size_;
                bytes_to_l2_ += bytes;
                pushes_completed_++;

                // Generate TILE_READY for Streamer
                output_events_.add_ready(output, current_cycle_ + config_.push_latency);
            }
        }

        // The L3 buffer is now free - signal back to DMA
        for (const auto& input : node.inputs) {
            if (input.location == Location::L3 && input.type != OperandType::BUFFER_TOKEN) {
                auto buffer_token = make_buffer_token(Location::L3, config_.node_id());
                output_events_.add_buffer_available(buffer_token, current_cycle_ + config_.push_latency);
            }
        }
    }

    void execute_pull_from_l2(uint32_t node_id, const FlowNode& node) {
        for (const auto& input : node.inputs) {
            if (input.location == Location::L2) {
                uint8_t bank_id = input.node_id;
                if (bank_id < l2_banks_.size()) {
                    l2_banks_[bank_id].remove_tile();
                }

                uint32_t bytes = input.size_bytes > 0 ? input.size_bytes : default_tile_size_;
                bytes_from_l2_ += bytes;
                pulls_completed_++;

                // Generate BUFFER_AVAILABLE for Streamer
                auto buffer_token = make_buffer_token(Location::L2, bank_id);
                output_events_.add_buffer_available(buffer_token, current_cycle_ + config_.pull_latency);
            }
        }

        // Tile is now in L3 (for potential STORE back to memory)
        for (const auto& output : node.outputs) {
            if (output.location == Location::L3) {
                output_events_.add_ready(output, current_cycle_ + config_.pull_latency);
            }
        }
    }

    void execute_send(uint32_t node_id, const FlowNode& node, std::optional<uint8_t> dest_node) {
        if (!dest_node) return;

        for (const auto& input : node.inputs) {
            if (input.type != OperandType::BUFFER_TOKEN) {
                // Create output operand at destination
                Operand dest_operand = input;
                dest_operand.location = Location::L3;
                dest_operand.node_id = *dest_node;

                uint32_t bytes = input.size_bytes > 0 ? input.size_bytes : default_tile_size_;
                bytes_mesh_ += bytes;
                mesh_sends_completed_++;

                // Generate TILE_READY at destination
                output_events_.add_ready(dest_operand, current_cycle_ + config_.mesh_latency_per_hop);
            }
        }

        // Also use outputs if specified
        for (const auto& output : node.outputs) {
            if (output.location == Location::L3 && output.node_id == *dest_node) {
                output_events_.add_ready(output, current_cycle_ + config_.mesh_latency_per_hop);
            }
        }
    }

    void execute_receive(uint32_t node_id, const FlowNode& node) {
        // RECEIVE is mostly a synchronization point - the tile is already
        // marked ready by the sender's SEND operation
        // We just need to make the tile available locally
        for (const auto& output : node.outputs) {
            output_events_.add_ready(output, current_cycle_ + 1);
        }
    }

private:
    BlockMoverExecutorConfig config_;
    std::vector<L2BankState> l2_banks_;

    // Statistics
    uint32_t pushes_completed_ = 0;
    uint32_t pulls_completed_ = 0;
    uint32_t mesh_sends_completed_ = 0;
    uint64_t bytes_to_l2_ = 0;
    uint64_t bytes_from_l2_ = 0;
    uint64_t bytes_mesh_ = 0;

    static constexpr uint32_t default_tile_size_ = 64 * 64 * 4;

    OutputEventCollector output_events_;
};

// ============================================================================
// BlockMover Flow Graph Builder
// ============================================================================

/// Helper to build C-stationary BlockMover flow graphs
class BlockMoverFlowGraphBuilder {
public:
    BlockMoverFlowGraphBuilder() {
        graph_.level = ExecutionLevel::BLOCK_MOVER;
    }

    /// Set mesh position
    BlockMoverFlowGraphBuilder& set_position(uint8_t row, uint8_t col, uint8_t mesh_cols = 4) {
        graph_.mesh_row = row;
        graph_.mesh_col = col;
        graph_.node_id = row * mesh_cols + col;
        mesh_cols_ = mesh_cols;
        return *this;
    }

    /// Set problem dimensions
    BlockMoverFlowGraphBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k;
        return *this;
    }

    /// Build C-stationary flow for one k iteration
    /// A flows West->East, B flows North->South
    BlockMoverFlowGraphBuilder& add_c_stationary_iteration(uint16_t k) {
        uint16_t i = graph_.mesh_row;
        uint16_t j = graph_.mesh_col;
        uint8_t node_id = graph_.node_id;

        // === A tile path ===
        // Wait for A tile (from West or DMA)
        auto a_l3 = tile_a(i, k, Location::L3, node_id);
        auto wait_a = graph_.add_wait({a_l3}, "wait_A[" + std::to_string(i) + "," + std::to_string(k) + "]");

        // Wait for L2 bank 0 available
        auto l2_avail_0 = make_buffer_token(Location::L2, 0);
        auto wait_l2_a = graph_.add_wait({l2_avail_0}, "wait_L2[0]");

        // Join
        auto join_a = graph_.add_join({a_l3, l2_avail_0}, "join_A");
        graph_.add_edge(wait_a, join_a);
        graph_.add_edge(wait_l2_a, join_a);

        // Push A to L2
        auto a_l2 = tile_a(i, k, Location::L2, 0);
        auto push_a = graph_.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2}, "push_A");
        graph_.add_edge(join_a, push_a);

        // Forward A east (if not rightmost column)
        if (j + 1 < mesh_cols_) {
            auto a_east = tile_a(i, k, Location::L3, node_id + 1);
            auto send_a = graph_.add_fire(Operation::SEND_EAST, {a_l3}, {a_east}, "send_A_east");
            graph_.add_edge(push_a, send_a);
        }

        // === B tile path ===
        // Wait for B tile (from North or DMA)
        auto b_l3 = tile_b(k, j, Location::L3, node_id);
        auto wait_b = graph_.add_wait({b_l3}, "wait_B[" + std::to_string(k) + "," + std::to_string(j) + "]");

        // Wait for L2 bank 1 available
        auto l2_avail_1 = make_buffer_token(Location::L2, 1);
        auto wait_l2_b = graph_.add_wait({l2_avail_1}, "wait_L2[1]");

        // Join
        auto join_b = graph_.add_join({b_l3, l2_avail_1}, "join_B");
        graph_.add_edge(wait_b, join_b);
        graph_.add_edge(wait_l2_b, join_b);

        // Push B to L2
        auto b_l2 = tile_b(k, j, Location::L2, 1);
        auto push_b = graph_.add_fire(Operation::PUSH_TO_L2, {b_l3}, {b_l2}, "push_B");
        graph_.add_edge(join_b, push_b);

        // Forward B south (if not bottom row).
        // graph_.m_tiles is uint16_t; using uint8_t here would silently
        // wrap to 0 for any mesh >= 256 rows and break the bounds check,
        // so match the source type. Also fixes MSVC C4244 narrowing
        // warning that propagated to every TU including this header.
        uint16_t mesh_rows = graph_.m_tiles;  // assumes square mesh
        if (i + 1 < mesh_rows) {
            auto b_south = tile_b(k, j, Location::L3, node_id + mesh_cols_);
            auto send_b = graph_.add_fire(Operation::SEND_SOUTH, {b_l3}, {b_south}, "send_B_south");
            graph_.add_edge(push_b, send_b);
        }

        return *this;
    }

    /// Build complete C-stationary flow for all k iterations
    BlockMoverFlowGraphBuilder& build_c_stationary() {
        for (uint16_t k = 0; k < graph_.k_tiles; ++k) {
            add_c_stationary_iteration(k);
        }
        return *this;
    }

    /// Add drain operation for C tile after compute
    BlockMoverFlowGraphBuilder& add_drain_c() {
        uint16_t i = graph_.mesh_row;
        uint16_t j = graph_.mesh_col;
        uint8_t node_id = graph_.node_id;

        // Wait for C in L2
        auto c_l2 = tile_c(i, j, Location::L2, 2);
        auto wait_c = graph_.add_wait({c_l2}, "wait_C_L2");

        // Pull C from L2 to L3
        auto c_l3 = tile_c(i, j, Location::L3, node_id);
        auto pull_c = graph_.add_fire(Operation::PULL_FROM_L2, {c_l2}, {c_l3}, "pull_C");
        graph_.add_edge(wait_c, pull_c);

        return *this;
    }

    OperandFlowGraph build() {
        return std::move(graph_);
    }

private:
    OperandFlowGraph graph_;
    uint8_t mesh_cols_ = 4;
};

// ============================================================================
// A-Stationary BlockMover Flow Graph Builder
// ============================================================================

/// Helper to build A-stationary BlockMover flow graphs
/// In A-stationary (mesh indexed by i,k):
///   - A[i,k] stays at node (i,k) - loaded once, reused for all j
///   - B[k,j] flows North→South along column k
///     * DMA loads B[k,j] to north edge node (i=0, k)
///     * B[k,j] flows down to all nodes in column k
///   - C[i,j] partial sums reduce across row i (West→East)
///
/// Data flow for B[k,j]:
///   - All nodes (0,k), (1,k), ... (M-1,k) need B[k,j]
///   - B enters at (0,k) from DMA
///   - Flows: (0,k) → (1,k) → ... → (M-1,k)
class AStationaryBlockMoverBuilder {
public:
    AStationaryBlockMoverBuilder() {
        graph_.level = ExecutionLevel::BLOCK_MOVER;
    }

    /// Set mesh position - in A-stationary, position is (i,k) not (i,j)
    AStationaryBlockMoverBuilder& set_position(uint8_t i, uint8_t k, uint8_t mesh_cols = 4) {
        i_ = i;
        k_ = k;
        graph_.mesh_row = i;
        graph_.mesh_col = k;
        graph_.node_id = i * mesh_cols + k;
        mesh_cols_ = mesh_cols;
        return *this;
    }

    /// Set problem dimensions
    AStationaryBlockMoverBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k_tiles) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k_tiles;
        n_tiles_ = n;
        return *this;
    }

    /// Load A tile once (stationary)
    AStationaryBlockMoverBuilder& load_stationary_a() {
        uint8_t node_id = graph_.node_id;

        // A[i,k] arrives from DMA
        auto a_l3 = tile_a(i_, k_, Location::L3, node_id);
        auto l2_avail = make_buffer_token(Location::L2, 0);
        auto a_l2 = tile_a(i_, k_, Location::L2, 0);

        auto wait_a = graph_.add_wait({a_l3}, "wait_A_stationary");
        auto wait_l2 = graph_.add_wait({l2_avail}, "wait_L2_A");
        auto join = graph_.add_join({a_l3, l2_avail}, "join_A");
        auto push = graph_.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2}, "push_A_stationary");

        graph_.add_edge(wait_a, join);
        graph_.add_edge(wait_l2, join);
        graph_.add_edge(join, push);

        return *this;
    }

    /// Add flow for one j iteration (B flows N→S, C partials reduce W→E)
    AStationaryBlockMoverBuilder& add_a_stationary_iteration(uint16_t j) {
        uint8_t node_id = graph_.node_id;

        // === B tile path ===
        // B[k,j] arrives from North (or DMA if i_==0)
        // Then flows South to next row
        auto b_l3 = tile_b(k_, j, Location::L3, node_id);
        auto l2_avail_b = make_buffer_token(Location::L2, 1);
        auto b_l2 = tile_b(k_, j, Location::L2, 1);

        auto wait_b = graph_.add_wait({b_l3}, "wait_B[" + std::to_string(k_) + "," + std::to_string(j) + "]");
        auto wait_l2_b = graph_.add_wait({l2_avail_b}, "wait_L2_B");
        auto join_b = graph_.add_join({b_l3, l2_avail_b}, "join_B");
        auto push_b = graph_.add_fire(Operation::PUSH_TO_L2, {b_l3}, {b_l2}, "push_B");

        graph_.add_edge(wait_b, join_b);
        graph_.add_edge(wait_l2_b, join_b);
        graph_.add_edge(join_b, push_b);

        // Forward B south (if not bottom row)
        // B[k,j] flows N→S along column k
        if (i_ + 1 < graph_.m_tiles) {
            auto b_south = tile_b(k_, j, Location::L3, node_id + mesh_cols_);
            auto send_b = graph_.add_fire(Operation::SEND_SOUTH, {b_l3}, {b_south}, "send_B_south");
            graph_.add_edge(push_b, send_b);
        }

        // === C partial sum path ===
        // After compute, C partials reduce West→East across row i
        // (not implemented in this simplified version)

        return *this;
    }

    /// Build complete A-stationary flow for all j iterations
    AStationaryBlockMoverBuilder& build_a_stationary() {
        load_stationary_a();
        for (uint16_t j = 0; j < n_tiles_; ++j) {
            add_a_stationary_iteration(j);
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
};

// ============================================================================
// B-Stationary BlockMover Flow Graph Builder
// ============================================================================

/// Helper to build B-stationary BlockMover flow graphs
/// In B-stationary (mesh indexed by k,j):
///   - B[k,j] stays at node (k,j) - loaded once, reused for all i
///   - A[i,k] flows West→East along row k
///     * DMA loads A[i,k] to west edge node (k, j=0)
///     * A[i,k] flows right to all nodes in row k
///   - C[i,j] partial sums reduce along column j (North→South)
///
/// Data flow for A[i,k]:
///   - All nodes (k,0), (k,1), ... (k,N-1) need A[i,k]
///   - A enters at (k,0) from DMA
///   - Flows: (k,0) → (k,1) → ... → (k,N-1)
class BStationaryBlockMoverBuilder {
public:
    BStationaryBlockMoverBuilder() {
        graph_.level = ExecutionLevel::BLOCK_MOVER;
    }

    /// Set mesh position - (k,j) indexing for B-stationary
    BStationaryBlockMoverBuilder& set_position(uint8_t k, uint8_t j, uint8_t mesh_cols = 4) {
        k_ = k;
        j_ = j;
        graph_.mesh_row = k;
        graph_.mesh_col = j;
        graph_.node_id = k * mesh_cols + j;
        mesh_cols_ = mesh_cols;
        return *this;
    }

    /// Set problem dimensions
    BStationaryBlockMoverBuilder& set_dimensions(uint16_t m, uint16_t n, uint16_t k_tiles) {
        graph_.m_tiles = m;
        graph_.n_tiles = n;
        graph_.k_tiles = k_tiles;
        m_tiles_ = m;
        return *this;
    }

    /// Load B tile once (stationary)
    BStationaryBlockMoverBuilder& load_stationary_b() {
        uint8_t node_id = graph_.node_id;

        // B[k,j] arrives from DMA
        auto b_l3 = tile_b(k_, j_, Location::L3, node_id);
        auto l2_avail = make_buffer_token(Location::L2, 1);
        auto b_l2 = tile_b(k_, j_, Location::L2, 1);

        auto wait_b = graph_.add_wait({b_l3}, "wait_B_stationary");
        auto wait_l2 = graph_.add_wait({l2_avail}, "wait_L2_B");
        auto join = graph_.add_join({b_l3, l2_avail}, "join_B");
        auto push = graph_.add_fire(Operation::PUSH_TO_L2, {b_l3}, {b_l2}, "push_B_stationary");

        graph_.add_edge(wait_b, join);
        graph_.add_edge(wait_l2, join);
        graph_.add_edge(join, push);

        return *this;
    }

    /// Add flow for one i iteration (A flows W→E, C partials reduce N→S)
    BStationaryBlockMoverBuilder& add_b_stationary_iteration(uint16_t i) {
        uint8_t node_id = graph_.node_id;

        // === A tile path ===
        // A[i,k] arrives from West (or DMA if j_==0)
        // Then flows East to next column
        auto a_l3 = tile_a(i, k_, Location::L3, node_id);
        auto l2_avail_a = make_buffer_token(Location::L2, 0);
        auto a_l2 = tile_a(i, k_, Location::L2, 0);

        auto wait_a = graph_.add_wait({a_l3}, "wait_A[" + std::to_string(i) + "," + std::to_string(k_) + "]");
        auto wait_l2_a = graph_.add_wait({l2_avail_a}, "wait_L2_A");
        auto join_a = graph_.add_join({a_l3, l2_avail_a}, "join_A");
        auto push_a = graph_.add_fire(Operation::PUSH_TO_L2, {a_l3}, {a_l2}, "push_A");

        graph_.add_edge(wait_a, join_a);
        graph_.add_edge(wait_l2_a, join_a);
        graph_.add_edge(join_a, push_a);

        // Forward A east (if not rightmost column)
        // A[i,k] flows W→E along row k
        if (j_ + 1 < mesh_cols_) {
            auto a_east = tile_a(i, k_, Location::L3, node_id + 1);
            auto send_a = graph_.add_fire(Operation::SEND_EAST, {a_l3}, {a_east}, "send_A_east");
            graph_.add_edge(push_a, send_a);
        }

        // === C partial sum path ===
        // After compute, C partials reduce North→South along column j
        // (not implemented in this simplified version)

        return *this;
    }

    /// Build complete B-stationary flow for all i iterations
    BStationaryBlockMoverBuilder& build_b_stationary() {
        load_stationary_b();
        for (uint16_t i = 0; i < m_tiles_; ++i) {
            add_b_stationary_iteration(i);
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
};

} // namespace sw::kpu::dataflow
