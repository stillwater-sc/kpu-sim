// ============================================================================
// include/sw/kpu/dataflow/tile_dataflow_graph.hpp
// Tile Data Flow Graph for representing tensor tile movement and computation
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>

namespace sw::kpu::dataflow {

// ============================================================================
// Data Flow Node Types
// ============================================================================

/// Types of operations in the data flow graph
enum class DFNodeType : uint8_t {
    // Data movement
    DMA_LOAD,           // External Memory → L3
    DMA_STORE,          // L3 → External Memory
    L3_TRANSFER,        // L3 → L3 (via BlockMover/Interconnect)
    L3_TO_L2,           // L3 → L2 (BlockMover PUSH_TO_L2)
    L2_TO_L3,           // L2 → L3 (BlockMover PULL_FROM_L2 / drain)

    // Computation
    MATMUL,             // Matrix multiplication on systolic array
    ELEMENTWISE,        // Element-wise operation (add, mul, etc.)
    ACTIVATION,         // Activation function (via VectorEngine)
    REDUCTION,          // Reduction operation

    // Synchronization
    BARRIER,            // Wait for all predecessors
    TRIGGER,            // Emit a trigger signal
};

inline const char* to_string(DFNodeType type) {
    switch (type) {
        case DFNodeType::DMA_LOAD: return "DMA_LOAD";
        case DFNodeType::DMA_STORE: return "DMA_STORE";
        case DFNodeType::L3_TRANSFER: return "L3_TRANSFER";
        case DFNodeType::L3_TO_L2: return "L3_TO_L2";
        case DFNodeType::L2_TO_L3: return "L2_TO_L3";
        case DFNodeType::MATMUL: return "MATMUL";
        case DFNodeType::ELEMENTWISE: return "ELEMENTWISE";
        case DFNodeType::ACTIVATION: return "ACTIVATION";
        case DFNodeType::REDUCTION: return "REDUCTION";
        case DFNodeType::BARRIER: return "BARRIER";
        case DFNodeType::TRIGGER: return "TRIGGER";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Data Flow Edge Types
// ============================================================================

/// Type of dependency between nodes
enum class DFEdgeType : uint8_t {
    DATA,               // Data dependency (consumer needs producer's output)
    CONTROL,            // Control dependency (ordering only, no data)
    ANTI,               // Anti-dependency (write-after-read)
    OUTPUT,             // Output dependency (write-after-write)
};

// ============================================================================
// Data Flow Node
// ============================================================================

/// A node in the tile data flow graph
struct DFNode {
    size_t id = 0;              // Unique node ID
    DFNodeType type = DFNodeType::BARRIER;

    // Tile information
    TileDescriptor tile;

    // Location (where this operation executes/resides)
    uint8_t l3_id = 0;          // L3 tile ID
    uint8_t l2_bank_id = 0;     // L2 bank ID (for L3↔L2 transfers)

    // For transfers: source and destination
    uint8_t src_l3_id = 0;
    uint8_t dst_l3_id = 0;

    // For compute operations: dimensions
    uint32_t M = 0, N = 0, K = 0;

    // Scheduling information (filled by scheduler)
    int64_t earliest_start = -1;    // Earliest possible start cycle
    int64_t scheduled_start = -1;   // Actual scheduled start cycle
    int64_t duration = 0;           // Estimated duration in cycles

    // Dependencies
    std::vector<size_t> predecessors;   // Nodes that must complete before this
    std::vector<size_t> successors;     // Nodes that depend on this

    // Metadata
    std::string name;
    uint32_t priority = 0;          // For scheduling (higher = more urgent)

    bool is_scheduled() const { return scheduled_start >= 0; }

    std::string to_string() const {
        std::string s = "[" + std::to_string(id) + "] ";
        s += sw::kpu::dataflow::to_string(type);
        s += " " + tile.to_string();
        s += " @L3[" + std::to_string(l3_id) + "]";
        if (type == DFNodeType::L3_TRANSFER) {
            s += " " + std::to_string(src_l3_id) + "→" + std::to_string(dst_l3_id);
        }
        return s;
    }
};

// ============================================================================
// Data Flow Edge
// ============================================================================

/// An edge (dependency) in the data flow graph
struct DFEdge {
    size_t from_node = 0;       // Producer node
    size_t to_node = 0;         // Consumer node
    DFEdgeType type = DFEdgeType::DATA;

    // For data edges: which tile is being passed
    TileDescriptor tile;

    // Latency of this edge (cycles from producer completion to consumer start)
    uint32_t latency = 0;
};

// ============================================================================
// Timing Model for Accurate Scheduling
// ============================================================================

/// Models timing characteristics for cycle-accurate systolic scheduling
struct TimingModel {
    // Bandwidth parameters (bytes per cycle)
    uint64_t dma_bandwidth_bytes_per_cycle = 32;     // External memory → L3 (32 GB/s @ 1GHz)
    uint64_t l3_bandwidth_bytes_per_cycle = 64;      // L3 ↔ L3 link bandwidth

    // Latency parameters
    uint64_t router_latency_cycles = 1;              // Per-router traversal latency
    uint64_t link_latency_cycles = 1;                // Per-link latency

    // Transfer mode
    bool store_and_forward = true;                   // If true, entire tile received before forwarding
                                                     // If false, flit-pipelined (forward as flits arrive)

    // Mesh configuration (for hop count calculation)
    uint8_t mesh_rows = 4;
    uint8_t mesh_cols = 4;

    /// Calculate time to inject a tile into the network
    uint64_t injection_duration(uint64_t tile_bytes) const {
        return (tile_bytes + l3_bandwidth_bytes_per_cycle - 1) / l3_bandwidth_bytes_per_cycle;
    }

    /// Calculate time for a tile to propagate through H hops
    uint64_t propagation_delay(uint64_t tile_bytes, uint8_t hops) const {
        if (hops == 0) return 0;
        if (store_and_forward) {
            // Store-and-forward: must wait for entire tile at each hop
            // Each hop takes: injection_duration (to receive) + injection_duration (to forward)
            // But pipelining means: first hop's forward overlaps with second hop's receive
            // Total: injection_duration * hops (each hop adds one tile transfer time)
            return injection_duration(tile_bytes) * hops;
        } else {
            // Flit-pipelined: tiles can forward as flits arrive
            // Just router + link latency per hop
            return (router_latency_cycles + link_latency_cycles) * hops;
        }
    }

    /// Calculate time for a DMA load
    uint64_t dma_load_duration(uint64_t tile_bytes) const {
        return (tile_bytes + dma_bandwidth_bytes_per_cycle - 1) / dma_bandwidth_bytes_per_cycle;
    }

    /// Calculate Manhattan distance (hop count) between two L3 tiles
    uint8_t hop_count(uint8_t src_l3, uint8_t dst_l3) const {
        uint8_t src_row = src_l3 / mesh_cols;
        uint8_t src_col = src_l3 % mesh_cols;
        uint8_t dst_row = dst_l3 / mesh_cols;
        uint8_t dst_col = dst_l3 % mesh_cols;
        return static_cast<uint8_t>(
            (src_row > dst_row ? src_row - dst_row : dst_row - src_row) +
            (src_col > dst_col ? src_col - dst_col : dst_col - src_col)
        );
    }

    /// Calculate total L3→L3 transfer duration including propagation
    uint64_t l3_transfer_duration(uint64_t tile_bytes, uint8_t src_l3, uint8_t dst_l3) const {
        uint8_t hops = hop_count(src_l3, dst_l3);
        // Total time = injection + propagation to destination
        return injection_duration(tile_bytes) + propagation_delay(tile_bytes, hops);
    }
};

// ============================================================================
// Tile Data Flow Graph
// ============================================================================

/// Represents the complete data flow for a tiled computation
class TileDataFlowGraph {
public:
    TileDataFlowGraph() = default;

    // ========== Node Management ==========

    /// Add a node to the graph, returns node ID
    size_t add_node(const DFNode& node);

    /// Add a node with minimal parameters
    size_t add_dma_load(const TileDescriptor& tile, uint8_t dest_l3);
    size_t add_dma_store(const TileDescriptor& tile, uint8_t src_l3);
    size_t add_l3_transfer(const TileDescriptor& tile, uint8_t src_l3, uint8_t dst_l3);
    size_t add_l3_to_l2(const TileDescriptor& tile, uint8_t l3_id, uint8_t l2_bank);
    size_t add_l2_to_l3(const TileDescriptor& tile, uint8_t l3_id, uint8_t l2_bank);
    size_t add_matmul(uint8_t l3_id, uint32_t M, uint32_t N, uint32_t K,
                      const TileDescriptor& a, const TileDescriptor& b, const TileDescriptor& c);
    size_t add_barrier(uint8_t l3_id);

    /// Get a node by ID
    DFNode& node(size_t id) { return nodes_[id]; }
    const DFNode& node(size_t id) const { return nodes_[id]; }

    /// Get all nodes
    const std::vector<DFNode>& nodes() const { return nodes_; }
    std::vector<DFNode>& nodes() { return nodes_; }

    size_t num_nodes() const { return nodes_.size(); }

    // ========== Edge Management ==========

    /// Add a data dependency edge
    void add_edge(size_t from, size_t to, DFEdgeType type = DFEdgeType::DATA);

    /// Add a data edge with tile information
    void add_data_edge(size_t from, size_t to, const TileDescriptor& tile);

    /// Get all edges
    const std::vector<DFEdge>& edges() const { return edges_; }

    size_t num_edges() const { return edges_.size(); }

    // ========== Graph Queries ==========

    /// Get nodes that have no predecessors (ready to execute)
    std::vector<size_t> get_sources() const;

    /// Get nodes that have no successors (final outputs)
    std::vector<size_t> get_sinks() const;

    /// Get all nodes of a specific type
    std::vector<size_t> get_nodes_by_type(DFNodeType type) const;

    /// Get all nodes assigned to a specific L3 tile
    std::vector<size_t> get_nodes_by_l3(uint8_t l3_id) const;

    /// Check if the graph is acyclic
    bool is_acyclic() const;

    /// Get topological order of nodes
    std::vector<size_t> topological_order() const;

    /// Get critical path length (longest path in cycles)
    int64_t critical_path_length() const;

    // ========== Graph Transformations ==========

    /// Compute earliest start times for all nodes
    void compute_earliest_starts();

    /// Assign priorities based on critical path
    void assign_critical_path_priorities();

    // ========== Matmul-Specific Builders ==========

    /// Build a DFG for tiled matmul C[M,N] = A[M,K] × B[K,N]
    static TileDataFlowGraph build_matmul(
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles,
        uint8_t num_l3_tiles = 16);

    /// Build a DFG for matmul with A-stationary dataflow
    static TileDataFlowGraph build_matmul_a_stationary(
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles);

    /// Build a DFG for matmul with B-stationary dataflow
    static TileDataFlowGraph build_matmul_b_stationary(
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles);

    /// Build a DFG for matmul with output-stationary dataflow
    static TileDataFlowGraph build_matmul_output_stationary(
        uint32_t M, uint32_t N, uint32_t K,
        uint32_t m_tiles, uint32_t n_tiles, uint32_t k_tiles);

    // ========== Timing Model ==========

    /// Set the timing model for accurate duration calculations
    void set_timing_model(const TimingModel& model) { timing_ = model; }

    /// Get the timing model
    const TimingModel& timing_model() const { return timing_; }
    TimingModel& timing_model() { return timing_; }

    // ========== Debug ==========

    std::string to_string() const;
    std::string to_dot() const;  // Generate Graphviz DOT format

    /// Print graph summary
    void print_summary() const;

private:
    std::vector<DFNode> nodes_;
    std::vector<DFEdge> edges_;

    // Index structures for fast lookup
    std::map<uint8_t, std::vector<size_t>> nodes_by_l3_;

    // Timing model for accurate duration calculations
    TimingModel timing_;

    // Helper for cycle detection
    bool has_cycle_from(size_t node, std::vector<int>& state) const;
};

// ============================================================================
// DFG Schedule
// ============================================================================

/// A schedule assigns start times to all nodes in a DFG
struct DFGSchedule {
    struct ScheduledNode {
        size_t node_id;
        int64_t start_cycle;
        int64_t end_cycle;
        uint8_t resource_id;    // L3 tile or other resource
    };

    std::vector<ScheduledNode> nodes;
    int64_t makespan = 0;       // Total execution time

    // Per-resource schedules
    std::map<uint8_t, std::vector<size_t>> l3_schedules;

    /// Get schedule for a specific L3 tile
    std::vector<ScheduledNode> get_l3_schedule(uint8_t l3_id) const;

    /// Find scheduled node by ID
    const ScheduledNode* find_node(size_t node_id) const {
        for (const auto& sn : nodes) {
            if (sn.node_id == node_id) {
                return &sn;
            }
        }
        return nullptr;
    }

    /// Validate schedule (check dependencies)
    bool validate(const TileDataFlowGraph& dfg) const;
};

// ============================================================================
// DFG Scheduler
// ============================================================================

/// Schedules nodes in a DFG to minimize makespan
class DFGScheduler {
public:
    enum class Algorithm {
        ASAP,               // As Soon As Possible
        ALAP,               // As Late As Possible
        LIST_SCHEDULING,    // Priority-based list scheduling
        CRITICAL_PATH,      // Critical path scheduling
    };

    struct Config {
        Algorithm algorithm = Algorithm::LIST_SCHEDULING;
        uint8_t num_l3_tiles = 16;

        // Resource constraints
        uint8_t max_concurrent_dma = 4;
        uint8_t max_concurrent_l3_transfers = 8;
    };

    /// Default constructor
    DFGScheduler();

    /// Constructor with custom configuration
    explicit DFGScheduler(const Config& config);

    /// Schedule the graph
    DFGSchedule schedule(const TileDataFlowGraph& dfg);

    /// Get estimated makespan without full scheduling
    int64_t estimate_makespan(const TileDataFlowGraph& dfg) const;

private:
    Config config_;

    DFGSchedule schedule_asap(const TileDataFlowGraph& dfg);
    DFGSchedule schedule_list(const TileDataFlowGraph& dfg);
};

} // namespace sw::kpu::dataflow
