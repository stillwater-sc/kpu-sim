// ============================================================================
// include/sw/kpu/models/dataflow/operand_flow_graph.hpp
// Operand Flow Graph - Dataflow representation for tile-based computation
// ============================================================================
//
// The Operand Flow Graph (OFG) represents computation as data-driven execution
// where operations fire when their input operands are ready, not when upstream
// operations complete.
//
// Key principle: "TILE_READY(A[i,k] @ L3)" not "DMA_DONE"
//
// ============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <optional>
#include <functional>

namespace sw::kpu::dataflow {

// ============================================================================
// Operand Types
// ============================================================================

/// Types of operands that flow through the graph
enum class OperandType : uint8_t {
    TILE_A = 0,         // A matrix tile (typically left operand)
    TILE_B = 1,         // B matrix tile (typically right operand)
    TILE_C = 2,         // C matrix tile (output/accumulator)
    TILE_BIAS = 3,      // Bias vector tile
    TILE_GENERIC = 4,   // Generic data tile
    BUFFER_TOKEN = 5,   // Buffer availability token (for backpressure)
    SYNC_TOKEN = 6,     // Synchronization token (control flow)
};

inline const char* to_string(OperandType type) {
    switch (type) {
        case OperandType::TILE_A: return "TILE_A";
        case OperandType::TILE_B: return "TILE_B";
        case OperandType::TILE_C: return "TILE_C";
        case OperandType::TILE_BIAS: return "TILE_BIAS";
        case OperandType::TILE_GENERIC: return "TILE_GENERIC";
        case OperandType::BUFFER_TOKEN: return "BUFFER_TOKEN";
        case OperandType::SYNC_TOKEN: return "SYNC_TOKEN";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Memory Hierarchy Locations
// ============================================================================

/// Location in the memory hierarchy
enum class Location : uint8_t {
    MEMORY = 0,     // External memory (HBM, DDR, etc.)
    L3 = 1,         // L3 tile (on-chip SRAM cache)
    L2 = 2,         // L2 bank (operand buffer near compute)
    L1 = 3,         // L1 buffer (systolic array edge registers)
    ACCUMULATOR = 4, // Accumulator register (inside compute unit)
};

inline const char* to_string(Location loc) {
    switch (loc) {
        case Location::MEMORY: return "MEMORY";
        case Location::L3: return "L3";
        case Location::L2: return "L2";
        case Location::L1: return "L1";
        case Location::ACCUMULATOR: return "ACCUMULATOR";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Tile Coordinates
// ============================================================================

/// Tile coordinate in the logical tensor space
struct TileCoord {
    uint16_t i = 0;     // Row tile index (M dimension)
    uint16_t j = 0;     // Column tile index (N dimension)
    uint16_t k = 0;     // Reduction tile index (K dimension)

    bool operator==(const TileCoord& other) const {
        return i == other.i && j == other.j && k == other.k;
    }

    bool operator!=(const TileCoord& other) const {
        return !(*this == other);
    }

    std::string to_string() const {
        return "[" + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k) + "]";
    }
};

/// Hash function for TileCoord (for use in unordered containers)
struct TileCoordHash {
    std::size_t operator()(const TileCoord& c) const {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(c.i) << 32) |
            (static_cast<uint64_t>(c.j) << 16) |
            static_cast<uint64_t>(c.k)
        );
    }
};

// ============================================================================
// Operand Descriptor
// ============================================================================

/// Complete description of an operand in the flow graph
struct Operand {
    OperandType type = OperandType::TILE_GENERIC;
    TileCoord coord;            // Logical tile coordinates
    Location location = Location::MEMORY;
    uint8_t node_id = 0;        // Physical node (L3 tile ID, L2 bank ID, etc.)

    // Optional: specific offset within the location
    uint64_t offset = 0;

    // Optional: size in bytes (0 = use default tile size)
    uint32_t size_bytes = 0;

    bool operator==(const Operand& other) const {
        return type == other.type &&
               coord == other.coord &&
               location == other.location &&
               node_id == other.node_id;
    }

    std::string to_string() const {
        std::string s = sw::kpu::dataflow::to_string(type);
        s += coord.to_string();
        s += "@" + std::string(sw::kpu::dataflow::to_string(location));
        s += "[" + std::to_string(node_id) + "]";
        return s;
    }
};

// ============================================================================
// Operations
// ============================================================================

/// Operations that can be performed in a flow node
enum class Operation : uint8_t {
    // No operation
    NOP = 0,

    // Memory → L3 operations (DMA level)
    LOAD = 1,               // Load tile from external memory to L3
    STORE = 2,              // Store tile from L3 to external memory

    // L3 → L2 operations (BlockMover level)
    PUSH_TO_L2 = 10,        // Push tile from L3 to L2 bank
    PULL_FROM_L2 = 11,      // Pull tile from L2 back to L3

    // L3 ↔ L3 operations (mesh communication)
    SEND_EAST = 20,         // Send tile to eastern neighbor
    SEND_WEST = 21,         // Send tile to western neighbor
    SEND_NORTH = 22,        // Send tile to northern neighbor
    SEND_SOUTH = 23,        // Send tile to southern neighbor
    SEND_TO = 24,           // Send tile to arbitrary node
    RECEIVE = 25,           // Receive tile from any direction

    // L2 → L1 operations (Streamer level)
    FEED_WEST = 30,         // Feed tile to systolic west edge (A operand)
    FEED_NORTH = 31,        // Feed tile to systolic north edge (B operand)
    DRAIN = 32,             // Drain result from accumulator to L2

    // Compute operations
    MATMUL = 40,            // Matrix multiply: C += A × B
    ACCUMULATE = 41,        // Accumulate partial sum
    APPLY_BIAS = 42,        // Add bias vector
    ACTIVATE = 43,          // Apply activation function

    // Control operations
    JOIN = 50,              // AND of multiple inputs (all must be ready)
    FORK = 51,              // Replicate to multiple outputs
    BARRIER = 52,           // Wait for all pending operations
};

inline const char* to_string(Operation op) {
    switch (op) {
        case Operation::NOP: return "NOP";
        case Operation::LOAD: return "LOAD";
        case Operation::STORE: return "STORE";
        case Operation::PUSH_TO_L2: return "PUSH_TO_L2";
        case Operation::PULL_FROM_L2: return "PULL_FROM_L2";
        case Operation::SEND_EAST: return "SEND_EAST";
        case Operation::SEND_WEST: return "SEND_WEST";
        case Operation::SEND_NORTH: return "SEND_NORTH";
        case Operation::SEND_SOUTH: return "SEND_SOUTH";
        case Operation::SEND_TO: return "SEND_TO";
        case Operation::RECEIVE: return "RECEIVE";
        case Operation::FEED_WEST: return "FEED_WEST";
        case Operation::FEED_NORTH: return "FEED_NORTH";
        case Operation::DRAIN: return "DRAIN";
        case Operation::MATMUL: return "MATMUL";
        case Operation::ACCUMULATE: return "ACCUMULATE";
        case Operation::APPLY_BIAS: return "APPLY_BIAS";
        case Operation::ACTIVATE: return "ACTIVATE";
        case Operation::JOIN: return "JOIN";
        case Operation::FORK: return "FORK";
        case Operation::BARRIER: return "BARRIER";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Flow Node Types
// ============================================================================

/// Types of nodes in the flow graph
enum class FlowNodeType : uint8_t {
    WAIT = 0,       // Wait for operands on input ports
    FIRE = 1,       // Execute operation when inputs ready
    PRODUCE = 2,    // Produce operands on output ports
    JOIN = 3,       // AND of multiple inputs (all must be ready)
    FORK = 4,       // Replicate single input to multiple outputs
    CONDITIONAL = 5, // Fire only if condition is true
};

inline const char* to_string(FlowNodeType type) {
    switch (type) {
        case FlowNodeType::WAIT: return "WAIT";
        case FlowNodeType::FIRE: return "FIRE";
        case FlowNodeType::PRODUCE: return "PRODUCE";
        case FlowNodeType::JOIN: return "JOIN";
        case FlowNodeType::FORK: return "FORK";
        case FlowNodeType::CONDITIONAL: return "CONDITIONAL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Flow Node
// ============================================================================

/// A node in the Operand Flow Graph
struct FlowNode {
    // Node identification
    uint32_t id = 0;
    std::string name;

    // Node type and operation
    FlowNodeType type = FlowNodeType::FIRE;
    Operation operation = Operation::NOP;

    // Input operands (what this node consumes/waits for)
    std::vector<Operand> inputs;

    // Output operands (what this node produces)
    std::vector<Operand> outputs;

    // Optional: iteration context (which k iteration, etc.)
    std::optional<uint16_t> iteration;

    // Optional: conditional predicate (for CONDITIONAL nodes)
    // e.g., "j < 3" for edge boundary checks
    std::string predicate;

    // Estimated latency in cycles (for scheduling hints)
    uint16_t latency_cycles = 1;

    std::string to_string() const {
        std::string s = "[" + std::to_string(id) + "] ";
        s += std::string(sw::kpu::dataflow::to_string(type));
        s += "(" + std::string(sw::kpu::dataflow::to_string(operation)) + ")";
        if (!name.empty()) {
            s += " \"" + name + "\"";
        }
        if (!inputs.empty()) {
            s += " in:{";
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i > 0) s += ", ";
                s += inputs[i].to_string();
            }
            s += "}";
        }
        if (!outputs.empty()) {
            s += " out:{";
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (i > 0) s += ", ";
                s += outputs[i].to_string();
            }
            s += "}";
        }
        return s;
    }
};

// ============================================================================
// Flow Edge
// ============================================================================

/// An edge connecting two nodes in the flow graph
struct FlowEdge {
    uint32_t from_node = 0;     // Source node ID
    uint32_t to_node = 0;       // Destination node ID
    uint8_t from_port = 0;      // Output port on source (for FORK nodes)
    uint8_t to_port = 0;        // Input port on destination (for JOIN nodes)

    // Optional: the operand flowing along this edge
    std::optional<Operand> operand;

    std::string to_string() const {
        std::string s = std::to_string(from_node);
        if (from_port > 0) s += ":" + std::to_string(from_port);
        s += " -> " + std::to_string(to_node);
        if (to_port > 0) s += ":" + std::to_string(to_port);
        if (operand) {
            s += " [" + operand->to_string() + "]";
        }
        return s;
    }
};

// ============================================================================
// Execution Level
// ============================================================================

/// Which level of the memory hierarchy this graph operates at
enum class ExecutionLevel : uint8_t {
    DMA = 1,            // Memory ↔ L3 (DMA engine)
    BLOCK_MOVER = 2,    // L3 ↔ L2 (BlockMover)
    STREAMER = 3,       // L2 ↔ L1 (Streamer)
    COMPUTE = 4,        // L1 ↔ Accumulator (Compute fabric)
};

inline const char* to_string(ExecutionLevel level) {
    switch (level) {
        case ExecutionLevel::DMA: return "DMA";
        case ExecutionLevel::BLOCK_MOVER: return "BLOCK_MOVER";
        case ExecutionLevel::STREAMER: return "STREAMER";
        case ExecutionLevel::COMPUTE: return "COMPUTE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Operand Flow Graph
// ============================================================================

/// Complete Operand Flow Graph for a sequencer
struct OperandFlowGraph {
    // Graph structure
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;

    // Graph metadata
    std::string name;
    ExecutionLevel level = ExecutionLevel::BLOCK_MOVER;
    uint8_t node_id = 0;        // Which physical node (L3 tile, CT, etc.)

    // Mesh position (for distributed graphs)
    uint8_t mesh_row = 0;
    uint8_t mesh_col = 0;

    // Problem dimensions (for context)
    uint16_t m_tiles = 0;       // Number of M tiles
    uint16_t n_tiles = 0;       // Number of N tiles
    uint16_t k_tiles = 0;       // Number of K tiles

    // ========== Graph Construction ==========

    /// Add a node and return its ID
    uint32_t add_node(FlowNode node) {
        node.id = static_cast<uint32_t>(nodes.size());
        nodes.push_back(std::move(node));
        return nodes.back().id;
    }

    /// Add a WAIT node
    uint32_t add_wait(const std::vector<Operand>& inputs, const std::string& name = "") {
        FlowNode node;
        node.type = FlowNodeType::WAIT;
        node.operation = Operation::NOP;
        node.inputs = inputs;
        node.name = name;
        return add_node(std::move(node));
    }

    /// Add a FIRE node with an operation
    uint32_t add_fire(Operation op, const std::vector<Operand>& inputs,
                      const std::vector<Operand>& outputs, const std::string& name = "") {
        FlowNode node;
        node.type = FlowNodeType::FIRE;
        node.operation = op;
        node.inputs = inputs;
        node.outputs = outputs;
        node.name = name;
        return add_node(std::move(node));
    }

    /// Add a PRODUCE node
    uint32_t add_produce(const std::vector<Operand>& outputs, const std::string& name = "") {
        FlowNode node;
        node.type = FlowNodeType::PRODUCE;
        node.operation = Operation::NOP;
        node.outputs = outputs;
        node.name = name;
        return add_node(std::move(node));
    }

    /// Add a JOIN node (AND of multiple inputs)
    uint32_t add_join(const std::vector<Operand>& inputs, const std::string& name = "") {
        FlowNode node;
        node.type = FlowNodeType::JOIN;
        node.operation = Operation::JOIN;
        node.inputs = inputs;
        node.name = name;
        return add_node(std::move(node));
    }

    /// Add a FORK node (replicate to multiple outputs)
    uint32_t add_fork(const Operand& input, const std::vector<Operand>& outputs,
                      const std::string& name = "") {
        FlowNode node;
        node.type = FlowNodeType::FORK;
        node.operation = Operation::FORK;
        node.inputs = {input};
        node.outputs = outputs;
        node.name = name;
        return add_node(std::move(node));
    }

    /// Add an edge between nodes
    void add_edge(uint32_t from, uint32_t to, uint8_t from_port = 0, uint8_t to_port = 0) {
        edges.push_back({from, to, from_port, to_port, std::nullopt});
    }

    /// Add an edge with operand annotation
    void add_edge(uint32_t from, uint32_t to, const Operand& operand) {
        edges.push_back({from, to, 0, 0, operand});
    }

    // ========== Graph Queries ==========

    /// Get all nodes that produce a given operand
    std::vector<uint32_t> find_producers(const Operand& operand) const {
        std::vector<uint32_t> result;
        for (const auto& node : nodes) {
            for (const auto& out : node.outputs) {
                if (out == operand) {
                    result.push_back(node.id);
                    break;
                }
            }
        }
        return result;
    }

    /// Get all nodes that consume a given operand
    std::vector<uint32_t> find_consumers(const Operand& operand) const {
        std::vector<uint32_t> result;
        for (const auto& node : nodes) {
            for (const auto& in : node.inputs) {
                if (in == operand) {
                    result.push_back(node.id);
                    break;
                }
            }
        }
        return result;
    }

    /// Get predecessor nodes (nodes with edges pointing to given node)
    std::vector<uint32_t> get_predecessors(uint32_t node_id) const {
        std::vector<uint32_t> result;
        for (const auto& edge : edges) {
            if (edge.to_node == node_id) {
                result.push_back(edge.from_node);
            }
        }
        return result;
    }

    /// Get successor nodes (nodes that given node points to)
    std::vector<uint32_t> get_successors(uint32_t node_id) const {
        std::vector<uint32_t> result;
        for (const auto& edge : edges) {
            if (edge.from_node == node_id) {
                result.push_back(edge.to_node);
            }
        }
        return result;
    }

    /// Get entry nodes (nodes with no predecessors)
    std::vector<uint32_t> get_entry_nodes() const {
        std::vector<uint32_t> result;
        for (const auto& node : nodes) {
            if (get_predecessors(node.id).empty()) {
                result.push_back(node.id);
            }
        }
        return result;
    }

    /// Get exit nodes (nodes with no successors)
    std::vector<uint32_t> get_exit_nodes() const {
        std::vector<uint32_t> result;
        for (const auto& node : nodes) {
            if (get_successors(node.id).empty()) {
                result.push_back(node.id);
            }
        }
        return result;
    }

    // ========== Validation ==========

    /// Check if the graph is well-formed
    struct ValidationResult {
        bool valid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    ValidationResult validate() const {
        ValidationResult result;

        // Check for duplicate node IDs
        std::vector<bool> seen(nodes.size(), false);
        for (const auto& node : nodes) {
            if (node.id >= nodes.size()) {
                result.valid = false;
                result.errors.push_back("Node ID " + std::to_string(node.id) + " out of range");
            } else if (seen[node.id]) {
                result.valid = false;
                result.errors.push_back("Duplicate node ID " + std::to_string(node.id));
            } else {
                seen[node.id] = true;
            }
        }

        // Check edges reference valid nodes
        for (const auto& edge : edges) {
            if (edge.from_node >= nodes.size()) {
                result.valid = false;
                result.errors.push_back("Edge from invalid node " + std::to_string(edge.from_node));
            }
            if (edge.to_node >= nodes.size()) {
                result.valid = false;
                result.errors.push_back("Edge to invalid node " + std::to_string(edge.to_node));
            }
        }

        // Check WAIT nodes have inputs
        for (const auto& node : nodes) {
            if (node.type == FlowNodeType::WAIT && node.inputs.empty()) {
                result.warnings.push_back("WAIT node " + std::to_string(node.id) + " has no inputs");
            }
            if (node.type == FlowNodeType::PRODUCE && node.outputs.empty()) {
                result.warnings.push_back("PRODUCE node " + std::to_string(node.id) + " has no outputs");
            }
        }

        return result;
    }

    // ========== Serialization ==========

    std::string to_string() const {
        std::string s = "OperandFlowGraph \"" + name + "\" {\n";
        s += "  level: " + std::string(sw::kpu::dataflow::to_string(level)) + "\n";
        s += "  node_id: " + std::to_string(node_id) + "\n";
        s += "  mesh: [" + std::to_string(mesh_row) + "," + std::to_string(mesh_col) + "]\n";
        s += "  tiles: M=" + std::to_string(m_tiles) + " N=" + std::to_string(n_tiles) +
             " K=" + std::to_string(k_tiles) + "\n";
        s += "\n  Nodes:\n";
        for (const auto& node : nodes) {
            s += "    " + node.to_string() + "\n";
        }
        s += "\n  Edges:\n";
        for (const auto& edge : edges) {
            s += "    " + edge.to_string() + "\n";
        }
        s += "}\n";
        return s;
    }

    // ========== Statistics ==========

    size_t num_nodes() const { return nodes.size(); }
    size_t num_edges() const { return edges.size(); }

    size_t count_nodes_by_type(FlowNodeType type) const {
        size_t count = 0;
        for (const auto& node : nodes) {
            if (node.type == type) ++count;
        }
        return count;
    }

    size_t count_nodes_by_operation(Operation op) const {
        size_t count = 0;
        for (const auto& node : nodes) {
            if (node.operation == op) ++count;
        }
        return count;
    }
};

// ============================================================================
// Coordination Events
// ============================================================================

/// Forward flow event: Tile T is available at location Loc
struct TileReadyEvent {
    Operand operand;
    uint64_t timestamp = 0;     // Cycle when event occurred

    std::string to_string() const {
        return "TILE_READY(" + operand.to_string() + ")@" + std::to_string(timestamp);
    }
};

/// Backpressure event: Buffer at Loc can accept a tile
struct BufferAvailableEvent {
    Location location = Location::L3;
    uint8_t node_id = 0;
    uint8_t bank_id = 0;        // For L2: which bank
    uint32_t capacity = 0;      // How many tiles can be accepted
    uint64_t timestamp = 0;

    std::string to_string() const {
        return "BUFFER_AVAILABLE(" + std::string(sw::kpu::dataflow::to_string(location)) +
               "[" + std::to_string(node_id) + "])@" + std::to_string(timestamp);
    }
};

// ============================================================================
// Operand Token (runtime state)
// ============================================================================

/// Runtime token representing an operand's presence
struct OperandToken {
    Operand operand;
    bool ready = false;         // Has the operand arrived?
    uint64_t ready_cycle = 0;   // When did it become ready?

    // For backpressure tokens
    bool consumed = false;      // Has the operand been consumed?
    uint64_t consumed_cycle = 0;
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an operand for a tile at a specific location
inline Operand make_tile_operand(OperandType type, uint16_t i, uint16_t j, uint16_t k,
                                  Location loc, uint8_t node_id) {
    return Operand{type, {i, j, k}, loc, node_id, 0, 0};
}

/// Create a buffer token for backpressure
inline Operand make_buffer_token(Location loc, uint8_t node_id) {
    return Operand{OperandType::BUFFER_TOKEN, {0, 0, 0}, loc, node_id, 0, 0};
}

/// Create an A tile operand
inline Operand tile_a(uint16_t i, uint16_t k, Location loc, uint8_t node_id) {
    return make_tile_operand(OperandType::TILE_A, i, 0, k, loc, node_id);
}

/// Create a B tile operand
inline Operand tile_b(uint16_t k, uint16_t j, Location loc, uint8_t node_id) {
    return make_tile_operand(OperandType::TILE_B, 0, j, k, loc, node_id);
}

/// Create a C tile operand
inline Operand tile_c(uint16_t i, uint16_t j, Location loc, uint8_t node_id) {
    return make_tile_operand(OperandType::TILE_C, i, j, 0, loc, node_id);
}

} // namespace sw::kpu::dataflow
