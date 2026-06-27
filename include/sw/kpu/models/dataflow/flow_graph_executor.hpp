// ============================================================================
// include/sw/kpu/models/dataflow/flow_graph_executor.hpp
// Base Flow Graph Executor - Dataflow execution engine for OperandFlowGraphs
// ============================================================================
//
// This executor implements dataflow semantics where nodes fire when all their
// input operands are ready. The execution model is:
//
// 1. Track ready state of all operands (tokens)
// 2. Find nodes whose inputs are all ready (fireable nodes)
// 3. Fire those nodes, which produces output operands
// 4. Repeat until no more nodes can fire
//
// ============================================================================

#pragma once

#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <optional>
#include <string>

namespace sw::kpu::dataflow {

// ============================================================================
// Operand Key for hashing
// ============================================================================

/// Unique key for identifying an operand in hash maps
struct OperandKey {
    OperandType type;
    uint16_t i, j, k;
    Location location;
    uint8_t node_id;

    static OperandKey from_operand(const Operand& op) {
        return {op.type, op.coord.i, op.coord.j, op.coord.k, op.location, op.node_id};
    }

    bool operator==(const OperandKey& other) const {
        return type == other.type && i == other.i && j == other.j && k == other.k &&
               location == other.location && node_id == other.node_id;
    }
};

struct OperandKeyHash {
    std::size_t operator()(const OperandKey& k) const {
        std::size_t h = static_cast<std::size_t>(k.type);
        h = h * 31 + k.i;
        h = h * 31 + k.j;
        h = h * 31 + k.k;
        h = h * 31 + static_cast<std::size_t>(k.location);
        h = h * 31 + k.node_id;
        return h;
    }
};

// ============================================================================
// Node State
// ============================================================================

/// State of a flow node during execution
enum class NodeState : uint8_t {
    WAITING = 0,    // Waiting for inputs
    READY = 1,      // All inputs ready, can fire
    FIRING = 2,     // Currently executing
    COMPLETED = 3,  // Finished execution
    BLOCKED = 4,    // Blocked (e.g., by backpressure)
};

inline const char* to_string(NodeState state) {
    switch (state) {
        case NodeState::WAITING: return "WAITING";
        case NodeState::READY: return "READY";
        case NodeState::FIRING: return "FIRING";
        case NodeState::COMPLETED: return "COMPLETED";
        case NodeState::BLOCKED: return "BLOCKED";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Execution Event
// ============================================================================

/// Types of events during execution
enum class ExecutionEventType : uint8_t {
    OPERAND_READY = 0,      // An operand became ready
    OPERAND_CONSUMED = 1,   // An operand was consumed
    NODE_FIRED = 2,         // A node started executing
    NODE_COMPLETED = 3,     // A node finished executing
    BUFFER_AVAILABLE = 4,   // A buffer became available
    EXECUTION_COMPLETE = 5, // All nodes completed
    DEADLOCK = 6,           // No progress possible
};

/// Event generated during execution
struct ExecutionEvent {
    ExecutionEventType type;
    uint64_t cycle = 0;
    uint32_t node_id = 0;
    std::optional<Operand> operand;
    std::string message;

    std::string to_string() const {
        std::string s = "@" + std::to_string(cycle) + " ";
        switch (type) {
            case ExecutionEventType::OPERAND_READY:
                s += "OPERAND_READY";
                if (operand) s += " " + operand->to_string();
                break;
            case ExecutionEventType::OPERAND_CONSUMED:
                s += "OPERAND_CONSUMED";
                if (operand) s += " " + operand->to_string();
                break;
            case ExecutionEventType::NODE_FIRED:
                s += "NODE_FIRED[" + std::to_string(node_id) + "]";
                break;
            case ExecutionEventType::NODE_COMPLETED:
                s += "NODE_COMPLETED[" + std::to_string(node_id) + "]";
                break;
            case ExecutionEventType::BUFFER_AVAILABLE:
                s += "BUFFER_AVAILABLE";
                if (operand) s += " " + operand->to_string();
                break;
            case ExecutionEventType::EXECUTION_COMPLETE:
                s += "EXECUTION_COMPLETE";
                break;
            case ExecutionEventType::DEADLOCK:
                s += "DEADLOCK: " + message;
                break;
        }
        return s;
    }
};

// ============================================================================
// Execution Statistics
// ============================================================================

struct ExecutionStats {
    uint64_t total_cycles = 0;
    uint32_t nodes_fired = 0;
    uint32_t operands_produced = 0;
    uint32_t operands_consumed = 0;

    // Per-operation counts
    std::unordered_map<Operation, uint32_t> operation_counts;

    // Timing
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;

    void record_operation(Operation op) {
        operation_counts[op]++;
    }

    std::string to_string() const {
        std::string s = "ExecutionStats {\n";
        s += "  cycles: " + std::to_string(end_cycle - start_cycle) + "\n";
        s += "  nodes_fired: " + std::to_string(nodes_fired) + "\n";
        s += "  operands_produced: " + std::to_string(operands_produced) + "\n";
        s += "  operands_consumed: " + std::to_string(operands_consumed) + "\n";
        s += "  operations:\n";
        for (const auto& [op, count] : operation_counts) {
            s += "    " + std::string(sw::kpu::dataflow::to_string(op)) + ": " +
                 std::to_string(count) + "\n";
        }
        s += "}\n";
        return s;
    }
};

// ============================================================================
// Flow Graph Executor (Base Class)
// ============================================================================

/// Base class for executing Operand Flow Graphs
class FlowGraphExecutor {
public:
    using EventCallback = std::function<void(const ExecutionEvent&)>;

    explicit FlowGraphExecutor(const OperandFlowGraph& graph)
        : graph_(graph), current_cycle_(0) {
        reset();
    }

    virtual ~FlowGraphExecutor() = default;

    // ========== Execution Control ==========

    /// Reset executor to initial state
    virtual void reset() {
        current_cycle_ = 0;
        node_states_.clear();
        node_states_.resize(graph_.num_nodes(), NodeState::WAITING);
        ready_operands_.clear();
        pending_events_.clear();
        stats_ = ExecutionStats{};

        // Initialize node states based on graph structure
        update_all_node_states();
    }

    /// Inject an external operand as ready (e.g., from another level)
    void inject_operand(const Operand& operand, uint64_t cycle = 0) {
        auto key = OperandKey::from_operand(operand);
        ready_operands_[key] = OperandToken{operand, true, cycle, false, 0};
        record_event(ExecutionEventType::OPERAND_READY, cycle, 0, operand);
        update_all_node_states();
    }

    /// Inject a buffer availability token
    void inject_buffer_available(Location loc, uint8_t node_id, uint64_t cycle = 0) {
        auto token = make_buffer_token(loc, node_id);
        inject_operand(token, cycle);
    }

    /// Execute one step (fire all ready nodes)
    /// Returns true if any nodes fired
    bool step() {
        auto ready_nodes = get_ready_nodes();
        if (ready_nodes.empty()) {
            return false;
        }

        for (uint32_t node_id : ready_nodes) {
            fire_node(node_id);
        }

        current_cycle_++;
        update_all_node_states();
        return true;
    }

    /// Execute until no more progress can be made
    /// Returns true if execution completed successfully
    bool run(uint64_t max_cycles = 10000) {
        stats_.start_cycle = current_cycle_;

        while (current_cycle_ < max_cycles) {
            if (is_complete()) {
                stats_.end_cycle = current_cycle_;
                record_event(ExecutionEventType::EXECUTION_COMPLETE, current_cycle_);
                return true;
            }

            if (!step()) {
                // No progress - check for deadlock
                if (!is_complete()) {
                    record_event(ExecutionEventType::DEADLOCK, current_cycle_, 0, std::nullopt,
                                "No nodes ready but execution not complete");
                    return false;
                }
                break;
            }
        }

        stats_.end_cycle = current_cycle_;
        return is_complete();
    }

    /// Check if all nodes have completed
    bool is_complete() const {
        for (const auto& state : node_states_) {
            if (state != NodeState::COMPLETED) {
                return false;
            }
        }
        return true;
    }

    // ========== State Queries ==========

    uint64_t current_cycle() const { return current_cycle_; }

    NodeState get_node_state(uint32_t node_id) const {
        if (node_id >= node_states_.size()) return NodeState::WAITING;
        return node_states_[node_id];
    }

    bool is_operand_ready(const Operand& operand) const {
        auto key = OperandKey::from_operand(operand);
        auto it = ready_operands_.find(key);
        return it != ready_operands_.end() && it->second.ready;
    }

    std::vector<uint32_t> get_ready_nodes() const {
        std::vector<uint32_t> result;
        for (size_t i = 0; i < node_states_.size(); ++i) {
            if (node_states_[i] == NodeState::READY) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
        return result;
    }

    std::vector<uint32_t> get_completed_nodes() const {
        std::vector<uint32_t> result;
        for (size_t i = 0; i < node_states_.size(); ++i) {
            if (node_states_[i] == NodeState::COMPLETED) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
        return result;
    }

    std::vector<uint32_t> get_waiting_nodes() const {
        std::vector<uint32_t> result;
        for (size_t i = 0; i < node_states_.size(); ++i) {
            if (node_states_[i] == NodeState::WAITING) {
                result.push_back(static_cast<uint32_t>(i));
            }
        }
        return result;
    }

    // ========== Statistics and Events ==========

    const ExecutionStats& stats() const { return stats_; }

    const std::vector<ExecutionEvent>& events() const { return pending_events_; }

    void set_event_callback(EventCallback callback) {
        event_callback_ = std::move(callback);
    }

    // ========== Graph Access ==========

    const OperandFlowGraph& graph() const { return graph_; }

protected:
    // ========== Virtual Methods for Specialization ==========

    /// Called when a node fires - override to implement actual operations
    virtual void execute_operation([[maybe_unused]] uint32_t node_id, [[maybe_unused]] const FlowNode& node) {
        // Default: just mark completion after latency
        // Subclasses override to perform actual memory/compute operations
    }

    /// Get the latency for a node's operation
    virtual uint16_t get_operation_latency(const FlowNode& node) const {
        return node.latency_cycles > 0 ? node.latency_cycles : 1;
    }

    /// Check if a node can fire (beyond just having inputs ready)
    /// Override for additional constraints like resource availability
    virtual bool can_fire([[maybe_unused]] uint32_t node_id, [[maybe_unused]] const FlowNode& node) const {
        return true;
    }

    // ========== Internal Methods ==========

    void fire_node(uint32_t node_id) {
        if (node_id >= graph_.nodes.size()) return;

        const FlowNode& node = graph_.nodes[node_id];
        node_states_[node_id] = NodeState::FIRING;

        record_event(ExecutionEventType::NODE_FIRED, current_cycle_, node_id);

        // Consume inputs
        for (const auto& input : node.inputs) {
            consume_operand(input);
        }

        // Execute the operation (subclass may override)
        execute_operation(node_id, node);

        // Produce outputs
        for (const auto& output : node.outputs) {
            produce_operand(output);
        }

        // Mark completed
        node_states_[node_id] = NodeState::COMPLETED;
        stats_.nodes_fired++;
        stats_.record_operation(node.operation);

        record_event(ExecutionEventType::NODE_COMPLETED, current_cycle_, node_id);
    }

    void produce_operand(const Operand& operand) {
        auto key = OperandKey::from_operand(operand);
        ready_operands_[key] = OperandToken{operand, true, current_cycle_, false, 0};
        stats_.operands_produced++;
        record_event(ExecutionEventType::OPERAND_READY, current_cycle_, 0, operand);
    }

    void consume_operand(const Operand& operand) {
        auto key = OperandKey::from_operand(operand);
        auto it = ready_operands_.find(key);
        if (it != ready_operands_.end()) {
            it->second.consumed = true;
            it->second.consumed_cycle = current_cycle_;
        }
        stats_.operands_consumed++;
        record_event(ExecutionEventType::OPERAND_CONSUMED, current_cycle_, 0, operand);
    }

    void update_all_node_states() {
        for (size_t i = 0; i < node_states_.size(); ++i) {
            if (node_states_[i] == NodeState::WAITING ||
                node_states_[i] == NodeState::BLOCKED) {
                update_node_state(static_cast<uint32_t>(i));
            }
        }
    }

    void update_node_state(uint32_t node_id) {
        if (node_id >= graph_.nodes.size()) return;
        if (node_states_[node_id] == NodeState::COMPLETED ||
            node_states_[node_id] == NodeState::FIRING) {
            return;
        }

        const FlowNode& node = graph_.nodes[node_id];

        // Check if all inputs are ready
        bool all_inputs_ready = true;
        for (const auto& input : node.inputs) {
            if (!is_operand_ready(input)) {
                all_inputs_ready = false;
                break;
            }
        }

        // Also check predecessor nodes (graph edges)
        for (const auto& edge : graph_.edges) {
            if (edge.to_node == node_id) {
                if (node_states_[edge.from_node] != NodeState::COMPLETED) {
                    all_inputs_ready = false;
                    break;
                }
            }
        }

        if (all_inputs_ready && can_fire(node_id, node)) {
            node_states_[node_id] = NodeState::READY;
        } else {
            node_states_[node_id] = NodeState::WAITING;
        }
    }

    void record_event(ExecutionEventType type, uint64_t cycle,
                      uint32_t node_id = 0,
                      std::optional<Operand> operand = std::nullopt,
                      const std::string& message = "") {
        ExecutionEvent event{type, cycle, node_id, operand, message};
        pending_events_.push_back(event);

        if (event_callback_) {
            event_callback_(event);
        }
    }

protected:
    const OperandFlowGraph& graph_;
    uint64_t current_cycle_;

    std::vector<NodeState> node_states_;
    std::unordered_map<OperandKey, OperandToken, OperandKeyHash> ready_operands_;

    std::vector<ExecutionEvent> pending_events_;
    EventCallback event_callback_;

    ExecutionStats stats_;
};

// ============================================================================
// Output Events (for inter-level communication)
// ============================================================================

/// Events produced by an executor for consumption by another level
struct OutputEvent {
    Operand operand;
    uint64_t cycle;
    bool is_ready;          // true = TILE_READY, false = BUFFER_AVAILABLE
};

/// Collector for output events
class OutputEventCollector {
public:
    void add_ready(const Operand& operand, uint64_t cycle) {
        events_.push_back({operand, cycle, true});
    }

    void add_buffer_available(const Operand& operand, uint64_t cycle) {
        events_.push_back({operand, cycle, false});
    }

    const std::vector<OutputEvent>& events() const { return events_; }

    void clear() { events_.clear(); }

    std::vector<OutputEvent> get_ready_events() const {
        std::vector<OutputEvent> result;
        for (const auto& e : events_) {
            if (e.is_ready) result.push_back(e);
        }
        return result;
    }

    std::vector<OutputEvent> get_buffer_events() const {
        std::vector<OutputEvent> result;
        for (const auto& e : events_) {
            if (!e.is_ready) result.push_back(e);
        }
        return result;
    }

private:
    std::vector<OutputEvent> events_;
};

} // namespace sw::kpu::dataflow
