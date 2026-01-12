// ============================================================================
// include/sw/kpu/models/dataflow/chrome_trace.hpp
// Chrome Trace Format (CTF) export for OFG execution tracing
// ============================================================================
//
// This module provides Chrome Trace Event Format export for visualizing
// distributed OFG execution. The trace format is compatible with:
// - Chrome's chrome://tracing
// - Perfetto (https://ui.perfetto.dev)
// - Trace Compass and other CTF viewers
//
// Architecture:
// - Each ExecutionLevel (DMA, BlockMover, Streamer, Compute) = process (pid)
// - Each physical node (L3 tile, L2 bank, compute tile) = thread (tid)
// - Node executions = duration events (B/E or X)
// - Operand ready/consumed = instant events (i)
//
// Usage:
//   ChromeTraceWriter writer("dma", ExecutionLevel::DMA, 0);  // L3 tile 0
//   writer.add_event(event);  // Add ExecutionEvents
//   writer.write_to_file("dma_trace.json");
//
//   // Or consolidate multiple traces:
//   TraceConsolidator consolidator;
//   consolidator.add_trace(dma_writer);
//   consolidator.add_trace(block_mover_writer);
//   consolidator.write_to_file("full_trace.json");
//
// ============================================================================

#pragma once

#include <sw/kpu/models/dataflow/operand_flow_graph.hpp>
#include <sw/kpu/models/dataflow/flow_graph_executor.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <algorithm>

namespace sw::kpu::dataflow {

// ============================================================================
// Chrome Trace Event Format
// ============================================================================

/// Phase types for Chrome Trace events
enum class TracePhase : char {
    BEGIN = 'B',           // Duration event begin
    END = 'E',             // Duration event end
    COMPLETE = 'X',        // Complete event (has dur field)
    INSTANT = 'i',         // Instant event
    COUNTER = 'C',         // Counter event
    ASYNC_BEGIN = 'b',     // Async event begin
    ASYNC_END = 'e',       // Async event end
    ASYNC_INSTANT = 'n',   // Async instant
    FLOW_BEGIN = 's',      // Flow event begin
    FLOW_END = 'f',        // Flow event end
    METADATA = 'M',        // Metadata event
    OBJECT_CREATE = 'N',   // Object created
    OBJECT_DESTROY = 'D',  // Object destroyed
    OBJECT_SNAPSHOT = 'O', // Object snapshot
};

/// Instant event scope
enum class InstantScope : char {
    GLOBAL = 'g',   // Global scope
    PROCESS = 'p',  // Process scope
    THREAD = 't',   // Thread scope
};

/// A single trace event in Chrome Trace format
struct TraceEvent {
    std::string name;           // Event name
    std::string category;       // Category (for filtering)
    TracePhase phase;           // Event phase (B/E/X/i/etc)
    uint64_t timestamp_us;      // Timestamp in microseconds
    uint64_t duration_us = 0;   // Duration for X phase
    uint32_t pid;               // Process ID (execution level)
    uint32_t tid;               // Thread ID (physical node)
    InstantScope scope = InstantScope::THREAD;  // For instant events
    std::string id;             // For async/flow events

    // Arguments (key-value pairs)
    std::unordered_map<std::string, std::string> args;

    /// Convert to JSON string (single event)
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"name\":\"" << escape_json(name) << "\",";
        ss << "\"cat\":\"" << escape_json(category) << "\",";
        ss << "\"ph\":\"" << static_cast<char>(phase) << "\",";
        ss << "\"ts\":" << timestamp_us << ",";
        if (phase == TracePhase::COMPLETE) {
            ss << "\"dur\":" << duration_us << ",";
        }
        ss << "\"pid\":" << pid << ",";
        ss << "\"tid\":" << tid;

        if (phase == TracePhase::INSTANT) {
            ss << ",\"s\":\"" << static_cast<char>(scope) << "\"";
        }

        if (!id.empty()) {
            ss << ",\"id\":\"" << escape_json(id) << "\"";
        }

        if (!args.empty()) {
            ss << ",\"args\":{";
            bool first = true;
            for (const auto& [key, value] : args) {
                if (!first) ss << ",";
                first = false;
                ss << "\"" << escape_json(key) << "\":\"" << escape_json(value) << "\"";
            }
            ss << "}";
        }

        ss << "}";
        return ss.str();
    }

private:
    static std::string escape_json(const std::string& s) {
        std::ostringstream ss;
        for (char c : s) {
            switch (c) {
                case '"': ss << "\\\""; break;
                case '\\': ss << "\\\\"; break;
                case '\b': ss << "\\b"; break;
                case '\f': ss << "\\f"; break;
                case '\n': ss << "\\n"; break;
                case '\r': ss << "\\r"; break;
                case '\t': ss << "\\t"; break;
                default: ss << c; break;
            }
        }
        return ss.str();
    }
};

// ============================================================================
// Process/Thread Naming for KPU Architecture
// ============================================================================

/// Get process ID for an execution level
inline uint32_t level_to_pid(ExecutionLevel level) {
    return static_cast<uint32_t>(level);
}

/// Get process name for an execution level
inline std::string level_to_process_name(ExecutionLevel level) {
    switch (level) {
        case ExecutionLevel::DMA: return "DMA Engine";
        case ExecutionLevel::BLOCK_MOVER: return "Block Mover";
        case ExecutionLevel::STREAMER: return "Streamer";
        case ExecutionLevel::COMPUTE: return "Compute Fabric";
        default: return "Unknown";
    }
}

/// Get thread name for a physical node at a given level
inline std::string node_to_thread_name(ExecutionLevel level, uint32_t node_id) {
    switch (level) {
        case ExecutionLevel::DMA:
            return "L3 Tile " + std::to_string(node_id);
        case ExecutionLevel::BLOCK_MOVER:
            return "Tile[" + std::to_string(node_id / 4) + "," +
                   std::to_string(node_id % 4) + "]";  // Assuming 4-wide mesh
        case ExecutionLevel::STREAMER:
            return "L2 Bank " + std::to_string(node_id);
        case ExecutionLevel::COMPUTE:
            return "PE[" + std::to_string(node_id / 16) + "," +
                   std::to_string(node_id % 16) + "]";  // Assuming 16-wide array
        default:
            return "Node " + std::to_string(node_id);
    }
}

// ============================================================================
// Chrome Trace Writer
// ============================================================================

/// Configuration for trace cycle-to-time conversion
struct TraceConfig {
    uint64_t cycle_time_ns = 1000;   // Nanoseconds per cycle (1 MHz default for visible traces)
    bool use_microseconds = true;     // Chrome trace uses microseconds

    uint64_t cycle_to_timestamp(uint64_t cycle) const {
        uint64_t ns = cycle * cycle_time_ns;
        // Chrome trace format uses microseconds for 'ts' field
        return use_microseconds ? ns / 1000 : ns;
    }
};

/// Writer for Chrome Trace events from a single executor
class ChromeTraceWriter {
public:
    /// Create a trace writer for a specific executor
    /// @param name Executor name (for display)
    /// @param level Execution level (determines pid)
    /// @param node_id Physical node ID (determines tid)
    /// @param config Timing configuration
    ChromeTraceWriter(const std::string& name,
                      ExecutionLevel level,
                      uint32_t node_id,
                      const TraceConfig& config = {})
        : name_(name)
        , level_(level)
        , node_id_(node_id)
        , pid_(level_to_pid(level))
        , tid_(node_id)
        , config_(config) {
        // Add metadata events for process/thread names
        add_metadata_events();
    }

    // ========== Event Recording ==========

    /// Record an ExecutionEvent from the flow graph executor
    void record_event(const ExecutionEvent& event) {
        switch (event.type) {
            case ExecutionEventType::NODE_FIRED:
                record_node_begin(event);
                break;
            case ExecutionEventType::NODE_COMPLETED:
                record_node_end(event);
                break;
            case ExecutionEventType::OPERAND_READY:
                record_operand_ready(event);
                break;
            case ExecutionEventType::OPERAND_CONSUMED:
                record_operand_consumed(event);
                break;
            case ExecutionEventType::BUFFER_AVAILABLE:
                record_buffer_available(event);
                break;
            case ExecutionEventType::EXECUTION_COMPLETE:
                record_execution_complete(event);
                break;
            case ExecutionEventType::DEADLOCK:
                record_deadlock(event);
                break;
        }
    }

    /// Record a custom duration event
    void record_duration(const std::string& name,
                         const std::string& category,
                         uint64_t start_cycle,
                         uint64_t end_cycle,
                         const std::unordered_map<std::string, std::string>& args = {}) {
        TraceEvent event;
        event.name = name;
        event.category = category;
        event.phase = TracePhase::COMPLETE;
        event.timestamp_us = config_.cycle_to_timestamp(start_cycle);
        event.duration_us = config_.cycle_to_timestamp(end_cycle - start_cycle);
        event.pid = pid_;
        event.tid = tid_;
        event.args = args;
        events_.push_back(event);
    }

    /// Record a custom instant event
    void record_instant(const std::string& name,
                        const std::string& category,
                        uint64_t cycle,
                        InstantScope scope = InstantScope::THREAD,
                        const std::unordered_map<std::string, std::string>& args = {}) {
        TraceEvent event;
        event.name = name;
        event.category = category;
        event.phase = TracePhase::INSTANT;
        event.timestamp_us = config_.cycle_to_timestamp(cycle);
        event.pid = pid_;
        event.tid = tid_;
        event.scope = scope;
        event.args = args;
        events_.push_back(event);
    }

    /// Record a flow event (for visualizing operand movement between levels)
    void record_flow_begin(const std::string& name,
                           const std::string& flow_id,
                           uint64_t cycle,
                           const std::unordered_map<std::string, std::string>& args = {}) {
        TraceEvent event;
        event.name = name;
        event.category = "flow";
        event.phase = TracePhase::FLOW_BEGIN;
        event.timestamp_us = config_.cycle_to_timestamp(cycle);
        event.pid = pid_;
        event.tid = tid_;
        event.id = flow_id;
        event.args = args;
        events_.push_back(event);
    }

    void record_flow_end(const std::string& name,
                         const std::string& flow_id,
                         uint64_t cycle,
                         const std::unordered_map<std::string, std::string>& args = {}) {
        TraceEvent event;
        event.name = name;
        event.category = "flow";
        event.phase = TracePhase::FLOW_END;
        event.timestamp_us = config_.cycle_to_timestamp(cycle);
        event.pid = pid_;
        event.tid = tid_;
        event.id = flow_id;
        event.args = args;
        events_.push_back(event);
    }

    /// Record a counter value (for visualizing buffer occupancy, etc.)
    void record_counter(const std::string& name,
                        uint64_t cycle,
                        const std::unordered_map<std::string, int64_t>& values) {
        TraceEvent event;
        event.name = name;
        event.category = "counter";
        event.phase = TracePhase::COUNTER;
        event.timestamp_us = config_.cycle_to_timestamp(cycle);
        event.pid = pid_;
        event.tid = tid_;
        for (const auto& [key, value] : values) {
            event.args[key] = std::to_string(value);
        }
        events_.push_back(event);
    }

    // ========== Access ==========

    const std::vector<TraceEvent>& events() const { return events_; }
    std::vector<TraceEvent>& events() { return events_; }

    ExecutionLevel level() const { return level_; }
    uint32_t node_id() const { return node_id_; }
    uint32_t pid() const { return pid_; }
    uint32_t tid() const { return tid_; }
    const std::string& name() const { return name_; }

    // ========== Output ==========

    /// Write trace to file (standalone, not for consolidation)
    bool write_to_file(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        file << "{\"traceEvents\":[\n";
        bool first = true;
        for (const auto& event : events_) {
            if (!first) file << ",\n";
            first = false;
            file << event.to_json();
        }
        file << "\n],\"displayTimeUnit\":\"ns\"}\n";

        return true;
    }

    /// Get JSON array of events (for consolidation)
    std::string to_json_array() const {
        std::ostringstream ss;
        bool first = true;
        for (const auto& event : events_) {
            if (!first) ss << ",\n";
            first = false;
            ss << event.to_json();
        }
        return ss.str();
    }

private:
    void add_metadata_events() {
        // Process name
        TraceEvent proc_name;
        proc_name.name = "process_name";
        proc_name.category = "__metadata";
        proc_name.phase = TracePhase::METADATA;
        proc_name.timestamp_us = 0;
        proc_name.pid = pid_;
        proc_name.tid = 0;
        proc_name.args["name"] = level_to_process_name(level_);
        events_.push_back(proc_name);

        // Thread name
        TraceEvent thread_name;
        thread_name.name = "thread_name";
        thread_name.category = "__metadata";
        thread_name.phase = TracePhase::METADATA;
        thread_name.timestamp_us = 0;
        thread_name.pid = pid_;
        thread_name.tid = tid_;
        thread_name.args["name"] = name_.empty() ?
            node_to_thread_name(level_, node_id_) : name_;
        events_.push_back(thread_name);
    }

    void record_node_begin(const ExecutionEvent& event) {
        // Track start cycle for duration calculation when node completes
        node_start_cycles_[event.node_id] = event.cycle;
    }

    void record_node_end(const ExecutionEvent& event) {
        // Use COMPLETE (X) phase with duration instead of B/E pairs
        // This is more reliable in Chrome tracing
        TraceEvent te;
        te.name = "Node " + std::to_string(event.node_id);
        te.category = "node";
        te.phase = TracePhase::COMPLETE;

        // Get start cycle, default to current if not found
        uint64_t start_cycle = event.cycle;
        auto it = node_start_cycles_.find(event.node_id);
        if (it != node_start_cycles_.end()) {
            start_cycle = it->second;
            node_start_cycles_.erase(it);
        }

        te.timestamp_us = config_.cycle_to_timestamp(start_cycle);
        // Ensure minimum duration of 1us for visibility
        uint64_t duration = config_.cycle_to_timestamp(event.cycle) - te.timestamp_us;
        te.duration_us = duration > 0 ? duration : 1;
        te.pid = pid_;
        te.tid = tid_;
        te.args["node_id"] = std::to_string(event.node_id);
        events_.push_back(te);
    }

    void record_operand_ready(const ExecutionEvent& event) {
        if (!event.operand) return;

        TraceEvent te;
        te.name = "Operand Ready";
        te.category = "operand";
        te.phase = TracePhase::INSTANT;
        te.timestamp_us = config_.cycle_to_timestamp(event.cycle);
        te.pid = pid_;
        te.tid = tid_;
        te.scope = InstantScope::THREAD;
        te.args["operand"] = event.operand->to_string();
        te.args["type"] = to_string(event.operand->type);
        te.args["location"] = to_string(event.operand->location);
        te.args["coord"] = "(" + std::to_string(event.operand->coord.i) + "," +
                          std::to_string(event.operand->coord.j) + "," +
                          std::to_string(event.operand->coord.k) + ")";
        events_.push_back(te);
    }

    void record_operand_consumed(const ExecutionEvent& event) {
        if (!event.operand) return;

        TraceEvent te;
        te.name = "Operand Consumed";
        te.category = "operand";
        te.phase = TracePhase::INSTANT;
        te.timestamp_us = config_.cycle_to_timestamp(event.cycle);
        te.pid = pid_;
        te.tid = tid_;
        te.scope = InstantScope::THREAD;
        te.args["operand"] = event.operand->to_string();
        events_.push_back(te);
    }

    void record_buffer_available(const ExecutionEvent& event) {
        TraceEvent te;
        te.name = "Buffer Available";
        te.category = "buffer";
        te.phase = TracePhase::INSTANT;
        te.timestamp_us = config_.cycle_to_timestamp(event.cycle);
        te.pid = pid_;
        te.tid = tid_;
        te.scope = InstantScope::THREAD;
        if (event.operand) {
            te.args["location"] = to_string(event.operand->location);
            te.args["node_id"] = std::to_string(event.operand->node_id);
        }
        events_.push_back(te);
    }

    void record_execution_complete(const ExecutionEvent& event) {
        TraceEvent te;
        te.name = "Execution Complete";
        te.category = "milestone";
        te.phase = TracePhase::INSTANT;
        te.timestamp_us = config_.cycle_to_timestamp(event.cycle);
        te.pid = pid_;
        te.tid = tid_;
        te.scope = InstantScope::PROCESS;
        events_.push_back(te);
    }

    void record_deadlock(const ExecutionEvent& event) {
        TraceEvent te;
        te.name = "DEADLOCK";
        te.category = "error";
        te.phase = TracePhase::INSTANT;
        te.timestamp_us = config_.cycle_to_timestamp(event.cycle);
        te.pid = pid_;
        te.tid = tid_;
        te.scope = InstantScope::GLOBAL;
        te.args["message"] = event.message;
        events_.push_back(te);
    }

    std::string name_;
    ExecutionLevel level_;
    uint32_t node_id_;
    uint32_t pid_;
    uint32_t tid_;
    TraceConfig config_;
    std::vector<TraceEvent> events_;
    std::unordered_map<uint32_t, uint64_t> node_start_cycles_;  // For duration calc
};

// ============================================================================
// Trace Consolidator
// ============================================================================

/// Consolidates traces from multiple executors into a single Chrome Trace file
class TraceConsolidator {
public:
    explicit TraceConsolidator(const TraceConfig& config = {})
        : config_(config) {}

    /// Add a trace writer's events to the consolidated trace
    void add_trace(const ChromeTraceWriter& writer) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& event : writer.events()) {
            events_.push_back(event);
        }

        // Track unique pid/tid pairs for metadata
        auto key = std::make_pair(writer.pid(), writer.tid());
        if (threads_.find(key) == threads_.end()) {
            threads_[key] = {writer.level(), writer.node_id(), writer.name()};
        }
    }

    /// Add events directly (for custom event generation)
    void add_events(const std::vector<TraceEvent>& events) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& event : events) {
            events_.push_back(event);
        }
    }

    /// Add a flow connection between two trace writers (operand movement)
    void add_operand_flow(const std::string& operand_id,
                          const ChromeTraceWriter& producer,
                          uint64_t produce_cycle,
                          const ChromeTraceWriter& consumer,
                          uint64_t consume_cycle) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Flow begin at producer
        TraceEvent begin;
        begin.name = "Operand Flow";
        begin.category = "flow";
        begin.phase = TracePhase::FLOW_BEGIN;
        begin.timestamp_us = config_.cycle_to_timestamp(produce_cycle);
        begin.pid = producer.pid();
        begin.tid = producer.tid();
        begin.id = operand_id;
        begin.args["operand"] = operand_id;
        events_.push_back(begin);

        // Flow end at consumer
        TraceEvent end;
        end.name = "Operand Flow";
        end.category = "flow";
        end.phase = TracePhase::FLOW_END;
        end.timestamp_us = config_.cycle_to_timestamp(consume_cycle);
        end.pid = consumer.pid();
        end.tid = consumer.tid();
        end.id = operand_id;
        end.args["operand"] = operand_id;
        events_.push_back(end);
    }

    /// Clear all events
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        threads_.clear();
    }

    /// Get total event count
    size_t event_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

    /// Write consolidated trace to file
    bool write_to_file(const std::string& filename) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream file(filename);
        if (!file.is_open()) return false;

        // Sort events by timestamp for better visualization
        std::vector<TraceEvent> sorted_events = events_;
        std::sort(sorted_events.begin(), sorted_events.end(),
            [](const TraceEvent& a, const TraceEvent& b) {
                if (a.timestamp_us != b.timestamp_us) return a.timestamp_us < b.timestamp_us;
                // Metadata events first
                if (a.phase == TracePhase::METADATA && b.phase != TracePhase::METADATA) return true;
                if (a.phase != TracePhase::METADATA && b.phase == TracePhase::METADATA) return false;
                return false;
            });

        // Chrome trace format: just an array of events (simpler, more compatible)
        file << "[\n";

        bool first = true;
        for (const auto& event : sorted_events) {
            if (!first) file << ",\n";
            first = false;
            file << event.to_json();
        }

        file << "\n]\n";

        return true;
    }

    /// Generate summary statistics
    std::string summary() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream ss;
        ss << "Trace Consolidator Summary\n";
        ss << "==========================\n";
        ss << "Total events: " << events_.size() << "\n";
        ss << "Unique threads: " << threads_.size() << "\n";

        // Count events by category
        std::unordered_map<std::string, size_t> by_category;
        for (const auto& event : events_) {
            by_category[event.category]++;
        }

        ss << "Events by category:\n";
        for (const auto& [cat, count] : by_category) {
            ss << "  " << cat << ": " << count << "\n";
        }

        // Find time range
        if (!events_.empty()) {
            uint64_t min_ts = UINT64_MAX, max_ts = 0;
            for (const auto& event : events_) {
                if (event.phase != TracePhase::METADATA) {
                    min_ts = std::min(min_ts, event.timestamp_us);
                    max_ts = std::max(max_ts, event.timestamp_us);
                }
            }
            ss << "Time range: " << min_ts << " - " << max_ts << " us\n";
            ss << "Duration: " << (max_ts - min_ts) << " us\n";
        }

        return ss.str();
    }

private:
    struct ThreadInfo {
        ExecutionLevel level;
        uint32_t node_id;
        std::string name;
    };

    TraceConfig config_;
    mutable std::mutex mutex_;
    std::vector<TraceEvent> events_;
    std::map<std::pair<uint32_t, uint32_t>, ThreadInfo> threads_;
};

// ============================================================================
// Tracing Mixin for FlowGraphExecutor
// ============================================================================

/// Mixin class that adds Chrome Trace support to any FlowGraphExecutor
template<typename BaseExecutor>
class TracingExecutor : public BaseExecutor {
public:
    template<typename... Args>
    TracingExecutor(const std::string& name,
                    ExecutionLevel level,
                    uint32_t node_id,
                    Args&&... args)
        : BaseExecutor(std::forward<Args>(args)...)
        , trace_writer_(name, level, node_id) {
        // Install event callback to capture all events
        this->set_event_callback([this](const ExecutionEvent& event) {
            trace_writer_.record_event(event);
        });
    }

    /// Get the trace writer for this executor
    ChromeTraceWriter& trace_writer() { return trace_writer_; }
    const ChromeTraceWriter& trace_writer() const { return trace_writer_; }

    /// Write trace to file
    bool write_trace(const std::string& filename) const {
        return trace_writer_.write_to_file(filename);
    }

    /// Add custom duration event
    void trace_duration(const std::string& name,
                        const std::string& category,
                        uint64_t start_cycle,
                        uint64_t end_cycle) {
        trace_writer_.record_duration(name, category, start_cycle, end_cycle);
    }

    /// Add custom instant event
    void trace_instant(const std::string& name,
                       const std::string& category,
                       uint64_t cycle) {
        trace_writer_.record_instant(name, category, cycle);
    }

private:
    ChromeTraceWriter trace_writer_;
};

// ============================================================================
// Convenience factory functions
// ============================================================================

/// Create a trace writer for a DMA executor
inline ChromeTraceWriter make_dma_trace_writer(uint32_t l3_tile_id,
                                                const TraceConfig& config = {}) {
    return ChromeTraceWriter("L3 Tile " + std::to_string(l3_tile_id),
                             ExecutionLevel::DMA, l3_tile_id, config);
}

/// Create a trace writer for a BlockMover executor
inline ChromeTraceWriter make_block_mover_trace_writer(uint32_t row, uint32_t col,
                                                        const TraceConfig& config = {}) {
    uint32_t node_id = row * 4 + col;  // Assuming 4-wide mesh
    return ChromeTraceWriter("Tile[" + std::to_string(row) + "," +
                             std::to_string(col) + "]",
                             ExecutionLevel::BLOCK_MOVER, node_id, config);
}

/// Create a trace writer for a Streamer executor
inline ChromeTraceWriter make_streamer_trace_writer(uint32_t l2_bank_id,
                                                     const TraceConfig& config = {}) {
    return ChromeTraceWriter("L2 Bank " + std::to_string(l2_bank_id),
                             ExecutionLevel::STREAMER, l2_bank_id, config);
}

/// Create a trace writer for a Compute executor
inline ChromeTraceWriter make_compute_trace_writer(uint32_t pe_row, uint32_t pe_col,
                                                    const TraceConfig& config = {}) {
    uint32_t node_id = pe_row * 16 + pe_col;  // Assuming 16-wide array
    return ChromeTraceWriter("PE[" + std::to_string(pe_row) + "," +
                             std::to_string(pe_col) + "]",
                             ExecutionLevel::COMPUTE, node_id, config);
}

} // namespace sw::kpu::dataflow
