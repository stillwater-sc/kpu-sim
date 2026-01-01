// ============================================================================
// include/sw/benchmark/noc_benchmark_tracer.hpp
// NoC Benchmark Tracing for Chrome Trace / Perfetto Visualization
// ============================================================================

#pragma once

#include <sw/kpu/noc/noc_interface.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw::benchmark {

using namespace sw::kpu::noc;

// ============================================================================
// Router State
// ============================================================================

enum class RouterState : uint8_t {
    IDLE = 0,
    ROUTING,
    BLOCKED,
    DRAINING
};

inline const char* to_string(RouterState state) {
    switch (state) {
        case RouterState::IDLE: return "IDLE";
        case RouterState::ROUTING: return "ROUTING";
        case RouterState::BLOCKED: return "BLOCKED";
        case RouterState::DRAINING: return "DRAINING";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Port Direction (for hop tracking)
// Use the PortDir from wormhole_router.hpp
// ============================================================================

// Include wormhole_router for PortDir definition
#include <sw/kpu/noc/wormhole_router.hpp>

// Bring PortDir into benchmark namespace for convenience
using sw::kpu::noc::PortDir;

// Provide to_string wrapper (wormhole_router has to_cstring)
inline const char* to_string(PortDir dir) {
    return sw::kpu::noc::to_cstring(dir);
}

// ============================================================================
// Tracer Configuration
// ============================================================================

/// Configuration for benchmark tracing
struct TracerConfig {
    bool enabled = true;
    bool trace_packets = true;        // Packet inject/deliver with flow arrows
    bool trace_hops = true;           // Per-hop routing events
    bool trace_router_state = true;   // Router activity
    bool trace_link_usage = true;     // Link occupancy
    bool trace_counters = true;       // Bandwidth/backpressure counters
    uint64_t counter_interval = 10;   // Cycles between counter samples
    size_t max_events = 10000000;     // Memory limit
};

// ============================================================================
// NoCBenchmarkTracer
// ============================================================================

/// Main tracer class for NoC benchmark visualization
///
/// Records NoC events during benchmark runs and exports to Chrome Trace
/// format for visualization in chrome://tracing or Perfetto.
///
/// Key features:
/// - Flow events showing packet injection to delivery (arrows in Perfetto)
/// - Per-router activity visualization
/// - Link utilization heatmaps
/// - Bandwidth/backpressure counter time series
class NoCBenchmarkTracer {
public:
    explicit NoCBenchmarkTracer(const TracerConfig& config = {})
        : config_(config)
    {
        packets_.reserve(10000);
        hops_.reserve(100000);
        link_transfers_.reserve(100000);
        counter_samples_.reserve(10000);
    }

    // ========== Event Recording ==========

    /// Record packet injection (returns packet_id for correlation)
    uint64_t record_inject(uint8_t src_router, uint8_t dst_router,
                           const TileDescriptor& tile, uint64_t cycle) {
        if (!config_.enabled || !config_.trace_packets) return 0;
        if (packets_.size() >= config_.max_events) return 0;

        uint64_t packet_id = next_packet_id_++;
        PacketInfo info;
        info.tile = tile;
        info.src_router = src_router;
        info.dst_router = dst_router;
        info.inject_cycle = cycle;
        info.deliver_cycle = 0;

        packets_[packet_id] = info;
        return packet_id;
    }

    /// Record packet delivery (uses packet_id from inject)
    void record_deliver(uint64_t packet_id, uint64_t cycle) {
        if (!config_.enabled || !config_.trace_packets) return;
        if (packet_id == 0) return;

        auto it = packets_.find(packet_id);
        if (it != packets_.end()) {
            it->second.deliver_cycle = cycle;
        }
    }

    /// Record a single hop through a router
    void record_hop(uint64_t packet_id, uint8_t router_id,
                    PortDir in_port, PortDir out_port,
                    uint64_t enter_cycle, uint64_t exit_cycle) {
        if (!config_.enabled || !config_.trace_hops) return;
        if (hops_.size() >= config_.max_events) return;

        HopInfo hop;
        hop.packet_id = packet_id;
        hop.router_id = router_id;
        hop.in_port = in_port;
        hop.out_port = out_port;
        hop.enter_cycle = enter_cycle;
        hop.exit_cycle = exit_cycle;
        hops_.push_back(hop);
    }

    /// Record router state change
    void record_router_state(uint8_t router_id, RouterState state,
                             uint64_t cycle, const std::string& reason = "") {
        if (!config_.enabled || !config_.trace_router_state) return;
        if (router_states_.size() >= config_.max_events) return;

        RouterStateInfo info;
        info.router_id = router_id;
        info.state = state;
        info.cycle = cycle;
        info.reason = reason;
        router_states_.push_back(info);
    }

    /// Record link transfer
    void record_link_transfer(uint8_t src_router, uint8_t dst_router,
                              PortDir direction, uint64_t start_cycle,
                              uint64_t end_cycle, uint32_t bytes) {
        if (!config_.enabled || !config_.trace_link_usage) return;
        if (link_transfers_.size() >= config_.max_events) return;

        LinkInfo info;
        info.src_router = src_router;
        info.dst_router = dst_router;
        info.direction = direction;
        info.start_cycle = start_cycle;
        info.end_cycle = end_cycle;
        info.bytes = bytes;
        link_transfers_.push_back(info);
    }

    /// Record counter sample
    void record_counters(uint64_t cycle, double bandwidth,
                         uint32_t blocked_routers, uint32_t active_flits) {
        if (!config_.enabled || !config_.trace_counters) return;
        if (counter_samples_.size() >= config_.max_events) return;

        CounterSample sample;
        sample.cycle = cycle;
        sample.bandwidth = bandwidth;
        sample.blocked_routers = blocked_routers;
        sample.active_flits = active_flits;
        counter_samples_.push_back(sample);
    }

    // ========== Export ==========

    /// Export to Chrome Trace JSON format
    bool export_chrome_trace(const std::string& filename,
                             uint8_t mesh_rows, uint8_t mesh_cols) const {
        std::ofstream file(filename);
        if (!file) return false;

        file << "{\"traceEvents\":[\n";

        bool first = true;

        // ========== Process Metadata ==========

        // Process 1: Routers
        emit_metadata(file, first, 1, "NoC Routers");

        // Process 2: East Links
        emit_metadata(file, first, 2, "East Links (\u2192)");

        // Process 3: South Links
        emit_metadata(file, first, 3, "South Links (\u2193)");

        // Process 4: Packets (for flow events)
        emit_metadata(file, first, 4, "Packet Flow");

        // Process 5: Counters
        emit_metadata(file, first, 5, "Counters");

        // ========== Thread Metadata ==========

        // Router threads
        for (uint8_t r = 0; r < mesh_rows; r++) {
            for (uint8_t c = 0; c < mesh_cols; c++) {
                uint8_t id = r * mesh_cols + c;
                std::ostringstream name;
                name << "Router[" << static_cast<int>(r) << "," << static_cast<int>(c) << "]";
                emit_thread_name(file, first, 1, id, name.str());
            }
        }

        // East link threads
        for (uint8_t r = 0; r < mesh_rows; r++) {
            for (uint8_t c = 0; c < mesh_cols - 1; c++) {
                int tid = r * (mesh_cols - 1) + c;
                std::ostringstream name;
                name << "[" << static_cast<int>(r) << "," << static_cast<int>(c)
                     << "]\u2192[" << static_cast<int>(r) << "," << static_cast<int>(c+1) << "]";
                emit_thread_name(file, first, 2, tid, name.str());
            }
        }

        // South link threads
        for (uint8_t r = 0; r < mesh_rows - 1; r++) {
            for (uint8_t c = 0; c < mesh_cols; c++) {
                int tid = r * mesh_cols + c;
                std::ostringstream name;
                name << "[" << static_cast<int>(r) << "," << static_cast<int>(c)
                     << "]\u2193[" << static_cast<int>(r+1) << "," << static_cast<int>(c) << "]";
                emit_thread_name(file, first, 3, tid, name.str());
            }
        }

        // Packet flow threads
        emit_thread_name(file, first, 4, 0, "Injected");
        emit_thread_name(file, first, 4, 1, "Delivered");

        // Counter threads
        emit_thread_name(file, first, 5, 0, "Bandwidth (B/cycle)");
        emit_thread_name(file, first, 5, 1, "Blocked Routers");
        emit_thread_name(file, first, 5, 2, "Active Flits");

        // ========== Packet Flow Events ==========

        for (const auto& [packet_id, info] : packets_) {
            if (info.deliver_cycle == 0) continue;  // Not delivered yet

            std::string tile_name = format_tile_name(info.tile);

            // Flow start (injection)
            emit_flow_start(file, first, packet_id, tile_name,
                            info.src_router, info.dst_router,
                            info.inject_cycle, info.tile.size, mesh_cols);

            // Flow end (delivery)
            emit_flow_end(file, first, packet_id, tile_name,
                          info.src_router, info.dst_router,
                          info.deliver_cycle,
                          info.deliver_cycle - info.inject_cycle, mesh_cols);
        }

        // ========== Hop Events ==========

        for (const auto& hop : hops_) {
            emit_hop_event(file, first, hop, mesh_cols);
        }

        // ========== Router State Events ==========

        for (const auto& state : router_states_) {
            emit_router_state_event(file, first, state);
        }

        // ========== Link Transfer Events ==========

        for (const auto& link : link_transfers_) {
            emit_link_event(file, first, link, mesh_cols);
        }

        // ========== Counter Events ==========

        for (const auto& sample : counter_samples_) {
            emit_counter_events(file, first, sample);
        }

        file << "\n]}\n";
        file.close();
        return true;
    }

    /// Export with NoCConfig
    bool export_chrome_trace(const std::string& filename,
                             const NoCConfig& config) const {
        return export_chrome_trace(filename, config.rows, config.cols);
    }

    // ========== Statistics ==========

    size_t num_packets() const { return packets_.size(); }
    size_t num_hops() const { return hops_.size(); }
    size_t num_link_events() const { return link_transfers_.size(); }
    size_t num_counter_samples() const { return counter_samples_.size(); }

    size_t num_events() const {
        return packets_.size() * 2 + hops_.size() +
               router_states_.size() + link_transfers_.size() +
               counter_samples_.size() * 3;
    }

    uint64_t total_hops() const { return hops_.size(); }

    void clear() {
        packets_.clear();
        hops_.clear();
        router_states_.clear();
        link_transfers_.clear();
        counter_samples_.clear();
        next_packet_id_ = 1;
    }

    void enable() { config_.enabled = true; }
    void disable() { config_.enabled = false; }
    bool is_enabled() const { return config_.enabled; }

    const TracerConfig& config() const { return config_; }

private:
    TracerConfig config_;

    struct PacketInfo {
        TileDescriptor tile;
        uint8_t src_router = 0;
        uint8_t dst_router = 0;
        uint64_t inject_cycle = 0;
        uint64_t deliver_cycle = 0;
    };
    std::unordered_map<uint64_t, PacketInfo> packets_;

    struct HopInfo {
        uint64_t packet_id = 0;
        uint8_t router_id = 0;
        PortDir in_port = PortDir::LOCAL;
        PortDir out_port = PortDir::LOCAL;
        uint64_t enter_cycle = 0;
        uint64_t exit_cycle = 0;
    };
    std::vector<HopInfo> hops_;

    struct RouterStateInfo {
        uint8_t router_id = 0;
        RouterState state = RouterState::IDLE;
        uint64_t cycle = 0;
        std::string reason;
    };
    std::vector<RouterStateInfo> router_states_;

    struct LinkInfo {
        uint8_t src_router = 0;
        uint8_t dst_router = 0;
        PortDir direction = PortDir::LOCAL;
        uint64_t start_cycle = 0;
        uint64_t end_cycle = 0;
        uint32_t bytes = 0;
    };
    std::vector<LinkInfo> link_transfers_;

    struct CounterSample {
        uint64_t cycle = 0;
        double bandwidth = 0.0;
        uint32_t blocked_routers = 0;
        uint32_t active_flits = 0;
    };
    std::vector<CounterSample> counter_samples_;

    uint64_t next_packet_id_ = 1;

    // ========== Helper Methods ==========

    static std::string format_tile_name(const TileDescriptor& tile) {
        std::ostringstream oss;
        switch (tile.tensor) {
            case TensorId::A:
                oss << "A[" << tile.m_tile << "," << tile.k_tile << "]";
                break;
            case TensorId::B:
                oss << "B[" << tile.k_tile << "," << tile.n_tile << "]";
                break;
            case TensorId::C:
                oss << "C[" << tile.m_tile << "," << tile.n_tile << "]";
                break;
            default:
                oss << "T[" << tile.m_tile << "," << tile.n_tile << "]";
        }
        return oss.str();
    }

    static std::string format_router_id(uint8_t router_id, uint8_t mesh_cols) {
        uint8_t row = router_id / mesh_cols;
        uint8_t col = router_id % mesh_cols;
        std::ostringstream oss;
        oss << "R[" << static_cast<int>(row) << "," << static_cast<int>(col) << "]";
        return oss.str();
    }

    // Emit process metadata
    static void emit_metadata(std::ostream& out, bool& first, int pid,
                               const std::string& name) {
        if (!first) out << ",\n";
        first = false;
        out << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << pid
            << ",\"args\":{\"name\":\"" << name << "\"}}";
    }

    // Emit thread name metadata
    static void emit_thread_name(std::ostream& out, bool& first, int pid, int tid,
                                  const std::string& name) {
        if (!first) out << ",\n";
        first = false;
        out << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":" << pid
            << ",\"tid\":" << tid << ",\"args\":{\"name\":\"" << name << "\"}}";
    }

    // Emit flow start event
    static void emit_flow_start(std::ostream& out, bool& first,
                                 uint64_t packet_id, const std::string& tile_name,
                                 uint8_t src_router, uint8_t dst_router,
                                 uint64_t cycle, uint32_t size, uint8_t mesh_cols) {
        if (!first) out << ",\n";
        first = false;

        out << "{\"name\":\"" << tile_name << "\",\"cat\":\"packet\",\"ph\":\"s\","
            << "\"id\":" << packet_id << ",\"ts\":" << cycle
            << ",\"pid\":1,\"tid\":" << static_cast<int>(src_router)
            << ",\"args\":{\"src\":\"" << format_router_id(src_router, mesh_cols)
            << "\",\"dst\":\"" << format_router_id(dst_router, mesh_cols)
            << "\",\"size\":" << size << "}}";
    }

    // Emit flow end event
    static void emit_flow_end(std::ostream& out, bool& first,
                               uint64_t packet_id, const std::string& tile_name,
                               uint8_t src_router, uint8_t dst_router,
                               uint64_t cycle, uint64_t latency, uint8_t mesh_cols) {
        if (!first) out << ",\n";
        first = false;

        out << "{\"name\":\"" << tile_name << "\",\"cat\":\"packet\",\"ph\":\"f\","
            << "\"id\":" << packet_id << ",\"ts\":" << cycle << ",\"bp\":\"e\""
            << ",\"pid\":1,\"tid\":" << static_cast<int>(dst_router)
            << ",\"args\":{\"src\":\"" << format_router_id(src_router, mesh_cols)
            << "\",\"dst\":\"" << format_router_id(dst_router, mesh_cols)
            << "\",\"latency\":" << latency << "}}";
    }

    // Emit hop event (complete event on router)
    void emit_hop_event(std::ostream& out, bool& first,
                        const HopInfo& hop, uint8_t mesh_cols) const {
        if (!first) out << ",\n";
        first = false;

        uint64_t duration = (hop.exit_cycle > hop.enter_cycle) ?
                            (hop.exit_cycle - hop.enter_cycle) : 1;

        out << "{\"name\":\"HOP\",\"cat\":\"routing\",\"ph\":\"X\","
            << "\"ts\":" << hop.enter_cycle << ",\"dur\":" << duration
            << ",\"pid\":1,\"tid\":" << static_cast<int>(hop.router_id)
            << ",\"args\":{\"packet\":" << hop.packet_id
            << ",\"in\":\"" << to_string(hop.in_port)
            << "\",\"out\":\"" << to_string(hop.out_port) << "\"}}";
    }

    // Emit router state event
    void emit_router_state_event(std::ostream& out, bool& first,
                                  const RouterStateInfo& state) const {
        if (!first) out << ",\n";
        first = false;

        out << "{\"name\":\"" << to_string(state.state)
            << "\",\"cat\":\"router_state\",\"ph\":\"i\","
            << "\"ts\":" << state.cycle << ",\"s\":\"t\""
            << ",\"pid\":1,\"tid\":" << static_cast<int>(state.router_id);

        if (!state.reason.empty()) {
            out << ",\"args\":{\"reason\":\"" << state.reason << "\"}";
        }
        out << "}";
    }

    // Emit link transfer event
    void emit_link_event(std::ostream& out, bool& first,
                          const LinkInfo& link, uint8_t mesh_cols) const {
        if (!first) out << ",\n";
        first = false;

        uint64_t duration = (link.end_cycle > link.start_cycle) ?
                            (link.end_cycle - link.start_cycle) : 1;

        // Determine process and thread based on direction
        int pid = 2;  // East links by default
        int tid = 0;
        uint8_t src_row = link.src_router / mesh_cols;
        uint8_t src_col = link.src_router % mesh_cols;

        if (link.direction == PortDir::EAST) {
            pid = 2;
            tid = src_row * (mesh_cols - 1) + src_col;
        } else if (link.direction == PortDir::SOUTH) {
            pid = 3;
            tid = src_row * mesh_cols + src_col;
        }

        double bw = (duration > 0) ? static_cast<double>(link.bytes) / duration : 0.0;

        out << "{\"name\":\"XFER\",\"cat\":\"link\",\"ph\":\"X\","
            << "\"ts\":" << link.start_cycle << ",\"dur\":" << duration
            << ",\"pid\":" << pid << ",\"tid\":" << tid
            << ",\"args\":{\"bytes\":" << link.bytes
            << ",\"bw\":" << std::fixed << std::setprecision(1) << bw << "}}";
    }

    // Emit counter events
    void emit_counter_events(std::ostream& out, bool& first,
                              const CounterSample& sample) const {
        // Bandwidth counter
        if (!first) out << ",\n";
        first = false;
        out << "{\"name\":\"bandwidth\",\"cat\":\"counter\",\"ph\":\"C\","
            << "\"ts\":" << sample.cycle << ",\"pid\":5,\"tid\":0,"
            << "\"args\":{\"bytes_per_cycle\":"
            << std::fixed << std::setprecision(2) << sample.bandwidth << "}}";

        // Blocked routers counter
        out << ",\n{\"name\":\"blocked\",\"cat\":\"counter\",\"ph\":\"C\","
            << "\"ts\":" << sample.cycle << ",\"pid\":5,\"tid\":1,"
            << "\"args\":{\"count\":" << sample.blocked_routers << "}}";

        // Active flits counter
        out << ",\n{\"name\":\"active_flits\",\"cat\":\"counter\",\"ph\":\"C\","
            << "\"ts\":" << sample.cycle << ",\"pid\":5,\"tid\":2,"
            << "\"args\":{\"count\":" << sample.active_flits << "}}";
    }
};

} // namespace sw::benchmark
