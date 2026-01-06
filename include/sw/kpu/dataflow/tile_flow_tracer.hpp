// ============================================================================
// include/sw/kpu/dataflow/tile_flow_tracer.hpp
// Tile Flow Tracer - Records tile movement events for visualization
// ============================================================================

#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "sw/kpu/components/block_mover_isa.hpp"

namespace sw::kpu::dataflow {

// ============================================================================
// Trace Event Types
// ============================================================================

/// Type of tile flow event
enum class TileEventType : uint8_t {
    // Memory Controller events (DRAM timing)
    MC_BANK_ACTIVATE,       // DRAM bank row activation (ACT command)
    MC_BANK_PRECHARGE,      // DRAM bank precharge (PRE command)
    MC_PAGE_HIT,            // Access to already-open row (fast)
    MC_PAGE_CONFLICT,       // Access to different row (requires precharge+activate)
    MC_READ_ISSUE,          // Memory controller issues read command
    MC_READ_COMPLETE,       // Data arrives from DRAM
    MC_WRITE_ISSUE,         // Memory controller issues write command
    MC_WRITE_COMPLETE,      // Write completes to DRAM
    MC_REFRESH_START,       // DRAM refresh begins
    MC_REFRESH_COMPLETE,    // DRAM refresh completes
    MC_QUEUE_FULL,          // Memory controller queue full (stall)

    // DMA events (External memory <-> L3)
    DMA_REQUEST,            // DMA request queued
    DMA_LOAD_START,         // Tile load from external memory begins
    DMA_LOAD_COMPLETE,      // Tile load from external memory completes
    DMA_STORE_START,        // Tile store to external memory begins
    DMA_STORE_COMPLETE,     // Tile store to external memory completes
    DMA_STALL,              // DMA stalled waiting for memory controller
    DMA_CHANNEL_IDLE,       // DMA channel becomes idle

    // L3-to-L3 transfer events
    L3_SEND_START,          // Tile transfer to neighbor L3 begins
    L3_SEND_COMPLETE,       // Tile transfer to neighbor L3 completes
    L3_RECEIVE,             // Tile received from neighbor L3

    // L3-to-L2 transfer events
    L3_TO_L2_START,         // Tile push to L2 begins
    L3_TO_L2_COMPLETE,      // Tile push to L2 completes
    L2_TO_L3_START,         // Tile pull from L2 begins
    L2_TO_L3_COMPLETE,      // Tile pull from L2 completes

    // Compute events
    COMPUTE_START,          // Matmul computation begins
    COMPUTE_COMPLETE,       // Matmul computation completes

    // Synchronization events
    BARRIER_START,          // BlockMover hits barrier
    BARRIER_COMPLETE,       // BlockMover exits barrier
    TRIGGER_SEND,           // Trigger sent to other BlockMovers
    TRIGGER_RECEIVE,        // Trigger received

    // State changes
    MOVER_IDLE,             // BlockMover becomes idle
    MOVER_RUNNING,          // BlockMover starts running
    MOVER_WAITING,          // BlockMover starts waiting
};

inline const char* to_string(TileEventType type) {
    switch (type) {
        // Memory Controller events
        case TileEventType::MC_BANK_ACTIVATE: return "MC_BANK_ACTIVATE";
        case TileEventType::MC_BANK_PRECHARGE: return "MC_BANK_PRECHARGE";
        case TileEventType::MC_PAGE_HIT: return "MC_PAGE_HIT";
        case TileEventType::MC_PAGE_CONFLICT: return "MC_PAGE_CONFLICT";
        case TileEventType::MC_READ_ISSUE: return "MC_READ_ISSUE";
        case TileEventType::MC_READ_COMPLETE: return "MC_READ_COMPLETE";
        case TileEventType::MC_WRITE_ISSUE: return "MC_WRITE_ISSUE";
        case TileEventType::MC_WRITE_COMPLETE: return "MC_WRITE_COMPLETE";
        case TileEventType::MC_REFRESH_START: return "MC_REFRESH_START";
        case TileEventType::MC_REFRESH_COMPLETE: return "MC_REFRESH_COMPLETE";
        case TileEventType::MC_QUEUE_FULL: return "MC_QUEUE_FULL";
        // DMA events
        case TileEventType::DMA_REQUEST: return "DMA_REQUEST";
        case TileEventType::DMA_LOAD_START: return "DMA_LOAD_START";
        case TileEventType::DMA_LOAD_COMPLETE: return "DMA_LOAD_COMPLETE";
        case TileEventType::DMA_STORE_START: return "DMA_STORE_START";
        case TileEventType::DMA_STORE_COMPLETE: return "DMA_STORE_COMPLETE";
        case TileEventType::DMA_STALL: return "DMA_STALL";
        case TileEventType::DMA_CHANNEL_IDLE: return "DMA_CHANNEL_IDLE";
        // L3-to-L3 events
        case TileEventType::L3_SEND_START: return "L3_SEND_START";
        case TileEventType::L3_SEND_COMPLETE: return "L3_SEND_COMPLETE";
        case TileEventType::L3_RECEIVE: return "L3_RECEIVE";
        // L3-to-L2 events
        case TileEventType::L3_TO_L2_START: return "L3_TO_L2_START";
        case TileEventType::L3_TO_L2_COMPLETE: return "L3_TO_L2_COMPLETE";
        case TileEventType::L2_TO_L3_START: return "L2_TO_L3_START";
        case TileEventType::L2_TO_L3_COMPLETE: return "L2_TO_L3_COMPLETE";
        // Compute events
        case TileEventType::COMPUTE_START: return "COMPUTE_START";
        case TileEventType::COMPUTE_COMPLETE: return "COMPUTE_COMPLETE";
        // Sync events
        case TileEventType::BARRIER_START: return "BARRIER_START";
        case TileEventType::BARRIER_COMPLETE: return "BARRIER_COMPLETE";
        case TileEventType::TRIGGER_SEND: return "TRIGGER_SEND";
        case TileEventType::TRIGGER_RECEIVE: return "TRIGGER_RECEIVE";
        // State events
        case TileEventType::MOVER_IDLE: return "MOVER_IDLE";
        case TileEventType::MOVER_RUNNING: return "MOVER_RUNNING";
        case TileEventType::MOVER_WAITING: return "MOVER_WAITING";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Trace Event
// ============================================================================

/// A single tile flow event
struct TileFlowEvent {
    uint64_t cycle = 0;             // Simulation cycle when event occurred
    uint8_t l3_id = 0;              // L3 tile where event occurred
    TileEventType type;             // Type of event
    TileDescriptor tile;            // Tile involved (if applicable)

    // Additional context
    uint8_t src_l3_id = 0;          // Source L3 for transfers
    uint8_t dst_l3_id = 0;          // Destination L3 for transfers
    uint8_t trigger_channel = 0;    // Trigger channel (for trigger events)
    uint32_t bytes = 0;             // Bytes transferred

    // Memory controller specific fields
    uint8_t bank_id = 0;            // DRAM bank ID
    uint32_t row = 0;               // DRAM row address
    uint8_t dma_channel = 0;        // DMA channel ID
    uint64_t address = 0;           // Memory address

    std::string to_csv() const {
        std::ostringstream oss;
        oss << cycle << ","
            << static_cast<int>(l3_id) << ","
            << to_string(type) << ","
            << static_cast<int>(tile.tensor) << ","
            << tile.m_tile << ","
            << tile.n_tile << ","
            << tile.k_tile << ","
            << static_cast<int>(src_l3_id) << ","
            << static_cast<int>(dst_l3_id) << ","
            << bytes << ","
            << static_cast<int>(bank_id) << ","
            << row << ","
            << static_cast<int>(dma_channel) << ","
            << address;
        return oss.str();
    }

    static std::string csv_header() {
        return "cycle,l3_id,event_type,tensor,m_tile,n_tile,k_tile,src_l3,dst_l3,bytes,bank_id,row,dma_channel,address";
    }
};

// ============================================================================
// Tile Flow Tracer
// ============================================================================

/// Configuration for TileFlowTracer
struct TileFlowTracerConfig {
    bool enabled = true;
    size_t max_events = 1000000;    // Limit to avoid memory issues
    bool trace_memory_controller = true;  // Memory controller DRAM timing events
    bool trace_dma = true;
    bool trace_l3_transfers = true;
    bool trace_l2_transfers = true;
    bool trace_compute = true;
    bool trace_sync = true;
    bool trace_state = true;
};

/// Records tile movement events during simulation for later visualization
class TileFlowTracer {
public:
    using Config = TileFlowTracerConfig;

    TileFlowTracer() : config_() {
        events_.reserve(10000);
    }

    explicit TileFlowTracer(const Config& config)
        : config_(config)
    {
        events_.reserve(10000);  // Pre-allocate for common case
    }

    // ========== Recording Events ==========

    void record(const TileFlowEvent& event) {
        if (!config_.enabled) return;
        if (events_.size() >= config_.max_events) return;

        // Filter by event type
        switch (event.type) {
            // Memory Controller events
            case TileEventType::MC_BANK_ACTIVATE:
            case TileEventType::MC_BANK_PRECHARGE:
            case TileEventType::MC_PAGE_HIT:
            case TileEventType::MC_PAGE_CONFLICT:
            case TileEventType::MC_READ_ISSUE:
            case TileEventType::MC_READ_COMPLETE:
            case TileEventType::MC_WRITE_ISSUE:
            case TileEventType::MC_WRITE_COMPLETE:
            case TileEventType::MC_REFRESH_START:
            case TileEventType::MC_REFRESH_COMPLETE:
            case TileEventType::MC_QUEUE_FULL:
                if (!config_.trace_memory_controller) return;
                break;

            // DMA events
            case TileEventType::DMA_REQUEST:
            case TileEventType::DMA_LOAD_START:
            case TileEventType::DMA_LOAD_COMPLETE:
            case TileEventType::DMA_STORE_START:
            case TileEventType::DMA_STORE_COMPLETE:
            case TileEventType::DMA_STALL:
            case TileEventType::DMA_CHANNEL_IDLE:
                if (!config_.trace_dma) return;
                break;

            case TileEventType::L3_SEND_START:
            case TileEventType::L3_SEND_COMPLETE:
            case TileEventType::L3_RECEIVE:
                if (!config_.trace_l3_transfers) return;
                break;

            case TileEventType::L3_TO_L2_START:
            case TileEventType::L3_TO_L2_COMPLETE:
            case TileEventType::L2_TO_L3_START:
            case TileEventType::L2_TO_L3_COMPLETE:
                if (!config_.trace_l2_transfers) return;
                break;

            case TileEventType::COMPUTE_START:
            case TileEventType::COMPUTE_COMPLETE:
                if (!config_.trace_compute) return;
                break;

            case TileEventType::BARRIER_START:
            case TileEventType::BARRIER_COMPLETE:
            case TileEventType::TRIGGER_SEND:
            case TileEventType::TRIGGER_RECEIVE:
                if (!config_.trace_sync) return;
                break;

            case TileEventType::MOVER_IDLE:
            case TileEventType::MOVER_RUNNING:
            case TileEventType::MOVER_WAITING:
                if (!config_.trace_state) return;
                break;
        }

        events_.push_back(event);
    }

    // ========== Memory Controller Events ==========

    void record_mc_bank_activate(uint64_t cycle, uint8_t bank_id, uint32_t row) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::MC_BANK_ACTIVATE;
        e.bank_id = bank_id;
        e.row = row;
        record(e);
    }

    void record_mc_bank_precharge(uint64_t cycle, uint8_t bank_id) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::MC_BANK_PRECHARGE;
        e.bank_id = bank_id;
        record(e);
    }

    void record_mc_page_hit(uint64_t cycle, uint8_t bank_id, uint32_t row) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::MC_PAGE_HIT;
        e.bank_id = bank_id;
        e.row = row;
        record(e);
    }

    void record_mc_page_conflict(uint64_t cycle, uint8_t bank_id, uint32_t old_row [[maybe_unused]], uint32_t new_row) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::MC_PAGE_CONFLICT;
        e.bank_id = bank_id;
        e.row = new_row;
        record(e);
    }

    void record_mc_read(uint64_t cycle, uint8_t bank_id, uint64_t address, uint32_t bytes, bool complete) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = complete ? TileEventType::MC_READ_COMPLETE : TileEventType::MC_READ_ISSUE;
        e.bank_id = bank_id;
        e.address = address;
        e.bytes = bytes;
        record(e);
    }

    void record_mc_write(uint64_t cycle, uint8_t bank_id, uint64_t address, uint32_t bytes, bool complete) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = complete ? TileEventType::MC_WRITE_COMPLETE : TileEventType::MC_WRITE_ISSUE;
        e.bank_id = bank_id;
        e.address = address;
        e.bytes = bytes;
        record(e);
    }

    void record_mc_refresh(uint64_t cycle, bool start) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = start ? TileEventType::MC_REFRESH_START : TileEventType::MC_REFRESH_COMPLETE;
        record(e);
    }

    void record_mc_queue_full(uint64_t cycle) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::MC_QUEUE_FULL;
        record(e);
    }

    // ========== DMA Events ==========

    void record_dma_request(uint64_t cycle, uint8_t dma_channel, uint8_t l3_id,
                            const TileDescriptor& tile, uint64_t address) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::DMA_REQUEST;
        e.dma_channel = dma_channel;
        e.l3_id = l3_id;
        e.tile = tile;
        e.address = address;
        e.bytes = tile.size;
        record(e);
    }

    void record_dma_stall(uint64_t cycle, uint8_t dma_channel) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::DMA_STALL;
        e.dma_channel = dma_channel;
        record(e);
    }

    void record_dma_channel_idle(uint64_t cycle, uint8_t dma_channel) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.type = TileEventType::DMA_CHANNEL_IDLE;
        e.dma_channel = dma_channel;
        record(e);
    }

    // Convenience methods for common events
    void record_dma_load_start(uint64_t cycle, uint8_t l3_id, const TileDescriptor& tile) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = TileEventType::DMA_LOAD_START;
        e.tile = tile;
        e.bytes = tile.size;
        record(e);
    }

    void record_dma_load_complete(uint64_t cycle, uint8_t l3_id, const TileDescriptor& tile) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = TileEventType::DMA_LOAD_COMPLETE;
        e.tile = tile;
        e.bytes = tile.size;
        record(e);
    }

    void record_l3_send(uint64_t cycle, uint8_t src_l3, uint8_t dst_l3,
                        const TileDescriptor& tile, bool start) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = src_l3;
        e.type = start ? TileEventType::L3_SEND_START : TileEventType::L3_SEND_COMPLETE;
        e.tile = tile;
        e.src_l3_id = src_l3;
        e.dst_l3_id = dst_l3;
        e.bytes = tile.size;
        record(e);
    }

    void record_l3_receive(uint64_t cycle, uint8_t l3_id, uint8_t src_l3,
                           const TileDescriptor& tile) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = TileEventType::L3_RECEIVE;
        e.tile = tile;
        e.src_l3_id = src_l3;
        e.dst_l3_id = l3_id;
        e.bytes = tile.size;
        record(e);
    }

    void record_l3_to_l2(uint64_t cycle, uint8_t l3_id, const TileDescriptor& tile, bool start) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = start ? TileEventType::L3_TO_L2_START : TileEventType::L3_TO_L2_COMPLETE;
        e.tile = tile;
        e.bytes = tile.size;
        record(e);
    }

    void record_l2_to_l3(uint64_t cycle, uint8_t l3_id, const TileDescriptor& tile, bool start) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = start ? TileEventType::L2_TO_L3_START : TileEventType::L2_TO_L3_COMPLETE;
        e.tile = tile;
        e.bytes = tile.size;
        record(e);
    }

    void record_barrier(uint64_t cycle, uint8_t l3_id, bool start) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = start ? TileEventType::BARRIER_START : TileEventType::BARRIER_COMPLETE;
        record(e);
    }

    void record_mover_state(uint64_t cycle, uint8_t l3_id, TileEventType state) {
        TileFlowEvent e;
        e.cycle = cycle;
        e.l3_id = l3_id;
        e.type = state;
        record(e);
    }

    // ========== Query ==========

    const std::vector<TileFlowEvent>& events() const { return events_; }
    size_t num_events() const { return events_.size(); }
    bool is_enabled() const { return config_.enabled; }

    void enable() { config_.enabled = true; }
    void disable() { config_.enabled = false; }
    void clear() { events_.clear(); }

    uint64_t start_cycle() const {
        return events_.empty() ? 0 : events_.front().cycle;
    }

    uint64_t end_cycle() const {
        return events_.empty() ? 0 : events_.back().cycle;
    }

    // Get events for a specific L3 tile
    std::vector<TileFlowEvent> events_for_l3(uint8_t l3_id) const {
        std::vector<TileFlowEvent> result;
        for (const auto& e : events_) {
            if (e.l3_id == l3_id) {
                result.push_back(e);
            }
        }
        return result;
    }

    // Get events in a cycle range
    std::vector<TileFlowEvent> events_in_range(uint64_t start, uint64_t end) const {
        std::vector<TileFlowEvent> result;
        for (const auto& e : events_) {
            if (e.cycle >= start && e.cycle <= end) {
                result.push_back(e);
            }
        }
        return result;
    }

    // Get events for specific tiles (by tile key: tensor, m, n, k)
    std::vector<TileFlowEvent> filter_by_tiles(
        const std::set<std::tuple<uint8_t, uint16_t, uint16_t, uint16_t>>& tile_keys) const {
        std::vector<TileFlowEvent> result;
        for (const auto& e : events_) {
            auto key = std::make_tuple(static_cast<uint8_t>(e.tile.tensor),
                                        e.tile.m_tile, e.tile.n_tile, e.tile.k_tile);
            if (tile_keys.count(key) > 0) {
                result.push_back(e);
            }
        }
        return result;
    }

    // Filter events for a specific output tile C[m,n] and its dependencies
    // This includes C[m,n], A[m,k] for all k, and B[k,n] for all k
    std::vector<TileFlowEvent> events_for_output_tile(
        uint16_t m_tile, uint16_t n_tile, size_t k_tiles) const {
        std::set<std::tuple<uint8_t, uint16_t, uint16_t, uint16_t>> tile_keys;

        // C[m,n] output tile
        tile_keys.insert({ static_cast<uint8_t>(2), m_tile, n_tile, static_cast<uint16_t>(0)});

        // A[m,k] input tiles for all k
        for (size_t k = 0; k < k_tiles; k++) {
            tile_keys.insert({ static_cast<uint8_t>(0), m_tile, static_cast<uint16_t>(0), static_cast<uint16_t>(k)});
        }

        // B[k,n] input tiles for all k
        for (size_t k = 0; k < k_tiles; k++) {
            tile_keys.insert({ static_cast<uint8_t>(1), static_cast<uint16_t>(0), n_tile, static_cast<uint16_t>(k)});
        }

        return filter_by_tiles(tile_keys);
    }

    // ========== Export ==========

    /// Export to CSV format
    bool export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) return false;

        file << TileFlowEvent::csv_header() << "\n";
        for (const auto& e : events_) {
            file << e.to_csv() << "\n";
        }

        return true;
    }

    /// Export to JSON format
    bool export_json(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) return false;

        file << "{\n";
        file << "  \"metadata\": {\n";
        file << "    \"num_events\": " << events_.size() << ",\n";
        file << "    \"start_cycle\": " << start_cycle() << ",\n";
        file << "    \"end_cycle\": " << end_cycle() << "\n";
        file << "  },\n";
        file << "  \"events\": [\n";

        for (size_t i = 0; i < events_.size(); ++i) {
            const auto& e = events_[i];
            file << "    {\n";
            file << "      \"cycle\": " << e.cycle << ",\n";
            file << "      \"event_type\": \"" << to_string(e.type) << "\",\n";
            file << "      \"l3_id\": " << static_cast<int>(e.l3_id) << ",\n";
            file << "      \"tensor\": " << static_cast<int>(e.tile.tensor) << ",\n";
            file << "      \"m_tile\": " << e.tile.m_tile << ",\n";
            file << "      \"n_tile\": " << e.tile.n_tile << ",\n";
            file << "      \"k_tile\": " << e.tile.k_tile << ",\n";
            file << "      \"src_l3\": " << static_cast<int>(e.src_l3_id) << ",\n";
            file << "      \"dst_l3\": " << static_cast<int>(e.dst_l3_id) << ",\n";
            file << "      \"bytes\": " << e.bytes << ",\n";
            file << "      \"bank_id\": " << static_cast<int>(e.bank_id) << ",\n";
            file << "      \"row\": " << e.row << ",\n";
            file << "      \"dma_channel\": " << static_cast<int>(e.dma_channel) << ",\n";
            file << "      \"address\": " << e.address << "\n";
            file << "    }" << (i < events_.size() - 1 ? "," : "") << "\n";
        }

        file << "  ]\n";
        file << "}\n";

        return true;
    }

    /// Export to Chrome Trace format (for chrome://tracing or Perfetto)
    /// Basic version - all events on L3 threads
    bool export_chrome_trace(const std::string& filename) const {
        return export_chrome_trace_v2(filename, 4);  // Default 4x4 mesh
    }

    /// Format tile name based on tensor type with correct semantics:
    /// - A[m,k]: A tiles indexed by (row, k-dim)
    /// - B[k,n]: B tiles indexed by (k-dim, column)
    /// - C[m,n]: C tiles indexed by (row, column)
    static std::string format_tile_name(const TileDescriptor& tile) {
        std::string name;
        switch (tile.tensor) {
            case TensorId::A:
                name = "A[" + std::to_string(tile.m_tile) + "," +
                       std::to_string(tile.k_tile) + "]";
                break;
            case TensorId::B:
                name = "B[" + std::to_string(tile.k_tile) + "," +
                       std::to_string(tile.n_tile) + "]";
                break;
            case TensorId::C:
                name = "C[" + std::to_string(tile.m_tile) + "," +
                       std::to_string(tile.n_tile) + "]";
                break;
            case TensorId::BIAS:
                name = "Bias[" + std::to_string(tile.n_tile) + "]";
                break;
            default:
                name = "?[" + std::to_string(tile.m_tile) + "," +
                       std::to_string(tile.n_tile) + "]";
        }
        return name;
    }

    /// Format L3 ID as grid coordinates [row,col]
    static std::string format_l3_id(uint8_t l3_id, size_t mesh_cols) {
        size_t row = l3_id / mesh_cols;
        size_t col = l3_id % mesh_cols;
        return "L3[" + std::to_string(row) + "," + std::to_string(col) + "]";
    }

    /// Get link ID for East (horizontal) links: row * (cols-1) + col
    static int get_east_link_id(uint8_t src_l3, size_t mesh_cols) {
        size_t row = src_l3 / mesh_cols;
        size_t col = src_l3 % mesh_cols;
        return static_cast<int>(row * (mesh_cols - 1) + col);
    }

    /// Get link ID for South (vertical) links: row * cols + col
    static int get_south_link_id(uint8_t src_l3, size_t mesh_cols) {
        size_t row = src_l3 / mesh_cols;
        size_t col = src_l3 % mesh_cols;
        return static_cast<int>(row * mesh_cols + col);
    }

    /// Enhanced Chrome Trace export with proper resource hierarchy:
    /// - Process 1: "L3 Cache" - shows tile presence in each L3
    /// - Process 2: "East Links" - horizontal interconnect link occupancy
    /// - Process 3: "South Links" - vertical interconnect link occupancy
    /// - Process 4: "Compute" - compute unit activity per L3
    /// - Process 5: "L2 Transfers" - L3<->L2 data movement
    bool export_chrome_trace_v2(const std::string& filename, size_t mesh_size = 4) const {
        std::ofstream file(filename);
        if (!file) return false;

        const size_t mesh_cols = mesh_size;
        const size_t mesh_rows = mesh_size;
        const size_t num_l3 = mesh_rows * mesh_cols;

        file << "{\"traceEvents\":[\n";

        // First, emit process and thread metadata
        bool first = true;

        // Process names
        auto emit_process_name = [&](int pid, const std::string& name) {
            if (!first) file << ",\n";
            first = false;
            file << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":" << pid
                 << ",\"args\":{\"name\":\"" << name << "\"}}";
        };

        emit_process_name(1, "L3 Cache State");
        emit_process_name(2, "East Links (A tiles →)");
        emit_process_name(3, "South Links (B tiles ↓)");
        emit_process_name(4, "Compute Units");
        emit_process_name(5, "L2 Transfers");

        // Thread names for L3 caches
        for (size_t i = 0; i < num_l3; i++) {
            size_t row = i / mesh_cols;
            size_t col = i % mesh_cols;
            file << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << i
                 << ",\"args\":{\"name\":\"L3[" << row << "," << col << "]\"}}";
        }

        // Thread names for East links (horizontal)
        for (size_t row = 0; row < mesh_rows; row++) {
            for (size_t col = 0; col < mesh_cols - 1; col++) {
                int link_id = static_cast<int>(row * (mesh_cols - 1) + col);
                file << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":2,\"tid\":" << link_id
                     << ",\"args\":{\"name\":\"[" << row << "," << col << "]→["
                     << row << "," << (col + 1) << "]\"}}";
            }
        }

        // Thread names for South links (vertical)
        for (size_t row = 0; row < mesh_rows - 1; row++) {
            for (size_t col = 0; col < mesh_cols; col++) {
                int link_id = static_cast<int>(row * mesh_cols + col);
                file << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":3,\"tid\":" << link_id
                     << ",\"args\":{\"name\":\"[" << row << "," << col << "]↓["
                     << (row + 1) << "," << col << "]\"}}";
            }
        }

        // Thread names for Compute units
        for (size_t i = 0; i < num_l3; i++) {
            size_t row = i / mesh_cols;
            size_t col = i % mesh_cols;
            file << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":4,\"tid\":" << i
                 << ",\"args\":{\"name\":\"Compute[" << row << "," << col << "]\"}}";
        }

        // Thread names for L2 transfers
        for (size_t i = 0; i < num_l3; i++) {
            size_t row = i / mesh_cols;
            size_t col = i % mesh_cols;
            file << ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":5,\"tid\":" << i
                 << ",\"args\":{\"name\":\"L2[" << row << "," << col << "]\"}}";
        }

        // Now emit actual events
        for (const auto& e : events_) {
            std::string tile_name = format_tile_name(e.tile);
            std::string src_l3_name = format_l3_id(e.src_l3_id, mesh_cols);
            std::string dst_l3_name = format_l3_id(e.dst_l3_id, mesh_cols);

            char phase = 'i';
            int pid = 1;
            int tid = e.l3_id;
            std::string name;
            std::string cat;

            switch (e.type) {
                case TileEventType::L3_SEND_START: {
                    // Determine if this is an East or South transfer
                    size_t src_row = e.src_l3_id / mesh_cols;
                    size_t src_col = e.src_l3_id % mesh_cols;
                    size_t dst_row = e.dst_l3_id / mesh_cols;
                    size_t dst_col = e.dst_l3_id % mesh_cols;

                    if (dst_col > src_col) {
                        // East transfer (A tiles)
                        pid = 2;
                        tid = get_east_link_id(e.src_l3_id, mesh_cols);
                        name = tile_name + " " + src_l3_name + "→" + dst_l3_name;
                    } else if (dst_row > src_row) {
                        // South transfer (B tiles)
                        pid = 3;
                        tid = get_south_link_id(e.src_l3_id, mesh_cols);
                        name = tile_name + " " + src_l3_name + "↓" + dst_l3_name;
                    } else {
                        // Other direction (shouldn't happen in systolic)
                        pid = 1;
                        tid = e.l3_id;
                        name = "SEND " + tile_name + " " + src_l3_name + "->" + dst_l3_name;
                    }
                    phase = 'B';
                    cat = "transfer";
                    break;
                }

                case TileEventType::L3_SEND_COMPLETE: {
                    size_t src_row = e.src_l3_id / mesh_cols;
                    size_t src_col = e.src_l3_id % mesh_cols;
                    size_t dst_row = e.dst_l3_id / mesh_cols;
                    size_t dst_col = e.dst_l3_id % mesh_cols;

                    if (dst_col > src_col) {
                        pid = 2;
                        tid = get_east_link_id(e.src_l3_id, mesh_cols);
                        name = tile_name + " " + src_l3_name + "→" + dst_l3_name;
                    } else if (dst_row > src_row) {
                        pid = 3;
                        tid = get_south_link_id(e.src_l3_id, mesh_cols);
                        name = tile_name + " " + src_l3_name + "↓" + dst_l3_name;
                    } else {
                        pid = 1;
                        tid = e.l3_id;
                        name = "SEND " + tile_name + " " + src_l3_name + "->" + dst_l3_name;
                    }
                    phase = 'E';
                    cat = "transfer";
                    break;
                }

                case TileEventType::L3_RECEIVE:
                    // Show as instant event on the destination L3's cache
                    pid = 1;
                    tid = e.dst_l3_id;
                    name = "RECV " + tile_name + " from " + src_l3_name;
                    phase = 'i';
                    cat = "cache";
                    break;

                case TileEventType::L3_TO_L2_START:
                    pid = 5;
                    tid = e.l3_id;
                    name = "L3→L2 " + tile_name;
                    phase = 'B';
                    cat = "l2";
                    break;

                case TileEventType::L3_TO_L2_COMPLETE:
                    pid = 5;
                    tid = e.l3_id;
                    name = "L3→L2 " + tile_name;
                    phase = 'E';
                    cat = "l2";
                    break;

                case TileEventType::L2_TO_L3_START:
                    pid = 5;
                    tid = e.l3_id;
                    name = "L2→L3 " + tile_name;
                    phase = 'B';
                    cat = "l2";
                    break;

                case TileEventType::L2_TO_L3_COMPLETE:
                    pid = 5;
                    tid = e.l3_id;
                    name = "L2→L3 " + tile_name;
                    phase = 'E';
                    cat = "l2";
                    break;

                case TileEventType::COMPUTE_START:
                    pid = 4;
                    tid = e.l3_id;
                    name = "Compute " + tile_name;
                    phase = 'B';
                    cat = "compute";
                    break;

                case TileEventType::COMPUTE_COMPLETE:
                    pid = 4;
                    tid = e.l3_id;
                    name = "Compute " + tile_name;
                    phase = 'E';
                    cat = "compute";
                    break;

                case TileEventType::BARRIER_START:
                    pid = 1;
                    tid = e.l3_id;
                    name = "BARRIER";
                    phase = 'B';
                    cat = "sync";
                    break;

                case TileEventType::BARRIER_COMPLETE:
                    pid = 1;
                    tid = e.l3_id;
                    name = "BARRIER";
                    phase = 'E';
                    cat = "sync";
                    break;

                case TileEventType::DMA_LOAD_START:
                    pid = 1;
                    tid = e.l3_id;
                    name = "DMA_LOAD " + tile_name;
                    phase = 'B';
                    cat = "dma";
                    break;

                case TileEventType::DMA_LOAD_COMPLETE:
                    pid = 1;
                    tid = e.l3_id;
                    name = "DMA_LOAD " + tile_name;
                    phase = 'E';
                    cat = "dma";
                    break;

                default:
                    pid = 1;
                    tid = e.l3_id;
                    name = to_string(e.type);
                    phase = 'i';
                    cat = "other";
                    break;
            }

            file << ",\n{\"name\":\"" << name << "\","
                 << "\"cat\":\"" << cat << "\","
                 << "\"ph\":\"" << phase << "\","
                 << "\"ts\":" << e.cycle << ","
                 << "\"pid\":" << pid << ","
                 << "\"tid\":" << tid;

            // Add args
            file << ",\"args\":{";
            file << "\"tile\":\"" << tile_name << "\"";
            if (e.src_l3_id != e.dst_l3_id) {
                file << ",\"src\":\"" << src_l3_name << "\"";
                file << ",\"dst\":\"" << dst_l3_name << "\"";
            }
            if (e.bytes > 0) {
                file << ",\"bytes\":" << e.bytes;
            }
            file << "}}";
        }

        file << "\n]}\n";

        return true;
    }

    // ========== Statistics ==========

    struct Stats {
        uint64_t total_events = 0;
        uint64_t dma_loads = 0;
        uint64_t dma_stores = 0;
        uint64_t l3_transfers = 0;
        uint64_t l2_transfers = 0;
        uint64_t barriers = 0;
        uint64_t total_bytes_dma = 0;
        uint64_t total_bytes_l3 = 0;
        uint64_t total_bytes_l2 = 0;

        // Per-L3 activity
        std::vector<uint64_t> events_per_l3;
    };

    Stats compute_stats(size_t num_l3_tiles = 16) const {
        Stats s;
        s.total_events = events_.size();
        s.events_per_l3.resize(num_l3_tiles, 0);

        for (const auto& e : events_) {
            if (e.l3_id < num_l3_tiles) {
                s.events_per_l3[e.l3_id]++;
            }

            switch (e.type) {
                case TileEventType::DMA_LOAD_START:
                    s.dma_loads++;
                    s.total_bytes_dma += e.bytes;
                    break;
                case TileEventType::DMA_STORE_START:
                    s.dma_stores++;
                    s.total_bytes_dma += e.bytes;
                    break;
                case TileEventType::L3_SEND_START:
                    s.l3_transfers++;
                    s.total_bytes_l3 += e.bytes;
                    break;
                case TileEventType::L3_TO_L2_START:
                    s.l2_transfers++;
                    s.total_bytes_l2 += e.bytes;
                    break;
                case TileEventType::BARRIER_START:
                    s.barriers++;
                    break;
                default:
                    break;
            }
        }

        return s;
    }

private:
    Config config_;
    std::vector<TileFlowEvent> events_;
};

} // namespace sw::kpu::dataflow
