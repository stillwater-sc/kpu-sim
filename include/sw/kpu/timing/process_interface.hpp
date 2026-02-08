// ============================================================================
// include/sw/kpu/timing/process_interface.hpp
// Process interface for CSP-style concurrent timing simulation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::timing {

// ============================================================================
// TimingEvent - Events emitted by processes during simulation
// ============================================================================

/**
 * @brief Type of timing event
 */
enum class EventType {
    // DMA / Memory Controller events
    DMA_LOAD_START,      ///< DMA started loading tile from DRAM
    DMA_LOAD_COMPLETE,   ///< DMA finished loading tile to L3
    DMA_STORE_START,     ///< DMA started storing tile to DRAM
    DMA_STORE_COMPLETE,  ///< DMA finished storing tile to DRAM
    DMA_STALL_CREDIT,    ///< DMA stalled waiting for L3 credit
    DMA_STALL_TAG,       ///< DMA stalled waiting for tile in L3 (for store)
    MC_ACCESS_TYPE,      ///< Memory Controller access classification (row hit/miss/empty)
    MC_BANK_CONFLICT,    ///< Memory Controller bank conflict (different row in same bank)

    // BlockMover events
    BM_MOVE_START,       ///< BlockMover started L3→L2 transfer
    BM_MOVE_COMPLETE,    ///< BlockMover finished L3→L2 transfer
    BM_WRITEBACK_START,  ///< BlockMover started L2→L3 writeback
    BM_WRITEBACK_COMPLETE, ///< BlockMover finished L2→L3 writeback
    BM_STALL_TAG,        ///< BlockMover stalled waiting for tile in L3
    BM_STALL_CREDIT,     ///< BlockMover stalled waiting for L2 credit

    // Streamer events
    STR_FEED_START,      ///< Streamer started feeding tile to compute
    STR_FEED_COMPLETE,   ///< Streamer finished feeding tile
    STR_DRAIN_START,     ///< Streamer started draining result
    STR_DRAIN_COMPLETE,  ///< Streamer finished draining result
    STR_STALL_TAG,       ///< Streamer stalled waiting for tile in L2
    STR_STALL_CREDIT,    ///< Streamer stalled waiting for L2 credit (drain)
    STR_STALL_COMPUTE,   ///< Streamer stalled waiting for compute result

    // Compute events
    COMPUTE_START,       ///< Systolic array started computation
    COMPUTE_COMPLETE,    ///< Systolic array finished computation

    // Credit events
    CREDIT_ACQUIRED,     ///< Credit acquired from pool
    CREDIT_RELEASED,     ///< Credit released to pool

    // Tile tracking
    TILE_ARRIVED_L3,     ///< Tile arrived at L3 buffer
    TILE_ARRIVED_L2,     ///< Tile arrived at L2 bank
    TILE_ARRIVED_L1,     ///< Tile arrived at L1 buffer
    TILE_FED_TO_COMPUTE, ///< Tile fed to compute unit
    TILE_DRAINED,        ///< Result tile drained from compute
    TILE_CONSUMED,       ///< Tile consumed (removed from memory level)
};

/**
 * @brief Convert event type to string for debugging/tracing
 */
inline const char* to_string(EventType type) {
    switch (type) {
        case EventType::DMA_LOAD_START: return "DMA_LOAD_START";
        case EventType::DMA_LOAD_COMPLETE: return "DMA_LOAD_COMPLETE";
        case EventType::DMA_STORE_START: return "DMA_STORE_START";
        case EventType::DMA_STORE_COMPLETE: return "DMA_STORE_COMPLETE";
        case EventType::DMA_STALL_CREDIT: return "DMA_STALL_CREDIT";
        case EventType::DMA_STALL_TAG: return "DMA_STALL_TAG";
        case EventType::MC_ACCESS_TYPE: return "MC_ACCESS_TYPE";
        case EventType::MC_BANK_CONFLICT: return "MC_BANK_CONFLICT";
        case EventType::BM_MOVE_START: return "BM_MOVE_START";
        case EventType::BM_MOVE_COMPLETE: return "BM_MOVE_COMPLETE";
        case EventType::BM_WRITEBACK_START: return "BM_WRITEBACK_START";
        case EventType::BM_WRITEBACK_COMPLETE: return "BM_WRITEBACK_COMPLETE";
        case EventType::BM_STALL_TAG: return "BM_STALL_TAG";
        case EventType::BM_STALL_CREDIT: return "BM_STALL_CREDIT";
        case EventType::STR_FEED_START: return "STR_FEED_START";
        case EventType::STR_FEED_COMPLETE: return "STR_FEED_COMPLETE";
        case EventType::STR_DRAIN_START: return "STR_DRAIN_START";
        case EventType::STR_DRAIN_COMPLETE: return "STR_DRAIN_COMPLETE";
        case EventType::STR_STALL_TAG: return "STR_STALL_TAG";
        case EventType::STR_STALL_CREDIT: return "STR_STALL_CREDIT";
        case EventType::STR_STALL_COMPUTE: return "STR_STALL_COMPUTE";
        case EventType::COMPUTE_START: return "COMPUTE_START";
        case EventType::COMPUTE_COMPLETE: return "COMPUTE_COMPLETE";
        case EventType::CREDIT_ACQUIRED: return "CREDIT_ACQUIRED";
        case EventType::CREDIT_RELEASED: return "CREDIT_RELEASED";
        case EventType::TILE_ARRIVED_L3: return "TILE_ARRIVED_L3";
        case EventType::TILE_ARRIVED_L2: return "TILE_ARRIVED_L2";
        case EventType::TILE_ARRIVED_L1: return "TILE_ARRIVED_L1";
        case EventType::TILE_FED_TO_COMPUTE: return "TILE_FED_TO_COMPUTE";
        case EventType::TILE_DRAINED: return "TILE_DRAINED";
        case EventType::TILE_CONSUMED: return "TILE_CONSUMED";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Timing event emitted by a process
 *
 * Events capture what happened during a simulation cycle for:
 * - Chrome trace visualization
 * - Statistics collection
 * - Debugging
 */
struct TimingEvent {
    EventType type;           ///< What happened
    Cycle cycle;              ///< When it happened
    uint32_t component_id;    ///< Which component (DMA engine 0, BlockMover 1, etc.)
    TileID tile_id;           ///< Which tile (if applicable)
    Cycle duration;           ///< Duration in cycles (for COMPLETE events)
    uint32_t slot_id;         ///< Buffer/bank slot (if applicable)
    std::string component_name; ///< Human-readable component name
    std::string detail;         ///< Additional detail (e.g., "ROW_HIT" for MC_ACCESS_TYPE)
    Address matrix_base_address = 0; ///< Base address of the matrix in DRAM
    Address dram_address = 0;  ///< DRAM address for this tile

    TimingEvent() = default;

    TimingEvent(EventType t, Cycle c, uint32_t comp_id, const std::string& comp_name = "")
        : type(t), cycle(c), component_id(comp_id), duration(0), slot_id(0),
          component_name(comp_name) {}

    TimingEvent(EventType t, Cycle c, uint32_t comp_id, const TileID& tile,
                const std::string& comp_name = "")
        : type(t), cycle(c), component_id(comp_id), tile_id(tile),
          duration(0), slot_id(0), component_name(comp_name) {}

    /// Create a duration event (for Chrome trace X events)
    static TimingEvent duration_event(EventType t, Cycle start, Cycle dur,
                                       uint32_t comp_id, const TileID& tile,
                                       const std::string& comp_name = "") {
        TimingEvent e(t, start, comp_id, tile, comp_name);
        e.duration = dur;
        return e;
    }

    /// Generate Chrome trace JSON format
    std::string to_chrome_trace_json() const {
        std::string json = "{";
        json += "\"name\":\"" + std::string(to_string(type)) + "\",";
        json += "\"cat\":\"" + component_name + "\",";
        json += "\"ph\":\"" + std::string(duration > 0 ? "X" : "i") + "\",";
        json += "\"ts\":" + std::to_string(cycle) + ",";
        if (duration > 0) {
            json += "\"dur\":" + std::to_string(duration) + ",";
        }
        json += "\"pid\":1,";
        json += "\"tid\":" + std::to_string(component_id) + ",";
        json += "\"args\":{";
        json += "\"tile\":\"" + tile_id.to_string() + "\"";
        if (matrix_base_address != 0) {
            char hex[32];
            snprintf(hex, sizeof(hex), "0x%lX", matrix_base_address);
            json += ",\"base\":\"" + std::string(hex) + "\"";
        }
        if (dram_address != 0) {
            char hex[32];
            snprintf(hex, sizeof(hex), "0x%lX", dram_address);
            json += ",\"addr\":\"" + std::string(hex) + "\"";
        }
        if (slot_id > 0) {
            json += ",\"slot\":" + std::to_string(slot_id);
        }
        json += "}}";
        return json;
    }
};

// ============================================================================
// IProcess - Interface for all CSP processes
// ============================================================================

/**
 * @brief Interface for CSP-style component processes
 *
 * All component processes (DMA, BlockMover, Streamer) implement this interface.
 * The executor calls tick() on each process every cycle to advance simulation.
 *
 * Key principles:
 * - Processes are independent and don't directly communicate
 * - Synchronization happens through shared CreditPools and TagCAMs
 * - Each tick() call may emit zero or more TimingEvents
 */
class IProcess {
public:
    virtual ~IProcess() = default;

    /**
     * @brief Advance simulation by one cycle
     * @param current_cycle The current simulation cycle
     * @return Vector of events that occurred this cycle
     *
     * Called by the executor every cycle. The process should:
     * 1. Check for completed transfers
     * 2. Try to issue new work if resources available
     * 3. Return events describing what happened
     */
    virtual std::vector<TimingEvent> tick(Cycle current_cycle) = 0;

    /**
     * @brief Check if process has no in-flight work
     * @return true if no transfers are in progress
     */
    [[nodiscard]] virtual bool is_idle() const = 0;

    /**
     * @brief Check if process has pending work in queues
     * @return true if work queues are non-empty
     */
    [[nodiscard]] virtual bool has_pending_work() const = 0;

    /**
     * @brief Check if process is complete (idle and no pending work)
     * @return true if process is done with all work
     */
    [[nodiscard]] virtual bool is_complete() const {
        return is_idle() && !has_pending_work();
    }

    /**
     * @brief Get process identifier
     * @return Unique ID for this process instance
     */
    [[nodiscard]] virtual uint32_t id() const = 0;

    /**
     * @brief Get human-readable process name
     * @return Name for debugging/tracing
     */
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Reset process to initial state
     *
     * Clears all queues and in-flight state. Used at simulation start.
     */
    virtual void reset() = 0;
};

// ============================================================================
// InFlightTransfer - Common structure for tracking in-progress transfers
// ============================================================================

/**
 * @brief Represents a transfer that is currently in progress
 */
struct InFlightTransfer {
    TileDescriptor tile;    ///< The tile being transferred
    Cycle start_cycle;      ///< When the transfer started
    Cycle complete_cycle;   ///< When the transfer will complete
    uint32_t slot_id;       ///< Destination slot (buffer/bank ID)
    bool is_load;           ///< true = load/move/feed, false = store/writeback/drain

    InFlightTransfer() = default;

    InFlightTransfer(const TileDescriptor& t, Cycle start, Cycle complete,
                     uint32_t slot, bool load)
        : tile(t), start_cycle(start), complete_cycle(complete),
          slot_id(slot), is_load(load) {}

    /// Check if transfer is complete at given cycle
    [[nodiscard]] bool is_complete(Cycle current_cycle) const {
        return current_cycle >= complete_cycle;
    }

    /// Get transfer duration
    [[nodiscard]] Cycle duration() const {
        return complete_cycle - start_cycle;
    }
};

} // namespace sw::kpu::timing
