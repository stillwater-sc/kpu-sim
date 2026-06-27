// ============================================================================
// include/sw/kpu/timing/memory_controller_process.hpp
// Transactional Memory Controller with correct resource contention
//
// Models a single LPDDR5 channel with:
// - Command bus: 1 command per cycle (shared resource)
// - Bank State Machines: Track open row per bank
// - Data bus occupancy: Burst transfers occupy the bus
//
// The MC is a "dumb" DRAM access resource. DMA engines submit requests and
// poll for completions. L3 credit/tag management is done by DMA engines.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/process_interface.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Bank state for DRAM bank state machine
 */
struct BankState {
    enum class State {
        IDLE,         ///< No row activated
        ACTIVE,       ///< Row is open and ready for access
        ACTIVATING,   ///< ACT command issued, waiting for tRCD
        PRECHARGING   ///< PRE command issued, waiting for tRP
    };

    State state = State::IDLE;
    uint32_t open_row = 0;         ///< Which row is open (valid when ACTIVE)
    Cycle ready_cycle = 0;         ///< When bank accepts next command
};

/**
 * @brief Access type classification for latency calculation
 */
enum class MemoryAccessType {
    ROW_HIT,      ///< Row already open, just RD/WR
    ROW_MISS,     ///< Different row open, need PRE + ACT + RD/WR
    ROW_EMPTY     ///< Bank idle, need ACT + RD/WR
};

/**
 * @brief Completed transfer notification from MC to DMA
 */
struct CompletedTransfer {
    TileDescriptor tile;
    bool is_load;              ///< true = load from DRAM, false = store to DRAM
    Cycle start_cycle;
    Cycle complete_cycle;
    uint32_t bank_id;
    MemoryAccessType access_type;
    uint32_t submitter_id;     ///< ID of the DMA engine that submitted this request
};

/**
 * @brief Memory Controller Process for DRAM access contention modeling
 *
 * This models a single LPDDR5 memory controller/channel with correct
 * resource contention:
 *
 * 1. **Command Bus**: Only 1 command per cycle (ACT, PRE, RD, WR)
 * 2. **Bank States**: 16 banks, each with open/closed row tracking
 * 3. **Data Bus**: Occupied during burst transfers
 *
 * DMA engines submit requests via submit_request() and poll for completions
 * via get_completed_transfer(). The MC does NOT handle L3 credits or tags -
 * that's the DMA engine's responsibility.
 */
class MemoryControllerProcess : public IProcess {
public:
    /**
     * @brief Memory Controller configuration
     */
    struct Config {
        uint32_t controller_id = 0;     ///< Unique MC identifier
        size_t num_banks = 16;          ///< Number of banks (LPDDR5: 4 BG × 4 banks)
        size_t num_bank_groups = 4;     ///< Number of bank groups
        size_t request_queue_depth = 32;///< Max pending requests

        // Simple timing model (cycles at reference clock)
        Cycle t_cl = 10;                ///< CAS latency (row hit to data)
        Cycle t_rcd = 15;               ///< RAS to CAS delay (ACT to RD/WR)
        Cycle t_rp = 15;                ///< Row precharge time
        Cycle t_burst = 4;              ///< Burst transfer duration
        Cycle startup_latency = 5;      ///< Command issue overhead

        // Bandwidth (for statistics, not used in simple model)
        double bandwidth_gbps = 25.6;   ///< Channel bandwidth
        double clock_ghz = 1.0;         ///< Reference clock

        // Address mapping (simplified)
        uint32_t row_bits = 14;         ///< Bits for row address
        uint32_t col_bits = 10;         ///< Bits for column address
        uint32_t bank_bits = 4;         ///< Bits for bank address (log2(16))

        std::string name = "MC";

        std::string display_name() const {
            return "MC" + std::to_string(controller_id);
        }
    };

    /**
     * @brief Pending memory request
     */
    struct PendingRequest {
        TileDescriptor tile;
        bool is_load;                   ///< true = load from DRAM, false = store to DRAM
        Cycle enqueue_cycle;
        uint32_t bank_id;               ///< Target bank (0-15)
        uint32_t row_id;                ///< Target row within bank
        uint32_t priority = 0;          ///< For future scheduling policies
        uint32_t request_id = 0;        ///< Unique request ID for tracking
        uint32_t submitter_id = 0;      ///< ID of the DMA engine that submitted this
    };

    /**
     * @brief In-flight transfer tracking
     */
    struct InFlightTransfer {
        TileDescriptor tile;
        bool is_load;
        Cycle start_cycle;
        Cycle complete_cycle;
        uint32_t bank_id;
        MemoryAccessType access_type;
        uint32_t submitter_id;
    };

    /**
     * @brief Construct a Memory Controller process
     * @param config MC configuration
     */
    explicit MemoryControllerProcess(const Config& config)
        : config_(config),
          bank_states_(config.num_banks) {
    }

    // ========================================================================
    // DMA Engine Interface (submit/poll pattern)
    // ========================================================================

    /**
     * @brief Submit a memory request from DMA to MC
     * @param tile Tile descriptor with DRAM address and size
     * @param is_load true = load from DRAM, false = store to DRAM
     * @param submitter_id ID of the DMA engine submitting this request
     * @return true if request was accepted, false if queue full
     *
     * DMA engines call this to submit DRAM access requests. The MC will
     * process them respecting command bus and bank state constraints.
     */
    bool submit_request(const TileDescriptor& tile, bool is_load, uint32_t submitter_id = 0) {
        if (request_queue_.size() >= config_.request_queue_depth) {
            return false;  // Queue full
        }

        PendingRequest req;
        req.tile = tile;
        req.is_load = is_load;
        req.enqueue_cycle = current_cycle_;
        req.bank_id = address_to_bank(tile.dram_address);
        req.row_id = address_to_row(tile.dram_address);
        req.request_id = next_request_id_++;
        req.submitter_id = submitter_id;

        request_queue_.push_back(req);
        return true;
    }

    /**
     * @brief Poll for completed transfers from a specific submitter
     * @param submitter_id ID of the DMA engine polling for its completions
     * @return CompletedTransfer if one is available for this submitter, std::nullopt otherwise
     *
     * DMA engines call this to check for completed DRAM accesses.
     * Only returns transfers that were submitted by the specified DMA engine.
     */
    std::optional<CompletedTransfer> get_completed_transfer(uint32_t submitter_id) {
        for (auto it = completed_transfers_.begin(); it != completed_transfers_.end(); ++it) {
            if (it->submitter_id == submitter_id) {
                CompletedTransfer ct = *it;
                completed_transfers_.erase(it);
                return ct;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Poll for any completed transfer (legacy API for tests)
     * @return CompletedTransfer if one is available, std::nullopt otherwise
     */
    std::optional<CompletedTransfer> get_completed_transfer() {
        if (completed_transfers_.empty()) {
            return std::nullopt;
        }
        CompletedTransfer ct = completed_transfers_.front();
        completed_transfers_.erase(completed_transfers_.begin());
        return ct;
    }

    /**
     * @brief Check if there are completed transfers waiting
     */
    [[nodiscard]] bool has_completed_transfers() const {
        return !completed_transfers_.empty();
    }

    /**
     * @brief Check if there are completed transfers waiting for a specific submitter
     */
    [[nodiscard]] bool has_completed_transfers(uint32_t submitter_id) const {
        for (const auto& ct : completed_transfers_) {
            if (ct.submitter_id == submitter_id) {
                return true;
            }
        }
        return false;
    }

    // ========================================================================
    // Legacy API (for backward compatibility with tests)
    // These wrap submit_request for convenience
    // ========================================================================

    /**
     * @brief Schedule a tile load from DRAM to L3 (legacy API)
     */
    void schedule_load(const TileDescriptor& tile) {
        submit_request(tile, true);
    }

    /**
     * @brief Schedule a tile store from L3 to DRAM (legacy API)
     */
    void schedule_store(const TileDescriptor& tile) {
        submit_request(tile, false);
    }

    // ========================================================================
    // IProcess Interface
    // ========================================================================

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfers
        check_completions(current_cycle, events);

        // Step 2: Try to issue a command (only 1 per cycle!)
        try_issue_command(current_cycle, events);

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        return in_flight_.empty();
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !request_queue_.empty();
    }

    /**
     * @brief Check if MC is complete (no pending or in-flight work)
     */
    [[nodiscard]] bool is_complete() const override {
        return request_queue_.empty() && in_flight_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.controller_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name;
    }

    void reset() override {
        request_queue_.clear();
        in_flight_.clear();
        completed_transfers_.clear();
        for (auto& bank : bank_states_) {
            bank = BankState{};
        }
        command_bus_ready_ = 0;
        data_bus_ready_ = 0;
        stall_cycles_cmd_bus_ = 0;
        stall_cycles_bank_ = 0;
        row_hits_ = 0;
        row_misses_ = 0;
        row_empty_ = 0;
        total_bytes_transferred_ = 0;
        next_request_id_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t pending_requests() const { return request_queue_.size(); }
    [[nodiscard]] size_t in_flight_count() const { return in_flight_.size(); }
    [[nodiscard]] Cycle stall_cycles_cmd_bus() const { return stall_cycles_cmd_bus_; }
    [[nodiscard]] Cycle stall_cycles_bank() const { return stall_cycles_bank_; }
    [[nodiscard]] size_t row_hits() const { return row_hits_; }
    [[nodiscard]] size_t row_misses() const { return row_misses_; }
    [[nodiscard]] size_t row_empty_accesses() const { return row_empty_; }

    [[nodiscard]] double row_hit_rate() const {
        size_t total = row_hits_ + row_misses_ + row_empty_;
        return total > 0 ? static_cast<double>(row_hits_) / total : 0.0;
    }

    [[nodiscard]] const Config& config() const { return config_; }

    // For compatibility with ConcurrentTimingExecutor statistics
    [[nodiscard]] Cycle stall_cycles() const {
        return stall_cycles_cmd_bus_ + stall_cycles_bank_;
    }
    [[nodiscard]] size_t total_bytes_transferred() const { return total_bytes_transferred_; }

private:
    Config config_;

    std::vector<BankState> bank_states_;
    std::vector<PendingRequest> request_queue_;
    std::vector<InFlightTransfer> in_flight_;
    std::vector<CompletedTransfer> completed_transfers_;  ///< Completed transfers for DMA to poll

    Cycle current_cycle_ = 0;
    Cycle command_bus_ready_ = 0;    ///< When command bus is free
    Cycle data_bus_ready_ = 0;       ///< When data bus is free
    uint32_t next_request_id_ = 0;

    // Statistics
    Cycle stall_cycles_cmd_bus_ = 0;
    Cycle stall_cycles_bank_ = 0;
    size_t row_hits_ = 0;
    size_t row_misses_ = 0;
    size_t row_empty_ = 0;
    size_t total_bytes_transferred_ = 0;

    // ========================================================================
    // Address Mapping (simplified linear mapping)
    // ========================================================================

    [[nodiscard]] uint32_t address_to_bank(uint64_t addr) const {
        // Simple interleaving: bank = (addr >> col_bits) & bank_mask
        uint32_t bank_mask = (1u << config_.bank_bits) - 1;
        return static_cast<uint32_t>((addr >> config_.col_bits) & bank_mask);
    }

    [[nodiscard]] uint32_t address_to_row(uint64_t addr) const {
        // Row = (addr >> (col_bits + bank_bits)) & row_mask
        uint32_t shift = config_.col_bits + config_.bank_bits;
        uint32_t row_mask = (1u << config_.row_bits) - 1;
        return static_cast<uint32_t>((addr >> shift) & row_mask);
    }

    // ========================================================================
    // Access Classification
    // ========================================================================

    [[nodiscard]] MemoryAccessType classify_access(uint32_t bank_id, uint32_t row_id) const {
        const auto& bank = bank_states_[bank_id];

        switch (bank.state) {
            case BankState::State::IDLE:
                return MemoryAccessType::ROW_EMPTY;

            case BankState::State::ACTIVE:
                if (bank.open_row == row_id) {
                    return MemoryAccessType::ROW_HIT;
                } else {
                    return MemoryAccessType::ROW_MISS;
                }

            case BankState::State::ACTIVATING:
            case BankState::State::PRECHARGING:
                // Bank is busy - treat as empty (will stall until ready)
                return MemoryAccessType::ROW_EMPTY;
        }
        return MemoryAccessType::ROW_EMPTY;
    }

    [[nodiscard]] Cycle compute_latency(MemoryAccessType type) const {
        switch (type) {
            case MemoryAccessType::ROW_HIT:
                // Just CAS latency + burst
                return config_.startup_latency + config_.t_cl + config_.t_burst;

            case MemoryAccessType::ROW_EMPTY:
                // ACT + CAS latency + burst
                return config_.startup_latency + config_.t_rcd + config_.t_cl + config_.t_burst;

            case MemoryAccessType::ROW_MISS:
                // PRE + ACT + CAS latency + burst
                return config_.startup_latency + config_.t_rp + config_.t_rcd +
                       config_.t_cl + config_.t_burst;
        }
        return config_.startup_latency + config_.t_rcd + config_.t_cl + config_.t_burst;
    }

    // ========================================================================
    // Command Processing
    // ========================================================================

    void check_completions(Cycle current_cycle, std::vector<TimingEvent>& events) {
        auto it = in_flight_.begin();
        while (it != in_flight_.end()) {
            if (current_cycle >= it->complete_cycle) {
                // Transfer complete - add to completed queue for DMA to poll
                CompletedTransfer ct;
                ct.tile = it->tile;
                ct.is_load = it->is_load;
                ct.start_cycle = it->start_cycle;
                ct.complete_cycle = it->complete_cycle;
                ct.bank_id = it->bank_id;
                ct.access_type = it->access_type;
                ct.submitter_id = it->submitter_id;
                completed_transfers_.push_back(ct);

                total_bytes_transferred_ += it->tile.size_bytes;

                // Emit MC completion event
                auto event_type = it->is_load ? EventType::DMA_LOAD_COMPLETE
                                              : EventType::DMA_STORE_COMPLETE;
                auto event = TimingEvent::duration_event(
                    event_type,
                    it->start_cycle,
                    it->complete_cycle - it->start_cycle,
                    config_.controller_id,
                    it->tile.tile_id,
                    name()
                );
                event.matrix_base_address = it->tile.matrix_base_address;
                event.dram_address = it->tile.dram_address;
                events.push_back(event);

                it = in_flight_.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * @brief Try to issue ONE command this cycle
     *
     * This is the key constraint: only 1 command per cycle on the command bus.
     * Uses simple FCFS scheduling - first ready request wins.
     */
    void try_issue_command(Cycle current_cycle, std::vector<TimingEvent>& events) {
        // Check command bus availability
        if (current_cycle < command_bus_ready_) {
            stall_cycles_cmd_bus_++;
            return;  // Command bus busy
        }

        // Find first ready request
        for (auto it = request_queue_.begin(); it != request_queue_.end(); ++it) {
            auto& req = *it;
            auto& bank = bank_states_[req.bank_id];

            // Check if bank is ready
            if (current_cycle < bank.ready_cycle) {
                continue;  // Bank busy, try next request
            }

            // Classify access type and compute latency
            MemoryAccessType access_type = classify_access(req.bank_id, req.row_id);
            Cycle latency = compute_latency(access_type);

            // Update statistics
            switch (access_type) {
                case MemoryAccessType::ROW_HIT:   row_hits_++; break;
                case MemoryAccessType::ROW_MISS:  row_misses_++; break;
                case MemoryAccessType::ROW_EMPTY: row_empty_++; break;
            }

            // Update bank state
            bank.state = BankState::State::ACTIVE;
            bank.open_row = req.row_id;
            bank.ready_cycle = current_cycle + latency;

            // Update bus occupancy
            command_bus_ready_ = current_cycle + 1;  // 1 command per cycle
            data_bus_ready_ = current_cycle + latency;

            // Create in-flight transfer
            InFlightTransfer xfer;
            xfer.tile = req.tile;
            xfer.is_load = req.is_load;
            xfer.start_cycle = current_cycle;
            xfer.complete_cycle = current_cycle + latency;
            xfer.bank_id = req.bank_id;
            xfer.access_type = access_type;
            xfer.submitter_id = req.submitter_id;
            in_flight_.push_back(xfer);

            // Emit start event
            auto event_type = req.is_load ? EventType::DMA_LOAD_START
                                          : EventType::DMA_STORE_START;
            auto event = TimingEvent(
                event_type,
                current_cycle,
                config_.controller_id,
                req.tile.tile_id,
                name()
            );
            event.matrix_base_address = req.tile.matrix_base_address;
            event.dram_address = req.tile.dram_address;
            events.push_back(event);

            // Emit access type info
            const char* access_str =
                (access_type == MemoryAccessType::ROW_HIT) ? "ROW_HIT" :
                (access_type == MemoryAccessType::ROW_MISS) ? "ROW_MISS" : "ROW_EMPTY";
            auto detail_event = TimingEvent(
                EventType::MC_ACCESS_TYPE,
                current_cycle,
                config_.controller_id,
                req.tile.tile_id,
                name()
            );
            detail_event.detail = access_str;
            events.push_back(detail_event);

            // Remove from queue
            request_queue_.erase(it);
            return;  // Only 1 command per cycle!
        }

        // No ready request found - all stalled on bank
        if (!request_queue_.empty()) {
            stall_cycles_bank_++;
        }
    }
};

} // namespace sw::kpu::timing
