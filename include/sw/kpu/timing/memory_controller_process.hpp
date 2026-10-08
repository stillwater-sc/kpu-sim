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
// TWO MODELS (docs/plans/dram-bank-model.md step 2). Without Config::hosted, the legacy
// model below: one request per tile, a size-independent latency, and a data bus that is
// written but never read -- blind to everything the bank structure does (§1.4). With
// Config::hosted, the process hosts the cycle-accurate LPDDR5 controller through a
// DramBridge: every tile is split into bursts the deployment's address map places, the
// controller schedules them on its banks and per-channel data buses in its own clock, and the
// tile completes when its last burst does. Opt-in for now; the legacy model stays the default
// until a follow-up flips it and rebaselines.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/dram_bridge.hpp>
#include <sw/kpu/timing/process_interface.hpp>

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
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

        // The declared DRAM to host (DramHosting::of(spec.device(i))). Absent = the legacy
        // model. `clock_ghz` is then the executor clock the bridge converts to.
        std::optional<DramHosting> hosted;

        // Record every DRAM command and every window burst (hosted only;
        // docs/plans/memory-side-debugger.md §3.1). Off = no cost.
        bool record = false;

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
        if (config_.hosted)
            bridge_ = std::make_unique<DramBridge>(*config_.hosted, config_.controller_id,
                                                   config_.clock_ghz,
                                                   static_cast<std::uint32_t>(config_.request_queue_depth));
    }

    [[nodiscard]] bool hosted() const { return bridge_ != nullptr; }
    [[nodiscard]] const DramBridge* bridge() const { return bridge_.get(); }

    // ========================================================================
    // Burst interface (hosted only; docs/plans/dram-bank-model.md step 3)
    // ========================================================================
    // A DMA engine with a burst window decomposes its tile, keeps up to W bursts in flight, and
    // counts them home. The controller sees bursts, reorders them by bank (FR-FCFS), and
    // reports each one's completion to its submitter. The tile is the engine's business.

    /// The DRAM burst: the unit of submit_burst.
    [[nodiscard]] std::uint64_t burst_bytes() const {
        if (!bridge_) throw std::logic_error(name() + ": bursts need a hosted DRAM");
        return bridge_->map().burst_bytes();
    }

    /// The bursts `tile` occupies: from the burst holding its first byte to the one holding
    /// its last.
    [[nodiscard]] std::uint64_t bursts_of(const TileDescriptor& tile) const {
        const std::uint64_t b = burst_bytes();
        if (tile.size_bytes == 0) return 0;
        const std::uint64_t first = tile.dram_address / b * b;
        const std::uint64_t end = (tile.dram_address + tile.size_bytes + b - 1) / b * b;
        return (end - first) / b;
    }

    /// Submit burst `index` of `tile`. False = the controller's queue is full (back-pressure).
    bool submit_burst(const TileDescriptor& tile, std::uint64_t index, bool is_load,
                      uint32_t submitter_id) {
        const std::uint64_t b = burst_bytes();
        const auto& map = bridge_->map();
        if (tile.dram_address + tile.size_bytes > map.capacity())
            throw std::out_of_range(
                config_.name + ": tile " + std::to_string(tile.dram_address) + " + " +
                std::to_string(tile.size_bytes) + " B runs past the declared DRAM (" +
                std::to_string(map.capacity()) + " B)");
        const std::uint64_t address = tile.dram_address / b * b + index * b;
        const std::uint64_t id = next_burst_id_++;
        ensure_recording();
        if (!bridge_->submit(address, is_load, kBurstTag | id)) {
            --next_burst_id_;
            return false;
        }
        bursts_[id] = BurstDone{tile.tile_id, is_load, submitter_id};
        if (config_.record) {
            BurstRecord r;
            r.id = id;
            r.submitter_id = submitter_id;
            r.tile = tile.tile_id;
            r.is_load = is_load;
            r.address = address;
            r.coord = map.decode(address);
            r.submitted = current_cycle_;
            burst_index_[id] = burst_records_.size();
            burst_records_.push_back(r);
        }
        return true;
    }

    struct BurstDone {
        TileID tile;
        bool is_load = true;
        uint32_t submitter_id = 0;
    };

    // ---- Recording (Config::record) ----
    enum class PageOutcome : uint8_t { Unknown, Hit, Empty, Conflict };
    struct BurstRecord {
        uint64_t id = 0;                    // this controller's burst id
        uint32_t submitter_id = 0;
        TileID tile;
        bool is_load = true;
        uint64_t address = 0;               // the burst's first byte
        program::platform::DramCoord coord; // decoded by the spec's map
        Cycle submitted = 0;                // into the controller's queue
        Cycle first_command = 0;            // its first command: ACT, or the CAS on a page hit
        Cycle data_start = 0, data_end = 0; // its CAS's data-bus window
        Cycle done = 0;                     // reported complete
        bool commanded = false, finished = false;
        PageOutcome outcome = PageOutcome::Unknown;
    };
    [[nodiscard]] const std::vector<BurstRecord>& recorded_bursts() const { return burst_records_; }
    [[nodiscard]] const std::vector<DramBridge::Command>& recorded_commands() const {
        return command_records_;
    }
    /// The oldest finished burst submitted by `submitter_id`, if any.
    std::optional<BurstDone> get_completed_burst(uint32_t submitter_id) {
        for (auto it = bursts_done_.begin(); it != bursts_done_.end(); ++it)
            if (it->submitter_id == submitter_id) {
                const BurstDone d = *it;
                bursts_done_.erase(it);
                return d;
            }
        return std::nullopt;
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
        if (bridge_) return submit_hosted(tile, is_load, submitter_id);
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
        if (bridge_) {
            tick_hosted(current_cycle, events);
            return events;
        }

        // Step 1: Check for completed transfers
        check_completions(current_cycle, events);

        // Step 2: Try to issue a command (only 1 per cycle!)
        try_issue_command(current_cycle, events);

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        if (bridge_) return hosted_.empty() && bursts_.empty() && !bridge_->busy();
        return in_flight_.empty();
    }

    [[nodiscard]] bool has_pending_work() const override {
        if (bridge_) return !hosted_.empty() || !bursts_done_.empty();
        return !request_queue_.empty();
    }

    /**
     * @brief Check if MC is complete (no pending or in-flight work)
     */
    [[nodiscard]] bool is_complete() const override {
        if (bridge_)
            return hosted_.empty() && bursts_.empty() && bursts_done_.empty() && !bridge_->busy();
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
        hosted_.clear();
        hosted_order_.clear();
        bursts_.clear();
        bursts_done_.clear();
        next_burst_id_ = 0;
        burst_records_.clear();
        burst_index_.clear();
        command_records_.clear();
        if (bridge_) bridge_->reset();
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t pending_requests() const { return request_queue_.size(); }
    [[nodiscard]] size_t in_flight_count() const { return in_flight_.size(); }
    [[nodiscard]] Cycle stall_cycles_cmd_bus() const { return stall_cycles_cmd_bus_; }
    [[nodiscard]] Cycle stall_cycles_bank() const { return stall_cycles_bank_; }
    // Hosted: the controller's per-BURST row-buffer classification.
    [[nodiscard]] size_t row_hits() const { return bridge_ ? bridge_->stats().page_hits : row_hits_; }
    [[nodiscard]] size_t row_misses() const { return bridge_ ? bridge_->stats().page_conflicts : row_misses_; }
    [[nodiscard]] size_t row_empty_accesses() const { return bridge_ ? bridge_->stats().page_empty : row_empty_; }

    [[nodiscard]] double row_hit_rate() const {
        size_t total = row_hits() + row_misses() + row_empty_accesses();
        return total > 0 ? static_cast<double>(row_hits()) / static_cast<double>(total) : 0.0;
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

    // Hosted mode: tiles in flight, by request id, in arrival order. A tile's bursts are fed to
    // the controller in order, tiles first-come first-served; the controller schedules them.
    struct HostedTile {
        PendingRequest req;
        std::uint64_t first = 0;        // first burst's address (burst-aligned)
        std::uint64_t bursts = 0, fed = 0, done = 0;
        Cycle start = 0;
        std::uint32_t bank = 0;         // flat bank of the first burst, for CompletedTransfer
    };
    std::unique_ptr<DramBridge> bridge_;
    std::unordered_map<std::uint32_t, HostedTile> hosted_;
    std::deque<std::uint32_t> hosted_order_;
    std::vector<std::uint64_t> burst_done_;
    // Window bursts (submit_burst): their own tag space, so the tile path's ids never collide.
    static constexpr std::uint64_t kBurstTag = std::uint64_t{1} << 63;
    std::uint64_t next_burst_id_ = 0;
    std::unordered_map<std::uint64_t, BurstDone> bursts_;     // in the controller
    std::deque<BurstDone> bursts_done_;                       // finished, for submitters to poll

    // Recording
    std::vector<BurstRecord> burst_records_;
    std::unordered_map<std::uint64_t, std::size_t> burst_index_;   // burst id -> record
    std::vector<DramBridge::Command> command_records_;
    bool recording_ = false;
    // Installed on first use, not in the constructor: the sink captures `this`, so it must be
    // installed where the process will stay.
    void ensure_recording() {
        if (!config_.record || recording_ || !bridge_) return;
        recording_ = true;
        bridge_->set_command_sink([this](const DramBridge::Command& c) {
            command_records_.push_back(c);
            if (!c.tag || !(*c.tag & kBurstTag)) return;
            auto it = burst_index_.find(*c.tag & ~kBurstTag);
            if (it == burst_index_.end()) return;
            BurstRecord& r = burst_records_[it->second];
            if (!r.commanded) {
                r.commanded = true;
                r.first_command = c.issue;
            }
            if (c.kind == DramBridge::Command::Kind::Read || c.kind == DramBridge::Command::Kind::Write) {
                r.data_start = c.data_start;
                r.data_end = c.data_end;
                r.outcome = c.conflicted ? PageOutcome::Conflict
                          : c.activated  ? PageOutcome::Empty
                                         : PageOutcome::Hit;
            }
        });
    }

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
    // Hosted mode
    // ========================================================================

    bool submit_hosted(const TileDescriptor& tile, bool is_load, uint32_t submitter_id) {
        if (hosted_.size() >= config_.request_queue_depth) return false;   // back-pressure
        const auto& map = bridge_->map();
        const std::uint64_t b = map.burst_bytes();
        if (tile.dram_address + tile.size_bytes > map.capacity())
            throw std::out_of_range(
                config_.name + ": tile " + std::to_string(tile.dram_address) + " + " +
                std::to_string(tile.size_bytes) + " B runs past the declared DRAM (" +
                std::to_string(map.capacity()) + " B)");
        HostedTile h;
        h.req.tile = tile;
        h.req.is_load = is_load;
        h.req.enqueue_cycle = current_cycle_;
        h.req.request_id = next_request_id_++;
        h.req.submitter_id = submitter_id;
        h.first = tile.dram_address / b * b;
        const std::uint64_t end = (tile.dram_address + tile.size_bytes + b - 1) / b * b;
        h.bursts = (end - h.first) / b;
        h.bank = h.bursts ? map.flat_bank(map.decode(h.first)) : 0;
        hosted_order_.push_back(h.req.request_id);
        hosted_.emplace(h.req.request_id, std::move(h));
        return true;
    }

    void tick_hosted(Cycle now, std::vector<TimingEvent>& events) {
        ensure_recording();
        const std::uint64_t b = bridge_->map().burst_bytes();
        // Feed bursts, oldest tile first, until the controller's queue refuses one.
        bool full = false;
        for (std::uint32_t id : hosted_order_) {
            HostedTile& h = hosted_.at(id);
            while (h.fed < h.bursts) {
                if (!bridge_->submit(h.first + h.fed * b, h.req.is_load, id)) { full = true; break; }
                if (h.fed++ == 0) {
                    h.start = now;
                    auto e = TimingEvent(h.req.is_load ? EventType::DMA_LOAD_START
                                                       : EventType::DMA_STORE_START,
                                         now, config_.controller_id, h.req.tile.tile_id, name());
                    e.matrix_base_address = h.req.tile.matrix_base_address;
                    e.dram_address = h.req.tile.dram_address;
                    events.push_back(e);
                }
            }
            if (full) break;
        }
        if (full) ++stall_cycles_bank_;

        // Time passes in the controller's clock; collect the bursts that finished.
        burst_done_.clear();
        bridge_->advance(now, burst_done_);
        for (std::uint64_t tag : burst_done_) {
            if (tag & kBurstTag) {
                auto it = bursts_.find(tag & ~kBurstTag);
                bursts_done_.push_back(it->second);
                bursts_.erase(it);
                total_bytes_transferred_ += b;
                if (config_.record) {
                    BurstRecord& r = burst_records_[burst_index_.at(tag & ~kBurstTag)];
                    r.done = now;
                    r.finished = true;
                }
                continue;
            }
            ++hosted_.at(static_cast<std::uint32_t>(tag)).done;
        }

        // A tile completes with its last burst (a zero-byte tile, on its first tick).
        for (auto it = hosted_order_.begin(); it != hosted_order_.end();) {
            HostedTile& h = hosted_.at(*it);
            if (h.done < h.bursts || h.fed < h.bursts) { ++it; continue; }
            if (h.bursts == 0) h.start = now;
            CompletedTransfer ct;
            ct.tile = h.req.tile;
            ct.is_load = h.req.is_load;
            ct.start_cycle = h.start;
            ct.complete_cycle = now;
            ct.bank_id = h.bank;
            // Row-buffer state is per burst in this model; see row_hits() and friends.
            ct.access_type = MemoryAccessType::ROW_EMPTY;
            ct.submitter_id = h.req.submitter_id;
            completed_transfers_.push_back(ct);
            total_bytes_transferred_ += h.req.tile.size_bytes;
            auto e = TimingEvent::duration_event(
                h.req.is_load ? EventType::DMA_LOAD_COMPLETE : EventType::DMA_STORE_COMPLETE,
                h.start, now - h.start, config_.controller_id, h.req.tile.tile_id, name());
            e.matrix_base_address = h.req.tile.matrix_base_address;
            e.dram_address = h.req.tile.dram_address;
            events.push_back(e);
            hosted_.erase(*it);
            it = hosted_order_.erase(it);
        }
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
