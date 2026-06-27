// ============================================================================
// include/sw/kpu/timing/dma_engine_process.hpp
// DMA Engine Process for CSP-style concurrent timing simulation
//
// The DMA Engine is a programmable ISA-driven process that:
// - Executes data movement operations (LOAD/STORE tiles)
// - Manages L3 credit acquisition and release
// - Tracks tile arrivals in L3 TagCAM
// - Uses a Memory Controller for actual DRAM access contention
//
// Architecture:
//   DMA Engine (CSP Process) --uses--> Memory Controller (Resource)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/process_interface.hpp>
#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>
#include <sw/kpu/timing/work_queue.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>

#include <cmath>
#include <optional>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief DMA Engine Process for DRAM ↔ L3 transfers
 *
 * The DMA engine handles:
 * - Loading tiles from DRAM to L3 (requires L3 credit)
 * - Storing tiles from L3 to DRAM (requires tile in L3 TagCAM)
 *
 * The DMA engine uses a Memory Controller for actual DRAM access. Multiple
 * DMA engines can share a single MC (realistic for multi-channel configs).
 *
 * Credit flow:
 * - LOAD: Must acquire L3 credit before submitting to MC (buffer space needed)
 * - STORE: Must match tile in L3 TagCAM before submitting to MC
 *
 * Concurrency:
 * - DMA engine queues requests, MC processes them with contention
 * - Multiple DMA engines can share one MC
 */
class DMAEngineProcess : public IProcess {
public:
    /**
     * @brief DMA Engine configuration
     */
    struct Config {
        uint32_t engine_id = 0;        ///< Unique engine identifier
        size_t queue_depth = 32;       ///< Max pending requests in queue
        std::string name = "DMA";      ///< Human-readable name

        /// Generate human-readable name
        std::string display_name() const {
            return "DMA" + std::to_string(engine_id);
        }
    };

    /**
     * @brief Pending request state
     */
    enum class RequestState {
        WAITING_CREDIT,   ///< Load waiting for L3 credit
        WAITING_TAG,      ///< Store waiting for tile in L3
        SUBMITTED,        ///< Submitted to MC, waiting for completion
        COMPLETED         ///< Completed (will be removed)
    };

    /**
     * @brief Pending request tracking
     */
    struct PendingRequest {
        TileDescriptor tile;
        bool is_load;
        RequestState state;
        Cycle enqueue_cycle;
        uint32_t slot_id = 0;    ///< L3 slot for this tile (for loads)
    };

    /**
     * @brief Construct a DMA engine process
     * @param config Engine configuration
     * @param mc Reference to shared Memory Controller
     * @param l3_credits Credit pool for L3 buffers (shared)
     * @param l3_tag_cam Tag CAM for L3 tile tracking (shared)
     */
    DMAEngineProcess(const Config& config,
                     MemoryControllerProcess& mc,
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam)
        : config_(config),
          mc_(mc),
          l3_credits_(l3_credits),
          l3_tag_cam_(l3_tag_cam),
          next_slot_id_(0) {
    }

    /**
     * @brief Schedule a tile load from DRAM to L3
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be loaded when:
     * 1. An L3 credit is available
     * 2. MC can accept the request
     */
    void schedule_load(const TileDescriptor& tile) {
        if (pending_requests_.size() >= config_.queue_depth) {
            return;  // Queue full
        }

        PendingRequest req;
        req.tile = tile;
        req.is_load = true;
        req.state = RequestState::WAITING_CREDIT;
        req.enqueue_cycle = current_cycle_;

        pending_requests_.push_back(req);
    }

    /**
     * @brief Schedule a tile store from L3 to DRAM
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be stored when:
     * 1. The tile is present in L3 (TagCAM match)
     * 2. MC can accept the request
     */
    void schedule_store(const TileDescriptor& tile) {
        if (pending_requests_.size() >= config_.queue_depth) {
            return;  // Queue full
        }

        PendingRequest req;
        req.tile = tile;
        req.is_load = false;
        req.state = RequestState::WAITING_TAG;
        req.enqueue_cycle = current_cycle_;

        pending_requests_.push_back(req);
    }

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfers from MC
        process_mc_completions(events);

        // Step 2: Try to acquire credits and submit new loads
        process_pending_loads(events);

        // Step 3: Try to match tags and submit new stores
        process_pending_stores(events);

        // Step 4: Remove completed requests
        remove_completed_requests();

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        // DMA is idle if no requests are submitted to MC
        bool has_submitted = false;
        for (const auto& req : pending_requests_) {
            if (req.state == RequestState::SUBMITTED) {
                has_submitted = true;
                break;
            }
        }
        return !has_submitted;
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !pending_requests_.empty();
    }

    /**
     * @brief Check if DMA engine is complete (no pending or in-flight work)
     */
    [[nodiscard]] bool is_complete() const override {
        return pending_requests_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.engine_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name;
    }

    void reset() override {
        pending_requests_.clear();
        next_slot_id_ = 0;
        stall_cycles_credit_ = 0;
        stall_cycles_tag_ = 0;
        total_bytes_loaded_ = 0;
        total_bytes_stored_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t pending_count() const {
        return pending_requests_.size();
    }

    [[nodiscard]] size_t submitted_count() const {
        size_t count = 0;
        for (const auto& req : pending_requests_) {
            if (req.state == RequestState::SUBMITTED) count++;
        }
        return count;
    }

    [[nodiscard]] Cycle stall_cycles_credit() const {
        return stall_cycles_credit_;
    }

    [[nodiscard]] Cycle stall_cycles_tag() const {
        return stall_cycles_tag_;
    }

    [[nodiscard]] Cycle stall_cycles() const {
        return stall_cycles_credit_ + stall_cycles_tag_;
    }

    [[nodiscard]] size_t total_bytes_loaded() const {
        return total_bytes_loaded_;
    }

    [[nodiscard]] size_t total_bytes_stored() const {
        return total_bytes_stored_;
    }

    [[nodiscard]] const Config& config() const {
        return config_;
    }

private:
    Config config_;
    MemoryControllerProcess& mc_;
    CreditPool& l3_credits_;
    TagCAM& l3_tag_cam_;

    std::vector<PendingRequest> pending_requests_;

    Cycle current_cycle_ = 0;
    uint32_t next_slot_id_ = 0;

    // Statistics
    Cycle stall_cycles_credit_ = 0;
    Cycle stall_cycles_tag_ = 0;
    size_t total_bytes_loaded_ = 0;
    size_t total_bytes_stored_ = 0;

    /**
     * @brief Process completed transfers from MC
     */
    void process_mc_completions(std::vector<TimingEvent>& events) {
        // Only poll for our own completions using our engine_id
        while (auto completed = mc_.get_completed_transfer(config_.engine_id)) {
            // Find matching pending request
            for (auto& req : pending_requests_) {
                if (req.state == RequestState::SUBMITTED &&
                    req.tile.tile_id == completed->tile.tile_id &&
                    req.is_load == completed->is_load) {

                    if (completed->is_load) {
                        // Load complete: tile arrived at L3
                        l3_tag_cam_.insert(completed->tile.tile_id, req.slot_id, current_cycle_);
                        total_bytes_loaded_ += completed->tile.size_bytes;

                        events.push_back(TimingEvent(
                            EventType::TILE_ARRIVED_L3,
                            current_cycle_,
                            config_.engine_id,
                            completed->tile.tile_id,
                            name()
                        ));
                    } else {
                        // Store complete: tile written to DRAM
                        total_bytes_stored_ += completed->tile.size_bytes;

                        // Invalidate tile in L3 and release credit
                        bool released = l3_tag_cam_.invalidate(completed->tile.tile_id);
                        if (released) {
                            l3_credits_.release();
                            events.push_back(TimingEvent(
                                EventType::CREDIT_RELEASED,
                                current_cycle_,
                                config_.engine_id,
                                completed->tile.tile_id,
                                name()
                            ));
                        }
                    }

                    req.state = RequestState::COMPLETED;
                    break;
                }
            }
        }
    }

    /**
     * @brief Try to acquire credits and submit pending loads to MC
     */
    void process_pending_loads(std::vector<TimingEvent>& events) {
        for (auto& req : pending_requests_) {
            if (!req.is_load || req.state != RequestState::WAITING_CREDIT) {
                continue;
            }

            // Check if tile is already in L3 (supports tile reuse)
            if (l3_tag_cam_.lookup(req.tile.tile_id)) {
                // Tile already in L3 - just increment ref_count, no credit needed
                auto entry = l3_tag_cam_.match(req.tile.tile_id);
                l3_tag_cam_.insert(req.tile.tile_id, entry->slot_id, current_cycle_);

                events.push_back(TimingEvent(
                    EventType::TILE_ARRIVED_L3,
                    current_cycle_,
                    config_.engine_id,
                    req.tile.tile_id,
                    name()
                ));

                req.state = RequestState::COMPLETED;
                continue;
            }

            // Need L3 credit
            if (!l3_credits_.acquire()) {
                events.push_back(TimingEvent(
                    EventType::DMA_STALL_CREDIT,
                    current_cycle_,
                    config_.engine_id,
                    req.tile.tile_id,
                    name()
                ));
                stall_cycles_credit_++;
                continue;  // Try other requests
            }

            // Got credit - submit to MC with our engine_id
            req.slot_id = allocate_slot();
            if (!mc_.submit_request(req.tile, true, config_.engine_id)) {
                // MC queue full - release credit and retry later
                l3_credits_.release();
                continue;
            }

            req.state = RequestState::SUBMITTED;

            events.push_back(TimingEvent(
                EventType::CREDIT_ACQUIRED,
                current_cycle_,
                config_.engine_id,
                req.tile.tile_id,
                name()
            ));
        }
    }

    /**
     * @brief Try to match tags and submit pending stores to MC
     */
    void process_pending_stores(std::vector<TimingEvent>& events) {
        for (auto& req : pending_requests_) {
            if (req.is_load || req.state != RequestState::WAITING_TAG) {
                continue;
            }

            // Need tile to be in L3 (TagCAM match)
            auto entry = l3_tag_cam_.match(req.tile.tile_id);
            if (!entry.has_value()) {
                events.push_back(TimingEvent(
                    EventType::DMA_STALL_TAG,
                    current_cycle_,
                    config_.engine_id,
                    req.tile.tile_id,
                    name()
                ));
                stall_cycles_tag_++;
                continue;  // Try other requests
            }

            // Tile is in L3 - submit to MC with our engine_id
            if (!mc_.submit_request(req.tile, false, config_.engine_id)) {
                // MC queue full - retry later
                continue;
            }

            req.state = RequestState::SUBMITTED;
        }
    }

    /**
     * @brief Remove completed requests from the list
     */
    void remove_completed_requests() {
        pending_requests_.erase(
            std::remove_if(pending_requests_.begin(), pending_requests_.end(),
                [](const PendingRequest& req) {
                    return req.state == RequestState::COMPLETED;
                }),
            pending_requests_.end()
        );
    }

    /**
     * @brief Allocate a buffer slot (round-robin)
     */
    uint32_t allocate_slot() {
        uint32_t slot = next_slot_id_;
        next_slot_id_ = (next_slot_id_ + 1) % static_cast<uint32_t>(l3_tag_cam_.capacity());
        return slot;
    }
};

} // namespace sw::kpu::timing
