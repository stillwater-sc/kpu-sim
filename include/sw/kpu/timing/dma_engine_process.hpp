// ============================================================================
// include/sw/kpu/timing/dma_engine_process.hpp
// DMA Engine Process for CSP-style concurrent timing simulation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/process_interface.hpp>
#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>
#include <sw/kpu/timing/work_queue.hpp>

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
 * Credit flow:
 * - LOAD: Must acquire L3 credit before starting (buffer space needed)
 * - STORE: Must match tile in L3 TagCAM, then invalidate and release credit
 *
 * Concurrency:
 * - Supports multiple outstanding transfers (queue_depth)
 * - Each DMA engine is independent
 * - Multiple engines can be used for A, B, C matrices
 */
class DMAEngineProcess : public IProcess {
public:
    /**
     * @brief DMA Engine configuration
     */
    struct Config {
        uint32_t engine_id = 0;        ///< Unique engine identifier
        Size bus_width_bytes = 64;     ///< Bus width in bytes (64B = 512-bit)
        Cycle startup_latency = 10;    ///< Cycles to start a transfer
        double bandwidth_gbps = 25.6;  ///< Bandwidth per channel in GB/s
        double clock_ghz = 1.0;        ///< Reference clock in GHz
        size_t queue_depth = 8;        ///< Max outstanding transfers
        std::string name = "DMA";      ///< Human-readable name
    };

    /**
     * @brief Construct a DMA engine process
     * @param config Engine configuration
     * @param l3_credits Credit pool for L3 buffers (shared)
     * @param l3_tag_cam Tag CAM for L3 tile tracking (shared)
     * @param l2_tag_cam Tag CAM for L2 tile tracking (for downstream check)
     */
    DMAEngineProcess(const Config& config,
                     CreditPool& l3_credits,
                     TagCAM& l3_tag_cam,
                     TagCAM& l2_tag_cam)
        : config_(config),
          l3_credits_(l3_credits),
          l3_tag_cam_(l3_tag_cam),
          l2_tag_cam_(l2_tag_cam),
          load_queue_(config.queue_depth * 4),  // Allow some buffering
          store_queue_(config.queue_depth * 4),
          next_slot_id_(0) {
        // Compute cycles per byte based on bandwidth and clock
        // bandwidth_gbps * 1e9 bytes/sec / (clock_ghz * 1e9 cycles/sec) = bytes/cycle
        double bytes_per_cycle = config_.bandwidth_gbps / config_.clock_ghz;
        cycles_per_byte_ = 1.0 / bytes_per_cycle;
    }

    /**
     * @brief Schedule a tile load from DRAM to L3
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be loaded when:
     * 1. An L3 credit is available
     * 2. Queue depth allows another in-flight transfer
     */
    void schedule_load(const TileDescriptor& tile) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        load_queue_.enqueue(t);
    }

    /**
     * @brief Schedule a tile store from L3 to DRAM
     * @param tile Tile descriptor with DRAM address and size
     *
     * The tile will be stored when:
     * 1. The tile is present in L3 (TagCAM match)
     * 2. Queue depth allows another in-flight transfer
     */
    void schedule_store(const TileDescriptor& tile) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        store_queue_.enqueue(t);
    }

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfers
        check_completions(current_cycle, events);

        // Step 2: Try to issue new loads
        try_issue_loads(current_cycle, events);

        // Step 3: Try to issue new stores
        try_issue_stores(current_cycle, events);

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        return in_flight_.empty();
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !load_queue_.empty() || !store_queue_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.engine_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name + "_" + std::to_string(config_.engine_id);
    }

    void reset() override {
        load_queue_.reset();
        store_queue_.reset();
        in_flight_.clear();
        next_slot_id_ = 0;
        stall_cycles_ = 0;
        total_bytes_loaded_ = 0;
        total_bytes_stored_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t in_flight_count() const {
        return in_flight_.size();
    }

    [[nodiscard]] size_t load_queue_depth() const {
        return load_queue_.size();
    }

    [[nodiscard]] size_t store_queue_depth() const {
        return store_queue_.size();
    }

    [[nodiscard]] Cycle stall_cycles() const {
        return stall_cycles_;
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
    CreditPool& l3_credits_;
    TagCAM& l3_tag_cam_;
    TagCAM& l2_tag_cam_;

    WorkQueue<TileDescriptor> load_queue_;
    WorkQueue<TileDescriptor> store_queue_;
    std::vector<InFlightTransfer> in_flight_;

    Cycle current_cycle_ = 0;
    uint32_t next_slot_id_ = 0;
    double cycles_per_byte_ = 0.04;  // ~25 bytes/cycle at 25.6 GB/s @ 1GHz

    // Statistics
    Cycle stall_cycles_ = 0;
    size_t total_bytes_loaded_ = 0;
    size_t total_bytes_stored_ = 0;

    /**
     * @brief Compute transfer time in cycles
     */
    [[nodiscard]] Cycle compute_transfer_cycles(Size bytes) const {
        // Transfer time = startup + data transfer
        Cycle data_cycles = static_cast<Cycle>(std::ceil(bytes * cycles_per_byte_));
        return config_.startup_latency + data_cycles;
    }

    /**
     * @brief Check for and process completed transfers
     */
    void check_completions(Cycle current_cycle, std::vector<TimingEvent>& events) {
        auto it = in_flight_.begin();
        while (it != in_flight_.end()) {
            if (it->is_complete(current_cycle)) {
                if (it->is_load) {
                    // Load complete: tile arrived at L3
                    // TagCAM now supports reference counting for duplicate tiles
                    l3_tag_cam_.insert(it->tile.tile_id, it->slot_id, current_cycle);
                    total_bytes_loaded_ += it->tile.size_bytes;

                    events.push_back(TimingEvent::duration_event(
                        EventType::DMA_LOAD_COMPLETE,
                        it->start_cycle,
                        it->duration(),
                        config_.engine_id,
                        it->tile.tile_id,
                        name()
                    ));
                    events.back().slot_id = it->slot_id;

                    events.push_back(TimingEvent(
                        EventType::TILE_ARRIVED_L3,
                        current_cycle,
                        config_.engine_id,
                        it->tile.tile_id,
                        name()
                    ));
                    events.back().slot_id = it->slot_id;
                } else {
                    // Store complete: tile written to DRAM
                    total_bytes_stored_ += it->tile.size_bytes;

                    events.push_back(TimingEvent::duration_event(
                        EventType::DMA_STORE_COMPLETE,
                        it->start_cycle,
                        it->duration(),
                        config_.engine_id,
                        it->tile.tile_id,
                        name()
                    ));
                }
                it = in_flight_.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * @brief Try to issue new load transfers
     */
    void try_issue_loads(Cycle current_cycle, std::vector<TimingEvent>& events) {
        while (!load_queue_.empty() && in_flight_.size() < config_.queue_depth) {
            const TileDescriptor& peek_tile = load_queue_.peek();

            // Check if tile is already in L3 (supports tile reuse)
            if (l3_tag_cam_.lookup(peek_tile.tile_id)) {
                // Tile already in L3 - just increment ref_count, no credit needed
                TileDescriptor tile = load_queue_.dequeue();
                auto entry = l3_tag_cam_.match(tile.tile_id);
                // Insert increments ref_count for existing tiles
                l3_tag_cam_.insert(tile.tile_id, entry->slot_id, current_cycle);

                events.push_back(TimingEvent(
                    EventType::TILE_ARRIVED_L3,  // Tile already there
                    current_cycle,
                    config_.engine_id,
                    tile.tile_id,
                    name()
                ));
                events.back().slot_id = entry->slot_id;
                continue;  // Process next tile
            }

            // Tile not in L3 - need to load it from DRAM
            // Need L3 credit to proceed
            if (!l3_credits_.acquire()) {
                // Stalled on credit
                events.push_back(TimingEvent(
                    EventType::DMA_STALL_CREDIT,
                    current_cycle,
                    config_.engine_id,
                    peek_tile.tile_id,
                    name()
                ));
                stall_cycles_++;
                break;
            }

            // Got credit - issue the load
            TileDescriptor tile = load_queue_.dequeue();
            Cycle transfer_cycles = compute_transfer_cycles(tile.size_bytes);
            uint32_t slot = allocate_slot();

            in_flight_.emplace_back(
                tile,
                current_cycle,
                current_cycle + transfer_cycles,
                slot,
                true  // is_load
            );

            events.push_back(TimingEvent(
                EventType::DMA_LOAD_START,
                current_cycle,
                config_.engine_id,
                tile.tile_id,
                name()
            ));
            events.back().slot_id = slot;

            events.push_back(TimingEvent(
                EventType::CREDIT_ACQUIRED,
                current_cycle,
                config_.engine_id,
                tile.tile_id,
                name()
            ));
        }
    }

    /**
     * @brief Try to issue new store transfers
     */
    void try_issue_stores(Cycle current_cycle, std::vector<TimingEvent>& events) {
        while (!store_queue_.empty() && in_flight_.size() < config_.queue_depth) {
            const TileDescriptor& tile = store_queue_.peek();

            // Need tile to be in L3 (TagCAM match)
            auto entry = l3_tag_cam_.match(tile.tile_id);
            if (!entry.has_value()) {
                // Stalled on tag match - tile not yet in L3
                events.push_back(TimingEvent(
                    EventType::DMA_STALL_CREDIT,  // Using same event, could add DMA_STALL_TAG
                    current_cycle,
                    config_.engine_id,
                    tile.tile_id,
                    name()
                ));
                stall_cycles_++;
                break;
            }

            // Tile is in L3 - start the store
            TileDescriptor store_tile = store_queue_.dequeue();
            Cycle transfer_cycles = compute_transfer_cycles(store_tile.size_bytes);
            uint32_t slot = entry->slot_id;

            // Invalidate tile in L3 (it's being written back)
            // Only release L3 credit if tile was fully removed (ref_count reached 0)
            bool credit_released = l3_tag_cam_.invalidate(store_tile.tile_id);
            if (credit_released) {
                l3_credits_.release();
            }

            in_flight_.emplace_back(
                store_tile,
                current_cycle,
                current_cycle + transfer_cycles,
                slot,
                false  // is_store
            );

            events.push_back(TimingEvent(
                EventType::DMA_STORE_START,
                current_cycle,
                config_.engine_id,
                store_tile.tile_id,
                name()
            ));

            if (credit_released) {
                events.push_back(TimingEvent(
                    EventType::CREDIT_RELEASED,
                    current_cycle,
                    config_.engine_id,
                    store_tile.tile_id,
                    name()
                ));
                events.back().slot_id = slot;
            }
        }
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
