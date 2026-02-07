// ============================================================================
// include/sw/kpu/timing/streamer_process.hpp
// Streamer Process for CSP-style concurrent timing simulation
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
#include <limits>
#include <optional>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Streamer type for row vs column streaming
 */
enum class StreamerType {
    ROW_STREAMER,  ///< West edge - streams rows of A matrix
    COL_STREAMER   ///< North edge - streams columns of B matrix
};

/**
 * @brief Streamer Process for L2 ↔ L1/Compute transfers
 *
 * The Streamer handles:
 * - Feeding tiles from L2 to L1/Compute (requires tile in L2)
 * - Draining result tiles from Compute to L2 (requires L2 credit)
 *
 * Credit flow:
 * - FEED (L2→Compute): Needs tile in L2 TagCAM, releases L2 credit when consumed
 * - DRAIN (Compute→L2): Acquires L2 credit, inserts result into L2 TagCAM
 *
 * Unlike DMA (which has multiple in-flight), Streamer handles one transfer at a time.
 * Multiple Streamers can operate concurrently (e.g., West and North edges).
 */
class StreamerProcess : public IProcess {
public:
    /**
     * @brief Streamer configuration
     */
    struct Config {
        uint32_t streamer_id = 0;      ///< Unique streamer identifier
        StreamerType type = StreamerType::ROW_STREAMER;  ///< Streamer type
        GridPosition compute_tile_pos; ///< Compute tile grid position
        Size bus_width_bytes = 64;     ///< Bus width in bytes
        Cycle startup_latency = 2;     ///< Cycles to start a transfer
        size_t l1_depth = 4;           ///< L1 buffer depth (for double-buffering)
        double bandwidth_gbps = 102.4; ///< L2↔L1 bandwidth (internal, very high)
        double clock_ghz = 1.0;        ///< Reference clock in GHz
        bool priority_aging = false;   ///< If true, prefer oldest ready tile (prevents starvation)
        std::string name = "STR";      ///< Human-readable name

        /// Generate human-readable name: "CT(0,0):RowSTR" or "CT(0,1):ColSTR" format
        std::string display_name() const {
            std::string type_str = (type == StreamerType::ROW_STREAMER) ? "RowSTR" : "ColSTR";
            return "CT" + compute_tile_pos.to_string() + ":" + type_str;
        }
    };

    /**
     * @brief Construct a Streamer process
     * @param config Streamer configuration
     * @param l2_tag_cam Tag CAM for L2 tile tracking (shared)
     * @param l2_credits Credit pool for L2 banks (shared)
     * @param compute_result_tag_cam Tag CAM for compute results (shared) - DRAIN waits for this
     */
    StreamerProcess(const Config& config,
                    TagCAM& l2_tag_cam,
                    CreditPool& l2_credits,
                    TagCAM& compute_result_tag_cam)
        : config_(config),
          l2_tag_cam_(l2_tag_cam),
          l2_credits_(l2_credits),
          compute_result_tag_cam_(compute_result_tag_cam) {
        // Compute cycles per byte based on bandwidth and clock
        double bytes_per_cycle = config_.bandwidth_gbps / config_.clock_ghz;
        cycles_per_byte_ = 1.0 / bytes_per_cycle;
    }

    /**
     * @brief Schedule a tile feed from L2 to Compute
     * @param tile Tile descriptor
     *
     * The tile will be fed when:
     * 1. The tile is present in L2 (TagCAM match)
     * 2. No other transfer is in progress
     */
    void schedule_feed(const TileDescriptor& tile) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        feed_queue_.enqueue(t);
    }

    /**
     * @brief Schedule a result tile drain from Compute to L2
     * @param tile Tile descriptor
     *
     * The result will be drained when:
     * 1. An L2 credit is available
     * 2. No other transfer is in progress
     */
    void schedule_drain(const TileDescriptor& tile) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        drain_queue_.enqueue(t);
    }

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfer
        check_completion(current_cycle, events);

        // Step 2: If idle, try to start new work
        if (!in_flight_.has_value()) {
            // Priority: feeds over drains (keep compute fed)
            if (!try_start_feed(current_cycle, events)) {
                try_start_drain(current_cycle, events);
            }
        }

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        return !in_flight_.has_value();
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !feed_queue_.empty() || !drain_queue_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.streamer_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name;  // Uses display_name() set during creation
    }

    void reset() override {
        feed_queue_.reset();
        drain_queue_.reset();
        in_flight_.reset();
        stall_cycles_tag_ = 0;
        stall_cycles_credit_ = 0;
        stall_cycles_compute_ = 0;
        total_tiles_fed_ = 0;
        total_tiles_drained_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t feed_queue_depth() const {
        return feed_queue_.size();
    }

    [[nodiscard]] size_t drain_queue_depth() const {
        return drain_queue_.size();
    }

    [[nodiscard]] Cycle stall_cycles_tag() const {
        return stall_cycles_tag_;
    }

    [[nodiscard]] Cycle stall_cycles_credit() const {
        return stall_cycles_credit_;
    }

    [[nodiscard]] Cycle stall_cycles_compute() const {
        return stall_cycles_compute_;
    }

    [[nodiscard]] size_t total_tiles_fed() const {
        return total_tiles_fed_;
    }

    [[nodiscard]] size_t total_tiles_drained() const {
        return total_tiles_drained_;
    }

    [[nodiscard]] const Config& config() const {
        return config_;
    }

    [[nodiscard]] StreamerType streamer_type() const {
        return config_.type;
    }

private:
    Config config_;
    TagCAM& l2_tag_cam_;
    CreditPool& l2_credits_;
    TagCAM& compute_result_tag_cam_;  ///< DRAIN waits for result in this TagCAM

    WorkQueue<TileDescriptor> feed_queue_;
    WorkQueue<TileDescriptor> drain_queue_;
    std::optional<InFlightTransfer> in_flight_;

    Cycle current_cycle_ = 0;
    double cycles_per_byte_ = 0.01;  // ~100 bytes/cycle at 102.4 GB/s @ 1GHz

    // For in-flight tracking
    bool in_flight_is_feed_ = true;
    uint32_t in_flight_l2_slot_ = 0;

    // Statistics
    Cycle stall_cycles_tag_ = 0;
    Cycle stall_cycles_credit_ = 0;
    Cycle stall_cycles_compute_ = 0;
    size_t total_tiles_fed_ = 0;
    size_t total_tiles_drained_ = 0;

    /**
     * @brief Compute transfer time in cycles
     */
    [[nodiscard]] Cycle compute_transfer_cycles(Size bytes) const {
        Cycle data_cycles = static_cast<Cycle>(std::ceil(bytes * cycles_per_byte_));
        return config_.startup_latency + data_cycles;
    }

    /**
     * @brief Check for and process completed transfer
     */
    void check_completion(Cycle current_cycle, std::vector<TimingEvent>& events) {
        if (!in_flight_.has_value()) return;

        if (in_flight_->is_complete(current_cycle)) {
            if (in_flight_is_feed_) {
                // Feed complete: tile consumed from L2, fed to compute
                // Only release L2 credit if tile was fully removed (ref_count reached 0)
                bool credit_released = l2_tag_cam_.invalidate(in_flight_->tile.tile_id);
                if (credit_released) {
                    l2_credits_.release();
                }
                total_tiles_fed_++;

                events.push_back(TimingEvent::duration_event(
                    EventType::STR_FEED_COMPLETE,
                    in_flight_->start_cycle,
                    in_flight_->duration(),
                    config_.streamer_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_l2_slot_;

                events.push_back(TimingEvent(
                    EventType::TILE_FED_TO_COMPUTE,
                    current_cycle,
                    config_.streamer_id,
                    in_flight_->tile.tile_id,
                    name()
                ));

                if (credit_released) {
                    events.push_back(TimingEvent(
                        EventType::CREDIT_RELEASED,
                        current_cycle,
                        config_.streamer_id,
                        in_flight_->tile.tile_id,
                        name()
                    ));
                }
            } else {
                // Drain complete: result tile arrived at L2
                // Note: slot_id was set when drain started
                l2_tag_cam_.insert(in_flight_->tile.tile_id, in_flight_->slot_id, current_cycle);
                total_tiles_drained_++;

                events.push_back(TimingEvent::duration_event(
                    EventType::STR_DRAIN_COMPLETE,
                    in_flight_->start_cycle,
                    in_flight_->duration(),
                    config_.streamer_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;

                events.push_back(TimingEvent(
                    EventType::TILE_DRAINED,
                    current_cycle,
                    config_.streamer_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;
            }

            in_flight_.reset();
        }
    }

    /**
     * @brief Try to start a feed transfer (L2→Compute) using work-conserving scan
     * @return true if a feed was started
     *
     * Work-conserving: Scan queue for any tile ready in L2, not just the head.
     * Priority aging: When enabled, prefers oldest ready tile to prevent starvation.
     */
    bool try_start_feed(Cycle current_cycle, std::vector<TimingEvent>& events) {
        if (feed_queue_.empty()) return false;

        // Work-conserving scan: find any tile that's ready in L2
        // With priority aging: find the oldest ready tile
        std::optional<TagCAMEntry> found_entry;
        size_t found_index = 0;
        Cycle oldest_enqueue = std::numeric_limits<Cycle>::max();

        for (size_t i = 0; i < feed_queue_.size(); ++i) {
            const auto& tile = feed_queue_.at(i);
            auto l2_entry = l2_tag_cam_.match(tile.tile_id);
            if (l2_entry.has_value()) {
                if (config_.priority_aging) {
                    // Priority aging: prefer oldest tile
                    if (tile.enqueue_cycle < oldest_enqueue) {
                        found_entry = l2_entry;
                        found_index = i;
                        oldest_enqueue = tile.enqueue_cycle;
                    }
                } else {
                    // Default: first ready tile in queue order
                    found_entry = l2_entry;
                    found_index = i;
                    break;
                }
            }
        }

        // No tile in queue is ready in L2
        if (!found_entry.has_value()) {
            const auto& head_tile = feed_queue_.peek();
            events.push_back(TimingEvent(
                EventType::STR_STALL_TAG,
                current_cycle,
                config_.streamer_id,
                head_tile.tile_id,
                name()
            ));
            stall_cycles_tag_++;
            return false;
        }

        // Start the feed - remove from queue at found position
        TileDescriptor feed_tile = feed_queue_.remove_at(found_index);
        Cycle transfer_cycles = compute_transfer_cycles(feed_tile.size_bytes);

        in_flight_ = InFlightTransfer(
            feed_tile,
            current_cycle,
            current_cycle + transfer_cycles,
            found_entry->slot_id,
            true  // is_feed
        );
        in_flight_is_feed_ = true;
        in_flight_l2_slot_ = found_entry->slot_id;

        events.push_back(TimingEvent(
            EventType::STR_FEED_START,
            current_cycle,
            config_.streamer_id,
            feed_tile.tile_id,
            name()
        ));
        events.back().slot_id = found_entry->slot_id;

        return true;
    }

    /**
     * @brief Try to start a drain transfer (Compute→L2)
     * @return true if a drain was started
     *
     * DRAIN requires:
     * 1. Result tile is ready in compute (COMPUTE operation completed)
     * 2. L2 credit is available
     */
    bool try_start_drain(Cycle current_cycle, std::vector<TimingEvent>& events) {
        if (drain_queue_.empty()) return false;

        const TileDescriptor& tile = drain_queue_.peek();

        // First check: Is the compute result ready?
        auto compute_entry = compute_result_tag_cam_.match(tile.tile_id);
        if (!compute_entry.has_value()) {
            // Result not yet ready - still computing
            events.push_back(TimingEvent(
                EventType::STR_STALL_COMPUTE,
                current_cycle,
                config_.streamer_id,
                tile.tile_id,
                name()
            ));
            stall_cycles_compute_++;
            return false;
        }

        // Second check: Is L2 credit available?
        if (!l2_credits_.acquire()) {
            events.push_back(TimingEvent(
                EventType::STR_STALL_CREDIT,
                current_cycle,
                config_.streamer_id,
                tile.tile_id,
                name()
            ));
            stall_cycles_credit_++;
            return false;
        }

        // Consume the compute result (remove from compute_result_tag_cam)
        compute_result_tag_cam_.invalidate(tile.tile_id);

        // Start the drain
        TileDescriptor drain_tile = drain_queue_.dequeue();
        Cycle transfer_cycles = compute_transfer_cycles(drain_tile.size_bytes);
        uint32_t l2_slot = allocate_l2_slot();

        in_flight_ = InFlightTransfer(
            drain_tile,
            current_cycle,
            current_cycle + transfer_cycles,
            l2_slot,
            false  // is_drain
        );
        in_flight_is_feed_ = false;

        events.push_back(TimingEvent(
            EventType::STR_DRAIN_START,
            current_cycle,
            config_.streamer_id,
            drain_tile.tile_id,
            name()
        ));
        events.back().slot_id = l2_slot;

        events.push_back(TimingEvent(
            EventType::CREDIT_ACQUIRED,
            current_cycle,
            config_.streamer_id,
            drain_tile.tile_id,
            name()
        ));

        return true;
    }

    /**
     * @brief Allocate an L2 bank slot (round-robin)
     */
    uint32_t allocate_l2_slot() {
        static uint32_t next_slot = 0;
        uint32_t slot = next_slot;
        next_slot = (next_slot + 1) % static_cast<uint32_t>(l2_tag_cam_.capacity());
        return slot;
    }
};

} // namespace sw::kpu::timing
