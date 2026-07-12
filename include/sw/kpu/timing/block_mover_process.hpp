// ============================================================================
// include/sw/kpu/timing/block_mover_process.hpp
// BlockMover Process for CSP-style concurrent timing simulation
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
 * @brief BlockMover Process for L3 ↔ L2 transfers
 *
 * The BlockMover handles:
 * - Moving tiles from L3 to L2 (ingress path, requires L2 credit + L3 tag match)
 * - Moving tiles from L2 to L3 (writeback path, requires L3 credit + L2 tag match)
 *
 * Credit flow:
 * - MOVE (L3→L2): Needs tile in L3 TagCAM, acquires L2 credit, releases L3 credit on complete
 * - WRITEBACK (L2→L3): Needs tile in L2 TagCAM, acquires L3 credit, releases L2 credit on complete
 *
 * Unlike DMA (which can have multiple in-flight), BlockMover handles one transfer at a time.
 * Multiple BlockMovers can operate concurrently.
 */
class BlockMoverProcess : public IProcess {
public:
    /**
     * @brief BlockMover configuration
     */
    struct Config {
        uint32_t mover_id = 0;         ///< Unique mover identifier
        GridPosition l3_tile_pos;      ///< L3 memory tile grid position
        Size bus_width_bytes = 64;     ///< Bus width in bytes
        Cycle startup_latency = 4;     ///< Cycles to start a transfer
        double bandwidth_gbps = 51.2;  ///< L3↔L2 bandwidth (internal, higher than DRAM)
        double clock_ghz = 1.0;        ///< Reference clock in GHz
        bool supports_transpose = true; ///< Can transpose during move
        bool priority_aging = false;   ///< If true, prefer oldest ready tile (prevents starvation)
        size_t l2_credit_reserve = 0;  ///< L2 credits moves may NOT consume (reserved for
                                       ///< compute-result drains; prevents credit starvation)
        std::string name = "BM";       ///< Human-readable name

        /// Generate human-readable name: "L3(0,0):BM" format
        std::string display_name() const {
            return "L3" + l3_tile_pos.to_string() + ":BM";
        }
    };

    /**
     * @brief Construct a BlockMover process
     * @param config Mover configuration
     * @param l3_tag_cam Tag CAM for L3 tile tracking (shared)
     * @param l3_credits Credit pool for L3 buffers (shared)
     * @param l2_credits Credit pool for L2 banks (shared)
     * @param l2_tag_cam Tag CAM for L2 tile tracking (shared)
     */
    BlockMoverProcess(const Config& config,
                      TagCAM& l3_tag_cam,
                      CreditPool& l3_credits,
                      CreditPool& l2_credits,
                      TagCAM& l2_tag_cam)
        : config_(config),
          l3_tag_cam_(l3_tag_cam),
          l3_credits_(l3_credits),
          l2_credits_(l2_credits),
          l2_tag_cam_(l2_tag_cam),
          next_l2_slot_(0) {
        // Compute cycles per byte based on bandwidth and clock
        double bytes_per_cycle = config_.bandwidth_gbps / config_.clock_ghz;
        cycles_per_byte_ = 1.0 / bytes_per_cycle;
    }

    /**
     * @brief Schedule a tile move from L3 to L2
     * @param tile Tile descriptor
     * @param transpose Whether to transpose during transfer
     *
     * The tile will be moved when:
     * 1. The tile is present in L3 (TagCAM match)
     * 2. An L2 credit is available
     * 3. No other transfer is in progress
     */
    void schedule_move(const TileDescriptor& tile, bool transpose = false) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        move_queue_.enqueue(t);
        transpose_flags_.push_back(transpose);
    }

    /**
     * @brief Schedule a tile writeback from L2 to L3
     * @param tile Tile descriptor
     *
     * The tile will be written back when:
     * 1. The tile is present in L2 (TagCAM match)
     * 2. An L3 credit is available
     * 3. No other transfer is in progress
     */
    void schedule_writeback(const TileDescriptor& tile) {
        TileDescriptor t = tile;
        t.enqueue_cycle = current_cycle_;
        writeback_queue_.enqueue(t);
    }

    /**
     * @brief Advance simulation by one cycle
     */
    std::vector<TimingEvent> tick(Cycle current_cycle) override {
        current_cycle_ = current_cycle;
        std::vector<TimingEvent> events;

        // Step 1: Check for completed transfer
        check_completion(current_cycle, events);

        // Step 2: Deduplicate moves for tiles already resident in L2
        // (tile reuse - no transfer, no L2 credit)
        process_dedup_moves(current_cycle, events);

        // Step 3: If idle, try to start new work
        if (!in_flight_.has_value()) {
            // Priority: moves over writebacks (keep pipeline fed)
            if (!try_start_move(current_cycle, events)) {
                try_start_writeback(current_cycle, events);
            }
        }

        return events;
    }

    [[nodiscard]] bool is_idle() const override {
        return !in_flight_.has_value();
    }

    [[nodiscard]] bool has_pending_work() const override {
        return !move_queue_.empty() || !writeback_queue_.empty();
    }

    [[nodiscard]] uint32_t id() const override {
        return config_.mover_id;
    }

    [[nodiscard]] std::string name() const override {
        return config_.name;  // Uses display_name() set during creation
    }

    void reset() override {
        move_queue_.reset();
        writeback_queue_.reset();
        transpose_flags_.clear();
        in_flight_.reset();
        next_l2_slot_ = 0;
        stall_cycles_tag_ = 0;
        stall_cycles_credit_ = 0;
        total_tiles_moved_ = 0;
        total_tiles_writeback_ = 0;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    [[nodiscard]] size_t move_queue_depth() const {
        return move_queue_.size();
    }

    [[nodiscard]] size_t writeback_queue_depth() const {
        return writeback_queue_.size();
    }

    [[nodiscard]] Cycle stall_cycles_tag() const {
        return stall_cycles_tag_;
    }

    [[nodiscard]] Cycle stall_cycles_credit() const {
        return stall_cycles_credit_;
    }

    [[nodiscard]] size_t total_tiles_moved() const {
        return total_tiles_moved_;
    }

    [[nodiscard]] size_t total_tiles_writeback() const {
        return total_tiles_writeback_;
    }

    [[nodiscard]] const Config& config() const {
        return config_;
    }

private:
    Config config_;
    TagCAM& l3_tag_cam_;
    CreditPool& l3_credits_;
    CreditPool& l2_credits_;
    TagCAM& l2_tag_cam_;

    WorkQueue<TileDescriptor> move_queue_;
    WorkQueue<TileDescriptor> writeback_queue_;
    std::vector<bool> transpose_flags_;  // Parallel to move_queue
    std::optional<InFlightTransfer> in_flight_;

    Cycle current_cycle_ = 0;
    uint32_t next_l2_slot_ = 0;
    double cycles_per_byte_ = 0.02;  // ~50 bytes/cycle at 51.2 GB/s @ 1GHz

    // For in-flight tracking
    bool in_flight_is_move_ = true;
    uint32_t in_flight_l3_slot_ = 0;

    // Statistics
    Cycle stall_cycles_tag_ = 0;
    Cycle stall_cycles_credit_ = 0;
    size_t total_tiles_moved_ = 0;
    size_t total_tiles_writeback_ = 0;

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
            if (in_flight_is_move_) {
                // Move complete: tile arrived at L2. The ref count is seeded
                // with the tile's consumer count (broadcast 1:1:k, #100);
                // the ordinary pipeline has consumer_count = 1.
                l2_tag_cam_.insert(in_flight_->tile.tile_id, in_flight_->slot_id,
                                   current_cycle,
                                   in_flight_->tile.consumer_count);
                // Only release L3 credit if tile was fully removed (ref_count reached 0)
                bool credit_released = l3_tag_cam_.invalidate(in_flight_->tile.tile_id);
                if (credit_released) {
                    l3_credits_.release(
                        static_cast<size_t>(in_flight_->tile.tile_id.matrix));
                }
                total_tiles_moved_++;

                events.push_back(TimingEvent::duration_event(
                    EventType::BM_MOVE_COMPLETE,
                    in_flight_->start_cycle,
                    in_flight_->duration(),
                    config_.mover_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;

                events.push_back(TimingEvent(
                    EventType::TILE_ARRIVED_L2,
                    current_cycle,
                    config_.mover_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;

                if (credit_released) {
                    events.push_back(TimingEvent(
                        EventType::CREDIT_RELEASED,
                        current_cycle,
                        config_.mover_id,
                        in_flight_->tile.tile_id,
                        name()
                    ));
                }
            } else {
                // Writeback complete: tile arrived at L3
                l3_tag_cam_.insert(in_flight_->tile.tile_id, in_flight_->slot_id, current_cycle);
                // Only release L2 credit if tile was fully removed (ref_count reached 0)
                bool credit_released = l2_tag_cam_.invalidate(in_flight_->tile.tile_id);
                if (credit_released) {
                    l2_credits_.release(
                        static_cast<size_t>(in_flight_->tile.tile_id.matrix));
                }
                total_tiles_writeback_++;

                events.push_back(TimingEvent::duration_event(
                    EventType::BM_WRITEBACK_COMPLETE,
                    in_flight_->start_cycle,
                    in_flight_->duration(),
                    config_.mover_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;

                events.push_back(TimingEvent(
                    EventType::TILE_ARRIVED_L3,
                    current_cycle,
                    config_.mover_id,
                    in_flight_->tile.tile_id,
                    name()
                ));
                events.back().slot_id = in_flight_->slot_id;

                if (credit_released) {
                    events.push_back(TimingEvent(
                        EventType::CREDIT_RELEASED,
                        current_cycle,
                        config_.mover_id,
                        in_flight_->tile.tile_id,
                        name()
                    ));
                }
            }

            in_flight_.reset();
        }
    }

    /**
     * @brief Complete moves for tiles already resident in L2 (tile reuse)
     *
     * The schedule emits one MOVE per tile *use*; the execution layer
     * deduplicates. A duplicate move increments the L2 entry's ref_count
     * (one ref per pending FEED) WITHOUT acquiring a new L2 credit - the
     * entry holds exactly one credit for its lifetime. Acquiring a credit
     * per duplicate move leaks credits (N acquires, 1 release when the
     * ref_count reaches 0) and exhausts the L2 pool on reuse-heavy
     * schedules (issue #61).
     *
     * The L3 side is still consumed per move: each dedup decrements the
     * L3 ref_count and releases the L3 credit when it reaches 0.
     */
    void process_dedup_moves(Cycle current_cycle, std::vector<TimingEvent>& events) {
        size_t i = 0;
        while (i < move_queue_.size()) {
            const auto& tile = move_queue_.at(i);
            auto l2_entry = l2_tag_cam_.match(tile.tile_id);
            auto l3_entry = l3_tag_cam_.match(tile.tile_id);
            // Dedup only when the tile is present on BOTH sides: resident in
            // L2 (reuse target) and arrived in L3 (this move's L3 ref is due)
            if (!l2_entry.has_value() || !l3_entry.has_value()) {
                ++i;
                continue;
            }

            TileDescriptor dedup_tile = move_queue_.remove_at(i);
            if (i < transpose_flags_.size()) {
                transpose_flags_.erase(transpose_flags_.begin() +
                                       static_cast<std::ptrdiff_t>(i));
            }

            // ref_count++ in L2 (no credit - entry already holds one)
            l2_tag_cam_.insert(dedup_tile.tile_id, l2_entry->slot_id, current_cycle);
            // Consume this move's L3 reference
            bool credit_released = l3_tag_cam_.invalidate(dedup_tile.tile_id);
            if (credit_released) {
                l3_credits_.release(static_cast<size_t>(dedup_tile.tile_id.matrix));
            }
            total_tiles_moved_++;

            events.push_back(TimingEvent(
                EventType::TILE_ARRIVED_L2,
                current_cycle,
                config_.mover_id,
                dedup_tile.tile_id,
                name()
            ));
            events.back().slot_id = l2_entry->slot_id;

            if (credit_released) {
                events.push_back(TimingEvent(
                    EventType::CREDIT_RELEASED,
                    current_cycle,
                    config_.mover_id,
                    dedup_tile.tile_id,
                    name()
                ));
            }
            // Do not advance i - remove_at shifted the next element into i
        }
    }

    /**
     * @brief Try to start a move transfer (L3→L2) using work-conserving scan
     * @return true if a move was started
     *
     * Work-conserving: Instead of only checking the head of the queue, scan the
     * entire queue for any tile that's ready (present in L3). This prevents
     * head-of-line blocking when tiles arrive out of order.
     *
     * Priority aging: When enabled, selects the oldest ready tile (by enqueue_cycle)
     * instead of the first ready tile in queue order. This prevents starvation.
     */
    bool try_start_move(Cycle current_cycle, std::vector<TimingEvent>& events) {
        if (move_queue_.empty()) return false;

        // Work-conserving scan: find any tile that's ready in L3
        // With priority aging: find the oldest ready tile
        std::optional<TagCAMEntry> found_entry;
        size_t found_index = 0;
        Cycle oldest_enqueue = std::numeric_limits<Cycle>::max();

        for (size_t i = 0; i < move_queue_.size(); ++i) {
            const auto& tile = move_queue_.at(i);
            auto l3_entry = l3_tag_cam_.match(tile.tile_id);
            if (l3_entry.has_value()) {
                if (config_.priority_aging) {
                    // Priority aging: prefer oldest tile (lowest enqueue_cycle)
                    if (tile.enqueue_cycle < oldest_enqueue) {
                        found_entry = l3_entry;
                        found_index = i;
                        oldest_enqueue = tile.enqueue_cycle;
                    }
                } else {
                    // Default: first ready tile in queue order
                    found_entry = l3_entry;
                    found_index = i;
                    break;
                }
            }
        }

        // No tile in queue is ready in L3
        if (!found_entry.has_value()) {
            // Report stall on head-of-queue tile for debugging
            const auto& head_tile = move_queue_.peek();
            events.push_back(TimingEvent(
                EventType::BM_STALL_TAG,
                current_cycle,
                config_.mover_id,
                head_tile.tile_id,
                name()
            ));
            stall_cycles_tag_++;
            return false;
        }

        // Check if L2 credit available (from the tile's matrix partition
        // when the pool is partitioned, per #89). Moves may not consume the
        // reserved credits - those are kept for compute-result drains
        // (Streamer), which otherwise starve because BlockMovers tick
        // before Streamers.
        const size_t move_part =
            static_cast<size_t>(move_queue_.at(found_index).tile_id.matrix);
        if (l2_credits_.available(move_part) <= config_.l2_credit_reserve ||
            !l2_credits_.acquire(move_part)) {
            const auto& tile = move_queue_.at(found_index);
            events.push_back(TimingEvent(
                EventType::BM_STALL_CREDIT,
                current_cycle,
                config_.mover_id,
                tile.tile_id,
                name()
            ));
            stall_cycles_credit_++;
            return false;
        }

        // Start the move - remove from queue at found position
        TileDescriptor move_tile = move_queue_.remove_at(found_index);

        // Also remove corresponding transpose flag if any
        if (found_index < transpose_flags_.size()) {
            transpose_flags_.erase(transpose_flags_.begin() +
                                   static_cast<std::ptrdiff_t>(found_index));
        }

        Cycle transfer_cycles = compute_transfer_cycles(move_tile.size_bytes);
        uint32_t l2_slot = allocate_l2_slot();

        in_flight_ = InFlightTransfer(
            move_tile,
            current_cycle,
            current_cycle + transfer_cycles,
            l2_slot,
            true  // is_move
        );
        in_flight_is_move_ = true;
        in_flight_l3_slot_ = found_entry->slot_id;

        events.push_back(TimingEvent(
            EventType::BM_MOVE_START,
            current_cycle,
            config_.mover_id,
            move_tile.tile_id,
            name()
        ));
        events.back().slot_id = l2_slot;

        events.push_back(TimingEvent(
            EventType::CREDIT_ACQUIRED,
            current_cycle,
            config_.mover_id,
            move_tile.tile_id,
            name()
        ));

        return true;
    }

    /**
     * @brief Try to start a writeback transfer (L2→L3) using work-conserving scan
     * @return true if a writeback was started
     *
     * Work-conserving: Scan queue for any tile ready in L2, not just the head.
     * Priority aging: When enabled, prefers oldest ready tile to prevent starvation.
     */
    bool try_start_writeback(Cycle current_cycle, std::vector<TimingEvent>& events) {
        if (writeback_queue_.empty()) return false;

        // Work-conserving scan: find any tile that's ready in L2
        // With priority aging: find the oldest ready tile
        std::optional<TagCAMEntry> found_entry;
        size_t found_index = 0;
        Cycle oldest_enqueue = std::numeric_limits<Cycle>::max();

        for (size_t i = 0; i < writeback_queue_.size(); ++i) {
            const auto& tile = writeback_queue_.at(i);
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
            const auto& head_tile = writeback_queue_.peek();
            events.push_back(TimingEvent(
                EventType::BM_STALL_TAG,
                current_cycle,
                config_.mover_id,
                head_tile.tile_id,
                name()
            ));
            stall_cycles_tag_++;
            return false;
        }

        // Check if L3 credit available (writebacks carry C tiles, which have
        // their own partition when the pool is partitioned, per #89)
        if (!l3_credits_.acquire(
                static_cast<size_t>(writeback_queue_.at(found_index).tile_id.matrix))) {
            const auto& tile = writeback_queue_.at(found_index);
            events.push_back(TimingEvent(
                EventType::BM_STALL_CREDIT,
                current_cycle,
                config_.mover_id,
                tile.tile_id,
                name()
            ));
            stall_cycles_credit_++;
            return false;
        }

        // Start the writeback - remove from queue at found position
        TileDescriptor wb_tile = writeback_queue_.remove_at(found_index);
        Cycle transfer_cycles = compute_transfer_cycles(wb_tile.size_bytes);
        uint32_t l3_slot = found_entry->slot_id;  // Reuse slot info for tracking

        in_flight_ = InFlightTransfer(
            wb_tile,
            current_cycle,
            current_cycle + transfer_cycles,
            l3_slot,
            false  // is_writeback
        );
        in_flight_is_move_ = false;

        events.push_back(TimingEvent(
            EventType::BM_WRITEBACK_START,
            current_cycle,
            config_.mover_id,
            wb_tile.tile_id,
            name()
        ));

        events.push_back(TimingEvent(
            EventType::CREDIT_ACQUIRED,
            current_cycle,
            config_.mover_id,
            wb_tile.tile_id,
            name()
        ));

        return true;
    }

    /**
     * @brief Allocate an L2 bank slot (round-robin)
     */
    uint32_t allocate_l2_slot() {
        uint32_t slot = next_l2_slot_;
        next_l2_slot_ = (next_l2_slot_ + 1) % static_cast<uint32_t>(l2_tag_cam_.capacity());
        return slot;
    }
};

} // namespace sw::kpu::timing
