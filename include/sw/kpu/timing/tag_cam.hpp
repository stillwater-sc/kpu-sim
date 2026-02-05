// ============================================================================
// include/sw/kpu/timing/tag_cam.hpp
// Tag Content-Addressable Memory for tile arrival tracking
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Tag CAM entry containing tile location and arrival information
 */
struct TagCAMEntry {
    TileID tile_id;        ///< The tile identifier
    uint32_t slot_id;      ///< Buffer/bank slot where tile resides
    Cycle arrival_cycle;   ///< When the tile arrived at this level
    bool valid;            ///< Whether this entry is valid

    TagCAMEntry() : slot_id(0), arrival_cycle(0), valid(false) {}

    TagCAMEntry(const TileID& id, uint32_t slot, Cycle arrival)
        : tile_id(id), slot_id(slot), arrival_cycle(arrival), valid(true) {}
};

/**
 * @brief Tag Content-Addressable Memory for tracking tile arrivals
 *
 * In the KPU's credit-based dataflow model, downstream consumers cannot
 * request data. Instead, they wait for tiles to arrive and then process
 * them. The TagCAM provides the hardware mechanism for this:
 *
 * 1. When a tile arrives at a memory level (L3, L2, L1), it's inserted
 *    into the TagCAM with its slot location and arrival time
 * 2. Downstream consumers check the TagCAM to see if their required
 *    tile has arrived
 * 3. When a tile is consumed and the buffer freed, the TagCAM entry
 *    is invalidated
 *
 * Example flow:
 * 1. DMA completes load of tile A[0,0,0] to L3 buffer 3
 * 2. DMA calls l3_tag_cam.insert({A,0,0,0}, 3, cycle_100)
 * 3. BlockMover checks l3_tag_cam.lookup({A,0,0,0}) -> true
 * 4. BlockMover gets location: l3_tag_cam.match({A,0,0,0}) -> (3, cycle_100)
 * 5. BlockMover moves tile from L3[3] to L2
 * 6. BlockMover calls l3_tag_cam.invalidate({A,0,0,0})
 *
 * Invariants:
 * - Each TileID can only be in the CAM once (no duplicates)
 * - Lookup returns true iff a valid entry with matching TileID exists
 * - Match returns the slot and arrival cycle for timing calculations
 */
class TagCAM {
public:
    /**
     * @brief Construct a TagCAM with given capacity
     * @param capacity Maximum number of entries (should match buffer count)
     *
     * The capacity determines how many tiles can be tracked simultaneously.
     * For L3 buffers, this might be 8; for L2 banks, 16; etc.
     */
    explicit TagCAM(size_t capacity)
        : capacity_(capacity) {
        entries_.reserve(capacity);
    }

    /**
     * @brief Insert a tile arrival record
     * @param tile_id The tile that arrived
     * @param slot_id The buffer/bank slot containing the tile
     * @param arrival_cycle When the tile arrived (for timing)
     * @return true if inserted, false if CAM is full or tile already present
     *
     * Called by the component that pushes data downstream (e.g., DMA engine
     * after completing a load to L3).
     */
    bool insert(const TileID& tile_id, uint32_t slot_id, Cycle arrival_cycle) {
        // Check if already present
        if (entries_.count(tile_id) > 0) {
            return false;  // Duplicate - shouldn't happen in correct operation
        }

        // Check capacity
        if (entries_.size() >= capacity_) {
            return false;  // Full
        }

        entries_[tile_id] = TagCAMEntry(tile_id, slot_id, arrival_cycle);
        return true;
    }

    /**
     * @brief Check if a tile is present in the CAM
     * @param tile_id The tile to look up
     * @return true if tile is present and valid
     *
     * Called by downstream consumer to check if required tile has arrived.
     * This is a fast existence check - use match() to get full details.
     */
    [[nodiscard]] bool lookup(const TileID& tile_id) const {
        auto it = entries_.find(tile_id);
        return it != entries_.end() && it->second.valid;
    }

    /**
     * @brief Get full entry for a tile (slot and arrival time)
     * @param tile_id The tile to look up
     * @return Entry if found, std::nullopt otherwise
     *
     * Used when the consumer needs the slot location and arrival time
     * for timing calculations or to know which buffer to read from.
     */
    [[nodiscard]] std::optional<TagCAMEntry> match(const TileID& tile_id) const {
        auto it = entries_.find(tile_id);
        if (it != entries_.end() && it->second.valid) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Get slot ID for a tile (convenience method)
     * @param tile_id The tile to look up
     * @return Slot ID if found, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<uint32_t> get_slot(const TileID& tile_id) const {
        auto entry = match(tile_id);
        if (entry) {
            return entry->slot_id;
        }
        return std::nullopt;
    }

    /**
     * @brief Get arrival cycle for a tile (convenience method)
     * @param tile_id The tile to look up
     * @return Arrival cycle if found, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<Cycle> get_arrival_cycle(const TileID& tile_id) const {
        auto entry = match(tile_id);
        if (entry) {
            return entry->arrival_cycle;
        }
        return std::nullopt;
    }

    /**
     * @brief Remove a tile from the CAM
     * @param tile_id The tile to invalidate
     * @return true if tile was found and invalidated
     *
     * Called when a tile is consumed and the buffer is freed.
     * This allows the upstream producer to reuse the buffer slot.
     */
    bool invalidate(const TileID& tile_id) {
        auto it = entries_.find(tile_id);
        if (it != entries_.end()) {
            entries_.erase(it);
            return true;
        }
        return false;
    }

    /**
     * @brief Get number of valid entries
     * @return Count of tiles currently tracked
     */
    [[nodiscard]] size_t size() const {
        return entries_.size();
    }

    /**
     * @brief Get maximum capacity
     * @return Maximum number of entries
     */
    [[nodiscard]] size_t capacity() const {
        return capacity_;
    }

    /**
     * @brief Check if CAM is empty
     * @return true if no tiles are tracked
     */
    [[nodiscard]] bool empty() const {
        return entries_.empty();
    }

    /**
     * @brief Check if CAM is full
     * @return true if at capacity
     */
    [[nodiscard]] bool full() const {
        return entries_.size() >= capacity_;
    }

    /**
     * @brief Clear all entries
     *
     * Used at simulation start or reset.
     */
    void reset() {
        entries_.clear();
    }

    /**
     * @brief Get all valid entries (for debugging/tracing)
     * @return Vector of all current entries
     */
    [[nodiscard]] std::vector<TagCAMEntry> all_entries() const {
        std::vector<TagCAMEntry> result;
        result.reserve(entries_.size());
        for (const auto& [id, entry] : entries_) {
            result.push_back(entry);
        }
        return result;
    }

    /**
     * @brief Find any ready tile (for work-conserving scan)
     * @return First valid entry found, or std::nullopt if empty
     *
     * Used by work-conserving processes that can process any available
     * tile rather than waiting for a specific one.
     */
    [[nodiscard]] std::optional<TagCAMEntry> find_any_ready() const {
        for (const auto& [id, entry] : entries_) {
            if (entry.valid) {
                return entry;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Find the oldest ready tile (for priority/aging)
     * @return Entry with smallest arrival_cycle, or std::nullopt if empty
     *
     * Used for priority aging - older tiles get processed first to
     * prevent starvation.
     */
    [[nodiscard]] std::optional<TagCAMEntry> find_oldest() const {
        std::optional<TagCAMEntry> oldest;
        for (const auto& [id, entry] : entries_) {
            if (entry.valid) {
                if (!oldest || entry.arrival_cycle < oldest->arrival_cycle) {
                    oldest = entry;
                }
            }
        }
        return oldest;
    }

    /**
     * @brief Find all tiles for a specific matrix (for work-conserving)
     * @param matrix The matrix type to filter by
     * @return Vector of matching entries
     */
    [[nodiscard]] std::vector<TagCAMEntry> find_by_matrix(isa::MatrixID matrix) const {
        std::vector<TagCAMEntry> result;
        for (const auto& [id, entry] : entries_) {
            if (entry.valid && entry.tile_id.matrix == matrix) {
                result.push_back(entry);
            }
        }
        return result;
    }

private:
    size_t capacity_;
    std::unordered_map<TileID, TagCAMEntry, TileIDHash> entries_;
};

} // namespace sw::kpu::timing
