// ============================================================================
// include/sw/kpu/timing/tile_descriptor.hpp
// Tile descriptor for concurrent timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>

#include <cstdint>
#include <functional>
#include <tuple>

namespace sw::kpu::timing {

using Cycle = uint64_t;
using Address = uint64_t;
using Size = uint32_t;

// ============================================================================
// TileID - Unique identifier for a tile
// ============================================================================

/**
 * @brief Unique identifier for a tile in the memory hierarchy
 *
 * A tile is identified by:
 * - matrix: Which matrix it belongs to (A, B, or C)
 * - ti, tj, tk: Tile coordinates in the tiled iteration space
 */
struct TileID {
    isa::MatrixID matrix = isa::MatrixID::A;
    Size ti = 0;  // Row tile index
    Size tj = 0;  // Column tile index
    Size tk = 0;  // K-dimension tile index

    bool operator==(const TileID& other) const {
        return matrix == other.matrix &&
               ti == other.ti && tj == other.tj && tk == other.tk;
    }

    bool operator!=(const TileID& other) const {
        return !(*this == other);
    }

    bool operator<(const TileID& other) const {
        return std::tie(matrix, ti, tj, tk) <
               std::tie(other.matrix, other.ti, other.tj, other.tk);
    }

    // Convert to string for debugging/tracing
    std::string to_string() const {
        const char* m = (matrix == isa::MatrixID::A) ? "A" :
                        (matrix == isa::MatrixID::B) ? "B" : "C";
        return std::string(m) + "[" + std::to_string(ti) + "," +
               std::to_string(tj) + "," + std::to_string(tk) + "]";
    }
};

// Hash function for TileID (for use in unordered_map/set)
struct TileIDHash {
    std::size_t operator()(const TileID& id) const {
        std::size_t h1 = std::hash<int>{}(static_cast<int>(id.matrix));
        std::size_t h2 = std::hash<Size>{}(id.ti);
        std::size_t h3 = std::hash<Size>{}(id.tj);
        std::size_t h4 = std::hash<Size>{}(id.tk);
        // Combine hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

// ============================================================================
// TileDescriptor - Full description of a tile operation
// ============================================================================

/**
 * @brief Complete description of a tile for scheduling and tracking
 *
 * Contains all information needed to:
 * - Issue a DMA transfer
 * - Track the tile through the memory hierarchy
 * - Record timing events
 */
struct TileDescriptor {
    TileID tile_id;           // Unique tile identifier
    Address dram_address = 0; // DRAM address for DMA operations
    Size size_bytes = 0;      // Transfer size in bytes

    // Tile dimensions (for compute operations)
    Size height = 16;         // Tile height (rows)
    Size width = 16;          // Tile width (columns)
    Size element_size = 4;    // Bytes per element (4 for FP32, 2 for FP16)

    // Scheduling metadata
    Cycle enqueue_cycle = 0;  // When this was added to work queue
    Size priority = 0;        // Higher = more urgent (for aging)

    // Compute size from dimensions
    Size compute_size_bytes() const {
        return height * width * element_size;
    }

    // Age-based priority (increases over time)
    Size age_priority(Cycle current_cycle) const {
        return static_cast<Size>(current_cycle - enqueue_cycle);
    }
};

// ============================================================================
// MemoryLevel - Where a tile resides
// ============================================================================

enum class MemoryLevel {
    DRAM,       // External memory
    L3,         // L3 tile buffers
    L2,         // L2 banks
    L1,         // L1 streaming buffers
    COMPUTE     // In systolic array / accumulator
};

inline const char* to_string(MemoryLevel level) {
    switch (level) {
        case MemoryLevel::DRAM: return "DRAM";
        case MemoryLevel::L3: return "L3";
        case MemoryLevel::L2: return "L2";
        case MemoryLevel::L1: return "L1";
        case MemoryLevel::COMPUTE: return "COMPUTE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// TileLocation - Where a tile is located
// ============================================================================

/**
 * @brief Tracks where a tile currently resides
 */
struct TileLocation {
    MemoryLevel level = MemoryLevel::DRAM;
    uint32_t slot_id = 0;     // Buffer/bank/slot ID at this level
    Cycle arrival_cycle = 0;  // When tile arrived at this location

    bool is_valid() const {
        return arrival_cycle > 0 || level == MemoryLevel::DRAM;
    }
};

} // namespace sw::kpu::timing
