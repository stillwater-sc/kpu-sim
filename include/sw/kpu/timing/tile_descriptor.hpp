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
#include <vector>

namespace sw::kpu::timing {

using Cycle = uint64_t;
using Address = uint64_t;
using Size = uint32_t;

// ============================================================================
// GridPosition - 2D position in the chip grid
// ============================================================================

/**
 * @brief 2D position on the chip grid
 *
 * Used for:
 * - Memory tiles: L3(row, col) - position in the memory tile grid
 * - Compute tiles: CT(row, col) - position in the compute tile grid
 * - Memory controllers: MC(index) - typically on chip edges
 */
struct GridPosition {
    uint32_t row = 0;
    uint32_t col = 0;

    GridPosition() = default;
    GridPosition(uint32_t r, uint32_t c) : row(r), col(c) {}

    std::string to_string() const {
        return "(" + std::to_string(row) + "," + std::to_string(col) + ")";
    }

    bool operator==(const GridPosition& other) const {
        return row == other.row && col == other.col;
    }
};

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
// VectorStage - one stage of a tile context (docs/plans/csp-language.md step 3): a vector
// operation applied to a tile as a move carries it. Add broadcasts a vector tile (`arg`, read
// from L3) down the rows; the rest are elementwise activations.
// ============================================================================
struct VectorStage {
    enum class Op : uint8_t { Add, Relu, Gelu, Silu, Atan };
    Op op = Op::Relu;
    TileID arg{};
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

    // Matrix base address (for trace display - shows where matrix starts in DRAM)
    Address matrix_base_address = 0;  // Base address of the matrix in DRAM

    // The tile's HOME L3 tile: where it is loaded to, written back to, and ejected from
    // (docs/plans/noc-port-arbitration.md §4.1, Q10). Placement is the compiler's decision, so
    // a schedule may state it; -1 = not stated, and the executor picks a deterministic default
    // (a hash of the tile id). Every operation on one tile must agree on it.
    int32_t l3_tile = -1;

    // The compute tile a COMPUTE on this result runs on (docs/plans/system-schedule-debugger.md
    // §3.2): a schedule's decision, like l3_tile. -1 = not stated, and the executor runs it on
    // the first free compute tile. Ignored by the legacy unbounded compute model.
    int32_t cf_tile = -1;

    // L3 residency decided by a CSP program (docs/plans/csp-language.md step 1c.2): the program
    // releases this LOAD's L3 entry with an explicit Release (schedule_release), so the entry
    // holds one reference that Moves out of L3 do not consume. false = the legacy schedule path:
    // each Move consumes the load's reference, and a load of a tile already in L3 is a tag-CAM
    // hit. A program load is never a hit -- the program decided to load -- so a load of a tile
    // still held by its previous residency waits for that residency's Release.
    bool l3_held = false;

    // The BlockMover's order between a tile's Moves and its Releases: the Releases of this
    // tile issued before this Move or Release (set by the mover at enqueue).
    uint32_t l3_epoch = 0;

    // Vector-unit time on this move (set by the executor from a tile context's stages and the
    // site's lanes and rate): the move takes the longer of its transfer and this, since the
    // unit works on the tile as it streams. `fabric_cycles`: a Drain's stages in the compute
    // fabric, on the accumulator before it leaves; they run first and charge the compute tile.
    Cycle ve_cycles = 0;
    Cycle fabric_cycles = 0;

    // A CSP program's action (CspDriver): ordered by the program, not only by tag match. A Feed
    // of such a tile takes the copy its own Move brought: it waits until the tile's Moves
    // scheduled before it (`after_moves`, set by the executor) have arrived in L2, so it never
    // feeds an earlier copy that is still in L2 for another purpose (a drained result on its way
    // to its writeback).
    bool ordered = false;
    uint64_t after_moves = 0;
    uint64_t program_seq = 0;   // the action's place in its program (ordered actions only)
    uint64_t after_writebacks = 0;  // a Move's or ejection's: the tile's Writebacks before it (set by the mover)
    uint64_t after_drains = 0;      // a Writeback's: the tile's Drains before it (set by the executor)

    // Tile dimensions (for compute operations)
    Size height = 16;         // Tile height (rows)
    Size width = 16;          // Tile width (columns)
    Size element_size = 4;    // Bytes per element (4 for FP32, 2 for FP16)

    // Scheduling metadata
    Cycle enqueue_cycle = 0;  // When this was added to work queue
    Size priority = 0;        // Higher = more urgent (for aging)

    // Broadcast consumer count (issue #100): how many downstream feeds
    // will consume this tile after one MOVE delivers it. The BlockMover
    // seeds the L2 TagCAM ref_count with this value, making the
    // one-load-many-feeds (1:1:k) broadcast discipline explicit while
    // preserving one-credit-per-entry conservation. Default 1 = the
    // ordinary 1:1:1 pipeline.
    Size consumer_count = 1;

    // Compute size from dimensions
    Size compute_size_bytes() const {
        return height * width * element_size;
    }

    // Age-based priority (increases over time)
    Size age_priority(Cycle current_cycle) const {
        return static_cast<Size>(current_cycle - enqueue_cycle);
    }

    // Format tile with base address for trace display
    // e.g., "A[0,0] @ 0x1000" or just "A[0,0,0]" if no base address set
    std::string to_string_with_address() const {
        std::string s = tile_id.to_string();
        if (matrix_base_address != 0) {
            char hex[32];
            snprintf(hex, sizeof(hex), " @ 0x%llX",
                     static_cast<unsigned long long>(matrix_base_address));
            s += hex;
        }
        return s;
    }
};

/**
 * @brief Numeric contents associated with a tile in the functional timing model.
 *
 * Payloads live in the executor rather than in TileDescriptor so descriptors
 * remain cheap scheduling messages while values have one authoritative home.
 */
struct TilePayload {
    Size rows = 0;
    Size cols = 0;
    std::vector<float> values;

    [[nodiscard]] bool valid() const {
        return rows > 0 && cols > 0 && values.size() ==
               static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
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
