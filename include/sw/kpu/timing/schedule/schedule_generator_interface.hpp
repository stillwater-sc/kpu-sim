// ============================================================================
// include/sw/kpu/timing/schedule/schedule_generator_interface.hpp
// Schedule generator interface for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

#include <string>
#include <vector>

namespace sw::kpu::timing::schedule {

// ============================================================================
// Resource Envelope
// ============================================================================

/**
 * @brief Per-matrix burst share for a resource envelope (issue #67)
 *
 * The canonical formula tying envelope-aware schedule generation to the
 * capacity-aware livelock check: each matrix may burst at most a quarter of
 * the smaller credit pool, so an A burst and a B burst can always be
 * resident simultaneously with half the pool left as headroom for in-flight
 * drains, C writebacks, and prefetch overlap. Generators bound their bursts
 * with this; is_livelock_safe(schedule, l3, l2) verifies against it. Keep
 * both sides on this single definition.
 */
[[nodiscard]] inline Size per_matrix_burst_share(Size l3_credits, Size l2_credits) {
    Size l3_share = l3_credits / 4;
    Size l2_share = l2_credits / 4;
    Size share = l3_share < l2_share ? l3_share : l2_share;
    return share > 0 ? share : 1;
}

// ============================================================================
// Schedule Operation Types
// ============================================================================

/**
 * @brief Types of operations in a schedule
 *
 * These operations are ISA-independent representations of data movement
 * that can be mapped to the ConcurrentTimingExecutor's schedule_* API.
 */
enum class ScheduleOpType {
    LOAD,       // DMA load: DRAM -> L3
    STORE,      // DMA store: L3 -> DRAM
    MOVE,       // BlockMover: L3 -> L2
    WRITEBACK,  // BlockMover: L2 -> L3
    FEED,       // Streamer: L2 -> Compute
    COMPUTE,    // Compute: signals result tile ready (after all inputs fed)
    DRAIN       // Streamer: Compute -> L2 (waits for COMPUTE)
};

/**
 * @brief Convert ScheduleOpType to string
 */
inline const char* to_string(ScheduleOpType type) {
    switch (type) {
        case ScheduleOpType::LOAD:      return "LOAD";
        case ScheduleOpType::STORE:     return "STORE";
        case ScheduleOpType::MOVE:      return "MOVE";
        case ScheduleOpType::WRITEBACK: return "WRITEBACK";
        case ScheduleOpType::FEED:      return "FEED";
        case ScheduleOpType::COMPUTE:   return "COMPUTE";
        case ScheduleOpType::DRAIN:     return "DRAIN";
        default:                        return "UNKNOWN";
    }
}

// ============================================================================
// Schedule Operation
// ============================================================================

/**
 * @brief A single operation in an execution schedule
 *
 * ScheduleOperation is the fundamental unit of a schedule. It pairs an
 * operation type with a tile descriptor, enabling:
 * - ISA-independent schedule representation
 * - Direct mapping to ConcurrentTimingExecutor API
 * - Static analysis for validation
 */
struct ScheduleOperation {
    ScheduleOpType type;                    ///< Operation type
    TileDescriptor tile;                    ///< Tile to operate on
    bool transpose = false;                 ///< Transpose during move (for B matrix)

    // Optional component targeting (use -1 for auto-select)
    int engine_id = -1;                     ///< Target DMA engine
    int mover_id = -1;                      ///< Target BlockMover
    int streamer_id = -1;                   ///< Target Streamer

    // For COMPUTE operations: ALL tiles that must be FED before compute
    // starts (every A and B K-slice contributing to the result tile).
    // dependency_tile is retained for backward compatibility and holds the
    // last entry of dependency_tiles.
    std::vector<TileID> dependency_tiles;   ///< COMPUTE waits for all of these (fed)
    TileID dependency_tile{};               ///< Last dependency (legacy field)

    // Compute-resident inputs (issue #155): tiles a COMPUTE consumes from
    // compute storage, produced by a PRIOR compute, rather than via a fresh
    // FEED - e.g. the running (m, l) state of online softmax handed to the
    // apply computes. Empty for ordinary computes.
    std::vector<TileID> resident_tiles;

    /**
     * @brief Create a LOAD operation
     */
    static ScheduleOperation load(const TileDescriptor& tile, int engine = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::LOAD;
        op.tile = tile;
        op.engine_id = engine;
        return op;
    }

    /**
     * @brief Create a STORE operation
     */
    static ScheduleOperation store(const TileDescriptor& tile, int engine = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::STORE;
        op.tile = tile;
        op.engine_id = engine;
        return op;
    }

    /**
     * @brief Create a MOVE operation (L3 -> L2)
     */
    static ScheduleOperation move(const TileDescriptor& tile, bool transpose = false, int mover = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::MOVE;
        op.tile = tile;
        op.transpose = transpose;
        op.mover_id = mover;
        return op;
    }

    /**
     * @brief Create a WRITEBACK operation (L2 -> L3)
     */
    static ScheduleOperation writeback(const TileDescriptor& tile, int mover = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::WRITEBACK;
        op.tile = tile;
        op.mover_id = mover;
        return op;
    }

    /**
     * @brief Create a FEED operation (L2 -> Compute)
     */
    static ScheduleOperation feed(const TileDescriptor& tile, int streamer = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::FEED;
        op.tile = tile;
        op.streamer_id = streamer;
        return op;
    }

    /**
     * @brief Create a COMPUTE operation (signals result tile ready)
     *
     * COMPUTE must be scheduled after all FEED operations for the input tiles
     * that contribute to this result tile. DRAIN waits for COMPUTE before
     * transferring the result from compute fabric to L2.
     *
     * @param tile Result tile (C matrix)
     * @param dependency Single input tile that must be FED before compute
     *                   starts (legacy single-dependency form)
     */
    static ScheduleOperation compute(const TileDescriptor& tile, const TileID& dependency) {
        ScheduleOperation op;
        op.type = ScheduleOpType::COMPUTE;
        op.tile = tile;
        op.dependency_tiles = {dependency};
        op.dependency_tile = dependency;
        return op;
    }

    /**
     * @brief Create a COMPUTE operation with the full dependency set
     *
     * @param tile Result tile (C matrix)
     * @param dependencies ALL input tiles (every A and B K-slice) that must
     *                     be FED before compute starts; the K-slice count
     *                     (dependencies.size() / 2) also scales compute
     *                     latency in the executor
     */
    static ScheduleOperation compute(const TileDescriptor& tile,
                                     std::vector<TileID> dependencies) {
        ScheduleOperation op;
        op.type = ScheduleOpType::COMPUTE;
        op.tile = tile;
        if (!dependencies.empty()) {
            op.dependency_tile = dependencies.back();
        }
        op.dependency_tiles = std::move(dependencies);
        return op;
    }

    /**
     * @brief Create a COMPUTE with both fed and compute-resident inputs
     *
     * @param tile Result tile
     * @param fed_dependencies input tiles that must be FED before compute
     * @param resident_dependencies input tiles produced by a prior COMPUTE
     *        that stay resident in the compute fabric (issue #155)
     */
    static ScheduleOperation compute(const TileDescriptor& tile,
                                     std::vector<TileID> fed_dependencies,
                                     std::vector<TileID> resident_dependencies) {
        ScheduleOperation op;
        op.type = ScheduleOpType::COMPUTE;
        op.tile = tile;
        if (!fed_dependencies.empty()) {
            op.dependency_tile = fed_dependencies.back();
        }
        op.dependency_tiles = std::move(fed_dependencies);
        op.resident_tiles = std::move(resident_dependencies);
        return op;
    }

    /**
     * @brief Create a DRAIN operation (Compute -> L2)
     */
    static ScheduleOperation drain(const TileDescriptor& tile, int streamer = -1) {
        ScheduleOperation op;
        op.type = ScheduleOpType::DRAIN;
        op.tile = tile;
        op.streamer_id = streamer;
        return op;
    }

    /**
     * @brief Get matrix type for this operation
     */
    [[nodiscard]] isa::MatrixID matrix() const {
        return tile.tile_id.matrix;
    }

    /**
     * @brief Human-readable description
     */
    [[nodiscard]] std::string to_string() const {
        return std::string(schedule::to_string(type)) + "(" + tile.tile_id.to_string() + ")";
    }
};

// ============================================================================
// Schedule Metadata
// ============================================================================

/**
 * @brief Metadata about a generated schedule
 *
 * Contains information needed to understand and validate a schedule,
 * including problem dimensions, tile counts, and configuration.
 */
struct ScheduleMetadata {
    std::string name;           ///< Schedule name (e.g., "matmul_128x128x128")
    std::string generator;      ///< Generator name (e.g., "MatMulScheduleGenerator")

    // Problem dimensions
    Size M = 0, N = 0, K = 0;   ///< Matrix dimensions

    // Tile dimensions
    Size Ti = 16, Tj = 16, Tk = 16;

    // Tile counts per matrix type
    Size a_tiles = 0;           ///< Total A tiles to load
    Size b_tiles = 0;           ///< Total B tiles to load
    Size c_tiles = 0;           ///< Total C tiles to produce

    // Resource requirements
    Size l3_buffers_needed = 0; ///< Minimum L3 buffers required
    Size l2_banks_needed = 0;   ///< Minimum L2 banks required

    // Generation envelope (issue #91): the credit pools this schedule was
    // generated against (#67/#90 constructive-safety guarantees hold only
    // under an executor configured with at least these pools). 0 = not
    // recorded (hand-built or legacy schedule).
    Size l3_buffer_count = 0;   ///< L3 envelope used at generation time
    Size l2_bank_count = 0;     ///< L2 envelope used at generation time

    // Scheduling strategy metadata
    std::string strategy;       ///< E.g., "OUTPUT_STATIONARY", "INTERLEAVED_AB"

    /**
     * @brief Calculate total number of tiles
     */
    [[nodiscard]] Size total_tiles() const {
        return a_tiles + b_tiles + c_tiles;
    }

    /**
     * @brief Calculate total operations (approximate)
     */
    [[nodiscard]] Size estimated_ops() const {
        // Each tile needs: load, move, feed (for A/B) or drain, writeback, store (for C)
        return (a_tiles + b_tiles) * 3 + c_tiles * 3;
    }
};

// ============================================================================
// Schedule Result
// ============================================================================

/**
 * @brief Result of schedule generation
 *
 * Contains the sequence of operations and associated metadata.
 */
struct ScheduleResult {
    std::vector<ScheduleOperation> operations;  ///< Ordered sequence of operations
    ScheduleMetadata metadata;                  ///< Schedule metadata
    bool valid = false;                         ///< Whether generation succeeded
    std::string error_message;                  ///< Error message if invalid

    /**
     * @brief Check if schedule is valid
     */
    [[nodiscard]] explicit operator bool() const {
        return valid;
    }

    /**
     * @brief Get number of operations
     */
    [[nodiscard]] size_t size() const {
        return operations.size();
    }

    /**
     * @brief Count operations by type
     */
    [[nodiscard]] size_t count_ops(ScheduleOpType type) const {
        size_t count = 0;
        for (const auto& op : operations) {
            if (op.type == type) ++count;
        }
        return count;
    }

    /**
     * @brief Count operations by matrix
     */
    [[nodiscard]] size_t count_matrix_ops(isa::MatrixID matrix) const {
        size_t count = 0;
        for (const auto& op : operations) {
            if (op.matrix() == matrix) ++count;
        }
        return count;
    }
};

// ============================================================================
// Schedule Generator Interface
// ============================================================================

/**
 * @brief Abstract interface for schedule generators
 *
 * Schedule generators produce sequences of ScheduleOperations for specific
 * computation patterns (matmul, conv2d, softmax, etc.). The generated
 * schedules are ISA-independent and can be:
 * - Executed on ConcurrentTimingExecutor
 * - Validated for correctness
 * - Analyzed for livelock potential
 *
 * Generator implementations should ensure their schedules are livelock-free
 * when using the recommended strategies (e.g., INTERLEAVED_AB for matmul).
 */
class IScheduleGenerator {
public:
    virtual ~IScheduleGenerator() = default;

    /**
     * @brief Generate the schedule
     * @return ScheduleResult containing operations and metadata
     */
    virtual ScheduleResult generate() = 0;

    /**
     * @brief Get generator name
     * @return Human-readable name (e.g., "MatMulScheduleGenerator")
     */
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Get a brief description of the generator
     */
    [[nodiscard]] virtual std::string description() const {
        return name();
    }
};

// ============================================================================
// Schedule Analysis Utilities
// ============================================================================

/**
 * @brief Analyze a schedule for potential issues
 */
struct ScheduleAnalysis {
    size_t total_ops = 0;
    size_t load_ops = 0;
    size_t store_ops = 0;
    size_t move_ops = 0;
    size_t writeback_ops = 0;
    size_t feed_ops = 0;
    size_t compute_ops = 0;
    size_t drain_ops = 0;

    size_t a_ops = 0;
    size_t b_ops = 0;
    size_t c_ops = 0;

    // Livelock risk indicators
    size_t max_consecutive_a = 0;   ///< Longest run of A ops
    size_t max_consecutive_b = 0;   ///< Longest run of B ops
    bool is_interleaved = true;     ///< A-B alternation detected

    /**
     * @brief Analyze a schedule
     */
    static ScheduleAnalysis analyze(const ScheduleResult& schedule) {
        ScheduleAnalysis result;
        result.total_ops = schedule.operations.size();

        size_t consecutive_a = 0;
        size_t consecutive_b = 0;
        isa::MatrixID last_matrix = isa::MatrixID::C;

        for (const auto& op : schedule.operations) {
            // Count by type
            switch (op.type) {
                case ScheduleOpType::LOAD:      result.load_ops++; break;
                case ScheduleOpType::STORE:     result.store_ops++; break;
                case ScheduleOpType::MOVE:      result.move_ops++; break;
                case ScheduleOpType::WRITEBACK: result.writeback_ops++; break;
                case ScheduleOpType::FEED:      result.feed_ops++; break;
                case ScheduleOpType::COMPUTE:   result.compute_ops++; break;
                case ScheduleOpType::DRAIN:     result.drain_ops++; break;
            }

            // Count by matrix and track consecutive runs
            isa::MatrixID current_matrix = op.matrix();

            switch (current_matrix) {
                case isa::MatrixID::A:
                    result.a_ops++;
                    if (last_matrix == isa::MatrixID::A) {
                        consecutive_a++;
                    } else {
                        // Switching to A - first update max for previous matrix
                        if (consecutive_b > result.max_consecutive_b) {
                            result.max_consecutive_b = consecutive_b;
                        }
                        consecutive_b = 0;
                        consecutive_a = 1;
                    }
                    break;
                case isa::MatrixID::B:
                    result.b_ops++;
                    if (last_matrix == isa::MatrixID::B) {
                        consecutive_b++;
                    } else {
                        // Switching to B - first update max for previous matrix
                        if (consecutive_a > result.max_consecutive_a) {
                            result.max_consecutive_a = consecutive_a;
                        }
                        consecutive_a = 0;
                        consecutive_b = 1;
                    }
                    break;
                case isa::MatrixID::C:
                    result.c_ops++;
                    // Update max for whichever was running
                    if (consecutive_a > result.max_consecutive_a) {
                        result.max_consecutive_a = consecutive_a;
                    }
                    if (consecutive_b > result.max_consecutive_b) {
                        result.max_consecutive_b = consecutive_b;
                    }
                    consecutive_a = 0;
                    consecutive_b = 0;
                    break;
            }
            last_matrix = current_matrix;
        }

        // Final consecutive counts
        if (consecutive_a > result.max_consecutive_a) {
            result.max_consecutive_a = consecutive_a;
        }
        if (consecutive_b > result.max_consecutive_b) {
            result.max_consecutive_b = consecutive_b;
        }

        // Check interleaving: A ops should not exceed 2x B ops consecutively
        result.is_interleaved = (result.max_consecutive_a <= 3 && result.max_consecutive_b <= 3);

        return result;
    }
};

} // namespace sw::kpu::timing::schedule
