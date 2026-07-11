// ============================================================================
// include/sw/kpu/timing/schedule/matmul_schedule_generator.hpp
// Matrix multiplication schedule generator for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sw::kpu::timing::schedule {

/**
 * @brief Matrix multiplication schedule generator
 *
 * Generates schedules for C[M,N] = A[M,K] × B[K,N] operations.
 *
 * Supports multiple scheduling strategies:
 * - OUTPUT_STATIONARY: C tiles stay in accumulators, A and B stream through
 * - INTERLEAVED_AB: A-B-A-B ordering for balanced buffer usage (livelock-safe)
 * - PREFETCH_NEXT: Overlap next tile load with current compute
 * - BLOCKED_AB: A bursts then B bursts, blocked over K with burst lengths
 *   derived from the resource envelope (livelock-safe by construction)
 *
 * The Config carries a resource envelope (l3_buffer_count/l2_bank_count).
 * BLOCKED_AB derives its burst lengths from it so the tile working set
 * provably fits the credit pools (issue #67); INTERLEAVED_AB and
 * OUTPUT_STATIONARY are envelope-safe by their op-level interleaving.
 * Extending envelope-derived blocking to the remaining strategies and
 * generators is follow-up work tracked in #67.
 *
 * Usage:
 * ```cpp
 * MatMulScheduleGenerator::Config config;
 * config.M = 128; config.N = 128; config.K = 128;
 * config.Ti = 16; config.Tj = 16; config.Tk = 16;
 * config.strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;
 *
 * MatMulScheduleGenerator generator(config);
 * auto schedule = generator.generate();
 * ```
 */
class MatMulScheduleGenerator : public IScheduleGenerator {
public:
    /**
     * @brief Scheduling strategy
     */
    enum class Strategy {
        OUTPUT_STATIONARY,  ///< C stays in accumulators
        INTERLEAVED_AB,     ///< A-B-A-B ordering (livelock-safe) - DEFAULT
        PREFETCH_NEXT,      ///< Overlap load with compute
        BLOCKED_AB          ///< A/B bursts blocked over K, burst length bounded
                            ///< by the resource envelope (livelock-safe by
                            ///< construction)
    };

    /**
     * @brief Configuration for matmul schedule generation
     */
    struct Config {
        // Problem dimensions
        Size M = 64;            ///< A rows, C rows
        Size N = 64;            ///< B columns, C columns
        Size K = 64;            ///< A columns, B rows (reduction dimension)

        // Tile dimensions
        Size Ti = 16;           ///< Tile height
        Size Tj = 16;           ///< Tile width
        Size Tk = 16;           ///< Tile depth (reduction)

        // Element size
        Size element_size = 4;  ///< Bytes per element (4 for FP32, 2 for FP16)

        // Scheduling strategy
        Strategy strategy = Strategy::INTERLEAVED_AB;

        // Resource envelope: the buffer capacities this schedule is generated
        // against. Blocked strategies derive their burst lengths from these so
        // the tile working set provably fits the credit pools (issue #67) -
        // the classic blocked-linear-algebra discipline of sizing blocks to
        // the memory hierarchy. Defaults match ConcurrentTimingExecutor::Config.
        Size l3_buffer_count = 32;   ///< L3 credit pool the schedule targets
        Size l2_bank_count = 64;     ///< L2 credit pool the schedule targets

        // Address generation (base addresses in DRAM)
        Address a_base = 0;
        Address b_base = 0;
        Address c_base = 0;

        /**
         * @brief Per-matrix burst bound derived from the resource envelope
         *
         * A burst of L same-matrix tiles can occupy up to L buffers
         * concurrently; bounding L keeps an A-burst and a B-burst
         * simultaneously resident so neither matrix can monopolize the
         * credits the other needs. Delegates to per_matrix_burst_share() -
         * the canonical formula shared with is_livelock_safe().
         */
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }

        /**
         * @brief Calculate tile size in bytes
         */
        [[nodiscard]] Size tile_size_bytes() const {
            return Ti * Tj * element_size;
        }

        /**
         * @brief Calculate number of tiles in M dimension
         */
        [[nodiscard]] Size m_tiles() const {
            return (M + Ti - 1) / Ti;
        }

        /**
         * @brief Calculate number of tiles in N dimension
         */
        [[nodiscard]] Size n_tiles() const {
            return (N + Tj - 1) / Tj;
        }

        /**
         * @brief Calculate number of tiles in K dimension
         */
        [[nodiscard]] Size k_tiles() const {
            return (K + Tk - 1) / Tk;
        }

        /**
         * @brief Calculate total A tiles
         */
        [[nodiscard]] Size total_a_tiles() const {
            return m_tiles() * k_tiles();
        }

        /**
         * @brief Calculate total B tiles
         */
        [[nodiscard]] Size total_b_tiles() const {
            return k_tiles() * n_tiles();
        }

        /**
         * @brief Calculate total C tiles
         */
        [[nodiscard]] Size total_c_tiles() const {
            return m_tiles() * n_tiles();
        }
    };

    /**
     * @brief Construct with configuration
     */
    explicit MatMulScheduleGenerator(const Config& config)
        : config_(config) {}

    /**
     * @brief Generate the schedule
     */
    ScheduleResult generate() override {
        ScheduleResult result;

        // Validate configuration
        if (config_.M == 0 || config_.N == 0 || config_.K == 0) {
            result.valid = false;
            result.error_message = "Matrix dimensions must be non-zero";
            return result;
        }

        if (config_.Ti == 0 || config_.Tj == 0 || config_.Tk == 0) {
            result.valid = false;
            result.error_message = "Tile dimensions must be non-zero";
            return result;
        }

        // Generate based on strategy
        switch (config_.strategy) {
            case Strategy::OUTPUT_STATIONARY:
                generate_output_stationary(result);
                break;
            case Strategy::INTERLEAVED_AB:
                generate_interleaved_ab(result);
                break;
            case Strategy::PREFETCH_NEXT:
                generate_prefetch_next(result);
                break;
            case Strategy::BLOCKED_AB:
                generate_blocked_ab(result);
                break;
        }

        // Set metadata
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.M;
        result.metadata.N = config_.N;
        result.metadata.K = config_.K;
        result.metadata.Ti = config_.Ti;
        result.metadata.Tj = config_.Tj;
        result.metadata.Tk = config_.Tk;
        result.metadata.a_tiles = config_.total_a_tiles();
        result.metadata.b_tiles = config_.total_b_tiles();
        result.metadata.c_tiles = config_.total_c_tiles();
        result.metadata.strategy = strategy_name(config_.strategy);

        result.valid = true;
        return result;
    }

    /**
     * @brief Get generator name
     */
    [[nodiscard]] std::string name() const override {
        return "MatMulScheduleGenerator";
    }

    /**
     * @brief Get description
     */
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "MatMul " << config_.M << "x" << config_.N << "x" << config_.K
           << " (" << strategy_name(config_.strategy) << ")";
        return ss.str();
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    /**
     * @brief Generate output-stationary schedule
     *
     * Loop order: for ti, for tj, for tk
     * C tiles stay in PE accumulators.
     *
     * Emits all operations; execution layer handles deduplication.
     */
    void generate_output_stationary(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        // For each output tile
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                // Accumulate over K
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    // Load A tile (execution layer deduplicates)
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(a_tile));

                    // Load B tile (execution layer deduplicates)
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                    result.operations.push_back(ScheduleOperation::load(b_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));  // Transpose B
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Signal compute complete, then drain C tile
                // COMPUTE depends on ALL K-slice A and B feeds for this C tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::compute(
                    c_tile, make_compute_dependencies(ti, tj, k_tiles)));
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate interleaved A-B schedule (livelock-safe)
     *
     * This schedule emits all operations for each tile use. The execution layer
     * handles deduplication:
     * - DMA skips loads for tiles already in L3 or in-flight
     * - BlockMover skips moves for tiles already in L2
     *
     * Resource reuse pattern:
     * - A[ti, tk] is used n_tiles times (once per tj)
     * - B[tk, tj] is used m_tiles times (once per ti)
     *
     * The interleaved A-B ordering alternates between matrix types to prevent
     * buffer monopolization and ensure livelock-free execution.
     */
    void generate_interleaved_ab(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        // For each output tile - process in interleaved A-B fashion
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                // Accumulate over K with interleaved A-B
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);

                    // Interleaved: Load A, Load B
                    // (Execution layer deduplicates if tile already in L3)
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::load(b_tile));

                    // Interleaved: Move A, Move B
                    // (Execution layer deduplicates if tile already in L2)
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));

                    // Interleaved: Feed A, Feed B
                    // (Each feed consumes the tile - no deduplication)
                    result.operations.push_back(ScheduleOperation::feed(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Signal compute complete, then drain and store C tile
                // COMPUTE depends on ALL K-slice A and B feeds for this C tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::compute(
                    c_tile, make_compute_dependencies(ti, tj, k_tiles)));
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate prefetch-next schedule
     *
     * Overlaps loading of next tiles with processing of current tiles.
     * Emits all operations; execution layer handles deduplication.
     */
    void generate_prefetch_next(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);

                    // Load current tiles only at tk == 0; tiles for tk >= 1
                    // were already loaded by the previous iteration's
                    // prefetch. Every LOAD must pair 1:1 with a MOVE:
                    // each load inserts an L3 TagCAM reference and each move
                    // consumes one, so duplicate loads leave references (and
                    // the L3 credit) stranded -> livelock at scale.
                    if (tk == 0) {
                        result.operations.push_back(ScheduleOperation::load(a_tile));
                        result.operations.push_back(ScheduleOperation::load(b_tile));
                    }

                    // Prefetch next tiles if available
                    if (tk + 1 < k_tiles) {
                        auto a_next = make_tile(isa::MatrixID::A, ti, 0, tk + 1);
                        auto b_next = make_tile(isa::MatrixID::B, 0, tj, tk + 1);
                        result.operations.push_back(ScheduleOperation::load(a_next));
                        result.operations.push_back(ScheduleOperation::load(b_next));
                    }

                    // Move and feed current tiles
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));
                    result.operations.push_back(ScheduleOperation::feed(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Signal compute complete, then drain and store C
                // COMPUTE depends on ALL K-slice A and B feeds for this C tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::compute(
                    c_tile, make_compute_dependencies(ti, tj, k_tiles)));
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate blocked A-B schedule (livelock-safe by construction)
     *
     * Classic blocked-linear-algebra structure: an outer K-block loop
     * sequences tile residency so the working set provably fits the
     * resource envelope (issue #67). Within each K block, an A burst is
     * followed by a B burst; the burst length is bounded by
     * Config::max_burst_tiles(), which is derived from the L3/L2 credit
     * pools. An A burst and a B burst can therefore always be resident
     * simultaneously - neither matrix can monopolize the credits the other
     * needs, so livelock-safety is a constructive property of the schedule
     * rather than an empirical hope.
     *
     * With a large envelope the block degenerates to the full K loop
     * (identical to the historical behavior); with a constrained envelope
     * the K loop is chunked.
     */
    void generate_blocked_ab(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();
        Size burst = config_.max_burst_tiles();

        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                // Outer K-block loop: each block's bursts fit the envelope
                for (Size kb = 0; kb < k_tiles; kb += burst) {
                    Size kend = kb + burst < k_tiles ? kb + burst : k_tiles;

                    // A burst for this K block (bounded by the envelope)
                    for (Size tk = kb; tk < kend; ++tk) {
                        auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                        result.operations.push_back(ScheduleOperation::load(a_tile));
                        result.operations.push_back(ScheduleOperation::move(a_tile));
                    }

                    // B burst for this K block
                    for (Size tk = kb; tk < kend; ++tk) {
                        auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                        result.operations.push_back(ScheduleOperation::load(b_tile));
                        result.operations.push_back(ScheduleOperation::move(b_tile, true));
                    }

                    // Feed this K block's tiles (consumes their residency)
                    for (Size tk = kb; tk < kend; ++tk) {
                        auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                        auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                        result.operations.push_back(ScheduleOperation::feed(a_tile));
                        result.operations.push_back(ScheduleOperation::feed(b_tile));
                    }
                }

                // Signal compute complete, then drain and store C
                // COMPUTE depends on ALL K-slice A and B feeds for this C tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::compute(
                    c_tile, make_compute_dependencies(ti, tj, k_tiles)));
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Build the full COMPUTE dependency set for C[ti,tj]
     *
     * Every A[ti,*,k] and B[*,tj,k] K-slice must be FED before the compute
     * for C[ti,tj] can start. The K-slice count (dependencies / 2) also
     * scales compute latency in the executor.
     */
    std::vector<TileID> make_compute_dependencies(Size ti, Size tj, Size k_tiles) const {
        std::vector<TileID> deps;
        deps.reserve(2 * k_tiles);
        for (Size tk = 0; tk < k_tiles; ++tk) {
            TileID a;
            a.matrix = isa::MatrixID::A;
            a.ti = ti;
            a.tj = 0;
            a.tk = tk;
            deps.push_back(a);

            TileID b;
            b.matrix = isa::MatrixID::B;
            b.ti = 0;
            b.tj = tj;
            b.tk = tk;
            deps.push_back(b);
        }
        return deps;
    }

    /**
     * @brief Create a tile descriptor
     */
    TileDescriptor make_tile(isa::MatrixID matrix, Size ti, Size tj, Size tk) const {
        TileDescriptor tile;
        tile.tile_id.matrix = matrix;
        tile.tile_id.ti = ti;
        tile.tile_id.tj = tj;
        tile.tile_id.tk = tk;

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();

        // Calculate DRAM address and set matrix base address
        tile.dram_address = calculate_address(matrix, ti, tj, tk);
        switch (matrix) {
            case isa::MatrixID::A:
                tile.matrix_base_address = config_.a_base;
                break;
            case isa::MatrixID::B:
                tile.matrix_base_address = config_.b_base;
                break;
            case isa::MatrixID::C:
                tile.matrix_base_address = config_.c_base;
                break;
        }

        return tile;
    }

    /**
     * @brief Calculate DRAM address for a tile
     */
    Address calculate_address(isa::MatrixID matrix, Size ti, Size tj, Size tk) const {
        Size tile_bytes = config_.tile_size_bytes();

        switch (matrix) {
            case isa::MatrixID::A:
                // A[M,K]: row-major, tile at (ti, tk)
                return config_.a_base +
                       (ti * config_.k_tiles() + tk) * tile_bytes;

            case isa::MatrixID::B:
                // B[K,N]: row-major, tile at (tk, tj)
                return config_.b_base +
                       (tk * config_.n_tiles() + tj) * tile_bytes;

            case isa::MatrixID::C:
                // C[M,N]: row-major, tile at (ti, tj)
                return config_.c_base +
                       (ti * config_.n_tiles() + tj) * tile_bytes;

            default:
                return 0;
        }
    }

    /**
     * @brief Generate schedule name
     */
    std::string generate_name() const {
        std::ostringstream ss;
        ss << "matmul_" << config_.M << "x" << config_.N << "x" << config_.K
           << "_" << strategy_name(config_.strategy);
        return ss.str();
    }

    /**
     * @brief Get strategy name
     */
    static const char* strategy_name(Strategy strategy) {
        switch (strategy) {
            case Strategy::OUTPUT_STATIONARY: return "output_stationary";
            case Strategy::INTERLEAVED_AB:    return "interleaved_ab";
            case Strategy::PREFETCH_NEXT:     return "prefetch_next";
            case Strategy::BLOCKED_AB:        return "blocked_ab";
            default:                          return "unknown";
        }
    }
};

} // namespace sw::kpu::timing::schedule
