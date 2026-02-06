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
 * - BLOCKED_AB: All A tiles then all B tiles (livelock-prone with limited credits)
 *
 * The default INTERLEAVED_AB strategy ensures livelock-free execution by
 * alternating between A and B tile operations, preventing either matrix
 * type from monopolizing buffer resources.
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
        BLOCKED_AB          ///< All A then all B (can livelock)
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

        // Address generation (base addresses in DRAM)
        Address a_base = 0;
        Address b_base = 0;
        Address c_base = 0;

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
                    // Load A tile
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(a_tile));

                    // Load B tile
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                    result.operations.push_back(ScheduleOperation::load(b_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));  // Transpose B
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Drain C tile after all K iterations
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate interleaved A-B schedule (livelock-safe)
     *
     * Alternates A and B operations to prevent buffer monopolization.
     * Each K iteration processes one A tile and one B tile before moving on.
     */
    void generate_interleaved_ab(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        // For each output tile
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                // Accumulate over K with interleaved A-B
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);

                    // Interleaved: Load A, Load B
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::load(b_tile));

                    // Interleaved: Move A, Move B
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));

                    // Interleaved: Feed A, Feed B
                    result.operations.push_back(ScheduleOperation::feed(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Drain and store C tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
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

                    // Load current tiles
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::load(b_tile));

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

                // Drain and store C
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate blocked A-B schedule (can cause livelock)
     *
     * Loads all A tiles for a row, then all B tiles for a column.
     * WARNING: This can cause livelock if buffer space is limited!
     */
    void generate_blocked_ab(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                // Load ALL A tiles for this row first
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                }

                // Then load ALL B tiles for this column
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                    result.operations.push_back(ScheduleOperation::load(b_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));
                }

                // Feed tiles
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                    result.operations.push_back(ScheduleOperation::feed(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Drain and store C
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
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

        // Calculate DRAM address
        tile.dram_address = calculate_address(matrix, ti, tj, tk);

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
