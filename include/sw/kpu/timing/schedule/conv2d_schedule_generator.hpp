// ============================================================================
// include/sw/kpu/timing/schedule/conv2d_schedule_generator.hpp
// Convolution schedule generator for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <algorithm>
#include <sstream>

namespace sw::kpu::timing::schedule {

/**
 * @brief 2D Convolution schedule generator
 *
 * Generates schedules for Conv2D operations using im2col transformation,
 * which converts the convolution into a matrix multiplication:
 *
 *   Output[N, H_out, W_out, C_out] =
 *       Conv2D(Input[N, H_in, W_in, C_in], Filter[C_out, Kh, Kw, C_in])
 *
 * Using im2col:
 *   - Input is transformed to [N * H_out * W_out, Kh * Kw * C_in]
 *   - Filter is reshaped to [Kh * Kw * C_in, C_out]
 *   - Output is [N * H_out * W_out, C_out]
 *
 * This allows reuse of the optimized matmul scheduling strategies.
 *
 * Usage:
 * ```cpp
 * Conv2DScheduleGenerator::Config config;
 * config.N = 1;                    // Batch size
 * config.H_in = 224; config.W_in = 224; config.C_in = 3;
 * config.C_out = 64;
 * config.Kh = 3; config.Kw = 3;
 * config.stride_h = 1; config.stride_w = 1;
 * config.padding_h = 1; config.padding_w = 1;
 *
 * Conv2DScheduleGenerator generator(config);
 * auto schedule = generator.generate();
 * ```
 */
class Conv2DScheduleGenerator : public IScheduleGenerator {
public:
    /**
     * @brief Scheduling strategy
     */
    enum class Strategy {
        IM2COL_INTERLEAVED,     ///< Im2col + interleaved matmul (default)
        IM2COL_OUTPUT_STATIONARY, ///< Im2col + output-stationary matmul
        DIRECT_CONV             ///< Direct convolution (future)
    };

    /**
     * @brief Configuration for conv2d schedule generation
     */
    struct Config {
        // Input dimensions (NHWC format)
        Size N = 1;             ///< Batch size
        Size H_in = 64;         ///< Input height
        Size W_in = 64;         ///< Input width
        Size C_in = 64;         ///< Input channels

        // Filter dimensions
        Size C_out = 64;        ///< Output channels
        Size Kh = 3;            ///< Kernel height
        Size Kw = 3;            ///< Kernel width

        // Stride and padding
        Size stride_h = 1;      ///< Vertical stride
        Size stride_w = 1;      ///< Horizontal stride
        Size padding_h = 0;     ///< Vertical padding
        Size padding_w = 0;     ///< Horizontal padding

        // Tiling (for the resulting matmul)
        Size Ti = 16;           ///< Output tile height
        Size Tj = 16;           ///< Output tile width
        Size Tk = 16;           ///< Reduction tile depth

        // Element size
        Size element_size = 4;  ///< Bytes per element

        // Strategy
        Strategy strategy = Strategy::IM2COL_INTERLEAVED;

        // Resource envelope (issue #90): the buffer capacities this schedule
        // is generated against. Defaults match ConcurrentTimingExecutor.
        // Both im2col strategies emit fine-grained per-tile interleaving
        // (working set of one A/B pair in flight), so no burst blocking is
        // required; the envelope is carried for the generator contract and
        // capacity-aware validation.
        Size l3_buffer_count = 32;  ///< L3 credit pool the schedule targets
        Size l2_bank_count = 64;    ///< L2 credit pool the schedule targets

        // Base addresses
        Address input_base = 0;
        Address filter_base = 0;
        Address output_base = 0;

        /**
         * @brief Calculate output height
         */
        [[nodiscard]] Size H_out() const {
            return (H_in + 2 * padding_h - Kh) / stride_h + 1;
        }

        /**
         * @brief Calculate output width
         */
        [[nodiscard]] Size W_out() const {
            return (W_in + 2 * padding_w - Kw) / stride_w + 1;
        }

        /**
         * @brief Calculate effective M dimension for matmul
         */
        [[nodiscard]] Size M() const {
            return N * H_out() * W_out();
        }

        /**
         * @brief Calculate effective K dimension for matmul
         */
        [[nodiscard]] Size K() const {
            return Kh * Kw * C_in;
        }

        /**
         * @brief Calculate tile counts
         */
        [[nodiscard]] Size m_tiles() const { return (M() + Ti - 1) / Ti; }
        [[nodiscard]] Size n_tiles() const { return (C_out + Tj - 1) / Tj; }
        [[nodiscard]] Size k_tiles() const { return (K() + Tk - 1) / Tk; }

        /**
         * @brief Calculate tile size in bytes
         */
        [[nodiscard]] Size tile_size_bytes() const {
            return Ti * Tj * element_size;
        }

        /**
         * @brief Per-matrix burst bound derived from the resource envelope
         */
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }

        /**
         * @brief Peak tile residency this schedule implies
         *
         * The im2col strategies stream one A/B tile pair plus the pending
         * C tile per iteration.
         */
        [[nodiscard]] Size required_working_set() const {
            return 3;
        }
    };

    /**
     * @brief Construct with configuration
     */
    explicit Conv2DScheduleGenerator(const Config& config)
        : config_(config) {}

    /**
     * @brief Generate the schedule
     */
    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "conv2d schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }

        // Validate configuration
        if (config_.H_in == 0 || config_.W_in == 0 || config_.C_in == 0) {
            result.valid = false;
            result.error_message = "Input dimensions must be non-zero";
            return result;
        }

        if (config_.Kh == 0 || config_.Kw == 0 || config_.C_out == 0) {
            result.valid = false;
            result.error_message = "Filter dimensions must be non-zero";
            return result;
        }

        if (config_.H_out() == 0 || config_.W_out() == 0) {
            result.valid = false;
            result.error_message = "Output dimensions must be non-zero (check padding/stride)";
            return result;
        }

        // Generate based on strategy
        switch (config_.strategy) {
            case Strategy::IM2COL_INTERLEAVED:
                generate_im2col_interleaved(result);
                break;
            case Strategy::IM2COL_OUTPUT_STATIONARY:
                generate_im2col_output_stationary(result);
                break;
            case Strategy::DIRECT_CONV:
                // Not implemented yet
                result.valid = false;
                result.error_message = "DIRECT_CONV not yet implemented";
                return result;
        }

        // Set metadata
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.M();
        result.metadata.N = config_.C_out;
        result.metadata.K = config_.K();
        result.metadata.Ti = config_.Ti;
        result.metadata.Tj = config_.Tj;
        result.metadata.Tk = config_.Tk;
        result.metadata.a_tiles = config_.m_tiles() * config_.k_tiles();
        result.metadata.b_tiles = config_.k_tiles() * config_.n_tiles();
        result.metadata.c_tiles = config_.m_tiles() * config_.n_tiles();
        result.metadata.strategy = strategy_name(config_.strategy);

        result.valid = true;
        return result;
    }

    /**
     * @brief Get generator name
     */
    [[nodiscard]] std::string name() const override {
        return "Conv2DScheduleGenerator";
    }

    /**
     * @brief Get description
     */
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "Conv2D " << config_.H_in << "x" << config_.W_in << "x" << config_.C_in
           << " -> " << config_.H_out() << "x" << config_.W_out() << "x" << config_.C_out
           << " (kernel " << config_.Kh << "x" << config_.Kw << ")";
        return ss.str();
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    /**
     * @brief Generate im2col + interleaved matmul schedule
     */
    void generate_im2col_interleaved(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        // For each output tile (using interleaved A-B ordering)
        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    // A = im2col(input) tile
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    // B = filter tile
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

                // Drain and store output tile
                auto c_tile = make_tile(isa::MatrixID::C, ti, tj, 0);
                result.operations.push_back(ScheduleOperation::drain(c_tile));
                result.operations.push_back(ScheduleOperation::writeback(c_tile));
                result.operations.push_back(ScheduleOperation::store(c_tile));
            }
        }
    }

    /**
     * @brief Generate im2col + output-stationary matmul schedule
     */
    void generate_im2col_output_stationary(ScheduleResult& result) {
        Size m_tiles = config_.m_tiles();
        Size n_tiles = config_.n_tiles();
        Size k_tiles = config_.k_tiles();

        for (Size ti = 0; ti < m_tiles; ++ti) {
            for (Size tj = 0; tj < n_tiles; ++tj) {
                for (Size tk = 0; tk < k_tiles; ++tk) {
                    // Load and process A tile
                    auto a_tile = make_tile(isa::MatrixID::A, ti, 0, tk);
                    result.operations.push_back(ScheduleOperation::load(a_tile));
                    result.operations.push_back(ScheduleOperation::move(a_tile));
                    result.operations.push_back(ScheduleOperation::feed(a_tile));

                    // Load and process B tile
                    auto b_tile = make_tile(isa::MatrixID::B, 0, tj, tk);
                    result.operations.push_back(ScheduleOperation::load(b_tile));
                    result.operations.push_back(ScheduleOperation::move(b_tile, true));
                    result.operations.push_back(ScheduleOperation::feed(b_tile));
                }

                // Output tile
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

        // Address calculation would involve im2col mapping
        tile.dram_address = calculate_address(matrix, ti, tj, tk);

        return tile;
    }

    /**
     * @brief Calculate address (simplified - would need im2col mapping)
     */
    Address calculate_address(isa::MatrixID matrix, Size ti, Size tj, Size tk) const {
        Size tile_bytes = config_.tile_size_bytes();

        switch (matrix) {
            case isa::MatrixID::A:
                // Im2col input - address computation simplified
                return config_.input_base +
                       (ti * config_.k_tiles() + tk) * tile_bytes;
            case isa::MatrixID::B:
                // Filter
                return config_.filter_base +
                       (tk * config_.n_tiles() + tj) * tile_bytes;
            case isa::MatrixID::C:
                // Output
                return config_.output_base +
                       (ti * config_.n_tiles() + tj) * tile_bytes;
            default:
                return 0;
        }
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "conv2d_" << config_.H_in << "x" << config_.W_in
           << "_k" << config_.Kh << "x" << config_.Kw
           << "_c" << config_.C_in << "to" << config_.C_out;
        return ss.str();
    }

    static const char* strategy_name(Strategy strategy) {
        switch (strategy) {
            case Strategy::IM2COL_INTERLEAVED:      return "im2col_interleaved";
            case Strategy::IM2COL_OUTPUT_STATIONARY: return "im2col_output_stationary";
            case Strategy::DIRECT_CONV:             return "direct_conv";
            default:                                return "unknown";
        }
    }
};

} // namespace sw::kpu::timing::schedule
