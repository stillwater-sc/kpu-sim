// ============================================================================
// include/sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp
// Batch Normalization schedule generator for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/elementwise_schedule_generator.hpp>  // emit_broadcast_tile

#include <sstream>

namespace sw::kpu::timing::schedule {

/**
 * @brief Batch Normalization schedule generator
 *
 * Generates schedules for BatchNorm operations:
 *   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
 *
 * BatchNorm differs from LayerNorm in two key ways:
 * 1. Normalization is per-channel, not per-instance
 * 2. Uses running statistics (training) or fixed statistics (inference)
 *
 * This generator targets inference mode where running_mean and running_var
 * are pre-computed and stored as parameters.
 *
 * For NCHW format input of shape [N, C, H, W]:
 * - Each channel c has its own mean[c], var[c], gamma[c], beta[c]
 * - Normalization: y[n,c,h,w] = gamma[c] * (x[n,c,h,w] - mean[c]) / sqrt(var[c] + eps) + beta[c]
 *
 * Usage:
 * ```cpp
 * BatchNormScheduleGenerator::Config config;
 * config.N = 32;       // Batch size
 * config.C = 64;       // Channels
 * config.H = 56;       // Height
 * config.W = 56;       // Width
 *
 * BatchNormScheduleGenerator generator(config);
 * auto schedule = generator.generate();
 * ```
 */
class BatchNormScheduleGenerator : public IScheduleGenerator {
public:
    /**
     * @brief Configuration for batch norm schedule generation
     */
    struct Config {
        // Input dimensions [N, C, H, W]
        Size N = 1;         ///< Batch size
        Size C = 64;        ///< Number of channels
        Size H = 56;        ///< Height
        Size W = 56;        ///< Width

        // Tiling
        Size Ti = 16;       ///< Tile size for spatial (H*W) dimension
        Size Tj = 16;       ///< Tile size for processing

        // Element size
        Size element_size = 4;  ///< Bytes per element

        // Epsilon for numerical stability
        float eps = 1e-5f;

        // Mode
        bool training = false;  ///< Training mode (compute batch stats) vs inference

        // Resource envelope (issue #90): the buffer capacities this schedule
        // is generated against. Defaults match ConcurrentTimingExecutor.
        Size l3_buffer_count = 32;  ///< L3 credit pool the schedule targets
        Size l2_bank_count = 64;    ///< L2 credit pool the schedule targets

        // Base addresses
        Address input_base = 0;
        Address output_base = 0;
        // Inference (folded) parameters: y = x*scale[c] + shift[c], where
        // scale = gamma/sqrt(var+eps), shift = beta - mean*scale (see
        // batchnorm_affine.hpp / docs/plans/e9_batchnorm_pattern.md). The fold
        // is done at param-prep time so inference loads two [C] vectors, not
        // four - halving resident params to 2C+1.
        Address scale_base = 0;         ///< Folded scale parameter [C]
        Address shift_base = 0;         ///< Folded shift parameter [C]
        // Raw parameters (used by training mode, which computes batch stats).
        Address gamma_base = 0;         ///< Scale parameter [C]
        Address beta_base = 0;          ///< Bias parameter [C]
        Address running_mean_base = 0;  ///< Running mean [C]
        Address running_var_base = 0;   ///< Running variance [C]
        Address scratch_base = 0;       ///< For training mode intermediates

        /**
         * @brief Calculate spatial size (H * W)
         */
        [[nodiscard]] Size spatial_size() const {
            return H * W;
        }

        /**
         * @brief Calculate total elements per sample
         */
        [[nodiscard]] Size elements_per_sample() const {
            return C * H * W;
        }

        /**
         * @brief Calculate spatial tiles per channel
         */
        [[nodiscard]] Size spatial_tiles() const {
            return (spatial_size() + Ti - 1) / Ti;
        }

        /**
         * @brief Calculate tile size in bytes
         */
        [[nodiscard]] Size tile_size_bytes() const {
            return Ti * element_size;
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
         * Inference preloads the FOLDED scale/shift for EVERY channel up front
         * and keeps them resident across all samples: 2*C tiles plus the
         * streaming input (the fold halves this from the raw 4*C+1). Training
         * processes channels sequentially: per channel, the gamma/beta pair
         * plus the mean/var scratch pair are live alongside the streaming input.
         */
        [[nodiscard]] Size required_working_set() const {
            return training ? 5 : 2 * C + 1;
        }
    };

    /**
     * @brief Construct with configuration
     */
    explicit BatchNormScheduleGenerator(const Config& config)
        : config_(config) {}

    /**
     * @brief Generate the schedule
     */
    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "batchnorm schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }

        // Validate configuration
        if (config_.N == 0 || config_.C == 0 || config_.H == 0 || config_.W == 0) {
            result.valid = false;
            result.error_message = "Dimensions must be non-zero";
            return result;
        }

        if (config_.training) {
            generate_training_mode(result);
        } else {
            generate_inference_mode(result);
        }

        // Set metadata
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.N * config_.C;
        result.metadata.N = config_.spatial_size();
        result.metadata.K = 1;
        result.metadata.Ti = config_.Ti;
        result.metadata.Tj = config_.Tj;
        result.metadata.Tk = 1;

        Size input_tiles = config_.N * config_.C * config_.spatial_tiles();
        result.metadata.a_tiles = input_tiles;
        // Two folded params per channel at inference (scale + shift); training
        // loads gamma + beta per channel (mean/var are computed into scratch).
        result.metadata.b_tiles = 2 * config_.C;
        result.metadata.c_tiles = input_tiles;

        result.metadata.strategy = config_.training ? "training_batchnorm" : "inference_batchnorm";
        result.metadata.l3_buffer_count = config_.l3_buffer_count;
        result.metadata.l2_bank_count = config_.l2_bank_count;
        result.valid = true;

        return result;
    }

    /**
     * @brief Get generator name
     */
    [[nodiscard]] std::string name() const override {
        return "BatchNormScheduleGenerator";
    }

    /**
     * @brief Get description
     */
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "BatchNorm [" << config_.N << " x " << config_.C
           << " x " << config_.H << " x " << config_.W << "]";
        return ss.str();
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    /**
     * @brief Generate inference mode schedule
     *
     * Uses the precomputed folded scale/shift: y = x*scale[c] + shift[c]. The
     * per-channel scale/shift are broadcast operands (P5): each is delivered
     * once (LOAD + MOVE with a consumer count) and stays resident across all
     * N*spatial-tile consumptions of its channel. Every output tile is an
     * executable per-channel affine COMPUTE depending on the streamed input
     * tile plus that channel's resident scale/shift (resolving the batchnorm
     * half of #139: the schedule no longer emits a DRAIN with no producer).
     */
    void generate_inference_mode(ScheduleResult& result) {
        Size spatial_tiles = config_.spatial_tiles();
        Size consumers_per_channel = config_.N * spatial_tiles;

        // Preload the folded scale/shift for every channel (all-channel
        // resident, 2C tiles). Broadcast delivery: one LOAD + one MOVE each,
        // the MOVE carrying the consumer count so the operand holds a single
        // L2 credit across its whole consumption span.
        for (Size c = 0; c < config_.C; ++c) {
            emit_broadcast_tile(result, make_param_tile(c, ParamType::SCALE),
                                consumers_per_channel);
            emit_broadcast_tile(result, make_param_tile(c, ParamType::SHIFT),
                                consumers_per_channel);
        }

        // Stream the input; each output tile is a per-channel affine COMPUTE.
        for (Size n = 0; n < config_.N; ++n) {
            for (Size c = 0; c < config_.C; ++c) {
                auto scale_tile = make_param_tile(c, ParamType::SCALE);
                auto shift_tile = make_param_tile(c, ParamType::SHIFT);

                for (Size si = 0; si < spatial_tiles; ++si) {
                    auto input_tile = make_input_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::load(input_tile));
                    result.operations.push_back(ScheduleOperation::move(input_tile));
                    result.operations.push_back(ScheduleOperation::feed(input_tile));

                    // The (n, si)-th consumption of this channel's resident
                    // scale/shift broadcast operands.
                    result.operations.push_back(ScheduleOperation::feed(scale_tile));
                    result.operations.push_back(ScheduleOperation::feed(shift_tile));

                    // y = x * scale[c] + shift[c] (a Vector Engine affine).
                    auto output_tile = make_output_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::compute(
                        output_tile, {input_tile.tile_id, scale_tile.tile_id,
                                      shift_tile.tile_id}));
                    result.operations.push_back(ScheduleOperation::drain(output_tile));
                    result.operations.push_back(ScheduleOperation::writeback(output_tile));
                    result.operations.push_back(ScheduleOperation::store(output_tile));
                }
            }
        }
    }

    /**
     * @brief Generate training mode schedule
     *
     * Computes batch statistics on the fly:
     * 1. Compute batch mean for each channel
     * 2. Compute batch variance for each channel
     * 3. Normalize and apply affine transformation
     */
    void generate_training_mode(ScheduleResult& result) {
        Size spatial_tiles = config_.spatial_tiles();

        // For each channel
        for (Size c = 0; c < config_.C; ++c) {
            // ========================================
            // Pass 1: Compute batch mean for channel c
            // ========================================
            for (Size n = 0; n < config_.N; ++n) {
                for (Size si = 0; si < spatial_tiles; ++si) {
                    auto tile = make_input_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::load(tile));
                    result.operations.push_back(ScheduleOperation::move(tile));
                    result.operations.push_back(ScheduleOperation::feed(tile));
                    // VE accumulates sum
                }
            }

            // Drain batch mean
            auto mean_tile = make_scratch_tile(c, 0);
            result.operations.push_back(ScheduleOperation::drain(mean_tile));
            result.operations.push_back(ScheduleOperation::writeback(mean_tile));

            // ========================================
            // Pass 2: Compute batch variance for channel c
            // ========================================
            for (Size n = 0; n < config_.N; ++n) {
                for (Size si = 0; si < spatial_tiles; ++si) {
                    auto tile = make_input_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::load(tile));
                    result.operations.push_back(ScheduleOperation::move(tile));
                    result.operations.push_back(ScheduleOperation::feed(tile));
                    // VE computes: (x - mean)^2, accumulates
                }
            }

            // Drain batch variance
            auto var_tile = make_scratch_tile(c, 1);
            result.operations.push_back(ScheduleOperation::drain(var_tile));
            result.operations.push_back(ScheduleOperation::writeback(var_tile));

            // ========================================
            // Pass 3: Normalize all samples for channel c
            // ========================================
            auto gamma_tile = make_param_tile(c, ParamType::GAMMA);
            auto beta_tile = make_param_tile(c, ParamType::BETA);
            result.operations.push_back(ScheduleOperation::load(gamma_tile));
            result.operations.push_back(ScheduleOperation::load(beta_tile));
            result.operations.push_back(ScheduleOperation::move(gamma_tile));
            result.operations.push_back(ScheduleOperation::move(beta_tile));

            for (Size n = 0; n < config_.N; ++n) {
                for (Size si = 0; si < spatial_tiles; ++si) {
                    auto input_tile = make_input_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::load(input_tile));
                    result.operations.push_back(ScheduleOperation::move(input_tile));
                    result.operations.push_back(ScheduleOperation::feed(input_tile));

                    auto output_tile = make_output_tile(n, c, si);
                    result.operations.push_back(ScheduleOperation::drain(output_tile));
                    result.operations.push_back(ScheduleOperation::writeback(output_tile));
                    result.operations.push_back(ScheduleOperation::store(output_tile));
                }
            }
        }
    }

    // GAMMA/BETA/MEAN/VAR are the raw params (training); SCALE/SHIFT are the
    // folded inference params. tj distinguishes them within a channel's B tiles.
    enum class ParamType { GAMMA, BETA, MEAN, VAR, SCALE, SHIFT };

    /**
     * @brief Create input tile descriptor
     */
    TileDescriptor make_input_tile(Size n, Size c, Size si) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::A;
        tile.tile_id.ti = n * config_.C + c;
        tile.tile_id.tj = si;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();

        // NCHW layout: address = base + (n*C*H*W + c*H*W + si*Ti) * elem_size
        Size spatial = config_.spatial_size();
        tile.dram_address = config_.input_base +
                            (n * config_.C * spatial + c * spatial + si * config_.Ti) *
                            config_.element_size;

        return tile;
    }

    /**
     * @brief Create output tile descriptor
     */
    TileDescriptor make_output_tile(Size n, Size c, Size si) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;
        tile.tile_id.ti = n * config_.C + c;
        tile.tile_id.tj = si;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();

        Size spatial = config_.spatial_size();
        tile.dram_address = config_.output_base +
                            (n * config_.C * spatial + c * spatial + si * config_.Ti) *
                            config_.element_size;

        return tile;
    }

    /**
     * @brief Create parameter tile descriptor
     */
    TileDescriptor make_param_tile(Size c, ParamType ptype) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::B;
        tile.tile_id.ti = c;
        tile.tile_id.tj = static_cast<Size>(ptype);
        tile.tile_id.tk = 0;

        tile.height = 1;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.element_size;

        Address base = 0;
        switch (ptype) {
            case ParamType::GAMMA: base = config_.gamma_base; break;
            case ParamType::BETA:  base = config_.beta_base; break;
            case ParamType::MEAN:  base = config_.running_mean_base; break;
            case ParamType::VAR:   base = config_.running_var_base; break;
            case ParamType::SCALE: base = config_.scale_base; break;
            case ParamType::SHIFT: base = config_.shift_base; break;
        }
        tile.dram_address = base + c * config_.element_size;

        return tile;
    }

    /**
     * @brief Create scratch tile descriptor
     */
    TileDescriptor make_scratch_tile(Size c, Size index) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;
        tile.tile_id.ti = c;
        tile.tile_id.tj = index;
        tile.tile_id.tk = 1;  // Distinguish from output

        tile.height = 1;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.element_size;
        tile.dram_address = config_.scratch_base + (c * 2 + index) * config_.element_size;

        return tile;
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "batchnorm_" << config_.N << "x" << config_.C
           << "x" << config_.H << "x" << config_.W;
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
