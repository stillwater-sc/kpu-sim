// ============================================================================
// include/sw/kpu/timing/schedule/layernorm_schedule_generator.hpp
// Layer Normalization schedule generator for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <sstream>

namespace sw::kpu::timing::schedule {

/**
 * @brief Layer Normalization schedule generator
 *
 * Generates schedules for LayerNorm operations:
 *   y = gamma * (x - mean) / sqrt(variance + eps) + beta
 *
 * Where:
 *   mean = mean(x) over normalized dimensions
 *   variance = var(x) over normalized dimensions
 *   gamma, beta = learnable affine parameters
 *
 * LayerNorm is a multi-pass operation:
 * 1. Pass 1: Compute mean over normalized dimensions
 * 2. Pass 2: Compute variance (using mean from pass 1)
 * 3. Pass 3: Normalize and apply affine transformation
 *
 * Typical use case: Transformer models normalize over hidden dimension.
 *
 * Usage:
 * ```cpp
 * LayerNormScheduleGenerator::Config config;
 * config.batch_size = 32;
 * config.sequence_length = 512;
 * config.hidden_size = 768;  // Normalized dimension
 *
 * LayerNormScheduleGenerator generator(config);
 * auto schedule = generator.generate();
 * ```
 */
class LayerNormScheduleGenerator : public IScheduleGenerator {
public:
    /**
     * @brief Configuration for layer norm schedule generation
     */
    struct Config {
        // Input dimensions [batch, sequence, hidden]
        Size batch_size = 1;        ///< Batch dimension
        Size sequence_length = 512; ///< Sequence/spatial dimension
        Size hidden_size = 768;     ///< Hidden dimension (normalized)

        // Tiling
        Size Ti = 16;               ///< Tile size for batch*sequence
        Size Tj = 16;               ///< Tile size for hidden dimension

        // Element size
        Size element_size = 4;      ///< Bytes per element

        // Epsilon for numerical stability
        float eps = 1e-5f;

        // Has learnable parameters
        bool affine = true;         ///< Whether to apply gamma/beta

        // Resource envelope (issue #90): the buffer capacities this schedule
        // is generated against. Defaults match ConcurrentTimingExecutor.
        Size l3_buffer_count = 32;  ///< L3 credit pool the schedule targets
        Size l2_bank_count = 64;    ///< L2 credit pool the schedule targets

        // Base addresses
        Address input_base = 0;
        Address output_base = 0;
        Address gamma_base = 0;     ///< Gamma (scale) parameter
        Address beta_base = 0;      ///< Beta (bias) parameter
        Address scratch_base = 0;   ///< For mean/variance intermediates

        /**
         * @brief Per-matrix burst bound derived from the resource envelope
         */
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }

        /**
         * @brief Peak tile residency this multi-pass schedule implies
         *
         * With affine parameters, gamma and beta tiles (2 x hidden_tiles)
         * are loaded once up front and stay resident across every instance,
         * plus the mean/variance scratch pair and the streaming input tile.
         * Without affine, only the scratch pair and the streaming input
         * are live. A schedule whose
         * working set exceeds the envelope share would wedge at runtime,
         * so generate() refuses it a priori (parameter re-streaming or
         * blocking is epic E9 scope, issue #78).
         */
        [[nodiscard]] Size required_working_set() const {
            return affine ? 2 * hidden_tiles() + 3 : 3;
        }

        /**
         * @brief Calculate total normalized instances (batch * sequence)
         */
        [[nodiscard]] Size normalized_instances() const {
            return batch_size * sequence_length;
        }

        /**
         * @brief Calculate instance tiles
         */
        [[nodiscard]] Size instance_tiles() const {
            return (normalized_instances() + Ti - 1) / Ti;
        }

        /**
         * @brief Calculate hidden tiles
         */
        [[nodiscard]] Size hidden_tiles() const {
            return (hidden_size + Tj - 1) / Tj;
        }

        /**
         * @brief Calculate tile size in bytes
         */
        [[nodiscard]] Size tile_size_bytes() const {
            return Ti * Tj * element_size;
        }
    };

    /**
     * @brief Construct with configuration
     */
    explicit LayerNormScheduleGenerator(const Config& config)
        : config_(config) {}

    /**
     * @brief Generate the schedule
     */
    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "layernorm multi-pass schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles (resident affine params + scratch) but the resource "
                "envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count or re-stream the "
                "parameters (epic E9, issue #78)";
            return result;
        }

        // Validate configuration
        if (config_.normalized_instances() == 0 || config_.hidden_size == 0) {
            result.valid = false;
            result.error_message = "Dimensions must be non-zero";
            return result;
        }

        // Generate multi-pass layer norm schedule
        generate_multi_pass(result);

        // Set metadata
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.normalized_instances();
        result.metadata.N = config_.hidden_size;
        result.metadata.K = 1;
        result.metadata.Ti = config_.Ti;
        result.metadata.Tj = config_.Tj;
        result.metadata.Tk = 1;

        Size input_tiles = config_.instance_tiles() * config_.hidden_tiles();
        result.metadata.a_tiles = input_tiles;
        result.metadata.b_tiles = config_.affine ? config_.hidden_tiles() : 0;
        result.metadata.c_tiles = input_tiles;

        result.metadata.strategy = "multi_pass_layernorm";
        result.valid = true;

        return result;
    }

    /**
     * @brief Get generator name
     */
    [[nodiscard]] std::string name() const override {
        return "LayerNormScheduleGenerator";
    }

    /**
     * @brief Get description
     */
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "LayerNorm [" << config_.batch_size << " x "
           << config_.sequence_length << " x " << config_.hidden_size << "]";
        return ss.str();
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    /**
     * @brief Generate multi-pass layer norm schedule
     *
     * Pass structure:
     * 1. Compute mean: Load input, reduce sum, divide by hidden_size
     * 2. Compute variance: Load input, subtract mean, square, reduce sum
     * 3. Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
     */
    void generate_multi_pass(ScheduleResult& result) {
        Size instance_tiles = config_.instance_tiles();
        Size hidden_tiles = config_.hidden_tiles();

        // Load gamma and beta once if affine
        if (config_.affine) {
            for (Size hi = 0; hi < hidden_tiles; ++hi) {
                auto gamma_tile = make_param_tile(hi, true);
                result.operations.push_back(ScheduleOperation::load(gamma_tile));
                result.operations.push_back(ScheduleOperation::move(gamma_tile));

                auto beta_tile = make_param_tile(hi, false);
                result.operations.push_back(ScheduleOperation::load(beta_tile));
                result.operations.push_back(ScheduleOperation::move(beta_tile));
            }
        }

        // For each normalized instance (batch * sequence)
        for (Size ii = 0; ii < instance_tiles; ++ii) {
            // ========================================
            // Pass 1: Compute mean
            // ========================================
            for (Size hi = 0; hi < hidden_tiles; ++hi) {
                auto tile = make_input_tile(ii, hi);
                result.operations.push_back(ScheduleOperation::load(tile));
                result.operations.push_back(ScheduleOperation::move(tile));
                result.operations.push_back(ScheduleOperation::feed(tile));
                // VE performs SUM reduction
            }

            // Drain mean to scratch (divide by hidden_size done in VE)
            auto mean_tile = make_scratch_tile(ii, 0);
            result.operations.push_back(ScheduleOperation::drain(mean_tile));
            result.operations.push_back(ScheduleOperation::writeback(mean_tile));

            // ========================================
            // Pass 2: Compute variance
            // ========================================
            for (Size hi = 0; hi < hidden_tiles; ++hi) {
                auto tile = make_input_tile(ii, hi);
                result.operations.push_back(ScheduleOperation::load(tile));
                result.operations.push_back(ScheduleOperation::move(tile));
                result.operations.push_back(ScheduleOperation::feed(tile));
                // VE performs: SUB mean, SQUARE, SUM
            }

            // Drain variance to scratch
            auto var_tile = make_scratch_tile(ii, 1);
            result.operations.push_back(ScheduleOperation::drain(var_tile));
            result.operations.push_back(ScheduleOperation::writeback(var_tile));

            // ========================================
            // Pass 3: Normalize and apply affine
            // ========================================
            for (Size hi = 0; hi < hidden_tiles; ++hi) {
                auto tile = make_input_tile(ii, hi);
                result.operations.push_back(ScheduleOperation::load(tile));
                result.operations.push_back(ScheduleOperation::move(tile));
                result.operations.push_back(ScheduleOperation::feed(tile));
                // VE performs: (x - mean) / sqrt(var + eps) * gamma + beta

                // Feed gamma and beta for this hidden tile
                if (config_.affine) {
                    auto gamma_tile = make_param_tile(hi, true);
                    result.operations.push_back(ScheduleOperation::feed(gamma_tile));
                }

                // Output normalized tile
                auto out_tile = make_output_tile(ii, hi);
                result.operations.push_back(ScheduleOperation::drain(out_tile));
                result.operations.push_back(ScheduleOperation::writeback(out_tile));
                result.operations.push_back(ScheduleOperation::store(out_tile));
            }
        }
    }

    /**
     * @brief Create input tile descriptor
     */
    TileDescriptor make_input_tile(Size ii, Size hi) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::A;
        tile.tile_id.ti = ii;
        tile.tile_id.tj = hi;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();
        tile.dram_address = config_.input_base +
                            (ii * config_.hidden_tiles() + hi) * config_.tile_size_bytes();

        return tile;
    }

    /**
     * @brief Create output tile descriptor
     */
    TileDescriptor make_output_tile(Size ii, Size hi) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;
        tile.tile_id.ti = ii;
        tile.tile_id.tj = hi;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();
        tile.dram_address = config_.output_base +
                            (ii * config_.hidden_tiles() + hi) * config_.tile_size_bytes();

        return tile;
    }

    /**
     * @brief Create parameter tile descriptor (gamma or beta)
     */
    TileDescriptor make_param_tile(Size hi, bool is_gamma) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::B;
        tile.tile_id.ti = 0;
        tile.tile_id.tj = hi;
        tile.tile_id.tk = is_gamma ? 0 : 1;

        tile.height = 1;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.Tj * config_.element_size;

        Address base = is_gamma ? config_.gamma_base : config_.beta_base;
        tile.dram_address = base + hi * tile.size_bytes;

        return tile;
    }

    /**
     * @brief Create scratch tile descriptor (for mean/variance)
     */
    TileDescriptor make_scratch_tile(Size ii, Size index) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;
        tile.tile_id.ti = ii;
        tile.tile_id.tj = index;
        tile.tile_id.tk = 1;  // Distinguish from output

        tile.height = config_.Ti;
        tile.width = 1;  // Mean/var are scalars per instance row
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.Ti * config_.element_size;
        tile.dram_address = config_.scratch_base +
                            (ii * 2 + index) * tile.size_bytes;

        return tile;
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "layernorm_" << config_.batch_size << "x"
           << config_.sequence_length << "x" << config_.hidden_size;
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
