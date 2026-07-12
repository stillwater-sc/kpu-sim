// ============================================================================
// include/sw/kpu/timing/schedule/softmax_schedule_generator.hpp
// Softmax schedule generator for CSP-style timing model
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <cmath>
#include <sstream>

namespace sw::kpu::timing::schedule {

/**
 * @brief Softmax schedule generator
 *
 * Generates schedules for Softmax operations:
 *   softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Softmax is a multi-pass operation:
 * 1. Pass 1: Find max value along reduction dimension
 * 2. Pass 2: Subtract max and compute exp
 * 3. Pass 3: Sum the exponentials
 * 4. Pass 4: Divide each exp by sum (normalize)
 *
 * For numerical stability, we use the "safe softmax" algorithm that
 * subtracts the max before exponentiating.
 *
 * Usage:
 * ```cpp
 * SoftmaxScheduleGenerator::Config config;
 * config.batch_size = 16;
 * config.sequence_length = 512;
 * config.axis = -1;  // Softmax along last dimension
 *
 * SoftmaxScheduleGenerator generator(config);
 * auto schedule = generator.generate();
 * ```
 */
class SoftmaxScheduleGenerator : public IScheduleGenerator {
public:
    /**
     * @brief Configuration for softmax schedule generation
     */
    struct Config {
        // Input dimensions
        Size batch_size = 1;        ///< Number of independent softmax operations
        Size reduction_dim = 1024;  ///< Size of dimension to reduce over

        // Tiling
        Size Ti = 16;               ///< Tile size for batch dimension
        Size Tj = 16;               ///< Tile size for reduction dimension

        // Element size
        Size element_size = 4;      ///< Bytes per element

        // Resource envelope (issue #90): the buffer capacities this schedule
        // is generated against. Defaults match ConcurrentTimingExecutor.
        Size l3_buffer_count = 32;  ///< L3 credit pool the schedule targets
        Size l2_bank_count = 64;    ///< L2 credit pool the schedule targets

        // Base addresses
        Address input_base = 0;
        Address output_base = 0;
        Address scratch_base = 0;   ///< For intermediate results (max, sum)

        /**
         * @brief Per-matrix burst bound derived from the resource envelope
         */
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }

        /**
         * @brief Peak tile residency this multi-pass schedule implies
         *
         * The exp scratch tiles written back in pass 2 stay resident until
         * pass 4 consumes them, plus the max and sum scratch tiles:
         * reduction_tiles + 2. A schedule whose working set exceeds the
         * envelope share would wedge at runtime, so generate() refuses it
         * a priori (the online single-pass softmax - epic E8, issue #77 -
         * removes this residency requirement).
         */
        [[nodiscard]] Size required_working_set() const {
            return reduction_tiles() + 2;
        }

        /**
         * @brief Calculate number of batch tiles
         */
        [[nodiscard]] Size batch_tiles() const {
            return (batch_size + Ti - 1) / Ti;
        }

        /**
         * @brief Calculate number of reduction tiles
         */
        [[nodiscard]] Size reduction_tiles() const {
            return (reduction_dim + Tj - 1) / Tj;
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
    explicit SoftmaxScheduleGenerator(const Config& config)
        : config_(config) {}

    /**
     * @brief Generate the schedule
     */
    ScheduleResult generate() override {
        ScheduleResult result;

        // Validate configuration
        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "softmax multi-pass schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles (reduction_tiles + 2 scratch) but the resource "
                "envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count or use the online "
                "single-pass softmax (epic E8, issue #77)";
            return result;
        }

        if (config_.batch_size == 0 || config_.reduction_dim == 0) {
            result.valid = false;
            result.error_message = "Dimensions must be non-zero";
            return result;
        }

        // Generate multi-pass softmax schedule
        generate_multi_pass(result);

        // Set metadata
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.batch_size;
        result.metadata.N = config_.reduction_dim;
        result.metadata.K = 1;  // Not a matmul
        result.metadata.Ti = config_.Ti;
        result.metadata.Tj = config_.Tj;
        result.metadata.Tk = 1;

        // Tile counts: input tiles + output tiles + scratch
        Size input_tiles = config_.batch_tiles() * config_.reduction_tiles();
        result.metadata.a_tiles = input_tiles;
        result.metadata.b_tiles = 0;  // No B matrix
        result.metadata.c_tiles = input_tiles;  // Same shape output

        result.metadata.strategy = "multi_pass_safe_softmax";
        result.valid = true;

        return result;
    }

    /**
     * @brief Get generator name
     */
    [[nodiscard]] std::string name() const override {
        return "SoftmaxScheduleGenerator";
    }

    /**
     * @brief Get description
     */
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "Softmax [" << config_.batch_size << " x " << config_.reduction_dim << "]";
        return ss.str();
    }

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    /**
     * @brief Generate multi-pass softmax schedule
     *
     * Pass structure:
     * 1. LOAD input tiles, compute running max (FEED -> Vector Engine MAX reduce)
     * 2. LOAD input tiles again, subtract max and exp (FEED -> VE SUB, EXP)
     * 3. Compute sum of exponentials (VE SUM reduce)
     * 4. Divide by sum to normalize (VE DIV)
     * 5. DRAIN and STORE output tiles
     */
    void generate_multi_pass(ScheduleResult& result) {
        Size batch_tiles = config_.batch_tiles();
        Size reduction_tiles = config_.reduction_tiles();

        // For each batch tile row
        for (Size bi = 0; bi < batch_tiles; ++bi) {
            // ========================================
            // Pass 1: Find max over reduction dimension
            // ========================================
            for (Size ri = 0; ri < reduction_tiles; ++ri) {
                auto tile = make_input_tile(bi, ri);
                result.operations.push_back(ScheduleOperation::load(tile));
                result.operations.push_back(ScheduleOperation::move(tile));
                result.operations.push_back(ScheduleOperation::feed(tile));
                // VE would perform MAX reduction - modeled as compute
            }

            // Drain max result to scratch
            auto max_tile = make_scratch_tile(bi, 0);
            result.operations.push_back(ScheduleOperation::drain(max_tile));
            result.operations.push_back(ScheduleOperation::writeback(max_tile));

            // ========================================
            // Pass 2: Compute exp(x - max)
            // ========================================
            for (Size ri = 0; ri < reduction_tiles; ++ri) {
                auto tile = make_input_tile(bi, ri);
                result.operations.push_back(ScheduleOperation::load(tile));
                result.operations.push_back(ScheduleOperation::move(tile));
                result.operations.push_back(ScheduleOperation::feed(tile));
                // VE performs: SUB max, then EXP
            }

            // Drain exp results to intermediate storage
            for (Size ri = 0; ri < reduction_tiles; ++ri) {
                auto exp_tile = make_scratch_tile(bi, ri + 1);
                result.operations.push_back(ScheduleOperation::drain(exp_tile));
                result.operations.push_back(ScheduleOperation::writeback(exp_tile));
            }

            // ========================================
            // Pass 3: Sum exponentials
            // ========================================
            for (Size ri = 0; ri < reduction_tiles; ++ri) {
                auto exp_tile = make_scratch_tile(bi, ri + 1);
                result.operations.push_back(ScheduleOperation::load(exp_tile));
                result.operations.push_back(ScheduleOperation::move(exp_tile));
                result.operations.push_back(ScheduleOperation::feed(exp_tile));
                // VE performs SUM reduction
            }

            // Drain sum to scratch
            auto sum_tile = make_scratch_tile(bi, reduction_tiles + 1);
            result.operations.push_back(ScheduleOperation::drain(sum_tile));
            result.operations.push_back(ScheduleOperation::writeback(sum_tile));

            // ========================================
            // Pass 4: Normalize (divide by sum)
            // ========================================
            for (Size ri = 0; ri < reduction_tiles; ++ri) {
                // Load exp values
                auto exp_tile = make_scratch_tile(bi, ri + 1);
                result.operations.push_back(ScheduleOperation::load(exp_tile));
                result.operations.push_back(ScheduleOperation::move(exp_tile));
                result.operations.push_back(ScheduleOperation::feed(exp_tile));
                // VE performs DIV by sum

                // Output normalized tile
                auto out_tile = make_output_tile(bi, ri);
                result.operations.push_back(ScheduleOperation::drain(out_tile));
                result.operations.push_back(ScheduleOperation::writeback(out_tile));
                result.operations.push_back(ScheduleOperation::store(out_tile));
            }
        }
    }

    /**
     * @brief Create input tile descriptor
     */
    TileDescriptor make_input_tile(Size bi, Size ri) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::A;  // Use A for input
        tile.tile_id.ti = bi;
        tile.tile_id.tj = ri;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();
        tile.dram_address = config_.input_base +
                            (bi * config_.reduction_tiles() + ri) * config_.tile_size_bytes();

        return tile;
    }

    /**
     * @brief Create output tile descriptor
     */
    TileDescriptor make_output_tile(Size bi, Size ri) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;  // Use C for output
        tile.tile_id.ti = bi;
        tile.tile_id.tj = ri;
        tile.tile_id.tk = 0;

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();
        tile.dram_address = config_.output_base +
                            (bi * config_.reduction_tiles() + ri) * config_.tile_size_bytes();

        return tile;
    }

    /**
     * @brief Create scratch tile descriptor (for max, exp, sum intermediates)
     */
    TileDescriptor make_scratch_tile(Size bi, Size index) const {
        TileDescriptor tile;
        tile.tile_id.matrix = isa::MatrixID::C;  // Reuse C for scratch
        tile.tile_id.ti = bi;
        tile.tile_id.tj = index;
        tile.tile_id.tk = 1;  // Use tk to distinguish scratch from output

        tile.height = config_.Ti;
        tile.width = config_.Tj;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_size_bytes();
        tile.dram_address = config_.scratch_base +
                            (bi * (config_.reduction_tiles() + 2) + index) * config_.tile_size_bytes();

        return tile;
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "softmax_" << config_.batch_size << "x" << config_.reduction_dim;
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
