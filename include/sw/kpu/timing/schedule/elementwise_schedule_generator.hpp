// ============================================================================
// include/sw/kpu/timing/schedule/elementwise_schedule_generator.hpp
// Elementwise schedule generator for CSP-style timing model (issue #101)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <sstream>
#include <string>

namespace sw::kpu::timing::schedule {

/**
 * @brief Emit a broadcast tile delivery: one LOAD + one MOVE feeding k
 *        consumers (pattern P5, issue #100/#101)
 *
 * The MOVE carries the consumer count, so the BlockMover seeds the L2
 * TagCAM reference count with k on arrival (the 1:1:k discipline from
 * docs/plans/e2_broadcast_elementwise_pattern.md). The caller emits the
 * k FEED operations at their consumption points; the tile holds exactly
 * one L2 credit for its whole consumer span, released after the k-th
 * feed. Usable by the norm/epilogue generators for resident parameters.
 */
inline void emit_broadcast_tile(ScheduleResult& result,
                                TileDescriptor tile, Size consumers) {
    tile.consumer_count = consumers > 0 ? consumers : 1;
    result.operations.push_back(ScheduleOperation::load(tile));
    result.operations.push_back(ScheduleOperation::move(tile));
}

/**
 * @brief Elementwise schedule generator: C = op(A, B), op(A), or
 *        op(A, broadcast B)
 *
 * Pattern P6 (two-operand aligned streaming): the binary form emits
 * paired A/B tile streams (load A[i], load B[i], ... interleaved per the
 * #67 discipline) so neither stream can monopolize the credit pools, and
 * COMPUTE for C[i] depends on BOTH operand feeds - the per-instance feed
 * accounting fires it only when the pair has arrived. There is no tile
 * reuse: the working set is one A + one B + one C tile in flight
 * regardless of tensor size.
 *
 * The broadcast form (P5) streams A against a single resident B tile
 * (e.g. a bias vector) delivered once via emit_broadcast_tile.
 *
 * Unlike the earlier multi-pass generators, every emitted schedule is
 * EXECUTABLE: COMPUTE operations carry their full dependency sets
 * (resolving #139 for the elementwise family).
 */
class ElementwiseScheduleGenerator : public IScheduleGenerator {
public:
    enum class Form {
        BINARY,      ///< C[i] = op(A[i], B[i]) - paired streams
        UNARY,       ///< C[i] = op(A[i]) - single stream
        BROADCAST_B  ///< C[i] = op(A[i], B) - A streams, B resident
    };

    struct Config {
        Size num_elements = 4096;   ///< Total elements per tensor
        Size tile_elems = 256;      ///< Elements per tile
        Form form = Form::BINARY;
        std::string op_name = "ADD";  ///< For naming/metadata only

        Size element_size = 4;

        // Resource envelope (issue #90 discipline)
        Size l3_buffer_count = 32;
        Size l2_bank_count = 64;

        Address a_base = 0;
        Address b_base = 0;
        Address c_base = 0;

        [[nodiscard]] Size data_tiles() const {
            return (num_elements + tile_elems - 1) / tile_elems;
        }

        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }

        /**
         * @brief Peak tile residency: one tile per active stream plus the
         *        pending output, plus the resident broadcast operand
         */
        [[nodiscard]] Size required_working_set() const {
            switch (form) {
                case Form::BINARY:      return 3;
                case Form::UNARY:       return 2;
                case Form::BROADCAST_B: return 3;  // A + resident B + C
            }
            return 3;
        }
    };

    explicit ElementwiseScheduleGenerator(const Config& config)
        : config_(config) {}

    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.num_elements == 0 || config_.tile_elems == 0) {
            result.valid = false;
            result.error_message = "Elementwise dimensions must be non-zero";
            return result;
        }

        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "elementwise schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }

        Size n = config_.data_tiles();

        // Resident broadcast operand delivered once, consumed n times
        TileDescriptor b_broadcast;
        if (config_.form == Form::BROADCAST_B) {
            b_broadcast = make_tile(isa::MatrixID::B, 0);
            emit_broadcast_tile(result, b_broadcast, n);
        }

        for (Size i = 0; i < n; ++i) {
            auto a_tile = make_tile(isa::MatrixID::A, i);

            std::vector<TileID> deps;
            if (config_.form == Form::BINARY) {
                auto b_tile = make_tile(isa::MatrixID::B, i);
                // Paired interleaved emission: A and B advance together
                result.operations.push_back(ScheduleOperation::load(a_tile));
                result.operations.push_back(ScheduleOperation::load(b_tile));
                result.operations.push_back(ScheduleOperation::move(a_tile));
                result.operations.push_back(ScheduleOperation::move(b_tile));
                result.operations.push_back(ScheduleOperation::feed(a_tile));
                result.operations.push_back(ScheduleOperation::feed(b_tile));
                deps = {a_tile.tile_id, b_tile.tile_id};
            } else {
                result.operations.push_back(ScheduleOperation::load(a_tile));
                result.operations.push_back(ScheduleOperation::move(a_tile));
                result.operations.push_back(ScheduleOperation::feed(a_tile));
                deps = {a_tile.tile_id};
                if (config_.form == Form::BROADCAST_B) {
                    // The i-th consumption of the resident broadcast tile
                    result.operations.push_back(
                        ScheduleOperation::feed(b_broadcast));
                    deps.push_back(b_broadcast.tile_id);
                }
            }

            auto c_tile = make_tile(isa::MatrixID::C, i);
            result.operations.push_back(
                ScheduleOperation::compute(c_tile, std::move(deps)));
            result.operations.push_back(ScheduleOperation::drain(c_tile));
            result.operations.push_back(ScheduleOperation::writeback(c_tile));
            result.operations.push_back(ScheduleOperation::store(c_tile));
        }

        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.num_elements;
        result.metadata.N = 1;
        result.metadata.K = 1;
        result.metadata.Ti = config_.tile_elems;
        result.metadata.Tj = 1;
        result.metadata.Tk = 1;
        result.metadata.a_tiles = n;
        result.metadata.b_tiles =
            config_.form == Form::BINARY ? n
            : config_.form == Form::BROADCAST_B ? 1 : 0;
        result.metadata.c_tiles = n;
        result.metadata.strategy = form_name(config_.form);
        result.metadata.l3_buffer_count = config_.l3_buffer_count;
        result.metadata.l2_bank_count = config_.l2_bank_count;

        result.valid = true;
        return result;
    }

    [[nodiscard]] std::string name() const override {
        return "ElementwiseScheduleGenerator";
    }

    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "Elementwise " << config_.op_name << " [" << config_.num_elements
           << "] (" << form_name(config_.form) << ")";
        return ss.str();
    }

    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    TileDescriptor make_tile(isa::MatrixID matrix, Size index) const {
        TileDescriptor tile;
        tile.tile_id.matrix = matrix;
        tile.tile_id.ti = index;
        tile.tile_id.tj = 0;
        tile.tile_id.tk = 0;
        tile.height = config_.tile_elems;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_elems * config_.element_size;

        Address base = matrix == isa::MatrixID::A ? config_.a_base
                     : matrix == isa::MatrixID::B ? config_.b_base
                                                  : config_.c_base;
        tile.dram_address = base + index * tile.size_bytes;
        tile.matrix_base_address = base;
        return tile;
    }

    static const char* form_name(Form form) {
        switch (form) {
            case Form::BINARY:      return "binary_paired";
            case Form::UNARY:       return "unary_stream";
            case Form::BROADCAST_B: return "broadcast_b";
        }
        return "unknown";
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "elementwise_" << config_.op_name << "_" << config_.num_elements
           << "_" << form_name(config_.form);
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
