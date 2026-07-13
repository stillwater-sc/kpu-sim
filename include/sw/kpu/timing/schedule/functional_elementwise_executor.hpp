// ============================================================================
// include/sw/kpu/timing/schedule/functional_elementwise_executor.hpp
// Value-producing elementwise execution on the CSP timing model (issue #102)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/elementwise_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::timing::schedule {

/**
 * @brief Apply a VEOp to one element (same IEEE semantics as
 *        BehavioralProgramExecutor::execute_ve_elementwise)
 *
 * Binary ops read both operands; unary ops read `a`; scalar-broadcast ops
 * read `a` and `scalar`. Division by zero, sqrt of negatives, and log of
 * non-positives follow IEEE-754 (inf/NaN propagate to the output, matching
 * the behavioral ISA executor so oracles can compare bit-for-bit).
 */
inline float apply_ve_op(isa::VEOp op, float a, float b, float scalar) {
    using isa::VEOp;
    switch (op) {
        case VEOp::ADD:   return a + b;
        case VEOp::SUB:   return a - b;
        case VEOp::MUL:   return a * b;
        case VEOp::DIV:   return a / b;
        case VEOp::MAX:   return a > b ? a : b;
        case VEOp::MIN:   return a < b ? a : b;
        case VEOp::NEG:   return -a;
        case VEOp::ABS:   return std::fabs(a);
        case VEOp::SQRT:  return std::sqrt(a);
        case VEOp::EXP:   return std::exp(a);
        case VEOp::LOG:   return std::log(a);
        case VEOp::ADD_S: return a + scalar;
        case VEOp::MUL_S: return a * scalar;
        case VEOp::POW_S: return std::pow(a, scalar);
    }
    throw std::invalid_argument("Unknown VEOp");
}

/**
 * @brief Value-producing elementwise execution over the real CSP data path
 *
 * Bridges the #101 ElementwiseScheduleGenerator to the #66 payload
 * machinery: the generator's schedule provides the credit-safe movement
 * plan (paired streams / broadcast delivery), input payloads ride
 * DRAM->L3->L2->L1->compute through the normal LOAD/MOVE/FEED path, each
 * COMPUTE is bound to a FunctionalComputeSpec applying `op`, and results
 * drain back to DRAM through DRAIN/WRITEBACK/STORE. Timing is identical to
 * the timing-only path - the value plane rides the same events.
 *
 * The trailing tile of a non-tile-aligned tensor is partial (the #101
 * clamp); payload slicing and result assembly honor the per-tile footprint.
 */
class FunctionalElementwiseExecutor {
public:
    using Form = ElementwiseScheduleGenerator::Form;

    struct Result {
        ExecutionResult execution;      ///< Timing/completion status
        std::vector<float> values;      ///< C tensor gathered from DRAM stores
    };

    FunctionalElementwiseExecutor(ElementwiseScheduleGenerator::Config generator_config,
                                  ConcurrentTimingExecutor::Config executor_config = {})
        : generator_config_(std::move(generator_config)),
          executor_config_(std::move(executor_config)) {}

    /**
     * @brief Execute C = op(A[, B]) end-to-end on the CSP executor
     *
     * @param op      VEOp consistent with the configured form: binary ops for
     *                BINARY/BROADCAST_B, unary and scalar ops for UNARY
     * @param a       A tensor, num_elements values
     * @param b       B tensor: num_elements (BINARY), one tile of
     *                min(tile_elems, num_elements) values (BROADCAST_B),
     *                empty (UNARY)
     * @param scalar  Operand for ADD_S / MUL_S / POW_S
     */
    Result run(isa::VEOp op, const std::vector<float>& a,
               const std::vector<float>& b = {}, float scalar = 0.0f) {
        validate_inputs(op, a, b);

        ElementwiseScheduleGenerator generator(generator_config_);
        auto schedule = generator.generate();

        Result result;
        if (!schedule.valid) {
            result.execution.error_message =
                "Schedule generation refused: " + schedule.error_message;
            return result;
        }

        ConcurrentTimingExecutor executor(executor_config_);
        seed_input_payloads(executor, schedule, a, b);

        ScheduleExecutor sched_exec(executor);
        sched_exec.set_functional_compute_binder(
            [op, scalar](const ScheduleOperation& compute_op)
                -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
                ConcurrentTimingExecutor::FunctionalComputeSpec spec;
                spec.input_tiles = compute_op.dependency_tiles;
                spec.operation =
                    [op, scalar](const std::vector<TilePayload>& inputs) {
                        return apply_elementwise(op, scalar, inputs);
                    };
                return spec;
            });

        result.execution = sched_exec.execute(schedule);
        if (result.execution.success) {
            result.values = gather_output(executor, schedule);
        }
        return result;
    }

    [[nodiscard]] const ElementwiseScheduleGenerator::Config& generator_config() const {
        return generator_config_;
    }

private:
    ElementwiseScheduleGenerator::Config generator_config_;
    ConcurrentTimingExecutor::Config executor_config_;

    void validate_inputs(isa::VEOp op, const std::vector<float>& a,
                         const std::vector<float>& b) const {
        const auto& cfg = generator_config_;
        if (a.size() != cfg.num_elements) {
            throw std::invalid_argument("A tensor size does not match num_elements");
        }
        const bool binary_op = static_cast<uint8_t>(op) <=
                               static_cast<uint8_t>(isa::VEOp::MIN);
        switch (cfg.form) {
            case Form::BINARY:
                if (!binary_op) {
                    throw std::invalid_argument("BINARY form requires a binary VEOp");
                }
                if (b.size() != cfg.num_elements) {
                    throw std::invalid_argument("B tensor size does not match num_elements");
                }
                break;
            case Form::BROADCAST_B: {
                if (!binary_op) {
                    throw std::invalid_argument("BROADCAST_B form requires a binary VEOp");
                }
                const Size b_elems = std::min(cfg.tile_elems, cfg.num_elements);
                if (b.size() != b_elems) {
                    throw std::invalid_argument(
                        "Broadcast B must be one tile (" + std::to_string(b_elems) +
                        " values)");
                }
                break;
            }
            case Form::UNARY:
                if (binary_op) {
                    throw std::invalid_argument("UNARY form requires a unary or scalar VEOp");
                }
                if (!b.empty()) {
                    throw std::invalid_argument("UNARY form takes no B tensor");
                }
                break;
        }
    }

    /**
     * @brief Seed DRAM payloads for every LOAD in the schedule, sliced by the
     *        tile's own footprint (partial trailing tiles included)
     */
    void seed_input_payloads(ConcurrentTimingExecutor& executor,
                             const ScheduleResult& schedule,
                             const std::vector<float>& a,
                             const std::vector<float>& b) const {
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto& tile = op.tile;
            const std::vector<float>& source =
                tile.tile_id.matrix == isa::MatrixID::A ? a : b;
            const Size offset = tile.tile_id.ti * generator_config_.tile_elems;
            const Size elems = tile.height;
            TilePayload payload{elems, 1,
                                std::vector<float>(source.begin() + offset,
                                                   source.begin() + offset + elems)};
            executor.set_tile_payload(tile.tile_id, std::move(payload));
        }
    }

    /**
     * @brief Elementwise kernel over the fed input payloads
     *
     * Inputs arrive in COMPUTE dependency order: {A} (unary/scalar) or
     * {A, B} (binary/broadcast). A partial trailing A tile pairs with the
     * leading elements of a full broadcast B tile.
     */
    static TilePayload apply_elementwise(isa::VEOp op, float scalar,
                                         const std::vector<TilePayload>& inputs) {
        if (inputs.empty()) {
            throw std::invalid_argument("Elementwise compute received no inputs");
        }
        const auto& a = inputs.front();
        const TilePayload* b = inputs.size() > 1 ? &inputs[1] : nullptr;
        if (b != nullptr && b->values.size() < a.values.size()) {
            throw std::invalid_argument("Elementwise B operand is smaller than A");
        }
        TilePayload out{a.rows, a.cols, std::vector<float>(a.values.size())};
        for (size_t i = 0; i < a.values.size(); ++i) {
            out.values[i] = apply_ve_op(op, a.values[i],
                                        b != nullptr ? b->values[i] : 0.0f, scalar);
        }
        return out;
    }

    /**
     * @brief Reassemble the C tensor from the per-tile DRAM store payloads
     */
    std::vector<float> gather_output(const ConcurrentTimingExecutor& executor,
                                     const ScheduleResult& schedule) const {
        std::vector<float> output(generator_config_.num_elements, 0.0f);
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto& tile = op.tile;
            const auto& payload =
                executor.tile_payload_at(MemoryLevel::DRAM, tile.tile_id);
            const Size offset = tile.tile_id.ti * generator_config_.tile_elems;
            if (payload.values.size() != tile.height) {
                throw std::runtime_error("Stored payload size does not match tile " +
                                         tile.tile_id.to_string());
            }
            std::copy(payload.values.begin(), payload.values.end(),
                      output.begin() + offset);
        }
        return output;
    }
};

} // namespace sw::kpu::timing::schedule
