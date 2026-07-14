// ============================================================================
// include/sw/kpu/timing/schedule/functional_reduction_executor.hpp
// Value-producing streaming reduction on the CSP timing model (issue #107)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/online_reduction_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::schedule {

/**
 * @brief Reduce a set of streamed tile payloads to a finalized statistic
 *
 * Matches the VE_REDUCE behavioral ABI (issue #105): population variance,
 * clamped so cancellation never yields a negative value; empty -> NaN. The
 * result rides lane 0 of a payload sized to the stat tile footprint.
 */
inline TilePayload reduce_payloads(OnlineReductionScheduleGenerator::ReduceOp op,
                                   const std::vector<TilePayload>& inputs) {
    using ReduceOp = OnlineReductionScheduleGenerator::ReduceOp;
    if (inputs.empty()) {
        throw std::invalid_argument("Reduction compute received no inputs");
    }

    double acc_max = -std::numeric_limits<double>::infinity();
    double acc_min =  std::numeric_limits<double>::infinity();
    double sum = 0.0, sumsq = 0.0;
    size_t count = 0;
    for (const auto& tile : inputs) {
        for (float v : tile.values) {
            acc_max = v > acc_max ? v : acc_max;
            acc_min = v < acc_min ? v : acc_min;
            sum += v;
            sumsq += static_cast<double>(v) * v;
            ++count;
        }
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    double stat = 0.0;
    switch (op) {
        case ReduceOp::MAX: stat = acc_max; break;
        case ReduceOp::MIN: stat = acc_min; break;
        case ReduceOp::SUM: stat = sum; break;
        case ReduceOp::MEAN: stat = count > 0 ? sum / count : nan; break;
        case ReduceOp::VAR: {
            if (count == 0) { stat = nan; }
            else if (count == 1) { stat = 0.0; }
            else {
                const double mean = sum / count;
                stat = std::max(0.0, sumsq / count - mean * mean);
            }
            break;
        }
    }

    // Size the payload to the stat tile footprint (lane 0 = stat, rest 0)
    const Size lanes = inputs.front().values.empty()
        ? 1 : static_cast<Size>(inputs.front().values.size());
    TilePayload out{lanes, 1, std::vector<float>(lanes, 0.0f)};
    out.values[0] = static_cast<float>(stat);
    return out;
}

/**
 * @brief Value-producing streaming reduction over the real CSP data path
 *
 * Bridges the #106 OnlineReductionScheduleGenerator to the #66 payload
 * machinery for the stats-producing forms (FULL_REDUCE, ROW_STATS): input
 * tiles ride DRAM->L3->L2->L1->compute, the per-row stat COMPUTE reduces
 * every streamed feed to a finalized statistic, and the stat drains back to
 * DRAM. Verified elementwise against a host oracle in the tests.
 *
 * ROW_NORMALIZE's apply phase is intentionally out of scope here: its stat
 * must reach the apply computes as a compute-resident dependency (no DRAM
 * round-trip race), which lands with the softmax/layernorm generators
 * (E8/E9). This executor rejects ROW_NORMALIZE configs.
 */
class FunctionalReductionExecutor {
public:
    using ReduceOp = OnlineReductionScheduleGenerator::ReduceOp;
    using Form = OnlineReductionScheduleGenerator::Form;

    struct Result {
        ExecutionResult execution;        ///< Timing/completion status
        std::vector<float> stats;         ///< One finalized stat per row
    };

    FunctionalReductionExecutor(OnlineReductionScheduleGenerator::Config generator_config,
                                ConcurrentTimingExecutor::Config executor_config = {})
        : generator_config_(std::move(generator_config)),
          executor_config_(std::move(executor_config)) {
        if (generator_config_.form == Form::ROW_NORMALIZE) {
            throw std::invalid_argument(
                "FunctionalReductionExecutor supports FULL_REDUCE and ROW_STATS "
                "only; ROW_NORMALIZE apply-phase numerics land with E8/E9");
        }
    }

    /**
     * @brief Reduce the input stream and return one stat per row
     * @param data  row-major: num_rows x reduction_elems values
     */
    Result run(const std::vector<float>& data) {
        const auto& cfg = generator_config_;
        const Size rows = cfg.form == Form::FULL_REDUCE ? 1 : cfg.num_rows;
        if (data.size() != static_cast<size_t>(rows) * cfg.reduction_elems) {
            throw std::invalid_argument(
                "input size does not match num_rows x reduction_elems");
        }

        OnlineReductionScheduleGenerator generator(cfg);
        auto schedule = generator.generate();

        Result result;
        if (!schedule.valid) {
            result.execution.error_message =
                "Schedule generation refused: " + schedule.error_message;
            return result;
        }

        ConcurrentTimingExecutor executor(executor_config_);
        seed_input_payloads(executor, schedule, data);

        ScheduleExecutor sched_exec(executor);
        const ReduceOp op = cfg.op;
        sched_exec.set_functional_compute_binder(
            [op](const ScheduleOperation& compute_op)
                -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
                ConcurrentTimingExecutor::FunctionalComputeSpec spec;
                spec.input_tiles = compute_op.dependency_tiles;
                spec.operation = [op](const std::vector<TilePayload>& inputs) {
                    return reduce_payloads(op, inputs);
                };
                return spec;
            });

        result.execution = sched_exec.execute(schedule);
        if (result.execution.success) {
            result.stats = gather_stats(executor, schedule, rows);
        }
        return result;
    }

private:
    OnlineReductionScheduleGenerator::Config generator_config_;
    ConcurrentTimingExecutor::Config executor_config_;

    void seed_input_payloads(ConcurrentTimingExecutor& executor,
                             const ScheduleResult& schedule,
                             const std::vector<float>& data) const {
        const Size rt = generator_config_.reduction_tiles();
        const Size te = generator_config_.tile_elems;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto& tile = op.tile;
            // Input tiles ride matrix A; index is (row=ti, reduction-tile=tj)
            const Size row = tile.tile_id.ti;
            const Size t = tile.tile_id.tj;
            const Size row_base = row * generator_config_.reduction_elems;
            const Size offset = row_base + t * te;
            const Size elems = tile.height;
            (void)rt;
            TilePayload payload{elems, 1,
                                std::vector<float>(data.begin() + offset,
                                                   data.begin() + offset + elems)};
            executor.set_tile_payload(tile.tile_id, std::move(payload));
        }
    }

    std::vector<float> gather_stats(const ConcurrentTimingExecutor& executor,
                                    const ScheduleResult& schedule, Size rows) const {
        std::vector<float> stats(rows, 0.0f);
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto& tile = op.tile;
            if (tile.tile_id.matrix != isa::MatrixID::B) continue;  // stat tile
            const auto& payload =
                executor.tile_payload_at(MemoryLevel::DRAM, tile.tile_id);
            stats[tile.tile_id.ti] = payload.values.at(0);
        }
        return stats;
    }
};

} // namespace sw::kpu::timing::schedule
