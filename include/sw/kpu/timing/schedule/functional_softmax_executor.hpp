// ============================================================================
// include/sw/kpu/timing/schedule/functional_softmax_executor.hpp
// Value-producing online softmax on the CSP timing model (issue #157)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/online_softmax_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::schedule {

/**
 * @brief Stats op: reduce the row's tiles to the (m, l) softmax state
 *
 * m = max over the row, l = sum_j exp(x_j - m). The stats COMPUTE receives
 * every row tile at once, so the mathematically-equivalent two-accumulator
 * safe form is used (no streaming rescale needed). The m == -inf guard (an
 * all-(-inf) / fully-masked row) leaves l = 0 rather than evaluating
 * exp(-inf - -inf) = NaN; the apply then emits a uniform distribution.
 * Result rides lanes [m, l] of a 2-element payload.
 */
inline TilePayload softmax_stats(const std::vector<TilePayload>& inputs) {
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& tile : inputs) {
        for (float v : tile.values) if (v > m) m = v;
    }
    double l = 0.0;
    if (m > -std::numeric_limits<double>::infinity()) {
        for (const auto& tile : inputs) {
            for (float v : tile.values) l += std::exp(static_cast<double>(v) - m);
        }
    }
    return TilePayload{2, 1, {static_cast<float>(m), static_cast<float>(l)}};
}

/**
 * @brief Apply op: normalize one row tile with the resident (m, l)
 *
 * inputs = [ x_tile (fed), (m, l) state (resident) ]. Emits
 * exp(x - m)/l elementwise, or the uniform 1/row_elems when l == 0
 * (nonempty all-(-inf) row).
 */
inline TilePayload softmax_apply(const std::vector<TilePayload>& inputs,
                                 Size row_elems) {
    if (inputs.size() < 2) {
        throw std::invalid_argument("softmax apply needs the tile and the (m,l) state");
    }
    const auto& x = inputs[0];
    const float m = inputs[1].values.at(0);
    const float l = inputs[1].values.at(1);
    TilePayload out{x.rows, x.cols, std::vector<float>(x.values.size())};
    if (l > 0.0f) {
        for (size_t i = 0; i < x.values.size(); ++i) {
            out.values[i] = std::exp(x.values[i] - m) / l;
        }
    } else {
        const float uniform = 1.0f / static_cast<float>(row_elems);
        for (float& v : out.values) v = uniform;
    }
    return out;
}

/**
 * @brief Value-producing online softmax over the real CSP data path
 *
 * Bridges the #156 OnlineSoftmaxScheduleGenerator to the #66 payload
 * machinery: the stats COMPUTE produces the (m, l) state, which the apply
 * COMPUTEs consume as a compute-RESIDENT dependency (the #155 mechanism) -
 * no DRAM round-trip - and each apply emits its normalized output tile.
 * Verified elementwise against a host safe-softmax oracle in the tests.
 */
class FunctionalSoftmaxExecutor {
public:
    struct Result {
        ExecutionResult execution;      ///< Timing/completion status
        std::vector<float> values;      ///< num_rows x reduction_elems softmax output
    };

    FunctionalSoftmaxExecutor(OnlineSoftmaxScheduleGenerator::Config generator_config,
                              ConcurrentTimingExecutor::Config executor_config = {})
        : generator_config_(std::move(generator_config)),
          executor_config_(std::move(executor_config)) {}

    Result run(const std::vector<float>& data) {
        const auto& cfg = generator_config_;
        if (data.size() != static_cast<size_t>(cfg.num_rows) * cfg.reduction_elems) {
            throw std::invalid_argument(
                "input size does not match num_rows x reduction_elems");
        }

        OnlineSoftmaxScheduleGenerator generator(cfg);
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
        const Size row_elems = cfg.reduction_elems;
        sched_exec.set_functional_compute_binder(
            [row_elems](const ScheduleOperation& op)
                -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
                ConcurrentTimingExecutor::FunctionalComputeSpec spec;
                // Fed inputs first, then the resident (m,l) state
                spec.input_tiles = op.dependency_tiles;
                for (const auto& r : op.resident_tiles) spec.input_tiles.push_back(r);
                spec.resident_tiles = op.resident_tiles;
                if (op.tile.tile_id.matrix == isa::MatrixID::B) {
                    spec.operation = softmax_stats;                    // (m, l)
                } else {
                    spec.operation = [row_elems](const std::vector<TilePayload>& in) {
                        return softmax_apply(in, row_elems);            // exp(x-m)/l
                    };
                }
                return spec;
            });

        result.execution = sched_exec.execute(schedule);
        if (result.execution.success) {
            result.values = gather_output(executor, schedule);
        }
        return result;
    }

private:
    OnlineSoftmaxScheduleGenerator::Config generator_config_;
    ConcurrentTimingExecutor::Config executor_config_;

    void seed_input_payloads(ConcurrentTimingExecutor& executor,
                             const ScheduleResult& schedule,
                             const std::vector<float>& data) const {
        const Size te = generator_config_.tile_elems;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto& tile = op.tile;
            if (tile.tile_id.matrix != isa::MatrixID::A) continue;  // input only
            const Size row = tile.tile_id.ti;
            const Size t = tile.tile_id.tj;
            const Size offset = row * generator_config_.reduction_elems + t * te;
            const Size elems = tile.height;
            TilePayload payload{elems, 1,
                                std::vector<float>(data.begin() + offset,
                                                   data.begin() + offset + elems)};
            executor.set_tile_payload(tile.tile_id, std::move(payload));
        }
    }

    std::vector<float> gather_output(const ConcurrentTimingExecutor& executor,
                                     const ScheduleResult& schedule) const {
        const Size te = generator_config_.tile_elems;
        std::vector<float> out(
            static_cast<size_t>(generator_config_.num_rows) *
            generator_config_.reduction_elems, 0.0f);
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto& tile = op.tile;
            if (tile.tile_id.matrix != isa::MatrixID::C) continue;
            const auto& payload =
                executor.tile_payload_at(MemoryLevel::DRAM, tile.tile_id);
            const Size offset = tile.tile_id.ti * generator_config_.reduction_elems +
                                tile.tile_id.tj * te;
            std::copy(payload.values.begin(), payload.values.end(),
                      out.begin() + offset);
        }
        return out;
    }
};

} // namespace sw::kpu::timing::schedule
