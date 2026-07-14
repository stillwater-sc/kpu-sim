// ============================================================================
// include/sw/kpu/timing/schedule/online_softmax_schedule_generator.hpp
// Online single-pass softmax schedule generator (issue #156, epic E8)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/elementwise_schedule_generator.hpp>  // emit_broadcast_tile
#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>

#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::timing::schedule {

/**
 * @brief Online (single-pass) softmax schedule generator (pattern P3)
 *
 * softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m), m = max_j x_j. The
 * online form computes m and the normalizer l = sum_j exp(x_j - m) in ONE
 * streaming stats pass (running max with sum rescaling), then an apply
 * pass emits exp(x - m)/l. The running (m, l) state produced by the stats
 * COMPUTE is handed to the apply COMPUTEs as a compute-RESIDENT dependency
 * (issue #155) - no drain/reload, no DRAM round-trip race - which is what
 * distinguishes this from the reduction ROW_NORMALIZE (E3) and supersedes
 * the 4-pass SoftmaxScheduleGenerator (resolving #139 for softmax).
 *
 * Realization is chosen a priori from the envelope (the #67 discipline):
 *  - ROW_RESIDENT (reduction_tiles + 2 <= per-matrix burst share): the row
 *    is delivered once with consumer_count=2 (fed for stats, fed again for
 *    apply). DRAM traffic 1 read + 1 write per element - the payoff.
 *  - RESTREAMED (otherwise): the row is re-read for the apply pass, so only
 *    3 tiles are ever in flight. The (m, l) state rides the resident path
 *    either way.
 *
 * Every emitted schedule is executable: COMPUTEs carry their full
 * dependency sets (the #101 discipline).
 */
class OnlineSoftmaxScheduleGenerator : public IScheduleGenerator {
public:
    enum class Realization { ROW_RESIDENT, RESTREAMED };

    struct Config {
        Size num_rows = 1;            ///< Independent softmax rows (batch)
        Size reduction_elems = 4096;  ///< Softmax dimension length per row
        Size tile_elems = 256;        ///< Elements per tile

        Size element_size = 4;

        // Resource envelope (issue #90 discipline)
        Size l3_buffer_count = 32;
        Size l2_bank_count = 64;

        Address in_base = 0;
        Address stat_base = 0;
        Address out_base = 0;

        [[nodiscard]] Size reduction_tiles() const {
            return (reduction_elems + tile_elems - 1) / tile_elems;
        }
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }
        [[nodiscard]] Realization realization() const {
            return reduction_tiles() + 2 <= max_burst_tiles()
                ? Realization::ROW_RESIDENT
                : Realization::RESTREAMED;
        }
        /**
         * @brief Peak tile residency
         *
         * ROW_RESIDENT holds the whole row resident across its two phases
         * (reduction_tiles) plus an in-flight output; the (m, l) stat is
         * compute-resident, not an L2 tile. RESTREAMED re-reads the row so
         * only 3 tiles are ever in flight.
         */
        [[nodiscard]] Size required_working_set() const {
            return realization() == Realization::ROW_RESIDENT
                ? reduction_tiles() + 2
                : 3;
        }
    };

    explicit OnlineSoftmaxScheduleGenerator(const Config& config)
        : config_(config) {}

    ScheduleResult generate() override {
        ScheduleResult result;
        if (config_.num_rows == 0 || config_.reduction_elems == 0 ||
            config_.tile_elems == 0) {
            result.valid = false;
            result.error_message = "Softmax dimensions must be non-zero";
            return result;
        }
        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "online softmax requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }

        for (Size r = 0; r < config_.num_rows; ++r) {
            if (config_.realization() == Realization::ROW_RESIDENT) {
                emit_row_resident(result, r);
            } else {
                emit_restreamed(result, r);
            }
        }

        stamp_metadata(result);
        result.valid = true;
        return result;
    }

    [[nodiscard]] std::string name() const override {
        return "OnlineSoftmaxScheduleGenerator";
    }
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "OnlineSoftmax [" << config_.num_rows << "x"
           << config_.reduction_elems << "] (" << strategy_name() << ")";
        return ss.str();
    }
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    // ROW_RESIDENT: deliver each row tile once (consumer_count=2), stats
    // COMPUTE produces (m, l), apply COMPUTEs consume the row tile's second
    // feed plus the resident (m, l).
    void emit_row_resident(ScheduleResult& result, Size row) {
        const Size rt = config_.reduction_tiles();

        std::vector<TileDescriptor> row_tiles;
        row_tiles.reserve(rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            in.consumer_count = 2;
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            row_tiles.push_back(in);
        }

        // Stats pass: one COMPUTE over the first consumption of every tile
        std::vector<TileID> stat_deps;
        stat_deps.reserve(rt);
        for (const auto& in : row_tiles) {
            result.operations.push_back(ScheduleOperation::feed(in));
            stat_deps.push_back(in.tile_id);
        }
        auto stat = make_stat_tile(row);
        result.operations.push_back(ScheduleOperation::compute(stat, std::move(stat_deps)));

        // Apply pass: (m, l) stays RESIDENT - no drain/store/reload
        for (Size t = 0; t < rt; ++t) {
            result.operations.push_back(ScheduleOperation::feed(row_tiles[t]));
            emit_apply(result, row, t, row_tiles[t].tile_id, stat.tile_id);
        }
    }

    // RESTREAMED: stats pass on a fresh delivery, then re-read the row for
    // the apply pass; (m, l) rides the resident path across both.
    void emit_restreamed(ScheduleResult& result, Size row) {
        const Size rt = config_.reduction_tiles();

        std::vector<TileID> stat_deps;
        stat_deps.reserve(rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            result.operations.push_back(ScheduleOperation::feed(in));
            stat_deps.push_back(in.tile_id);
        }
        auto stat = make_stat_tile(row);
        result.operations.push_back(ScheduleOperation::compute(stat, std::move(stat_deps)));

        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            result.operations.push_back(ScheduleOperation::feed(in));
            emit_apply(result, row, t, in.tile_id, stat.tile_id);
        }
    }

    // One normalized output tile: apply COMPUTE consuming the fed input and
    // the resident (m, l), then drain/writeback/store.
    void emit_apply(ScheduleResult& result, Size row, Size t,
                    const TileID& in_id, const TileID& stat_id) {
        auto out = make_output_tile(row, t);
        result.operations.push_back(ScheduleOperation::compute(
            out, /*fed*/ std::vector<TileID>{in_id},
            /*resident*/ std::vector<TileID>{stat_id}));
        result.operations.push_back(ScheduleOperation::drain(out));
        result.operations.push_back(ScheduleOperation::writeback(out));
        result.operations.push_back(ScheduleOperation::store(out));
    }

    TileDescriptor make_input_tile(Size row, Size t) const {
        return make_tile(isa::MatrixID::A, row, t, config_.in_base);
    }
    TileDescriptor make_stat_tile(Size row) const {
        return make_tile(isa::MatrixID::B, row, 0, config_.stat_base);
    }
    TileDescriptor make_output_tile(Size row, Size t) const {
        return make_tile(isa::MatrixID::C, row, t, config_.out_base);
    }

    TileDescriptor make_tile(isa::MatrixID matrix, Size row, Size t,
                             Address base) const {
        TileDescriptor tile;
        tile.tile_id.matrix = matrix;
        tile.tile_id.ti = row;
        tile.tile_id.tj = t;
        tile.tile_id.tk = 0;

        // A/C tiles span the reduction dim: clamp the trailing partial tile
        // (full-tile stride preserved). Stat tiles (B) are one per row.
        Size elems = config_.tile_elems;
        if (matrix != isa::MatrixID::B) {
            const Size within_row = t * config_.tile_elems;
            if (within_row + elems > config_.reduction_elems) {
                elems = config_.reduction_elems - within_row;
            }
        }
        tile.height = elems;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = elems * config_.element_size;

        const Size full_bytes = config_.tile_elems * config_.element_size;
        const Size linear = matrix == isa::MatrixID::B
            ? row
            : row * config_.reduction_tiles() + t;
        tile.dram_address = base + linear * full_bytes;
        tile.matrix_base_address = base;
        return tile;
    }

    void stamp_metadata(ScheduleResult& result) const {
        const Size rt = config_.reduction_tiles();
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.num_rows;
        result.metadata.N = config_.reduction_elems;
        result.metadata.K = 1;
        result.metadata.Ti = config_.tile_elems;
        result.metadata.Tj = 1;
        result.metadata.Tk = 1;
        result.metadata.a_tiles = config_.num_rows * rt;
        result.metadata.b_tiles = config_.num_rows;   // one (m,l) per row
        result.metadata.c_tiles = config_.num_rows * rt;
        result.metadata.strategy = strategy_name();
        result.metadata.l3_buffer_count = config_.l3_buffer_count;
        result.metadata.l2_bank_count = config_.l2_bank_count;
    }

    std::string strategy_name() const {
        return config_.realization() == Realization::ROW_RESIDENT
            ? "online_row_resident" : "online_restreamed";
    }
    std::string generate_name() const {
        std::ostringstream ss;
        ss << "online_softmax_" << config_.num_rows << "x"
           << config_.reduction_elems << "_" << strategy_name();
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
