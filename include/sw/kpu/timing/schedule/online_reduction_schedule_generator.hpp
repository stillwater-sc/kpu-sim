// ============================================================================
// include/sw/kpu/timing/schedule/online_reduction_schedule_generator.hpp
// Online/streaming reduction schedule generator (issue #106, epic E3)
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
 * @brief Streaming reduction schedule generator (pattern class P3)
 *
 * A reduction consumes N input tiles and produces ONE running statistic
 * (max/min/sum/mean/var). Per the E3 design
 * (docs/plans/e3_online_reduction_pattern.md) the accumulation is modelled
 * matmul-shaped: a single COMPUTE depends on every streamed input feed and
 * its latency scales with the stream length (the executor's K-scaled
 * latency), so no resident-accumulator chain is needed at the schedule
 * tier and the stats pass has a constant working set of 2 (one streaming
 * tile in flight + the output stat) regardless of stream length.
 *
 * Forms:
 *  - FULL_REDUCE   : one stat over the whole stream (global pooling, losses)
 *  - ROW_STATS     : per-row stats over the reduction dim (softmax max/sum,
 *                    norm mean/var), rows batched
 *  - ROW_NORMALIZE : ROW_STATS + an apply phase (softmax/layernorm
 *                    substrate). The data is seen twice; the realization is
 *                    chosen a priori from the envelope:
 *                      ROW_RESIDENT (row delivered once, consumer_count=2)
 *                        when reduction_tiles + 2 <= per-matrix burst share,
 *                      RESTREAMED (re-read the row from DRAM, constant
 *                        working set 3) otherwise.
 *
 * Every emitted schedule is executable: COMPUTE operations carry their full
 * feed-dependency sets (the #101 discipline).
 */
class OnlineReductionScheduleGenerator : public IScheduleGenerator {
public:
    enum class Form { FULL_REDUCE, ROW_STATS, ROW_NORMALIZE };
    enum class Realization { ROW_RESIDENT, RESTREAMED };
    enum class ReduceOp { MAX, MIN, SUM, MEAN, VAR };

    struct Config {
        Size num_rows = 1;            ///< Row batch (FULL_REDUCE uses 1)
        Size reduction_elems = 4096;  ///< Elements per row along the reduction dim
        Size tile_elems = 256;        ///< Elements per tile
        Form form = Form::FULL_REDUCE;
        ReduceOp op = ReduceOp::SUM;

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

        /**
         * @brief Realization chosen a priori for ROW_NORMALIZE (the #67
         *        constructive-safety discipline); other forms are single-pass
         */
        [[nodiscard]] Realization realization() const {
            return reduction_tiles() + 2 <= max_burst_tiles()
                ? Realization::ROW_RESIDENT
                : Realization::RESTREAMED;
        }

        /**
         * @brief Peak tile residency
         *
         * Stats passes stream one tile at a time against a resident stat
         * (working set 2). ROW_NORMALIZE in ROW_RESIDENT holds the whole row
         * resident across its two phases (reduction_tiles + stat + output);
         * RESTREAMED re-reads the row so only 3 tiles are ever in flight.
         */
        [[nodiscard]] Size required_working_set() const {
            switch (form) {
                case Form::FULL_REDUCE:
                case Form::ROW_STATS:
                    return 2;
                case Form::ROW_NORMALIZE:
                    return realization() == Realization::ROW_RESIDENT
                        ? reduction_tiles() + 2
                        : 3;
            }
            return 2;
        }
    };

    explicit OnlineReductionScheduleGenerator(const Config& config)
        : config_(config) {}

    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.reduction_elems == 0 || config_.tile_elems == 0 ||
            config_.num_rows == 0) {
            result.valid = false;
            result.error_message = "Reduction dimensions must be non-zero";
            return result;
        }
        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "reduction schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }

        const Size rows = config_.form == Form::FULL_REDUCE ? 1 : config_.num_rows;
        for (Size r = 0; r < rows; ++r) {
            switch (config_.form) {
                case Form::FULL_REDUCE:
                case Form::ROW_STATS:
                    emit_stats_row(result, r);
                    break;
                case Form::ROW_NORMALIZE:
                    if (config_.realization() == Realization::ROW_RESIDENT) {
                        emit_normalize_row_resident(result, r);
                    } else {
                        emit_normalize_restreamed(result, r);
                    }
                    break;
            }
        }

        stamp_metadata(result, rows);
        result.valid = true;
        return result;
    }

    [[nodiscard]] std::string name() const override {
        return "OnlineReductionScheduleGenerator";
    }

    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << "Reduction " << op_name(config_.op) << " ["
           << config_.num_rows << "x" << config_.reduction_elems << "] ("
           << form_name(config_.form) << ")";
        return ss.str();
    }

    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    // ------------------------------------------------------------------
    // Stats-only row (FULL_REDUCE / ROW_STATS): stream the row's tiles, one
    // COMPUTE over all their feeds, drain/store the stat.
    // ------------------------------------------------------------------
    void emit_stats_row(ScheduleResult& result, Size row) {
        const Size rt = config_.reduction_tiles();
        std::vector<TileID> deps;
        deps.reserve(rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            result.operations.push_back(ScheduleOperation::feed(in));
            deps.push_back(in.tile_id);
        }
        auto stat = make_stat_tile(row);
        result.operations.push_back(ScheduleOperation::compute(stat, std::move(deps)));
        result.operations.push_back(ScheduleOperation::drain(stat));
        result.operations.push_back(ScheduleOperation::writeback(stat));
        result.operations.push_back(ScheduleOperation::store(stat));
    }

    // ------------------------------------------------------------------
    // ROW_NORMALIZE, ROW_RESIDENT: the row fits the L2 share, so each tile
    // is delivered ONCE with consumer_count=2 (the E2 1:1:k discipline,
    // k=2): fed first to accumulate the stat, fed again to normalize. The
    // stat produced by the stats compute is drained to L2 and broadcast to
    // every apply compute.
    // ------------------------------------------------------------------
    void emit_normalize_row_resident(ScheduleResult& result, Size row) {
        const Size rt = config_.reduction_tiles();

        // Deliver each row tile once, to be consumed twice
        std::vector<TileDescriptor> row_tiles;
        row_tiles.reserve(rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            in.consumer_count = 2;
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            row_tiles.push_back(in);
        }

        // Phase 1: stats over the first consumption of every tile, then the
        // stat round-trips to DRAM (compute -> drain -> writeback -> store)
        std::vector<TileID> stat_deps;
        stat_deps.reserve(rt);
        for (const auto& in : row_tiles) {
            result.operations.push_back(ScheduleOperation::feed(in));
            stat_deps.push_back(in.tile_id);
        }
        emit_stat_compute_and_store(result, row, std::move(stat_deps));

        // Phase 2: reload the stat as a broadcast operand (a distinct tile
        // id, delivered once, consumed by every apply), and apply against
        // the row tiles' second consumption
        auto stat_op = make_stat_operand_tile(row);
        emit_broadcast_tile(result, stat_op, rt);
        for (Size t = 0; t < rt; ++t) {
            result.operations.push_back(ScheduleOperation::feed(row_tiles[t]));
            result.operations.push_back(ScheduleOperation::feed(stat_op));
            emit_apply_output(result, row, t, row_tiles[t].tile_id, stat_op.tile_id);
        }
    }

    // ------------------------------------------------------------------
    // ROW_NORMALIZE, RESTREAMED: the row does not fit, so it is read twice
    // from DRAM. Phase 1 streams for the stat (working set 2); phase 2
    // re-streams and applies with the stat as a resident (broadcast)
    // operand. Constant working set 3.
    // ------------------------------------------------------------------
    void emit_normalize_restreamed(ScheduleResult& result, Size row) {
        const Size rt = config_.reduction_tiles();

        // Phase 1: stats (fresh delivery, consumed once), stat -> DRAM
        std::vector<TileID> stat_deps;
        stat_deps.reserve(rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            result.operations.push_back(ScheduleOperation::feed(in));
            stat_deps.push_back(in.tile_id);
        }
        emit_stat_compute_and_store(result, row, std::move(stat_deps));

        // Phase 2: reload the stat as a broadcast operand, re-stream the row
        // (second DRAM read), and apply
        auto stat_op = make_stat_operand_tile(row);
        emit_broadcast_tile(result, stat_op, rt);
        for (Size t = 0; t < rt; ++t) {
            auto in = make_input_tile(row, t);
            result.operations.push_back(ScheduleOperation::load(in));
            result.operations.push_back(ScheduleOperation::move(in));
            result.operations.push_back(ScheduleOperation::feed(in));
            result.operations.push_back(ScheduleOperation::feed(stat_op));
            emit_apply_output(result, row, t, in.tile_id, stat_op.tile_id);
        }
    }

    // Stat compute over the given feed deps, then round-trip to DRAM so a
    // distinct reload tile can broadcast it to the apply phase.
    void emit_stat_compute_and_store(ScheduleResult& result, Size row,
                                     std::vector<TileID> deps) {
        auto stat = make_stat_tile(row);
        result.operations.push_back(ScheduleOperation::compute(stat, std::move(deps)));
        result.operations.push_back(ScheduleOperation::drain(stat));
        result.operations.push_back(ScheduleOperation::writeback(stat));
        result.operations.push_back(ScheduleOperation::store(stat));
    }

    void emit_apply_output(ScheduleResult& result, Size row, Size t,
                           const TileID& in_id, const TileID& stat_id) {
        auto out = make_output_tile(row, t);
        result.operations.push_back(ScheduleOperation::compute(out, {in_id, stat_id}));
        result.operations.push_back(ScheduleOperation::drain(out));
        result.operations.push_back(ScheduleOperation::writeback(out));
        result.operations.push_back(ScheduleOperation::store(out));
    }

    // ------------------------------------------------------------------
    // Tile descriptors
    // ------------------------------------------------------------------
    TileDescriptor make_input_tile(Size row, Size t) const {
        // Input rides matrix A; index encodes (row, reduction-tile)
        return make_tile(isa::MatrixID::A, row, t, config_.in_base);
    }
    TileDescriptor make_stat_tile(Size row) const {
        // Stat rides matrix B (an intermediate); one per row, tk=0
        return make_tile(isa::MatrixID::B, row, 0, config_.stat_base);
    }
    TileDescriptor make_stat_operand_tile(Size row) const {
        // Distinct tile id (tk=1) reading the SAME DRAM slot as the stored
        // stat, so the apply phase can broadcast it without reusing the
        // stat's compute-target tile id
        auto tile = make_tile(isa::MatrixID::B, row, 0, config_.stat_base);
        tile.tile_id.tk = 1;
        return tile;
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
        tile.height = config_.tile_elems;
        tile.width = 1;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.tile_elems * config_.element_size;
        // Row-major over (row, reduction-tile); stat tiles are one per row
        const Size linear = matrix == isa::MatrixID::B
            ? row
            : row * config_.reduction_tiles() + t;
        tile.dram_address = base + linear * tile.size_bytes;
        tile.matrix_base_address = base;
        return tile;
    }

    void stamp_metadata(ScheduleResult& result, Size rows) const {
        const Size rt = config_.reduction_tiles();
        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.num_rows;
        result.metadata.N = config_.reduction_elems;
        result.metadata.K = 1;
        result.metadata.Ti = config_.tile_elems;
        result.metadata.Tj = 1;
        result.metadata.Tk = 1;
        result.metadata.a_tiles = rows * rt;
        result.metadata.b_tiles = rows;  // one stat per row
        result.metadata.c_tiles =
            config_.form == Form::ROW_NORMALIZE ? rows * rt : 0;
        result.metadata.strategy = strategy_name();
        result.metadata.l3_buffer_count = config_.l3_buffer_count;
        result.metadata.l2_bank_count = config_.l2_bank_count;
    }

    static const char* form_name(Form form) {
        switch (form) {
            case Form::FULL_REDUCE:   return "full_reduce";
            case Form::ROW_STATS:     return "row_stats";
            case Form::ROW_NORMALIZE: return "row_normalize";
        }
        return "unknown";
    }
    static const char* op_name(ReduceOp op) {
        switch (op) {
            case ReduceOp::MAX:  return "MAX";
            case ReduceOp::MIN:  return "MIN";
            case ReduceOp::SUM:  return "SUM";
            case ReduceOp::MEAN: return "MEAN";
            case ReduceOp::VAR:  return "VAR";
        }
        return "?";
    }
    std::string strategy_name() const {
        std::string s = form_name(config_.form);
        if (config_.form == Form::ROW_NORMALIZE) {
            s += config_.realization() == Realization::ROW_RESIDENT
                ? "_resident" : "_restreamed";
        }
        return s;
    }
    std::string generate_name() const {
        std::ostringstream ss;
        ss << "reduction_" << op_name(config_.op) << "_" << config_.num_rows
           << "x" << config_.reduction_elems << "_" << strategy_name();
        return ss.str();
    }
};

} // namespace sw::kpu::timing::schedule
