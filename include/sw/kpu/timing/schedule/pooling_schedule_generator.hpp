// ============================================================================
// include/sw/kpu/timing/schedule/pooling_schedule_generator.hpp
// Pooling schedule generator for CSP-style timing model (E7-T3, #193)
//
// Pooling reduces each channel over a spatial window (see
// docs/plans/e7_pooling_pattern.md). The generator emits an EXECUTABLE schedule
// from the start (no #139): per output tile it streams the window rows and emits
// a reduce COMPUTE before the drain.
//
//   WINDOWED  (max/avg): per channel, per Ti-block of output positions, the
//             input is the [Ti, Kh*Kw] window block; the COMPUTE reduces each
//             row along the window axis (MAX / MEAN) to one output per position.
//   GLOBAL_AVG:          per channel, the whole H*W plane is streamed and a
//             single COMPUTE reduces it to one value (mean).
//
// The pool op (MAX/MEAN) rides in Config/metadata; the reduce itself is the
// existing VE_REDUCE bound at execution (no new executor kernel). Movement is
// pool-op-agnostic.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/schedule/schedule_generator_interface.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>  // Pool2DGeometry, PoolType

#include <sstream>
#include <vector>

namespace sw::kpu::timing::schedule {

class PoolingScheduleGenerator : public IScheduleGenerator {
public:
    enum class Mode { WINDOWED, GLOBAL_AVG };

    struct Config {
        Pool2DGeometry geom;             ///< N/C/H/W/Kh/Kw/stride/pad
        PoolType pool_type = PoolType::MAX;
        Mode mode = Mode::WINDOWED;
        Size Ti = 16;                    ///< output positions per tile
        Size element_size = 4;

        // Resource envelope (issue #90).
        Size l3_buffer_count = 32;
        Size l2_bank_count = 64;

        Address input_base = 0;
        Address output_base = 0;

        /// Output positions per channel (N*Hout*Wout for windowed; N for gap).
        [[nodiscard]] Size out_positions() const {
            return mode == Mode::GLOBAL_AVG ? geom.N : geom.N * geom.out_spatial();
        }
        [[nodiscard]] Size out_tiles() const {
            return (out_positions() + Ti - 1) / Ti;
        }
        /// Plane tiles per channel for GLOBAL_AVG (H*W streamed in Ti-chunks).
        [[nodiscard]] Size plane_tiles() const {
            const Size plane = geom.H * geom.W;
            return (plane + Ti - 1) / Ti;
        }
        [[nodiscard]] Size max_burst_tiles() const {
            return per_matrix_burst_share(l3_buffer_count, l2_bank_count);
        }
        /// Windowed streams one window tile + one output; gap streams one plane
        /// tile + the running output.
        [[nodiscard]] Size required_working_set() const { return 3; }
    };

    explicit PoolingScheduleGenerator(const Config& config) : config_(config) {}

    ScheduleResult generate() override {
        ScheduleResult result;

        if (config_.required_working_set() > config_.max_burst_tiles()) {
            result.valid = false;
            result.error_message =
                "pooling schedule requires a working set of " +
                std::to_string(config_.required_working_set()) +
                " tiles but the resource envelope share is " +
                std::to_string(config_.max_burst_tiles()) +
                "; enlarge l3_buffer_count/l2_bank_count";
            return result;
        }
        if (!config_.geom.valid()) {
            result.valid = false;
            result.error_message = "pooling geometry invalid (check sizes/padding)";
            return result;
        }

        if (config_.mode == Mode::GLOBAL_AVG) generate_global_avg(result);
        else                                  generate_windowed(result);

        result.metadata.name = generate_name();
        result.metadata.generator = name();
        result.metadata.M = config_.out_positions();
        result.metadata.N = config_.geom.C;
        result.metadata.K = (config_.mode == Mode::GLOBAL_AVG)
                                ? config_.geom.H * config_.geom.W : config_.geom.window();
        result.metadata.Ti = config_.Ti;
        result.metadata.strategy = strategy_name();
        result.metadata.l3_buffer_count = config_.l3_buffer_count;
        result.metadata.l2_bank_count = config_.l2_bank_count;
        result.valid = true;
        return result;
    }

    [[nodiscard]] std::string name() const override { return "PoolingScheduleGenerator"; }
    [[nodiscard]] std::string description() const override {
        std::ostringstream ss;
        ss << strategy_name() << " [" << config_.geom.N << "x" << config_.geom.C
           << "x" << config_.geom.H << "x" << config_.geom.W << "]";
        return ss.str();
    }
    [[nodiscard]] const Config& config() const { return config_; }

private:
    Config config_;

    // Windowed pooling: per channel, per output-position tile, reduce the
    // [Ti, Kh*Kw] window block along the window axis.
    void generate_windowed(ScheduleResult& result) {
        const Size out_tiles = config_.out_tiles();
        for (Size c = 0; c < config_.geom.C; ++c) {
            for (Size ti = 0; ti < out_tiles; ++ti) {
                auto in = make_tile(isa::MatrixID::A, c, ti, config_.geom.window());
                result.operations.push_back(ScheduleOperation::load(in));
                result.operations.push_back(ScheduleOperation::move(in));
                result.operations.push_back(ScheduleOperation::feed(in));

                auto out = make_tile(isa::MatrixID::C, c, ti, 1);
                result.operations.push_back(ScheduleOperation::compute(out, {in.tile_id}));
                result.operations.push_back(ScheduleOperation::drain(out));
                result.operations.push_back(ScheduleOperation::writeback(out));
                result.operations.push_back(ScheduleOperation::store(out));
            }
        }
    }

    // Global average pool: per channel, stream the whole H*W plane and reduce
    // it to one value. The COMPUTE depends on every plane tile.
    void generate_global_avg(ScheduleResult& result) {
        const Size plane_tiles = config_.plane_tiles();
        for (Size n = 0; n < config_.geom.N; ++n) {
            for (Size c = 0; c < config_.geom.C; ++c) {
                std::vector<TileID> deps;
                deps.reserve(plane_tiles);
                for (Size pt = 0; pt < plane_tiles; ++pt) {
                    auto in = make_tile(isa::MatrixID::A, n * config_.geom.C + c, pt,
                                        config_.Ti);
                    result.operations.push_back(ScheduleOperation::load(in));
                    result.operations.push_back(ScheduleOperation::move(in));
                    result.operations.push_back(ScheduleOperation::feed(in));
                    deps.push_back(in.tile_id);
                }
                auto out = make_tile(isa::MatrixID::C, n * config_.geom.C + c, 0, 1);
                result.operations.push_back(ScheduleOperation::compute(out, std::move(deps)));
                result.operations.push_back(ScheduleOperation::drain(out));
                result.operations.push_back(ScheduleOperation::writeback(out));
                result.operations.push_back(ScheduleOperation::store(out));
            }
        }
    }

    TileDescriptor make_tile(isa::MatrixID matrix, Size ti, Size tj, Size width) const {
        TileDescriptor tile;
        tile.tile_id.matrix = matrix;
        tile.tile_id.ti = ti;
        tile.tile_id.tj = tj;
        tile.tile_id.tk = 0;
        tile.height = config_.Ti;
        tile.width = width;
        tile.element_size = config_.element_size;
        tile.size_bytes = config_.Ti * width * config_.element_size;
        tile.dram_address = (matrix == isa::MatrixID::A ? config_.input_base
                                                        : config_.output_base) +
                            (ti * 0x10000 + tj * 0x100) * config_.element_size;
        return tile;
    }

    std::string generate_name() const {
        std::ostringstream ss;
        ss << "pooling_" << strategy_name() << "_" << config_.geom.N << "x"
           << config_.geom.C << "x" << config_.geom.H << "x" << config_.geom.W;
        return ss.str();
    }

    const char* strategy_name() const {
        if (config_.mode == Mode::GLOBAL_AVG) return "global_avg_pool";
        return config_.pool_type == PoolType::MAX ? "max_pool" : "avg_pool";
    }
};

} // namespace sw::kpu::timing::schedule
