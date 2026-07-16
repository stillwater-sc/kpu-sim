// ============================================================================
// examples/schedule/pooling_simulator.cpp
// Standalone pooling simulator with a horizontal tile-state debug log
// (companion to conv2d/batchnorm simulators; see docs/tile-state-tracking.md).
//
// Pooling reduces each channel over a spatial window (max/avg) - see
// docs/plans/e7_pooling_pattern.md. This driver seeds the per-channel window
// rows, runs the generated pooling schedule one executor cycle at a time, and
// after each cycle records what tiles occupy L3 / L2 / L1 / array via the
// TileTracker. Watch a channel's window block X[c,t] stream DRAM->L3->L2->L1/
// array, a per-row reduce in the array, and the pooled output Y[c,t] drain back
// out. Ends on a pool2d host-oracle check.
//
// Usage: pooling_simulator [--c C] [--h H] [--w W] [--kh Kh] [--kw Kw]
//                          [--stride S] [--pad P] [--tile T] [--avg]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {
long arg(int argc, char** argv, const char* flag, long def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atol(argv[i + 1]);
    return def;
}
bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}
} // namespace

int main(int argc, char** argv) {
    try {
        Pool2DGeometry g;
        g.N = 1;
        g.C = static_cast<Size>(arg(argc, argv, "--c", 2));
        g.H = static_cast<Size>(arg(argc, argv, "--h", 8));
        g.W = static_cast<Size>(arg(argc, argv, "--w", 8));
        g.Kh = static_cast<Size>(arg(argc, argv, "--kh", 2));
        g.Kw = static_cast<Size>(arg(argc, argv, "--kw", 2));
        g.stride_h = g.stride_w = static_cast<Size>(arg(argc, argv, "--stride", 2));
        g.pad_h = g.pad_w = static_cast<Size>(arg(argc, argv, "--pad", 0));
        const Size T = static_cast<Size>(arg(argc, argv, "--tile", 16));
        const PoolType type = has_flag(argc, argv, "--avg") ? PoolType::AVG : PoolType::MAX;
        if (T == 0) { std::cerr << "error: --tile must be > 0\n"; return 2; }
        if (!g.valid()) { std::cerr << "error: invalid geometry\n"; return 2; }
        if (g.out_spatial() % T) {
            std::cerr << "error: --tile " << T << " must divide Hout*Wout="
                      << g.out_spatial() << "\n";
            return 2;
        }

        std::vector<float> input(g.elems());
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = -1.5f + 0.5f * static_cast<float>(i % 7);
        const auto ref = pool2d_reference(input, g, type);
        const Size K = g.window(), M = g.out_spatial();

        std::vector<std::vector<float>> win(g.C);
        for (Size c = 0; c < g.C; ++c) win[c] = pool_window_channel(input, g, 0, c, type).rows;

        PoolingScheduleGenerator::Config cfg;
        cfg.geom = g; cfg.pool_type = type; cfg.Ti = T;
        cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED;
        cfg.input_base = 0x100000; cfg.output_base = 0x400000;
        auto schedule = PoolingScheduleGenerator(cfg).generate();
        if (!schedule.valid) { std::cerr << "schedule refused: " << schedule.error_message << "\n"; return 1; }

        std::cout << "Pooling simulator  -  " << (type == PoolType::MAX ? "max" : "avg")
                  << "-pool  (C" << g.C << " " << g.H << "x" << g.W << " -> "
                  << g.H_out() << "x" << g.W_out() << ", k" << g.Kh << "x" << g.Kw
                  << " s" << g.stride_h << " p" << g.pad_h << ", tile " << T << ")\n"
                  << "  each output = reduce over its " << K << "-tap window\n\n";

        ConcurrentTimingExecutor::Config ecfg; ecfg.max_cycles = 2'000'000;
        ConcurrentTimingExecutor exec(ecfg);
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto id = op.tile.tile_id;
            const Size c = id.ti, ti = id.tj;
            std::vector<float> blk(static_cast<std::size_t>(T) * K);
            for (Size r = 0; r < T; ++r)
                for (Size k = 0; k < K; ++k)
                    blk[r * K + k] = win[c][static_cast<std::size_t>(ti * T + r) * K + k];
            exec.set_tile_payload(id, TilePayload{T, K, std::move(blk)});
        }
        for (const auto& op : schedule.operations) {
            switch (op.type) {
                case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
                case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
                case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
                case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
                case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
                case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
                case ScheduleOpType::COMPUTE: {
                    ConcurrentTimingExecutor::FunctionalComputeSpec spec;
                    spec.input_tiles = op.dependency_tiles;
                    spec.operation = [type, K](const std::vector<TilePayload>& in) {
                        const auto& x = in.at(0);
                        const Size rows = x.rows;
                        TilePayload out{rows, 1, std::vector<float>(rows)};
                        for (Size r = 0; r < rows; ++r) {
                            const float* row = &x.values[static_cast<std::size_t>(r) * K];
                            if (type == PoolType::MAX) {
                                float m = -std::numeric_limits<float>::infinity();
                                for (Size k = 0; k < K; ++k) m = std::max(m, row[k]);
                                out.values[r] = m;
                            } else {
                                float s = 0.0f;
                                for (Size k = 0; k < K; ++k) s += row[k];
                                out.values[r] = s / static_cast<float>(K);
                            }
                        }
                        return out;
                    };
                    exec.schedule_functional_compute(op.tile, spec);
                    break;
                }
            }
        }

        TileTracker::Config tcfg;
        tcfg.label = [](const TileID& id) {
            const std::string s = "[" + std::to_string(id.ti) + "," + std::to_string(id.tj) + "]";
            return (id.matrix == MatrixID::A ? "X" : "Y") + s;
        };
        TileTracker tracker(tcfg);
        tracker.observe(exec);
        while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
            exec.step();
            tracker.observe(exec);
        }
        std::cout << tracker.log() << "\n";
        if (!exec.is_complete()) { std::cerr << "schedule did not complete\n"; return 1; }

        std::cout << "Result: y = " << (type == PoolType::MAX ? "max" : "avg") << "_pool(x)\n";
        double max_err = 0.0;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto id = op.tile.tile_id;
            const Size c = id.ti, ti = id.tj;
            const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
            for (Size r = 0; r < T; ++r)
                max_err = std::max(max_err, static_cast<double>(
                    std::abs(p.values[r] - ref[static_cast<std::size_t>(c) * M + ti * T + r])));
        }
        const bool ok = max_err < 1e-4;
        std::cout << "  max abs error vs host oracle: " << max_err
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n\n"
                  << (ok ? "POOLING OK" : "POOLING MISMATCH") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
