// ============================================================================
// examples/schedule/batchnorm_simulator.cpp
// Standalone BatchNorm-inference simulator with a horizontal tile-state debug
// log (companion to conv2d_simulator; see docs/tile-state-tracking.md).
//
// BatchNorm inference folds to a per-channel affine y = x*scale[c] + shift[c]
// (see docs/plans/e9_batchnorm_pattern.md). This driver seeds the streamed
// input tiles and the folded per-channel scale/shift params (the E9-T2 fold),
// runs the generated BN schedule one executor cycle at a time, and after each
// cycle records any change in what tiles occupy L3 / L2 / L1 / array via the
// TileTracker. Watch the per-channel scale/shift broadcast params arrive and
// stay RESIDENT while the input tiles stream DRAM->L3->L2->L1/array, a
// per-channel affine compute in the array, and the output drain back out. Ends
// on a direct-conv-style host oracle check.
//
// Usage: batchnorm_simulator [--n N] [--c C] [--h H] [--w W] [--tile T]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/batchnorm_schedule_generator.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
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

} // namespace

int main(int argc, char** argv) {
    try {
        BatchNormGeometry g;
        g.N = static_cast<Size>(arg(argc, argv, "--n", 1));
        g.C = static_cast<Size>(arg(argc, argv, "--c", 2));
        g.H = static_cast<Size>(arg(argc, argv, "--h", 4));
        g.W = static_cast<Size>(arg(argc, argv, "--w", 4));
        const Size T = static_cast<Size>(arg(argc, argv, "--tile", 16));
        if (T == 0) { std::cerr << "error: --tile must be > 0\n"; return 2; }
        if (!g.valid()) { std::cerr << "error: invalid geometry\n"; return 2; }
        if (g.spatial() % T) {
            std::cerr << "error: --tile " << T << " must divide H*W=" << g.spatial() << "\n";
            return 2;
        }

        // Host operands (bounded) and the folded scale/shift.
        std::vector<float> input(g.elems());
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = -1.0f + 0.5f * static_cast<float>(i % 7);
        std::vector<float> gamma(g.C), beta(g.C), mean(g.C), var(g.C);
        for (Size c = 0; c < g.C; ++c) {
            gamma[c] = 0.5f + 0.5f * static_cast<float>(c % 4);
            beta[c]  = -1.0f + 0.75f * static_cast<float>(c % 3);
            mean[c]  = 0.25f + 0.5f * static_cast<float>(c % 5);
            var[c]   = 0.5f + 0.25f * static_cast<float>(c % 4);
        }
        const float eps = 1e-3f;
        const auto affine = bn_fold(gamma, beta, mean, var, eps);
        const auto ref = batchnorm_reference(input, gamma, beta, mean, var, eps, g);

        BatchNormScheduleGenerator::Config cfg;
        cfg.N = g.N; cfg.C = g.C; cfg.H = g.H; cfg.W = g.W;
        cfg.Ti = T; cfg.Tj = T; cfg.training = false;
        const Size credits = 4 * (2 * g.C + 1) + 4;
        cfg.l3_buffer_count = cfg.l2_bank_count = credits < 32 ? 32 : credits;
        auto schedule = BatchNormScheduleGenerator(cfg).generate();
        if (!schedule.valid) {
            std::cerr << "schedule refused: " << schedule.error_message << "\n";
            return 1;
        }

        std::cout << "BatchNorm simulator  -  inference (folded affine)  (N" << g.N
                  << " C" << g.C << " " << g.H << "x" << g.W << ", tile " << T
                  << ")\n  y = x*scale[c] + shift[c]  (scale/shift resident per channel)\n\n";

        ConcurrentTimingExecutor::Config ecfg;
        ecfg.l3_buffer_count = ecfg.l2_bank_count = cfg.l3_buffer_count;
        ecfg.max_cycles = 2'000'000;
        ConcurrentTimingExecutor exec(ecfg);

        const Size spatial = g.spatial();
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto id = op.tile.tile_id;
            if (id.matrix == MatrixID::A) {
                const Size n = id.ti / g.C, c = id.ti % g.C, si = id.tj;
                const std::size_t base =
                    (static_cast<std::size_t>(n) * g.C + c) * spatial + si * T;
                exec.set_tile_payload(id, TilePayload{T, 1,
                    std::vector<float>(input.begin() + static_cast<std::ptrdiff_t>(base),
                                       input.begin() + static_cast<std::ptrdiff_t>(base + T))});
            } else {  // param: ti = channel, tj = SCALE(4)/SHIFT(5) ordinal
                const Size c = id.ti;
                const bool is_scale = (id.tj == 4);
                exec.set_tile_payload(id, TilePayload{1, 1,
                    {is_scale ? affine.scale[c] : affine.shift[c]}});
            }
        }

        // Enqueue; a COMPUTE becomes a value-producing FunctionalComputeSpec
        // applying the per-channel affine to the streamed input tile.
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
                    spec.input_tiles = op.dependency_tiles;  // {input, scale, shift}
                    spec.operation = [](const std::vector<TilePayload>& in) {
                        const auto& x = in.at(0);
                        const float scale = in.at(1).values.at(0);
                        const float shift = in.at(2).values.at(0);
                        TilePayload out{x.rows, x.cols, std::vector<float>(x.values.size())};
                        for (std::size_t i = 0; i < x.values.size(); ++i)
                            out.values[i] = x.values[i] * scale + shift;
                        return out;
                    };
                    exec.schedule_functional_compute(op.tile, spec);
                    break;
                }
            }
        }

        // Label input as X[c,si], params as scale[c]/shift[c], output as Y[c,si].
        TileTracker::Config tcfg;
        tcfg.label = [C = g.C](const TileID& id) -> std::string {
            auto ij = [](Size a, Size b) {
                return "[" + std::to_string(a) + "," + std::to_string(b) + "]";
            };
            if (id.matrix == MatrixID::A) return "X" + ij(id.ti % C, id.tj);
            if (id.matrix == MatrixID::C) return "Y" + ij(id.ti % C, id.tj);
            return (id.tj == 4 ? "scale[" : "shift[") + std::to_string(id.ti) + "]";
        };
        TileTracker tracker(tcfg);
        tracker.observe(exec);
        while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
            exec.step();
            tracker.observe(exec);
        }
        std::cout << tracker.log() << "\n";

        if (!exec.is_complete()) {
            std::cerr << "schedule did not complete (deadlock or max_cycles)\n";
            return 1;
        }

        std::cout << "Result: y = batchnorm_inference(x)\n";
        double max_err = 0.0;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto id = op.tile.tile_id;
            const Size n = id.ti / g.C, c = id.ti % g.C, si = id.tj;
            const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
            const std::size_t base =
                (static_cast<std::size_t>(n) * g.C + c) * spatial + si * T;
            for (Size i = 0; i < T; ++i)
                max_err = std::max(max_err,
                    static_cast<double>(std::abs(p.values[i] - ref[base + i])));
        }
        const bool ok = max_err < 1e-3;
        std::cout << "  max abs error vs host oracle: " << max_err
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n\n"
                  << (ok ? "BATCHNORM OK" : "BATCHNORM MISMATCH") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
