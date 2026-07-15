// ============================================================================
// examples/schedule/conv2d_simulator.cpp
// Standalone conv2d simulator with a horizontal tile-state debug log
// (companion to matmul_simulator; see docs/tile-state-tracking.md).
//
// Conv2D is lowered to a single GEMM C = A_col @ B_w (im2col; see
// docs/plans/e6_conv2d_pattern.md). This driver seeds the im2col A_col patches
// and the reshaped B_w weights (the E6-T2 helpers), runs the generated conv2d
// schedule one executor cycle at a time, and after each cycle records any change
// in what tiles occupy L3 / L2 / L1 / array via the TileTracker. Watch the
// im2col patch tiles A[patch,k] and weight tiles B[k,cout] stream
// DRAM->L3->L2->L1/array, a C[patch,cout] accumulate over the K-slices in the
// array, and the C result drain back out. Ends on a direct-conv host oracle
// check (with optional bias + ReLU).
//
// Usage: conv2d_simulator [--cin C] [--h H] [--w W] [--cout O]
//                         [--kh Kh] [--kw Kw] [--stride S] [--pad P]
//                         [--tile T] [--bias] [--relu]
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>

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
bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

// Extract the [T x T] block (br, bc) from a row-major [rows x cols] matrix.
std::vector<float> block(const std::vector<float>& mat, Size cols,
                         Size br, Size bc, Size T) {
    std::vector<float> b(static_cast<std::size_t>(T) * T);
    for (Size r = 0; r < T; ++r)
        for (Size c = 0; c < T; ++c)
            b[r * T + c] = mat[(br * T + r) * cols + (bc * T + c)];
    return b;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Conv2DGeometry g;
        g.N = 1;
        g.C_in  = static_cast<Size>(arg(argc, argv, "--cin", 4));
        g.H_in  = static_cast<Size>(arg(argc, argv, "--h", 4));
        g.W_in  = static_cast<Size>(arg(argc, argv, "--w", 4));
        g.C_out = static_cast<Size>(arg(argc, argv, "--cout", 4));
        g.Kh    = static_cast<Size>(arg(argc, argv, "--kh", 3));
        g.Kw    = static_cast<Size>(arg(argc, argv, "--kw", 3));
        g.stride_h = g.stride_w = static_cast<Size>(arg(argc, argv, "--stride", 1));
        g.pad_h = g.pad_w = static_cast<Size>(arg(argc, argv, "--pad", 1));
        const Size T = static_cast<Size>(arg(argc, argv, "--tile", 4));
        const bool use_bias = has_flag(argc, argv, "--bias");
        const bool use_relu = has_flag(argc, argv, "--relu");

        if (!g.valid()) {
            std::cerr << "error: invalid conv geometry (check sizes/padding)\n";
            return 2;
        }
        if (g.M() % T || g.C_out % T || g.K() % T) {
            std::cerr << "error: --tile " << T << " must divide M=" << g.M()
                      << ", C_out=" << g.C_out << ", and K=" << g.K() << "\n";
            return 2;
        }

        // Host operands (bounded, deterministic) and the im2col lowering.
        std::vector<float> input(g.input_elems()), filter(g.filter_elems());
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = 0.5f + 0.5f * static_cast<float>(i % 7);
        for (std::size_t i = 0; i < filter.size(); ++i)
            filter[i] = -1.0f + 0.5f * static_cast<float>(i % 5);
        std::vector<float> bias;
        if (use_bias) {
            bias.resize(g.C_out);
            for (Size i = 0; i < g.C_out; ++i)
                bias[i] = -2.0f + 0.5f * static_cast<float>(i % 6);
        }
        const auto a_col = im2col_nchw(input, g);
        const auto b_w = filter_to_bw_nchw(filter, g);
        const auto ref = conv2d_reference(input, filter, bias, g, use_relu);

        Conv2DScheduleGenerator::Config cfg;
        cfg.N = g.N; cfg.H_in = g.H_in; cfg.W_in = g.W_in; cfg.C_in = g.C_in;
        cfg.C_out = g.C_out; cfg.Kh = g.Kh; cfg.Kw = g.Kw;
        cfg.stride_h = g.stride_h; cfg.stride_w = g.stride_w;
        cfg.padding_h = g.pad_h; cfg.padding_w = g.pad_w;
        cfg.Ti = T; cfg.Tj = T; cfg.Tk = T;
        cfg.input_base = 0x100000; cfg.filter_base = 0x400000; cfg.output_base = 0x700000;
        auto schedule = Conv2DScheduleGenerator(cfg).generate();
        if (!schedule.valid) {
            std::cerr << "schedule refused: " << schedule.error_message << "\n";
            return 1;
        }

        std::cout << "Conv2D simulator  -  im2col+GEMM  (N" << g.N << " C" << g.C_in
                  << " " << g.H_in << "x" << g.W_in << " -> C" << g.C_out << " "
                  << g.H_out() << "x" << g.W_out() << ", k" << g.Kh << "x" << g.Kw
                  << " s" << g.stride_h << " p" << g.pad_h << ", tile " << T << ")\n"
                  << "  lowered GEMM: C[" << g.M() << "x" << g.C_out << "] = A_col["
                  << g.M() << "x" << g.K() << "] @ B_w[" << g.K() << "x" << g.C_out
                  << "]" << (use_bias ? " + bias" : "") << (use_relu ? " , ReLU" : "")
                  << "\n\n";

        ConcurrentTimingExecutor::Config ecfg;
        ecfg.max_cycles = 2'000'000;
        ConcurrentTimingExecutor exec(ecfg);

        // Seed A_col patch tiles and B_w weight tiles from the host operands.
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::LOAD) continue;
            const auto id = op.tile.tile_id;
            if (id.matrix == MatrixID::A)
                exec.set_tile_payload(id, TilePayload{T, T, block(a_col, g.K(), id.ti, id.tk, T)});
            else
                exec.set_tile_payload(id, TilePayload{T, T, block(b_w, g.C_out, id.tk, id.tj, T)});
        }

        // Enqueue; a COMPUTE becomes a value-producing MatMulComputeSpec carrying
        // the tj-slice of the per-output-channel bias and the activation.
        for (const auto& op : schedule.operations) {
            switch (op.type) {
                case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
                case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
                case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
                case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
                case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
                case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
                case ScheduleOpType::COMPUTE: {
                    ConcurrentTimingExecutor::MatMulComputeSpec spec;
                    for (const auto& dep : op.dependency_tiles) {
                        if (dep.matrix == MatrixID::A) spec.a_tiles.push_back(dep);
                        else                            spec.b_tiles.push_back(dep);
                    }
                    if (use_bias) {
                        const Size tj = op.tile.tile_id.tj;
                        spec.bias.assign(bias.begin() + static_cast<std::ptrdiff_t>(tj * T),
                                         bias.begin() + static_cast<std::ptrdiff_t>(tj * T + T));
                    }
                    if (use_relu)
                        spec.activation = ConcurrentTimingExecutor::FunctionalActivation::RELU;
                    exec.schedule_matmul_compute(op.tile, spec);
                    break;
                }
            }
        }

        // Label tiles by their 2D role: A = im2col patches [patch-row, k-slice],
        // B = weights [k-slice, out-channel], C = output [patch-row, out-channel].
        TileTracker::Config tcfg;
        tcfg.label = [](const TileID& id) {
            auto ij = [](Size a, Size b) {
                return "[" + std::to_string(a) + "," + std::to_string(b) + "]";
            };
            switch (id.matrix) {
                case MatrixID::A: return "A" + ij(id.ti, id.tk);
                case MatrixID::B: return "B" + ij(id.tk, id.tj);
                default:          return "C" + ij(id.ti, id.tj);
            }
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

        // Correctness: each stored C tile matches the direct-conv host oracle.
        std::cout << "Result: C = conv2d(input, filter)"
                  << (use_bias ? " + bias" : "") << (use_relu ? " , ReLU" : "") << "\n";
        const Size Hout = g.H_out(), Wout = g.W_out();
        double max_err = 0.0;
        for (const auto& op : schedule.operations) {
            if (op.type != ScheduleOpType::STORE) continue;
            const auto id = op.tile.tile_id;
            const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
            for (Size r = 0; r < T; ++r)
                for (Size c = 0; c < T; ++c) {
                    const Size m = id.ti * T + r, co = id.tj * T + c;
                    const Size n = m / (Hout * Wout), rem = m % (Hout * Wout);
                    const Size ho = rem / Wout, wo = rem % Wout;
                    const double want =
                        ref[((static_cast<std::size_t>(n) * g.C_out + co) * Hout + ho) *
                                Wout + wo];
                    max_err = std::max(max_err, std::abs(p.values[r * T + c] - want));
                }
        }
        const bool ok = max_err < 1e-3;
        std::cout << "  max abs error vs host oracle: " << max_err
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n\n"
                  << (ok ? "CONV2D OK" : "CONV2D MISMATCH") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
