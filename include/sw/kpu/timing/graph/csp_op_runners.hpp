// ============================================================================
// include/sw/kpu/timing/graph/csp_op_runners.hpp
// Value-producing single-op runners on the CSP executor (M2-T2, #201)
//
// The workhorses the graph->CSP bridge (graph_csp_executor.hpp) dispatches to.
// Each runs one operator through the credit-based ConcurrentTimingExecutor and
// returns the output tensor (real fp32 values), reusing the schedule-generator
// value path proven by the E6/E9/E10 functional tests.
//
//   run_conv2d_fused: conv (im2col -> GEMM) with an OPTIONAL folded BatchNorm and
//     an OPTIONAL bias + activation applied in-compute (the fused epilogue) - so
//     conv+BN+ReLU is one GEMM, no DRAM round-trip for the intermediate.
//   run_elementwise: a binary VE op (ADD for the residual join, MAX for ReLU).
//
// Tensors are row-major NCHW fp32. Tile size T divides M / Cout / K; the M2 demo
// picks aligned shapes (documented in docs/plans/m2_resnet_dfg.md).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/functional_elementwise_executor.hpp>
#include <sw/kpu/timing/schedule/pooling_window.hpp>
#include <sw/kpu/timing/schedule/pooling_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::graph {

using schedule::Conv2DGeometry;
using schedule::BatchNormAffine;

/// Accumulated CSP-execution stats across the ops of a graph run.
struct RunStats {
    Cycle total_cycles = 0;
    Cycle dma_stalls = 0;
    Cycle bm_stalls = 0;
    Cycle str_stalls = 0;
    std::size_t ops = 0;
};

namespace detail {

// [T x T] block (br, bc) of a row-major [rows x cols] matrix.
inline std::vector<float> block(const std::vector<float>& mat, Size cols,
                                Size br, Size bc, Size T) {
    std::vector<float> b(static_cast<std::size_t>(T) * T);
    for (Size r = 0; r < T; ++r)
        for (Size c = 0; c < T; ++c)
            b[r * T + c] = mat[(br * T + r) * cols + (bc * T + c)];
    return b;
}

inline void accumulate(RunStats& s, const ConcurrentTimingExecutor& exec) {
    const auto st = exec.get_statistics();
    s.total_cycles += exec.current_cycle();
    s.dma_stalls += st.dma_credit_stalls;
    s.bm_stalls += st.bm_tag_stalls + st.bm_credit_stalls;
    s.str_stalls += st.str_tag_stalls + st.str_credit_stalls;
    ++s.ops;
}

} // namespace detail

/**
 * @brief Run conv2d (im2col -> GEMM) with a fused epilogue, on the CSP executor.
 *
 * y = act( conv(x, filter) * bn.scale + bn.shift )     (if bn != nullptr)
 * y = act( conv(x, filter) + bias )                    (otherwise)
 *
 * The BatchNorm fold (E9) scales the weight columns and moves the shift into the
 * per-output-channel bias, so conv+BN is a single GEMM; bias + activation apply
 * in-compute (E10). Input/output are NCHW.
 *
 * @param bn  optional folded BatchNorm affine (size C_out each), or nullptr
 * @param bias optional per-output-channel conv bias (size C_out), or empty
 * @param relu apply ReLU as the in-compute activation
 * @param T   tile size (must divide M = N*Hout*Wout, C_out, and K = Cin*Kh*Kw)
 */
[[nodiscard]] inline std::vector<float>
run_conv2d_fused(const std::vector<float>& input, const std::vector<float>& filter,
                 const Conv2DGeometry& geom, const BatchNormAffine* bn,
                 const std::vector<float>& bias, bool relu, Size T, RunStats& stats) {
    using schedule::MatMulScheduleGenerator;
    using MatrixID = sw::kpu::isa::MatrixID;

    const Size M = geom.M(), N = geom.C_out, K = geom.K();
    if (T == 0 || M % T || N % T || K % T)
        throw std::invalid_argument("run_conv2d_fused: T must be > 0 and divide M, C_out, K");
    if (input.size() != static_cast<std::size_t>(geom.N) * geom.C_in * geom.H_in * geom.W_in)
        throw std::invalid_argument("run_conv2d_fused: input size does not match geometry");
    if (filter.size() != static_cast<std::size_t>(N) * geom.C_in * geom.Kh * geom.Kw)
        throw std::invalid_argument("run_conv2d_fused: filter size does not match geometry");
    if (bn && (bn->scale.size() != static_cast<std::size_t>(N) ||
               bn->shift.size() != static_cast<std::size_t>(N)))
        throw std::invalid_argument("run_conv2d_fused: bn scale/shift size must be C_out");
    if (!bias.empty() && bias.size() != static_cast<std::size_t>(N))
        throw std::invalid_argument("run_conv2d_fused: bias size must be C_out");

    // im2col A_col [M, K] and reshaped weights B_w [K, N].
    auto a_col = schedule::im2col_nchw(input, geom);
    auto b_w = schedule::filter_to_bw_nchw(filter, geom);

    // Fold: scale the weight columns; fused bias = bias*scale + shift.
    std::vector<float> fused_bias(bias.empty() ? std::vector<float>(N, 0.0f) : bias);
    bool has_bias = !bias.empty();
    if (bn) {
        for (Size k = 0; k < K; ++k)
            for (Size co = 0; co < N; ++co)
                b_w[static_cast<std::size_t>(k) * N + co] *= bn->scale[co];
        for (Size co = 0; co < N; ++co)
            fused_bias[co] = fused_bias[co] * bn->scale[co] + bn->shift[co];
        has_bias = true;
    }

    MatMulScheduleGenerator::Config cfg;
    cfg.M = M; cfg.N = N; cfg.K = K; cfg.Ti = cfg.Tj = cfg.Tk = T;
    cfg.a_base = 0x100000; cfg.b_base = 0x400000; cfg.c_base = 0x700000;
    auto schedule = MatMulScheduleGenerator(cfg).generate();
    if (!schedule.valid)
        throw std::runtime_error("run_conv2d_fused: schedule refused: " + schedule.error_message);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 20'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    for (const auto& op : schedule.operations) {
        if (op.type != schedule::ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A)
            exec.set_tile_payload(id, TilePayload{T, T, detail::block(a_col, K, id.ti, id.tk, T)});
        else
            exec.set_tile_payload(id, TilePayload{T, T, detail::block(b_w, N, id.tk, id.tj, T)});
    }
    for (const auto& op : schedule.operations) {
        using schedule::ScheduleOpType;
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
                if (has_bias) {
                    const Size tj = op.tile.tile_id.tj;
                    spec.bias.assign(fused_bias.begin() + static_cast<std::ptrdiff_t>(tj * T),
                                     fused_bias.begin() + static_cast<std::ptrdiff_t>(tj * T + T));
                }
                if (relu) spec.activation = ConcurrentTimingExecutor::FunctionalActivation::RELU;
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles)
        exec.step();
    if (!exec.is_complete())
        throw std::runtime_error("run_conv2d_fused: schedule did not complete");
    detail::accumulate(stats, exec);

    // Read C [M, N] from DRAM stores; reshape to NCHW y[n, co, ho, wo].
    const Size Hout = geom.H_out(), Wout = geom.W_out();
    std::vector<float> y(static_cast<std::size_t>(geom.N) * N * Hout * Wout);
    for (const auto& op : schedule.operations) {
        if (op.type != schedule::ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c) {
                const Size m = id.ti * T + r, co = id.tj * T + c;
                const Size n = m / (Hout * Wout), rem = m % (Hout * Wout);
                const Size ho = rem / Wout, wo = rem % Wout;
                y[((static_cast<std::size_t>(n) * N + co) * Hout + ho) * Wout + wo] =
                    p.values[r * T + c];
            }
    }
    return y;
}

/**
 * @brief Run a binary elementwise op (ADD residual join / MAX for ReLU) on the
 *        CSP executor via the value-producing functional path.
 *
 * For ReLU pass op = MAX and b = zeros. Both operands must be the same length.
 */
[[nodiscard]] inline std::vector<float>
run_elementwise(sw::kpu::isa::VEOp op, const std::vector<float>& a,
                const std::vector<float>& b, RunStats& stats) {
    using schedule::ElementwiseScheduleGenerator;
    using schedule::FunctionalElementwiseExecutor;
    if (a.size() != b.size())
        throw std::invalid_argument("run_elementwise: operand size mismatch");

    ElementwiseScheduleGenerator::Config gcfg;
    gcfg.num_elements = static_cast<Size>(a.size());
    gcfg.tile_elems = 256;
    gcfg.form = ElementwiseScheduleGenerator::Form::BINARY;
    gcfg.a_base = 0x100000; gcfg.b_base = 0x400000; gcfg.c_base = 0x700000;

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 20'000'000;

    FunctionalElementwiseExecutor exec(gcfg, ecfg);
    auto result = exec.run(op, a, b);
    if (!result.execution.success)
        throw std::runtime_error("run_elementwise: execution failed: " +
                                 result.execution.error_message);
    stats.total_cycles += result.execution.total_cycles;
    ++stats.ops;
    return result.values;
}

/// ReLU as MAX(x, 0) on the CSP executor.
[[nodiscard]] inline std::vector<float>
run_relu(const std::vector<float>& x, RunStats& stats) {
    return run_elementwise(sw::kpu::isa::VEOp::MAX, x,
                           std::vector<float>(x.size(), 0.0f), stats);
}

/**
 * @brief Run a matmul C = act(A @ W + bias) on the CSP executor (the FC layer).
 *
 * A is [M, K] row-major, W is [K, N] row-major. Reuses the value-producing GEMM
 * path (schedule_matmul_compute) with the bias + activation epilogue.
 *
 * @param T tile size (must divide M, N, K)
 */
[[nodiscard]] inline std::vector<float>
run_matmul(const std::vector<float>& a, const std::vector<float>& w,
           Size M, Size N, Size K, const std::vector<float>& bias, bool relu,
           Size T, RunStats& stats) {
    using schedule::MatMulScheduleGenerator;
    using MatrixID = sw::kpu::isa::MatrixID;
    if (T == 0 || M % T || N % T || K % T)
        throw std::invalid_argument("run_matmul: T must be > 0 and divide M, N, K");
    if (a.size() != static_cast<std::size_t>(M) * K)
        throw std::invalid_argument("run_matmul: A size does not match M*K");
    if (w.size() != static_cast<std::size_t>(K) * N)
        throw std::invalid_argument("run_matmul: W size does not match K*N");
    if (!bias.empty() && bias.size() != static_cast<std::size_t>(N))
        throw std::invalid_argument("run_matmul: bias size must be N or empty");

    MatMulScheduleGenerator::Config cfg;
    cfg.M = M; cfg.N = N; cfg.K = K; cfg.Ti = cfg.Tj = cfg.Tk = T;
    cfg.a_base = 0x100000; cfg.b_base = 0x400000; cfg.c_base = 0x700000;
    auto schedule = MatMulScheduleGenerator(cfg).generate();
    if (!schedule.valid)
        throw std::runtime_error("run_matmul: schedule refused: " + schedule.error_message);

    ConcurrentTimingExecutor::Config ecfg; ecfg.max_cycles = 20'000'000;
    ConcurrentTimingExecutor exec(ecfg);
    for (const auto& op : schedule.operations) {
        if (op.type != schedule::ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A)
            exec.set_tile_payload(id, TilePayload{T, T, detail::block(a, K, id.ti, id.tk, T)});
        else
            exec.set_tile_payload(id, TilePayload{T, T, detail::block(w, N, id.tk, id.tj, T)});
    }
    for (const auto& op : schedule.operations) {
        using schedule::ScheduleOpType;
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
                if (!bias.empty()) {
                    const Size tj = op.tile.tile_id.tj;
                    spec.bias.assign(bias.begin() + static_cast<std::ptrdiff_t>(tj * T),
                                     bias.begin() + static_cast<std::ptrdiff_t>(tj * T + T));
                }
                if (relu) spec.activation = ConcurrentTimingExecutor::FunctionalActivation::RELU;
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles)
        exec.step();
    if (!exec.is_complete()) throw std::runtime_error("run_matmul: did not complete");
    detail::accumulate(stats, exec);

    std::vector<float> c(static_cast<std::size_t>(M) * N);
    for (const auto& op : schedule.operations) {
        if (op.type != schedule::ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size col = 0; col < T; ++col)
                c[(id.ti * T + r) * N + (id.tj * T + col)] = p.values[r * T + col];
    }
    return c;
}

/**
 * @brief Global average pool on the CSP executor: mean over the H*W plane per
 *        (n, c). Input is NCHW; output is [N*C] row-major (the [N,C,1,1] tensor).
 *
 * Tiles the plane in Ti = H*W chunks so any spatial extent works (plane % Ti == 0).
 */
[[nodiscard]] inline std::vector<float>
run_global_avg_pool(const std::vector<float>& input,
                    const schedule::Pool2DGeometry& geom, RunStats& stats) {
    using namespace sw::kpu::timing::schedule;
    if (input.size() != geom.elems())
        throw std::invalid_argument("run_global_avg_pool: input size does not match geometry");
    const Size plane = geom.H * geom.W;
    const Size Ti = plane;  // one chunk per channel: plane % Ti == 0 always

    PoolingScheduleGenerator::Config cfg;
    cfg.geom = geom; cfg.mode = PoolingScheduleGenerator::Mode::GLOBAL_AVG; cfg.Ti = Ti;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    auto schedule = PoolingScheduleGenerator(cfg).generate();
    if (!schedule.valid)
        throw std::runtime_error("run_global_avg_pool: schedule refused: " + schedule.error_message);

    ConcurrentTimingExecutor::Config ecfg; ecfg.max_cycles = 20'000'000;
    ConcurrentTimingExecutor exec(ecfg);
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        const std::size_t base = static_cast<std::size_t>(id.ti) * plane + id.tj * Ti;
        exec.set_tile_payload(id, TilePayload{Ti, 1,
            std::vector<float>(input.begin() + static_cast<std::ptrdiff_t>(base),
                               input.begin() + static_cast<std::ptrdiff_t>(base + Ti))});
    }

    ScheduleExecutor sched_exec(exec);
    sched_exec.set_functional_compute_binder(
        [plane](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;
            spec.operation = [plane](const std::vector<TilePayload>& in) {
                float s = 0.0f;
                for (const auto& t : in) for (float v : t.values) s += v;
                return TilePayload{1, 1, {s / static_cast<float>(plane)}};
            };
            return spec;
        });
    auto result = sched_exec.execute(schedule);
    if (!result.success)
        throw std::runtime_error("run_global_avg_pool: execution failed: " + result.error_message);
    stats.total_cycles += result.total_cycles;
    ++stats.ops;

    std::vector<float> out(static_cast<std::size_t>(geom.N) * geom.C);
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        out[id.ti] = exec.tile_payload_at(MemoryLevel::DRAM, id).values.at(0);
    }
    return out;
}

} // namespace sw::kpu::timing::graph
