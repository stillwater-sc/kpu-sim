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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sw::kpu::timing::graph {

using schedule::Conv2DGeometry;
using schedule::BatchNormAffine;

/// Optional per-cycle observer, invoked with the live executor after each step()
/// (and once before the first). Used to drive a TileTracker occupancy timeline;
/// empty by default so the normal execution path pays nothing.
using CycleObserver = std::function<void(const ConcurrentTimingExecutor&)>;

/// Accumulated CSP-execution stats across the ops of a graph run.
///
/// Nodes execute sequentially (GraphCspExecutor topological walk), so each op runs
/// on a fresh ConcurrentTimingExecutor and its per-op Statistics are additive. The
/// movement-fabric activity below is summed the same way as the stall/cycle totals.
///
/// Utilization is Sum(busy) / Sum(total_cycles) per mover. busy is the executor's
/// DIRECTLY MEASURED active-cycle count (each component counts, in its tick(), the
/// cycles a transfer actually occupied it) averaged per component — not derived
/// from total - stalls, so idle cycles are excluded. Because nodes run
/// sequentially this captures within-op activity, not cross-branch overlap
/// (see docs/benchmarking/resnet-benchmarking-guide.md, section 6).
struct RunStats {
    Cycle total_cycles = 0;
    Cycle dma_stalls = 0;
    Cycle bm_stalls = 0;
    Cycle str_stalls = 0;
    std::size_t ops = 0;

    // Idealized concurrent-branch-overlap latency: the DAG critical path assuming
    // execution-level-independent ops run concurrently with unbounded resources.
    // <= total_cycles; equal when the graph is a chain. An upper bound on what
    // concurrent branch scheduling could buy (ignores DMA/mover contention).
    Cycle critical_path_cycles = 0;

    // Movement-fabric activity (summed per-op). busy is fractional: each op's
    // per-component mean active cycles is exact (not integer-floored), so the
    // whole-network sum carries that precision.
    double dma_busy = 0.0;
    double bm_busy = 0.0;
    double str_busy = 0.0;
    std::size_t tiles_loaded = 0;
    std::size_t tiles_stored = 0;
    std::size_t tiles_moved = 0;
    std::size_t tiles_fed = 0;
    std::size_t bytes_loaded = 0;
    std::size_t bytes_stored = 0;

    // Multiply-accumulates performed by the GEMM ops (conv im2col + FC matmul),
    // accumulated during the graph walk. Fused BatchNorm, ReLU epilogues, residual
    // adds, and pooling are not GEMMs and contribute no MACs.
    std::uint64_t total_macs = 0;

    // Utilization (0.0 - 1.0) = busy / total_cycles, over the whole run.
    [[nodiscard]] double dma_utilization() const {
        return total_cycles ? static_cast<double>(dma_busy) / static_cast<double>(total_cycles) : 0.0;
    }
    [[nodiscard]] double bm_utilization() const {
        return total_cycles ? static_cast<double>(bm_busy) / static_cast<double>(total_cycles) : 0.0;
    }
    [[nodiscard]] double str_utilization() const {
        return total_cycles ? static_cast<double>(str_busy) / static_cast<double>(total_cycles) : 0.0;
    }

    // Effective DRAM bandwidth (GB/s) at an assumed clock: bytes / (cycles/clock).
    // A non-positive clock (or zero cycles) yields 0.0 rather than inf/NaN.
    [[nodiscard]] double effective_load_bandwidth(double clock_ghz) const {
        if (clock_ghz <= 0.0 || total_cycles == 0) return 0.0;
        const double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        return static_cast<double>(bytes_loaded) / (seconds * 1e9);
    }
    [[nodiscard]] double effective_store_bandwidth(double clock_ghz) const {
        if (clock_ghz <= 0.0 || total_cycles == 0) return 0.0;
        const double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        return static_cast<double>(bytes_stored) / (seconds * 1e9);
    }

    // ---- Compute FLOP efficiency (roofline) ---------------------------------
    // Conventions match the matmul benchmark harness (sw::benchmark::HardwareSpec):
    // 2 FLOPs per MAC, achieved GFLOP/s = FLOP/cycle * clock, peak/bandwidth
    // supplied by the caller (no PE-array peak lives in the timing config).

    /// Best-case speedup from overlapping execution-level-independent branches
    /// (resource-unconstrained upper bound): sequential cycles / critical path.
    /// 1.0 for a chain graph; > 1 when independent branches could overlap.
    [[nodiscard]] double overlap_speedup() const {
        return critical_path_cycles
            ? static_cast<double>(total_cycles) / static_cast<double>(critical_path_cycles)
            : 1.0;
    }

    /// Total floating-point operations (2 per MAC).
    [[nodiscard]] double total_flops() const { return 2.0 * static_cast<double>(total_macs); }

    /// Achieved compute throughput (GFLOP/s) at the given clock.
    [[nodiscard]] double achieved_gflops(double clock_ghz) const {
        return total_cycles ? total_flops() / static_cast<double>(total_cycles) * clock_ghz : 0.0;
    }

    /// Arithmetic intensity: FLOPs per byte of external (DRAM) traffic.
    [[nodiscard]] double arithmetic_intensity() const {
        const std::size_t bytes = bytes_loaded + bytes_stored;
        return bytes ? total_flops() / static_cast<double>(bytes) : 0.0;
    }

    /// Fraction of the compute fabric's peak (clock-independent): achieved
    /// FLOP/cycle divided by peak FLOP/cycle (e.g. 16*16*2 = 512 for a 16x16 PE).
    [[nodiscard]] double compute_efficiency(double peak_flops_per_cycle) const {
        return (total_cycles && peak_flops_per_cycle > 0.0)
            ? total_flops() / (static_cast<double>(total_cycles) * peak_flops_per_cycle)
            : 0.0;
    }

    /// Fraction of the roofline ceiling min(AI * bandwidth, peak) at the given
    /// clock - i.e. how close to the attainable (memory- or compute-bound) limit.
    [[nodiscard]] double roofline_efficiency(double peak_gflops, double ext_bw_gbs,
                                             double clock_ghz) const {
        const double roof = std::min(arithmetic_intensity() * ext_bw_gbs, peak_gflops);
        return roof > 0.0 ? achieved_gflops(clock_ghz) / roof : 0.0;
    }
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

// Fold one op's executor Statistics into the running RunStats. EVERY
// RunStats-producing runner must call this (directly, or via the executor
// overload) so busy cycles / tiles / bytes land in the numerators for the same
// ops whose total_cycles land in the denominator. Instrumenting total_cycles
// without busy understates utilization (see resnet-benchmarking-guide.md).
inline void accumulate(RunStats& s, const ConcurrentTimingExecutor::Statistics& st) {
    s.total_cycles += st.total_cycles;
    s.dma_stalls += st.dma_credit_stalls;
    s.bm_stalls += st.bm_tag_stalls + st.bm_credit_stalls;
    s.str_stalls += st.str_tag_stalls + st.str_credit_stalls;
    s.dma_busy += st.dma_busy_cycles;
    s.bm_busy += st.bm_busy_cycles;
    s.str_busy += st.str_busy_cycles;
    s.tiles_loaded += st.tiles_loaded;
    s.tiles_stored += st.tiles_stored;
    s.tiles_moved += st.tiles_moved;
    s.tiles_fed += st.tiles_fed;
    s.bytes_loaded += st.bytes_loaded;
    s.bytes_stored += st.bytes_stored;
    ++s.ops;
}

// Statistics.total_cycles == executor.current_cycle(), so this is equivalent to
// accumulating from the executor's live statistics.
inline void accumulate(RunStats& s, const ConcurrentTimingExecutor& exec) {
    accumulate(s, exec.get_statistics());
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
                 const std::vector<float>& bias, bool relu, Size T, RunStats& stats,
                 const CycleObserver& on_cycle = {}) {
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
    if (on_cycle) on_cycle(exec);
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
        exec.step();
        if (on_cycle) on_cycle(exec);
    }
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
    detail::accumulate(stats, result.execution.stats);
    return result.values;
}

/// ReLU as MAX(x, 0) on the CSP executor.
[[nodiscard]] inline std::vector<float>
run_relu(const std::vector<float>& x, RunStats& stats) {
    return run_elementwise(sw::kpu::isa::VEOp::MAX, x,
                           std::vector<float>(x.size(), 0.0f), stats);
}

/**
 * @brief Run a unary/scalar VE op (NEG/EXP/... or ADD_S/MUL_S/POW_S) on the CSP
 *        executor via the value-producing functional path (UNARY form).
 */
[[nodiscard]] inline std::vector<float>
run_ve_unary(sw::kpu::isa::VEOp op, const std::vector<float>& x, float scalar,
             RunStats& stats) {
    using schedule::ElementwiseScheduleGenerator;
    using schedule::FunctionalElementwiseExecutor;

    ElementwiseScheduleGenerator::Config gcfg;
    gcfg.num_elements = static_cast<Size>(x.size());
    gcfg.tile_elems = 256;
    gcfg.form = ElementwiseScheduleGenerator::Form::UNARY;
    gcfg.a_base = 0x100000; gcfg.b_base = 0x400000; gcfg.c_base = 0x700000;

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 20'000'000;

    FunctionalElementwiseExecutor exec(gcfg, ecfg);
    auto result = exec.run(op, x, {}, scalar);
    if (!result.execution.success)
        throw std::runtime_error("run_ve_unary: execution failed: " +
                                 result.execution.error_message);
    detail::accumulate(stats, result.execution.stats);
    return result.values;
}

/**
 * @brief Sigmoid = 1/(1 + e^-x) on the CSP executor (EfficientNet SE gate).
 *
 * The Vector Engine has no sigmoid op, so it is composed from four unary/scalar
 * VE ops on the value path: POW_S(ADD_S(EXP(NEG(x)), 1), -1).
 */
[[nodiscard]] inline std::vector<float>
run_sigmoid(const std::vector<float>& x, RunStats& stats) {
    using VEOp = sw::kpu::isa::VEOp;
    auto t = run_ve_unary(VEOp::NEG, x, 0.0f, stats);          // -x
    t = run_ve_unary(VEOp::EXP, t, 0.0f, stats);               // e^-x
    t = run_ve_unary(VEOp::ADD_S, t, 1.0f, stats);             // 1 + e^-x
    return run_ve_unary(VEOp::POW_S, t, -1.0f, stats);         // 1/(1 + e^-x)
}

/**
 * @brief SiLU / swish = x * sigmoid(x) on the CSP executor (EfficientNet activation).
 *
 * Composed on the value path: sigmoid(x) (four VE ops, run_sigmoid) then a binary
 * multiply against x - five VE ops total.
 */
[[nodiscard]] inline std::vector<float>
run_silu(const std::vector<float>& x, RunStats& stats) {
    auto sig = run_sigmoid(x, stats);
    return run_elementwise(sw::kpu::isa::VEOp::MUL, x, sig, stats);
}

/**
 * @brief Channel-broadcast multiply for the squeeze-and-excitation gate: scale
 *        an [N,C,H,W] activation by a per-channel [N,C] gate on the CSP executor.
 *
 * Each activation element (n,c,h,w) is multiplied by gate[n,c]. Because both are
 * row-major, the gate index of activation element i is i / (H*W) where the plane
 * size H*W = activation.size()/gate.size(), so no explicit geometry is needed.
 * The gate is expanded host-side (like run_relu's zeros / run_matmul's blocking)
 * and the C*H*W multiplies run on the CSP data path via a binary MUL.
 */
[[nodiscard]] inline std::vector<float>
run_channel_broadcast_mul(const std::vector<float>& activation,
                          const std::vector<float>& gate, RunStats& stats) {
    if (gate.empty() || activation.size() % gate.size() != 0)
        throw std::invalid_argument(
            "run_channel_broadcast_mul: activation size must be a multiple of gate size");
    const std::size_t plane = activation.size() / gate.size();
    std::vector<float> gate_full(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i)
        gate_full[i] = gate[i / plane];
    return run_elementwise(sw::kpu::isa::VEOp::MUL, activation, gate_full, stats);
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
 * @brief Depthwise convolution on the CSP executor (M3-T2, #209).
 *
 * Each output channel is the 2D conv of a single input channel with its own
 * Kh*Kw filter (groups = C). Reuses the E7 pooling per-channel window unfold for
 * movement (pool_window_channel, 0-filled at padding) and reduces each window row
 * by a dot-product with that channel's folded filter (a weighted sum) - the M3
 * design's decision. Optional folded BatchNorm and a ReLU6 epilogue apply in the
 * reduce, so depthwise+BN+ReLU6 is one CSP op.
 *
 *   y[n,c,ho,wo] = act( sum_k window(n,c,ho,wo)[k] * filter[c][k] + bias[c] )
 *
 * @param filter per-channel filters, size C*Kh*Kw (row-major [C, Kh, Kw])
 * @param bn     optional folded BatchNorm affine (size C), or nullptr
 * @param bias   optional per-channel bias (size C), or empty
 * @param relu6  apply ReLU6 = min(max(x,0),6) as the epilogue
 * @param T      tile size (must divide N*Hout*Wout)
 *
 * @note The #210 compute-result-CAM cap is fixed, so high channel counts work
 *       (validated to C=48, and hidden=64 at 8x8 = 65536 outputs / 4096 output
 *       tiles once the livelock detector was taught to count backward-path and
 *       compute progress, not just load/move/feed).
 */
[[nodiscard]] inline std::vector<float>
run_depthwise_conv(const std::vector<float>& input,
                   const schedule::Pool2DGeometry& geom,
                   const std::vector<float>& filter, const BatchNormAffine* bn,
                   const std::vector<float>& bias, bool relu6, Size T, RunStats& stats) {
    using namespace sw::kpu::timing::schedule;
    const Size C = geom.C, K = geom.window();
    const Size Hout = geom.H_out(), Wout = geom.W_out();
    const Size Mpos = geom.N * Hout * Wout;                 // positions per channel
    if (input.size() != geom.elems())
        throw std::invalid_argument("run_depthwise_conv: input size does not match geometry");
    if (filter.size() != static_cast<std::size_t>(C) * K)
        throw std::invalid_argument("run_depthwise_conv: filter size must be C*Kh*Kw");
    if (T == 0 || Mpos % T != 0)
        throw std::invalid_argument("run_depthwise_conv: T must be > 0 and divide N*Hout*Wout");
    if (bn && bn->scale.size() != static_cast<std::size_t>(C))
        throw std::invalid_argument("run_depthwise_conv: bn size must be C");
    if (!bias.empty() && bias.size() != static_cast<std::size_t>(C))
        throw std::invalid_argument("run_depthwise_conv: bias size must be C");

    // Fold BN into the per-channel filter + bias.
    std::vector<float> fw(filter);
    std::vector<float> fb(bias.empty() ? std::vector<float>(C, 0.0f) : bias);
    if (bn) {
        for (Size c = 0; c < C; ++c) {
            for (Size k = 0; k < K; ++k) fw[c * K + k] *= bn->scale[c];
            fb[c] = fb[c] * bn->scale[c] + bn->shift[c];
        }
    }

    // Per-(n,c) window rows [Hout*Wout, K], 0-filled at padding (AVG mode).
    std::vector<std::vector<float>> win(static_cast<std::size_t>(geom.N) * C);
    for (Size n = 0; n < geom.N; ++n)
        for (Size c = 0; c < C; ++c)
            win[static_cast<std::size_t>(n) * C + c] =
                pool_window_channel(input, geom, n, c, PoolType::AVG).rows;

    PoolingScheduleGenerator::Config cfg;
    cfg.geom = geom; cfg.mode = PoolingScheduleGenerator::Mode::WINDOWED; cfg.Ti = T;
    cfg.l3_buffer_count = 128; cfg.l2_bank_count = 128;
    cfg.input_base = 0x100000; cfg.output_base = 0x400000;
    auto schedule = PoolingScheduleGenerator(cfg).generate();
    if (!schedule.valid)
        throw std::runtime_error("run_depthwise_conv: schedule refused: " + schedule.error_message);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 50'000'000;
    ecfg.l3_buffer_count = 128; ecfg.l2_bank_count = 128;
    ConcurrentTimingExecutor exec(ecfg);
    // Seed each window tile (channel c = id.ti, output-tile ti = id.tj).
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        const Size c = id.ti, tb = id.tj;
        std::vector<float> blk(static_cast<std::size_t>(T) * K);
        for (Size r = 0; r < T; ++r) {
            const Size p = tb * T + r;                       // flattened position
            const Size n = p / (Hout * Wout), m = p % (Hout * Wout);
            const auto& wr = win[static_cast<std::size_t>(n) * C + c];
            for (Size k = 0; k < K; ++k) blk[r * K + k] = wr[static_cast<std::size_t>(m) * K + k];
        }
        exec.set_tile_payload(id, TilePayload{T, K, std::move(blk)});
    }

    ScheduleExecutor se(exec);
    se.set_functional_compute_binder(
        [&fw, &fb, K, relu6](const ScheduleOperation& op)
            -> std::optional<ConcurrentTimingExecutor::FunctionalComputeSpec> {
            const Size c = op.tile.tile_id.ti;               // output channel
            ConcurrentTimingExecutor::FunctionalComputeSpec spec;
            spec.input_tiles = op.dependency_tiles;          // {window tile}
            spec.operation = [&fw, &fb, K, c, relu6](const std::vector<TilePayload>& in) {
                const auto& x = in.at(0);                    // [Ti, K]
                const float* f = &fw[static_cast<std::size_t>(c) * K];
                const float b = fb[c];
                TilePayload out{x.rows, 1, std::vector<float>(x.rows)};
                for (Size r = 0; r < x.rows; ++r) {
                    const float* row = &x.values[static_cast<std::size_t>(r) * K];
                    float acc = b;
                    for (Size k = 0; k < K; ++k) acc += row[k] * f[k];
                    if (relu6) acc = std::min(std::max(0.0f, acc), 6.0f);
                    out.values[r] = acc;
                }
                return out;
            };
            return spec;
        });
    auto result = se.execute(schedule);
    if (!result.success)
        throw std::runtime_error("run_depthwise_conv: execution failed: " + result.error_message);
    detail::accumulate(stats, exec);

    // Read output [C, Mpos] (per-channel positions) -> NCHW.
    std::vector<float> y(static_cast<std::size_t>(geom.N) * C * Hout * Wout);
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const Size c = id.ti, tb = id.tj;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r) {
            const Size pos = tb * T + r;
            const Size n = pos / (Hout * Wout), m = pos % (Hout * Wout);
            const Size ho = m / Wout, wo = m % Wout;
            y[((static_cast<std::size_t>(n) * C + c) * Hout + ho) * Wout + wo] = p.values[r];
        }
    }
    return y;
}

/// ReLU6 = min(max(x, 0), 6) on the CSP executor (two elementwise ops).
[[nodiscard]] inline std::vector<float>
run_relu6(const std::vector<float>& x, RunStats& stats) {
    auto r = run_elementwise(sw::kpu::isa::VEOp::MAX, x, std::vector<float>(x.size(), 0.0f), stats);
    return run_elementwise(sw::kpu::isa::VEOp::MIN, r, std::vector<float>(x.size(), 6.0f), stats);
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
    detail::accumulate(stats, exec);

    std::vector<float> out(static_cast<std::size_t>(geom.N) * geom.C);
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        out[id.ti] = exec.tile_payload_at(MemoryLevel::DRAM, id).values.at(0);
    }
    return out;
}

} // namespace sw::kpu::timing::graph
