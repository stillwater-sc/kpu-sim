// ============================================================================
// tests/timing/test_epilogue_fused_regression.cpp
// Fused-epilogue regression: matmul + bias + activation applied in the compute
// (no DRAM round-trip for the intermediate), plus a conv+BN+ReLU composition
// cell (issue #188, epic E10). See docs/plans/e10_fused_epilogue_pattern.md.
//
// The fused epilogue is already the value path (MatMulComputeSpec bias +
// activation, validated in M1 and E6-T4). This locks it in as its own coverage
// row: a bias x activation x shape x envelope matrix checked elementwise vs a
// host oracle Y = act(A@B + bias), the fusion invariant (no extra STORE for the
// epilogue), per-stage tile accounting, credit conservation, and characterization.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/batchnorm_affine.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;
using Activation = ConcurrentTimingExecutor::FunctionalActivation;

namespace {

std::vector<float> ramp(std::size_t n, int period, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = base + step * static_cast<float>(i % static_cast<std::size_t>(period));
    return v;
}

std::vector<float> block(const std::vector<float>& mat, Size cols,
                         Size br, Size bc, Size T) {
    std::vector<float> b(static_cast<std::size_t>(T) * T);
    for (Size r = 0; r < T; ++r)
        for (Size c = 0; c < T; ++c)
            b[r * T + c] = mat[(br * T + r) * cols + (bc * T + c)];
    return b;
}

// Enqueue a matmul schedule with a fused bias/activation epilogue and run it.
struct CellStats {
    std::size_t store_ops = 0, out_tiles = 0;  // count_ops() is size_t (MSVC C4267)
    Cycle cycles = 0, stalls = 0;
    double dma_util = 0.0, str_util = 0.0;
};

// Runs C = act(A@B + bias); fills stats; returns max abs error vs the oracle.
double run_matmul_epilogue(Size M, Size N, Size K, Size T,
                           const std::vector<float>& bias, bool relu,
                           Size l3, Size l2, bool partitioned, CellStats& out) {
    auto A = ramp(static_cast<std::size_t>(M) * K, 7, -1.0f, 0.5f);
    auto B = ramp(static_cast<std::size_t>(K) * N, 5, -1.0f, 0.5f);

    // Host oracle Y = act(A@B + bias[col]).
    std::vector<float> Y(static_cast<std::size_t>(M) * N, 0.0f);
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k) {
            const float a = A[static_cast<std::size_t>(i) * K + k];
            for (Size j = 0; j < N; ++j)
                Y[static_cast<std::size_t>(i) * N + j] += a * B[static_cast<std::size_t>(k) * N + j];
        }
    for (Size i = 0; i < M; ++i)
        for (Size j = 0; j < N; ++j) {
            float& v = Y[static_cast<std::size_t>(i) * N + j];
            if (!bias.empty()) v += bias[j];
            if (relu && v < 0.0f) v = 0.0f;
        }

    MatMulScheduleGenerator::Config cfg;
    cfg.M = M; cfg.N = N; cfg.K = K; cfg.Ti = cfg.Tj = cfg.Tk = T;
    cfg.l3_buffer_count = l3; cfg.l2_bank_count = l2;
    cfg.a_base = 0x100000; cfg.b_base = 0x400000; cfg.c_base = 0x700000;
    auto schedule = MatMulScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.l3_buffer_count = l3; ecfg.l2_bank_count = l2;
    ecfg.partition_l3_credits = partitioned; ecfg.partition_l2_credits = partitioned;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A)
            exec.set_tile_payload(id, TilePayload{T, T, block(A, K, id.ti, id.tk, T)});
        else
            exec.set_tile_payload(id, TilePayload{T, T, block(B, N, id.tk, id.tj, T)});
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
                if (relu) spec.activation = Activation::RELU;
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles)
        exec.step();
    REQUIRE(exec.is_complete());

    // Fusion invariant: exactly one STORE per output tile - the pre-bias
    // accumulator is never materialized to DRAM by an extra store.
    const std::size_t m_tiles = static_cast<std::size_t>((M + T - 1) / T);
    const std::size_t n_tiles = static_cast<std::size_t>((N + T - 1) / T);
    out.store_ops = schedule.count_ops(ScheduleOpType::STORE);
    out.out_tiles = m_tiles * n_tiles;
    REQUIRE(out.store_ops == out.out_tiles);

    // Per-stage tile accounting + credit conservation.
    const auto stats = exec.get_statistics();
    REQUIRE(stats.tiles_drained == schedule.count_ops(ScheduleOpType::DRAIN));
    REQUIRE(stats.tiles_stored == schedule.count_ops(ScheduleOpType::STORE));
    REQUIRE(exec.l3_credits().available() == l3);
    REQUIRE(exec.l2_credits().available() == l2);

    out.cycles = exec.current_cycle();
    out.stalls = stats.dma_credit_stalls + stats.bm_tag_stalls + stats.bm_credit_stalls +
                 stats.str_tag_stalls + stats.str_credit_stalls;
    out.dma_util = stats.dma_utilization();
    out.str_util = stats.str_utilization();

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c) {
                const double got = p.values[r * T + c];
                REQUIRE(std::isfinite(got));
                const double want = Y[(id.ti * T + r) * N + (id.tj * T + c)];
                max_err = std::max(max_err, std::abs(got - want));
            }
    }
    return max_err;
}

struct CharRow { std::string cell; Cycle cycles; Cycle stalls; double dma, str; };
std::vector<CharRow>& characterization() { static std::vector<CharRow> r; return r; }

} // namespace

TEST_CASE("Fused epilogue: matmul + bias + activation matches host oracle",
          "[timing][regression][epilogue][matrix]") {
    struct Shape { const char* name; Size M, N, K, T; };
    const Shape shapes[] = {{"32x32x32", 32, 32, 32, 16},
                            {"64x32x48", 64, 32, 48, 16}};
    struct Env { const char* name; Size l3, l2; bool part; };
    const Env envs[] = {{"default", 32, 64, false}, {"partitioned", 32, 64, true}};

    for (const auto& s : shapes) {
        // Per-output-column bias; RELU cases use a bias that drives negatives.
        std::vector<float> bias = ramp(s.N, 6, -2.0f, 0.5f);
        std::vector<float> neg_bias(s.N, -1000.0f);  // forces the ReLU clamp
        for (const auto& e : envs) {
            DYNAMIC_SECTION(s.name << "/" << e.name << "/no-bias-linear") {
                CellStats st;
                REQUIRE(run_matmul_epilogue(s.M, s.N, s.K, s.T, {}, false,
                                            e.l3, e.l2, e.part, st) < 1e-3);
                characterization().push_back(
                    {std::string(s.name) + "/" + e.name + "/linear",
                     st.cycles, st.stalls, st.dma_util, st.str_util});
            }
            DYNAMIC_SECTION(s.name << "/" << e.name << "/bias") {
                CellStats st;
                REQUIRE(run_matmul_epilogue(s.M, s.N, s.K, s.T, bias, false,
                                            e.l3, e.l2, e.part, st) < 1e-3);
            }
            DYNAMIC_SECTION(s.name << "/" << e.name << "/bias+relu") {
                CellStats st;
                REQUIRE(run_matmul_epilogue(s.M, s.N, s.K, s.T, bias, true,
                                            e.l3, e.l2, e.part, st) < 1e-3);
            }
            DYNAMIC_SECTION(s.name << "/" << e.name << "/relu-clamp-all") {
                CellStats st;
                // All outputs clamp to 0; the drained tile must be exactly 0.
                REQUIRE(run_matmul_epilogue(s.M, s.N, s.K, s.T, neg_bias, true,
                                            e.l3, e.l2, e.part, st) < 1e-3);
            }
        }
    }
}

TEST_CASE("Fused epilogue: conv + BN + ReLU composition (ResNet block)",
          "[timing][regression][epilogue][conv-bn-relu]") {
    // conv -> BN -> ReLU folds to a single GEMM with a scaled weight and a fused
    // bias + ReLU: y = relu(conv(x) * bn_scale[co] + bn_shift[co]).
    Conv2DGeometry cg;
    cg.N = 1; cg.C_in = 16; cg.H_in = 8; cg.W_in = 8; cg.C_out = 16;
    cg.Kh = 3; cg.Kw = 3; cg.pad_h = 1; cg.pad_w = 1;  // M=64, K=144, Cout=16
    const Size T = 16;

    auto input = ramp(cg.input_elems(), 7, 0.5f, 0.5f);
    auto filter = ramp(cg.filter_elems(), 5, -1.0f, 0.5f);
    // BN params for the Cout channels; fold to scale/shift.
    std::vector<float> gamma(cg.C_out), beta(cg.C_out), mean(cg.C_out), var(cg.C_out);
    for (Size c = 0; c < cg.C_out; ++c) {
        gamma[c] = 0.5f + 0.25f * static_cast<float>(c % 4);
        beta[c]  = -3.0f + 0.5f * static_cast<float>(c % 6);  // some channels negative
        mean[c]  = 0.5f * static_cast<float>(c % 3);
        var[c]   = 0.5f + 0.25f * static_cast<float>(c % 4);
    }
    const auto bn = bn_fold(gamma, beta, mean, var, 1e-3f);

    const auto a_col = im2col_nchw(input, cg);         // [M, K]
    auto b_w = filter_to_bw_nchw(filter, cg);          // [K, Cout]
    // Fold BN scale into the weight columns; fused bias = bn shift.
    for (Size k = 0; k < cg.K(); ++k)
        for (Size co = 0; co < cg.C_out; ++co)
            b_w[static_cast<std::size_t>(k) * cg.C_out + co] *= bn.scale[co];

    // Host oracle: relu(conv * bn_scale + bn_shift) via the folded GEMM.
    const Size M = cg.M(), N = cg.C_out, K = cg.K();
    std::vector<float> Y(static_cast<std::size_t>(M) * N, 0.0f);
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k) {
            const float a = a_col[static_cast<std::size_t>(i) * K + k];
            for (Size j = 0; j < N; ++j)
                Y[static_cast<std::size_t>(i) * N + j] += a * b_w[static_cast<std::size_t>(k) * N + j];
        }
    for (Size i = 0; i < M; ++i)
        for (Size j = 0; j < N; ++j) {
            float& v = Y[static_cast<std::size_t>(i) * N + j];
            v += bn.shift[j];
            if (v < 0.0f) v = 0.0f;
        }

    MatMulScheduleGenerator::Config cfg;
    cfg.M = M; cfg.N = N; cfg.K = K; cfg.Ti = cfg.Tj = cfg.Tk = T;
    cfg.a_base = 0x100000; cfg.b_base = 0x400000; cfg.c_base = 0x700000;
    auto schedule = MatMulScheduleGenerator(cfg).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A)
            exec.set_tile_payload(id, TilePayload{T, T, block(a_col, K, id.ti, id.tk, T)});
        else
            exec.set_tile_payload(id, TilePayload{T, T, block(b_w, N, id.tk, id.tj, T)});
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
                ConcurrentTimingExecutor::MatMulComputeSpec spec;
                for (const auto& dep : op.dependency_tiles) {
                    if (dep.matrix == MatrixID::A) spec.a_tiles.push_back(dep);
                    else                            spec.b_tiles.push_back(dep);
                }
                const Size tj = op.tile.tile_id.tj;
                spec.bias.assign(bn.shift.begin() + static_cast<std::ptrdiff_t>(tj * T),
                                 bn.shift.begin() + static_cast<std::ptrdiff_t>(tj * T + T));
                spec.activation = Activation::RELU;
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles)
        exec.step();
    REQUIRE(exec.is_complete());

    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c) {
                const double got = p.values[r * T + c];
                REQUIRE(std::isfinite(got));
                max_err = std::max(max_err, std::abs(got - Y[(id.ti * T + r) * N + (id.tj * T + c)]));
            }
    }
    REQUIRE(max_err < 1e-3);
}

TEST_CASE("Fused epilogue characterization report",
          "[timing][regression][epilogue][report]") {
    const auto& rows = characterization();
    if (rows.empty()) { SUCCEED("matrix test did not run"); return; }
    std::printf("\n%-28s %9s %8s %6s %6s\n", "cell", "cycles", "stalls", "dma%", "str%");
    for (const auto& r : rows)
        std::printf("%-28s %9llu %8llu %6.1f %6.1f\n", r.cell.c_str(),
                    static_cast<unsigned long long>(r.cycles),
                    static_cast<unsigned long long>(r.stalls),
                    100.0 * r.dma, 100.0 * r.str);
    SUCCEED("characterization recorded");
}
