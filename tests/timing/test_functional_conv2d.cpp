// ============================================================================
// tests/timing/test_functional_conv2d.cpp
// Value-producing conv2d on the CSP executor vs. a host oracle (E6-T4, #122).
//
// Seeds the im2col A_col operand and the reshaped B_w weights (E6-T2 helpers)
// as tile payloads, executes the Conv2DScheduleGenerator schedule through the
// value-producing matmul path (schedule_matmul_compute, with per-output-channel
// bias + ReLU on the COMPUTE), and checks every drained C tile elementwise
// against conv2d_reference. This is the conv2d.functional coverage gate.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/conv2d_im2col.hpp>
#include <sw/kpu/timing/schedule/conv2d_schedule_generator.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;

namespace {

// Extract the [T x T] block (br, bc) from a row-major [rows x cols] matrix.
std::vector<float> block(const std::vector<float>& mat, Size cols,
                         Size br, Size bc, Size T) {
    std::vector<float> b(static_cast<std::size_t>(T) * T);
    for (Size r = 0; r < T; ++r)
        for (Size c = 0; c < T; ++c)
            b[r * T + c] = mat[(br * T + r) * cols + (bc * T + c)];
    return b;
}

// Bounded deterministic fill. Values are kept small (a short repeating set) so
// the K-reduction stays in a range where fp32 is exact-ish: the executor sums
// the K products then adds bias, while the host reference seeds acc=bias first,
// so with large partial sums the two rounding orders diverge by a few ULPs. A
// bounded magnitude keeps that divergence well under the 1e-3 tolerance and
// exercises the same functional path.
std::vector<float> fill(std::size_t n, int period, float base, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = base + step * static_cast<float>(i % static_cast<std::size_t>(period));
    return v;
}

// Convert an NCHW conv2d_reference tensor into GEMM row-major [M x Cout], the
// layout the C tiles are stored in (C[m, co], m = (n*Hout+ho)*Wout+wo).
std::vector<float> ref_to_gemm(const std::vector<float>& y, const Conv2DGeometry& g) {
    const Size Hout = g.H_out(), Wout = g.W_out();
    std::vector<float> gemm(static_cast<std::size_t>(g.M()) * g.C_out);
    for (Size n = 0; n < g.N; ++n)
        for (Size co = 0; co < g.C_out; ++co)
            for (Size ho = 0; ho < Hout; ++ho)
                for (Size wo = 0; wo < Wout; ++wo) {
                    const Size m = (n * Hout + ho) * Wout + wo;
                    const std::size_t yi =
                        ((static_cast<std::size_t>(n) * g.C_out + co) * Hout + ho) *
                            Wout + wo;
                    gemm[static_cast<std::size_t>(m) * g.C_out + co] = y[yi];
                }
    return gemm;
}

Conv2DScheduleGenerator::Config gen_config(const Conv2DGeometry& g, Size T,
                                           Conv2DScheduleGenerator::Strategy s) {
    Conv2DScheduleGenerator::Config c;
    c.N = g.N; c.H_in = g.H_in; c.W_in = g.W_in; c.C_in = g.C_in;
    c.C_out = g.C_out; c.Kh = g.Kh; c.Kw = g.Kw;
    c.stride_h = g.stride_h; c.stride_w = g.stride_w;
    c.padding_h = g.pad_h; c.padding_w = g.pad_w;
    c.Ti = T; c.Tj = T; c.Tk = T;
    c.strategy = s;
    c.input_base = 0x00001000; c.filter_base = 0x00100000; c.output_base = 0x00200000;
    return c;
}

// Execute one conv2d and return the max abs error vs. the host oracle.
double run_and_compare(const Conv2DGeometry& g, Size T,
                       Conv2DScheduleGenerator::Strategy strategy,
                       const std::vector<float>& bias, bool relu) {
    // Tile sizing must be square and dividing M/Cout/K for direct seeding.
    REQUIRE(g.M() % T == 0);
    REQUIRE(g.C_out % T == 0);
    REQUIRE(g.K() % T == 0);

    auto input = fill(g.input_elems(), 7, 0.5f, 0.5f);    // 0.5 .. 3.5
    auto filter = fill(g.filter_elems(), 5, -1.0f, 0.5f);  // -1 .. 1
    const auto a_col = im2col_nchw(input, g);        // [M, K]
    const auto b_w = filter_to_bw_nchw(filter, g);   // [K, Cout]
    const auto ref = ref_to_gemm(conv2d_reference(input, filter, bias, g, relu), g);

    auto schedule = Conv2DScheduleGenerator(gen_config(g, T, strategy)).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ecfg;
    ecfg.max_cycles = 2'000'000;
    ConcurrentTimingExecutor exec(ecfg);

    // Seed A_col and B_w input tiles from the host operands.
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        if (id.matrix == MatrixID::A) {
            exec.set_tile_payload(id, TilePayload{T, T, block(a_col, g.K(), id.ti, id.tk, T)});
        } else {  // B_w block (tk, tj)
            exec.set_tile_payload(id, TilePayload{T, T, block(b_w, g.C_out, id.tk, id.tj, T)});
        }
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
    REQUIRE(exec.is_complete());

    // Compare each stored C tile to the oracle.
    double max_err = 0.0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c) {
                const double got = p.values[r * T + c];
                const double want =
                    ref[(id.ti * T + r) * g.C_out + (id.tj * T + c)];
                max_err = std::max(max_err, std::abs(got - want));
            }
    }
    return max_err;
}

Conv2DGeometry base_geom() {
    Conv2DGeometry g;
    g.N = 1; g.C_in = 16; g.H_in = 8; g.W_in = 8; g.C_out = 16; g.Kh = 3; g.Kw = 3;
    return g;  // M=64, K=144, Cout=16 -> all multiples of 16
}

constexpr Size T = 16;

} // namespace

TEST_CASE("Conv2D functional: im2col+GEMM on CSP executor matches host oracle",
          "[timing][conv2d][functional]") {
    using S = Conv2DScheduleGenerator::Strategy;
    for (S strat : {S::IM2COL_INTERLEAVED, S::IM2COL_OUTPUT_STATIONARY}) {
        DYNAMIC_SECTION("strategy=" << static_cast<int>(strat)) {
            SECTION("3x3 stride1 pad1") {
                auto g = base_geom(); g.pad_h = 1; g.pad_w = 1;
                REQUIRE(run_and_compare(g, T, strat, {}, false) < 1e-3);
            }
            SECTION("1x1 pointwise, C_in=16") {
                auto g = base_geom(); g.Kh = 1; g.Kw = 1;  // K=16
                REQUIRE(run_and_compare(g, T, strat, {}, false) < 1e-3);
            }
            SECTION("3x3 stride2 pad1, C_out=32") {
                auto g = base_geom(); g.stride_h = 2; g.stride_w = 2;
                g.pad_h = 1; g.pad_w = 1; g.C_out = 32;  // M=16, Cout=32
                REQUIRE(run_and_compare(g, T, strat, {}, false) < 1e-3);
            }
            SECTION("with per-output-channel bias") {
                auto g = base_geom(); g.pad_h = 1; g.pad_w = 1;
                auto bias = fill(g.C_out, 8, -2.0f, 0.5f);
                REQUIRE(run_and_compare(g, T, strat, bias, false) < 1e-3);
            }
            SECTION("conv + bias + ReLU (fused epilogue)") {
                auto g = base_geom(); g.pad_h = 1; g.pad_w = 1;
                // filter averages 0, so pre-activation sums straddle 0; a mixed
                // bias guarantees both signs, exercising the ReLU clamp.
                auto bias = fill(g.C_out, 6, -6.0f, 2.0f);  // -6 .. 4
                REQUIRE(run_and_compare(g, T, strat, bias, true) < 1e-3);
            }
            SECTION("batch N=2") {
                auto g = base_geom(); g.N = 2; g.pad_h = 1; g.pad_w = 1;  // M=128
                REQUIRE(run_and_compare(g, T, strat, {}, false) < 1e-3);
            }
        }
    }
}
