// ============================================================================
// tests/program/test_stream_signature.cpp
// L1 stream signatures: matmul derivation (output-stationary systolic schedule),
// wavefront latency, and value-orthogonality (L1 does not change L0 results).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::stream;

namespace {
StreamSignature sig_for(const StreamProgram& sp, const TileProgram& l0,
                        TileOpKind kind, const std::string& port) {
    const auto& ops = l0.ops();
    for (const auto& [idx, s] : sp.streams)
        if (ops[idx].kind == kind && s.port == port) return s;
    FAIL("no stream signature for the requested op/port");
    return {};   // unreachable
}
} // namespace

TEST_CASE("matmul L1: A/B/C stream signatures match the output-stationary schedule",
          "[program][stream]") {
    const Dim N = 64, T = 16;   // 4x4x4 tile grid; each tile fits a 16x16 array
    TileProgram l0 = derive_matmul_tile_program(N, N, N, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0);

    CHECK(sp.array_rows == T);
    CHECK(sp.array_cols == T);

    // counts: feeds = 2 * mt*nt*kt, drains = mt*nt, computes = mt*nt*kt
    CHECK(sp.streams.size() == static_cast<std::size_t>(2 * 4 * 4 * 4 + 4 * 4));
    CHECK(sp.computes.size() == static_cast<std::size_t>(4 * 4 * 4));

    // A -> West, lane = row, skew (1,1)
    const StreamSignature A = sig_for(sp, l0, TileOpKind::Feed, "West");
    CHECK(A.edge == Edge::West);
    CHECK(A.lane_axis == LaneAxis::Row);
    CHECK(A.lanes == T);
    CHECK(A.rows == T);
    CHECK(A.cols == T);
    CHECK(A.skew_row == 1);
    CHECK(A.skew_col == 1);
    CHECK_FALSE(A.is_output);
    CHECK(A.element_count() == T * T);
    CHECK(A.time_span() == 2 * (T - 1));          // (i+k) span over the tile
    CHECK(A.lane_of(3, 5) == 3);                   // lane = row index
    CHECK(A.time_of(3, 5) == 8);                   // t = i + k

    // B -> North, lane = col
    const StreamSignature B = sig_for(sp, l0, TileOpKind::Feed, "North");
    CHECK(B.edge == Edge::North);
    CHECK(B.lane_axis == LaneAxis::Col);
    CHECK(B.lanes == T);
    CHECK(B.lane_of(3, 5) == 5);                   // lane = col index
    CHECK(B.time_of(3, 5) == 8);                   // t = j + k

    // C drain -> South (output)
    const StreamSignature C = sig_for(sp, l0, TileOpKind::Drain, "South");
    CHECK(C.edge == Edge::South);
    CHECK(C.is_output);
    CHECK(C.lane_axis == LaneAxis::Col);
}

TEST_CASE("matmul L1: wavefront latency is the systolic fill+reduce+drain", "[program][stream]") {
    const Dim T = 16;
    TileProgram l0 = derive_matmul_tile_program(64, 64, 64, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0);

    // every compute is a full 16x16x16 tile -> latency (16-1)*3 + 1 = 46
    for (const auto& [idx, w] : sp.computes) {
        (void)idx;
        CHECK(w.array_rows == T);
        CHECK(w.array_cols == T);
        CHECK(w.k_depth == T);
        CHECK(w.latency() == static_cast<Dim>(3 * (T - 1) + 1));
        // physically shaped: a wavefront costs far more than a single MAC cycle
        CHECK(w.latency() > 1);
    }
}

TEST_CASE("matmul L1: clamped trailing tiles size the wavefront correctly", "[program][stream]") {
    // N=40, T=16 -> 3x3x3 grid, trailing tile dim = 8.
    const Dim N = 40, T = 16;
    TileProgram l0 = derive_matmul_tile_program(N, N, N, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0);

    // find the compute writing the corner tile C[2,2] with the trailing A K-slice tk=2
    const auto& ops = l0.ops();
    bool saw_corner = false;
    for (const auto& [idx, w] : sp.computes) {
        const auto& out = ops[idx].outputs[0];
        const auto& ain = ops[idx].inputs[0];
        if (out.ti == 2 && out.tj == 2 && ain.tj == 2) {   // all trailing -> 8x8x8
            CHECK(w.array_rows == 8);
            CHECK(w.array_cols == 8);
            CHECK(w.k_depth == 8);
            CHECK(w.latency() == static_cast<Dim>(3 * 7 + 1));
            saw_corner = true;
        }
    }
    CHECK(saw_corner);
}

TEST_CASE("matmul L1 is value-orthogonal (derivation does not change L0 results)",
          "[program][stream]") {
    const Dim M = 6, N = 4, K = 8, Ti = 2, Tj = 2, Tk = 4;
    TileProgram l0 = derive_matmul_tile_program(M, N, K, Ti, Tj, Tk);

    std::vector<float> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
    for (Dim i = 0; i < M; ++i)
        for (Dim k = 0; k < K; ++k) A[static_cast<std::size_t>(i) * K + k] = float((i + 2 * k) % 7);
    for (Dim k = 0; k < K; ++k)
        for (Dim j = 0; j < N; ++j) B[static_cast<std::size_t>(k) * N + j] = float((3 * k + j) % 5);
    l0.operand("A").values = A;
    l0.operand("B").values = B;

    // deriving L1 must not touch the L0 program (it takes it by const&)
    StreamProgram sp = derive_matmul_streams(l0);
    CHECK(!sp.streams.empty());

    TileProgramReference ref;
    ref.run(l0);

    float max_err = 0.0f;
    const auto& C = l0.operand("C").values;
    for (Dim i = 0; i < M; ++i)
        for (Dim j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (Dim k = 0; k < K; ++k) acc += A[std::size_t(i) * K + k] * B[std::size_t(k) * N + j];
            max_err = std::max(max_err, std::fabs(acc - C[std::size_t(i) * N + j]));
        }
    CHECK(max_err == 0.0f);
}

TEST_CASE("StreamProgram disassembly lists the stream layer", "[program][stream]") {
    TileProgram l0 = derive_matmul_tile_program(32, 32, 32, 16, 16, 16);
    StreamProgram sp = derive_matmul_streams(l0);
    const std::string text = sp.disassemble(l0);
    CHECK(text.find("StreamProgram") != std::string::npos);
    CHECK(text.find("@West") != std::string::npos);
    CHECK(text.find("@North") != std::string::npos);
    CHECK(text.find("@South") != std::string::npos);
    CHECK(text.find("wavefront") != std::string::npos);
    CHECK(text.find("latency=") != std::string::npos);
}
