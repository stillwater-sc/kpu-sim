// ============================================================================
// tests/program/test_stream_signature.cpp
// L1 stream signatures: the space-time-mapping taxonomy (which operand is
// stationary), the output-stationary result-evacuation bubble, the mesh-vs-hex
// network requirement, wavefront latency, and value-orthogonality.
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

TEST_CASE("output-stationary: A/B stream in, C evacuates North with a bubble", "[program][stream]") {
    const Dim T = 16;
    TileProgram l0 = derive_matmul_tile_program(64, 64, 64, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0, SpaceTimeMap::output_stationary());

    const StreamSignature* A = sp.signature("A");
    const StreamSignature* B = sp.signature("B");
    const StreamSignature* C = sp.signature("C");
    REQUIRE((A && B && C));

    // A -> West, streams in, dense
    CHECK(A->role == FlowRole::StreamIn);
    CHECK(A->edge == Edge::West);
    CHECK(A->dense());
    CHECK(A->lanes == T);

    // B -> North, streams in, dense
    CHECK(B->role == FlowRole::StreamIn);
    CHECK(B->edge == Edge::North);
    CHECK(B->dense());

    // C -> evacuates NORTH (not South — that collides with B and sibling C),
    // and picks up a bubble because it traverses the filled array.
    CHECK(C->role == FlowRole::StreamOut);
    CHECK(C->edge == Edge::North);
    CHECK(C->flow == std::array<int, 2>{-1, 0});   // northbound
    CHECK(C->element_stride == 2);
    CHECK(C->bubble() == 1);
    CHECK_FALSE(C->dense());

    // output-stationary fits a plain 2-D mesh
    CHECK(sp.network.required == FabricTopology::Mesh2D);
    CHECK_FALSE(sp.network.needs_overlay_on_mesh);
}

TEST_CASE("weight(B)-stationary: B is held, C exits East dense", "[program][stream]") {
    const Dim T = 16;
    TileProgram l0 = derive_matmul_tile_program(64, 64, 64, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0, SpaceTimeMap::b_stationary());

    CHECK(sp.signature("B")->role == FlowRole::Stationary);
    CHECK(sp.signature("A")->role == FlowRole::StreamIn);
    const StreamSignature* C = sp.signature("C");
    CHECK(C->role == FlowRole::StreamOut);
    CHECK(C->edge == Edge::East);
    CHECK(C->dense());                       // accumulate-and-exit, no traversal bubble
    CHECK(C->bubble() == 0);
    CHECK(sp.network.required == FabricTopology::Mesh2D);
}

TEST_CASE("A-stationary: A is held, mesh network", "[program][stream]") {
    TileProgram l0 = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    StreamProgram sp = derive_matmul_streams(l0, SpaceTimeMap::a_stationary());
    CHECK(sp.signature("A")->role == FlowRole::Stationary);
    CHECK(sp.signature("B")->role == FlowRole::StreamIn);
    CHECK(sp.signature("C")->role == FlowRole::StreamOut);
    CHECK(sp.network.required == FabricTopology::Mesh2D);
}

TEST_CASE("fully-streaming: hexagonal network, all operands stream, no bubbles",
          "[program][stream]") {
    TileProgram l0 = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    SpaceTimeMap hex = SpaceTimeMap::fully_streaming();
    StreamProgram sp = derive_matmul_streams(l0, hex);

    // τ ∥ proj = [1,1,1]  -> aligned, contention-free
    CHECK(hex.aligned());

    // no operand is stationary
    CHECK(sp.signature("A")->role != FlowRole::Stationary);
    CHECK(sp.signature("B")->role != FlowRole::Stationary);
    CHECK(sp.signature("C")->role == FlowRole::StreamOut);

    // aligned schedule -> every stream is dense (no bubbles)
    CHECK(sp.signature("A")->dense());
    CHECK(sp.signature("B")->dense());
    CHECK(sp.signature("C")->dense());

    // three stream directions -> hexagonal, needing an overlay on a 2-D mesh
    CHECK(sp.network.required == FabricTopology::Hexagonal);
    CHECK(sp.network.needs_overlay_on_mesh);
    CHECK(sp.network.stream_directions.size() == 3u);
}

TEST_CASE("wavefront latency is the systolic fill+reduce+drain (incl. clamped tiles)",
          "[program][stream]") {
    const Dim T = 16;
    // full 16^3 tiles
    StreamProgram full = derive_matmul_streams(derive_matmul_tile_program(64, 64, 64, T, T, T));
    for (const auto& [idx, w] : full.computes) {
        (void)idx;
        CHECK(w.latency() == static_cast<Dim>(3 * (T - 1) + 1));   // 46
        CHECK(w.latency() > 1);                                    // physically shaped, not lumped
    }

    // N=40 -> trailing 8x8x8 corner tile
    TileProgram l0 = derive_matmul_tile_program(40, 40, 40, T, T, T);
    StreamProgram sp = derive_matmul_streams(l0);
    const auto& ops = l0.ops();
    bool saw_corner = false;
    for (const auto& [idx, w] : sp.computes) {
        const auto& out = ops[idx].outputs[0];
        const auto& ain = ops[idx].inputs[0];
        if (out.ti == 2 && out.tj == 2 && ain.tj == 2) {
            CHECK(w.array_rows == 8);
            CHECK(w.k_depth == 8);
            CHECK(w.latency() == static_cast<Dim>(3 * 7 + 1));
            saw_corner = true;
        }
    }
    CHECK(saw_corner);
}

TEST_CASE("L1 is value-orthogonal (derivation does not change L0 results)",
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

    // deriving any mapping's L1 must not touch the L0 program (const&)
    StreamProgram sp = derive_matmul_streams(l0, SpaceTimeMap::fully_streaming());
    CHECK(!sp.signatures.empty());

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

TEST_CASE("StreamProgram disassembly names mapping, roles, bubbles, and network",
          "[program][stream]") {
    TileProgram l0 = derive_matmul_tile_program(32, 32, 32, 16, 16, 16);
    const std::string os = derive_matmul_streams(l0, SpaceTimeMap::output_stationary()).disassemble();
    CHECK(os.find("output-stationary") != std::string::npos);
    CHECK(os.find("network=Mesh2D") != std::string::npos);
    CHECK(os.find("bubble=1") != std::string::npos);
    CHECK(os.find("wavefront") != std::string::npos);

    const std::string hx = derive_matmul_streams(l0, SpaceTimeMap::fully_streaming()).disassemble();
    CHECK(hx.find("Hexagonal") != std::string::npos);
    CHECK(hx.find("overlay-on-mesh") != std::string::npos);
}
