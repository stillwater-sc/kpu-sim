// ============================================================================
// tests/program/test_tile_program.cpp
// L0 TileProgram + functional reference: matmul (exact) and explicit tile-LU
// ({GETRF,LASWP,TRSM,GEMM} kernels; P.A = L.U reconstruction). Validates that the
// device-independent tile sequence alone computes the correct result (issue #230).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>

#include <cmath>
#include <vector>

using namespace sw::kpu::program;

namespace {

// Ground-truth naive matmul, C = A . B (row-major).
std::vector<float> naive_matmul(const std::vector<float>& A, const std::vector<float>& B,
                                Dim M, Dim N, Dim K) {
    std::vector<float> C(static_cast<std::size_t>(M) * N, 0.0f);
    for (Dim i = 0; i < M; ++i)
        for (Dim j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (Dim k = 0; k < K; ++k)
                acc += A[static_cast<std::size_t>(i) * K + k] * B[static_cast<std::size_t>(k) * N + j];
            C[static_cast<std::size_t>(i) * N + j] = acc;
        }
    return C;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Fill an operand's value buffer (row-major) from a flat vector.
void fill(TensorOperand& op, const std::vector<float>& v) {
    REQUIRE(v.size() == op.values.size());
    op.values = v;
}

} // namespace

// ----------------------------------------------------------------------------
TEST_CASE("TileProgram matmul reproduces naive matmul exactly", "[program][matmul]") {
    const Dim M = 6, N = 4, K = 8;
    const Dim Ti = 2, Tj = 2, Tk = 4;   // evenly divisible

    // Small integer-valued inputs so the sum-of-products is exact in float.
    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    for (Dim i = 0; i < M; ++i)
        for (Dim k = 0; k < K; ++k) A[static_cast<std::size_t>(i) * K + k] = float((i + 2 * k) % 7);
    for (Dim k = 0; k < K; ++k)
        for (Dim j = 0; j < N; ++j) B[static_cast<std::size_t>(k) * N + j] = float((3 * k + j) % 5);

    TileProgram prog = derive_matmul_tile_program(M, N, K, Ti, Tj, Tk);
    fill(prog.operand("A"), A);
    fill(prog.operand("B"), B);

    TileProgramReference ref;
    auto sum = ref.run(prog);

    // Structural checks: the outer-loop tile sequence has the expected shape.
    const Dim mt = 3, nt = 2, kt = 2;
    CHECK(sum.feeds == static_cast<std::size_t>(2) * mt * nt * kt);
    CHECK(sum.computes == static_cast<std::size_t>(mt) * nt * kt);
    CHECK(sum.drains == static_cast<std::size_t>(mt) * nt);
    CHECK(prog.count(TileOpKind::MatMulAccum) == sum.computes);

    CHECK(max_abs_diff(prog.operand("C").values, naive_matmul(A, B, M, N, K)) == 0.0f);
}

// ----------------------------------------------------------------------------
TEST_CASE("TileProgram matmul handles non-divisible tiling", "[program][matmul]") {
    const Dim M = 7, N = 5, K = 6;
    const Dim Ti = 3, Tj = 2, Tk = 4;   // none divide evenly -> trailing tiles clamp

    std::vector<float> A(static_cast<std::size_t>(M) * K);
    std::vector<float> B(static_cast<std::size_t>(K) * N);
    for (std::size_t i = 0; i < A.size(); ++i) A[i] = std::sin(0.7f * float(i)) * 2.0f;
    for (std::size_t i = 0; i < B.size(); ++i) B[i] = std::cos(0.4f * float(i)) * 1.5f;

    TileProgram prog = derive_matmul_tile_program(M, N, K, Ti, Tj, Tk);
    fill(prog.operand("A"), A);
    fill(prog.operand("B"), B);

    TileProgramReference ref;
    ref.run(prog);

    CHECK(max_abs_diff(prog.operand("C").values, naive_matmul(A, B, M, N, K)) < 1e-4f);
}

// ----------------------------------------------------------------------------
// LU: run the tile program, extract L (unit lower) and U (upper) in place, and
// verify the reconstruction P.A0 = L.U where P is the reported pivot permutation.
// This validates the explicit cross-tile dependencies (down-column TRSM, per-tile
// trailing GEMM), the within-tile pivoting, and the LASWP propagation together.
namespace {

void check_lu_reconstructs(const std::vector<float>& A0, Dim N, Dim T,
                           bool expect_swaps) {
    TileProgram prog = derive_lu_tile_program(N, T);
    fill(prog.operand("A"), A0);

    TileProgramReference ref;
    auto sum = ref.run(prog);

    REQUIRE(sum.permutation.size() == N);
    if (expect_swaps) CHECK(sum.row_swaps >= 1);

    const auto& F = prog.operand("A").values;   // in-place factored L\U
    auto idx = [N](Dim i, Dim j) { return static_cast<std::size_t>(i) * N + j; };

    // L.U reconstruction.
    std::vector<float> LU(static_cast<std::size_t>(N) * N, 0.0f);
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (Dim k = 0; k < N; ++k) {
                float l = (i > k) ? F[idx(i, k)] : (i == k ? 1.0f : 0.0f);  // unit lower
                float u = (k <= j) ? F[idx(k, j)] : 0.0f;                    // upper
                acc += l * u;
            }
            LU[idx(i, j)] = acc;
        }

    // P.A0 : row i of the permuted matrix is original row perm[i].
    std::vector<float> PA(static_cast<std::size_t>(N) * N, 0.0f);
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j)
            PA[idx(i, j)] = A0[idx(sum.permutation[i], j)];

    CHECK(max_abs_diff(LU, PA) < 1e-4f);
}

} // namespace

TEST_CASE("TileProgram tile-LU factors a blocked matrix (swaps exercised)",
          "[program][lu]") {
    // Classic well-conditioned 4x4; the large sub-diagonal in the diagonal tile's
    // column 0 forces a within-tile pivot. T=2 -> 2x2 blocks so GETRF / LASWP /
    // TRSM / GEMM trailing update all run across the tile grid.
    const Dim N = 4, T = 2;
    std::vector<float> A0 = {
        2, 1, 1, 0,
        4, 3, 3, 1,
        8, 7, 9, 5,
        6, 7, 9, 8,
    };
    check_lu_reconstructs(A0, N, T, /*expect_swaps=*/true);
}

TEST_CASE("TileProgram tile-LU factors a single-tile matrix", "[program][lu]") {
    // T == N -> one diagonal tile, no trailing tiles: exercises GETRF on its own.
    const Dim N = 4, T = 4;
    std::vector<float> A0 = {
        2, 1, 1, 0,
        4, 3, 3, 1,
        8, 7, 9, 5,
        6, 7, 9, 8,
    };
    check_lu_reconstructs(A0, N, T, /*expect_swaps=*/true);
}

TEST_CASE("TileProgram tile-LU factors a 3x3 tile grid", "[program][lu]") {
    // N=6, T=2 -> 3x3 tile grid: multiple trailing-update tiles and left-column
    // LASWP ops. Diagonally strengthened so within-tile pivots stay nonzero; a
    // spike in each diagonal tile's column triggers within-tile swaps.
    const Dim N = 6, T = 2;
    std::vector<float> A0(static_cast<std::size_t>(N) * N);
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j)
            A0[static_cast<std::size_t>(i) * N + j] =
                (i == j) ? 5.0f + float(i)
                         : 1.0f / (1.0f + std::fabs(float(int(i) - int(j))));
    // spike the sub-diagonal inside each 2x2 diagonal tile to force within-tile swaps
    A0[static_cast<std::size_t>(1) * N + 0] = 9.0f;   // diag tile (0,0)
    A0[static_cast<std::size_t>(3) * N + 2] = 11.0f;  // diag tile (1,1)
    A0[static_cast<std::size_t>(5) * N + 4] = 13.0f;  // diag tile (2,2)
    check_lu_reconstructs(A0, N, T, /*expect_swaps=*/true);
}

TEST_CASE("TileProgram tile-LU handles a clamped trailing block", "[program][lu]") {
    // N=6, T=4 -> 2x2 tile grid with a clamped 2x2 trailing block (rows/cols 4..5).
    const Dim N = 6, T = 4;
    std::vector<float> A0(static_cast<std::size_t>(N) * N);
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j)
            A0[static_cast<std::size_t>(i) * N + j] =
                (i == j) ? 5.0f + float(i)
                         : 1.0f / (1.0f + std::fabs(float(int(i) - int(j))));
    A0[static_cast<std::size_t>(1) * N + 0] = 9.0f;
    check_lu_reconstructs(A0, N, T, /*expect_swaps=*/true);
}

// ----------------------------------------------------------------------------
TEST_CASE("TileProgram disassembly lists the tile sequence", "[program][disasm]") {
    TileProgram prog = derive_matmul_tile_program(4, 4, 4, 2, 2, 2);
    const std::string text = prog.disassemble();
    CHECK(text.find("TileProgram") != std::string::npos);
    CHECK(text.find("MATMUL_ACCUM") != std::string::npos);
    CHECK(text.find("FEED") != std::string::npos);
    CHECK(text.find("DRAIN") != std::string::npos);

    TileProgram lu = derive_lu_tile_program(6, 2);
    const std::string lut = lu.disassemble();
    CHECK(lut.find("LU_DIAG_FACTOR") != std::string::npos);
    CHECK(lut.find("PIVOT_APPLY") != std::string::npos);
    CHECK(lut.find("TRSM_LOWER_LEFT") != std::string::npos);
    CHECK(lut.find("TRSM_UPPER_RIGHT") != std::string::npos);
    CHECK(lut.find("pivot#") != std::string::npos);
}
