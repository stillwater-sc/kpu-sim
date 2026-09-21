// ============================================================================
// tests/program/test_tile_transaction_executor.cpp
// The TRANSACTIONAL tier executor (increment 3): bit-exactness against the
// functional reference, determinism, and timing that scales.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/tile_transaction_executor.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/stream/derive/matmul_streams.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace sw::kpu::program;
using characterize::DeviceDescriptor;

namespace {

// Deterministic, non-trivial fill; the same values for both tiers.
void fill_matmul(TileProgram& p, Dim K, Dim N) {
    (void)K; (void)N;
    auto& A = p.operand("A");
    auto& B = p.operand("B");
    for (std::size_t i = 0; i < A.values.size(); ++i)
        A.values[i] = float((i * 7 + 1) % 13) - 6.0f + 0.25f * float(i % 3);
    for (std::size_t i = 0; i < B.values.size(); ++i)
        B.values[i] = float((i * 5 + 2) % 11) - 5.0f - 0.125f * float(i % 5);
}

void fill_lu(TileProgram& p, Dim N) {
    auto& A = p.operand("A");
    for (Dim i = 0; i < N; ++i)
        for (Dim j = 0; j < N; ++j)
            A.at(i, j) = (i == j) ? 4.0f + float((i * 3) % 5)
                                  : 1.0f / (1.0f + std::fabs(float(int(i) - int(j))));
    if (N > 1) A.at(1, 0) = 7.0f;          // force a within-tile row swap
}

// Bit-for-bit, not "close": both tiers run the same kernels, so any difference
// is a bug rather than rounding (ADR 0001 D5).
bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

TileRunResult run_transactional(TileProgram& prog, const DeviceDescriptor& dev,
                                const Placement& pl,
                                const stream::StreamProgram* l1 = nullptr) {
    TileTransactionExecutor exec;
    TileExecutionRequest req{prog, pl, dev, l1, /*seed=*/1234};
    return exec.run(req);
}

} // namespace

TEST_CASE("GEMM is bit-exact against the functional reference", "[program][transactional]") {
    // Ragged on purpose: none of 37, 29, 23 divides the 16-tile, so every trailing
    // tile is clamped — the case a tile executor is most likely to get wrong.
    const Dim M = 37, N = 29, K = 23, T = 16;

    TileProgram ref_prog = derive_matmul_tile_program(M, N, K, T, T, T);
    TileProgram txn_prog = derive_matmul_tile_program(M, N, K, T, T, T);
    fill_matmul(ref_prog, K, N);
    fill_matmul(txn_prog, K, N);
    REQUIRE(bit_identical(ref_prog.operand("A").values, txn_prog.operand("A").values));

    TileProgramReference ref;
    const auto ref_summary = ref.run(ref_prog);

    const DeviceDescriptor dev = DeviceDescriptor::single();
    const auto result = run_transactional(txn_prog, dev, Placement::single(dev.compute_tiles));

    // Guard against a vacuous comparison: two all-zero buffers would "match".
    const auto& ref_c = ref_prog.operand("C").values;
    REQUIRE(std::any_of(ref_c.begin(), ref_c.end(), [](float v) { return v != 0.0f; }));

    CHECK(bit_identical(ref_c, txn_prog.operand("C").values));
    CHECK(result.summary.ops == ref_summary.ops);
    CHECK(result.summary.computes == ref_summary.computes);

    SECTION("with several compute tiles the ORDER changes but the values do not") {
        TileProgram par_prog = derive_matmul_tile_program(M, N, K, T, T, T);
        fill_matmul(par_prog, K, N);
        DeviceDescriptor par = DeviceDescriptor::checkerboard(4);
        const auto par_result = run_transactional(par_prog, par, Placement::single(4));
        CHECK(bit_identical(ref_prog.operand("C").values, par_prog.operand("C").values));
        // and it really did run differently
        CHECK(par_result.stats.makespan < result.stats.makespan);
    }
}

TEST_CASE("tile LU is bit-exact against the functional reference", "[program][transactional]") {
    const Dim N = 64, T = 32;

    TileProgram ref_prog = derive_lu_tile_program(N, T);
    TileProgram txn_prog = derive_lu_tile_program(N, T);
    fill_lu(ref_prog, N);
    fill_lu(txn_prog, N);

    TileProgramReference ref;
    const auto ref_summary = ref.run(ref_prog);

    const DeviceDescriptor dev = DeviceDescriptor::single();
    const auto result = run_transactional(txn_prog, dev, Placement::single(dev.compute_tiles));

    CHECK(bit_identical(ref_prog.operand("A").values, txn_prog.operand("A").values));

    SECTION("the pivot permutation and swap count match too") {
        // Non-vacuous: the fill forces a swap, so the permutation is NOT the identity
        // and row_swaps is positive — otherwise this section would compare nothing.
        REQUIRE(ref_summary.row_swaps > 0);
        bool permuted = false;
        for (Dim i = 0; i < ref_summary.permutation.size(); ++i)
            if (ref_summary.permutation[i] != i) { permuted = true; break; }
        REQUIRE(permuted);

        CHECK(result.summary.permutation == ref_summary.permutation);
        CHECK(result.summary.row_swaps == ref_summary.row_swaps);
        CHECK(result.summary.diag_factors == ref_summary.diag_factors);
        CHECK(result.summary.pivot_applies == ref_summary.pivot_applies);
    }
}

TEST_CASE("compute time is non-zero and scales with the problem", "[program][transactional]") {
    const DeviceDescriptor dev = DeviceDescriptor::single();
    const Placement pl = Placement::single(dev.compute_tiles);

    auto run_gemm = [&](Dim M, Dim N, Dim K, Dim T) {
        TileProgram p = derive_matmul_tile_program(M, N, K, T, T, T);
        fill_matmul(p, K, N);
        return run_transactional(p, dev, pl);
    };

    const auto base = run_gemm(64, 64, 64, 16);
    CHECK(base.stats.compute_cycles > 0);          // the bug this tier exists to avoid
    CHECK(base.stats.computes > 0);
    CHECK(base.stats.makespan >= base.stats.compute_cycles / dev.compute_tiles);

    SECTION("more accumulation depth costs more compute") {
        CHECK(run_gemm(64, 64, 128, 16).stats.compute_cycles > base.stats.compute_cycles);
    }
    SECTION("larger output costs more compute") {
        CHECK(run_gemm(128, 64, 64, 16).stats.compute_cycles > base.stats.compute_cycles);
    }
    SECTION("tile size changes the per-op cost, not just the op count") {
        const auto big_tiles = run_gemm(64, 64, 64, 32);
        bool found_bigger = false;
        for (const auto& r : big_tiles.timeline)
            if (r.kind == TileOpKind::MatMulAccum && (r.finish - r.start) > 0) {
                for (const auto& b : base.timeline)
                    if (b.kind == TileOpKind::MatMulAccum)
                        { found_bigger = (r.finish - r.start) > (b.finish - b.start); break; }
                break;
            }
        CHECK(found_bigger);
    }
    SECTION("makespan never beats the analytical floor") {
        CHECK(double(base.stats.makespan) >= base.stats.lower_bound - 1.0);
    }
}

TEST_CASE("runs are deterministic", "[program][transactional]") {
    const DeviceDescriptor dev = DeviceDescriptor::checkerboard(4);
    const Placement pl = Placement::single(4);

    auto once = [&]() {
        TileProgram p = derive_lu_tile_program(96, 32);
        fill_lu(p, 96);
        return run_transactional(p, dev, pl);
    };
    const auto a = once();
    const auto b = once();

    REQUIRE(a.timeline.size() == b.timeline.size());
    CHECK(a.stats.makespan == b.stats.makespan);
    CHECK(a.stats.compute_cycles == b.stats.compute_cycles);
    for (std::size_t i = 0; i < a.timeline.size(); ++i) {
        CHECK(a.timeline[i].start == b.timeline[i].start);
        CHECK(a.timeline[i].finish == b.timeline[i].finish);
        CHECK(a.timeline[i].resource_id == b.timeline[i].resource_id);
    }
}

TEST_CASE("placement decides where compute lands", "[program][transactional]") {
    TileProgram unpinned_prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    TileProgram pinned_prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    fill_matmul(unpinned_prog, 64, 64);
    fill_matmul(pinned_prog, 64, 64);

    DeviceDescriptor dev = DeviceDescriptor::checkerboard(4);

    const auto unpinned = run_transactional(unpinned_prog, dev, Placement::single(4));

    // Pin every op to compute tile 0: same values, but no compute concurrency.
    std::vector<Dim> all_on_zero(pinned_prog.ops().size(), 0);
    const Placement pinned = Placement::pinned(all_on_zero, 4);
    const auto pinned_run = run_transactional(pinned_prog, dev, pinned);

    CHECK(bit_identical(unpinned_prog.operand("C").values, pinned_prog.operand("C").values));
    CHECK(pinned_run.stats.makespan > unpinned.stats.makespan);
    for (const auto& r : pinned_run.timeline)
        if (r.resource == ResourceKind::ComputeTile) CHECK(r.resource_id == 0);

    SECTION("an out-of-range pin is refused") {
        CHECK_THROWS_AS(Placement::pinned({0, 9}, 4), std::invalid_argument);
    }
    SECTION("placement and device appear in the provenance") {
        CHECK(pinned_run.provenance.placement == "pinned/cf4");
        CHECK(unpinned.provenance.placement == "unpinned/cf4");
        CHECK(unpinned.provenance.device == dev.label());
        CHECK(unpinned.provenance.seed == 1234);
        CHECK_FALSE(unpinned.provenance.calibrated);      // increment 6
        CHECK(unpinned.provenance.extrapolated);          // >1 compute tile
    }
}

TEST_CASE("L1 stream signatures change the timing, not the values",
          "[program][transactional]") {
    const Dim M = 64, N = 64, K = 64, T = 16;
    TileProgram lumped_prog = derive_matmul_tile_program(M, N, K, T, T, T);
    TileProgram l1_prog = derive_matmul_tile_program(M, N, K, T, T, T);
    fill_matmul(lumped_prog, K, N);
    fill_matmul(l1_prog, K, N);

    const DeviceDescriptor dev = DeviceDescriptor::single();
    const Placement pl = Placement::single(dev.compute_tiles);

    const auto lumped = run_transactional(lumped_prog, dev, pl);
    const stream::StreamProgram l1 =
        stream::derive_matmul_streams(l1_prog, stream::SpaceTimeMap::output_stationary());
    const auto timed = run_transactional(l1_prog, dev, pl, &l1);

    CHECK(bit_identical(lumped_prog.operand("C").values, l1_prog.operand("C").values));
    CHECK(timed.provenance.l1_timing);
    CHECK_FALSE(lumped.provenance.l1_timing);
    CHECK(timed.stats.makespan != lumped.stats.makespan);   // systolic latencies applied
}
