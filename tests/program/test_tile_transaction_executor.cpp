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
#include <sw/kpu/program/characterize/characterization.hpp>

#include <algorithm>
#include <chrono>
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
    // Provision the movement processes generously, so COMPUTE is the bottleneck and
    // placement is what the makespan is measuring. With one lane per process the real
    // three-hop chain dominates, and pinning every op to one compute tile costs nothing
    // measurable — the assertion below would pass or fail for reasons unrelated to
    // placement.
    dev.dma_engines = 8;   dev.dma_bytes_per_cycle = 1024.0;
    dev.block_movers = 8;  dev.bm_bytes_per_cycle = 1024.0;
    dev.streamers = 8;     dev.str_bytes_per_cycle = 1024.0;

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

TEST_CASE("one executor can run repeatedly without leaking state",
          "[program][transactional]") {
    // The kernels carry pivot slots and a row permutation between ops, so an
    // executor that does not reset them produces WRONG values on its second LU run.
    // Every other test here uses a fresh executor, which is exactly why this case
    // needs its own: the leak is invisible unless an instance is reused.
    const Dim N = 64, T = 32;
    const DeviceDescriptor dev = DeviceDescriptor::single();
    const Placement pl = Placement::single(dev.compute_tiles);

    TileProgram fresh_prog = derive_lu_tile_program(N, T);
    fill_lu(fresh_prog, N);
    TileTransactionExecutor fresh;
    TileExecutionRequest fresh_req{fresh_prog, pl, dev, nullptr, 7};
    const auto expected = fresh.run(fresh_req);

    TileTransactionExecutor reused;
    for (int pass = 0; pass < 3; ++pass) {
        TileProgram prog = derive_lu_tile_program(N, T);
        fill_lu(prog, N);
        TileExecutionRequest req{prog, pl, dev, nullptr, 7};
        const auto got = reused.run(req);

        CHECK(bit_identical(fresh_prog.operand("A").values, prog.operand("A").values));
        CHECK(got.summary.permutation == expected.summary.permutation);
        CHECK(got.summary.row_swaps == expected.summary.row_swaps);
        CHECK(got.stats.makespan == expected.stats.makespan);
    }
}

TEST_CASE("the ready set scales to large programs", "[program][transactional][scale]") {
    // Many movement ops sit ready while one lane retires them one at a time. A sorted
    // ready vector rescans and re-sorts per completion, which is O(n^2 log n): this
    // program took 116 s that way versus 0.17 s with index-keyed heaps. The budget is a
    // regression guard with ~170x headroom, not a benchmark.
    const Dim M = 128, T = 4;
    TileProgram prog = derive_matmul_tile_program(M, M, M, T, T, T);
    fill_matmul(prog, M, M);
    REQUIRE(prog.ops().size() > 50000);

    const DeviceDescriptor dev = DeviceDescriptor::single();
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = run_transactional(prog, dev, Placement::single(dev.compute_tiles));
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    CHECK(result.stats.ops == prog.ops().size());
    CHECK(result.stats.compute_cycles > 0);
    CHECK(seconds < 30.0);
}

// ============================================================================
// Increment 4 — L3 capacity, credits, residency reuse, refusal
// ============================================================================

TEST_CASE("capacity at the program's peak live set completes; one below refuses",
          "[program][transactional][capacity]") {
    // The harness computes peak_live_tiles statically; the executor enforces capacity
    // dynamically. They are independent implementations of the same question, so their
    // feasibility boundary must coincide EXACTLY — if it does not, one of them is wrong.
    const Dim M = 64, N = 64, K = 64, T = 16;
    const std::size_t peak =
        characterize::peak_live_tiles(derive_matmul_tile_program(M, N, K, T, T, T));
    REQUIRE(peak > 1);

    auto run_with = [&](Dim l3_tiles) {
        TileProgram p = derive_matmul_tile_program(M, N, K, T, T, T);
        fill_matmul(p, K, N);
        DeviceDescriptor dev = DeviceDescriptor::single();
        dev.l3_tiles = l3_tiles;
        return run_transactional(p, dev, Placement::single(dev.compute_tiles));
    };

    SECTION("exactly the static peak is enough") {
        const auto at_peak = run_with(static_cast<Dim>(peak));
        CHECK(at_peak.stats.peak_l3_residency <= peak);
        CHECK(at_peak.stats.l3_credit_stalls > 0);      // it was genuinely tight
    }

    SECTION("one tile short is refused, with a diagnosis naming the budget") {
        try {
            run_with(static_cast<Dim>(peak - 1));
            FAIL("expected a refusal at one tile below the peak live set");
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            CHECK(msg.find("no op can fire") != std::string::npos);
            CHECK(msg.find("L3 slot") != std::string::npos);      // says WHY, not just that
            CHECK(msg.find("live set") != std::string::npos);     // and what to do about it
        }
    }
}

TEST_CASE("capacity pressure throttles rather than corrupting",
          "[program][transactional][capacity]") {
    const Dim M = 64, N = 64, K = 64, T = 16;
    const std::size_t peak =
        characterize::peak_live_tiles(derive_matmul_tile_program(M, N, K, T, T, T));

    TileProgram ref_prog = derive_matmul_tile_program(M, N, K, T, T, T);
    fill_matmul(ref_prog, K, N);
    TileProgramReference ref;
    ref.run(ref_prog);

    auto run_with = [&](Dim l3_tiles) {
        TileProgram p = derive_matmul_tile_program(M, N, K, T, T, T);
        fill_matmul(p, K, N);
        DeviceDescriptor dev = DeviceDescriptor::single();
        dev.l3_tiles = l3_tiles;
        auto r = run_transactional(p, dev, Placement::single(dev.compute_tiles));
        return std::pair{r, std::move(p)};
    };

    auto [unbounded, up] = run_with(0);
    auto [tight, tp]     = run_with(static_cast<Dim>(peak));

    SECTION("values are unaffected by how tight the buffer is") {
        CHECK(bit_identical(ref_prog.operand("C").values, up.operand("C").values));
        CHECK(bit_identical(ref_prog.operand("C").values, tp.operand("C").values));
    }
    SECTION("a tighter buffer costs time and reports the stalls that caused it") {
        CHECK(tight.stats.makespan > unbounded.stats.makespan);
        CHECK(tight.stats.l3_credit_stalls > 0);
        CHECK(unbounded.stats.l3_credit_stalls == 0);
    }
    SECTION("residency is bounded by the budget, and unbounded runs exceed it") {
        CHECK(tight.stats.peak_l3_residency <= peak);
        CHECK(unbounded.stats.peak_l3_residency > peak);   // the natural live set is larger
    }
}

TEST_CASE("a tile already resident re-enters the chain; it is not re-fed for free",
          "[program][transactional][capacity]") {
    // Tiled GEMM re-feeds one B[tk,tj] across a column of output tiles, and this tier
    // exists partly to reward that reuse. But the reward is SKIPPING THE DMA LEG, not
    // skipping the chain: L2 and L1 are separate memories reached by separate processes,
    // so the tile still crosses the BlockMover and the Streamer to reach a stream buffer.
    //
    // An earlier version of this test asserted such a feed was free and zero-work. That
    // followed from the collapsed-hop model, which described a machine that cannot be
    // built.
    TileProgram p = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    fill_matmul(p, 64, 64);
    const DeviceDescriptor dev = DeviceDescriptor::single();
    const auto r = run_transactional(p, dev, Placement::single(dev.compute_tiles));

    REQUIRE(r.stats.resident_feeds > 0);

    std::size_t reuse = 0;
    Cycle reuse_span = 0, fresh_span = 0;
    for (const auto& rec : r.timeline) {
        if (rec.kind != TileOpKind::Feed || rec.hops.empty()) continue;
        if (rec.hops.front().hop == Hop::BlockMoverL3ToL2) {
            ++reuse;
            CHECK(rec.hops.size() == 2);                   // BlockMover + Streamer
            CHECK(rec.finish > rec.start);                 // cheaper, NOT free
            if (reuse_span == 0) reuse_span = rec.finish - rec.start;
        } else {
            CHECK(rec.hops.front().hop == Hop::DmaDramToL3);
            CHECK(rec.hops.size() == 3);
            if (fresh_span == 0) fresh_span = rec.finish - rec.start;
        }
    }
    CHECK(reuse == r.stats.resident_feeds);
    // The saving is exactly one leg's worth of work.
    REQUIRE(fresh_span > 0);
    REQUIRE(reuse_span > 0);
    CHECK(reuse_span < fresh_span);
}

TEST_CASE("runs stay deterministic under capacity pressure",
          "[program][transactional][capacity]") {
    const std::size_t peak =
        characterize::peak_live_tiles(derive_lu_tile_program(96, 32));
    auto once = [&]() {
        TileProgram p = derive_lu_tile_program(96, 32);
        fill_lu(p, 96);
        DeviceDescriptor dev = DeviceDescriptor::checkerboard(4);
        dev.l3_tiles = static_cast<Dim>(peak);
        return run_transactional(p, dev, Placement::single(4));
    };
    const auto a = once();
    const auto b = once();
    CHECK(a.stats.makespan == b.stats.makespan);
    CHECK(a.stats.l3_credit_stalls == b.stats.l3_credit_stalls);
    CHECK(a.stats.peak_l3_residency == b.stats.peak_l3_residency);
    REQUIRE(a.timeline.size() == b.timeline.size());
    for (std::size_t i = 0; i < a.timeline.size(); ++i)
        CHECK(a.timeline[i].start == b.timeline[i].start);
}

// ----------------------------------------------------------------------------
// Regression: a tile is released when its LAST USER COMPLETES, not at the
// highest-indexed user.
//
// `TileDependencies` orders writer->reader and deliberately leaves reader->reader
// unordered, so two readers of one tile are concurrent and the higher-INDEXED one can
// finish FIRST. Releasing the tile on it frees a slot an earlier reader still holds,
// which under-counts residency and lets a run fit in an L3 budget that cannot actually
// hold its live set.
//
// Getting this observable took some care, and the shape of the program is the argument:
//
//   - the short reader must be UNGATED, so it can start early under capacity pressure.
//     It accumulates into a tile that was fed explicitly, so it needs no new slot and
//     program-order slot acquisition does not hold it behind the long reader. Every
//     earlier attempt failed here: capacity pressure serialised the two readers and
//     destroyed the out-of-order completion the bug needs.
//   - the claimant must become ready exactly when the short reader completes, so it is
//     the op that would consume a prematurely freed slot.
//
// This case was VERIFIED TO DISCRIMINATE: built against the old highest-index release,
// `l3_tiles = 4` completes with `peak_l3_residency` reported as 4 while five tiles are
// really live. With release-on-last-completion it is correctly refused.
// ----------------------------------------------------------------------------
namespace {

TileProgram build_shared_reader_program() {
    TileProgram p("shared-reader");
    p.add_operand(TensorOperand("A", 16, 16, 16, 16));
    p.add_operand(TensorOperand("B", 16, 24, 16, 16));   // B[0,1] is 16x8 -> less work
    p.add_operand(TensorOperand("C", 16, 24, 16, 16));   // C[0,1] is 16x8
    p.add_operand(TensorOperand("H",  8, 16,  8, 16));
    p.add_operand(TensorOperand("G", 16, 16, 16, 16));
    auto feed = [&](const char* o, Dim i, Dim j) {
        TileOp f; f.kind = TileOpKind::Feed; f.port_kind = PortKind::Input; f.port = "West";
        f.inputs = {TileCoord{o, i, j}}; p.push(std::move(f));
    };
    auto drain = [&](const char* o, Dim i, Dim j) {
        TileOp d; d.kind = TileOpKind::Drain; d.port_kind = PortKind::Output; d.port = "South";
        d.outputs = {TileCoord{o, i, j}}; p.push(std::move(d));
    };
    auto mac = [&](TileCoord i0, TileCoord i1, TileCoord o) {
        TileOp m; m.kind = TileOpKind::MatMulAccum; m.inputs = {i0, i1}; m.outputs = {o};
        p.push(std::move(m));
    };
    feed("A", 0, 0);                              // 0
    feed("B", 0, 1);                              // 1
    feed("C", 0, 1);                              // 2  so op6 needs no new slot
    feed("H", 0, 0);                              // 3
    feed("B", 0, 0);                              // 4
    mac({"A",0,0}, {"B",0,0}, {"C",0,0});         // 5  LONG  reader of A[0,0]
    mac({"A",0,0}, {"B",0,1}, {"C",0,1});         // 6  SHORT reader of A[0,0], ungated
    mac({"C",0,1}, {"H",0,0}, {"G",0,0});         // 7  claimant of the freed slot
    drain("C", 0, 0); drain("C", 0, 1); drain("G", 0, 0);   // 8, 9, 10
    return p;
}

TileRunResult run_shared_reader(Dim l3_tiles) {
    TileProgram p = build_shared_reader_program();
    auto& A = p.operand("A");
    for (std::size_t i = 0; i < A.values.size(); ++i) A.values[i] = float(i % 7) - 3.0f;
    auto& B = p.operand("B");
    for (std::size_t i = 0; i < B.values.size(); ++i) B.values[i] = float(i % 5) - 2.0f;
    DeviceDescriptor dev = DeviceDescriptor::single();
    dev.compute_tiles = 2;                        // the two readers must be able to overlap
    dev.l3_tiles = l3_tiles;
    return run_transactional(p, dev, Placement::single(2));
}

} // namespace

TEST_CASE("a shared tile is held until its last reader completes, not its last index",
          "[program][transactional][capacity]") {
    SECTION("the premise: the higher-indexed reader really does finish first") {
        const auto r = run_shared_reader(0);          // unbounded: no capacity interference
        CHECK(r.timeline[6].finish < r.timeline[5].finish);
        CHECK(r.timeline[6].start < r.timeline[5].start);
    }

    SECTION("five tiles are genuinely live, so five is enough and four is refused") {
        const auto ok = run_shared_reader(5);
        CHECK(ok.stats.peak_l3_residency == 5);
        CHECK(ok.timeline[6].finish < ok.timeline[5].finish);   // still out of order

        // The regression. Releasing A[0,0] at op6 (its highest-indexed user) while op5
        // still reads it makes this budget look sufficient: the old logic completed here
        // and reported a peak of 4.
        REQUIRE_THROWS_AS(run_shared_reader(4), std::runtime_error);
    }
}

// ----------------------------------------------------------------------------
// Regression: the two scheduling rules that make credits gate WITHOUT idling
// resources. Both of these were shipped without a test and both were caught in
// review; each fixture below was verified to FAIL against the implementation that
// preceded the fix, which is the only thing that makes it a regression test.
// ----------------------------------------------------------------------------
namespace {

// One firing hands the seeker to a movement op whose queue this pass already walked
// past. Movement is scanned before compute, so without repeating the pass the lane
// waits for an unrelated completion.
TileProgram build_same_pass_program() {
    TileProgram p("same-pass");
    p.add_operand(TensorOperand("A", 16, 16, 16, 16));
    p.add_operand(TensorOperand("B", 16, 16, 16, 16));
    p.add_operand(TensorOperand("C", 16, 16, 16, 16));
    p.add_operand(TensorOperand("D", 16, 16, 16, 16));
    auto feed = [&](const char* o) {
        TileOp f; f.kind = TileOpKind::Feed; f.port_kind = PortKind::Input; f.port = "West";
        f.inputs = {TileCoord{o, 0, 0}}; p.push(std::move(f));
    };
    auto drain = [&](const char* o) {
        TileOp d; d.kind = TileOpKind::Drain; d.port_kind = PortKind::Output; d.port = "South";
        d.outputs = {TileCoord{o, 0, 0}}; p.push(std::move(d));
    };
    feed("A");                                          // 0
    feed("B");                                          // 1
    TileOp m; m.kind = TileOpKind::MatMulAccum;         // 2  the seeker; fires this pass
    m.inputs = {TileCoord{"A",0,0}, TileCoord{"B",0,0}}; m.outputs = {TileCoord{"C",0,0}};
    p.push(std::move(m));
    feed("D");                                          // 3  the seeker advances to here
    drain("C"); drain("D");                             // 4, 5
    return p;
}

// n feeds, each needing a slot, ahead of the drains that return the credits. At a small
// budget the queued blocked prefix is far longer than the 32-entry scan window, so the
// op that would release a slot is only reachable by the exhaustive pass.
TileProgram build_deep_queue_program(Dim n) {
    TileProgram p("deep-queue");
    p.add_operand(TensorOperand("A", 16 * n, 16, 16, 16));      // n tiles, A[i,0]
    for (Dim i = 0; i < n; ++i) {
        TileOp f; f.kind = TileOpKind::Feed; f.port_kind = PortKind::Input; f.port = "West";
        f.inputs = {TileCoord{"A", i, 0}}; p.push(std::move(f));
    }
    for (Dim i = 0; i < n; ++i) {
        TileOp d; d.kind = TileOpKind::Drain; d.port_kind = PortKind::Output; d.port = "South";
        d.outputs = {TileCoord{"A", i, 0}}; p.push(std::move(d));
    }
    return p;
}

} // namespace

TEST_CASE("work unblocked by a firing starts in the same cycle, not after the next completion",
          "[program][transactional][capacity]") {
    TileProgram p = build_same_pass_program();
    DeviceDescriptor dev = DeviceDescriptor::single();
    dev.l3_tiles = 4;                       // finite, so the program-order seeker binds
    const auto r = run_transactional(p, dev, Placement::single(dev.compute_tiles));

    // op2 (the mac) is the seeker and fires; the seeker then advances to op3 (feed D),
    // whose queue was scanned BEFORE op2's. Both must start in the same cycle.
    CHECK(r.timeline[3].start == r.timeline[2].start);
    CHECK(r.timeline[3].start < r.timeline[2].finish);   // i.e. no wait for a completion

    // Against a single-pass try_fire, feed D starts at op2's FINISH rather than its start,
    // which pushes the whole run out. Asserted relationally: the exact makespan moved when
    // hops stopped collapsing, and a number that has to be re-derived on every model change
    // tests the model less than the relation does.
    CHECK(r.timeline[3].finish <= r.stats.makespan);
}

TEST_CASE("a feasible op past the scan window is still found, not refused",
          "[program][transactional][capacity]") {
    const Dim n = 40, budget = 5;
    // The blocked prefix must exceed the 32-entry window for this to test anything.
    STATIC_REQUIRE(n - budget > 32);

    TileProgram p = build_deep_queue_program(n);
    DeviceDescriptor dev = DeviceDescriptor::single();
    dev.l3_tiles = budget;
    const auto r = run_transactional(p, dev, Placement::single(dev.compute_tiles));

    CHECK(r.stats.ops == std::size_t(2 * n));
    CHECK(r.stats.peak_l3_residency == budget);         // the budget was respected
    CHECK(r.timeline.back().finish == r.stats.makespan);

    // Without the exhaustive pass that runs before declaring a wedge, the drain that
    // would return a credit sits 35 entries deep and this run is REFUSED outright.
}

// ============================================================================
// Increment 5 — movement as a chain of CSP processes (design note §6)
//
// The first version of this increment modelled a collapsible chain, with a
// single-pool "collapsed" mode selected by the descriptor. That was wrong: the
// physical pathways for a shortcut do not exist. DMA (DRAM<->L3), BlockMover
// (L3<->L2) and Streamer (L2<->L1) are distinct processes, so a span always
// contains all of its hops. These tests pin that, not the old flexibility.
// ============================================================================
namespace {

TileRunResult run_gemm_on(const DeviceDescriptor& dev) {
    TileProgram p = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    fill_matmul(p, 64, 64);
    return run_transactional(p, dev, Placement::single(dev.compute_tiles));
}

DeviceDescriptor dma_starved(Dim engines) {
    DeviceDescriptor d = DeviceDescriptor::single();
    d.dma_engines = engines;
    d.dma_bytes_per_cycle = 16.0;      // the scarce process
    d.block_movers = 4;
    d.bm_bytes_per_cycle = 256.0;
    d.streamers = 4;
    d.str_bytes_per_cycle = 256.0;
    return d;
}

std::size_t count_hop(const TileRunResult& r, Hop h) {
    std::size_t n = 0;
    for (const auto& rec : r.timeline)
        for (const auto& hr : rec.hops)
            if (hr.hop == h) ++n;
    return n;
}

} // namespace

TEST_CASE("a span contains all of its hops; there is no collapse",
          "[program][transactional][hops]") {
    const auto r = run_gemm_on(DeviceDescriptor::single());

    // Every inbound tile that actually comes from DRAM crosses all three legs, in order.
    // Every outbound tile crosses all three of the reverse legs.
    std::size_t inbound_full = 0, outbound_full = 0, resident_entry = 0;
    for (const auto& rec : r.timeline) {
        if (rec.kind == TileOpKind::MatMulAccum) { CHECK(rec.hops.empty()); continue; }
        if (rec.kind == TileOpKind::Drain) {
            REQUIRE(rec.hops.size() == 3);
            CHECK(rec.hops[0].hop == Hop::StreamerL1ToL2);
            CHECK(rec.hops[1].hop == Hop::BlockMoverL2ToL3);
            CHECK(rec.hops[2].hop == Hop::DmaL3ToDram);
            ++outbound_full;
            continue;
        }
        if (rec.hops.size() == 3) {
            CHECK(rec.hops[0].hop == Hop::DmaDramToL3);
            CHECK(rec.hops[1].hop == Hop::BlockMoverL3ToL2);
            CHECK(rec.hops[2].hop == Hop::StreamerL2ToL1);
            ++inbound_full;
        } else {
            // Reuse: the tile is already in L3, so the chain STARTS at the BlockMover.
            // It does not skip an interior hop — L2 and L1 are separate memories reached
            // by separate processes.
            REQUIRE(rec.hops.size() == 2);
            CHECK(rec.hops[0].hop == Hop::BlockMoverL3ToL2);
            CHECK(rec.hops[1].hop == Hop::StreamerL2ToL1);
            ++resident_entry;
        }
    }
    CHECK(inbound_full > 0);
    CHECK(outbound_full > 0);
    CHECK(resident_entry == r.stats.resident_feeds);   // reuse re-enters, never shortcuts

    // No leg is ever absent from the middle of a chain.
    CHECK(count_hop(r, Hop::StreamerL2ToL1) == inbound_full + resident_entry);
    CHECK(count_hop(r, Hop::BlockMoverL3ToL2) == inbound_full + resident_entry);
    CHECK(count_hop(r, Hop::DmaDramToL3) == inbound_full);
}

TEST_CASE("movement stops at L1, because that is all the fabric reads",
          "[program][transactional][hops]") {
    const auto r = run_gemm_on(dma_starved(2));
    for (const auto& rec : r.timeline) {
        // The L1 buffers push elements into the fabric; that is not a mover, so a compute
        // op carries no transfer at all.
        if (rec.kind == TileOpKind::MatMulAccum) { CHECK(rec.hops.empty()); continue; }
        if (rec.hops.empty()) continue;
        if (rec.kind == TileOpKind::Drain) {
            // Results leave FROM L1: the outbound chain starts at the Streamer.
            CHECK(rec.hops.front().hop == Hop::StreamerL1ToL2);
        } else {
            // Operands arrive AT L1 and stop: the inbound chain's last leg is the
            // Streamer writing the stream buffer, never a leg into the fabric.
            CHECK(rec.hops.back().hop == Hop::StreamerL2ToL1);
        }
    }
}

TEST_CASE("each movement process is its own bottleneck candidate",
          "[program][transactional][hops]") {
    const auto one = run_gemm_on(dma_starved(1));
    const auto two = run_gemm_on(dma_starved(2));
    const auto four = run_gemm_on(dma_starved(4));

    CHECK(one.stats.mover_utilization.at(Mover::Dma) > 0.95);       // DMA saturated
    CHECK(one.stats.mover_utilization.at(Mover::Streamer) < 0.30);  // streamers idle
    CHECK(two.stats.makespan < one.stats.makespan);
    CHECK(four.stats.makespan < two.stats.makespan);

    // More engines overlap more transfers; they never shorten the work itself.
    CHECK(two.stats.mover_busy_cycles.at(Mover::Dma) ==
          one.stats.mover_busy_cycles.at(Mover::Dma));
    CHECK(four.stats.mover_busy_cycles.at(Mover::Dma) ==
          one.stats.mover_busy_cycles.at(Mover::Dma));
    CHECK(one.stats.lower_bound <= double(one.stats.makespan));
}

TEST_CASE("lanes give concurrency, never speed-up (§6.3, normative)",
          "[program][transactional][hops]") {
    auto first_transfer_cycles = [](const TileRunResult& r) {
        for (const auto& rec : r.timeline)
            if (!rec.hops.empty()) return rec.hops[0].finish - rec.hops[0].start;
        return Cycle(0);
    };
    const Cycle d1 = first_transfer_cycles(run_gemm_on(dma_starved(1)));
    const Cycle d2 = first_transfer_cycles(run_gemm_on(dma_starved(2)));
    const Cycle d8 = first_transfer_cycles(run_gemm_on(dma_starved(8)));
    REQUIRE(d1 > 0);
    CHECK(d1 == d2);
    CHECK(d1 == d8);
}

TEST_CASE("hops are pipelined for one tile, not serialized end-to-end",
          "[program][transactional][hops]") {
    const auto r = run_gemm_on(dma_starved(2));
    std::size_t multi_hop = 0;
    for (const auto& rec : r.timeline) {
        if (rec.hops.size() < 2) continue;
        ++multi_hop;
        for (std::size_t i = 1; i < rec.hops.size(); ++i)
            CHECK(rec.hops[i].start >= rec.hops[i - 1].finish);
        CHECK(rec.start == rec.hops.front().start);
        CHECK(rec.finish == rec.hops.back().finish);
    }
    REQUIRE(multi_hop > 0);
}

TEST_CASE("two legs of one process share its lanes", "[program][transactional][hops]") {
    // There is one set of BlockMovers, not one per direction, so inbound L3->L2 and
    // outbound L2->L3 compete. The process's busy cycles must therefore be the sum of
    // both legs' cycles.
    const auto r = run_gemm_on(dma_starved(2));
    const Cycle bm = r.stats.mover_busy_cycles.at(Mover::BlockMover);
    const Cycle legs = r.stats.hop_busy_cycles.at(Hop::BlockMoverL3ToL2) +
                       r.stats.hop_busy_cycles.at(Hop::BlockMoverL2ToL3);
    CHECK(bm == legs);
    const Cycle str = r.stats.mover_busy_cycles.at(Mover::Streamer);
    CHECK(str == r.stats.hop_busy_cycles.at(Hop::StreamerL2ToL1) +
                 r.stats.hop_busy_cycles.at(Hop::StreamerL1ToL2));
}

TEST_CASE("the stream cost lands on the Streamer legs",
          "[program][transactional][hops]") {
    // l1_duration() models the L1 stream buffers: a Drain costs the C signature's element
    // stride, so an output-stationary drain bubble stretches it. That belongs to the legs
    // that touch those buffers, not to the DMA or BlockMover legs upstream.
    auto run = [](bool use_streams) {
        TileProgram p = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
        fill_matmul(p, 64, 64);
        stream::StreamProgram sp;
        if (use_streams)
            sp = stream::derive_matmul_streams(p, stream::SpaceTimeMap::output_stationary());
        const DeviceDescriptor dev = DeviceDescriptor::single();
        TileTransactionExecutor exec;
        TileExecutionRequest req{p, Placement::single(dev.compute_tiles), dev,
                                 use_streams ? &sp : nullptr, 1234};
        return exec.run(req);
    };
    const auto bytes_model = run(false);
    const auto stream_model = run(true);

    // The drain bubble shows up on the Streamer, and the upstream legs are untouched.
    CHECK(stream_model.stats.hop_busy_cycles.at(Hop::StreamerL1ToL2) >
          bytes_model.stats.hop_busy_cycles.at(Hop::StreamerL1ToL2));
    CHECK(stream_model.stats.hop_busy_cycles.at(Hop::DmaL3ToDram) ==
          bytes_model.stats.hop_busy_cycles.at(Hop::DmaL3ToDram));
    CHECK(stream_model.stats.hop_busy_cycles.at(Hop::BlockMoverL2ToL3) ==
          bytes_model.stats.hop_busy_cycles.at(Hop::BlockMoverL2ToL3));
}

TEST_CASE("the chain changes timing only, never values", "[program][transactional][hops]") {
    TileProgram ref_prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    fill_matmul(ref_prog, 64, 64);
    TileProgramReference ref;
    ref.run(ref_prog);

    TileProgram txn_prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    fill_matmul(txn_prog, 64, 64);
    const DeviceDescriptor dev = dma_starved(2);
    run_transactional(txn_prog, dev, Placement::single(dev.compute_tiles));

    const auto& a = ref_prog.operand("C").values;
    const auto& b = txn_prog.operand("C").values;
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        REQUIRE(std::memcmp(&a[i], &b[i], sizeof(float)) == 0);
}

TEST_CASE("chained runs stay deterministic", "[program][transactional][hops]") {
    const auto a = run_gemm_on(dma_starved(2));
    const auto b = run_gemm_on(dma_starved(2));
    CHECK(a.stats.makespan == b.stats.makespan);
    CHECK(a.stats.hop_lane_stalls == b.stats.hop_lane_stalls);
    REQUIRE(a.timeline.size() == b.timeline.size());
    for (std::size_t i = 0; i < a.timeline.size(); ++i) {
        CHECK(a.timeline[i].start == b.timeline[i].start);
        REQUIRE(a.timeline[i].hops.size() == b.timeline[i].hops.size());
        for (std::size_t h = 0; h < a.timeline[i].hops.size(); ++h) {
            CHECK(a.timeline[i].hops[h].hop == b.timeline[i].hops[h].hop);
            CHECK(a.timeline[i].hops[h].lane == b.timeline[i].hops[h].lane);
            CHECK(a.timeline[i].hops[h].start == b.timeline[i].hops[h].start);
        }
    }
}

TEST_CASE("a full pool does not block a transfer bound for another process",
          "[program][transactional][hops]") {
    // ready_move is ordered by op index and take_admissible checks CREDITS, not lanes, so
    // ending the scan on a lane-blocked op would stop an op whose first hop belongs to a
    // different process — a reuse feed starts at the BlockMover while the DMA is busy.
    const auto r = run_gemm_on(dma_starved(1));
    CHECK(r.stats.mover_utilization.at(Mover::Dma) > 0.999);   // never idle with work queued
    CHECK(double(r.stats.makespan) == r.stats.lower_bound);    // and so it attains the floor
}
