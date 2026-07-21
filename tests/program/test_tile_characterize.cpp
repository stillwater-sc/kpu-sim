// ============================================================================
// tests/program/test_tile_characterize.cpp
// Tile-DAG concurrency analysis + characterization metrics.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>
#include <sw/kpu/program/characterize/characterization.hpp>

using namespace sw::kpu::program;
using namespace sw::kpu::program::characterize;
using Catch::Approx;

namespace {
// A device where movement is effectively free, so the compute DAG governs the
// schedule — isolates the compute-concurrency behavior under test. (That matmul is
// movement-bound under realistic coefficients is itself a finding, exercised by the
// harness, not here.)
DeviceDescriptor compute_bound(Dim cf) {
    DeviceDescriptor d = DeviceDescriptor::checkerboard(cf);
    d.bytes_per_cycle = 1e12;   // movement ~ 0 cycles
    d.move_lanes = cf;
    return d;
}
} // namespace

TEST_CASE("matmul tile-DAG exposes output-tile parallelism", "[program][characterize]") {
    // 64^3, T=16 -> 4x4x4 tile grid: 16 output tiles, each an accumulate chain of
    // kt=4 GEMMs. So depth=4, width=16 independent chains.
    TileProgram prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);

    TileDag dag(prog, compute_bound(1));
    const double cp = dag.critical_path_cycles();
    const double compute_work = dag.compute_work_cycles();
    CHECK(cp > 0.0);
    CHECK(compute_work > cp);   // there IS parallelism to exploit

    // makespan must fall monotonically as we add compute tiles, then bottom out at
    // the critical path once we have >= width workers.
    auto makespan = [&](Dim cf) {
        TileDag g(prog, compute_bound(cf));
        return g.list_schedule().makespan;
    };
    const double m1 = makespan(1), m4 = makespan(4), m16 = makespan(16), m64 = makespan(64);
    CHECK(m1 > m4);
    CHECK(m4 > m16);
    // >= 16 workers cannot beat the dependency chain
    CHECK(m16 == Approx(m64));
    CHECK(m16 == Approx(cp));
    // single worker serializes all compute
    CHECK(m1 == Approx(compute_work));
}

TEST_CASE("LU tile-DAG is far more serial than matmul", "[program][characterize]") {
    const Dim N = 64, T = 16;   // 4x4 tile grid
    TileProgram mm = derive_matmul_tile_program(N, N, N, T, T, T);
    TileProgram lu = derive_lu_tile_program(N, T);

    DeviceDescriptor big = compute_bound(64);
    const double mm_cp = TileDag(mm, big).critical_path_cycles();
    const double mm_work = TileDag(mm, big).compute_work_cycles();
    const double lu_cp = TileDag(lu, big).critical_path_cycles();
    const double lu_work = TileDag(lu, big).compute_work_cycles();

    // matmul: shallow (critical path is a small fraction of total work).
    // LU: the panel dependencies make the critical path a large fraction.
    CHECK((mm_cp / mm_work) < (lu_cp / lu_work));
}

TEST_CASE("characterize reports structural + modeled metrics", "[program][characterize]") {
    TileProgram prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    DeviceDescriptor d = DeviceDescriptor::checkerboard(4);
    Metrics m = characterize_program(prog, d);

    CHECK(m.computes == 4u * 4u * 4u);            // mt*nt*kt GEMMs
    CHECK(m.total_macs == Approx(64.0 * 64.0 * 64.0));  // full matmul MAC count
    CHECK(m.total_move_bytes > 0.0);
    CHECK(m.arithmetic_intensity > 0.0);
    CHECK(m.peak_live_tiles > 0);
    CHECK(m.peak_live_tiles <= m.distinct_tiles);
    CHECK(m.makespan_cycles >= m.lower_bound_cycles);   // schedule >= lower bound
    CHECK(m.energy_total_pj > 0.0);
    CHECK(m.energy_total_pj == Approx(m.energy_compute_pj + m.energy_move_pj + m.energy_leak_pj));
}

TEST_CASE("feasibility gates on L3 tile capacity", "[program][characterize]") {
    TileProgram prog = derive_matmul_tile_program(64, 64, 64, 16, 16, 16);
    const std::size_t peak = peak_live_tiles(prog);
    REQUIRE(peak > 1);

    DeviceDescriptor tight = DeviceDescriptor::single();
    tight.l3_tiles = static_cast<Dim>(peak - 1);        // one short
    CHECK(characterize_program(prog, tight).feasible == false);

    DeviceDescriptor roomy = DeviceDescriptor::single();
    roomy.l3_tiles = static_cast<Dim>(peak);            // exactly fits
    CHECK(characterize_program(prog, roomy).feasible == true);

    DeviceDescriptor unbounded = DeviceDescriptor::single();  // l3_tiles == 0
    CHECK(characterize_program(prog, unbounded).feasible == true);
}
