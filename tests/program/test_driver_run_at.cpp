// ============================================================================
// tests/program/test_driver_run_at.cpp
// The run_at seam (#285 §D2): one function maps a level to an interpreter, and
// it refuses a level it cannot run rather than substituting one it can.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;

namespace {

bool bit_identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

} // namespace

TEST_CASE("every implemented level computes identical values", "[program][driver]") {
    // The whole reason the driver exists: values are level-invariant (ADR 0002 §2), so
    // this is a differential test of the models, not a smoke test of the CLI.
    for (const char* algo : {"matmul", "lu"}) {
        ProgramSpec ps;
        ps.algo = algo;
        ps.size = 48;
        ps.tile = 16;
        DeviceSpec ds;
        const auto device = make_device(ds);

        std::vector<float> authority;
        std::size_t authority_swaps = 0;
        std::vector<Dim> authority_perm;
        for (ExecutionLevel l : all_levels()) {
            if (!level_implemented(l)) continue;
            TileProgram p = derive(ps);
            fill(p, ps);
            const auto out = run_at(l, p, device, Placement::single(device.compute_tiles));
            const auto& vals = p.operand(result_operand(ps)).values;
            if (l == ExecutionLevel::Behavioral) {
                authority = vals;
                authority_swaps = out.summary.row_swaps;
                authority_perm = out.summary.permutation;
                continue;
            }
            CHECK(bit_identical(authority, vals));
            // LU's pivoting is state a value diff alone would not catch.
            CHECK(out.summary.row_swaps == authority_swaps);
            CHECK(out.summary.permutation == authority_perm);
        }
        REQUIRE_FALSE(authority.empty());
    }
}

TEST_CASE("a level with no interpreter is refused, never substituted",
          "[program][driver]") {
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    // Snapshot the result operand, so "nothing ran" is checkable. Asserting only that the
    // vector is non-empty would pass even if run_at had computed a full result into it --
    // the vector is sized by derive(), not by the run.
    const std::vector<float> c_before = p.operand("C").values;
    const DeviceSpec ds;
    const auto device = make_device(ds);

    // The failure mode this guards against is a driver that quietly answers a
    // different question than it was asked.
    for (ExecutionLevel l : {ExecutionLevel::ResourceTransactional,
                             ExecutionLevel::CycleAccurate}) {
        REQUIRE_FALSE(level_implemented(l));
        CHECK_THROWS_AS(run_at(l, p, device, Placement::single(1)), std::invalid_argument);
        // The refusal has to say where to follow it up, or it is just a dead end.
        CHECK(not_implemented_reason(l).find("#283") != std::string::npos);
    }
    // And nothing ran: the operand is bit-for-bit what fill() left there.
    REQUIRE_FALSE(c_before.empty());
    CHECK(bit_identical(p.operand("C").values, c_before));
}

TEST_CASE("L-B reports that it models no timing, rather than a timing of zero",
          "[program][driver]") {
    // makespan 0 would read as "instant". has_timing == false is a different claim,
    // and the distinction matters when a report is compared across levels.
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    const DeviceSpec ds;
    const auto out = run_at(ExecutionLevel::Behavioral, p, make_device(ds),
                            Placement::single(1));
    CHECK_FALSE(out.has_timing);
    CHECK_FALSE(out.stats.has_value());
    CHECK_FALSE(out.provenance.has_value());
    CHECK(out.ops > 0);

    TileProgram q = derive(ps);
    fill(q, ps);
    const auto t1 = run_at(ExecutionLevel::BlockSequential, q, make_device(ds),
                           Placement::single(1));
    CHECK(t1.has_timing);
    CHECK(t1.makespan > 0);
    REQUIRE(t1.stats.has_value());
    REQUIRE(t1.provenance.has_value());
    CHECK_FALSE(t1.provenance->calibrated);      // increment 6 has not happened
}

TEST_CASE("level names round-trip, and an unknown name is rejected",
          "[program][driver]") {
    for (ExecutionLevel l : all_levels()) {
        const auto back = parse_level(to_string(l));
        REQUIRE(back.has_value());
        CHECK(*back == l);
        const auto shortly = parse_level(short_name(l));
        REQUIRE(shortly.has_value());
        CHECK(*shortly == l);
    }
    CHECK_FALSE(parse_level("transactional").has_value());   // ambiguous under ADR 0002
    CHECK_FALSE(parse_level("csp").has_value());             // CSP is not a level at all
    CHECK_FALSE(parse_level("").has_value());
}

TEST_CASE("the shared spec builds the device the executor actually schedules on",
          "[program][driver]") {
    // §D1: one mapping, shared with the characterizer. Movement is per process, so the
    // spec must carry per-process lanes and no aggregate movement pool.
    DeviceSpec ds;
    ds.dma_engines = 2;
    ds.block_movers = 3;
    ds.streamers = 4;
    ds.l3_tiles = 24;
    const auto d = make_device(ds);
    CHECK(d.dma_engines == 2);
    CHECK(d.block_movers == 3);
    CHECK(d.streamers == 4);
    CHECK(d.l3_tiles == 24);
    CHECK(d.label().find("dma2") != std::string::npos);
    CHECK(d.label().find("bm3") != std::string::npos);
    CHECK(d.label().find("str4") != std::string::npos);
}
