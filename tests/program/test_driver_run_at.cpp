// ============================================================================
// tests/program/test_driver_run_at.cpp
// The run_at seam (#285 §D2): one function maps a level to an interpreter, and
// it refuses a level it cannot run rather than substituting one it can. An L0 program
// runs at L-B only (kpu-run-csp-programs step 4e): the timed levels run CSP programs.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/driver/step_cursor.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>

#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <variant>
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
    for (ExecutionLevel l : {ExecutionLevel::BlockSequential, ExecutionLevel::ResourceTransactional,
                             ExecutionLevel::CycleAccurate}) {
        REQUIRE_FALSE(level_implemented(l));
        CHECK_THROWS_AS(run_at(l, p, device, Placement::single(1)), std::invalid_argument);
        // The refusal has to say where to follow it up, or it is just a dead end: L-T2's is
        // #283; L-T1 and L-CA run CSP programs, not this L0 trace (kpu-run-csp-programs 4e).
        CHECK(not_implemented_reason(l).find(l == ExecutionLevel::ResourceTransactional ? "#283" : "kpu-csp-gen") !=
              std::string::npos);
    }
    CHECK(level_implemented(ExecutionLevel::Behavioral));
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
    CHECK(out.makespan == 0);
    CHECK(out.ops > 0);
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

TEST_CASE("the shared spec builds the device the levels schedule on",
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

TEST_CASE("an option present without a value is an error, not an absence",
          "[program][driver]") {
    // arg() cannot tell those apart: it returns the fallback both when an option is absent
    // and when it is the last token. For a flag that must carry a value those are
    // different errors, and only one of them is silent.
    std::string out, err;
    const std::vector<std::string> terminal{"--size", "32", "--timeline"};
    CHECK(arg_present(terminal, "--timeline"));
    CHECK_FALSE(arg_required(terminal, "--timeline", out, err));
    CHECK(err.find("missing value") != std::string::npos);

    // A following flag is not a value either, or `--timeline --step` writes a trace to a
    // file called "--step".
    const std::vector<std::string> flag_as_value{"--timeline", "--step"};
    err.clear();
    CHECK_FALSE(arg_required(flag_as_value, "--timeline", out, err));
    CHECK(err.find("--step") != std::string::npos);

    // Absent leaves the default alone and is NOT an error.
    out = "untouched";
    err.clear();
    const std::vector<std::string> absent{"--size", "32"};
    CHECK(arg_required(absent, "--timeline", out, err));
    CHECK(out == "untouched");
    CHECK(err.empty());

    // And a real value is taken.
    const std::vector<std::string> given{"--timeline", "run.json"};
    CHECK(arg_required(given, "--timeline", out, err));
    CHECK(out == "run.json");
}

// ============================================================================
// Increment 3 — single-stepping (§D4)
//
// One step is one transaction AT THAT LEVEL. An L0 program steps at L-B, by executing;
// a CSP program's L-T1 replay is csp_step.hpp's, tested with the CSP driver.
// ============================================================================

TEST_CASE("behavioral stepping executes, one op at a time", "[program][driver][step]") {
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;

    // Stepping every op must land in the same place as running the level outright. If it
    // did not, the stepper would be showing a different computation than the one it claims
    // to be stepping through.
    TileProgram whole = derive(ps);
    fill(whole, ps);
    const DeviceSpec ds;
    run_at(ExecutionLevel::Behavioral, whole, make_device(ds), Placement::single(1));

    TileProgram stepped = derive(ps);
    fill(stepped, ps);
    BehavioralStepper cur(stepped);
    CHECK(cur.size() == stepped.ops().size());

    std::size_t steps = 0;
    while (cur.step()) {
        CHECK(cur.current().kind == StepKind::OpApplied);
        CHECK(cur.current().op_index == steps);
        ++steps;
    }
    CHECK(steps == stepped.ops().size());
    CHECK(bit_identical(whole.operand(result_operand(ps)).values,
                        stepped.operand(result_operand(ps)).values));
}

TEST_CASE("behavioral stepping shows values forming", "[program][driver][step]") {
    // The reason real execution beats a replay at L-B: the result is observably
    // incomplete partway through, which is what makes it useful for debugging arithmetic.
    ProgramSpec ps;
    ps.size = 32;
    ps.tile = 16;
    TileProgram p = derive(ps);
    fill(p, ps);
    BehavioralStepper cur(p);

    const std::vector<float> before = p.operand("C").values;
    std::size_t taken = 0;
    while (taken < cur.size() / 2 && cur.step()) ++taken;
    const std::vector<float> midway = p.operand("C").values;
    while (cur.step()) {}
    const std::vector<float> after = p.operand("C").values;

    CHECK_FALSE(bit_identical(before, midway));    // something happened
    CHECK_FALSE(bit_identical(midway, after));     // and it was not finished
}

TEST_CASE("a level with no interpreter has no stepper either",
          "[program][driver][step]") {
    ProgramSpec ps;
    TileProgram p = derive(ps);
    fill(p, ps);
    for (ExecutionLevel l : {ExecutionLevel::BlockSequential, ExecutionLevel::ResourceTransactional,
                             ExecutionLevel::CycleAccurate})
        CHECK_THROWS_AS(make_stepper(l, p), std::invalid_argument);
}

// ----------------------------------------------------------------------------
// A program that came from a FILE (#265 increment 5 / #285 increment 4)
// ----------------------------------------------------------------------------
TEST_CASE("the program's inputs are the operands it READS", "[program][driver][file]") {
    // The rule is "read at all", not "read before it is written", and tile LU is why.
    ProgramSpec mm;
    mm.algo = "matmul"; mm.size = 32; mm.tile = 16;
    CHECK(program_inputs(derive(mm)) == std::vector<std::string>{"A", "B"});

    ProgramSpec lu;
    lu.algo = "lu"; lu.size = 32; lu.tile = 16;
    // LU factors A IN PLACE, and its FIRST op (LuDiagFactor) declares A[k,k] as an OUTPUT.
    // A read-before-written rule would classify A as produced and return nothing here --
    // leaving a loaded LU kernel to factor a zero matrix and report success at every level.
    CHECK(program_inputs(derive(lu)) == std::vector<std::string>{"A"});
}

TEST_CASE("synthesized inputs fill only what the program reads",
          "[program][driver][file]") {
    ProgramSpec ps;
    ps.algo = "matmul"; ps.size = 32; ps.tile = 16;
    TileProgram p = derive(ps);
    fill_inputs(p);

    // C is produced, so pre-filling it would make a value comparison compare the fill.
    for (float v : p.operand("C").values) CHECK(v == 0.0f);

    for (const char* name : {"A", "B"}) {
        const auto& t = p.operand(name);
        bool any_nonzero = false;
        for (float v : t.values) {
            any_nonzero = any_nonzero || (v != 0.0f);
            // EXACTLY REPRESENTABLE: a multiple of 1/8 needs no rounding, so the fill is
            // reproducible on a host whose instruction selection differs (-march=native).
            CHECK(v * 8.0f == std::floor(v * 8.0f));
            CHECK(std::isfinite(v));
        }
        CHECK(any_nonzero);
    }

    // Two operands must not receive the same pattern, or a level that swapped them would
    // still compute the right answer.
    CHECK_FALSE(bit_identical(p.operand("A").values, p.operand("B").values));

    // And a row must not be nearly constant. `h + 131*r + 17*c` passed every check above
    // while producing an integer part CONSTANT ALONG EACH ROW -- 17*c vanishes mod 17 --
    // so each row held eight distinct values spanning a range of 1.0. A level that read a
    // neighbouring element would have computed almost the right answer.
    const auto& A = p.operand("A");
    std::set<float> in_row_0, in_col_0;
    for (Dim c = 0; c < A.cols; ++c) in_row_0.insert(A.at(0, c));
    for (Dim r = 0; r < A.rows; ++r) in_col_0.insert(A.at(r, 0));
    CHECK(in_row_0.size() > 8);
    CHECK(in_col_0.size() > 8);
}

TEST_CASE("synthesized inputs are the same on every call", "[program][driver][file]") {
    // The driver fills each level's copy independently, so a fill that varied between
    // calls would make every comparison meaningless -- and it would fail as a DISAGREEMENT
    // between the models, which is the most expensive way to find out.
    ProgramSpec ps;
    ps.algo = "lu"; ps.size = 32; ps.tile = 16;
    TileProgram a = derive(ps), b = derive(ps);
    fill_inputs(a);
    fill_inputs(b);
    CHECK(bit_identical(a.operand("A").values, b.operand("A").values));
}

TEST_CASE("a kernel from a file, filled, computes what the same program computes in memory",
          "[program][driver][file]") {
    // The path the driver takes for `--program kernel.l0 --fill-inputs`: nothing here
    // calls fill(), because a loaded program has no ProgramSpec behind it to fill from.
    using namespace sw::kpu::program::serialize;
    for (const char* algo : {"matmul", "lu"}) {
        ProgramSpec ps;
        ps.algo = algo; ps.size = 32; ps.tile = 16;

        // Written as a KERNEL: structure only, no values (increment 1's mode).
        const std::string text = to_string(derive(ps));
        LoadInfo info;
        const TileProgram kernel = from_string(text, &info);
        REQUIRE_FALSE(info.has_values);

        const DeviceSpec ds;
        const auto device = make_device(ds);
        // The file's program, and the derivation's own, each filled the way kpu-run fills a
        // kernel and run at L-B: reading the file must not change what is computed.
        std::vector<TileProgram> ran{kernel, derive(ps)};
        for (TileProgram& p : ran) {
            fill_inputs(p);
            run_at(ExecutionLevel::Behavioral, p, device, Placement::single(device.compute_tiles));
        }

        // Every operand, not just the result: a level that scribbled on an INPUT while
        // computing the right output is a model bug the old one-operand check could not see.
        for (std::size_t i = 1; i < ran.size(); ++i)
            for (const std::string& key : ran[0].operand_order())
                CHECK(bit_identical(ran[0].operand(key).values,
                                    ran[i].operand(key).values));

        // The synthesized inputs must actually have driven a computation. A kernel run on
        // zeros also agrees at every level, and agreeing on nothing is the failure this
        // whole path exists to avoid.
        bool produced_something = false;
        for (float v : ran[0].operand(result_operand(ps)).values)
            produced_something = produced_something || (v != 0.0f);
        CHECK(produced_something);
    }
}

TEST_CASE("a test case from a file runs without being filled at all",
          "[program][driver][file]") {
    // `--program case.l0` with no --fill-inputs, and the driver REFUSES the other
    // combination: a file carrying values must not have them replaced, or the run is no
    // longer the one the file describes.
    using namespace sw::kpu::program::serialize;
    ProgramSpec ps;
    ps.algo = "matmul"; ps.size = 32; ps.tile = 16;
    TileProgram authored = derive(ps);
    fill(authored, ps);

    LoadInfo info;
    TileProgram loaded = from_string(to_test_case(authored), &info);
    REQUIRE(info.has_values);

    const DeviceSpec ds;
    const auto device = make_device(ds);
    TileProgram reference = authored;
    run_at(ExecutionLevel::Behavioral, loaded, device,
           Placement::single(device.compute_tiles));
    run_at(ExecutionLevel::Behavioral, reference, device,
           Placement::single(device.compute_tiles));
    for (const std::string& key : reference.operand_order())
        CHECK(bit_identical(reference.operand(key).values, loaded.operand(key).values));
}
