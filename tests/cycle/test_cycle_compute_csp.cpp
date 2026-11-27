// ============================================================================
// tests/cycle/test_cycle_compute_csp.cpp
// The compute subsystem orchestrated by a CSP program, step 2e (#546; docs/plans/cycle-based-kpu-
// prototype.md §5.2-5.4, rungs 5-6): the walker's Move, Feed, Call, Drain, Writeback and L2Release
// actions as the BlockMover's, the L2's and the Streamer's programs (csp_compute_program.hpp), run
// on the whole path L3 -> L2 -> L1 -> array -> Combiner -> L2 -> L3, the L3 starting with A and B in
// the program's slots.
//
// Rung 5: single and double buffering in L2 (the program's `l2 = 3` and `l2 = 6`), and early release.
// Rung 6: the RTL's 8x8x16 int8 matmul (kpu_soc_2x2, a 4x4 array), then an int8 64^3 on kpu_s1's
// 128x64 array, its 16x16 tiles on the array's corner. On each: C in L3 equals L-B's per tile, int32
// exact, and the RTL's properties hold as behaviour, measured from the run: the L1 credit and no north
// collision (checked every cycle: a breach faults the run), the 2R - 1 drain, `completed` before
// `result` with the operands' L2 slots freed before it, and a chain writing its tile only at its end.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "compute_rig.hpp"

#include <sw/kpu/cycle/compute/csp_compute_program.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
namespace program = sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 100'000'000'000;   // 100 ms

platform::DeviceSpecification device(const std::string& file) {
    return platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file).device(0);
}

// The RTL's matmul: C[8,8] = A[8,16] . B[16,8], int8 in 4x4 tiles (tests/program/test_csp_double_buffer.cpp),
// with `l2` L2 slots: 3 holds one round of a call (A's copy, B's, the result), 6 two.
std::string rtl_8x8x16(int l2) {
    return "csp 1.1\nprogram mm machine flat(l3 = 32, l2 = " + std::to_string(l2) +
           ") {\n"
           "  tensor A[8,16] int8 tile 4x4 in;\n  tensor B[16,8] int8 tile 4x4 in;\n"
           "  tensor C[8,8] int32 tile 4x4 out;\n"
           "  for j in 0..2 {\n    resident B[:, j];\n    for i in 0..2 {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

// An int8 64^3 in 16x16 tiles; `l3` L3 slots, `buffers` rounds of a call in L2.
std::string int8_64cubed(std::size_t l3, int buffers) {
    csp::gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = l3;
    o.dtype = "int8";
    o.l2_buffers = buffers;
    return csp::gen::generate(o);
}

struct Case {
    csp::CspProgram p;
    program::TileProgram in, lb;   // the operands before the run, and L-B's result
};

Case compile(const std::string& text, const platform::DeviceSpecification& dev) {
    const lang::Program ast = lang::parse(text);
    const auto target = lang::target_from(dev);
    const program::TileProgram in = driver::csp_inputs(ast);
    auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    REQUIRE(!lb.skipped);
    return Case{lang::compile(ast, &target), in, lb.values};
}

// The measured run: what the tests check.
struct Run {
    std::vector<Streamer::Record> commands;
    std::vector<Combiner::Record> results;
    std::map<std::uint32_t, std::vector<std::pair<Cycle, Cycle>>> live;   // L2 slot -> [acquired, freed) kpu clocks
    Cycle end = 0;
    Cycle max_round_trip = 0;
};

struct Setup {
    std::uint32_t read_delay = 40;
    std::uint32_t banks = 4;
    std::uint32_t bus_bytes = 16;
    bool log = true;
};

// Runs the plan on the device's array; checks C in L3 against L-B's, every written-back tile.
Run run_and_check(const CspComputePlan& plan, const platform::DeviceSpecification& dev, const Setup& s) {
    const PEArray::Config array = PEArray::Config::from(dev);
    ComputeRig::Options o;
    o.l2 = L2::Config{plan.l2_slots, plan.rows_per_slot, s.banks, plan.row_bytes, 4, 2, 1};
    o.lanes = array.rows;
    o.cols = array.cols;
    o.read_delay = s.read_delay;
    o.bus_bytes = s.bus_bytes;
    o.log = s.log;
    ComputeRig r(plan.l2, plan.commands, o);
    for (const auto& t : plan.inputs) r.l3.load(t.address, t.bytes);

    Run run;
    std::vector<int> was(plan.l2_slots, 0);
    r.sim.add_invariant("watch", [&] {
        const Cycle now = r.kpu.cycle();
        for (std::uint32_t slot = 0; slot < plan.l2_slots; ++slot) {
            const int live = r.l2.live(slot) ? 1 : 0;
            if (live && !was[slot]) run.live[slot].push_back({now, now});
            if (!live && was[slot]) run.live[slot].back().second = now;
            was[slot] = live;
        }
    });
    REQUIRE(r.sim.run_until_idle(kLimit));
    for (std::uint32_t slot = 0; slot < plan.l2_slots; ++slot) CHECK(!r.l2.live(slot));

    REQUIRE(!plan.outputs.empty());
    for (const auto& t : plan.outputs) {
        CAPTURE(t.tile.to_string(), t.l3_slot);
        CHECK(r.l3.peek(t.address, t.bytes.size()) == t.bytes);
    }
    run.commands = r.st.records();
    run.results = r.combiner.records();
    run.end = r.kpu.cycle();
    run.max_round_trip = r.st.max_round_trip();
    CHECK(run.max_round_trip <= s.read_delay);
    CHECK(run.results.size() == plan.drains);
    CHECK(r.combiner.dropped() == std::uint64_t(plan.drains) * plan.max_h * (array.cols - plan.max_w));
    return run;
}

// The RTL's properties, measured from a run.
void check_properties(const CspComputePlan& plan, const Run& run, std::uint32_t R) {
    // A chain writes its tile only at its end: the Combiner's rows of a tile all come after its last
    // command fired, and every command before it in the chain names no result.
    std::size_t tile = 0;
    Cycle prev_last = 0;
    bool any_last = false;
    for (std::size_t n = 0; n < plan.commands.size(); ++n) {
        const StreamerCommand& c = plan.commands[n];
        REQUIRE(run.commands[n].completed);
        if (!c.last) {
            CHECK(!c.result);
            continue;
        }
        REQUIRE(c.result);
        REQUIRE(tile < run.results.size());
        const Combiner::Record& res = run.results[tile++];
        CHECK(res.command == n);
        REQUIRE(res.result);
        CHECK(res.rows.size() == plan.commands[n].a.lanes);
        CHECK(res.rows.front() > run.commands[n].accepted);
        // Early release: the operands' reads are done (`completed`) before the result is.
        CHECK(*run.commands[n].completed < *res.result);
        // The drain: reduction ends fire at least 2R - 1 clocks apart.
        if (any_last) CHECK(run.commands[n].accepted - prev_last >= 2 * R - 1);
        prev_last = run.commands[n].accepted;
        any_last = true;
    }
    CHECK(tile == run.results.size());
}

// Early L2 release: each chain's last command's operand slots are freed before its result is.
void check_early_release(const CspComputePlan& plan, const Run& run) {
    std::size_t tile = 0;
    for (std::size_t n = 0; n < plan.commands.size(); ++n) {
        const StreamerCommand& c = plan.commands[n];
        if (!c.last) continue;
        const Cycle result = *run.results[tile++].result;
        for (const auto* op : {&c.a, &c.b}) {
            // The residency the command read: the slot's span live when the command fired.
            const auto& spans = run.live.at(op->slot);
            const auto it = std::find_if(spans.begin(), spans.end(), [&](const auto& sp) {
                return sp.first <= run.commands[n].accepted && run.commands[n].accepted < sp.second;
            });
            REQUIRE(it != spans.end());
            CAPTURE(n, op->slot, it->second, result);
            CHECK(it->second < result);
        }
    }
}

}  // namespace

TEST_CASE("The plan maps the walker's actions onto the BlockMover's, the L2's and the Streamer's queues",
          "[cycle][csp][compute]") {
    const auto dev = device("kpu_soc_2x2.json");
    const Case c = compile(rtl_8x8x16(3), dev);
    const CspComputePlan plan = plan_compute(c.p, c.in, c.lb);
    // C[8,8] is 4 tiles, each a chain of 4 calls over K = 16 in 4-element blocks.
    CHECK(plan.calls == 16);
    CHECK(plan.commands.size() == 16);
    CHECK(plan.drains == 4);
    CHECK(plan.writebacks == 4);
    CHECK(plan.moves == 32);    // each call's A and B copies
    CHECK(plan.feeds == 32);
    CHECK(plan.l2_releases == 36);
    CHECK(plan.l2_slots == 3);
    CHECK(plan.rows_per_slot == 4);
    CHECK(plan.row_bytes == 16);   // 4 int32 a C row
    for (std::size_t n = 0; n < plan.commands.size(); ++n) {
        const StreamerCommand& cmd = plan.commands[n];
        CAPTURE(n);
        CHECK(cmd.k == 4);
        CHECK(cmd.a.lanes == 4);
        CHECK(cmd.b.lanes == 4);
        CHECK(cmd.first == (n % 4 == 0));
        CHECK(cmd.last == (n % 4 == 3));
        CHECK(cmd.result.has_value() == cmd.last);
        CHECK(cmd.a.slot != cmd.b.slot);
    }
    // The L3 image is Phase 1's (csp_l3_program.hpp): each tile densely, row-major, a byte an int8
    // element, then zeros to the 64-byte beat. The BlockMover makes the lanes (#548).
    CHECK(plan.slot_bytes == 64);   // a 4x4 int32 C tile; an int8 one is 16 bytes, padded
    std::size_t a = 0, b = 0;
    for (const auto& t : plan.inputs) {
        CAPTURE(t.tile.to_string());
        REQUIRE(t.bytes.size() == 64);
        CHECK(t.address == std::uint64_t(t.l3_slot) * 64);
        const auto& op = c.in.operand(t.tile.operand);
        for (std::uint32_t r = 0; r < 4; ++r)
            for (std::uint32_t k = 0; k < 4; ++k)
                CHECK(t.bytes[r * 4 + k] == static_cast<std::uint8_t>(op.word(t.tile.ti * 4 + r, t.tile.tj * 4 + k)));
        for (std::size_t e = 16; e < 64; ++e) CHECK(t.bytes[e] == 0);
        (t.tile.operand == "A" ? a : b)++;
    }
    // A's Moves are its rows, B's its columns (the transpose); C's Writebacks its rows, int32.
    for (const L2Action& x : plan.l2) {
        if (x.kind == L2Action::Kind::Move) {
            CHECK(x.layout == (x.tile.matrix == sw::kpu::isa::MatrixID::B ? L2Action::Layout::DenseColumns
                                                                          : L2Action::Layout::DenseRows));
            CHECK(x.element_bytes == 1);
        } else if (x.kind == L2Action::Kind::Writeback) {
            CHECK(x.layout == L2Action::Layout::DenseRows);
            CHECK(x.element_bytes == 4);
            CHECK(x.dense_bytes() == 64);
        }
    }
    CHECK(a == 16);   // A[i, :] is resident again for each j
    CHECK(b == 8);
    CHECK(plan.outputs.size() == 4);
    CHECK_NOTHROW(check_feeds(plan.commands, plan.l2));
}

TEST_CASE("The plan refuses what step 2e does not build, before any cycle", "[cycle][csp][compute]") {
    SECTION("an L3 slot that holds two residencies over the run (the Loads run in time at the join)") {
        const auto dev = device("kpu_s1.json");
        const Case c = compile(int8_64cubed(9, 1), dev);
        CHECK_THROWS_WITH(plan_compute(c.p, c.in, c.lb), ContainsSubstring("opens a second residency in L3 slot"));
    }
    SECTION("a program whose A and B copies share one L2 slot: a command reads both at once") {
        // `l2 = 2`: the result's slot and one more, which A's copy and then B's take in turn. The CSP
        // feeds each into L1 whole and frees the slot; the RTL's Streamer reads A's and B's rows in
        // the same clocks, so each needs a slot of its own.
        const auto dev = device("kpu_soc_2x2.json");
        const Case c = compile(rtl_8x8x16(2), dev);
        CHECK_THROWS_WITH(plan_compute(c.p, c.in, c.lb), ContainsSubstring("reads A and B from one L2 slot"));
        // The same with one round of a call in L2 (`l2 = 3`) when the stores are late: the last
        // chain's result holds its slot past the next chain's first call.
        const auto s1 = device("kpu_s1.json");
        const Case d = compile(int8_64cubed(96, 1), s1);
        CHECK_THROWS_WITH(plan_compute(d.p, d.in, d.lb), ContainsSubstring("reads A and B from one L2 slot"));
    }
}

TEST_CASE("Rungs 5-6: the RTL's 8x8x16 int8 matmul from its CSP program, single and double buffered, C exact",
          "[cycle][csp][compute]") {
    const auto dev = device("kpu_soc_2x2.json");
    std::map<int, Cycle> end;
    for (const int l2 : {3, 6}) {
        CAPTURE(l2);
        const Case c = compile(rtl_8x8x16(l2), dev);
        const CspComputePlan plan = plan_compute(c.p, c.in, c.lb);
        const Run run = run_and_check(plan, dev, Setup{});
        check_properties(plan, run, 4);
        check_early_release(plan, run);
        end[l2] = run.end;
    }
    // Double buffering: the second round of copies moves while the first is read, so the run is shorter.
    CHECK(end[6] < end[3]);
}

TEST_CASE("Rung 6: an int8 64^3 matmul on kpu_s1's 128x64 array, its 16x16 tiles on the corner, C exact",
          "[cycle][csp][compute]") {
    const auto dev = device("kpu_s1.json");
    const Case c = compile(int8_64cubed(96, 2), dev);
    const CspComputePlan plan = plan_compute(c.p, c.in, c.lb);
    CHECK(plan.commands.size() == 64);   // 16 C tiles, 4 calls each
    CHECK(plan.drains == 16);
    CHECK(plan.max_h == 16);
    CHECK(plan.max_w == 16);
    CHECK(plan.row_bytes == 64);         // 16 int32 a C row
    Setup s;
    s.log = false;
    s.bus_bytes = 64;
    const Run run = run_and_check(plan, dev, s);
    check_properties(plan, run, 128);
    check_early_release(plan, run);
}
