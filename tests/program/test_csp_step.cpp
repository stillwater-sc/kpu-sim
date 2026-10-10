// ============================================================================
// tests/program/test_csp_step.cpp
// Stepping a CSP program and its timeline (docs/plans/kpu-run-csp-programs.md step 4c).
//   - L-B: one action per step, applied. Stepping to the end leaves the values L-B's run
//     computes, and station occupancy never exceeds the program's L3.
//   - L-T1: one record per step, in start order. Every record of the run is replayed once;
//     lanes busy never exceed the process's lanes, and L3 slots held never exceed L-T1's peak.
//   - L-CA steps cycles (#283): refused, with the reason.
//   - The timeline: one Chrome-trace event per leg; a Release is not an interval.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/csp_timeline.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace lang = sw::kpu::program::csp::lang;
using sw::kpu::program::driver::ExecutionLevel;
using Catch::Matchers::ContainsSubstring;
using Proc = sw::kpu::program::csp::TransactionalInterpreter::Proc;

namespace {

lang::Program corpus(const std::string& name) {
    std::ifstream in("tests/program/corpus/" + name, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return lang::parse(s.str());
}

const char* kPrograms[] = {"matmul_48x48x48_t16.csp", "linear_64_t16_relu.csp", "lu_64_t16.csp"};

}  // namespace

TEST_CASE("csp step: L-B applies one action per step and ends with L-B's values", "[csp][step]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        platform::VirtualPlatform vp(platform::read_spec_file("tests/program/deploy/kpu_s1.json"));
        const lang::Program ast = corpus(name);
        const auto h = vp.load_csp(ast, driver::csp_inputs(ast));
        const auto run = vp.run_csp(h, ExecutionLevel::Behavioral);

        auto cur = vp.step_begin(h, ExecutionLevel::Behavioral);
        CHECK(cur.executes());
        CHECK_FALSE(cur.run_result().has_value());
        CHECK(cur.size() == run.outcome.actions);
        std::size_t peak = 0;
        while (cur.step()) {
            CHECK(cur.current().action + 1 == cur.position());
            CHECK(cur.current().l3_held <= static_cast<std::size_t>(ast.l3));
            peak = std::max(peak, cur.current().l3_held);
        }
        CHECK(cur.position() == cur.size());
        CHECK(cur.current().l3_held == 0);          // the program released everything
        CHECK(peak == run.outcome.peak_l3);
        const auto& lb = dynamic_cast<const driver::CspBehavioralStepper&>(cur.stepper());
        REQUIRE(lb.finished());
        for (const auto& op : run.outcome.values.operand_order())
            CHECK(lb.values().operand(op).values == run.outcome.values.operand(op).values);
    }
}

TEST_CASE("csp step: L-T1 replays every record in start order, within its lanes and slots", "[csp][step]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        const auto spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
        platform::VirtualPlatform vp(spec);
        const lang::Program ast = corpus(name);
        const auto h = vp.load_csp(ast, driver::csp_inputs(ast));

        auto cur = vp.step_begin(h, ExecutionLevel::BlockSequential);
        CHECK_FALSE(cur.executes());
        REQUIRE(cur.run_result().has_value());
        const auto& o = cur.run_result()->outcome;
        CHECK(cur.size() == o.records.size());
        CHECK(cur.run_result()->identity.level == ExecutionLevel::BlockSequential);

        std::uint64_t last = 0;
        std::map<std::size_t, std::size_t> seen;    // records replayed per action
        std::size_t releases = 0, store_legs = 0;
        while (cur.step()) {
            const auto& s = cur.current();
            CHECK(s.start >= last);
            last = s.start;
            CHECK(s.finish >= s.start);
            ++seen[s.action];
            CHECK(s.l3_held <= o.peak_l3);
            for (const auto& [p, n] : cur.stepper().lanes_busy()) {
                CAPTURE(p);
                CHECK(n >= 1);
                CHECK(n <= o.lanes.at(p));
            }
            if (s.kind == csp::Action::Kind::Release) {
                ++releases;
                CHECK_FALSE(s.proc.has_value());
                CHECK(s.start == s.finish);
            } else {
                REQUIRE(s.proc.has_value());
            }
            if (s.kind == csp::Action::Kind::Store) {
                CHECK(*s.proc == (s.leg == 0 ? Proc::Bm : Proc::Dma));
                ++store_legs;
            }
        }
        CHECK(cur.position() == cur.size());
        CHECK(seen.size() == o.actions);             // every action replayed
        CHECK(releases == o.slots.size());           // one Release per residency
        CHECK(store_legs == 2 * o.dram_stores);
    }
}

TEST_CASE("csp step: L-CA's step is a cycle, and is refused", "[csp][step]") {
    platform::VirtualPlatform vp(platform::read_spec_file("tests/program/deploy/kpu_s1.json"));
    const lang::Program ast = corpus("matmul_48x48x48_t16.csp");
    const auto h = vp.load_csp(ast, driver::csp_inputs(ast));
    CHECK_THROWS_WITH(vp.step_begin(h, ExecutionLevel::CycleAccurate), ContainsSubstring("#283"));
}

TEST_CASE("csp timeline: one event per leg on its process and lane", "[csp][timeline]") {
    const auto spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    platform::VirtualPlatform vp(spec);
    const lang::Program ast = corpus("linear_64_t16_relu.csp");
    const auto inputs = driver::csp_inputs(ast);
    const auto h = vp.load_csp(ast, inputs);
    const auto o = vp.run_csp(h, ExecutionLevel::BlockSequential).outcome;
    const auto entries = driver::csp_trace_entries(o.records, inputs, spec.device_view().element_bytes);

    std::size_t releases = 0;
    for (const auto& r : o.records) releases += r.kind == csp::Action::Kind::Release;
    REQUIRE(entries.size() == o.records.size() - releases);

    std::size_t i = 0;
    for (const auto& r : o.records) {
        if (r.kind == csp::Action::Kind::Release) continue;
        const auto& e = entries[i++];
        CAPTURE(e.description);
        CHECK(e.cycle_issue == r.start);
        CHECK(e.cycle_complete == r.finish);
        CHECK(e.component_type == driver::component_of(r.proc));
        CHECK(e.component_id == r.lane);
        CHECK_THAT(e.description, ContainsSubstring("[action " + std::to_string(r.action) + "]"));
        if (r.proc == Proc::Cf)
            CHECK(e.transaction_type == sw::trace::TransactionType::MATMUL);
        else
            CHECK(std::get<sw::trace::DMAPayload>(e.payload).bytes_transferred > 0);
    }
}
