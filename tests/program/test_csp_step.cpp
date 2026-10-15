// ============================================================================
// tests/program/test_csp_step.cpp
// Stepping a CSP program and its timeline (docs/plans/kpu-run-csp-programs.md step 4c), and the
// state transition each L-B step carries (docs/plans/domain-flow-debugger.md step 1).
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
#include <set>
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

// ---- the CSP state transition (docs/plans/domain-flow-debugger.md step 1) -------------------

namespace {

using Kind = csp::Action::Kind;
using Lazy = driver::CspBehavioralStepper;

std::vector<std::string> vars(const lang::Where& w) {
    std::vector<std::string> v;
    for (const auto& l : w.loops) v.push_back(l.var);
    return v;
}

long long index_of(const lang::Where& w, const std::string& var) {
    for (const auto& l : w.loops)
        if (l.var == var) return l.value;
    FAIL("no loop " << var);
    return -1;
}

}  // namespace

TEST_CASE("csp step: the walker reports where each action is in the program", "[csp][step][where]") {
    SECTION("matmul: j, i, k") {
        // matmul_48x48x48_t16.csp: for j (line 7) { resident B (8); for i (9) { resident A (10);
        // acc (11) { for k (12) { call } } store (14); release A (15) } release B (17) }
        const lang::Program ast = corpus("matmul_48x48x48_t16.csp");
        Lazy st(ast, driver::csp_inputs(ast), Lazy::Count::Lazy);
        std::size_t calls = 0, b_loads = 0, stores = 0;
        while (st.step()) {
            const auto& s = st.current();
            REQUIRE(s.where.has_value());
            const lang::Where& w = *s.where;
            CAPTURE(s.action, lang::to_string(w));
            for (const auto& l : w.loops) CHECK((l.value >= 0 && l.value < l.hi));
            if (s.kind == Kind::Call) {
                ++calls;
                CHECK(w.line == 12);
                CHECK(vars(w) == std::vector<std::string>{"j", "i", "k"});
                CHECK(s.tile.ti == static_cast<std::size_t>(index_of(w, "i")));    // C[i, j]
                CHECK(s.tile.tj == static_cast<std::size_t>(index_of(w, "j")));
                CHECK(w.loops[0].line == 7);
                CHECK(w.loops[1].line == 9);
                CHECK(w.loops[2].line == 12);
            } else if (s.kind == Kind::Load && s.tile.operand == "B") {
                ++b_loads;
                CHECK(w.line == 8);
                CHECK(vars(w) == std::vector<std::string>{"j"});
                CHECK(s.tile.tj == static_cast<std::size_t>(index_of(w, "j")));    // B[:, j]
            } else if (s.kind == Kind::Store) {
                ++stores;
                CHECK(w.line == 14);
                CHECK(vars(w) == std::vector<std::string>{"j", "i"});
            } else if (s.kind == Kind::Release && s.tile.operand == "B") {
                CHECK(w.line == 17);
            }
        }
        CHECK(calls == 27);
        CHECK(b_loads == 9);
        CHECK(stores == 9);
    }
    SECTION("lu: getrf on the diagonal at each k") {
        const lang::Program ast = corpus("lu_64_t16.csp");
        Lazy st(ast, driver::csp_inputs(ast), Lazy::Count::Lazy);
        std::size_t getrf = 0;
        while (st.step()) {
            const auto& s = st.current();
            REQUIRE(s.where.has_value());
            if (s.kind == Kind::Call && s.where->line == 7) {     // call getrf(A[k, k])
                CHECK(vars(*s.where) == std::vector<std::string>{"k"});
                CHECK(s.tile.ti == static_cast<std::size_t>(getrf));
                CHECK(s.tile.tj == static_cast<std::size_t>(getrf));
                CHECK(index_of(*s.where, "k") == static_cast<long long>(getrf));
                ++getrf;
            }
            if (s.where->line == 5) CHECK(s.where->loops.empty());   // resident A[:, :], outside every loop
        }
        CHECK(getrf == 4);
    }
}

TEST_CASE("csp step: a lazy L-B step carries the state transition and changes no value", "[csp][step]") {
    for (const char* name : kPrograms) {
        CAPTURE(name);
        platform::VirtualPlatform vp(platform::read_spec_file("tests/program/deploy/kpu_s1.json"));
        const lang::Program ast = corpus(name);
        const auto run = vp.run_csp(vp.load_csp(ast, driver::csp_inputs(ast)), ExecutionLevel::Behavioral);

        Lazy st(ast, driver::csp_inputs(ast), Lazy::Count::Lazy);
        CHECK_FALSE(st.size_known());
        std::size_t open = 0, retained = 0, next_residency = 0;
        std::set<std::size_t> live;
        while (st.step()) {
            const auto& s = st.current();
            CAPTURE(s.action, csp::to_string(s.kind), s.tile.to_string());
            if (st.finished()) {
                CHECK(st.size_known());
                CHECK(st.size() == st.position());
            } else {
                CHECK_FALSE(st.size_known());
            }
            // Who acted: every kind but the credit actions has a process, and it is the kind's.
            const bool credit = s.kind == Kind::Release || s.kind == Kind::Inherit || s.kind == Kind::Retain;
            CHECK(s.process.has_value() == !credit);
            CHECK(s.from.has_value() == (s.kind != Kind::Call && !credit));
            if (s.from) CHECK(*s.from != *s.to);
            // Residencies open in order, once, and end once.
            CHECK(s.place.has_value() == (s.residency != csp::kNone));
            if (s.opens) {
                CHECK(s.residency == next_residency++);
                CHECK(live.insert(s.residency).second);
                ++open;
            } else if (s.residency != csp::kNone) {
                CHECK(live.count(s.residency) == 1);
            }
            if (s.closes || s.retains) {
                CHECK(live.erase(s.residency) == 1);
                --open;
                retained += s.retains;
            }
            // Occupancy is the residencies: L3 holds the open ones and the retained ones.
            CHECK(s.held[static_cast<std::size_t>(csp::Chan::L3)] == open + retained);
            CHECK(s.held[static_cast<std::size_t>(csp::Chan::L3)] == s.l3_held);
            CHECK(s.l3_held <= static_cast<std::size_t>(ast.l3));
        }
        CHECK(live.empty());
        CHECK(st.size_known());
        CHECK(st.size() == run.outcome.actions);
        REQUIRE(st.finished());
        // Same computation, so bit-identical.
        for (const auto& op : run.outcome.values.operand_order())
            CHECK(st.values().operand(op).values == run.outcome.values.operand(op).values);
    }
}

TEST_CASE("csp step: a walker refusal is thrown at its own step, not one early", "[csp][step]") {
    const lang::Program ast = lang::parse(R"(csp 1.0
program bad machine flat(l3 = 2) {
  tensor A[16,16] tile 16x16 in;
  resident A[0, 0];
  release A[0, 0];
  release A[0, 0];
}
)");
    Lazy st(ast, driver::csp_inputs(ast), Lazy::Count::Lazy);
    REQUIRE(st.step());
    CHECK(st.current().kind == Kind::Load);
    REQUIRE(st.step());
    CHECK(st.current().kind == Kind::Release);
    CHECK_THROWS_WITH(st.step(), ContainsSubstring("line 6") && ContainsSubstring("not resident"));
}
