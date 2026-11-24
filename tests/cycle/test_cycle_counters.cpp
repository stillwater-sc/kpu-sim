// ============================================================================
// tests/cycle/test_cycle_counters.cpp
// The cycle core's operational-analysis counters, step 0b (#534;
// docs/plans/cycle-based-kpu-prototype.md §3.3). Every expected count is worked out by hand from
// the rules in link.hpp and pipeline.hpp, and Little's law is checked against residence times
// measured independently, beat by beat.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pipeline.hpp"

#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/xue_export.hpp>

#include <cstdint>
#include <map>
#include <string>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;

namespace {

constexpr Picoseconds kLimit = 100'000'000;

std::map<std::string, Counters> counts(const Simulator& sim) {
    std::map<std::string, Counters> m;
    for (const auto& r : report(sim)) m[r.name] = r.c;
    return m;
}

// One state per cycle: busy + stalls + idle = T, for every component and link.
void partitioned(const Simulator& sim) {
    for (const auto& r : report(sim)) {
        CAPTURE(r.name);
        CHECK(r.c.busy + r.c.stalled() + r.c.idle == r.c.cycles);
    }
}

}  // namespace

TEST_CASE("Counters: one credit, a 2-cycle loop, every count by hand", "[cycle][counters]") {
    // Producer -> consumer over one link of capacity 1, latencies 1 and 1. The producer pushes at
    // 0, 2, ..., 198; at 1, 3, ..., 197 its beat is due and its credit is still on the way back.
    // The consumer pops at 1, 3, ..., 199, and the run ends after cycle 199: T = 200.
    Simulator sim;
    Domain& d = sim.add_domain(Domain("clk", 1000));
    auto& p = sim.add<Producer>("producer", d, 100);
    auto& c = sim.add<Consumer>("consumer", d);
    auto& l = sim.link<Beat>("p->c", p, c, 1);
    p.connect(l);
    c.connect(l);
    REQUIRE(sim.run_until_idle(kLimit));
    c.finish(100);
    const auto m = counts(sim);
    const Counters& cp = m.at("producer");
    CHECK(cp.cycles == 200);
    CHECK(cp.busy == 100);
    CHECK(cp.stalled(Stall::NoCredit) == 99);
    CHECK(cp.idle == 1);                                  // cycle 199: nothing left to push
    CHECK(cp.completions == 100);
    const Counters& cc = m.at("consumer");
    CHECK(cc.cycles == 200);
    CHECK(cc.busy == 100);
    CHECK(cc.idle == 100);
    CHECK(cc.arrivals == 100);
    CHECK(cc.completions == 100);
    CHECK(cc.occupancy_integral == 0);                    // a sink holds nothing past its pop
    CHECK_THAT(cc.throughput(), WithinRel(0.5));          // X = C / T
    const Counters& cl = m.at("p->c");
    CHECK(cl.arrivals == 100);
    CHECK(cl.completions == 100);
    CHECK(cl.busy == 100);                                // the cycles it carried a push
    CHECK(cl.occupancy_integral == 100);                  // each beat is in the link for one cycle
    CHECK(cl.max_occupancy == 1);
    CHECK_THAT(cl.mean_residence(), WithinRel(1.0));      // R = Q / C
    CHECK_THAT(cl.mean_occupancy(), WithinRel(0.5));      // N = Q / T = X R
    partitioned(sim);
}

TEST_CASE("Little's law is exact: the occupancy integrals are the residence times, beat by beat",
          "[cycle][counters]") {
    // A beat is in exactly one place at the end of each cycle from its push to its pop: the link
    // into the stage, the stage, or the link out. So the three integrals add up to the sum over
    // beats of (consumer's pop - producer's push), measured from the two logs.
    for (const bool stalled : {false, true}) {
        CAPTURE(stalled);
        Simulator sim;
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 80);
        auto& s = sim.add<Stage>("stage", d, 5, 3, 4);
        auto& c = sim.add<Consumer>("consumer", d);
        auto& a = sim.link<Beat>("p->s", p, s, 4);
        auto& b = sim.link<Beat>("s->c", s, c, 3, 2, 2);
        p.connect(a);
        s.connect(a, b);
        c.connect(b);
        if (stalled) c.stall(40, 120);
        REQUIRE(sim.run_until_idle(kLimit));
        c.finish(80);
        std::uint64_t residence = 0;
        for (const auto& [id, at] : c.received) residence += at - p.pushed_at[id];
        const auto m = counts(sim);
        CHECK(m.at("p->s").occupancy_integral + m.at("stage").occupancy_integral + m.at("s->c").occupancy_integral ==
              residence);
        // At quiescence everything that arrived has completed: A = C, nothing held.
        for (const char* n : {"p->s", "stage", "s->c"}) {
            CAPTURE(n);
            CHECK(m.at(n).arrivals == 80);
            CHECK(m.at(n).completions == 80);
            CHECK(m.at(n).occupancy == 0);
        }
        // The stage's service: it accepts one beat every 3 cycles, so it is never empty in steady
        // state, and holds at most ceil(5 / 3) = 2 beats when unblocked.
        if (!stalled) CHECK(m.at("stage").max_occupancy == 2);
        partitioned(sim);
    }
}

TEST_CASE("A stage is busy from the cycle it accepts a beat to the cycle it forwards it", "[cycle][counters]") {
    // One beat through a stage of latency 3: pushed at 0, accepted at 1, held at the end of 1, 2
    // and 3, forwarded at 4 with nothing left -- a productive cycle, so busy (#550's review): 4.
    Simulator sim;
    Domain& d = sim.add_domain(Domain("clk", 1000));
    auto& p = sim.add<Producer>("producer", d, 1);
    auto& s = sim.add<Stage>("stage", d, 3);
    auto& c = sim.add<Consumer>("consumer", d);
    auto& a = sim.link<Beat>("p->s", p, s, 1);
    auto& b = sim.link<Beat>("s->c", s, c, 1);
    p.connect(a);
    s.connect(a, b);
    c.connect(b);
    REQUIRE(sim.run_until_idle(kLimit));
    c.finish(1);
    CHECK(c.received.front().second == 5);
    const Counters st = counts(sim).at("stage");
    CHECK(st.busy == 4);
    CHECK(st.occupancy_integral == 3);                     // held at the end of 1, 2 and 3
    CHECK(st.idle == st.cycles - 4);
    partitioned(sim);
}

TEST_CASE("Stalls are counted under their cause", "[cycle][counters]") {
    // The consumer is held for cycles [20, 80) with a beat waiting in every one of them; behind it
    // the stage and the producer run out of credits.
    Simulator sim;
    Domain& d = sim.add_domain(Domain("clk", 1000));
    auto& p = sim.add<Producer>("producer", d, 200);
    auto& s = sim.add<Stage>("stage", d, 2, 1, 2);
    auto& c = sim.add<Consumer>("consumer", d);
    auto& a = sim.link<Beat>("p->s", p, s, 4);
    auto& b = sim.link<Beat>("s->c", s, c, 4);
    p.connect(a);
    s.connect(a, b);
    c.connect(b);
    c.stall(20, 80);
    REQUIRE(sim.run_until_idle(kLimit));
    c.finish(200);
    const auto m = counts(sim);
    CHECK(m.at("consumer").stalled(Stall::External) == 60);
    CHECK(m.at("consumer").busy == 200);
    CHECK(m.at("stage").stalled(Stall::NoCredit) > 0);
    CHECK(m.at("producer").stalled(Stall::NoCredit) > 0);
    // Only the consumer is held; nothing else claims that cause, and no other cause occurs.
    for (const char* n : {"producer", "stage"}) {
        CAPTURE(n);
        CHECK(m.at(n).stalled(Stall::External) == 0);
        CHECK(m.at(n).stalled() == m.at(n).stalled(Stall::NoCredit));
    }
    partitioned(sim);
}

TEST_CASE("Each component counts its own domain's cycles", "[cycle][counters]") {
    // Two pipelines, at 1 GHz and 250 MHz, in one simulator: it runs until the slow one is done,
    // so the fast domain is clocked about four times as often.
    Simulator sim;
    Domain& f = sim.add_domain(Domain::from_ghz("fast", 1.0));
    Domain& s = sim.add_domain(Domain::from_ghz("slow", 0.25));
    auto& pf = sim.add<Producer>("pf", f, 20);
    auto& cf = sim.add<Consumer>("cf", f);
    auto& ps = sim.add<Producer>("ps", s, 20);
    auto& cs = sim.add<Consumer>("cs", s);
    auto& lf = sim.link<Beat>("f", pf, cf, 1);
    auto& ls = sim.link<Beat>("s", ps, cs, 1);
    pf.connect(lf);
    cf.connect(lf);
    ps.connect(ls);
    cs.connect(ls);
    REQUIRE(sim.run_until_idle(kLimit));
    const auto m = counts(sim);
    for (const char* n : {"pf", "cf", "f"}) CHECK(m.at(n).cycles == f.cycle());
    for (const char* n : {"ps", "cs", "s"}) CHECK(m.at(n).cycles == s.cycle());
    CHECK(s.cycle() == 40);                    // the slow consumer's last pop is at its cycle 39
    CHECK(f.cycle() == 157);                   // 1 GHz edges through 156 ns, the slow domain's last edge
    CHECK(m.at("cf").busy == m.at("cs").busy);
    partitioned(sim);
}

TEST_CASE("The counters summary, the XUE export, and reset", "[cycle][counters]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain("clk", 1000));
    auto& p = sim.add<Producer>("producer", d, 10);
    auto& c = sim.add<Consumer>("consumer", d);
    auto& l = sim.link<Beat>("p->c", p, c, 1);
    p.connect(l);
    c.connect(l);
    REQUIRE(sim.run_until_idle(kLimit));
    const std::string table = format_report(report(sim));
    CHECK_THAT(table, ContainsSubstring("producer") && ContainsSubstring("no credit 9") &&
                          ContainsSubstring("p->c") && ContainsSubstring("link"));
    // XUE's EventStats: its throughput and average delay (Q / A) agree with ours at quiescence.
    // (Its utilization() approximates U from the occupancy; ours counts the busy cycles.)
    const Counters lc = counts(sim).at("p->c");
    sw::xue::EventStats x;
    to_xue(lc, x);
    CHECK_THAT(x.throughput(lc.cycles), WithinRel(lc.throughput()));
    CHECK_THAT(x.average_delay(), WithinRel(lc.mean_residence()));
    CHECK_THAT(x.average_occupancy(lc.cycles), WithinRel(lc.mean_occupancy()));
    CHECK(x.total_cycles.load() == lc.busy);
    // A reset run counts from zero, and counts the same.
    sim.reset();
    for (const auto& r : report(sim)) CHECK(r.c.cycles == 0);
    REQUIRE(sim.run_until_idle(kLimit));
    CHECK(format_report(report(sim)) == table);
}
