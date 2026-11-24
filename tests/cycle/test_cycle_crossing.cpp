// ============================================================================
// tests/cycle/test_cycle_crossing.cpp
// The cycle core's clock-domain crossings, step 0d (#536; docs/plans/cycle-based-kpu-prototype.md
// §3.2, §3.5): an async FIFO between two domains, with a synchronizer each way (link.hpp), and
// each domain's clock track in the trace.
//
// The two clocks are the plan's 4:1 case: the array at 1 GHz (1000 ps) and the L2 banks at
// 250 MHz (4000 ps). Their edges coincide on every fourth array edge. Every expected cycle below
// is worked out by hand from link.hpp's rule: a beat pushed at the sender's edge t is seen at the
// receiver's sync-th edge strictly after t, and a credit freed at the receiver's edge t is the
// sender's at its credit_sync-th edge strictly after t.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pipeline.hpp"

#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 100'000'000;

std::map<std::string, Counters> counts(const Simulator& sim) {
    std::map<std::string, Counters> m;
    for (const auto& r : report(sim)) m[r.name] = r.c;
    return m;
}

// Producer -> crossing -> consumer, the producer in one domain and the consumer in the other.
// `array_first`: the array's domain is added first (the scheduler's order at coinciding edges).
struct Cross {
    Simulator sim;
    Domain* array = nullptr;
    Domain* banks = nullptr;
    Producer* p = nullptr;
    Consumer* c = nullptr;
    Link<Beat>* x = nullptr;
    Cross(bool fast_to_slow, std::uint64_t n, std::size_t capacity, bool array_first = true, Cycle sync = 2,
          Cycle credit_sync = 2, Tracer* tracer = nullptr) {
        if (tracer) sim.attach(*tracer);
        if (array_first) {
            array = &sim.add_domain(Domain("array", 1000));
            banks = &sim.add_domain(Domain("banks", 4000));
        } else {
            banks = &sim.add_domain(Domain("banks", 4000));
            array = &sim.add_domain(Domain("array", 1000));
        }
        Domain& tx = fast_to_slow ? *array : *banks;
        Domain& rx = fast_to_slow ? *banks : *array;
        p = &sim.add<Producer>("producer", tx, n);
        c = &sim.add<Consumer>("consumer", rx);
        x = &sim.crossing<Beat>("crossing", *p, *c, capacity, sync, credit_sync);
        p->connect(*x);
        c->connect(*x);
    }
    [[nodiscard]] std::vector<Cycle> pops() const {
        std::vector<Cycle> v;
        for (const auto& [id, at] : c->received) v.push_back(at);
        return v;
    }
};

// A crossing is sampled at its sender's edges, so after a fast receiver's last pop a slow
// sender's count is as of its last edge until it has another: clock up to it.
void close_sender(Simulator& sim, const Domain& tx) {
    const Cycle c = tx.cycle();
    while (tx.cycle() == c) REQUIRE(sim.step());
}

std::string slurp_and_remove(const std::string& path) {
    std::string text;
    {
        std::ifstream in(path);
        std::stringstream ss;
        ss << in.rdbuf();
        text = ss.str();
    }
    std::remove(path.c_str());
    return text;
}

}  // namespace

TEST_CASE("Domains: the first edge strictly after a time", "[cycle][crossing]") {
    const Domain d("d", 1000);
    CHECK(d.first_after(0) == 1);
    CHECK(d.first_after(999) == 1);
    CHECK(d.first_after(1000) == 2);   // an edge at t itself is not after t
    CHECK(d.first_after(1500) == 2);

    const Domain ph("phase", 1000, 1, 300);   // edges at 300, 1300, ...
    CHECK(ph.first_after(0) == 0);
    CHECK(ph.first_after(299) == 0);
    CHECK(ph.first_after(300) == 1);
    CHECK(ph.first_after(1299) == 1);
    CHECK(ph.first_after(1300) == 2);

    const Domain lp("lpddr", 625, 2);         // 312.5 ps: edges at 0, 312, 625, 937, 1250
    CHECK(lp.first_after(311) == 1);
    CHECK(lp.first_after(312) == 2);
    CHECK(lp.first_after(624) == 2);
    CHECK(lp.first_after(625) == 3);
    CHECK(lp.first_after(936) == 3);
    CHECK(lp.first_after(937) == 4);

    // The definition, for every picosecond over a few periods: edge(n - 1) <= t < edge(n).
    for (const Domain& x : {d, ph, lp, Domain("third", 1000, 3), Domain::from_ghz("ddr", 3.2), Domain("p7", 7, 1, 5)}) {
        CAPTURE(x.name());
        for (Picoseconds t = 0; t < 5000; ++t) {
            const Cycle n = x.first_after(t);
            CHECK(x.edge(n) > t);
            if (n > 0) CHECK(x.edge(n - 1) <= t);
        }
    }
}

TEST_CASE("A crossing is made between two domains, and only there", "[cycle][crossing]") {
    Simulator sim;
    Domain& a = sim.add_domain(Domain("array", 1000));
    Domain& b = sim.add_domain(Domain("banks", 4000));
    auto& pa = sim.add<Producer>("pa", a, 1);
    auto& ca = sim.add<Consumer>("ca", a);
    auto& cb = sim.add<Consumer>("cb", b);
    CHECK_THROWS_WITH(sim.link<Beat>("x", pa, cb, 1), ContainsSubstring("crossing()"));
    CHECK_THROWS_WITH(sim.crossing<Beat>("y", pa, ca, 1), ContainsSubstring("use link()"));
    CHECK_THROWS_WITH(sim.crossing<Beat>("z", pa, cb, 1, 0, 2), ContainsSubstring("at least 1"));
    CHECK_THROWS_WITH(sim.crossing<Beat>("z", pa, cb, 1, 2, 0), ContainsSubstring("at least 1"));
    // The receiver's domain must be this simulator's.
    Simulator elsewhere;
    Domain& o = elsewhere.add_domain(Domain("other", 4000));
    auto& co = elsewhere.add<Consumer>("co", o);
    CHECK_THROWS_WITH(sim.crossing<Beat>("w", pa, co, 1), ContainsSubstring("add_domain first"));

    auto& x = sim.crossing<Beat>("x", pa, cb, 2);
    CHECK(x.crossing());
    CHECK(&x.domain() == &a);                 // counted with the sender
    CHECK(&x.rx_domain() == &b);
    CHECK_THAT(x.describe(0), ContainsSubstring("crossing array -> banks"));
}

TEST_CASE("A crossing's beat and credit, cycle by cycle on each side", "[cycle][crossing]") {
    // Array (1000 ps) to banks (4000 ps), two stages each way. A beat pushed at array cycle 0 is
    // seen at the banks' second edge after 0 ps: cycle 2, 8000 ps. Until then it is in flight, by
    // the sender's cycles 0-7; from array cycle 8 (8000 ps) it has landed.
    const Domain a("array", 1000), b("banks", 4000);
    Link<Beat> x("x", a, b, 2, 2, 2);
    Beat beat;
    beat.id = 7;
    x.push(0, beat);
    for (Cycle c = 0; c < 8; ++c) {
        CAPTURE(c);
        CHECK(x.in_flight(c) == 1);
        CHECK(x.buffered(c) == 0);
        CHECK(x.transporting(c));
    }
    CHECK(x.in_flight(8) == 0);
    CHECK(x.buffered(8) == 1);
    CHECK_FALSE(x.can_pop(1));                 // the receiver's cycles
    REQUIRE(x.can_pop(2));
    CHECK(x.pop(2).id == 7);
    // Freed at 8000 ps, the credit is the sender's at its second edge after: array cycle 10.
    CHECK(x.credits(9) == 1);
    CHECK(x.credits_returning() == 1);
    CHECK(x.transporting(9));
    CHECK(x.credits(10) == 2);
    CHECK_FALSE(x.transporting(10));
    x.check(10);
}

TEST_CASE("Fast to slow at 4:1: every beat, in order, at the banks' rate", "[cycle][crossing]") {
    // Beat 0 is pushed at array cycle 0 (0 ps). The banks' first edge after 0 ps is cycle 1, so
    // with two stages it is seen at banks cycle 2 (8000 ps). Its credit, freed at 8000 ps, is back
    // at the array's second edge after it: cycle 10. Array cycles 0-3 push beats 0-3 with the four
    // credits; all land at banks cycle 2, and the consumer takes one per banks cycle. A credit
    // freed at banks cycle r is spent at array cycle 4r + 2, and that beat lands at banks cycle
    // r + 2: the FIFO never runs dry, so beat k is popped at banks cycle 2 + k.
    Cross x(true, 100, 4);
    REQUIRE(x.sim.run_until_idle(kLimit));
    x.c->finish(100);
    const auto pops = x.pops();
    for (std::uint64_t k = 0; k < 100; ++k) {
        CAPTURE(k);
        CHECK(pops[k] == 2 + k);
    }
    CHECK(x.p->pushed_at[4] == 10);
    // One beat per banks cycle: the slower side's throughput. The producer waits on credits.
    const auto m = counts(x.sim);
    CHECK(m.at("consumer").busy == 100);
    CHECK(m.at("producer").stalls[static_cast<int>(Stall::NoCredit)] > 0);
}

TEST_CASE("Slow to fast at 4:1: every beat, in order, at the banks' rate", "[cycle][crossing]") {
    // Beat k is pushed at banks cycle k (4000k ps), seen at the array's second edge after it,
    // 4k + 2, and popped there. Its credit, freed at 4000k + 2000 ps, is the producer's at the
    // banks' second edge after: k + 2. Two credits keep a push every banks cycle.
    Cross x(false, 100, 2);
    REQUIRE(x.sim.run_until_idle(kLimit));
    x.c->finish(100);
    const auto pops = x.pops();
    for (std::uint64_t k = 0; k < 100; ++k) {
        CAPTURE(k);
        CHECK(pops[k] == 4 * k + 2);
        CHECK(x.p->pushed_at[k] == k);
    }
    // The consumer is busy one array cycle in four: the slower side's throughput.
    const auto m = counts(x.sim);
    CHECK(m.at("consumer").busy == 100);
    CHECK(m.at("consumer").cycles >= 4 * 99 + 3);
}

TEST_CASE("A crossing's credit loop bounds its throughput: too few entries for the round trip",
          "[cycle][crossing]") {
    SECTION("fast to slow, one entry") {
        // Pop at banks cycle r -> credit at array cycle 4r + 2 -> push lands at banks r + 2.
        Cross x(true, 20, 1);
        REQUIRE(x.sim.run_until_idle(kLimit));
        const auto pops = x.pops();
        for (std::uint64_t k = 0; k < 20; ++k) CHECK(pops[k] == 2 + 2 * k);
    }
    SECTION("slow to fast, one entry") {
        // Push at banks 2k -> popped at array 8k + 2 -> credit at banks 2k + 2.
        Cross x(false, 20, 1);
        REQUIRE(x.sim.run_until_idle(kLimit));
        const auto pops = x.pops();
        for (std::uint64_t k = 0; k < 20; ++k) CHECK(pops[k] == 8 * k + 2);
    }
    SECTION("one synchronizer stage each way halves the round trip") {
        // Fast to slow, sync 1: beat 0 lands at banks 1; a pop at r returns its credit at array
        // 4r + 1, which lands at banks r + 1. One entry: a beat every banks cycle from cycle 1.
        Cross x(true, 20, 1, true, 1, 1);
        REQUIRE(x.sim.run_until_idle(kLimit));
        const auto pops = x.pops();
        for (std::uint64_t k = 0; k < 20; ++k) CHECK(pops[k] == 1 + k);
    }
}

TEST_CASE("Backpressure across a crossing: no beat lost or duplicated, credits conserved each cycle",
          "[cycle][crossing]") {
    // A stage on the sender's side and the consumer held on the receiver's, in both directions.
    // The checks (on) test every link's conservation each cycle; the consumer checks order.
    for (const bool fast_to_slow : {true, false}) {
        CAPTURE(fast_to_slow);
        Simulator sim;
        Domain& a = sim.add_domain(Domain("array", 1000));
        Domain& b = sim.add_domain(Domain("banks", 4000));
        Domain& tx = fast_to_slow ? a : b;
        Domain& rx = fast_to_slow ? b : a;
        auto& p = sim.add<Producer>("producer", tx, 60);
        auto& s = sim.add<Stage>("stage", tx, 3, 1, 2);
        auto& c = sim.add<Consumer>("consumer", rx, 2);
        auto& l = sim.link<Beat>("p->s", p, s, 2);
        auto& x = sim.crossing<Beat>("s->c", s, c, 3);
        p.connect(l);
        s.connect(l, x);
        c.connect(x);
        c.stall(10, 40);
        c.stall(60, 61);
        REQUIRE(sim.run_until_idle(kLimit));
        c.finish(60);
        close_sender(sim, tx);
        const auto m = counts(sim);
        CHECK(m.at("s->c").arrivals == 60);
        CHECK(m.at("s->c").completions == 60);
        CHECK(m.at("s->c").occupancy == 0);
        CHECK(m.at("stage").stalls[static_cast<int>(Stall::NoCredit)] > 0);   // held up by the crossing
        CHECK(m.at("consumer").stalls[static_cast<int>(Stall::External)] > 0);
        CHECK(x.credits(tx.cycle()) + x.credits_returning() == 3);               // every credit home or on its way
    }
}

TEST_CASE("The operational laws hold across a crossing; Little's law beat by beat", "[cycle][crossing]") {
    // A crossing is counted in its sender's cycles, closed after every domain at that instant has
    // been clocked. A beat pushed at the sender's cycle p and popped at the receiver's edge t is
    // held at the end of every sender cycle from p up to, not including, the sender's first edge
    // at or after t: a pop at an instant both clocks share is counted before the sender's cycle
    // there closes, whichever domain the scheduler clocks first, as a pop is within one domain. So
    // the occupancy integral is the sum over beats of (tx.first_after(t - 1) - p).
    for (const bool fast_to_slow : {true, false}) {
        for (const bool array_first : {true, false}) {
            CAPTURE(fast_to_slow, array_first);
            Cross x(fast_to_slow, 80, 3, array_first);
            x.c->stall(15, 30);
            REQUIRE(x.sim.run_until_idle(kLimit));
            x.c->finish(80);
            const Domain& tx = x.x->domain();
            close_sender(x.sim, tx);
            const Domain& rx = x.x->rx_domain();
            std::uint64_t residence = 0;
            for (const auto& [id, at] : x.c->received) residence += tx.first_after(rx.edge(at) - 1) - x.p->pushed_at[id];
            const auto m = counts(x.sim);
            const Counters& k = m.at("crossing");
            CHECK(k.occupancy_integral == residence);
            CHECK(k.arrivals == 80);
            CHECK(k.completions == 80);
            CHECK(k.busy == 80);                    // one push per busy sender cycle
            CHECK(k.cycles == tx.cycle());
            CHECK(k.occupancy == 0);                // nothing held at quiescence
            for (const auto& r : report(x.sim)) {
                CAPTURE(r.name);
                CHECK(r.c.busy + r.c.stalled() + r.c.idle == r.c.cycles);   // one state per cycle
            }
        }
    }
}

TEST_CASE("Determinism: the order of domains whose edges coincide changes nothing", "[cycle][crossing]") {
    for (const bool fast_to_slow : {true, false}) {
        CAPTURE(fast_to_slow);
        Cross one(fast_to_slow, 50, 2, true);
        Cross two(fast_to_slow, 50, 2, false);
        one.c->stall(5, 12);
        two.c->stall(5, 12);
        REQUIRE(one.sim.run_until_idle(kLimit));
        REQUIRE(two.sim.run_until_idle(kLimit));
        CHECK(one.c->received == two.c->received);
        CHECK(one.p->pushed_at == two.p->pushed_at);
        const auto m1 = counts(one.sim), m2 = counts(two.sim);
        for (const auto& [name, k] : m1) {
            CAPTURE(name);
            const Counters& j = m2.at(name);
            CHECK(k.arrivals == j.arrivals);
            CHECK(k.completions == j.completions);
            CHECK(k.busy == j.busy);
            CHECK(k.idle == j.idle);
            CHECK(k.cycles == j.cycles);
            CHECK(k.occupancy_integral == j.occupancy_integral);
            CHECK(k.max_occupancy == j.max_occupancy);
            for (std::size_t i = 0; i < kStallCauses; ++i) CHECK(k.stalls[i] == j.stalls[i]);
        }
    }
}

TEST_CASE("No false deadlock while beats and credits cross a slow synchronizer", "[cycle][crossing]") {
    // Four synchronizer stages each way on the slow side, and a window shorter than the round trip
    // in array edges: the beats and credits in transit are progress.
    Cross x(true, 10, 1, true, 4, 4);
    x.sim.set_deadlock_window(8);
    REQUIRE(x.sim.run_until_idle(kLimit));
    x.c->finish(10);
}

TEST_CASE("The trace: a crossing's beats, each domain's cycle numbers and, as an option, its clock",
          "[cycle][crossing][trace]") {
    for (const bool waves : {false, true}) {
        CAPTURE(waves);
        Tracer tracer;
        tracer.clock_tracks(waves);
        auto x = std::make_unique<Cross>(true, 8, 2, true, 2, 2, &tracer);
        REQUIRE(x->sim.run_until_idle(kLimit));
        const std::string path = std::string("test_cycle_crossing_") + (waves ? "waves" : "cycles") + ".json";
        tracer.write(path);
        const std::string text = slurp_and_remove(path);

        CHECK_THAT(text, ContainsSubstring(R"j("name":"link crossing (array -> banks)")j"));
        CHECK_THAT(text, ContainsSubstring(R"("name":"clock array")"));
        // Each domain's cycle numbers, one per cycle, at its edges; its clock's rising edges.
        std::map<std::string, std::vector<std::pair<double, std::uint64_t>>> cycles, levels;
        const std::regex counter(R"re("ph":"C".*"name":"(array|banks)","ts":([0-9.]+),"args":\{"(cycle|clock)":([0-9]+)\})re");
        std::istringstream lines(text);
        for (std::string line; std::getline(lines, line);) {
            std::smatch mt;
            if (!std::regex_search(line, mt, counter)) continue;
            auto& into = mt[3] == "cycle" ? cycles : levels;
            into[mt[1]].emplace_back(std::stod(mt[2]), std::stoull(mt[4]));
        }
        for (const Domain* d : {x->array, x->banks}) {
            CAPTURE(d->name());
            const auto& cs = cycles[d->name()];
            REQUIRE(cs.size() == d->cycle());
            for (std::uint64_t n = 0; n < cs.size(); ++n) {
                CHECK(cs[n].second == n);
                CHECK(static_cast<Picoseconds>(std::llround(cs[n].first * 1e6)) == d->edge(n));
            }
            CHECK(levels[d->name()].size() == (waves ? 2 * d->cycle() : 0));
        }
        if (waves) {
            // At 4:1 the banks' clock rises on every fourth rise of the array's.
            std::vector<double> rise_a, rise_b;
            for (const auto& [ts, v] : levels["array"]) if (v == 1) rise_a.push_back(ts);
            for (const auto& [ts, v] : levels["banks"]) if (v == 1) rise_b.push_back(ts);
            REQUIRE(!rise_b.empty());
            for (std::size_t n = 0; n < rise_b.size(); ++n) {
                REQUIRE(4 * n < rise_a.size());
                CHECK(rise_b[n] == rise_a[4 * n]);
            }
            // Low from the middle of each cycle: the array at 500 ps, the banks at 2000 ps.
            CHECK(levels["array"][1] == std::pair<double, std::uint64_t>{0.0005, 0});
            CHECK(levels["banks"][1] == std::pair<double, std::uint64_t>{0.002, 0});
        }
    }
    SECTION("the cycle numbers can be turned off") {
        Tracer tracer;
        tracer.cycle_tracks(false);
        Cross x(true, 4, 2, true, 2, 2, &tracer);
        REQUIRE(x.sim.run_until_idle(kLimit));
        const std::string path = "test_cycle_crossing_off.json";
        tracer.write(path);
        CHECK_THAT(slurp_and_remove(path), !ContainsSubstring(R"("args":{"cycle")"));
    }
}

TEST_CASE("A tracer attached after the domains still gets their clock tracks", "[cycle][crossing][trace]") {
    Tracer tracer;
    Simulator sim;
    Domain& d = sim.add_domain(Domain("early", 1000));
    sim.attach(tracer);
    Domain& e = sim.add_domain(Domain("late", 2000));
    auto& p = sim.add<Producer>("p", d, 2);
    auto& c = sim.add<Consumer>("c", e);
    auto& x = sim.crossing<Beat>("x", p, c, 2);
    p.connect(x);
    c.connect(x);
    REQUIRE(sim.run_until_idle(kLimit));
    const std::string path = "test_cycle_crossing_attach.json";
    tracer.write(path);
    const std::string text = slurp_and_remove(path);
    CHECK_THAT(text, ContainsSubstring(R"("name":"clock early")"));
    CHECK_THAT(text, ContainsSubstring(R"("name":"clock late")"));
    CHECK_THAT(text, ContainsSubstring(R"("name":"late","ts":)"));
}
