// ============================================================================
// tests/cycle/test_cycle_core.cpp
// The cycle core, step 0a (#533; docs/plans/cycle-based-kpu-prototype.md §3.5): time, links,
// the scheduler, and the test structure's throughput, latency, backpressure and determinism.
// Every expected cycle below is worked out by hand from the rules in link.hpp and pipeline.hpp.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pipeline.hpp"

#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 100'000'000;   // 100 us: far past any run here

// Producer -> consumer, directly, over one link.
struct Direct {
    Simulator sim;
    Domain& d;
    Producer& p;
    Consumer& c;
    Direct(std::uint64_t n, std::size_t capacity, Cycle latency, Cycle credit_latency)
        : d(sim.add_domain(Domain("clk", 1000))), p(sim.add<Producer>("producer", d, n)),
          c(sim.add<Consumer>("consumer", d)) {
        auto& l = sim.link<Beat>("p->c", p, c, capacity, latency, credit_latency);
        p.connect(l);
        c.connect(l);
    }
};

// Producer -> stage -> consumer, with the stage's latency, interval and depth.
struct Pipe {
    Simulator sim;
    Domain& d;
    Producer* p = nullptr;
    Stage* s = nullptr;
    Consumer* c = nullptr;
    // `reversed` adds the components in the opposite order.
    Pipe(std::uint64_t n, Cycle latency, Cycle interval, std::size_t depth, bool reversed = false,
         Domain dom = Domain("clk", 1000))
        : d(sim.add_domain(std::move(dom))) {
        if (reversed) {
            c = &sim.add<Consumer>("consumer", d);
            s = &sim.add<Stage>("stage", d, latency, interval, depth);
            p = &sim.add<Producer>("producer", d, n);
        } else {
            p = &sim.add<Producer>("producer", d, n);
            s = &sim.add<Stage>("stage", d, latency, interval, depth);
            c = &sim.add<Consumer>("consumer", d);
        }
        auto& a = sim.link<Beat>("p->s", *p, *s, 4);
        auto& b = sim.link<Beat>("s->c", *s, *c, 4);
        p->connect(a);
        s->connect(a, b);
        c->connect(b);
    }
};

}  // namespace

TEST_CASE("Domains: edges in picoseconds, exact for rational periods", "[cycle][core]") {
    const Domain array = Domain::from_ghz("array", 1.0);
    CHECK(array.edge(0) == 0);
    CHECK(array.edge(7) == 7000);
    const Domain banks = Domain::from_ghz("l2", 0.25);
    CHECK(banks.edge(3) == 12000);
    // LPDDR5's 3.2 GHz controller: 312.5 ps, so edge 5 is 1562.5 ps, reported as 1562; edge 10 is exact.
    const Domain dram = Domain::from_ghz("dram", 3.2);
    CHECK(dram.edge(5) == 1562);
    CHECK(dram.edge(10) == 3125);
    CHECK(dram.edge(3'200'000'000ULL) == 1'000'000'000'000ULL);   // one second, no overflow
    CHECK_THROWS(Domain("bad", 0));
}

TEST_CASE("Domains: one period, written three ways, has the same edges, and no edge overflows", "[cycle][core]") {
    // 312.5 ps as 10^12 / 3.2 10^9 (#549's review: the remainder product overflowed), 625 / 2, and
    // from_ghz(3.2) = 10^9 / 3.2 10^6.
    const Domain a("a", 1'000'000'000'000ULL, 3'200'000'000ULL), b("b", 625, 2);
    const Domain c = Domain::from_ghz("c", 3.2);
    for (const Domain* d : {&a, &b, &c}) {
        CAPTURE(d->name());
        CHECK(d->edge(20'000'000) == 6'250'000'000ULL);
        CHECK(d->edge(3) == 937);
    }
    // The scheduler's next edge, stepped, is edge(n) exactly, for every n.
    for (Domain d : {a, b, c, Domain("p", 1000, 3, 17)}) {
        CAPTURE(d.name());
        for (Cycle n = 0; n < 20'000; ++n) {
            REQUIRE(d.next_edge() == d.edge(n));
            d.advance();
        }
    }
    // (2^64 - 1) * 3 / 4 = 3 * 2^62 - 3/4: a 66-bit product, floored to 3 * 2^62 - 1.
    CHECK(mul_div(~0ULL, 3, 4) == 0xBFFF'FFFF'FFFF'FFFFULL);
    CHECK_THROWS_AS(mul_div(~0ULL, 2, 1), std::overflow_error);
    // Past 2^64 ps the clock refuses to advance, rather than wrapping to an earlier time (#549's review).
    Domain big("big", 1ULL << 63);
    big.advance();
    CHECK(big.next_edge() == 1ULL << 63);
    CHECK_THROWS_AS(big.advance(), std::overflow_error);
    CHECK(big.cycle() == 1);                           // unchanged by the refusal
    CHECK(big.next_edge() == 1ULL << 63);
    CHECK_THROWS_AS(big.edge(2), std::overflow_error);
    CHECK_THROWS_AS(Domain("phased", 1, 1, ~0ULL).edge(1), std::overflow_error);   // the phase's addition
}

TEST_CASE("Links: latency, credits, and their conservation", "[cycle][core]") {
    Domain d("clk", 1000);
    Link<int> l("l", d, 2, 3, 2);
    CHECK(l.can_push(0));
    l.push(0, 10);
    l.push(0, 11);
    CHECK_FALSE(l.can_push(0));                            // two credits spent
    CHECK_THROWS_WITH(l.push(0, 12), ContainsSubstring("push without a credit"));
    CHECK_FALSE(l.can_pop(2));                             // lands at 0 + 3
    CHECK(l.can_pop(3));
    l.check(3);
    CHECK(l.pop(3) == 10);                                 // its credit is back at 3 + 2
    CHECK_FALSE(l.can_push(4));
    CHECK(l.can_push(5));
    l.check(4);
    l.check(5);
    CHECK_THROWS_WITH(Link<int>("z", d, 0), ContainsSubstring("capacity"));
    CHECK_THROWS_WITH(Link<int>("z", d, 1, 0), ContainsSubstring("at least 1 cycle"));
}

TEST_CASE("The credit loop bounds a link's throughput: capacity / (latency + credit latency)", "[cycle][core]") {
    SECTION("one credit, a 2-cycle loop: one beat every 2 cycles") {
        Direct x(100, 1, 1, 1);
        REQUIRE(x.sim.run_until_idle(kLimit));
        REQUIRE(x.c.received.size() == 100);
        for (std::size_t i = 0; i < 100; ++i) CHECK(x.c.received[i].second == 1 + 2 * i);
    }
    SECTION("two credits cover the loop: one beat every cycle") {
        Direct x(100, 2, 1, 1);
        REQUIRE(x.sim.run_until_idle(kLimit));
        for (std::size_t i = 0; i < 100; ++i) CHECK(x.c.received[i].second == 1 + i);
    }
    SECTION("three credits, a 4-cycle loop: three beats every four cycles") {
        // Pushes at 0, 1, 2 land at 2, 3, 4; their credits are back at 4, 5, 6, so the next three
        // push at 4, 5, 6 and land at 6, 7, 8.
        Direct x(300, 3, 2, 2);
        REQUIRE(x.sim.run_until_idle(kLimit));
        REQUIRE(x.c.received.size() == 300);
        for (std::size_t i = 0; i < 300; ++i) CHECK(x.c.received[i].second == 2 + 4 * (i / 3) + i % 3);
        CHECK(x.c.received.back().second == 400);
    }
}

TEST_CASE("A pipeline's throughput is its slowest stage's, and its latency the sum", "[cycle][core]") {
    // The stage takes a beat every 3 cycles and holds it 5. Beat 0: pushed at 0, landed and
    // accepted at 1, offered at 6, landed and popped at 7. The stage accepts at 1, 4, 7, ...
    Pipe x(50, 5, 3, 4);
    REQUIRE(x.sim.run_until_idle(kLimit));
    x.c->finish(50);
    CHECK(x.c->received.front().second == 7);
    for (std::size_t i = 1; i < 50; ++i) CHECK(x.c->received[i].second - x.c->received[i - 1].second == 3);
    CHECK(x.c->received.back().second == 7 + 3 * 49);
    // The producer is paced by the stage, through the link's 4 credits: beats 0-3 take them at
    // 0-3; the stage's pops at 1 and 4 return credits at 2 and 5, so beats 4 and 5 push at 4 and 5;
    // from beat 6 on, one credit comes back every 3 cycles: 8, 11, 14, ...
    for (std::size_t i = 0; i < 6; ++i) CHECK(x.p->pushed_at[i] == i);
    for (std::size_t i = 6; i < 50; ++i) CHECK(x.p->pushed_at[i] == 8 + 3 * (i - 6));
}

TEST_CASE("Backpressure: a stalled consumer stops the producer, and no beat is lost or duplicated",
          "[cycle][core]") {
    Pipe x(200, 2, 1, 2);
    x.c->stall(20, 80);
    REQUIRE(x.sim.run_until_idle(kLimit));           // the consumer's check() fails on loss or reordering
    x.c->finish(200);
    std::size_t before = 0;
    for (const auto& [id, at] : x.c->received) before += at < 20;
    // While the consumer is stalled, the producer can push only into the two links' 8 slots and the
    // stage's 2: everything else waits for credits.
    std::size_t pushed_by_79 = 0;
    for (const Cycle at : x.p->pushed_at) pushed_by_79 += at < 80;
    CHECK(pushed_by_79 <= before + 4 + 4 + 2);
    // Nothing is popped in the stall, and popping resumes at its end.
    for (const auto& [id, at] : x.c->received) CHECK((at < 20 || at >= 80));
    CHECK(x.c->received[before].second == 80);
}

TEST_CASE("Determinism: two runs agree, and the order components are added in does not matter",
          "[cycle][core]") {
    Pipe a(100, 3, 2, 2), b(100, 3, 2, 2), r(100, 3, 2, 2, true);
    a.c->stall(30, 45);
    b.c->stall(30, 45);
    r.c->stall(30, 45);
    REQUIRE(a.sim.run_until_idle(kLimit));
    REQUIRE(b.sim.run_until_idle(kLimit));
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(a.c->received == b.c->received);
    CHECK(a.p->pushed_at == b.p->pushed_at);
    CHECK(a.c->received == r.c->received);
    CHECK(a.p->pushed_at == r.p->pushed_at);
    // A reset run repeats itself.
    const auto first = a.c->received;
    a.sim.reset();
    a.c->stall(30, 45);
    REQUIRE(a.sim.run_until_idle(kLimit));
    CHECK(a.c->received == first);
}

TEST_CASE("Domains run side by side, each on its own clock", "[cycle][core]") {
    // The same pipeline at 1 GHz and at 250 MHz takes the same number of cycles; at 250 MHz each
    // is four times as long in picoseconds.
    Pipe fast(20, 5, 3, 4, false, Domain::from_ghz("fast", 1.0));
    Pipe slow(20, 5, 3, 4, false, Domain::from_ghz("slow", 0.25));
    REQUIRE(fast.sim.run_until_idle(kLimit));
    REQUIRE(slow.sim.run_until_idle(kLimit));
    CHECK(fast.c->received == slow.c->received);
    // One simulator with both: its edges interleave, and each pipeline keeps its own cycles.
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
    CHECK(cf.received == cs.received);                   // the same cycles in each domain
    CHECK(f.edge(cf.received.back().second) == 39'000);   // cycle 39 at 1 GHz
    CHECK(s.edge(cs.received.back().second) == 156'000);  // cycle 39 at 250 MHz
    // A link across the two is a crossing, which this step refuses.
    CHECK_THROWS_WITH(sim.link<Beat>("x", pf, cs, 1), ContainsSubstring("crossing"));
}

TEST_CASE("A component or link outside the simulator's domains is refused", "[cycle][core]") {
    Simulator sim;
    Domain other("other", 1000);
    CHECK_THROWS_WITH(sim.add<Consumer>("c", other), ContainsSubstring("add_domain first"));
    // Two components of another simulator share a domain, but not this simulator's (#549's review).
    Simulator elsewhere;
    Domain& d = elsewhere.add_domain(Domain("clk", 1000));
    auto& p = elsewhere.add<Producer>("p", d, 1);
    auto& c = elsewhere.add<Consumer>("c", d);
    CHECK_THROWS_WITH(sim.link<Beat>("p->c", p, c, 1), ContainsSubstring("link p->c"));
}

namespace {
// A stage that loses beat `drop`: it pops it and never forwards it.
class Dropper : public Component {
public:
    Dropper(std::string name, Domain& d, std::uint64_t drop) : Component(std::move(name), d), drop_(drop) {}
    void connect(Link<Beat>& in, Link<Beat>& out) {
        in_ = &in;
        out_ = &out;
    }
    void evaluate(Cycle now) override {
        if (!in_->can_pop(now) || !out_->can_push(now)) return;
        Beat b = in_->pop(now);
        if (b.id != drop_) out_->push(now, std::move(b));
    }
    [[nodiscard]] bool idle() const override { return true; }

private:
    Link<Beat>* in_ = nullptr;
    Link<Beat>* out_ = nullptr;
    std::uint64_t drop_;
};
}  // namespace

TEST_CASE("The test structure catches a lost beat, by name and cycle", "[cycle][core]") {
    struct Lossy {
        Simulator sim;
        Consumer* c = nullptr;
        Lossy(std::uint64_t n, std::uint64_t drop) {
            Domain& d = sim.add_domain(Domain("clk", 1000));
            auto& p = sim.add<Producer>("producer", d, n);
            auto& x = sim.add<Dropper>("dropper", d, drop);
            c = &sim.add<Consumer>("consumer", d);
            auto& a = sim.link<Beat>("p->x", p, x, 2);
            auto& b = sim.link<Beat>("x->c", x, *c, 2);
            p.connect(a);
            x.connect(a, b);
            c->connect(b);
        }
    };
    SECTION("a beat in the middle: the next one arrives out of place") {
        Lossy l(20, 7);
        CHECK_THROWS_WITH(l.sim.run_until_idle(kLimit), ContainsSubstring("consumer at cycle") &&
                                                            ContainsSubstring("beat 8 arrived as number 7"));
    }
    // No later beat shows the gap (#549's review): the count at quiescence does.
    SECTION("the last beat") {
        Lossy l(20, 19);
        REQUIRE(l.sim.run_until_idle(kLimit));
        CHECK_THROWS_WITH(l.c->finish(20), ContainsSubstring("19 of 20 beats arrived"));
    }
    SECTION("the only beat") {
        Lossy l(1, 0);
        REQUIRE(l.sim.run_until_idle(kLimit));
        CHECK_THROWS_WITH(l.c->finish(1), ContainsSubstring("0 of 1 beats arrived"));
    }
    SECTION("none lost") {
        Lossy l(20, 99);
        REQUIRE(l.sim.run_until_idle(kLimit));
        CHECK_NOTHROW(l.c->finish(20));
    }
}
