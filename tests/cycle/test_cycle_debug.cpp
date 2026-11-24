// ============================================================================
// tests/cycle/test_cycle_debug.cpp
// The cycle core's debug support, step 0c (#535; docs/plans/cycle-based-kpu-prototype.md §3.3):
// the deadlock detector, invariant violations, the state dump, and the trace.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pipeline.hpp"

#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;

// A node of a ring: it starts with `beats`, forwards its oldest when it holds a credit, and
// accepts a landed beat while it holds fewer than `depth`.
class Node : public Component {
public:
    Node(std::string name, Domain& d, std::size_t beats, std::size_t depth)
        : Component(std::move(name), d), depth_(depth) {
        for (std::size_t i = 0; i < beats; ++i) {
            Beat b;
            b.id = i;
            held_.push_back(std::move(b));
        }
    }
    void connect(Link<Beat>& in, Link<Beat>& out) {
        in_ = &in;
        out_ = &out;
    }
    void evaluate(Cycle now) override {
        bool moved = false;
        if (!held_.empty()) {
            if (out_->can_push(now)) {
                out_->push(now, std::move(held_.front()));
                held_.pop_front();
                moved = true;
            } else {
                counters().stall(Stall::NoCredit);
            }
        }
        if (held_.size() < depth_ && in_->can_pop(now)) {
            held_.push_back(in_->pop(now));
            moved = true;
        }
        if (moved) counters().busy();
    }
    [[nodiscard]] bool idle() const override { return held_.empty(); }
    [[nodiscard]] std::string describe() const override {
        return "holds " + std::to_string(held_.size()) + " of " + std::to_string(depth_);
    }

private:
    Link<Beat>* in_ = nullptr;
    Link<Beat>* out_ = nullptr;
    std::size_t depth_;
    std::deque<Beat> held_;
};

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
        counters().busy();
    }
    [[nodiscard]] bool idle() const override { return true; }

private:
    Link<Beat>* in_ = nullptr;
    Link<Beat>* out_ = nullptr;
    std::uint64_t drop_;
};

// Three nodes in a ring over links of capacity 1.
struct Ring {
    Simulator sim;
    Ring(std::size_t beats, std::size_t depth, std::uint64_t window) {
        sim.set_deadlock_window(window);
        Domain& d = sim.add_domain(Domain("clk", 1000));
        Node* n[3];
        for (int i = 0; i < 3; ++i) n[i] = &sim.add<Node>("n" + std::to_string(i), d, beats, depth);
        Link<Beat>* l[3];
        for (int i = 0; i < 3; ++i)
            l[i] = &sim.link<Beat>("n" + std::to_string(i) + "->n" + std::to_string((i + 1) % 3), *n[i], *n[(i + 1) % 3], 1);
        for (int i = 0; i < 3; ++i) n[i]->connect(*l[(i + 2) % 3], *l[i]);
    }
};

}  // namespace

TEST_CASE("The deadlock detector names a credit cycle", "[cycle][debug]") {
    // Each node starts with 3 beats and holds at most 2; at cycle 0 each pushes one, filling its
    // link. From cycle 1 every node is full (it cannot accept) and every link is full (it cannot
    // push): a credit cycle, the reason the NoC needs its bubble rule. Nothing moves again.
    Ring r(3, 2, 20);
    try {
        r.sim.run_until_idle(kLimit);
        FAIL("the ring should deadlock");
    } catch (const Deadlock& e) {
        CHECK(e.blocked() == std::vector<std::string>{"n0", "n1", "n2"});
        const std::string what = e.what();
        CHECK_THAT(what, ContainsSubstring("no progress for 20 edges"));
        CHECK_THAT(what, ContainsSubstring("n0 stalled (no credit)"));
        CHECK_THAT(what, ContainsSubstring("link n0->n1 credits 0/1, in flight 0, buffered 1, returning 0"));
        CHECK_THAT(e.dump(), ContainsSubstring("component n1: stalled (no credit)") &&
                                 ContainsSubstring("holds 2 of 2"));
        // It stopped 20 edges after the last progress (cycle 0): at cycle 20.
        CHECK(r.sim.now() == 20'000);
    }
}

TEST_CASE("The deadlock detector does not fire on a live system", "[cycle][debug]") {
    SECTION("a ring with room circulates forever") {
        Ring r(2, 2, 20);                           // 6 beats in 9 slots: always room to move
        CHECK(r.sim.run_until([] { return false; }, 5'000'000) == false);   // ran to the limit
        CHECK(r.sim.now() > 5'000'000);
    }
    SECTION("a latency longer than the window is busy, not deadlocked") {
        Simulator sim;
        sim.set_deadlock_window(100);
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 3);
        auto& s = sim.add<Stage>("stage", d, 1000);       // holds each beat 1000 cycles
        auto& c = sim.add<Consumer>("consumer", d);
        auto& a = sim.link<Beat>("p->s", p, s, 1);
        auto& b = sim.link<Beat>("s->c", s, c, 1);
        p.connect(a);
        s.connect(a, b);
        c.connect(b);
        REQUIRE(sim.run_until_idle(kLimit));
        c.finish(3);
    }
    SECTION("a consumer held by an outside control longer than the window") {
        Simulator sim;
        sim.set_deadlock_window(100);
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 10);
        auto& c = sim.add<Consumer>("consumer", d);
        auto& l = sim.link<Beat>("p->c", p, c, 1);
        p.connect(l);
        c.connect(l);
        c.stall(5, 1005);
        REQUIRE(sim.run_until_idle(kLimit));
        c.finish(10);
    }
    SECTION("off") {
        Ring r(3, 2, 0);
        CHECK(r.sim.run_until_idle(100'000) == false);  // still stuck, but nothing stops it
    }
}

TEST_CASE("A failing invariant stops the run with the cycle and the component", "[cycle][debug]") {
    SECTION("a component's own check: a lost beat") {
        // A stage that drops beat 3: the consumer's arrival-order check fails when beat 4 arrives.
        Simulator sim;
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 10);
        auto& x = sim.add<Dropper>("dropper", d, 3);
        auto& c = sim.add<Consumer>("consumer", d);
        auto& a = sim.link<Beat>("p->x", p, x, 2);
        auto& b = sim.link<Beat>("x->c", x, c, 2);
        p.connect(a);
        x.connect(a, b);
        c.connect(b);
        try {
            sim.run_until_idle(kLimit);
            FAIL("the consumer's check should fail");
        } catch (const InvariantViolation& e) {
            CHECK(e.where() == "consumer");
            CHECK(e.domain() == "clk");
            CHECK_THAT(e.what(), ContainsSubstring("consumer (clk cycle ") &&
                                     ContainsSubstring("beat 4 arrived as number 3"));
            CHECK(e.cycle() == c.received.back().second);   // the cycle beat 4 arrived
            CHECK_THAT(e.dump(), ContainsSubstring("component consumer: busy") && ContainsSubstring("received 4"));
        }
    }
    SECTION("an invariant hook") {
        Simulator sim;
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 50);
        auto& c = sim.add<Consumer>("consumer", d, 3);    // pops every 3 cycles: the link fills
        auto& l = sim.link<Beat>("p->c", p, c, 4);
        p.connect(l);
        c.connect(l);
        sim.add_invariant("at most 3 beats waiting", [&] {
            if (l.buffered(d.cycle() - 1) > 3) throw std::logic_error("4 beats waiting");
        });
        try {
            sim.run_until_idle(kLimit);
            FAIL("the hook should fail");
        } catch (const InvariantViolation& e) {
            CHECK(e.where() == "invariant at most 3 beats waiting");
            CHECK_THAT(e.what(), ContainsSubstring("4 beats waiting"));
            CHECK_THAT(e.dump(), ContainsSubstring("link p->c: credits 0/4"));
        }
    }
    SECTION("checks off: the hook is not run") {
        Simulator sim;
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 5);
        auto& c = sim.add<Consumer>("consumer", d);
        auto& l = sim.link<Beat>("p->c", p, c, 1);
        p.connect(l);
        c.connect(l);
        sim.add_invariant("always fails", [] { throw std::logic_error("no"); });
        sim.set_checks(false);
        CHECK(sim.run_until_idle(kLimit));
    }
}

TEST_CASE("The state dump shows every domain, component and link", "[cycle][debug]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain("clk", 1000));
    auto& p = sim.add<Producer>("producer", d, 10);
    auto& s = sim.add<Stage>("stage", d, 4, 1, 2);
    auto& c = sim.add<Consumer>("consumer", d);
    auto& a = sim.link<Beat>("p->s", p, s, 2);
    auto& b = sim.link<Beat>("s->c", s, c, 2);
    p.connect(a);
    s.connect(a, b);
    c.connect(b);
    for (int i = 0; i < 4; ++i) sim.step();
    // After cycles 0-3: the producer pushed beats 0-3 (credits back at 2 and 3 from the stage's
    // pops at 1 and 2); the stage holds beats 0 and 1, the oldest ready at cycle 5.
    const std::string dump = sim.dump();
    CHECK_THAT(dump, ContainsSubstring("domain clk: cycle 4, next edge 4000 ps"));
    CHECK_THAT(dump, ContainsSubstring("component producer: busy; A 0, C 4, held 0; pushed 4 of 10"));
    CHECK_THAT(dump, ContainsSubstring("component stage: busy; A 2, C 0, held 2; holds 2 of 2, oldest beat 0 ready at cycle 5"));
    CHECK_THAT(dump, ContainsSubstring("component consumer: idle"));
    CHECK_THAT(dump, ContainsSubstring("link p->s: credits 0/2"));
    CHECK_THAT(dump, ContainsSubstring("link s->c: credits 2/2, in flight 0, buffered 0, returning 0"));
}

TEST_CASE("The trace: states, beats and occupancy, bounded", "[cycle][debug][trace]") {
    const auto dir = std::filesystem::temp_directory_path();
    auto build = [](Simulator& sim) {
        Domain& d = sim.add_domain(Domain("clk", 1000));
        auto& p = sim.add<Producer>("producer", d, 20);
        auto& c = sim.add<Consumer>("consumer", d);
        auto& l = sim.link<Beat>("p->c", p, c, 1);
        p.connect(l);
        c.connect(l);
        return &c;
    };
    Tracer t;
    Simulator sim;
    sim.attach(t);
    build(sim)->stall(10, 15);
    REQUIRE(sim.run_until_idle(kLimit));
    const auto path = (dir / "cycle_trace_unit.json").string();
    t.write(path);
    CHECK(t.dropped() == 0);
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();
    CHECK_THAT(json, ContainsSubstring("\"traceEvents\""));
    CHECK_THAT(json, ContainsSubstring(R"j("name":"thread_name","args":{"name":"producer (clk)"})j"));
    CHECK_THAT(json, ContainsSubstring(R"j("name":"stall: no credit")j"));
    CHECK_THAT(json, ContainsSubstring(R"j("name":"stall: held")j"));
    CHECK_THAT(json, ContainsSubstring(R"j("ph":"b")j") && ContainsSubstring(R"j("beat":"beat 19")j"));
    CHECK_THAT(json, ContainsSubstring(R"j("name":"p->c occupancy")j"));
    std::filesystem::remove(path);
    // Bounded: past its limit the trace counts what it dropped, and says so.
    Tracer small(8);
    Simulator s2;
    s2.attach(small);
    build(s2);
    REQUIRE(s2.run_until_idle(kLimit));
    CHECK(small.events() == 8);
    CHECK(small.dropped() > 0);
    const auto p2 = (dir / "cycle_trace_small.json").string();
    small.write(p2);
    std::ifstream in2(p2);
    std::stringstream s2s;
    s2s << in2.rdbuf();
    CHECK_THAT(s2s.str(), ContainsSubstring("\"dropped\":" + std::to_string(small.dropped())));
    std::filesystem::remove(p2);
    // Attached after the model is built: refused, since its components would have no tracks.
    Tracer late;
    CHECK_THROWS_WITH(s2.attach(late), ContainsSubstring("before adding"));
}
