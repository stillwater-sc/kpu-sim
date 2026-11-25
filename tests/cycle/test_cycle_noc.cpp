// ============================================================================
// tests/cycle/test_cycle_noc.cpp
// The NoC on the cycle core, step 1c (#539; docs/plans/cycle-based-kpu-prototype.md §3.4, §4.3
// rung 3).
//
// First the fabric on its own, on the T4's and the T16's folded tori: every port reaches every
// hub's L3 and every hub's L3 reaches every port, each beat exactly once and only there. Then rung
// 3: DMA engines and L3 tiles over the NoC, each engine with its own LPDDR5 channel, every tile
// checked bit for bit in its slot and in DRAM, the slot credits and the bubble rule (every ring
// keeps a free slot) checked every cycle. Then the bubble rule's reason: with it off, saturating
// traffic fills a ring. Then the counters, determinism and the refusals.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "l3_dma_rig.hpp"

#include <lpddr5_configs.hpp>

#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/noc/noc_fabric.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using sw::kpu::program::platform::ArrayLayout;
using sw::kpu::program::platform::DeviceSpecification;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 2'000'000'000;   // 2 ms

DeviceSpecification device(const std::string& file) {
    return sw::kpu::program::platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file)
        .device(0);
}
ArrayLayout layout(const std::string& file) { return *ArrayLayout::of(device(file)); }

// Pushes its beats onto a link, one a cycle while it has a credit.
class Pusher : public Component {
public:
    Pusher(std::string name, Domain& d, std::vector<L3Beat> beats) : Component(std::move(name), d), beats_(std::move(beats)) {}
    void connect(Link<L3Beat>& out) { out_ = &out; }
    void evaluate(Cycle now) override {
        if (next_ == beats_.size()) return;
        if (out_->can_push(now)) {
            out_->push(now, beats_[next_++]);
            counters().busy();
        } else {
            counters().stall(Stall::NoCredit);
        }
    }
    [[nodiscard]] bool idle() const override { return next_ == beats_.size(); }
    void reset() override { next_ = 0; }

private:
    std::vector<L3Beat> beats_;
    std::size_t next_ = 0;
    Link<L3Beat>* out_ = nullptr;
};

// Pops every beat that lands on a link, from cycle `from`, and keeps them.
class Catcher : public Component {
public:
    Catcher(std::string name, Domain& d, Cycle from = 0) : Component(std::move(name), d), from_(from) {}
    void connect(Link<L3Beat>& in) { in_ = &in; }
    void evaluate(Cycle now) override {
        if (now < from_) {
            if (in_->can_pop(now)) counters().stall(Stall::External);
            return;
        }
        if (in_->can_pop(now)) {
            got.push_back(in_->pop(now));
            counters().busy();
        }
    }
    [[nodiscard]] bool idle() const override { return true; }
    void reset() override { got.clear(); }
    std::vector<L3Beat> got;

private:
    Cycle from_;
    Link<L3Beat>* in_ = nullptr;
};

// `n` beats from source `src` (in `ticket`), each with its index in `offset`.
std::vector<L3Beat> stream(std::uint64_t src, std::size_t n) {
    std::vector<L3Beat> v;
    for (std::size_t i = 0; i < n; ++i) {
        L3Beat b;
        b.id = i + 1;
        b.ticket = src;
        b.offset = static_cast<std::uint32_t>(i);
        b.bytes.assign(kL3BeatBytes, static_cast<std::uint8_t>(src * 7 + i));
        v.push_back(std::move(b));
    }
    return v;
}

// Every catcher got exactly the beats meant for it, each once and in its source's order.
void check_caught(const std::vector<Catcher*>& catchers, const std::map<std::uint64_t, std::size_t>& source_to_catcher,
                  std::size_t n) {
    std::map<std::size_t, std::map<std::uint64_t, std::vector<std::uint32_t>>> seen;
    for (std::size_t c = 0; c < catchers.size(); ++c)
        for (const auto& b : catchers[c]->got) seen[c][b.ticket].push_back(b.offset);
    for (const auto& [src, c] : source_to_catcher) {
        CAPTURE(src, c);
        REQUIRE(seen[c].count(src));
        const auto& offs = seen[c][src];
        REQUIRE(offs.size() == n);
        for (std::size_t i = 0; i < n; ++i) CHECK(offs[i] == i);   // one path per pair: in order
        for (std::size_t o = 0; o < catchers.size(); ++o)
            if (o != c) CHECK(!seen[o].count(src));
    }
}

// Rung 3's rig: DMA engines on ports, each with its own LPDDR5 channel and its own L3 tile at a
// hub, over the NoC of `file`'s layout.
struct NocRig {
    struct Pair {
        NocDim port, hub;
    };
    Simulator sim;
    Domain& dram;
    Domain& kpu;
    std::unique_ptr<NocFabric> noc;
    std::vector<Lpddr5Channel*> channels;
    std::vector<DmaEngine*> dmas;
    std::vector<L3Tile*> l3s;

    NocRig(const std::string& file, const std::vector<Pair>& pairs, const std::vector<L3Program>& programs,
           NocFabric::Config cfg, std::uint32_t slots = 2, std::size_t window = 8)
        : dram(sim.add_domain(Domain::from_ghz("dram", 3.2))), kpu(sim.add_domain(Domain::from_ghz("kpu", 1.0))) {
        noc = std::make_unique<NocFabric>(sim, kpu, layout(file), cfg);
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            const std::string n = std::to_string(i);
            auto& ch = sim.add<Lpddr5Channel>("lpddr5 " + n, dram, sw::kpu::patterns::lpddr5::single_channel_config());
            auto& dma = sim.add<DmaEngine>("dma " + n, kpu, DmaEngine::Config{slots, 4096, window, 4}, programs[i]);
            auto& l3 = sim.add<L3Tile>("l3 " + n, kpu, L3Tile::Config{slots, 4096, 4}, programs[i]);
            auto& req = sim.crossing<DramRequest>("dma " + n + " -> lpddr5", dma, ch, 4);
            auto& rsp = sim.crossing<DramResponse>("lpddr5 -> dma " + n, ch, dma, 4);
            auto& cr = sim.link<SlotCredit>("l3 " + n + " credits", l3, dma, slots);
            const auto e = noc->attach_engine(pairs[i].port, dma, pairs[i].hub);
            const auto t = noc->attach_l3(pairs[i].hub, l3, pairs[i].port, e.engine);
            ch.connect(req, rsp);
            dma.connect(req, rsp, e.to_noc, e.from_noc, cr);
            l3.connect(t.to_l3, t.from_l3, cr);
            channels.push_back(&ch);
            dmas.push_back(&dma);
            l3s.push_back(&l3);
            sim.add_invariant("slot credits " + n, [&dma, &l3] { check_slot_credits(dma, l3); });
        }
    }
};

L3Program ring_program(std::uint32_t tiles, std::uint32_t bytes, std::uint32_t slots) {
    L3Program p;
    for (std::uint32_t t = 0; t < tiles; ++t) {
        p.push_back(l3_load(t % slots, 0x10000 + std::uint64_t(t) * bytes, bytes));
        p.push_back(l3_store(t % slots, 0x400000 + std::uint64_t(t) * bytes, bytes));
        p.push_back(l3_release(t % slots));
    }
    return p;
}

std::uint32_t seed(std::size_t engine, std::size_t ticket) { return static_cast<std::uint32_t>(engine * 1000 + ticket); }

// Preloads every engine's tiles and watches every slot: the bytes of each tile when it is READY.
struct Watch {
    std::vector<std::map<std::uint64_t, std::vector<std::uint8_t>>> at_ready;
};
void preload_and_watch(NocRig& r, const std::vector<L3Program>& programs, Watch& w) {
    w.at_ready.assign(programs.size(), {});
    for (std::size_t i = 0; i < programs.size(); ++i)
        for (std::size_t a = 0; a < programs[i].size(); ++a)
            if (programs[i][a].kind == L3Action::Kind::Load)
                r.channels[i]->preload(programs[i][a].dram_address, tile_bytes(seed(i, a), programs[i][a].bytes));
    r.sim.add_invariant("watch", [&r, &w] {
        for (std::size_t i = 0; i < r.l3s.size(); ++i)
            for (std::uint32_t s = 0; s < r.l3s[i]->config().slots; ++s)
                if (r.l3s[i]->ready(s) && !w.at_ready[i].count(r.l3s[i]->ticket(s)))
                    w.at_ready[i][r.l3s[i]->ticket(s)] = r.l3s[i]->peek(s);
    });
}
void check_data(NocRig& r, const std::vector<L3Program>& programs, const Watch& w) {
    for (std::size_t i = 0; i < programs.size(); ++i) {
        CAPTURE(i);
        std::map<std::uint32_t, std::size_t> last;
        for (std::size_t a = 0; a < programs[i].size(); ++a) {
            const L3Action& x = programs[i][a];
            if (x.kind == L3Action::Kind::Load) {
                last[x.slot] = a;
                REQUIRE(w.at_ready[i].count(a));
                CHECK(w.at_ready[i].at(a) == tile_bytes(seed(i, a), x.bytes));
            } else if (x.kind == L3Action::Kind::Store) {
                CHECK(r.channels[i]->peek(x.dram_address, x.bytes) == tile_bytes(seed(i, last.at(x.slot)), x.bytes));
            }
        }
    }
}

}  // namespace

TEST_CASE("The NoC delivers every port's beats to every hub's L3, once and only there", "[cycle][noc]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json"}) {
        CAPTURE(file);
        Simulator sim;
        Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
        NocFabric noc(sim, d, layout(file), {});
        const NocTopology& t = noc.topology();
        std::vector<Catcher*> l3s;
        for (NocDim h = 0; h < t.hub_count(); ++h) {
            auto& c = sim.add<Catcher>("l3 " + std::to_string(h), d);
            const auto links = noc.attach_l3(h, c, 0, 0);
            c.connect(links.to_l3);
            l3s.push_back(&c);
        }
        // One engine per (port, hub) pair, each streaming 6 beats to its hub.
        std::map<std::uint64_t, std::size_t> where;
        std::vector<Catcher*> sinks;
        for (NocDim k = 0; k < t.port_count(); ++k)
            for (NocDim h = 0; h < t.hub_count(); ++h) {
                const std::uint64_t src = k * 100 + h;
                auto& p = sim.add<Pusher>("engine " + std::to_string(src), d, stream(src, 6));
                auto& s = sim.add<Catcher>("engine sink " + std::to_string(src), d);
                const auto e = noc.attach_engine(k, p, h);
                p.connect(e.to_noc);
                s.connect(e.from_noc);
                sinks.push_back(&s);
                where[src] = h;
            }
        REQUIRE(sim.run_until_idle(kLimit));
        check_caught(l3s, where, 6);
        for (const auto* s : sinks) CHECK(s->got.empty());
        for (const auto& r : noc.ring_peaks()) CHECK(r.peak < r.slots);
    }
}

TEST_CASE("The NoC delivers every hub's L3 beats to every port's engine, once and only there", "[cycle][noc]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json"}) {
        CAPTURE(file);
        const ArrayLayout L = layout(file);
        const NocTopology probe(L);
        // Each hub stores to one port; rotate which, so every pair is crossed.
        for (NocDim shift = 0; shift < probe.port_count(); ++shift) {
            CAPTURE(shift);
            Simulator sim;
            Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
            NocFabric noc(sim, d, L, {});
            const NocTopology& t = noc.topology();
            std::vector<Catcher*> engines;
            std::vector<NocDim> engine_of_port(t.port_count());
            for (NocDim k = 0; k < t.port_count(); ++k) {
                auto& c = sim.add<Catcher>("engine " + std::to_string(k), d);
                const auto e = noc.attach_engine(k, c, 0);
                c.connect(e.from_noc);
                engine_of_port[k] = e.engine;
                engines.push_back(&c);
            }
            std::map<std::uint64_t, std::size_t> where;
            for (NocDim h = 0; h < t.hub_count(); ++h) {
                const NocDim k = static_cast<NocDim>((h + shift) % t.port_count());
                auto& p = sim.add<Pusher>("l3 " + std::to_string(h), d, stream(h, 6));
                const auto links = noc.attach_l3(h, p, k, engine_of_port[k]);
                p.connect(links.from_l3);
                where[h] = k;
            }
            REQUIRE(sim.run_until_idle(kLimit));
            check_caught(engines, where, 6);
        }
    }
}

TEST_CASE("Rung 3: two DMA engines and two L3 tiles over the T4's NoC, data in the right slots", "[cycle][noc][dma]") {
    // Engine 0 on port 1 (row0.E) and engine 1 on port 2 (col0.N); each engine's L3 tile at the
    // hub across the fold or at its own end, so traffic turns, crosses, and shares the rings.
    using P = NocRig::Pair;
    const auto [pairs, label] = GENERATE(
        std::make_pair(std::vector<P>{{1, 1}, {2, 0}}, "each L3 across the fold"),
        std::make_pair(std::vector<P>{{1, 0}, {2, 1}}, "each L3 at the port's own end"),
        std::make_pair(std::vector<P>{{0, 1}, {1, 0}}, "both engines on the row loop"));
    CAPTURE(label);
    const std::vector<L3Program> programs = {ring_program(8, 1024, 2), ring_program(8, 1024, 2)};
    NocRig r("kpu_t4.json", pairs, programs, NocFabric::Config::from(device("kpu_t4.json")));
    Watch w;
    preload_and_watch(r, programs, w);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_data(r, programs, w);
    for (std::size_t i = 0; i < 2; ++i) {
        CHECK(r.dmas[i]->dma().loads == 8);
        CHECK(r.dmas[i]->dma().stores == 8);
        CHECK(r.dmas[i]->held_count() == 2);
    }
    // The NoC carried every beat: 8 tiles x 16 beats each way, per engine.
    std::uint64_t injected = 0, ejected = 0, delivered = 0, from_l3 = 0;
    for (NocDim k = 0; k < r.noc->topology().port_count(); ++k) {
        injected += r.noc->port(k).noc().injected;
        ejected += r.noc->port(k).noc().ejected;
    }
    for (NocDim h = 0; h < r.noc->topology().hub_count(); ++h) {
        delivered += r.noc->hub(h).noc().delivered;
        from_l3 += r.noc->hub(h).noc().from_l3;
    }
    CHECK(injected == 2 * 128);
    CHECK(delivered == 2 * 128);
    CHECK(from_l3 == 2 * 128);
    CHECK(ejected == 2 * 128);
}

TEST_CASE("Eight engines and eight L3 tiles over the T16's NoC, its rings at two flits a queue", "[cycle][noc][dma]") {
    // Every port has an engine, every hub an L3 tile, each engine's tile three hubs on from
    // its own (h = 3k + 5 mod 8, a permutation): the routes turn, cross and contend.
    std::vector<NocRig::Pair> pairs;
    std::vector<L3Program> programs;
    for (NocDim k = 0; k < 8; ++k) {
        pairs.push_back({k, static_cast<NocDim>((3 * k + 5) % 8)});
        programs.push_back(ring_program(6, 2048, 2));
    }
    NocRig r("kpu_t16.json", pairs, programs, NocFabric::Config::from(device("kpu_t16.json")), 2, 16);
    Watch w;
    preload_and_watch(r, programs, w);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_data(r, programs, w);
    std::size_t busiest = 0;
    for (const auto& ring : r.noc->ring_peaks()) {
        CAPTURE(ring.label);
        CHECK(ring.peak < ring.slots);
        busiest = std::max(busiest, ring.peak);
    }
    CHECK(busiest >= 2);   // the rings were used, more than one flit at a time
    // Per-link utilization and occupancy, from the core's counters: every NoC link moved flits,
    // and Little's law holds on each (link.hpp's counters, report.hpp's table).
    const auto rows = report(r.sim);
    std::size_t noc_links = 0;
    std::uint64_t flits = 0;
    for (const auto& row : rows)
        if (row.name.rfind("ch", 0) == 0) {
            ++noc_links;
            CHECK(row.c.completions == row.c.arrivals);
            flits += row.c.completions;
        }
    // A link per channel, and a second on each of the two fold channels of every port.
    CHECK(noc_links == r.noc->topology().channels().size() + 2 * r.noc->topology().port_count());
    CHECK(flits > 8 * 6 * 32 * 2);   // 8 engines x 6 tiles x 32 beats, each way, over several hops
}

TEST_CASE("Without the bubble rule, saturating traffic fills a ring", "[cycle][noc]") {
    // The T16 with every port streaming to every hub and every hub's L3 streaming back, the L3
    // tiles holding what reaches them for the first 3000 cycles, so the rings back up. With the
    // rule, every ring keeps a free slot through it, and everything arrives once the tiles drain;
    // without it, entering flits take the last slots and a ring fills.
    auto run = [](bool bubble, std::size_t queue) {
        Simulator sim;
        sim.set_deadlock_window(2000);
        Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
        NocFabric::Config c;
        c.bubble = bubble;
        c.ring_queue = queue;
        auto noc = std::make_unique<NocFabric>(sim, d, layout("kpu_t16.json"), c);
        const NocTopology& t = noc->topology();
        std::vector<NocDim> engine_of_port;
        for (NocDim k = 0; k < t.port_count(); ++k)
            for (NocDim h = 0; h < t.hub_count(); ++h) {
                const std::string n = std::to_string(k) + "." + std::to_string(h);
                auto& p = sim.add<Pusher>("engine " + n, d, stream(k * 100 + h, 200));
                auto& s = sim.add<Catcher>("engine sink " + n, d);
                const auto e = noc->attach_engine(k, p, h);
                p.connect(e.to_noc);
                s.connect(e.from_noc);
                if (h == 0) engine_of_port.push_back(e.engine);
            }
        for (NocDim h = 0; h < t.hub_count(); ++h) {
            auto& p = sim.add<Pusher>("l3 out " + std::to_string(h), d, stream(100 + h, 400));
            auto& c = sim.add<Catcher>("l3 " + std::to_string(h), d, 3000);
            const NocDim k = static_cast<NocDim>((h + 3) % t.port_count());
            const auto links = noc->attach_l3(h, c, k, engine_of_port[k]);
            p.connect(links.from_l3);
            c.connect(links.to_l3);
        }
        const bool ok = sim.run_until_idle(kLimit);
        return ok;
    };
    SECTION("with it: every ring keeps a slot") { CHECK(run(true, 2)); }
    SECTION("without it: the check finds a full ring") {
        try {
            run(false, 2);
            FAIL("a ring should fill");
        } catch (const InvariantViolation& e) {
            CHECK(e.where() == "invariant noc bubble: every ring keeps a free slot");
            CHECK_THAT(e.what(), ContainsSubstring("no bubble left"));
        }
    }
}

TEST_CASE("Two runs over the NoC, and a run after reset, are identical", "[cycle][noc][dma]") {
    const std::vector<L3Program> programs = {ring_program(4, 1024, 2), ring_program(4, 1024, 2)};
    auto spans = [](const NocRig& r) {
        std::vector<std::pair<Cycle, Cycle>> v;
        for (const auto* d : r.dmas)
            for (const auto& s : d->spans()) v.emplace_back(s.started, *s.completed);
        return v;
    };
    const auto cfg = NocFabric::Config::from(device("kpu_t4.json"));
    NocRig a("kpu_t4.json", {{1, 1}, {2, 0}}, programs, cfg), b("kpu_t4.json", {{1, 1}, {2, 0}}, programs, cfg);
    Watch wa, wb;
    preload_and_watch(a, programs, wa);
    preload_and_watch(b, programs, wb);
    REQUIRE(a.sim.run_until_idle(kLimit));
    REQUIRE(b.sim.run_until_idle(kLimit));
    CHECK(a.sim.now() == b.sim.now());
    CHECK(spans(a) == spans(b));
    const auto first = spans(a);
    a.sim.reset();
    for (auto& m : wa.at_ready) m.clear();
    REQUIRE(a.sim.run_until_idle(kLimit));
    CHECK(spans(a) == first);
    check_data(a, programs, wa);
}

TEST_CASE("The NoC refuses what it cannot build", "[cycle][noc]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    NocFabric::Config one;
    one.ring_queue = 1;
    CHECK_THROWS_WITH(NocFabric(sim, d, layout("kpu_t4.json"), one), ContainsSubstring("cannot keep a bubble"));
    CHECK_THROWS_WITH(NocFabric(sim, d, layout("kpu_soc_2x2.json"), {}), ContainsSubstring("NocTopology"));
    CHECK_THROWS_WITH(NocFabric::Config::from(device("kpu_soc_2x2.json")), ContainsSubstring("declares no noc"));
    NocFabric noc(sim, d, layout("kpu_t4.json"), {});
    auto& c = sim.add<Catcher>("c", d);
    CHECK_THROWS_WITH(noc.attach_l3(0, c, 9, 0), ContainsSubstring("there is no port 9"));
    CHECK_THROWS_WITH(noc.attach_engine(0, c, 7), ContainsSubstring("there is no hub 7"));
}
