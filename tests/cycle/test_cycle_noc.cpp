// ============================================================================
// tests/cycle/test_cycle_noc.cpp
// The NoC on the cycle core, step 1c (#539; docs/plans/cycle-based-kpu-prototype.md §3.4, §4.3
// rung 3).
//
// First the fabric on its own, on the T4's, T16's and T64's folded tori: every port reaches every
// hub's L3 and every hub's L3 reaches every engine of every port, each beat exactly once and only
// there; and one port by hand, for the ring-through that DMA <-> L3 traffic never makes. Then rung
// 3: DMA engines and L3 tiles over the NoC, each engine with its own LPDDR5 channel, every tile
// checked bit for bit in its slot and in DRAM, the slot credits and the bubble rule (every ring
// keeps a free slot; a link takes one flit a cycle) checked every cycle. Then the bubble rule
// under saturation: the flits it holds back, and the slots they take without it. Then the
// counters, determinism and the refusals.
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

// Pushes its items onto a link, one a cycle while it has a credit.
template <class T>
class PusherOf : public Component {
public:
    PusherOf(std::string name, Domain& d, std::vector<T> beats) : Component(std::move(name), d), beats_(std::move(beats)) {}
    void connect(Link<T>& out) { out_ = &out; }
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
    std::vector<T> beats_;
    std::size_t next_ = 0;
    Link<T>* out_ = nullptr;
};
using Pusher = PusherOf<L3Beat>;

// Pops every item that lands on a link, from cycle `from`, and keeps them.
template <class T>
class CatcherOf : public Component {
public:
    CatcherOf(std::string name, Domain& d, Cycle from = 0) : Component(std::move(name), d), from_(from) {}
    void connect(Link<T>& in) { in_ = &in; }
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
    std::vector<T> got;

private:
    Cycle from_;
    Link<T>* in_ = nullptr;
};
using Catcher = CatcherOf<L3Beat>;

// A component that does nothing: a subject for attachments that are refused.
class Inert : public Component {
public:
    Inert(std::string name, Domain& d) : Component(std::move(name), d) {}
    void evaluate(Cycle) override {}
    [[nodiscard]] bool idle() const override { return true; }
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
        for (const auto& b : catchers[c]->got) {
            // Every beat is one of the expected sources', at its catcher (#556's review).
            const auto it = source_to_catcher.find(b.ticket);
            CAPTURE(c, b.ticket);
            REQUIRE(it != source_to_catcher.end());
            CHECK(it->second == c);
            seen[c][b.ticket].push_back(b.offset);
        }
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
    for (const char* file : {"kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
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
        // Each hub stores to one engine of one port (two engines a port); rotate which, so every
        // hub-port pair is crossed and both engines of a port are used.
        for (NocDim shift = 0; shift < probe.port_count(); ++shift) {
            CAPTURE(shift);
            Simulator sim;
            Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
            NocFabric noc(sim, d, L, {});
            const NocTopology& t = noc.topology();
            std::vector<Catcher*> engines;   // port k's engine e at 2k + e
            for (NocDim k = 0; k < t.port_count(); ++k)
                for (NocDim e = 0; e < 2; ++e) {
                    auto& c = sim.add<Catcher>("engine " + std::to_string(k) + "." + std::to_string(e), d);
                    const auto links = noc.attach_engine(k, c, 0);
                    REQUIRE(links.engine == e);
                    c.connect(links.from_noc);
                    engines.push_back(&c);
                }
            std::map<std::uint64_t, std::size_t> where;
            for (NocDim h = 0; h < t.hub_count(); ++h) {
                const NocDim k = static_cast<NocDim>((h + shift) % t.port_count());
                const NocDim e = static_cast<NocDim>((h + shift) % 2);
                auto& p = sim.add<Pusher>("l3 " + std::to_string(h), d, stream(h, 6));
                const auto links = noc.attach_l3(h, p, k, e);
                p.connect(links.from_l3);
                where[h] = 2 * k + e;
            }
            REQUIRE(sim.run_until_idle(kLimit));
            check_caught(engines, where, 6);
        }
    }
}

TEST_CASE("A port on its own: ring-through, ejection at its exit hub, ring traffic first", "[cycle][noc]") {
    // DMA <-> L3 traffic never passes through a port (it enters and leaves at the fold hub nearer
    // its end), so the port's ring-through path is driven here by hand: the T4's port 1, row0.E,
    // on the fold link hub 0 -> hub 1 (and back), with flits pushed in from hub 0's side and hub
    // 1's, and one DMA engine injecting toward hub 1.
    const ArrayLayout L = layout("kpu_t4.json");
    const NocTopology t(L);
    REQUIRE(t.port(1).hub_a == 0);
    REQUIRE(t.port(1).hub_b == 1);
    auto flit = [](std::uint64_t id, NocFlit::Dest dest, NocDim hub, NocDim port = 0) {
        NocFlit f;
        f.id = id;
        f.dest = dest;
        f.dst_hub = hub;
        f.dst_port = port;
        f.fresh = false;
        f.beat.ticket = id;
        f.beat.bytes.assign(kL3BeatBytes, static_cast<std::uint8_t>(id));
        return f;
    };
    struct Bench {
        Simulator sim;
        Domain& d;
        NocPort& port;
        PusherOf<NocFlit>& from_a;
        PusherOf<NocFlit>& from_b;
        CatcherOf<NocFlit>& to_b;
        CatcherOf<NocFlit>& to_a;
        Pusher& engine;
        Catcher& engine_out;
        Pusher* engine2 = nullptr;
        // `hold_b`: hub 1's side takes nothing before that cycle. `second`: a second engine, with
        // that many beats for hub 0.
        Bench(const NocTopology& t, std::vector<NocFlit> a, std::vector<NocFlit> b, std::size_t injections,
              Cycle hold_b = 0, std::size_t second = 0)
            : d(sim.add_domain(Domain::from_ghz("kpu", 1.0))),
              port(sim.add<NocPort>("port", d, t, 1)),
              from_a(sim.add<PusherOf<NocFlit>>("from hub 0", d, std::move(a))),
              from_b(sim.add<PusherOf<NocFlit>>("from hub 1", d, std::move(b))),
              to_b(sim.add<CatcherOf<NocFlit>>("hub 1", d, hold_b)),
              to_a(sim.add<CatcherOf<NocFlit>>("hub 0", d)),
              engine(sim.add<Pusher>("engine", d, stream(7, injections))),
              engine_out(sim.add<Catcher>("engine out", d)) {
            auto& ia = sim.link<NocFlit>("0->port", from_a, port, 4);
            auto& ob = sim.link<NocFlit>("port->1", port, to_b, 4);
            auto& ib = sim.link<NocFlit>("1->port", from_b, port, 4);
            auto& oa = sim.link<NocFlit>("port->0", port, to_a, 4);
            auto& ei = sim.link<L3Beat>("engine->port", engine, port, 2);
            auto& eo = sim.link<L3Beat>("port->engine", port, engine_out, 2);
            from_a.connect(ia);
            from_b.connect(ib);
            to_b.connect(ob);
            to_a.connect(oa);
            engine.connect(ei);
            engine_out.connect(eo);
            port.connect_ring(ia, ob, ib, oa);
            REQUIRE(port.connect_engine(ei, eo, 1) == 0);
            if (second) {
                engine2 = &sim.add<Pusher>("engine 2", d, stream(8, second));
                auto& c2 = sim.add<Catcher>("engine 2 out", d);
                auto& e2i = sim.link<L3Beat>("engine 2->port", *engine2, port, 2);
                auto& e2o = sim.link<L3Beat>("port->engine 2", port, c2, 2);
                engine2->connect(e2i);
                c2.connect(e2o);
                REQUIRE(port.connect_engine(e2i, e2o, 0) == 1);
            }
        }
    };
    SECTION("ring traffic first: an injection waits for a cycle the channel is free") {
        std::vector<NocFlit> ring;
        for (std::uint64_t i = 1; i <= 20; ++i) ring.push_back(flit(i, NocFlit::Dest::L3, 1));
        Bench b(t, ring, {}, 5);
        REQUIRE(b.sim.run_until_idle(kLimit));
        REQUIRE(b.to_b.got.size() == 25);
        for (std::size_t i = 0; i < 20; ++i) CHECK(b.to_b.got[i].id == i + 1);   // ring flits, in order
        for (std::size_t i = 20; i < 25; ++i) {
            CHECK(b.to_b.got[i].dest == NocFlit::Dest::L3);
            CHECK(b.to_b.got[i].dst_hub == 1);
            CHECK(b.to_b.got[i].fresh);                                       // routed by its first hub
            CHECK(b.to_b.got[i].beat.ticket == 7);
        }
        CHECK(b.port.noc().ring_through == 20);
        CHECK(b.port.noc().injected == 5);
        CHECK(b.port.noc().injection_waits > 0);
    }
    SECTION("ejection only from the exit hub's side; from the other it rings through") {
        Bench b(t, {flit(1, NocFlit::Dest::Engine, 0, 1)}, {flit(2, NocFlit::Dest::Engine, 0, 1)}, 0);
        REQUIRE(b.sim.run_until_idle(kLimit));
        REQUIRE(b.engine_out.got.size() == 1);
        CHECK(b.engine_out.got[0].ticket == 1);   // from hub 0, its exit hub: ejected
        REQUIRE(b.to_a.got.size() == 1);
        CHECK(b.to_a.got[0].id == 2);             // from hub 1: on to hub 0, which turns it back
        CHECK(b.port.noc().ejected == 1);
        CHECK(b.port.noc().ring_through == 1);
    }
    SECTION("a head the bubble holds is counted while a younger one injects") {
        // #556's review. Hub 1's side holds 3 of its 4 slots (ring flits it does not take until
        // cycle 100): engine 0's beats for hub 1 have one credit, and the rule wants two. Engine 1's
        // for hub 0 go. Engine 0's wait is counted, as a bubble wait, while engine 1 injects.
        std::vector<NocFlit> ring;
        for (std::uint64_t i = 1; i <= 3; ++i) ring.push_back(flit(i, NocFlit::Dest::L3, 1));
        // Engine 1 injects every cycle through the hold, so every one of engine 0's waits is in a
        // cycle another engine injected.
        Bench b(t, ring, {}, 1, 100, 200);
        REQUIRE(b.sim.run_until_idle(kLimit));
        CHECK(b.to_a.got.size() == 200);   // engine 1's
        CHECK(b.to_b.got.size() == 4);     // the ring flits, then engine 0's
        CHECK(b.port.noc().bubble_waits >= 90);
        CHECK(b.port.noc().injection_waits >= b.port.noc().bubble_waits);
    }
    SECTION("a flit no hub has routed is a fault") {
        auto f = flit(1, NocFlit::Dest::L3, 1);
        f.fresh = true;
        Bench b(t, {f}, {}, 0);
        CHECK_THROWS_WITH(b.sim.run_until_idle(kLimit), ContainsSubstring("reached it unrouted"));
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

TEST_CASE("The bubble rule under saturation: what it refuses, and what goes without it", "[cycle][noc]") {
    // The T16 with every port streaming to every hub and every hub's L3 streaming back, the L3
    // tiles holding what reaches them for the first 3000 cycles, so the rings back up. With the
    // rule, every ring keeps a free slot through it (checked every cycle), entering flits are
    // refused a last slot, and everything arrives once the tiles drain. Without it, entering flits
    // take those slots: the rings fill further than the rule ever lets them.
    //
    // (A ring full to its last slot, and the deadlock that follows, is not built here: on these
    // tori dimension-ordered routes are at most two hops a loop, and a ring jams only when every
    // flit at every head continues along it. The check is what would catch it.)
    struct Run {
        bool done = false;
        std::uint64_t hub_bubble_waits = 0, port_bubble_waits = 0, turn_bubble_waits = 0;
        std::size_t deepest = 0;   // the most flits any ring held, less its slots
        std::size_t slots = 0;
    };
    auto run = [](bool bubble, std::size_t queue = 2) {
        Simulator sim;
        sim.set_deadlock_window(2000);
        Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
        NocFabric::Config c;
        c.bubble = bubble;
        c.ring_queue = queue;
        NocFabric noc(sim, d, layout("kpu_t16.json"), c);
        const NocTopology& t = noc.topology();
        std::vector<NocDim> engine_of_port;
        for (NocDim k = 0; k < t.port_count(); ++k)
            for (NocDim h = 0; h < t.hub_count(); ++h) {
                const std::string n = std::to_string(k) + "." + std::to_string(h);
                auto& p = sim.add<Pusher>("engine " + n, d, stream(k * 100 + h, 200));
                auto& s = sim.add<Catcher>("engine sink " + n, d);
                const auto e = noc.attach_engine(k, p, h);
                p.connect(e.to_noc);
                s.connect(e.from_noc);
                if (h == 0) engine_of_port.push_back(e.engine);
            }
        for (NocDim h = 0; h < t.hub_count(); ++h) {
            auto& p = sim.add<Pusher>("l3 out " + std::to_string(h), d, stream(1000 + h, 400));
            auto& c = sim.add<Catcher>("l3 " + std::to_string(h), d, 3000);
            const NocDim k = static_cast<NocDim>((h + 3) % t.port_count());
            const auto links = noc.attach_l3(h, p, k, engine_of_port[k]);
            p.connect(links.from_l3);
            c.connect(links.to_l3);
        }
        Run r;
        r.done = sim.run_until_idle(kLimit);
        for (NocDim h = 0; h < t.hub_count(); ++h) {
            r.hub_bubble_waits += noc.hub(h).noc().bubble_waits;
            r.turn_bubble_waits += noc.hub(h).noc().turn_bubble_waits;
        }
        for (NocDim k = 0; k < t.port_count(); ++k) r.port_bubble_waits += noc.port(k).noc().bubble_waits;
        for (const auto& ring : noc.ring_peaks()) {
            r.deepest = std::max(r.deepest, ring.peak);
            r.slots = ring.slots;
        }
        return r;
    };
    const Run with = run(true);
    const Run without = run(false);
    CHECK(with.done);
    CHECK(with.hub_bubble_waits > 0);        // the rule refused flits a last slot, at hubs
    CHECK(with.port_bubble_waits > 0);       // and at the ports' injections
    CHECK(with.turn_bubble_waits > 0);       // and at the turns from one loop onto another
    CHECK(without.hub_bubble_waits == 0);
    CHECK(without.port_bubble_waits == 0);
    CHECK(with.deepest < with.slots);        // and every ring kept one (also checked every cycle)
    CHECK(without.done);
    CHECK(without.deepest > with.deepest);   // without it, entering flits took those slots
    // Deeper queues: ring traffic and an injection now both find room on a fold channel in one
    // cycle, and the channel still takes one flit a cycle (the fabric's check), ring first.
    const Run deep = run(true, 4);
    CHECK(deep.done);
    CHECK(deep.deepest < deep.slots);
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

TEST_CASE("The NoC's checks outlive the fabric object, and start afresh on reset", "[cycle][noc]") {
    // #556's review. The fabric's checks (and the topology its hubs and ports point into) are held
    // by the simulator, so the fabric object may go first.
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    Catcher* l3 = nullptr;
    {
        NocFabric noc(sim, d, layout("kpu_t16.json"), {});
        l3 = &sim.add<Catcher>("l3", d);
        auto& p = sim.add<Pusher>("engine", d, stream(1, 40));
        const auto e = noc.attach_engine(0, p, 5);
        p.connect(e.to_noc);
        l3->connect(noc.attach_l3(5, *l3, 0, e.engine).to_l3);
    }
    REQUIRE(sim.run_until_idle(kLimit));
    CHECK(l3->got.size() == 40);

    // Reset: the rings' peaks start again from zero.
    Simulator s2;
    Domain& d2 = s2.add_domain(Domain::from_ghz("kpu", 1.0));
    NocFabric noc(s2, d2, layout("kpu_t4.json"), {});
    auto& c = s2.add<Catcher>("l3", d2);
    auto& p = s2.add<Pusher>("engine", d2, stream(2, 40));
    const auto e = noc.attach_engine(1, p, 1);
    p.connect(e.to_noc);
    c.connect(noc.attach_l3(1, c, 1, e.engine).to_l3);
    REQUIRE(s2.run_until_idle(kLimit));
    std::size_t peak = 0;
    for (const auto& r : noc.ring_peaks()) peak = std::max(peak, r.peak);
    REQUIRE(peak > 0);
    s2.reset();
    for (const auto& r : noc.ring_peaks()) CHECK(r.peak == 0);
    REQUIRE(s2.run_until_idle(kLimit));
    std::size_t again = 0;
    for (const auto& r : noc.ring_peaks()) again = std::max(again, r.peak);
    CHECK(again == peak);
}

TEST_CASE("The NoC refuses an attachment before it adds anything", "[cycle][noc]") {
    // #556's review: a refused attach leaves the simulator as it was (no links), a hub takes one L3
    // tile, and a store engine that does not exist is found before the first cycle closes.
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    NocFabric noc(sim, d, layout("kpu_t4.json"), {});
    auto& c = sim.add<Inert>("c", d);
    const std::size_t rows = report(sim).size();
    CHECK_THROWS_WITH(noc.attach_l3(0, c, 9, 0), ContainsSubstring("there is no port 9"));
    CHECK_THROWS_WITH(noc.attach_l3(5, c, 0, 0), ContainsSubstring("there is no hub 5"));
    CHECK_THROWS_WITH(noc.attach_engine(0, c, 7), ContainsSubstring("there is no hub 7"));
    CHECK_THROWS_WITH(noc.attach_engine(8, c, 0), ContainsSubstring("there is no port 8"));
    CHECK(report(sim).size() == rows);
    auto& l3 = sim.add<Catcher>("l3", d);
    l3.connect(noc.attach_l3(0, l3, 0, 1).to_l3);   // port 0's engine 1: not attached yet
    auto& other = sim.add<Inert>("other", d);
    CHECK_THROWS_WITH(noc.attach_l3(0, other, 0, 0), ContainsSubstring("hub 0 already has an L3 tile"));
    auto& p = sim.add<Pusher>("engine", d, stream(1, 4));
    p.connect(noc.attach_engine(0, p, 0).to_noc);   // engine 0 only
    try {
        sim.run_until_idle(kLimit);
        FAIL("the store target should be refused");
    } catch (const InvariantViolation& e) {
        CHECK(e.cycle() == 0);
        CHECK_THAT(e.what(), ContainsSubstring("hub 0's L3 tile stores to engine 1 of noc port") &&
                                 ContainsSubstring("which has 1"));
    }
}

TEST_CASE("The NoC refuses what it cannot build", "[cycle][noc]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    NocFabric::Config one;
    one.ring_queue = 1;
    CHECK_THROWS_WITH(NocFabric(sim, d, layout("kpu_t4.json"), one), ContainsSubstring("cannot keep a bubble"));
    CHECK_THROWS_WITH(NocFabric(sim, d, layout("kpu_soc_2x2.json"), {}), ContainsSubstring("NocTopology"));
    CHECK_THROWS_WITH(NocFabric::Config::from(device("kpu_soc_2x2.json")), ContainsSubstring("declares no noc"));
}
