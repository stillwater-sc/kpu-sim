// ============================================================================
// tests/timing/test_noc_fabric.cpp
// NoC port plan step 4a: the CSP hub and port processes on the folded torus, on their own
// (docs/plans/noc-port-arbitration.md §3, §5, §6). Each bus holds one block, ring traffic
// crosses before an injection, injection goes oldest first whatever the engine index,
// ejection never stalls at the derived depth when the engine keeps up, a slow engine's stalls
// are charged to rate, ring-first's injection wait is measured, and a saturated T64 stays
// live -- while one buffer per hub does not.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/noc_fabric.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using sw::kpu::program::platform::ArrayLayout;
using sw::kpu::program::platform::DeviceSpecification;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;
using Kind = NocPortProcess::Kind;

namespace {

DeviceSpecification device(const char* file) {
    return read_spec_file(std::string("tests/program/deploy/") + file).device(0);
}
ArrayLayout layout(const char* file) { return *ArrayLayout::of(device(file)); }

NocFabric::Config cfg(Cycle block_cycles, NocDim ports, NocDim engines = 1) {
    NocFabric::Config c;
    c.block_cycles = block_cycles;
    c.engines_per_port.assign(ports, engines);
    return c;
}

std::vector<NocPortProcess::Crossing> of_kind(const NocPortProcess& p, Kind k) {
    std::vector<NocPortProcess::Crossing> out;
    for (const auto& c : p.crossings())
        if (c.kind == k) out.push_back(c);
    return out;
}

std::size_t delivered(const NocFabric& f) {
    std::size_t n = 0;
    for (const auto& h : f.hubs()) n += h.delivered().size();
    return n;
}

}  // namespace

TEST_CASE("NoC topology: four channels per hub, two per port, dimension-ordered routes",
          "[noc][topology]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
        CAPTURE(file);
        const ArrayLayout L = layout(file);
        const NocTopology T(L);
        REQUIRE(T.hub_count() == L.l3_count());
        REQUIRE(T.port_count() == L.ports().size());

        std::size_t loop_links = 0;
        for (const auto& loop : L.loops()) loop_links += loop.hubs.size();
        REQUIRE(T.channels().size() == 2 * loop_links);

        std::vector<int> in(T.hub_count(), 0);
        for (const auto& c : T.channels()) ++in[c.to_hub];
        for (NocDim h = 0; h < T.hub_count(); ++h) {
            CHECK(T.out_channels(h).size() == 4);
            CHECK(in[h] == 4);
        }
        std::map<NocDim, int> owned;
        for (const auto& c : T.channels())
            if (c.port) ++owned[*c.port];
        REQUIRE(owned.size() == T.port_count());
        for (const auto& [k, n] : owned) CHECK(n == 2);

        // Following next_hop reaches the target in exactly distance() hops, row loop first:
        // once a block is on a column loop it never takes a row channel again, and it never
        // reverses direction on a loop.
        for (NocDim s = 0; s < T.hub_count(); ++s)
            for (NocDim t = 0; t < T.hub_count(); ++t) {
                NocDim h = s, hops = 0;
                std::optional<NocDim> prev;
                while (h != t) {
                    const NocChannel& c = T.channel(T.next_hop(h, t, prev && T.channel(*prev).column));
                    if (prev) {
                        const NocChannel& p = T.channel(*prev);
                        REQUIRE((!p.column || c.column));
                        if (p.loop == c.loop) REQUIRE(p.forward == c.forward);
                    }
                    prev = c.id;
                    h = c.to_hub;
                    REQUIRE(++hops <= T.distance(s, t));
                }
                CHECK(hops == T.distance(s, t));
            }
    }

    // Deterministic: a second build routes identically.
    const ArrayLayout L = layout("kpu_t64.json");
    const NocTopology a(L), b(L);
    for (NocDim s = 0; s < a.hub_count(); ++s)
        for (NocDim t = 0; t < a.hub_count(); ++t)
            if (s != t) REQUIRE(a.next_hop(s, t, false) == b.next_hop(s, t, false));
}

TEST_CASE("NoC topology: a port injects toward the fold hub nearer the destination (Q4)",
          "[noc][topology]") {
    const ArrayLayout L = layout("kpu_t64.json");
    const NocTopology T(L);
    for (NocDim k = 0; k < T.port_count(); ++k) {
        const auto& p = T.port(k);
        CHECK(T.injection_hub(k, p.hub_a) == p.hub_a);
        CHECK(T.injection_hub(k, p.hub_b) == p.hub_b);
        for (NocDim d = 0; d < T.hub_count(); ++d) {
            const NocDim via = T.injection_hub(k, d);
            CHECK(T.distance(via, d) == std::min(T.distance(p.hub_a, d), T.distance(p.hub_b, d)));
            if (T.distance(p.hub_a, d) == T.distance(p.hub_b, d)) CHECK(via == p.hub_a);
        }
    }
}

TEST_CASE("NoC fabric config comes from the declared noc section", "[noc][spec]") {
    const DeviceSpecification t4 = device("kpu_t4.json");
    std::string why;
    // 4 KiB blocks: 32 cycles on a 128 B/cycle link, 64 cycles per engine write at 64 B/cycle.
    const auto c = NocFabric::Config::from(t4, 4096, 100, &why);
    REQUIRE(c);
    CHECK(c->hub_buffer_blocks == 8);
    CHECK(c->input_queue_blocks == 2);
    CHECK(c->output_queue_blocks == 0);
    CHECK(c->block_cycles == 32);
    CHECK(c->dma_write_interval == 64);

    const NocFabric f(layout("kpu_t4.json"), *c);
    CHECK(f.derived_output_queue_depth() == 5);     // ceil(100 / 32) + 1
    CHECK(f.output_queue_depth() == 5);
    CHECK(derived_output_queue_blocks(100, 128.0, 4096.0) == 5);
    CHECK(derived_output_queue_blocks(0, 128.0, 4096.0) == 1);

    DeviceSpecification none = t4;
    none.noc.reset();
    CHECK_FALSE(NocFabric::Config::from(none, 4096, 100, &why));
    CHECK_THAT(why, ContainsSubstring("no noc"));

    // One buffer per input is refused at validation; only the fabric itself can be built so.
    DeviceSpecification thin = t4;
    thin.noc->hub_buffer_blocks = 4;
    CHECK_FALSE(NocFabric::Config::from(thin, 4096, 100, &why));
    CHECK_THAT(why, ContainsSubstring("hub_buffer_blocks"));
}

TEST_CASE("TF-PORT-1: a port's injection bus carries one block; the second waits a block time",
          "[noc][port]") {
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    NocFabric f(L, cfg(B, L.ports().size(), 2));
    const auto& p = f.topology().port(0);
    // Different fold hubs, so different fold channels: only the bus can serialize them.
    REQUIRE(f.inject(0, 0, p.hub_a));
    REQUIRE(f.inject(0, 1, p.hub_b));
    REQUIRE(f.run_until_quiescent(1000));

    const auto inj = of_kind(f.ports()[0], Kind::Injection);
    REQUIRE(inj.size() == 2);
    CHECK(inj[0].start == 0);
    CHECK(inj[1].start == B);
    CHECK(f.ports()[0].stats().injection_busy_cycles == 2 * B);
    CHECK(delivered(f) == 2);
}

TEST_CASE("Ring first: a ring block at the fold link crosses before a queued injection",
          "[noc][port]") {
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    NocFabric f(L, cfg(B, L.ports().size()));
    const NocTopology& T = f.topology();
    const NocDim c = T.next_hop(0, 1, false);
    REQUIRE(T.channel(c).port);
    const NocDim k = *T.channel(c).port;
    // The injection toward hub 1 needs the same channel the ring block is about to take.
    REQUIRE(T.fold_from(k, T.port(k).hub_a == 1 ? T.port(k).hub_b : T.port(k).hub_a) == c);

    REQUIRE(f.transfer(0, 1));                      // lands in hub 0 at cycle B
    while (f.now() < B) f.tick();
    REQUIRE(f.inject(k, 0, 1));                     // queued in the same cycle the ring block is ready
    REQUIRE(f.run_until_quiescent(1000));

    const auto& cr = f.ports()[k].crossings();
    REQUIRE(cr.size() == 2);
    CHECK(cr[0].kind == Kind::RingThrough);
    CHECK(cr[0].start == B);
    CHECK(cr[1].kind == Kind::Injection);
    CHECK(cr[1].start == 2 * B);
    CHECK(delivered(f) == 2);
}

TEST_CASE("Ring first on a proper ring: a block passing THROUGH a port crosses before an injection",
          "[noc][port][t16]") {
    // On the T4 every channel is a fold link, so no block ever passes through a port: the
    // ring-first test above races a block that only just entered. The T16's four-hub rings
    // have true ring-through traffic. Find a route whose fold crossing continues the ring the
    // block was already on.
    const ArrayLayout L = layout("kpu_t16.json");
    constexpr Cycle B = 8;
    NocFabric f(L, cfg(B, L.ports().size()));
    const NocTopology& T = f.topology();

    struct Through { NocDim src = 0, dst = 0, hop = 0, channel = 0; };
    std::optional<Through> pick;
    for (NocDim s = 0; s < T.hub_count() && !pick; ++s)
        for (NocDim d = 0; d < T.hub_count() && !pick; ++d) {
            NocDim h = s, hop = 0;
            std::optional<NocDim> prev;
            while (h != d && !pick) {
                const NocDim c = T.next_hop(h, d, prev && T.channel(*prev).column);
                if (prev && T.channel(c).port && T.channel(c).same_ring(T.channel(*prev)))
                    pick = Through{s, d, hop, c};
                prev = c;
                h = T.channel(c).to_hub;
                ++hop;
            }
        }
    REQUIRE(pick);
    const NocChannel& fold = T.channel(pick->channel);
    const NocDim k = *fold.port;
    // The injection lands in the hub the ring block is crossing into, over the same channel.
    REQUIRE(T.injection_hub(k, fold.to_hub) == fold.to_hub);
    REQUIRE(T.fold_from(k, fold.from_hub) == pick->channel);

    // Uncontended, the block enters at B and starts hop i at (i + 1) B.
    const Cycle cross = (pick->hop + 1) * B;
    REQUIRE(f.transfer(pick->src, pick->dst));
    while (f.now() < cross) f.tick();
    REQUIRE(f.inject(k, 0, fold.to_hub));
    REQUIRE(f.run_until_quiescent(1000));

    const auto& cr = f.ports()[k].crossings();
    REQUIRE(cr.size() == 2);
    CHECK(cr[0].kind == Kind::RingThrough);
    CHECK(cr[0].start == cross);
    CHECK(cr[1].kind == Kind::Injection);
    CHECK(cr[1].start == cross + B);
    CHECK(delivered(f) == 2);
}

TEST_CASE("Oldest first, stateless: injection order is by age, not engine index",
          "[noc][port]") {
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    NocFabric f(L, cfg(B, L.ports().size(), 4));
    const NocDim dst = f.topology().port(0).hub_a;
    REQUIRE(f.inject(0, 3, dst));                   // holds the bus for cycles [0, B)
    f.tick();
    // While the bus is busy, engine 1 queues first, then 2, then 0. When it frees at cycle B
    // their heads are B-1, B-2 and B-3 cycles old, so the lowest index is the youngest.
    REQUIRE(f.inject(0, 1, dst));
    f.tick();
    REQUIRE(f.inject(0, 2, dst));
    f.tick();
    REQUIRE(f.inject(0, 0, dst));
    REQUIRE(f.run_until_quiescent(1000));

    const auto inj = of_kind(f.ports()[0], Kind::Injection);
    REQUIRE(inj.size() == 4);
    CHECK(inj[0].engine == 3);
    CHECK(inj[1].engine == 1);
    CHECK(inj[2].engine == 2);
    CHECK(inj[3].engine == 0);
    for (std::size_t i = 1; i < inj.size(); ++i) CHECK(inj[i].start == inj[i - 1].start + B);
}

namespace {

// Saturate port k's ejection bus: every hub keeps pushing blocks to engine 0 on port k until
// `n` have been accepted, then run dry.
void eject_stream(NocFabric& f, NocDim k, std::size_t n) {
    std::size_t sent = 0;
    for (Cycle i = 0; i < 100000 && sent < n; ++i) {
        for (NocDim h = 0; h < f.topology().hub_count() && sent < n; ++h)
            if (f.eject(h, k, 0, sent)) ++sent;
        f.tick();
    }
    REQUIRE(sent == n);
    REQUIRE(f.run_until_quiescent(100000));
}

}  // namespace

TEST_CASE("TF-PORT-2: at the derived output depth, ejection never stalls the NoC",
          "[noc][port][eject]") {
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    NocFabric::Config c = cfg(B, L.ports().size());
    c.dma_write_latency = 3 * B + 1;                // derived depth = ceil(25 / 8) + 1 = 5
    c.dma_write_interval = B;                       // the engine keeps up with the bus
    constexpr std::size_t N = 64;

    SECTION("derived depth: zero stalls") {
        NocFabric f(L, c);
        REQUIRE(f.output_queue_depth() == 5);
        eject_stream(f, 0, N);
        const auto& s = f.ports()[0].stats();
        CHECK(s.ejected == N);
        CHECK(s.written == N);
        CHECK(s.eject_stall_cycles == 0);
        // The bus really was saturated: back-to-back ejections for the whole stream.
        CHECK(s.ejection_busy_cycles == N * B);
    }
    SECTION("one block below derived: stalls appear, charged to depth") {
        c.output_queue_blocks = 4;
        NocFabric f(L, c);
        eject_stream(f, 0, N);
        const auto& s = f.ports()[0].stats();
        CHECK(s.written == N);
        CHECK(s.eject_stall_cycles > 0);
        CHECK(s.eject_stalls_depth == s.eject_stall_cycles);
        CHECK(s.eject_stalls_unattributed == 0);
    }
}

TEST_CASE("Rate, not depth: an engine slower than the bus stalls ejection at any depth",
          "[noc][port][eject]") {
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    for (std::size_t depth : {std::size_t{0}, std::size_t{20}}) {
        CAPTURE(depth);
        NocFabric::Config c = cfg(B, L.ports().size());
        c.dma_write_latency = 3 * B + 1;
        c.dma_write_interval = 2 * B;               // retires a block every two block times
        c.output_queue_blocks = depth;
        NocFabric f(L, c);
        eject_stream(f, 0, 64);
        const auto& s = f.ports()[0].stats();
        CHECK(s.written == 64);
        CHECK(s.eject_stall_cycles > 0);
        CHECK(s.eject_stalls_rate == s.eject_stall_cycles);
        CHECK(s.eject_stalls_depth == 0);
        CHECK(s.eject_stalls_unattributed == 0);
    }
}

TEST_CASE("TF-PORT-3: ring-first holds a queued injection while ring traffic lasts",
          "[noc][port]") {
    // The ring traffic is an ejection stream: blocks leaving the ring across the same fold
    // channel the injection needs, at one block per block time. Ring first, so the injection
    // waits out the whole stream, and TF-PORT-3 reports how long.
    const ArrayLayout L = layout("kpu_t4.json");
    constexpr Cycle B = 8;
    NocFabric f(L, cfg(B, L.ports().size()));
    const NocTopology& T = f.topology();
    const NocDim k = 0;
    const NocDim a = T.port(k).hub_a, b = T.port(k).hub_b;
    REQUIRE(T.exit_hub(k, a) == a);
    REQUIRE(T.injection_hub(k, b) == b);        // both need the fold channel a -> b

    constexpr std::size_t N = 32;
    std::size_t sent = 0;
    bool queued = false;
    for (Cycle i = 0; i < 10000 && sent < N; ++i) {
        if (f.eject(a, k, 0, sent)) ++sent;
        if (!queued && f.now() == B) queued = f.inject(k, 0, b).has_value();
        f.tick();
    }
    REQUIRE(queued);
    REQUIRE(f.run_until_quiescent(10000));

    const auto& port = f.ports()[k];
    const auto ej = of_kind(port, Kind::Ejection);
    const auto inj = of_kind(port, Kind::Injection);
    REQUIRE(ej.size() == N);
    REQUIRE(inj.size() == 1);
    for (std::size_t i = 1; i < N; ++i) CHECK(ej[i].start == ej[i - 1].start + B);  // no gap
    CHECK(inj[0].start == ej.back().start + B);
    CHECK(port.stats().max_input_wait == inj[0].start - B);
    CHECK(port.stats().max_input_wait >= (N - 1) * B);
    CHECK(port.stats().eject_stall_cycles == 0);
}

namespace {

struct Lcg {
    std::uint64_t s;
    std::uint32_t next(std::uint32_t n) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>((s >> 33) % n);
    }
};

// Every engine of every port always has a block ready, to a pseudo-random hub, until each has
// injected `per_engine`. Returns the tag -> destination of everything injected.
std::map<std::uint64_t, NocDim> saturate(NocFabric& f, NocDim engines, std::size_t per_engine,
                                         Cycle max_cycles) {
    const NocDim ports = f.topology().port_count(), hubs = f.topology().hub_count();
    std::vector<std::size_t> sent(ports * engines, 0);
    std::vector<NocDim> pending(ports * engines);
    Lcg rng{12345};
    for (auto& p : pending) p = rng.next(hubs);
    std::map<std::uint64_t, NocDim> dst;
    std::uint64_t tag = 0;
    for (Cycle i = 0; i < max_cycles; ++i) {
        bool more = false;
        for (NocDim k = 0; k < ports; ++k)
            for (NocDim e = 0; e < engines; ++e) {
                const std::size_t slot = k * engines + e;
                if (sent[slot] == per_engine) continue;
                more = true;
                if (f.inject(k, e, pending[slot], tag)) {
                    dst[tag++] = pending[slot];
                    pending[slot] = rng.next(hubs);
                    ++sent[slot];
                }
            }
        if (!more) break;
        f.tick();
    }
    return dst;
}

}  // namespace

TEST_CASE("Liveness: saturating injection from every T64 port drains, and every block arrives",
          "[noc][liveness]") {
    const ArrayLayout L = layout("kpu_t64.json");
    constexpr Cycle B = 4;
    NocFabric::Config c = cfg(B, L.ports().size(), 2);
    c.hub_buffer_blocks = 8;
    c.watchdog_cycles = 200 * B;
    NocFabric f(L, c);

    const auto dst = saturate(f, 2, 256, 400000);
    REQUIRE(dst.size() == L.ports().size() * 2 * 256);
    REQUIRE(f.run_until_quiescent(400000));
    CHECK_FALSE(f.watchdog_fired());

    // Every block arrived once, at the hub it was sent to, with its tag intact: the NoC moves
    // blocks, it does not change them.
    std::map<std::uint64_t, int> seen;
    for (NocDim h = 0; h < f.hubs().size(); ++h)
        for (const auto& d : f.hubs()[h].delivered()) {
            ++seen[d.block.tag];
            CHECK(dst.at(d.block.tag) == h);
        }
    CHECK(seen.size() == dst.size());
    for (const auto& [t, n] : seen) CHECK(n == 1);

    // TF-HUB-1 and TF-PORT-1 held throughout.
    for (const auto& h : f.hubs()) {
        CHECK(h.stats().peak_ring_queue <= 2);
        CHECK(h.stats().peak_ring_occupancy <= c.hub_buffer_blocks);
    }
    for (const auto& p : f.ports()) {
        CHECK(p.stats().injection_busy_cycles == p.stats().injected * B);
        CHECK_FALSE(p.injection_bus_busy());
    }
}

TEST_CASE("Liveness: the T16 drains saturating injection mixed with L3 -> L3 moves",
          "[noc][liveness][t16]") {
    // Injection alone never passes THROUGH a T16 port: a port injects into its nearer fold hub,
    // and on a four-hub ring no shortest route from there crosses a fold link. L3 -> L3 moves
    // do, so they are mixed in to load the ports with ring-through traffic too.
    const ArrayLayout L = layout("kpu_t16.json");
    constexpr Cycle B = 4;
    NocFabric::Config c = cfg(B, L.ports().size(), 2);
    c.watchdog_cycles = 200 * B;
    NocFabric f(L, c);
    const NocDim ports = f.topology().port_count(), hubs = f.topology().hub_count();

    Lcg rng{4242};
    std::map<std::uint64_t, NocDim> dst;
    std::uint64_t tag = 0;
    constexpr std::size_t N = 4000;     // of each kind
    std::size_t injected = 0, moved = 0;
    for (Cycle i = 0; i < 400000 && (injected < N || moved < N); ++i) {
        for (NocDim k = 0; k < ports && injected < N; ++k) {
            const NocDim d = rng.next(hubs);
            if (f.inject(k, rng.next(2), d, tag)) { dst[tag++] = d; ++injected; }
        }
        for (NocDim h = 0; h < hubs && moved < N; ++h) {
            const NocDim d = rng.next(hubs);
            if (f.transfer(h, d, tag)) { dst[tag++] = d; ++moved; }
        }
        f.tick();
    }
    REQUIRE(injected == N);
    REQUIRE(moved == N);
    REQUIRE(f.run_until_quiescent(400000));
    CHECK_FALSE(f.watchdog_fired());

    std::size_t through = 0;
    for (const auto& p : f.ports()) through += p.stats().ring_through;
    CHECK(through > 0);
    std::map<std::uint64_t, int> seen;
    for (NocDim h = 0; h < f.hubs().size(); ++h)
        for (const auto& d : f.hubs()[h].delivered()) {
            ++seen[d.block.tag];
            CHECK(dst.at(d.block.tag) == h);
        }
    CHECK(seen.size() == dst.size());
    for (const auto& [t, n] : seen) CHECK(n == 1);
}

TEST_CASE("Liveness: injection, L3 -> L3 moves and ejection together drain on the T64",
          "[noc][liveness]") {
    const ArrayLayout L = layout("kpu_t64.json");
    constexpr Cycle B = 4;
    NocFabric::Config c = cfg(B, L.ports().size(), 2);
    c.watchdog_cycles = 200 * B;
    c.dma_write_latency = 3 * B;
    NocFabric f(L, c);
    const NocDim ports = f.topology().port_count(), hubs = f.topology().hub_count();

    Lcg rng{777};
    std::size_t injected = 0, moved = 0, ejected = 0;
    constexpr std::size_t N = 2000;     // of each kind
    for (Cycle i = 0; i < 400000 && (injected < N || moved < N || ejected < N); ++i) {
        if (injected < N && f.inject(rng.next(ports), rng.next(2), rng.next(hubs))) ++injected;
        for (NocDim h = 0; h < hubs; ++h) {
            if (h % 2 == 0 && moved < N && f.transfer(h, rng.next(hubs))) ++moved;
            if (h % 2 == 1 && ejected < N && f.eject(h, rng.next(ports), rng.next(2))) ++ejected;
        }
        f.tick();
    }
    REQUIRE(injected == N);
    REQUIRE(moved == N);
    REQUIRE(ejected == N);
    REQUIRE(f.run_until_quiescent(400000));
    CHECK_FALSE(f.watchdog_fired());

    std::size_t written = 0;
    for (const auto& p : f.ports()) {
        written += p.written().size();
        CHECK(p.stats().eject_stalls_unattributed == 0);
    }
    CHECK(delivered(f) == 2 * N);
    CHECK(written == N);
}

namespace {

// Every hub of every row loop keeps sending to the hub `dist` places ahead on its own loop:
// every block continues along one ring, the pattern the bubble exists for.
void same_ring(NocFabric& f, const ArrayLayout& L, NocDim dist, Cycle cycles) {
    std::uint64_t tag = 0;
    for (Cycle i = 0; i < cycles && !f.watchdog_fired(); ++i) {
        for (const auto& loop : L.loops()) {
            if (loop.axis != sw::kpu::program::platform::NocLoop::Axis::Row) continue;
            const auto n = static_cast<NocDim>(loop.hubs.size());
            for (NocDim j = 0; j < n; ++j)
                if (f.transfer(loop.hubs[j], loop.hubs[(j + dist) % n], tag)) ++tag;
        }
        f.tick();
    }
}

}  // namespace

TEST_CASE("Liveness: the bubble is what keeps a full ring moving", "[noc][liveness]") {
    const ArrayLayout L = layout("kpu_t64.json");
    constexpr Cycle B = 4;
    NocFabric::Config c = cfg(B, L.ports().size());
    c.watchdog_cycles = 200 * B;
    for (NocDim dist : {2u, 3u, 4u}) {
        CAPTURE(dist);
        SECTION("two slots per input, bubble on: live") {
            NocFabric f(L, c);
            same_ring(f, L, dist, 20000);
            CHECK_FALSE(f.watchdog_fired());
            REQUIRE(f.run_until_quiescent(20000));
            CHECK(delivered(f) > 0);
        }
        SECTION("one slot per input, bubble off: the ring fills and wedges") {
            c.hub_buffer_blocks = 4;
            c.bubble = false;
            NocFabric f(L, c);
            same_ring(f, L, dist, 20000);
            CHECK(f.watchdog_fired());
        }
        SECTION("one slot per input, bubble on: nothing can enter (the spec refuses this)") {
            c.hub_buffer_blocks = 4;
            NocFabric f(L, c);
            same_ring(f, L, dist, 2000);
            CHECK_FALSE(f.watchdog_fired());
            CHECK_FALSE(f.run_until_quiescent(2000));
            CHECK(delivered(f) == 0);
        }
    }
}
