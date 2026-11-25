// ============================================================================
// tests/cycle/test_cycle_l3_dma.cpp
// The L3 tile and the DMA engine on a direct link, step 1b (#538;
// docs/plans/cycle-based-kpu-prototype.md §4.1, §4.3 rungs 1-2).
//
// Rung 1: one tile is loaded into L3 and stored back to DRAM, bit for bit. Rung 2: several tiles
// go through one L3's slots, a Release freeing each slot for the next Load. In every run the slot
// credits are checked every cycle (check_slot_credits, as an invariant hook), and the bytes each
// slot holds are compared with what the program places there at the cycle the tile is READY.
// Then: the window and the store buffer under backpressure, the waits that must not be faults
// (a Load without its credit, a Store before its tile) found by the deadlock detector, the L3
// tile's faults on a beat it must never receive, determinism, and the refusals.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "l3_dma_rig.hpp"

#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
constexpr std::uint64_t kIn = 0x10000;          // where the tiles are in DRAM
constexpr std::uint64_t kOut = 0x400000;        // where they are stored back

// `tiles` tiles of `bytes` each through `slots` slots, round robin: Load, Store, Release.
L3Program ring(std::uint32_t tiles, std::uint32_t bytes, std::uint32_t slots) {
    L3Program p;
    for (std::uint32_t t = 0; t < tiles; ++t) {
        p.push_back(l3_load(t % slots, kIn + std::uint64_t(t) * bytes, bytes));
        p.push_back(l3_store(t % slots, kOut + std::uint64_t(t) * bytes, bytes));
        p.push_back(l3_release(t % slots));
    }
    return p;
}

// Preloads each Load's tile, and records the bytes of every tile at the cycle its slot is READY.
struct Observed {
    std::map<std::uint64_t, std::vector<std::uint8_t>> at_ready;   // by the Load's ticket
};

void preload_and_watch(L3DmaRig& r, const L3Program& p, Observed& seen) {
    for (std::size_t i = 0; i < p.size(); ++i)
        if (p[i].kind == L3Action::Kind::Load)
            r.channel.preload(p[i].dram_address, tile_bytes(static_cast<std::uint32_t>(i), p[i].bytes));
    r.sim.add_invariant("watch", [&r, &seen] {
        for (std::uint32_t s = 0; s < r.l3.config().slots; ++s)
            if (r.l3.ready(s) && !seen.at_ready.count(r.l3.ticket(s))) seen.at_ready[r.l3.ticket(s)] = r.l3.peek(s);
    });
}

// Every Load's tile was in its slot, bit for bit, and every Store's bytes are in DRAM.
void check_data(L3DmaRig& r, const L3Program& p, const Observed& seen) {
    std::map<std::uint32_t, std::size_t> last_load;   // slot -> the ticket of the Load it holds
    for (std::size_t i = 0; i < p.size(); ++i) {
        const L3Action& a = p[i];
        if (a.kind == L3Action::Kind::Load) {
            last_load[a.slot] = i;
            REQUIRE(seen.at_ready.count(i));
            CHECK(seen.at_ready.at(i) == tile_bytes(static_cast<std::uint32_t>(i), a.bytes));
        } else if (a.kind == L3Action::Kind::Store) {
            CHECK(r.channel.peek(a.dram_address, a.bytes) ==
                  tile_bytes(static_cast<std::uint32_t>(last_load.at(a.slot)), a.bytes));
        }
    }
}

// A source of hand-made beats into an L3 tile, one a cycle while it has a credit, none before
// cycle `from`[i] for beat i (0 where `from` is short).
class BeatSource : public Component {
public:
    BeatSource(std::string name, Domain& d, std::vector<L3Beat> beats, std::vector<Cycle> from = {})
        : Component(std::move(name), d), beats_(std::move(beats)), from_(std::move(from)) {}
    void connect(Link<L3Beat>& out, Link<SlotCredit>& credits) {
        out_ = &out;
        credits_ = &credits;
    }
    void evaluate(Cycle now) override {
        while (credits_->can_pop(now)) credits_->pop(now);   // a freed slot's credit, taken back
        if (next_ < beats_.size() && next_ < from_.size() && now < from_[next_]) {
            counters().stall(Stall::External);
            return;
        }
        if (next_ < beats_.size() && out_->can_push(now)) {
            out_->push(now, beats_[next_++]);
            counters().busy();
        }
    }
    [[nodiscard]] bool idle() const override { return next_ == beats_.size(); }

private:
    std::vector<L3Beat> beats_;
    std::vector<Cycle> from_;
    std::size_t next_ = 0;
    Link<L3Beat>* out_ = nullptr;
    Link<SlotCredit>* credits_ = nullptr;
};

// The beats of a whole tile (ticket, slot, bytes), each byte the ticket.
std::vector<L3Beat> beats_of(std::uint64_t ticket, std::uint32_t slot, std::uint32_t bytes) {
    std::vector<L3Beat> v;
    for (std::uint32_t o = 0; o < bytes; o += kL3BeatBytes) {
        L3Beat b;
        b.id = v.size() + 1;
        b.ticket = ticket;
        b.slot = slot;
        b.offset = o;
        b.tile_bytes = bytes;
        b.bytes.assign(kL3BeatBytes, static_cast<std::uint8_t>(ticket));
        v.push_back(std::move(b));
    }
    return v;
}

// An L3 tile fed by a BeatSource, its eject link left unread.
struct L3Alone {
    Simulator sim;
    Domain& d;
    L3Tile& l3;
    BeatSource& src;
    L3Alone(std::vector<L3Beat> beats, const L3Program& program = {}, std::vector<Cycle> from = {})
        : d(sim.add_domain(Domain::from_ghz("kpu", 1.0))),
          l3(sim.add<L3Tile>("l3", d, L3Tile::Config{2, 4096, 4}, program)),
          src(sim.add<BeatSource>("src", d, std::move(beats), std::move(from))) {
        auto& in = sim.link<L3Beat>("src->l3", src, l3, 2);
        auto& out = sim.link<L3Beat>("l3 ejects", l3, src, 2);
        auto& cr = sim.link<SlotCredit>("l3 credits", l3, src, 2);
        src.connect(in, cr);
        l3.connect(in, out, cr);
    }
};

void expect_fault(L3Alone& a, const std::string& what) {
    try {
        a.sim.run_until_idle(kLimit);
        FAIL("the L3 tile should fault: " + what);
    } catch (const InvariantViolation& e) {
        CHECK(e.where() == "l3");
        CHECK_THAT(e.what(), ContainsSubstring(what));
    }
}

}  // namespace

TEST_CASE("Rung 1: one tile is loaded into L3 and stored back, bit for bit", "[cycle][l3][dma]") {
    const L3Program p = {l3_load(0, kIn, 4096), l3_store(0, kOut, 4096), l3_release(0)};
    L3DmaRig r(p, {});
    Observed seen;
    preload_and_watch(r, p, seen);
    REQUIRE(r.sim.run_until_idle(kLimit));

    check_data(r, p, seen);
    CHECK(r.channel.peek(kIn, 4096) == tile_bytes(0, 4096));   // the source is untouched
    CHECK(r.dma.dma().loads == 1);
    CHECK(r.dma.dma().stores == 1);
    CHECK(r.dma.dma().read_bursts == 64);
    CHECK(r.dma.dma().write_bursts == 64);
    CHECK(r.channel.dram().reads == 64);
    CHECK(r.channel.dram().writes == 64);

    // The tile's life: READY after the Load's last beat was pushed, released after the Store's
    // last write was acknowledged, and the credit back with the engine.
    REQUIRE(r.l3.residencies().size() == 1);
    const auto& life = r.l3.residencies()[0];
    REQUIRE(r.dma.spans().size() == 2);
    const auto& load = r.dma.spans()[0];
    const auto& store = r.dma.spans()[1];
    CHECK(load.ticket == 0);
    CHECK(store.ticket == 1);
    CHECK(life.ready_at > *load.completed);   // the last beat lands a cycle after its push
    CHECK(*store.completed > life.ready_at);
    REQUIRE(life.released_at);
    CHECK(*life.released_at > life.ready_at);
    CHECK(r.dma.holds(0));
    CHECK(!r.l3.live(0));
    CHECK(r.l3.counts().arrivals == 1);
    CHECK(r.l3.counts().completions == 1);
    CHECK(r.dma.counts().completions == 2);
}

TEST_CASE("Rung 2: tiles through one L3's slots, a Release freeing each slot for the next Load",
          "[cycle][l3][dma]") {
    const std::uint32_t slots = GENERATE(1u, 2u, 3u);
    const L3Program p = ring(8, 1024, slots);
    L3DmaRig::Options o;
    o.slots = slots;
    L3DmaRig r(p, o);
    Observed seen;
    preload_and_watch(r, p, seen);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_data(r, p, seen);

    // Each Load into a slot starts only once the Release before it has freed the slot, and its
    // credit has come back (a cycle, on the credit link).
    const auto& lives = r.l3.residencies();
    REQUIRE(lives.size() == 8);
    std::map<std::uint64_t, Cycle> started;
    for (const auto& s : r.dma.spans()) started[s.ticket] = s.started;
    for (std::uint32_t t = slots; t < 8; ++t) {
        const auto& before = lives[t - slots];   // the slot's previous tile
        REQUIRE(before.slot == t % slots);
        REQUIRE(before.released_at);
        CHECK(started.at(3 * t) > *before.released_at);
    }
    // Never more tiles than slots, and with several slots, tiles overlap in L3: the next Load
    // fills one slot while the last tile is stored from another. (Two at most here: the engine
    // sends the oldest command's bursts first, so a Store drains before the Load after it.)
    CHECK(r.l3.counts().max_occupancy <= slots);
    if (slots > 1) CHECK(r.l3.counts().max_occupancy >= 2);
    CHECK(r.l3.counts().arrivals == 8);
    CHECK(r.l3.counts().completions == 8);
    CHECK(r.dma.held_count() == slots);
    CHECK(r.l3.live_count() == 0);
}

TEST_CASE("Slots that are only filled: Loads and Releases, the last tiles left resident", "[cycle][l3][dma]") {
    // Five Loads through two slots with no Stores; the last two tiles stay in L3.
    L3Program p;
    for (std::uint32_t t = 0; t < 5; ++t) {
        p.push_back(l3_load(t % 2, kIn + t * 512, 512));
        if (t < 3) p.push_back(l3_release(t % 2));
    }
    L3DmaRig r(p, {});
    Observed seen;
    preload_and_watch(r, p, seen);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_data(r, p, seen);
    CHECK(r.l3.ready(0));
    CHECK(r.l3.ready(1));
    CHECK(r.l3.peek(0) == tile_bytes(static_cast<std::uint32_t>(p.size() - 1), 512));   // the fifth Load
    CHECK(r.dma.held_count() == 0);
    CHECK(r.l3.counts().occupancy == 2);
}

TEST_CASE("The burst window and the store buffer bound what is in flight", "[cycle][l3][dma]") {
    const L3Program p = ring(4, 2048, 2);
    auto run = [&](std::size_t window, std::size_t buffer, std::size_t l3_link) {
        L3DmaRig::Options o;
        o.window = window;
        o.store_buffer_beats = buffer;
        o.l3_link = l3_link;
        auto r = std::make_unique<L3DmaRig>(p, o);
        Observed seen;
        preload_and_watch(*r, p, seen);
        REQUIRE(r->sim.run_until_idle(kLimit));
        check_data(*r, p, seen);
        CHECK(r->dma.dma().max_in_flight == window);   // reached, never passed (the engine's check)
        return r;
    };
    const auto one = run(1, 1, 1);
    const auto wide = run(16, 8, 4);
    // One burst at a time pays DRAM's whole latency per burst; a window overlaps it.
    CHECK(wide->sim.now() * 2 < one->sim.now());
    CHECK(one->dma.dma().waited(DmaWait::Window) > 0);
    CHECK(one->dma.dma().waited(DmaWait::StoreBuffer) > 0);
}

TEST_CASE("Backpressure from L3: a narrow link holds DRAM's responses, nothing lost", "[cycle][l3][dma]") {
    // The L3 link has one credit and the tile's Stores read the bank a Load writes; the engine
    // waits on the link, and every byte still arrives.
    L3Program p = ring(4, 4096, 2);
    L3DmaRig::Options o;
    o.l3_link = 1;
    o.banks = 1;   // every read and write meet in the one bank
    o.window = 16;
    L3DmaRig r(p, o);
    Observed seen;
    preload_and_watch(r, p, seen);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_data(r, p, seen);
    CHECK(r.dma.dma().waited(DmaWait::L3Credit) > 0);
    CHECK(r.l3.bank_conflicts() > 0);
}

TEST_CASE("A Load waits for its slot's credit: one never released is a deadlock, not a fault",
          "[cycle][l3][dma]") {
    // Two Loads into the one slot with no Release between: the second waits for a credit that
    // never comes. The deadlock detector names the engine and what it waits for.
    const L3Program p = {l3_load(0, kIn, 256), l3_load(0, kIn + 256, 256)};
    L3DmaRig::Options o;
    o.slots = 1;
    L3DmaRig r(p, o);
    r.sim.set_deadlock_window(500);
    try {
        r.sim.run_until_idle(kLimit);
        FAIL("the second Load should wait forever");
    } catch (const Deadlock& e) {
        CHECK(std::find(e.blocked().begin(), e.blocked().end(), "dma") != e.blocked().end());
        CHECK_THAT(e.dump(), ContainsSubstring("Load slot 0: waiting for its credit"));
    }
    CHECK(r.l3.ready(0));
    CHECK(r.l3.ticket(0) == 0);   // the first tile is intact
    CHECK(r.dma.dma().waited(DmaWait::SlotCredit) > 0);
}

TEST_CASE("A Store waits for its slot's tile: one never loaded is a deadlock, not a fault", "[cycle][l3][dma]") {
    const L3Program p = {l3_store(1, kOut, 256)};
    L3DmaRig r(p, {});
    r.sim.set_deadlock_window(500);
    try {
        r.sim.run_until_idle(kLimit);
        FAIL("the Store should wait forever");
    } catch (const Deadlock& e) {
        CHECK(std::find(e.blocked().begin(), e.blocked().end(), "l3") != e.blocked().end());
        CHECK_THAT(e.dump(), ContainsSubstring("Store slot 1: waiting for its tile"));
    }
    CHECK(r.dma.dma().write_bursts == 0);
    CHECK(r.l3.counts().stalled(Stall::NoData) > 0);
}

TEST_CASE("The L3 tile faults a beat it must never receive", "[cycle][l3]") {
    SECTION("a Load into a READY slot") {
        auto beats = beats_of(1, 0, 256);
        const auto second = beats_of(2, 0, 256);
        beats.insert(beats.end(), second.begin(), second.end());
        L3Alone a(beats);
        expect_fault(a, "the beat at offset 0 of Load 2 into slot 0, which holds Load 1's tile (ready): a Load into a live slot");
    }
    SECTION("a Load into a FILLING slot") {
        auto beats = beats_of(1, 0, 256);
        beats.resize(2);
        const auto second = beats_of(2, 0, 256);
        beats.insert(beats.end(), second.begin(), second.end());
        L3Alone a(beats);
        expect_fault(a, "which holds Load 1's tile (filling): a Load into a live slot");
    }
    SECTION("a beat that lands twice") {
        auto beats = beats_of(1, 0, 256);
        beats.insert(beats.begin() + 2, beats[1]);
        L3Alone a(beats);
        expect_fault(a, "the beat at offset 64 of Load 1 into slot 0 landed twice");
    }
    SECTION("a beat outside its tile, or its slot") {
        auto beats = beats_of(1, 0, 256);
        beats[1].offset = 256;
        L3Alone a(beats);
        expect_fault(a, "is outside its tile or its slot");
    }
    SECTION("a beat into a slot that does not exist") {
        L3Alone a(beats_of(1, 5, 256));
        expect_fault(a, "there is no such slot");
    }
    SECTION("a tile whose beats disagree on its size") {
        auto beats = beats_of(1, 0, 256);
        beats[1].tile_bytes = 512;
        L3Alone a(beats);
        expect_fault(a, "says its tile is 512 bytes, its first beat 256");
    }
    SECTION("but a Release frees the slot for the next Load") {
        // Tile 1's four beats from cycle 0, tile 2's from cycle 20, after tile 1's Release.
        auto beats = beats_of(1, 0, 256);
        const auto second = beats_of(2, 0, 256);
        beats.insert(beats.end(), second.begin(), second.end());
        L3Alone a(beats, {l3_load(0, 0, 256), l3_release(0), l3_load(0, 0, 256)}, {0, 0, 0, 0, 20});
        REQUIRE(a.sim.run_until_idle(kLimit));
        REQUIRE(a.l3.residencies().size() == 2);
        CHECK(*a.l3.residencies()[0].released_at < 20);
        CHECK(a.l3.ticket(0) == 2);
        CHECK(a.l3.peek(0) == std::vector<std::uint8_t>(256, 2));
    }
}

TEST_CASE("A Store of a different size than the tile in its slot is a fault", "[cycle][l3]") {
    L3Alone a(beats_of(1, 0, 256), {l3_load(0, 0, 256), l3_store(0, 0, 128)});
    expect_fault(a, "action 1 stores 128 bytes from slot 0, which holds a tile of 256");
}

TEST_CASE("Two runs, and a run after reset, are identical", "[cycle][l3][dma]") {
    const L3Program p = ring(6, 1024, 2);
    auto spans_of = [](const L3DmaRig& r) {
        std::vector<std::pair<Cycle, Cycle>> v;
        for (const auto& s : r.dma.spans()) v.emplace_back(s.started, *s.completed);
        return v;
    };
    L3DmaRig a(p, {}), b(p, {});
    Observed sa, sb;
    preload_and_watch(a, p, sa);
    preload_and_watch(b, p, sb);
    REQUIRE(a.sim.run_until_idle(kLimit));
    REQUIRE(b.sim.run_until_idle(kLimit));
    CHECK(a.sim.now() == b.sim.now());
    CHECK(spans_of(a) == spans_of(b));
    const auto first = spans_of(a);
    const Picoseconds end = a.sim.now();
    a.sim.reset();
    sa.at_ready.clear();
    REQUIRE(a.sim.run_until_idle(kLimit));
    CHECK(a.sim.now() == end);
    CHECK(spans_of(a) == first);
    check_data(a, p, sa);
}

TEST_CASE("The engine and the tile refuse a program or a configuration they cannot run", "[cycle][l3][dma]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    auto refuse = [&](const L3Program& p, const std::string& what) {
        CHECK_THROWS_WITH(sim.add<DmaEngine>("dma", d, DmaEngine::Config{2, 4096, 8, 4}, p), ContainsSubstring(what));
        CHECK_THROWS_WITH(sim.add<L3Tile>("l3", d, L3Tile::Config{2, 4096, 4}, p), ContainsSubstring(what));
    };
    refuse({l3_load(2, 0, 64)}, "action 0 (Load slot 2): there are 2 slots");
    refuse({l3_release(0), l3_store(0, 0, 100)}, "action 1 (Store slot 0): 100 bytes, not a positive multiple of 64");
    refuse({l3_load(0, 0, 0)}, "0 bytes, not a positive multiple of 64");
    refuse({l3_load(0, 0, 8192)}, "8192 bytes, past the slot's 4096");
    refuse({l3_load(0, 32, 64)}, "DRAM address 32 is not on a 64-byte burst boundary");
    CHECK_THROWS_WITH(sim.add<DmaEngine>("dma", d, DmaEngine::Config{2, 4096, 0, 4}, L3Program{}),
                      ContainsSubstring("the window and the store buffer must be positive"));
    CHECK_THROWS_WITH(sim.add<L3Tile>("l3", d, L3Tile::Config{2, 100, 4}, L3Program{}),
                      ContainsSubstring("the bytes whole beats"));
    SECTION("not connected") {
        auto& dma = sim.add<DmaEngine>("dma", d, DmaEngine::Config{}, L3Program{});
        auto& l3 = sim.add<L3Tile>("l3", d, L3Tile::Config{}, L3Program{});
        CHECK_THROWS_WITH(dma.evaluate(0), ContainsSubstring("not connected"));
        CHECK_THROWS_WITH(l3.evaluate(0), ContainsSubstring("not connected"));
    }
}
