// ============================================================================
// tests/cycle/test_cycle_streamer.cpp
// The Streamer and the L1 lane buffers on the cycle core, step 2b (#543;
// docs/plans/cycle-based-kpu-prototype.md §5.1, §7; docs/plans/systolic-tile-at-lca.md §3.2-3.3).
//
// The operands go L3 -> L2 by the BlockMover (step 2a), and the Streamer's commands read them, one A
// row and one B row a clock, into the L1 lane buffers, which stream them to a stand-in for the array.
// Every clock the lanes check their credit is conserved (it is held exactly when left <= 1), the L2
// its banks' ports and slot states, and the rig that the BlockMover holds no live slot's credit.
// Every clock below is measured from the run, never computed by the component from a formula: the
// reads, the loads, `completed`, the stream's skew, the commands' spacing.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "streamer_rig.hpp"

#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::isa::MatrixID;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
constexpr std::uint64_t kA = 0x10000;           // where A's rows are in L3
constexpr std::uint64_t kB = 0x20000;           // B's, one row per column (pre-arranged)
constexpr std::uint32_t kRow = 16;              // the L2's row: 16 int8 elements

const TileID kTileA{MatrixID::A, 0, 0, 0};
const TileID kTileB{MatrixID::B, 0, 0, 0};

// Element k of L3 row r of the operand: distinct per operand, row and element.
std::uint8_t element(char side, std::uint32_t r, std::uint32_t k) {
    return static_cast<std::uint8_t>((side == 'A' ? 1 : 101) + r * 7 + k * 3);
}

// A reduction chain of `blocks` commands on an h x w tile: command n reads rows n*h.. of A's slot 0
// and n*w.. of B's slot 1, K elements a lane.
struct Chain {
    std::uint32_t h = 4, w = 4, k = 16, blocks = 1;
};

L2Program l2_program(const Chain& c) {
    L2Program p{l2_move(0, kTileA, kA, kRow, c.blocks * c.h), l2_move(1, kTileB, kB, kRow, c.blocks * c.w)};
    for (std::uint32_t n = 0; n < c.blocks; ++n) {
        p.push_back(l2_feed(0, kTileA, c.h));
        p.push_back(l2_feed(1, kTileB, c.w));
    }
    p.push_back(l2_release(0, kTileA));
    p.push_back(l2_release(1, kTileB));
    return p;
}

StreamerProgram commands(const Chain& c) {
    StreamerProgram p;
    for (std::uint32_t n = 0; n < c.blocks; ++n)
        p.push_back({{0, kTileA, n * c.h, c.h}, {1, kTileB, n * c.w, c.w}, c.k, n == 0, n + 1 == c.blocks});
    return p;
}

void load(StreamerRig& r, const Chain& c) {
    for (std::uint32_t row = 0; row < c.blocks * c.h; ++row) {
        std::vector<std::uint8_t> v(kRow);
        for (std::uint32_t k = 0; k < kRow; ++k) v[k] = element('A', row, k);
        r.l3.load(kA + std::uint64_t(row) * kRow, v);
    }
    for (std::uint32_t row = 0; row < c.blocks * c.w; ++row) {
        std::vector<std::uint8_t> v(kRow);
        for (std::uint32_t k = 0; k < kRow; ++k) v[k] = element('B', row, k);
        r.l3.load(kB + std::uint64_t(row) * kRow, v);
    }
}

// Lane m of `side` streamed its blocks' elements, in order, one a clock with no gap, `first` on the
// chain's first element and `last` on its last. Returns the clock the sink took its first element.
Cycle check_lane(const StreamerRig& r, const Chain& c, char side, std::uint32_t m) {
    CAPTURE(side, m);
    const auto got = r.sink.lane(side, m);
    const std::uint32_t lanes = side == 'A' ? c.h : c.w;
    REQUIRE(got.size() == c.blocks * c.k);
    for (std::size_t i = 0; i < got.size(); ++i) {
        CAPTURE(i);
        const std::uint32_t n = static_cast<std::uint32_t>(i / c.k), k = static_cast<std::uint32_t>(i % c.k);
        CHECK(got[i].e.bits == element(side, n * lanes + m, k));
        CHECK(got[i].e.command == n);
        CHECK(got[i].e.k == k);
        CHECK(got[i].e.first == (i == 0));
        CHECK(got[i].e.last == (i + 1 == got.size()));
        if (i > 0) CHECK(got[i].at == got[i - 1].at + 1);
    }
    return got.front().at;
}

StreamerRig::Options options() { return StreamerRig::Options{}; }

}  // namespace

TEST_CASE("One command: one A row and one B row a clock, loaded D later, streamed with the array's skew",
          "[cycle][streamer][l1]") {
    const Chain c{4, 4, 16, 1};
    StreamerRig r(l2_program(c), commands(c), options());
    load(r, c);
    REQUIRE(r.sim.run_until_idle(kLimit));
    const Cycle D = r.st.config().read_delay;
    REQUIRE(r.st.records().size() == 1);
    const Cycle a = r.st.records()[0].accepted;

    // Read slot m reads A row m and B row m at a + 1 + m: one of each per clock, never more.
    std::map<std::pair<Cycle, char>, int> per_clock;
    for (const auto& i : r.st.issues()) {
        CAPTURE(i.at, i.side, i.lane);
        CHECK(i.at == a + 1 + i.lane);
        ++per_clock[{i.at, i.side}];
    }
    CHECK(r.st.issues().size() == 8);
    for (const auto& [key, n] : per_clock) CHECK(n == 1);

    // `completed` is the clock the last row loaded: the last read's, D later.
    REQUIRE(r.st.records()[0].completed);
    CHECK(*r.st.records()[0].completed == a + D + 4);
    CHECK(r.st.max_round_trip() <= D);

    // Lane m loads at a + 1 + D + m and streams from the next clock; the sink takes each element a
    // clock after it leaves. So lane m's stream starts m clocks after lane 0's: the skew.
    for (std::uint32_t m = 0; m < 4; ++m) {
        CHECK(check_lane(r, c, 'A', m) == a + 3 + D + m);
        CHECK(check_lane(r, c, 'B', m) == a + 3 + D + m);
    }
    // The operand slots were released, their credits are back at the BlockMover, and the Streamer
    // waited for its tiles (the tile-ready tokens) before it fired.
    CHECK(r.l2.state(0) == L2::State::Free);
    CHECK(r.l2.state(1) == L2::State::Free);
    CHECK(r.bm.holds(0));
    CHECK(r.bm.holds(1));
    CHECK(r.st.counts().stalls[static_cast<std::size_t>(Stall::NoData)] > 0);
    CHECK(r.st.counts().completions == 1);
}

TEST_CASE("A tile smaller than the array runs on its corner: L = max(h, w) read slots", "[cycle][streamer][l1]") {
    const Chain c{2, 4, 8, 1};
    StreamerRig r(l2_program(c), commands(c), options());
    load(r, c);
    REQUIRE(r.sim.run_until_idle(kLimit));
    const Cycle a = r.st.records()[0].accepted, D = r.st.config().read_delay;
    std::uint32_t reads_a = 0, reads_b = 0;
    for (const auto& i : r.st.issues()) (i.side == 'A' ? reads_a : reads_b)++;
    CHECK(reads_a == 2);
    CHECK(reads_b == 4);
    CHECK(*r.st.records()[0].completed == a + D + 4);
    for (std::uint32_t m = 0; m < 2; ++m) check_lane(r, c, 'A', m);
    for (std::uint32_t m = 0; m < 4; ++m) check_lane(r, c, 'B', m);
    CHECK(r.sink.lane('A', 2).empty());   // the lanes past the tile idle
    CHECK(r.sink.lane('A', 3).empty());
}

TEST_CASE("Early release: `completed` comes before the lanes finish, and the operand slots are freed by then",
          "[cycle][streamer][l1][l2]") {
    const Chain c{4, 4, 16, 1};
    StreamerRig r(l2_program(c), commands(c), options());
    load(r, c);
    std::optional<Picoseconds> freed[2];
    bool live[2] = {false, false};
    r.sim.add_invariant("slot freed", [&] {
        for (std::uint32_t s = 0; s < 2; ++s) {
            if (r.l2.live(s)) live[s] = true;
            else if (live[s] && !freed[s]) freed[s] = r.sim.now();
        }
    });
    REQUIRE(r.sim.run_until_idle(kLimit));
    const Picoseconds completed = r.kpu.edge(*r.st.records()[0].completed);
    Cycle last = 0;
    for (const auto& g : r.sink.got()) last = std::max(last, g.at);
    for (std::uint32_t s = 0; s < 2; ++s) {
        CAPTURE(s);
        REQUIRE(freed[s]);
        CHECK(*freed[s] <= completed);           // the slot can be released at `completed`
        CHECK(*freed[s] < r.kpu.edge(last));     // long before the array has its last element
    }
}

TEST_CASE("A K chain of commands back to back: every lane streams with no gap, the chain marked once",
          "[cycle][streamer][l1]") {
    const Chain c{4, 4, 16, 3};
    StreamerRig r(l2_program(c), commands(c), options());
    load(r, c);
    REQUIRE(r.sim.run_until_idle(kLimit));
    const Cycle D = r.st.config().read_delay;
    const auto& rec = r.st.records();
    REQUIRE(rec.size() == 3);
    for (std::size_t n = 0; n < 3; ++n) {
        CAPTURE(n);
        if (n > 0) CHECK(rec[n].accepted == rec[n - 1].accepted + 16);   // I = max(K, h, w)
        CHECK(*rec[n].completed == rec[n].accepted + D + 4);
    }
    for (std::uint32_t m = 0; m < 4; ++m) {
        check_lane(r, c, 'A', m);   // 48 elements a lane, consecutive clocks
        check_lane(r, c, 'B', m);
        // Each block after the first loaded in the clock its lane's last element left.
        CHECK(r.st.a_lanes().stats(m).seamless == 2);
        CHECK(r.st.b_lanes().stats(m).seamless == 2);
    }
}

TEST_CASE("A command fired a clock early loads a lane without its credit: an L1Fault", "[cycle][streamer][l1]") {
    const Chain c{4, 4, 16, 2};
    StreamerRig::Options o = options();
    o.interval_adjust = -1;
    StreamerRig r(l2_program(c), commands(c), o);
    load(r, c);
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit),
                        ContainsSubstring("a load into lane 0 for command 1 without its credit") &&
                            ContainsSubstring("1 more element to stream"));
}

TEST_CASE("A row not back by its load is an L1Fault: the read delay a configuration needs is measured",
          "[cycle][streamer][l1]") {
    const Chain c{4, 4, 16, 2};
    const auto run = [&](std::uint32_t delay) {
        StreamerRig::Options o = options();
        o.read_delay = delay;
        StreamerRig r(l2_program(c), commands(c), o);
        load(r, c);
        REQUIRE(r.sim.run_until_idle(kLimit));
        for (std::uint32_t m = 0; m < 4; ++m) check_lane(r, c, 'A', m);
        return r.st.max_round_trip();
    };
    const Cycle need = run(64);
    CAPTURE(need);
    CHECK(need > 16);   // the L2's 4 clocks of 4 ns, and the crossings both ways
    CHECK(run(static_cast<std::uint32_t>(need)) == need);   // the round trip does not depend on D
    StreamerRig::Options o = options();
    o.read_delay = static_cast<std::uint32_t>(need - 1);
    StreamerRig r(l2_program(c), commands(c), o);
    load(r, c);
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("is not back from L2 by its load"));
}

TEST_CASE("The lane buffer's credit is left <= 1: a load before it is a fault, at it a seamless block",
          "[cycle][l1]") {
    L1LaneBuffers l("A", 1, 8);
    l.load(0, 0, {1, 2, 3, 4}, 7, true, true);
    l.settle();
    l.check(0);
    CHECK_FALSE(l.credit(0));
    std::vector<L1Element> out;
    for (Cycle t = 1; t <= 2; ++t) {
        out.push_back(*l.stream(t, 0));
        l.settle();
        l.check(t);
    }
    // Clock 3: two elements left at its start, so no credit; the load faults.
    out.push_back(*l.stream(3, 0));
    REQUIRE_THROWS_AS(l.load(3, 0, {5, 6}, 8, false, true), L1Fault);
    REQUIRE_THROWS_WITH(l.load(3, 0, {5, 6}, 8, false, true), ContainsSubstring("lane 0 for command 8 without its credit"));
    l.settle();
    l.check(3);
    CHECK(l.credit(0));   // one left: its last element leaves next clock
    // Clock 4: the last element leaves and the next block loads, streaming from clock 5.
    out.push_back(*l.stream(4, 0));
    l.load(4, 0, {5, 6}, 8, false, true);
    l.settle();
    l.check(4);
    CHECK(l.stats(0).seamless == 1);
    out.push_back(*l.stream(5, 0));
    out.push_back(*l.stream(6, 0));
    REQUIRE(out.size() == 6);
    for (std::uint32_t i = 0; i < 6; ++i) CHECK(out[i].bits == i + 1);
    CHECK(out[0].first);
    CHECK(out[3].last);   // command 7's block is a whole chain: its k = 3 is `last`
    CHECK_FALSE(out[4].first);
    CHECK(out[5].last);
    CHECK_FALSE(l.stream(7, 0));
}

TEST_CASE("The L2's two readers share its banks' ports: a Writeback of A while the Streamer reads it",
          "[cycle][streamer][l2][blockmover]") {
    // Both read slot 0 at once: the slot counts its two readers, and the Release waits for both.
    const Chain c{4, 4, 16, 1};
    L2Program p{l2_move(0, kTileA, kA, kRow, 4), l2_move(1, kTileB, kB, kRow, 4), l2_feed(0, kTileA, 4),
                l2_feed(1, kTileB, 4), l2_writeback(0, kTileA, 0x80000, kRow, 4), l2_release(0, kTileA),
                l2_release(1, kTileB)};
    StreamerRig::Options o = options();
    o.read_delay = 64;
    StreamerRig r(p, commands(c), o);
    load(r, c);
    bool both = false;
    r.sim.add_invariant("two readers", [&] { both = both || r.l2.readers(0) == 2; });
    REQUIRE(r.sim.run_until_idle(kLimit));
    for (std::uint32_t m = 0; m < 4; ++m) {
        check_lane(r, c, 'A', m);
        check_lane(r, c, 'B', m);
        std::vector<std::uint8_t> v(kRow);
        for (std::uint32_t k = 0; k < kRow; ++k) v[k] = element('A', m, k);
        CHECK(r.l3.peek(0x80000 + std::uint64_t(m) * kRow, kRow) == v);
    }
    CHECK(both);
    std::uint64_t reads = 0;
    for (const auto& b : r.l2.banks()) reads += b.reads;
    CHECK(reads == 12);   // the Streamer's 8 and the Writeback's 4
    CHECK(r.l2.state(0) == L2::State::Free);
}

TEST_CASE("A tile moved into its slot again is read only after its own Move: a token admits its residency's Feeds",
          "[cycle][streamer][l2]") {
    // A[0] and B[0] each go through their slot twice, the second time with other data (L3 rows 4-7).
    // The first residency's token must not admit the second command: it would read the first
    // residency's data, or overrun its Release's count of consumers (CodeRabbit, #593).
    L2Program p;
    for (std::uint32_t n = 0; n < 2; ++n) {
        p.push_back(l2_move(0, kTileA, kA + std::uint64_t(n) * 4 * kRow, kRow, 4));
        p.push_back(l2_move(1, kTileB, kB + std::uint64_t(n) * 4 * kRow, kRow, 4));
        p.push_back(l2_feed(0, kTileA, 4));
        p.push_back(l2_feed(1, kTileB, 4));
        p.push_back(l2_release(0, kTileA));
        p.push_back(l2_release(1, kTileB));
    }
    const StreamerCommand cmd{{0, kTileA, 0, 4}, {1, kTileB, 0, 4}, 16, true, true};
    StreamerRig r(p, {cmd, cmd}, options());
    load(r, Chain{4, 4, 16, 2});
    REQUIRE(r.sim.run_until_idle(kLimit));
    for (const char side : {'A', 'B'})
        for (std::uint32_t m = 0; m < 4; ++m) {
            CAPTURE(side, m);
            const auto got = r.sink.lane(side, m);
            REQUIRE(got.size() == 32);
            for (std::size_t i = 0; i < 32; ++i) {
                CAPTURE(i);
                const std::uint32_t n = static_cast<std::uint32_t>(i / 16), k = static_cast<std::uint32_t>(i % 16);
                CHECK(got[i].e.bits == element(side, n * 4 + m, k));
                CHECK(got[i].e.first == (k == 0));
                CHECK(got[i].e.last == (k == 15));
            }
        }
    // The second command fired only after the second Moves made the slots READY again.
    REQUIRE(r.st.records().size() == 2);
    CHECK(r.l2.acquires() == 4);
    CHECK(r.l2.releases() == 4);
}

TEST_CASE("Two runs are identical", "[cycle][streamer]") {
    const Chain c{4, 4, 16, 3};
    const auto run = [&] {
        StreamerRig r(l2_program(c), commands(c), options());
        load(r, c);
        REQUIRE(r.sim.run_until_idle(kLimit));
        std::vector<std::uint64_t> v{static_cast<std::uint64_t>(r.sim.now()), r.st.counts().busy,
                                     static_cast<std::uint64_t>(r.st.max_round_trip())};
        for (const auto& g : r.sink.got()) v.insert(v.end(), {static_cast<std::uint64_t>(g.at), g.e.bits});
        for (const auto& b : r.l2.banks()) v.insert(v.end(), {b.reads, b.stalls, b.in_flight_integral});
        return v;
    };
    CHECK(run() == run());
}

TEST_CASE("The Streamer's program is checked before anything runs", "[cycle][streamer]") {
    Domain d = Domain::from_ghz("kpu", 1.0);
    const Streamer::Config cfg{};
    const auto cmd = [](std::uint32_t h, std::uint32_t w, std::uint32_t k) {
        return StreamerCommand{{0, kTileA, 0, h}, {1, kTileB, 0, w}, k, true, true};
    };
    CHECK_NOTHROW(Streamer("st", d, cfg, {cmd(4, 4, 16)}));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {cmd(5, 4, 16)}), ContainsSubstring("5 A lanes, the array has 1 to 4"));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {cmd(4, 0, 16)}), ContainsSubstring("0 B lanes"));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {cmd(4, 4, 17)}), ContainsSubstring("K = 17, a row holds 1 to 16"));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {cmd(4, 4, 0)}), ContainsSubstring("K = 0"));
    StreamerCommand past = cmd(4, 4, 16);
    past.a.row = 13;
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {past}), ContainsSubstring("A rows 13 to 16, past the slot's 16"));
    StreamerCommand one = cmd(4, 4, 16);
    one.b.slot = 0;
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {one}), ContainsSubstring("A and B in one slot"));
    // A chain opened and never closed, one continued that was never opened, and one opened inside another.
    StreamerCommand open = cmd(4, 4, 16), cont = cmd(4, 4, 16);
    open.last = false;
    cont.first = false;
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {open}), ContainsSubstring("leaves its chain open"));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {cont}), ContainsSubstring("continues a chain no command opened"));
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {open, cmd(4, 4, 16)}), ContainsSubstring("opens a chain inside"));
    CHECK_NOTHROW(Streamer("st", d, cfg, {open, cont}));
    // The L2 program must feed what the commands read, and a Feed no more rows than its Move wrote.
    const Chain c{4, 4, 16, 2};
    CHECK_NOTHROW(check_feeds(commands(c), l2_program(c)));
    CHECK_THROWS_WITH(check_feeds(commands(c), l2_program(Chain{4, 4, 16, 1})),
                      ContainsSubstring("slot 0: the commands read it 2 times, the L2 program's Feeds 1"));
    CHECK_THROWS_WITH(check_feeds(commands(Chain{4, 2, 16, 2}), l2_program(c)),
                      ContainsSubstring("slot 1: read 0 of B[0,0,0] is 2 rows by the commands, 4 by its Feed"));
    // The order across a slot's tiles is checked too: commands that read A1 before A0, where the L2
    // feeds A0, releases it and only then moves A1 in, would wait forever (CodeRabbit, #593).
    const TileID a1{MatrixID::A, 1, 0, 0}, b1{MatrixID::B, 1, 0, 0};
    const L2Program two{l2_move(0, kTileA, kA, kRow, 4), l2_move(1, kTileB, kB, kRow, 4), l2_feed(0, kTileA, 4),
                        l2_feed(1, kTileB, 4), l2_release(0, kTileA), l2_release(1, kTileB),
                        l2_move(0, a1, kA, kRow, 4), l2_move(1, b1, kB, kRow, 4), l2_feed(0, a1, 4),
                        l2_feed(1, b1, 4), l2_release(0, a1), l2_release(1, b1)};
    const StreamerCommand first{{0, kTileA, 0, 4}, {1, kTileB, 0, 4}, 16, true, true};
    const StreamerCommand second{{0, a1, 0, 4}, {1, b1, 0, 4}, 16, true, true};
    CHECK_NOTHROW(check_feeds({first, second}, two));
    CHECK_THROWS_WITH(check_feeds({second, first}, two),
                      ContainsSubstring("slot 0: read 0 is of A[1,0,0] by the commands, of A[0,0,0] by the L2"));
    // Each slot's order can match while the order across slots cannot complete: the first command
    // needs B1, whose Move waits behind A1's, which waits for A0's Release, which waits for the
    // first command (CodeRabbit, #593).
    const L2Program three{l2_move(0, kTileA, kA, kRow, 4), l2_move(1, kTileB, kB, kRow, 4), l2_feed(0, kTileA, 4),
                          l2_feed(1, kTileB, 4), l2_release(0, kTileA), l2_release(1, kTileB),
                          l2_move(0, a1, kA, kRow, 4), l2_move(2, b1, kB, kRow, 4), l2_feed(0, a1, 4),
                          l2_feed(2, b1, 4), l2_release(0, a1), l2_release(2, b1)};
    const StreamerCommand x{{0, kTileA, 0, 4}, {2, b1, 0, 4}, 16, true, true};
    const StreamerCommand y{{0, a1, 0, 4}, {1, kTileB, 0, 4}, 16, true, true};
    CHECK_THROWS_WITH(check_feeds({x, y}, three),
                      ContainsSubstring("the schedule cannot complete: the Streamer waits at command 0") &&
                          ContainsSubstring("the BlockMover at action 6 (Move A[1,0,0] in slot 0)"));
    const StreamerCommand x_ok{{0, kTileA, 0, 4}, {1, kTileB, 0, 4}, 16, true, true};
    const StreamerCommand y_ok{{0, a1, 0, 4}, {2, b1, 0, 4}, 16, true, true};
    CHECK_NOTHROW(check_feeds({x_ok, y_ok}, three));
    // A Writeback is a consumer the Release waits for, as the L2 counts it.
    const L2Program wb{l2_move(0, kTileA, kA, kRow, 4), l2_move(1, kTileB, kB, kRow, 4), l2_feed(0, kTileA, 4),
                       l2_feed(1, kTileB, 4), l2_writeback(0, kTileA, 0x80000, kRow, 4), l2_release(0, kTileA),
                       l2_release(1, kTileB)};
    CHECK_NOTHROW(check_feeds({first}, wb));
    CHECK_THROWS_WITH(validate_l2_program({l2_move(0, kTileA, 0, 16, 4), l2_feed(0, kTileA, 8)}, 2, 16, 16, "l2"),
                      ContainsSubstring("8 rows, but its Move wrote 4"));
    CHECK_THROWS_WITH(validate_l2_program({l2_move(0, kTileA, 0, 16, 4), l2_feed(0, kTileB, 4)}, 2, 16, 16, "l2"),
                      ContainsSubstring("the slot holds A[0,0,0]"));
}
