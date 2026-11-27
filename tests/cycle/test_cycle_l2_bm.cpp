// ============================================================================
// tests/cycle/test_cycle_l2_bm.cpp
// The L2 and the BlockMover on the cycle core, step 2a (#542; docs/plans/cycle-based-kpu-prototype.md
// §5.1, §7): L3 -> L2 by strided rows under the slot's credit, and back.
//
// One tile, then several through two slots, each moved into L2 and written back to L3, bit for bit;
// every row in the bank its slot and row name. In every run the L2 checks every clock that no bank
// served more than its ports and that every slot's rows agree with its #387 state, and the rig
// checks the BlockMover never holds a live slot's credit. Then: bank contention, the reads in flight
// per bank (#495's question, by measurement), the faults, determinism and the refusals.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "l2_bm_rig.hpp"

#include <sw/kpu/cycle/compute/block_mover.hpp>
#include <sw/kpu/cycle/compute/l2.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::isa::MatrixID;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
constexpr std::uint64_t kIn = 0x10000;          // where the tiles are in L3
constexpr std::uint64_t kOut = 0x400000;        // where they are written back

TileID tile(std::uint32_t t) { return TileID{t % 2 ? MatrixID::B : MatrixID::A, t, 0, 0}; }

// Row `r` of tile `t`: distinct bytes per tile and row.
std::vector<std::uint8_t> row_bytes(std::uint32_t t, std::uint32_t r, std::uint32_t n) {
    std::vector<std::uint8_t> v(n);
    for (std::uint32_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(t * 53 + r * 17 + i * 3 + (i >> 5));
    return v;
}

struct Tiles {
    std::uint32_t count = 1, rows = 8, pitch = 512, slots = 2;
};

// `n.count` tiles of `n.rows` rows through `n.slots` slots, round robin: Move, Writeback, Release.
// Tile t's rows are at kIn + t * 64 KiB, `pitch` apart; its writeback packs them at kOut + t * 64 KiB.
L2Program ring(const Tiles& n, std::uint32_t row) {
    L2Program p;
    for (std::uint32_t t = 0; t < n.count; ++t) {
        const std::uint32_t s = t % n.slots;
        p.push_back(l2_move(s, tile(t), kIn + std::uint64_t(t) * 0x10000, n.pitch, n.rows));
        p.push_back(l2_writeback(s, tile(t), kOut + std::uint64_t(t) * 0x10000, row, n.rows));
        p.push_back(l2_release(s, tile(t)));
    }
    return p;
}

void load(L2BmRig& r, const Tiles& n) {
    const std::uint32_t row = r.l2.config().row_bytes;
    for (std::uint32_t t = 0; t < n.count; ++t)
        for (std::uint32_t k = 0; k < n.rows; ++k)
            r.l3.load(kIn + std::uint64_t(t) * 0x10000 + std::uint64_t(k) * n.pitch, row_bytes(t, k, row));
}

void check_written_back(const L2BmRig& r, const Tiles& n) {
    const std::uint32_t row = r.l2.config().row_bytes;
    for (std::uint32_t t = 0; t < n.count; ++t)
        for (std::uint32_t k = 0; k < n.rows; ++k) {
            CAPTURE(t, k);
            CHECK(r.l3.peek(kOut + std::uint64_t(t) * 0x10000 + std::uint64_t(k) * row, row) == row_bytes(t, k, row));
        }
}

L2BmRig::Options options(Tiles n) {
    L2BmRig::Options o;
    o.l2.slots = n.slots;
    return o;
}

}  // namespace

TEST_CASE("One tile: L3 -> L2 by strided rows, each in its bank, and back to L3 bit for bit", "[cycle][l2][blockmover]") {
    const Tiles n{1, 8, 512, 2};
    const L2Program p = ring(n, 256);
    L2BmRig r(p, options(n));
    load(r, n);
    // Every row in its bank, with the Move's bytes, at the clock its slot becomes READY (or READING:
    // the Writeback's first read can be served in the clock the last row lands).
    bool seen_ready = false;
    r.sim.add_invariant("rows in their banks", [&] {
        if (seen_ready || (r.l2.state(0) != L2::State::Ready && r.l2.state(0) != L2::State::Reading)) return;
        seen_ready = true;
        for (std::uint32_t k = 0; k < n.rows; ++k) {
            CAPTURE(k);
            CHECK(r.l2.bank(0, k) == k % r.l2.config().banks);
            CHECK(r.l2.bank_row(r.l2.bank(0, k), r.l2.index(0, k)) == row_bytes(0, k, 256));
        }
    });
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(seen_ready);
    check_written_back(r, n);
    // The slot went FREE -> OWNED -> READY -> READING -> READY -> FREE, and its credit is back.
    CHECK(r.l2.state(0) == L2::State::Free);
    CHECK(r.l2.acquires() == 1);
    CHECK(r.l2.releases() == 1);
    CHECK(r.bm.holds(0));
    CHECK(r.bm.counts().completions == 2);   // the Move and the Writeback
    CHECK(r.l2.counts().completions == 1);   // the tile's residency
    std::uint64_t reads = 0, writes = 0;
    for (const auto& b : r.l2.banks()) {
        reads += b.reads;
        writes += b.writes;
    }
    CHECK(reads == n.rows);
    CHECK(writes == n.rows);
    CHECK(r.l3.beats_written() == n.rows * 256 / 64);
}

TEST_CASE("A slot's rows map onto the banks slot-major", "[cycle][l2]") {
    Domain d = Domain::from_ghz("l2", 0.25);
    L2 l2("l2", d, L2::Config{3, 6, 4, 64, 4, 2, 1}, L2Program{});
    // slot s row k is global row s * 6 + k: bank (s * 6 + k) mod 4, index (s * 6 + k) / 4.
    for (std::uint32_t s = 0; s < 3; ++s)
        for (std::uint32_t k = 0; k < 6; ++k) {
            CAPTURE(s, k);
            CHECK(l2.bank(s, k) == (s * 6 + k) % 4);
            CHECK(l2.index(s, k) == (s * 6 + k) / 4);
        }
}

TEST_CASE("Several tiles through two slots, each slot reused after its Release", "[cycle][l2][blockmover]") {
    const Tiles n{6, 16, 1024, 2};
    L2BmRig r(ring(n, 256), options(n));
    load(r, n);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_written_back(r, n);
    CHECK(r.l2.acquires() == n.count);
    CHECK(r.l2.releases() == n.count);
    CHECK(r.bm.holds(0));
    CHECK(r.bm.holds(1));
}

TEST_CASE("A Move waits, with no fault, for its slot's credit", "[cycle][l2][blockmover]") {
    // Four tiles through one slot: each Move after the first needs the credit the Release before it
    // returns. With the credit crossing slowed, the mover waits for it, and the data is still right;
    // every 40 edges more the credit takes adds 40 edges to each of the three waits.
    const Tiles n{4, 8, 512, 1};
    const auto waits = [&](Cycle sync) {
        L2BmRig::Options o = options(n);
        o.credit_sync = sync;
        L2BmRig r(ring(n, 256), o);
        load(r, n);
        REQUIRE(r.sim.run_until_idle(kLimit));
        check_written_back(r, n);
        return r.bm.counts().stalls[static_cast<std::size_t>(Stall::NoCredit)];
    };
    const std::uint64_t fast = waits(2), slow = waits(40), slower = waits(80);
    CHECK(fast == 0);   // the credit is back before the mover finishes the writeback
    CHECK(slow > 0);
    CHECK(slower - slow == 3 * 40);
}

TEST_CASE("No bank serves more than its ports: contention is a wait, counted on the bank", "[cycle][l2]") {
    // One bank, so every row of the tile is in it; the mover asks for rows faster (1 GHz) than the
    // bank (250 MHz, one or two read ports) serves them.
    const Tiles n{1, 16, 256, 1};
    const auto run = [&](std::uint32_t ports) {
        L2BmRig::Options o = options(n);
        o.l2.banks = 1;
        o.l2.read_ports = ports;
        o.read_window = 16;
        L2BmRig r(ring(n, 256), o);
        load(r, n);
        REQUIRE(r.sim.run_until_idle(kLimit));
        check_written_back(r, n);
        REQUIRE(r.l2.banks().size() == 1);
        CHECK(r.l2.banks()[0].reads == n.rows);
        return r.l2.banks()[0].stalls;
    };
    const std::uint64_t one = run(1), two = run(2);
    CHECK(one > 0);
    CHECK(one > two);
}

TEST_CASE("Reads in flight per bank, measured: Little's law holds exactly", "[cycle][l2]") {
    // A read is in flight from the clock it is served to the clock its row leaves. With the data
    // link never full, that is the read latency for every read, so the reads-in-flight integral over
    // the banks is reads x latency, and no bank ever holds more than its ports times the latency.
    for (std::uint32_t latency : {1u, 4u, 8u}) {
        CAPTURE(latency);
        const Tiles n{1, 16, 256, 1};
        L2BmRig::Options o = options(n);
        o.l2.read_latency = latency;
        o.l2_link = 32;
        o.read_window = 16;
        o.row_buffers = 16;
        L2BmRig r(ring(n, 256), o);
        load(r, n);
        REQUIRE(r.sim.run_until_idle(kLimit));
        check_written_back(r, n);
        std::uint64_t reads = 0, integral = 0;
        for (const auto& b : r.l2.banks()) {
            reads += b.reads;
            integral += b.in_flight_integral;
            CHECK(b.max_in_flight <= std::uint64_t(r.l2.config().read_ports) * latency);
        }
        CHECK(reads == n.rows);
        CHECK(integral == reads * latency);
    }
}

TEST_CASE("The mover's requests in flight bound the writeback's rate (#495, measured)", "[cycle][l2][blockmover]") {
    // A window of one request leaves the banks idle while each row's latency passes; a window that
    // covers the latency keeps them busy. The measured reads in flight show which.
    const Tiles n{1, 16, 256, 1};
    const auto run = [&](std::uint32_t window) {
        L2BmRig::Options o = options(n);
        o.read_window = window;
        o.row_buffers = 16;
        o.l2_link = 16;
        L2BmRig r(ring(n, 256), o);
        load(r, n);
        REQUIRE(r.sim.run_until_idle(kLimit));
        check_written_back(r, n);
        std::uint64_t max_total = 0;
        for (const auto& b : r.l2.banks()) max_total = std::max(max_total, b.max_in_flight);
        return std::pair{r.sim.now(), max_total};
    };
    const auto [t1, m1] = run(1);
    const auto [t8, m8] = run(8);
    CHECK(m1 == 1);
    CHECK(m8 > 1);
    CHECK(t8 < t1);
}

TEST_CASE("A row of another tile into a live slot is a SlotFault", "[cycle][l2]") {
    const Tiles n{1, 4, 256, 2};
    L2BmRig r(ring(n, 256), options(n));
    load(r, n);
    // Before the mover's Move of A[0] into slot 0, a row of another tile claims the slot.
    r.l2_writes.push(0, L2RowWrite{999, 0, 0, 4, tile(7), std::vector<std::uint8_t>(256, 0)});
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("a Move into a live slot"));
}

TEST_CASE("A read waits for its tile, READY, with no fault", "[cycle][l2]") {
    // The writeback's reads reach L2 while its Move's rows still wait for the one bank's single
    // write port: they wait for the tile-ready token, and the data is right.
    const Tiles n{1, 16, 256, 1};
    L2BmRig::Options o = options(n);
    o.l2.banks = 1;
    o.row_buffers = 4;
    L2BmRig r(ring(n, 256), o);
    load(r, n);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_written_back(r, n);
}

TEST_CASE("Two runs are identical", "[cycle][l2][blockmover]") {
    const Tiles n{4, 8, 384, 2};
    const auto run = [&] {
        L2BmRig r(ring(n, 256), options(n));
        load(r, n);
        REQUIRE(r.sim.run_until_idle(kLimit));
        std::vector<std::uint64_t> v{static_cast<std::uint64_t>(r.sim.now()), r.bm.counts().busy, r.l2.counts().busy};
        for (const auto& b : r.l2.banks()) v.insert(v.end(), {b.reads, b.writes, b.stalls, b.in_flight_integral});
        return v;
    };
    CHECK(run() == run());
}

TEST_CASE("The program and the configurations are checked before anything runs", "[cycle][l2][blockmover]") {
    Domain d = Domain::from_ghz("l2", 0.25);
    const L2::Config c{};
    CHECK_THROWS_WITH(L2("l2", d, L2::Config{0, 16, 4, 256, 4, 2, 1}, {}), ContainsSubstring("must be positive"));
    CHECK_THROWS_WITH(BlockMover("bm", d, BlockMover::Config{2, 16, 256, 96, 2, 8}, {}),
                      ContainsSubstring("whole beats"));
    // A Move into a slot still holding a tile.
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(0, tile(0), 0, 256, 4), l2_move(0, tile(1), 0, 256, 4)}),
                      ContainsSubstring("no Release before this Move"));
    // Rows that overlap in L3, rows past the slot, a slot that does not exist, a Writeback of a tile
    // the slot does not hold.
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(0, tile(0), 0, 128, 4)}), ContainsSubstring("rows overlap"));
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(0, tile(0), 0, 256, 17)}), ContainsSubstring("a slot holds 1 to 16"));
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(2, tile(0), 0, 256, 4)}), ContainsSubstring("there are 2 slots"));
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(0, tile(0), 0, 256, 4), l2_writeback(0, tile(1), 0, 256, 4)}),
                      ContainsSubstring("the slot holds A[0,0,0]"));
    // A Writeback of more rows than its Move wrote is refused before anything runs (CodeRabbit,
    // #592); one of the same rows or fewer is a program.
    CHECK_THROWS_WITH(L2("l2", d, c, {l2_move(0, tile(0), 0, 256, 4), l2_writeback(0, tile(0), 0, 256, 8)}),
                      ContainsSubstring("8 rows, but its Move wrote 4"));
    CHECK_NOTHROW(L2("l2", d, c, {l2_move(0, tile(0), 0, 256, 4), l2_writeback(0, tile(0), 0, 256, 4),
                                  l2_writeback(0, tile(0), 0, 256, 2), l2_release(0, tile(0))}));
}

// --- the dense layouts (#548, the join): L3 holds a tile densely, L2 one row a lane ---------------

namespace {

// A dense R x C tile of `e`-byte elements, row-major, padded to 64-byte beats; element (r, c)'s
// bytes are distinct.
std::vector<std::uint8_t> dense_tile(std::uint32_t R, std::uint32_t C, std::uint32_t e) {
    std::vector<std::uint8_t> v((R * C * e + 63) / 64 * 64, 0);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t c = 0; c < C; ++c)
            for (std::uint32_t b = 0; b < e; ++b) v[(std::size_t(r) * C + c) * e + b] = static_cast<std::uint8_t>(r * 29 + c * 7 + b * 101 + 1);
    return v;
}
// Its transpose, C x R, packed the same way.
std::vector<std::uint8_t> transposed(const std::vector<std::uint8_t>& t, std::uint32_t R, std::uint32_t C, std::uint32_t e) {
    std::vector<std::uint8_t> v(t.size(), 0);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t c = 0; c < C; ++c)
            for (std::uint32_t b = 0; b < e; ++b) v[(std::size_t(c) * R + r) * e + b] = t[(std::size_t(r) * C + c) * e + b];
    return v;
}

}  // namespace

TEST_CASE("A dense tile into L2 lanes: its rows, or its columns (the transpose), and back to L3 dense",
          "[cycle][l2][blockmover][dense]") {
    // 16 x 16 int8, 256 bytes in L3: 16 lanes of 16 bytes in a slot of 16 rows of 256.
    const std::vector<std::uint8_t> t = dense_tile(16, 16, 1);
    for (const bool transpose : {false, true}) {
        CAPTURE(transpose);
        const L2Program p{l2_move_dense(0, tile(0), kIn, 16, 16, 1, transpose),
                          l2_writeback_dense(0, tile(0), kOut, 16, 16, 1), l2_release(0, tile(0))};
        L2BmRig::Options o;
        L2BmRig r(p, o);
        r.l3.load(kIn, t);
        bool seen = false;
        r.sim.add_invariant("lanes", [&] {
            if (seen || (r.l2.state(0) != L2::State::Ready && r.l2.state(0) != L2::State::Reading)) return;
            seen = true;
            // Lane m in L2 row m: the tile's row m, or its column m, then zeros to the row.
            for (std::uint32_t m = 0; m < 16; ++m) {
                CAPTURE(m);
                const auto row = r.l2.bank_row(r.l2.bank(0, m), r.l2.index(0, m));
                for (std::uint32_t k = 0; k < 16; ++k) CHECK(row[k] == t[transpose ? k * 16 + m : m * 16 + k]);
                for (std::uint32_t k = 16; k < 256; ++k) CHECK(row[k] == 0);
            }
        });
        REQUIRE(r.sim.run_until_idle(kLimit));
        CHECK(seen);
        // The Writeback packs each row's lane back densely: the tile, or its transpose.
        CHECK(r.l3.peek(kOut, t.size()) == (transpose ? transposed(t, 16, 16, 1) : t));
        CHECK(r.l3.beats_written() == t.size() / 64);
    }
}

TEST_CASE("A dense Writeback of int32 rows: each row's first w elements, the last beat padded", "[cycle][l2][blockmover][dense]") {
    // A 4 x 4 int32 result: 4 L2 rows whose first 16 bytes are the row, 64 bytes dense in L3.
    const std::vector<std::uint8_t> t = dense_tile(4, 4, 4);
    REQUIRE(t.size() == 64);
    L2BmRig::Options o;
    o.l2.row_bytes = 64;
    o.bus_bytes = 16;
    const L2Program p{l2_move(0, tile(0), kIn, 64, 4), l2_writeback_dense(0, tile(0), kOut, 4, 4, 4), l2_release(0, tile(0))};
    L2BmRig r(p, o);
    for (std::uint32_t i = 0; i < 4; ++i) {
        std::vector<std::uint8_t> row(64, 0xEE);   // past the lane: never written back
        std::copy(t.begin() + i * 16, t.begin() + (i + 1) * 16, row.begin());
        r.l3.load(kIn + i * 64, row);
    }
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(r.l3.peek(kOut, 64) == t);
    CHECK(r.l3.beats_written() == 4);   // 64 bytes in 16-byte beats
}

TEST_CASE("The transpose waits for the whole tile; a row lane goes as soon as its bytes are in",
          "[cycle][l2][blockmover][dense]") {
    // 16 x 16 int8 in 16-byte beats: 16 beats. Lane 0 of the rows is in after one beat; any
    // column needs all 16. The slot is OWNED from its first row's write: measured, never coded.
    const std::vector<std::uint8_t> t = dense_tile(16, 16, 1);
    std::map<bool, Cycle> owned;
    for (const bool transpose : {false, true}) {
        L2BmRig::Options o;
        o.bus_bytes = 16;
        o.l2.row_bytes = 16;
        const L2Program p{l2_move_dense(0, tile(0), kIn, 16, 16, 1, transpose), l2_release(0, tile(0))};
        L2BmRig r(p, o);
        r.l3.load(kIn, t);
        std::optional<Cycle> at;
        r.sim.add_invariant("owned", [&] {
            if (!at && r.l2.state(0) != L2::State::Free) at = r.kpu.cycle();
        });
        REQUIRE(r.sim.run_until_idle(kLimit));
        REQUIRE(at);
        owned[transpose] = *at;
    }
    // Fifteen more beats, a clock each, before a column is whole.
    CHECK(owned[true] >= owned[false] + 15);
}

TEST_CASE("Dense layouts are checked before anything runs", "[cycle][l2][blockmover][dense]") {
    L2BmRig::Options o;
    SECTION("a lane past the L2's row") {
        o.l2.row_bytes = 64;
        const L2Program p{l2_move_dense(0, tile(0), kIn, 4, 128, 1, false)};
        CHECK_THROWS_WITH(L2BmRig(p, o), ContainsSubstring("past the L2's 64-byte row"));
    }
    SECTION("L2 rows that are not the tile's lanes") {
        L2Action a = l2_move_dense(0, tile(0), kIn, 4, 8, 1, true);
        a.rows = 4;   // a transpose's lanes are its 8 columns
        CHECK_THROWS_WITH(L2BmRig(L2Program{a}, o), ContainsSubstring("a lane a column"));
    }
    SECTION("a Writeback by columns") {
        L2Action w = l2_writeback_dense(0, tile(0), kOut, 4, 4, 4);
        w.layout = L2Action::Layout::DenseColumns;
        w.rows = 4;
        const L2Program p{l2_move(0, tile(0), kIn, 256, 4), w};
        CHECK_THROWS_WITH(L2BmRig(p, o), ContainsSubstring("writes its rows back as the tile's rows"));
    }
    SECTION("a tile past the mover's tile buffer") {
        // The buffer is one L2 slot. Lanes that fit their rows always fit it, but the dense tile is
        // padded to 64-byte beats: a slot of one 16-byte row holds a 1 x 16 tile's lane, not its beat.
        o.l2.rows_per_slot = 1;
        o.l2.row_bytes = 16;
        o.bus_bytes = 16;
        CHECK_THROWS_WITH(L2BmRig(L2Program{l2_move_dense(0, tile(0), kIn, 1, 16, 1, false)}, o),
                          ContainsSubstring("for a tile buffer of one slot (16)"));
        o.l2.rows_per_slot = 4;   // 64 bytes: the beat fits
        CHECK_NOTHROW(L2BmRig(L2Program{l2_move_dense(0, tile(0), kIn, 1, 16, 1, false)}, o));
    }
    SECTION("an element that is not 1, 2 or 4 bytes") {
        CHECK_THROWS_WITH(L2BmRig(L2Program{l2_move_dense(0, tile(0), kIn, 4, 4, 3, false)}, o),
                          ContainsSubstring("a dense tile of 4 x 4 elements of 3 bytes"));
    }
}
