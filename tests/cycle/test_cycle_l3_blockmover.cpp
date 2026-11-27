// ============================================================================
// tests/cycle/test_cycle_l3_blockmover.cpp
// The L3 tile's BlockMover side, step 3b of the join (#548; docs/plans/cycle-based-kpu-prototype.md
// §6): DRAM -> DMA -> L3 -> BlockMover -> L2 -> BlockMover -> L3 -> DMA -> DRAM, bit for bit.
//
// The BlockMover's Reads are consumers of a residency (its Release waits for them), its Writes are
// producers (under the slot's credit, which the Release before them returns to the mover). The
// slot credits are checked every cycle three ways: the engine's, the mover's and the tile's.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "l3_bm_rig.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::isa::MatrixID;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
constexpr std::uint32_t kSlot = 1024;            // the L3 slot's bytes
constexpr std::uint64_t kDram = 0x100000;

TileID tile(MatrixID m, std::uint32_t n) { return TileID{m, n, 0, 0}; }
std::uint64_t at(std::uint32_t slot) { return std::uint64_t(slot) * kSlot; }

// A dense 16 x 16 int8 tile (256 bytes: four 64-byte beats), distinct per seed.
std::vector<std::uint8_t> tile16(std::uint32_t seed) {
    std::vector<std::uint8_t> v(256);
    for (std::uint32_t i = 0; i < 256; ++i) v[i] = static_cast<std::uint8_t>(seed * 41 + i * 3 + (i >> 4));
    return v;
}
std::vector<std::uint8_t> transposed16(const std::vector<std::uint8_t>& t) {
    std::vector<std::uint8_t> v(256);
    for (std::uint32_t r = 0; r < 16; ++r)
        for (std::uint32_t c = 0; c < 16; ++c) v[c * 16 + r] = t[r * 16 + c];
    return v;
}

}  // namespace

TEST_CASE("A tile from DRAM through L3 into L2 lanes and back through L3 to DRAM, bit for bit",
          "[cycle][l3][blockmover][join]") {
    // Slot 0 is the DMA's (a Load), slot 1 the mover's (a Write): each holds its credit at the start.
    for (const bool transpose : {false, true}) {
        CAPTURE(transpose);
        const std::vector<std::uint8_t> a = tile16(1);
        const L3Program l3p{l3_load(0, kDram, 256), l3_read(0, 256), l3_write(1, 256), l3_store(1, kDram + 0x1000, 256),
                            l3_release(0), l3_release(1)};
        L2Action wb = l2_writeback_dense(0, tile(MatrixID::A, 0), at(1), 16, 16, 1);
        wb.l3_slot = 1;
        const L2Program l2p{l2_move_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1, transpose), wb,
                            l2_release(0, tile(MatrixID::A, 0))};
        L3BmRig r(l3p, l2p, L3BmRig::Options{});
        CHECK(r.dma.holds(0));
        CHECK(!r.dma.holds(1));
        CHECK(r.bm.holds_l3(1));
        r.channel.preload(kDram, a);
        REQUIRE(r.sim.run_until_idle(kLimit));
        // The mover's lanes went into L2 as rows or columns; written back densely, they are the tile
        // or its transpose, stored by the engine.
        CHECK(r.channel.peek(kDram + 0x1000, 256) == (transpose ? transposed16(a) : a));
        CHECK(r.dma.holds(0));          // slot 0's next producer: none, so the engine's
        CHECK(r.dma.holds(1));
        CHECK(!r.bm.holds_l3(1));
        CHECK(r.l3.counts().completions == 2);   // two residencies, each released
    }
}

TEST_CASE("A Release waits for the BlockMover's Reads of its tile: the slot is not refilled under it",
          "[cycle][l3][blockmover][join]") {
    // The mover is slow to read A0 from slot 0: first it moves X into its one L2 slot and writes X
    // back, and only then moves A0. The engine meanwhile has A1 to load into slot 0, after A0's
    // Release. If the Release did not wait for the Read, A1 would land in slot 0 before the mover
    // read A0, and the mover's Read, bound to A0, would never be served.
    const std::vector<std::uint8_t> x = tile16(7), a0 = tile16(1), a1 = tile16(2);
    const L3Program l3p{l3_load(1, kDram + 0x2000, 256),  // X
                        l3_load(0, kDram, 256),           // A0
                        l3_read(1, 256),                  // the mover's Move of X
                        l3_write(2, 256),                 // its Writeback of X
                        l3_read(0, 256),                  // its Move of A0
                        l3_release(0),                    // A0's Release: after its Read
                        l3_load(0, kDram + 0x1000, 256),  // A1, into slot 0 again
                        l3_read(0, 256),                  // the Move of A1
                        l3_write(3, 256),                 // the Writeback of A0
                        l3_store(2, kDram + 0x10000, 256), l3_store(3, kDram + 0x11000, 256),
                        l3_release(1), l3_release(2), l3_release(3), l3_release(0)};
    auto wb = [](MatrixID m, std::uint32_t l3) {
        L2Action w = l2_writeback_dense(0, tile(m, 0), at(l3), 16, 16, 1);
        w.l3_slot = l3;
        return w;
    };
    // One L2 slot for everything, in turn: X (written back into slot 2), A0 (into slot 3), then A1,
    // moved and released (its Read is what frees slot 0 at the end).
    const L2Program l2p{l2_move_dense(0, tile(MatrixID::B, 0), at(1), 16, 16, 1, false), wb(MatrixID::B, 2),
                        l2_release(0, tile(MatrixID::B, 0)),
                        l2_move_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1, false), wb(MatrixID::A, 3),
                        l2_release(0, tile(MatrixID::A, 0)),
                        l2_move_dense(0, tile(MatrixID::A, 1), at(0), 16, 16, 1, false),
                        l2_release(0, tile(MatrixID::A, 1))};
    L3BmRig::Options o;
    o.l2.slots = 1;
    L3BmRig r(l3p, l2p, o);
    r.channel.preload(kDram, a0);
    r.channel.preload(kDram + 0x1000, a1);
    r.channel.preload(kDram + 0x2000, x);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(r.channel.peek(kDram + 0x10000, 256) == x);
    CHECK(r.channel.peek(kDram + 0x11000, 256) == a0);   // A0, as the mover read it: not A1
    // Five residencies: X, A0, X's copy, A1, A0's copy; A0's in slot 0 ended before A1's began.
    const auto& res = r.l3.residencies();
    REQUIRE(res.size() == 5);
    std::optional<Cycle> a0_released, a1_filling;
    for (const auto& x : res)
        if (x.slot == 0) (x.ticket == 1 ? a0_released = x.released_at : a1_filling = x.filling_from);
    REQUIRE(a0_released);
    REQUIRE(a1_filling);
    CHECK(*a1_filling > *a0_released);
}

TEST_CASE("A Release returns the slot's credit to its next producer: the engine's Load, then the mover's Write",
          "[cycle][l3][blockmover][join]") {
    // One L3 slot, reused: A is loaded into it, read into L2, released; the result is written back
    // into the same slot (its credit now the mover's), stored, released (back to the engine).
    const std::vector<std::uint8_t> a = tile16(3);
    const L3Program l3p{l3_load(0, kDram, 256), l3_read(0, 256), l3_release(0), l3_write(0, 256),
                        l3_store(0, kDram + 0x1000, 256), l3_release(0)};
    L2Action wb = l2_writeback_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1);
    wb.l3_slot = 0;
    const L2Program l2p{l2_move_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1, true), wb, l2_release(0, tile(MatrixID::A, 0))};
    L3BmRig::Options o;
    o.l3_slots = 1;
    L3BmRig r(l3p, l2p, o);
    CHECK(r.dma.holds(0));
    CHECK(!r.bm.holds_l3(0));
    r.channel.preload(kDram, a);
    bool mover_held = false;
    r.sim.add_invariant("the mover's turn", [&] { mover_held = mover_held || r.bm.holds_l3(0); });
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(mover_held);   // between the first Release and the Write
    CHECK(r.channel.peek(kDram + 0x1000, 256) == transposed16(a));
    CHECK(r.dma.holds(0));   // the last Release: no producer follows, so the engine's
}

TEST_CASE("A BlockMover read that is not the program's next Read is a fault", "[cycle][l3][blockmover][join]") {
    // The L3 program reads slot 1; the mover's Move reads slot 0.
    const L3Program l3p{l3_load(0, kDram, 256), l3_load(1, kDram + 0x1000, 256), l3_read(1, 256), l3_release(0),
                        l3_release(1)};
    const L2Program l2p{l2_move_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1, false), l2_release(0, tile(MatrixID::A, 0))};
    L3BmRig r(l3p, l2p, L3BmRig::Options{});
    r.channel.preload(kDram, tile16(1));
    r.channel.preload(kDram + 0x1000, tile16(2));
    CHECK_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("reads 256 bytes of slot 1"));
}

TEST_CASE("The slot credits start with each slot's first producer", "[cycle][l3][blockmover][join]") {
    const L3Program p{l3_write(0, 64), l3_load(1, kDram, 64), l3_release(0), l3_load(0, kDram + 64, 64),
                      l3_release(1), l3_release(0)};
    CHECK(blockmover_holds_at_start(p, 0));
    CHECK(!blockmover_holds_at_start(p, 1));
    CHECK(!blockmover_holds_at_start(p, 2));   // no producer: the engine's
    // Reads and Writes have no DRAM, so they need no address.
    CHECK_NOTHROW(validate_l3_program(p, 3, 1024, "test"));
    CHECK_THROWS_WITH(validate_l3_program(L3Program{l3_read(0, 100)}, 1, 1024, "test"),
                      ContainsSubstring("not a positive multiple of 64"));
}

TEST_CASE("A strided Move's per-row reads make one Read; it completes when all its bytes are served",
          "[cycle][l3][blockmover][join]") {
    // #600's review: a strided Move asks L3 for each row at l3_address + k * pitch. Two 64-byte rows
    // of slot 0 (the program's Read of 128 bytes), written back strided into slot 1.
    std::vector<std::uint8_t> t(128);
    for (std::uint32_t i = 0; i < 128; ++i) t[i] = static_cast<std::uint8_t>(i * 5 + 3);
    const L3Program l3p{l3_load(0, kDram, 128), l3_read(0, 128), l3_write(1, 128), l3_store(1, kDram + 0x1000, 128),
                        l3_release(0), l3_release(1)};
    L2Action wb = l2_writeback(0, tile(MatrixID::A, 0), at(1), 64, 2);
    wb.l3_slot = 1;
    const L2Program l2p{l2_move(0, tile(MatrixID::A, 0), at(0), 64, 2), wb, l2_release(0, tile(MatrixID::A, 0))};
    L3BmRig r(l3p, l2p, L3BmRig::Options{});
    r.channel.preload(kDram, t);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(r.channel.peek(kDram + 0x1000, 128) == t);
}

TEST_CASE("A read past its Read's bytes, or past the program's Reads, is a fault", "[cycle][l3][blockmover][join]") {
    SECTION("one request larger than the Read") {
        // The program reads 64 bytes of slot 0; the dense Move asks for its whole 256-byte tile.
        const L3Program l3p{l3_load(0, kDram, 256), l3_read(0, 64), l3_release(0)};
        const L2Program l2p{l2_move_dense(0, tile(MatrixID::A, 0), at(0), 16, 16, 1, false), l2_release(0, tile(MatrixID::A, 0))};
        L3BmRig r(l3p, l2p, L3BmRig::Options{});
        r.channel.preload(kDram, tile16(1));
        CHECK_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("past the 64 bytes action 1 reads"));
    }
    SECTION("more rows than the Read's bytes") {
        // The program reads 64 bytes of slot 0; the strided Move asks for two 64-byte rows: the first
        // completes the Read, the second has none left.
        const L3Program l3p{l3_load(0, kDram, 128), l3_read(0, 64), l3_release(0)};
        const L2Program l2p{l2_move(0, tile(MatrixID::A, 0), at(0), 64, 2), l2_release(0, tile(MatrixID::A, 0))};
        L3BmRig r(l3p, l2p, L3BmRig::Options{});
        r.channel.preload(kDram, std::vector<std::uint8_t>(128, 1));
        CHECK_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("past the program's Reads"));
    }
}

TEST_CASE("A read whose address wraps the address space is outside its slot", "[cycle][l3][blockmover][join]") {
    // #600's review: address - base + bytes wrapped below the slot's size near 2^64.
    const L3Program l3p{l3_load(0, kDram, 64), l3_read(0, 64), l3_release(0)};
    const L2Program l2p{l2_move_dense(0, tile(MatrixID::A, 0), ~std::uint64_t{0} - 63, 1, 16, 1, false),
                        l2_release(0, tile(MatrixID::A, 0))};
    L3BmRig r(l3p, l2p, L3BmRig::Options{});
    r.channel.preload(kDram, std::vector<std::uint8_t>(64, 1));
    CHECK_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("outside its slot"));
}

TEST_CASE("A strided Writeback into an L3 slot is contiguous rows, or refused", "[cycle][l3][blockmover][join]") {
    // #600's review: an L3 Write is a whole tile, its beats in order, so rows `l3_pitch` apart with a
    // pitch past the row would leave gaps the Write never fills. Refused before anything runs.
    L2Action wb = l2_writeback(0, tile(MatrixID::A, 0), at(1), 128, 2);
    wb.l3_slot = 1;
    const L2Program l2p{l2_move(0, tile(MatrixID::A, 0), at(0), 64, 2), wb, l2_release(0, tile(MatrixID::A, 0))};
    const L3Program l3p{l3_load(0, kDram, 128), l3_read(0, 128), l3_write(1, 128), l3_release(0), l3_release(1)};
    CHECK_THROWS_WITH(L3BmRig(l3p, l2p, L3BmRig::Options{}), ContainsSubstring("rows 128 bytes apart"));
}
