// ============================================================================
// tests/timing/test_l2_banks.cpp
// L-CA's banked L2 (#401 slice 1; docs/plans/banked-l2-at-lca.md), mirroring #383 and #388:
// rows walk the banks by their low bits, each bank serves 2 reads and 1 write per clock (esg#24),
// a row that finds its bank's ports taken waits a clock and is counted, a bank count must be a
// power of two (esg#14), and a merge is a read-modify-write.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/timing/l2_banks.hpp>

using sw::kpu::timing::L2Banks;
using A = L2Banks::Access;
using C = L2Banks::Client;
using Catch::Matchers::ContainsSubstring;

namespace {
L2Banks banks(std::uint32_t n, std::uint32_t row_bytes = 64, sw::kpu::timing::Cycle latency = 4) {
    L2Banks::Geometry g;
    g.banks = n;
    g.row_bytes = row_bytes;
    g.read_latency = latency;
    return L2Banks(g, 1);
}
}  // namespace

TEST_CASE("L2 banks: a row's bank is its low bits (#383)", "[timing][l2][banks]") {
    const auto b = banks(4);
    for (std::uint64_t r = 0; r < 32; ++r) CHECK(b.bank_of(r) == r % 4);
    CHECK(banks(1).bank_of(7) == 0);                    // one bank takes no address bits
    CHECK(b.rows_of(256) == 4);                         // a 256-byte tile, 64-byte rows
    CHECK(b.rows_of(1) == 1);
    CHECK(b.first_row(3, 4) == 12);                     // slot 3 of 4-row tiles
}

TEST_CASE("L2 banks: the count must be a power of two (esg#14)", "[timing][l2][banks]") {
    for (std::uint32_t n : {3u, 6u, 12u, 0u}) {
        CAPTURE(n);
        L2Banks::Geometry g;
        g.banks = n;
        CHECK_THROWS_WITH(L2Banks(g, 1), ContainsSubstring(n ? "power of two" : "power of two"));
    }
    for (std::uint32_t n : {1u, 2u, 4u, 8u, 64u}) CHECK_NOTHROW(banks(n));
}

TEST_CASE("L2 banks: an uncontended transfer takes its rows at its rate", "[timing][l2][banks]") {
    auto b = banks(4);
    // 4 rows at 2 clocks a row: 8 clocks; a read adds its latency once (pipelined).
    CHECK(b.book(0, 0, 256, A::Write, C::BlockMover, 100, 2) == 8);
    CHECK(b.book(0, 1, 256, A::Read, C::Streamer, 200, 1) == 4 + 4);
    CHECK(b.stats().stalls_total() == 0);
    CHECK(b.stats().accesses == 8);
}

TEST_CASE("L2 banks: two reads and one write per bank per clock (esg#24)", "[timing][l2][banks]") {
    auto b = banks(4, 64, 0);
    // Slot 0 and slot 1 of one-row tiles are rows 0 and 1: banks 0 and 1. Slot 4 is row 4, bank 0.
    // Two reads of bank 0 in clock 10 fit; a third waits a clock.
    CHECK(b.book(0, 0, 64, A::Read, C::Streamer, 10, 1) == 1);
    CHECK(b.book(0, 4, 64, A::Read, C::Streamer, 10, 1) == 1);
    CHECK(b.book(0, 8, 64, A::Read, C::Combiner, 10, 1) == 2);
    CHECK(b.stats().stalls[static_cast<std::size_t>(C::Combiner)] == 1);
    // One write of bank 1 in clock 20; a second waits; a read beside them does not.
    CHECK(b.book(0, 1, 64, A::Write, C::BlockMover, 20, 1) == 1);
    CHECK(b.book(0, 5, 64, A::Write, C::Drain, 20, 1) == 2);
    CHECK(b.book(0, 9, 64, A::Read, C::Streamer, 20, 1) == 1);
    CHECK(b.stats().stalls[static_cast<std::size_t>(C::Drain)] == 1);
    CHECK(b.stats().bank_stalls[0] == 1);
    CHECK(b.stats().bank_stalls[1] == 1);
    // Other banks in the same clock are free.
    CHECK(b.book(0, 2, 64, A::Write, C::BlockMover, 20, 1) == 1);
}

TEST_CASE("L2 banks: a merge reads and writes each row in one clock", "[timing][l2][banks]") {
    auto b = banks(2, 64, 3);
    // A read-modify-write needs a read and the write port of its bank: a write already there
    // makes it wait, and it pays the read latency.
    CHECK(b.book(0, 0, 64, A::Write, C::Drain, 50, 1) == 1);
    CHECK(b.book(0, 2, 64, A::ReadModifyWrite, C::Combiner, 50, 1) == 2 + 3);
    CHECK(b.stats().stalls[static_cast<std::size_t>(C::Combiner)] == 1);
}

TEST_CASE("L2 banks: each compute tile's L2 has its own banks", "[timing][l2][banks]") {
    L2Banks::Geometry g;
    g.banks = 1;
    g.row_bytes = 64;
    g.read_latency = 0;
    L2Banks b(g, 2);
    CHECK(b.book(0, 0, 64, A::Write, C::Drain, 5, 1) == 1);
    CHECK(b.book(1, 0, 64, A::Write, C::Drain, 5, 1) == 1);     // another tile: no conflict
    CHECK(b.book(0, 1, 64, A::Write, C::Drain, 5, 1) == 2);     // the same tile's one bank: waits
}
