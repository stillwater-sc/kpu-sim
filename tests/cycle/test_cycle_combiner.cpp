// ============================================================================
// tests/cycle/test_cycle_combiner.cpp
// The north result FIFOs and the Combiner on the cycle core, step 2d (#545;
// docs/plans/cycle-based-kpu-prototype.md §5.1, §5.3 rungs 3-4): the array's results into the result
// tile's L2 slot, and from there back to L3 by the BlockMover's Writeback. So C in L3 is checked
// against the int8 operands' product, int32 exact, on the whole path from L3 to L3.
//
// Rung 3: a K chain writes only at its end. Rung 4: back-to-back commands run with no bubble, and the
// Streamer's drain rule spaces reduction ends 2R - 1 clocks apart, where a forced 2R - 2 collides.
// Every clock is measured from the run: each row leaves the FIFOs at s + K + C + 2i, `result` is its
// last, and `completed` comes before it (early release).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "compute_rig.hpp"

#include <sw/kpu/cycle/compute/combiner.hpp>
#include <sw/kpu/cycle/compute/l2_program.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::isa::MatrixID;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
using Matrix = std::vector<std::vector<std::int32_t>>;

std::int32_t value(std::uint32_t seed, std::uint32_t r, std::uint32_t c) {
    return static_cast<std::int8_t>(static_cast<std::uint8_t>(seed * 59 + r * 37 + c * 11 + 5));
}
Matrix operand(std::uint32_t seed, std::uint32_t rows, std::uint32_t cols) {
    Matrix m(rows, std::vector<std::int32_t>(cols));
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < cols; ++c) m[r][c] = value(seed, r, c);
    return m;
}
Matrix product(const Matrix& a, const Matrix& b) {
    Matrix c(a.size(), std::vector<std::int32_t>(b[0].size(), 0));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b[0].size(); ++j)
            for (std::size_t k = 0; k < b.size(); ++k) c[i][j] += a[i][k] * b[k][j];
    return c;
}
Matrix transpose(const Matrix& m) {
    Matrix t(m[0].size(), std::vector<std::int32_t>(m.size()));
    for (std::size_t r = 0; r < m.size(); ++r)
        for (std::size_t c = 0; c < m[0].size(); ++c) t[c][r] = m[r][c];
    return t;
}

TileID tile(MatrixID m, std::uint32_t n) { return TileID{m, n, 0, 0}; }

// An operand's lanes (A's rows, or B's columns: transpose(B)) into L3 as the L2 lays them out: row
// n * lanes + m is lane m's K-block n, `k` int8 elements in a 16-byte row.
void load_lanes(L3Standin& l3, std::uint64_t at, const Matrix& lanes, std::uint32_t k) {
    const std::uint32_t blocks = static_cast<std::uint32_t>(lanes[0].size() / k);
    for (std::uint32_t n = 0; n < blocks; ++n)
        for (std::uint32_t m = 0; m < lanes.size(); ++m) {
            std::vector<std::uint8_t> row(16, 0);
            for (std::uint32_t e = 0; e < k; ++e) row[e] = static_cast<std::uint8_t>(lanes[m][n * k + e]);
            l3.load(at + std::uint64_t(n * lanes.size() + m) * 16, row);
        }
}

// The result tile written back to L3: row i is w int32, little-endian, in a 16-byte row.
Matrix read_result(const L3Standin& l3, std::uint64_t at, std::uint32_t h, std::uint32_t w) {
    Matrix c(h, std::vector<std::int32_t>(w));
    for (std::uint32_t i = 0; i < h; ++i) {
        const auto row = l3.peek(at + std::uint64_t(i) * 16, 16);
        for (std::uint32_t j = 0; j < w; ++j) {
            std::uint32_t u = 0;
            for (std::uint32_t b = 0; b < 4; ++b) u |= std::uint32_t(row[j * 4 + b]) << (8 * b);
            c[i][j] = static_cast<std::int32_t>(u);
        }
    }
    return c;
}

// s, measured: the clock row 0's first element of `command` was presented to the array.
Cycle measured_s(const PEArray& a, std::uint32_t command) {
    for (const auto& m : a.macs())
        if (m.row == 0 && m.col == 0 && m.k == 0 && m.command == command) return m.at - 1;
    FAIL("PE(0,0) never multiplied command " << command << "'s first element");
    return 0;
}

constexpr std::uint64_t kA = 0x10000, kB = 0x20000, kC = 0x40000;

// One result tile C = A B, h x w, K = 16 * blocks, a chain of `blocks` commands: A in slot 0, B in
// slot 1, C in slot 2, written back to kC.
struct One {
    std::uint32_t h = 4, w = 4, blocks = 1;
    StreamerCommand::Epilogue epilogue{};
};
L2Program one_program(const One& o) {
    const TileID ta = tile(MatrixID::A, 0), tb = tile(MatrixID::B, 0), tc = tile(MatrixID::C, 0);
    L2Program p{l2_move(0, ta, kA, 16, o.h * o.blocks), l2_move(1, tb, kB, 16, o.w * o.blocks)};
    for (std::uint32_t n = 0; n < o.blocks; ++n) {
        p.push_back(l2_feed(0, ta, o.h));
        p.push_back(l2_feed(1, tb, o.w));
    }
    p.push_back(l2_result(2, tc, o.h));
    p.push_back(l2_release(0, ta));
    p.push_back(l2_release(1, tb));
    p.push_back(l2_writeback(2, tc, kC, 16, o.h));
    p.push_back(l2_release(2, tc));
    return p;
}
StreamerProgram one_commands(const One& o) {
    const TileID ta = tile(MatrixID::A, 0), tb = tile(MatrixID::B, 0), tc = tile(MatrixID::C, 0);
    StreamerProgram p;
    for (std::uint32_t n = 0; n < o.blocks; ++n) {
        StreamerCommand c{{0, ta, n * o.h, o.h}, {1, tb, n * o.w, o.w}, 16, n == 0, n + 1 == o.blocks, std::nullopt};
        if (n + 1 == o.blocks) c.result = StreamerCommand::Result{2, tc, o.epilogue};
        p.push_back(c);
    }
    return p;
}

}  // namespace

TEST_CASE("Rung 3: a K chain writes its result tile only at its end, into L2 and back to L3 exact",
          "[cycle][combiner]") {
    for (std::uint32_t blocks : {1u, 3u}) {
        CAPTURE(blocks);
        const One o{4, 4, blocks, {}};
        const Matrix a = operand(1, 4, 16 * blocks), b = operand(2, 16 * blocks, 4);
        ComputeRig r(one_program(o), one_commands(o), ComputeRig::Options{});
        load_lanes(r.l3, kA, a, 16);
        load_lanes(r.l3, kB, transpose(b), 16);
        REQUIRE(r.sim.run_until_idle(kLimit));
        CHECK(read_result(r.l3, kC, 4, 4) == product(a, b));

        // The tile's rows, written once each, after the chain's last command fired.
        REQUIRE(r.combiner.records().size() == 1);
        const auto& rec = r.combiner.records()[0];
        REQUIRE(rec.rows.size() == 4);
        const auto& cmds = r.st.records();
        for (Cycle at : rec.rows) CHECK(at > cmds.back().accepted);
        for (const auto& l : r.array.launches()) CHECK(l.command == blocks - 1);
        // Row i leaves the FIFOs at s + K + C + 2i (s the chain's, K the whole reduction's), and
        // `result` is the last row's clock.
        const Cycle s = measured_s(r.array, 0);
        for (std::uint32_t i = 0; i < 4; ++i) {
            CAPTURE(i);
            CHECK(rec.rows[i] == s + 16 * blocks + 4 + 2 * i);
        }
        REQUIRE(rec.result);
        CHECK(*rec.result == rec.rows.back());
        // Early release: the operands' slots return at `completed`, before `result`.
        CHECK(*cmds.back().completed < *rec.result);
        // Every slot is free at the end, and its credit back with its next producer (none: the mover).
        for (std::uint32_t sl = 0; sl < 3; ++sl) {
            CAPTURE(sl);
            CHECK(r.l2.state(sl) == L2::State::Free);
            CHECK(r.bm.holds(sl));
            CHECK_FALSE(r.st.holds_credit(sl));
        }
    }
}

TEST_CASE("Rung 4: back-to-back commands run with no bubble, two result tiles", "[cycle][combiner]") {
    // C0 = A B0 and C1 = A B1, A read by both: the second command fires I = K clocks after the first.
    const TileID ta = tile(MatrixID::A, 0), b0 = tile(MatrixID::B, 0), b1 = tile(MatrixID::B, 1),
                 c0 = tile(MatrixID::C, 0), c1 = tile(MatrixID::C, 1);
    const L2Program p{l2_move(0, ta, kA, 16, 4),    l2_move(1, b0, kB, 16, 4),         l2_move(2, b1, kB + 0x1000, 16, 4),
                      l2_feed(0, ta, 4),            l2_feed(1, b0, 4),                 l2_feed(0, ta, 4),
                      l2_feed(2, b1, 4),            l2_result(3, c0, 4),               l2_result(4, c1, 4),
                      l2_release(0, ta),            l2_release(1, b0),                 l2_release(2, b1),
                      l2_writeback(3, c0, kC, 16, 4), l2_writeback(4, c1, kC + 0x1000, 16, 4), l2_release(3, c0),
                      l2_release(4, c1)};
    const StreamerProgram cmds{{{0, ta, 0, 4}, {1, b0, 0, 4}, 16, true, true, StreamerCommand::Result{3, c0, {}}},
                               {{0, ta, 0, 4}, {2, b1, 0, 4}, 16, true, true, StreamerCommand::Result{4, c1, {}}}};
    const Matrix a = operand(3, 4, 16), mb0 = operand(4, 16, 4), mb1 = operand(5, 16, 4);
    ComputeRig r(p, cmds, ComputeRig::Options{});
    load_lanes(r.l3, kA, a, 16);
    load_lanes(r.l3, kB, transpose(mb0), 16);
    load_lanes(r.l3, kB + 0x1000, transpose(mb1), 16);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(read_result(r.l3, kC, 4, 4) == product(a, mb0));
    CHECK(read_result(r.l3, kC + 0x1000, 4, 4) == product(a, mb1));
    REQUIRE(r.st.records().size() == 2);
    CHECK(r.st.records()[1].accepted == r.st.records()[0].accepted + 16);   // no bubble
    // PE(0,0) multiplies all 32 elements on consecutive clocks.
    std::vector<Cycle> at;
    for (const auto& m : r.array.macs())
        if (m.row == 0 && m.col == 0) at.push_back(m.at);
    REQUIRE(at.size() == 32);
    for (std::size_t i = 1; i < at.size(); ++i) CHECK(at[i] == at[i - 1] + 1);
    // The second tile's rows follow the first's by the commands' spacing.
    for (std::uint32_t i = 0; i < 4; ++i)
        CHECK(r.combiner.records()[1].rows[i] == r.combiner.records()[0].rows[i] + 16);
}

TEST_CASE("Rung 4: the drain rule spaces reduction ends 2R - 1 apart; a forced 2R - 2 collides",
          "[cycle][combiner]") {
    // One-element reductions: their interval is max(K, h, w) = 4, but their results must leave the
    // array 2R - 1 = 7 clocks apart, so the Streamer holds the second command for the drain.
    const TileID ta = tile(MatrixID::A, 0), b0 = tile(MatrixID::B, 0), c0 = tile(MatrixID::C, 0),
                 c1 = tile(MatrixID::C, 1);
    const L2Program p{l2_move(0, ta, kA, 16, 4), l2_move(1, b0, kB, 16, 4), l2_feed(0, ta, 4), l2_feed(1, b0, 4),
                      l2_feed(0, ta, 4), l2_feed(1, b0, 4), l2_result(3, c0, 4), l2_result(4, c1, 4),
                      l2_release(0, ta), l2_release(1, b0), l2_writeback(3, c0, kC, 16, 4),
                      l2_writeback(4, c1, kC + 0x1000, 16, 4), l2_release(3, c0), l2_release(4, c1)};
    const StreamerProgram cmds{{{0, ta, 0, 4}, {1, b0, 0, 4}, 1, true, true, StreamerCommand::Result{3, c0, {}}},
                               {{0, ta, 0, 4}, {1, b0, 0, 4}, 1, true, true, StreamerCommand::Result{4, c1, {}}}};
    const Matrix a = operand(6, 4, 1), b = operand(7, 1, 4);
    const auto rig = [&](std::int32_t adjust) {
        ComputeRig::Options o;
        o.drain_adjust = adjust;
        auto r = std::make_unique<ComputeRig>(p, cmds, o);
        load_lanes(r->l3, kA, a, 1);
        load_lanes(r->l3, kB, transpose(b), 1);
        return r;
    };
    {
        auto r = rig(0);
        REQUIRE(r->sim.run_until_idle(kLimit));
        CHECK(r->st.records()[1].accepted - r->st.records()[0].accepted == 2 * 4 - 1);
        CHECK(read_result(r->l3, kC, 4, 4) == product(a, b));
        CHECK(read_result(r->l3, kC + 0x1000, 4, 4) == product(a, b));
    }
    auto r = rig(-1);
    REQUIRE_THROWS_WITH(r->sim.run_until_idle(kLimit), ContainsSubstring("a north collision at PE(0,0)"));
}

TEST_CASE("The Combiner applies the tile's epilogue as it writes: bias, then ReLU", "[cycle][combiner]") {
    const One o{4, 4, 1, StreamerCommand::Epilogue{{-1000, 0, 500, 7}, true}};
    const Matrix a = operand(8, 4, 16), b = operand(9, 16, 4);
    ComputeRig r(one_program(o), one_commands(o), ComputeRig::Options{});
    load_lanes(r.l3, kA, a, 16);
    load_lanes(r.l3, kB, transpose(b), 16);
    REQUIRE(r.sim.run_until_idle(kLimit));
    Matrix want = product(a, b);
    bool clipped = false, kept = false;
    for (auto& row : want)
        for (std::size_t j = 0; j < 4; ++j) {
            row[j] = std::max(0, row[j] + o.epilogue.bias[j]);
            (row[j] == 0 ? clipped : kept) = true;
        }
    CHECK(clipped);   // the operands give both signs
    CHECK(kept);
    CHECK(read_result(r.l3, kC, 4, 4) == want);
}

TEST_CASE("A tile on the array's corner: the padded columns are dropped, the tile's rows are whole",
          "[cycle][combiner]") {
    const One o{2, 3, 1, {}};
    const Matrix a = operand(10, 2, 16), b = operand(11, 16, 3);
    ComputeRig r(one_program(o), one_commands(o), ComputeRig::Options{});
    load_lanes(r.l3, kA, a, 16);
    load_lanes(r.l3, kB, transpose(b), 16);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(read_result(r.l3, kC, 2, 3) == product(a, b));
    CHECK(r.combiner.dropped() == 2);   // column 3 of rows 0 and 1
    CHECK(r.combiner.records()[0].rows.size() == 2);
    CHECK(r.l3.peek(kC + 12, 4) == std::vector<std::uint8_t>(4, 0));   // the row past the tile's columns
}

TEST_CASE("A result slot used twice: the second tile's command waits for the slot's credit", "[cycle][combiner]") {
    // C0 then C1 in slot 2: the second command fires only after C0 is written back and its slot
    // released, its credit routed to the Streamer (the slot's next producer is a Result).
    const TileID ta = tile(MatrixID::A, 0), tb = tile(MatrixID::B, 0), c0 = tile(MatrixID::C, 0),
                 c1 = tile(MatrixID::C, 1);
    const L2Program p{l2_move(0, ta, kA, 16, 4),      l2_move(1, tb, kB, 16, 4),   l2_feed(0, ta, 4),
                      l2_feed(1, tb, 4),              l2_result(2, c0, 4),         l2_writeback(2, c0, kC, 16, 4),
                      l2_release(2, c0),              l2_feed(0, ta, 4),           l2_feed(1, tb, 4),
                      l2_result(2, c1, 4),            l2_release(0, ta),           l2_release(1, tb),
                      l2_writeback(2, c1, kC + 0x1000, 16, 4), l2_release(2, c1)};
    const StreamerCommand::Epilogue relu{{}, true};
    const StreamerProgram cmds{{{0, ta, 0, 4}, {1, tb, 0, 4}, 16, true, true, StreamerCommand::Result{2, c0, {}}},
                               {{0, ta, 0, 4}, {1, tb, 0, 4}, 16, true, true, StreamerCommand::Result{2, c1, relu}}};
    const Matrix a = operand(12, 4, 16), b = operand(13, 16, 4);
    ComputeRig r(p, cmds, ComputeRig::Options{});
    load_lanes(r.l3, kA, a, 16);
    load_lanes(r.l3, kB, transpose(b), 16);
    REQUIRE(r.sim.run_until_idle(kLimit));
    Matrix c = product(a, b), relu_c = c;
    for (auto& row : relu_c)
        for (auto& v : row) v = std::max(0, v);
    CHECK(read_result(r.l3, kC, 4, 4) == c);
    CHECK(read_result(r.l3, kC + 0x1000, 4, 4) == relu_c);
    CHECK(r.st.records()[1].accepted > *r.combiner.records()[0].result);   // after C0 was written
    CHECK(r.st.counts().stalls[static_cast<std::size_t>(Stall::NoCredit)] > 0);
    CHECK(r.l2.acquires() == 4);
    CHECK(r.bm.holds(2));   // slot 2's last Release: no producer follows, so the mover's
}

TEST_CASE("A result with no result tile to write is an L1Fault", "[cycle][combiner]") {
    // The command ends its reduction, so the array launches its results, but it names no result tile.
    One o{4, 4, 1, {}};
    StreamerProgram cmds = one_commands(o);
    cmds.back().result.reset();
    L2Program p = one_program(o);
    p.erase(std::remove_if(p.begin(), p.end(), [](const L2Action& x) { return x.slot == 2; }), p.end());
    ComputeRig r(p, cmds, ComputeRig::Options{});
    load_lanes(r.l3, kA, operand(14, 4, 16), 16);
    load_lanes(r.l3, kB, transpose(operand(15, 16, 4)), 16);
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("which produces no result tile"));
}

TEST_CASE("A north FIFO past its depth is an L1Fault: results never wait", "[cycle][combiner]") {
    // Column 0's results come, column 1's never: the tile's rows cannot leave, and column 0's FIFO,
    // two deep, overflows at its third result.
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("kpu", 1.0));
    struct Source : Component {
        using Component::Component;
        Link<ArrayResult>* out = nullptr;
        std::uint32_t sent = 0;
        void evaluate(Cycle now) override {
            if (sent < 3 && out->can_push(now)) {
                ++sent;
                out->push(now, ArrayResult{sent, sent - 1, 0, 0, 1, false});
            }
        }
        [[nodiscard]] bool idle() const override { return sent == 3; }
    };
    const TileID ta = tile(MatrixID::A, 0), tb = tile(MatrixID::B, 0), tc = tile(MatrixID::C, 0);
    const StreamerProgram cmds{{{0, ta, 0, 4}, {1, tb, 0, 2}, 16, true, true, StreamerCommand::Result{2, tc, {}}}};
    auto& src = sim.add<Source>("array", d);
    auto& cmb = sim.add<Combiner>("combiner", d, Combiner::Config{2, 2, 16}, cmds);
    auto& l0 = sim.link<ArrayResult>("north 0", src, cmb, 8);
    auto& l1 = sim.link<ArrayResult>("north 1", src, cmb, 8);
    auto& w = sim.link<L2RowWrite>("writes", cmb, src, 4);
    src.out = &l0;
    cmb.connect({&l0, &l1}, w);
    REQUIRE_THROWS_WITH(sim.run_until_idle(kLimit), ContainsSubstring("column 0's north FIFO overflows its 2 results"));
}

TEST_CASE("The result tiles are checked before anything runs", "[cycle][combiner][streamer]") {
    Domain d = Domain::from_ghz("kpu", 1.0);
    const TileID ta = tile(MatrixID::A, 0), tb = tile(MatrixID::B, 0), tc = tile(MatrixID::C, 0);
    const Streamer::Config cfg{4, 4, 3, 16, 16, 1, 32, 0, 0};
    StreamerCommand c{{0, ta, 0, 4}, {1, tb, 0, 4}, 16, true, true, StreamerCommand::Result{2, tc, {}}};
    CHECK_NOTHROW(Streamer("st", d, cfg, {c}));
    StreamerCommand open = c;
    open.last = false;
    StreamerCommand close = c;
    close.first = false;
    close.result.reset();
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {open, close}), ContainsSubstring("does not end its chain"));
    StreamerCommand same = c;
    same.result->slot = 1;
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {same}), ContainsSubstring("in an operand's slot"));
    StreamerCommand bias = c;
    bias.result->epilogue.bias = {1, 2};
    CHECK_THROWS_WITH(Streamer("st", d, cfg, {bias}), ContainsSubstring("a bias of 2 for 4 columns"));
    CHECK_THROWS_WITH(Combiner("cmb", d, Combiner::Config{4, 4, 8}, {c}), ContainsSubstring("8-byte row"));
    // The L2 program's Results agree with the commands' result tiles.
    const One o{4, 4, 1, {}};
    CHECK_NOTHROW(check_feeds(one_commands(o), one_program(o)));
    L2Program fewer = one_program(o);
    fewer.erase(std::remove_if(fewer.begin(), fewer.end(),
                               [](const L2Action& x) { return x.kind == L2Action::Kind::Result; }),
                fewer.end());
    CHECK_THROWS(check_feeds(one_commands(o), fewer));
    L2Program rows = one_program(o);
    for (L2Action& x : rows)
        if (x.kind == L2Action::Kind::Result) x.rows = 2;
    CHECK_THROWS_WITH(check_feeds(one_commands(o), rows), ContainsSubstring("the L2 program's Result is C[0,0,0], 2 rows"));
    CHECK_THROWS_WITH(validate_l2_program({l2_result(0, tc, 4), l2_result(0, tc, 4)}, 2, 16, 16, "l2"),
                      ContainsSubstring("no Release before this Move"));
}
