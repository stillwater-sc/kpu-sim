// ============================================================================
// tests/cycle/test_cycle_pe_array.cpp
// The PE array on the cycle core, step 2c (#544; docs/plans/cycle-based-kpu-prototype.md §5.1, §5.3
// rungs 1-2): R x C registered MAC PEs, output-stationary, as the RTL builds them.
//
// Rung 1 is one PE, rung 2 a 4 x 4 array with one command. The wavefront is MEASURED from the array's
// log of what each PE did, never coded: PE(i,j) multiplies element k at s + 1 + i + j + k, column j
// carries row i's result at s + K + j + 2i, and row i is complete at the north edge (its results
// taken by the result buffers) at s + K + C + 2i. Here s is the clock row 0's first element is
// presented at the west edge, also measured. Then the same on the whole path from L3, with int32
// values checked exactly against the product of the int8 operands; the corner of the array; the
// north collision; and operands that do not meet.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "array_rig.hpp"

#include <sw/kpu/cycle/compute/l1_lanes.hpp>
#include <sw/kpu/cycle/compute/pe_array.hpp>
#include <sw/kpu/cycle/compute/streamer.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::isa::MatrixID;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms
using Matrix = std::vector<std::vector<std::int32_t>>;

// An int8 operand with negative and positive values: element (r, c) of operand `seed`.
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

// Presents scheduled elements at the array's edges: each pushed the clock before it is presented.
class EdgeDriver : public Component {
public:
    EdgeDriver(std::string name, Domain& d) : Component(std::move(name), d) {}
    void connect(std::vector<Link<L1Element>*> west, std::vector<Link<L1Element>*> north) {
        west_ = std::move(west);
        north_ = std::move(north);
    }
    // Operand A (h x K) on rows 0..h-1 and B (K x w) on columns 0..w-1, row i's element k presented
    // at s + i + k and column j's at s + j + k: the L1 lanes' stagger.
    void stream(Cycle s, const Matrix& a, const Matrix& b, std::uint32_t command, bool first, bool last) {
        const std::uint32_t K = static_cast<std::uint32_t>(b.size());
        for (std::uint32_t i = 0; i < a.size(); ++i)
            for (std::uint32_t k = 0; k < K; ++k)
                at_[s + i + k - 1].push_back({'A', i, element(i, command, k, K, a[i][k], first, last)});
        for (std::uint32_t j = 0; j < b[0].size(); ++j)
            for (std::uint32_t k = 0; k < K; ++k)
                at_[s + j + k - 1].push_back({'B', j, element(j, command, k, K, b[k][j], first, last)});
    }
    // One element moved a clock later (to break the skew).
    void delay(char side, std::uint32_t lane) { late_ = {side, lane}; }
    void evaluate(Cycle now) override {
        const auto it = at_.find(now);
        if (it == at_.end()) return;
        for (const Push& p : it->second) {
            if (late_ && late_->first == p.side && late_->second == p.lane && p.e.k == 0 && !moved_) {
                moved_ = true;
                at_[now + 1].push_back(p);
                continue;
            }
            Link<L1Element>* l = (p.side == 'A' ? west_ : north_)[p.lane];
            l->push(now, p.e);
        }
        at_.erase(it);
    }
    [[nodiscard]] bool idle() const override { return at_.empty(); }

private:
    struct Push {
        char side;
        std::uint32_t lane;
        L1Element e;
    };
    static L1Element element(std::uint32_t lane, std::uint32_t command, std::uint32_t k, std::uint32_t K,
                             std::int32_t v, bool first, bool last) {
        return {0, lane, command, k, static_cast<std::uint32_t>(static_cast<std::uint8_t>(v)), first && k == 0,
                last && k + 1 == K};
    }
    std::map<Cycle, std::vector<Push>> at_;
    std::vector<Link<L1Element>*> west_, north_;
    std::optional<std::pair<char, std::uint32_t>> late_;
    bool moved_ = false;
};

// The driver, an R x C array and the result sink.
struct Bench {
    Simulator sim;
    Domain& kpu;
    EdgeDriver& drv;
    PEArray& array;
    ResultSink& north;
    explicit Bench(std::uint32_t R, std::uint32_t C, Tracer* tracer = nullptr)
        : kpu(traced(sim, tracer).add_domain(Domain::from_ghz("kpu", 1.0))),
          drv(sim.add<EdgeDriver>("edges", kpu)),
          array(sim.add<PEArray>("array", kpu, PEArray::Config{R, C, 1, true})),
          north(sim.add<ResultSink>("north", kpu)) {
        std::vector<Link<L1Element>*> w, n;
        std::vector<Link<ArrayResult>*> c;
        for (std::uint32_t i = 0; i < R; ++i) w.push_back(&sim.link<L1Element>("west " + std::to_string(i), drv, array, 4));
        for (std::uint32_t j = 0; j < C; ++j) {
            n.push_back(&sim.link<L1Element>("north in " + std::to_string(j), drv, array, 4));
            c.push_back(&sim.link<ArrayResult>("north " + std::to_string(j), array, north, 2));
        }
        drv.connect(w, n);
        array.connect(w, n, c);
        north.connect(c);
    }
    static Simulator& traced(Simulator& s, Tracer* t) {
        if (t) s.attach(*t);
        return s;
    }
};

// s, measured: the clock row 0's first element of `command` was presented, one before PE(0,0)
// multiplied it.
Cycle measured_s(const PEArray& a, std::uint32_t command) {
    for (const auto& m : a.macs())
        if (m.row == 0 && m.col == 0 && m.k == 0 && m.command == command) return m.at - 1;
    FAIL("PE(0,0) never multiplied command " << command << "'s first element");
    return 0;
}

// The wavefront of one K-element reduction on an h x w tile, measured: every PE of the tile
// multiplies element k at s + 1 + i + j + k; each launches once, and column j carries row i's
// result at s + K + j + 2i; the result buffers have row i whole at s + K + C + 2i, C the array's.
void check_wavefront(const PEArray& a, const ResultSink& north, std::uint32_t command, std::uint32_t h,
                     std::uint32_t w, std::uint32_t K, const Matrix& c) {
    const Cycle s = measured_s(a, command);
    const std::uint32_t C = a.config().cols;
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, Cycle> mac;
    for (const auto& m : a.macs())
        if (m.command == command) mac[{m.row, m.col, m.k}] = m.at;
    for (std::uint32_t i = 0; i < h; ++i)
        for (std::uint32_t j = 0; j < w; ++j)
            for (std::uint32_t k = 0; k < K; ++k) {
                CAPTURE(i, j, k);
                REQUIRE(mac.count({i, j, k}));
                CHECK(mac[{i, j, k}] == s + 1 + i + j + k);
            }
    std::map<std::pair<std::uint32_t, std::uint32_t>, Cycle> left;
    for (const auto& l : a.launches())
        if (l.command == command && !l.padded) {
            CAPTURE(l.row, l.col);
            CHECK(l.left == s + K + l.col + 2 * l.row);
            CHECK(l.value == c[l.row][l.col]);
            left[{l.row, l.col}] = l.left;
        }
    CHECK(left.size() == std::size_t(h) * w);
    // Row i whole at the result buffers: its last column's result taken, a clock after it left.
    std::map<std::uint32_t, Cycle> whole;
    for (const auto& g : north.got())
        if (g.r.command == command) whole[g.r.row] = std::max(whole[g.r.row], g.at);
    for (std::uint32_t i = 0; i < h; ++i) {
        CAPTURE(i);
        CHECK(whole[i] == s + K + C + 2 * i);
    }
}

}  // namespace

TEST_CASE("Rung 1, one PE: a MAC a clock, `first` overwrites, `last` launches", "[cycle][array]") {
    const Matrix a = operand(1, 1, 8), b = operand(2, 8, 1), b2 = operand(3, 8, 1);
    SECTION("one reduction") {
        Bench r(1, 1);
        r.drv.stream(1, a, b, 0, true, true);
        REQUIRE(r.sim.run_until_idle(kLimit));
        REQUIRE(r.array.macs().size() == 8);
        for (std::uint32_t k = 0; k < 8; ++k) CHECK(r.array.macs()[k].at == 2 + k);   // s + 1 + k, s = 1
        REQUIRE(r.array.launches().size() == 1);
        CHECK(r.array.launches()[0].at == 1 + 8);    // the last product's clock, s + K
        CHECK(r.array.launches()[0].left == 1 + 8);  // row 0 drives the north edge at once
        CHECK(r.array.launches()[0].value == product(a, b)[0][0]);
    }
    SECTION("two reductions back to back: the second needs no clear") {
        Bench r(1, 1);
        r.drv.stream(1, a, b, 0, true, true);
        r.drv.stream(9, a, b2, 1, true, true);
        REQUIRE(r.sim.run_until_idle(kLimit));
        REQUIRE(r.array.launches().size() == 2);
        CHECK(r.array.launches()[0].value == product(a, b)[0][0]);
        CHECK(r.array.launches()[1].value == product(a, b2)[0][0]);
        CHECK(r.array.launches()[1].at == r.array.launches()[0].at + 8);
    }
    SECTION("a chain of two commands: only the last launches, the sum of both") {
        Bench r(1, 1);
        r.drv.stream(1, a, b, 0, true, false);
        r.drv.stream(9, a, b2, 1, false, true);
        REQUIRE(r.sim.run_until_idle(kLimit));
        REQUIRE(r.array.launches().size() == 1);
        CHECK(r.array.launches()[0].value == product(a, b)[0][0] + product(a, b2)[0][0]);
        CHECK(r.array.launches()[0].command == 1);
    }
}

TEST_CASE("Rung 2, a 4 x 4 array and one command: the wavefront, measured", "[cycle][array]") {
    const Matrix a = operand(4, 4, 16), b = operand(5, 16, 4);
    Bench r(4, 4);
    r.drv.stream(3, a, b, 0, true, true);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(measured_s(r.array, 0) == 3);
    CHECK(r.array.macs().size() == 4 * 4 * 16);
    check_wavefront(r.array, r.north, 0, 4, 4, 16, product(a, b));
    for (std::uint32_t i = 0; i < 4; ++i)
        for (std::uint32_t j = 0; j < 4; ++j) CHECK(r.array.macs_at(i, j) == 16);
}

TEST_CASE("A tile on the array's corner: the columns past it launch padded results, the rows past it idle",
          "[cycle][array]") {
    const Matrix a = operand(6, 2, 8), b = operand(7, 8, 3);
    Bench r(4, 4);
    r.drv.stream(1, a, b, 0, true, true);
    REQUIRE(r.sim.run_until_idle(kLimit));
    check_wavefront(r.array, r.north, 0, 2, 3, 8, product(a, b));
    std::uint32_t padded = 0;
    for (const auto& l : r.array.launches()) {
        CAPTURE(l.row, l.col);
        CHECK(l.row < 2);                        // rows past the tile have no A
        CHECK(l.padded == (l.col == 3));         // A flows on past the tile's last column
        if (l.padded) {
            CHECK(l.value == 0);
            ++padded;
        }
    }
    CHECK(padded == 2);
    CHECK(r.array.macs_at(2, 0) == 0);
}

TEST_CASE("The north drain: results 2R - 1 clocks apart never collide; 2R - 2 apart is an L1Fault",
          "[cycle][array]") {
    // One-element reductions on a 4 x 4 array, the second started g clocks after the first: its row 0
    // launches as the first's row 3 result passes it when g = 2 * 3.
    const Matrix a = operand(8, 4, 1), b = operand(9, 1, 4);
    {
        Bench r(4, 4);
        r.drv.stream(1, a, b, 0, true, true);
        r.drv.stream(1 + 2 * 4 - 1, a, b, 1, true, true);
        REQUIRE(r.sim.run_until_idle(kLimit));
        check_wavefront(r.array, r.north, 0, 4, 4, 1, product(a, b));
        check_wavefront(r.array, r.north, 1, 4, 4, 1, product(a, b));
    }
    Bench r(4, 4);
    r.drv.stream(1, a, b, 0, true, true);
    r.drv.stream(1 + 2 * 4 - 2, a, b, 1, true, true);
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("a north collision at PE(0,0)") &&
                                            ContainsSubstring("PE(3,0)'s for command 0 passed it"));
}

TEST_CASE("Operands that do not meet are a defect of the skew, raised by name", "[cycle][array]") {
    const Matrix a = operand(10, 4, 8), b = operand(11, 8, 4);
    Bench r(4, 4);
    r.drv.stream(1, a, b, 0, true, true);
    r.drv.delay('B', 2);   // column 2's first element a clock late
    REQUIRE_THROWS_WITH(r.sim.run_until_idle(kLimit), ContainsSubstring("the operands do not meet at PE(0,2)"));
}

TEST_CASE("The whole path, L3 to the array: one command's int32 results exact, and the wavefront",
          "[cycle][array][streamer]") {
    // A (4 x 16) and B (16 x 4, one L2 row per column) moved into L2, read by the Streamer, streamed
    // through the L1 lanes into the array.
    constexpr std::uint64_t kA = 0x10000, kB = 0x20000;
    const TileID ta{MatrixID::A, 0, 0, 0}, tb{MatrixID::B, 0, 0, 0};
    const auto run = [&](std::uint32_t blocks) {
        const std::uint32_t K = 16 * blocks;
        const Matrix a = operand(12, 4, K), b = operand(13, K, 4);
        L2Program p{l2_move(0, ta, kA, 16, 4 * blocks), l2_move(1, tb, kB, 16, 4 * blocks)};
        StreamerProgram cmds;
        for (std::uint32_t n = 0; n < blocks; ++n) {
            p.push_back(l2_feed(0, ta, 4));
            p.push_back(l2_feed(1, tb, 4));
            cmds.push_back({{0, ta, n * 4, 4}, {1, tb, n * 4, 4}, 16, n == 0, n + 1 == blocks});
        }
        p.push_back(l2_release(0, ta));
        p.push_back(l2_release(1, tb));
        ArrayRig r(p, cmds, ArrayRig::Options{});
        // L2 row n*4 + m holds lane m's block n: A's row m, B's column m.
        for (std::uint32_t n = 0; n < blocks; ++n)
            for (std::uint32_t m = 0; m < 4; ++m) {
                std::vector<std::uint8_t> ra(16), rb(16);
                for (std::uint32_t k = 0; k < 16; ++k) {
                    ra[k] = static_cast<std::uint8_t>(a[m][n * 16 + k]);
                    rb[k] = static_cast<std::uint8_t>(b[n * 16 + k][m]);
                }
                r.l3.load(kA + std::uint64_t(n * 4 + m) * 16, ra);
                r.l3.load(kB + std::uint64_t(n * 4 + m) * 16, rb);
            }
        REQUIRE(r.sim.run_until_idle(kLimit));
        const Matrix c = product(a, b);
        if (blocks == 1) check_wavefront(r.array, r.north, 0, 4, 4, 16, c);
        // One launch a PE, by the chain's last command, with the whole reduction's sum.
        REQUIRE(r.array.launches().size() == 16);
        for (const auto& l : r.array.launches()) {
            CAPTURE(l.row, l.col);
            CHECK(l.command == blocks - 1);
            CHECK(l.value == c[l.row][l.col]);
        }
        // The array's s is the Streamer's: lane 0 streams from a + 2 + D (step 2b), and its element is
        // presented a clock later.
        const Cycle a0 = r.st.records()[0].accepted;
        CHECK(measured_s(r.array, 0) == a0 + 3 + r.st.config().read_delay);
        // The chain streams with no gap: PE(0,0) multiplies every element on consecutive clocks.
        Cycle prev = 0;
        std::uint32_t n = 0;
        for (const auto& m : r.array.macs())
            if (m.row == 0 && m.col == 0) {
                if (n++ > 0) CHECK(m.at == prev + 1);
                prev = m.at;
            }
        CHECK(n == K);
    };
    SECTION("one command") { run(1); }
    SECTION("a chain of three commands") { run(3); }
}

TEST_CASE("The array is sized from the deployment's fabric", "[cycle][array]") {
    const auto device = [](const std::string& f) {
        return sw::kpu::program::platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + f)
            .device(0);
    };
    const PEArray::Config soc = PEArray::Config::from(device("kpu_soc_2x2.json"));
    CHECK(soc.rows == 4);
    CHECK(soc.cols == 4);
    const PEArray::Config s1 = PEArray::Config::from(device("kpu_s1.json"));
    CHECK(s1.rows == 128);
    CHECK(s1.cols == 64);
    CHECK_THROWS_WITH(PEArray::Config::from(device("kpu_t4.json")), ContainsSubstring("declares no fabric"));
    Domain d = Domain::from_ghz("kpu", 1.0);
    CHECK_THROWS_WITH(PEArray("a", d, PEArray::Config{0, 4, 1, true}), ContainsSubstring("must be positive"));
    CHECK_THROWS_WITH(PEArray("a", d, PEArray::Config{4, 4, 3, true}), ContainsSubstring("1, 2 or 4 bytes"));
}

TEST_CASE("The array's Perfetto lanes: a span per multiply, one lane per PE", "[cycle][array]") {
    Tracer t;
    const Matrix a = operand(14, 2, 4), b = operand(15, 4, 2);
    Bench r(2, 2, &t);
    r.array.trace_lanes(t);
    r.drv.stream(1, a, b, 0, true, true);
    REQUIRE(r.sim.run_until_idle(kLimit));
    CHECK(r.array.macs().size() == 2 * 2 * 4);
    CHECK_NOTHROW(t.finish());
}
