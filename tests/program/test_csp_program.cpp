// ============================================================================
// tests/program/test_csp_program.cpp
// The CSP program, level 1 (docs/plans/csp-program-tile-sequencing.md step 1): lowered from L0
// with Belady residency against the L3 capacity, and run behaviorally to the L0 reference's
// values bit for bit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lower.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;
using Kind = Action::Kind;

namespace {

TileProgram program(const std::string& algo, Dim size, Dim tile) {
    driver::ProgramSpec s;
    s.algo = algo;
    s.size = size;
    s.tile = tile;
    TileProgram p = driver::derive(s);
    driver::fill(p, s);
    return p;
}

std::size_t count(const CspProgram& p, Kind k) {
    return static_cast<std::size_t>(std::count_if(p.actions.begin(), p.actions.end(),
                                                  [k](const Action& a) { return a.kind == k; }));
}

// The CSP program run at L-B must reproduce the L0 reference bit for bit, every operand.
void check_values(const TileProgram& l0, const CspProgram& p) {
    TileProgram ref = l0;
    TileProgramReference().run(ref);
    BehavioralInterpreter lb;
    const auto sum = lb.run(p);
    for (const auto& name : ref.operand_order()) {
        CAPTURE(name);
        CHECK(lb.result().operand(name).values == ref.operand(name).values);
    }
    if (p.channel(Chan::L3).capacity) CHECK(sum.peak_l3 <= p.channel(Chan::L3).capacity);
}

// The fewest DRAM loads any eviction policy can achieve for an explicit (matmul) program's
// order with `cap` L3 slots: dynamic programming over (position, resident set). A Feed needs
// its tile resident (a load if it is not); a Drain needs a slot for the result's writeback,
// which is then stored and freed. Dead tiles leave for free.
std::size_t optimal_loads(const TileProgram& l0, std::size_t cap) {
    std::map<std::string, int> id;
    struct Req { int tile; bool feed; };
    std::vector<Req> seq;
    for (const auto& op : l0.ops()) {
        if (op.kind == TileOpKind::Feed)
            for (const auto& t : op.inputs) seq.push_back({id.emplace(t.to_string(), int(id.size())).first->second, true});
        if (op.kind == TileOpKind::Drain)
            for (const auto& t : op.outputs) seq.push_back({id.emplace(t.to_string(), int(id.size())).first->second, false});
    }
    REQUIRE(id.size() <= 64);
    std::vector<std::uint64_t> live_after(seq.size() + 1, 0);    // tiles used after position i
    for (std::size_t i = seq.size(); i-- > 0;) live_after[i] = live_after[i + 1] | (std::uint64_t{1} << seq[i].tile);
    std::unordered_map<std::uint64_t, std::size_t> memo;         // (i, set) -> min loads; set fits 48 bits here
    const std::size_t inf = std::numeric_limits<std::size_t>::max() / 2;
    std::function<std::size_t(std::size_t, std::uint64_t)> best = [&](std::size_t i, std::uint64_t set) -> std::size_t {
        set &= live_after[i];                                     // dead tiles go for free
        if (i == seq.size()) return 0;
        const std::uint64_t key = (static_cast<std::uint64_t>(i) << 48) | set;
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        const std::uint64_t bit = std::uint64_t{1} << seq[i].tile;
        std::size_t r = inf;
        if (seq[i].feed && (set & bit)) {
            r = best(i + 1, set);
        } else {
            const std::size_t cost = seq[i].feed ? 1 : 0;
            const auto n = static_cast<std::size_t>(std::popcount(set));
            if (n < cap) {
                r = cost + best(i + 1, seq[i].feed ? (set | bit) : set);
            } else {
                for (std::uint64_t s = set; s; s &= s - 1) {
                    const std::uint64_t v = s & (~s + 1);
                    r = std::min(r, cost + best(i + 1, seq[i].feed ? ((set & ~v) | bit) : (set & ~v)));
                }
            }
        }
        memo[key] = r;
        return r;
    };
    return best(0, 0);
}

}  // namespace

TEST_CASE("CSP lowering: matmul on S1's 128 slots loads every tile exactly once",
          "[program][csp]") {
    const TileProgram l0 = program("matmul", 256, 32);
    std::size_t feeds = l0.count(TileOpKind::Feed);
    REQUIRE(feeds == 1024);                                  // the L0 order re-feeds every use
    const CspProgram p = lower(l0, {128});
    const ReuseReport r = p.reuse();
    CHECK(r.loads == 128);                                   // 64 A + 64 B tiles, each once
    CHECK(r.distinct_loaded == 128);
    CHECK(r.reloads == 0);
    CHECK(r.moves == 1024);                                  // every feed is a Move out of L3
    CHECK(r.stores == 64);                                   // each C tile stored once
    CHECK(r.calls == 512);
    CHECK(r.peak_l3 <= 128);
    // Each A[i,k] feeds the 8 computes of its row, each B[k,j] the 8 of its column: one
    // residency each, with 8 consumers.
    for (const Residency& res : p.residencies) {
        CAPTURE(res.tile.to_string());
        if (res.loaded) CHECK(res.consumers == 8);
        else CHECK(res.consumers == 1);                      // a C writeback's only consumer: its Store
    }
}

TEST_CASE("CSP lowering: a smaller L3 reloads exactly what Belady must, the fewest possible",
          "[program][csp]") {
    const TileProgram l0 = program("matmul", 48, 16);       // 3 x 3 x 3 tiles: 18 distinct inputs
    for (std::size_t cap : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
                            std::size_t{6}, std::size_t{10}, std::size_t{18}}) {
        CAPTURE(cap);
        const CspProgram p = lower(l0, {cap});
        const std::size_t opt = optimal_loads(l0, cap);
        std::printf("csp belady matmul 48^3/16^3, L3 %2zu slots: %3zu loads (optimal %3zu) of 18 tiles\n", cap,
                    p.reuse().loads, opt);
        CHECK(p.reuse().loads == opt);
        CHECK(p.reuse().peak_l3 <= cap);
        check_values(l0, p);
    }
    CHECK(lower(l0, {18}).reuse().reloads == 0);
}

TEST_CASE("CSP behavioral: the program computes the L0 reference's values, matmul and LU",
          "[program][csp]") {
    SECTION("matmul, unbounded and tight") {
        const TileProgram l0 = program("matmul", 128, 32);
        check_values(l0, lower(l0, {0}));
        check_values(l0, lower(l0, {3}));
    }
    SECTION("LU, unbounded and tight") {
        const TileProgram l0 = program("lu", 64, 16);
        const CspProgram all = lower(l0, {0});
        check_values(l0, all);
        CHECK(all.reuse().reloads == 0);
        CHECK(all.reuse().loads == 16);                       // every A tile once
        CHECK(count(all, Kind::Store) == 16);                 // and stored once, after its last use
        const CspProgram tight = lower(l0, {3});              // a trailing update holds 3 tiles
        check_values(l0, tight);
        CHECK(tight.reuse().reloads > 0);
        std::printf("csp LU 64^2/16^2: L3 unbounded %zu loads; L3 3 slots %zu loads (%zu reloads)\n",
                    all.reuse().loads, tight.reuse().loads, tight.reuse().reloads);
    }
}

TEST_CASE("CSP program: actions belong to their process, and a tile is fed only from L3",
          "[program][csp]") {
    const TileProgram l0 = program("matmul", 64, 32);
    const CspProgram p = lower(l0, {4});
    REQUIRE(p.processes.size() == 4);
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        CAPTURE(i, to_string(a.kind));
        if (a.kind == Kind::Release) { CHECK(a.process == kNone); continue; }
        const ProcessKind k = p.processes.at(a.process).kind;
        switch (a.kind) {
            case Kind::Load: case Kind::Store:         CHECK(k == ProcessKind::Dma); break;
            case Kind::Move: case Kind::Writeback:     CHECK(k == ProcessKind::BlockMover); break;
            case Kind::Feed: case Kind::Drain:         CHECK(k == ProcessKind::Streamer); break;
            case Kind::Call:                           CHECK(k == ProcessKind::Compute); break;
            default: break;
        }
        if (a.kind == Kind::Feed) CHECK((i > 0 && p.actions[i - 1].kind == Kind::Move));
    }
    // Each process's actions are in program order.
    for (const Process& proc : p.processes)
        CHECK(std::is_sorted(proc.actions.begin(), proc.actions.end()));
    // Every residency is released, after its last consumer.
    for (const Residency& r : p.residencies) {
        REQUIRE(r.release != kNone);
        CHECK(r.release > r.open);
    }
    const std::string dis = p.disassemble();
    CHECK_THAT(dis, ContainsSubstring("process dma"));
    CHECK_THAT(dis, ContainsSubstring("channel l3  capacity 4"));
    CHECK_THAT(dis, ContainsSubstring("reuse: "));
}

TEST_CASE("CSP lowering: an op whose tiles cannot all fit the L3 is refused by name",
          "[program][csp]") {
    const TileProgram l0 = program("lu", 64, 16);
    CHECK_THROWS_WITH(lower(l0, {2}), ContainsSubstring("needs 3 tiles in L3 at once; the L3 holds 2"));
}

TEST_CASE("CSP behavioral: a program that reads a released tile is refused",
          "[program][csp]") {
    const TileProgram l0 = program("matmul", 64, 32);
    CspProgram p = lower(l0, {0});
    // Move the first Release ahead of the Move that reads its tile.
    auto rel = std::find_if(p.actions.begin(), p.actions.end(), [&](const Action& a) {
        return a.kind == Kind::Release && p.residencies.at(a.residency).loaded;
    });
    REQUIRE(rel != p.actions.end());
    const Action released = *rel;
    p.actions.erase(rel);
    auto load = std::find_if(p.actions.begin(), p.actions.end(), [&](const Action& a) {
        return a.kind == Kind::Load && a.tile.to_string() == released.tile.to_string();
    });
    REQUIRE(load != p.actions.end());
    p.actions.insert(load + 1, released);
    CHECK_THROWS_WITH(BehavioralInterpreter().run(p), ContainsSubstring("the tile is not in l3"));
}

TEST_CASE("CSP lowering: an explicit program whose call reads an unfed tile is refused by name",
          "[program][csp]") {
    // A program with Feeds is explicit: every kernel input must be fed. Here the call reads B,
    // which no Feed brought to the fabric.
    TileProgram l0("mixed");
    l0.add_operand(TensorOperand("A", 16, 16, 16, 16));
    l0.add_operand(TensorOperand("B", 16, 16, 16, 16));
    l0.add_operand(TensorOperand("C", 16, 16, 16, 16));
    TileOp feed;
    feed.kind = TileOpKind::Feed;
    feed.port = "West";
    feed.inputs = {TileCoord{"A", 0, 0}};
    l0.push(feed);
    TileOp mac;
    mac.kind = TileOpKind::MatMulAccum;
    mac.inputs = {TileCoord{"A", 0, 0}, TileCoord{"B", 0, 0}};
    mac.outputs = {TileCoord{"C", 0, 0}};
    mac.label = "gemm 0";
    l0.push(mac);
    CHECK_THROWS_WITH(lower(l0), ContainsSubstring("L0 op 1 (MATMUL_ACCUM, gemm 0) reads B[0,0], which no Feed"));
    // A fed input is consumed by its call: a second call on the same feed is refused too.
    TileProgram twice("twice");
    twice.add_operand(TensorOperand("A", 16, 16, 16, 16));
    twice.add_operand(TensorOperand("B", 16, 16, 16, 16));
    twice.add_operand(TensorOperand("C", 16, 16, 16, 16));
    TileOp fb = feed;
    fb.inputs = {TileCoord{"B", 0, 0}};
    fb.port = "North";
    twice.push(feed);
    twice.push(fb);
    twice.push(mac);
    twice.push(mac);
    CHECK_THROWS_WITH(lower(twice), ContainsSubstring("L0 op 3"));
    // A Drain of a tile no call produced is refused.
    TileProgram drain("drain");
    drain.add_operand(TensorOperand("C", 16, 16, 16, 16));
    TileOp d;
    d.kind = TileOpKind::Drain;
    d.port_kind = PortKind::Output;
    d.port = "South";
    d.outputs = {TileCoord{"C", 0, 0}};
    drain.push(d);
    CHECK_THROWS_WITH(lower(drain), ContainsSubstring("drains C[0,0], which no call has produced"));
}
