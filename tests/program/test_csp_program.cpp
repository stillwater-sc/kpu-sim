// ============================================================================
// tests/program/test_csp_program.cpp
// The CSP program's trace form (csp_program.hpp; docs/plans/csp-program-tile-sequencing.md
// step 1): the actions a compiled program holds, each owned by its process, and run by L-B to
// the values L-B computes from the program's stream.
//
// The programs are written (kpu-csp-gen), never lowered from an L0: an L0 is a program's trace,
// not its source (docs/plans/retire-l0-derivations.md, PR 2).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/tile_check.hpp>

#include "trace_fixtures.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;
using Kind = Action::Kind;

namespace {

std::size_t count(const CspProgram& p, Kind k) {
    return static_cast<std::size_t>(std::count_if(p.actions.begin(), p.actions.end(),
                                                  [k](const Action& a) { return a.kind == k; }));
}

using trace_fixtures::compiled;

// The trace form run at L-B computes, tile by tile, what L-B computes from the program's stream.
void check_values(const csp::gen::Options& o, const CspProgram& p) {
    BehavioralInterpreter lb;
    const auto sum = lb.run(p);
    const TileProgram oracle = trace_fixtures::lb_values(o);
    for (const auto& name : oracle.operand_order()) {
        const TileCheck c = check_tiles(lb.result(), oracle, name);
        CAPTURE(c.summary());
        CHECK(c.pass);
    }
    if (p.channel(Chan::L3).capacity) CHECK(sum.peak_l3 <= p.channel(Chan::L3).capacity);
}

}  // namespace

TEST_CASE("CSP behavioral: the program's trace form computes what L-B computes, matmul and LU", "[program][csp]") {
    SECTION("matmul, column and row panels") {
        for (const char* orient : {"col", "row"}) {
            CAPTURE(orient);
            csp::gen::Options o = trace_fixtures::gen("matmul", 128, 32);
            o.orient = orient;
            check_values(o, compiled(o));
        }
    }
    SECTION("LU: every tile loaded once, stored once") {
        const csp::gen::Options o = trace_fixtures::gen("lu", 64, 16);
        const CspProgram p = compiled(o);
        check_values(o, p);
        CHECK(p.reuse().reloads == 0);
        CHECK(p.reuse().loads == 16);                       // every A tile once
        CHECK(count(p, Kind::Store) == 16);                 // and stored once
        std::printf("csp LU 64^2/16^2: %zu loads, peak L3 %zu\n", p.reuse().loads, p.reuse().peak_l3);
    }
}

TEST_CASE("CSP program: actions belong to their process, and a tile is fed only from L3", "[program][csp]") {
    const csp::gen::Options o = trace_fixtures::gen("matmul", 64, 32);
    const CspProgram p = compiled(o);
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
    // Every residency is released, after it opens.
    for (const Residency& r : p.residencies) {
        REQUIRE(r.release != kNone);
        CHECK(r.release > r.open);
    }
    const std::string dis = p.disassemble();
    CHECK_THAT(dis, ContainsSubstring("process dma"));
    CHECK_THAT(dis, ContainsSubstring("channel l3  capacity " + std::to_string(o.l3)));
    CHECK_THAT(dis, ContainsSubstring("reuse: "));
}

TEST_CASE("CSP behavioral: a program that reads a released tile is refused", "[program][csp]") {
    CspProgram p = compiled(trace_fixtures::gen("matmul", 64, 32));
    // Move the first Release of a loaded tile ahead of the Move that reads it.
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
    // The read finds its slot free, and L-B names the slot and the tile (ADR 0004 §5).
    CHECK_THROWS_WITH(BehavioralInterpreter().run(p),
                      ContainsSubstring("MOVE " + released.tile.to_string()) &&
                          ContainsSubstring(released.place->to_string() + ": the slot is free (" +
                                            released.tile.to_string() + " is not resident)"));
}
