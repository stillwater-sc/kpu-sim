// ============================================================================
// tests/program/test_tile_dependencies.cpp
// The L0 dependency model: tile hazards, pivot-slot hazards, diagnosis.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/tile_dependencies.hpp>
#include <sw/kpu/program/derive/lu_tile_program.hpp>
#include <sw/kpu/program/derive/matmul_tile_program.hpp>
#include <sw/kpu/program/characterize/tile_dag.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::characterize;

namespace {

bool has_edge(const TileDependencies& d, std::size_t from, std::size_t to, TileDepKind kind) {
    return std::any_of(d.edges.begin(), d.edges.end(), [&](const TileDepEdge& e) {
        return e.from == from && e.to == to && e.kind == kind;
    });
}

std::size_t count_kind(const TileDependencies& d, TileDepKind kind) {
    return static_cast<std::size_t>(std::count_if(d.edges.begin(), d.edges.end(),
        [&](const TileDepEdge& e) { return e.kind == kind; }));
}

// A two-panel LU-shaped program that REUSES pivot slot 0 for both panels. The
// derivations in-tree use a unique slot per panel, so this is the case the
// anti-dependency edges exist for, constructed explicitly.
TileProgram slot_reusing_program() {
    TileProgram prog("slot reuse");
    prog.add_operand(TensorOperand("A", 4, 4, 2, 2));

    auto op = [](TileOpKind k, TileCoord out, int slot) {
        TileOp o;
        o.kind = k;
        o.outputs = {out};
        o.pivot_slot = slot;
        return o;
    };
    prog.push(op(TileOpKind::LuDiagFactor, TileCoord{"A", 0, 0}, 0));   // 0: writes slot 0
    prog.push(op(TileOpKind::PivotApply,   TileCoord{"A", 0, 1}, 0));   // 1: reads slot 0
    prog.push(op(TileOpKind::LuDiagFactor, TileCoord{"A", 1, 1}, 0));   // 2: REWRITES slot 0
    prog.push(op(TileOpKind::PivotApply,   TileCoord{"A", 1, 0}, 0));   // 3: reads slot 0 again
    return prog;
}

} // namespace

TEST_CASE("tile hazards are recovered from declared tile I/O", "[program][deps]") {
    // C[0,0] accumulates over two K-slices, so the second must follow the first.
    TileProgram prog = derive_matmul_tile_program(32, 32, 32, 16, 16, 16);
    const TileDependencies d = build_tile_dependencies(prog);

    REQUIRE(d.op_count() == prog.ops().size());
    CHECK(count_kind(d, TileDepKind::TileRaw) > 0);
    CHECK(count_kind(d, TileDepKind::FeedAvailable) > 0);
    CHECK(count_kind(d, TileDepKind::TileWaw) > 0);   // accumulation into one output tile

    SECTION("preds and succs are consistent and acyclic in program order") {
        for (std::size_t i = 0; i < d.op_count(); ++i)
            for (std::size_t p : d.preds[i]) {
                CHECK(p < i);                       // a producer always precedes its consumer
                const auto& s = d.succs[p];
                CHECK(std::find(s.begin(), s.end(), i) != s.end());
            }
    }

    SECTION("pairs are deduplicated in preds, and every reason is kept in edges") {
        std::size_t pair_count = 0;
        for (std::size_t i = 0; i < d.op_count(); ++i) {
            auto v = d.preds[i];
            const auto before = v.size();
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            CHECK(v.size() == before);          // no duplicate pair in preds
            pair_count += before;
        }
        // At least one typed edge per deduplicated pair, and strictly more when a pair
        // is ordered for several reasons. Comparing against op_count would prove nothing.
        CHECK(d.edges.size() >= pair_count);

        // Every typed edge must be reflected in BOTH adjacency lists.
        for (const TileDepEdge& e : d.edges) {
            const auto& p = d.preds[e.to];
            const auto& s = d.succs[e.from];
            CHECK(std::find(p.begin(), p.end(), e.from) != p.end());
            CHECK(std::find(s.begin(), s.end(), e.to) != s.end());
        }
    }

    SECTION("matmul has no pivot edges at all") {
        CHECK(count_kind(d, TileDepKind::PivotRaw) == 0);
        CHECK(count_kind(d, TileDepKind::PivotWar) == 0);
        CHECK(count_kind(d, TileDepKind::PivotWaw) == 0);
        CHECK(d.pivot_slots_single_assignment());
    }
}

TEST_CASE("pivot-slot dataflow is recovered for LU", "[program][deps]") {
    TileProgram prog = derive_lu_tile_program(64, 32);
    const TileDependencies d = build_tile_dependencies(prog);

    // GETRF -> LASWP on the same slot is the data-dependent control edge.
    CHECK(count_kind(d, TileDepKind::PivotRaw) > 0);

    SECTION("the in-tree derivation assigns each slot once, so no slot anti-deps arise") {
        CHECK(d.pivot_slots_single_assignment());
        CHECK(d.reused_pivot_slots().empty());
        CHECK(count_kind(d, TileDepKind::PivotWar) == 0);
        CHECK(count_kind(d, TileDepKind::PivotWaw) == 0);
    }

    SECTION("every slot edge names the slot it came from") {
        for (const TileDepEdge& e : d.edges)
            if (is_pivot_dependency(e.kind))
                CHECK(e.subject.rfind("pivot#", 0) == 0);
    }
}

TEST_CASE("pivot-slot reuse is ordered by anti-dependencies", "[program][deps]") {
    // The §3.3 hazard: LuDiagFactor CLEARS its slot on entry, so a later writer
    // must not run before earlier readers of that slot have replayed the swaps.
    TileProgram prog = slot_reusing_program();
    const TileDependencies d = build_tile_dependencies(prog);

    CHECK(has_edge(d, 0, 1, TileDepKind::PivotRaw));   // op1 replays op0's swaps
    CHECK(has_edge(d, 1, 2, TileDepKind::PivotWar));   // op2 must wait for reader op1
    CHECK(has_edge(d, 0, 2, TileDepKind::PivotWaw));   // writers stay ordered
    CHECK(has_edge(d, 2, 3, TileDepKind::PivotRaw));   // op3 replays op2's swaps

    SECTION("reuse is reported, since it serializes otherwise-independent panels") {
        CHECK_FALSE(d.pivot_slots_single_assignment());
        REQUIRE(d.reused_pivot_slots().size() == 1);
        CHECK(d.reused_pivot_slots()[0] == 0);
        REQUIRE(d.slot_writers.at(0).size() == 2);
        CHECK(d.slot_writers.at(0)[0] == 0);
        CHECK(d.slot_writers.at(0)[1] == 2);
    }

    SECTION("without the WAR edge op2 could clear the slot before op1 read it") {
        // op2 is transitively after op1, which is the property that makes reuse safe.
        const auto& p = d.preds[2];
        CHECK(std::find(p.begin(), p.end(), std::size_t{1}) != p.end());
    }
}

TEST_CASE("dependency diagnosis explains why an op is blocked", "[program][deps]") {
    TileProgram prog = derive_lu_tile_program(64, 32);
    const TileDependencies d = build_tile_dependencies(prog);

    std::vector<bool> completed(d.op_count(), false);

    SECTION("the first op is blocked by nothing") {
        CHECK(d.blocking_edges(0, completed).empty());
        CHECK(d.explain_blocked(prog, 0, completed).find("not blocked") != std::string::npos);
    }

    SECTION("a later op names its blockers, its edge kinds and its subjects") {
        const std::size_t last = d.op_count() - 1;
        const auto blockers = d.blocking_edges(last, completed);
        REQUIRE_FALSE(blockers.empty());
        const std::string why = d.explain_blocked(prog, last, completed);
        CHECK(why.find("waits on") != std::string::npos);
        CHECK(why.find("op " + std::to_string(blockers.front().from)) != std::string::npos);
    }

    SECTION("a wrong-sized completion snapshot is refused, not silently mis-answered") {
        std::vector<bool> too_short(d.op_count() - 1, false);
        CHECK_THROWS_AS(d.blocking_edges(d.op_count() - 1, too_short), std::invalid_argument);
        CHECK_THROWS_AS(d.explain_blocked(prog, 0, too_short), std::invalid_argument);
        CHECK_THROWS_AS(d.blocking_edges(d.op_count(), completed), std::out_of_range);
    }

    SECTION("completing the predecessors clears the blockers") {
        const std::size_t last = d.op_count() - 1;
        for (std::size_t p : d.preds[last]) completed[p] = true;
        CHECK(d.blocking_edges(last, completed).empty());
    }
}

TEST_CASE("TileDag reuses the shared dependency model", "[program][deps][characterize]") {
    // One recovery, not two: the analysis must not disagree with the executor
    // about what is legal.
    TileProgram prog = derive_lu_tile_program(64, 32);
    TileDag dag(prog, DeviceDescriptor::single());
    const TileDependencies& d = dag.dependencies();

    REQUIRE(dag.nodes().size() == d.op_count());
    for (std::size_t i = 0; i < d.op_count(); ++i) {
        CHECK(dag.nodes()[i].preds == d.preds[i]);
        CHECK(dag.nodes()[i].succs == d.succs[i]);
    }

    SECTION("edge recovery is deterministic") {
        const TileDependencies again = build_tile_dependencies(prog);
        REQUIRE(again.edges.size() == d.edges.size());
        for (std::size_t i = 0; i < d.edges.size(); ++i) {
            CHECK(again.edges[i].from == d.edges[i].from);
            CHECK(again.edges[i].to == d.edges[i].to);
            CHECK(again.edges[i].kind == d.edges[i].kind);
            CHECK(again.edges[i].subject == d.edges[i].subject);
        }
    }
}
