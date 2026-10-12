// ============================================================================
// tests/program/test_csp_lb_place.cpp
// L-B executes L3 placement (docs/plans/l3-placement-at-lb.md step 3; ADR 0004 §5): its L3 is
// addressed storage. Every action on a residency writes, reads or frees the slot the walker
// placed it at, and a program that writes a live slot, reads a slot holding another tile, or
// frees the wrong slot (#279's class of defect) is refused at that action, naming the slot and
// both tiles.
//
// The trace records it (step 4, opset 1.2): one L3 transaction per action on a residency, with
// its address. It round-trips through the L0 text, and the reference's replay executes it on an
// addressed L3, refusing what L-B refuses.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/csp_step.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include "trace_fixtures.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Catch::Matchers::ContainsSubstring;
using Kind = Action::Kind;

namespace {

// The program text with its machine clause's options added.
std::string with_machine(std::string text, const std::string& options) {
    const auto at = text.find("machine flat(l3 = ");
    REQUIRE(at != std::string::npos);
    const auto close = text.find(')', at);
    return text.insert(close, options);
}

CspProgram compiled_text(const std::string& text) {
    return trace_fixtures::compiled(lang::parse(text));
}

// The live slots just before action `upto`: slot -> the tile in it.
std::map<std::uint32_t, std::string> live_before(const CspProgram& p, std::size_t upto) {
    std::map<std::uint32_t, std::string> live;
    for (std::size_t i = 0; i < upto; ++i) {
        const Action& a = p.actions[i];
        if (!a.place) continue;
        if (a.kind == Kind::Release) live.erase(a.place->slot);
        else live[a.place->slot] = a.tile.to_string();
    }
    return live;
}

std::size_t first(const CspProgram& p, Kind k, std::size_t from = 0) {
    for (std::size_t i = from; i < p.actions.size(); ++i)
        if (p.actions[i].kind == k) return i;
    FAIL("no " << to_string(k) << " from action " << from);
    return 0;
}

// A live slot at action `i` other than the action's own.
L3Address other_live(const CspProgram& p, std::size_t i) {
    for (const auto& [slot, tile] : live_before(p, i))
        if (slot != p.actions[i].place->slot) return L3Address{0, slot, slot % p.l3_banks};
    FAIL("no other live slot at action " << i);
    return {};
}

bool same_bits(const TileProgram& a, const TileProgram& b) {
    for (const auto& name : a.operand_order()) {
        const auto& x = a.operand(name).values;
        const auto& y = b.operand(name).values;
        if (x.size() != y.size() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) != 0) return false;
    }
    return true;
}

TileProgram lb_run(const CspProgram& p) {
    BehavioralInterpreter lb;
    (void)lb.run(p);
    return lb.result();
}

}  // namespace

TEST_CASE("L-B: every action lands in the slot the walker placed it at", "[csp][placement][behavioral]") {
    std::vector<std::string> texts;
    for (const char* f : {"tests/program/corpus/matmul_48x48x48_t16.csp", "tests/program/corpus/lu_64_t16.csp",
                          "tests/program/corpus/linear_64_t16_relu.csp"}) {
        std::ifstream in(f, std::ios::binary);
        REQUIRE(in);
        std::ostringstream ss;
        ss << in.rdbuf();
        texts.push_back(ss.str());
    }
    for (const std::string& text : texts) {
        for (const char* opts : {"", ", place = lowest", ", banks = 4, place = bank-spread"}) {
            const CspProgram p = compiled_text(with_machine(text, opts));
            CAPTURE(p.source.name(), opts);
            BehavioralInterpreter lb;
            lb.begin(p.source, false, l3_geometry(p));
            std::map<std::uint32_t, std::string> live;
            for (std::size_t i = 0; i < p.actions.size(); ++i) {
                const Action& a = p.actions[i];
                lb.step(a, a.kind == Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
                if (!a.place) continue;
                CAPTURE(i, to_string(a.kind), a.tile.to_string());
                if (a.kind == Kind::Release) {
                    live.erase(a.place->slot);
                    CHECK_FALSE(lb.resident_at(a.tile).has_value());
                } else {
                    live[a.place->slot] = a.tile.to_string();
                    REQUIRE(lb.resident_at(a.tile).has_value());
                    CHECK(*lb.resident_at(a.tile) == *a.place);
                }
                CHECK(lb.held(Chan::L3) == live.size());     // occupancy is the tagged slots
            }
            (void)lb.finish();
        }
    }
}

TEST_CASE("L-B: placement does not change values", "[csp][placement][behavioral]") {
    // The same tiles are read and written in the same order, whatever their slots: the same
    // computation, so the values agree bit for bit.
    for (const char* algo : {"matmul", "lu"}) {
        CAPTURE(algo);
        const auto o = trace_fixtures::gen(algo, algo == std::string("lu") ? 64 : 96, algo == std::string("lu") ? 16 : 32);
        const std::string text = gen::generate(o);
        const TileProgram fifo = lb_run(compiled_text(text));
        CHECK(same_bits(fifo, *trace_fixtures::reference_of(lang::parse(text)).values));
        for (const char* opts : {", place = lowest", ", banks = 4, place = bank-spread"}) {
            CAPTURE(opts);
            const CspProgram p = compiled_text(with_machine(text, opts));
            CHECK(same_bits(lb_run(p), fifo));
        }
    }
}

TEST_CASE("L-B refuses an action that breaks placement, naming the slot and the tiles", "[csp][placement][behavioral]") {
    const std::string text = gen::generate(trace_fixtures::gen("matmul", 64, 32));
    const CspProgram good = compiled_text(text);
    REQUIRE_NOTHROW(lb_run(good));

    SECTION("a Load into a live slot") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Load, first(p, Kind::Load) + 1);
        const L3Address taken = other_live(p, i);
        const std::string holder = live_before(p, i).at(taken.slot);
        p.actions[i].place = taken;
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring("csp action " + std::to_string(i) + " LOAD") &&
                                         ContainsSubstring(taken.to_string() + " holds " + holder + ", which is live"));
    }
    SECTION("a Move that reads a slot holding another tile") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Move);
        const L3Address wrong = other_live(p, i);
        const std::string holder = live_before(p, i).at(wrong.slot);
        p.actions[i].place = wrong;
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring(wrong.to_string() + " holds " + holder + ", not " +
                                                       p.actions[i].tile.to_string()));
    }
    SECTION("#279: a Release that frees another tile's slot") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Release);
        const L3Address wrong = other_live(p, i);
        const std::string holder = live_before(p, i).at(wrong.slot);
        p.actions[i].place = wrong;
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring("RELEASE") &&
                                         ContainsSubstring(wrong.to_string() + " holds " + holder + ", not " +
                                                           p.actions[i].tile.to_string()));
    }
    SECTION("a Release of a slot already free") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Release);
        p.actions.insert(p.actions.begin() + static_cast<std::ptrdiff_t>(i) + 1, p.actions[i]);
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring(p.actions[i].place->to_string() +
                                                       ": nothing resident to release"));
    }
    SECTION("a residency never released") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Release);
        const std::string tile = p.actions[i].tile.to_string();
        const L3Address at = *p.actions[i].place;
        p.actions.erase(p.actions.begin() + static_cast<std::ptrdiff_t>(i));
        // The slot stays live: the next tile the walker placed there collides with it, or, if none
        // was, the program ends with it resident. Either way L-B names the tile and its slot.
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring(at.to_string()) && ContainsSubstring(tile));
    }
    SECTION("an action with no address, or one outside the L3") {
        CspProgram p = good;
        const std::size_t i = first(p, Kind::Load);
        p.actions[i].place.reset();
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring("the action carries no L3 address"));
        const auto slots = static_cast<std::uint32_t>(good.channel(Chan::L3).capacity);
        p.actions[i].place = L3Address{0, slots, 0};
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring("is outside the L3's " + std::to_string(slots) + " slots"));
    }
    SECTION("a bank that is not the slot's") {
        CspProgram p = compiled_text(with_machine(text, ", banks = 4"));
        const std::size_t i = first(p, Kind::Load);
        p.actions[i].place->bank = (p.actions[i].place->bank + 1) % 4;
        CHECK_THROWS_WITH(lb_run(p), ContainsSubstring("is in bank"));
    }
}

TEST_CASE("L-B stepping names each action's slot", "[csp][placement][behavioral]") {
    const lang::Program ast = trace_fixtures::program_of(trace_fixtures::gen("matmul", 64, 32));
    driver::CspBehavioralStepper st(ast, trace_fixtures::compiled(ast).source);
    std::size_t placed = 0;
    while (st.step()) {
        const driver::CspStep& s = st.current();
        if (s.place) {
            ++placed;
            CHECK_THAT(driver::describe(s, false), ContainsSubstring(" @ " + s.place->to_string()));
        } else {
            CHECK(driver::describe(s, false).find(" @ l3[") == std::string::npos);
        }
    }
    CHECK(placed > 0);
}

namespace {

std::size_t first_op(const TileProgram& t, TileOpKind k, std::size_t from = 0) {
    for (std::size_t i = from; i < t.ops().size(); ++i)
        if (t.ops()[i].kind == k) return i;
    FAIL("no " << to_string(k) << " from op " << from);
    return 0;
}

// The tiles live in L3 just before op `upto` of a trace: slot -> tile.
std::map<std::uint32_t, std::string> live_ops_before(const TileProgram& t, std::size_t upto) {
    std::map<std::uint32_t, std::string> live;
    for (std::size_t i = 0; i < upto; ++i) {
        const TileOp& op = t.ops()[i];
        if (op.kind == TileOpKind::L3Release) live.erase(op.l3.slot);
        else if (op.kind == TileOpKind::L3Load || op.kind == TileOpKind::L3Writeback || op.kind == TileOpKind::L3Inherit)
            live[op.l3.slot] = op.outputs.at(0).to_string();
    }
    return live;
}

}  // namespace

TEST_CASE("the trace records every L3 transaction with its address", "[csp][placement][l0]") {
    for (const char* algo : {"matmul", "lu"}) {
        CAPTURE(algo);
        const auto o = trace_fixtures::gen(algo, algo == std::string("lu") ? 64 : 48, 16);
        const CspProgram p = trace_fixtures::compiled(o);
        std::size_t placed = 0;
        for (const Action& a : p.actions) {
            if (!a.place) continue;
            ++placed;
            CAPTURE(to_string(a.kind), a.tile.to_string());
            REQUIRE(a.l0_op != kNone);
            const TileOp& op = p.source.ops().at(a.l0_op);
            CHECK(is_l3(op.kind));
            CHECK(op.l3 == *a.place);
            REQUIRE(op.outputs.size() == 1);
            CHECK(op.outputs[0].to_string() == a.tile.to_string());
            CHECK(op.inputs.empty());
        }
        // The trace L-B records holds the same transactions, in program order.
        const TileProgram trace = trace_fixtures::trace_of(o);
        std::size_t l3_ops = 0;
        for (const TileOp& op : trace.ops()) l3_ops += is_l3(op.kind) ? 1 : 0;
        CHECK(l3_ops == placed);
        CHECK(serialize::opset_for(trace).str() == "1.2.0");
        // A Feed follows the Move that took its tile out of L3. (LU's in-place calls record no
        // Feed: only an accumulating call's operands are fed in the trace.)
        if (std::string(algo) == "matmul") {
            const std::size_t feed = first_op(trace, TileOpKind::Feed);
            REQUIRE(feed > 0);
            CHECK(trace.ops()[feed - 1].kind == TileOpKind::L3Move);
            CHECK(trace.ops()[feed - 1].outputs[0].to_string() == trace.ops()[feed].inputs[0].to_string());
        }
    }
}

TEST_CASE("L3 transactions round-trip through the L0 text and replay", "[csp][placement][l0]") {
    const TileProgram trace = trace_fixtures::trace_of(trace_fixtures::gen("lu", 64, 16));
    serialize::WriteOptions opt;
    opt.include_values = true;
    const std::string text = serialize::to_string(trace, opt);
    const TileProgram back = serialize::from_string(text);
    REQUIRE(back.ops().size() == trace.ops().size());
    for (std::size_t i = 0; i < back.ops().size(); ++i) {
        CAPTURE(i);
        CHECK(back.ops()[i].kind == trace.ops()[i].kind);
        CHECK(back.ops()[i].l3 == trace.ops()[i].l3);
    }
    CHECK(serialize::to_string(back, opt) == text);
    TileProgram a = trace, b = back;
    const auto sa = TileProgramReference().run(a);
    const auto sb = TileProgramReference().run(b);
    CHECK(sa.l3_ops == sb.l3_ops);
    CHECK(sa.peak_l3 == sb.peak_l3);
    CHECK(sa.l3_ops > 0);

    SECTION("an L3 transaction without its address, or naming its tile as in=, is refused") {
        const auto at = text.find(" slot=");
        REQUIRE(at != std::string::npos);
        std::string no_slot = text;
        no_slot.erase(at, no_slot.find(' ', at + 1) - at);
        CHECK_THROWS_AS(serialize::from_string(no_slot), serialize::FormatError);
        std::string as_input = text;
        const auto op = as_input.find("OP kind=L3_LOAD out=");
        as_input.replace(op, 20, "OP kind=L3_LOAD in=");
        CHECK_THROWS_WITH(serialize::from_string(as_input), ContainsSubstring("names one tile, as out="));
    }
}

TEST_CASE("the replay refuses a trace that breaks placement", "[csp][placement][l0]") {
    const TileProgram good = trace_fixtures::trace_of(trace_fixtures::gen("matmul", 64, 32));
    {
        TileProgram t = good;
        REQUIRE_NOTHROW(TileProgramReference().run(t));
    }
    SECTION("#279: a release of another tile's slot") {
        TileProgram t = good;
        const std::size_t i = first_op(t, TileOpKind::L3Release);
        const auto live = live_ops_before(t, i);
        std::uint32_t other = 0;
        for (const auto& [slot, tile] : live)
            if (slot != t.ops()[i].l3.slot) { other = slot; break; }
        REQUIRE(other != t.ops()[i].l3.slot);
        t.ops()[i].l3.slot = other;
        CHECK_THROWS_WITH(TileProgramReference().run(t),
                          ContainsSubstring("L3_RELEASE") && ContainsSubstring(" holds " + live.at(other) + ", not " +
                                                                               t.ops()[i].outputs[0].to_string()));
    }
    SECTION("a load into a live slot") {
        TileProgram t = good;
        const std::size_t first = first_op(t, TileOpKind::L3Load);
        const std::size_t i = first_op(t, TileOpKind::L3Load, first + 1);
        t.ops()[i].l3 = t.ops()[first].l3;
        CHECK_THROWS_WITH(TileProgramReference().run(t),
                          ContainsSubstring(" holds " + t.ops()[first].outputs[0].to_string() + ", which is live"));
    }
    SECTION("a read or release whose bank is not the slot's") {
        // The slot is keyed by module and slot, so the bank is compared as part of the address.
        TileProgram t = good;
        const std::size_t i = first_op(t, TileOpKind::L3Move);
        const L3Address opened = t.ops()[i].l3;
        t.ops()[i].l3.bank += 1;
        CHECK_THROWS_WITH(TileProgramReference().run(t),
                          ContainsSubstring("L3_MOVE") && ContainsSubstring("'s address, " + opened.to_string()));
        CspProgram p = trace_fixtures::compiled(trace_fixtures::gen("matmul", 64, 32));
        const std::size_t r = first(p, Kind::Release);
        const L3Address at = *p.actions[r].place;
        p.actions[r].place->bank += 1;
        BehavioralInterpreter lb;                       // no geometry: the bank is not range-checked
        lb.begin(p.source);
        CHECK_THROWS_WITH([&] {
            for (const Action& a : p.actions) lb.step(a, a.kind == Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
        }(), ContainsSubstring("RELEASE") && ContainsSubstring("'s address, " + at.to_string()));
    }
    SECTION("a release of a free slot, and a slot never freed") {
        TileProgram twice = good;
        const std::size_t i = first_op(twice, TileOpKind::L3Release);
        twice.ops().insert(twice.ops().begin() + static_cast<std::ptrdiff_t>(i) + 1, twice.ops()[i]);
        CHECK_THROWS_WITH(TileProgramReference().run(twice), ContainsSubstring("the slot is free"));
        TileProgram never = good;
        never.ops().erase(never.ops().begin() + static_cast<std::ptrdiff_t>(i));
        CHECK_THROWS_AS(TileProgramReference().run(never), PlacementError);
    }
}
