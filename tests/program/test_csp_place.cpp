// ============================================================================
// tests/program/test_csp_place.cpp
// L3 placement (docs/plans/l3-placement-at-lb.md step 2; ADR 0004 §5): every residency has an L3
// address -- module, slot, bank -- allocated once by the walker, by the program's policy or where
// the program names it, and carried on every action that touches the residency.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/place.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>

#include "trace_fixtures.hpp"

#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using namespace sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;
using Kind = Action::Kind;

namespace {

// The program text with its machine clause's options replaced.
std::string with_machine(std::string text, const std::string& options) {
    const auto at = text.find("machine flat(l3 = ");
    REQUIRE(at != std::string::npos);
    const auto close = text.find(')', at);
    const auto eq = text.find("= ", at) + 2;
    const std::string l3 = text.substr(eq, text.find_first_of(",)", eq) - eq);
    return text.replace(at, close - at + 1, "machine flat(l3 = " + l3 + options + ")");
}

// Walk the actions: no two live residencies share a slot, every action carries its residency's
// address, a Release frees exactly its slot, and no more slots are live than the L3 holds.
void check_placement(const CspProgram& p) {
    const std::size_t cap = p.channel(Chan::L3).capacity;
    std::map<std::uint32_t, std::size_t> live;           // slot -> residency
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        const Action& a = p.actions[i];
        CAPTURE(i, to_string(a.kind), a.tile.to_string());
        if (a.residency == kNone) {
            CHECK_FALSE(a.place.has_value());
            continue;
        }
        const Residency& r = p.residencies.at(a.residency);
        REQUIRE(a.place.has_value());
        CHECK(*a.place == r.place);
        CHECK(a.place->slot < cap);
        CHECK(a.place->bank == a.place->slot % p.l3_banks);
        if (r.open == i) {
            CHECK(live.count(a.place->slot) == 0);       // a slot holds one live tile
            live[a.place->slot] = a.residency;
        } else {
            REQUIRE(live.count(a.place->slot) == 1);     // read where it was placed
            CHECK(live.at(a.place->slot) == a.residency);
        }
        if (a.kind == Kind::Release) live.erase(a.place->slot);
        CHECK(live.size() <= cap);
    }
}

std::vector<std::uint32_t> opened_slots(const CspProgram& p) {
    std::vector<std::uint32_t> out;
    for (const Residency& r : p.residencies) out.push_back(r.place.slot);
    return out;
}

}  // namespace

TEST_CASE("placement: the allocator's policies", "[csp][placement]") {
    SECTION("fifo takes the slot freed longest ago") {
        L3Allocator a(3, 1, PlacePolicy::Fifo);
        CHECK(a.take().slot == 0);
        CHECK(a.take().slot == 1);
        CHECK(a.take().slot == 2);
        a.give(a.address(1));
        a.give(a.address(0));
        CHECK(a.take().slot == 1);
        CHECK(a.take().slot == 0);
    }
    SECTION("lowest takes the lowest free slot") {
        L3Allocator a(3, 1, PlacePolicy::Lowest);
        for (int i = 0; i < 3; ++i) (void)a.take();
        a.give(a.address(2));
        a.give(a.address(0));
        CHECK(a.take().slot == 0);
        CHECK(a.take().slot == 2);
    }
    SECTION("bank-spread takes a slot in the emptiest bank") {
        L3Allocator a(4, 2, PlacePolicy::BankSpread);    // banks: slots 0,2 -> b0; 1,3 -> b1
        const L3Address x = a.take();
        const L3Address y = a.take();
        CHECK(x.bank != y.bank);
        CHECK(x.slot == 0);
        CHECK(y.slot == 1);
        CHECK(a.held() == 2);
    }
    SECTION("an explicit slot, and the refusals") {
        L3Allocator a(4, 1, PlacePolicy::Fifo);
        L3Address got;
        CHECK_FALSE(a.take_at(2, got).has_value());
        CHECK(got.slot == 2);
        CHECK_THAT(*a.take_at(2, got), ContainsSubstring("holds a live tile"));
        CHECK_THAT(*a.take_at(9, got), ContainsSubstring("outside the L3 of 4"));
        CHECK(a.take().slot == 0);                       // the policy skips the taken slot
        CHECK(a.take().slot == 1);
        CHECK(a.take().slot == 3);
        CHECK_THROWS(a.take());                          // the walker checks capacity first
        CHECK_THROWS(L3Allocator(2, 1, PlacePolicy::Fifo).give(L3Address{0, 1, 0}));
    }
}

TEST_CASE("placement: every residency of matmul and LU is placed, under each policy", "[csp][placement]") {
    for (const char* algo : {"matmul", "lu"}) {
        for (const char* opts : {"", ", place = lowest", ", banks = 4, place = bank-spread", ", banks = 4"}) {
            CAPTURE(algo, opts);
            const auto o = trace_fixtures::gen(algo, algo == std::string("lu") ? 64 : 128, algo == std::string("lu") ? 16 : 32);
            const CspProgram p = compile(with_machine(gen::generate(o), opts));
            REQUIRE_FALSE(p.residencies.empty());
            check_placement(p);
        }
    }
}

TEST_CASE("placement: the policy decides where reuse lands", "[csp][placement]") {
    const std::string text = gen::generate(trace_fixtures::gen("matmul", 128, 32));
    const CspProgram fifo = compile(text);
    const CspProgram lowest = compile(with_machine(text, ", place = lowest"));
    CHECK(fifo.place == "fifo");
    CHECK(lowest.place == "lowest");
    // Same actions, same residencies; different slots once the first one is reused.
    REQUIRE(fifo.residencies.size() == lowest.residencies.size());
    CHECK(opened_slots(fifo) != opened_slots(lowest));
    // fifo cycles through every slot before reusing one; lowest reuses the lowest freed slot.
    const std::size_t cap = fifo.channel(Chan::L3).capacity;
    const std::vector<std::uint32_t> all = opened_slots(fifo);
    REQUIRE(all.size() >= cap);
    const std::vector<std::uint32_t> first(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(cap));
    std::vector<std::uint32_t> in_order(cap);
    for (std::uint32_t s = 0; s < cap; ++s) in_order[s] = s;
    CHECK(first == in_order);
}

TEST_CASE("placement: a program may name its slots", "[csp][placement]") {
    const std::string base = R"(csp 1.0
program p machine flat(l3 = 6) {
  tensor A[4,4] tile 2x2 in;
  tensor B[4,4] tile 2x2 in;
  tensor C[4,4] tile 2x2 out;
  for j in 0..2 {
    resident B[:, j] @ slots(4..6);
    for i in 0..2 {
      resident A[i, :];
      acc C[i, j] in fabric {
        for k in 0..2 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }
      }
      store C[i, j];
      release A[i, :];
    }
    release B[:, j];
  }
}
)";
    const CspProgram p = compile(base);
    check_placement(p);
    for (const Residency& r : p.residencies)
        if (r.tile.operand == "B") {
            CHECK(r.pinned);
            CHECK(r.place.slot == 4 + r.tile.ti);        // one slot per tile, in order
        } else {
            CHECK_FALSE(r.pinned);
            CHECK(r.place.slot < 4);                     // the policy places the rest around them
        }

    SECTION("the text keeps the placement, and print reproduces it") {
        CHECK_THAT(format(parse(base)), ContainsSubstring("resident B[:, j] @ slots(4..6);"));
        const CspProgram again = compile(print(p));
        CHECK(opened_slots(again) == opened_slots(p));
    }
    SECTION("a slot that holds a live tile is refused, naming it") {
        std::string clash = base;
        clash.replace(clash.find("resident A[i, :];"), 17, "resident A[i, :] @ slots(3..5);");
        CHECK_THROWS_WITH(compile(clash), ContainsSubstring("slot 4 holds a live tile (B[0,0])"));
    }
    SECTION("one slot per tile, inside the L3") {
        std::string few = base;
        few.replace(few.find("@ slots(4..6)"), 13, "@ slots(4..5)");
        CHECK_THROWS_WITH(compile(few), ContainsSubstring("1 slots for 2 tiles"));
        std::string outside = base;
        outside.replace(outside.find("@ slots(4..6)"), 13, "@ slots(5..7)");
        CHECK_THROWS_WITH(compile(outside), ContainsSubstring("outside the L3 of 6"));
    }
}

TEST_CASE("placement: the machine clause's options, and their text", "[csp][placement]") {
    const std::string text = gen::generate(trace_fixtures::gen("matmul", 64, 32));
    // Defaults are not printed: a program that names none keeps its text, and its digest.
    CHECK(format(parse(text)).find("place =") == std::string::npos);
    CHECK(format(parse(text)).find("banks =") == std::string::npos);
    const std::string named = with_machine(text, ", banks = 4, place = bank-spread");
    CHECK_THAT(format(parse(named)), ContainsSubstring(", banks = 4, place = bank-spread)"));
    CHECK_THAT(print(compile(named)), ContainsSubstring(", banks = 4, place = bank-spread)"));
    CHECK_THROWS_WITH(parse(with_machine(text, ", place = random")), ContainsSubstring("fifo, lowest or bank-spread"));
    CHECK_THROWS_WITH(parse(with_machine(text, ", banks = 0")), ContainsSubstring("banks must be positive"));
    CHECK_THROWS_WITH(parse(with_machine(text, ", ways = 2")), ContainsSubstring("banks or place"));
}
