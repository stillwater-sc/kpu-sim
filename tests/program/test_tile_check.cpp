// ============================================================================
// tests/program/test_tile_check.cpp
// The functional check on tile contents (docs/plans/retire-l0-derivations.md §3.8): each tile
// read in the program's format, every element within a proximity bound of the oracle's, a miss
// named by its tile.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/tile_check.hpp>

#include <cmath>
#include <limits>

using namespace sw::kpu::program;
using Catch::Matchers::ContainsSubstring;

namespace {

// 5x7 in 2x3 tiles: a 3x3 grid whose last row and column of tiles clamp.
TensorOperand operand() {
    TensorOperand t("C", 5, 7, 2, 3);
    for (std::size_t i = 0; i < t.values.size(); ++i) t.values[i] = 0.125f * float(i) - 2.0f;
    return t;
}

}  // namespace

TEST_CASE("tile check: identical tiles pass, every tile visited, clamped ones included", "[program][tile_check]") {
    const TensorOperand a = operand();
    const TileCheck r = check_tiles(a, a, ElementFormat{}, proximity_for(ElementFormat{}));
    CHECK(r.pass);
    CHECK(r.tiles == 9);
    CHECK(r.elements == 35);
    CHECK(r.max_abs == 0.0);
    CHECK(r.misses.empty());
    CHECK_THAT(r.summary(), ContainsSubstring("within bound"));
}

TEST_CASE("tile check: a reordered reduction's rounding is within bound; the bound is the bar", "[program][tile_check]") {
    const TensorOperand oracle = operand();
    TensorOperand a = oracle;
    // A last-place difference, as a different accumulation order leaves: within 1e-6 + 1e-4|x|.
    a.at(1, 1) = std::nextafter(a.at(1, 1), 1e9f);
    CHECK(check_tiles(a, oracle, ElementFormat{}, proximity_for(ElementFormat{})).pass);
    // Beyond the bound, in the clamped corner tile (rows 4, cols 6): named by tile and element.
    a.at(4, 6) += 0.5f;
    const TileCheck r = check_tiles(a, oracle, ElementFormat{}, proximity_for(ElementFormat{}));
    CHECK_FALSE(r.pass);
    REQUIRE(r.misses.size() == 1);
    CHECK(r.misses[0].tile.to_string() == TileCoord{"C", 2, 2}.to_string());
    CHECK(r.misses[0].row == 4);
    CHECK(r.misses[0].col == 6);
    CHECK(r.misses[0].elements == 1);
    CHECK_THAT(r.summary(), ContainsSubstring("1 tile(s) out of bound"));
    // A looser bound, stated by the test, admits it.
    CHECK(check_tiles(a, oracle, ElementFormat{}, Proximity{0.0, 0.5}).pass);
}

TEST_CASE("tile check: non-finite values are out of bound, whichever side", "[program][tile_check]") {
    const TensorOperand oracle = operand();
    TensorOperand a = oracle;
    a.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(check_tiles(a, oracle, ElementFormat{}, proximity_for(ElementFormat{})).pass);
    a = oracle;
    a.at(0, 0) = std::numeric_limits<float>::infinity();
    TensorOperand o = oracle;
    o.at(0, 0) = std::numeric_limits<float>::infinity();
    CHECK_FALSE(check_tiles(a, o, ElementFormat{}, proximity_for(ElementFormat{})).pass);
}

TEST_CASE("tile check: a different grid is structural, and an uncarried format is refused", "[program][tile_check]") {
    const TensorOperand oracle = operand();
    const TensorOperand other("C", 5, 7, 3, 3);
    CHECK_THROWS_WITH(check_tiles(other, oracle, ElementFormat{}, Proximity{}), ContainsSubstring("the oracle's is 5x7 in 2x3"));
    ElementFormat bf16;
    bf16.type = sw::kpu::DataType::BFLOAT16;
    CHECK_THROWS_WITH(check_tiles(oracle, oracle, bf16, Proximity{}), ContainsSubstring("not carried yet"));
    CHECK_THROWS_WITH(proximity_for(bf16), ContainsSubstring("no proximity bound"));
    // Values that do not fill the declared shape are structural, not a value to bound.
    TensorOperand empty = oracle;
    empty.values.clear();
    CHECK_THROWS_WITH(check_tiles(empty, empty, ElementFormat{}, Proximity{}), ContainsSubstring("declares 35 elements"));
}

TEST_CASE("tile check: a bound that would pass anything is refused", "[program][tile_check]") {
    const TensorOperand oracle = operand();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_WITH(check_tiles(oracle, oracle, ElementFormat{}, Proximity{nan, 1e-4}), ContainsSubstring("finite and non-negative"));
    CHECK_THROWS_WITH(check_tiles(oracle, oracle, ElementFormat{}, Proximity{1e-6, nan}), ContainsSubstring("finite and non-negative"));
    CHECK_THROWS_WITH(check_tiles(oracle, oracle, ElementFormat{}, Proximity{-1.0, 1e-4}), ContainsSubstring("finite and non-negative"));
    CHECK_THROWS_WITH(check_tiles(oracle, oracle, ElementFormat{},
                                  Proximity{std::numeric_limits<double>::infinity(), 0.0}),
                      ContainsSubstring("finite and non-negative"));
}

TEST_CASE("tile check: two programs' operand, at the format's default bound", "[program][tile_check]") {
    TileProgram a("p"), b("p");
    a.add_operand(operand());
    b.add_operand(operand());
    CHECK(check_tiles(a, b, "C").pass);
}
