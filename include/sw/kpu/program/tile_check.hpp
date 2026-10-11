// ============================================================================
// include/sw/kpu/program/tile_check.hpp
// Functional checks on TILE CONTENTS (docs/plans/retire-l0-derivations.md §3.8).
//
// A machine's functional behaviour has two parts. The sequence of tile moves through the
// resources is structure, and is checked exactly elsewhere (loads, stores, credit order). The
// bits of the result are checked here: floating point is not associative, and a data path
// legitimately reorders reductions and converts between formats, so bit equality is not the
// functional bar. Instead each tile's stored bits are read through an INTERPRETATION LAYER, in
// the format the program intends, and every element is asserted to lie within a PROXIMITY bound
// of the oracle's (L-B's) value for the same tile:
//
//     |actual - oracle| <= atol + rtol * |oracle|          (ADR 0001 D5, per element)
//
// Operands store float32 today, so float32 is the one format decoded. Any other format is
// refused by name until operands carry it -- never checked as though it were float32.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/data_types.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/value_tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program {

// How a tile's stored bits are meant: the element format, and its decoding.
struct ElementFormat {
    DataType type = DataType::FLOAT32;

    // The value a stored element means. Operands hold float32 elements, so the bits are read as
    // a float32; a format the operands do not carry yet is refused.
    double decode(const float& stored) const {
        if (type != DataType::FLOAT32)
            throw std::invalid_argument("tile_check: operands store float32; " + dtype_name(type) +
                                        " tiles are not carried yet, so they cannot be interpreted");
        return static_cast<double>(stored);         // the 32 bits, as IEEE 754 binary32
    }
};

// The proximity bound, per element: |actual - oracle| <= atol + rtol * |oracle|.
struct Proximity {
    double atol = kAtolFloat32;
    double rtol = kRtolMatmul;
};

// The format's default bound (ADR 0001 D5): float32 atol 1e-6, rtol 1e-4. A test states a looser
// one where its data path justifies it (a deep reduction, a narrowing conversion).
inline Proximity proximity_for(const ElementFormat& f) {
    if (f.type == DataType::FLOAT32) return Proximity{kAtolFloat32, kRtolMatmul};
    throw std::invalid_argument("tile_check: no proximity bound for " + dtype_name(f.type) + " yet");
}

// One tile out of bound.
struct TileMiss {
    TileCoord tile;
    std::size_t elements = 0;       // elements out of bound in this tile
    Dim row = 0, col = 0;           // the first, in operand coordinates
    double actual = 0.0, oracle = 0.0;
    double max_abs = 0.0;
};

struct TileCheck {
    bool pass = true;
    std::string operand;
    std::size_t tiles = 0, elements = 0;
    double max_abs = 0.0;           // worst |actual - oracle|, over every element
    double max_rel = 0.0;           // worst relative error, over elements with |oracle| >= atol
    std::vector<TileMiss> misses;   // the tiles out of bound, in tile order

    // One line, or the first misses by tile: what a failing CHECK prints.
    std::string summary(std::size_t show = 4) const {
        std::ostringstream o;
        o << operand << ": " << tiles << " tiles, " << elements << " elements, max abs " << max_abs << ", max rel "
          << max_rel;
        if (pass) return o.str() + ", within bound";
        o << "; " << misses.size() << " tile(s) out of bound";
        for (std::size_t i = 0; i < misses.size() && i < show; ++i) {
            const TileMiss& m = misses[i];
            o << "\n  " << m.tile.to_string() << ": " << m.elements << " element(s), first at (" << m.row << ","
              << m.col << ") got " << m.actual << " expected " << m.oracle << ", max abs " << m.max_abs;
        }
        return o.str();
    }
};

// Every tile of `operand`, actual against oracle, interpreted in `format`, within `bound`.
// Shapes and tilings must agree: a different grid is a structural failure, thrown, not a value
// to bound.
inline TileCheck check_tiles(const TensorOperand& actual, const TensorOperand& oracle,
                             const ElementFormat& format, const Proximity& bound) {
    if (actual.rows != oracle.rows || actual.cols != oracle.cols || actual.tile_rows != oracle.tile_rows ||
        actual.tile_cols != oracle.tile_cols || actual.values.size() != oracle.values.size())
        throw std::invalid_argument("tile_check: " + actual.name + " is " + std::to_string(actual.rows) + "x" +
                                    std::to_string(actual.cols) + " in " + std::to_string(actual.tile_rows) + "x" +
                                    std::to_string(actual.tile_cols) + " tiles; the oracle's is " +
                                    std::to_string(oracle.rows) + "x" + std::to_string(oracle.cols) + " in " +
                                    std::to_string(oracle.tile_rows) + "x" + std::to_string(oracle.tile_cols));
    const std::size_t declared = static_cast<std::size_t>(oracle.rows) * oracle.cols;
    if (actual.values.size() != declared || oracle.values.size() != declared)
        throw std::invalid_argument("tile_check: " + actual.name + " declares " + std::to_string(declared) +
                                    " elements; actual holds " + std::to_string(actual.values.size()) +
                                    ", the oracle " + std::to_string(oracle.values.size()));
    // A NaN or negative bound would pass anything (NaN compares false): refuse it.
    if (!std::isfinite(bound.atol) || !std::isfinite(bound.rtol) || bound.atol < 0.0 || bound.rtol < 0.0)
        throw std::invalid_argument("tile_check: the proximity bound must be finite and non-negative (atol " +
                                    std::to_string(bound.atol) + ", rtol " + std::to_string(bound.rtol) + ")");
    TileCheck out;
    out.operand = actual.name;
    for (Dim ti = 0; ti < oracle.n_tile_rows(); ++ti)
        for (Dim tj = 0; tj < oracle.n_tile_cols(); ++tj) {
            ++out.tiles;
            TileMiss miss;
            miss.tile = TileCoord{actual.name, ti, tj};
            for (Dim r = oracle.row_begin(ti); r < oracle.row_end(ti); ++r)
                for (Dim c = oracle.col_begin(tj); c < oracle.col_end(tj); ++c) {
                    ++out.elements;
                    const double a = format.decode(actual.at(r, c));
                    const double e = format.decode(oracle.at(r, c));
                    const double err = std::abs(a - e);
                    // Non-finite on either side is out of bound: with NaN, `err > bound` is false.
                    const bool out_of_bound =
                        !std::isfinite(a) || !std::isfinite(e) || err > bound.atol + bound.rtol * std::abs(e);
                    if (std::isfinite(err)) {
                        out.max_abs = std::max(out.max_abs, err);
                        if (std::abs(e) >= bound.atol) out.max_rel = std::max(out.max_rel, err / std::abs(e));
                    }
                    if (!out_of_bound) continue;
                    if (miss.elements++ == 0) {
                        miss.row = r;
                        miss.col = c;
                        miss.actual = a;
                        miss.oracle = e;
                    }
                    if (std::isfinite(err)) miss.max_abs = std::max(miss.max_abs, err);
                }
            if (miss.elements) {
                out.pass = false;
                out.misses.push_back(miss);
            }
        }
    return out;
}

// The same operand of two programs (a level's result against L-B's), float32 at its default bound.
inline TileCheck check_tiles(const TileProgram& actual, const TileProgram& oracle, const std::string& operand,
                             const ElementFormat& format = {}, std::optional<Proximity> bound = std::nullopt) {
    return check_tiles(actual.operand(operand), oracle.operand(operand), format,
                       bound ? *bound : proximity_for(format));
}

}  // namespace sw::kpu::program
