// ============================================================================
// include/sw/kpu/program/numeric.hpp
// The number systems an operand's elements are in, and the fabric's typed MAC
// (#394; docs/plans/typed-value-path.md).
//
// The arithmetic is stillwater-sc/universal's: each number system is a Universal type, named as
// universal_dtypes names it (the NumPy side of the same types), so a Python reference computes
// with the same names. The registry is compiled in (src/program/numeric.cpp, the one file that
// includes Universal): a name maps to the type's width and its encode/decode, and a gemm's
// (input, accumulator) pair maps to a MAC instantiated for those Universal types. A pair that is
// not registered is refused by name; adding one is one table entry.
//
// Storage (plan §3): an element is a 32-bit WORD. fp32's word is the float itself, so every
// fp32 program computes exactly what it did before. Any other number system's word is its
// Universal bit pattern, zero-extended -- int8 -1 is 0x000000FF -- and only the code that
// interprets values (the kernels, checks, synthesis, the L0 text, printing) decodes it. Code that
// moves tiles copies words and never looks.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::numeric {

using Word = std::uint32_t;

inline constexpr const char* kFp32 = "fp32";       // the default: a declaration without a type

// How values of a number system are compared and written: an integer is exact, a real is read
// through the proximity bound (tile_check).
enum class Kind : std::uint8_t { Real, Integer };

struct NumberSystem {
    std::string name;               // as universal_dtypes names it: "fp32", "int8", "int32", ...
    unsigned bits = 32;             // the encoding's width (<= 32: plan §3)
    Kind kind = Kind::Real;
    std::string universal;          // the Universal type, for messages: "integer<8>"
};

// The registered number systems, in a stable order. find: nullptr when `name` is not one.
const std::vector<NumberSystem>& systems();
const NumberSystem* find(const std::string& name);
// get: throws std::invalid_argument naming the registered systems.
const NumberSystem& get(const std::string& name);
std::string names();                                 // "fp32, int8, int32", for messages

// A value <-> its word in `type`. encode converts as Universal does: an integer type takes the
// value's integer part modulo 2^bits (two's complement), a real type rounds.
double decode(const std::string& type, Word w);
Word encode(const std::string& type, double v);

// The fabric's MAC on one tile: c[i][j] += sum_k a[i][k] * b[k][j], with a (m x k) and b (k x n)
// in the input type and c (m x n) in the accumulator's, every array row-major and contiguous.
// The product and the sum are the accumulator type's (int8 x int8 summed in int32, wrapping at
// 32 bits, as the KPU RTL's systolic_int8 does).
using Mac = void (*)(const Word* a, const Word* b, Word* c, std::uint32_t m, std::uint32_t n, std::uint32_t k);
// nullptr when the pair is not registered.
Mac mac(const std::string& in, const std::string& acc);
std::vector<std::pair<std::string, std::string>> mac_pairs();
std::string mac_names();                             // "fp32 -> fp32, int8 -> int32"

// fp32's word is the float's bits.
inline Word word_of(float f) { Word w; std::memcpy(&w, &f, sizeof w); return w; }
inline float float_of(Word w) { float f; std::memcpy(&f, &w, sizeof f); return f; }

}  // namespace sw::kpu::program::numeric
