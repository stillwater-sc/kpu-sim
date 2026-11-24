#pragma once
// ============================================================================
// include/sw/kpu/cycle/time.hpp
// Time on the cycle core (docs/plans/cycle-based-kpu-prototype.md §3.2, decision 1).
//
// Global time is in picoseconds. Each clock DOMAIN has a period and a phase: its edges are at
// phase, phase + period, phase + 2 period, ... A component lives in one domain and counts that
// domain's cycles. A 1 GHz array has a period of 1000 ps, 250 MHz L2 banks 4000 ps, and a
// 3.2 GHz LPDDR5 controller 312.5 ps -- so periods that are not whole picoseconds are given as
// a rational number of picoseconds (num / den), and an edge's time is rounded down only when it
// is reported, never accumulated.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sw::kpu::cycle {

// floor(a * b / c), exact, without a 128-bit integer type (MSVC has none): the 128-bit product
// in two 64-bit halves, then long division. Raises std::overflow_error if the quotient does not
// fit in 64 bits.
inline std::uint64_t mul_div(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    if (c == 0) throw std::invalid_argument("cycle::mul_div: division by zero");
    const std::uint64_t a_lo = a & 0xFFFFFFFFULL, a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xFFFFFFFFULL, b_hi = b >> 32;
    const std::uint64_t p0 = a_lo * b_lo, p1 = a_lo * b_hi, p2 = a_hi * b_lo, p3 = a_hi * b_hi;
    const std::uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    const std::uint64_t lo = (p0 & 0xFFFFFFFFULL) | (mid << 32);
    const std::uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    if (hi >= c) throw std::overflow_error("cycle::mul_div: the result does not fit in 64 bits");
    // hi < c, so the quotient fits: shift the low half's bits through the remainder.
    std::uint64_t q = 0, r = hi;
    for (int i = 63; i >= 0; --i) {
        const bool carry = (r >> 63) != 0;
        r = (r << 1) | ((lo >> i) & 1ULL);
        if (carry || r >= c) {
            r -= c;
            q |= 1ULL << i;
        }
    }
    return q;
}

using Picoseconds = std::uint64_t;
using Cycle = std::uint64_t;        // a domain's cycle count

class Domain {
public:
    // A domain whose period is num / den picoseconds (den = 1: a whole number of picoseconds).
    Domain(std::string name, std::uint64_t period_num, std::uint64_t period_den = 1, Picoseconds phase = 0)
        : name_(std::move(name)), num_(period_num), den_(period_den), phase_(phase) {
        if (num_ == 0 || den_ == 0) throw std::invalid_argument("cycle::Domain " + name_ + ": the period must be positive");
        if (den_ > (std::uint64_t{1} << 62))
            throw std::invalid_argument("cycle::Domain " + name_ + ": the period's denominator is too large");
        whole_ = num_ / den_;
        frac_ = num_ % den_;
        reset();
    }
    static Domain from_ghz(std::string name, double ghz);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Cycle cycle() const { return cycle_; }
    // The time of edge n, in picoseconds (exact to the picosecond below).
    // Edge n at phase + floor(n * num / den), exact (mul_div): never a rounded period accumulated.
    [[nodiscard]] Picoseconds edge(Cycle n) const { return add(phase_, mul_div(n, num_, den_)); }
    // The next edge, kept as a whole part and a remainder that advance() steps exactly, so the
    // scheduler's hot path neither multiplies nor divides. Equal to edge(cycle()).
    [[nodiscard]] Picoseconds next_edge() const { return next_; }
    [[nodiscard]] double period_ps() const { return static_cast<double>(num_) / static_cast<double>(den_); }

    // Raises std::overflow_error, leaving the clock as it was, if the next edge is past 2^64 ps
    // (about 213 days), rather than wrapping to an earlier time.
    void advance() {
        std::uint64_t rem = rem_ + frac_;   // rem_, frac_ < den_ <= 2^62: no overflow
        std::uint64_t next = add(next_, whole_);
        if (rem >= den_) {
            rem -= den_;
            next = add(next, 1);
        }
        ++cycle_;
        next_ = next;
        rem_ = rem;
    }
    void reset() {
        cycle_ = 0;
        next_ = phase_;
        rem_ = 0;
    }

private:
    [[nodiscard]] std::uint64_t add(std::uint64_t a, std::uint64_t b) const {
        if (a > ~std::uint64_t{0} - b)
            throw std::overflow_error("cycle::Domain " + name_ + ": an edge past 2^64 ps");
        return a + b;
    }

    std::string name_;
    std::uint64_t num_, den_;
    Picoseconds phase_;
    std::uint64_t whole_ = 0, frac_ = 0;   // the period as whole_ + frac_ / den_
    Cycle cycle_ = 0;                      // the next edge to take
    Picoseconds next_ = 0;                 // its time, whole picoseconds ...
    std::uint64_t rem_ = 0;                // ... plus rem_ / den_
};

// A frequency in GHz as a period of 1000 / ghz ps, kept rational to a millionth of a GHz.
inline Domain Domain::from_ghz(std::string name, double ghz) {
    if (!(ghz > 0.0)) throw std::invalid_argument("cycle::Domain " + name + ": the frequency must be positive");
    const auto mhz_e3 = static_cast<std::uint64_t>(ghz * 1e6 + 0.5);   // GHz in millionths
    return Domain(std::move(name), 1000ULL * 1000000ULL, mhz_e3);
}

}  // namespace sw::kpu::cycle
