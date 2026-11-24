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

using Picoseconds = std::uint64_t;
using Cycle = std::uint64_t;        // a domain's cycle count

class Domain {
public:
    // A domain whose period is num / den picoseconds (den = 1: a whole number of picoseconds).
    Domain(std::string name, std::uint64_t period_num, std::uint64_t period_den = 1, Picoseconds phase = 0)
        : name_(std::move(name)), num_(period_num), den_(period_den), phase_(phase) {
        if (num_ == 0 || den_ == 0) throw std::invalid_argument("cycle::Domain " + name_ + ": the period must be positive");
    }
    static Domain from_ghz(std::string name, double ghz);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] Cycle cycle() const { return cycle_; }
    // The time of edge n, in picoseconds (exact to the picosecond below).
    // n * num / den without forming n * num, which overflows for long runs of a rational period.
    [[nodiscard]] Picoseconds edge(Cycle n) const {
        return phase_ + (n / den_) * num_ + ((n % den_) * num_) / den_;
    }
    [[nodiscard]] Picoseconds next_edge() const { return edge(cycle_); }
    [[nodiscard]] double period_ps() const { return static_cast<double>(num_) / static_cast<double>(den_); }

    void advance() { ++cycle_; }
    void reset() { cycle_ = 0; }

private:
    std::string name_;
    std::uint64_t num_, den_;
    Picoseconds phase_;
    Cycle cycle_ = 0;               // the next edge to take
};

// A frequency in GHz as a period of 1000 / ghz ps, kept rational to a millionth of a GHz.
inline Domain Domain::from_ghz(std::string name, double ghz) {
    if (!(ghz > 0.0)) throw std::invalid_argument("cycle::Domain " + name + ": the frequency must be positive");
    const auto mhz_e3 = static_cast<std::uint64_t>(ghz * 1e6 + 0.5);   // GHz in millionths
    return Domain(std::move(name), 1000ULL * 1000000ULL, mhz_e3);
}

}  // namespace sw::kpu::cycle
