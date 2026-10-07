// ============================================================================
// include/sw/kpu/program/value_tolerance.hpp
// The ADR 0001 value comparator (D5; §7 answer 5): how a level's results are held to the L0
// TileProgramReference.
//
//   transactional (L-T1)  vs L0: bit-exact, no tolerance
//   cycle-accurate (CSP)  vs L0: elementwise  |actual - reference| <= atol + rtol * |reference|
//
// atol = 1e-6 for float32. rtol = 1e-4 for the MLP/matmul oracle and 5e-3 for the composed CNN
// references. A pure relative error is undefined at a zero reference and unstable near zero,
// so the bar is mixed. Both max errors are reported, and the relative figure skips every
// element whose reference magnitude is below atol.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program {

inline constexpr double kAtolFloat32 = 1e-6;
inline constexpr double kRtolMatmul = 1e-4;
inline constexpr double kRtolComposedCnn = 5e-3;

struct ToleranceReport {
    bool pass = true;               // every element within atol + rtol * |reference|
    bool bit_identical = true;      // every element the same bits
    double max_abs = 0.0;           // max |actual - reference|
    double max_rel = 0.0;           // max |actual - reference| / |reference|, |reference| >= atol
    std::size_t failures = 0;       // elements outside the bar
    std::size_t first_failure = 0;  // index of the first, when failures > 0
};

inline ToleranceReport compare_within(const std::vector<float>& actual,
                                      const std::vector<float>& reference,
                                      double atol, double rtol) {
    if (actual.size() != reference.size())
        throw std::invalid_argument("compare_within: " + std::to_string(actual.size()) +
                                    " values against a reference of " +
                                    std::to_string(reference.size()));
    ToleranceReport r;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], ref = reference[i];
        if (std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0) r.bit_identical = false;
        const double err = std::abs(a - ref);
        // A non-finite reference is a failure too: with NaN, `err > bound` is false, and the
        // element would otherwise pass unseen.
        if (!std::isfinite(a) || !std::isfinite(ref) || err > atol + rtol * std::abs(ref)) {
            if (r.failures++ == 0) r.first_failure = i;
            r.pass = false;
        }
        if (std::isfinite(err)) {
            if (err > r.max_abs) r.max_abs = err;
            if (std::abs(ref) >= atol && err / std::abs(ref) > r.max_rel)
                r.max_rel = err / std::abs(ref);
        }
    }
    return r;
}

} // namespace sw::kpu::program
