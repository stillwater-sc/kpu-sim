#pragma once
// ============================================================================
// include/sw/kpu/cycle/xue_export.hpp
// The cycle core's counters as XUE EventStats (include/sw/xue/event_counter.hpp), for XUE's
// tooling: arrivals as XUE's count, completions, busy cycles as its total cycles, and the
// occupancy integral and peak. Kept out of counters.hpp so the core does not depend on XUE.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/counters.hpp>
#include <sw/xue/event_counter.hpp>

namespace sw::kpu::cycle {

inline void to_xue(const Counters& c, sw::xue::EventStats& x) {
    constexpr auto r = std::memory_order_relaxed;
    x.count.store(c.arrivals, r);
    x.completions.store(c.completions, r);
    x.total_cycles.store(c.busy, r);
    x.occupancy_integral.store(c.occupancy_integral, r);
    x.current_occupancy.store(static_cast<std::int64_t>(c.occupancy), r);
    x.max_occupancy.store(c.max_occupancy, r);
}

}  // namespace sw::kpu::cycle
