// ============================================================================
// tests/timing/test_tile_tracker.cpp
// TileTracker horizontal L3 | L2 | L1/array occupancy log (issue #165).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/tile_tracker.hpp>

#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using sw::kpu::isa::MatrixID;

namespace {

ConcurrentTimingExecutor::Config exec_config() {
    ConcurrentTimingExecutor::Config c;
    c.max_cycles = 200000;
    return c;
}

TileDescriptor scalar_tile(MatrixID m, Size ti) {
    TileDescriptor t;
    t.tile_id = {m, ti, 0, 0};
    t.height = 1; t.width = 1; t.element_size = 4; t.size_bytes = 4;
    t.dram_address = 0x100000 + ti * 0x1000;
    return t;
}

// Run one A tile through load/move/feed, tracking every occupancy change.
std::string run_and_track(TileTracker& tracker) {
    ConcurrentTimingExecutor exec(exec_config());
    auto a = scalar_tile(MatrixID::A, 0);
    exec.set_tile_payload(a.tile_id, TilePayload{1, 1, {7.0f}});
    exec.schedule_load(a);
    exec.schedule_move(a);
    exec.schedule_feed(a);

    tracker.observe(exec);
    while (!exec.is_complete() && exec.current_cycle() < exec.config().max_cycles) {
        exec.step();
        tracker.observe(exec);
    }
    return tracker.log();
}

// Split a band line "cyc | L3 | L2 | L1" into its column segments.
std::vector<std::string> columns(const std::string& line) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string seg;
    while (std::getline(ss, seg, '|')) {
        size_t a = seg.find_first_not_of(" ");
        size_t b = seg.find_last_not_of(" ");
        cols.push_back(a == std::string::npos ? "" : seg.substr(a, b - a + 1));
    }
    return cols;
}

} // namespace

TEST_CASE("TileTracker renders a horizontal L3|L2|L1 progression",
          "[timing][tracker]") {
    TileTracker tracker;
    const std::string log = run_and_track(tracker);

    // Header names the columns left-to-right: L3, then L2, then L1/array
    const auto hpos_l3 = log.find("L3 buffers");
    const auto hpos_l2 = log.find("L2 banks");
    const auto hpos_l1 = log.find("L1 / array");
    REQUIRE(hpos_l3 != std::string::npos);
    REQUIRE(hpos_l3 < hpos_l2);      // L3 left of L2
    REQUIRE(hpos_l2 < hpos_l1);      // L2 left of L1/array

    // The tile appears in each column at some snapshot, in flow order.
    // Parse bands (skip header + rule) and record when A[0,0,0] shows in
    // the L3, L2, and L1/array segments.
    bool in_l3 = false, in_l2 = false, in_l1 = false;
    std::istringstream ls(log);
    std::string line;
    while (std::getline(ls, line)) {
        if (line.find('|') == std::string::npos) continue;
        if (line.find("L3 buffers") != std::string::npos) continue;  // header
        auto cols = columns(line);
        if (cols.size() < 4) continue;  // cyc, L3, L2, L1
        if (cols[1].find("A[0,0,0]") != std::string::npos) in_l3 = true;
        if (cols[2].find("A[0,0,0]") != std::string::npos) in_l2 = true;
        if (cols[3].find("A[0,0,0]") != std::string::npos) in_l1 = true;
    }
    REQUIRE(in_l3);
    REQUIRE(in_l2);
    REQUIRE(in_l1);
}

TEST_CASE("TileTracker shows tile content and marks the array",
          "[timing][tracker]") {
    TileTracker tracker;
    const std::string log = run_and_track(tracker);

    // Value summary for the scalar payload (7) appears in a cell
    REQUIRE(log.find("A[0,0,0]=(7)") != std::string::npos);
    // COMPUTE (array) residency is marked with '*'
    REQUIRE(log.find("A[0,0,0]=(7)*") != std::string::npos);
}

TEST_CASE("TileTracker dedupes on occupancy change and is deterministic",
          "[timing][tracker]") {
    // observe() appends only when occupancy changed
    TileTracker t;
    ConcurrentTimingExecutor exec(exec_config());
    auto a = scalar_tile(MatrixID::A, 0);
    exec.set_tile_payload(a.tile_id, TilePayload{1, 1, {1.0f}});
    exec.schedule_load(a); exec.schedule_move(a); exec.schedule_feed(a);
    REQUIRE(t.observe(exec));            // first observation appends
    REQUIRE_FALSE(t.observe(exec));      // unchanged -> no band

    // Deterministic: two independent runs produce identical logs
    TileTracker t1, t2;
    REQUIRE(run_and_track(t1) == run_and_track(t2));
}
