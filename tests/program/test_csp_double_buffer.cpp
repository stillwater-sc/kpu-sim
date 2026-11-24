// ============================================================================
// tests/program/test_csp_double_buffer.cpp
// Double buffering in L2 at L-CA (docs/plans/slot-credits-at-lca.md §6, #387 step 3c).
//
// The RTL's test: an 8x8x16 int8 matmul on a 4x4 tile, with one and two L2 buffers. kpu-sim runs it
// on kpu_soc_2x2 (the RTL-derived 4x4 tile, int8 -> int32) with l2 = 3 and l2 = 6: an output-
// stationary call holds A's copy, B's copy and its result, so B buffers take 3B slots.
//
// What this asserts, as it carries over (decided 2026-11-24):
//   - values: L-B's, bit for bit, with one buffer and with two;
//   - two buffers take the L2 slot waits away (the bubble report's "L2 slot credit") and overlap
//     more Moves with Feeds;
//   - the run is bound by its one BlockMover, not by the array (decided 2026-11-24, #458): with the
//     array timed as the RTL's COMPUTE commands the 16 calls compute for 64 cycles, and the
//     BlockMover is busy for over 70% of either run, so one buffer and two take within 3% of each
//     other. (Under the MAC-rate model compute took 96 cycles, and two buffers hid some of it.)
//   - with either, no Move lands in a slot while a Feed is still reading it (the slot invariant).
// What it does not, yet: the RTL's "one buffer never overlaps a Move with a Feed" -- kpu-sim keeps
// A's and B's credits separate (decision 2), so B's Move overlaps A's Feed by design -- and "two
// buffers take under 0.85x the cycles": here the run is bound by the BlockMover and DMA, with
// int8 still moved as 4 bytes (#415) and kpu-sim's default mover rates. That clause is #401's,
// and so is the makespan relation it implies.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/bubbles.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using driver::ExecutionLevel;

namespace {

// The RTL's double-buffering test: C[8,8] = A[8,16] . B[16,8], int8 in 4x4 tiles, `l2` slots.
std::string matmul(int l2) {
    return "csp 1.1\nprogram mm machine flat(l3 = 32, l2 = " + std::to_string(l2) + ") {\n"
           "  tensor A[8,16] int8 tile 4x4 in;\n  tensor B[16,8] int8 tile 4x4 in;\n"
           "  tensor C[8,8] int32 tile 4x4 out;\n"
           "  for j in 0..2 {\n    resident B[:, j];\n    for i in 0..2 {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

const platform::DeviceSpecification& tile4x4() {
    static const auto d = platform::read_spec_file("tests/program/deploy/kpu_soc_2x2.json").device(0);
    return d;
}

struct Run {
    driver::CspLevelOutcome lca;
    std::vector<sw::kpu::timing::TimingEvent> events;
};

Run run(int l2) {
    const lang::Program ast = lang::parse(matmul(l2));
    Run r;
    r.lca = driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &tile4x4(), driver::csp_inputs(ast), 256, true);
    REQUIRE_FALSE(r.lca.skipped);
    const auto cfg = sw::kpu::timing::csp_config_from(tile4x4());
    REQUIRE(cfg.has_value());
    sw::kpu::timing::ConcurrentTimingExecutor exec(cfg->config);
    lang::ActionStream s(ast, lang::target_from(tile4x4()));
    REQUIRE(sw::kpu::timing::CspDriver(exec, s, driver::csp_inputs(ast)).run().completed);
    r.events = exec.events();
    return r;
}

std::size_t move_feed_overlaps(const driver::CspLevelOutcome& o) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> moves, feeds;
    for (const auto& r : o.records) {
        if (r.kind == csp::Action::Kind::Move) moves.emplace_back(r.start, r.finish);
        if (r.kind == csp::Action::Kind::Feed) feeds.emplace_back(r.start, r.finish);
    }
    std::size_t n = 0;
    for (const auto& m : moves)
        for (const auto& f : feeds) n += m.first < f.second && f.first < m.second;
    return n;
}

std::uint64_t l2_slot_wait(const driver::CspLevelOutcome& o) {
    const auto b = csp::bubble_report(o.records, o.makespan, o.busy, o.lanes, "L-CA");
    const auto it = b.idle_by_cause.find(csp::BubbleReport::Cause::L2Slot);
    return it == b.idle_by_cause.end() ? 0 : it->second;
}

// From the executor's events: no Move lands in an L2 slot while a Feed is still reading it.
void no_move_into_a_slot_being_read(const std::vector<sw::kpu::timing::TimingEvent>& events) {
    using sw::kpu::timing::EventType;
    std::map<std::uint32_t, int> reading;      // slot -> Feeds in flight from it
    for (const auto& e : events) {
        CAPTURE(e.cycle, e.tile_id.to_string(), e.slot_id);
        if (e.type == EventType::STR_FEED_START) ++reading[e.slot_id];
        else if (e.type == EventType::STR_FEED_COMPLETE) --reading[e.slot_id];
        else if (e.type == EventType::TILE_ARRIVED_L2) CHECK(reading[e.slot_id] == 0);
    }
}

}  // namespace

TEST_CASE("Double buffering at L-CA: the RTL's 8x8x16 int8 matmul, one L2 buffer and two",
          "[csp][lca][l2][double-buffer]") {
    const Run one = run(3), two = run(6);
    // Values: L-B's (int8 and int32 exact, per tile), and bit for bit between the two runs. A typed
    // payload holds each element's bit pattern, so they compare as bytes, not as floats.
    const lang::Program ast = lang::parse(matmul(3));
    const auto lb = driver::csp_run_level(ExecutionLevel::Behavioral, ast, &tile4x4(), driver::csp_inputs(ast));
    for (const Run* r : {&one, &two}) {
        const OperandsCheck c = check_operands(r->lca.values, lb.values);
        INFO(c.summary());
        CHECK(c.pass);
    }
    const auto& c1 = one.lca.values.operand("C").values;
    const auto& c2 = two.lca.values.operand("C").values;
    REQUIRE(c1.size() == c2.size());
    CHECK(std::memcmp(c1.data(), c2.data(), c1.size() * sizeof(c1[0])) == 0);
    // Two buffers: the L2 slot waits nearly gone, more Moves overlapping Feeds, and faster.
    const std::uint64_t wait1 = l2_slot_wait(one.lca), wait2 = l2_slot_wait(two.lca);
    const std::size_t ov1 = move_feed_overlaps(one.lca), ov2 = move_feed_overlaps(two.lca);
    std::printf("double buffering at L-CA (kpu_soc_2x2, 8x8x16 int8): 1 buffer %llu cycles (L2 slot wait %llu, "
                "%zu Move/Feed overlaps); 2 buffers %llu (wait %llu, %zu overlaps): %.3fx\n",
                (unsigned long long)one.lca.makespan, (unsigned long long)wait1, ov1,
                (unsigned long long)two.lca.makespan, (unsigned long long)wait2, ov2,
                double(two.lca.makespan) / double(one.lca.makespan));
    CHECK(wait1 > 0);
    CHECK(wait2 * 4 < wait1);
    CHECK(ov2 > ov1);
    // The BlockMover bound: its busy cycles are over 70% of each run, the array's under 10%, and the
    // two makespans are within 3% of each other.
    for (const Run* r : {&one, &two}) {
        const std::uint64_t bm = r->lca.busy.count("bm") ? r->lca.busy.at("bm") : 0;
        const std::uint64_t cf = r->lca.busy.count("cf") ? r->lca.busy.at("cf") : 0;
        CAPTURE(r->lca.makespan, bm, cf);
        CHECK(bm * 10 > r->lca.makespan * 7);
        CHECK(cf * 10 < r->lca.makespan);
    }
    const std::uint64_t lo = std::min(one.lca.makespan, two.lca.makespan);
    const std::uint64_t hi = std::max(one.lca.makespan, two.lca.makespan);
    CHECK((hi - lo) * 100 <= lo * 3);
    // Either way, a slot is never written while it is read.
    no_move_into_a_slot_being_read(one.events);
    no_move_into_a_slot_being_read(two.events);
}
