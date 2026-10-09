// ============================================================================
// tests/timing/test_schedule_dispatcher.cpp
// The schedule-paced dispatcher (docs/plans/system-schedule-debugger.md §3.3, step 3): every
// operation is released against the oldest unfinished compute, at most P computes ahead. An
// unlimited P reproduces releasing everything up front, exactly; a small P releases no load
// early and shortens the time an L3 credit is held for a tile not yet arrived.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_dispatcher.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;
using sw::kpu::program::platform::read_spec_file;

namespace {

ConcurrentTimingExecutor::Config s1() {
    return csp_config_from(read_spec_file("tests/program/deploy/kpu_s1.json").device(0))->config;
}

ScheduleResult matmul(Size size, Size T, MatMulScheduleGenerator::Strategy s =
                                             MatMulScheduleGenerator::Strategy::INTERLEAVED_AB) {
    MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = size;
    g.Ti = g.Tj = g.Tk = T;
    g.strategy = s;
    g.l3_buffer_count = 128;
    g.a_base = 0x100000;
    g.b_base = 0x400000;
    g.c_base = 0x700000;
    auto r = MatMulScheduleGenerator(g).generate();
    REQUIRE(r.valid);
    return r;
}

struct Run {
    ExecutionResult result;
    std::vector<TimingEvent> events;
    std::vector<ScheduleDispatcher::Release> releases;
    std::size_t forced = 0;
};

Run run(const ScheduleResult& s, std::optional<std::size_t> P) {
    ConcurrentTimingExecutor exec(s1());
    ScheduleExecutor::Config c;
    c.prefetch_depth = P;
    ScheduleExecutor x(exec, c);
    Run out;
    out.result = x.execute(s);
    out.events = exec.events();
    out.releases = x.releases();
    out.forced = x.forced_releases();
    return out;
}

// Loads that went to DRAM (the rest found their tile already in L3: the tag-CAM reuse path).
std::size_t dram_loads(const std::vector<TimingEvent>& ev) {
    return static_cast<std::size_t>(std::count_if(ev.begin(), ev.end(), [](const TimingEvent& e) {
        return e.type == EventType::DMA_LOAD_COMPLETE;
    }));
}

// Mean cycles an L3 credit is held for a tile not yet arrived: from the load's credit to the
// tile's arrival in L3, matched per tile in order.
double mean_credit_idle(const std::vector<TimingEvent>& ev) {
    std::unordered_map<TileID, std::deque<Cycle>, TileIDHash> credit;
    double total = 0;
    std::size_t n = 0;
    for (const auto& e : ev) {
        if (e.type == EventType::CREDIT_ACQUIRED && e.component_name.rfind("DMA", 0) == 0)
            credit[e.tile_id].push_back(e.cycle);
        if (e.type == EventType::TILE_ARRIVED_L3) {
            auto& q = credit[e.tile_id];
            if (q.empty()) continue;            // a tag-CAM hit: no credit was taken
            total += static_cast<double>(e.cycle - q.front());
            q.pop_front();
            ++n;
        }
    }
    return n ? total / static_cast<double>(n) : 0.0;
}

}  // namespace

TEST_CASE("Dispatcher: every operation serves the compute that consumes or produced its tile",
          "[timing][schedule][dispatcher]") {
    const auto s = matmul(64, 16);
    ScheduleDispatcher d(s.operations, 1);
    REQUIRE(d.computes() == 16);
    std::vector<std::size_t> compute_op;
    for (std::size_t i = 0; i < s.operations.size(); ++i)
        if (s.operations[i].type == ScheduleOpType::COMPUTE) compute_op.push_back(i);
    for (std::size_t i = 0; i < s.operations.size(); ++i) {
        const auto& op = s.operations[i];
        const std::size_t c = d.serves(i);
        REQUIRE(c < compute_op.size());
        const auto& comp = s.operations[compute_op[c]];
        CAPTURE(i, c);
        switch (op.type) {
            case ScheduleOpType::LOAD:
            case ScheduleOpType::MOVE:
            case ScheduleOpType::FEED:
                CHECK(compute_op[c] > i);
                CHECK(std::find(comp.dependency_tiles.begin(), comp.dependency_tiles.end(),
                                op.tile.tile_id) != comp.dependency_tiles.end());
                break;
            case ScheduleOpType::COMPUTE:
                CHECK(compute_op[c] == i);
                break;
            default:
                CHECK(compute_op[c] < i);
                CHECK(comp.tile.tile_id == op.tile.tile_id);
                break;
        }
    }
}

TEST_CASE("Dispatcher: an unlimited P reproduces enqueueing everything up front, event for event",
          "[timing][schedule][dispatcher]") {
    const auto s = matmul(128, 32);
    const Run upfront = run(s, std::nullopt);
    const Run paced = run(s, ScheduleDispatcher::kUnlimited);
    REQUIRE(upfront.result.success);
    REQUIRE(paced.result.success);
    CHECK(upfront.releases.empty());
    CHECK(paced.releases.size() == s.operations.size());
    CHECK(paced.result.total_cycles == upfront.result.total_cycles);
    REQUIRE(paced.events.size() == upfront.events.size());
    for (std::size_t i = 0; i < paced.events.size(); ++i) {
        const auto& a = upfront.events[i];
        const auto& b = paced.events[i];
        CAPTURE(i);
        CHECK((a.type == b.type && a.cycle == b.cycle && a.tile_id == b.tile_id &&
               a.component_id == b.component_id && a.duration == b.duration));
    }
}

TEST_CASE("Dispatcher: at P = 1 no operation is released more than one compute ahead",
          "[timing][schedule][dispatcher]") {
    const auto s = matmul(256, 32);
    const Run r = run(s, 1);
    REQUIRE(r.result.success);
    REQUIRE(r.releases.size() == s.operations.size());
    CHECK(r.forced == 0);
    ScheduleDispatcher d(s.operations, 1);
    Cycle last = 0;
    for (const auto& rel : r.releases) {
        CAPTURE(rel.op, rel.cycle, rel.cursor);
        CHECK(d.serves(rel.op) <= rel.cursor + 1);
        CHECK(rel.cycle >= last);                       // released in order
        last = rel.cycle;
    }
    // Pacing holds operations back: not everything is released at cycle 0.
    CHECK(r.releases.back().cycle > 0);
}

TEST_CASE("Dispatcher: a smaller P holds L3 credits for less time, and P = 0 still finishes",
          "[timing][schedule][dispatcher]") {
    const auto s = matmul(256, 32);
    const Run all = run(s, ScheduleDispatcher::kUnlimited);
    const Run two = run(s, 2);
    const Run one = run(s, 1);
    REQUIRE((all.result.success && two.result.success && one.result.success));
    const double idle_all = mean_credit_idle(all.events), idle_two = mean_credit_idle(two.events),
                 idle_one = mean_credit_idle(one.events);
    CAPTURE(idle_all, idle_two, idle_one, all.result.total_cycles, two.result.total_cycles,
            one.result.total_cycles);
    std::printf("dispatcher 256^3/32^3 on S1: P=inf %llu cycles, credit idle %.0f, %zu DRAM loads; "
                "P=2 %llu, %.0f, %zu; P=1 %llu, %.0f, %zu\n",
                static_cast<unsigned long long>(all.result.total_cycles), idle_all, dram_loads(all.events),
                static_cast<unsigned long long>(two.result.total_cycles), idle_two, dram_loads(two.events),
                static_cast<unsigned long long>(one.result.total_cycles), idle_one, dram_loads(one.events));
    CHECK(idle_one < idle_all);
    CHECK(idle_two <= idle_all);
    // P = 0: a compute's inputs are released only once it is the oldest unfinished one. The
    // schedule's order puts no later compute's operation ahead of it, so nothing is forced.
    const Run zero = run(s, 0);
    REQUIRE(zero.result.success);
    CHECK(zero.forced == 0);
    CHECK(zero.result.total_cycles >= one.result.total_cycles);
    std::printf("dispatcher 256^3/32^3 on S1: P=0 %llu cycles, credit idle %.0f\n",
                static_cast<unsigned long long>(zero.result.total_cycles), mean_credit_idle(zero.events));
}

TEST_CASE("Dispatcher: a schedule that prefetches deeper than P is forced through, not wedged",
          "[timing][schedule][dispatcher]") {
    // Release is in schedule order. Move a load for the SECOND compute ahead of the first
    // compute's operations: at P = 0 it is not eligible (it serves compute 1, the cursor is 0),
    // and holding it back would hold back compute 0 forever. The progress rule releases it, and
    // says so.
    auto s = matmul(64, 32);
    ScheduleDispatcher probe(s.operations, 0);
    // A load that serves compute 1 and whose tile compute 0 does not use (a B tile of the next
    // column): moved to the front, it still serves compute 1.
    const auto& first = *std::find_if(s.operations.begin(), s.operations.end(),
                                      [](const ScheduleOperation& op) { return op.type == ScheduleOpType::COMPUTE; });
    std::size_t moved = s.operations.size();
    for (std::size_t i = 0; i < s.operations.size(); ++i) {
        const auto& op = s.operations[i];
        if (op.type == ScheduleOpType::LOAD && probe.serves(i) == 1 &&
            std::find(first.dependency_tiles.begin(), first.dependency_tiles.end(), op.tile.tile_id) ==
                first.dependency_tiles.end()) {
            moved = i;
            break;
        }
    }
    REQUIRE(moved < s.operations.size());
    const ScheduleOperation early = s.operations[moved];
    s.operations.erase(s.operations.begin() + static_cast<std::ptrdiff_t>(moved));
    s.operations.insert(s.operations.begin(), early);

    const Run r = run(s, 0);
    REQUIRE(r.result.success);
    CHECK(r.releases.size() == s.operations.size());
    CHECK(r.forced >= 1);
    CHECK(r.releases.front().forced);                   // the early load went first, forced
    // With a lookahead of one compute it is simply eligible.
    const Run one = run(s, 1);
    REQUIRE(one.result.success);
    CHECK(one.forced == 0);
}

TEST_CASE("Dispatcher: P against reuse -- a tile reloads when its next consumer is beyond P",
          "[timing][schedule][dispatcher][characterization]") {
    // 256^3 with 32^3 tiles on S1: 1,024 LOAD operations of 128 distinct tiles. With every
    // operation released up front, all 128 fit S1's 128-slot L3 and every repeat load is a
    // tag-CAM hit -- the schedule's reuse is accidental, and pacing exposes it: a tile's credit
    // returns once its RELEASED consumers finish, so a consumer more than P computes later
    // reloads it from DRAM.
    const auto s = matmul(256, 32);
    std::size_t loads = 0;
    for (const auto& op : s.operations) loads += op.type == ScheduleOpType::LOAD;
    CHECK(loads == 1024);
    std::map<std::size_t, std::size_t> dram_at;
    for (std::size_t P : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8},
                          std::size_t{16}, ScheduleDispatcher::kUnlimited}) {
        const Run r = run(s, P);
        REQUIRE(r.result.success);
        const std::size_t dram = dram_loads(r.events);
        std::printf("dispatcher sweep S1 256^3/32^3: P=%-6s %7llu cycles, %4zu DRAM loads, credit idle %.0f\n",
                    P == ScheduleDispatcher::kUnlimited ? "inf" : std::to_string(P).c_str(),
                    static_cast<unsigned long long>(r.result.total_cycles), dram, mean_credit_idle(r.events));
        CHECK(dram >= 128);
        dram_at[P] = dram;
    }
    // Which repeat loads still find their tile depends on timing, so the count is not strictly
    // monotone in P (2 can load a few more than 1); the trend is: each tile once with everything
    // released, and far fewer loads with a lookahead long enough to span B's reuse distance (8
    // computes in this order) than with none.
    CHECK(dram_at.at(ScheduleDispatcher::kUnlimited) == 128);
    CHECK(dram_at.at(16) < dram_at.at(1));
    CHECK(dram_at.at(1) < dram_at.at(0));
    CHECK(dram_at.at(0) == loads);                      // no lookahead: no reuse at all
}
