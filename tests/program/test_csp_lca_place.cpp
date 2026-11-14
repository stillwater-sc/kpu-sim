// ============================================================================
// tests/program/test_csp_lca_place.cpp
// L-CA executes L3 placement (docs/plans/l3-placement-at-lb.md step 6; #399): a credit is a
// slot. The Load, Writeback or Inherit that opens a residency takes ITS slot's credit -- it
// waits until the tile placed there before has left L3 -- and no L-CA process picks a slot of
// its own. Every slot holds the same tiles, in the same order, at L-B, L-T1 and L-CA.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>
#include <sw/kpu/timing/tag_cam.hpp>

#include "trace_fixtures.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using driver::ExecutionLevel;

namespace {

const platform::DeploymentSpec& s1() {
    static const platform::DeploymentSpec spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    return spec;
}

driver::CspLevelOutcome run(ExecutionLevel level, const std::string& text) {
    const lang::Program ast = lang::parse(text);
    auto o = driver::csp_run_level(level, ast, &s1().device(0), driver::csp_inputs(ast));
    REQUIRE_FALSE(o.skipped.has_value());
    return o;
}

std::string with_machine(std::string text, const std::string& options) {
    const auto at = text.find("machine flat(l3 = ");
    REQUIRE(at != std::string::npos);
    return text.insert(text.find(')', at), options);
}

using BySlot = std::map<std::uint32_t, std::vector<std::string>>;   // slot -> its tiles, in order

std::string show(const BySlot& by) {
    std::string out;
    for (const auto& [slot, tiles] : by) {
        out += std::to_string(slot) + ":";
        for (const auto& t : tiles) out += " " + t;
        out += "\n";
    }
    return out;
}

// The walker's placement: each residency in the order it opened.
BySlot walker(const CspProgram& p) {
    BySlot by;
    for (const Residency& r : p.residencies) by[r.place.slot].push_back(r.tile.to_string());
    return by;
}

// L-T1's: each slot's residencies in time.
BySlot lt1(const driver::CspLevelOutcome& o) {
    std::map<std::uint32_t, std::vector<const TransactionalInterpreter::Slot*>> v;
    for (const auto& s : o.slots) v[s.place.slot].push_back(&s);
    BySlot by;
    for (auto& [slot, ss] : v) {
        std::stable_sort(ss.begin(), ss.end(), [](auto* a, auto* b) { return a->t0 < b->t0; });
        for (auto* s : ss) by[slot].push_back(s->tile.to_string());
    }
    return by;
}

// L-CA's: each slot's tiles in the order the hardware claimed it.
BySlot lca(const driver::CspLevelOutcome& o) {
    BySlot by;
    for (const auto& [tile, slot] : o.l3_placement) by[slot].push_back(tile.to_string());
    return by;
}

const char* kStreamed = R"(csp 1.0
program streamed machine flat(l3 = 5) {
  tensor A[32,256] tile 32x32 in;
  tensor B[256,32] tile 32x32 in;
  tensor C[32,32] tile 32x32 out;
  acc C[0, 0] in fabric {
    for k in 0..8 {
      resident A[0, k], B[k, 0]@@;
      call gemm(A[0, k], B[k, 0]) +-> C[0, 0];
      release A[0, k], B[k, 0];
    }
  }
  store C[0, 0];
}
)";

}  // namespace

TEST_CASE("TagCAM: a placed slot holds one tile until its entry leaves", "[csp][placement][lca][tag_cam]") {
    using sw::kpu::timing::TagCAM;
    using sw::kpu::timing::TileID;
    TagCAM cam(4);
    TileID x, y;
    x.ti = 1;
    y.ti = 2;
    CHECK(cam.slot_free(2));
    REQUIRE(cam.claim_slot(2, x));
    CHECK_FALSE(cam.slot_free(2));
    CHECK_FALSE(cam.claim_slot(2, y));          // held by another tile
    CHECK_FALSE(cam.claim_slot(4, y));          // past the module's slots
    REQUIRE(cam.slot_owner(2).has_value());
    CHECK(*cam.slot_owner(2) == x);
    REQUIRE(cam.insert(x, 2, 0, 2));            // two references
    CHECK_FALSE(cam.invalidate(x));
    CHECK_FALSE(cam.slot_free(2));              // one reference left: still x's
    CHECK(cam.invalidate(x));
    CHECK(cam.slot_free(2));                    // the entry left: the slot's credit is back
    CHECK_FALSE(cam.claim_slot(2, y, 0));       // the slot's first turn was x's
    REQUIRE(cam.claim_slot(2, y, 1));
    REQUIRE(cam.claim_log().size() == 2);
    CHECK(cam.claim_log()[0].second == x);
    CHECK(cam.claim_log()[1].second == y);
    cam.unclaim_slot(2, y);                     // a transfer that did not start
    CHECK(cam.slot_free(2));
    CHECK(cam.slot_ready(2, 1));                // its turn is back
    CHECK(cam.claim_log().size() == 1);
}

TEST_CASE("L-CA executes the walker's placement, the same slots as L-T1 and L-B", "[csp][placement][lca]") {
    for (const char* opts : {"", ", place = lowest", ", banks = 4, place = bank-spread"}) {
        for (const char* init : {"none", "bias-row"}) {
            CAPTURE(opts, init);
            auto o = trace_fixtures::gen("matmul", 96, 32);
            o.init = init;
            o.l3 = gen::slots_needed(o);                // the init needs one more
            const std::string text = with_machine(gen::generate(o), opts);
            const CspProgram p = lang::compile(lang::parse(text));
            const auto t1 = run(ExecutionLevel::BlockSequential, text);
            const auto ca = run(ExecutionLevel::CycleAccurate, text);
            REQUIRE(ca.l3_placement.size() == p.residencies.size());
            CHECK(show(lt1(t1)) == show(walker(p)));
            CHECK(show(lca(ca)) == show(walker(p)));
        }
    }
    {
        // A streamed program, and the same one with its tiles pinned to two slots.
        std::string text = kStreamed;
        text.replace(text.find("@@"), 2, " @ slots(0..2)");
        const CspProgram p = lang::compile(lang::parse(text));
        const auto ca = run(ExecutionLevel::CycleAccurate, text);
        CHECK(lca(ca) == walker(p));
        CHECK(lca(ca) == lt1(run(ExecutionLevel::BlockSequential, text)));
    }
}

namespace {

// L-CA run on the executor directly, for its events and statistics.
struct Direct {
    sw::kpu::timing::CspDriver::Result r;
    sw::kpu::timing::ConcurrentTimingExecutor::Statistics stats;
    std::vector<sw::kpu::timing::TimingEvent> events;
};

Direct direct(const std::string& text) {
    const lang::Program ast = lang::parse(text);
    const auto cfg = sw::kpu::timing::csp_config_from(s1().device(0));
    REQUIRE(cfg.has_value());
    sw::kpu::timing::ConcurrentTimingExecutor exec(cfg->config);
    lang::ActionStream s(ast, lang::target_from(s1().device(0)));
    Direct d;
    d.r = sw::kpu::timing::CspDriver(exec, s, driver::csp_inputs(ast)).run();
    REQUIRE(d.r.completed);
    d.stats = exec.get_statistics();
    d.events = exec.events();
    return d;
}

// From the executor's events: a placed slot holds one tile at a time. A tile enters its slot when
// its Load lands in L3 (or its opening Writeback does) and leaves with its entry's CREDIT_RELEASED
// from a BlockMover (a Release, or a written residency's ejection). A Writeback's own
// CREDIT_RELEASED, right after its TILE_ARRIVED_L3, is the L2 credit it returns: not L3's.
std::size_t check_one_tile_per_slot(const Direct& d) {
    using sw::kpu::timing::EventType;
    std::map<std::uint32_t, std::string> held;
    std::size_t released = 0;
    for (std::size_t i = 0; i < d.events.size(); ++i) {
        const auto& e = d.events[i];
        const bool l2_return = i > 0 && d.events[i - 1].type == EventType::TILE_ARRIVED_L3 &&
                               d.events[i - 1].tile_id == e.tile_id &&
                               d.events[i - 1].component_name == e.component_name;
        const bool l3_arrival = e.type == EventType::TILE_ARRIVED_L3;
        const bool l3_release = e.type == EventType::CREDIT_RELEASED && !l2_return &&
                                e.component_name.find(":BM") != std::string::npos &&
                                held.count(e.slot_id) && held.at(e.slot_id) == e.tile_id.to_string();
        if (l3_arrival) {
            const auto h = held.find(e.slot_id);
            CAPTURE(e.cycle, e.slot_id, e.tile_id.to_string());
            if (h != held.end()) CHECK(h->second == e.tile_id.to_string());   // in place: its own slot
            held[e.slot_id] = e.tile_id.to_string();
        } else if (l3_release) {
            held.erase(e.slot_id);
            ++released;
        }
    }
    return released;
}

}  // namespace

TEST_CASE("L-CA: a slot reused early serializes on it", "[csp][placement][lca]") {
    // Pinned to slots 0 and 1, each k-step's Loads wait for the Releases of the step before,
    // though two other slots stand free. A pooled credit (any free slot) would not wait.
    std::string free_text = kStreamed, pinned_text = kStreamed;
    free_text.replace(free_text.find("@@"), 2, "");
    pinned_text.replace(pinned_text.find("@@"), 2, " @ slots(0..2)");
    const Direct free = direct(free_text);
    const Direct pinned = direct(pinned_text);
    for (const auto& [tile, slot] : pinned.r.l3_placement)
        if (tile.operand != "C") CHECK(slot < 2);
    CHECK(check_one_tile_per_slot(free) >= 16);       // every A and B tile left its slot
    CHECK(check_one_tile_per_slot(pinned) >= 16);
    // The pinned Loads wait for their slots: the dma process stalls on credit longer.
    CHECK(pinned.stats.dma_credit_stalls > free.stats.dma_credit_stalls);
    // Measured, not asserted: at L-CA the pinned program is not slower. Its Loads arrive one pair
    // at a time, each at the DMA's full bandwidth, where the free program's share it.
    std::printf("L-CA slot reuse: free makespan %llu (%llu dma credit-stall cycles), pinned %llu (%llu)\n",
                (unsigned long long)free.r.cycles, (unsigned long long)free.stats.dma_credit_stalls,
                (unsigned long long)pinned.r.cycles, (unsigned long long)pinned.stats.dma_credit_stalls);
}
