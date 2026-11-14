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

// ---- L2 staging (#399's L2 part) -------------------------------------------------------------

namespace {

// Row panels of A outer; staged, A's panel is copied into L2 once for the loop over B's panels
// (test_csp_l2_staging.cpp's program).
std::string staged_matmul(bool staged) {
    std::string s = std::string("csp 1.0\nprogram mm machine flat(l3 = 9") + (staged ? ", l2 = 4" : "") + ") {\n"
                    "  tensor A[64,64] tile 16x16 in;\n  tensor B[64,64] tile 16x16 in;\n  tensor C[64,64] tile 16x16 out;\n"
                    "  for i in 0..4 {\n"
                    "    resident A[i, :];\n";
    if (staged) s += "    stage A[i, :] in l2 {\n";
    s += "    for j in 0..4 {\n"
         "      resident B[:, j];\n"
         "      acc C[i, j] in fabric {\n"
         "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
         "      }\n"
         "      store C[i, j] late;\n"
         "      release B[:, j];\n"
         "    }\n";
    if (staged) s += "    }\n";
    s += "    release A[i, :];\n  }\n}\n";
    return s;
}

}  // namespace

TEST_CASE("L-CA executes the walker's L2 staging, the same slots as L-T1", "[csp][placement][lca][l2]") {
    const std::string text = staged_matmul(true);
    const CspProgram p = lang::compile(lang::parse(text));
    BySlot w;
    for (const Staging& st : p.stagings) w[st.place.slot].push_back(st.tile.to_string());
    const auto t1 = run(ExecutionLevel::BlockSequential, text);
    std::map<std::uint32_t, std::vector<const TransactionalInterpreter::L2Slot*>> v;
    for (const auto& sl : t1.l2_slots) v[sl.place.slot].push_back(&sl);
    BySlot lt1_l2;
    for (auto& [slot, ss] : v) {
        std::stable_sort(ss.begin(), ss.end(), [](auto* a, auto* b) { return a->t0 < b->t0; });
        for (auto* sl : ss) lt1_l2[slot].push_back(sl->tile.to_string());
    }
    const auto ca = run(ExecutionLevel::CycleAccurate, text);
    REQUIRE(ca.l2_placement.size() == p.stagings.size());
    BySlot lca_l2;
    for (const auto& [tile, slot] : ca.l2_placement) lca_l2[slot].push_back(tile.to_string());
    CHECK(show(lt1_l2) == show(w));
    CHECK(show(lca_l2) == show(w));
    // The L3 placement is still the walker's.
    CHECK(lca(ca) == walker(p));
}

TEST_CASE("L-CA: a staged copy is moved once and holds its L2 slot until its L2Release", "[csp][placement][lca][l2]") {
    using sw::kpu::timing::EventType;
    const Direct staged = direct(staged_matmul(true));
    const Direct plain = direct(staged_matmul(false));
    auto moves = [](const Direct& d) {
        std::size_t n = 0;
        for (const auto& e : d.events) n += e.type == EventType::TILE_ARRIVED_L2;
        return n;
    };
    // Unstaged, the program moves each call's A and B tiles (64 + 64); staged, A's 16 tiles move
    // once.
    CHECK(moves(plain) == 128);
    CHECK(moves(staged) == 80);
    CHECK(staged.r.staged_feeds == 64);
    // From the events: each placed L2 slot (0-3) holds one staged copy at a time, from its
    // Move's arrival to its L2Release; every staged Feed reads its copy's slot; an unstaged
    // copy (a B tile, a drained C) never takes a placed slot.
    std::map<std::uint32_t, std::string> held;
    std::map<std::string, std::uint32_t> slot_of;
    std::size_t released = 0, fed = 0;
    for (const auto& e : staged.events) {
        const std::string t = e.tile_id.to_string();
        const bool is_a = e.tile_id.matrix == sw::kpu::isa::MatrixID::A;
        CAPTURE(e.cycle, t, e.slot_id);
        if (e.type == EventType::TILE_ARRIVED_L2 || e.type == EventType::TILE_DRAINED) {
            CHECK((e.slot_id < 4) == is_a);
            if (is_a) {
                CHECK_FALSE(held.count(e.slot_id));
                held[e.slot_id] = t;
                slot_of[t] = e.slot_id;
            }
        } else if (e.type == EventType::STR_FEED_START && is_a) {
            REQUIRE(slot_of.count(t));
            CHECK(e.slot_id == slot_of.at(t));
            CHECK(held.count(e.slot_id));
            ++fed;
        } else if (e.type == EventType::CREDIT_RELEASED && e.component_name == "L2:release") {
            REQUIRE(held.count(e.slot_id));
            CHECK(held.at(e.slot_id) == t);
            held.erase(e.slot_id);
            ++released;
        }
    }
    CHECK(released == 16);
    CHECK(fed == 64);
    CHECK(held.empty());
    // Staged, the mover transfers each A tile once and each B tile per call: 16 + 64.
    auto transfers = [](const Direct& d) {
        std::size_t n = 0;
        for (const auto& e : d.events) n += e.type == EventType::BM_MOVE_COMPLETE;
        return n;
    };
    CHECK(transfers(staged) == 80);
    // Measured, not asserted: unstaged, the executor's dedup (process_dedup_moves) completes 48 of
    // A's Moves on the copy still in L2, without a transfer -- reuse the program did not decide --
    // so on this program staging leaves L-CA's timing as it was.
    std::printf("L-CA L2 staging: unstaged makespan %llu (bm busy %.0f, dma busy %.0f, cf busy %llu), "
                "staged %llu (bm busy %.0f, dma busy %.0f, cf busy %llu)\n",
                (unsigned long long)plain.r.cycles, plain.stats.bm_busy_cycles, plain.stats.dma_busy_cycles,
                (unsigned long long)plain.r.cf_busy, (unsigned long long)staged.r.cycles,
                staged.stats.bm_busy_cycles, staged.stats.dma_busy_cycles, (unsigned long long)staged.r.cf_busy);
}

// ---- L2 staging on the executor: orderings a program may produce (#419 review) -------------

namespace {

struct StagingRig {
    sw::kpu::timing::ConcurrentTimingExecutor exec;
    sw::kpu::timing::TileDescriptor a;          // one A tile, resident in L3

    StagingRig() : exec(sw::kpu::timing::csp_config_from(s1().device(0))->config) {
        using namespace sw::kpu::timing;
        exec.reset();
        exec.l2_tag_cam().set_placed_slots(4);
        a.tile_id.matrix = sw::kpu::isa::MatrixID::A;
        a.ordered = true;
        a.height = a.width = 16;
        a.element_size = 4;
        a.size_bytes = 16 * 16 * 4;
        a.matrix_base_address = a.dram_address = 0x100000;
        TilePayload p;
        p.rows = p.cols = 16;
        p.values.assign(256, 1.0f);
        exec.seed_l3(a, p);
        a.l3_held = true;
    }
    sw::kpu::timing::TileDescriptor staged(std::uint32_t slot, std::uint32_t turn) const {
        auto d = a;
        d.l2_slot = static_cast<std::int32_t>(slot);
        d.l2_slot_turn = turn;
        return d;
    }
    std::vector<sw::kpu::timing::TimingEvent> run() {
        for (int c = 0; c < 200000 && !exec.is_complete(); ++c) exec.step();
        REQUIRE(exec.is_complete());
        return exec.events();
    }
};

}  // namespace

TEST_CASE("L-CA L2: an ordinary Move after a staging Move does not open the staged Feed early", "[csp][lca][l2]") {
    using sw::kpu::timing::EventType;
    StagingRig g;
    // M1 (ordinary), F1, M2 (staged, slot 0), F2 (staged), M3 (ordinary), F3, the L2Release.
    g.exec.schedule_move(g.a);
    g.exec.schedule_feed(g.a);
    g.exec.schedule_move(g.staged(0, 0));
    auto f2 = g.a;
    f2.l2_held = true;
    g.exec.schedule_feed(f2);
    g.exec.schedule_move(g.a);
    g.exec.schedule_feed(g.a);
    g.exec.schedule_l2_release(g.staged(0, 0));
    const auto events = g.run();
    std::vector<std::uint32_t> feeds;           // each Feed's L2 slot, in start order
    for (const auto& e : events)
        if (e.type == EventType::STR_FEED_START) feeds.push_back(e.slot_id);
    REQUIRE(feeds.size() == 3);
    CHECK(feeds[0] >= 4);                       // F1: M1's transient copy
    CHECK(feeds[1] == 0);                       // F2: the staged copy, in its placed slot
}

TEST_CASE("L-CA L2: an L2Release retires its own staging's copy, not another of the tile", "[csp][lca][l2]") {
    using sw::kpu::timing::EventType;
    StagingRig g;
    // M1 (ordinary), M2 (staged, slot 1), the L2Release (no Feeds of the staging), then F1, which
    // reads M1's copy. The release must wait for M2's copy, not take M1's.
    g.exec.schedule_move(g.a);
    g.exec.schedule_move(g.staged(1, 0));
    g.exec.schedule_l2_release(g.staged(1, 0));
    auto f1 = g.a;
    f1.ordered = false;                         // reads whichever copy is there: M1's
    g.exec.schedule_feed(f1);
    const auto events = g.run();
    std::size_t staged_arrival = events.size(), release = events.size();
    for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];
        if (e.type == EventType::TILE_ARRIVED_L2 && e.slot_id == 1) staged_arrival = i;
        if (e.type == EventType::CREDIT_RELEASED && e.component_name == "L2:release") {
            CHECK(e.slot_id == 1);
            release = i;
        }
    }
    REQUIRE(release < events.size());
    CHECK(staged_arrival < release);
}
