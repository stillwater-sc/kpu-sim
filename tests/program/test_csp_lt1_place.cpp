// ============================================================================
// tests/program/test_csp_lt1_place.cpp
// L-T1 executes L3 placement (docs/plans/l3-placement-at-lb.md step 5; ADR 0004 §5): a credit is
// a slot. Every action that opens a residency waits for ITS slot's Release, not for whichever
// slot frees first, so a program or a placement policy that reuses a slot early serializes on
// it -- a register's write-after-read, in L3. Placement is the walker's, the same as L-B's.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include "trace_fixtures.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using Kind = Action::Kind;
using LT = TransactionalInterpreter;

namespace {

struct Run {
    LT::Stats stats;
    std::vector<LT::Record> records;
    std::vector<LT::Slot> slots;
};

// kpu_s1's lanes and rates: enough DMA bandwidth that a Load can run ahead of the fabric, so a
// slot, not the DMA, can be what decides when it starts.
characterize::DeviceDescriptor s1() {
    return platform::read_spec_file("tests/program/deploy/kpu_s1.json").device_view(0);
}

Run lt1(const std::string& text, const characterize::DeviceDescriptor& dev = s1()) {
    const lang::Program ast = lang::parse(text);
    LT lt(dev, static_cast<std::size_t>(ast.l3),
          static_cast<std::uint32_t>(ast.banks));
    lt.begin(driver::csp_inputs(ast));
    lang::ActionStream s(ast);
    while (auto e = s.next()) lt.step(e->action, e->op ? &*e->op : nullptr);
    Run r;
    r.stats = lt.finish();
    r.records = lt.records();
    r.slots = lt.slots();
    return r;
}

std::string with_machine(std::string text, const std::string& options) {
    const auto at = text.find("machine flat(l3 = ");
    REQUIRE(at != std::string::npos);
    return text.insert(text.find(')', at), options);
}

// Within each slot, a residency starts no earlier than the previous one in that slot ended: a
// slot holds one tile at a time, in time as well as in program order.
void check_one_tile_per_slot(const std::vector<LT::Slot>& slots) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<const LT::Slot*>> by;
    for (const auto& s : slots) by[{s.place.l3_tile, s.place.slot}].push_back(&s);
    for (auto& [at, v] : by) {
        std::sort(v.begin(), v.end(), [](const LT::Slot* a, const LT::Slot* b) { return a->t0 < b->t0; });
        for (std::size_t i = 1; i < v.size(); ++i) {
            CAPTURE(at.second, v[i - 1]->tile.to_string(), v[i]->tile.to_string());
            CHECK(v[i]->t0 >= v[i - 1]->t1);
        }
    }
}

}  // namespace

TEST_CASE("L-T1 executes the walker's placement, the same as L-B's", "[csp][placement][lt1]") {
    for (const char* opts : {"", ", place = lowest", ", banks = 4, place = bank-spread"}) {
        for (const char* algo : {"matmul", "lu"}) {
            CAPTURE(algo, opts);
            const auto o = trace_fixtures::gen(algo, algo == std::string("lu") ? 64 : 96, algo == std::string("lu") ? 16 : 32);
            const std::string text = with_machine(gen::generate(o), opts);
            const CspProgram p = lang::compile(lang::parse(text));
            const Run r = lt1(text);
            // Every record names the slot its action carries.
            for (const LT::Record& rec : r.records) {
                const Action& a = p.actions.at(rec.action);
                REQUIRE(rec.place.has_value() == a.place.has_value());
                if (rec.place) CHECK(*rec.place == *a.place);
            }
            // Every residency's slot is the one the walker placed it in.
            std::multiset<std::tuple<std::string, std::uint32_t, std::uint32_t>> walker, executed;
            for (const Residency& res : p.residencies) walker.insert({res.tile.to_string(), res.place.slot, res.place.bank});
            for (const LT::Slot& s : r.slots) executed.insert({s.tile.to_string(), s.place.slot, s.place.bank});
            CHECK(executed == walker);
            check_one_tile_per_slot(r.slots);
        }
    }
}

TEST_CASE("L-T1: a slot reused early serializes on it", "[csp][placement][lt1]") {
    // Four slots, two tiles at a time. With the policy, each k-step's tiles land in the slots freed
    // longest ago, so their Loads wait on nothing. Pinned to slots 0 and 1, each k-step's Loads
    // wait for the Releases of the step before, though two other slots stand free.
    const std::string base = R"(csp 1.0
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
    std::string free_text = base, pinned_text = base;
    free_text.replace(free_text.find("@@"), 2, "");
    pinned_text.replace(pinned_text.find("@@"), 2, " @ slots(0..2)");
    const Run free = lt1(free_text);
    const Run pinned = lt1(pinned_text);
    const Run lowest = lt1(with_machine(free_text, ", place = lowest"));
    check_one_tile_per_slot(pinned.slots);
    for (const LT::Slot& s : pinned.slots)
        if (s.tile.operand != "C") CHECK(s.place.slot < 2);
    // The early reuse costs time: its Loads wait on their slots, and the run is longer.
    CHECK(pinned.stats.slot_wait_cycles > free.stats.slot_wait_cycles);
    CHECK(pinned.stats.makespan > free.stats.makespan);
    // `lowest` reuses the lowest freed slot, so on a stream it reuses slots 0 and 1 as the pinned
    // program does; fifo reuses the slot freed longest ago.
    CHECK(lowest.stats.slot_wait_cycles > free.stats.slot_wait_cycles);
    CHECK(lowest.stats.makespan > free.stats.makespan);
    std::printf("streamed, lowest: %zu slot-bound (%llu cycles), makespan %llu\n", lowest.stats.slot_bound,
                (unsigned long long)lowest.stats.slot_wait_cycles, (unsigned long long)lowest.stats.makespan);
    std::printf("slot reuse: free %zu stalls, %zu slot-bound (%llu cycles), makespan %llu; pinned %zu stalls, "
                "%zu slot-bound (%llu cycles), makespan %llu\n",
                free.stats.credit_stalls, free.stats.slot_bound, (unsigned long long)free.stats.slot_wait_cycles,
                (unsigned long long)free.stats.makespan, pinned.stats.credit_stalls, pinned.stats.slot_bound,
                (unsigned long long)pinned.stats.slot_wait_cycles, (unsigned long long)pinned.stats.makespan);
}

TEST_CASE("L-T1: fifo against lowest on matmul (measured)", "[csp][placement][lt1]") {
    // The plan's measurement: the policy decides how soon a freed slot is reused. Measured on
    // kpu_s1 (2026-10-12): kpu-csp-gen's matmul 128/32 has no slot-bound Load under either policy,
    // at its own L3 or twice it, so the makespans are equal. Its Releases come early enough that
    // a reused slot is always free before the DMA reaches its Load. The streamed program above is
    // where the policy shows.
    const std::string text = gen::generate(trace_fixtures::gen("matmul", 128, 32));
    const Run fifo = lt1(text);
    const Run lowest = lt1(with_machine(text, ", place = lowest"));
    check_one_tile_per_slot(fifo.slots);
    check_one_tile_per_slot(lowest.slots);
    for (const char* l3 : {"", "x2"}) {
        std::string t = text;
        if (std::string(l3) == "x2") {
            const auto at = t.find("machine flat(l3 = ") + 18;
            const auto end = t.find_first_of(",)", at);
            t.replace(at, end - at, std::to_string(2 * std::stoul(t.substr(at, end - at))));
        }
        const Run f = lt1(t);
        const Run lo = lt1(with_machine(t, ", place = lowest"));
        std::printf("matmul 128/32 l3%s: fifo %zu slot-bound (%llu cyc) mk %llu; lowest %zu slot-bound (%llu cyc) mk %llu\n",
                    l3, f.stats.slot_bound, (unsigned long long)f.stats.slot_wait_cycles,
                    (unsigned long long)f.stats.makespan, lo.stats.slot_bound,
                    (unsigned long long)lo.stats.slot_wait_cycles, (unsigned long long)lo.stats.makespan);
    }
}
