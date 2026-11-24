// ============================================================================
// tests/timing/test_slot_pool.cpp
// Each L2 slot a CSP program places is always in exactly one state, FREE, OWNED, READY or READING,
// and a transition from the wrong state is a SlotFault by name (docs/plans/slot-credits-at-lca.md
// §3.4, §6; #387 step 3b). L-CA checks every slot against its tag CAM every cycle.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>
#include <sw/kpu/timing/slot_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>

#include <string>

using namespace sw::kpu::timing;
using Catch::Matchers::ContainsSubstring;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;

namespace {

TileID tile(int ti, int tj, sw::kpu::isa::MatrixID m = sw::kpu::isa::MatrixID::A) {
    TileID t;
    t.matrix = m;
    t.ti = static_cast<decltype(t.ti)>(ti);
    t.tj = static_cast<decltype(t.tj)>(tj);
    return t;
}

}  // namespace

TEST_CASE("SlotPool: a slot's life is FREE, OWNED, READY, READING, READY, FREE", "[timing][slots]") {
    SlotPool p(2);
    const TileID a = tile(0, 0);
    CHECK(p.state(0) == SlotPool::State::Free);
    p.acquire(0, a);
    CHECK(p.state(0) == SlotPool::State::Owned);
    CHECK(p.owner(0) == a);
    p.signal(0, a);
    CHECK(p.state(0) == SlotPool::State::Ready);
    p.take(0, a);
    p.take(0, a);                               // two consumers of one copy may overlap
    CHECK(p.state(0) == SlotPool::State::Reading);
    CHECK(p.readers(0) == 2);
    p.done(0, a);
    CHECK(p.state(0) == SlotPool::State::Reading);
    p.done(0, a);
    CHECK(p.state(0) == SlotPool::State::Ready);
    p.release(0, a);
    CHECK(p.state(0) == SlotPool::State::Free);
    CHECK(p.all_free());
    CHECK(p.acquires() == 1);
    CHECK(p.releases() == 1);
}

TEST_CASE("SlotPool: a transition from the wrong state is a fault, by name", "[timing][slots]") {
    SlotPool p(2);
    const TileID a = tile(0, 0), b = tile(0, 1);
    CHECK_THROWS_WITH(p.signal(0, a), ContainsSubstring("L2 slot 0: SIGNAL by") && ContainsSubstring("from FREE"));
    CHECK_THROWS_WITH(p.release(0, a), ContainsSubstring("RELEASE") && ContainsSubstring("from FREE"));
    CHECK_THROWS_WITH(p.take(0, a), ContainsSubstring("TAKE") && ContainsSubstring("from FREE"));
    p.acquire(0, a);
    CHECK_THROWS_WITH(p.acquire(0, b), ContainsSubstring("ACQUIRE") && ContainsSubstring("from OWNED") &&
                                           ContainsSubstring("held by " + a.to_string()));
    CHECK_THROWS_WITH(p.signal(0, b), ContainsSubstring("but " + a.to_string() + " holds it"));
    CHECK_THROWS_WITH(p.release(0, a), ContainsSubstring("from OWNED"));      // not landed yet
    p.signal(0, a);
    p.take(0, a);
    CHECK_THROWS_WITH(p.release(0, a), ContainsSubstring("from READING"));    // a consumer is reading it
    CHECK_THROWS_WITH(p.acquire(5, a), ContainsSubstring("past the program's 2 slots"));
}

TEST_CASE("TagCAM: placed slots drive the pool, and check_slots ties each state to the CAM", "[timing][slots]") {
    TagCAM cam(8);
    cam.set_placed_slots(2);
    REQUIRE(cam.slot_pool() != nullptr);
    const TileID a = tile(0, 0);
    REQUIRE(cam.claim_slot(1, a));
    CHECK(cam.slot_pool()->state(1) == SlotPool::State::Owned);
    cam.check_slots();
    REQUIRE(cam.insert(a, 1, 10));
    CHECK(cam.slot_pool()->state(1) == SlotPool::State::Ready);
    cam.take_slot(1, a);
    CHECK(cam.slot_pool()->state(1) == SlotPool::State::Reading);
    cam.done_slot(1, a);
    cam.check_slots();
    CHECK(cam.invalidate(a));                   // the L2Release: the entry and its claim go
    CHECK(cam.slot_pool()->state(1) == SlotPool::State::Free);
    cam.check_slots();
    // A slot no program places (past 2) has no state: the legacy and transient paths are untouched.
    REQUIRE(cam.claim_slot(5, tile(1, 1)));
    cam.check_slots();
    // An entry landed in a placed slot its tile never claimed: the pool stays FREE, so the CAM
    // and the pool disagree, and the check names the slot.
    TagCAM bad(8);
    bad.set_placed_slots(2);
    REQUIRE(bad.claim_slot(0, a));
    REQUIRE(bad.insert(a, 1, 5));               // lands in slot 1, though it claimed 0
    CHECK_THROWS_WITH(bad.check_slots(), ContainsSubstring("L2 slot 1 is FREE, but " + a.to_string() + " is in it"));
    // And a claimed slot whose tile has landed while the pool still says OWNED.
    TagCAM early(8);
    early.set_placed_slots(2);
    REQUIRE(early.claim_slot(0, a));
    REQUIRE(early.insert(a, 0, 5));             // SIGNAL: READY
    early.check_slots();
}

TEST_CASE("L-CA: every placed slot is acquired and released once per copy, and FREE at the end",
          "[timing][slots][csp]") {
    // A row-panel matmul with A staged: 16 stagings, 64 B copies and 16 results, each placed.
    const std::string src =
        "csp 1.1\nprogram mm machine flat(l3 = 9, l2 = 11) {\n"
        "  tensor A[64,64] tile 16x16 in;\n  tensor B[64,64] tile 16x16 in;\n  tensor C[64,64] tile 16x16 out;\n"
        "  for i in 0..4 {\n    resident A[i, :];\n    stage A[i, :] in l2 {\n"
        "      for j in 0..4 {\n        resident B[:, j];\n"
        "        acc C[i, j] in fabric {\n          for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n        }\n"
        "        store C[i, j] late;\n        release B[:, j];\n      }\n    }\n    release A[i, :];\n  }\n}\n";
    const lang::Program ast = lang::parse(src);
    const csp::CspProgram p = lang::compile(ast);
    const auto dev = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    const auto cfg = csp_config_from(dev);
    REQUIRE(cfg.has_value());
    ConcurrentTimingExecutor exec(cfg->config);
    lang::ActionStream s(ast, lang::target_from(dev));
    const auto r = CspDriver(exec, s, sw::kpu::program::driver::csp_inputs(ast)).run();
    REQUIRE(r.completed);
    const SlotPool* pool = exec.l2_tag_cam().slot_pool();
    REQUIRE(pool != nullptr);
    CHECK(pool->size() == 11);
    CHECK(pool->acquires() == p.stagings.size());
    CHECK(pool->releases() == p.stagings.size());
    CHECK(p.stagings.size() == 16 + 64 + 16);
    CHECK(pool->all_free());
}
