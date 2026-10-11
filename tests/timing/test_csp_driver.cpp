// ============================================================================
// tests/timing/test_csp_driver.cpp
// L-CA from the CSP program (docs/plans/csp-program-tile-sequencing.md step 2): the cycle-level
// executor runs the program's actions, process by process -- no ScheduleResult, no generator.
//
// Two checks, made differently (docs/plans/retire-l0-derivations.md §3.8):
//   the sequence of tile moves   exactly: the program's DRAM loads and stores, the DMA's credit
//                                order, every L3 credit returned
//   the bits of the result       per tile, read as float32, within the proximity bound of L-B's
//                                values for the same program (check_tiles)
//
// The programs are written (kpu-csp-gen, or by hand), on the inputs kpu-run synthesizes.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include "../program/trace_fixtures.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
namespace csp = sw::kpu::program::csp;
using sw::kpu::program::TileProgram;
using sw::kpu::program::check_tiles;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;

namespace {

ConcurrentTimingExecutor::Config machine(const char* file) {
    auto c = csp_config_from(read_spec_file(std::string("tests/program/deploy/") + file).device(0))->config;
    c.max_cycles = 5'000'000;
    return c;
}

// kpu-csp-gen's matmul, column panels unless asked otherwise.
csp::gen::Options matmul(sw::kpu::program::Dim size, sw::kpu::program::Dim tile, const char* orient = "col") {
    csp::gen::Options o = trace_fixtures::gen("matmul", size, tile);
    o.orient = orient;
    return o;
}

// C at L-CA, tile by tile, against L-B's C for the same program.
void check_c(const TileProgram& lca, const TileProgram& lb) {
    const auto c = check_tiles(lca, lb, "C");
    CAPTURE(c.summary());
    CHECK(c.pass);
}

// A written matmul (docs/plans/csp-language.md step 1), 128^3 in 32 x 32 tiles: A held whole,
// B's column panel per j -- 16 + 16 loads.
const char* kWrittenMatmul = R"(csp 1.0
program matmul machine flat(l3 = 128) {
  tensor A[128,128] tile 32x32 in;
  tensor B[128,128] tile 32x32 in;
  tensor C[128,128] tile 32x32 out;
  resident A[:, :];
  for j in 0..4 {
    resident B[:, j];
    for i in 0..4 {
      acc C[i, j] in fabric {
        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }
      }
      store C[i, j];
    }
    release B[:, j];
  }
  release A[:, :];
}
)";

}  // namespace

TEST_CASE("CSP driver: S1 runs a written program at L-CA -- L-B's values, the program's loads",
          "[timing][csp][driver]") {
    const auto o = matmul(256, 32);
    const csp::CspProgram p = trace_fixtures::compiled(o);
    const auto cfg = machine("kpu_s1.json");
    ConcurrentTimingExecutor exec(cfg);
    CspDriver driver(exec, p);
    const auto r = driver.run();
    REQUIRE(r.completed);
    CHECK_FALSE(r.livelock);

    check_c(r.values, trace_fixtures::lb_values(o));

    // Reuse is the program's: the executor read DRAM exactly as often as the program loads.
    // Column panels: B's 64 tiles once, A's row panel (8 tiles) once per (j, i): 64 + 512.
    CHECK(r.dram_loads == p.reuse().loads);
    CHECK(r.dram_loads == 576);
    CHECK(r.dram_stores == 64);
    CHECK(exec.l3_credits_available() == cfg.l3_buffer_count);              // every slot came home
    Cycle last_load = 0, first_compute = ~Cycle{0}, last_feed = 0, last_move = 0;
    for (const auto& e : exec.events()) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) last_load = std::max(last_load, e.cycle + e.duration);
        if (e.type == EventType::COMPUTE_START) first_compute = std::min(first_compute, e.cycle);
        if (e.type == EventType::STR_FEED_COMPLETE) last_feed = std::max(last_feed, e.cycle + e.duration);
        if (e.type == EventType::BM_MOVE_COMPLETE) last_move = std::max(last_move, e.cycle + e.duration);
    }
    std::printf("csp driver S1 256^3/32^3 (kpu-csp-gen, col): %llu cycles, %zu DRAM loads; "
                "last load %llu, first compute %llu, last move %llu, last feed %llu, cf busy %llu\n",
                static_cast<unsigned long long>(r.cycles), r.dram_loads, static_cast<unsigned long long>(last_load),
                static_cast<unsigned long long>(first_compute), static_cast<unsigned long long>(last_move),
                static_cast<unsigned long long>(last_feed),
                static_cast<unsigned long long>(exec.compute_tile_busy_cycles(0)));
    auto st = exec.get_statistics();
    std::printf("  stalls: dma credit %llu, bm tag %llu bm credit %llu, str tag %llu str credit %llu\n",
                static_cast<unsigned long long>(st.dma_credit_stalls), static_cast<unsigned long long>(st.bm_tag_stalls),
                static_cast<unsigned long long>(st.bm_credit_stalls), static_cast<unsigned long long>(st.str_tag_stalls),
                static_cast<unsigned long long>(st.str_credit_stalls));
}

TEST_CASE("CSP driver: a program's loads are the executor's DRAM reads, one for one, whatever its reuse",
          "[timing][csp][driver]") {
    const auto cfg = machine("kpu_s1.json");
    // Three schedules of one matmul: A held whole (no reloads), and column and row panels, which
    // reload the panel they do not hold.
    struct Case { const char* name; csp::lang::Program ast; };
    const std::vector<Case> cases = {{"written, A whole", csp::lang::parse(kWrittenMatmul)},
                                     {"kpu-csp-gen col", trace_fixtures::program_of(matmul(128, 32, "col"))},
                                     {"kpu-csp-gen row", trace_fixtures::program_of(matmul(128, 32, "row"))}};
    for (const Case& c : cases) {
        CAPTURE(c.name);
        const csp::CspProgram p = trace_fixtures::compiled(c.ast);
        ConcurrentTimingExecutor exec(cfg);
        const auto r = CspDriver(exec, p).run();
        REQUIRE(r.completed);
        CHECK(r.dram_loads == p.reuse().loads);
        CHECK(r.dram_stores == p.reuse().stores);
        CHECK(exec.l3_credits_available() == cfg.l3_buffer_count);
        if (std::string(c.name) != "written, A whole") CHECK(p.reuse().reloads > 0);
        check_c(r.values, trace_fixtures::lb_values(c.ast));
        std::printf("csp driver S1 128^3/32^3, %s: %llu cycles, %zu DRAM loads (%zu reloads)\n", c.name,
                    static_cast<unsigned long long>(r.cycles), r.dram_loads, p.reuse().reloads);
    }
}

TEST_CASE("CSP driver: the dma process takes its credits in program order",
          "[timing][csp][driver]") {
    const csp::CspProgram p = trace_fixtures::compiled(matmul(128, 32));
    ConcurrentTimingExecutor exec(machine("kpu_s1.json"));
    REQUIRE(CspDriver(exec, p).run().completed);
    std::vector<TileID> program_order, credit_order;
    for (const auto& a : p.actions)
        if (a.kind == csp::Action::Kind::Load) {
            TileID t;
            t.matrix = a.tile.operand == "A" ? sw::kpu::isa::MatrixID::A : sw::kpu::isa::MatrixID::B;
            t.ti = a.tile.ti;
            t.tj = a.tile.tj;
            program_order.push_back(t);
        }
    for (const auto& e : exec.events())
        if (e.type == EventType::CREDIT_ACQUIRED && e.component_name.rfind("DMA", 0) == 0 &&
            e.tile_id.matrix != sw::kpu::isa::MatrixID::C)
            credit_order.push_back(e.tile_id);
    REQUIRE_FALSE(program_order.empty());
    CHECK(credit_order == program_order);
}

TEST_CASE("CSP driver: a program not written for this machine is refused by name",
          "[timing][csp][driver]") {
    SECTION("more L3 than the machine has, or an unbounded one") {
        ConcurrentTimingExecutor exec(machine("kpu_s1.json"));
        csp::gen::Options o = matmul(64, 32);
        o.l3 = 4096;
        const csp::CspProgram big = trace_fixtures::compiled(o);
        CHECK_THROWS_WITH(CspDriver(exec, big), ContainsSubstring("written for an L3 of 4096 tiles"));
        csp::CspProgram unbounded = trace_fixtures::compiled(matmul(64, 32));
        unbounded.channels.at(static_cast<std::size_t>(csp::Chan::L3)).capacity = 0;
        CHECK_THROWS_WITH(CspDriver(exec, unbounded), ContainsSubstring("unbounded size"));
    }
    SECTION("a machine with several L3 tiles is level 2's") {
        ConcurrentTimingExecutor exec(machine("kpu_t4.json"));
        CHECK_THROWS_WITH(CspDriver(exec, trace_fixtures::compiled(matmul(64, 32))),
                          ContainsSubstring("level 2 distributes"));
    }
}

TEST_CASE("CSP driver: timeline against the generator path (characterization)",
          "[timing][csp][driver][characterization]") {
    using namespace sw::kpu::timing::schedule;
    const auto cfg = machine("kpu_s1.json");
    MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = 256;
    g.Ti = g.Tj = g.Tk = 32;
    g.l3_buffer_count = 128;
    g.a_base = 0x100000; g.b_base = 0x400000; g.c_base = 0x700000;
    const auto s = MatMulScheduleGenerator(g).generate();
    ConcurrentTimingExecutor exec(cfg);
    ScheduleExecutor x(exec);
    const auto r = x.execute(s);
    REQUIRE(r.success);
    Cycle last_load = 0, first_compute = ~Cycle{0}, last_feed = 0, last_move = 0, first_store = ~Cycle{0};
    std::size_t dram = 0;
    for (const auto& e : exec.events()) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) { ++dram; last_load = std::max(last_load, e.cycle + e.duration); }
        if (e.type == EventType::COMPUTE_START) first_compute = std::min(first_compute, e.cycle);
        if (e.type == EventType::STR_FEED_COMPLETE) last_feed = std::max(last_feed, e.cycle + e.duration);
        if (e.type == EventType::BM_MOVE_COMPLETE) last_move = std::max(last_move, e.cycle + e.duration);
        if (e.type == EventType::DMA_STORE_COMPLETE) first_store = std::min(first_store, e.cycle);
    }
    auto st = exec.get_statistics();
    std::printf("generator S1 256^3/32^3: %llu cycles, %zu DRAM loads; last load %llu, first compute %llu, last move %llu, "
                "last feed %llu, first store %llu, cf busy %llu; stalls bm tag %llu credit %llu, str tag %llu\n",
                static_cast<unsigned long long>(r.total_cycles), dram, static_cast<unsigned long long>(last_load),
                static_cast<unsigned long long>(first_compute), static_cast<unsigned long long>(last_move),
                static_cast<unsigned long long>(last_feed), static_cast<unsigned long long>(first_store),
                static_cast<unsigned long long>(exec.compute_tile_busy_cycles(0)),
                static_cast<unsigned long long>(st.bm_tag_stalls), static_cast<unsigned long long>(st.bm_credit_stalls),
                static_cast<unsigned long long>(st.str_tag_stalls));
}

TEST_CASE("CSP driver: with the DMA burst window on (characterization)", "[timing][csp][driver][characterization]") {
    using namespace sw::kpu::timing::schedule;
    auto cfg = machine("kpu_s1.json");
    cfg.dma_window = 32;
    const csp::CspProgram p = trace_fixtures::compiled(matmul(256, 32));
    ConcurrentTimingExecutor e1(cfg);
    const auto r = CspDriver(e1, p).run();
    REQUIRE(r.completed);
    MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = 256; g.Ti = g.Tj = g.Tk = 32; g.l3_buffer_count = 128;
    g.a_base = 0x100000; g.b_base = 0x400000; g.c_base = 0x700000;
    ConcurrentTimingExecutor e2(cfg);
    ScheduleExecutor x(e2);
    const auto gr = x.execute(MatMulScheduleGenerator(g).generate());
    REQUIRE(gr.success);
    std::printf("window 32 on S1 256^3/32^3: csp %llu cycles, generator %llu cycles\n",
                static_cast<unsigned long long>(r.cycles), static_cast<unsigned long long>(gr.total_cycles));
}

TEST_CASE("CSP driver: a WRITTEN program runs at L-CA -- its values, its loads", "[timing][csp][driver][lang]") {
    // The CSP language (docs/plans/csp-language.md step 1): a program someone wrote by hand.
    // A held whole, B's column panel per j: 16 + 16 loads, as written.
    const csp::lang::Program ast = csp::lang::parse(kWrittenMatmul);
    const csp::CspProgram p = trace_fixtures::compiled(ast);
    ConcurrentTimingExecutor exec(machine("kpu_s1.json"));
    const auto r = CspDriver(exec, p).run();
    REQUIRE(r.completed);
    check_c(r.values, trace_fixtures::lb_values(ast));
    CHECK(r.dram_loads == p.reuse().loads);
    CHECK(r.dram_loads == 32);
    std::printf("csp driver, written matmul 128^3/32^3 on S1: %llu cycles, %zu DRAM loads\n",
                static_cast<unsigned long long>(r.cycles), r.dram_loads);
}

TEST_CASE("CSP driver: the STREAM runs at L-CA -- the trace's cycles, in a bounded window",
          "[timing][csp][driver][lang][stream]") {
    // docs/plans/csp-language.md step 1c.2: the structured program executes at L-CA without
    // its trace. The driver pulls actions as the executor's backlog allows; L3 entries are
    // retired by the program's Releases, so nothing needs lookahead.
    const csp::lang::Program ast = csp::lang::parse(kWrittenMatmul);
    const TileProgram inputs = sw::kpu::program::driver::csp_inputs(ast);
    const TileProgram ref = trace_fixtures::lb_values(ast);
    const auto cfg = machine("kpu_s1.json");

    const csp::CspProgram trace = trace_fixtures::compiled(ast);
    ConcurrentTimingExecutor e0(cfg);
    const auto t = CspDriver(e0, trace).run();
    REQUIRE(t.completed);

    SECTION("the whole stream at once is the trace, cycle for cycle") {
        csp::lang::ActionStream s(csp::lang::parse(kWrittenMatmul));
        ConcurrentTimingExecutor e(cfg);
        const auto r = CspDriver(e, s, inputs, CspDriver::kWholeProgram).run();
        REQUIRE(r.completed);
        CHECK(r.cycles == t.cycles);
        CHECK(r.actions == trace.actions.size());
        CHECK(r.dram_loads == t.dram_loads);
        check_c(r.values, ref);
    }
    SECTION("a window bounds what the driver holds; values and loads do not change") {
        for (const std::size_t w : {1u, 2u, 4u, 16u, 64u, 256u}) {
            CAPTURE(w);
            csp::lang::ActionStream s(csp::lang::parse(kWrittenMatmul));
            ConcurrentTimingExecutor e(cfg);
            const auto r = CspDriver(e, s, inputs, w).run();
            REQUIRE(r.completed);
            CHECK(r.actions == trace.actions.size());
            CHECK(r.peak_held <= w + 1);        // one pull hands over at most one more
            CHECK(r.dram_loads == 32);
            check_c(r.values, ref);
            CHECK(r.cycles >= t.cycles);
            std::printf("csp driver, streamed matmul 128^3/32^3 on S1, window %2zu: %llu cycles (trace %llu)\n", w,
                        static_cast<unsigned long long>(r.cycles), static_cast<unsigned long long>(t.cycles));
        }
    }
    SECTION("the inputs must be the operands the program declares") {
        csp::lang::ActionStream s(csp::lang::parse(kWrittenMatmul));
        ConcurrentTimingExecutor e(cfg);
        CHECK_THROWS_WITH(CspDriver(e, s, sw::kpu::program::driver::csp_inputs(
                                             trace_fixtures::program_of(matmul(64, 32)))),
                          ContainsSubstring("not the shape the program declares"));
    }
}

TEST_CASE("CSP driver: a Release retires its entry after the moves before it, and before the moves after it",
          "[timing][csp][driver][release]") {
    // The executor's Release (step 1c.2), directly: one tile, two residencies. The first is
    // moved twice and released; the second is loaded again and moved once. The Release waits
    // for both earlier moves, and the later move waits for the second load -- it never moves
    // the first residency's copy, though that copy sits in L3 until its Release.
    ConcurrentTimingExecutor exec(machine("kpu_s1.json"));
    TileDescriptor d;
    d.tile_id.matrix = sw::kpu::isa::MatrixID::A;
    d.height = d.width = 32;
    d.size_bytes = 32 * 32 * 4;
    d.matrix_base_address = 0x100000;
    d.dram_address = 0x100000;
    d.l3_held = true;
    TilePayload v;
    v.rows = v.cols = 32;
    v.values.assign(32 * 32, 1.0f);
    exec.set_tile_payload(d.tile_id, v);
    exec.schedule_load(d);
    exec.schedule_move(d);
    exec.schedule_move(d);
    exec.schedule_release(d);
    exec.schedule_move(d);         // the second residency's
    const std::size_t credits = exec.l3_credits_available();
    for (int i = 0; i < 20000; ++i) exec.step();
    REQUIRE(exec.backlog() == 1);  // the last move waits on a load the program has not issued
    exec.schedule_load(d);
    exec.schedule_release(d);
    while (!exec.is_complete() && exec.current_cycle() < 1'000'000) exec.step();
    REQUIRE(exec.is_complete());
    CHECK(exec.l3_credits_available() == credits);

    std::vector<Cycle> in_l3, in_l2, released;
    for (const auto& e : exec.events()) {
        if (!(e.tile_id == d.tile_id)) continue;
        if (e.type == EventType::TILE_ARRIVED_L3) in_l3.push_back(e.cycle);
        if (e.type == EventType::TILE_ARRIVED_L2) in_l2.push_back(e.cycle);
        if (e.type == EventType::CREDIT_RELEASED && e.component_name.find(":BM") != std::string::npos) released.push_back(e.cycle);
    }
    REQUIRE(in_l3.size() == 2);
    REQUIRE(in_l2.size() == 3);
    REQUIRE(released.size() == 2);
    CHECK(released[0] >= in_l2[1]);    // after both of the first residency's moves
    CHECK(in_l2[2] >= in_l3[1]);       // the later move takes the second load's copy
    CHECK(released[1] >= in_l2[2]);
}
