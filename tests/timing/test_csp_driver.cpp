// ============================================================================
// tests/timing/test_csp_driver.cpp
// L-CA from the CSP program (docs/plans/csp-program-tile-sequencing.md step 2): the cycle-level
// executor runs the program's actions, process by process -- no ScheduleResult, no generator --
// and computes the L0 reference's values, with exactly the program's DRAM loads.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lower.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
namespace csp = sw::kpu::program::csp;
using sw::kpu::program::TileProgram;
using sw::kpu::program::TileProgramReference;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;

namespace {

TileProgram matmul(sw::kpu::program::Dim size, sw::kpu::program::Dim tile) {
    sw::kpu::program::driver::ProgramSpec s;
    s.algo = "matmul";
    s.size = size;
    s.tile = tile;
    TileProgram p = sw::kpu::program::driver::derive(s);
    sw::kpu::program::driver::fill(p, s);
    return p;
}

ConcurrentTimingExecutor::Config machine(const char* file) {
    auto c = csp_config_from(read_spec_file(std::string("tests/program/deploy/") + file).device(0))->config;
    c.max_cycles = 5'000'000;
    return c;
}

}  // namespace

TEST_CASE("CSP driver: S1 runs the program at L-CA -- the reference's values, the program's loads",
          "[timing][csp][driver]") {
    const TileProgram l0 = matmul(256, 32);
    const auto cfg = machine("kpu_s1.json");
    const csp::CspProgram p = csp::lower(l0, {cfg.l3_buffer_count});
    ConcurrentTimingExecutor exec(cfg);
    CspDriver driver(exec, p);
    const auto r = driver.run();
    REQUIRE(r.completed);
    CHECK_FALSE(r.livelock);

    TileProgram ref = l0;
    TileProgramReference().run(ref);
    CHECK(r.values.operand("C").values == ref.operand("C").values);          // bit for bit
    csp::BehavioralInterpreter lb;
    lb.run(p);
    CHECK(r.values.operand("C").values == lb.result().operand("C").values);  // L-B == L-CA

    // Reuse is the program's: the executor read DRAM exactly as often as the program loads.
    CHECK(r.dram_loads == p.reuse().loads);
    CHECK(r.dram_loads == 128);
    CHECK(r.dram_stores == 64);
    CHECK(exec.l3_credits_available() == cfg.l3_buffer_count);              // every slot came home
    Cycle last_load = 0, first_compute = ~Cycle{0}, last_feed = 0, last_move = 0;
    for (const auto& e : exec.events()) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) last_load = std::max(last_load, e.cycle + e.duration);
        if (e.type == EventType::COMPUTE_START) first_compute = std::min(first_compute, e.cycle);
        if (e.type == EventType::STR_FEED_COMPLETE) last_feed = std::max(last_feed, e.cycle + e.duration);
        if (e.type == EventType::BM_MOVE_COMPLETE) last_move = std::max(last_move, e.cycle + e.duration);
    }
    std::printf("csp driver S1 256^3/32^3: %llu cycles, %zu DRAM loads (generator path: 36,668 cycles, 128); "
                "last load %llu, first compute %llu, last move %llu, last feed %llu, cf busy %llu\n",
                static_cast<unsigned long long>(r.cycles), r.dram_loads, static_cast<unsigned long long>(last_load),
                static_cast<unsigned long long>(first_compute), static_cast<unsigned long long>(last_move),
                static_cast<unsigned long long>(last_feed),
                static_cast<unsigned long long>(exec.compute_tile_busy_cycles(0)));
    {
        Cycle first_store = ~Cycle{0}, last_store = 0, last_compute = 0, first_drain = ~Cycle{0}, first_wb = ~Cycle{0};
        for (const auto& e : exec.events()) {
            if (e.type == EventType::DMA_STORE_COMPLETE) { first_store = std::min(first_store, e.cycle); last_store = std::max(last_store, e.cycle + e.duration); }
            if (e.type == EventType::COMPUTE_COMPLETE) last_compute = std::max(last_compute, e.cycle + e.duration);
            if (e.type == EventType::STR_DRAIN_COMPLETE) first_drain = std::min(first_drain, e.cycle);
            if (e.type == EventType::BM_WRITEBACK_COMPLETE) first_wb = std::min(first_wb, e.cycle);
        }
        std::printf("  first drain %llu, first writeback %llu, first store %llu, last store %llu, last compute %llu\n",
                    static_cast<unsigned long long>(first_drain), static_cast<unsigned long long>(first_wb),
                    static_cast<unsigned long long>(first_store), static_cast<unsigned long long>(last_store),
                    static_cast<unsigned long long>(last_compute));
    }
    auto st = exec.get_statistics();
    std::printf("  stalls: dma credit %llu, bm tag %llu bm credit %llu, str tag %llu str credit %llu\n",
                static_cast<unsigned long long>(st.dma_credit_stalls), static_cast<unsigned long long>(st.bm_tag_stalls),
                static_cast<unsigned long long>(st.bm_credit_stalls), static_cast<unsigned long long>(st.str_tag_stalls),
                static_cast<unsigned long long>(st.str_credit_stalls));
}

TEST_CASE("CSP driver: a tighter program's reloads are the executor's DRAM reads, one for one",
          "[timing][csp][driver]") {
    const TileProgram l0 = matmul(128, 32);
    const auto cfg = machine("kpu_s1.json");
    for (std::size_t cap : {std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
        CAPTURE(cap);
        const csp::CspProgram p = csp::lower(l0, {cap});
        REQUIRE(p.reuse().reloads > 0);
        ConcurrentTimingExecutor exec(cfg);
        const auto r = CspDriver(exec, p).run();
        REQUIRE(r.completed);
        CHECK(r.dram_loads == p.reuse().loads);
        TileProgram ref = l0;
        TileProgramReference().run(ref);
        CHECK(r.values.operand("C").values == ref.operand("C").values);
        std::printf("csp driver S1 128^3/32^3, program L3 %2zu: %llu cycles, %zu DRAM loads\n", cap,
                    static_cast<unsigned long long>(r.cycles), r.dram_loads);
    }
}

TEST_CASE("CSP driver: the dma process takes its credits in program order",
          "[timing][csp][driver]") {
    const TileProgram l0 = matmul(128, 32);
    const auto cfg = machine("kpu_s1.json");
    const csp::CspProgram p = csp::lower(l0, {8});
    ConcurrentTimingExecutor exec(cfg);
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
    CHECK(credit_order == program_order);
}

TEST_CASE("CSP driver: a program not lowered for this machine is refused by name",
          "[timing][csp][driver]") {
    const TileProgram l0 = matmul(64, 32);
    SECTION("more L3 than the machine has, or an unbounded one") {
        ConcurrentTimingExecutor exec(machine("kpu_s1.json"));
        const csp::CspProgram big = csp::lower(l0, {4096});
        CHECK_THROWS_WITH(CspDriver(exec, big), ContainsSubstring("lowered for an L3 of 4096 tiles"));
        const csp::CspProgram unbounded = csp::lower(l0, {0});
        CHECK_THROWS_WITH(CspDriver(exec, unbounded), ContainsSubstring("unbounded size"));
    }
    SECTION("a machine with several L3 tiles is level 2's") {
        ConcurrentTimingExecutor exec(machine("kpu_t4.json"));
        CHECK_THROWS_WITH(CspDriver(exec, csp::lower(l0, {8})), ContainsSubstring("level 2 distributes"));
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
    const TileProgram l0 = matmul(256, 32);
    const csp::CspProgram p = csp::lower(l0, {cfg.l3_buffer_count});
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
