// ============================================================================
// tests/timing/test_csp_spec_oracle.cpp
// NoC port plan step 4b.1: a deployment's device drives the CSP cycle-accurate executor, and
// a matmul on it computes the L0 reference's values (ADR 0001 D5: within atol + rtol).
//
// This is the oracle the rest of step 4b keeps: push-only writeback (4b.2), L3 placement
// (4b.3) and the NoC (4b.4) each change WHEN things happen on T4, T16 and T64, and never WHAT
// is computed. 4b.4 additionally holds NoC-on against NoC-off bit for bit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/floorplan.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/program/value_tolerance.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;
using sw::kpu::program::TensorOperand;
using sw::kpu::program::TileProgram;
using sw::kpu::program::TileProgramReference;
using sw::kpu::program::platform::DeviceSpecification;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;

namespace {

DeviceSpecification device(const std::string& file) {
    return read_spec_file("tests/program/deploy/" + file).device(0);
}

// One T x T block of a row-major operand.
std::vector<float> block(const TensorOperand& t, Size br, Size bc, Size T) {
    std::vector<float> b(static_cast<std::size_t>(T) * T);
    for (Size r = 0; r < T; ++r)
        for (Size c = 0; c < T; ++c)
            b[static_cast<std::size_t>(r) * T + c] =
                t.values[static_cast<std::size_t>(br * T + r) * t.cols + (bc * T + c)];
    return b;
}

struct OracleRun {
    std::vector<float> csp, reference;      // C, row-major
    Cycle cycles = 0;
    std::size_t l3_capacity = 0, l3_free_after = 0;
};

// C = A @ B (size^3, tile^3) on the CSP executor configured from `d`, and on the L0 reference
// with the same inputs.
OracleRun run_matmul(const DeviceSpecification& d, Size size, Size T) {
    std::string why;
    const auto csp = csp_config_from(d, &why);
    INFO(why);
    REQUIRE(csp);

    sw::kpu::program::driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = size;
    ps.tile = T;
    TileProgram prog = sw::kpu::program::driver::derive(ps);
    sw::kpu::program::driver::fill(prog, ps);
    TileProgram ref = prog;
    TileProgramReference().run(ref);

    MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = size;
    g.Ti = g.Tj = g.Tk = T;
    g.l3_buffer_count = static_cast<Size>(csp->config.l3_buffer_count);
    g.l2_bank_count = static_cast<Size>(csp->config.l2_bank_count);
    g.a_base = 0x100000;
    g.b_base = 0x400000;
    g.c_base = 0x700000;
    const auto schedule = MatMulScheduleGenerator(g).generate();
    REQUIRE(schedule.valid);

    ConcurrentTimingExecutor::Config ec = csp->config;
    ec.max_cycles = 20'000'000;
    ConcurrentTimingExecutor exec(ec);
    const TensorOperand& A = prog.operand("A");
    const TensorOperand& B = prog.operand("B");
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::LOAD) continue;
        const auto id = op.tile.tile_id;
        exec.set_tile_payload(id, id.matrix == MatrixID::A
                                      ? TilePayload{T, T, block(A, id.ti, id.tk, T)}
                                      : TilePayload{T, T, block(B, id.tk, id.tj, T)});
    }
    for (const auto& op : schedule.operations) {
        switch (op.type) {
            case ScheduleOpType::LOAD:      exec.schedule_load(op.tile, op.engine_id); break;
            case ScheduleOpType::MOVE:      exec.schedule_move(op.tile, op.transpose, op.mover_id); break;
            case ScheduleOpType::FEED:      exec.schedule_feed(op.tile, op.streamer_id); break;
            case ScheduleOpType::DRAIN:     exec.schedule_drain(op.tile, op.streamer_id); break;
            case ScheduleOpType::WRITEBACK: exec.schedule_writeback(op.tile, op.mover_id); break;
            case ScheduleOpType::STORE:     exec.schedule_store(op.tile, op.engine_id); break;
            case ScheduleOpType::COMPUTE: {
                ConcurrentTimingExecutor::MatMulComputeSpec spec;
                for (const auto& dep : op.dependency_tiles)
                    (dep.matrix == MatrixID::A ? spec.a_tiles : spec.b_tiles).push_back(dep);
                exec.schedule_matmul_compute(op.tile, spec);
                break;
            }
        }
    }
    while (!exec.is_complete() && exec.current_cycle() < ec.max_cycles) exec.step();
    REQUIRE(exec.is_complete());

    OracleRun out;
    out.cycles = exec.current_cycle();
    out.l3_capacity = ec.l3_buffer_count;
    out.l3_free_after = exec.l3_credits().available();
    out.reference = ref.operand("C").values;
    out.csp.assign(out.reference.size(), 0.0f);
    std::size_t stored = 0;
    for (const auto& op : schedule.operations) {
        if (op.type != ScheduleOpType::STORE) continue;
        ++stored;
        const auto id = op.tile.tile_id;
        const auto& p = exec.tile_payload_at(MemoryLevel::DRAM, id);
        for (Size r = 0; r < T; ++r)
            for (Size c = 0; c < T; ++c)
                out.csp[static_cast<std::size_t>(id.ti * T + r) * size + (id.tj * T + c)] =
                    p.values[static_cast<std::size_t>(r) * T + c];
    }
    REQUIRE(stored == static_cast<std::size_t>(size / T) * (size / T));
    return out;
}

}  // namespace

TEST_CASE("csp_config_from maps a deployment's device onto the CSP executor",
          "[timing][csp][spec]") {
    SECTION("T4: one controller, eight engines, the hosted DRAM") {
        const auto c = csp_config_from(device("kpu_t4.json"));
        REQUIRE(c);
        CHECK(c->config.num_memory_controllers == 1);
        CHECK(c->config.num_dma_engines == 8);
        CHECK(c->config.dma_engine_controller == std::vector<std::size_t>(8, 0));
        CHECK(c->config.dram.has_value());
        CHECK(c->config.l3_buffer_count == 128);
        CHECK(c->config.num_block_movers == 4);
        CHECK(c->config.num_row_streamers == 1);
        CHECK(c->config.num_col_streamers == 1);
    }
    SECTION("T64: engines numbered per controller, like the floorplan") {
        const auto c = csp_config_from(device("kpu_t64.json"));
        REQUIRE(c);
        CHECK(c->config.num_memory_controllers == 4);
        CHECK(c->config.num_dma_engines == 32);
        for (std::size_t i = 0; i < 32; ++i) CHECK(c->config.dma_engine_controller[i] == i / 8);
        CHECK(c->config.l3_buffer_count == 2016);
        CHECK(c->config.num_block_movers == 112);
        CHECK(c->config.num_row_streamers + c->config.num_col_streamers == 32);
    }
    SECTION("what it does not map, it says") {
        const auto c = csp_config_from(device("kpu_t16.json"));
        REQUIRE(c);
        bool rates = false, l2 = false;
        for (const auto& u : c->unmapped) {
            rates = rates || u.find("bytes_per_cycle") != std::string::npos;
            l2 = l2 || u.find("l2.banks_per_tile") != std::string::npos;
        }
        CHECK(rates);
        CHECK(l2);
    }
    SECTION("refused with the reason") {
        std::string why;
        DeviceSpecification d = device("kpu_t16.json");
        d.dma.engines = 15;
        CHECK_FALSE(csp_config_from(d, &why));
        CHECK_THAT(why, ContainsSubstring("does not split evenly over 2 memory controllers"));
        d = device("kpu_t16.json");
        d.dma.engines = 0;
        CHECK_FALSE(csp_config_from(d, &why));
        CHECK_THAT(why, ContainsSubstring("dma.engines is zero"));
        d = device("kpu_t16.json");
        d.l3.capacity_tiles = 0;
        CHECK_FALSE(csp_config_from(d, &why));
        CHECK_THAT(why, ContainsSubstring("l3.capacity_tiles"));
    }
}

TEST_CASE("the executor puts each engine on the controller it was given", "[timing][csp]") {
    ConcurrentTimingExecutor::Config c;
    c.num_memory_controllers = 2;
    c.num_dma_engines = 4;
    c.dma_engine_controller = {0, 0, 1, 2};
    CHECK_THROWS_WITH(ConcurrentTimingExecutor(c), ContainsSubstring("names controller 2"));
    c.dma_engine_controller = {0, 1};
    CHECK_THROWS_WITH(ConcurrentTimingExecutor(c), ContainsSubstring("names 2 engines"));

    // Obeyed, not just validated: both engines on controller 1, so controller 0 moves nothing
    // (round-robin would have given engine 0 to controller 0).
    c.num_dma_engines = 2;
    c.dma_engine_controller = {1, 1};
    ConcurrentTimingExecutor exec(c);
    for (int e = 0; e < 2; ++e) {
        TileDescriptor t;
        t.tile_id = TileID{MatrixID::A, static_cast<Size>(e), 0, 0};
        t.dram_address = 0x10000 * static_cast<Address>(e + 1);
        t.size_bytes = 4096;
        exec.schedule_load(t, e);
    }
    while (!exec.is_complete() && exec.current_cycle() < 100000) exec.step();
    REQUIRE(exec.is_complete());
    CHECK(exec.memory_controller(0).total_bytes_transferred() == 0);
    CHECK(exec.memory_controller(1).total_bytes_transferred() == 2 * 4096);
}

TEST_CASE("Values oracle: a matmul on T4, T16 and T64 computes the L0 reference's values",
          "[timing][csp][oracle]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
        CAPTURE(file);
        const OracleRun r = run_matmul(device(file), 128, 32);
        const auto cmp = sw::kpu::program::compare_within(
            r.csp, r.reference, sw::kpu::program::kAtolFloat32, sw::kpu::program::kRtolMatmul);
        CAPTURE(cmp.max_abs, cmp.max_rel, cmp.failures, r.cycles);
        CHECK(cmp.pass);
        // fill()'s inputs are exactly representable and their products and partial sums stay
        // exact, so accumulation order cannot round: the CSP result is the reference's, bit
        // for bit. A change that breaks this has changed the arithmetic, not the timing.
        CHECK(cmp.bit_identical);
        CHECK(r.l3_free_after == r.l3_capacity);        // every L3 credit came home
        std::printf("csp oracle %-14s 128^3/32^3: %8llu cycles, max abs %.3g, max rel %.3g\n",
                    file, static_cast<unsigned long long>(r.cycles), cmp.max_abs, cmp.max_rel);
    }
}

TEST_CASE("compare_within holds the ADR bar, and a non-finite value on either side fails",
          "[timing][csp][oracle]") {
    using sw::kpu::program::compare_within;
    const std::vector<float> ref = {1.0f, 0.0f, 100.0f};
    auto r = compare_within({1.0f, 0.0f, 100.0f}, ref, 1e-6, 1e-4);
    CHECK(r.pass);
    CHECK(r.bit_identical);
    r = compare_within({1.00005f, 5e-7f, 100.009f}, ref, 1e-6, 1e-4);   // inside atol + rtol|r|
    CHECK(r.pass);
    CHECK_FALSE(r.bit_identical);
    r = compare_within({1.0f, 0.0f, 100.02f}, ref, 1e-6, 1e-4);         // 0.02 > 1e-6 + 0.01
    CHECK_FALSE(r.pass);
    CHECK(r.first_failure == 2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(compare_within({nan, 0.0f, 100.0f}, ref, 1e-6, 1e-4).pass);
    CHECK_FALSE(compare_within({1.0f, 0.0f, 100.0f}, {nan, 0.0f, 100.0f}, 1e-6, 1e-4).pass);
    CHECK_FALSE(compare_within({1.0f}, {std::numeric_limits<float>::infinity()}, 1e-6, 1e-4).pass);
}
