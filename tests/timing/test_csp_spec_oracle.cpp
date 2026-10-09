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
#include <sw/kpu/timing/schedule/schedule_dispatcher.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <cstdio>
#include <algorithm>
#include <set>
#include <limits>
#include <map>
#include <optional>
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
    std::size_t l3_capacity = 0, l3_free_after = 0, l3_tiles = 0;
    // With the NoC on (4b.4)
    std::size_t dram_loads = 0, dram_stores = 0, noc_delivered = 0, noc_ejected = 0, noc_handed = 0;
    std::size_t eject_stalls = 0, peak_ring_queue = 0, ring_queue_depth = 0;
    bool watchdog = false, buses_serial = true;
    std::map<NocDim, std::size_t> injected_per_port;
    // The compute fabric (system-schedule step 2): each compute's tile and [start, end).
    struct Compute { std::uint32_t cf; Cycle start, end; };
    std::vector<Compute> computes;
    std::size_t compute_tiles = 0;
    std::vector<Cycle> busy;
};

// C = A @ B (size^3, tile^3) on the CSP executor configured from `d`, and on the L0 reference
// with the same inputs.
OracleRun run_matmul(const DeviceSpecification& d, Size size, Size T, bool noc = false,
                     int32_t cf_tile = -1, std::optional<std::size_t> prefetch = std::nullopt) {
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
    if (noc) {
        const auto w = csp_noc_wiring(d, static_cast<double>(T) * T * 4, ec.dma_store_buffer_blocks, &why);
        INFO(why);
        REQUIRE(w);
        ec.noc = *w;
    }
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
    auto enqueue = [&](const ScheduleOperation& op) {
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
                TileDescriptor result = op.tile;
                result.cf_tile = cf_tile;
                exec.schedule_matmul_compute(result, spec);
                break;
            }
        }
    };
    // Every operation up front, or paced by the dispatcher (system-schedule step 3).
    ScheduleDispatcher dispatch(schedule.operations, prefetch.value_or(ScheduleDispatcher::kUnlimited));
    while (exec.current_cycle() < ec.max_cycles) {
        dispatch.release(exec, enqueue);
        if (dispatch.done() && exec.is_complete()) break;
        exec.step();
    }
    REQUIRE(dispatch.done());
    REQUIRE(exec.is_complete());

    OracleRun out;
    out.cycles = exec.current_cycle();
    for (const auto& e : exec.events()) {
        out.dram_loads += e.type == EventType::DMA_LOAD_COMPLETE;
        out.dram_stores += e.type == EventType::DMA_STORE_COMPLETE;
        if (e.type == EventType::COMPUTE_COMPLETE)
            out.computes.push_back({e.component_id, e.cycle, e.cycle + e.duration});
    }
    out.compute_tiles = exec.compute_tiles();
    for (std::size_t t = 0; t < out.compute_tiles; ++t) out.busy.push_back(exec.compute_tile_busy_cycles(t));
    if (const NocFabric* f = exec.noc()) {
        out.watchdog = f->watchdog_fired();
        out.ring_queue_depth = ec.noc->fabric.hub_buffer_blocks / 4;
        for (const auto& h : f->hubs()) {
            out.noc_delivered += h.delivered().size();
            out.peak_ring_queue = std::max(out.peak_ring_queue, h.stats().peak_ring_queue);
        }
        for (const auto& p : f->ports()) {
            out.noc_ejected += p.stats().ejected;
            out.noc_handed += p.written().size();
            out.eject_stalls += p.stats().eject_stall_cycles;
            out.buses_serial = out.buses_serial &&
                               p.stats().injection_busy_cycles == p.stats().injected * f->config().block_cycles;
            if (p.stats().injected) out.injected_per_port[p.id()] = p.stats().injected;
        }
    }
    out.l3_capacity = ec.l3_buffer_count;
    out.l3_free_after = exec.l3_credits_available();
    out.l3_tiles = exec.l3_tiles();
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
        // Two L3 tiles, each with the two movers on its edges that abut a compute tile.
        CHECK(c->config.l3_tiles == 2);
        CHECK(c->config.block_mover_l3_tile == std::vector<std::size_t>{0, 0, 1, 1});
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
        CHECK(c->config.l3_tiles == 32);
        std::vector<int> movers_per_tile(32, 0);
        for (std::size_t h : c->config.block_mover_l3_tile) ++movers_per_tile.at(h);
        for (int n : movers_per_tile) CHECK((n >= 2 && n <= 4));    // corner 2, edge 3, inner 4
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

TEST_CASE("Values oracle: a matmul on S1, T4, T16 and T64 computes the L0 reference's values",
          "[timing][csp][oracle]") {
    for (const char* file : {"kpu_s1.json", "kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
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
        CHECK(r.l3_tiles == csp_config_from(device(file))->config.l3_tiles);   // placed, not pooled
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

TEST_CASE("L3 placement: a tile lives in its home L3 tile, and only there", "[timing][csp][l3]") {
    ConcurrentTimingExecutor::Config c;
    c.l3_buffer_count = 8;
    c.l3_tiles = 2;
    c.num_block_movers = 2;
    c.block_mover_l3_tile = {0, 1};

    auto tile = [](Size ti, int home) {
        TileDescriptor t;
        t.tile_id = TileID{MatrixID::A, ti, 0, 0};
        t.dram_address = 0x1000 * (ti + 1);
        t.size_bytes = 1024;
        t.l3_tile = home;
        return t;
    };

    SECTION("a declared home is where it lands and whose credit it takes") {
        ConcurrentTimingExecutor exec(c);
        CHECK(exec.l3_tile_credits(0).capacity() == 4);
        CHECK(exec.l3_tile_credits(1).capacity() == 4);
        exec.schedule_load(tile(0, 1));
        while (!exec.is_complete() && exec.current_cycle() < 10000) exec.step();
        CHECK(exec.l3_tile_tag_cam(1).lookup(TileID{MatrixID::A, 0, 0, 0}));
        CHECK_FALSE(exec.l3_tile_tag_cam(0).lookup(TileID{MatrixID::A, 0, 0, 0}));
        CHECK(exec.l3_tile_credits(1).available() == 3);
        CHECK(exec.l3_tile_credits(0).available() == 4);
        CHECK_THROWS_WITH(exec.l3_credits(), ContainsSubstring("2 L3 tiles"));
    }
    SECTION("undeclared, every operation on a tile resolves the same default home") {
        ConcurrentTimingExecutor exec(c);
        const auto t = tile(3, -1);
        const uint32_t h = exec.home_l3_tile(t);
        exec.schedule_load(t);
        exec.schedule_move(t);      // moved by a mover of the same tile
        while (!exec.is_complete() && exec.current_cycle() < 10000) exec.step();
        CHECK(exec.l3_credits_available() == 8);
        CHECK(exec.block_mover_l3_tile(h) == h);
    }
    SECTION("one tile, one home: a conflicting declaration is refused") {
        ConcurrentTimingExecutor exec(c);
        exec.schedule_load(tile(0, 0));
        CHECK_THROWS_WITH(exec.schedule_move(tile(0, 1)), ContainsSubstring("a tile has one home"));
    }
    SECTION("a named mover must sit on the tile's home") {
        ConcurrentTimingExecutor exec(c);
        CHECK_THROWS_WITH(exec.schedule_move(tile(0, 1), false, 0),
                          ContainsSubstring("BlockMover 0 is on L3 tile 0"));
        CHECK_NOTHROW(exec.schedule_move(tile(1, 1), false, 1));
    }
    SECTION("every L3 tile needs a mover") {
        c.block_mover_l3_tile = {1, 1};
        CHECK_THROWS_WITH(ConcurrentTimingExecutor(c), ContainsSubstring("L3 tile 0 has no BlockMover"));
    }
}

TEST_CASE("NoC wired: loads and ejections cross the fabric, and values never move",
          "[timing][csp][oracle][noc]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
        CAPTURE(file);
        const OracleRun off = run_matmul(device(file), 128, 32, false);
        const OracleRun on = run_matmul(device(file), 128, 32, true);

        // When, never what (ADR 0002): bit-identical with and without the NoC, and to L0.
        CHECK(on.csp == off.csp);
        const auto cmp = sw::kpu::program::compare_within(
            on.csp, on.reference, sw::kpu::program::kAtolFloat32, sw::kpu::program::kRtolMatmul);
        CHECK(cmp.bit_identical);
        CHECK(on.l3_free_after == on.l3_capacity);

        // Every DRAM read crossed to its hub, and every store's ejection crossed to its port and
        // was handed into the engine's store buffer.
        CHECK(on.dram_loads > 0);
        CHECK(on.noc_delivered == on.dram_loads);
        CHECK(on.noc_ejected == on.dram_stores);
        CHECK(on.noc_handed == on.dram_stores);

        // TF-PORT-1 (one block per bus), TF-PORT-2 (the store buffer slot is reserved before the
        // block leaves L3, so no ejection waits on a full output queue), TF-HUB-1 and -2.
        CHECK(on.buses_serial);
        CHECK(on.eject_stalls == 0);
        CHECK(on.peak_ring_queue <= on.ring_queue_depth);
        CHECK_FALSE(on.watchdog);

        // The fabric costs time; it does not save any.
        CHECK(on.cycles >= off.cycles);
        std::printf("noc %-14s 128^3/32^3: %8llu cycles (off %llu), %zu loads delivered, "
                    "%zu ejected, ports used %zu\n",
                    file, static_cast<unsigned long long>(on.cycles),
                    static_cast<unsigned long long>(off.cycles), on.noc_delivered, on.noc_ejected,
                    on.injected_per_port.size());
    }

    // The T4: all eight engines attach to one port, so every load enters through one bus.
    const OracleRun t4 = run_matmul(device("kpu_t4.json"), 128, 32, true);
    REQUIRE(t4.injected_per_port.size() == 1);
    CHECK(t4.injected_per_port.begin()->second == t4.dram_loads);
}

TEST_CASE("csp_noc_wiring refuses a device the NoC cannot be wired on", "[timing][csp][noc]") {
    std::string why;
    DeviceSpecification d = device("kpu_t16.json");
    d.noc.reset();
    CHECK_FALSE(csp_noc_wiring(d, 4096, 2, &why));
    CHECK_THAT(why, ContainsSubstring("no noc"));

    d = device("kpu_t16.json");
    d.memory.controllers = 0;
    CHECK_FALSE(csp_noc_wiring(d, 4096, 2, &why));
    CHECK_THAT(why, ContainsSubstring("does not split evenly over 0 memory controllers"));
    d = device("kpu_t16.json");
    d.dma.engines = 15;
    CHECK_FALSE(csp_noc_wiring(d, 4096, 2, &why));
    CHECK_THAT(why, ContainsSubstring("does not split evenly over 2"));

    const auto w = csp_noc_wiring(device("kpu_t16.json"), 4096, 2, &why);
    REQUIRE(w);
    CHECK(w->engine_port.size() == 16);
    CHECK(w->fabric.output_queue_blocks == 2);
    CHECK(w->fabric.dma_write_latency == 0);
    std::size_t attached = 0;
    for (auto n : w->fabric.engines_per_port) attached += n;
    CHECK(attached == 16);
}


TEST_CASE("NoC wired: a load is in L3 only once its hub delivers it", "[timing][csp][noc]") {
    const DeviceSpecification d = device("kpu_t4.json");
    auto cfg = csp_config_from(d)->config;
    const double block = 32.0 * 32 * 4;
    cfg.noc = *csp_noc_wiring(d, block, cfg.dma_store_buffer_blocks);
    ConcurrentTimingExecutor exec(cfg);

    TileDescriptor t;
    t.tile_id = TileID{MatrixID::A, 0, 0, 0};
    t.dram_address = 0x100000;
    t.size_bytes = 32 * 32 * 4;
    t.l3_tile = 1;
    exec.schedule_load(t);
    while (!exec.is_complete() && exec.current_cycle() < 100000) exec.step();
    REQUIRE(exec.is_complete());

    Cycle read = 0, arrived = 0;
    for (const auto& e : exec.events()) {
        if (e.type == EventType::DMA_LOAD_COMPLETE) read = e.cycle + e.duration;
        if (e.type == EventType::TILE_ARRIVED_L3) arrived = e.cycle;
    }
    REQUIRE(read > 0);
    // At least one block time on the fold link after DRAM delivered it; never at MC completion.
    CHECK(arrived >= read + exec.noc()->config().block_cycles);
    CHECK(exec.l3_tile_tag_cam(1).lookup(t.tile_id));
    CHECK(exec.noc()->hubs()[1].delivered().size() == 1);
}

TEST_CASE("Burst window: values never move, with the NoC off and on", "[timing][csp][oracle][window]") {
    for (const char* file : {"kpu_t4.json", "kpu_t16.json", "kpu_t64.json"}) {
        CAPTURE(file);
        const OracleRun tiles = run_matmul(device(file), 128, 32, false);
        DeviceSpecification d = device(file);
        d.dma.window = 32;
        for (const bool noc : {false, true}) {
            CAPTURE(noc);
            const OracleRun w = run_matmul(d, 128, 32, noc);
            CHECK(w.csp == tiles.csp);
            CHECK(w.csp == w.reference);            // bit-identical: fill() keeps sums exact
            CHECK(w.l3_free_after == w.l3_capacity);
            std::printf("window %-14s noc=%d: %8llu cycles (tile-level, no NoC: %llu)\n", file,
                        noc ? 1 : 0, static_cast<unsigned long long>(w.cycles),
                        static_cast<unsigned long long>(tiles.cycles));
        }
    }
}

// docs/plans/system-schedule-debugger.md step 2: a compute tile runs one compute at a time, for
// as long as its MACs take at the spec's rate.
TEST_CASE("Compute fabric: one compute at a time per compute tile, latency from the MAC rate",
          "[timing][csp][compute]") {
    using Compute = OracleRun::Compute;
    auto no_overlap = [](std::vector<Compute> v, std::uint32_t cf) {
        std::vector<std::pair<Cycle, Cycle>> on;
        for (const auto& c : v) if (c.cf == cf) on.push_back({c.start, c.end});
        std::sort(on.begin(), on.end());
        for (std::size_t i = 1; i < on.size(); ++i)
            if (on[i].first < on[i - 1].second) return false;
        return true;
    };

    SECTION("kpu_s1: one large compute tile") {
        const DeviceSpecification d = device("kpu_s1.json");
        const OracleRun r = run_matmul(d, 128, 32);
        REQUIRE(r.compute_tiles == 1);
        REQUIRE(r.computes.size() == 16);                 // (128/32)^2 output tiles
        CHECK(no_overlap(r.computes, 0));
        // Each compute: fill (2 x 32) + 32 x 32 x 128 MACs at 8192 per cycle = 64 + 16.
        Cycle busy = 0;
        for (const auto& c : r.computes) {
            CHECK(c.cf == 0);
            CHECK(c.end - c.start == 64 + (32 * 32 * 128) / 8192);
            busy += c.end - c.start;
        }
        CHECK(r.busy.at(0) == busy);
        CHECK(sw::kpu::program::compare_within(r.csp, r.reference, sw::kpu::program::kAtolFloat32,
                                               sw::kpu::program::kRtolMatmul).bit_identical);
    }
    SECTION("kpu_t4: two compute tiles, both used, neither double-booked") {
        const OracleRun r = run_matmul(device("kpu_t4.json"), 128, 32);
        REQUIRE(r.compute_tiles == 2);
        CHECK(no_overlap(r.computes, 0));
        CHECK(no_overlap(r.computes, 1));
        std::set<std::uint32_t> used;
        for (const auto& c : r.computes) used.insert(c.cf);
        CHECK(used == std::set<std::uint32_t>{0, 1});
        // T4's tile is 4096 MACs per cycle: the same compute takes 64 + 32.
        for (const auto& c : r.computes) CHECK(c.end - c.start == 64 + (32 * 32 * 128) / 4096);
    }
    SECTION("a schedule that names a compute tile runs there") {
        const OracleRun r = run_matmul(device("kpu_t4.json"), 128, 32, false, 1);
        for (const auto& c : r.computes) CHECK(c.cf == 1);
        CHECK(no_overlap(r.computes, 1));
        CHECK(r.busy.at(0) == 0);
        CHECK(sw::kpu::program::compare_within(r.csp, r.reference, sw::kpu::program::kAtolFloat32,
                                               sw::kpu::program::kRtolMatmul).bit_identical);
    }
    SECTION("a compute tile the fabric does not have is refused by name") {
        auto c = csp_config_from(device("kpu_t4.json"))->config;
        ConcurrentTimingExecutor exec(c);
        TileDescriptor t;
        t.tile_id.matrix = MatrixID::C;
        t.cf_tile = 2;
        CHECK_THROWS_WITH(exec.schedule_compute(t, std::vector<TileID>{}),
                          ContainsSubstring("names compute tile 2; the fabric has 2"));
    }
}

// Review of step 2: K comes from the A inputs' widths as fed, so a non-square schedule must give
// A its true shape (Ti x Tk), not Ti x Tj.
TEST_CASE("Compute fabric: K is the A tiles' width, with non-square tiles",
          "[timing][csp][compute]") {
    MatMulScheduleGenerator::Config g;
    g.M = 64; g.N = 64; g.K = 64;
    g.Ti = 16; g.Tj = 32; g.Tk = 8;
    const auto schedule = MatMulScheduleGenerator(g).generate();
    REQUIRE(schedule.valid);
    for (const auto& op : schedule.operations) {
        const auto& t = op.tile;
        if (t.tile_id.matrix == MatrixID::A) { CHECK(t.height == 16); CHECK(t.width == 8); CHECK(t.size_bytes == 16 * 8 * 4); }
        if (t.tile_id.matrix == MatrixID::B) { CHECK(t.height == 8); CHECK(t.width == 32); CHECK(t.size_bytes == 8 * 32 * 4); }
        if (t.tile_id.matrix == MatrixID::C) { CHECK(t.height == 16); CHECK(t.width == 32); CHECK(t.size_bytes == 16 * 32 * 4); }
    }
    // A[ti, tk] tiles are packed by their own size: consecutive k tiles are 512 bytes apart.
    std::map<std::pair<Size, Size>, Address> a_at;
    for (const auto& op : schedule.operations)
        if (op.tile.tile_id.matrix == MatrixID::A) a_at[{op.tile.tile_id.ti, op.tile.tile_id.tk}] = op.tile.dram_address;
    CHECK(a_at.at({0, 1}) - a_at.at({0, 0}) == 16 * 8 * 4);

    // On the fabric, each compute is 16 x 32 x 64 MACs: fill 2 x 32, then 32768 / 8192.
    auto c = csp_config_from(device("kpu_s1.json"))->config;
    ConcurrentTimingExecutor exec(c);
    ScheduleExecutor run(exec);
    REQUIRE(run.execute(schedule).success);
    std::size_t computes = 0;
    for (const auto& e : exec.events())
        if (e.type == EventType::COMPUTE_COMPLETE) {
            ++computes;
            CHECK(e.duration == 64 + (16 * 32 * 64) / 8192);
        }
    CHECK(computes == (64 / 16) * (64 / 32));
}

// docs/plans/system-schedule-debugger.md step 3: pacing changes when operations are released,
// never what is computed.
TEST_CASE("Dispatcher: values never move with the prefetch depth", "[timing][csp][oracle][dispatcher]") {
    for (const char* file : {"kpu_s1.json", "kpu_t4.json"})
        for (std::size_t P : {std::size_t{0}, std::size_t{1}, std::size_t{4}}) {
            CAPTURE(file, P);
            const OracleRun r = run_matmul(device(file), 128, 32, false, -1, P);
            const auto cmp = sw::kpu::program::compare_within(
                r.csp, r.reference, sw::kpu::program::kAtolFloat32, sw::kpu::program::kRtolMatmul);
            CHECK(cmp.bit_identical);
            CHECK(r.l3_free_after == r.l3_capacity);
        }
}
