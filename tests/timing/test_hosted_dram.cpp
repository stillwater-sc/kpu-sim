// ============================================================================
// tests/timing/test_hosted_dram.cpp
// DRAM plan step 2: the CSP memory controller hosting the cycle-accurate LPDDR5 controller.
// A tile is its bursts, the data bus is a ceiling, banks matter, the clocks agree, refresh
// happens, and the controller's own invariant checker stays clean -- on the T4's declared DRAM.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <string>
#include <vector>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;
using sw::kpu::isa::MatrixID;
using sw::kpu::program::platform::DeviceSpecification;
using sw::kpu::program::platform::DramCoord;
using sw::kpu::program::platform::DramMapError;
using sw::kpu::program::platform::read_spec_file;
using Catch::Matchers::ContainsSubstring;

namespace {

DeviceSpecification device(const char* file) {
    return read_spec_file(std::string("tests/program/deploy/") + file).device(0);
}
DramHosting t4() { return DramHosting::of(device("kpu_t4.json")); }

MemoryControllerProcess hosted_mc(const DramHosting& h, unsigned id = 0) {
    MemoryControllerProcess::Config c;
    c.controller_id = id;
    c.clock_ghz = 1.0;
    c.hosted = h;
    return MemoryControllerProcess(c);
}

TileDescriptor tile_at(std::uint64_t addr, std::uint64_t bytes, Size ti = 0) {
    TileDescriptor t;
    t.tile_id.matrix = MatrixID::A;
    t.tile_id.ti = ti;
    t.dram_address = addr;
    t.size_bytes = static_cast<Size>(bytes);
    t.element_size = 4;
    return t;
}

// Run the controller until every submitted tile completed; the cycle each one finished.
std::vector<Cycle> run(MemoryControllerProcess& mc, const std::vector<TileDescriptor>& tiles) {
    for (const auto& t : tiles) REQUIRE(mc.submit_request(t, true, 0));
    std::vector<Cycle> done;
    for (Cycle c = 1; done.size() < tiles.size() && c < 1'000'000; ++c) {
        mc.tick(c);
        while (auto ct = mc.get_completed_transfer(0)) done.push_back(ct->complete_cycle);
    }
    REQUIRE(done.size() == tiles.size());
    return done;
}

} // namespace

TEST_CASE("the bridge runs the controller in its own clock", "[timing][dram][hosted]") {
    const DramHosting h = t4();
    DramBridge b(h, 0, 1.0);
    // LPDDR5X-8533: the controller clock is half the data rate, 4.2665 GHz, against 1 GHz.
    CHECK(b.ticks_per_cycle() == Catch::Approx(4.2665));
    CHECK_THAT(b.timing_note(), ContainsSubstring("DERIVED"));
    DramBridge slow(h, 0, 2.0);
    CHECK(slow.ticks_per_cycle() == Catch::Approx(4.2665 / 2));
}

TEST_CASE("a hosted tile is its bursts, and the data bus is a ceiling", "[timing][dram][hosted]") {
    const DramHosting h = t4();
    auto small = hosted_mc(h);
    const Cycle t_small = run(small, {tile_at(0, 64)}).front();

    auto big = hosted_mc(h);
    const Cycle t_big = run(big, {tile_at(0, 64 * 1024)}).front();
    CHECK(big.bridge()->stats().bursts == 1024);

    // 64 KiB over two x16 channels at 8533 MT/s (34.1 GB/s) takes at least 1920 ns -- the
    // legacy model finished any tile in one latency, whatever its size.
    CHECK(t_big >= 65536 * 1000 / (2 * 8533 * 2));
    CHECK(t_big > 50 * t_small);
    CHECK(big.bridge()->stats().violations == 0);
    CHECK(small.bridge()->stats().violations == 0);
}

TEST_CASE("banks matter: a row conflict in one bank costs more than two banks",
          "[timing][dram][hosted]") {
    const DramHosting h = t4();
    // Two one-page tiles (32 bursts each). Same bank, different rows: the second needs a
    // precharge and an activate. Different bank groups: only an activate.
    DramCoord a{}, same_bank_next_row{}, other_group{};
    same_bank_next_row.row = 1;
    other_group.bank_group = 1;
    const std::uint64_t page = 2048;

    auto conflict = hosted_mc(h);
    const auto tc = run(conflict, {tile_at(h.map.encode(a), page, 0),
                                   tile_at(h.map.encode(same_bank_next_row), page, 1)});
    auto spread = hosted_mc(h);
    const auto ts = run(spread, {tile_at(h.map.encode(a), page, 0),
                                 tile_at(h.map.encode(other_group), page, 1)});

    CHECK(h.map.flat_bank(h.map.decode(h.map.encode(same_bank_next_row))) ==
          h.map.flat_bank(h.map.decode(h.map.encode(a))));
    CHECK(conflict.row_misses() >= 1);
    CHECK(spread.row_misses() == 0);
    CHECK(tc.back() > ts.back());
}

TEST_CASE("refresh happens while the controller idles", "[timing][dram][hosted]") {
    auto mc = hosted_mc(t4());
    for (Cycle c = 1; c <= 20'000; ++c) mc.tick(c);
    CHECK(mc.bridge()->stats().refreshes > 0);
    CHECK(mc.bridge()->stats().violations == 0);
    CHECK(mc.is_complete());
}

TEST_CASE("a tile past the declared DRAM is refused, not wrapped", "[timing][dram][hosted]") {
    auto mc = hosted_mc(t4());
    CHECK_THROWS_WITH(mc.submit_request(tile_at((std::uint64_t{4} << 30) - 32, 64), true, 0),
                      ContainsSubstring("runs past the declared DRAM"));
}

TEST_CASE("a burst whose address names another controller is counted", "[timing][dram][hosted]") {
    const DramHosting h = DramHosting::of(device("kpu_t64.json"));
    REQUIRE(h.map.controllers() == 4);
    DramCoord on_mc1{};
    on_mc1.mc = 1;
    auto mc0 = hosted_mc(h, 0);
    run(mc0, {tile_at(h.map.encode(on_mc1), 64)});
    CHECK(mc0.bridge()->stats().misrouted == 1);
}

TEST_CASE("a DRAM the hosted controller cannot model is refused by name", "[timing][dram][hosted]") {
    DeviceSpecification d = device("kpu_t4.json");
    d.memory.dram->technology = "hbm3";
    CHECK_THROWS_WITH(DramHosting::of(d), ContainsSubstring("'hbm3' needs its own controller"));
    d = device("kpu_t4.json");
    d.memory.dram->ranks = 2;
    d.memory.dram->capacity_bytes *= 2;
    CHECK_THROWS_WITH(DramHosting::of(d), ContainsSubstring("one rank"));
    d = device("kpu_t4.json");
    d.memory.dram->burst_bytes = 128;
    d.memory.dram->page_bytes = 4096;
    CHECK_THROWS_WITH(DramHosting::of(d), ContainsSubstring("BL64"));
    d = device("kpu_t4.json");
    d.memory.dram.reset();
    CHECK_THROWS_AS(DramHosting::of(d), DramMapError);
}

TEST_CASE("a matmul runs to completion on the hosted DRAM", "[timing][dram][hosted][executor]") {
    MatMulScheduleGenerator::Config g;
    g.M = g.N = g.K = 64;
    g.Ti = g.Tj = g.Tk = 16;
    g.a_base = 0x00001000;
    g.b_base = 0x00100000;
    g.c_base = 0x00200000;
    MatMulScheduleGenerator gen(g);
    const auto schedule = gen.generate();
    REQUIRE(schedule.valid);

    auto config_for = [](bool hosted) {
        ConcurrentTimingExecutor::Config c;
        c.num_memory_controllers = 1;
        c.l3_buffer_count = 32;
        c.num_block_movers = 4;
        c.l2_bank_count = 64;
        c.num_row_streamers = 2;
        c.num_col_streamers = 2;
        c.max_cycles = 2'000'000;
        if (hosted) c.dram = t4();
        return c;
    };

    ConcurrentTimingExecutor legacy(config_for(false));
    ScheduleExecutor run_legacy(legacy);
    const auto rl = run_legacy.execute(schedule);
    REQUIRE(rl.success);

    ConcurrentTimingExecutor hosted(config_for(true));
    ScheduleExecutor run_hosted(hosted);
    const auto rh = run_hosted.execute(schedule);
    INFO("legacy " << rl.total_cycles << " cycles, hosted " << rh.total_cycles << " cycles");
    REQUIRE(rh.success);
    REQUIRE_FALSE(rh.livelock_detected);

    // Same work, every stage exactly once; only the time differs.
    const auto sl = legacy.get_statistics(), sh = hosted.get_statistics();
    CHECK(sh.tiles_loaded == sl.tiles_loaded);
    CHECK(sh.tiles_moved == sl.tiles_moved);
    CHECK(sh.tiles_fed == sl.tiles_fed);
    CHECK(sh.tiles_drained == sl.tiles_drained);
    CHECK(sh.tiles_writeback == sl.tiles_writeback);
    CHECK(rh.total_cycles != rl.total_cycles);

    const auto* mc = hosted.memory_controller(0).bridge();
    REQUIRE(mc != nullptr);
    CHECK(mc->stats().violations == 0);
    CHECK(mc->stats().bursts * 64 >= hosted.memory_controller(0).total_bytes_transferred());
}

TEST_CASE("the declared DRAM must have as many controllers as the executor",
          "[timing][dram][hosted][executor]") {
    ConcurrentTimingExecutor::Config c;
    c.num_memory_controllers = 1;
    c.dram = DramHosting::of(device("kpu_t64.json"));    // 4 controllers
    CHECK_THROWS_WITH(ConcurrentTimingExecutor(c), ContainsSubstring("4 memory controllers"));
}
