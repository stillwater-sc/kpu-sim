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
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/memory_controller_process.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>

#include <algorithm>
#include <map>
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

// ============================================================================
// DRAM plan step 3: the DMA burst window
// ============================================================================
namespace {

struct Stream {
    Cycle cycles = 0;
    double bytes_per_cycle = 0;
    std::size_t max_in_flight = 0;          // bursts, over all engines and cycles
    bool credit_before_first_burst = true;
};

// `n` distinct 4 KiB loads through the T4's hosted controller, from `engines` engines with a
// burst window of `window` (0 = tile-level requests).
Stream stream(std::size_t engines, std::size_t window, std::size_t n = 128) {
    auto c = csp_config_from(device("kpu_t4.json"))->config;
    c.num_dma_engines = engines;
    c.dma_engine_controller.assign(engines, 0);
    c.dma_window = window;
    c.l3_buffer_count = 4096;
    c.l3_tiles = 1;
    c.block_mover_l3_tile.clear();
    c.num_block_movers = 1;
    ConcurrentTimingExecutor ex(c);
    const Size T = 4096;
    for (std::size_t i = 0; i < n; ++i) {
        TileDescriptor t;
        t.tile_id = TileID{MatrixID::A, static_cast<Size>(i), 0, 0};
        t.dram_address = 0x100000 + static_cast<sw::kpu::timing::Address>(i) * T;
        t.size_bytes = T;
        ex.schedule_load(t, static_cast<int>(i % engines));
    }
    Stream s;
    while (!ex.is_complete() && ex.current_cycle() < 5'000'000) {
        ex.step();
        for (std::size_t e = 0; e < engines; ++e)
            s.max_in_flight = std::max(s.max_in_flight, ex.dma_engine(e).bursts_in_flight());
    }
    REQUIRE(ex.is_complete());
    s.cycles = ex.current_cycle();
    s.bytes_per_cycle = static_cast<double>(n) * T / static_cast<double>(s.cycles);
    // Per tile: its L3 credit is taken before its first burst goes out (plan §2).
    std::map<std::string, Cycle> credit;
    for (const auto& e : ex.events()) {
        if (e.type == EventType::CREDIT_ACQUIRED) credit.emplace(e.tile_id.to_string(), e.cycle);
        if (e.type == EventType::DMA_LOAD_START) {
            auto it = credit.find(e.tile_id.to_string());
            s.credit_before_first_burst = s.credit_before_first_burst && it != credit.end() &&
                                          it->second <= e.cycle;
        }
    }
    return s;
}

// The controller's data-bus ceiling in bytes per executor cycle: channels x width x rate.
double peak_bytes_per_cycle() {
    const DeviceSpecification d = device("kpu_t4.json");    // held: `m` points into it
    const auto& m = *d.memory.dram;
    return static_cast<double>(m.channels) * (m.channel_width_bits / 8.0) * m.data_rate_mtps * 1e6 / 1e9;
}

}  // namespace

TEST_CASE("Burst window: one engine with W = 32 saturates the controller; 32 with W = 1 do not",
          "[timing][dram][hosted][window]") {
    // The plan's row (dram-bank-model.md step 3). Both have 32 bursts in flight, so Little's
    // law alone does not separate them: locality does. One engine streams a tile's bursts
    // through open rows; thirty-two single-burst engines scatter over rows, and FR-FCFS has
    // nothing to reorder.
    const double peak = peak_bytes_per_cycle();
    const Stream one = stream(1, 32);
    const Stream many = stream(32, 1);
    CAPTURE(peak, one.bytes_per_cycle, many.bytes_per_cycle);
    CHECK(one.bytes_per_cycle >= 0.90 * peak);
    CHECK(many.bytes_per_cycle < 0.80 * peak);
    CHECK(one.max_in_flight <= 32);
    CHECK(many.max_in_flight <= 1);
}

TEST_CASE("Burst window: bandwidth grows with W until the latency is covered, and W bounds it",
          "[timing][dram][hosted][window]") {
    double last = 0;
    for (std::size_t w : {1u, 2u, 4u, 8u, 16u, 32u}) {
        CAPTURE(w);
        const Stream s = stream(1, w);
        CHECK(s.max_in_flight <= w);
        CHECK(s.max_in_flight == w);                // and the engine does fill its window
        CHECK(s.bytes_per_cycle >= last * 0.98);    // non-decreasing, within noise
        CHECK(s.credit_before_first_burst);
        last = s.bytes_per_cycle;
    }
    // A single-burst engine is latency-bound: a small fraction of the ceiling.
    CHECK(stream(1, 1).bytes_per_cycle < 0.30 * peak_bytes_per_cycle());
}

TEST_CASE("Burst window: a window needs a hosted controller", "[timing][dram][window]") {
    ConcurrentTimingExecutor::Config c;
    c.dma_window = 8;                               // no `dram`: the legacy controller
    CHECK_THROWS_WITH(ConcurrentTimingExecutor(c), ContainsSubstring("needs a hosted DRAM"));
}


TEST_CASE("Burst window: a tile completes with its last burst, not its first",
          "[timing][dram][hosted][window]") {
    auto c = csp_config_from(device("kpu_t4.json"))->config;
    c.num_dma_engines = 1;
    c.dma_engine_controller = {0};
    c.dma_window = 32;
    c.l3_tiles = 1;
    c.block_mover_l3_tile.clear();
    c.num_block_movers = 1;
    ConcurrentTimingExecutor ex(c);
    TileDescriptor t;
    t.tile_id = TileID{MatrixID::A, 0, 0, 0};
    t.dram_address = 0x100000;
    t.size_bytes = 4096;                            // 64 bursts of 64 B
    ex.schedule_load(t, 0);
    while (!ex.is_complete() && ex.current_cycle() < 100000) ex.step();
    REQUIRE(ex.is_complete());

    Cycle start = 0, end = 0;
    std::size_t completions = 0;
    for (const auto& e : ex.events())
        if (e.type == EventType::DMA_LOAD_COMPLETE) {
            ++completions;
            start = e.cycle;
            end = e.cycle + e.duration;
        }
    CHECK(completions == 1);                        // one tile, one completion
    // Its 4 KiB cannot cross the data bus faster than the controller's ceiling.
    CHECK(static_cast<double>(end - start) >= 4096.0 / peak_bytes_per_cycle());
    CHECK(ex.memory_controller(0).total_bytes_transferred() == 4096);
}

// ============================================================================
// Memory-side debugger step 1: every command and every burst, observed
// ============================================================================
namespace {

using Cmd = DramBridge::Command;
using Outcome = MemoryControllerProcess::PageOutcome;

// One 64 B burst at `addr`, run to completion on its own.
void one_burst(MemoryControllerProcess& mc, std::uint64_t addr, Cycle& clock) {
    REQUIRE(mc.submit_burst(tile_at(addr, 64), 0, true, 0));
    for (int i = 0; i < 100000; ++i) {
        mc.tick(++clock);
        if (mc.get_completed_burst(0)) return;
    }
    FAIL("burst never completed");
}

}  // namespace

TEST_CASE("Recording: a burst's commands, coordinates and page outcome are observed",
          "[timing][dram][hosted][record]") {
    MemoryControllerProcess::Config c;
    c.clock_ghz = 1.0;
    c.hosted = t4();
    c.record = true;
    MemoryControllerProcess mc(c);
    const auto& map = c.hosted->map;

    DramCoord a{};                      // channel 0, bank group 1, bank 2, row 5, column 0
    a.bank_group = 1;
    a.bank = 2;
    a.row = 5;
    DramCoord b = a;                    // the same row: a page hit
    b.col = 1;
    DramCoord k = a;                    // another row in the same bank: a page conflict
    k.row = 9;

    Cycle clock = 0;
    one_burst(mc, map.encode(a), clock);
    one_burst(mc, map.encode(b), clock);
    one_burst(mc, map.encode(k), clock);

    const auto& bursts = mc.recorded_bursts();
    REQUIRE(bursts.size() == 3);
    CHECK(bursts[0].coord == a);
    CHECK(bursts[1].coord == b);
    CHECK(bursts[2].coord == k);
    CHECK(bursts[0].outcome == Outcome::Empty);
    CHECK(bursts[1].outcome == Outcome::Hit);
    CHECK(bursts[2].outcome == Outcome::Conflict);
    for (const auto& r : bursts) {
        CHECK(r.finished);
        CHECK(r.commanded);
        CHECK(r.submitted <= r.first_command);
        CHECK(r.first_command <= r.data_start);
        CHECK(r.data_start < r.data_end);
        CHECK(r.data_end <= r.done + 1);
    }

    // The command chain: ACT row 5, RD; RD (no ACT on a hit); PRE row 5, ACT row 9, RD.
    std::vector<std::pair<Cmd::Kind, std::uint64_t>> chain;
    Cycle last = 0;
    for (const auto& cmd : mc.recorded_commands()) {
        if (cmd.kind == Cmd::Kind::Refresh) continue;
        CHECK(cmd.issue >= last);                       // issued in order
        last = cmd.issue;
        CHECK(cmd.channel == a.channel);
        CHECK(cmd.bank_group == a.bank_group);
        CHECK(cmd.bank == a.bank);
        chain.emplace_back(cmd.kind, cmd.row);
    }
    const std::vector<std::pair<Cmd::Kind, std::uint64_t>> want = {
        {Cmd::Kind::Activate, 5}, {Cmd::Kind::Read, 5}, {Cmd::Kind::Read, 5},
        {Cmd::Kind::Precharge, 5}, {Cmd::Kind::Activate, 9}, {Cmd::Kind::Read, 9}};
    CHECK(chain == want);
}

TEST_CASE("Recording is off by default and costs nothing", "[timing][dram][hosted][record]") {
    MemoryControllerProcess mc = hosted_mc(t4());
    Cycle clock = 0;
    one_burst(mc, 0x1000, clock);
    CHECK(mc.recorded_bursts().empty());
    CHECK(mc.recorded_commands().empty());
}
