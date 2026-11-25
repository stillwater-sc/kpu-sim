// ============================================================================
// tests/cycle/test_cycle_mflow_recorder.cpp
// The cycle core's memory-flow recorder, step 1 (docs/plans/cycle-mflow-and-debug-server.md):
// a hand-built run's rows, the DMA engine's waits as intervals, recording that changes nothing
// (I2), records whose totals are the counters' (I3), on a direct link (kpu_s1) and over the NoC
// (the T4). The bundles it writes are checked by tools/trace/mflow_check.py (CTest
// cycle_mflow_check_*).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/cycle/memory/memory_subsystem.hpp>
#include <sw/kpu/cycle/record/mflow_recorder.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/memory_flow_record.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
namespace rec = sw::kpu::program::record;
using MFR = rec::MemoryFlowRecord;

namespace {

platform::DeviceSpecification device(const std::string& file) {
    return platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file).device(0);
}

CspL3Plan plan_for(const platform::DeviceSpecification& dev, unsigned size, unsigned tile) {
    csp::gen::Options g;
    g.size = size;
    g.tile = tile;
    g.l3 = 9;
    g.dtype = "int8";
    const lang::Program ast = lang::parse(csp::gen::generate(g));
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const auto in = driver::csp_inputs(ast);
    const auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    REQUIRE(!lb.skipped);
    return plan_l3(p, in, lb.values);
}

MflowSetup setup_for(const std::string& name, const MemoryRunOptions& o) {
    return MflowSetup{name, o.window, o.window, o.store_buffer_beats, lpddr5_one_channel()};
}

struct Recorded {
    std::size_t window = 0, write_window = 0;
    MemoryRun run;
    MFR record;
    std::vector<MflowRecorder::Wait> waits;
    std::vector<MflowRecorder::Action> actions;
};

Recorded run_recorded(const CspL3Plan& plan, const platform::DeviceSpecification* dev, MemoryRunOptions o,
                      const std::string& name) {
    MflowRecorder recorder(setup_for(name, o));
    o.events = &recorder;
    Recorded out;
    out.window = o.window;
    out.write_window = o.window;   // run_memory_subsystem gives the engine both windows the same
    out.run = run_memory_subsystem(plan, dev, o);
    out.record = recorder.record(out.run.end);
    out.waits = recorder.waits();
    out.actions = recorder.actions();
    return out;
}

std::size_t count(const MFR& r, MFR::CommandKind k) {
    return static_cast<std::size_t>(std::count_if(r.commands.begin(), r.commands.end(),
                                                  [&](const MFR::Command& c) { return c.kind == k; }));
}

// What every finished run's record must satisfy, and agree on with the counters (I3).
void check_record(const Recorded& x, const CspL3Plan& plan) {
    const MFR& r = x.record;
    const MemoryRun& run = x.run;
    REQUIRE(run.finished);
    // Bursts: as many as the engine sent, each served by exactly one CAS, its times in order.
    std::size_t loads = 0, stores = 0;
    for (const auto& b : r.bursts) (b.is_load ? loads : stores)++;
    CHECK(loads == run.dma.read_bursts);
    CHECK(stores == run.dma.write_bursts);
    std::vector<int> served(r.bursts.size(), 0);
    for (const auto& c : r.commands)
        if ((c.kind == MFR::CommandKind::Read || c.kind == MFR::CommandKind::Write) && c.burst != rec::kNone)
            ++served.at(c.burst);
    CHECK(std::all_of(served.begin(), served.end(), [](int n) { return n == 1; }));
    std::array<std::uint64_t, 4> outcomes{};
    for (const auto& b : r.bursts) {
        CHECK(b.posted <= b.submitted);
        CHECK(b.submitted <= b.first_command);
        CHECK(b.first_command <= b.data_start);
        CHECK(b.data_start <= b.data_end);
        CHECK(b.data_end <= b.done);
        CHECK(b.done <= r.makespan);
        CHECK(b.request != rec::kNone);
        ++outcomes[static_cast<std::size_t>(b.outcome)];
    }
    CHECK(outcomes[static_cast<std::size_t>(MFR::Outcome::Unknown)] == 0);
    CHECK(outcomes[static_cast<std::size_t>(MFR::Outcome::Hit)] == run.dram.page_hits);
    CHECK(outcomes[static_cast<std::size_t>(MFR::Outcome::Empty)] == run.dram.page_empty);
    CHECK(outcomes[static_cast<std::size_t>(MFR::Outcome::Conflict)] == run.dram.page_conflicts);
    // Commands: the controller's reads, writes and refreshes.
    CHECK(count(r, MFR::CommandKind::Read) == run.dram.reads);
    CHECK(count(r, MFR::CommandKind::Write) == run.dram.writes);
    CHECK(count(r, MFR::CommandKind::Refresh) == run.dram.refreshes);
    // Requests: one per Load, Store and Fill, each with exactly the bursts its bytes span (a
    // Fill: none), in order, naming its action and slot.
    CHECK(r.requests.size() == plan.loads + plan.stores + plan.fills);
    std::vector<std::uint32_t> of(r.requests.size(), 0);
    for (const auto& b : r.bursts) ++of.at(b.request);
    std::size_t fills = 0;
    for (std::size_t q = 0; q < r.requests.size(); ++q) {
        const auto& rq = r.requests[q];
        const bool fill = rq.kind == MFR::RequestKind::Fill;
        fills += fill ? 1 : 0;
        CHECK(of[q] == (fill ? 0u : rq.bytes / kL3BeatBytes));
        CHECK(rq.is_load == (rq.kind != MFR::RequestKind::Store));
        REQUIRE(rq.ticket < plan.program.size());
        CHECK(rq.slot == plan.program[rq.ticket].slot);
        CHECK(static_cast<int>(plan.program[rq.ticket].kind) == static_cast<int>(rq.kind) +
                  (fill ? 1 : 0));   // L3Action::Kind is Load, Store, Release, Fill
        CHECK(rq.offered <= rq.credit);
        CHECK(rq.credit <= rq.first_burst);
        CHECK(rq.first_burst <= rq.last_burst);
        CHECK(rq.last_burst <= rq.retired);
        CHECK(rq.retired <= r.makespan);
    }
    CHECK(fills == plan.fills);
    // Every action, Fills too, started and completed.
    CHECK(x.actions.size() == plan.loads + plan.stores + plan.fills);
    CHECK(std::all_of(x.actions.begin(), x.actions.end(), [](const auto& a) { return a.completed.has_value(); }));
    // Each window on its own: reads within `window`, writes within `write_window`, from the
    // engine's post to the burst's retirement (half-open, as mflow_check.py's M1 counts).
    auto peak = [&](bool load) {
        std::vector<std::pair<Cycle, int>> ev;
        for (const auto& b : r.bursts)
            if (b.is_load == load && b.done > b.posted) {
                ev.emplace_back(b.posted, 1);
                ev.emplace_back(b.done, -1);
            }
        std::sort(ev.begin(), ev.end());
        int level = 0, most = 0;
        for (const auto& e : ev) most = std::max(most, level += e.second);
        return static_cast<std::size_t>(most);
    };
    CHECK(peak(true) <= x.window);
    CHECK(peak(false) <= x.write_window);
    CHECK(peak(true) <= run.dma.max_reads_in_flight);
    // The store buffer within its depth.
    for (const auto& s : r.buffers) CHECK(s.held <= r.stations.back().capacity);
    // The waits: one cause's intervals never overlap, and add up to the engine's counts.
    std::array<std::uint64_t, kDmaWaits> total{};
    std::array<Cycle, kDmaWaits> last_end{};
    for (const auto& w : x.waits) {
        const auto k = static_cast<std::size_t>(w.why);
        CHECK(w.from < w.to);
        CHECK(w.from >= last_end[k]);
        last_end[k] = w.to;
        total[k] += w.to - w.from;
    }
    for (std::size_t k = 0; k < kDmaWaits; ++k) CHECK(total[k] == run.dma.waits[k]);
    // The record's waits are the recorder's, reason for reason.
    REQUIRE(r.waits.size() == x.waits.size());
    for (std::size_t i = 0; i < r.waits.size(); ++i) {
        CHECK(static_cast<std::size_t>(r.waits[i].reason) == static_cast<std::size_t>(x.waits[i].why));
        CHECK(r.waits[i].from == x.waits[i].from);
        CHECK(r.waits[i].to == x.waits[i].to);
    }
    // Both windows, and who recorded it.
    CHECK(r.source == "cycle-core");
    CHECK(r.window == x.window);
    CHECK(r.write_window == x.write_window);
    // The manifest: the timing table M5 needs, and the clock ratio.
    CHECK(r.ticks_per_cycle == 3.2);
    CHECK(std::any_of(r.dram_timing.begin(), r.dram_timing.end(), [](const auto& t) { return t.first == "tFAW"; }));
}

}  // namespace

TEST_CASE("A hand-built run's record: a Load, then a Store of its tile", "[cycle][mflow]") {
    CspL3Plan plan;
    plan.slots = 2;
    plan.slot_bytes = 256;
    plan.program = {l3_load(0, 0x0, 256), l3_store(0, 0x10000, 256), l3_release(0)};
    plan.loads = 1;
    plan.stores = 1;
    plan.releases = 1;
    const Recorded x = run_recorded(plan, nullptr, {}, "hand");
    REQUIRE(x.run.ok());
    check_record(x, plan);
    const MFR& r = x.record;
    REQUIRE(r.bursts.size() == 8);
    REQUIRE(r.requests.size() == 2);
    CHECK(r.requests[0].is_load);
    CHECK(!r.requests[1].is_load);
    // The Store starts after the Load: in the program's order.
    CHECK(r.requests[1].credit >= r.requests[0].credit);
    CHECK(r.requests[1].offered == r.requests[0].credit);
    // The first read opens its row: an empty page, after an ACTIVATE that names its burst.
    CHECK(r.bursts[0].outcome == MFR::Outcome::Empty);
    const auto act = std::find_if(r.commands.begin(), r.commands.end(),
                                  [](const auto& c) { return c.kind == MFR::CommandKind::Activate; });
    REQUIRE(act != r.commands.end());
    CHECK(act->burst == 0);
    CHECK(r.bursts[0].first_command == act->issue);
    // The Store's beats pass through the store buffer.
    CHECK(!r.buffers.empty());
    CHECK(std::any_of(r.buffers.begin(), r.buffers.end(), [](const auto& s) { return s.held > 0; }));
    CHECK(r.buffers.back().held == 0);
    // Stations: 16 banks, the bus, the engine (both windows), its store buffer, its waits.
    REQUIRE(r.stations.size() == 20);
    CHECK(r.stations[0].name == "hand/mc[0]/ch[0]/bg[0]/ba[0]");
    CHECK(r.stations[16].kind == "dram_bus");
    CHECK(r.stations[17].kind == "dma");
    CHECK(r.stations[17].capacity == 32);   // 16 reads and 16 writes in flight
    CHECK(r.stations[18].kind == "dmabuf");
    CHECK(r.stations[19].kind == "dmawait");
    CHECK(r.stations[19].capacity == kDmaWaits);
    // A CAS's exact data window is one burst of the controller's ticks.
    for (const auto& c : r.commands)
        if (c.kind == MFR::CommandKind::Read || c.kind == MFR::CommandKind::Write) {
            CHECK(c.tick_data_end > c.tick_data_start);
            CHECK(c.tick_data_start >= c.tick);
        }
}

TEST_CASE("The DMA engine's waits are intervals that add up to its counts", "[cycle][mflow]") {
    const auto dev = device("kpu_s1.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    MemoryRunOptions o;
    o.window = 1;   // one read and one write in flight: the window holds the engine up
    const Recorded x = run_recorded(plan, &dev, o, "kpu_s1");
    REQUIRE(x.run.ok());
    check_record(x, plan);
    CHECK(x.run.dma.waited(DmaWait::Window) > 0);
    CHECK(std::any_of(x.waits.begin(), x.waits.end(), [](const auto& w) { return w.why == DmaWait::Window; }));
    // An interval spans several cycles when a cause holds on: a run of cycles is one row.
    CHECK(std::any_of(x.waits.begin(), x.waits.end(), [](const auto& w) { return w.to - w.from > 1; }));
}

TEST_CASE("Recording changes nothing (I2)", "[cycle][mflow]") {
    const auto dev = device("kpu_s1.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    const MemoryRun bare = run_memory_subsystem(plan, &dev, {});
    const Recorded x = run_recorded(plan, &dev, {}, "kpu_s1");
    REQUIRE(bare.ok());
    REQUIRE(x.run.ok());
    CHECK(x.run.end == bare.end);
    CHECK(x.run.dma.waits == bare.dma.waits);
    CHECK(x.run.dma.read_bursts == bare.dma.read_bursts);
    CHECK(x.run.dram.data_bus_busy == bare.dram.data_bus_busy);
    CHECK(x.run.slots_checked == bare.slots_checked);
    CHECK(x.run.tiles_checked == bare.tiles_checked);
    // Every counter of every component and link, as the summary prints them.
    CHECK(format_memory_summary(plan, x.run) == format_memory_summary(plan, bare));
}

TEST_CASE("kpu_s1's record agrees with its counters, and is written and read back", "[cycle][mflow]") {
    const auto dev = device("kpu_s1.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    const Recorded x = run_recorded(plan, &dev, {}, "kpu_s1");
    REQUIRE(x.run.ok());
    check_record(x, plan);
    // C is filled from L-B: Fills are actions, never requests or bursts.
    CHECK(plan.fills > 0);
    // The bundle mflow_check.py checks (CTest cycle_mflow_check_s1).
    const std::string dir = std::string(KPU_BINARY_DIR) + "/cycle_mflow_s1.mflow";
    std::filesystem::remove_all(dir);
    rec::write_mflow(x.record, dir);
    const MFR back = rec::read_mflow(dir);
    CHECK(back.bursts.size() == x.record.bursts.size());
    CHECK(back.commands.size() == x.record.commands.size());
    CHECK(back.requests.size() == x.record.requests.size());
    CHECK(back.makespan == x.record.makespan);
    CHECK(back.source == "cycle-core");
    CHECK(back.write_window == x.record.write_window);
    REQUIRE(back.waits.size() == x.record.waits.size());
    CHECK(!back.waits.empty());
    for (std::size_t i = 0; i < back.waits.size(); ++i) {
        CHECK(back.waits[i].reason == x.record.waits[i].reason);
        CHECK(back.waits[i].from == x.record.waits[i].from);
        CHECK(back.waits[i].to == x.record.waits[i].to);
    }
    for (std::size_t q = 0; q < back.requests.size(); ++q) {
        CHECK(back.requests[q].kind == x.record.requests[q].kind);
        CHECK(back.requests[q].ticket == x.record.requests[q].ticket);
        CHECK(back.requests[q].slot == x.record.requests[q].slot);
    }
}

TEST_CASE("Version 4 reads version 3, and refuses a request whose kind and direction disagree", "[cycle][mflow]") {
    CspL3Plan plan;
    plan.slots = 2;
    plan.slot_bytes = 256;
    plan.program = {l3_load(0, 0x0, 256), l3_store(0, 0x10000, 256), l3_release(0)};
    plan.loads = plan.stores = plan.releases = 1;
    const Recorded x = run_recorded(plan, nullptr, {}, "hand");
    REQUIRE(x.run.ok());
    const std::string dir = std::string(KPU_BINARY_DIR) + "/cycle_mflow_v3.mflow";
    std::filesystem::remove_all(dir);
    rec::write_mflow(x.record, dir);
    // The same bundle declared version 3: the reader ignores v4's columns and its waits table, and
    // takes each request's kind from is_load.
    const std::string manifest = dir + "/manifest.json";
    std::string text;
    {
        std::ifstream in(manifest);
        text.assign(std::istreambuf_iterator<char>(in), {});
    }
    const std::string v4 = "\"version\": 4";
    REQUIRE(text.find(v4) != std::string::npos);
    {
        std::string v3 = text;
        v3.replace(v3.find(v4), v4.size(), "\"version\": 3");
        std::ofstream(manifest) << v3;
    }
    const MFR old = rec::read_mflow(dir);
    CHECK(old.source == "memory-side-harness");
    CHECK(old.write_window == 0);
    CHECK(old.waits.empty());
    REQUIRE(old.requests.size() == 2);
    CHECK(old.requests[0].kind == MFR::RequestKind::Load);
    CHECK(old.requests[1].kind == MFR::RequestKind::Store);
    CHECK(old.requests[1].ticket == rec::kNone);
    // Versions outside 3..4 are refused.
    {
        std::string v2 = text;
        v2.replace(v2.find(v4), v4.size(), "\"version\": 2");
        std::ofstream(manifest) << v2;
    }
    CHECK_THROWS_WITH(rec::read_mflow(dir), Catch::Matchers::ContainsSubstring("versions 3 to 4"));
    // The writer refuses a Store that says it is a load, and a Fill that says it is not.
    MFR bad = x.record;
    bad.requests[1].is_load = true;
    CHECK_THROWS_WITH(rec::write_mflow(bad, dir), Catch::Matchers::ContainsSubstring("of kind store"));
    bad = x.record;
    bad.requests[0].kind = MFR::RequestKind::Fill;
    bad.requests[0].is_load = false;
    CHECK_THROWS_WITH(rec::write_mflow(bad, dir), Catch::Matchers::ContainsSubstring("of kind fill"));
}

TEST_CASE("The T4's record, over the NoC, agrees with its counters", "[cycle][mflow][noc]") {
    const auto dev = device("kpu_t4.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    const Recorded x = run_recorded(plan, &dev, {}, "kpu_t4");
    REQUIRE(x.run.ok());
    REQUIRE(x.run.noc);
    check_record(x, plan);
    const std::string dir = std::string(KPU_BINARY_DIR) + "/cycle_mflow_t4.mflow";
    std::filesystem::remove_all(dir);
    rec::write_mflow(x.record, dir);
}

TEST_CASE("The recorder keeps its own clocks: the run's domains are gone when it records", "[cycle][mflow]") {
    // run_memory_subsystem's domains belong to its Simulator, destroyed when it returns, before
    // record() is called. The recorder must not refer to them: free them, put other domains in
    // their place (the allocator reuses the blocks), and the record still has the run's clocks.
    MflowRecorder r(MflowSetup{"x", 8, 0, 4, lpddr5_one_channel()});
    auto kpu = std::make_unique<Domain>(Domain::from_ghz("kpu", 1.0));
    auto dram = std::make_unique<Domain>(Domain::from_ghz("dram", 3.2));
    r.clocks(*kpu, *dram);
    kpu.reset();
    dram.reset();
    auto other_a = std::make_unique<Domain>("other a", 7);
    auto other_b = std::make_unique<Domain>("other b", 11);
    const MFR rec = r.record(5000);   // 5 ns at 1 GHz
    CHECK(rec.ticks_per_cycle == 3.2);
    CHECK(rec.makespan == 5);
    // And so does a real run, recorded after run_memory_subsystem has returned.
    CspL3Plan plan;
    plan.slots = 1;
    plan.slot_bytes = 64;
    plan.program = {l3_load(0, 0x0, 64), l3_release(0)};
    plan.loads = plan.releases = 1;
    MemoryRunOptions o;
    MflowRecorder after(setup_for("after", o));
    o.events = &after;
    const MemoryRun run = run_memory_subsystem(plan, nullptr, o);
    auto reuse_a = std::make_unique<Domain>("reuse a", 13);
    auto reuse_b = std::make_unique<Domain>("reuse b", 17);
    REQUIRE(run.finished);
    const MFR x = after.record(run.end);
    CHECK(x.ticks_per_cycle == 3.2);
    CHECK(x.makespan == run.end / 1000);   // the KPU cycle the run ends in, at 1 GHz
}

TEST_CASE("The recorder refuses events it cannot place", "[cycle][mflow]") {
    MflowRecorder r(MflowSetup{"x", 8, 0, 4, lpddr5_one_channel()});
    CHECK_THROWS(r.record(0));   // no clocks
    Domain kpu = Domain::from_ghz("kpu", 1.0);
    Domain dram = Domain::from_ghz("dram", 3.2);
    r.clocks(kpu, dram);
    CHECK_THROWS(r.dram_accepted(0, 0, 1, 99, false, 0));   // a request no engine posted
    CHECK_THROWS(r.burst_retired(0, 0, 99));
    r.action_started(0, 0, 0, l3_load(0, 0, 64));
    r.burst_posted(0, 1000, 0, 1, 0, false);
    CHECK_THROWS(r.burst_posted(0, 2000, 0, 1, 0, false));  // posted twice
    CHECK_THROWS(r.burst_posted(1, 2000, 0, 2, 64, false)); // a second engine
}
