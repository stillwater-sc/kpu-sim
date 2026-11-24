// ============================================================================
// tests/cycle/test_cycle_lpddr5.cpp
// The LPDDR5 controller as a cycle-core component, step 1a (#537;
// docs/plans/cycle-based-kpu-prototype.md §3.4, §4.1).
//
// PORTED, NOT REWRITTEN, is tested as an equivalence: each workload of the controller's pattern
// suite (patterns/memory/lpddr5/common/workloads.hpp) runs through the channel on the core, and
// again on the bare controller, each request submitted at the cycle the channel accepted it. Every
// request completes at the same cycle in both, with the same statistics and trace, and with the
// controller's invariants checked every cycle. Around that: the values the channel keeps, its
// bound, backpressure, its counters, and a channel fed across a clock-domain crossing.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "dram_pipeline.hpp"

#include <lpddr5_configs.hpp>
#include <workloads.hpp>

#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
namespace pl = sw::kpu::patterns::lpddr5;
using Catch::Matchers::ContainsSubstring;
using Controller = Lpddr5Channel::Controller;

namespace {

constexpr Picoseconds kLimit = 1'000'000'000;   // 1 ms

std::vector<Requester::Item> items_of(const pl::Workload& w) {
    std::vector<Requester::Item> v;
    std::uint64_t id = 0;
    for (const auto& a : w.accesses) {
        Requester::Item it;
        it.request.id = id++;
        it.request.write = a.type == pl::AccessType::WRITE;
        it.request.address = a.address;
        it.request.size = a.size;
        if (it.request.write) it.request.bytes.assign(a.size, static_cast<std::uint8_t>(0xA0 + (it.request.id & 0x0F)));
        it.request.tag = 1000 + it.request.id;
        it.at = a.relative_cycle;
        v.push_back(std::move(it));
    }
    return v;
}

// Requester -> channel -> sink, all in the DRAM domain (3.2 GHz) unless `requester_ghz` differs:
// then the requester and the sink are in their own domain, across two crossings.
struct Rig {
    Simulator sim;
    Domain& dram;
    Domain* host = nullptr;
    Requester* req = nullptr;
    Lpddr5Channel* ch = nullptr;
    Sink* sink = nullptr;
    Link<DramRequest>* in = nullptr;
    Link<DramResponse>* out = nullptr;
    Rig(std::vector<Requester::Item> items, const Controller::Config& cfg = pl::single_channel_config(),
        double requester_ghz = 0.0, std::size_t in_cap = 4, std::size_t out_cap = 4)
        : dram(sim.add_domain(Domain::from_ghz("dram", 3.2))) {
        host = requester_ghz > 0.0 ? &sim.add_domain(Domain::from_ghz("host", requester_ghz)) : &dram;
        req = &sim.add<Requester>("requester", *host, std::move(items));
        ch = &sim.add<Lpddr5Channel>("lpddr5", dram, cfg);
        sink = &sim.add<Sink>("sink", *host);
        if (host == &dram) {
            in = &sim.link<DramRequest>("requests", *req, *ch, in_cap);
            out = &sim.link<DramResponse>("responses", *ch, *sink, out_cap);
        } else {
            in = &sim.crossing<DramRequest>("requests", *req, *ch, in_cap);
            out = &sim.crossing<DramResponse>("responses", *ch, *sink, out_cap);
        }
        req->connect(*in);
        ch->connect(*in, *out);
        sink->connect(*out);
    }
};

// The bare controller, each request submitted at the cycle the channel accepted it: the cycle each
// request (by the channel's response id) completes at.
std::map<std::uint64_t, std::uint64_t> replay(const Controller::Config& cfg, const std::vector<Requester::Item>& items,
                                              const std::vector<DramResponse>& got, Controller& mc) {
    std::map<std::uint64_t, Cycle> accepted;
    for (const auto& r : got) accepted[r.id] = r.accepted;
    std::multimap<Cycle, const DramRequest*> by_cycle;
    for (const auto& it : items) by_cycle.emplace(accepted.at(it.request.id), &it.request);
    std::map<std::uint64_t, std::uint64_t> done;
    (void)cfg;
    auto next = by_cycle.begin();
    while (next != by_cycle.end() || mc.has_pending()) {
        for (; next != by_cycle.end() && next->first == mc.current_cycle(); ++next) {
            const DramRequest& q = *next->second;
            const std::uint64_t id = q.id;
            auto cb = [&done, &mc, id] { done[id] = mc.current_cycle(); };
            const auto ok = q.write ? mc.submit_write(q.address, q.bytes.data(), static_cast<std::uint32_t>(q.bytes.size()), cb)
                                    : mc.submit_read(q.address, q.size, cb);
            REQUIRE(ok.has_value());
        }
        REQUIRE((next == by_cycle.end() || next->first >= mc.current_cycle()));
        mc.tick();
        REQUIRE(mc.current_cycle() < 10'000'000);
    }
    return done;
}

}  // namespace

TEST_CASE("Every pattern workload runs on the core exactly as on the bare controller", "[cycle][lpddr5]") {
    for (const auto& w : pl::all_workloads()) {
        CAPTURE(w.name);
        const auto items = items_of(w);
        Rig rig(items);
        rig.ch->enable_dram_trace(true);
        REQUIRE(rig.sim.run_until_idle(kLimit));
        REQUIRE(rig.sink->received.size() == items.size());

        Controller mc(pl::single_channel_config());
        mc.enable_invariants(true);
        mc.enable_tracing(true);
        const auto done = replay(pl::single_channel_config(), items, rig.sink->received, mc);
        CHECK_FALSE(mc.has_violations());
        for (const auto& r : rig.sink->received) {
            CAPTURE(r.id);
            CHECK(r.completed == done.at(r.id));
            CHECK(r.tag == 1000 + r.id);
        }
        const auto& a = rig.ch->controller().lpddr5_stats();
        const auto& b = mc.lpddr5_stats();
        CHECK(a.reads == b.reads);
        CHECK(a.writes == b.writes);
        CHECK(a.page_hits == b.page_hits);
        CHECK(a.page_empty == b.page_empty);
        CHECK(a.page_conflicts == b.page_conflicts);
        CHECK(a.refreshes == b.refreshes);
        CHECK(rig.ch->controller().trace_entries().size() == mc.trace_entries().size());
        // The pattern suite's own expected page outcomes, where it states them.
        if (w.expected.page_empty + w.expected.page_hits + w.expected.page_conflicts > 0) {
            CHECK(a.page_hits == w.expected.page_hits);
            CHECK(a.page_empty == w.expected.page_empty);
            CHECK(a.page_conflicts == w.expected.page_conflicts);
        }
    }
}

TEST_CASE("The controller suite's level-1 cases, through the channel", "[cycle][lpddr5]") {
    auto run = [](std::vector<std::uint64_t> addrs) {
        std::vector<Requester::Item> items;
        for (std::uint64_t i = 0; i < addrs.size(); ++i) {
            Requester::Item it;
            it.request.id = i;
            it.request.address = addrs[i];
            items.push_back(it);
        }
        Rig rig(items);
        REQUIRE(rig.sim.run_until_idle(kLimit));
        return rig.ch->dram();
    };
    SECTION("sixteen reads of one row: one page empty, fifteen hits") {
        std::vector<std::uint64_t> a;
        for (std::uint32_t i = 0; i < 16; ++i) a.push_back(pl::make_address(0, 100, i));
        const auto d = run(a);
        CHECK(d.reads == 16);
        CHECK(d.page_empty == 1);
        CHECK(d.page_hits == 15);
    }
    SECTION("a second row in the same bank: a conflict") {
        const auto d = run({pl::make_address(0, 100, 0), pl::make_address(0, 200, 0)});
        CHECK(d.page_empty == 1);
        CHECK(d.page_conflicts == 1);
    }
}

TEST_CASE("Values: writes land, reads see them in the order requests arrive, preloads persist", "[cycle][lpddr5]") {
    const std::uint64_t a = pl::make_address(2, 7, 0), b = pl::make_address(2, 9, 0);   // two rows: a conflict
    std::vector<Requester::Item> items(5);
    for (std::uint64_t i = 0; i < items.size(); ++i) items[i].request.id = i;
    items[0].request.address = a;                               // the preload
    items[1].request = {1, true, a, 64, std::vector<std::uint8_t>(64, 0x5A), 0};
    items[2].request.address = a;                               // after the write: its bytes
    items[3].request.address = b;                               // never written: zeros
    items[4].request.address = a + 8;                           // a read across the write's bytes
    items[4].request.size = 16;
    Rig rig(items);
    std::vector<std::uint8_t> pre(64);
    for (std::size_t i = 0; i < pre.size(); ++i) pre[i] = static_cast<std::uint8_t>(i);
    rig.ch->preload(a, pre);
    REQUIRE(rig.sim.run_until_idle(kLimit));
    std::map<std::uint64_t, DramResponse> by_id;
    for (const auto& r : rig.sink->received) by_id[r.id] = r;
    REQUIRE(by_id.size() == 5);
    CHECK(by_id[0].bytes == pre);
    CHECK(by_id[1].write);
    CHECK(by_id[1].bytes.empty());
    CHECK(by_id[2].bytes == std::vector<std::uint8_t>(64, 0x5A));
    CHECK(by_id[3].bytes == std::vector<std::uint8_t>(64, 0));
    CHECK(by_id[4].bytes == std::vector<std::uint8_t>(16, 0x5A));
    CHECK(rig.ch->peek(a, 64) == std::vector<std::uint8_t>(64, 0x5A));

    // reset() restores the preload, not the run's writes, and the run repeats exactly.
    const auto first = rig.sink->received;
    rig.sim.reset();
    CHECK(rig.ch->peek(a, 64) == pre);
    REQUIRE(rig.sim.run_until_idle(kLimit));
    REQUIRE(rig.sink->received.size() == first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        CHECK(rig.sink->received[i].id == first[i].id);
        CHECK(rig.sink->received[i].completed == first[i].completed);
        CHECK(rig.sink->received[i].bytes == first[i].bytes);
    }
}

TEST_CASE("Backpressure: a held sink fills the channel to its bound and stops the requester, losing nothing",
          "[cycle][lpddr5]") {
    auto cfg = pl::single_channel_config();
    cfg.queue_depth = 6;
    std::vector<Requester::Item> items;
    for (std::uint64_t i = 0; i < 40; ++i) {
        Requester::Item it;
        it.request.id = i;
        it.request.address = pl::make_address(static_cast<std::uint8_t>(i % 4), 100, static_cast<std::uint32_t>(i));
        items.push_back(it);
    }
    Rig rig(items, cfg, 0.0, 2, 2);
    rig.sink->stall(20, 400);
    std::size_t peak = 0;
    rig.sim.add_invariant("bound", [&] {
        peak = std::max(peak, rig.ch->outstanding());
        if (rig.ch->outstanding() > 6) throw std::logic_error("past the bound");
    });
    REQUIRE(rig.sim.run_until_idle(kLimit));
    CHECK(peak == 6);
    REQUIRE(rig.sink->received.size() == 40);
    std::vector<bool> seen(40, false);
    for (const auto& r : rig.sink->received) {
        CHECK_FALSE(seen[r.id]);
        seen[r.id] = true;
    }
    std::map<std::string, Counters> m;
    for (const auto& r : report(rig.sim)) m[r.name] = r.c;
    CHECK(m.at("lpddr5").stalls[static_cast<int>(Stall::NoCredit)] > 0);     // responses held by the sink
    CHECK(m.at("requester").stalls[static_cast<int>(Stall::NoCredit)] > 0);  // the channel full
    CHECK(m.at("lpddr5").arrivals == 40);
    CHECK(m.at("lpddr5").completions == 40);
    for (const auto& r : report(rig.sim)) {
        CAPTURE(r.name);
        CHECK(r.c.busy + r.c.stalled() + r.c.idle == r.c.cycles);
    }
}

TEST_CASE("Counters: the data bus carries one burst per request, and refresh runs while the channel idles",
          "[cycle][lpddr5]") {
    const auto cfg = pl::single_channel_config();
    SECTION("streaming page hits") {
        std::vector<Requester::Item> items;
        for (std::uint64_t i = 0; i < 64; ++i) {
            Requester::Item it;
            it.request.id = i;
            it.request.address = pl::make_address(0, 100, static_cast<std::uint32_t>(i));
            items.push_back(it);
        }
        Rig rig(items, cfg, 0.0, 8, 8);
        REQUIRE(rig.sim.run_until_idle(kLimit));
        const auto d = rig.ch->dram();
        CHECK(d.data_bus_busy == 64 * cfg.timing.tBurst_BL16);
        CHECK(d.page_hits == 63);
        CHECK(d.cycles == rig.dram.cycle());
        // The bursts back to back once the row is open: near the bus's capacity.
        const double window = static_cast<double>(d.cycles) - cfg.timing.tRCD - cfg.timing.tCL;
        CHECK(static_cast<double>(d.data_bus_busy) / window >= 0.85);
        CHECK(d.data_bus_utilization() > 0.0);
    }
    SECTION("refresh, with nothing to do") {
        Rig rig({}, cfg);
        const Cycle horizon = 20ull * 16 * cfg.timing.tREFIpb;
        while (rig.dram.cycle() < horizon) REQUIRE(rig.sim.step());
        const auto d = rig.ch->dram();
        CHECK(d.refreshes > 0);
        CHECK(d.refreshes == rig.ch->controller().lpddr5_stats().refreshes);
        CHECK(d.data_bus_busy == 0);
    }
}

TEST_CASE("A 1 GHz requester across clock-domain crossings: every request served, the same DRAM outcomes",
          "[cycle][lpddr5][crossing]") {
    const auto w = pl::make_four_banks_across_groups_workload();
    const auto items = items_of(w);
    Rig rig(items, pl::single_channel_config(), 1.0);
    REQUIRE(rig.sim.run_until_idle(kLimit));
    REQUIRE(rig.sink->received.size() == items.size());
    const auto d = rig.ch->dram();
    CHECK(d.page_hits == w.expected.page_hits);
    CHECK(d.page_empty == w.expected.page_empty);
    CHECK(d.page_conflicts == w.expected.page_conflicts);
    for (const auto& r : rig.sink->received) CHECK(r.tag == 1000 + r.id);
}

TEST_CASE("The channel refuses what it cannot model", "[cycle][lpddr5]") {
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("dram", 3.2));
    CHECK_THROWS_WITH(sim.add<Lpddr5Channel>("two", d, pl::dual_channel_config()), ContainsSubstring("one channel per component"));
    auto zero = pl::single_channel_config();
    zero.queue_depth = 0;
    CHECK_THROWS_WITH(sim.add<Lpddr5Channel>("zero", d, zero), ContainsSubstring("queue depth"));
    // Not connected: the first cycle says so, as an InvariantViolation naming the channel.
    sim.add<Lpddr5Channel>("loose", d, pl::single_channel_config());
    CHECK_THROWS_WITH(sim.step(), ContainsSubstring("not connected"));
}
