#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/memory_subsystem.hpp
// The memory subsystem on the cycle core, run on a program's L3 traffic (#541;
// docs/plans/cycle-based-kpu-prototype.md §4.4): what kpu-run --memory-subsystem runs.
//
// THE MACHINE. One LPDDR5 channel in its own 3.2 GHz domain; one DMA engine and one L3 tile in the
// KPU's domain (1 GHz), over a direct link, or over the NoC when the deployment declares one: the
// engine on port 0, the L3 tile at hub 0 (a level-1 program addresses L3 module 0 only). Its
// program is a CspL3Plan (csp_l3_program.hpp): the walker's L3 actions, C filled from L-B.
//
// WHAT A RUN CHECKS:
//   data        every Load's and Fill's slot holds the bytes the program places, at the cycle its
//               tile is ready; DRAM ends with every stored tile as L-B leaves it
//   invariants  every cycle: the slot credits (check_slot_credits), each component's and link's
//               own checks, the NoC's bubble rule and its one flit per link per cycle
//   the laws    on every component's and link's counters: the cycles add up (busy + stalled + idle
//               = T), flow balances (arrivals - completions = what is held at the end), and the
//               utilization law U = X S and Little's law N = X R hold on the derived figures
//   deadlock    the detector's report, if no progress is made
//
// THE TRACE (a Tracer in the options): a track per component (DMA engine, channel, L3 tile, hub,
// port) and per link, each domain's clock, and a lane per L3 slot: each tile's span "filling"
// from its first beat to ready, then resident until its Release. The channel's own trace, a track
// per bank and bank group, is a separate file (MemoryRunOptions::dram_trace) for the LPDDR5 trace
// validator.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/cycle/memory/dma_engine.hpp>
#include <sw/kpu/cycle/memory/l3_tile.hpp>
#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/noc/noc_fabric.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::cycle {

// LPDDR5-6400, one channel of 16 banks in 4 groups (patterns/memory/lpddr5's single channel).
inline Lpddr5Channel::Controller::Config lpddr5_one_channel() {
    Lpddr5Channel::Controller::Config c;
    c.num_channels = 1;
    c.banks_per_channel = 16;
    c.bank_groups = 4;
    c.burst_length = lpddr5::BurstLength::BL16;
    c.queue_depth = 64;
    return c;
}

struct MemoryRunOptions {
    double kpu_ghz = 1.0;
    std::size_t window = 16;                 // the DMA engine's reads (and writes) in flight
    std::size_t store_buffer_beats = 4;
    std::uint64_t deadlock_window = 10'000;  // edges without progress
    Picoseconds limit = 1'000'000'000'000;   // 1 s of simulated time
    Tracer* tracer = nullptr;
    std::string dram_trace;                  // where to write the channel's own trace ("" = none)
    std::string counters;                    // where to write the counters (write_counters_json; "" = none)
};

struct MemoryRun {
    bool finished = false;
    std::optional<std::string> deadlock;     // the detector's report: its message and the state dump
    std::optional<std::string> violation;    // an invariant that failed, with the state dump
    bool noc = false;
    Picoseconds end = 0;
    std::vector<std::string> data_errors;
    std::vector<std::string> law_errors;
    std::size_t slots_checked = 0, tiles_checked = 0;
    std::vector<ReportRow> rows;
    DmaCounts dma;
    DramCounts dram;
    std::uint64_t l3_max_live = 0;
    double l3_mean_live = 0;
    [[nodiscard]] bool ok() const {
        return finished && !deadlock && !violation && data_errors.empty() && law_errors.empty();
    }
};

// The laws on the counters, row by row; an empty list when every one holds. Flow balance: a
// component holds what it accepted and has not completed; a link, whose holding is counted in its
// sender's cycles (a crossing's receiver may pop after the sender's last), is empty at the end of
// a run that `finished`.
inline std::vector<std::string> check_laws(const std::vector<ReportRow>& rows, bool finished) {
    std::vector<std::string> bad;
    auto close = [](double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(b)); };
    for (const ReportRow& r : rows) {
        const Counters& c = r.c;
        const std::string who = (r.link ? "link " : "component ") + r.name;
        if (!r.link && c.busy + c.stalled() + c.idle != c.cycles)
            bad.push_back(who + ": busy " + std::to_string(c.busy) + " + stalled " + std::to_string(c.stalled()) +
                          " + idle " + std::to_string(c.idle) + " != T " + std::to_string(c.cycles));
        if (c.arrivals < c.completions || (!r.link && c.arrivals - c.completions != c.occupancy) ||
            (r.link && finished && c.arrivals != c.completions))
            bad.push_back(who + ": " + std::to_string(c.arrivals) + " arrivals - " + std::to_string(c.completions) +
                          " completions, " + std::to_string(c.occupancy) + " held");
        if (c.completions > 0) {
            if (!close(c.utilization(), c.throughput() * c.service_time()))
                bad.push_back(who + ": U != X S");
            if (!close(c.mean_occupancy(), c.throughput() * c.mean_residence()))
                bad.push_back(who + ": N != X R");
        }
        if (c.utilization() > 1.0 + 1e-12) bad.push_back(who + ": U > 1");
    }
    return bad;
}

inline MemoryRun run_memory_subsystem(const CspL3Plan& plan, const program::platform::DeviceSpecification* device,
                                      const MemoryRunOptions& o) {
    MemoryRun out;
    Simulator sim;
    if (o.tracer) sim.attach(*o.tracer);
    sim.set_deadlock_window(o.deadlock_window);
    Domain& dram_d = sim.add_domain(Domain::from_ghz("dram", 3.2));
    Domain& kpu = sim.add_domain(Domain::from_ghz("kpu", o.kpu_ghz));
    auto& ch = sim.add<Lpddr5Channel>("lpddr5", dram_d, lpddr5_one_channel());
    auto& dma = sim.add<DmaEngine>("dma", kpu,
                                   DmaEngine::Config{plan.slots, plan.slot_bytes, o.window, o.store_buffer_beats, o.window},
                                   plan.program);
    auto& l3 = sim.add<L3Tile>("l3", kpu, L3Tile::Config{plan.slots, plan.slot_bytes, plan.banks}, plan.program);
    auto& req = sim.crossing<DramRequest>("dma -> lpddr5", dma, ch, 4);
    auto& rsp = sim.crossing<DramResponse>("lpddr5 -> dma", ch, dma, 4);
    auto& credits = sim.link<SlotCredit>("l3 credits", l3, dma, plan.slots);
    ch.connect(req, rsp);
    std::unique_ptr<NocFabric> noc;
    std::optional<program::platform::ArrayLayout> layout;
    if (device && device->noc) layout = program::platform::ArrayLayout::of(*device);
    if (layout && layout->has_noc()) {
        out.noc = true;
        noc = std::make_unique<NocFabric>(sim, kpu, *layout, NocFabric::Config::from(*device));
        const auto e = noc->attach_engine(0, dma, 0);
        const auto t = noc->attach_l3(0, l3, 0, e.engine);
        dma.connect(req, rsp, e.to_noc, e.from_noc, credits);
        l3.connect(t.to_l3, t.from_l3, credits);
    } else {
        auto& loads = sim.link<L3Beat>("dma -> l3", dma, l3, 2);
        auto& ejects = sim.link<L3Beat>("l3 -> dma", l3, dma, 2);
        dma.connect(req, rsp, loads, ejects, credits);
        l3.connect(loads, ejects, credits);
    }
    sim.add_invariant("slot credits", [&] { check_slot_credits(dma, l3); });
    for (const auto& t : plan.inputs) ch.preload(t.address, t.data);
    std::map<std::size_t, std::vector<std::uint8_t>> seen;
    sim.add_invariant("slots", [&] {
        for (std::uint32_t s = 0; s < plan.slots; ++s)
            if (l3.ready(s) && !seen.count(l3.ticket(s))) seen[l3.ticket(s)] = l3.peek(s);
    });
    if (!o.dram_trace.empty()) ch.enable_dram_trace(true);

    try {
        out.finished = sim.run_until_idle(o.limit);
    } catch (const Deadlock& e) {
        out.deadlock = std::string(e.what()) + "\n" + e.dump();
    } catch (const InvariantViolation& e) {
        out.violation = std::string(e.what()) + "\n" + e.dump();
    }
    out.end = sim.now();

    for (const auto& [ticket, bytes] : plan.placed) {
        const auto it = seen.find(ticket);
        const std::string tile = plan.what.at(ticket).to_string();
        if (it == seen.end()) out.data_errors.push_back(tile + " (action " + std::to_string(ticket) + ") never ready in its slot");
        else if (it->second != bytes) out.data_errors.push_back(tile + " (action " + std::to_string(ticket) + ") is not the program's bytes in its slot");
        else ++out.slots_checked;
    }
    if (out.finished)
        for (const auto& t : plan.outputs) {
            if (ch.peek(t.address, t.bytes) != t.data) out.data_errors.push_back(t.tile.to_string() + " in DRAM is not L-B's");
            else ++out.tiles_checked;
        }
    out.rows = report(sim);
    if (!o.counters.empty()) write_counters_json(sim, o.counters);
    out.law_errors = check_laws(out.rows, out.finished);
    out.dma = dma.dma();
    out.dram = ch.dram();
    out.l3_max_live = l3.counts().max_occupancy;
    out.l3_mean_live = l3.counts().mean_occupancy();

    if (o.tracer) {
        std::vector<std::uint32_t> lanes;
        for (std::uint32_t s = 0; s < plan.slots; ++s) lanes.push_back(o.tracer->add_lane("l3 slot " + std::to_string(s), kpu.name()));
        const Cycle last = kpu.cycle();
        for (const auto& r : l3.residencies()) {
            const std::string tile = plan.what.count(r.ticket) ? plan.what.at(r.ticket).to_string() : "?";
            o.tracer->span(lanes[r.slot], "filling " + tile, kpu.edge(r.filling_from), kpu.edge(r.ready_at));
            o.tracer->span(lanes[r.slot], tile, kpu.edge(r.ready_at), kpu.edge(r.released_at ? *r.released_at : last));
        }
    }
    if (!o.dram_trace.empty() && !ch.write_dram_trace(o.dram_trace))
        throw std::runtime_error("run_memory_subsystem: cannot write " + o.dram_trace);
    return out;
}

// The summary kpu-run prints: what ran, the data and the laws, X, U and the stalls of the DMA
// engine, the channel, the NoC and the L3 tile, then every component's and link's row.
inline std::string format_memory_summary(const CspL3Plan& plan, const MemoryRun& r) {
    std::string s;
    char b[512];
    auto row = [&](const std::string& name) -> const ReportRow* {
        for (const auto& x : r.rows)
            if (x.name == name) return &x;
        return nullptr;
    };
    auto stalls = [](const Counters& c) {
        std::string t;
        for (std::size_t k = 0; k < kStallCauses; ++k)
            if (c.stalls[k]) t += std::string(t.empty() ? "" : ", ") + to_string(static_cast<Stall>(k)) + " " + std::to_string(c.stalls[k]);
        return t.empty() ? std::string("none") : t;
    };
    std::snprintf(b, sizeof b, "memory subsystem: %zu Loads, %zu Fills (C, from L-B), %zu Stores, %zu Releases; %u L3 slots of %u bytes, %s\n",
                  plan.loads, plan.fills, plan.stores, plan.releases, plan.slots, plan.slot_bytes,
                  r.noc ? "over the NoC" : "on a direct link");
    s += b;
    if (r.deadlock) s += "DEADLOCK: " + *r.deadlock + "\n";
    if (r.violation) s += "INVARIANT VIOLATED: " + *r.violation + "\n";
    if (!r.finished && !r.deadlock && !r.violation) s += "the run did not finish within its limit\n";
    std::snprintf(b, sizeof b, "  run          %.3f us; data: %zu of %zu slots and %zu of %zu stored tiles bit for bit; laws: %s\n",
                  double(r.end) / 1e6, r.slots_checked, plan.placed.size(), r.tiles_checked, plan.outputs.size(),
                  r.law_errors.empty() ? "hold on every row" : "VIOLATED");
    s += b;
    for (const auto& e : r.data_errors) s += "  DATA: " + e + "\n";
    for (const auto& e : r.law_errors) s += "  LAW: " + e + "\n";
    if (const ReportRow* d = row("dma")) {
        std::snprintf(b, sizeof b, "  DMA          X %.4f/cycle, U %.3f; %llu read and %llu write bursts, at most %llu in flight; stalls: %s\n",
                      d->c.throughput(), d->c.utilization(), (unsigned long long)r.dma.read_bursts,
                      (unsigned long long)r.dma.write_bursts, (unsigned long long)r.dma.max_in_flight, stalls(d->c).c_str());
        s += b;
        std::string w;
        for (std::size_t k = 0; k < kDmaWaits; ++k)
            if (r.dma.waits[k]) w += std::string(w.empty() ? "" : ", ") + to_string(static_cast<DmaWait>(k)) + " " + std::to_string(r.dma.waits[k]);
        s += "               held up by: " + (w.empty() ? std::string("nothing") : w) + "\n";
    }
    if (const ReportRow* d = row("lpddr5")) {
        std::snprintf(b, sizeof b, "  DRAM         X %.4f/cycle, U %.3f; data bus %.1f%%; pages: %llu hit, %llu empty, %llu conflict; %llu refreshes; stalls: %s\n",
                      d->c.throughput(), d->c.utilization(), 100.0 * r.dram.data_bus_utilization(),
                      (unsigned long long)r.dram.page_hits, (unsigned long long)r.dram.page_empty,
                      (unsigned long long)r.dram.page_conflicts, (unsigned long long)r.dram.refreshes, stalls(d->c).c_str());
        s += b;
    }
    if (r.noc) {
        double umax = 0, nmax = 0;
        std::size_t links = 0;
        std::uint64_t flits = 0;
        for (const auto& x : r.rows)
            if (x.link && x.name.rfind("ch", 0) == 0) {
                ++links;
                flits += x.c.completions;
                umax = std::max(umax, x.c.utilization());
                nmax = std::max(nmax, x.c.mean_occupancy());
            }
        std::uint64_t hub_stalls = 0;
        for (const auto& x : r.rows)
            if (!x.link && (x.name.rfind("noc hub", 0) == 0 || x.name.rfind("noc port", 0) == 0)) hub_stalls += x.c.stalled();
        std::snprintf(b, sizeof b, "  NoC          %zu ring links, %llu flit hops; busiest link U %.3f, deepest mean queue %.3f; hub and port stalled cycles %llu\n",
                      links, (unsigned long long)flits, umax, nmax, (unsigned long long)hub_stalls);
        s += b;
    }
    if (const ReportRow* d = row("l3")) {
        std::snprintf(b, sizeof b, "  L3           X %.4f tiles/cycle, U %.3f; slots live: mean %.2f, at most %llu of %u; a tile's mean residence %.1f cycles; stalls: %s\n",
                      d->c.throughput(), d->c.utilization(), r.l3_mean_live, (unsigned long long)r.l3_max_live, plan.slots,
                      d->c.mean_residence(), stalls(d->c).c_str());
        s += b;
    }
    s += format_report(r.rows);
    return s;
}

}  // namespace sw::kpu::cycle
