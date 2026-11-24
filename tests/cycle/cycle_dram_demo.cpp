// ============================================================================
// tests/cycle/cycle_dram_demo.cpp
// One LPDDR5 channel on the cycle core (#537): its traces and counters, for the validators.
//
//   cycle_dram_demo <dram-trace.json> <core-trace.json> <counters.json>
//
// A 1 GHz requester sends every workload of the controller's pattern suite, one after another
// (page hits, conflicts, mixed reads and writes, two to four banks, strided), across a clock-domain
// crossing into the 3.2 GHz channel; the responses cross back to a 1 GHz sink, which is held for a
// while so the channel backs up. Writes:
//   - the controller's own trace, for patterns/memory/lpddr5/common/trace_validator.py;
//   - the cycle core's trace and counters, for check_trace.py (open either in ui.perfetto.dev).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "dram_pipeline.hpp"

#include <lpddr5_configs.hpp>
#include <workloads.hpp>

#include <sw/kpu/cycle/memory/lpddr5_channel.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstdio>
#include <exception>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
namespace pl = sw::kpu::patterns::lpddr5;

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::fprintf(stderr, "usage: cycle_dram_demo <dram-trace.json> <core-trace.json> <counters.json>\n");
        return 2;
    }
    std::vector<Requester::Item> items;
    for (const auto& w : pl::all_workloads())
        for (const auto& a : w.accesses) {
            Requester::Item it;
            it.request.id = items.size();
            it.request.write = a.type == pl::AccessType::WRITE;
            it.request.address = a.address;
            it.request.size = a.size;
            if (it.request.write) it.request.bytes.assign(a.size, static_cast<std::uint8_t>(it.request.id));
            it.request.tag = it.request.id;
            items.push_back(std::move(it));
        }
    const std::size_t n = items.size();

    Tracer tracer;
    Simulator sim;
    sim.attach(tracer);
    Domain& host = sim.add_domain(Domain::from_ghz("host", 1.0));
    Domain& dram = sim.add_domain(Domain::from_ghz("dram", 3.2));
    auto& req = sim.add<Requester>("requester", host, std::move(items));
    auto& ch = sim.add<Lpddr5Channel>("lpddr5", dram, pl::single_channel_config());
    auto& sink = sim.add<Sink>("sink", host);
    auto& in = sim.crossing<DramRequest>("requests", req, ch, 4);
    auto& out = sim.crossing<DramResponse>("responses", ch, sink, 4);
    req.connect(in);
    ch.connect(in, out);
    sink.connect(out);
    sink.stall(60, 160);
    ch.enable_dram_trace(true);

    if (!sim.run_until_idle(1'000'000'000)) {
        std::fprintf(stderr, "cycle_dram_demo: the run did not finish\n");
        return 1;
    }
    if (sink.received.size() != n) {
        std::fprintf(stderr, "cycle_dram_demo: %zu of %zu responses\n", sink.received.size(), n);
        return 1;
    }
    if (!ch.write_dram_trace(argv[1])) {
        std::fprintf(stderr, "cycle_dram_demo: cannot write %s\n", argv[1]);
        return 1;
    }
    tracer.write(argv[2]);
    write_counters_json(sim, argv[3]);
    const DramCounts d = ch.dram();
    std::printf("cycle_dram_demo: %zu requests; %llu reads, %llu writes; page hits %llu, empty %llu, conflicts %llu; "
                "refreshes %llu; data bus %llu of %llu cycles (%.1f%%)\n",
                n, static_cast<unsigned long long>(d.reads), static_cast<unsigned long long>(d.writes),
                static_cast<unsigned long long>(d.page_hits), static_cast<unsigned long long>(d.page_empty),
                static_cast<unsigned long long>(d.page_conflicts), static_cast<unsigned long long>(d.refreshes),
                static_cast<unsigned long long>(d.data_bus_busy), static_cast<unsigned long long>(d.cycles),
                100.0 * d.data_bus_utilization());
    std::printf("%s", format_report(report(sim)).c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cycle_dram_demo: %s\n", e.what());
    return 1;
}
