// ============================================================================
// tests/cycle/cycle_trace_demo.cpp
// The test structure's trace and counters files (#535; docs/plans/cycle-based-kpu-prototype.md
// §3.3): open the trace in https://ui.perfetto.dev (or chrome://tracing).
//
//   cycle_trace_demo <trace.json> <counters.json>
//
// Three pipelines on two clocks: at 1 GHz a producer -> stage -> consumer whose consumer is held
// for cycles [40, 70), so the stage and the producer stall for want of credits; at 250 MHz a
// producer -> consumer whose one credit paces it; and a 4:1 clock-domain crossing (#536), a 1 GHz
// producer into a 250 MHz consumer through an async FIFO of four entries, at the banks' rate.
// Each domain has its cycle-number track and its square-wave clock: the banks' clock rises on
// every fourth rise of the array's. check_trace.py validates the files against each other.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "pipeline.hpp"

#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>
#include <sw/kpu/cycle/trace.hpp>

#include <cstdio>
#include <exception>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: cycle_trace_demo <trace.json> <counters.json>\n");
        return 2;
    }
    Tracer tracer;
    tracer.clock_tracks(true);   // a short run: the square waves too
    Simulator sim;
    sim.attach(tracer);
    Domain& fast = sim.add_domain(Domain::from_ghz("array", 1.0));
    Domain& slow = sim.add_domain(Domain::from_ghz("banks", 0.25));

    auto& p = sim.add<Producer>("producer", fast, 60);
    auto& s = sim.add<Stage>("stage", fast, 4, 2, 2);
    auto& c = sim.add<Consumer>("consumer", fast);
    auto& a = sim.link<Beat>("producer->stage", p, s, 3);
    auto& b = sim.link<Beat>("stage->consumer", s, c, 2);
    p.connect(a);
    s.connect(a, b);
    c.connect(b);
    c.stall(40, 70);

    auto& ps = sim.add<Producer>("slow producer", slow, 20);
    auto& cs = sim.add<Consumer>("slow consumer", slow);
    auto& ls = sim.link<Beat>("slow link", ps, cs, 1);
    ps.connect(ls);
    cs.connect(ls);

    auto& px = sim.add<Producer>("array producer", fast, 24);
    auto& cx = sim.add<Consumer>("banks consumer", slow);
    auto& lx = sim.crossing<Beat>("array->banks", px, cx, 4);
    px.connect(lx);
    cx.connect(lx);

    if (!sim.run_until_idle(10'000'000)) {
        std::fprintf(stderr, "cycle_trace_demo: the run did not finish\n");
        return 1;
    }
    c.finish(60);
    cs.finish(20);
    cx.finish(24);
    tracer.write(argv[1]);
    write_counters_json(sim, argv[2]);
    std::printf("cycle_trace_demo: %zu trace events (%zu dropped) to %s; counters to %s\n", tracer.events(),
                tracer.dropped(), argv[1], argv[2]);
    std::printf("%s", format_report(report(sim)).c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cycle_trace_demo: %s\n", e.what());
    return 1;
}
