// ============================================================================
// tests/cycle/bench_cycle_core.cpp
// The cycle core's throughput (#534; docs/plans/cycle-based-kpu-prototype.md §3.3): how fast the
// simulator itself runs, so that its design choices rest on numbers.
//
// A chain of N stages (producer -> N stages -> consumer, each stage one cycle of latency, one
// beat per cycle, two credits per link so the credit loop never limits it) carries beats. It is
// run with the per-cycle invariant checks on and off, under two kinds of traffic:
//   dense    a beat every cycle, max(M, 4N) of them so that a long chain is mostly in steady state
//            (filling and draining take 2N cycles each way, one hop being a stage and a link)
//   sparse   M beats, one every 20 cycles: most stages are idle most of the time
// Each pair of settings runs in the order checks on, off, off, on, so that drift (warm-up, clock
// changes) falls on both alike; a row is the mean of its two runs, with their spread as an estimate
// of the noise. Reported: simulated cycles, wall time, cycles per second, component evaluations per
// second and ns per evaluation, and the IDLE SHARE -- the fraction of component evaluations in
// which the component was idle (its counters), which is what an opt-in sleep (simulator.hpp,
// step()) could save at most.
//
//   bench_cycle_core           sizes 10, 100, 1000, 10000; M = 1000
//   bench_cycle_core --quick   sizes 10, 100; M = 100 (the CTest smoke run)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "pipeline.hpp"

#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/simulator.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;

namespace {

struct Result {
    std::size_t stages = 0;
    bool dense = true, checks = true;
    Cycle cycles = 0;
    double wall_s = 0, idle_share = 0;
    std::uint64_t evaluations = 0;
};

Result run(std::size_t stages, std::uint64_t beats, bool dense, bool checks) {
    Simulator sim;
    Domain& d = sim.add_domain(Domain::from_ghz("clk", 1.0));
    auto& p = sim.add<Producer>("producer", d, beats, dense ? 1 : 20);
    std::vector<Stage*> chain;
    for (std::size_t i = 0; i < stages; ++i) chain.push_back(&sim.add<Stage>("s" + std::to_string(i), d, 1, 1, 1));
    auto& c = sim.add<Consumer>("consumer", d);
    Component* prev = &p;
    Link<Beat>* in = nullptr;
    for (std::size_t i = 0; i <= stages; ++i) {
        Component* next = i < stages ? static_cast<Component*>(chain[i]) : &c;
        auto& l = sim.link<Beat>("l" + std::to_string(i), *prev, *next, 2);
        if (i == 0) p.connect(l);
        else chain[i - 1]->connect(*in, l);
        in = &l;
        prev = next;
    }
    c.connect(*in);
    sim.set_checks(checks);

    const auto t0 = std::chrono::steady_clock::now();
    if (!sim.run_until_idle(~Picoseconds{0} - 1)) std::fprintf(stderr, "bench: the run did not finish\n");
    const auto t1 = std::chrono::steady_clock::now();
    c.finish(beats);

    Result r;
    r.stages = stages;
    r.dense = dense;
    r.checks = checks;
    r.cycles = d.cycle();
    r.wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::uint64_t idle = 0, all = 0;
    for (const auto& row : report(sim))
        if (!row.link) {
            idle += row.c.idle;
            all += row.c.cycles;
        }
    r.evaluations = all;
    r.idle_share = all ? double(idle) / double(all) : 0.0;
    return r;
}

}  // namespace

int main(int argc, char** argv) try {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const std::vector<std::size_t> sizes = quick ? std::vector<std::size_t>{10, 100}
                                                 : std::vector<std::size_t>{10, 100, 1000, 10000};
    const std::uint64_t beats = quick ? 100 : 1000;

    std::printf("cycle core throughput: a chain of N stages, M = %llu beats (%s)\n",
                static_cast<unsigned long long>(beats), quick ? "--quick" : "full");
    std::printf("each pair runs checks on, off, off, on: the mean of its two runs, and their spread\n");
    std::printf("%8s %-7s %-6s %10s %10s %8s %14s %16s %10s %8s\n", "stages", "traffic", "checks", "cycles",
                "wall ms", "spread", "cycles/s", "evaluations/s", "ns/eval", "idle");
    for (const std::size_t n : sizes)
        for (const bool dense : {true, false}) {
            const std::uint64_t m = dense ? std::max<std::uint64_t>(beats, 4 * n) : beats;
            // Balanced order, so drift (warm-up, clock changes) falls on both settings alike.
            Result on[2], off[2];
            on[0] = run(n, m, dense, true);
            off[0] = run(n, m, dense, false);
            off[1] = run(n, m, dense, false);
            on[1] = run(n, m, dense, true);
            for (const Result* r : {on, off}) {
                const double wall = (r[0].wall_s + r[1].wall_s) / 2;
                const double spread = wall > 0 ? std::abs(r[0].wall_s - r[1].wall_s) / wall : 0.0;
                const double eps = wall > 0 ? double(r[0].evaluations) / wall : 0.0;
                std::printf("%8zu %-7s %-6s %10llu %10.2f %7.1f%% %14.0f %16.0f %10.2f %7.1f%%\n", r[0].stages,
                            dense ? "dense" : "sparse", r[0].checks ? "on" : "off",
                            static_cast<unsigned long long>(r[0].cycles), wall * 1e3, 100.0 * spread,
                            wall > 0 ? double(r[0].cycles) / wall : 0.0, eps, eps > 0 ? 1e9 / eps : 0.0,
                            100.0 * r[0].idle_share);
            }
        }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "bench_cycle_core: %s\n", e.what());
    return 1;
}
