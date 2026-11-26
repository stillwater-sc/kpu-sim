// ============================================================================
// tests/cycle/cycle_csp_l3_demo.cpp
// Rung 4 on the cycle core (#540): an int8 64^3 matmul's L3 traffic on kpu_s1, with the
// hypothesized C, written out for the validators.
//
//   cycle_csp_l3_demo <source-dir> <core-trace.json> <counters.json>
//
// The walker's Loads, Stores and Releases drive the DMA engine and the L3 tile; L-B's C tiles are
// filled into their slots at the program's Writebacks. The data is checked (every slot at ready,
// DRAM's C at the end), and the run writes:
//   - the cycle core's trace and counters, for check_trace.py (open the trace in ui.perfetto.dev).
// The channel checks its controller's DRAM timing invariants as it runs.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include "l3_dma_rig.hpp"

#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/cycle/report.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdio>
#include <exception>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace driver = sw::kpu::program::driver;

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::fprintf(stderr, "usage: cycle_csp_l3_demo <source-dir> <core-trace.json> <counters.json>\n");
        return 2;
    }
    const auto dev =
        sw::kpu::program::platform::read_spec_file(std::string(argv[1]) + "/tests/program/deploy/kpu_s1.json").device(0);
    csp::gen::Options g;
    g.size = 64;
    g.tile = 16;
    g.l3 = 9;
    g.dtype = "int8";
    const lang::Program ast = lang::parse(csp::gen::generate(g));
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const auto in = driver::csp_inputs(ast);
    const auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    if (lb.skipped) {
        std::fprintf(stderr, "cycle_csp_l3_demo: L-B did not run: %s\n", lb.skipped->c_str());
        return 1;
    }
    const CspL3Plan plan = plan_l3(p, in, lb.values);

    Tracer tracer;
    L3DmaRig::Options o;
    o.slots = plan.slots;
    o.slot_bytes = plan.slot_bytes;
    o.banks = plan.banks;
    o.window = 16;
    o.tracer = &tracer;
    L3DmaRig r(plan.program, o);
    for (const auto& t : plan.inputs) r.channel.preload(t.address, t.data);
    std::map<std::size_t, bool> good;
    r.sim.add_invariant("watch", [&] {
        for (std::uint32_t s = 0; s < plan.slots; ++s)
            if (r.l3.ready(s) && !good.count(r.l3.ticket(s))) good[r.l3.ticket(s)] = r.l3.peek(s) == plan.placed.at(r.l3.ticket(s));
    });
    if (!r.sim.run_until_idle(20'000'000'000)) {
        std::fprintf(stderr, "cycle_csp_l3_demo: the run did not finish\n");
        return 1;
    }
    for (const auto& [ticket, bytes] : plan.placed)
        if (!good.count(ticket) || !good[ticket]) {
            std::fprintf(stderr, "cycle_csp_l3_demo: %s was not in its slot, bit for bit\n", plan.what.at(ticket).to_string().c_str());
            return 1;
        }
    for (const auto& t : plan.outputs)
        if (r.channel.peek(t.address, t.bytes) != t.data) {
            std::fprintf(stderr, "cycle_csp_l3_demo: %s is not L-B's in DRAM\n", t.tile.to_string().c_str());
            return 1;
        }
    tracer.write(argv[2]);
    write_counters_json(r.sim, argv[3]);
    std::printf("cycle_csp_l3_demo: int8 64^3 on kpu_s1: %zu Loads, %zu Fills (C), %zu Stores, %zu Releases through %u "
                "slots, bit for bit; %.1f%% of the data bus\n",
                plan.loads, plan.fills, plan.stores, plan.releases, plan.slots, 100.0 * r.channel.dram().data_bus_utilization());
    std::printf("%s", format_report(report(r.sim)).c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cycle_csp_l3_demo: %s\n", e.what());
    return 1;
}
