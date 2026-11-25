// ============================================================================
// tests/cycle/test_cycle_memory_subsystem.cpp
// The memory subsystem as kpu-run runs it, step 1e (#541; docs/plans/cycle-based-kpu-prototype.md
// §4.4): a program's L3 traffic through DRAM, the DMA engine and the L3 tile, on a direct link
// (kpu_s1) and over the NoC (the T4). The data checks, the laws on every row, the summary's figures
// for the DMA engine, the channel, the NoC and the L3 tile, the trace's L3 slot lanes, and the
// deadlock report.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/cycle/memory/memory_subsystem.hpp>
#include <sw/kpu/cycle/trace.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::cycle;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

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

}  // namespace

TEST_CASE("The memory subsystem on kpu_s1: data, laws and the summary", "[cycle][memory]") {
    const auto dev = device("kpu_s1.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    Tracer tracer;
    MemoryRunOptions o;
    o.tracer = &tracer;
    const MemoryRun r = run_memory_subsystem(plan, &dev, o);
    REQUIRE(r.ok());
    CHECK(!r.noc);
    CHECK(r.slots_checked == plan.placed.size());
    CHECK(r.tiles_checked == plan.outputs.size());
    CHECK(r.law_errors.empty());
    CHECK(r.dma.read_bursts == plan.loads * 4);     // 256-byte int8 tiles
    CHECK(r.dma.write_bursts == plan.stores * 16);  // 1 KiB int32 tiles
    CHECK(r.dram.data_bus_utilization() > 0.5);
    const std::string s = format_memory_summary(plan, r);
    CHECK_THAT(s, ContainsSubstring("on a direct link") && ContainsSubstring("laws: hold on every row") &&
                      ContainsSubstring("  DMA ") && ContainsSubstring("  DRAM ") && ContainsSubstring("  L3 ") &&
                      ContainsSubstring("held up by:"));
    // The trace: a lane per L3 slot, each tile filling, then resident.
    const std::string path = std::string(KPU_BINARY_DIR) + "/memory_subsystem_s1.json";
    tracer.write(path);
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    CHECK_THAT(text.str(), ContainsSubstring("\"l3 slot 0 (kpu)\"") && ContainsSubstring("\"filling A[0,0]\"") &&
                               ContainsSubstring("\"name\":\"C[0,0]\""));
}

TEST_CASE("The memory subsystem over the T4's NoC", "[cycle][memory][noc]") {
    const auto dev = device("kpu_t4.json");
    const CspL3Plan plan = plan_for(dev, 64, 16);
    const MemoryRun r = run_memory_subsystem(plan, &dev, {});
    REQUIRE(r.ok());
    CHECK(r.noc);
    CHECK_THAT(format_memory_summary(plan, r), ContainsSubstring("over the NoC") && ContainsSubstring("  NoC "));
}

TEST_CASE("A deadlock prints the detector's report", "[cycle][memory]") {
    // A Store from a slot nothing fills: the L3 tile waits for its tile forever.
    CspL3Plan plan;
    plan.slots = 1;
    plan.slot_bytes = 64;
    plan.program = {l3_store(0, 0x100000, 64)};
    MemoryRunOptions o;
    o.deadlock_window = 500;
    const MemoryRun r = run_memory_subsystem(plan, nullptr, o);
    CHECK(!r.ok());
    REQUIRE(r.deadlock);
    const std::string s = format_memory_summary(plan, r);
    CHECK_THAT(s, ContainsSubstring("DEADLOCK") && ContainsSubstring("Store slot 0: waiting for its tile"));
}

TEST_CASE("The laws find a row that does not add up", "[cycle][memory]") {
    ReportRow row;
    row.name = "x";
    row.c.cycles = 10;
    row.c.busy = 4;
    row.c.idle = 5;   // one cycle missing
    CHECK_THAT(check_laws({row}, true).at(0), ContainsSubstring("!= T 10"));
    ReportRow link;
    link.name = "l";
    link.link = true;
    link.c.arrivals = 3;
    link.c.completions = 2;
    CHECK(check_laws({link}, false).empty());   // still running: a beat may be on it
    CHECK_THAT(check_laws({link}, true).at(0), ContainsSubstring("3 arrivals - 2 completions"));
}

TEST_CASE("A slot or a stored tile that is not the program's bytes fails the run", "[cycle][memory]") {
    const auto dev = device("kpu_s1.json");
    CspL3Plan plan = plan_for(dev, 64, 16);
    SECTION("a slot") {
        auto& bytes = plan.placed.begin()->second;
        bytes[0] ^= 1;   // what the oracle expects, not what the program places
        const MemoryRun r = run_memory_subsystem(plan, &dev, {});
        CHECK(!r.ok());
        REQUIRE(r.data_errors.size() == 1);
        CHECK_THAT(r.data_errors[0], ContainsSubstring("is not the program's bytes in its slot"));
        CHECK_THAT(format_memory_summary(plan, r), ContainsSubstring("DATA: "));
    }
    SECTION("a stored tile") {
        plan.outputs.front().data[0] ^= 1;
        const MemoryRun r = run_memory_subsystem(plan, &dev, {});
        CHECK(!r.ok());
        REQUIRE(r.data_errors.size() == 1);
        CHECK_THAT(r.data_errors[0], ContainsSubstring("in DRAM is not L-B's"));
    }
}
