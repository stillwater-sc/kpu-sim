// ============================================================================
// tests/cycle/test_cycle_system.cpp
// The join, step 3c (#548; docs/plans/cycle-based-kpu-prototype.md §6): the memory subsystem and the
// compute subsystem on one simulator, DRAM -> L3 -> L2 -> L1 -> array -> Combiner -> L2 -> L3 ->
// DRAM, from one CSP program. Nothing hypothesized: C reaches DRAM from the array, and DRAM ends
// with L-B's result bit for bit.
//
// The RTL's 8x8x16 int8 matmul on kpu_soc_2x2, single and double buffered in L2; then an int8 64^3
// on kpu_s1 with Phase 1's own L3 of 9 tiles, so its slots are reused, each after its Release,
// under the credits the three of them hand round: the DMA engine, the L3 tile, the BlockMover.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/cycle/system/kpu_system.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <string>

using namespace sw::kpu::cycle;
namespace program = sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const std::string& file) {
    return platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file).device(0);
}

std::string rtl_8x8x16(int l2) {
    return "csp 1.1\nprogram mm machine flat(l3 = 32, l2 = " + std::to_string(l2) +
           ") {\n"
           "  tensor A[8,16] int8 tile 4x4 in;\n  tensor B[16,8] int8 tile 4x4 in;\n"
           "  tensor C[8,8] int32 tile 4x4 out;\n"
           "  for j in 0..2 {\n    resident B[:, j];\n    for i in 0..2 {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

std::string int8_64cubed(std::size_t l3, int buffers) {
    csp::gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = l3;
    o.dtype = "int8";
    o.l2_buffers = buffers;
    return csp::gen::generate(o);
}

SystemPlan plan_of(const std::string& text, const platform::DeviceSpecification& dev) {
    const lang::Program ast = lang::parse(text);
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const program::TileProgram in = driver::csp_inputs(ast);
    auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    REQUIRE(!lb.skipped);
    return plan_system(p, in, lb.values);
}

}  // namespace

TEST_CASE("The join: the RTL's 8x8x16 int8 matmul from DRAM to DRAM, C from the array bit for bit",
          "[cycle][system][join]") {
    const auto dev = device("kpu_soc_2x2.json");
    std::map<int, Cycle> clocks;
    for (const int l2 : {3, 6}) {
        CAPTURE(l2);
        const SystemPlan plan = plan_of(rtl_8x8x16(l2), dev);
        CHECK(plan.stores == 4);
        CHECK(plan.fills == 0);          // nothing hypothesized
        CHECK(plan.writes == 4);         // C written into L3 by the BlockMover
        CHECK(plan.reads == plan.compute.moves);
        const SystemRun r = run_system(plan, dev, SystemOptions{});
        INFO(format_system_summary(plan, r));
        REQUIRE(r.ok());
        CHECK(r.memory.tiles_checked == 4);
        CHECK(r.compute.results == 4);
        CHECK(r.compute.pe_total.useful == 16ull * 4 * 4 * 4);   // 16 commands of 4 x 4 lanes, K = 4
        clocks[l2] = r.compute.kpu_cycles;
    }
    CHECK(clocks[6] < clocks[3]);   // double buffering in L2, end to end
}

TEST_CASE("The join: an int8 64^3 on kpu_s1 with Phase 1's L3 of 9 tiles, its slots reused", "[cycle][system][join]") {
    const auto dev = device("kpu_s1.json");
    const SystemPlan plan = plan_of(int8_64cubed(9, 2), dev);
    CHECK(plan.memory.slots == 9);
    CHECK(plan.stores == 16);
    // More residencies than slots: each slot holds several over the run.
    CHECK(plan.loads + plan.writes > plan.memory.slots);
    const SystemRun r = run_system(plan, dev, SystemOptions{});
    INFO(format_system_summary(plan, r));
    REQUIRE(r.ok());
    CHECK(r.memory.tiles_checked == 16);
    CHECK(r.compute.pe_total.useful == 64ull * 16 * 16 * 16);
    CHECK(r.memory.l3_max_live <= 9);
    const std::string s = format_system_summary(plan, r);
    CHECK_THAT(s, ContainsSubstring("16 of 16 stored tiles in DRAM bit for bit, L-B's; laws: hold on every row"));
    CHECK_THAT(s, ContainsSubstring("DMA "));
    CHECK_THAT(s, ContainsSubstring("array        utilization"));
}

TEST_CASE("The join refuses a deployment with a NoC and tiles past the array", "[cycle][system][join]") {
    const SystemPlan plan = plan_of(int8_64cubed(9, 2), device("kpu_s1.json"));
    CHECK_THROWS_WITH(run_system(plan, device("kpu_soc_2x2.json"), SystemOptions{}), ContainsSubstring("past"));
}
