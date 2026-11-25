// ============================================================================
// tests/cycle/test_cycle_csp_l3.cpp
// Rung 4 of the memory subsystem, step 1d (#540; docs/plans/cycle-based-kpu-prototype.md §4.2-4.4):
// the DMA engine and the L3 tile run a real matmul program's L3 traffic -- the walker's Loads,
// Stores and Releases, in the walker's slots -- with the hypothesized C, L-B's result, filled into
// its slots at the program's Writebacks.
//
// On the RTL's 8x8x16 int8 matmul (kpu_soc_2x2) and an int8 64^3 (kpu_s1): every L3 slot holds
// exactly the bytes the program places there, at the cycle its tile is ready; DRAM ends with C bit
// for bit; the slot credits hold every cycle; the counts match the program's actions.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "l3_dma_rig.hpp"

#include <sw/kpu/cycle/memory/csp_l3_program.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::cycle;
using namespace sw::kpu::cycle::test;
namespace program = sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Picoseconds kLimit = 20'000'000'000;   // 20 ms

platform::DeviceSpecification device(const std::string& file) {
    return platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file).device(0);
}

// The RTL's matmul: C[8,8] = A[8,16] . B[16,8], int8 in 4x4 tiles (tests/program/test_csp_double_buffer.cpp).
std::string rtl_8x8x16() {
    return "csp 1.1\nprogram mm machine flat(l3 = 32, l2 = 2) {\n"
           "  tensor A[8,16] int8 tile 4x4 in;\n  tensor B[16,8] int8 tile 4x4 in;\n"
           "  tensor C[8,8] int32 tile 4x4 out;\n"
           "  for j in 0..2 {\n    resident B[:, j];\n    for i in 0..2 {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

std::string int8_64cubed() {
    csp::gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.dtype = "int8";
    return csp::gen::generate(o);
}

struct Case {
    CspL3Plan plan;
    program::TileProgram lb;   // L-B's result, every operand
};

Case prepare(const std::string& text, const platform::DeviceSpecification& dev) {
    const lang::Program ast = lang::parse(text);
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const program::TileProgram in = driver::csp_inputs(ast);
    auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    REQUIRE(!lb.skipped);
    Case c{plan_l3(p, in, lb.values), lb.values};
    return c;
}

// Runs the plan on DRAM -> DMA -> L3, and checks every slot at ready and DRAM at the end.
void run_and_check(const CspL3Plan& plan, std::size_t window) {
    L3DmaRig::Options o;
    o.slots = plan.slots;
    o.slot_bytes = plan.slot_bytes;
    o.banks = plan.banks;
    o.window = window;
    L3DmaRig r(plan.program, o);
    for (const auto& t : plan.inputs) r.channel.preload(t.address, t.data);
    std::map<std::size_t, std::vector<std::uint8_t>> seen;
    r.sim.add_invariant("watch", [&] {
        for (std::uint32_t s = 0; s < plan.slots; ++s)
            if (r.l3.ready(s) && !seen.count(r.l3.ticket(s))) seen[r.l3.ticket(s)] = r.l3.peek(s);
    });
    REQUIRE(r.sim.run_until_idle(kLimit));

    REQUIRE(seen.size() == plan.placed.size());
    for (const auto& [ticket, bytes] : plan.placed) {
        CAPTURE(ticket, plan.what.at(ticket).to_string());
        REQUIRE(seen.count(ticket));
        CHECK(seen.at(ticket) == bytes);
    }
    REQUIRE(!plan.outputs.empty());
    for (const auto& t : plan.outputs) {
        CAPTURE(t.tile.to_string());
        CHECK(r.channel.peek(t.address, t.bytes) == t.data);
    }
    CHECK(r.dma.dma().loads == plan.loads);
    CHECK(r.dma.dma().stores == plan.stores);
    CHECK(r.dma.dma().fills == plan.fills);
    CHECK(r.l3.counts().completions == plan.releases);
    CHECK(r.l3.counts().max_occupancy <= plan.slots);
}

}  // namespace

TEST_CASE("Rung 4: the RTL's 8x8x16 int8 matmul's L3 traffic, C hypothesized, bit for bit", "[cycle][csp][l3][dma]") {
    const Case c = prepare(rtl_8x8x16(), device("kpu_soc_2x2.json"));
    const CspL3Plan& plan = c.plan;
    // A[8,16] and B[16,8] are 8 tiles each; C[8,8] is 4, each stored once.
    CHECK(plan.stores == 4);
    CHECK(plan.fills == 4);
    CHECK(plan.outputs.size() == 4);
    CHECK(plan.loads >= 16);
    // 4x4 int8 tiles are 16 bytes, one beat padded; int32 C tiles are 64.
    CHECK(plan.slot_bytes == 64);
    for (const auto& t : plan.outputs) {
        // C is int32, four bytes an element, little-endian: L-B's words.
        const auto& op = c.lb.operand("C");
        for (std::uint32_t r = 0; r < 4; ++r)
            for (std::uint32_t k = 0; k < 4; ++k) {
                const std::uint32_t w = op.word(t.tile.ti * 4 + r, t.tile.tj * 4 + k);
                const std::size_t at = (r * 4 + k) * 4;
                CHECK((std::uint32_t(t.data[at]) | std::uint32_t(t.data[at + 1]) << 8 |
                       std::uint32_t(t.data[at + 2]) << 16 | std::uint32_t(t.data[at + 3]) << 24) == w);
            }
    }
    // An int8 A tile in DRAM: its 16 elements a byte each, row-major, then zeros to the beat.
    for (const auto& t : plan.inputs) {
        if (t.tile.operand != "A") continue;
        CAPTURE(t.tile.to_string());
        REQUIRE(t.bytes == 64);
        const auto& op = c.lb.operand("A");
        for (std::uint32_t r = 0; r < 4; ++r)
            for (std::uint32_t k = 0; k < 4; ++k)
                CHECK(t.data[r * 4 + k] == static_cast<std::uint8_t>(op.word(t.tile.ti * 4 + r, t.tile.tj * 4 + k)));
        for (std::size_t b = 16; b < 64; ++b) CHECK(t.data[b] == 0);
    }
    for (const std::size_t window : {1u, 8u}) {
        CAPTURE(window);
        run_and_check(plan, window);
    }
}

TEST_CASE("Rung 4: an int8 64^3 matmul's L3 traffic on kpu_s1, C hypothesized, bit for bit", "[cycle][csp][l3][dma]") {
    const Case c = prepare(int8_64cubed(), device("kpu_s1.json"));
    const CspL3Plan& plan = c.plan;
    CHECK(plan.stores == 16);   // C[64,64] in 16x16 tiles
    CHECK(plan.outputs.size() == 16);
    CHECK(plan.slot_bytes == 1024);   // a 16x16 int32 C tile
    for (const auto& t : plan.inputs)
        if (t.tile.operand == "A" || t.tile.operand == "B") CHECK(t.bytes == 256);   // 16x16 int8: a byte each
    run_and_check(plan, 16);
}

TEST_CASE("The plan refuses a DRAM layout past the end of the address space", "[cycle][csp][l3]") {
    // #557's review: a base near the top would wrap its alignment, and the tiles, to address 0.
    const auto dev = device("kpu_soc_2x2.json");
    const lang::Program ast = lang::parse(rtl_8x8x16());
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const program::TileProgram in = driver::csp_inputs(ast);
    const auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    CHECK_THROWS_WITH(plan_l3(p, in, lb.values, ~std::uint64_t{0}), ContainsSubstring("does not fit in the address space"));
    CHECK_THROWS_WITH(plan_l3(p, in, lb.values, ~std::uint64_t{0} - 0x1FFFF), ContainsSubstring("does not fit"));
    const CspL3Plan ok = plan_l3(p, in, lb.values, 0x100000);   // an ordinary layout is unchanged
    CHECK(ok.inputs.front().address == 0x100000);
}
