// ============================================================================
// tests/timing/test_csp_linear_lca.cpp
// The linear operator at L-CA (docs/plans/csp-language.md step 3): the epilogue's placement,
// timed. A tile context's stages run on the vector unit of the site the program put them on
// (movers.vector), overlapping the move they ride on; `fabric` stages charge the compute tile.
// Every placement computes the L0 reference bit for bit; what differs is the time and the
// traffic, which is the fusion benefit the driver exists to show.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace sw::kpu::timing;
namespace prog = sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

prog::TileProgram linear_inputs(prog::ActivationFn act) {
    prog::driver::ProgramSpec s;
    s.algo = "linear";
    s.size = 128;
    s.tile = 32;
    s.act = act;
    prog::TileProgram p = prog::driver::derive(s);
    prog::driver::fill(p, s);
    return p;
}

std::vector<float> reference(prog::ActivationFn act) {
    prog::TileProgram p = linear_inputs(act);
    prog::TileProgramReference().run(p);
    return p.operand("C").values;
}

std::string linear_source(const std::string& via, const std::string& unfused = "") {
    return std::string(R"(csp 1.0
program linear machine flat(l3 = 64) {
  tensor A[128,128] tile 32x32 in;
  tensor B[128,128] tile 32x32 in;
  vector b[128] tile 32 in;
  tensor C[128,128] tile 32x32 inout;
  for j in 0..4 {
    resident B[:, j], b[j];
    for i in 0..4 {
      resident A[i, :];
      acc C[i, j] in fabric {
        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }
      }
      store C[i, j])") + via + R"(;
      release A[i, :];
    }
)" + unfused + R"(    release B[:, j], b[j];
  }
}
)";
}

std::string unfused_epilogue(const std::string& act) {
    return "    for i in 0..4 {\n"
           "      resident C[i, j];\n"
           "      call add(C[i, j], b[j]) -> C[i, j];\n"
           "      call " + act + "(C[i, j]) -> C[i, j];\n"
           "      store C[i, j];\n"
           "      release C[i, j];\n"
           "    }\n";
}

// S1 with vector units: the BlockMovers' lack atan, the streamers' have it.
prog::platform::DeviceSpecification s1(std::size_t lanes = 16) {
    auto d = prog::platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    using VU = prog::platform::DeviceSpecification::Movers::VectorUnit;
    d.movers.bm_vector = VU{static_cast<prog::Dim>(lanes), 1.0, {"add", "relu", "gelu", "silu"}};
    d.movers.str_vector = VU{static_cast<prog::Dim>(lanes), 1.0, {"add", "relu", "gelu", "silu", "atan"}};
    return d;
}

ConcurrentTimingExecutor::Config machine(const prog::platform::DeviceSpecification& d) {
    auto c = csp_config_from(d)->config;
    c.max_cycles = 5'000'000;
    return c;
}

struct Run {
    CspDriver::Result r;
    std::size_t program_loads = 0;
};

Run run(const std::string& src, prog::ActivationFn act, const prog::platform::DeviceSpecification& d) {
    const lang::Target target = lang::target_from(d);
    csp::CspProgram p = lang::compile(src, &target);
    const prog::TileProgram in = linear_inputs(act);
    for (const char* name : {"A", "B", "b"}) p.source.operand(name).values = in.operand(name).values;
    ConcurrentTimingExecutor exec(machine(d));
    Run out;
    out.r = CspDriver(exec, p).run();
    out.program_loads = p.reuse().loads;
    return out;
}

}  // namespace

TEST_CASE("Linear at L-CA: every placement computes the reference; the placements differ in time and traffic",
          "[timing][csp][linear][lca]") {
    const auto d = s1();
    struct Case { const char* name; prog::ActivationFn act; std::string src; };
    const std::vector<Case> cases = {
        {"fabric", prog::ActivationFn::Relu, linear_source(" via add(b[j]) @ fabric, relu @ fabric")},
        {"str.drain", prog::ActivationFn::Relu, linear_source(" via add(b[j]) @ str.drain, relu @ str.drain")},
        {"bm.egress", prog::ActivationFn::Relu, linear_source(" via add(b[j]) @ bm.egress, relu @ bm.egress")},
        {"unfused", prog::ActivationFn::Relu, linear_source("", unfused_epilogue("relu"))},
        // atan where the machine has it: the streamers' unit (the BlockMovers' lacks it).
        {"atan@str", prog::ActivationFn::Atan, linear_source(" via add(b[j]) @ str.drain, atan @ str.drain")},
        {"gelu split", prog::ActivationFn::Gelu, linear_source(" via add(b[j]) @ fabric, gelu @ bm.egress")},
    };
    std::printf("linear 128^3/32^3 on S1 (VE 16 lanes): placement, cycles, DRAM bytes, cf busy, "
                "fabric epilogue, str VE busy/bound, bm VE busy/bound\n");
    std::uint64_t fused_bytes = 0;
    Cycle fused_cycles = 0;
    for (const Case& c : cases) {
        CAPTURE(c.name);
        const Run x = run(c.src, c.act, d);
        REQUIRE(x.r.completed);
        CHECK(x.r.values.operand("C").values == reference(c.act));
        CHECK(x.r.dram_loads == x.program_loads);
        std::printf("  %-10s %7llu  %7llu  %6llu  %4llu  %5llu/%-4llu  %5llu/%-4llu\n", c.name,
                    static_cast<unsigned long long>(x.r.cycles), static_cast<unsigned long long>(x.r.dram_bytes),
                    static_cast<unsigned long long>(x.r.cf_busy), static_cast<unsigned long long>(x.r.ve.fabric),
                    static_cast<unsigned long long>(x.r.ve.str_busy), static_cast<unsigned long long>(x.r.ve.str_bound),
                    static_cast<unsigned long long>(x.r.ve.bm_busy), static_cast<unsigned long long>(x.r.ve.bm_bound));
        const std::string name = c.name;
        // 16 result tiles of 32 x 32, two stages each.
        if (name == "fabric") {
            CHECK(x.r.ve.fabric == 16 * 2 * 1);           // 1024 elements at 8192 per cycle
            CHECK(x.r.ve.str_busy == 0);
            CHECK(x.r.ve.bm_busy == 0);
            fused_bytes = x.r.dram_bytes;
            fused_cycles = x.r.cycles;
        }
        if (name == "str.drain") {
            CHECK(x.r.ve.str_busy == 16 * 2 * 64);        // 1024 elements at 16 per cycle
            CHECK(x.r.ve.bm_busy == 0);
            CHECK(x.r.dram_bytes == fused_bytes);
        }
        if (name == "bm.egress") {
            CHECK(x.r.ve.bm_busy == 16 * 2 * 64);
            CHECK(x.r.ve.str_busy == 0);
            CHECK(x.r.dram_bytes == fused_bytes);
        }
        if (name == "unfused") {
            // C makes a second DRAM round trip: 16 tiles of 4 KiB read back and written again.
            CHECK(x.r.dram_bytes == fused_bytes + 2 * 16 * 4096);
            CHECK(x.r.ve.str_busy + x.r.ve.bm_busy + x.r.ve.fabric == 0);
            CHECK(x.r.cycles > fused_cycles);           // the round trip costs time as well as bytes
        }
    }
}

TEST_CASE("Linear at L-CA: the stream runs it too, in a window", "[timing][csp][linear][lca][stream]") {
    const auto d = s1();
    const lang::Target target = lang::target_from(d);
    const prog::TileProgram in = linear_inputs(prog::ActivationFn::Silu);
    for (const std::string& src : {linear_source(" via add(b[j]) @ str.drain, silu @ bm.egress"),
                                   linear_source("", unfused_epilogue("silu"))}) {
        for (std::size_t w : {4u, 64u}) {
            CAPTURE(w);
            lang::ActionStream s(lang::parse(src), target);
            prog::TileProgram operands = s.operands();
            for (const auto& name : operands.operand_order()) operands.operand(name).values = in.operand(name).values;
            ConcurrentTimingExecutor exec(machine(d));
            const auto r = CspDriver(exec, s, operands, w).run();
            REQUIRE(r.completed);
            CHECK(r.values.operand("C").values == reference(prog::ActivationFn::Silu));
        }
    }
}

TEST_CASE("Linear at L-CA: a vector unit slower than its move sets the pace", "[timing][csp][linear][lca]") {
    const std::string src = linear_source(" via add(b[j]) @ bm.egress, relu @ bm.egress");
    const Run fast = run(src, prog::ActivationFn::Relu, s1(16));
    const Run slow = run(src, prog::ActivationFn::Relu, s1(1));
    REQUIRE(fast.r.completed);
    REQUIRE(slow.r.completed);
    CHECK(slow.r.values.operand("C").values == fast.r.values.operand("C").values);
    CHECK(slow.r.ve.bm_busy == 16 * 2 * 1024);
    CHECK(slow.r.ve.bm_bound > fast.r.ve.bm_bound);
    CHECK(slow.r.cycles > fast.r.cycles);
    std::printf("bm.egress at 1 lane: %llu cycles (bound %llu) vs 16 lanes: %llu (bound %llu)\n",
                static_cast<unsigned long long>(slow.r.cycles), static_cast<unsigned long long>(slow.r.ve.bm_bound),
                static_cast<unsigned long long>(fast.r.cycles), static_cast<unsigned long long>(fast.r.ve.bm_bound));
}

TEST_CASE("Linear at L-CA: a stage on a site with no vector unit is refused", "[timing][csp][linear][lca]") {
    auto d = s1();
    d.movers.bm_vector.reset();
    // Compiled without the target (the language check is skipped), the executor still refuses.
    csp::CspProgram p = lang::compile(linear_source(" via add(b[j]) @ bm.egress, relu @ bm.egress"));
    const prog::TileProgram in = linear_inputs(prog::ActivationFn::Relu);
    for (const char* name : {"A", "B", "b"}) p.source.operand(name).values = in.operand(name).values;
    ConcurrentTimingExecutor exec(machine(d));
    CHECK_THROWS_WITH(CspDriver(exec, p).run(), ContainsSubstring("runs on the BlockMovers, which have no vector unit"));
}

