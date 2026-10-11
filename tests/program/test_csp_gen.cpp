// ============================================================================
// tests/program/test_csp_gen.cpp
// csp-gen (docs/plans/kpu-run-csp-programs.md step 1): every program the generator writes
// validates symbolically, compiles, and computes the same values however it is executed: the
// compiled action list at L-B, the stream at L-B (the oracle, docs/plans/retire-l0-derivations.md
// §3.2), and its own L0 trace replayed, bit for bit. Every schedule of one operator -- both
// orientations, every epilogue placement -- computes the same values, bit for bit. Its schedules
// use exactly the L3 they claim, and one slot less is refused.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace gen = sw::kpu::program::csp::gen;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::csp::VeOp;

namespace {

const char* result_of(const gen::Options& o) { return o.algo == "lu" ? "A" : "C"; }

// Generate, compile, run at L-B on the program's synthesized inputs (csp_inputs), through the
// action list, the stream and the trace. Returns the result operand's values: the oracle's.
std::vector<float> check_program(const gen::Options& o) {
    const std::string text = gen::generate(o);
    CHECK(text.find("for ") != std::string::npos);         // written with its loops
    const auto v = lang::validate(text);
    CHECK(v.peak_l3 == gen::slots_needed(o));

    const lang::Program ast = lang::parse(text);
    const TileProgram in = driver::csp_inputs(ast);
    const driver::CspReference ref = driver::csp_reference(ast, in, 1'000'000);
    REQUIRE(ref.values.has_value());
    REQUIRE(ref.trace.has_value());                         // its replay reproduced L-B
    const auto want = ref.values->operand(result_of(o)).values;

    csp::CspProgram p = lang::compile(text);
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    CHECK(p.reuse().peak_l3 == v.peak_l3);
    if (v.loads) CHECK(*v.loads == p.reuse().loads);

    csp::BehavioralInterpreter lb;
    lb.run(p);
    CHECK(lb.result().operand(result_of(o)).values == want);

    // The program's own trace, replayed, agrees with L-B.
    TileProgram trace_ref = *ref.trace;
    TileProgramReference().run(trace_ref);
    CHECK(trace_ref.operand(result_of(o)).values == want);
    return want;
}

}  // namespace

TEST_CASE("csp-gen: matmul, both orientations, compute the same values", "[program][csp][gen]") {
    // Inputs are exact eighths, so a matmul is exact in any order: the schedules agree bit for bit.
    std::map<Dim, std::vector<float>> first;               // size -> the first schedule's C
    for (const char* orient : {"col", "row"})
        for (auto [size, tile] : std::vector<std::pair<Dim, Dim>>{{64, 16}, {128, 32}, {80, 32}}) {
            gen::Options o;
            o.algo = "matmul";
            o.size = size;
            o.tile = tile;
            o.orient = orient;
            o.l3 = gen::slots_needed(o);                    // the smallest L3 the schedule accepts
            CAPTURE(orient, size, tile, o.l3);
            const auto c = check_program(o);
            if (!first.count(size)) first[size] = c;
            else CHECK(c == first[size]);
        }
}

TEST_CASE("csp-gen: linear, every placement and orientation, computes the same values",
          "[program][csp][gen]") {
    // A fused epilogue computes with the unfused one's element functions, so every placement and
    // both orientations of one activation agree bit for bit.
    std::map<std::string, std::vector<float>> first;       // activation -> the first schedule's C
    for (const char* place : {"fabric", "str.drain", "bm.egress", "unfused"})
        for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Gelu, ActivationFn::Silu, ActivationFn::Atan})
            for (const char* orient : {"col", "row"}) {
                gen::Options o;
                o.algo = "linear";
                o.size = 96;
                o.tile = 32;
                o.act = act;
                o.place = place;
                o.orient = orient;
                o.l3 = gen::slots_needed(o);
                CAPTURE(place, to_string(act), orient);
                const auto c = check_program(o);
                const std::string key = to_string(act);
                if (!first.count(key)) first[key] = c;
                else CHECK(c == first[key]);
            }
}

TEST_CASE("csp-gen: LU computes the same values through every path", "[program][csp][gen]") {
    for (auto [size, tile] : std::vector<std::pair<Dim, Dim>>{{64, 16}, {96, 32}, {48, 16}}) {
        gen::Options o;
        o.algo = "lu";
        o.size = size;
        o.tile = tile;
        o.l3 = gen::slots_needed(o);
        CAPTURE(size, tile);
        check_program(o);
    }
}

TEST_CASE("csp-gen: a schedule that does not fit the L3 is refused, with its slot count", "[program][csp][gen]") {
    gen::Options o;
    o.algo = "matmul";
    o.size = 256;
    o.tile = 32;
    o.l3 = 16;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("needs 17 L3 slots (2 panels of 8 tiles and the store's "
                                                          "writeback); the L3 holds 16"));
    o.l3 = 17;
    CHECK_NOTHROW(gen::generate(o));
    o.algo = "linear";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("needs 18 L3 slots"));
    o.place = "unfused";
    CHECK_NOTHROW(gen::generate(o));
    o.algo = "lu";
    o.size = 64;
    o.tile = 16;
    o.l3 = 15;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("the whole matrix, 4 x 4 tiles"));
}

TEST_CASE("csp-gen: placement is checked against the target", "[program][csp][gen]") {
    lang::Target t;
    t.bm = std::set<VeOp>{VeOp::Add, VeOp::Relu, VeOp::Gelu, VeOp::Silu};
    t.str = std::set<VeOp>{VeOp::Add, VeOp::Relu, VeOp::Gelu, VeOp::Silu, VeOp::Atan};
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 32;
    o.l3 = 64;
    o.act = ActivationFn::Atan;
    o.place = "bm.egress";
    CHECK_THROWS_WITH(gen::generate(o, &t), ContainsSubstring("atan @ bm.egress: the target's BlockMover vector unit "
                                                              "runs add, relu, gelu, silu, not atan"));
    o.place = "str.drain";
    CHECK_NOTHROW(gen::generate(o, &t));
    o.place = "fabric";
    CHECK_THROWS_WITH(gen::generate(o, &t), ContainsSubstring("atan @ fabric"));
    o.place = "unfused";                                   // calls in the fabric: no placement
    CHECK_NOTHROW(gen::generate(o, &t));
}

TEST_CASE("csp-gen: bad options are refused by name", "[program][csp][gen]") {
    gen::Options o;
    o.l3 = 64;
    o.algo = "conv";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("unknown algo 'conv'"));
    o.algo = "matmul";
    o.orient = "diagonal";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("orient 'diagonal'"));
    o.orient = "col";
    o.l3 = 0;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("L3 capacity (tiles) is required"));
    o.l3 = 64;
    o.algo = "linear";
    o.place = "dma";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("place 'dma'"));
}
