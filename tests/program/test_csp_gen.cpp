// ============================================================================
// tests/program/test_csp_gen.cpp
// csp-gen (docs/plans/kpu-run-csp-programs.md step 1): every program the generator writes
// validates symbolically, compiles, and -- executed at L-B -- computes what the L0 derivation
// of the same operator computes, bit for bit. Its schedules use exactly the L3 they claim, and
// one slot less is refused.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace gen = sw::kpu::program::csp::gen;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::csp::VeOp;

namespace {

// The L0 derivation of the same operator, filled and run: the values the program must compute.
TileProgram derived(const gen::Options& o, bool run) {
    driver::ProgramSpec s;
    s.algo = o.algo;
    s.size = o.size;
    s.tile = o.tile;
    s.act = o.act;
    TileProgram p = driver::derive(s);
    driver::fill(p, s);
    if (run) TileProgramReference().run(p);
    return p;
}

const char* result_of(const gen::Options& o) { return o.algo == "lu" ? "A" : "C"; }

// Generate, compile, run at L-B on the derivation's inputs, through the trace and the stream.
void check_program(const gen::Options& o) {
    const std::string text = gen::generate(o);
    CHECK(text.find("for ") != std::string::npos);         // written with its loops
    const auto v = lang::validate(text);
    CHECK(v.peak_l3 == gen::slots_needed(o));

    csp::CspProgram p = lang::compile(text);
    const TileProgram in = derived(o, false);
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    CHECK(p.reuse().peak_l3 == v.peak_l3);
    if (v.loads) CHECK(*v.loads == p.reuse().loads);

    const auto want = derived(o, true).operand(result_of(o)).values;
    csp::BehavioralInterpreter lb;
    lb.run(p);
    CHECK(lb.result().operand(result_of(o)).values == want);

    // The program's own trace is its oracle, and agrees with the derivation.
    TileProgram trace_ref = p.source;
    TileProgramReference().run(trace_ref);
    CHECK(trace_ref.operand(result_of(o)).values == want);

    lang::ActionStream s(lang::parse(text));
    TileProgram operands = s.operands();
    for (const auto& name : operands.operand_order()) operands.operand(name).values = in.operand(name).values;
    csp::BehavioralInterpreter ls;
    ls.begin(operands);
    while (auto e = s.next()) ls.step(e->action, e->op ? &*e->op : nullptr);
    ls.finish();
    CHECK(ls.result().operand(result_of(o)).values == want);
}

}  // namespace

TEST_CASE("csp-gen: matmul, both orientations, computes the derivation's values", "[program][csp][gen]") {
    for (const char* orient : {"col", "row"})
        for (auto [size, tile] : std::vector<std::pair<Dim, Dim>>{{64, 16}, {128, 32}, {80, 32}}) {
            gen::Options o;
            o.algo = "matmul";
            o.size = size;
            o.tile = tile;
            o.orient = orient;
            o.l3 = gen::slots_needed(o);                    // the smallest L3 the schedule accepts
            CAPTURE(orient, size, tile, o.l3);
            check_program(o);
        }
}

TEST_CASE("csp-gen: linear, every placement and activation, computes the derivation's values",
          "[program][csp][gen]") {
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
                check_program(o);
            }
}

TEST_CASE("csp-gen: LU computes the derivation's values", "[program][csp][gen]") {
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
