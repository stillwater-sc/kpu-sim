// ============================================================================
// tests/program/test_csp_run.cpp
// Running a CSP program at each level (docs/plans/kpu-run-csp-programs.md step 2): the program
// is the input, L0 its oracle. L-B and L-CA compute the trace's L0 reference bit for bit, on
// every operand; L-CA reads DRAM exactly as often as the program loads; a level that cannot
// run a program on a machine says why.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <algorithm>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
namespace gen = sw::kpu::program::csp::gen;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification s1(bool vector_units = true) {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    if (vector_units) {
        using VU = platform::DeviceSpecification::Movers::VectorUnit;
        d.movers.bm_vector = VU{16, 1.0, {"add", "relu", "gelu", "silu"}};
        d.movers.str_vector = VU{16, 1.0, {"add", "relu", "gelu", "silu", "atan"}};
    }
    return d;
}

CspRunResult run(const std::string& text, const platform::DeviceSpecification* d,
                 std::vector<ExecutionLevel> levels = {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                                       ExecutionLevel::CycleAccurate}) {
    const lang::Program ast = lang::parse(text);
    CspRunRequest req{ast, d, csp_inputs(ast), levels};
    return run_csp(req);
}

// Every level that ran computes the oracle, on every operand.
void check_values(const CspRunResult& r) {
    REQUIRE(r.reference.has_value());
    for (const auto& o : r.levels) {
        if (o.skipped) continue;
        CAPTURE(to_string(o.level));
        for (const auto& name : r.reference->operand_order())
            CHECK(o.values.operand(name).values == r.reference->operand(name).values);
    }
}

}  // namespace

TEST_CASE("CSP run: matmul and the linear operator at L-B, L-T1 and L-CA compute the oracle", "[program][csp][run]") {
    const auto d = s1();
    std::vector<gen::Options> cases;
    for (const char* orient : {"col", "row"}) {
        gen::Options o;
        o.algo = "matmul";
        o.size = 96;
        o.tile = 32;
        o.orient = orient;
        o.l3 = 64;
        cases.push_back(o);
    }
    for (const char* place : {"fabric", "str.drain", "bm.egress", "unfused"}) {
        gen::Options o;
        o.algo = "linear";
        o.size = 96;
        o.tile = 32;
        o.l3 = 64;
        o.place = place;
        o.act = std::string(place) == "fabric" ? ActivationFn::Relu : ActivationFn::Silu;
        cases.push_back(o);
    }
    for (const auto& o : cases) {
        CAPTURE(o.algo, o.orient, o.place);
        const std::string text = gen::generate(o);
        const auto r = run(text, &d);
        REQUIRE(r.levels.size() == 3);
        CHECK_FALSE(r.levels[0].skipped);
        REQUIRE_FALSE(r.levels[1].skipped);
        REQUIRE_FALSE(r.levels[2].skipped);
        check_values(r);
        // L-T1: the program's residency -- its loads, its L3 -- and a deterministic schedule.
        const auto& lt1 = r.levels[1];
        CHECK(lt1.has_timing);
        CHECK(lt1.makespan > 0);
        CHECK(lt1.dram_loads == *r.validation.loads);
        CHECK(lt1.dram_stores == *r.validation.stores);
        CHECK(lt1.peak_l3 >= r.validation.peak_l3);
        CHECK(lt1.peak_l3 <= static_cast<std::size_t>(o.l3));
        CHECK(run(text, &d, {ExecutionLevel::BlockSequential}).levels[0].makespan == lt1.makespan);
        const auto& lca = r.levels[2];
        CHECK(lca.has_timing);
        CHECK(lca.makespan > 0);
        CHECK(lca.dram_loads == *r.validation.loads);         // the program's loads, exactly
        CHECK(lca.dram_stores == *r.validation.stores);
        CHECK(r.levels[0].peak_l3 == r.validation.peak_l3);    // L-B's residency is the program's
    }
}

TEST_CASE("CSP run: inputs are the operands read first; results written first start at zero",
          "[program][csp][run]") {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 32;
    o.l3 = 64;
    o.place = "unfused";                                       // C is inout, written first
    const TileProgram unfused = csp_inputs(lang::parse(gen::generate(o)));
    for (float v : unfused.operand("C").values) REQUIRE(v == 0.0f);
    CHECK(unfused.operand("A").values != std::vector<float>(unfused.operand("A").values.size(), 0.0f));
    CHECK(unfused.operand("b").values != std::vector<float>(unfused.operand("b").values.size(), 0.0f));
    o.algo = "lu";                                             // A is inout, read first
    o.l3 = 16;
    const TileProgram lu = csp_inputs(lang::parse(gen::generate(o)));
    CHECK(lu.operand("A").values != std::vector<float>(lu.operand("A").values.size(), 0.0f));
}

TEST_CASE("CSP run: LU runs at L-B and L-T1; L-CA says why it does not", "[program][csp][run]") {
    const auto d = s1();
    gen::Options o;
    o.algo = "lu";
    o.size = 64;
    o.tile = 16;
    o.l3 = 64;
    const auto r = run(gen::generate(o), &d);
    check_values(r);                                           // L-B and L-T1
    CHECK_FALSE(r.levels[1].skipped);
    CHECK(r.levels[1].dram_loads == 16);                       // A loaded once, whole
    REQUIRE(r.levels[2].skipped);
    CHECK_THAT(*r.levels[2].skipped, ContainsSubstring("getrf (LU) at L-CA is a later increment"));
}

TEST_CASE("CSP run: a level that cannot run this program on this machine says why", "[program][csp][run]") {
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 32;
    o.l3 = 64;
    const lang::Program ast = lang::parse(gen::generate(o));
    CHECK_THAT(*csp_level_supported(ExecutionLevel::CycleAccurate, ast, nullptr),
               ContainsSubstring("deployment spec (--deploy)"));
    CHECK_FALSE(csp_level_supported(ExecutionLevel::BlockSequential, ast, nullptr));   // a default device
    const auto t4 = platform::read_spec_file("tests/program/deploy/kpu_t4.json").device(0);
    CHECK_THAT(*csp_level_supported(ExecutionLevel::CycleAccurate, ast, &t4), ContainsSubstring("level 2 distributes"));
    o.l3 = 200;                                                // more L3 than S1 has
    const auto d = s1();
    CHECK_THAT(*csp_level_supported(ExecutionLevel::CycleAccurate, lang::parse(gen::generate(o)), &d),
               ContainsSubstring("written for an L3 of 200 tiles; the machine's holds 128"));
    const std::string renamed = R"(csp 1.0
program t machine flat(l3 = 8) {
  tensor X[32,32] tile 32x32 in;
  tensor W[32,32] tile 32x32 in;
  tensor Y[32,32] tile 32x32 out;
  resident X[0, 0], W[0, 0];
  acc Y[0, 0] in fabric { call gemm(X[0, 0], W[0, 0]) +-> Y[0, 0]; }
  store Y[0, 0];
  release X[0, 0], W[0, 0];
}
)";
    CHECK_THAT(*csp_level_supported(ExecutionLevel::CycleAccurate, lang::parse(renamed), &d),
               ContainsSubstring("'X' is not one of them"));
    // ...and L-B runs it, computing the oracle.
    check_values(run(renamed, &d, {ExecutionLevel::Behavioral}));
}

TEST_CASE("CSP run: above the trace limit there is no oracle, and the levels still agree", "[program][csp][run]") {
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 32;
    o.l3 = 64;
    const auto d = s1();
    const lang::Program ast = lang::parse(gen::generate(o));
    CspRunRequest req{ast, &d, csp_inputs(ast), {ExecutionLevel::Behavioral, ExecutionLevel::CycleAccurate}};
    req.trace_limit = 10;
    const auto r = run_csp(req);
    CHECK_FALSE(r.reference.has_value());
    CHECK_THAT(r.reference_note, ContainsSubstring("more than 10 actions: no L0 reference"));
    REQUIRE_FALSE(r.levels[1].skipped);
    CHECK(r.levels[0].values.operand("C").values == r.levels[1].values.operand("C").values);
}

TEST_CASE("CSP run: a program the validator refuses is refused against the machine", "[program][csp][run]") {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 32;
    o.l3 = 64;
    o.act = ActivationFn::Atan;
    o.place = "bm.egress";
    const auto d = s1();                                       // the BlockMovers lack atan
    CHECK_THROWS_WITH(run(gen::generate(o), &d), ContainsSubstring("atan @ bm.egress"));
}

TEST_CASE("CSP run: L-T1's credits are the program's -- a Load waits for the Release that frees its slot",
          "[program][csp][run][lt1]") {
    // Two slots. Each k-step loads A[0,k] and B[k,0] and releases them, so every Load after the
    // first two waits for a Release: L-T1 takes credits only from the program's Releases.
    const std::string text = R"(csp 1.0
program streamed machine flat(l3 = 3) {
  tensor A[32,128] tile 32x32 in;
  tensor B[128,32] tile 32x32 in;
  tensor C[32,32] tile 32x32 out;
  acc C[0, 0] in fabric {
    for k in 0..4 {
      resident A[0, k], B[k, 0];
      call gemm(A[0, k], B[k, 0]) +-> C[0, 0];
      release A[0, k], B[k, 0];
    }
  }
  store C[0, 0];
}
)";
    const lang::Program ast = lang::parse(text);
    TileProgram inputs = csp_inputs(ast);
    csp::TransactionalInterpreter lt(characterize::DeviceDescriptor::single(), 3);
    lt.begin(inputs);
    lang::ActionStream s(ast);
    while (auto e = s.next()) lt.step(e->action, e->op ? &*e->op : nullptr);
    const auto st = lt.finish();
    CHECK(st.dram_loads == 8);
    CHECK(st.peak_l3 <= 3);
    CHECK(st.credit_stalls > 0);
    // Every Load starts no earlier than the credit it took: the k-th Load (k >= 3) after the
    // Release that freed a slot.
    std::vector<csp::TransactionalInterpreter::Cycle> loads, releases;
    for (const auto& rec : lt.records()) {
        if (rec.kind == csp::Action::Kind::Load) loads.push_back(rec.start);
        if (rec.kind == csp::Action::Kind::Release) releases.push_back(rec.start);
    }
    REQUIRE(loads.size() == 8);
    REQUIRE(releases.size() >= 2);
    std::sort(releases.begin(), releases.end());
    // Three slots: the 4th Load needs the 1st Release, the 5th the 2nd, and so on.
    for (std::size_t i = 3; i < loads.size(); ++i) CHECK(loads[i] >= releases[i - 3]);
    // ...and the values are L-B's.
    csp::BehavioralInterpreter lb;
    lb.begin(inputs);
    lang::ActionStream s2(ast);
    while (auto e = s2.next()) lb.step(e->action, e->op ? &*e->op : nullptr);
    lb.finish();
    CHECK(lt.result().operand("C").values == lb.result().operand("C").values);
}
