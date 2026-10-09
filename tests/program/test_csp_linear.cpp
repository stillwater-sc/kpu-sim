// ============================================================================
// tests/program/test_csp_linear.cpp
// The linear operator, Y = act(X . W + b), at L-B (docs/plans/csp-language.md step 2).
//
// L0 gains the epilogue ops (BiasAdd, Activation: relu, gelu, silu, atan). The CSP language
// gains vector operands and tile contexts, `via op @ place`: the epilogue runs on the result's
// way out, in the fabric, on the streamer's drain, or on the BlockMover's egress. Every
// placement, and the unfused program, computes the L0 reference bit for bit: one set of
// element functions serves the L0 kernels and every context stage. Where a stage may run is
// the machine's: `atan @ bm.egress` on a BlockMover without it is refused before anything runs.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/emit.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lower.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <string>
#include <vector>

using namespace sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;

namespace {

// The derived linear operator (128^3 in 32 x 32 tiles), its inputs filled, before the run.
TileProgram linear_inputs(ActivationFn act) {
    driver::ProgramSpec s;
    s.algo = "linear";
    s.size = 128;
    s.tile = 32;
    s.act = act;
    TileProgram p = driver::derive(s);
    driver::fill(p, s);
    return p;
}

TileProgram linear_reference(ActivationFn act) {
    TileProgram p = linear_inputs(act);
    TileProgramReference().run(p);
    return p;
}

// The written linear operator. `via` is the store's context; `unfused` (if set) is the
// epilogue as calls on the stored result, read back.
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

// Run a compiled program at L-B on the derived inputs; return C as DRAM holds it.
std::vector<float> run_lb(csp::CspProgram p, ActivationFn act) {
    const TileProgram in = linear_inputs(act);
    for (const char* name : {"A", "B", "b"}) p.source.operand(name).values = in.operand(name).values;
    csp::BehavioralInterpreter lb;
    lb.run(p);
    return lb.result().operand("C").values;
}

// The same, pulled from the program's stream.
std::vector<float> run_lb_stream(const std::string& src, ActivationFn act) {
    const TileProgram in = linear_inputs(act);
    lang::ActionStream s(lang::parse(src));
    TileProgram operands = s.operands();
    for (const auto& name : operands.operand_order()) {
        auto& o = operands.operand(name);
        o.values = in.operand(name).values;
    }
    csp::BehavioralInterpreter lb;
    lb.begin(operands);
    while (auto e = s.next()) lb.step(e->action, e->op ? &*e->op : nullptr);
    lb.finish();
    return lb.result().operand("C").values;
}

std::string compile_error(const std::string& src, const lang::Target* t = nullptr) {
    try {
        (void)lang::compile(src, t);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "compiled";
}
std::string validate_error(const std::string& src, const lang::Target* t = nullptr) {
    try {
        (void)lang::validate(src, t);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "validated";
}

// S1 with vector units: the BlockMovers' lack atan, the streamers' have it.
platform::DeviceSpecification s1_with_vector_units() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.movers.bm_vector = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu", "gelu", "silu"}};
    d.movers.str_vector =
        platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu", "gelu", "silu", "atan"}};
    return d;
}

}  // namespace

TEST_CASE("Linear: the epilogue's element functions", "[program][csp][linear]") {
    CHECK(activate(-2.5f, ActivationFn::Relu) == 0.0f);
    CHECK(activate(3.25f, ActivationFn::Relu) == 3.25f);
    // Against values computed independently of the kernels.
    CHECK(activate(1.0f, ActivationFn::Gelu) == Catch::Approx(0.8413447460685429).epsilon(1e-6));
    CHECK(activate(-1.0f, ActivationFn::Gelu) == Catch::Approx(-0.15865525393145707).epsilon(1e-6));
    CHECK(activate(1.0f, ActivationFn::Silu) == Catch::Approx(0.7310585786300049).epsilon(1e-6));
    CHECK(activate(-2.0f, ActivationFn::Silu) == Catch::Approx(-0.23840584404423515).epsilon(1e-6));
    CHECK(activate(1.0f, ActivationFn::Atan) == Catch::Approx(0.7853981633974483).epsilon(1e-6));
    CHECK(activate(-10.0f, ActivationFn::Atan) == Catch::Approx(-1.4711276743037347).epsilon(1e-6));
}

TEST_CASE("Linear: L0 computes act(A . B + b)", "[program][csp][linear]") {
    for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Gelu, ActivationFn::Silu, ActivationFn::Atan}) {
        CAPTURE(to_string(act));
        const TileProgram in = linear_inputs(act);
        TileProgram ref = in;
        const auto sum = TileProgramReference().run(ref);
        CHECK(sum.epilogues == 2 * 16);
        // Directly, in double: the kernels' float accumulation agrees within ADR 0001 D5's rtol.
        const auto& A = in.operand("A");
        const auto& B = in.operand("B");
        const auto& b = in.operand("b");
        bool any_negative = false, any_positive = false;
        for (Dim r = 0; r < 128; r += 7)
            for (Dim c = 0; c < 128; c += 5) {
                double y = b.at(c, 0);
                for (Dim k = 0; k < 128; ++k) y += double(A.at(r, k)) * B.at(k, c);
                any_negative = any_negative || y < 0;
                any_positive = any_positive || y > 0;
                const float want = activate(static_cast<float>(y), act);
                CHECK(ref.operand("C").at(r, c) == Catch::Approx(want).epsilon(1e-4).margin(1e-6));
            }
        CHECK(any_negative);            // the activation sees both sides of zero
        CHECK(any_positive);
    }
}

TEST_CASE("Linear: the epilogue ops serialize, opset 1.1.0", "[program][csp][linear][l0]") {
    const TileProgram in = linear_inputs(ActivationFn::Silu);
    serialize::WriteOptions opt;
    opt.include_values = true;
    const std::string text = serialize::to_string(in, opt);
    CHECK(text.find("OPSET tile 1.1.0\n") != std::string::npos);
    CHECK(text.find("OP kind=BIAS_ADD in=") != std::string::npos);
    CHECK(text.find("OP kind=ACTIVATION out=") != std::string::npos);
    CHECK(text.find(" act=silu") != std::string::npos);
    TileProgram back = serialize::from_string(text);
    REQUIRE(back.ops().size() == in.ops().size());
    TileProgram ref = in;
    TileProgramReference().run(ref);
    TileProgramReference().run(back);
    CHECK(back.operand("C").values == ref.operand("C").values);

    std::string no_act = text;
    no_act.erase(no_act.find(" act=silu"), 9);
    CHECK_THROWS_WITH(serialize::from_string(no_act), ContainsSubstring("act"));
    std::string bad_act = text;
    bad_act.replace(bad_act.find(" act=silu"), 9, " act=tanh");
    CHECK_THROWS_WITH(serialize::from_string(bad_act), ContainsSubstring("act=tanh: relu, gelu, silu or atan"));
}

TEST_CASE("Linear: the lowered L0 runs at L-B -- the reference, bit for bit", "[program][csp][linear]") {
    for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Atan}) {
        CAPTURE(to_string(act));
        const TileProgram in = linear_inputs(act);
        for (std::size_t cap : {8u, 64u}) {
            const csp::CspProgram p = csp::lower(in, {cap});
            csp::BehavioralInterpreter lb;
            lb.run(p);
            CHECK(lb.result().operand("C").values == linear_reference(act).operand("C").values);
        }
    }
    // The emitter does not choose a placement for the epilogue: the program is written.
    CHECK_THROWS_WITH(lang::emit(linear_inputs(ActivationFn::Relu), 64),
                      ContainsSubstring("placed by a tile context"));
}

TEST_CASE("Linear: every placement of the epilogue, and none, computes the reference",
          "[program][csp][linear][lang]") {
    struct Case { ActivationFn act; std::string via; };
    std::vector<Case> cases;
    // The fabric runs add and relu (what execute_matmul builds today).
    cases.push_back({ActivationFn::Relu, " via add(b[j]) @ fabric, relu @ fabric"});
    for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Gelu, ActivationFn::Silu, ActivationFn::Atan})
        for (const char* place : {"str.drain", "bm.egress"})
            cases.push_back({act, std::string(" via add(b[j]) @ ") + place + ", " + to_string(act) + " @ " + place});
    // Split across places, in path order.
    cases.push_back({ActivationFn::Gelu, " via add(b[j]) @ fabric, gelu @ bm.egress"});
    cases.push_back({ActivationFn::Atan, " via add(b[j]) @ str.drain, atan @ bm.egress"});

    for (const Case& c : cases) {
        CAPTURE(c.via);
        const std::string src = linear_source(c.via);
        const csp::CspProgram p = lang::compile(src);
        const auto want = linear_reference(c.act).operand("C").values;
        CHECK(run_lb(p, c.act) == want);
        CHECK(run_lb_stream(src, c.act) == want);
        // Symbolic validation agrees with the trace.
        const auto v = lang::validate(src);
        const auto r = p.reuse();
        CHECK(v.peak_l3 == r.peak_l3);
        CHECK(v.loads == std::optional<std::uint64_t>(r.loads));
        CHECK(v.stores == std::optional<std::uint64_t>(r.stores));
        // The printed program compiles to the same actions, contexts included, and is a fixed point.
        const std::string printed = lang::print(p);
        const csp::CspProgram q = lang::compile(printed);
        REQUIRE(q.actions.size() == p.actions.size());
        for (std::size_t i = 0; i < p.actions.size(); ++i) {
            CHECK(q.actions[i].context.size() == p.actions[i].context.size());
            for (std::size_t k = 0; k < std::min(q.actions[i].context.size(), p.actions[i].context.size()); ++k) {
                CHECK(q.actions[i].context[k].op == p.actions[i].context[k].op);
                CHECK(q.actions[i].context[k].place == p.actions[i].context[k].place);
                CHECK(q.actions[i].context[k].arg.to_string() == p.actions[i].context[k].arg.to_string());
            }
        }
        CHECK(lang::print(q) == printed);
    }

    SECTION("unfused: the result makes a second DRAM round trip, and computes the same") {
        for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Gelu, ActivationFn::Silu, ActivationFn::Atan}) {
            CAPTURE(to_string(act));
            const std::string src = linear_source("", unfused_epilogue(to_string(act)));
            const csp::CspProgram p = lang::compile(src);
            const auto want = linear_reference(act).operand("C").values;
            CHECK(run_lb(p, act) == want);
            CHECK(run_lb_stream(src, act) == want);
            const csp::CspProgram fused = lang::compile(linear_source(" via add(b[j]) @ bm.egress, relu @ bm.egress"));
            // 16 more loads (C read back) and 16 more stores than the fused program.
            CHECK(p.reuse().loads == fused.reuse().loads + 16);
            CHECK(p.reuse().stores == fused.reuse().stores + 16);
            CHECK(lang::validate(src).loads == std::optional<std::uint64_t>(p.reuse().loads));
            const std::string printed = lang::print(p);
            CHECK(lang::print(lang::compile(printed)) == printed);
        }
    }
}

TEST_CASE("Linear: a stage runs where the machine has it, or the program is refused",
          "[program][csp][linear][lang][placement]") {
    const auto spec = s1_with_vector_units();
    const lang::Target target = lang::target_from(spec);

    // atan on the BlockMover, which lacks it: refused by line, by the walker and the validator.
    const std::string misplaced = linear_source(" via add(b[j]) @ bm.egress, atan @ bm.egress");
    const char* why = "csp line 14: atan @ bm.egress: the target's BlockMover vector unit runs add, relu, gelu, silu, "
                      "not atan; place it where the machine has it";
    CHECK_THAT(compile_error(misplaced, &target), ContainsSubstring(why));
    CHECK_THAT(validate_error(misplaced, &target), ContainsSubstring(why));
    CHECK_THROWS_WITH(([&] {
                          lang::ActionStream s(lang::parse(misplaced), target);
                          while (s.next()) {}
                      }()),
                      ContainsSubstring(why));
    // Without a target, placement is not checked: the values do not depend on it.
    CHECK(compile_error(misplaced) == "compiled");

    // Its legal placement, on the streamer's drain, runs and computes the reference.
    const std::string legal = linear_source(" via add(b[j]) @ bm.egress, atan @ str.drain");
    CHECK_THAT(compile_error(legal, &target), ContainsSubstring("listed after a stage @ bm.egress"));
    const std::string ordered = linear_source(" via add(b[j]) @ str.drain, atan @ str.drain");
    const csp::CspProgram p = lang::compile(ordered, &target);
    CHECK(lang::validate(ordered, &target).peak_l3 == p.reuse().peak_l3);
    CHECK(run_lb(p, ActivationFn::Atan) == linear_reference(ActivationFn::Atan).operand("C").values);

    // The fabric runs add and relu only.
    CHECK_THAT(compile_error(linear_source(" via add(b[j]) @ fabric, gelu @ fabric"), &target),
               ContainsSubstring("gelu @ fabric: the target's compute fabric vector unit runs add, relu, not gelu"));
    // A machine without a streamer vector unit.
    auto bare = spec;
    bare.movers.str_vector.reset();
    const lang::Target bare_target = lang::target_from(bare);
    CHECK_THAT(compile_error(ordered, &bare_target),
               ContainsSubstring("add @ str.drain: the target's streamer has no vector unit"));
}

TEST_CASE("Linear: movers.vector in the deployment spec", "[program][csp][linear][spec]") {
    platform::DeploymentSpec spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    spec.devices.at(0).movers.bm_vector =
        platform::DeviceSpecification::Movers::VectorUnit{32, 0.5, {"add", "relu"}};
    const std::string text = platform::to_json(spec);
    CHECK(text.find("\"vector\"") != std::string::npos);
    const platform::DeploymentSpec back = platform::from_json(text);
    REQUIRE(back.devices.at(0).movers.bm_vector.has_value());
    CHECK(back.devices.at(0).movers.bm_vector->lanes == 32);
    CHECK(back.devices.at(0).movers.bm_vector->rate == 0.5);
    CHECK(back.devices.at(0).movers.bm_vector->ops == std::vector<std::string>{"add", "relu"});
    CHECK_FALSE(back.devices.at(0).movers.str_vector.has_value());
    CHECK(platform::deployment_digest(back) == platform::deployment_digest(spec));

    std::string bad = text;
    bad.replace(bad.find("\"relu\""), 6, "\"tanh\"");
    CHECK_THROWS_WITH(platform::from_json(bad), ContainsSubstring("'tanh' is not a vector operation"));
}

TEST_CASE("Linear: what the language refuses, by line", "[program][csp][linear][lang]") {
    auto err = [](const std::string& via) { return compile_error(linear_source(via)); };
    CHECK_THAT(err(" via relu @ bm.egress, add(b[j]) @ fabric"),
               ContainsSubstring("csp line 14: add @ fabric is listed after a stage @ bm.egress"));
    CHECK_THAT(err(" via add(B[0, j]) @ fabric"), ContainsSubstring("add takes a vector operand"));
    CHECK_THAT(err(" via add(b[0]) @ fabric"), ContainsSubstring("add(b[0,0]) @ fabric reads b[0,0], which is not resident"));
    CHECK_THAT(err(" via add @ fabric"), ContainsSubstring("add takes one vector tile"));
    CHECK_THAT(err(" via tanh @ fabric"), ContainsSubstring("'tanh' is not a vector operation"));
    CHECK_THAT(err(" via relu @ dma"), ContainsSubstring("'dma' is not a place"));
    CHECK_THAT(err(" via relu @ bm.ingress"), ContainsSubstring("bm.ingress is an operand's way into the fabric"));
    // The same refusals from the validator.
    CHECK_THAT(validate_error(linear_source(" via add(b[0]) @ fabric")),
               ContainsSubstring("add(b[0]) @ fabric reads b[0], which is not provably resident"));
    CHECK_THAT(validate_error(linear_source(" via relu @ bm.egress, add(b[j]) @ fabric")),
               ContainsSubstring("listed after a stage @ bm.egress"));

    const std::string head = R"(csp 1.0
program t machine flat(l3 = 16) {
  tensor A[64,64] tile 32x32 in;
  tensor B[64,64] tile 32x32 in;
  vector b[64] tile 32 in;
  tensor C[64,64] tile 32x32 inout;
)";
    auto prog = [&](const std::string& body) { return head + body + "}\n"; };
    // An accumulator takes gemm; its epilogue goes on its store.
    const std::string in_acc = prog("  resident A[0, 0], B[0, 0];\n  acc C[0, 0] in fabric {\n"
                                    "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n    call relu(C[0, 0]) -> C[0, 0];\n"
                                    "  }\n  store C[0, 0];\n  release A[0, 0], B[0, 0];\n");
    CHECK_THAT(compile_error(in_acc), ContainsSubstring("csp line 10: call relu on the open accumulator C[0,0]"));
    CHECK_THAT(validate_error(in_acc), ContainsSubstring("csp line 10: call relu on the open accumulator C[0, 0]"));
    // A resident tile's store is a DMA write: no vector unit there.
    const std::string resident_store =
        prog("  resident C[0, 0], b[0];\n  call add(C[0, 0], b[0]) -> C[0, 0];\n  store C[0, 0] via relu @ bm.egress;\n"
             "  release C[0, 0], b[0];\n");
    CHECK_THAT(compile_error(resident_store), ContainsSubstring("csp line 9: store C[0,0] via ...: a resident tile's store"));
    CHECK_THAT(validate_error(resident_store), ContainsSubstring("csp line 9: store C[0, 0] via ...: a resident tile's store"));
    // The in-place call carries the context instead, and runs.
    const std::string on_call =
        prog("  resident C[0, 0], b[0];\n  call add(C[0, 0], b[0]) -> C[0, 0] via relu @ bm.egress;\n"
             "  store C[0, 0];\n  release C[0, 0], b[0];\n");
    CHECK(compile_error(on_call) == "compiled");
    CHECK(validate_error(on_call) == "validated");
    const csp::CspProgram oc = lang::compile(on_call);
    CHECK(lang::print(lang::compile(lang::print(oc))) == lang::print(oc));
    // Indexing.
    CHECK_THAT(compile_error(prog("  resident b[0, 0];\n  release b[0, 0];\n")),
               ContainsSubstring("b is a vector: index it [j]"));
    CHECK_THAT(validate_error(prog("  resident b[0, 0];\n  release b[0, 0];\n")),
               ContainsSubstring("b is a vector: index it [j]"));
    CHECK_THAT(compile_error(prog("  resident C[0, 0], A[0, 0];\n  call add(C[0, 0], A[0, 0]) -> C[0, 0];\n"
                                  "  store C[0, 0];\n  release C[0, 0], A[0, 0];\n")),
               ContainsSubstring("add(y, b): b must be a vector operand"));
}
