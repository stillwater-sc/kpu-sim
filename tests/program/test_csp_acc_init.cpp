// ============================================================================
// tests/program/test_csp_acc_init.cpp
// An accumulator's init (#406; docs/plans/accumulator-init-stream.md): `acc Y in fabric init X`
// starts Y from a bias matrix D (C = A.B + D), a bias row b broadcast down the rows (C = A.B + b),
// or Y itself (C += A.B). The init is one more edge stream -- a Move and a Feed into the fabric,
// before the accumulator's first call -- and no init means a clear (zero, nothing streamed).
//
// Values are checked against a reference computed here, from the inputs, in the RTL's order
// (init + sum over k), not against another level.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/characterize/device_model.hpp>
#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <cmath>
#include <cstring>
#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;
using Kind = Action::Kind;

namespace {

gen::Options matmul(const std::string& init, const std::string& orient = "col") {
    gen::Options o;
    o.size = 8;
    o.tile = 4;
    o.l3 = 8;
    o.orient = orient;
    o.init = init;
    return o;
}

struct LbRun {
    TileProgram in;          // the inputs, as synthesized
    TileProgram out;         // the operands L-B leaves
    TileProgram trace;       // the L0 trace L-B recorded
};

LbRun lb(const std::string& src) {
    const lang::Program ast = lang::parse(src);
    LbRun r;
    r.in = driver::csp_inputs(ast);
    BehavioralInterpreter b;
    b.begin(r.in, /*record=*/true, lang::l3_geometry(ast), static_cast<std::uint32_t>(ast.l2));
    lang::ActionStream s(ast);
    while (auto e = s.next()) b.step(e->action, e->op ? &*e->op : nullptr, &e->l0);
    (void)b.finish();
    r.out = b.result();
    r.trace = b.trace();
    return r;
}

// C[r][c] = init[r][c] + sum_k A[r][k] B[k][c], from the inputs alone.
std::vector<float> reference(const TileProgram& in, const std::string& init) {
    const TensorOperand& A = in.operand("A");
    const TensorOperand& B = in.operand("B");
    const Dim n = A.rows;
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.0f);
    for (Dim r = 0; r < n; ++r)
        for (Dim j = 0; j < n; ++j) {
            float v = 0.0f;
            if (init == "bias") v = in.operand("D").at(r, j);
            if (init == "bias-row") v = in.operand("b").at(j, 0);
            if (init == "accumulate") v = in.operand("C").at(r, j);
            for (Dim k = 0; k < n; ++k) v += A.at(r, k) * B.at(k, j);
            c[static_cast<std::size_t>(r) * n + j] = v;
        }
    return c;
}

void check_close(const std::vector<float>& got, const std::vector<float>& want) {
    REQUIRE(got.size() == want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        CAPTURE(i, got[i], want[i]);
        CHECK(std::fabs(got[i] - want[i]) <= 1e-5f + 1e-5f * std::fabs(want[i]));
    }
}

std::string error_of(const std::string& src, bool symbolic) {
    try {
        if (symbolic) (void)lang::validate(src);
        else (void)lang::compile(src);
    } catch (const lang::CompileError& e) {
        return e.what();
    }
    return "accepted";
}

std::size_t first(const CspProgram& p, Kind kind, const std::string& t) {
    for (std::size_t i = 0; i < p.actions.size(); ++i)
        if (p.actions[i].kind == kind && p.actions[i].tile.to_string() == t) return i;
    FAIL("no " << to_string(kind) << " of " << t);
    return 0;
}

std::size_t count(const CspProgram& p, Kind kind, const std::string& operand) {
    std::size_t n = 0;
    for (const Action& a : p.actions) n += a.kind == kind && a.tile.operand == operand;
    return n;
}

const char* const kHead = "csp 1.0\nprogram t machine flat(l3 = 6) {\n"
                          "  tensor A[16,16] tile 16x16 in;\n  tensor B[16,16] tile 16x16 in;\n"
                          "  tensor D[16,16] tile 16x16 in;\n  tensor E[16,32] tile 16x16 in;\n"
                          "  vector b[16] tile 16 in;\n  tensor C[16,16] tile 16x16 out;\n"
                          "  tensor S[16,1] tile 16x1 out;\n  tensor X[16,16] tile 16x16 inout;\n";
// Operands of more than one tile, for the position refusals.
const char* const kWide = "  tensor F[32,32] tile 16x16 in;\n  tensor G[32,32] tile 16x16 out;\n"
                          "  vector v[32] tile 16 in;\n";

}  // namespace

TEST_CASE("acc init: parsed and formatted", "[program][csp][lang][init]") {
    const std::string src = gen::generate(matmul("bias"));
    const lang::Program ast = lang::parse(src);
    const std::string text = lang::format(ast);
    CHECK_THAT(text, ContainsSubstring("acc C[i, j] in fabric init D[i, j] {"));
    CHECK(lang::format(lang::parse(text)) == text);
    CHECK_THAT(gen::generate(matmul("bias-row")), ContainsSubstring("acc C[i, j] in fabric init b[j] {"));
    CHECK_THAT(gen::generate(matmul("accumulate")), ContainsSubstring("acc C[i, j] in fabric init C[i, j] {"));
    CHECK_THAT(gen::generate(matmul("none")), !ContainsSubstring(" init "));
}

TEST_CASE("acc init: L-B computes init + A.B, for every form and orientation", "[program][csp][init][values]") {
    for (const char* init : {"bias", "bias-row", "accumulate"})
        for (const char* orient : {"col", "row"}) {
            CAPTURE(init, orient);
            const LbRun r = lb(gen::generate(matmul(init, orient)));
            check_close(r.out.operand("C").values, reference(r.in, init));
        }
    // No init is a clear: the plain matmul, unchanged.
    const LbRun r = lb(gen::generate(matmul("none")));
    check_close(r.out.operand("C").values, reference(r.in, "none"));
}

TEST_CASE("acc init: the L0 trace records ACC_INIT, replays to L-B's values, and round-trips",
          "[program][csp][init][l0]") {
    for (const char* init : {"bias", "bias-row", "accumulate"}) {
        CAPTURE(init);
        const LbRun r = lb(gen::generate(matmul(init)));
        std::size_t inits = 0;
        for (const TileOp& op : r.trace.ops()) inits += op.kind == TileOpKind::AccInit;
        CHECK(inits == 4);                               // one per accumulator
        CHECK(serialize::opset_for(r.trace).str() == "1.4.0");
        TileProgram replay = r.trace;
        const auto sum = TileProgramReference().run(replay);
        CHECK(sum.acc_inits == 4);
        CHECK(replay.operand("C").values == r.out.operand("C").values);   // bit for bit: the same computation
        std::ostringstream os;
        serialize::write_l0(os, r.trace, {true, ""});
        std::istringstream is(os.str());
        TileProgram back = serialize::read_l0(is);
        (void)TileProgramReference().run(back);
        CHECK(back.operand("C").values == r.out.operand("C").values);
    }
    // A program without an init keeps its opset: the corpus does not change.
    CHECK(serialize::opset_for(lb(gen::generate(matmul("none"))).trace).str() != "1.4.0");
}

TEST_CASE("acc init: one more edge stream, before the accumulator's first call", "[program][csp][init][actions]") {
    const CspProgram p = lang::compile(gen::generate(matmul("bias")));
    const std::size_t feed = first(p, Kind::Feed, "D[1,0]");
    REQUIRE(p.actions[feed].seeds);
    CHECK(p.actions[feed].seeds->to_string() == "C[1,0]");
    CHECK(p.actions[feed - 1].kind == Kind::Move);       // L3 -> L2, then L2 -> L1
    CHECK(feed < first(p, Kind::Call, "C[1,0]"));
    // Only an init's Feed seeds.
    for (const Action& a : p.actions) CHECK((a.seeds.has_value() == (a.kind == Kind::Feed && a.tile.operand == "D")));
    // A bias row is read once per output tile: one Move per accumulator, one Load per panel.
    const CspProgram row = lang::compile(gen::generate(matmul("bias-row")));
    CHECK(count(row, Kind::Move, "b") == 4);
    CHECK(count(row, Kind::Load, "b") == 2);
    // Accumulating onto C: the fabric takes C's residency over, so its Release follows the Feed.
    const CspProgram acc = lang::compile(gen::generate(matmul("accumulate")));
    const std::size_t f = first(acc, Kind::Feed, "C[0,1]");
    REQUIRE(acc.actions[f].seeds);
    CHECK(acc.actions[f + 1].kind == Kind::Release);
    CHECK(acc.actions[f + 1].tile.to_string() == "C[0,1]");
    CHECK(f < first(acc, Kind::Call, "C[0,1]"));
}

TEST_CASE("acc init: a staged init is fed from its L2 copy, which stays for the next", "[program][csp][init][l2]") {
    // One bias row b[0], staged once, starts two accumulators.
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 6, l2 = 1) {\n"
                            "  tensor A[32,16] tile 16x16 in;\n  tensor B[16,16] tile 16x16 in;\n"
                            "  vector b[16] tile 16 in;\n  tensor C[32,16] tile 16x16 out;\n"
                            "  resident A[:, 0], B[0, 0], b[0];\n"
                            "  stage b[0] in l2 {\n"
                            "    for i in 0..2 {\n"
                            "      acc C[i, 0] in fabric init b[0] { call gemm(A[i, 0], B[0, 0]) +-> C[i, 0]; }\n"
                            "      store C[i, 0];\n"
                            "    }\n"
                            "  }\n"
                            "  release A[:, 0], B[0, 0], b[0];\n}\n";
    REQUIRE(error_of(src, true) == "accepted");
    const CspProgram p = lang::compile(src);
    CHECK(count(p, Kind::Move, "b") == 1);               // into L2 once
    std::size_t seeded = 0;
    for (const Action& a : p.actions)
        if (a.kind == Kind::Feed && a.seeds) {
            CHECK(a.l2.has_value());                     // from the staged copy
            ++seeded;
        }
    CHECK(seeded == 2);
    const LbRun r = lb(src);
    const TensorOperand& A = r.in.operand("A");
    const TensorOperand& B = r.in.operand("B");
    for (Dim row = 0; row < 32; ++row)
        for (Dim c = 0; c < 16; ++c) {
            float v = r.in.operand("b").at(c, 0);
            for (Dim k = 0; k < 16; ++k) v += A.at(row, k) * B.at(k, c);
            CAPTURE(row, c);
            CHECK(std::fabs(r.out.operand("C").at(row, c) - v) <= 1e-5f + 1e-5f * std::fabs(v));
        }
}

TEST_CASE("acc init: the printer writes the init back", "[program][csp][init][print]") {
    for (const char* init : {"bias", "bias-row", "accumulate"}) {
        CAPTURE(init);
        const CspProgram p = lang::compile(gen::generate(matmul(init)));
        const std::string printed = lang::print(p);
        CHECK_THAT(printed, ContainsSubstring(" init "));
        const CspProgram again = lang::compile(printed);
        REQUIRE(again.actions.size() == p.actions.size());
        for (std::size_t i = 0; i < p.actions.size(); ++i) {
            CAPTURE(i);
            CHECK(again.actions[i].kind == p.actions[i].kind);
            CHECK(again.actions[i].tile.to_string() == p.actions[i].tile.to_string());
            CHECK(again.actions[i].seeds.has_value() == p.actions[i].seeds.has_value());
        }
    }
}

TEST_CASE("acc init: what it refuses, symbolically and as it runs", "[program][csp][init][refuse]") {
    const std::string acc = "  acc C[0, 0] in fabric init ";
    const std::string body = " { call gemm(A[0, 0], B[0, 0]) +-> C[0, 0]; }\n  store C[0, 0];\n";
    struct Case { std::string src, symbolic, walked; };
    const std::vector<Case> cases = {
        // The init is resident when the block opens.
        {std::string(kHead) + "  resident A[0, 0], B[0, 0];\n" + acc + "D[0, 0]" + body +
             "  release A[0, 0], B[0, 0];\n}\n",
         "reads D[0, 0], which is not provably resident", "reads D[0,0], which is not resident"},
        // A tensor init is tiled as the accumulator.
        {std::string(kHead) + "  resident A[0, 0], B[0, 0], E[0, 0];\n" + acc + "E[0, 0]" + body +
             "  release A[0, 0], B[0, 0], E[0, 0];\n}\n",
         "a tensor init is tiled as the accumulator", "a tensor init is tiled as the accumulator"},
        // A vector init is a bias row: not for a one-column accumulator.
        {std::string(kHead) + "  resident A[0, 0], B[0, 0], b[0];\n  acc S[0, 0] in fabric init b[0]"
                              " { call gemm(A[0, 0], B[0, 0]) +-> S[0, 0]; }\n  store S[0, 0];\n"
                              "  release A[0, 0], B[0, 0], b[0];\n}\n",
         "a vector init is a bias row", "a vector init is a bias row"},
        // Accumulating onto an `out` operand: it has no value.
        {std::string(kHead) + "  resident A[0, 0], B[0, 0], C[0, 0];\n" + acc + "C[0, 0]" + body +
             "  release A[0, 0], B[0, 0];\n}\n",
         "C is declared out", "C is declared out"},
        // A tensor init starts the tile in the same place (review of #407: the validator agrees).
        {std::string(kHead) + kWide + "  resident A[0, 0], B[0, 0], F[1, 1];\n"
                              "  acc G[0, 0] in fabric init F[1, 1] { call gemm(A[0, 0], B[0, 0]) +-> G[0, 0]; }\n"
                              "  store G[0, 0];\n  release A[0, 0], B[0, 0], F[1, 1];\n}\n",
         "a tensor init starts the tile in the same place", "a tensor init starts the tile in the same place"},
        // A bias row is indexed by the accumulator's column.
        {std::string(kHead) + kWide + "  resident A[0, 0], B[0, 0], v[0];\n"
                              "  acc G[0, 1] in fabric init v[0] { call gemm(A[0, 0], B[0, 0]) +-> G[0, 1]; }\n"
                              "  store G[0, 1];\n  release A[0, 0], B[0, 0], v[0];\n}\n",
         "a bias row is indexed by the accumulator's column, v[1]",
         "a bias row is indexed by the accumulator's column, v[1]"},
    };
    for (const Case& c : cases) {
        CAPTURE(c.src);
        CHECK_THAT(error_of(c.src, true), ContainsSubstring(c.symbolic));
        CHECK_THAT(error_of(c.src, false), ContainsSubstring(c.walked));
    }
    // An accumulator on a resident tile says how to accumulate onto it.
    const std::string resident = std::string(kHead) + "  resident A[0, 0], B[0, 0], X[0, 0];\n"
                                 "  acc X[0, 0] in fabric { call gemm(A[0, 0], B[0, 0]) +-> X[0, 0]; }\n"
                                 "  store X[0, 0];\n  release A[0, 0], B[0, 0];\n}\n";
    CHECK_THAT(error_of(resident, false), ContainsSubstring("write `init X[0,0]`"));
    // And with the init, it runs: the residency is the fabric's, so it is never released.
    std::string onto = resident;
    onto.replace(onto.find("in fabric {"), 11, "in fabric init X[0, 0] {");
    CHECK(error_of(onto, true) == "accepted");
    CHECK(error_of(onto, false) == "accepted");
}

TEST_CASE("acc init: L-T1 times the init as an operand stream; values are L-B's", "[program][csp][init][lt1]") {
    const characterize::DeviceDescriptor dev = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device_view(0);
    for (const char* init : {"bias", "bias-row", "accumulate"}) {
        CAPTURE(init);
        const std::string src = gen::generate(matmul(init));
        const lang::Program ast = lang::parse(src);
        TransactionalInterpreter lt(dev, static_cast<std::size_t>(ast.l3), static_cast<std::uint32_t>(ast.banks));
        lt.begin(driver::csp_inputs(ast));
        lang::ActionStream s(ast);
        while (auto e = s.next()) lt.step(e->action, e->op ? &*e->op : nullptr);
        (void)lt.finish();
        // Each accumulator's first call starts after its init's Feed finishes.
        std::map<std::string, TransactionalInterpreter::Cycle> init_fed;
        std::map<std::string, bool> called;
        const CspProgram p = lang::compile(src);
        for (const auto& rec : lt.records()) {
            const Action& a = p.actions.at(rec.action);
            if (a.kind == Kind::Feed && a.seeds) init_fed[a.seeds->to_string()] = rec.finish;
            if (a.kind == Kind::Call && !called[a.tile.to_string()]) {
                called[a.tile.to_string()] = true;
                REQUIRE(init_fed.count(a.tile.to_string()));
                CHECK(rec.start >= init_fed[a.tile.to_string()]);
            }
        }
        CHECK(init_fed.size() == 4);
        check_close(lt.result().operand("C").values, reference(lb(src).in, init));
    }
}

TEST_CASE("acc init: L-CA feeds the init stream; values are the reference's", "[program][csp][init][lca]") {
    namespace timing = sw::kpu::timing;
    const auto dev = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    for (const char* orient : {"col", "row"})
        for (const char* init : {"none", "bias", "bias-row", "accumulate"}) {
            CAPTURE(orient, init);
            const std::string src = gen::generate(matmul(init, orient));
            const lang::Program ast = lang::parse(src);
            REQUIRE_FALSE(driver::csp_level_supported(driver::ExecutionLevel::CycleAccurate, ast, &dev));
            const TileProgram in = driver::csp_inputs(ast);
            const auto cfg = timing::csp_config_from(dev);
            REQUIRE(cfg);
            timing::ConcurrentTimingExecutor exec(cfg->config);
            lang::ActionStream s(ast);
            const auto r = timing::CspDriver(exec, s, in).run();
            REQUIRE(r.completed);
            check_close(r.values.operand("C").values, reference(in, init));
            if (std::string(init) == "none") continue;
            // The init's tile, as the driver names it: the one operand it tags (tk 1) is D or b,
            // and C is C.
            auto init_of = [&](const timing::TileID& c) {
                timing::TileID x = c;
                if (std::string(init) == "bias") { x.matrix = sw::kpu::isa::MatrixID::A; x.tk = 1; }
                if (std::string(init) == "bias-row") { x = {sw::kpu::isa::MatrixID::A, c.tj, 0, 1}; }
                return x;
            };
            // Each init tile's feeds, and the first k-steps of the accumulators it starts, in
            // time: the n-th first k-step starts no earlier than the n-th feed has finished.
            std::map<std::string, std::vector<timing::Cycle>> feeds, firsts;
            std::set<std::string> started;
            for (const auto& e : exec.events()) {
                if (e.type == timing::EventType::STR_FEED_COMPLETE) feeds[e.tile_id.to_string()].push_back(e.cycle + e.duration);
                if (e.type == timing::EventType::COMPUTE_START && e.tile_id.matrix == sw::kpu::isa::MatrixID::C &&
                    started.insert(e.tile_id.to_string()).second)
                    firsts[init_of(e.tile_id).to_string()].push_back(e.cycle);
            }
            REQUIRE(started.size() == 4);
            for (const auto& [x, starts] : firsts) {
                CAPTURE(x);
                REQUIRE(feeds[x].size() >= starts.size());
                for (std::size_t n = 0; n < starts.size(); ++n) CHECK(starts[n] >= feeds[x][n]);
            }
        }
    // An executor reused for another run keeps its DRAM: the next run's C is its own inputs'
    // (review of #409), not the C the run before stored.
    {
        const lang::Program acc = lang::parse(gen::generate(matmul("accumulate")));
        TileProgram in = driver::csp_inputs(acc);
        const auto cfg = timing::csp_config_from(dev);
        timing::ConcurrentTimingExecutor exec(cfg->config);
        for (int run = 0; run < 2; ++run) {
            CAPTURE(run);
            for (float& v : in.operand("C").values) v += 1.0f;
            lang::ActionStream s(acc);
            const auto r = timing::CspDriver(exec, s, in).run();
            REQUIRE(r.completed);
            check_close(r.values.operand("C").values, reference(in, "accumulate"));
        }
    }
    // The run's own entry point admits it, and a matrix L-CA does not map is still refused
    // when it is computed on rather than streamed in as an init.
    const lang::Program bias = lang::parse(gen::generate(matmul("bias")));
    const auto o = driver::csp_run_level(driver::ExecutionLevel::CycleAccurate, bias, &dev, driver::csp_inputs(bias));
    REQUIRE_FALSE(o.skipped);
    check_close(o.values.operand("C").values, reference(driver::csp_inputs(bias), "bias"));
    std::string computed = gen::generate(matmul("bias"));
    const std::size_t at = computed.find("call gemm(A[");
    REQUIRE(at != std::string::npos);
    computed.replace(at, 12, "call gemm(D[");
    const auto why = driver::csp_level_supported(driver::ExecutionLevel::CycleAccurate, lang::parse(computed), &dev);
    REQUIRE(why);
    CHECK_THAT(*why, ContainsSubstring("'D' is not one of them"));
}

TEST_CASE("acc init: kpu-csp-gen's options", "[program][csp][gen][init]") {
    gen::Options o = matmul("sideways");
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("init 'sideways'"));
    o = matmul("bias");
    o.algo = "linear";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("init is matmul's"));
    // The init's tile needs a slot beside the panels.
    o = matmul("bias");
    o.l3 = 5;
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("and the init's tile"));
    o = matmul("accumulate");
    o.l3 = 5;
    CHECK_NOTHROW(gen::generate(o));                     // C's slot is handed to the fabric
}

namespace {

// An L2 accumulator's init (#427): the weight- or input-stationary matmul, its panel's partials
// merged by the Combiner, which adds the init on each tile's first merge.
std::string stationary(const std::string& df, const std::string& init, const std::string& dtype) {
    gen::Options o;
    o.size = 16;
    o.tile = 4;
    o.l3 = 16;
    o.dataflow = df;
    o.init = init;
    o.dtype = dtype;
    return gen::generate(o);
}

platform::DeviceSpecification s1_combiner() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu"}};
    return d;
}

}  // namespace

TEST_CASE("acc init: an L2 accumulator's init, added by the Combiner on the first merge, at every level",
          "[program][csp][init][l2acc]") {
    using driver::ExecutionLevel;
    const auto d = s1_combiner();
    CHECK_THAT(stationary("ws", "bias", "fp32"), ContainsSubstring("acc C[:, j] in l2 init D[:, j] {"));
    CHECK_THAT(stationary("is", "bias-row", "fp32"), ContainsSubstring("acc C[i, :] in l2 init b[:] {"));
    for (const char* df : {"ws", "is"})
        for (const char* init : {"bias", "bias-row"})
            for (const char* dtype : {"fp32", "int8"}) {
                CAPTURE(df, init, dtype);
                const lang::Program ast = lang::parse(stationary(df, init, dtype));
                const TileProgram in = driver::csp_inputs(ast);
                driver::CspRunRequest req{ast, &d, in,
                                          {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                           ExecutionLevel::CycleAccurate}};
                const auto r = driver::run_csp(req);
                REQUIRE(r.reference);
                CHECK(r.reference_note.empty());                  // the trace (MERGE_INIT) replays L-B
                // The expectation, computed here: the plain matmul, plus the init.
                TileProgram expect = *r.reference;
                TensorOperand& C = expect.operand("C");
                if (std::string(dtype) == "fp32") {
                    const auto ref = reference(in, init);
                    for (Dim rr = 0; rr < C.rows; ++rr)
                        for (Dim cc = 0; cc < C.cols; ++cc) C.at(rr, cc) = ref[static_cast<std::size_t>(rr) * C.rows + cc];
                } else {
                    const lang::Program plain_ast = lang::parse(stationary(df, "none", dtype));
                    TileProgram plain_in = driver::csp_inputs(plain_ast);
                    for (const char* n : {"A", "B"}) plain_in.operand(n).values = in.operand(n).values;
                    const auto plain = driver::csp_reference(plain_ast, plain_in, 1'000'000);
                    REQUIRE(plain.values);
                    const TensorOperand& p = plain.values->operand("C");
                    for (Dim rr = 0; rr < C.rows; ++rr)
                        for (Dim cc = 0; cc < C.cols; ++cc) {
                            const numeric::Word x = std::string(init) == "bias" ? in.operand("D").word(rr, cc)
                                                                                : in.operand("b").word(cc, 0);
                            C.set_word(rr, cc, numeric::encode("int32", numeric::decode("int32", p.word(rr, cc)) +
                                                                            numeric::decode("int32", x)));
                        }
                }
                CHECK(check_operands(r.reference.value(), expect).pass);    // L-B against the reference
                for (const auto& o : r.levels) {
                    CAPTURE(driver::to_string(o.level));
                    INFO(o.skipped.value_or(""));
                    REQUIRE_FALSE(o.skipped);
                    CHECK(check_operands(o.values, *r.reference).pass);
                }
            }
}

TEST_CASE("acc init: an L2 accumulator's init in the trace is MERGE_INIT (opset 1.10)", "[program][csp][init][l2acc]") {
    const lang::Program ast = lang::parse(stationary("ws", "bias", "fp32"));
    const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
    REQUIRE(ref.trace);
    std::size_t inits = 0;
    for (const TileOp& op : ref.trace->ops()) inits += op.kind == TileOpKind::MergeInit;
    CHECK(inits == 16);                                       // one per C tile, on its first merge
    std::ostringstream os;
    serialize::WriteOptions with_values;
    with_values.include_values = true;
    serialize::write_l0(os, *ref.trace, with_values);
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.10.0"));
    CHECK_THAT(os.str(), ContainsSubstring("kind=MERGE_INIT in=D:0,0 out=C:0,0"));
    std::istringstream is(os.str());
    TileProgram back = serialize::read_l0(is);
    (void)TileProgramReference().run(back);
    const auto& a = back.operand("C").values;
    const auto& w = ref.values->operand("C").values;
    CHECK((a.size() == w.size() && std::memcmp(a.data(), w.data(), a.size() * sizeof(float)) == 0));
}

TEST_CASE("acc init: what an L2 accumulator's init refuses", "[program][csp][init][l2acc][refuse]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                             "  tensor A[8,8] tile 4x4 in;\n  tensor B[8,8] tile 4x4 in;\n"
                             "  tensor C[8,8] tile 4x4 out;\n  tensor D[8,8] tile 4x4 in;\n"
                             "  tensor E[8,8] int32 tile 4x4 in;\n  vector b[8] tile 4 in;\n";
    auto program = [&](const std::string& before, const std::string& init, const std::string& after) {
        return head + before + "  acc C[0, 0] in l2 init " + init + " {\n    resident A[0, 0], B[0, 0];\n"
               "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n    release A[0, 0], B[0, 0];\n  }\n" + after +
               "  store C[0, 0];\n}\n";
    };
    auto both = [&](const std::string& src, const std::string& text) {
        std::string walk = "accepted", sym = "accepted";
        try { (void)lang::compile(src); } catch (const std::exception& e) { walk = e.what(); }
        try { (void)lang::validate(src); } catch (const std::exception& e) { sym = e.what(); }
        INFO("walker: " << walk);
        CHECK_THAT(walk, ContainsSubstring(text));
        INFO("validator: " << sym);
        CHECK_THAT(sym, ContainsSubstring(text));
    };
    const std::string ok = program("  resident D[0, 0];\n", "D[0, 0]", "  release D[0, 0];\n");
    CHECK_NOTHROW(lang::compile(ok));
    CHECK_NOTHROW(lang::validate(ok));
    CHECK_NOTHROW(lang::compile(program("  resident b[0];\n", "b[0]", "  release b[0];\n")));
    both(program("", "D[0, 0]", ""), "resident; make it resident first");
    both(program("  resident E[0, 0];\n", "E[0, 0]", "  release E[0, 0];\n"), "an init is in the accumulator's number system");
    // Onto the accumulator itself: not built yet, said so.
    CHECK_THAT(lang::format(lang::parse(ok)), ContainsSubstring("acc C[0, 0] in l2 init D[0, 0] {"));
    gen::Options o;
    o.size = 16;
    o.tile = 4;
    o.l3 = 16;
    o.dataflow = "ws";
    o.init = "accumulate";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("init accumulate (C += A . B) onto an L2 accumulator"));
}

TEST_CASE("acc init: an L2 init is checked where the walker checks it (#446's review)", "[program][csp][init][l2acc][refuse]") {
    auto verdicts = [](const std::string& src) {
        std::pair<std::string, std::string> v{"accepted", "accepted"};
        try { (void)lang::compile(src); } catch (const std::exception& e) { v.first = e.what(); }
        try { (void)lang::validate(src); } catch (const std::exception& e) { v.second = e.what(); }
        return v;
    };
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 12, l2 = 4) {\n"
                             "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                             "  tensor C[32,32] tile 16x16 out;\n  tensor D[32,32] tile 16x16 in;\n";
    // 1. Released before the first merge: resident at the acc statement is not enough.
    {
        const auto v = verdicts(head + "  resident D[0, 0];\n  acc C[0, 0] in l2 init D[0, 0] {\n    release D[0, 0];\n"
                                       "    resident A[0, 0], B[0, 0];\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
                                       "    release A[0, 0], B[0, 0];\n  }\n  store C[0, 0];\n}\n");
        INFO("walker: " << v.first << "\nvalidator: " << v.second);
        CHECK_THAT(v.first, ContainsSubstring("its first merge adds the init"));
        CHECK_THAT(v.second, ContainsSubstring("its first merge adds the init"));
    }
    // 2. In the wrong place: D[1, 1] for C[0, 0], though D is shaped as C and D[1, 1] is resident.
    {
        const auto v = verdicts(head + "  resident D[1, 1];\n  acc C[0, 0] in l2 init D[1, 1] {\n"
                                       "    resident A[0, 0], B[0, 0];\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
                                       "    release A[0, 0], B[0, 0];\n  }\n  release D[1, 1];\n  store C[0, 0];\n}\n");
        INFO("walker: " << v.first << "\nvalidator: " << v.second);
        CHECK_THAT(v.first, ContainsSubstring("the init names no tile in its place"));
        CHECK_THAT(v.second, ContainsSubstring("a tensor init starts each tile in the same place"));
    }
    // 3. A slice is resident only where all its tiles are: D[0, 0] does not hold D[0, :].
    {
        const auto v = verdicts(head + "  resident D[0, 0];\n  acc C[0, :] in l2 init D[0, :] {\n"
                                       "    resident A[0, 0], B[0, :];\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
                                       "    call gemm(A[0, 0], B[0, 1]) +-> C[0, 1];\n    release A[0, 0], B[0, :];\n  }\n"
                                       "  release D[0, 0];\n  store C[0, :];\n}\n");
        INFO("walker: " << v.first << "\nvalidator: " << v.second);
        CHECK_THAT(v.first, ContainsSubstring("which is not resident"));
        CHECK_THAT(v.second, ContainsSubstring("which is not provably resident"));
    }
}

TEST_CASE("acc init: L-T1's first merge waits for its init's Load (#446's review)", "[program][csp][init][l2acc][lt1]") {
    // One DMA engine, slow, and D loaded last: the gemm and its Drain would otherwise finish
    // before D is in L3.
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                            "  tensor A[16,16] tile 16x16 in;\n  tensor B[16,16] tile 16x16 in;\n"
                            "  tensor C[16,16] tile 16x16 out;\n  tensor D[16,16] tile 16x16 in;\n"
                            "  resident A[0, 0], B[0, 0], D[0, 0];\n  acc C[0, 0] in l2 init D[0, 0] {\n"
                            "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n"
                            "  release A[0, 0], B[0, 0], D[0, 0];\n  store C[0, 0];\n}\n";
    auto dev = characterize::DeviceDescriptor::single();
    dev.dma_engines = 1;
    dev.dma_bytes_per_cycle = 1.0;
    dev.bm_bytes_per_cycle = 4096.0;
    dev.str_bytes_per_cycle = 4096.0;
    CspProgram p = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    TransactionalInterpreter lt(dev, p.channel(Chan::L3).capacity, 0);
    lt.begin(p.source);
    for (const Action& a : p.actions) lt.step(a, a.kind == Action::Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
    (void)lt.finish();
    std::uint64_t d_loaded = 0, drain_start = 0;
    bool drained = false;
    for (const auto& r : lt.records()) {
        if (r.kind == Action::Kind::Load && r.tile.operand == "D") d_loaded = r.finish;
        if (r.kind == Action::Kind::Drain && r.tile.operand == "C") {
            drain_start = r.start;
            drained = true;
        }
    }
    REQUIRE(drained);
    REQUIRE(d_loaded > 0);
    CHECK(drain_start >= d_loaded);
}

TEST_CASE("acc init: each tile's first merge finds its init resident, and a vector takes no tensor init",
          "[program][csp][init][l2acc][refuse]") {
    auto verdicts = [](const std::string& src) {
        std::pair<std::string, std::string> v{"accepted", "accepted"};
        try { (void)lang::compile(src); } catch (const std::exception& e) { v.first = e.what(); }
        try { (void)lang::validate(src); } catch (const std::exception& e) { v.second = e.what(); }
        return v;
    };
    // The second review of #446: C[0, 1]'s first merge comes after the init is released, in a
    // second call statement. Marking the whole family merged at the first call let it through.
    {
        const auto v = verdicts("csp 1.0\nprogram t machine flat(l3 = 12, l2 = 4) {\n"
                                "  tensor A[16,16] tile 16x16 in;\n  tensor B[16,32] tile 16x16 in;\n"
                                "  tensor C[16,32] tile 16x16 out;\n  tensor D[16,32] tile 16x16 in;\n"
                                "  resident D[0, :];\n  acc C[0, :] in l2 init D[0, :] {\n"
                                "    resident A[0, 0], B[0, :];\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
                                "    release D[0, :];\n    call gemm(A[0, 0], B[0, 1]) +-> C[0, 1];\n"
                                "    release A[0, 0], B[0, :];\n  }\n  store C[0, :];\n}\n");
        INFO("walker: " << v.first << "\nvalidator: " << v.second);
        CHECK_THAT(v.first, ContainsSubstring("its first merge adds the init"));
        CHECK_THAT(v.second, ContainsSubstring("call gemm +-> C[0, 1]: its first merge adds the init"));
    }
    // A tensor init on a vector accumulator: refused, not compared past the vector's one dimension.
    {
        const std::string src = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                                "  vector Y[16] tile 16 out;\n  tensor X[16,1] tile 16x1 in;\n"
                                "  resident X[0, 0];\n  acc Y[0] in l2 init X[0, 0] { }\n  release X[0, 0];\n}\n";
        std::string sym = "accepted";
        try { (void)lang::validate(src); } catch (const std::exception& e) { sym = e.what(); }
        CHECK_THAT(sym, ContainsSubstring("a tensor init needs a tensor accumulator, and Y is a vector"));
    }
}

TEST_CASE("acc init: an in-place writeback of the init's tile waits for the merge that reads it",
          "[program][csp][init][l2acc][lt1]") {
    // The second review of #446: D is the init of C's first merge, then updated in place. The
    // Writeback of D must not finish before C's first merge has read D. A slow gemm makes that late.
    // The streamer's in-order issue orders them already (D's Feed follows C's Drain); the in-place
    // Writeback's bound on the stages' reads, as Release and Retain have it, keeps this so if
    // that order ever loosens. The test pins the invariant.
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                            "  tensor A[16,16] tile 16x16 in;\n  tensor B[16,16] tile 16x16 in;\n"
                            "  tensor C[16,16] tile 16x16 out;\n  tensor D[16,16] tile 16x16 inout;\n"
                            "  resident A[0, 0], B[0, 0], D[0, 0];\n  acc C[0, 0] in l2 init D[0, 0] {\n"
                            "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n  }\n"
                            "  call relu(D[0, 0]) -> D[0, 0];\n  store D[0, 0];\n"
                            "  release A[0, 0], B[0, 0], D[0, 0];\n  store C[0, 0];\n}\n";
    auto dev = characterize::DeviceDescriptor::single();
    dev.fabric_macs_per_cycle = 1.0;               // the gemm, and so C's first merge, is late
    dev.compute_tiles = 2;                         // and the relu on D runs beside it, on the other tile
    CspProgram p = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    TransactionalInterpreter lt(dev, p.channel(Chan::L3).capacity, 0);
    lt.begin(p.source);
    for (const Action& a : p.actions) lt.step(a, a.kind == Action::Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
    (void)lt.finish();
    std::uint64_t merge_read = 0, wb_finish = 0;
    bool merged = false, wrote = false;
    for (const auto& r : lt.records()) {
        if (r.kind == Action::Kind::Drain && r.tile.operand == "C") {
            merge_read = r.finish;                 // the init is read as the merge's Drain runs
            merged = true;
        }
        if (r.kind == Action::Kind::Writeback && r.tile.operand == "D") {
            wb_finish = r.finish;
            wrote = true;
        }
    }
    REQUIRE(merged);
    REQUIRE(wrote);
    CHECK(wb_finish >= merge_read);
}
