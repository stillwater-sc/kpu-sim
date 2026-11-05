// ============================================================================
// tests/program/test_csp_l2_staging.cpp
// L2 staging (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 3): `stage T in l2 { }`
// copies resident tiles into L2 slots the walker allocates, the calls inside the block feed the
// copy, and the block's end frees the slots. L-B executes the addresses and refuses what breaks
// them; the trace records them (opset 1.3) and its replay checks them; every level computes the
// same values, staged or not.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/behavioral.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/format.hpp>
#include <sw/kpu/program/csp/lang/print.hpp>
#include <sw/kpu/program/csp/lang/validate.hpp>
#include <sw/kpu/program/csp/lang/walk.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/serialize/l0_format.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace gen = sw::kpu::program::csp::gen;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::driver::ExecutionLevel;

namespace {

// Row panels of A outer; staged, A's panel is copied into L2 once for the loop over B's panels.
std::string matmul(bool staged) {
    std::string s = std::string("csp 1.0\nprogram mm machine flat(l3 = 9") + (staged ? ", l2 = 4" : "") + ") {\n"
                    "  tensor A[64,64] tile 16x16 in;\n  tensor B[64,64] tile 16x16 in;\n  tensor C[64,64] tile 16x16 out;\n"
                    "  for i in 0..4 {\n"
                    "    resident A[i, :];\n";
    if (staged) s += "    stage A[i, :] in l2 {\n";
    s += "    for j in 0..4 {\n"
         "      resident B[:, j];\n"
         "      acc C[i, j] in fabric {\n"
         "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n"
         "      }\n"
         "      store C[i, j] late;\n"
         "      release B[:, j];\n"
         "    }\n";
    if (staged) s += "    }\n";
    s += "    release A[i, :];\n  }\n}\n";
    return s;
}

std::string error_of(const std::string& src, bool symbolic) {
    try {
        if (symbolic) (void)lang::validate(src);
        else (void)lang::compile(src);
    } catch (const lang::CompileError& e) {
        return e.what();
    } catch (const lang::SyntaxError& e) {
        return e.what();
    }
    return "accepted";
}

CspProgram with_inputs(const std::string& src) {
    CspProgram p = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    return p;
}

std::string lb_error(const CspProgram& p) {
    try {
        BehavioralInterpreter b;
        b.run(p);
    } catch (const BehavioralError& e) {
        return e.what();
    }
    return "ran";
}

std::size_t index_of(const CspProgram& p, Action::Kind k, const std::string& tile, bool staged, std::size_t nth = 0) {
    for (std::size_t i = 0; i < p.actions.size(); ++i)
        if (p.actions[i].kind == k && p.actions[i].tile.to_string() == tile && p.actions[i].l2.has_value() == staged &&
            nth-- == 0)
            return i;
    FAIL("no such action");
    return 0;
}

}  // namespace

TEST_CASE("L2 staging: parsed, formatted and validated", "[program][csp][lang][l2]") {
    const lang::Program ast = lang::parse(matmul(true));
    CHECK(ast.l2 == 4);
    const std::string text = lang::format(ast);
    CHECK_THAT(text, ContainsSubstring("machine flat(l3 = 9, l2 = 4)"));
    CHECK_THAT(text, ContainsSubstring("stage A[i, :] in l2 {"));
    CHECK(lang::format(lang::parse(text)) == text);
    const lang::Validation v = lang::validate(matmul(true));
    CHECK(v.peak_l2 == 4);
    CHECK(v.peak_l3 == lang::validate(matmul(false)).peak_l3);   // staging takes no L3 slot
    CHECK(lang::validate(matmul(false)).peak_l2 == 0);
}

TEST_CASE("L2 staging: one Move per tile per stage, and Feeds of the copy", "[program][csp][lang][l2]") {
    const CspProgram s = lang::compile(matmul(true)), u = lang::compile(matmul(false));
    const ReuseReport rs = s.reuse(), ru = u.reuse();
    // A: 64 Moves unstaged (one per call), 16 staged (one per tile per row); B unchanged.
    CHECK(ru.moves == 128);
    CHECK(rs.moves == 80);
    CHECK(rs.stagings == 16);
    CHECK(rs.staged_feeds == 64);
    CHECK(rs.peak_l2 == 4);
    CHECK(rs.loads == ru.loads);
    REQUIRE(s.stagings.size() == 16);
    for (const Staging& st : s.stagings) {
        CHECK(st.feeds == 4);                            // each A tile feeds the four C tiles of its row
        CHECK(st.release != kNone);
        CHECK(s.actions[st.open].kind == Action::Kind::Move);
        CHECK(s.actions[st.release].kind == Action::Kind::L2Release);
        CHECK(s.residencies.at(st.residency).tile.to_string() == st.tile.to_string());
        CHECK(s.residencies.at(st.residency).release > st.release);   // L3 stays live through the block
    }
    // The stage's Moves open the block; A[0,1]'s Feeds read slot 1 and nothing moves it again.
    const std::size_t m = index_of(s, Action::Kind::Move, "A[0,1]", true);
    CHECK(s.actions[m].l2->slot == 1);
    for (std::size_t n = 0; n < 4; ++n) {
        const std::size_t f = index_of(s, Action::Kind::Feed, "A[0,1]", true, n);
        CHECK(f > m);
        CHECK(s.actions[f].l2->slot == 1);
        CHECK(s.actions[f].staging == s.actions[m].staging);
    }
    // The trace records the L2 transactions after the L3 one, and the release.
    CHECK(s.source.ops().at(s.actions[m].l0_op).kind == TileOpKind::L3Move);
    CHECK(s.source.ops().at(s.actions[m].l0_op + 1).kind == TileOpKind::L2Stage);
    const std::size_t r = index_of(s, Action::Kind::L2Release, "A[0,1]", true);
    CHECK(s.source.ops().at(s.actions[r].l0_op).kind == TileOpKind::L2Release);
    CHECK(s.actions[r].process == kNone);
}

TEST_CASE("L2 staging: the stream yields the trace's actions", "[program][csp][lang][l2]") {
    const std::string src = matmul(true);
    const CspProgram trace = lang::compile(src);
    lang::ActionStream stream(lang::parse(src));
    std::size_t i = 0;
    while (auto e = stream.next()) {
        REQUIRE(i < trace.actions.size());
        CAPTURE(i);
        const Action& t = trace.actions[i];
        CHECK(e->action.kind == t.kind);
        CHECK(e->action.tile.to_string() == t.tile.to_string());
        CHECK(e->action.l2.has_value() == t.l2.has_value());
        if (t.l2) CHECK(*e->action.l2 == *t.l2);
        CHECK(e->action.staging == t.staging);
        CHECK(e->opens_staging == (t.kind == Action::Kind::Move && t.l2.has_value()));
        ++i;
    }
    CHECK(i == trace.actions.size());
}

TEST_CASE("L2 staging: L-B computes what the unstaged program does, bit for bit", "[program][csp][l2]") {
    BehavioralInterpreter s, u;
    const auto sum = s.run(with_inputs(matmul(true)));
    u.run(with_inputs(matmul(false)));
    CHECK(s.result().operand("C").values == u.result().operand("C").values);
    CHECK(sum.peak_l2 == 4);
    CHECK(sum.staged_feeds == 64);
    CHECK(s.l2_contents().empty());
}

TEST_CASE("L2 staging: the trace's replay checks the L2 transactions", "[program][csp][l2]") {
    const lang::Program ast = lang::parse(matmul(true));
    const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
    REQUIRE(ref.trace);
    CHECK(ref.replay_mismatch.empty());
    TileProgram replay = *ref.trace;
    const auto sum = TileProgramReference().run(replay);
    CHECK(sum.l2_ops == 32);                             // 16 stages, 16 releases
    CHECK(sum.peak_l2 == 4);
    // Opset 1.3, and a round trip keeps the addresses.
    std::ostringstream os;
    serialize::write_l0(os, *ref.trace);
    CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.3.0"));
    CHECK_THAT(os.str(), ContainsSubstring("L2_STAGE"));
    std::istringstream is(os.str());
    TileProgram back = serialize::read_l0(is);
    CHECK(TileProgramReference().run(back).l2_ops == 32);
    // A stage into a live slot is refused by the replay.
    TileProgram bad = *ref.trace;
    std::vector<TileOp> ops = bad.ops();
    std::size_t second = 0, seen = 0;
    for (std::size_t i = 0; i < ops.size(); ++i)
        if (ops[i].kind == TileOpKind::L2Stage && seen++ == 1) second = i;
    ops[second].l2.slot = 0;                             // where A[0,0] is staged
    TileProgram tampered(bad.name());
    for (const auto& n : bad.operand_order()) tampered.add_operand(bad.operand(n));
    for (const auto& op : ops) tampered.push(op);
    CHECK_THROWS_WITH(TileProgramReference().run(tampered), ContainsSubstring("l2[0]:0 holds A[0,0], which is live"));
}

TEST_CASE("L2 staging: L-B refuses a broken staging, by name", "[program][csp][l2]") {
    const CspProgram good = with_inputs(matmul(true));
    REQUIRE(lb_error(good) == "ran");
    {   // A Feed of a slot that holds another tile.
        CspProgram p = good;
        p.actions[index_of(p, Action::Kind::Feed, "A[0,1]", true)].l2->slot = 2;
        CHECK_THAT(lb_error(p), ContainsSubstring("FEED A[0,1]: l2[0]:2 holds A[0,2], not A[0,1]"));
    }
    {   // A stage into a live slot.
        CspProgram p = good;
        p.actions[index_of(p, Action::Kind::Move, "A[0,1]", true)].l2->slot = 0;
        CHECK_THAT(lb_error(p), ContainsSubstring("MOVE A[0,1]: l2[0]:0 holds A[0,0], which is staged"));
    }
    {   // A stage outside the program's L2.
        CspProgram p = good;
        p.actions[index_of(p, Action::Kind::Move, "A[0,1]", true)].l2->slot = 4;
        CHECK_THAT(lb_error(p), ContainsSubstring("l2[0]:4 is outside the L2's 4 slots"));
    }
    {   // A program that ends with a tile staged.
        CspProgram p = good;
        const std::size_t r = index_of(p, Action::Kind::L2Release, "A[3,3]", true);
        p.actions.erase(p.actions.begin() + static_cast<std::ptrdiff_t>(r));
        CHECK_THAT(lb_error(p), ContainsSubstring("ends with A[3,3] still staged at l2[0]:3"));
    }
}

TEST_CASE("L2 staging: what the language refuses, by line", "[program][csp][lang][l2]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 2) {\n"
                             "  tensor A[32,32] tile 16x16 inout;\n  tensor B[32,32] tile 16x16 in;\n"
                             "  tensor C[32,32] tile 16x16 out;\n";
    auto both = [&](const std::string& body, const std::string& walk, const std::string& symbolic) {
        CHECK_THAT(error_of(head + body, false), ContainsSubstring(walk));
        CHECK_THAT(error_of(head + body, true), ContainsSubstring(symbolic));
    };
    // No L2 declared.
    const std::string no_l2 = "csp 1.0\nprogram t machine flat(l3 = 8) {\n  tensor A[32,32] tile 16x16 in;\n"
                              "  resident A[0, 0];\n  stage A[0, 0] in l2 { }\n  release A[0, 0];\n}\n";
    CHECK_THAT(error_of(no_l2, false), ContainsSubstring("csp line 5: stage ... in l2: the machine clause declares no L2"));
    CHECK_THAT(error_of(no_l2, true), ContainsSubstring("csp line 5: stage ... in l2: the machine clause declares no L2"));
    // Not resident.
    both("  stage A[0, 0] in l2 { }\n}\n", "csp line 6: stage A[0,0]: it is not resident",
         "csp line 6: stage A[0, 0]: it is not provably resident");
    // Staged twice.
    both("  resident A[0, :];\n  stage A[0, :] in l2 { stage A[0, 1] in l2 { } }\n  release A[0, :];\n}\n",
         "csp line 7: stage A[0,1]: it is already staged in L2", "csp line 7: stage A[0, 1]: it may already be staged");
    // More than the L2 holds.
    both("  resident A[:, :];\n  stage A[:, :] in l2 { }\n  release A[:, :];\n}\n",
         "csp line 7: stage A[1,0] needs an L2 slot, and all 2 are held",
         "csp line 7: stage needs 4 L2 slots, and 2 of 2 are free");
    // Released inside its block: the L3 slot stays live.
    both("  resident A[0, 0];\n  stage A[0, 0] in l2 { release A[0, 0]; }\n}\n",
         "csp line 7: release A[0,0]: it is staged in L2",
         "csp line 7: release A[0, 0], which can be staged in L2");
    // Written inside its block: the copy would be stale.
    both("  resident A[0, 0];\n  stage A[0, 0] in l2 { call relu(A[0, 0]) -> A[0, 0]; }\n  store A[0, 0];\n"
         "  release A[0, 0];\n}\n",
         "csp line 7: call relu writes A[0,0], which is staged in L2",
         "csp line 7: call relu writes A[0, 0], which can be staged in L2");
}

TEST_CASE("L2 staging: every level computes the oracle; L-CA re-Moves from L3", "[program][csp][run][l2]") {
    const auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    for (bool staged : {true, false}) {
        CAPTURE(staged);
        const lang::Program ast = lang::parse(matmul(staged));
        driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast),
                                  {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
        const auto r = driver::run_csp(req);
        REQUIRE(r.reference);
        REQUIRE(r.levels.size() == 3);
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            REQUIRE_FALSE(o.skipped);
            CHECK(o.values.operand("C").values == r.reference->operand("C").values);
        }
        CHECK(r.levels[2].staged_feeds == (staged ? 64u : 0u));
        bool noted = false;
        for (const auto& u : r.levels[2].unmodelled) noted = noted || u.find("L2 staging") != std::string::npos;
        CHECK(noted == staged);
    }
    // L-T1: the block mover moves A once per row, so it is busy for less.
    auto bm = [&](bool staged) {
        const lang::Program ast = lang::parse(matmul(staged));
        driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast), {ExecutionLevel::BlockSequential}};
        return driver::run_csp(req).levels[0].busy.at("bm");
    };
    // 48 fewer Moves of one size (80 against 128); the 16 Writebacks are the same in both.
    const auto saved = bm(false) - bm(true);
    CHECK(saved > 0);
    CHECK(saved % 48 == 0);
}

TEST_CASE("L2 staging: L-T1 holds an L2 credit per slot, stage Move to L2Release", "[program][csp][run][l2]") {
    const auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    const lang::Program ast = lang::parse(matmul(true));
    const auto o = driver::csp_run_level(ExecutionLevel::BlockSequential, ast, &d, driver::csp_inputs(ast));
    REQUIRE_FALSE(o.skipped);
    // A's 16 tiles, each staged once (one per row and k) and fed once per column j.
    REQUIRE(o.l2_slots.size() == 16);
    CHECK(o.peak_l2 == 4);
    std::map<std::uint32_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> by_slot;
    for (const auto& sl : o.l2_slots) {
        CAPTURE(sl.tile.to_string());
        CHECK(sl.feeds == 4);
        CHECK(sl.t1 > sl.t0);
        by_slot[sl.place.slot].emplace_back(sl.t0, sl.t1);
    }
    // Every row reuses slots 0-3: the next row's stage into a slot starts no earlier than the
    // L2Release of the copy before it -- a write-after-read on the slot, as for L3.
    REQUIRE(by_slot.size() == 4);
    for (auto& [slot, iv] : by_slot) {
        CAPTURE(slot);
        REQUIRE(iv.size() == 4);
        std::sort(iv.begin(), iv.end());
        for (std::size_t i = 1; i < iv.size(); ++i) CHECK(iv[i].first >= iv[i - 1].second);
    }
    // Each stage Move starts no earlier than its copy's credit.
    std::size_t k = 0;
    for (const auto& rec : o.records)
        if (rec.kind == Action::Kind::Move && k < o.l2_slots.size() &&
            rec.tile.to_string() == o.l2_slots[k].tile.to_string()) {
            CHECK(rec.start >= o.l2_slots[k].t0);
            ++k;
        }
    CHECK(k == o.l2_slots.size());
}

TEST_CASE("L2 staging: the trace prints as a program with its stage blocks", "[program][csp][lang][l2]") {
    const CspProgram p = lang::compile(matmul(true));
    const std::string text = lang::print(p);
    CHECK_THAT(text, ContainsSubstring("l2 = 4"));
    CHECK_THAT(text, ContainsSubstring("stage A[0, 0], A[0, 1], A[0, 2], A[0, 3] in l2 {"));
    const CspProgram again = lang::compile(text);
    REQUIRE(again.actions.size() == p.actions.size());
    for (std::size_t i = 0; i < p.actions.size(); ++i) {
        CAPTURE(i);
        CHECK(again.actions[i].kind == p.actions[i].kind);
        CHECK(again.actions[i].tile.to_string() == p.actions[i].tile.to_string());
        CHECK(again.actions[i].l2.has_value() == p.actions[i].l2.has_value());
    }
}

TEST_CASE("L2 staging: kpu-csp-gen --stage-a stages A's row panel", "[program][csp][gen][l2]") {
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.orient = "row";
    const std::string plain = gen::generate(o);
    o.stage_a = true;
    const std::string staged = gen::generate(o);
    CHECK_THAT(staged, ContainsSubstring("machine flat(l3 = 9, l2 = 4)"));
    CHECK_THAT(staged, ContainsSubstring("stage A[i, :] in l2 {"));
    const ReuseReport s = lang::compile(staged).reuse(), u = lang::compile(plain).reuse();
    CHECK(u.moves - s.moves == 64 - 16);                 // A: once per tile per row, not per call
    BehavioralInterpreter a, b;
    a.run(with_inputs(staged));
    b.run(with_inputs(plain));
    CHECK(a.result().operand("C").values == b.result().operand("C").values);
    o.orient = "col";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("needs orient row"));
}
