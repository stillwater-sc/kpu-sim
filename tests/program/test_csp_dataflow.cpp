// ============================================================================
// tests/program/test_csp_dataflow.cpp
// Weight- and input-stationary matmul programs (#430), and the Combiner's merge of partials they
// reduce through (#429): `acc C[:, j] in l2 { ... }` holds C's partials in L2 slots, every call's
// result is drained from the fabric and merged into its slot by the Combiner (overwrite for the
// first K block, accumulate for the rest), and the store writes the slot back. The stationary
// operand is staged in L2 across the loop that reuses it, so it leaves L3 once per reuse loop.
//
// Values answer to L-B: WS and IS compute what OS does, per tile within the format's bound (exact
// for the integer path), the trace's replay reproduces L-B bit for bit, and L-T1 is L-B's.
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
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/program/tile_program_reference.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
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

// 64 in 16-tiles: 4 x 4 x 4 tile calls.
std::string program(const std::string& dataflow, const std::string& dtype = "fp32") {
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.dataflow = dataflow;
    o.dtype = dtype;
    return gen::generate(o);
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

// L-B's values of the program, streamed, on the inputs the run driver fills.
TileProgram lb_values(const std::string& src) {
    const lang::Program ast = lang::parse(src);
    const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
    REQUIRE(ref.values);
    return *ref.values;
}

// Bit for bit: a typed operand's elements are words in the floats' bits, which may read as NaN.
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// S1 with a Combiner: the unit that merges the partials.
platform::DeviceSpecification s1_with_combiner() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu"}};
    return d;
}

std::size_t count(const CspProgram& p, Action::Kind k, const std::string& operand) {
    std::size_t n = 0;
    for (const Action& a : p.actions)
        if (a.kind == k && a.tile.operand == operand) ++n;
    return n;
}

}  // namespace

TEST_CASE("Dataflow: the WS and IS programs are generated, validated and described", "[program][csp][dataflow]") {
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const std::string src = program(df);
        // The stationary operand staged across the reuse loop; C's partials an L2 accumulator.
        CHECK_THAT(src, ContainsSubstring("machine flat(l3 = 9, l2 = 5)"));
        CHECK_THAT(src, ContainsSubstring(std::string(df) == "ws" ? "acc C[:, j] in l2 {" : "acc C[i, :] in l2 {"));
        CHECK_THAT(src, ContainsSubstring(std::string(df) == "ws" ? "stage B[k, j] in l2 {" : "stage A[i, k] in l2 {"));
        CHECK_THAT(src, ContainsSubstring(std::string(df) == "ws" ? "weight-stationary" : "input-stationary"));
        CHECK_THAT(src, ContainsSubstring("(5 used at peak)"));
        const lang::Validation v = lang::validate(src);
        CHECK(v.peak_l3 == 5);                           // a panel of 4 and the stationary tile
        CHECK(v.peak_l2 == 5);                           // 4 partials and the stationary tile
        REQUIRE(v.calls);
        CHECK(*v.calls == 64);
        CHECK(*v.stores == 16);
        // The formatter round-trips `acc ... in l2`.
        const std::string text = lang::format(lang::parse(src));
        CHECK(lang::format(lang::parse(text)) == text);
        CHECK(lang::compile(text).actions.size() == lang::compile(src).actions.size());
    }
    // os is the default and its program is unchanged by the option.
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    const std::string def = gen::generate(o);
    CHECK(program("os") == def);
    CHECK(def.find(" in l2") == std::string::npos);
}

TEST_CASE("Dataflow: the generator refuses what WS and IS do not take", "[program][csp][dataflow]") {
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    auto refused = [&](gen::Options x) {
        try {
            (void)gen::generate(x);
        } catch (const gen::GenError& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };
    gen::Options x = o;
    x.dataflow = "rs";
    CHECK_THAT(refused(x), ContainsSubstring("dataflow 'rs' (os | ws | is)"));
    x = o;
    x.dataflow = "ws";
    x.orient = "row";
    CHECK_THAT(refused(x), ContainsSubstring("dataflow ws fixes its own loop order"));
    x = o;
    x.dataflow = "is";
    x.init = "accumulate";                                    // init bias and bias-row are built (#427)
    CHECK_THAT(refused(x), ContainsSubstring("init accumulate (C += A . B) onto an L2 accumulator"));
    x = o;
    x.dataflow = "ws";
    x.algo = "lu";
    CHECK_THAT(refused(x), ContainsSubstring("dataflow ws is matmul's and linear's"));
    x = o;
    x.dataflow = "ws";
    x.l3 = 4;
    CHECK_THAT(refused(x), ContainsSubstring("needs 5 L3 slots (a panel of 4 tiles and the stationary tile)"));
}

TEST_CASE("Dataflow: the stationary operand leaves L3 once per reuse loop; every call merges", "[program][csp][dataflow]") {
    const CspProgram os = lang::compile(program("os"));
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const CspProgram p = lang::compile(program(df));
        const bool ws = std::string(df) == "ws";
        const std::string held = ws ? "B" : "A", moving = ws ? "A" : "B";
        // Counted from the trace: one Move of each stationary tile per (k, panel) -- 16 -- against
        // one per call -- 64 -- for the streamed operand, and for both under OS.
        CHECK(count(p, Action::Kind::Move, held) == 16);
        CHECK(count(p, Action::Kind::Move, moving) == 64);
        CHECK(count(os, Action::Kind::Move, "A") == 64);
        CHECK(count(os, Action::Kind::Move, "B") == 64);
        // Every call's partial is drained and merged: 64 Drains of C, against OS's 16.
        CHECK(count(p, Action::Kind::Drain, "C") == 64);
        CHECK(count(os, Action::Kind::Drain, "C") == 16);
        const ReuseReport r = p.reuse();
        CHECK(r.stagings == 16);
        CHECK(r.staged_feeds == 64);
        CHECK(r.partials == 16);
        CHECK(r.merges == 64);
        CHECK(r.peak_l2 == 5);
        CHECK(r.stores == 16);
        // Each partial: opened by its first merge (overwrite), merged into 4 times, written back,
        // then its L2 slot freed.
        std::size_t partials = 0;
        for (const Staging& st : p.stagings) {
            if (!st.partial) {
                CHECK(st.feeds == 4);                    // the stationary tile, fed across the reuse loop
                continue;
            }
            ++partials;
            CAPTURE(st.tile.to_string());
            CHECK(st.merges == 4);
            CHECK(st.feeds == 0);
            CHECK(st.residency == kNone);
            const Action& open = p.actions.at(st.open);
            CHECK(open.kind == Action::Kind::Drain);
            CHECK(open.merge == Merge::Overwrite);
            const Action& rel = p.actions.at(st.release);
            CHECK(rel.kind == Action::Kind::L2Release);
            const Action& wb = p.actions.at(st.release - 1);
            CHECK(wb.kind == Action::Kind::Writeback);
            REQUIRE(wb.l2);
            CHECK(*wb.l2 == st.place);
        }
        CHECK(partials == 16);
        // The Combiner's merges: the first into each slot overwrites, the rest accumulate.
        std::map<std::string, std::vector<Merge>> merges;
        for (const Action& a : p.actions)
            if (a.merge != Merge::None) merges[a.tile.to_string()].push_back(a.merge);
        REQUIRE(merges.size() == 16);
        for (const auto& [k, m] : merges) {
            CAPTURE(k);
            REQUIRE(m.size() == 4);
            CHECK(m[0] == Merge::Overwrite);
            for (std::size_t i = 1; i < m.size(); ++i) CHECK(m[i] == Merge::Accumulate);
        }
        // The disassembly names the merge.
        CHECK_THAT(p.disassemble(), ContainsSubstring("merge accumulate @ combiner"));
        CHECK_THAT(p.disassemble(), ContainsSubstring("16 partials in L2 for 64 merges"));
    }
}

TEST_CASE("Dataflow: the stream yields the trace's actions", "[program][csp][dataflow]") {
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const std::string src = program(df);
        const CspProgram trace = lang::compile(src);
        lang::ActionStream stream(lang::parse(src));
        std::size_t i = 0;
        while (auto e = stream.next()) {
            REQUIRE(i < trace.actions.size());
            CAPTURE(i);
            const Action& t = trace.actions[i];
            CHECK(e->action.kind == t.kind);
            CHECK(e->action.tile.to_string() == t.tile.to_string());
            CHECK(e->action.merge == t.merge);
            CHECK(e->action.l2.has_value() == t.l2.has_value());
            if (t.l2) CHECK(*e->action.l2 == *t.l2);
            CHECK(e->action.staging == t.staging);
            ++i;
        }
        CHECK(i == trace.actions.size());
    }
}

TEST_CASE("Dataflow: at L-B, WS and IS compute OS's values, per tile", "[program][csp][dataflow]") {
    for (const char* dtype : {"fp32", "int8"}) {
        CAPTURE(dtype);
        const TileProgram os = lb_values(program("os", dtype));
        for (const char* df : {"ws", "is"}) {
            CAPTURE(df);
            const TileProgram v = lb_values(program(df, dtype));
            if (std::string(dtype) == "fp32") {
                const OperandsCheck c = check_operands(v, os);
                CHECK(c.pass);
                INFO(c.summary());
            } else {
                // The integer path is exact: int8 x int8 summed in int32, merged in int32.
                CHECK(same_bits(v.operand("C").values, os.operand("C").values));
            }
            CHECK(same_bits(v.operand("A").values, os.operand("A").values));
            CHECK(same_bits(v.operand("B").values, os.operand("B").values));
        }
    }
}

TEST_CASE("Dataflow: the trace records MATMUL_MERGE, and its replay reproduces L-B", "[program][csp][dataflow]") {
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const lang::Program ast = lang::parse(program(df));
        const auto ref = driver::csp_reference(ast, driver::csp_inputs(ast), 1'000'000);
        REQUIRE(ref.trace);
        CHECK(ref.replay_mismatch.empty());
        TileProgram replay = *ref.trace;
        const auto sum = TileProgramReference().run(replay);
        CHECK(sum.merges == 64);
        CHECK(sum.computes == 64);
        CHECK(sum.peak_l2 == 5);
        // Opset 1.5, and a round trip keeps each merge's mode and slot.
        std::ostringstream os;
        serialize::WriteOptions with_values;
        with_values.include_values = true;
        serialize::write_l0(os, *ref.trace, with_values);
        CHECK_THAT(os.str(), ContainsSubstring("OPSET tile 1.5.0"));
        CHECK_THAT(os.str(), ContainsSubstring("kind=MATMUL_MERGE"));
        CHECK_THAT(os.str(), ContainsSubstring("merge=overwrite l2=0 slot=0"));
        CHECK_THAT(os.str(), ContainsSubstring("merge=accumulate"));
        std::istringstream is(os.str());
        TileProgram back = serialize::read_l0(is);
        TileProgram again = *ref.trace;
        (void)TileProgramReference().run(back);
        (void)TileProgramReference().run(again);
        CHECK(same_bits(back.operand("C").values, again.operand("C").values));
        CHECK(same_bits(again.operand("C").values, ref.values->operand("C").values));
        // An accumulate into a slot the first merge did not open is refused by the replay.
        std::vector<TileOp> ops = ref.trace->ops();
        for (TileOp& op : ops)
            if (op.kind == TileOpKind::MatMulMerge) {
                op.overwrite = false;                    // the first merge, made an accumulate
                break;
            }
        TileProgram tampered(ref.trace->name());
        for (const auto& n : ref.trace->operand_order()) tampered.add_operand(ref.trace->operand(n));
        for (const auto& op : ops) tampered.push(op);
        CHECK_THROWS_WITH(TileProgramReference().run(tampered), ContainsSubstring("the first merge overwrites"));
    }
}

TEST_CASE("Dataflow: L-B refuses a broken merge, by name", "[program][csp][dataflow]") {
    const std::string src = program("ws");
    CspProgram good = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : good.source.operand_order()) good.source.operand(name).values = in.operand(name).values;
    auto lb_error = [](const CspProgram& p) {
        try {
            BehavioralInterpreter b;
            b.run(p);
        } catch (const BehavioralError& e) {
            return std::string(e.what());
        }
        return std::string("ran");
    };
    REQUIRE(lb_error(good) == "ran");
    auto nth_merge = [](const CspProgram& p, Merge m, std::size_t nth) {
        for (std::size_t i = 0; i < p.actions.size(); ++i)
            if (p.actions[i].merge == m && nth-- == 0) return i;
        FAIL("no such merge");
        return std::size_t{0};
    };
    {   // An accumulate into a slot that holds another partial.
        CspProgram p = good;
        Action& a = p.actions[nth_merge(p, Merge::Accumulate, 0)];
        a.l2->slot = (a.l2->slot + 1) % 4;
        CHECK_THAT(lb_error(p), ContainsSubstring("merge accumulate into l2[0]:"));
        CHECK_THAT(lb_error(p), ContainsSubstring(", not C["));
    }
    {   // A first merge into a live slot.
        CspProgram p = good;
        Action& a = p.actions[nth_merge(p, Merge::Overwrite, 1)];
        a.l2->slot = 0;
        CHECK_THAT(lb_error(p), ContainsSubstring("l2[0]:0 holds C[0,0], which is staged"));
    }
    {   // An accumulate whose slot was never opened.
        CspProgram p = good;
        p.actions[nth_merge(p, Merge::Overwrite, 0)].merge = Merge::Accumulate;
        CHECK_THAT(lb_error(p), ContainsSubstring("the slot is free (C[0,0]'s first merge overwrites it)"));
    }
}

TEST_CASE("Dataflow: what the language refuses for an L2 accumulator, by line", "[program][csp][lang][dataflow]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 4) {\n"
                             "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                             "  tensor C[32,32] tile 16x16 out;\n";
    auto both = [&](const std::string& body, const std::string& walk, const std::string& symbolic) {
        CHECK_THAT(error_of(head + body, false), ContainsSubstring(walk));
        CHECK_THAT(error_of(head + body, true), ContainsSubstring(symbolic));
    };
    const std::string ok = "  acc C[0, :] in l2 {\n    resident A[0, 0], B[0, :];\n"
                           "    for j in 0..2 { call gemm(A[0, 0], B[0, j]) +-> C[0, j]; }\n"
                           "    release A[0, 0], B[0, :];\n  }\n";
    REQUIRE(error_of(head + ok + "  store C[0, :];\n}\n", false) == "accepted");
    REQUIRE(error_of(head + ok + "  store C[0, :];\n}\n", true) == "accepted");
    // No L2 declared.
    const std::string no_l2 = "csp 1.0\nprogram t machine flat(l3 = 8) {\n  tensor C[32,32] tile 16x16 out;\n"
                              "  acc C[0, 0] in l2 { }\n}\n";
    CHECK_THAT(error_of(no_l2, false), ContainsSubstring("csp line 4: acc ... in l2: the machine clause declares no L2"));
    CHECK_THAT(error_of(no_l2, true), ContainsSubstring("csp line 4: acc ... in l2: the machine clause declares no L2"));
    // An init onto the accumulator itself is refused: the Combiner adds another tile's (#427).
    CHECK_THAT(error_of(head + "  acc C[0, 0] in l2 init C[0, 0] { }\n}\n", false),
               ContainsSubstring("csp line 6: acc C[0,0] in l2 init C[0,0]: accumulating onto C itself"));
    // More partials than the L2 holds.
    both("  acc C[:, :] in l2 {\n  }\n}\n", "csp line 6: acc C[0,0] in l2 receives no call",
         "csp line 6: acc C[:, :] in l2 receives no call");
    both("  resident A[0, 0];\n  stage A[0, 0] in l2 {\n    acc C[:, :] in l2 { }\n  }\n  release A[0, 0];\n}\n",
         "csp line 8: acc C[1,1] in l2 needs an L2 slot for its partial, and all 4 are held",
         "csp line 8: acc C[:, :] in l2 needs 4 L2 slots for its partials, and 3 of 4 are free");
    // A tile with no call.
    both("  acc C[0, :] in l2 {\n    resident A[0, 0], B[0, :];\n    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
         "    release A[0, 0], B[0, :];\n  }\n  store C[0, :];\n}\n",
         "csp line 6: acc C[0,1] in l2 receives no call", "accepted");
    // Not a gemm.
    both("  acc C[0, :] in l2 {\n    call relu(C[0, 0]) -> C[0, 0];\n  }\n}\n",
         "csp line 7: call relu on the L2 accumulator C[0,0]",
         "csp line 7: call relu on the L2 accumulator C[0, 0]");
    // A late store, and an epilogue on the final merge.
    both(ok + "  store C[0, :] late;\n}\n", "csp line 11: store C[0,0] late: C[0,0] is an L2 accumulator",
         "csp line 11: store C[0, :] late: C[0, :] is an L2 accumulator");
    both(ok + "  store C[0, :] via relu @ combiner;\n}\n",
         "csp line 11: store C[0,0] via ... @ combiner: C[0,0] is an L2 accumulator",
         "csp line 11: store C[0, :] via ... @ combiner: C[0, :] is an L2 accumulator");
    // A retain.
    both(ok + "  retain C[0, :];\n}\n", "csp line 11: retain C[0,0]: C[0,0] is an L2 accumulator",
         "csp line 11: retain C[0, :]: C[0, :] is an L2 accumulator");
    // The flat printer has no form for it; the formatter prints the source.
    CHECK_THROWS_WITH(lang::print(lang::compile(program("ws"))), ContainsSubstring("has no flat text form"));
}

TEST_CASE("Dataflow: L-T1 computes L-B's values, and holds each partial's L2 slot", "[program][csp][run][dataflow]") {
    const auto d = s1_with_combiner();
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const lang::Program ast = lang::parse(program(df));
        driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast),
                                  {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential}};
        const auto r = driver::run_csp(req);
        REQUIRE(r.reference);
        REQUIRE(r.levels.size() == 2);
        for (const auto& o : r.levels) {
            CAPTURE(to_string(o.level));
            REQUIRE_FALSE(o.skipped);
            CHECK(o.values.operand("C").values == r.reference->operand("C").values);
        }
        const auto& lt = r.levels[1];
        // 16 stationary copies, fed 4 times each, and 16 partials, merged 4 times each.
        std::size_t partials = 0, stagings = 0;
        std::map<std::uint32_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> by_slot;
        for (const auto& sl : lt.l2_slots) {
            CAPTURE(sl.tile.to_string());
            CHECK(sl.t1 > sl.t0);
            by_slot[sl.place.slot].emplace_back(sl.t0, sl.t1);
            if (sl.partial) {
                ++partials;
                CHECK(sl.merges == 4);
                CHECK(sl.feeds == 0);
            } else {
                ++stagings;
                CHECK(sl.feeds == 4);
            }
        }
        CHECK(partials == 16);
        CHECK(stagings == 16);
        CHECK(lt.peak_l2 <= 5);
        // A slot's next holder starts no earlier than its last one's L2Release.
        for (auto& [slot, iv] : by_slot) {
            CAPTURE(slot);
            std::sort(iv.begin(), iv.end());
            for (std::size_t i = 1; i < iv.size(); ++i) CHECK(iv[i].first >= iv[i - 1].second);
        }
        // The L2 waits counted (#432's review): each is a record's, they sum to the total, and a
        // record that counts one starts exactly when its slot's credit returned -- the credit was
        // its binding constraint, not the drain lane or the issue order.
        std::uint64_t waits = 0;
        std::size_t bound = 0;
        for (const auto& rec : lt.records) {
            if (rec.l2_wait == 0) continue;
            waits += rec.l2_wait;
            ++bound;
            CAPTURE(rec.tile.to_string(), rec.start, rec.l2_wait);
            CHECK(std::any_of(lt.l2_slots.begin(), lt.l2_slots.end(), [&](const auto& sl) {
                return sl.tile.to_string() == rec.tile.to_string() && sl.t0 == rec.start;
            }));
        }
        CHECK(waits == lt.l2_wait_cycles);
        CHECK(bound == lt.l2_bound);
        CHECK(lt.merges == 64);
        CHECK(lt.merge_accumulates == 48);
        // An accumulate reads its partial and writes it back; every merge writes a 16x16 fp32 tile.
        CHECK(lt.merge_l2_read_bytes == 48u * 16 * 16 * 4);
        CHECK(lt.merge_l2_write_bytes == 64u * 16 * 16 * 4);
    }
}

TEST_CASE("Dataflow: L-CA computes L-B's values, merging each partial in its L2 slot", "[program][csp][run][dataflow]") {
    const auto d = s1_with_combiner();
    for (const char* dtype : {"fp32", "int8"})
        for (const char* df : {"os", "ws", "is"}) {
            CAPTURE(dtype, df);
            const lang::Program ast = lang::parse(program(df, dtype));
            driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast), {ExecutionLevel::CycleAccurate}};
            const auto r = driver::run_csp(req);
            REQUIRE(r.reference);
            REQUIRE(r.levels.size() == 1);
            const auto& o = r.levels[0];
            REQUIRE_FALSE(o.skipped);
            // Values answer to L-B, per tile; the fabric's values are exact here, so bit for bit.
            if (std::string(dtype) == "fp32") {
                const OperandsCheck c = check_operands(o.values, *r.reference);
                INFO(c.summary());
                CHECK(c.pass);
            }
            CHECK(same_bits(o.values.operand("C").values, r.reference->operand("C").values));
            if (std::string(df) == "os") {
                CHECK(o.merges == 0);
                continue;
            }
            CHECK(o.merges == 64);                       // every call's partial, merged by the Combiner
            CHECK(o.merge_accumulates == 48);            // all but each tile's first: read-modify-writes
            CHECK(o.staged_feeds == 64);                 // the stationary tile, fed from its L2 copy
            // Each partial's L2 slot, as L-CA claimed it: the program's slots 0-3 for C's panel.
            std::size_t partial_claims = 0;
            for (const auto& [t, slot] : o.l2_placement)
                if (t.operand == "C") {
                    ++partial_claims;
                    CHECK(slot < 4);
                }
            CHECK(partial_claims == 16);
        }
    // A machine with no Combiner cannot merge: L-CA says so, and the language refuses it on that
    // target.
    const auto bare = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    const lang::Program ast = lang::parse(program("ws"));
    const auto o = driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &bare, driver::csp_inputs(ast));
    REQUIRE(o.skipped);
    CHECK_THAT(*o.skipped, ContainsSubstring("merges its partials on the Combiner, which the machine does not declare"));
    const lang::Target t = lang::target_from(bare);
    CHECK_THROWS_WITH(lang::validate(program("ws"), &t), ContainsSubstring("the target has no Combiner"));
}

TEST_CASE("Dataflow: a merge's result is never one of its inputs", "[program][csp][dataflow]") {
    TileProgram p("overlap");
    p.add_operand(TensorOperand("A", 1, 1, 1, 1)).at(0, 0) = 2.0f;
    p.add_operand(TensorOperand("B", 1, 1, 1, 1)).at(0, 0) = 3.0f;
    TileOp op;
    op.kind = TileOpKind::MatMulMerge;
    op.inputs = {TileCoord{"A", 0, 0}, TileCoord{"B", 0, 0}};
    op.outputs = {TileCoord{"A", 0, 0}};
    op.overwrite = true;
    TileKernelState st;
    CHECK_THROWS_WITH(apply(p, op, st), ContainsSubstring("the result A[0,0] is also an input"));
    CHECK(p.operand("A").at(0, 0) == 2.0f);              // refused before anything is written
}

TEST_CASE("Dataflow: L-CA keeps each drain's tile context with its own drain", "[program][csp][run][dataflow]") {
    // C[0, 0] is merged in L2 (two drains, no stages) and stored, then read back and written in
    // place through a ReLU on the Combiner. Every drain of a tile takes its own entry in the
    // tile's stage queue, so the merges' completions never consume the later drain's ReLU: a
    // ReLU applied to the first partial would give relu(relu(p0) + p1), not relu(p0 + p1).
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 4) {\n"
                            "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                            "  tensor C[32,32] tile 16x16 inout;\n"
                            "  acc C[0, 0] in l2 {\n    resident A[0, :], B[:, 0];\n"
                            "    for k in 0..2 { call gemm(A[0, k], B[k, 0]) +-> C[0, 0]; }\n"
                            "    release A[0, :], B[:, 0];\n  }\n  store C[0, 0];\n"
                            "  resident C[0, 0];\n  call relu(C[0, 0]) -> C[0, 0] via relu @ combiner;\n"
                            "  store C[0, 0];\n  release C[0, 0];\n}\n";
    const auto d = s1_with_combiner();
    const lang::Program ast = lang::parse(src);
    driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast), {ExecutionLevel::CycleAccurate}};
    const auto r = driver::run_csp(req);
    REQUIRE(r.reference);
    CHECK(r.reference_note.empty());
    INFO(r.levels.at(0).skipped.value_or(""));
    REQUIRE_FALSE(r.levels.at(0).skipped);
    CHECK(r.levels[0].merges == 2);
    CHECK(same_bits(r.levels[0].values.operand("C").values, r.reference->operand("C").values));
    // The data must reach the case: some element of the first partial is negative where the sum
    // is not, or the two orders would agree.
    bool negative = false;
    for (float v : r.reference->operand("C").values) negative = negative || v == 0.0f;
    CHECK(negative);
}

// ---- #427: the epilogue on the final merge, and the fused-operation matrix ----------------------

namespace {

// S1 with both vector units: the Combiner, which runs every activation, and the BlockMovers'.
platform::DeviceSpecification s1_with_vector_units() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu", "gelu", "silu", "atan"}};
    d.movers.bm_vector = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu", "gelu", "silu", "atan"}};
    return d;
}

std::string linear(const std::string& dataflow, const std::string& place, ActivationFn act) {
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dataflow = dataflow;
    o.place = place;
    o.act = act;
    return gen::generate(o);
}

}  // namespace

TEST_CASE("Final merge: the epilogue rides each tile's last call, on the Combiner", "[program][csp][dataflow][epilogue]") {
    for (const char* df : {"ws", "is"}) {
        CAPTURE(df);
        const std::string src = linear(df, "combiner", ActivationFn::Relu);
        CHECK_THAT(src, ContainsSubstring("for k in 0..3 {"));        // the last K block written out
        CHECK_THAT(src, ContainsSubstring("+-> C[i, j] via add(b[j]) @ combiner, relu @ combiner;"));
        CHECK_THAT(src, ContainsSubstring("on each tile's final merge"));
        const lang::Validation v = lang::validate(src);
        CHECK(v.peak_l3 == 6);                                         // a panel, the stationary tile, b's tile
        const CspProgram p = lang::compile(src);
        // Exactly one Drain per tile carries the epilogue: its fourth merge, an accumulate.
        std::map<std::string, std::size_t> fused, merges;
        for (const Action& a : p.actions) {
            if (a.merge == Merge::None) continue;
            const std::string k = a.tile.to_string();
            ++merges[k];
            if (!a.context.empty()) {
                ++fused[k];
                CHECK(merges[k] == 4);
                CHECK(a.merge == Merge::Accumulate);
                REQUIRE(a.context.size() == 2);
                CHECK(a.context[0].op == VeOp::Add);
                CHECK(a.context[0].place == Place::Combiner);
                CHECK(a.context[1].op == VeOp::Relu);
            }
        }
        CHECK(fused.size() == 16);
        for (const auto& [k, n] : fused) CHECK(n == 1);
    }
    // @ bm.egress under ws: the panel's store carries it, and no merge does.
    const CspProgram eg = lang::compile(linear("ws", "bm.egress", ActivationFn::Relu));
    for (const Action& a : eg.actions) {
        if (a.merge != Merge::None) CHECK(a.context.empty());
        if (a.kind == Action::Kind::Writeback && a.tile.operand == "C") CHECK(a.context.size() == 2);
    }
}

TEST_CASE("Final merge: the fused-operation matrix computes one result in every placement and dataflow",
          "[program][csp][run][dataflow][epilogue]") {
    const auto d = s1_with_vector_units();
    for (ActivationFn act : {ActivationFn::Relu, ActivationFn::Gelu, ActivationFn::Silu, ActivationFn::Atan}) {
        // The oracle's form: output-stationary, the epilogue unfused (a second pass of calls).
        const std::string oracle_src = linear("os", "unfused", act);
        const lang::Program oracle_ast = lang::parse(oracle_src);
        const TileProgram inputs = driver::csp_inputs(oracle_ast);
        const auto oracle = driver::csp_reference(oracle_ast, inputs, 1'000'000);
        REQUIRE(oracle.values);
        struct Cell { const char* df; const char* place; };
        for (const Cell c : {Cell{"os", "fabric"}, Cell{"os", "combiner"}, Cell{"os", "bm.egress"},
                             Cell{"ws", "combiner"}, Cell{"ws", "bm.egress"}, Cell{"ws", "unfused"},
                             Cell{"is", "combiner"}, Cell{"is", "unfused"}}) {
            CAPTURE(to_string(act), c.df, c.place);
            // The fabric's epilogue is the matmul's bias and ReLU (lang::Target::fabric).
            if (std::string(c.place) == "fabric" && act != ActivationFn::Relu) continue;
            const lang::Program ast = lang::parse(linear(c.df, c.place, act));
            // The same inputs for every cell: C's role (out or inout) does not change what A, B and b hold.
            TileProgram in = driver::csp_inputs(ast);
            for (const char* n : {"A", "B", "b"}) in.operand(n).values = inputs.operand(n).values;
            driver::CspRunRequest req{ast, &d, in,
                                      {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential,
                                       ExecutionLevel::CycleAccurate}};
            const auto r = driver::run_csp(req);
            REQUIRE(r.reference);
            CHECK(r.reference_note.empty());
            // L-B, per tile within the format's bound, against the unfused output-stationary oracle.
            TileProgram got = *oracle.values;
            got.operand("C").values = r.reference->operand("C").values;
            const OperandsCheck chk = check_operands(got, *oracle.values);
            INFO(chk.summary());
            CHECK(chk.pass);
            // The refining levels: L-B's values, bit for bit.
            for (const auto& o : r.levels) {
                CAPTURE(to_string(o.level));
                INFO(o.skipped.value_or(""));
                REQUIRE_FALSE(o.skipped);
                CHECK(same_bits(o.values.operand("C").values, r.reference->operand("C").values));
            }
            if (std::string(c.df) != "os" && std::string(c.place) == "combiner")
                CHECK(r.levels[2].ve.combiner_busy > 0);          // the merges and the epilogue on the Combiner
        }
    }
}

TEST_CASE("Final merge: what the language and the generator refuse", "[program][csp][lang][dataflow][epilogue]") {
    const std::string head = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 4) {\n"
                             "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                             "  vector b[32] tile 16 in;\n  tensor C[32,32] tile 16x16 out;\n";
    const std::string open = "  acc C[0, 0] in l2 {\n    resident A[0, :], B[:, 0], b[0];\n";
    const std::string close = "    release A[0, :], B[:, 0], b[0];\n  }\n  store C[0, 0];\n}\n";
    auto both = [&](const std::string& body, const std::string& walk, const std::string& symbolic) {
        CHECK_THAT(error_of(head + open + body + close, false), ContainsSubstring(walk));
        CHECK_THAT(error_of(head + open + body + close, true), ContainsSubstring(symbolic));
    };
    const std::string fin = "    call gemm(A[0, 1], B[1, 0]) +-> C[0, 0] via add(b[0]) @ combiner, relu @ combiner;\n";
    REQUIRE(error_of(head + open + "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n" + fin + close, false) == "accepted");
    REQUIRE(error_of(head + open + "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n" + fin + close, true) == "accepted");
    // A call after the final merge, written after it or in a later iteration of its loop.
    both(fin + "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n",
         "csp line 10: call gemm into C[0,0] after its final merge (csp line 9)",
         "csp line 10: call gemm into C[0, 0], which can be after its final merge (C[0, 0], csp line 9)");
    both("    for k in 0..2 { call gemm(A[0, k], B[k, 0]) +-> C[0, 0] via relu @ combiner; }\n",
         "csp line 9: call gemm into C[0,0] after its final merge (csp line 9)",
         "csp line 9: call gemm into C[0, 0] via ...: the final merge runs in every iteration of the loop over k");
    // After a loop of final merges over j, every tile it ranged over is final.
    const std::string row = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 4) {\n"
                            "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                            "  tensor C[32,32] tile 16x16 out;\n"
                            "  acc C[0, :] in l2 {\n    resident A[0, 0], B[0, :];\n"
                            "    for j in 0..2 { call gemm(A[0, 0], B[0, j]) +-> C[0, j] via relu @ combiner; }\n"
                            "    call gemm(A[0, 0], B[0, 1]) +-> C[0, 1];\n"
                            "    release A[0, 0], B[0, :];\n  }\n  store C[0, :];\n}\n";
    CHECK_THAT(error_of(row, true), ContainsSubstring("csp line 9: call gemm into C[0, 1], which can be after its final merge"));
    CHECK_THAT(error_of(row, false), ContainsSubstring("csp line 9: call gemm into C[0,1] after its final merge"));
    // An epilogue stage off the Combiner.
    both("    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0] via relu @ fabric;\n",
         "csp line 9: call gemm +-> C[0,0] via ... @ fabric: the epilogue on an L2 accumulator's final merge runs on the "
         "Combiner", "csp line 9: call gemm +-> C[0, 0] via ... @ fabric");
    both("    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0] via relu @ bm.egress;\n",
         "a bm.egress stage goes on the store", "a bm.egress stage goes on the store");
    // The add's vector must be resident at the merge.
    CHECK_THAT(error_of(head + "  acc C[0, 0] in l2 {\n    resident A[0, :], B[:, 0];\n"
                        "    call gemm(A[0, 0], B[0, 0]) +-> C[0, 0] via add(b[0]) @ combiner;\n"
                        "    release A[0, :], B[:, 0];\n  }\n  store C[0, 0];\n}\n", false),
               ContainsSubstring("add(b[0,0]) @ combiner reads b[0,0], which is not resident"));
    // The generator: no fabric epilogue under ws or is, and no bm.egress under is.
    gen::Options o;
    o.algo = "linear";
    o.size = 64;
    o.tile = 16;
    o.l3 = 12;
    o.dataflow = "ws";
    o.place = "fabric";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("a fabric epilogue would act on one partial"));
    o.dataflow = "is";
    o.place = "bm.egress";
    CHECK_THROWS_WITH(gen::generate(o), ContainsSubstring("b[j] varies by tile"));
}

TEST_CASE("L-CA: a vector tile resident for one result's context, then released (#434)", "[program][csp][run]") {
    // b[j] is made resident for each C tile's store, read by its add on the Combiner, and released
    // right after. The window hands the next tile's store over before this Release is due: the
    // Release waits for the reads scheduled before it only, never for a later residency's, which
    // wait for this Release to load b[j] again.
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 12) {\n"
                            "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                            "  vector b[32] tile 16 in;\n  tensor C[32,32] tile 16x16 out;\n"
                            "  for i in 0..2 {\n    resident A[i, :];\n    for j in 0..2 {\n      resident B[:, j];\n"
                            "      acc C[i, j] in fabric { for k in 0..2 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; } }\n"
                            "      resident b[j];\n      store C[i, j] via add(b[j]) @ combiner, relu @ combiner;\n"
                            "      release b[j];\n      release B[:, j];\n    }\n    release A[i, :];\n  }\n}\n";
    const auto d = s1_with_combiner();
    const lang::Program ast = lang::parse(src);
    driver::CspRunRequest req{ast, &d, driver::csp_inputs(ast),
                              {ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
    const auto r = driver::run_csp(req);
    REQUIRE(r.reference);
    for (const auto& o : r.levels) {
        CAPTURE(to_string(o.level));
        INFO(o.skipped.value_or(""));
        REQUIRE_FALSE(o.skipped);
        CHECK(same_bits(o.values.operand("C").values, r.reference->operand("C").values));
    }
}

TEST_CASE("L-CA: a tile context's add waits for its vector's Load to land", "[program][csp][run]") {
    // #435's review: nothing ordered a drain after the Load of the vector its stage reads. With a
    // slow memory, C's gemm is done long before b[j] -- made resident just before the store --
    // has landed in L3, and the drain read the previous residency's b, or none. The drain now
    // waits for the Loads of b scheduled before it.
    // A's tiles 2..5 are filler the gemm does not read: their Loads queue ahead of b[j]'s on the
    // DMA, so b[j] lands well after C's gemm is done.
    const std::string src = "csp 1.0\nprogram t machine flat(l3 = 16) {\n"
                            "  tensor A[32,96] tile 16x16 in;\n  tensor B[96,32] tile 16x16 in;\n"
                            "  vector b[32] tile 16 in;\n  tensor C[32,32] tile 16x16 out;\n"
                            "  for i in 0..2 {\n    for j in 0..2 {\n"
                            "      resident A[i, 0], A[i, 1], B[0, j], B[1, j];\n"
                            "      acc C[i, j] in fabric { for k in 0..2 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; } }\n"
                            "      resident A[i, 2], A[i, 3], A[i, 4], A[i, 5];\n"
                            "      resident b[j];\n      store C[i, j] via add(b[j]) @ combiner;\n"
                            "      release b[j];\n      release A[i, 2], A[i, 3], A[i, 4], A[i, 5];\n"
                            "      release A[i, 0], A[i, 1], B[0, j], B[1, j];\n    }\n  }\n}\n";
    const auto d = s1_with_combiner();
    const lang::Program ast = lang::parse(src);
    const TileProgram inputs = driver::csp_inputs(ast);
    const auto ref = driver::csp_reference(ast, inputs, 1'000'000);
    REQUIRE(ref.values);
    auto cfg = sw::kpu::timing::csp_config_from(d);
    REQUIRE(cfg);
    cfg->config.mc_bandwidth_gbps = 0.25;                // a slow memory: every Load takes long
    cfg->config.mc_startup_latency = 400;
    sw::kpu::timing::ConcurrentTimingExecutor exec(cfg->config);
    lang::ActionStream s(ast, lang::target_from(d));
    const auto r = sw::kpu::timing::CspDriver(exec, s, inputs, 256).run();
    REQUIRE(r.completed);
    CHECK(same_bits(r.values.operand("C").values, ref.values->operand("C").values));
}
