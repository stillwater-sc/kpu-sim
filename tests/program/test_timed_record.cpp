// ============================================================================
// tests/program/test_timed_record.cpp
// A timed level's own domain-flow record (#402; docs/plans/shared-tile-flow-record.md, 6.2): L-B's
// steps, the level's events with their cycles, and its final values. L-T1's events are its legs;
// L-CA's are its protocol events matched back to the actions (timing/csp_records.hpp), with each
// Release at the cycle its credit returned, not when the driver handed it over.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/record/timed_record.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using record::DomainFlowRecord;
using Catch::Matchers::ContainsSubstring;
using driver::ExecutionLevel;
namespace ev = record::ev;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

struct Runs {
    DomainFlowRecord lb;
    driver::CspLevelOutcome lt1, lca;
};

// The corpus's fused linear on S1: L-B recorded, L-T1, and L-CA with its events kept.
Runs runs(std::uint64_t limit = 0) {
    const std::string text = slurp("tests/program/corpus/linear_64_t16_relu.csp");
    const csp::lang::Program ast = csp::lang::parse(text);
    const auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    const TileProgram in = driver::csp_inputs(ast);
    record::DomainFlowRecorder r(ast, text, limit);
    const auto ref = driver::csp_reference(ast, in, 1'000'000, &r);
    REQUIRE(ref.values);
    Runs out;
    out.lb = r.finish(&*ref.values);
    out.lt1 = driver::csp_run_level(ExecutionLevel::BlockSequential, ast, &d, in);
    out.lca = driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &d, in, 256, true);
    REQUIRE_FALSE(out.lt1.skipped);
    REQUIRE_FALSE(out.lca.skipped);
    return out;
}

bool data_kind(csp::Action::Kind k) {
    using K = csp::Action::Kind;
    return k == K::Load || k == K::Store || k == K::Move || k == K::Writeback || k == K::Feed || k == K::Call ||
           k == K::Drain;
}

void check_record(const DomainFlowRecord& r, const driver::CspLevelOutcome& o, const std::string& level) {
    CAPTURE(level);
    CHECK(r.level == level);
    CHECK(r.timed());
    CHECK_FALSE(r.lt1);
    REQUIRE_FALSE(r.events.empty());
    // Every data step has exactly one transaction, joined to it; nothing else has one.
    std::map<std::uint64_t, int> tx;
    std::uint64_t latest = 0;
    for (std::size_t i = 0; i < r.events.size(); ++i) {
        const auto& e = r.events[i];
        CHECK(e.cycle <= e.end);
        CHECK(e.row == record::kNone32);                    // no row events
        CHECK(e.kind <= ev::Token);
        if (i) CHECK(r.events[i - 1].cycle <= e.cycle);     // in cycle order
        REQUIRE(e.action < r.steps.size());
        CHECK(e.tile == r.steps[e.action].tile);
        if (e.kind <= ev::Drain) ++tx[e.action];
        latest = std::max(latest, e.end);
    }
    CHECK(r.cycles == latest);
    // A token or credit a process sends is on the lane of its action's transaction (#469's review).
    std::map<std::uint64_t, std::uint32_t> lane_of;
    for (const auto& e : r.events)
        if (e.kind <= ev::Drain) lane_of[e.action] = e.lane;
    for (const auto& e : r.events)
        if ((e.kind == ev::Credit || e.kind == ev::Token) && e.process != record::kNone8) {
            CAPTURE(e.action);
            CHECK(e.lane == lane_of.at(e.action));
        }
    for (const auto& st : r.steps) {
        const auto k = static_cast<csp::Action::Kind>(st.kind);
        CAPTURE(st.action, csp::to_string(k));
        CHECK(tx[st.action] == (data_kind(k) ? 1 : 0));
    }
    // The values are the level's own finals, word for word.
    std::size_t finals = 0;
    for (const auto& s : r.snapshots) {
        CHECK(s.step == record::kNone64);
        const auto& t = r.tiles.at(s.tile);
        const TensorOperand& op = o.values.operand(r.operands.at(t.operand).name);
        std::size_t n = s.first;
        for (Dim rr = op.row_begin(t.ti); rr < op.row_end(t.ti); ++rr)
            for (Dim cc = op.col_begin(t.tj); cc < op.col_end(t.tj); ++cc) {
                const numeric::Word w = op.word(rr, cc);
                numeric::Word got = 0;
                std::memcpy(&got, &r.elements.at(n++), sizeof got);
                if (got != w) FAIL("a final element differs from " << level << "'s value");
            }
        ++finals;
    }
    CHECK(finals > 0);
}

}  // namespace

TEST_CASE("Timed record: L-T1's and L-CA's events join every data step, with the level's values",
          "[program][record][timed]") {
    const Runs x = runs();
    const auto lt1 = record::timed_record(x.lb, "L-T1", x.lt1.records, x.lt1.values, x.lt1.pe_rows, x.lt1.pe_cols);
    const auto lca = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values);
    check_record(lt1, x.lt1, "L-T1");
    check_record(lca, x.lca, "L-CA");
    // The steps are the program's, the same at every level.
    REQUIRE(lt1.steps.size() == x.lb.steps.size());
    for (std::size_t i = 0; i < lt1.steps.size(); ++i) CHECK(lt1.steps[i].tile == x.lb.steps[i].tile);
    // L-T1's cycles to result are its makespan.
    CHECK(lt1.cycles == x.lt1.makespan);
    // The record says what it does not model.
    CHECK_THAT(lca.level_notes.front(), ContainsSubstring("row progress"));
}

TEST_CASE("Timed record: it round-trips through the bundle", "[program][record][timed]") {
    const Runs x = runs();
    // With the array's edge, which only geometry.l1 carries in a timed record (#469's review).
    REQUIRE(x.lt1.pe_rows > 0);
    const auto lca = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, x.lt1.pe_rows, x.lt1.pe_cols);
    const std::string dir = (std::filesystem::temp_directory_path() / "kpu_timed_record_rt.dflow").string();
    std::filesystem::remove_all(dir);
    record::write_dflow(lca, dir);
    const DomainFlowRecord back = record::read_dflow(dir);
    std::filesystem::remove_all(dir);
    CHECK(back.level == "L-CA");
    CHECK(back.fabric);
    CHECK(back.pe_rows == lca.pe_rows);
    CHECK(back.pe_cols == lca.pe_cols);
    CHECK(back.cycles == lca.cycles);
    CHECK(back.level_notes == lca.level_notes);
    REQUIRE(back.events.size() == lca.events.size());
    for (std::size_t i = 0; i < back.events.size(); ++i) {
        const auto &a = back.events[i], &b = lca.events[i];
        CHECK(std::tie(a.kind, a.process, a.src, a.dst, a.src_addr, a.dst_addr, a.tile, a.action, a.where, a.lane,
                       a.cycle, a.end) ==
              std::tie(b.kind, b.process, b.src, b.dst, b.src_addr, b.dst_addr, b.tile, b.action, b.where, b.lane,
                       b.cycle, b.end));
    }
}

TEST_CASE("Timed record: L-CA's Release is at the cycle its credit returned", "[program][record][timed][lca]") {
    // csp_records used to stamp a Release at its handover, before the BlockMover had retired the
    // entry: a credit ahead of its tile's Move out of L3.
    const Runs x = runs();
    std::map<std::string, std::set<std::uint64_t>> credits;      // tile -> its L3 credit returns
    for (const auto& e : x.lca.events)
        if (e.type == sw::kpu::timing::EventType::CREDIT_RELEASED && e.tile && e.detail == "l3")
            credits[e.tile->to_string()].insert(e.cycle);
    std::map<std::string, std::uint64_t> moved_out;              // tile -> its last Move's finish so far
    std::size_t releases = 0;
    std::vector<csp::TransactionalInterpreter::Record> recs = x.lca.records;
    std::stable_sort(recs.begin(), recs.end(), [](const auto& a, const auto& b) { return a.action < b.action; });
    for (const auto& r : recs) {
        const std::string t = r.tile.to_string();
        if (r.kind == csp::Action::Kind::Move) moved_out[t] = std::max(moved_out[t], std::uint64_t(r.finish));
        if (r.kind != csp::Action::Kind::Release) continue;
        CAPTURE(r.action, t);
        ++releases;
        CHECK(credits[t].count(r.start) == 1);
        if (moved_out.count(t)) CHECK(r.start >= moved_out[t]);
    }
    CHECK(releases > 0);
}

TEST_CASE("Timed record: it needs every step of a run L-B finished", "[program][record][timed]") {
    const Runs x = runs(5);
    REQUIRE(x.lb.cut);
    CHECK_THROWS_WITH(record::timed_record(x.lb, "L-T1", x.lt1.records, x.lt1.values),
                      ContainsSubstring("needs every step"));
}

TEST_CASE("Timed record: each Release's credit follows its own tile's last move out", "[program][record][timed]") {
    // C[0, 0] is stored twice: merged in L2 and stored, then read back, put through a ReLU on the
    // Combiner, and stored again (#469's review). Each Release's credit is no earlier than the end
    // of its own Store, the second included, not only the first.
    const std::string text = "csp 1.0\nprogram t machine flat(l3 = 8, l2 = 4) {\n"
                             "  tensor A[32,32] tile 16x16 in;\n  tensor B[32,32] tile 16x16 in;\n"
                             "  tensor C[32,32] tile 16x16 inout;\n"
                             "  acc C[0, 0] in l2 {\n    resident A[0, :], B[:, 0];\n"
                             "    for k in 0..2 { call gemm(A[0, k], B[k, 0]) +-> C[0, 0]; }\n"
                             "    release A[0, :], B[:, 0];\n  }\n  store C[0, 0];\n"
                             "  resident C[0, 0];\n  call relu(C[0, 0]) -> C[0, 0] via relu @ combiner;\n"
                             "  store C[0, 0];\n  release C[0, 0];\n}\n";
    const csp::lang::Program ast = csp::lang::parse(text);
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu"}};
    const TileProgram in = driver::csp_inputs(ast);
    record::DomainFlowRecorder rr(ast, text, 0);
    const auto ref = driver::csp_reference(ast, in, 1'000'000, &rr);
    REQUIRE(ref.values);
    const DomainFlowRecord lb = rr.finish(&*ref.values);
    for (const auto level : {ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}) {
        const auto o = driver::csp_run_level(level, ast, &d, in, 256, true);
        REQUIRE_FALSE(o.skipped);
        const std::string name = level == ExecutionLevel::CycleAccurate ? "L-CA" : "L-T1";
        CAPTURE(name);
        const auto r = record::timed_record(lb, name, o.records, o.values);
        // C's transactions out of L3, and its L3 credits, in program order.
        std::uint32_t c = record::kNone32;
        for (std::uint32_t t = 0; t < r.tiles.size(); ++t)
            if (r.tile_name(t) == "C[0,0]") c = t;
        REQUIRE(c != record::kNone32);
        std::vector<const DomainFlowRecord::Event*> mine;
        for (const auto& e : r.events)
            if (e.tile == c && ((e.kind <= ev::Drain && e.src == ev::L3) || (e.kind == ev::Credit && e.src == ev::L3)))
                mine.push_back(&e);
        std::stable_sort(mine.begin(), mine.end(), [](auto* a, auto* b) { return a->action < b->action; });
        std::size_t stores = 0, credits = 0;
        std::uint64_t last_out = 0;
        for (const auto* e : mine) {
            if (e->kind == ev::Credit) {
                ++credits;
                CHECK(e->cycle >= last_out);
            } else {
                stores += e->kind == ev::Dma;
                last_out = e->end;
            }
        }
        CHECK(stores == 2);
        CHECK(credits >= 1);
    }
}

TEST_CASE("Timed record: a Release uses up its tile's last move out", "[program][record][timed]") {
    // One tile, two residencies: the first moved out of L3 and released, the second only read in
    // place (as the Combiner reads an init) and released (#469's review). The second Release has
    // no transaction to follow, so its credit is left out and counted, not stamped after the first.
    using K = csp::Action::Kind;
    DomainFlowRecord lb;
    lb.tiles.push_back({0, 0, 0});
    lb.l3 = 4;
    const K kinds[] = {K::Load, K::Move, K::Release, K::Load, K::Release};
    std::vector<csp::TransactionalInterpreter::Record> recs;
    for (std::uint64_t a = 0; a < 5; ++a) {
        DomainFlowRecord::Step st;
        st.action = a;
        st.kind = static_cast<std::uint8_t>(kinds[a]);
        st.tile = 0;
        st.flags = DomainFlowRecord::kHasPlace;
        st.slot = 0;
        lb.steps.push_back(st);
        csp::TransactionalInterpreter::Record r;
        r.action = a;
        r.kind = kinds[a];
        r.start = 10 * a;
        r.finish = 10 * a + (kinds[a] == K::Release ? 0 : 5);
        recs.push_back(r);
    }
    lb.actions = 5;
    const auto r = record::timed_record(lb, "L-T1", recs, TileProgram{});
    std::size_t credits = 0;
    for (const auto& e : r.events) credits += e.kind == ev::Credit;
    CHECK(credits == 1);                    // the first residency's
    bool counted = false;
    for (const auto& n : r.level_notes) counted = counted || n.rfind("1 credits left out", 0) == 0;
    CHECK(counted);
}
