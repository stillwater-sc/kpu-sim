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
    const auto lca = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values);
    const std::string dir = (std::filesystem::temp_directory_path() / "kpu_timed_record_rt.dflow").string();
    std::filesystem::remove_all(dir);
    record::write_dflow(lca, dir);
    const DomainFlowRecord back = record::read_dflow(dir);
    std::filesystem::remove_all(dir);
    CHECK(back.level == "L-CA");
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
