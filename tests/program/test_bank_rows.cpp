// ============================================================================
// tests/program/test_bank_rows.cpp
// L-CA's banked L2 in its records (#401 slice 1, second part): each L2 transfer's booking -- its
// rows and the clocks they waited for their banks -- becomes the timed record's row events and L2
// addresses (L2 in the rows form), the leg's bank wait in the L-CA records, and a cause in the
// bubble report.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/bubbles.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/record/timed_record.hpp>

#include <filesystem>
#include <map>
#include <string>

using namespace sw::kpu::program;
using record::DomainFlowRecord;
using driver::ExecutionLevel;
namespace ev = record::ev;
using Rec = csp::TransactionalInterpreter::Record;
using K = csp::Action::Kind;
namespace gen = sw::kpu::program::csp::gen;

namespace {

struct Run {
    DomainFlowRecord lb;
    driver::CspLevelOutcome lca;
};

// S1 with a Combiner, banked (l2.banks_per_tile = 4): a ws matmul, whose merges compete for the
// read ports, recorded at L-B and at L-CA with its events.
Run run() {
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 32;
    o.dataflow = "ws";
    const std::string text = gen::generate(o);
    const csp::lang::Program ast = csp::lang::parse(text);
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1_combiner.json").device(0);
    const TileProgram in = driver::csp_inputs(ast);
    record::DomainFlowRecorder r(ast, text, 0);
    const auto ref = driver::csp_reference(ast, in, 1'000'000, &r);
    REQUIRE(ref.values);
    Run out;
    out.lb = r.finish(&*ref.values);
    out.lca = driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &d, in, 256, true);
    REQUIRE_FALSE(out.lca.skipped);
    REQUIRE(out.lca.l2_geometry);
    REQUIRE_FALSE(out.lca.l2_bookings.empty());
    return out;
}

}  // namespace

TEST_CASE("Bank rows: every booking is a leg's bank wait", "[program][record][timed][banks]") {
    const Run x = run();
    std::uint64_t waited = 0;
    for (const Rec& r : x.lca.records) waited += r.bank_wait;
    REQUIRE(x.lca.l2_banks);
    // Every booking matched one leg: the legs' waits are the calendar's stalls.
    CHECK(waited == x.lca.l2_banks->stalls_total());
}

TEST_CASE("Bank rows: a banked L-CA record has L2 by rows, and its transfers' row events", "[program][record][timed][banks]") {
    const Run x = run();
    const auto r = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings,
                                        &*x.lca.l2_geometry);
    REQUIRE(r.l2_rows);
    CHECK(r.l2_rows->banks == x.lca.l2_geometry->banks);
    CHECK(r.l2_rows->row_bytes == x.lca.l2_geometry->row_bytes);
    std::size_t rows = 0, under = 0;
    for (const auto& e : r.events) {
        if (e.kind < ev::BmRow) continue;
        ++rows;
        REQUIRE(e.parent < r.events.size());
        const auto& p = r.events[e.parent];
        const std::uint8_t want = e.kind == ev::BmRow ? ev::BlockMove : e.kind == ev::StrRow ? ev::Stream : ev::Drain;
        CHECK(p.kind == want);
        CHECK(p.action == e.action);
        CHECK(p.tile == e.tile);
        CHECK(p.cycle <= e.cycle);
        CHECK(e.end <= p.end);
        // Its L2 address is a row: its transfer's first row, plus its own.
        const std::uint64_t mine = e.src == ev::L2 ? e.src_addr : e.dst_addr;
        const std::uint64_t first = p.src == ev::L2 ? p.src_addr : p.dst_addr;
        CHECK(mine == first + e.row);
        CHECK(mine < r.l2_rows->rows);
        ++under;
    }
    CHECK(rows > 0);
    CHECK(under == rows);
    // Round trip: the rows form survives the bundle.
    const std::string dir = (std::filesystem::temp_directory_path() / "kpu_bank_rows_rt.dflow").string();
    std::filesystem::remove_all(dir);
    record::write_dflow(r, dir);
    const DomainFlowRecord back = record::read_dflow(dir);
    std::filesystem::remove_all(dir);
    REQUIRE(back.l2_rows);
    CHECK(back.l2_rows->rows == r.l2_rows->rows);
    CHECK(back.events.size() == r.events.size());
}

TEST_CASE("Bank rows: a banked record keeps an unstaged feed's credit", "[program][record][timed][banks]") {
    // An unstaged copy's L2 rows are free once it is fed: its Feed returns the L2 credit. A banked
    // L2 gives every transfer an address (its first row), so being staged is the step's own flag,
    // not the address: the banked record has as many of these credits as the slotted one.
    const Run x = run();
    const auto banked = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings,
                                             &*x.lca.l2_geometry);
    const auto slotted = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values);
    auto feed_credits = [](const DomainFlowRecord& r) {
        std::size_t n = 0;
        for (const auto& e : r.events)
            n += e.kind == ev::Credit && e.src == ev::L2 && e.process == ev::PStr &&
                 static_cast<K>(r.steps.at(e.action).kind) == K::Feed;
        return n;
    };
    CHECK(feed_credits(slotted) > 0);
    CHECK(feed_credits(banked) == feed_credits(slotted));
}

TEST_CASE("Bank rows: a feed whose rows waited for their bank is the bubble's cause", "[program][csp][bubbles][banks]") {
    // Two calls on one compute lane; the second waits for its operand's Feed, which started when
    // it was ready but ran long because its rows waited for their bank's ports.
    auto rec = [](std::size_t action, K kind, csp::TransactionalInterpreter::Proc proc, const char* op,
                  std::uint64_t start, std::uint64_t finish) {
        Rec r;
        r.action = action;
        r.kind = kind;
        r.tile = TileCoord{op, 0, 0};
        r.proc = proc;
        r.start = r.ready = start;
        r.finish = finish;
        return r;
    };
    using P = csp::TransactionalInterpreter::Proc;
    std::vector<Rec> rs = {rec(0, K::Move, P::Bm, "A", 0, 4),  rec(1, K::Feed, P::Str, "A", 4, 8),
                           rec(2, K::Call, P::Cf, "C", 8, 18), rec(3, K::Move, P::Bm, "B", 4, 8),
                           rec(4, K::Feed, P::Str, "B", 8, 40), rec(5, K::Call, P::Cf, "D", 40, 50)};
    rs[4].bank_wait = 24;                   // the Feed of B: its rows waited 24 clocks for their bank
    std::map<std::string, std::uint64_t> busy{{"bm", 8}, {"str", 36}, {"cf", 20}};
    std::map<std::string, std::size_t> lanes{{"dma", 1}, {"bm", 1}, {"str", 1}, {"cf", 1}};
    const auto rep = csp::bubble_report(rs, 50, busy, lanes);
    using Cause = csp::BubbleReport::Cause;
    bool found = false;
    for (const auto& b : rep.bubbles)
        if (b.call == 5) {
            found = true;
            CHECK(b.cause == Cause::L2Bank);
        }
    CHECK(found);
    CHECK(std::string(csp::BubbleReport::to_string(Cause::L2Bank)) == "L2 bank ports");
}
