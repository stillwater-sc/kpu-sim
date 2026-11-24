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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/bubbles.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/domain_flow_record.hpp>
#include <sw/kpu/program/record/timed_record.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
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
    // A wait delays its leg at most by itself.
    for (const Rec& r : x.lca.records) CHECK(r.bank_delay <= r.bank_wait);
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
        CHECK(p.lane == e.lane);                // its transaction's process instance
        CHECK(e.end <= p.end);
        // Its L2 address is a row: its transfer's first row, plus its own.
        const std::uint64_t mine = e.src == ev::L2 ? e.src_addr : e.dst_addr;
        const std::uint64_t first = p.src == ev::L2 ? p.src_addr : p.dst_addr;
        CHECK(mine == x.lca.l2_geometry->row(first, e.row));
        CHECK(mine < r.l2_rows->rows);
        ++under;
    }
    CHECK(rows > 0);
    CHECK(under == rows);
    // Round trip: the rows form survives the bundle.
    struct Scratch {
        std::filesystem::path p;
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove_all(p, ec);
        }
    } tmp{std::filesystem::temp_directory_path() /
          ("kpu_bank_rows_rt_" + std::to_string(std::random_device{}()) + "_" +
           std::to_string(reinterpret_cast<std::uintptr_t>(&r)) + ".dflow")};
    const std::string dir = tmp.p.string();
    record::write_dflow(r, dir);
    const DomainFlowRecord back = record::read_dflow(dir);
    REQUIRE(back.l2_rows);
    CHECK(back.l2_rows->rows == r.l2_rows->rows);
    CHECK(back.events.size() == r.events.size());
}

TEST_CASE("Bank rows: a banked record keeps every L2 credit", "[program][record][timed][banks]") {
    // An L2 slot's credit returns when its copy is done with: at the L2Release of a copy the level
    // placed (a staging, and since #387 every copy the walker places), at the Feed of one it did
    // not. A banked L2 gives every transfer an address (its first row), so which of the two it is
    // is the step's own flag, not the address: the banked record has as many credits as the slotted
    // one, and both name the same steps.
    const Run x = run();
    const auto banked = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings,
                                             &*x.lca.l2_geometry);
    const auto slotted = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values);
    auto l2_credits = [](const DomainFlowRecord& r, K at) {
        std::size_t n = 0;
        for (const auto& e : r.events)
            n += e.kind == ev::Credit && e.src == ev::L2 && static_cast<K>(r.steps.at(e.action).kind) == at;
        return n;
    };
    CHECK(l2_credits(slotted, K::L2Release) > 0);
    CHECK(l2_credits(slotted, K::Feed) == 0);       // L-CA places every copy
    CHECK(l2_credits(banked, K::L2Release) == l2_credits(slotted, K::L2Release));
    CHECK(l2_credits(banked, K::Feed) == l2_credits(slotted, K::Feed));
    // Recorded unplaced (as L-T1 runs copies on more than one compute tile), a copy's credit is its
    // Feed's again.
    const auto unplaced = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, nullptr, nullptr, false);
    CHECK(l2_credits(unplaced, K::Feed) > 0);
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
    rs[4].bank_wait = rs[4].bank_delay = 24;    // the Feed of B: its rows waited 24 clocks, all added to it
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

TEST_CASE("Bank rows: a fixed-size L2 wraps a transfer's rows, walking the banks in order", "[program][timing][banks]") {
    sw::kpu::timing::L2Banks::Geometry g;
    g.banks = 4;
    g.rows_per_bank = 1;                    // four rows
    CHECK(g.row(3, 0) == 3);
    CHECK(g.row(3, 1) == 0);
    CHECK(g.row(3, 2) == 1);
    g.rows_per_bank = 0;                    // as many as the slots need: no wrap
    CHECK(g.row(3, 2) == 5);
}

TEST_CASE("Bank rows: a staged copy's L2 credit names its rows", "[program][record][timed][banks]") {
    // An L2Release frees a staged copy: the banked record's credit names the copy's first row, as
    // the slotted one names its slot.
    const Run x = run();
    const auto banked = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings,
                                             &*x.lca.l2_geometry);
    const auto slotted = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values);
    auto releases = [](const DomainFlowRecord& r, bool addressed) {
        std::size_t n = 0;
        for (const auto& e : r.events)
            if (e.kind == ev::Credit && e.src == ev::L2 && static_cast<K>(r.steps.at(e.action).kind) == K::L2Release)
                n += !addressed || e.src_addr != record::kNone64;
        return n;
    };
    CHECK(releases(slotted, false) > 0);
    CHECK(releases(banked, true) == releases(slotted, true));
    CHECK(releases(banked, true) == releases(banked, false));
}

TEST_CASE("Bank rows: a result ahead whose rows waited for their banks is the bubble's cause", "[program][csp][bubbles][banks]") {
    // A ready Feed held on the streamer behind a Drain whose rows waited for their bank: the
    // result's way out ran long, at the L2 bank ports.
    using P = csp::TransactionalInterpreter::Proc;
    auto rec = [](std::size_t action, K kind, P proc, const char* op, std::uint64_t ready, std::uint64_t start,
                  std::uint64_t finish) {
        Rec r;
        r.action = action;
        r.kind = kind;
        r.tile = TileCoord{op, 0, 0};
        r.proc = proc;
        r.ready = ready;
        r.start = start;
        r.finish = finish;
        return r;
    };
    std::vector<Rec> rs = {rec(0, K::Move, P::Bm, "A", 0, 0, 4),   rec(1, K::Feed, P::Str, "A", 4, 4, 8),
                           rec(2, K::Call, P::Cf, "C", 8, 8, 18),  rec(3, K::Drain, P::Str, "C", 18, 18, 60),
                           rec(4, K::Move, P::Bm, "B", 4, 4, 8),   rec(5, K::Feed, P::Str, "B", 8, 60, 64),
                           rec(6, K::Call, P::Cf, "D", 64, 64, 74)};
    rs[3].bank_wait = rs[3].bank_delay = 30;    // the Drain's rows waited for their bank, delaying it
    std::map<std::string, std::uint64_t> busy{{"bm", 8}, {"str", 50}, {"cf", 20}};
    std::map<std::string, std::size_t> lanes{{"dma", 1}, {"bm", 1}, {"str", 2}, {"cf", 1}};
    using Cause = csp::BubbleReport::Cause;
    auto cause_of = [&](const std::vector<Rec>& v) {
        const auto rep = csp::bubble_report(v, 74, busy, lanes);
        for (const auto& b : rep.bubbles)
            if (b.call == 6) return b.cause;
        return Cause::Unknown;
    };
    CHECK(cause_of(rs) == Cause::L2Bank);
    rs[3].bank_delay = 0;                   // its rows waited, but the Combiner's time hid it: the result ahead
    CHECK(cause_of(rs) == Cause::Result);
    rs[3].bank_wait = 0;                    // no bank wait at all: the result ahead
    CHECK(cause_of(rs) == Cause::Result);
}

TEST_CASE("Bank rows: a stall the Combiner's time hides does not delay the drain", "[program][timing][banks]") {
    using sw::kpu::timing::L2Banks;
    L2Banks::Geometry g;
    g.banks = 1;
    g.read_latency = 0;
    L2Banks b(g, 1);
    b.keep_log(true);
    // Hold the one bank's write port at clock 0, so a drain's first row stalls one clock.
    b.book(0, 0, g.row_bytes, L2Banks::Access::Write, L2Banks::Client::BlockMover, 0, 1, 0);
    const auto move = b.book(0, 1, 2 * g.row_bytes, L2Banks::Access::Write, L2Banks::Client::Drain, 0, 1, 0);
    REQUIRE(b.log().back().stalls == 1);
    CHECK(b.log().back().delay == 1);          // alone, the stall lengthens the transfer by its clock
    b.overlap_last(move, move + 10);           // a Combiner longer than the transfer hides it
    CHECK(b.log().back().delay == 0);
    const auto again = b.book(0, 2, 2 * g.row_bytes, L2Banks::Access::Write, L2Banks::Client::Drain, 0, 1, 0);
    const auto stalled = b.log().back().delay;
    REQUIRE(stalled > 0);
    b.overlap_last(again, again - 1);          // a Combiner one clock shorter: one clock of it shows
    CHECK(b.log().back().delay == std::min<sw::kpu::timing::Cycle>(stalled, 1));
}

TEST_CASE("Bank rows: the banked form follows the geometry, not the bookings", "[program][record][timed][banks]") {
    const Run x = run();
    // A banked run with no booking (no L2 transfer) still declares its L2 by rows, with no row events.
    const auto r = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, nullptr, &*x.lca.l2_geometry);
    REQUIRE(r.l2_rows);
    for (const auto& e : r.events) CHECK(e.kind < ev::BmRow);
    // An L2 whose rows the record cannot count is refused, not written as rows: 0.
    auto big = *x.lca.l2_geometry;
    big.banks = 4;
    big.rows_per_bank = 1u << 30;
    CHECK_THROWS_AS(record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings, &big),
                    record::RecordError);
}

TEST_CASE("Bank rows: a record whose L2 is declared both ways is refused", "[program][record][timed][banks]") {
    const Run x = run();
    const auto r = record::timed_record(x.lb, "L-CA", x.lca.records, x.lca.values, 0, 0, &x.lca.l2_bookings,
                                        &*x.lca.l2_geometry);
    struct Scratch {
        std::filesystem::path p;
        ~Scratch() {
            std::error_code ec;
            std::filesystem::remove_all(p, ec);
        }
    } tmp{std::filesystem::temp_directory_path() /
          ("kpu_bank_rows_mixed_" + std::to_string(std::random_device{}()) + ".dflow")};
    const std::string dir = tmp.p.string();
    record::write_dflow(r, dir);
    nlohmann::json m;
    {
        std::ifstream in(dir + "/manifest.json");
        in >> m;
    }
    m["geometry"]["l2"]["slots"] = 4;
    {
        std::ofstream out(dir + "/manifest.json");
        out << m.dump();
    }
    CHECK_THROWS_WITH(record::read_dflow(dir), Catch::Matchers::ContainsSubstring("declares both slots and rows"));
    // A rows-form L2 with a zero dimension is refused too.
    m["geometry"]["l2"].erase("slots");
    m["geometry"]["l2"]["banks"] = 0;
    {
        std::ofstream out(dir + "/manifest.json");
        out << m.dump();
    }
    CHECK_THROWS_WITH(record::read_dflow(dir), Catch::Matchers::ContainsSubstring("must be positive"));
    // Out of 32 bits either way: refused, not narrowed.
    for (const std::int64_t bad : {std::int64_t(-1), std::int64_t(4294967297)}) {
        m["geometry"]["l2"]["banks"] = 4;
        m["geometry"]["l2"]["rows"] = bad;
        {
            std::ofstream out(dir + "/manifest.json");
            out << m.dump();
            out.close();
            REQUIRE(out);                   // the bad row count reached the file
        }
        CHECK_THROWS_WITH(record::read_dflow(dir), Catch::Matchers::ContainsSubstring("geometry.l2.rows"));
    }
}

TEST_CASE("Bank rows: a result ahead that finished before the operand was ready is not its bank cause", "[program][csp][bubbles][banks]") {
    // A Move ready at 25 waits for a BlockMover lane until 35. Its predecessor on the process, a
    // bank-delayed Writeback on the other lane, finished at 20: its stalls cannot have held the Move.
    using P = csp::TransactionalInterpreter::Proc;
    auto rec = [](std::size_t action, K kind, P proc, const char* op, std::uint64_t ready, std::uint64_t start,
                  std::uint64_t finish, std::uint32_t lane) {
        Rec r;
        r.action = action;
        r.kind = kind;
        r.tile = TileCoord{op, 0, 0};
        r.proc = proc;
        r.ready = ready;
        r.start = start;
        r.finish = finish;
        r.lane = lane;
        return r;
    };
    // A first call opens the compute lane's span, so the second's wait is a bubble.
    std::vector<Rec> rs = {rec(0, K::Call, P::Cf, "E", 0, 0, 10, 0),       rec(1, K::Writeback, P::Bm, "C", 0, 0, 20, 1),
                           rec(2, K::Move, P::Bm, "A", 25, 35, 40, 0),     rec(3, K::Feed, P::Str, "A", 40, 40, 44, 0),
                           rec(4, K::Call, P::Cf, "D", 44, 44, 54, 0)};
    rs[1].bank_wait = rs[1].bank_delay = 6;
    std::map<std::string, std::uint64_t> busy{{"bm", 25}, {"str", 4}, {"cf", 20}};
    std::map<std::string, std::size_t> lanes{{"dma", 1}, {"bm", 2}, {"str", 1}, {"cf", 1}};
    using Cause = csp::BubbleReport::Cause;
    auto cause_of = [&](const std::vector<Rec>& v) {
        for (const auto& b : csp::bubble_report(v, 54, busy, lanes).bubbles)
            if (b.call == 4) return b.cause;
        return Cause::Unknown;
    };
    CHECK(cause_of(rs) == Cause::Result);
    rs[1].finish = 30;                      // still writing back when the Move was ready, but the Move
    CHECK(cause_of(rs) == Cause::Result);   // started at 35, on another lane: something else held it
    rs[1].finish = 35;                      // finished as the Move started, but on the other lane:
    CHECK(cause_of(rs) == Cause::Result);   // the Move did not take its lane
    rs[1].lane = 0;                         // the same lane: the Move took the lane it freed
    CHECK(cause_of(rs) == Cause::L2Bank);
}

TEST_CASE("Bank rows: an egress unit's time hides a writeback's bank stalls", "[program][record][timed][banks]") {
    // With the egress unit's work longer than the transfer, a writeback's stalls add nothing to its
    // finish (the BlockMover calls overlap_last as the drain does). In the run, every Writeback's
    // bank delay is at most its wait; on the calendar, the unit's longer time hides the stall.
    const Run x = run();
    for (const Rec& r : x.lca.records)
        if (r.kind == K::Writeback) CHECK(r.bank_delay <= r.bank_wait);
    sw::kpu::timing::L2Banks::Geometry g;
    g.banks = 1;
    g.read_latency = 0;
    sw::kpu::timing::L2Banks b(g, 1);
    b.keep_log(true);
    b.book(0, 0, g.row_bytes, sw::kpu::timing::L2Banks::Access::Read, sw::kpu::timing::L2Banks::Client::Streamer, 0, 1, 0);
    b.book(0, 0, g.row_bytes, sw::kpu::timing::L2Banks::Access::Read, sw::kpu::timing::L2Banks::Client::Streamer, 0, 1, 0);
    const auto move = b.book(0, 0, 2 * g.row_bytes, sw::kpu::timing::L2Banks::Access::Read,
                             sw::kpu::timing::L2Banks::Client::BlockMover, 0, 1, 0);
    REQUIRE(b.log().back().delay > 0);       // both read ports taken at clock 0: its first row waits
    b.overlap_last(move, move + 5);
    CHECK(b.log().back().delay == 0);
}

TEST_CASE("Bank rows: a move whose writeback ran long at the banks names them", "[program][csp][bubbles][banks]") {
    // A Move of a tile a Writeback just put back in L3: the Move's data waited for the Writeback,
    // whose rows waited for their banks.
    using P = csp::TransactionalInterpreter::Proc;
    auto rec = [](std::size_t action, K kind, P proc, const char* op, std::uint64_t ready, std::uint64_t start,
                  std::uint64_t finish) {
        Rec r;
        r.action = action;
        r.kind = kind;
        r.tile = TileCoord{op, 0, 0};
        r.proc = proc;
        r.ready = ready;
        r.start = start;
        r.finish = finish;
        return r;
    };
    std::vector<Rec> rs = {rec(0, K::Call, P::Cf, "E", 0, 0, 10),       rec(1, K::Writeback, P::Bm, "C", 0, 0, 30),
                           rec(2, K::Move, P::Bm, "C", 30, 30, 34),     rec(3, K::Feed, P::Str, "C", 34, 34, 38),
                           rec(4, K::Call, P::Cf, "D", 38, 38, 48)};
    std::map<std::string, std::uint64_t> busy{{"bm", 34}, {"str", 4}, {"cf", 20}};
    std::map<std::string, std::size_t> lanes{{"dma", 1}, {"bm", 1}, {"str", 1}, {"cf", 1}};
    using Cause = csp::BubbleReport::Cause;
    auto cause_of = [&](const std::vector<Rec>& v) {
        for (const auto& b : csp::bubble_report(v, 48, busy, lanes).bubbles)
            if (b.call == 4) return b.cause;
        return Cause::Unknown;
    };
    CHECK(cause_of(rs) == Cause::Result);
    rs[1].bank_wait = rs[1].bank_delay = 8;
    CHECK(cause_of(rs) == Cause::L2Bank);
    // L-CA's one-tick convention: a Move that starts the cycle after its Writeback finished has
    // ready = start, one past the finish. The Writeback still set it.
    rs[2].ready = rs[2].start = 31;
    rs[2].finish = 35;
    CHECK(cause_of(rs) == Cause::L2Bank);
    // A later credit set the Move's ready (at L-T1 it can do so without an l2_wait, when a lane
    // was as late): the slot, not the banks.
    rs[2].ready_by = Rec::ReadyBy::Credit;
    CHECK(cause_of(rs) == Cause::L2Slot);
    // Its handover set it (the program's issue order): the stalls did not delay it.
    rs[2].ready_by = Rec::ReadyBy::Handover;
    rs[2].ready = rs[2].start = 32;         // ready, and started, at its handover
    rs[2].finish = 36;
    rs[3].ready = rs[3].start = 36;
    rs[3].finish = 40;
    rs[4].ready = rs[4].start = 40;
    rs[4].finish = 50;
    CHECK(cause_of(rs) == Cause::Issue);
}

TEST_CASE("Bank rows: each level records what set a step's ready", "[program][csp][bubbles][banks]") {
    // L-T1: a staged Move made ready by its L2 slot's credit says so. L-CA: a step's handover or a
    // credit stall later than its inputs says so; the rest are their inputs'.
    gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 32;
    o.dataflow = "ws";
    const std::string text = gen::generate(o);
    const csp::lang::Program ast = csp::lang::parse(text);
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1_combiner.json").device(0);
    const auto lt1 = driver::csp_run_level(ExecutionLevel::BlockSequential, ast, &d, driver::csp_inputs(ast), 256, true);
    REQUIRE_FALSE(lt1.skipped);
    std::size_t credit = 0;
    for (const Rec& r : lt1.records) credit += r.kind == K::Move && r.ready_by == Rec::ReadyBy::Credit;
    CHECK(credit > 0);
    const Run x = run();
    std::map<Rec::ReadyBy, std::size_t> by;
    for (const Rec& r : x.lca.records) ++by[r.ready_by];
    CHECK(by[Rec::ReadyBy::Input] > 0);
    CHECK(by[Rec::ReadyBy::Handover] + by[Rec::ReadyBy::Credit] > 0);
}
