// ============================================================================
// tests/program/test_csp_bubbles.cpp
// The reuse and bubble report of an L-T1 run (#425): each compute lane's idle time before a
// call, what it waited for, the reuse each operand's program states, and each process's load.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/bubbles.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <map>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace gen = sw::kpu::program::csp::gen;
using Catch::Matchers::ContainsSubstring;
using Cause = BubbleReport::Cause;

namespace {

std::string matmul(const std::string& dataflow, bool stage_a = false) {
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.dataflow = dataflow;
    o.stage_a = stage_a;
    if (stage_a) o.orient = "row";
    return gen::generate(o);
}

// An L-T1 run of `src` on `dev`, and its report.
struct Run {
    TransactionalInterpreter::Stats stats;
    std::vector<TransactionalInterpreter::Record> records;
    BubbleReport report;
};
Run run(const std::string& src, const characterize::DeviceDescriptor& dev) {
    CspProgram p = lang::compile(src);
    const TileProgram in = driver::csp_inputs(lang::parse(src));
    for (const auto& name : p.source.operand_order()) p.source.operand(name).values = in.operand(name).values;
    TransactionalInterpreter lt(dev, p.channel(Chan::L3).capacity, 0);
    lt.begin(p.source);
    for (const Action& a : p.actions) lt.step(a, a.kind == Action::Kind::Call ? &p.source.ops().at(a.l0_op) : nullptr);
    Run r;
    r.stats = lt.finish();
    r.records = lt.records();
    r.report = bubble_report(r.records, r.stats.makespan, r.stats.busy, r.stats.lanes);
    return r;
}

const BubbleReport::Reuse& reuse_of(const BubbleReport& r, const std::string& operand) {
    for (const auto& u : r.reuse)
        if (u.operand == operand) return u;
    FAIL("no reuse row for " + operand);
    return r.reuse.front();
}

Cause dominant(const BubbleReport& r) {
    Cause c = Cause::Unknown;
    BubbleReport::Cycle most = 0;
    for (const auto& [k, n] : r.idle_by_cause)
        if (n > most) {
            most = n;
            c = k;
        }
    return c;
}

}  // namespace

TEST_CASE("Bubbles: the reuse each operand's program states", "[program][csp][bubbles]") {
    const auto dev = characterize::DeviceDescriptor::single();
    // Output-stationary, column panels: B's panel stays in L3 for the i loop (4 feeds per load),
    // but every call Moves both operands out of L3.
    const auto os = run(matmul("os"), dev).report;
    CHECK(os.calls == 64);
    CHECK(reuse_of(os, "A").loads == 64);
    CHECK(reuse_of(os, "A").moves == 64);
    CHECK(reuse_of(os, "B").loads == 16);
    CHECK(reuse_of(os, "B").moves == 64);
    CHECK(reuse_of(os, "B").feeds_per_load() == 4.0);
    CHECK(reuse_of(os, "B").feeds_per_move() == 1.0);
    // A's row panel staged in L2: each A tile leaves L3 once per row, and feeds 4 calls.
    const auto staged = run(matmul("os", true), dev).report;
    CHECK(reuse_of(staged, "A").moves == 16);
    CHECK(reuse_of(staged, "A").feeds_per_move() == 4.0);
    // Weight-stationary: B's tile is held across the i loop.
    const auto ws = run(matmul("ws"), dev).report;
    CHECK(reuse_of(ws, "B").moves == 16);
    CHECK(reuse_of(ws, "B").feeds_per_move() == 4.0);
    CHECK(reuse_of(ws, "A").feeds_per_move() == 1.0);
}

TEST_CASE("Bubbles: the report adds up, and every bubble names a cause", "[program][csp][bubbles]") {
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        const Run r = run(matmul(df), characterize::DeviceDescriptor::single());
        for (const auto& rec : r.records) CHECK(rec.ready <= rec.start);   // a leg never starts before it is ready
        BubbleReport::Cycle sum = 0, by_cause = 0;
        for (const auto& b : r.report.bubbles) {
            CHECK(b.idle > 0);
            REQUIRE_FALSE(b.chain.empty());
            CHECK(b.chain.front() == b.call);
            CHECK(b.cause != Cause::Unknown);
            sum += b.idle;
        }
        for (const auto& [c, n] : r.report.idle_by_cause) by_cause += n;
        CHECK(sum == r.report.cf_idle);
        CHECK(by_cause == r.report.cf_idle);
        CHECK(r.report.cf_busy == r.stats.busy.at("cf"));
        CHECK(r.report.bubble_fraction() >= 0.0);
        CHECK(r.report.bubble_fraction() < 1.0);
        for (const auto& l : r.report.loads) CHECK(l.utilization <= 1.0);
        const std::string t = r.report.text();
        CHECK_THAT(t, ContainsSubstring("bubbles  64 calls"));
        CHECK_THAT(t, ContainsSubstring("feeds/move"));
    }
}

TEST_CASE("Bubbles: with a declared fabric, feeds and drains are scheduled apart", "[program][csp][bubbles]") {
    // #438's review: the streamer's drains have lanes of their own under a fabric, issued in their
    // own order. The report must still add up, with every bubble's cause known.
    auto dev = characterize::DeviceDescriptor::single();
    dev.pe_rows = 16;
    dev.pe_cols = 16;
    dev.streamers = 2;
    for (const char* df : {"os", "ws"}) {
        CAPTURE(df);
        const Run r = run(matmul(df), dev);
        bool split = false;
        for (const auto& rec : r.records)
            if (rec.kind == Action::Kind::Drain && rec.lane >= 2) split = true;
        CHECK(split);                                    // the drains ran on their own lanes
        BubbleReport::Cycle sum = 0;
        for (const auto& b : r.report.bubbles) {
            CHECK(b.cause != Cause::Unknown);
            sum += b.idle;
        }
        CHECK(sum == r.report.cf_idle);
        for (const auto& l : r.report.loads) CHECK(l.utilization <= 1.0);
    }
}

TEST_CASE("Bubbles: the cause follows the binding resource", "[program][csp][bubbles]") {
    // A starved BlockMover: every call waits for its operands' Moves.
    auto slow_bm = characterize::DeviceDescriptor::single();
    slow_bm.bm_bytes_per_cycle = 4.0;
    const auto bm = run(matmul("os"), slow_bm).report;
    CHECK(bm.cf_idle > 0);
    CHECK(dominant(bm) == Cause::MoverBusy);
    // A slow DRAM: with one DMA lane the Loads queue for it; with a lane per Load, and fast movers
    // behind them, the calls wait for the transfers themselves.
    auto one_dma = characterize::DeviceDescriptor::single();
    one_dma.dma_bytes_per_cycle = 1.0;
    one_dma.dma_engines = 1;
    const auto queued = run(matmul("os"), one_dma).report;
    INFO(queued.text());
    CHECK(dominant(queued) == Cause::DmaBusy);
    auto slow_dram = one_dma;
    slow_dram.dma_engines = 64;
    slow_dram.bm_bytes_per_cycle = 4096.0;
    slow_dram.str_bytes_per_cycle = 4096.0;
    const auto dram = run(matmul("os"), slow_dram).report;
    INFO(dram.text());
    CHECK(dram.cf_idle > 0);
    CHECK((dominant(dram) == Cause::Dram || dominant(dram) == Cause::L3Slot));
    // Stating reuse cuts the idle time the BlockMover causes.
    const auto ws = run(matmul("ws"), slow_bm).report;
    const auto in_ws = ws.idle_by_cause.find(Cause::MoverBusy);
    CHECK((in_ws == ws.idle_by_cause.end() ? 0 : in_ws->second) < bm.idle_by_cause.at(Cause::MoverBusy));
}



namespace {

// S1 with a Combiner (the WS and IS schedules merge their partials there).
platform::DeviceSpecification s1_combiner() {
    auto d = platform::read_spec_file("tests/program/deploy/kpu_s1.json").device(0);
    d.combiner = platform::DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add", "relu"}};
    return d;
}

}  // namespace

TEST_CASE("Bubbles: L-CA's run, read from its protocol events, is the program's actions", "[program][csp][bubbles][lca]") {
    // #425's report at L-CA: the events matched back to the actions (timing/csp_records.hpp). The
    // same program at L-T1 is the check: one leg per action, the same reuse, a report that adds up.
    const auto d = s1_combiner();
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        const std::string src = matmul(df);
        const lang::Program ast = lang::parse(src);
        const TileProgram in = driver::csp_inputs(ast);
        const auto lt1 = driver::csp_run_level(driver::ExecutionLevel::BlockSequential, ast, &d, in);
        const auto lca = driver::csp_run_level(driver::ExecutionLevel::CycleAccurate, ast, &d, in, 256, true);
        REQUIRE_FALSE(lca.skipped);
        CHECK(lca.unmatched_events == 0);
        // One leg per action, as at L-T1 (a Store has two), kind by kind.
        std::map<std::pair<std::size_t, int>, std::size_t> a, b;
        for (const auto& r : lt1.records) ++a[{r.action, static_cast<int>(r.kind)}];
        for (const auto& r : lca.records) ++b[{r.action, static_cast<int>(r.kind)}];
        CHECK(a == b);
        for (const auto& r : lca.records) {
            CHECK(r.ready <= r.start);
            CHECK(r.start <= r.finish);
            CHECK(r.finish <= lca.makespan);
        }
        const BubbleReport rep = bubble_report(lca.records, lca.makespan, lca.busy, lca.lanes, "L-CA");
        const BubbleReport ref = bubble_report(lt1.records, lt1.makespan, lt1.busy, lt1.lanes);
        CHECK(rep.calls == ref.calls);
        REQUIRE(rep.reuse.size() == ref.reuse.size());
        for (std::size_t i = 0; i < rep.reuse.size(); ++i) {     // the reuse is the program's, not a level's
            CHECK(rep.reuse[i].operand == ref.reuse[i].operand);
            CHECK(rep.reuse[i].loads == ref.reuse[i].loads);
            CHECK(rep.reuse[i].moves == ref.reuse[i].moves);
            CHECK(rep.reuse[i].feeds == ref.reuse[i].feeds);
        }
        CHECK(reuse_of(rep, "A").loads == lca.dram_loads - reuse_of(rep, "B").loads);
        BubbleReport::Cycle sum = 0;
        for (const auto& bb : rep.bubbles) {
            CHECK(bb.idle > 0);
            CHECK(bb.cause != Cause::Unknown);
            sum += bb.idle;
        }
        CHECK(sum == rep.cf_idle);
        CHECK(rep.cf_busy == lca.busy.at("cf"));
        for (const auto& l : rep.loads) CHECK(l.utilization <= 1.0);
        CHECK_THAT(rep.text(), ContainsSubstring("(L-CA, uncalibrated)"));
    }
    // A run that keeps no events has no records: they are the report's, not every run's.
    const lang::Program ast = lang::parse(matmul("os"));
    const auto plain = driver::csp_run_level(driver::ExecutionLevel::CycleAccurate, ast, &d, driver::csp_inputs(ast));
    CHECK(plain.records.empty());
}

TEST_CASE("Bubbles: an L-CA credit stall belongs to the action that was pending", "[program][csp][bubbles][lca]") {
    // #443's review: stalls were kept by process and tile and read from each action's handover, so
    // a later Move on the tile took an earlier Move's credit stalls as its own L2 wait (and ready).
    using sw::kpu::timing::CspDriver;
    using sw::kpu::timing::EventType;
    using K = Action::Kind;
    const TileCoord t{"A", 0, 0};
    std::vector<CspDriver::Result::Handover> hand = {{0, K::Load, t, 0}, {1, K::Move, t, 0}, {2, K::Load, t, 0},
                                                     {3, K::Move, t, 0}};
    std::vector<CspDriver::ProtocolEvent> ev;
    auto add = [&](std::uint64_t c, EventType ty, const std::string& comp, std::uint32_t id) {
        CspDriver::ProtocolEvent e;
        e.cycle = c;
        e.type = ty;
        e.component = comp;
        e.component_id = id;
        e.tile = t;
        ev.push_back(e);
    };
    add(0, EventType::DMA_LOAD_START, "DMA0", 0);
    add(5, EventType::DMA_LOAD_COMPLETE, "DMA0", 0);
    for (std::uint64_t c = 10; c < 20; ++c) add(c, EventType::BM_STALL_CREDIT, "BM", 100);   // the first Move's
    add(20, EventType::BM_MOVE_START, "BM", 100);
    add(30, EventType::BM_MOVE_COMPLETE, "BM", 100);
    add(50, EventType::DMA_LOAD_START, "DMA0", 0);
    add(99, EventType::DMA_LOAD_COMPLETE, "DMA0", 0);
    add(100, EventType::BM_MOVE_START, "BM", 100);
    add(110, EventType::BM_MOVE_COMPLETE, "BM", 100);
    const auto out = sw::kpu::timing::csp_records(hand, ev, 1, 1, 1);
    CHECK(out.unmatched == 0);
    REQUIRE(out.records.size() == 4);
    const auto& first = out.records[1];
    const auto& second = out.records[3];
    CHECK(first.l2_wait == 10);                       // it waited 10..19 for its L2 slot
    CHECK(first.ready == 20);
    CHECK(second.l2_wait == 0);                       // the earlier stalls are not its own
    CHECK(second.ready == 100);                       // its Load finished at 99: ready on the next tick
    CHECK(second.lane == 0);                          // BlockMovers numbered from 0
}
