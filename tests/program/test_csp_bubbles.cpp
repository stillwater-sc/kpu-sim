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

