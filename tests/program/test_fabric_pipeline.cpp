// ============================================================================
// tests/program/test_fabric_pipeline.cpp
// L-T1 times the fabric as a systolic pipeline (docs/plans/l2-reuse-and-fabric-pipeline-view.md
// step 2). With the deployment's PE array declared, an accumulator's gemm chain is one
// output-stationary stream: it owns a compute tile from its first call to its Drain, fills once
// and drains once, and its results leave north on their own links, so the next chain's operands
// follow right behind. A call takes its reduction depth in cycles on the PEs its C tile occupies.
// Values never change: the timing is the only thing that moves.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/csp/transactional.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cstdint>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::csp;
using LT = TransactionalInterpreter;
using Cycle = LT::Cycle;

namespace {

struct Run {
    LT::Stats stats;
    std::vector<LT::Record> records;
    std::vector<LT::Chain> chains;
    TileProgram values;
};

Run lt1(const std::string& text, const characterize::DeviceDescriptor& dev) {
    const lang::Program ast = lang::parse(text);
    LT lt(dev, static_cast<std::size_t>(ast.l3), static_cast<std::uint32_t>(ast.banks));
    lt.begin(driver::csp_inputs(ast));
    lang::ActionStream s(ast);
    while (auto e = s.next()) lt.step(e->action, e->op ? &*e->op : nullptr);
    Run r;
    r.stats = lt.finish();
    r.records = lt.records();
    r.chains = lt.chains();
    r.values = lt.result();
    return r;
}

// C[m,n] = A[m,k] . B[k,n], one accumulator chain per C tile, k innermost. `store` after each
// chain, `store ... late` (as kpu-csp-gen writes it), or deferred until every chain has run.
enum class Store { EachChain, Late, Deferred };
std::string matmul(int m, int n, int k, int tm, int tn, int tk, Store store = Store::EachChain) {
    const int ni = m / tm, nj = n / tn, nk = k / tk;
    auto s = [](int v) { return std::to_string(v); };
    const bool each = store != Store::Deferred;
    std::string p = "csp 1.0\nprogram mm machine flat(l3 = " + s(2 * nk + 2) + ") {\n"
           "  tensor A[" + s(m) + "," + s(k) + "] tile " + s(tm) + "x" + s(tk) + " in;\n"
           "  tensor B[" + s(k) + "," + s(n) + "] tile " + s(tk) + "x" + s(tn) + " in;\n"
           "  tensor C[" + s(m) + "," + s(n) + "] tile " + s(tm) + "x" + s(tn) + " out;\n"
           "  for j in 0.." + s(nj) + " {\n    resident B[:, j];\n"
           "    for i in 0.." + s(ni) + " {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric {\n        for k in 0.." + s(nk) + " { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n" +
           std::string(!each ? "" : store == Store::Late ? "      store C[i, j] late;\n" : "      store C[i, j];\n") +
           "      release A[i, :];\n    }\n    release B[:, j];\n  }\n";
    if (!each) p += "  for j in 0.." + s(nj) + " { for i in 0.." + s(ni) + " { store C[i, j]; } }\n";
    return p + "}\n";
}

// A device whose movers never hold the fabric up, so the chains' timing is the fabric's alone.
characterize::DeviceDescriptor fast(Dim pe_rows, Dim pe_cols) {
    characterize::DeviceDescriptor d = characterize::DeviceDescriptor::single();
    d.fabric_macs_per_cycle = double(pe_rows * pe_cols);
    d.pe_rows = pe_rows;
    d.pe_cols = pe_cols;
    d.dma_engines = 8;
    d.dma_bytes_per_cycle = 1e6;
    d.bm_bytes_per_cycle = 1e6;
    d.str_bytes_per_cycle = 1e6;
    return d;
}

const LT::Record& leg_of(const Run& r, std::size_t action) {
    for (const auto& rec : r.records)
        if (rec.action == action && rec.proc == LT::Proc::Cf) return rec;
    FAIL("no cf leg for action " << action);
    return r.records.front();
}

bool same_values(const TileProgram& a, const TileProgram& b) {
    for (const auto& name : a.operand_order())
        if (a.operand(name).values != b.operand(name).values) return false;
    return true;
}

}  // namespace

TEST_CASE("a chain fills once and drains once, on its own compute tile", "[fabric][lt1]") {
    // 64x64 C tiles on a 64x64 array, four k-steps of depth 32 each.
    const Run r = lt1(matmul(128, 128, 128, 64, 64, 32), fast(64, 64));
    REQUIRE(r.chains.size() == 4);
    CHECK(r.stats.stream_chains == 4);
    CHECK(r.stats.one_leg.empty());
    for (const auto& c : r.chains) {
        CAPTURE(c.tile.to_string());
        REQUIRE(c.closed);
        CHECK(c.calls == 4);
        CHECK(c.rows == 64);
        CHECK(c.cols == 64);
        const Cycle skew = 63 + 63;
        CHECK(c.front - c.in0 == skew);                  // the wavefront crosses once
        CHECK(c.last - c.in1 == skew);                   // and leaves once
        CHECK(c.out1 - c.last == 64 + c.link_delay);     // the far PE's result: 64 hops north
        CHECK(c.link_delay == 0);                        // K = 128 >= 64: the top link keeps up
        // Each call is its reduction depth at PE(0,0): 32 cycles, back to back within the chain.
        const auto& first = leg_of(r, c.first_action);
        const auto& last = leg_of(r, c.last_action);
        CHECK(first.start == c.in0);
        CHECK(last.finish == c.in1);
        CHECK(c.in1 - c.in0 == 4 * 32);
        CHECK(first.chain != LT::kNoChain);
        CHECK(first.chain_pos == 1);
        CHECK(last.chain_pos == 4);
        // The Drain waits for the last result to leave the array, not for the last call.
        bool drained = false;
        for (const auto& rec : r.records)
            if (rec.action == c.drain_action) { CHECK(rec.start >= c.out1); drained = true; }
        CHECK(drained);
    }
}

TEST_CASE("the next chain's operands follow right behind, under the previous drain", "[fabric][lt1]") {
    // Stores deferred: nothing between two chains holds the movers, so the array decides.
    const Run r = lt1(matmul(128, 128, 128, 64, 64, 32, Store::Deferred), fast(64, 64));
    REQUIRE(r.chains.size() == 4);
    Cycle steady = 0;
    for (std::size_t i = 0; i < r.chains.size(); ++i) {
        steady += r.chains[i].in1 - r.chains[i].in0;
        if (i == 0) continue;
        const auto& prev = r.chains[i - 1];
        const auto& next = r.chains[i];
        CHECK(next.lane == prev.lane);
        CHECK(next.in0 == prev.in1);                     // no bubble: drains never stall operands
        CHECK(next.in0 < prev.out1);                     // the three-tile moment
    }
    // The array never idles between chains: utilization over the operand streams is exactly 1.
    CHECK(steady == r.chains.back().in1 - r.chains.front().in0);
}

TEST_CASE("a chain shorter than the drain waits on the top link, in the grid's registers", "[fabric][lt1]") {
    // One k-step of depth 16 per chain on a 64x64 array: K = 16 < 64 results per column.
    const Run r = lt1(matmul(128, 128, 16, 64, 64, 16, Store::Deferred), fast(64, 64));
    REQUIRE(r.chains.size() == 4);
    CHECK(r.chains[0].link_delay == 0);
    Cycle total = 0;
    for (std::size_t i = 1; i < r.chains.size(); ++i) {
        const auto& prev = r.chains[i - 1];
        const auto& next = r.chains[i];
        CHECK(next.in0 == prev.in1);                     // operands are never held back
        CHECK(next.out1 == prev.out1 + 64);              // 64 results through each top link
        // Results come 16 cycles apart but leave 64 apart: the queue in the grid grows by 48 a chain.
        CHECK(next.link_delay == prev.link_delay + (64 - 16));
        total += next.link_delay;
    }
    CHECK(r.stats.link_delay_cycles == total);
    // K < skew: the chain never fills the array; its fill and its tail overlap.
    CHECK(r.chains[1].front > r.chains[1].in1);
}

TEST_CASE("a store between chains serializes them: the program is the schedule", "[fabric][lt1]") {
    // Every mover issues in program order. `store C` after each chain puts its Writeback (bm) and
    // its Store (dma) ahead of the next chain's Moves and Loads, and both wait for the Drain, which
    // waits for the results. So the next chain cannot start until the previous one is out: the
    // fabric could overlap them, the program does not let it.
    const Run r = lt1(matmul(128, 128, 128, 64, 64, 32), fast(64, 64));
    REQUIRE(r.chains.size() == 4);
    for (std::size_t i = 1; i < r.chains.size(); ++i) CHECK(r.chains[i].in0 >= r.chains[i - 1].out1);
}

TEST_CASE("a late store lets the chains pipeline, one chain deep", "[fabric][lt1][late]") {
    // Step 2b: each store leaves after the next chain's calls, so the movers bring that chain's
    // tiles in first. 8 k-steps of 32: a chain's calls (256) outlast its drain (126 + 64), so the
    // array goes from one chain to the next exactly as it does with no stores in between.
    const Run late = lt1(matmul(128, 128, 256, 64, 64, 32, Store::Late), fast(64, 64));
    const Run deferred = lt1(matmul(128, 128, 256, 64, 64, 32, Store::Deferred), fast(64, 64));
    REQUIRE(late.chains.size() == 4);
    for (std::size_t i = 1; i < late.chains.size(); ++i) {
        CHECK(late.chains[i].in0 == late.chains[i - 1].in1);
        CHECK(late.chains[i].in0 < late.chains[i - 1].out1);   // the three-tile moment
        CHECK(late.chains[i].in0 == deferred.chains[i].in0);
    }
    CHECK(same_values(late.values, lt1(matmul(128, 128, 256, 64, 64, 32), fast(64, 64)).values));

    // One chain deep, not more: with 4 k-steps a chain's calls (128) end before the chain ahead
    // of it has drained (190), and the store that chain's Writeback leads in the block mover's
    // order holds the third chain's Moves until the first chain's results are out.
    const Run shallow = lt1(matmul(128, 128, 128, 64, 64, 32, Store::Late), fast(64, 64));
    REQUIRE(shallow.chains.size() == 4);
    CHECK(shallow.chains[1].in0 == shallow.chains[0].in1);
    CHECK(shallow.chains[2].in0 > shallow.chains[1].in1);
    CHECK(shallow.chains[2].in0 >= shallow.chains[0].out1);
}

TEST_CASE("a C tile smaller than the array leaves the other PEs idle", "[fabric][lt1]") {
    // 32x32 tiles on kpu_s1's 128x64 array: a call is its depth (32), not macs / 8192 (4).
    const Run r = lt1(matmul(64, 64, 64, 32, 32, 32), fast(128, 64));
    REQUIRE_FALSE(r.chains.empty());
    for (const auto& c : r.chains) {
        CHECK(c.rows == 32);
        CHECK(c.cols == 32);
        CHECK(c.front - c.in0 == 31 + 31);
        const auto& first = leg_of(r, c.first_action);
        CHECK(first.finish - first.start == 32);
    }
}

TEST_CASE("without a declared fabric every call is one leg, as before", "[fabric][lt1]") {
    characterize::DeviceDescriptor d = fast(64, 64);
    d.pe_rows = d.pe_cols = 0;
    const Run r = lt1(matmul(128, 128, 128, 64, 64, 32), d);
    CHECK(r.chains.empty());
    REQUIRE(r.stats.one_leg.count("no fabric declared"));
    CHECK(r.stats.one_leg.at("no fabric declared") == 16);
    for (const auto& rec : r.records)
        if (rec.kind == Action::Kind::Call) {             // a credit action's record has no process
            CHECK(rec.chain == LT::kNoChain);
            CHECK(rec.finish - rec.start == (64 * 64 * 32) / 4096);   // macs / macs_per_cycle
        }
}

TEST_CASE("the pipeline changes timing, never values", "[fabric][lt1]") {
    const std::string p = matmul(128, 128, 128, 64, 64, 32);
    characterize::DeviceDescriptor plain = fast(64, 64);
    plain.pe_rows = plain.pe_cols = 0;
    const Run a = lt1(p, fast(64, 64));
    const Run b = lt1(p, plain);
    CHECK(a.stats.makespan != b.stats.makespan);
    CHECK(same_values(a.values, b.values));
}

TEST_CASE("an in-place gemm and a C tile larger than the array stay one leg", "[fabric][lt1]") {
    // A C tile of 64x64 on a 32x32 array.
    const Run big = lt1(matmul(128, 128, 64, 64, 64, 32), fast(32, 32));
    CHECK(big.chains.empty());
    CHECK(big.stats.one_leg.count("C tile larger than the array"));
    // C resident in L3 and updated in place: its preload into the array has no stream design.
    const std::string in_place =
        "csp 1.0\nprogram ip machine flat(l3 = 4) {\n"
        "  tensor A[32,32] tile 32x32 in;\n  tensor B[32,32] tile 32x32 in;\n  tensor C[32,32] tile 32x32 inout;\n"
        "  resident A[0, 0], B[0, 0], C[0, 0];\n  call gemm(A[0, 0], B[0, 0]) +-> C[0, 0];\n"
        "  store C[0, 0];\n  release A[0, 0], B[0, 0], C[0, 0];\n}\n";
    const Run ip = lt1(in_place, fast(32, 32));
    CHECK(ip.chains.empty());
    CHECK(ip.stats.one_leg.count("in-place gemm (C preloaded)"));
}

TEST_CASE("a deployment's fabric must agree with its MAC rate", "[fabric][deployment]") {
    platform::DeploymentSpec spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");
    REQUIRE(spec.devices[0].fabric.pe_rows == Dim(128));
    REQUIRE(spec.devices[0].fabric.pe_cols == Dim(64));
    CHECK(spec.validate().empty());
    const auto view = spec.device_view(0);
    CHECK(view.pe_rows == 128);
    CHECK(view.pe_cols == 64);

    platform::DeploymentSpec wrong = spec;
    wrong.devices[0].fabric.pe_rows = 64;               // 64 x 64 = 4096, not 8192
    CHECK(wrong.validate().find("macs_per_cycle") != std::string::npos);
    platform::DeploymentSpec half = spec;
    half.devices[0].fabric.pe_cols.reset();
    CHECK(half.validate().find("declared together") != std::string::npos);

    // It survives the canonical round trip.
    const platform::DeploymentSpec back = platform::from_json(platform::to_json(spec));
    CHECK(back.devices[0].fabric.pe_rows == Dim(128));
    CHECK(back.devices[0].fabric.pe_cols == Dim(64));
}
