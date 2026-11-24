// ============================================================================
// tests/program/test_lca_systolic_array.cpp
// L-CA times an output-stationary gemm on the deployment's systolic array (#458;
// docs/plans/systolic-tile-at-lca.md §3.1-3.2, step 1).
//
// The RTL's COMPUTE commands (embodied-ai-soc-generator streamer.py, b57c881), for an h x w result
// on an R x C array with K split into commands of KB = C elements, D the L2 read latency:
//   - command m is accepted at a_m; the tile accepts the next I = max(K_m, h, w) later;
//   - its lanes stream from s = a + 2 + D, and only the reduction's last command makes results;
//   - result row i leaves the north FIFOs at s + K + w + 2i, so the last at s + K + w + 2h - 2;
//   - a reduction's first row leaves at least 2h' - 1 clocks after the previous one's (h' rows).
// A compute's record spans its stream [start, start + busy); its result is ready at result_cycle.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/tile_check.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>
#include <sw/kpu/timing/csp_driver.hpp>

#include <map>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::timing;
namespace lang = sw::kpu::program::csp::lang;
using Catch::Matchers::ContainsSubstring;
using driver::ExecutionLevel;
using sw::kpu::isa::MatrixID;

namespace {

const platform::DeviceSpecification& device(const std::string& file) {
    static std::map<std::string, platform::DeviceSpecification> cache;
    auto it = cache.find(file);
    if (it == cache.end())
        it = cache.emplace(file, platform::read_spec_file("tests/program/deploy/" + file).device(0)).first;
    return it->second;
}

// C[M,N] = A[M,K] . B[K,N], int8 -> int32, an output-stationary accumulator per C tile: C in h x w
// tiles, K in kt-wide slices.
std::string matmul(long m, long n, long k, long h, long w, long kt) {
    auto s = [](long v) { return std::to_string(v); };
    return "csp 1.1\nprogram mm machine flat(l3 = 64, l2 = 6) {\n"
           "  tensor A[" + s(m) + "," + s(k) + "] int8 tile " + s(h) + "x" + s(kt) + " in;\n"
           "  tensor B[" + s(k) + "," + s(n) + "] int8 tile " + s(kt) + "x" + s(w) + " in;\n"
           "  tensor C[" + s(m) + "," + s(n) + "] int32 tile " + s(h) + "x" + s(w) + " out;\n"
           "  for j in 0.." + s(n / w) + " {\n    resident B[:, j];\n    for i in 0.." + s(m / h) + " {\n"
           "      resident A[i, :];\n      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0.." + s(k / kt) + " { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

struct Compute {
    TileID tile;
    Cycle start, end, result;
};

// The program at L-CA on `d`: its computes in completion order, values checked against L-B.
std::vector<Compute> computes(const std::string& program, const platform::DeviceSpecification& d) {
    const lang::Program ast = lang::parse(program);
    const auto cfg = csp_config_from(d);
    REQUIRE(cfg.has_value());
    ConcurrentTimingExecutor exec(cfg->config);
    lang::ActionStream s(ast, lang::target_from(d));
    const auto r = CspDriver(exec, s, driver::csp_inputs(ast)).run();
    REQUIRE(r.completed);
    const auto lb = driver::csp_run_level(ExecutionLevel::Behavioral, ast, &d, driver::csp_inputs(ast));
    const auto lca = driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &d, driver::csp_inputs(ast));
    REQUIRE_FALSE(lca.skipped);
    const OperandsCheck c = check_operands(lca.values, lb.values);
    INFO(c.summary());
    CHECK(c.pass);
    std::vector<Compute> out;
    for (const auto& e : exec.events())
        if (e.type == EventType::COMPUTE_COMPLETE)
            out.push_back({e.tile_id, e.cycle, e.cycle + e.duration, e.result_cycle});
    return out;
}

constexpr Cycle kD = 4;     // kpu_soc_2x2's spad_read_delay, and the banked L2's default (kpu_s1)

}  // namespace

TEST_CASE("L-CA reads the systolic array from the deployment", "[lca][fabric][systolic]") {
    const auto s1 = csp_config_from(device("kpu_s1.json"));
    REQUIRE(s1.has_value());
    CHECK(s1->config.array_rows == 128);
    CHECK(s1->config.array_cols == 64);
    const auto soc = csp_config_from(device("kpu_soc_2x2.json"));
    REQUIRE(soc.has_value());
    CHECK(soc->config.array_rows == 4);
    CHECK(soc->config.array_cols == 4);
    REQUIRE(soc->config.l2_banks.has_value());
    CHECK(soc->config.l2_banks->read_latency == kD);
    // What L-CA does not model yet is named, and nothing says the fabric is ignored.
    auto names = [](const std::vector<std::string>& v, const std::string& what) {
        for (const auto& u : v) if (u.find(what) != std::string::npos) return true;
        return false;
    };
    CHECK(names(soc->unmapped, "COMPUTE commands on the 4x4 array"));
    CHECK_FALSE(names(soc->unmapped, "keeps its own timing"));
    // No fabric declared: the MAC-rate model, and it says so.
    const auto t4 = csp_config_from(device("kpu_t4.json"));
    REQUIRE(t4.has_value());
    CHECK(t4->config.array_rows == 0);
    CHECK(names(t4->unmapped, "fabric.pe_rows/pe_cols: not declared"));
}

TEST_CASE("An OS gemm runs as the RTL's COMPUTE commands on kpu_soc_2x2", "[lca][fabric][systolic]") {
    SECTION("one command: a 4x4x4 call") {
        const auto c = computes(matmul(4, 4, 4, 4, 4, 4), device("kpu_soc_2x2.json"));
        REQUIRE(c.size() == 1);
        CHECK(c[0].end - c[0].start == 4);                       // I = max(4, 4, 4)
        // s = a + 2 + D; the last row at s + K + w + 2h - 2.
        CHECK(c[0].result - c[0].start == 2 + kD + 4 + 4 + 2 * 4 - 2);
    }
    SECTION("a chain over K: four calls, one reduction") {
        const auto c = computes(matmul(4, 4, 16, 4, 4, 4), device("kpu_soc_2x2.json"));
        REQUIRE(c.size() == 4);
        for (std::size_t i = 0; i < 3; ++i) {                    // k-slices with more to come: no results
            CAPTURE(i);
            CHECK(c[i].end - c[i].start == 4);
            CHECK(c[i].result == c[i].end);
            CHECK(c[i + 1].start >= c[i].end);                   // one call at a time on the array
        }
        CHECK(c[3].end - c[3].start == 4);
        CHECK(c[3].result - c[3].start == 2 + kD + 4 + 4 + 2 * 4 - 2);
    }
    SECTION("a call whose K is two commands: A's tile is 4 x 8, KB = 4") {
        const auto c = computes(matmul(4, 4, 8, 4, 4, 8), device("kpu_soc_2x2.json"));
        REQUIRE(c.size() == 1);
        CHECK(c[0].end - c[0].start == 4 + 4);
        CHECK(c[0].result - c[0].start == 4 + 2 + kD + 4 + 4 + 2 * 4 - 2);
    }
    SECTION("the RTL's 8x8x16 matmul: values, and results never closer than 2h - 1 on the north links") {
        const auto c = computes(matmul(8, 8, 16, 4, 4, 4), device("kpu_soc_2x2.json"));
        REQUIRE(c.size() == 16);
        std::vector<Cycle> row0;                                  // each reduction's first result row
        for (const auto& x : c)
            if (x.result > x.end) row0.push_back(x.result - (4 + 2 * 4 - 2));
        REQUIRE(row0.size() == 4);
        for (std::size_t i = 1; i < row0.size(); ++i) CHECK(row0[i] >= row0[i - 1] + 2 * 4 - 1);
    }
}

TEST_CASE("A tile smaller than the array runs on its corner: 16x16 on kpu_s1's 128x64", "[lca][fabric][systolic]") {
    const auto c = computes(matmul(16, 16, 16, 16, 16, 16), device("kpu_s1.json"));
    REQUIRE(c.size() == 1);
    CHECK(c[0].end - c[0].start == 16);                          // I = max(16, 16, 16)
    // Its 16 rows drain north through 16 PEs, not the array's 128: s + K + w + 2h - 2.
    CHECK(c[0].result - c[0].start == 2 + kD + 16 + 16 + 2 * 16 - 2);
}

TEST_CASE("A reduction's first row waits 2h - 1 after the previous one's", "[lca][fabric][systolic]") {
    // Two independent 4x4 results back to back, their operands already in place (no Feeds): the
    // second could launch 4 clocks after the first, and is held at accept until 2h - 1 = 7.
    ConcurrentTimingExecutor::Config cfg;
    cfg.num_compute_tiles = 1;
    cfg.macs_per_cycle = 16.0;
    cfg.array_rows = cfg.array_cols = 4;
    ConcurrentTimingExecutor e(cfg);
    auto tile = [](MatrixID m, Size ti, Size tj) {
        TileDescriptor d;
        d.tile_id.matrix = m;
        d.tile_id.ti = ti;
        d.tile_id.tj = tj;
        d.height = d.width = 4;
        d.size_bytes = 4 * 4 * 4;
        return d;
    };
    const auto a = tile(MatrixID::A, 0, 0), b = tile(MatrixID::B, 0, 0);
    e.set_tile_payload(a.tile_id, TilePayload{4, 4, std::vector<float>(16, 1.0f)});
    e.set_tile_payload(b.tile_id, TilePayload{4, 4, std::vector<float>(16, 1.0f)});
    ConcurrentTimingExecutor::MatMulComputeSpec spec;
    spec.a_tiles = {a.tile_id};
    spec.b_tiles = {b.tile_id};
    for (const auto* t : {&a, &b}) {
        e.schedule_load(*t);
        e.schedule_move(*t);
        e.schedule_feed(*t);
    }
    const auto c0 = tile(MatrixID::C, 0, 0), c1 = tile(MatrixID::C, 0, 1);
    e.schedule_matmul_compute(c0, spec);
    e.schedule_matmul_compute(c1, spec);
    e.schedule_drain(c0);
    e.schedule_drain(c1);
    REQUIRE(e.run());
    std::vector<TimingEvent> done;
    for (const auto& ev : e.events())
        if (ev.type == EventType::COMPUTE_COMPLETE) done.push_back(ev);
    REQUIRE(done.size() == 2);
    const Cycle d = cfg.str_startup_latency;                     // the pooled L2's read latency
    CHECK(done[0].duration == 4);
    CHECK(done[0].result_cycle - done[0].cycle == 2 + d + 4 + 4 + 2 * 4 - 2);
    CHECK(done[1].cycle == done[0].cycle + 4);                   // accepted I = 4 later
    CHECK(done[1].duration == 3 + 4);                            // held 3 clocks at accept
    CHECK(done[1].result_cycle - done[0].result_cycle == 2 * 4 - 1);
}

TEST_CASE("A tile larger than the array is refused", "[lca][fabric][systolic]") {
    const lang::Program ast = lang::parse(matmul(8, 8, 8, 8, 8, 8));
    const auto why = driver::csp_level_supported(ExecutionLevel::CycleAccurate, ast, &device("kpu_soc_2x2.json"));
    REQUIRE(why.has_value());
    CHECK_THAT(*why, ContainsSubstring("8x8 tile of C on the 4x4 systolic array"));
    CHECK_THAT(*why, ContainsSubstring("csp line"));
    // The same program fits kpu_s1's array.
    CHECK_FALSE(driver::csp_level_supported(ExecutionLevel::CycleAccurate, ast, &device("kpu_s1.json")));
    // The executor refuses one too, if it is handed one.
    ConcurrentTimingExecutor::Config cfg;
    cfg.num_compute_tiles = 1;
    cfg.macs_per_cycle = 16.0;
    cfg.array_rows = cfg.array_cols = 4;
    ConcurrentTimingExecutor e(cfg);
    TileDescriptor a, b, c;
    a.tile_id.matrix = MatrixID::A;
    b.tile_id.matrix = MatrixID::B;
    c.tile_id.matrix = MatrixID::C;
    a.height = a.width = b.height = b.width = c.height = c.width = 8;
    e.set_tile_payload(a.tile_id, TilePayload{8, 8, std::vector<float>(64, 1.0f)});
    e.set_tile_payload(b.tile_id, TilePayload{8, 8, std::vector<float>(64, 1.0f)});
    a.size_bytes = b.size_bytes = c.size_bytes = 8 * 8 * 4;
    for (const auto* t : {&a, &b}) {
        e.schedule_load(*t);
        e.schedule_move(*t);
        e.schedule_feed(*t);
    }
    ConcurrentTimingExecutor::MatMulComputeSpec spec;
    spec.a_tiles = {a.tile_id};
    spec.b_tiles = {b.tile_id};
    e.schedule_matmul_compute(c, spec);
    CHECK_THROWS_WITH(e.run(), ContainsSubstring("is a 8x8 result; the array is 4x4"));
}

TEST_CASE("No fabric declared: the MAC-rate model, unchanged", "[lca][fabric][systolic]") {
    // kpu_soc_2x2 without pe_rows/pe_cols: a 4x4x4 call keeps fill (2 x 4) + 64 MACs / 16 per cycle.
    platform::DeviceSpecification d = device("kpu_soc_2x2.json");
    d.fabric.pe_rows.reset();
    d.fabric.pe_cols.reset();
    const auto c = computes(matmul(4, 4, 4, 4, 4, 4), d);
    REQUIRE(c.size() == 1);
    CHECK(c[0].end - c[0].start == 8 + 4);
    CHECK(c[0].result == c[0].end);
}
