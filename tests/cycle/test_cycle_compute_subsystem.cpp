// ============================================================================
// tests/cycle/test_cycle_compute_subsystem.cpp
// The compute subsystem as kpu-run --compute-subsystem runs it, step 2f (#547;
// docs/plans/cycle-based-kpu-prototype.md §5.4): run_compute_subsystem's checks, its counters
// summary, its trace's lanes and its utilization heatmap.
//
// The PE-clocks by cause are checked against what the program implies, exactly: each command of
// h x w lanes and K elements is h * w * K useful MACs and h * (C - w) * K padded ones, on the PEs its
// tile covers, and nothing on the rows past it; every PE's causes add up to the array's clocks.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/cycle/compute/compute_subsystem.hpp>
#include <sw/kpu/cycle/compute/csp_compute_program.hpp>
#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/compile.hpp>
#include <sw/kpu/program/csp/lang/context.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <string>

using namespace sw::kpu::cycle;
namespace program = sw::kpu::program;
namespace csp = sw::kpu::program::csp;
namespace lang = sw::kpu::program::csp::lang;
namespace platform = sw::kpu::program::platform;
namespace driver = sw::kpu::program::driver;
using Catch::Matchers::ContainsSubstring;

namespace {

platform::DeviceSpecification device(const std::string& file) {
    return platform::read_spec_file(std::string(KPU_SOURCE_DIR) + "/tests/program/deploy/" + file).device(0);
}

std::string rtl_8x8x16() {
    return "csp 1.1\nprogram mm machine flat(l3 = 32, l2 = 6) {\n"
           "  tensor A[8,16] int8 tile 4x4 in;\n  tensor B[16,8] int8 tile 4x4 in;\n"
           "  tensor C[8,8] int32 tile 4x4 out;\n"
           "  for j in 0..2 {\n    resident B[:, j];\n    for i in 0..2 {\n      resident A[i, :];\n"
           "      acc C[i, j] in fabric dataflow os {\n"
           "        for k in 0..4 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
           "      store C[i, j];\n      release A[i, :];\n    }\n    release B[:, j];\n  }\n}\n";
}

std::string int8_64cubed() {
    csp::gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 96;
    o.dtype = "int8";
    o.l2_buffers = 2;
    return csp::gen::generate(o);
}

CspComputePlan plan_of(const std::string& text, const platform::DeviceSpecification& dev, std::uint32_t align) {
    const lang::Program ast = lang::parse(text);
    const auto target = lang::target_from(dev);
    const csp::CspProgram p = lang::compile(ast, &target);
    const program::TileProgram in = driver::csp_inputs(ast);
    auto lb = driver::csp_run_level(driver::ExecutionLevel::Behavioral, ast, &dev, in);
    REQUIRE(!lb.skipped);
    return plan_compute(p, in, lb.values, align);
}

std::string scratch(const std::string& name) { return std::string(KPU_BINARY_DIR) + "/" + name; }

}  // namespace

TEST_CASE("The compute subsystem on kpu_s1: the int8 64^3's PE-clocks by cause, exactly as the program implies",
          "[cycle][compute][counters]") {
    const auto dev = device("kpu_s1.json");
    ComputeRunOptions o;
    const CspComputePlan plan = plan_of(int8_64cubed(), dev, o.bus_bytes);
    const ComputeRun r = run_compute_subsystem(plan, dev, o);
    REQUIRE(r.ok());
    CHECK(r.tiles_checked == 16);
    CHECK(r.rows_pe == 128);
    CHECK(r.cols_pe == 64);
    REQUIRE(r.array_cycles > 0);

    // 64 commands, each 16 x 16 lanes of K = 16, on the array's corner.
    const std::uint64_t cmds = 64, h = 16, w = 16, K = 16, C = 64;
    CHECK(r.pe_total.useful == cmds * h * w * K);
    CHECK(r.pe_total.padded == cmds * h * (C - w) * K);
    CHECK(r.pe_total.draining > 0);
    CHECK(r.pe_total.total() == std::uint64_t(128) * 64 * r.array_cycles);
    auto pe = [&](std::uint32_t i, std::uint32_t j) { return r.pes.at(std::size_t(i) * r.cols_pe + j); };
    CHECK(pe(0, 0).useful == cmds * K);
    CHECK(pe(15, 15).useful == cmds * K);
    CHECK(pe(0, 16).useful == 0);
    CHECK(pe(0, 16).padded == cmds * K);
    CHECK(pe(16, 0).useful + pe(16, 0).padded + pe(16, 0).draining == 0);   // below the tile: no operand, no result
    CHECK(pe(127, 63).waiting == r.array_cycles);
    // Draining: each result moves north a row a clock, so it passes each PE above it for one clock.
    // The chains run far apart here, so no PE multiplies while one passes: PE(0, j) sees the 15
    // results under it of each of the 16 tiles, PE(15, j) none.
    CHECK(pe(0, 0).draining == 16 * (h - 1));
    CHECK(pe(7, 3).draining == 16 * (h - 8));
    CHECK(pe(15, 0).draining == 0);
    for (const auto& c : r.pes) REQUIRE(c.total() == r.array_cycles);
    CHECK(std::fabs(r.utilization() - double(r.pe_total.useful) / (128.0 * 64 * double(r.array_cycles))) < 1e-12);

    // Each L2 bank's reads in flight: every read is in flight its read latency, so the integral is
    // reads x latency exactly (step 2a's measurement of #495's question).
    std::uint64_t reads = 0, integral = 0;
    for (const auto& b : r.banks) {
        reads += b.reads;
        integral += b.in_flight_integral;
        CHECK(b.max_in_flight >= 1);
    }
    CHECK(integral == reads * o.read_latency);
    // The Streamer reads each command's 16 A and 16 B rows, and the BlockMover each result's 16.
    CHECK(reads == cmds * (h + w) + 16 * h);

    // The north FIFOs: a row leaves when its last column is in, so column 0 holds up to w / 2 rows.
    CHECK(r.fifo_max.at(0) == w / 2);
    CHECK(r.fifo_max.at(w) == 0);   // a column past the tile: its results are padded, dropped

    const std::string s = format_compute_summary(plan, r);
    CHECK_THAT(s, ContainsSubstring("16 of 16 result tiles exact; laws: hold on every row"));
    CHECK_THAT(s, ContainsSubstring("array        utilization"));
    CHECK_THAT(s, ContainsSubstring("waiting on L1"));
    CHECK_THAT(s, ContainsSubstring("L2 bank 3"));
    CHECK_THAT(s, ContainsSubstring("reads in flight: mean"));
    CHECK_THAT(s, ContainsSubstring("Streamer     64 commands"));
    CHECK_THAT(s, ContainsSubstring("BlockMover   busy"));
    CHECK_THAT(s, ContainsSubstring("Combiner     16 result tiles"));
}

TEST_CASE("The compute subsystem's trace: lanes for the L1 buffers, PE rows, north FIFOs, L2 banks and slots",
          "[cycle][compute][trace]") {
    const auto dev = device("kpu_soc_2x2.json");
    Tracer t;
    ComputeRunOptions o;
    o.tracer = &t;
    const CspComputePlan plan = plan_of(rtl_8x8x16(), dev, o.bus_bytes);
    const ComputeRun r = run_compute_subsystem(plan, dev, o);
    REQUIRE(r.ok());
    const std::string path = scratch("compute_subsystem_trace.json");
    t.write(path);
    REQUIRE(t.dropped() == 0);

    std::ifstream in(path);
    const nlohmann::json j = nlohmann::json::parse(in);
    std::map<int, std::string> threads;
    for (const auto& e : j["traceEvents"])
        if (e["ph"] == "M" && e["name"] == "thread_name") threads[e["tid"].get<int>()] = e["args"]["name"];
    std::map<std::string, int> tid;
    for (const auto& [id, name] : threads) tid[name] = id;
    for (const char* lane : {"L1 A lane 0 (kpu)", "L1 B lane 3 (kpu)", "PE row 3 (kpu)", "north FIFO 0 (kpu)",
                             "L2 bank 0 reads in flight (l2)", "L2 slot 0 (kpu)"})
        CHECK(tid.count(lane));

    // Spans on a lane: each L1 lane streams each command's block, K = 4 clocks (4 ns) of "cmd n",
    // once per command, in order; PE row 0's west PE multiplies each the same 4 clocks.
    auto spans = [&](const std::string& lane) {
        std::vector<std::pair<std::string, double>> v;
        for (const auto& e : j["traceEvents"])
            if (e["ph"] == "X" && e["tid"].get<int>() == tid.at(lane)) v.push_back({e["name"], e["dur"].get<double>()});
        return v;
    };
    for (const char* lane : {"L1 A lane 0 (kpu)", "L1 B lane 3 (kpu)", "PE row 0 (kpu)", "PE row 3 (kpu)"}) {
        CAPTURE(lane);
        const auto v = spans(lane);
        REQUIRE(v.size() == plan.commands.size());
        for (std::size_t n = 0; n < v.size(); ++n) {
            CHECK(v[n].first == "cmd " + std::to_string(n));
            CHECK(std::fabs(v[n].second - 0.004) < 1e-9);
        }
    }
    // The result slot: each C tile filled by the Combiner, then read by its Writeback.
    std::set<std::string> slot_spans;
    for (std::uint32_t s = 0; s < plan.l2_slots; ++s)
        for (const auto& [name, dur] : spans("L2 slot " + std::to_string(s) + " (kpu)")) slot_spans.insert(name);
    CHECK(slot_spans.count("filling C[0,0,0]"));
    CHECK(slot_spans.count("C[0,0,0] read"));
    // The north FIFOs' occupancy as counters: the peak is the FIFO's, as the Combiner counted it.
    std::uint64_t peak = 0;
    for (const auto& e : j["traceEvents"])
        if (e["ph"] == "C" && e["tid"].get<int>() == tid.at("north FIFO 0 (kpu)"))
            peak = std::max<std::uint64_t>(peak, e["args"]["beats"].get<std::uint64_t>());
    CHECK(peak == r.fifo_max.at(0));
    CHECK(peak > 0);
}

TEST_CASE("The array heatmap: each PE's clocks by cause, as JSON and as an SVG", "[cycle][compute][heatmap]") {
    const auto dev = device("kpu_soc_2x2.json");
    ComputeRunOptions o;
    const CspComputePlan plan = plan_of(rtl_8x8x16(), dev, o.bus_bytes);
    const ComputeRun r = run_compute_subsystem(plan, dev, o);
    REQUIRE(r.ok());
    const std::string json = scratch("compute_heatmap.json"), svg = scratch("compute_heatmap.svg");
    write_heatmap_json(r, json);
    write_heatmap_svg(r, svg);
    std::ifstream in(json);
    const nlohmann::json h = nlohmann::json::parse(in);
    CHECK(h["format"] == "kpu-array-heatmap");
    CHECK(h["rows"] == 4);
    CHECK(h["cols"] == 4);
    CHECK(h["cycles"] == r.array_cycles);
    // A 4 x 4 tile fills the 4 x 4 array: every PE multiplies 16 commands x K = 4, none padded.
    for (std::uint32_t i = 0; i < 4; ++i)
        for (std::uint32_t j = 0; j < 4; ++j) {
            CHECK(h["useful"][i][j] == 64);
            CHECK(h["padded"][i][j] == 0);
            CHECK(h["useful"][i][j].get<std::uint64_t>() + h["padded"][i][j].get<std::uint64_t>() +
                      h["draining"][i][j].get<std::uint64_t>() + h["waiting"][i][j].get<std::uint64_t>() ==
                  r.array_cycles);
        }
    std::ifstream s(svg);
    const std::string text((std::istreambuf_iterator<char>(s)), std::istreambuf_iterator<char>());
    CHECK_THAT(text, ContainsSubstring("<svg"));
    CHECK_THAT(text, ContainsSubstring("PE(3,3): useful 64"));
}

TEST_CASE("A run that breaks the hardware's timing reports it, and is not ok", "[cycle][compute]") {
    // A read delay under the measured round trip: a row is not back from L2 by its load.
    const auto dev = device("kpu_soc_2x2.json");
    ComputeRunOptions o;
    o.read_delay = 12;
    const CspComputePlan plan = plan_of(rtl_8x8x16(), dev, o.bus_bytes);
    const ComputeRun r = run_compute_subsystem(plan, dev, o);
    CHECK(!r.ok());
    REQUIRE(r.violation);
    CHECK_THAT(*r.violation, ContainsSubstring("st"));
    CHECK_THAT(format_compute_summary(plan, r), ContainsSubstring("INVARIANT VIOLATED"));
}

TEST_CASE("The run refuses a program whose tiles do not fit the deployment's array", "[cycle][compute]") {
    // The 64^3's 16 x 16 tiles on kpu_soc_2x2's 4 x 4 array.
    const auto s1 = device("kpu_s1.json");
    const CspComputePlan plan = plan_of(int8_64cubed(), s1, 64);
    CHECK_THROWS_WITH(run_compute_subsystem(plan, device("kpu_soc_2x2.json"), ComputeRunOptions{}),
                      ContainsSubstring("past kpu_2x2_systolic.tile0's 4 x 4 array"));
}

TEST_CASE("The run refuses a zero-byte BlockMover beat before dividing by it", "[cycle][compute]") {
    // #547's review: row_bytes % bus_bytes with a zero beat is undefined behaviour.
    const auto dev = device("kpu_soc_2x2.json");
    const CspComputePlan plan = plan_of(rtl_8x8x16(), dev, 64);
    ComputeRunOptions o;
    o.bus_bytes = 0;
    CHECK_THROWS_WITH(run_compute_subsystem(plan, dev, o), ContainsSubstring("beat must be positive"));
}

TEST_CASE("A trace or heatmap whose last bytes cannot be flushed is a failed write", "[cycle][compute]") {
    // #547's review: the stream was checked before it was closed, so a failure flushing its buffer
    // passed unseen and a truncated file was published as written. /dev/full takes the open and
    // refuses every write: the small files below stay in the buffer until the close.
    if (!std::filesystem::exists("/dev/full")) SKIP("no /dev/full on this platform");
    const auto dev = device("kpu_soc_2x2.json");
    ComputeRunOptions o;
    const CspComputePlan plan = plan_of(rtl_8x8x16(), dev, o.bus_bytes);
    const ComputeRun r = run_compute_subsystem(plan, dev, o);
    CHECK_THROWS_WITH(write_heatmap_json(r, "/dev/full"), ContainsSubstring("failed"));
    CHECK_THROWS_WITH(write_heatmap_svg(r, "/dev/full"), ContainsSubstring("failed"));
    Tracer t;
    CHECK_THROWS_WITH(t.write("/dev/full"), ContainsSubstring("failed"));
}
