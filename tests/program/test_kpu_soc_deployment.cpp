// ============================================================================
// tests/program/test_kpu_soc_deployment.cpp
// The KPU RTL generator's machine as a kpu-sim deployment (#396): kpu_soc_2x2.json is one
// compute tile of the generator's kpu_2x2_systolic, converted by tools/deploy/
// kpuarch_to_deployment.py (a ctest keeps it in step with the vendored generator config). Its
// `systolic` block states the RTL tile's own configuration; the reader refuses it malformed, every
// level reports it as declared but not modelled, and a program runs on the deployment at L-B, L-T1
// and L-CA with the same values.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>

#include <fstream>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using Catch::Matchers::ContainsSubstring;
using driver::ExecutionLevel;

namespace {

const char* const kPath = "tests/program/deploy/kpu_soc_2x2.json";

std::string read(const std::string& path) {
    std::ifstream f(path);
    REQUIRE(f.good());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// The deployment's text with one replacement: the first `from` becomes `to`.
std::string edited(const std::string& from, const std::string& to) {
    std::string s = read(kPath);
    const auto at = s.find(from);
    REQUIRE(at != std::string::npos);
    return s.replace(at, from.size(), to);
}

std::string refusal(const std::string& text) {
    try {
        (void)platform::from_json(text);
    } catch (const platform::SpecError& e) {
        return e.what();
    }
    return "accepted";
}

}  // namespace

TEST_CASE("kpu_soc_2x2: the generator's tile, read with its systolic block", "[program][deploy][kpu_soc]") {
    const auto spec = platform::read_spec_file(kPath);
    const auto& d = spec.device(0);
    CHECK(d.name == "kpu_2x2_systolic.tile0");
    CHECK(d.compute_tiles == 1);
    CHECK(d.macs_per_cycle == 16.0);
    REQUIRE(d.systolic.has_value());
    const auto& y = *d.systolic;
    CHECK(y.dataflow == "weight_stationary");
    CHECK((y.input_bits == 8 && y.weight_bits == 8 && y.acc_bits == 32));
    CHECK((y.l2_slots == 4 && y.acc_slots == 2 && y.sp_banks == 4 && y.acc_banks == 2));
    CHECK((y.spad_read_delay == 4 && y.acc_latency == 2));
    CHECK((y.sp_rows_per_bank == 64 && y.acc_rows_per_bank == 64));
    CHECK(d.l3.capacity_tiles == 1024);
    // Written back, the block is the same: the canonical form carries it.
    const auto again = platform::from_json(platform::to_json(spec));
    REQUIRE(again.device(0).systolic.has_value());
    CHECK(again.device(0).systolic->dataflow == y.dataflow);
    CHECK(platform::deployment_digest(again) == platform::deployment_digest(spec));
}

TEST_CASE("kpu_soc_2x2: a malformed systolic block is refused by name", "[program][deploy][kpu_soc]") {
    CHECK_THAT(refusal(edited("\"dataflow\": \"weight_stationary\"", "\"dataflow\": \"row_stationary\"")),
               ContainsSubstring("systolic.dataflow is 'row_stationary'"));
    CHECK_THAT(refusal(edited("\"acc_bits\": 32", "\"acc_bits\": 24")), ContainsSubstring("systolic.acc_bits is 24"));
    CHECK_THAT(refusal(edited("\"l2_slots\": 4", "\"l2_slots\": 0")), ContainsSubstring("systolic.l2_slots must be non-zero"));
    CHECK_THAT(refusal(edited("\"grid_rows\": 4", "\"grid_rows\": 8")), ContainsSubstring("but macs_per_cycle is"));
    CHECK_THAT(refusal(edited("\"acc_latency\": 2,", "")), ContainsSubstring("systolic.acc_latency is required"));
    CHECK_THAT(refusal(edited("\"acc_latency\": 2,", "\"acc_latency\": 2, \"acc_ports\": 1,")),
               ContainsSubstring("acc_ports"));
}

TEST_CASE("kpu_soc_2x2: every level reports the systolic block as not modelled", "[program][deploy][kpu_soc]") {
    const auto spec = platform::read_spec_file(kPath);
    for (ExecutionLevel l : {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}) {
        CAPTURE(driver::to_string(l));
        bool named = false;
        for (const auto& u : driver::unmodelled_fields(l, spec)) named = named || u.rfind("systolic declared", 0) == 0;
        CHECK(named);
    }
    std::string why;
    const auto cfg = sw::kpu::timing::csp_config_from(spec.device(0), &why);
    CAPTURE(why);
    REQUIRE(cfg.has_value());
    bool noted = false;
    for (const auto& u : cfg->unmapped) noted = noted || u.find("systolic") != std::string::npos;
    CHECK(noted);
}

TEST_CASE("kpu_soc_2x2: an int8 program runs on the generator's tile at every level", "[program][deploy][kpu_soc]") {
    const auto spec = platform::read_spec_file(kPath);
    const std::string text =
        "csp 1.0\nprogram mm8 machine flat(l3 = 12) {\n"
        "  tensor A[8,8] int8 tile 4x4 in;\n  tensor B[8,8] int8 tile 4x4 in;\n  tensor C[8,8] int32 tile 4x4 out;\n"
        "  resident A[:, :], B[:, :];\n"
        "  for i in 0..2 {\n    for j in 0..2 {\n"
        "      acc C[i, j] in fabric {\n        for k in 0..2 { call gemm(A[i, k], B[k, j]) +-> C[i, j]; }\n      }\n"
        "      store C[i, j];\n    }\n  }\n"
        "  release A[:, :], B[:, :];\n}\n";
    const auto ast = csp::lang::parse(text);
    driver::CspRunRequest req{ast, &spec.device(0), driver::csp_inputs(ast),
                              {ExecutionLevel::Behavioral, ExecutionLevel::BlockSequential, ExecutionLevel::CycleAccurate}};
    const auto r = driver::run_csp(req);
    REQUIRE(r.reference);
    REQUIRE(r.levels.size() == 3);
    // Every level computes L-B's int32 result, word for word (an int32 word is not a float).
    const auto& want = r.reference->operand("C");
    for (const auto& o : r.levels) {
        CAPTURE(driver::to_string(o.level), o.skipped.value_or(""));
        REQUIRE_FALSE(o.skipped);
        const auto& got = o.values.operand("C");
        std::size_t differ = 0;
        for (Dim i = 0; i < want.rows; ++i)
            for (Dim j = 0; j < want.cols; ++j) differ += got.word(i, j) != want.word(i, j);
        CHECK(differ == 0);
    }
}
