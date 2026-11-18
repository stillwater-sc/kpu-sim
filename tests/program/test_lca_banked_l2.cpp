// ============================================================================
// tests/program/test_lca_banked_l2.cpp
// L-CA's banked L2 from the deployment (#401 slice 1; docs/plans/banked-l2-at-lca.md): the RTL
// tile's systolic block, else l2.banks_per_tile; a bank count that is not a power of two is
// refused (#383); one bank still computes L-B's values (#383's test_one_bank); banks time the
// transfers and never change a value (#388); the Streamer's reads are not stalled when it is the
// only reader of a bank (#388: two vectors per clock, under OS).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/timing/csp_config_from_spec.hpp>

#include <algorithm>
#include <string>

using namespace sw::kpu;
using program::platform::DeviceSpecification;
using program::platform::read_spec_file;
using program::driver::ExecutionLevel;
using Catch::Matchers::ContainsSubstring;

namespace {

DeviceSpecification s1() { return read_spec_file("tests/program/deploy/kpu_s1_combiner.json").device(0); }

bool lists(const std::vector<std::string>& v, const std::string& needle) {
    return std::any_of(v.begin(), v.end(), [&](const std::string& s) { return s.find(needle) != std::string::npos; });
}

program::driver::CspLevelOutcome lca(const DeviceSpecification& d, const std::string& dataflow = "os") {
    program::csp::gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 32;
    o.dataflow = dataflow;
    const auto ast = program::csp::lang::parse(program::csp::gen::generate(o));
    const auto out = program::driver::csp_run_level(ExecutionLevel::CycleAccurate, ast, &d,
                                                    program::driver::csp_inputs(ast));
    REQUIRE_FALSE(out.skipped);
    return out;
}

}  // namespace

TEST_CASE("Banked L2: l2.banks_per_tile banks L-CA's L2, its defaults named", "[program][lca][l2][banks]") {
    const auto cfg = timing::csp_config_from(s1());
    REQUIRE(cfg);
    REQUIRE(cfg->config.l2_banks);
    const auto& g = *cfg->config.l2_banks;
    CHECK(g.banks == 4);
    CHECK(g.row_bytes == 64 * 4);                       // the array's 64 columns of fp32
    CHECK(g.read_ports == 2);
    CHECK(g.write_ports == 1);
    CHECK(lists(cfg->unmapped, "l2.banks_per_tile: L2 is banked"));
    // Neither declared: the pooled L2, as before.
    auto pooled = s1();
    pooled.l2.banks_per_tile.reset();
    const auto p = timing::csp_config_from(pooled);
    REQUIRE(p);
    CHECK_FALSE(p->config.l2_banks);
}

TEST_CASE("Banked L2: the RTL's systolic block gives the scratchpad's geometry", "[program][lca][l2][banks]") {
    const auto d = read_spec_file("tests/program/deploy/kpu_soc_2x2.json").device(0);
    REQUIRE(d.systolic);
    const auto cfg = timing::csp_config_from(d);
    REQUIRE(cfg);
    REQUIRE(cfg->config.l2_banks);
    const auto& g = *cfg->config.l2_banks;
    CHECK(g.banks == d.systolic->sp_banks);
    CHECK(g.rows_per_bank == d.systolic->sp_rows_per_bank);
    CHECK(g.read_latency == d.systolic->spad_read_delay);
    CHECK(g.row_bytes == d.systolic->grid_cols * d.systolic->tile_cols * (d.systolic->input_bits / 8));
    CHECK(lists(cfg->unmapped, "systolic: L2 is banked from its scratchpad"));
}

TEST_CASE("Banked L2: a bank count that is not a power of two is refused (#383, esg#14)", "[program][lca][l2][banks]") {
    auto spec = read_spec_file("tests/program/deploy/kpu_s1.json");
    spec.devices[0].l2.banks_per_tile = 3;
    CHECK_THAT(spec.validate(), ContainsSubstring("l2.banks_per_tile is 3"));
    CHECK_THAT(spec.validate(), ContainsSubstring("power of two"));
    auto soc = read_spec_file("tests/program/deploy/kpu_soc_2x2.json");
    soc.devices[0].systolic->sp_banks = 6;
    CHECK_THAT(soc.validate(), ContainsSubstring("systolic.sp_banks is 6"));
}

TEST_CASE("Banked L2: one bank computes L-B's values (#383 test_one_bank)", "[program][lca][l2][banks]") {
    auto d = s1();
    d.l2.banks_per_tile = 1;
    const auto o = lca(d);
    REQUIRE(o.l2_banks);
    CHECK(o.l2_banks->bank_stalls.size() == 1);
    CHECK(o.l2_banks->accesses > 0);
    // The pooled L2's values, which every other test holds to L-B's.
    auto pooled = s1();
    pooled.l2.banks_per_tile.reset();
    const auto ref = lca(pooled);
    for (const auto& name : o.values.operand_order())
        CHECK(o.values.operand(name).values == ref.values.operand(name).values);
}

TEST_CASE("Banked L2: banks time the transfers, never the values (#388)", "[program][lca][l2][banks]") {
    for (const char* df : {"os", "ws", "is"}) {
        CAPTURE(df);
        auto pooled = s1();
        pooled.l2.banks_per_tile.reset();
        const auto a = lca(s1(), df), b = lca(pooled, df);
        REQUIRE(a.l2_banks);
        CHECK_FALSE(b.l2_banks);
        // Bit for bit: the same computation, timed differently.
        for (const auto& name : a.values.operand_order())
            CHECK(a.values.operand(name).values == b.values.operand(name).values);
        // Two vectors per clock: with the Streamer the only reader (OS: no merges), its reads never
        // wait, two read ports per bank for its two streams, the writes on their own port. Under WS
        // and IS the Combiner's read-modify-write reads too, and the first of them to book a bank's
        // port takes it, so either may wait, and the wait is counted. #388's priority for the
        // Streamer over the Combiner is a later refinement.
        if (std::string(df) == "os")
            CHECK(a.l2_banks->stalls[static_cast<std::size_t>(timing::L2Banks::Client::Streamer)] == 0);
    }
}
