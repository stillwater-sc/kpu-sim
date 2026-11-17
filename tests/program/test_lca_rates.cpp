// ============================================================================
// tests/program/test_lca_rates.cpp
// L-CA takes the deployment's rates (#458): the BlockMovers' and the Streamers', as GB/s at the
// executor's reference clock, and each moves L-CA's makespan. The DMA's reaches L-CA through a
// hosted DRAM; without one it is named as unmodelled. What L-CA does not read is named, not hidden.
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
using Catch::Matchers::ContainsSubstring;

namespace {

DeviceSpecification s1() { return read_spec_file("tests/program/deploy/kpu_s1.json").device(0); }

// The same device with no hosted DRAM: the controllers' channel rate is then the DMA's.
DeviceSpecification s1_no_dram() {
    DeviceSpecification d = s1();
    d.memory.dram.reset();
    d.dma.window.reset();
    return d;
}

bool lists(const std::vector<std::string>& v, const std::string& needle) {
    return std::any_of(v.begin(), v.end(), [&](const std::string& s) { return s.find(needle) != std::string::npos; });
}

std::uint64_t lca_makespan(const DeviceSpecification& d) {
    program::csp::gen::Options o;
    o.algo = "matmul";
    o.size = 64;
    o.tile = 16;
    o.l3 = 16;
    const auto ast = program::csp::lang::parse(program::csp::gen::generate(o));
    const auto inputs = program::driver::csp_inputs(ast);
    const auto out = program::driver::csp_run_level(program::driver::ExecutionLevel::CycleAccurate, ast, &d, inputs);
    REQUIRE_FALSE(out.skipped);
    return out.makespan;
}

}  // namespace

TEST_CASE("L-CA rates: the movers' and the DMA's bytes per cycle are the executor's GB/s", "[program][lca][rates]") {
    const DeviceSpecification d = s1_no_dram();
    const auto cfg = timing::csp_config_from(d);
    REQUIRE(cfg);
    const auto& c = cfg->config;
    REQUIRE(c.clock_ghz == 1.0);
    CHECK(c.bm_bandwidth_gbps == d.movers.bm_bytes_per_cycle * c.clock_ghz);
    CHECK(c.str_bandwidth_gbps == d.movers.str_bytes_per_cycle * c.clock_ghz);
    // Each controller's channel: its engines at the engine rate (its statistics). Without a hosted
    // DRAM the legacy controller times a transfer by latency, not size, and kpu-run says so.
    const std::size_t mcs = d.memory.controllers ? *d.memory.controllers : 1;
    CHECK(c.mc_bandwidth_gbps == static_cast<double>(d.dma.engines / mcs) * d.dma.bytes_per_cycle * c.clock_ghz);
    CHECK(lists(cfg->unmapped, "dma.bytes_per_cycle: no memory.dram"));
    CHECK_FALSE(lists(cfg->unmapped, "movers."));
    // Still not read, and named (with #401 and #388/#417).
    CHECK(lists(cfg->unmapped, "l2.banks_per_tile"));
}

TEST_CASE("L-CA rates: with a hosted DRAM, its model sets the transfers, and kpu-run says so", "[program][lca][rates]") {
    const auto cfg = timing::csp_config_from(s1());
    REQUIRE(cfg);
    REQUIRE(cfg->config.dram);
    CHECK(lists(cfg->unmapped, "dma.bytes_per_cycle: memory.dram is hosted"));
    CHECK(cfg->config.bm_bandwidth_gbps == s1().movers.bm_bytes_per_cycle);
}

TEST_CASE("L-CA rates: each rate moves L-CA's makespan", "[program][lca][rates]") {
    // Halving a rate the matmul's movement depends on must not leave L-CA's makespan unchanged:
    // the rate reaches the executor. (Before #458 each was an executor default.)
    const DeviceSpecification base = s1_no_dram();
    const std::uint64_t t0 = lca_makespan(base);
    REQUIRE(t0 > 0);
    SECTION("the BlockMovers'") {
        DeviceSpecification d = base;
        d.movers.bm_bytes_per_cycle /= 8.0;
        CHECK(lca_makespan(d) > t0);
    }
    SECTION("the Streamers'") {
        DeviceSpecification d = base;
        d.movers.str_bytes_per_cycle /= 8.0;
        CHECK(lca_makespan(d) > t0);
    }
    // The DMA's reaches L-CA through the hosted DRAM, which is rate-true; without one it is named
    // as unmodelled (above), so it is not expected to move the makespan here.
}
