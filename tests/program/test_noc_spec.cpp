// ============================================================================
// tests/program/test_noc_spec.cpp
// The deployment's NoC section (docs/plans/noc-port-arbitration.md step 2): store-and-forward
// hub buffering, the port controller's queues and arbitration -- declared, validated, written
// and read back, and reported as unmodelled by every level until the CSP processes land.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <functional>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::platform;
using Catch::Matchers::ContainsSubstring;

namespace {
DeploymentSpec t4() { return read_spec_file("tests/program/deploy/kpu_t4.json"); }

std::string spec_with(const std::string& noc) {
    return std::string(R"({"devices":[{"name":"d","topology":"checkerboard","compute_tiles":2,)") +
           R"("noc":)" + noc + "}]}";
}
} // namespace

TEST_CASE("the T4 declares its NoC, and it round-trips", "[program][platform][noc]") {
    const DeploymentSpec s = t4();
    REQUIRE(s.validate().empty());
    REQUIRE(s.device(0).noc.has_value());
    const auto& n = *s.device(0).noc;
    CHECK(n.hub_buffer_blocks == 4);
    CHECK(n.port.input_queue_blocks == 2);
    CHECK(n.port.output_queue_blocks == 0);          // derived from the DMA write latency
    CHECK(n.port.arbitration == "ring_first_oldest");

    const std::string text = to_json(s);
    CHECK(to_json(from_json(text)) == text);

    // Not declared, not written: an old spec keeps its bytes and its digest.
    DeploymentSpec plain = s;
    plain.device(0).noc.reset();
    CHECK_THAT(to_json(plain), !ContainsSubstring("\"noc\""));
}

TEST_CASE("an empty NoC section takes the plan's defaults, and writes them out",
          "[program][platform][noc]") {
    const DeploymentSpec s = from_json(spec_with("{}"));
    REQUIRE(s.validate().empty());
    const auto& n = *s.device(0).noc;
    CHECK(n.hub_buffer_blocks == 4);
    CHECK(n.port.input_queue_blocks == 2);
    CHECK(n.port.output_queue_blocks == 0);
    CHECK(n.port.arbitration == "ring_first_oldest");
    // A round trip states the defaults it relied on, so the bytes say what was assumed.
    CHECK_THAT(to_json(s), ContainsSubstring("\"hub_buffer_blocks\": 4"));
    CHECK_THAT(to_json(s), ContainsSubstring("\"arbitration\": \"ring_first_oldest\""));
}

TEST_CASE("a NoC that cannot work is refused with the field's own words",
          "[program][platform][noc]") {
    using Edit = std::function<void(DeviceSpecification&)>;
    const std::vector<std::pair<Edit, std::string>> cases = {
        {[](auto& d) { d.noc->hub_buffer_blocks = 3; }, "hub_buffer_blocks (3) must be at least the hub's 4 inputs"},
        {[](auto& d) { d.noc->port.input_queue_blocks = 0; }, "input_queue_blocks must be at least 1"},
        {[](auto& d) { d.noc->port.arbitration = "round_robin"; }, "arbitration 'round_robin'"},
        {[](auto& d) { d.topology = "news"; d.array.rows.reset(); d.array.cols.reset(); },
         "noc applies to the checkerboard"},
    };
    for (const auto& [edit, want] : cases) {
        DeploymentSpec s = t4();
        edit(s.device(0));
        INFO(want);
        CHECK_THAT(s.validate(), ContainsSubstring(want));
    }
    // The boundary itself is legal: four buffers is exactly one per input.
    DeploymentSpec ok = t4();
    ok.device(0).noc->hub_buffer_blocks = 4;
    CHECK(ok.validate().empty());
}

TEST_CASE("a NoC section in JSON is refused when it cannot be read as one",
          "[program][platform][noc]") {
    CHECK_THROWS_WITH(from_json(spec_with(R"({"hub_buffers":4})")),
                      ContainsSubstring("unknown key 'hub_buffers'"));
    CHECK_THROWS_WITH(from_json(spec_with(R"({"port":{"queue":2}})")),
                      ContainsSubstring("unknown key 'queue'"));
    CHECK_THROWS_WITH(from_json(spec_with(R"({"hub_buffer_blocks":-4})")),
                      ContainsSubstring("must not be negative"));
    CHECK_THROWS_WITH(from_json(spec_with(R"({"port":{"arbitration":3}})")),
                      ContainsSubstring("must be a string"));
}

TEST_CASE("every level admits it does not model the declared NoC yet",
          "[program][platform][noc]") {
    const DeploymentSpec s = t4();
    for (auto l : {driver::ExecutionLevel::Behavioral, driver::ExecutionLevel::BlockSequential,
                   driver::ExecutionLevel::CycleAccurate}) {
        bool found = false;
        for (const std::string& line : driver::unmodelled_fields(l, s))
            if (line.rfind("noc declared (hub 4 blocks, port in 2 / out derived per engine, "
                           "ring_first_oldest)", 0) == 0)
                found = true;
        CHECK(found);
    }
}
