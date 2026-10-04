// ============================================================================
// tests/program/test_dram_address_map.cpp
// The DRAM geometry in the deployment spec and the one address map derived from it
// (docs/plans/dram-bank-model.md step 1): a bijection on [0, capacity), XOR folds that spread
// rows across banks, refusals that name the field, and a JSON form that round-trips.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/dram_address_map.hpp>

#include <functional>
#include <set>
#include <string>
#include <tuple>

using namespace sw::kpu::program;
using namespace sw::kpu::program::platform;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kDeploy = "tests/program/deploy/";

DeploymentSpec t64() { return read_spec_file(std::string(kDeploy) + "kpu_t64.json"); }

// A geometry small enough to enumerate: every field at least one bit wide, 1024 bursts.
DeviceSpecification tiny(const std::string& map, bool folded) {
    DeviceSpecification d;
    d.memory.controllers = 2;
    DeviceSpecification::Memory::Dram m;
    m.channels = 2;
    m.ranks = 2;
    m.bank_groups = 2;
    m.banks_per_group = 2;
    m.page_bytes = 256;
    m.burst_bytes = 64;
    m.capacity_bytes = 1u << 16;
    m.map = map;
    if (folded) {
        m.xor_folds.push_back({"ba", "ro", 1, 0});
        m.xor_folds.push_back({"mc", "ro", 1, 2});
    }
    d.memory.dram = m;
    return d;
}

auto key(const DramCoord& c) {
    return std::make_tuple(c.mc, c.channel, c.rank, c.bank_group, c.bank, c.row, c.col);
}

} // namespace

TEST_CASE("the T64 map has the declared geometry, bit for bit", "[program][platform][dram]") {
    const DramAddressMap m = DramAddressMap::of(t64().device(0));
    CHECK(m.capacity() == (std::uint64_t{16} << 30));
    CHECK(m.controllers() == 4);
    CHECK(m.channels() == 2);
    CHECK(m.ranks() == 1);
    CHECK(m.bank_groups() * m.banks_per_group() == 16);
    CHECK(m.burst_bytes() == 64);
    CHECK(m.bursts_per_page() == 32);
    // 16 GiB over 4 controllers x 2 channels x 16 banks x 2 KiB pages = 64K rows, which is
    // a 16 Gb x16 LPDDR5X die per channel.
    CHECK(m.rows() == 65536);
    CHECK(m.describe() == "off[0,6) co[6,11) ch[11,12) bg[12,14) ba[14,16) mc[16,18) ro[18,34)"
                          " xor ba[0,2)^=ro[0,2) xor bg[0,2)^=ro[2,4)");
}

TEST_CASE("the T64 map is linear below the row", "[program][platform][dram]") {
    const DramAddressMap m = DramAddressMap::of(t64().device(0));
    // Row 0, so the folds contribute nothing and the fields read straight off the address.
    const std::uint64_t a = (std::uint64_t{3} << 16) | (2u << 14) | (1u << 12) | (1u << 11) |
                            (5u << 6) | 17;
    const DramCoord c = m.decode(a);
    CHECK(c.mc == 3);
    CHECK(c.bank == 2);
    CHECK(c.bank_group == 1);
    CHECK(c.channel == 1);
    CHECK(c.col == 5);
    CHECK(c.row == 0);
    CHECK(m.flat_bank(c) == 1 * 4 + 2);
    CHECK(m.encode(c, 17) == a);
}

TEST_CASE("the fold spreads rows that a linear map would stack on one bank",
          "[program][platform][dram]") {
    // The same linear bank bits, sixteen consecutive rows: a strided tensor's worst case.
    // Linear, they are sixteen different rows of ONE bank -- a page conflict per access.
    // Folded, they land in sixteen different banks.
    const DramAddressMap folded = DramAddressMap::of(t64().device(0));
    DeviceSpecification lin = t64().device(0);
    lin.memory.dram->xor_folds.clear();
    const DramAddressMap linear = DramAddressMap::of(lin);

    std::set<Dim> banks_folded, banks_linear;
    for (std::uint64_t row = 0; row < 16; ++row) {
        const std::uint64_t a = row << 18;
        banks_folded.insert(folded.flat_bank(folded.decode(a)));
        banks_linear.insert(linear.flat_bank(linear.decode(a)));
        CHECK(folded.decode(a).row == row);   // the fold moves the bank, never the row
    }
    CHECK(banks_linear.size() == 1);
    CHECK(banks_folded.size() == 16);
}

TEST_CASE("every burst decodes to a distinct coordinate and encodes back",
          "[program][platform][dram]") {
    for (const std::string& map : {std::string("co:ch:rk:bg:ba:mc:ro"),
                                   std::string("ro:co:mc:ba:bg:rk:ch")})
        for (bool folded : {false, true}) {
            INFO("map " << map << (folded ? " folded" : " linear"));
            const DeviceSpecification d = tiny(map, folded);
            const DramAddressMap m = DramAddressMap::of(d);
            const std::uint64_t bursts = m.capacity() / m.burst_bytes();
            REQUIRE(bursts == 1024);
            std::set<decltype(key(DramCoord{}))> seen;
            for (std::uint64_t i = 0; i < bursts; ++i) {
                const std::uint64_t a = i * m.burst_bytes() + (i % m.burst_bytes());
                const DramCoord c = m.decode(a);
                seen.insert(key(c));
                REQUIRE(m.encode(c, a % m.burst_bytes()) == a);
            }
            // A bijection: as many coordinates as bursts, so every coordinate is reachable.
            CHECK(seen.size() == bursts);
        }
}

TEST_CASE("an address or coordinate outside the map is refused", "[program][platform][dram]") {
    const DramAddressMap m = DramAddressMap::of(t64().device(0));
    CHECK_THROWS_WITH(m.decode(m.capacity()), ContainsSubstring("past the top of memory"));
    DramCoord c;
    c.mc = 4;
    CHECK_THROWS_WITH(m.encode(c), ContainsSubstring("mc = 4 does not fit"));
    CHECK_THROWS_WITH(m.encode(DramCoord{}, 64), ContainsSubstring("not within one burst"));
    CHECK_THROWS_WITH(DramAddressMap::of(DeviceSpecification{}),
                      ContainsSubstring("declares no memory.dram"));
}

TEST_CASE("an inconsistent DRAM is refused with the field's own words",
          "[program][platform][dram]") {
    using Edit = std::function<void(DeviceSpecification&)>;
    const std::vector<std::pair<Edit, std::string>> cases = {
        {[](auto& d) { d.memory.dram->channels = 3; }, "channels (3) must be a non-zero power of two"},
        {[](auto& d) { d.memory.dram->capacity_bytes = 0; }, "capacity_bytes (0) must be"},
        {[](auto& d) { d.memory.controllers = 3; d.dma.engines = 3; }, "memory.controllers (3) must be a power of two"},
        {[](auto& d) { d.memory.dram->technology = "sram"; }, "technology 'sram'"},
        {[](auto& d) { d.memory.dram->page_bytes = 32; }, "smaller than a burst"},
        {[](auto& d) { d.memory.dram->map = "co:ch:bg:ba:mc:ro"; }, "exactly once"},
        {[](auto& d) { d.memory.dram->map = "co:co:rk:bg:ba:mc:ro"; }, "exactly once"},
        {[](auto& d) { d.memory.dram->capacity_bytes = 1u << 16; }, "smaller than one row"},
        {[](auto& d) { d.dma.burst_bytes = 96; }, "whole number of DRAM bursts"},
        {[](auto& d) { d.memory.dram->xor_folds[0].into = "xx"; }, "unknown field"},
        {[](auto& d) { d.memory.dram->xor_folds[0].from = "ba"; }, "folds a field into itself"},
        {[](auto& d) { d.memory.dram->xor_folds[0].bits = 3; }, "folds 3 bits into 'ba', which has 2"},
        {[](auto& d) { d.memory.dram->xor_folds[1].from_lsb = 15; }, "reads bits [15, 17) of 'ro'"},
        {[](auto& d) { auto& f = d.memory.dram->xor_folds[1]; f.from = "ba"; f.from_lsb = 0; }, "never both"},
    };
    for (const auto& [edit, want] : cases) {
        DeploymentSpec s = t64();
        edit(s.device(0));
        INFO(want);
        CHECK_THAT(s.validate(), ContainsSubstring(want));
        CHECK_THROWS_AS(DramAddressMap::of(s.device(0)), DramMapError);
    }
    CHECK(t64().validate().empty());
}

TEST_CASE("the DRAM round-trips through JSON and is absent unless declared",
          "[program][platform][dram]") {
    const DeploymentSpec s = t64();
    REQUIRE(s.device(0).memory.dram.has_value());
    const std::string text = to_json(s);
    const DeploymentSpec back = from_json(text);
    CHECK(to_json(back) == text);
    CHECK(DramAddressMap::of(back.device(0)).describe() ==
          DramAddressMap::of(s.device(0)).describe());

    // Not declared, not written: an old spec keeps its bytes and its digest.
    DeploymentSpec plain = s;
    plain.device(0).memory.dram.reset();
    CHECK_THAT(to_json(plain), !ContainsSubstring("dram"));
}

TEST_CASE("a DRAM in JSON is refused when it cannot be read as one",
          "[program][platform][dram]") {
    auto with = [](const std::string& dram) {
        return std::string(R"({"devices":[{"name":"d","memory":{"controllers":1,"dram":)") +
               dram + "}}]}";
    };
    CHECK_THROWS_WITH(from_json(with(R"({"channels":2})")),
                      ContainsSubstring("capacity_bytes is required"));
    CHECK_THROWS_WITH(from_json(with(R"({"capacity_bytes":-1})")),
                      ContainsSubstring("must not be negative"));
    CHECK_THROWS_WITH(from_json(with(R"({"capacity_bytes":1.5})")),
                      ContainsSubstring("must be an integer"));
    CHECK_THROWS_WITH(from_json(with(R"({"capacity_bytes":4096,"bankz":4})")),
                      ContainsSubstring("unknown key 'bankz'"));
    CHECK_THROWS_WITH(from_json(with(R"({"capacity_bytes":4096,"xor_folds":{}})")),
                      ContainsSubstring("xor_folds must be an array"));
    CHECK_THROWS_WITH(from_json(with(R"({"capacity_bytes":4096,"xor_folds":[{"into":"ba"}]})")),
                      ContainsSubstring("needs into, from and bits"));
    // Past 4 GiB, which a 32-bit count could not hold.
    const DeploymentSpec big = from_json(with(R"({"capacity_bytes":17179869184})"));
    CHECK(big.device(0).memory.dram->capacity_bytes == (std::uint64_t{16} << 30));
}

TEST_CASE("every level admits it does not model the declared DRAM yet",
          "[program][platform][dram]") {
    // Until the controller schedules on banks (plan step 2), a declared geometry is a
    // statement no level acts on -- and a run must say so rather than look complete.
    const DeploymentSpec s = t64();
    for (auto l : {driver::ExecutionLevel::Behavioral, driver::ExecutionLevel::BlockSequential,
                   driver::ExecutionLevel::CycleAccurate}) {
        const auto u = driver::unmodelled_fields(l, s);
        bool found = false;
        for (const std::string& line : u)
            if (line.rfind("memory.dram declared (lpddr5x, 4 mc x 2 ch x 16 banks", 0) == 0)
                found = true;
        CHECK(found);
    }
}
