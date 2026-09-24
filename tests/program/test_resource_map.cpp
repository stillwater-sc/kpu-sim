// ============================================================================
// tests/program/test_resource_map.cpp
// The global naming map (#282 increment 3): identity now, state binding in #283.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/platform/resource_map.hpp>

#include <set>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::platform;

namespace {

// Two devices, fully declared, so every kind of resource has something to name. This is
// the deployment the issue's definition of done asks about: "an L3 tile, an L2 bank, an L1
// vector and a compute-tile register file, on a two-device deployment".
DeploymentSpec two_devices() {
    DeviceSpecification a;
    a.name = "left";
    a.topology = "checkerboard";
    a.compute_tiles = 4;
    a.l3.tiles = 2;
    a.l3.banks = 8;
    a.l3.capacity_tiles = 32;
    a.l2.banks_per_tile = 8;
    a.l1.vectors = 4;

    DeviceSpecification b;
    b.name = "right";
    b.topology = "single";
    b.compute_tiles = 1;
    b.l3.tiles = 1;
    b.l3.banks = 2;
    b.l2.banks_per_tile = 2;
    b.l1.vectors = 1;

    DeploymentSpec spec;
    spec.devices = {a, b};
    return spec;
}

} // namespace

TEST_CASE("the definition of done: four kinds of resource, two devices",
          "[program][platform][names]") {
    const ResourceMap map(two_devices());
    for (const char* text : {"left/l3[1]", "left/cf[3]/l2[7]", "left/cf[0]/l1[3]",
                             "left/cf[2]/regs", "left/dram", "left/l3[0]/bank[7]",
                             "right/l3[0]", "right/cf[0]/l2[1]", "right/cf[0]/l1[0]",
                             "right/cf[0]/regs"}) {
        INFO("name " << text);
        CHECK_NOTHROW(map.require(text));
    }
    // Both devices are reachable, and by NAME rather than by position -- a positional
    // address would silently move when a deployment is reordered.
    CHECK(map.exists(map.require("right/cf[0]/regs")));
    CHECK_FALSE(map.exists(parse_resource_name("middle/dram")));
}

TEST_CASE("a name round-trips through text", "[program][platform][names]") {
    const ResourceMap map(two_devices());
    for (const ResourceName& n : map.enumerate()) {
        const std::string text = format(n);
        INFO("name " << text);
        const ResourceName back = parse_resource_name(text);
        CHECK(back == n);
        CHECK(format(back) == text);
    }
    // An offset survives, and a zero offset has ONE spelling -- otherwise "dev/l3[0]" and
    // "dev/l3[0]+0" would be the same resource under two names, and anything keyed on the
    // text would hold it twice.
    ResourceName with_offset = map.require("left/cf[1]/l2[2]");
    with_offset.offset = 64;
    CHECK(format(with_offset) == "left/cf[1]/l2[2]+64");
    CHECK(parse_resource_name("left/cf[1]/l2[2]+64").offset == 64);
    CHECK(format(map.require("left/l3[0]")) == "left/l3[0]");
    CHECK(parse_resource_name("left/l3[0]+0").offset == 0);
    CHECK(format(parse_resource_name("left/l3[0]+0")) == "left/l3[0]");
}

TEST_CASE("the map's domain is exactly what the deployment declares",
          "[program][platform][names]") {
    // The discipline of increment 1, carried through: an undeclared field means the machine's
    // structure is unspecified there, so a name for it NAMES NOTHING. Resolving it against a
    // default would hand the backdoor an address for a resource nobody described.
    DeploymentSpec spec;
    spec.device(0).compute_tiles = 2;            // declared, always
    const ResourceMap bare(spec);

    // Compute tiles, their register files, and DRAM are declared by any device.
    CHECK(bare.exists(parse_resource_name("dev0/cf[0]")));
    CHECK(bare.exists(parse_resource_name("dev0/cf[1]/regs")));
    CHECK(bare.exists(parse_resource_name("dev0/dram")));
    // Everything the spec is silent about does NOT resolve...
    CHECK_FALSE(bare.exists(parse_resource_name("dev0/l3[0]")));
    CHECK_FALSE(bare.exists(parse_resource_name("dev0/l3[0]/bank[0]")));
    CHECK_FALSE(bare.exists(parse_resource_name("dev0/cf[0]/l2[0]")));
    CHECK_FALSE(bare.exists(parse_resource_name("dev0/cf[0]/l1[0]")));

    // ...and the map contains exactly the declared set, counted rather than sampled: 1 DRAM
    // + 2 compute tiles + 2 register files.
    CHECK(bare.size() == 5);

    // Declaring the fields makes the names resolve, and nothing else changed.
    spec.device(0).l3.tiles = 2;
    spec.device(0).l2.banks_per_tile = 3;
    spec.device(0).l1.vectors = 2;
    const ResourceMap declared(spec);
    CHECK(declared.exists(parse_resource_name("dev0/l3[1]")));
    CHECK(declared.exists(parse_resource_name("dev0/cf[1]/l2[2]")));
    CHECK(declared.exists(parse_resource_name("dev0/cf[0]/l1[1]")));
    // Banks are still undeclared, so an L3 bank still names nothing even though L3 modules
    // now exist. Declaring one level does not imply the next.
    CHECK_FALSE(declared.exists(parse_resource_name("dev0/l3[1]/bank[0]")));
    // 1 dram + 2 l3 + 2 cf + 2 regs + 2*3 l2 + 2*2 l1 = 17
    CHECK(declared.size() == 17);
}

TEST_CASE("the map enumerates every declared resource and nothing twice",
          "[program][platform][names]") {
    // Enumeration is what #286 needs to lay out its stations, so it has to be complete and
    // free of duplicates -- a station shown twice is occupancy counted twice.
    const ResourceMap map(two_devices());

    std::set<std::string> seen;
    for (const ResourceName& n : map.enumerate())
        CHECK(seen.insert(format(n)).second);
    CHECK(seen.size() == map.size());

    // Counted from the spec rather than pinned to a literal, so the arithmetic is visible:
    //   left:  1 dram + 2 l3 + 2*8 l3 banks + 4 cf + 4 regs + 4*8 l2 + 4*4 l1 = 75
    //   right: 1 dram + 1 l3 + 1*2 l3 banks + 1 cf + 1 regs + 1*2 l2 + 1*1 l1 = 9
    const DeploymentSpec spec = two_devices();
    std::size_t expected = 0;
    for (const DeviceSpecification& d : spec.devices) {
        expected += 1;                                            // dram
        if (d.l3.tiles) {
            expected += *d.l3.tiles;                              // modules
            if (d.l3.banks) expected += *d.l3.tiles * *d.l3.banks;
        }
        expected += d.compute_tiles * 2;                          // tile + register file
        if (d.l2.banks_per_tile) expected += d.compute_tiles * *d.l2.banks_per_tile;
        if (d.l1.vectors) expected += d.compute_tiles * *d.l1.vectors;
    }
    CHECK(map.size() == expected);
    CHECK(map.size() == 84);

    // Every kind that the deployment declares actually appears -- a map that enumerated
    // only the easy kinds would pass every count above if the count were wrong the same way.
    std::set<int> kinds;
    for (const ResourceName& n : map.enumerate()) kinds.insert(static_cast<int>(n.kind));
    CHECK(kinds.size() == all_resource_kinds().size());
}

TEST_CASE("a dense index identifies a resource, and ignores the offset",
          "[program][platform][names]") {
    // #286 keys event records on a small integer rather than re-formatting a string per
    // event. An OFFSET is not part of identity: two writes at different offsets are two
    // writes to the SAME resource, and counting them as two stations would be wrong.
    const ResourceMap map(two_devices());
    const ResourceName bank = map.require("left/cf[2]/l2[5]");
    ResourceName deeper = bank;
    deeper.offset = 4096;
    REQUIRE(map.index_of(bank).has_value());
    CHECK(map.index_of(deeper) == map.index_of(bank));
    CHECK(map.exists(deeper));

    // The index is dense and addresses the enumeration.
    for (std::size_t i = 0; i < map.size(); ++i)
        CHECK(map.index_of(map.enumerate()[i]) == i);
}

TEST_CASE("a name that does not resolve says why, and undeclared is not out-of-range",
          "[program][platform][names]") {
    // "l3.banks is not declared" and "there are only 8 banks" are different problems with
    // different fixes. Collapsing them into one message sends the reader to the wrong one.
    DeploymentSpec spec;
    spec.device(0).compute_tiles = 2;
    spec.device(0).l3.tiles = 2;
    const ResourceMap map(spec);

    CHECK(map.why_not(parse_resource_name("dev0/l3[0]/bank[0]")).find("not declared") !=
          std::string::npos);
    CHECK(map.why_not(parse_resource_name("dev0/cf[0]/l2[0]")).find("not declared") !=
          std::string::npos);
    CHECK(map.why_not(parse_resource_name("dev0/cf[0]/l1[0]")).find("not declared") !=
          std::string::npos);

    // Out of range is reported with the count, so the reader can see what they had.
    const std::string too_far = map.why_not(parse_resource_name("dev0/l3[9]"));
    CHECK(too_far.find("2") != std::string::npos);
    CHECK(too_far.find("not declared") == std::string::npos);
    const std::string no_tile = map.why_not(parse_resource_name("dev0/cf[7]"));
    CHECK(no_tile.find("compute tile") != std::string::npos);

    // An unknown device names the ones that exist rather than leaving the caller guessing.
    const std::string no_device = map.why_not(parse_resource_name("elsewhere/dram"));
    CHECK(no_device.find("dev0") != std::string::npos);

    // A resolving name has no complaint, and require() throws with the diagnosis rather
    // than with nothing.
    CHECK(map.why_not(parse_resource_name("dev0/l3[1]")).empty());
    CHECK_THROWS_AS(map.require("dev0/l3[9]"), NameError);
    try {
        map.require("dev0/cf[0]/l2[0]");
        FAIL("an undeclared resource must be refused");
    } catch (const NameError& e) {
        CHECK(std::string(e.what()).find("l2.banks_per_tile") != std::string::npos);
    }
}

TEST_CASE("a malformed address is refused with the text in the message",
          "[program][platform][names]") {
    for (const char* bad : {"",                       // nothing
                            "dev0",                   // no resource
                            "/dram",                  // no device
                            "dev0/",                  // empty resource
                            "dev0/l4[0]",             // no such resource
                            "dev0/l3",                // missing index
                            "dev0/l3[]",              // empty index
                            "dev0/l3[x]",             // not a number
                            "dev0/l3[-1]",            // must not wrap to a huge index
                            "dev0/l3[99999999999]",   // out of Dim range
                            "dev0/l3[0]/vector[0]",   // wrong child
                            "dev0/cf[0]/l9[0]",       // wrong child
                            "dev0/dram/bank[0]",      // dram has no children
                            "dev0/cf[0]/l2[0]/x[1]",  // too deep
                            "dev0/l3[0]+",            // '+' with no offset
                            "dev0/l3[0]+x"}) {        // offset not a number
        INFO("address '" << bad << "'");
        CHECK_THROWS_AS(parse_resource_name(bad), NameError);
    }
    // The message carries the offending text, because a caller with a list of names needs
    // to know WHICH one was wrong.
    try {
        parse_resource_name("dev0/l3[x]");
        FAIL("must throw");
    } catch (const NameError& e) {
        CHECK(std::string(e.what()).find("dev0/l3[x]") != std::string::npos);
    }
}

TEST_CASE("a mis-shaped name cannot be formatted", "[program][platform][names]") {
    // path_arity() is stated once so the parser, formatter and resolver cannot disagree.
    // A hand-built name with the wrong path length is a programming error, and formatting it
    // would produce an address that parses back as something else.
    ResourceName wrong;
    wrong.device = "dev0";
    wrong.kind = ResourceKind::L2Bank;
    wrong.path = {1};                            // needs {cf, bank}
    CHECK_THROWS_AS(format(wrong), NameError);
    CHECK_FALSE(ResourceMap(DeploymentSpec{}).exists(wrong));

    ResourceName too_many;
    too_many.device = "dev0";
    too_many.kind = ResourceKind::Dram;
    too_many.path = {0};                         // dram takes none
    CHECK_THROWS_AS(format(too_many), NameError);
}

TEST_CASE("a device name that cannot be addressed cannot be deployed",
          "[program][platform][names]") {
    // A device name IS part of an address, so a name containing the grammar would be
    // unparseable -- and learning that at the first backdoor write (#284) is far worse than
    // learning it at deployment.
    for (const char* bad : {"dev/0", "dev[0]", "dev+0", "a/b"}) {
        DeploymentSpec spec;
        spec.device(0).name = bad;
        INFO("device name '" << bad << "'");
        CHECK_FALSE(spec.validate().empty());
        CHECK_THROWS_AS(ResourceMap(spec), NameError);
    }
    // An ordinary name with punctuation the grammar does not use is fine.
    DeploymentSpec ok;
    ok.device(0).name = "kpu-0_east.2";
    CHECK(ok.validate().empty());
    const ResourceMap map(ok);
    CHECK(map.exists(parse_resource_name("kpu-0_east.2/dram")));
}

TEST_CASE("an impossible deployment has no map", "[program][platform][names]") {
    DeploymentSpec zero;
    zero.device(0).compute_tiles = 0;
    CHECK_THROWS_AS(ResourceMap(zero), NameError);
}
