// ============================================================================
// tests/program/test_resource_map.cpp
// The global naming map (#282 increment 3): identity now, state binding in #283.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/platform/resource_map.hpp>

#include <cstdint>
#include <limits>
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
    // The largest offset that fits is accepted, so the bound is exact rather than
    // conservative -- an off-by-one there would reject a legal address.
    CHECK(parse_resource_name("left/l3[0]+18446744073709551615").offset ==
          std::numeric_limits<std::uint64_t>::max());
    CHECK_THROWS_AS(parse_resource_name("left/l3[0]+18446744073709551616"), NameError);
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
    //          + 1 BlockMover: a single layout (one L3 beside one compute tile) has one
    //          L3 edge that abuts a compute tile. "left" has no layout -- 2 L3 tiles cannot
    //          alternate with 4 compute tiles -- so it gains nothing.
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
        if (const auto L = ArrayLayout::of(d)) expected += L->block_movers().size();
    }
    CHECK(map.size() == expected);
    CHECK(map.size() == 85);

    // Every kind that the deployment declares actually appears -- a map that enumerated
    // only the easy kinds would pass every count above if the count were wrong the same way.
    // This fixture declares no memory controllers, CPU or torus, so those kinds are absent;
    // the T64 test below declares everything and requires every kind.
    std::set<int> kinds;
    for (const ResourceName& n : map.enumerate()) kinds.insert(static_cast<int>(n.kind));
    CHECK(kinds.size() == 8);   // the seven original kinds + BlockMover
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
                            "dev0/l3[0]+x",           // offset not a number
                            // WRAPS PAST 2^64 WITHOUT EXCEEDING THE PREVIOUS VALUE. The
                            // first overflow test compared v*10+d against v, which is not an
                            // overflow test: 3689348814741910323*10 wraps to a LARGER
                            // number, so this parsed clean and produced a wrong offset --
                            // the exact failure a checked parser exists to prevent.
                            "dev0/dram+36893488147419103230",
                            "dev0/dram+99999999999999999999999"}) {
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


// ---- the physical shape (#286 step 1) -----------------------------------------------
namespace {

// The KPU-T64 as kpu-architecture.md §5.2.1 now describes it: an 8×8 alternating board,
// 32 L3 tiles and 32 compute tiles, every optional field declared.
DeploymentSpec t64() {
    DeviceSpecification d;
    d.name = "t64";
    d.topology = "checkerboard";
    d.compute_tiles = 32;
    d.l3.tiles = 32;
    d.l3.banks = 4;
    d.l3.capacity_tiles = 32 * 63;
    d.l2.banks_per_tile = 4;
    d.l1.vectors = 2;
    d.dma.engines = 8;
    d.memory.controllers = 4;
    d.cpu.harts = 4;
    d.array.rows = 8;
    d.array.cols = 8;
    DeploymentSpec spec;
    spec.devices = {d};
    return spec;
}

} // namespace

TEST_CASE("the T64 names every kind, and each new kind round-trips through its text",
          "[program][platform][names][layout]") {
    const ResourceMap map(t64());
    std::set<int> kinds;
    for (const ResourceName& n : map.enumerate()) {
        kinds.insert(static_cast<int>(n.kind));
        CHECK(parse_resource_name(format(n)) == n);     // one spelling, both ways
    }
    CHECK(kinds.size() == all_resource_kinds().size());

    for (const char* text : {"t64/l3[0]/bm[1]", "t64/l3[5]/noc", "t64/noc/port[15]",
                             "t64/mc[3]", "t64/mc[3]/dma[1]", "t64/cpu/hart[3]",
                             "t64/cpu/sram", "t64/cpu/dring", "t64/cpu/cring"})
        CHECK(map.exists(parse_resource_name(text)));

    // Counted: 112 BlockMovers (every horizontal and vertical neighbour pair on an 8×8
    // alternating board is one L3 and one compute tile: 2 × 8 × 7), 32 hubs, 16 ports,
    // 4 controllers × 2 engines, 4 harts + sram + 2 rings.
    std::size_t bm = 0, hubs = 0, ports = 0, mcs = 0, dmas = 0;
    for (const ResourceName& n : map.enumerate()) {
        bm += n.kind == ResourceKind::BlockMover;
        hubs += n.kind == ResourceKind::NocRouter;
        ports += n.kind == ResourceKind::NocPort;
        mcs += n.kind == ResourceKind::MemoryController;
        dmas += n.kind == ResourceKind::DmaEngine;
    }
    CHECK(bm == 112);
    CHECK(hubs == 32);
    CHECK(ports == 16);
    CHECK(mcs == 4);
    CHECK(dmas == 8);
}

TEST_CASE("a missing physical name says which part of the shape is missing",
          "[program][platform][names][layout]") {
    const ResourceMap map(t64());
    // The top-left L3 tile is at a corner: compute tiles abut only its E and S edges.
    CHECK(map.exists(parse_resource_name("t64/l3[0]/bm[1]")));
    CHECK(map.exists(parse_resource_name("t64/l3[0]/bm[2]")));
    CHECK(map.why_not(parse_resource_name("t64/l3[0]/bm[0]")).find("no compute tile on its N edge") !=
          std::string::npos);
    CHECK(map.why_not(parse_resource_name("t64/l3[0]/bm[7]")).find("named by its edge") !=
          std::string::npos);
    CHECK(map.why_not(parse_resource_name("t64/mc[0]/dma[2]")).find("2 DMA engine") !=
          std::string::npos);
    CHECK(map.why_not(parse_resource_name("t64/noc/port[16]")).find("16 fold-end port") !=
          std::string::npos);

    // A deployment with no layout names none of it, and says why rather than "out of range".
    const ResourceMap plain(two_devices());
    const std::string why = plain.why_not(parse_resource_name("left/l3[0]/noc"));
    CHECK(why.find("no array layout") != std::string::npos);
    CHECK(why.find("as many L3 tiles as compute tiles") != std::string::npos);
    CHECK(plain.why_not(parse_resource_name("left/mc[0]")).find("memory.controllers is not declared") !=
          std::string::npos);
    CHECK(plain.why_not(parse_resource_name("left/cpu/sram")).find("cpu.harts is not declared") !=
          std::string::npos);
}

TEST_CASE("pre-existing resources keep their dense index when the shape is declared",
          "[program][platform][names][layout]") {
    // The new kinds are appended after every original kind, per device, so declaring the
    // physical shape does not renumber a resource an event record already refers to. Stated
    // as the ordering it relies on: in the enumeration, no physical-shape name precedes an
    // original-kind name.
    const ResourceMap map(t64());
    const auto original = [](ResourceKind k) {
        return static_cast<int>(k) <= static_cast<int>(ResourceKind::RegisterFile);
    };
    std::size_t last_original = 0, first_new = map.size();
    for (std::size_t i = 0; i < map.size(); ++i) {
        if (original(map.enumerate()[i].kind)) last_original = i;
        else first_new = std::min(first_new, i);
    }
    CHECK(last_original < first_new);

    // And the original names are exactly the ones a shape-less spec of the same counts gives,
    // in the same order.
    DeploymentSpec bare = t64();
    bare.device(0).memory.controllers.reset();
    bare.device(0).cpu.harts.reset();
    bare.device(0).topology = "single";     // no checkerboard layout for 32 tiles
    bare.device(0).array.rows.reset();
    bare.device(0).array.cols.reset();
    const ResourceMap plain(bare);
    REQUIRE(plain.size() == last_original + 1);
    for (std::size_t i = 0; i < plain.size(); ++i)
        CHECK(plain.enumerate()[i] == map.enumerate()[i]);
}
