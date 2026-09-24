// ============================================================================
// tests/program/test_deployment_spec.cpp
// A deployment is data (#282 increment 1): one machine description, JSON at its
// edge, and every other description of a device a VIEW of it.
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR,
// following tests/program/test_l0_corpus.cpp.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace sw::kpu::program;
using namespace sw::kpu::program::driver;
using namespace sw::kpu::program::platform;

namespace {

const char* kDeploy = "tests/program/deploy/";

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());                 // a missing fixture is a failure, not a skip
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// A spec with every optional field set, so a round trip has something to lose.
DeploymentSpec rich_spec() {
    DeviceSpecification a;
    a.name = "left";
    a.topology = "checkerboard";
    a.compute_tiles = 16;
    a.macs_per_cycle = 512.0;
    a.element_bytes = 2;
    a.dma.engines = 4;
    a.dma.bytes_per_cycle = 96.0;
    a.dma.burst_bytes = 256;
    a.l3.tiles = 8;
    a.l3.banks = 8;
    a.l3.capacity_tiles = 32;
    a.l2.banks_per_tile = 8;
    a.l1.vectors = 4;
    a.movers.block_movers = 4;
    a.movers.bm_bytes_per_cycle = 192.0;
    a.movers.streamers = 4;
    a.movers.str_bytes_per_cycle = 320.0;
    a.movers.noc_links = 2;
    a.movers.noc_bytes_per_cycle = 144.0;
    a.analytical.bytes_per_cycle = 80.0;
    a.analytical.pj_per_mac = 1.5;
    a.analytical.pj_per_byte = 22.0;

    DeviceSpecification b;
    b.name = "right";
    b.topology = "news";
    b.compute_tiles = 1;

    DeploymentSpec spec;
    spec.devices = {a, b};
    return spec;
}

} // namespace

TEST_CASE("canonical JSON round-trips byte for byte", "[program][platform][deploy]") {
    // The claim the cache key rests on: the canonical bytes are a function of the spec, so
    // two specs that ARE the same have the same bytes and therefore the same digest.
    for (const DeploymentSpec& spec : {DeploymentSpec{}, rich_spec()}) {
        const std::string once = to_json(spec);
        const std::string twice = to_json(from_json(once));
        CHECK(once == twice);
        CHECK(deployment_digest(spec) == deployment_digest(from_json(once)));
    }
}

TEST_CASE("a checked-in canonical spec round-trips, so a format change cannot be quiet",
          "[program][platform][deploy]") {
    // Deliberately strict, exactly like the L0 corpus: it will fail on any change to the
    // canonical form, which is the point. The file is the evidence that an existing spec
    // still loads and still means the same machine.
    const std::string text = read_file(std::string(kDeploy) + "canonical_single.json");
    const DeploymentSpec spec = from_json(text);
    CHECK(to_json(spec) == text);
    // And it is the default machine, so a drifting default is caught here too.
    CHECK(to_json(spec) == to_json(DeploymentSpec{}));
}

TEST_CASE("the ADR's own spelling loads", "[program][platform][deploy]") {
    // A flat object is one device. Refusing the spelling the authority document shows
    // would make the documentation a trap -- someone pastes §3.5's example and it fails.
    const DeploymentSpec spec = read_spec_file(std::string(kDeploy) + "adr_checkerboard16.json");
    REQUIRE(spec.device_count() == 1);
    const DeviceSpecification& d = spec.device(0);
    CHECK(d.topology == "checkerboard");
    CHECK(d.compute_tiles == 16);
    CHECK(d.dma.engines == 4);
    // "burst" is read as burst_bytes: the ADR writes the short name, the canonical form
    // puts the unit in it.
    REQUIRE(d.dma.burst_bytes.has_value());
    CHECK(*d.dma.burst_bytes == 256);
    REQUIRE(d.l3.tiles.has_value());
    CHECK(*d.l3.tiles == 8);
    CHECK(d.l3.capacity_tiles == 32);
    REQUIRE(d.l2.banks_per_tile.has_value());
    CHECK(*d.l2.banks_per_tile == 8);
    REQUIRE(d.l1.vectors.has_value());
    CHECK(*d.l1.vectors == 4);
    CHECK(d.movers.block_movers == 4);
    CHECK(d.movers.streamers == 4);

    // NORMALIZATION IS IDEMPOTENT. A non-canonical spec is rewritten once and is stable
    // after that -- which is the honest form of the round-trip claim, since `"engines": 4`
    // is an integer literal where `bytes_per_cycle` is a double and no writer can reproduce
    // both spellings.
    const std::string first = to_json(spec);
    CHECK(to_json(from_json(first)) == first);
    // The canonical rewrite is a different file from the original, and saying so beats
    // implying the bytes survived.
    CHECK(first != read_file(std::string(kDeploy) + "adr_checkerboard16.json"));
}

TEST_CASE("'burst' and 'burst_bytes' are one field, and giving both is refused",
          "[program][platform][deploy]") {
    // Two spellings of one field, silently resolved, is a machine that differs from the
    // file whenever they disagree.
    CHECK_THROWS_AS(from_json(R"({"dma": {"burst": 128, "burst_bytes": 256}})"), SpecError);
    // Either alone is fine and means the same thing.
    CHECK(*from_json(R"({"dma": {"burst": 128}})").device(0).dma.burst_bytes == 128);
    CHECK(*from_json(R"({"dma": {"burst_bytes": 128}})").device(0).dma.burst_bytes == 128);
    CHECK(deployment_digest(from_json(R"({"dma": {"burst": 128}})")) ==
          deployment_digest(from_json(R"({"dma": {"burst_bytes": 128}})")));
}

TEST_CASE("absent is not the same as a default value", "[program][platform][deploy]") {
    // The mechanism behind "declared". Without it every run would report the resource
    // fields as unmodelled forever, the report would be noise, and noise is how a real one
    // gets missed.
    const DeploymentSpec undeclared = from_json(R"({"l3": {"capacity_tiles": 4}})");
    CHECK_FALSE(undeclared.device(0).l3.banks.has_value());
    CHECK(to_json(undeclared).find("banks") == std::string::npos);

    // Declaring a value -- even one that looks like a default -- is a statement, and it
    // survives the round trip and changes the digest.
    const DeploymentSpec declared_eight = from_json(R"({"l3": {"capacity_tiles": 4, "banks": 8}})");
    REQUIRE(declared_eight.device(0).l3.banks.has_value());
    CHECK(to_json(declared_eight).find("\"banks\": 8") != std::string::npos);
    CHECK(deployment_digest(declared_eight) != deployment_digest(undeclared));
}

TEST_CASE("an unknown key is refused, and the refusal says what would work",
          "[program][platform][deploy]") {
    // A typo silently ignored means the machine is not the one the file describes while
    // the run reports success.
    try {
        from_json(R"({"compute_tile": 16})");
        FAIL("an unknown key must be refused");
    } catch (const SpecError& e) {
        const std::string what = e.what();
        CHECK(what.find("compute_tile") != std::string::npos);
        CHECK(what.find("compute_tiles") != std::string::npos);   // the key that works
    }
    CHECK_THROWS_AS(from_json(R"({"l3": {"bank": 8}})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"movers": {"blockmovers": 8}})"), SpecError);
    // A "devices" array beside stray device fields is refused rather than half-read: one of
    // the two is being ignored and the file does not say which.
    CHECK_THROWS_AS(from_json(R"({"devices": [{"name": "a"}], "compute_tiles": 4})"), SpecError);
}

TEST_CASE("a hostile or wrongly typed number cannot become a machine",
          "[program][platform][deploy]") {
    // -1 must not wrap into an enormous count, which is the std::stoul failure one layer
    // over -- this repo has now met it twice.
    try {
        from_json(R"({"compute_tiles": -1})");
        FAIL("a negative count must be refused");
    } catch (const SpecError& e) {
        CHECK(std::string(e.what()).find("compute_tiles") != std::string::npos);
    }
    CHECK_THROWS_AS(from_json(R"({"compute_tiles": 99999999999})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"compute_tiles": "sixteen"})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"topology": 4})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"dma": {"bytes_per_cycle": "fast"}})"), SpecError);
    CHECK_THROWS_AS(from_json("not json at all"), SpecError);
    CHECK_THROWS_AS(from_json("[1, 2, 3]"), SpecError);
    CHECK_THROWS_AS(read_spec_file("tests/program/deploy/there_is_no_such_file.json"), SpecError);
}

TEST_CASE("an impossible machine is refused where the field name still exists",
          "[program][platform][deploy]") {
    // Validated at the edge rather than deep inside an executor, where a division by zero
    // would name an executor internal instead of the field that was wrong.
    CHECK_THROWS_AS(from_json(R"({"compute_tiles": 0})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"dma": {"engines": 0}})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"dma": {"bytes_per_cycle": 0}})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"movers": {"streamers": 0}})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"topology": "hexagonal"})"), SpecError);
    // A declared zero of a bank count is not a statement about a machine that can hold
    // anything.
    CHECK_THROWS_AS(from_json(R"({"l3": {"banks": 0}})"), SpecError);
    // Two devices with one name make an address ambiguous, not merely confusing.
    CHECK_THROWS_AS(
        from_json(R"({"devices": [{"name": "same"}, {"name": "same"}]})"), SpecError);
    CHECK_THROWS_AS(from_json(R"({"devices": []})"), SpecError);

    // noc_links MAY be zero: a topology with no L3<->L3 path is a real machine.
    CHECK_NOTHROW(from_json(R"({"movers": {"noc_links": 0}})"));
    // ...and so is an unbounded L3, which is what a design-space sweep wants.
    CHECK_NOTHROW(from_json(R"({"l3": {"capacity_tiles": 0}})"));
}

TEST_CASE("device_view projects capacity, and never confuses it with the module count",
          "[program][platform][deploy]") {
    // `l3.tiles` (how many L3 modules) and `l3.capacity_tiles` (how many tile-sized
    // buffers) are different things, and DeviceDescriptor::l3_tiles is the CAPACITY. Wiring
    // the module count into it would bound the credit model by the wrong number -- a
    // silently different machine, with plausible timing.
    const DeploymentSpec spec = from_json(R"({"l3": {"tiles": 8, "capacity_tiles": 32}})");
    const auto d = spec.device_view();
    CHECK(d.l3_tiles == 32);

    const DeploymentSpec modules_only = from_json(R"({"l3": {"tiles": 8}})");
    CHECK(modules_only.device_view().l3_tiles == 0);      // 0 = unbounded, not 8
}

TEST_CASE("the CLI's device is unchanged by going through a deployment",
          "[program][platform][deploy]") {
    // A REFACTOR GUARD. make_device() used to build a DeviceDescriptor directly; it now
    // projects one out of a DeploymentSpec. Every timing number in this repo depends on
    // these fields, so a shifted one would move calibration and makespans everywhere while
    // every other test still passed.
    for (const char* topo : {"single", "news", "checkerboard"}) {
        DeviceSpec ds;
        ds.topology = topo;
        ds.compute_tiles = 4;
        ds.l3_tiles = 12;
        ds.dma_engines = 2;
        ds.dma_bytes_per_cycle = 64.0;
        ds.block_movers = 3;
        ds.bm_bytes_per_cycle = 128.0;
        ds.streamers = 5;
        ds.str_bytes_per_cycle = 256.0;
        ds.noc_links = 1;
        ds.noc_bytes_per_cycle = 128.0;
        ds.macs_per_cycle = 256.0;
        ds.bytes_per_cycle = 64.0;
        ds.pj_per_mac = 1.0;
        ds.pj_per_byte = 20.0;

        const auto d = make_device(ds);
        INFO("topology " << topo);
        CHECK(d.compute_tiles == 4);
        CHECK(d.l3_tiles == 12);
        CHECK(d.dma_engines == 2);
        CHECK(d.dma_bytes_per_cycle == 64.0);
        CHECK(d.block_movers == 3);
        CHECK(d.bm_bytes_per_cycle == 128.0);
        CHECK(d.streamers == 5);
        CHECK(d.str_bytes_per_cycle == 256.0);
        CHECK(d.noc_links == 1);
        CHECK(d.noc_bytes_per_cycle == 128.0);
        CHECK(d.fabric_macs_per_cycle == 256.0);
        CHECK(d.bytes_per_cycle == 64.0);
        CHECK(d.pj_per_mac == 1.0);
        CHECK(d.pj_per_byte == 20.0);
        CHECK(d.element_bytes == 4.0);
        // The aggregate lane count is topology-derived, and the analytical harness reads it.
        const Dim expected_lanes = std::string(topo) == "single"  ? 1u
                                 : std::string(topo) == "news"    ? 4u
                                                                  : 4u;   // checkerboard: cf
        CHECK(d.move_lanes == expected_lanes);
        // And the same descriptor comes out of the spec the CLI now builds.
        CHECK(make_deployment(ds).device_view().label() == d.label());
    }

    // An impossible machine is refused with the flag's own words, not an executor's.
    DeviceSpec bad;
    bad.compute_tiles = 0;
    CHECK_THROWS_AS(make_deployment(bad), std::invalid_argument);
    DeviceSpec wrong_topo;
    wrong_topo.topology = "hexagonal";
    CHECK_THROWS_AS(make_deployment(wrong_topo), std::invalid_argument);
}

TEST_CASE("a level says which declared fields it does not model",
          "[program][platform][deploy]") {
    // The same statement as not_implemented_reason(), one layer down: that one keeps a clean
    // report from being mistaken for full coverage across LEVELS, this one across the
    // MACHINE.
    const DeploymentSpec rich = read_spec_file(std::string(kDeploy) + "adr_checkerboard16.json");

    const auto at_t1 = unmodelled_fields(ExecutionLevel::BlockSequential, rich);
    // L-T1 models the L3 capacity (#264 increment 4) and nothing else the spec declared.
    for (const std::string& line : at_t1) CHECK(line.find("capacity_tiles") == std::string::npos);
    CHECK(at_t1.size() == 5);          // l3.tiles, l3.banks, l2.banks_per_tile, l1.vectors, dma.burst_bytes

    const auto at_b = unmodelled_fields(ExecutionLevel::Behavioral, rich);
    // L-B has no buffers to bound, so the capacity joins the list rather than vanishing.
    CHECK(at_b.size() == at_t1.size() + 1);
    bool mentions_capacity = false;
    for (const std::string& line : at_b)
        mentions_capacity = mentions_capacity || line.find("capacity_tiles") != std::string::npos;
    CHECK(mentions_capacity);

    // AN UNDECLARED FIELD IS NEVER REPORTED. This is what keeps the report from being noise
    // on every run: the default deployment declares nothing, so it has nothing to admit.
    CHECK(unmodelled_fields(ExecutionLevel::BlockSequential, DeploymentSpec{}).empty());
    CHECK(unmodelled_fields(ExecutionLevel::Behavioral, DeploymentSpec{}).empty());

    // A flags-built deployment declares the L3 capacity when --l3-tiles was given, and that
    // is worth reporting at L-B -- which is the one case this already catches without
    // --deploy.
    DeviceSpec ds;
    ds.l3_tiles = 8;
    const auto flagged = make_deployment(ds);
    CHECK(unmodelled_fields(ExecutionLevel::BlockSequential, flagged).empty());
    CHECK(unmodelled_fields(ExecutionLevel::Behavioral, flagged).size() == 1);
}

TEST_CASE("a deployment holds more than one device", "[program][platform][deploy]") {
    // Multi-device from the start: the backdoor (#284) and the naming map (increment 3)
    // address resources as (device, kind, instance, offset), and retrofitting a device axis
    // later is a format migration.
    const DeploymentSpec spec = from_json(to_json(rich_spec()));
    REQUIRE(spec.device_count() == 2);
    CHECK(spec.device(0).name == "left");
    CHECK(spec.device(1).name == "right");
    CHECK(spec.index_of("right") == 1);
    CHECK(spec.index_of("absent") == spec.devices.size());

    // Each device projects its own descriptor -- the second is a different machine.
    CHECK(spec.device_view(0).compute_tiles == 16);
    CHECK(spec.device_view(1).compute_tiles == 1);
    CHECK(spec.device_view(0).element_bytes == 2.0);
    CHECK(spec.device_view(1).element_bytes == 4.0);
    CHECK(spec.label().find("/x2") != std::string::npos);
    CHECK_THROWS(spec.device_view(2));      // out of range is an error, not device 0
}

TEST_CASE("the digest tracks the spec, not the spelling", "[program][platform][deploy]") {
    // What it is for: labelling a run and looking one up. What it is NOT: the identity
    // claim -- see digest.hpp. So the test that matters is that it MOVES when the machine
    // moves, and does not when only the text does.
    const DeploymentSpec base = from_json(R"({"compute_tiles": 4})");
    const DeploymentSpec spaced = from_json("{\n\n  \"compute_tiles\":    4\n}\n");
    CHECK(deployment_digest(base) == deployment_digest(spaced));

    const DeploymentSpec more = from_json(R"({"compute_tiles": 5})");
    CHECK(deployment_digest(base) != deployment_digest(more));

    // Including a field no level models yet: the deployment is still a different
    // deployment, and a cache must not serve one for the other.
    const DeploymentSpec banked = from_json(R"({"compute_tiles": 4, "l3": {"banks": 8}})");
    CHECK(deployment_digest(base) != deployment_digest(banked));

    CHECK(deployment_digest(base).size() == 16);
}

// ----------------------------------------------------------------------------
// Review of #302
// ----------------------------------------------------------------------------
TEST_CASE("a non-finite number is not a fast machine", "[program][platform][deploy]") {
    // `!(x > 0.0)` lets +inf through, and it is REACHABLE: std::stod parses "inf", so
    // `--dma-bytes-per-cycle inf` reached the old check and passed it. An infinite
    // bandwidth is a makespan of 0 or a NaN, reported as a result.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    for (double bad : {inf, -inf, nan}) {
        DeploymentSpec spec;
        spec.device(0).dma.bytes_per_cycle = bad;
        CHECK_FALSE(spec.validate().empty());

        DeploymentSpec macs;
        macs.device(0).macs_per_cycle = bad;
        CHECK_FALSE(macs.validate().empty());

        DeploymentSpec bm;
        bm.device(0).movers.bm_bytes_per_cycle = bad;
        CHECK_FALSE(bm.validate().empty());

        DeploymentSpec str;
        str.device(0).movers.str_bytes_per_cycle = bad;
        CHECK_FALSE(str.validate().empty());

        DeploymentSpec noc;
        noc.device(0).movers.noc_bytes_per_cycle = bad;
        CHECK_FALSE(noc.validate().empty());

        DeploymentSpec an;
        an.device(0).analytical.bytes_per_cycle = bad;
        CHECK_FALSE(an.validate().empty());

        // The energy coefficients were not validated AT ALL.
        DeploymentSpec mac_energy;
        mac_energy.device(0).analytical.pj_per_mac = bad;
        CHECK_FALSE(mac_energy.validate().empty());

        DeploymentSpec byte_energy;
        byte_energy.device(0).analytical.pj_per_byte = bad;
        CHECK_FALSE(byte_energy.validate().empty());
    }

    // Zero energy is a legitimate modelling choice -- "ignore compute energy" -- so the
    // coefficients are finite and NON-NEGATIVE rather than positive. A negative one is not.
    DeploymentSpec free_compute;
    free_compute.device(0).analytical.pj_per_mac = 0.0;
    free_compute.device(0).analytical.pj_per_byte = 0.0;
    CHECK(free_compute.validate().empty());

    DeploymentSpec negative;
    negative.device(0).analytical.pj_per_mac = -1.0;
    CHECK_FALSE(negative.validate().empty());
}

TEST_CASE("anything validate() accepts can be written and read back",
          "[program][platform][deploy]") {
    // THE PROPERTY THE NON-FINITE CHECK REALLY PROTECTS. nlohmann writes a non-finite
    // double as JSON `null`, and null is not a number -- so a spec that validate() accepted
    // could serialize to bytes from_json() then REFUSED. A validator that admits values the
    // format cannot represent is not a validator, and the digest that keys a cache would be
    // taken over bytes nobody can load.
    const double inf = std::numeric_limits<double>::infinity();
    DeploymentSpec poisoned;
    poisoned.device(0).dma.bytes_per_cycle = inf;
    REQUIRE_FALSE(poisoned.validate().empty());          // refused, so it never gets written
    CHECK(to_json(poisoned).find("null") != std::string::npos);   // and this is why

    // Every spec this file builds and accepts survives the round trip.
    for (const DeploymentSpec& spec : {DeploymentSpec{}, rich_spec()}) {
        REQUIRE(spec.validate().empty());
        CHECK_NOTHROW(from_json(to_json(spec)));
    }
}

TEST_CASE("the canonical key order is part of the format",
          "[program][platform][deploy]") {
    // The output is ordered_json, so the bytes depend on the order write_device() assigns
    // in -- reordering an assignment changes every deployment_digest. That makes the order
    // a format decision, so it is asserted rather than left to whoever edits next. The
    // checked-in fixture would also catch it; this says WHY it failed.
    const std::string text = to_json(DeploymentSpec{});
    const std::size_t name = text.find("\"name\"");
    const std::size_t topology = text.find("\"topology\"");
    const std::size_t compute = text.find("\"compute_tiles\"");
    const std::size_t dma = text.find("\"dma\"");
    const std::size_t analytical = text.find("\"analytical\"");
    REQUIRE(name != std::string::npos);
    CHECK(name < topology);
    CHECK(topology < compute);
    CHECK(compute < dma);
    CHECK(dma < analytical);
}
