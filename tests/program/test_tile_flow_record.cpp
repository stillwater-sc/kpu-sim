// ============================================================================
// tests/program/test_tile_flow_record.cpp
// The tile-flow record (#286 step 2): a run's L3 residency series, its transits and its
// computes, agreeing with the executor's own statistics, and a .tflow bundle that round-trips
// byte for byte.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/record/tile_flow_record.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

using namespace sw::kpu::program;
using namespace sw::kpu::program::record;
using Catch::Matchers::ContainsSubstring;

namespace {

struct Run {
    TileProgram prog;
    driver::RunOutcome outcome;
    platform::DeploymentSpec spec;
    Placement placement = Placement::single(1);
};

Run run_matmul(unsigned n, unsigned t, Dim l3_cap, const std::set<std::string>& seeded = {},
               const std::set<std::string>& retained = {}, std::size_t foreign = 0,
               driver::ExecutionLevel level = driver::ExecutionLevel::BlockSequential) {
    driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = n;
    ps.tile = t;
    Run r;
    r.prog = driver::derive(ps);
    driver::DeviceSpec ds;
    ds.l3_tiles = l3_cap;
    r.spec = driver::make_deployment(ds);
    const auto dev = r.spec.device_view();
    r.placement = Placement::single(dev.compute_tiles);
    r.outcome = driver::run_at(level, r.prog, dev, r.placement, nullptr, 0, seeded, retained, foreign);
    return r;
}

std::string slurp(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string scratch(const std::string& name) {
    const auto dir = std::filesystem::temp_directory_path() / ("kpu-tflow-test-" + name);
    std::filesystem::remove_all(dir);
    return dir.string();
}

} // namespace

TEST_CASE("the record's L3 series agrees with the executor's own numbers",
          "[program][record]") {
    // Pinned against the statistics the executor already publishes, at a capacity tight
    // enough to stall (6 is the matmul's live set) and at an unbounded one.
    for (Dim cap : {Dim{0}, Dim{6}, Dim{8}, Dim{12}}) {
        INFO("L3 capacity " << cap);
        const Run r = run_matmul(32, 16, cap);
        REQUIRE(r.outcome.stats.has_value());
        const TileFlowRecord rec = build_record(r.prog, r.outcome, r.spec, r.placement);

        // The peak of the series IS the executor's peak -- the number #279 showed can look
        // healthy while the series behind it is not. Now the series is there to check.
        CHECK(peak_l3_occupancy(rec) == r.outcome.stats->peak_l3_residency);
        if (cap) {
            for (const Residency& x : rec.residency) {
                CHECK(l3_occupancy_at(rec, x.t0) <= cap);
            }
        }

        // Every tile the program touches held an L3 slot at some point, inside the run.
        std::set<std::uint32_t> resident_tiles;
        for (const Residency& x : rec.residency) {
            CHECK(x.t0 <= x.t1);
            CHECK(x.t1 <= rec.makespan);
            CHECK(rec.stations[x.station].kind == "l3");
            resident_tiles.insert(x.tile);
        }
        CHECK(resident_tiles.size() == rec.tiles.size());

        // One transit per hop the executor counted; one compute per compute op.
        std::size_t hops = 0;
        for (const auto& [h, n] : r.outcome.stats->hop_transfers) hops += n;
        CHECK(rec.transits.size() == hops);
        CHECK(rec.computes.size() == r.outcome.stats->computes);
        for (const Compute& c : rec.computes) CHECK(rec.stations[c.station].kind == "cf");
    }
}

TEST_CASE("a transit's endpoints are the stations its hop connects, and L2/L1 say unmodelled",
          "[program][record]") {
    const Run r = run_matmul(32, 16, 0);
    const TileFlowRecord rec = build_record(r.prog, r.outcome, r.spec, r.placement);
    const auto kind = [&](std::uint32_t s) { return rec.stations[s].kind; };
    for (const Transit& x : rec.transits) {
        switch (static_cast<Hop>(x.hop)) {
            case Hop::DmaDramToL3:      CHECK((kind(x.src) == "dram" && kind(x.dst) == "l3")); break;
            case Hop::BlockMoverL3ToL2: CHECK((kind(x.src) == "l3" && kind(x.dst) == "l2")); break;
            case Hop::StreamerL2ToL1:   CHECK((kind(x.src) == "l2" && kind(x.dst) == "l1")); break;
            case Hop::StreamerL1ToL2:   CHECK((kind(x.src) == "l1" && kind(x.dst) == "l2")); break;
            case Hop::BlockMoverL2ToL3: CHECK((kind(x.src) == "l2" && kind(x.dst) == "l3")); break;
            case Hop::BlockMoverL3ToDmaBuffer: CHECK((kind(x.src) == "l3" && kind(x.dst) == "dmabuf")); break;
            case Hop::DmaBufferToDram:  CHECK((kind(x.src) == "dmabuf" && kind(x.dst) == "dram")); break;
            case Hop::BlockMoverL3ToL3: CHECK((kind(x.src) == "l3" && kind(x.dst) == "l3")); break;
        }
        CHECK(static_cast<Mover>(x.mover) == mover_of(static_cast<Hop>(x.hop)));
    }
    // What L-T1 does not model is said, not drawn as empty.
    CHECK(rec.unmodelled == std::vector<std::string>{"l2", "l1", "dmabuf"});
    CHECK_FALSE(rec.stations[rec.station(r.spec.device(0).name + "/l2[*]")].modelled);
    CHECK(rec.stations[rec.station(r.spec.device(0).name + "/l3[*]")].pooled);
}

TEST_CASE("the caller's tiles are in the series: seeded from cycle 0, held to the end",
          "[program][record]") {
    // An orchestrator seeded A#0#0, will keep B#0#0, and holds 3 slots this program never
    // names. All three facts must reach the record, or its occupancy is low by them.
    const Run r = run_matmul(32, 16, 0, {"A#0#0"}, {"B#0#0"}, 3);
    const TileFlowRecord rec = build_record(r.prog, r.outcome, r.spec, r.placement, 0, 3);
    std::map<std::string, Residency> by_tile;
    for (const Residency& x : rec.residency) by_tile[rec.tiles[x.tile].key()] = x;

    const Residency a = by_tile.at("A#0#0");
    CHECK(a.t0 == 0);
    CHECK((a.flags & Residency::Seeded));
    CHECK((a.flags & Residency::Held));
    CHECK(a.t1 == rec.makespan);
    const Residency b = by_tile.at("B#0#0");
    CHECK_FALSE((b.flags & Residency::Seeded));
    CHECK((b.flags & Residency::Held));
    CHECK(b.t1 == rec.makespan);
    // A tile neither seeded nor retained is returned before the end.
    CHECK_FALSE((by_tile.at("A#0#1").flags & Residency::Held));

    CHECK(rec.foreign_slots == 3);
    CHECK(peak_l3_occupancy(rec) == r.outcome.stats->peak_l3_residency);
}

TEST_CASE("a level that models no time has no record, and says so", "[program][record]") {
    const Run r = run_matmul(32, 16, 0, {}, {}, 0, driver::ExecutionLevel::Behavioral);
    CHECK_THROWS_WITH(build_record(r.prog, r.outcome, r.spec, r.placement),
                      ContainsSubstring("models no time"));
}

TEST_CASE("the .tflow bundle round-trips, and the same run writes the same bytes",
          "[program][record]") {
    const Run r = run_matmul(64, 16, 0);
    const TileFlowRecord rec = build_record(r.prog, r.outcome, r.spec, r.placement);
    const std::string a = scratch("a"), b = scratch("b");
    write_tflow(rec, a);
    write_tflow(build_record(r.prog, run_matmul(64, 16, 0).outcome, r.spec, r.placement), b);
    for (const char* f : {"manifest.json", "residency.bin", "transit.bin", "compute.bin", "op_tiles.bin"}) {
        INFO(f);
        CHECK(slurp(a + "/" + f) == slurp(b + "/" + f));
        CHECK(!slurp(a + "/" + f).empty());
    }

    const TileFlowRecord back = read_tflow(a);
    CHECK(back.level == rec.level);
    CHECK(back.makespan == rec.makespan);
    CHECK(back.stations.size() == rec.stations.size());
    REQUIRE(back.operands.size() == 3);                      // A, B, C of the matmul
    CHECK(back.operands[0].name == rec.operands[0].name);
    CHECK(back.operands[0].tile_rows == 16);
    CHECK(back.element_bytes == rec.element_bytes);
    REQUIRE(back.tiles.size() == rec.tiles.size());
    for (std::size_t i = 0; i < rec.tiles.size(); ++i) CHECK(back.tiles[i].key() == rec.tiles[i].key());
    REQUIRE(back.residency.size() == rec.residency.size());
    for (std::size_t i = 0; i < rec.residency.size(); ++i) {
        CHECK(back.residency[i].tile == rec.residency[i].tile);
        CHECK(back.residency[i].t0 == rec.residency[i].t0);
        CHECK(back.residency[i].t1 == rec.residency[i].t1);
        CHECK(back.residency[i].flags == rec.residency[i].flags);
    }
    REQUIRE(back.transits.size() == rec.transits.size());
    for (std::size_t i = 0; i < rec.transits.size(); ++i) {
        CHECK(back.transits[i].op == rec.transits[i].op);
        CHECK(back.transits[i].hop == rec.transits[i].hop);
        CHECK(back.transits[i].lane == rec.transits[i].lane);
        CHECK(back.transits[i].t1 == rec.transits[i].t1);
    }
    REQUIRE(back.ops.size() == rec.ops.size());
    for (std::size_t i = 0; i < rec.ops.size(); ++i) CHECK(back.ops[i].tiles == rec.ops[i].tiles);
    CHECK(peak_l3_occupancy(back) == peak_l3_occupancy(rec));

    // A newer bundle is refused rather than misread, and so are older ones: version 1 has no
    // `written` column, and version 2 records writeback as a DMA read (its hop 5 is not an
    // ejection). Reading either would be a wrong answer.
    const std::string original = slurp(a + "/manifest.json");
    auto with_version = [&](const char* v) {
        std::string m = original;
        m.replace(m.find("\"version\": 3"), 12, std::string("\"version\": ") + v);
        std::ofstream(a + "/manifest.json", std::ios::binary) << m;
    };
    with_version("4");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("version 4"));
    with_version("2");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("records writeback as a DMA read"));
    with_version("1");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("re-record"));
}

TEST_CASE("a corrupt or hostile bundle is refused, not trusted", "[program][record]") {
    // The reader is a trust boundary: everything after it indexes without checking.
    const Run r = run_matmul(32, 16, 0);
    const TileFlowRecord rec = build_record(r.prog, r.outcome, r.spec, r.placement);
    const std::string dir = scratch("corrupt");
    auto fresh = [&] { std::filesystem::remove_all(dir); write_tflow(rec, dir); };
    auto edit_file = [&](const char* f, const std::function<void(std::string&)>& e) {
        std::string s = slurp(dir + "/" + f);
        e(s);
        std::ofstream(dir + "/" + f, std::ios::binary | std::ios::trunc) << s;
    };
    auto put_f64 = [](std::string& s, std::size_t at, double v) { std::memcpy(s.data() + at, &v, 8); };
    auto put_u32 = [](std::string& s, std::size_t at, std::uint32_t v) { std::memcpy(s.data() + at, &v, 4); };

    SECTION("a truncated column") {
        fresh();
        edit_file("transit.bin", [](std::string& s) { s.resize(s.size() / 2); });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("runs past the end"));
    }
    SECTION("an offset that would overflow the bounds check") {
        fresh();
        edit_file("manifest.json", [](std::string& s) {
            const auto p = s.find("\"name\": \"t0\"");
            const auto o = s.find("\"offset\": ", p);
            const auto e = s.find_first_of("\n}", o);
            s.replace(o, e - o, "\"offset\": 18446744073709551600");
        });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("runs past the end"));
    }
    SECTION("a time that is not a cycle count") {
        fresh();
        // residency.bin: tile u32 (rows*4, padded to 8), station u32 (same), then t0 f64.
        const std::size_t rows = rec.residency.size(), pad = (rows * 4 + 7) / 8 * 8;
        edit_file("residency.bin", [&](std::string& s) { put_f64(s, 2 * pad, 2.5); });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("not a cycle count"));
        edit_file("residency.bin", [&](std::string& s) { put_f64(s, 2 * pad, -1.0); });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("not a cycle count"));
    }
    SECTION("an op offset that decreases") {
        fresh();
        // op_tiles.bin: kind u8 (ops, padded to 8), then offset u32 (ops+1).
        const std::size_t base = (rec.ops.size() + 7) / 8 * 8;
        edit_file("op_tiles.bin", [&](std::string& s) { put_u32(s, base + 4 * 2, 0); });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("decreases"));
    }
    SECTION("an index past its table") {
        fresh();
        edit_file("compute.bin", [&](std::string& s) {
            const std::size_t pad = (rec.computes.size() * 4 + 7) / 8 * 8;
            put_u32(s, pad, 9999);                     // station column, first row
        });
        CHECK_THROWS_WITH(read_tflow(dir), ContainsSubstring("station index 9999"));
    }
    SECTION("a time past 2^53 is refused by the writer, not rounded") {
        TileFlowRecord bad = rec;
        bad.transits.front().t1 = (Cycle{1} << 53) + 1;
        CHECK_THROWS_WITH(write_tflow(bad, scratch("toolong")), ContainsSubstring("transit t1"));
    }
}
