// ============================================================================
// tests/program/test_tile_flow_record.cpp
// The tile-flow record (#286 step 2): a CSP program's L-T1 run as its L3 residency series, its
// transits and its computes, agreeing with the run's own numbers, and a .tflow bundle that
// round-trips byte for byte. (The L0 L-T1 executor's records, versions 1-3, retired with it in
// kpu-run-csp-programs step 4e; version 3 is still read.)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/csp/gen/generate.hpp>
#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/record/tile_flow_lod.hpp>
#include <sw/kpu/program/record/tile_flow_record.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/hop.hpp>

#include <algorithm>
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

namespace gen = sw::kpu::program::csp::gen;
namespace lang = sw::kpu::program::csp::lang;

struct Run {
    lang::Program ast;
    TileProgram inputs;
    platform::DeploymentSpec spec;
    driver::CspLevelOutcome outcome;
    TileFlowRecord record() const { return build_csp_record(ast, outcome, inputs, spec); }
};

Run run_text(const std::string& text,
             driver::ExecutionLevel level = driver::ExecutionLevel::BlockSequential) {
    Run r;
    r.ast = lang::parse(text);
    r.inputs = driver::csp_inputs(r.ast);
    r.spec = driver::make_deployment(driver::DeviceSpec{});
    r.outcome = driver::csp_run_level(level, r.ast, &r.spec.device(0), r.inputs);
    return r;
}

Run run_matmul(unsigned n, unsigned t, int l3,
               driver::ExecutionLevel level = driver::ExecutionLevel::BlockSequential) {
    gen::Options o;
    o.algo = "matmul";
    o.size = n;
    o.tile = t;
    o.l3 = l3;
    return run_text(gen::generate(o), level);
}

} // namespace

TEST_CASE("the record's L3 series agrees with the run's own numbers", "[program][record]") {
    const Run r = run_matmul(64, 16, 12);
    REQUIRE_FALSE(r.outcome.skipped);
    const TileFlowRecord rec = r.record();
    CHECK(rec.makespan == r.outcome.makespan);
    CHECK(peak_l3_occupancy(rec) == r.outcome.peak_l3);
    CHECK(peak_l3_occupancy(rec) <= 12u);
    CHECK(rec.foreign_slots == 0);
    // Occupancy is a step function over the residency intervals: never above the peak.
    for (const Residency& x : rec.residency) {
        CHECK(x.t0 <= x.t1);
        CHECK(l3_occupancy_at(rec, x.t0) <= peak_l3_occupancy(rec));
    }
}

TEST_CASE("a transit's endpoints are the stations its hop connects, and L2/L1 say unmodelled",
          "[program][record]") {
    const Run r = run_matmul(32, 16, 8);
    const TileFlowRecord rec = r.record();
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

TEST_CASE("an inherited tile is seeded from cycle 0, a retained one held to the end",
          "[program][record]") {
    // The chain's two halves (kpu-run-csp-programs step 4d.1): the first operator retains C, the
    // second inherits it. Both facts must reach the record, or its occupancy is low by them.
    const Run producer = run_text(slurp("tests/program/csp/matmul_relu_retain_32_t16.csp"));
    REQUIRE_FALSE(producer.outcome.skipped);
    const TileFlowRecord held = producer.record();
    std::map<std::string, Residency> by_tile;
    for (const Residency& x : held.residency) by_tile[held.tiles[x.tile].key()] = x;
    for (const char* c : {"C#0#0", "C#0#1", "C#1#0", "C#1#1"}) {
        CAPTURE(c);
        CHECK((by_tile.at(c).flags & Residency::Held));
        CHECK_FALSE((by_tile.at(c).flags & Residency::Seeded));
        CHECK(by_tile.at(c).t1 == held.makespan);
    }
    // A tile neither inherited nor retained is returned before the end.
    CHECK_FALSE((by_tile.at("A#0#0").flags & Residency::Held));

    const Run consumer = run_text(slurp("tests/program/csp/bias_inherit_32_t16.csp"));
    REQUIRE_FALSE(consumer.outcome.skipped);
    const TileFlowRecord seeded = consumer.record();
    std::size_t inherited = 0;
    for (const Residency& x : seeded.residency)
        if (x.flags & Residency::Seeded) {
            ++inherited;
            CHECK(x.t0 == 0);
            CHECK(seeded.tiles[x.tile].operand == "C");
        }
    CHECK(inherited == 4);
}

TEST_CASE("a level that models no time has no record, and says so", "[program][record]") {
    const Run r = run_matmul(32, 16, 8, driver::ExecutionLevel::Behavioral);
    CHECK_THROWS_WITH(r.record(), ContainsSubstring("built from its L-T1 run"));
}

TEST_CASE("the .tflow bundle round-trips, and the same run writes the same bytes",
          "[program][record]") {
    const Run r = run_matmul(64, 16, 12);
    const TileFlowRecord rec = r.record();
    const std::string a = scratch("a"), b = scratch("b");
    write_tflow(rec, a);
    write_tflow(run_matmul(64, 16, 12).record(), b);
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

    // Version 5: every residency carries the L3 address it held, and it round-trips.
    CHECK(back.l3_banks == rec.l3_banks);
    REQUIRE(back.residency.size() == rec.residency.size());
    for (std::size_t i = 0; i < rec.residency.size(); ++i)
        CHECK((back.residency[i].l3_tile == rec.residency[i].l3_tile && back.residency[i].slot == rec.residency[i].slot &&
               back.residency[i].bank == rec.residency[i].bank));

    // A newer bundle (version 7) is refused rather than misread, and so are older ones: version 1 has no
    // `written` column, and version 2 records writeback as a DMA read (its hop 5 is not an
    // ejection). Reading either would be a wrong answer.
    const std::string original = slurp(a + "/manifest.json");
    auto with_version = [&](const char* v, bool says_ops = true) {
        std::string m = original;
        m.replace(m.find("\"version\": 6"), 12, std::string("\"version\": ") + v);
        if (!says_ops) m.replace(m.find("\"ops\": "), 7, "\"opz\": ");
        std::ofstream(a + "/manifest.json", std::ios::binary) << m;
    };
    with_version("7");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("version 7"));
    // Version 6 must say its L2 slots.
    {
        std::string m = original;
        m.replace(m.find("\"l2_slots\": "), 12, "\"l2_slotz\": ");
        std::ofstream(a + "/manifest.json", std::ios::binary) << m;
        CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("l2_slots"));
    }
    // Version 5 still reads (its records have no L2 or chain tables), and must say its banks.
    with_version("5");
    {
        const TileFlowRecord v5 = read_tflow(a);
        CHECK_FALSE(v5.phases);
        CHECK(v5.l2_residency.empty());
        CHECK(v5.chains.empty());
    }
    {
        std::string m = original;
        m.replace(m.find("\"l3_banks\": "), 12, "\"l3_bankz\": ");
        std::ofstream(a + "/manifest.json", std::ios::binary) << m;
        CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("l3_banks"));
    }
    // Version 4 still reads; its residencies carry no address.
    with_version("4");
    {
        const TileFlowRecord v4 = read_tflow(a);
        CHECK(v4.l3_banks == 0);
        for (const Residency& x : v4.residency) CHECK(x.slot == kNoSlot);
    }
    // Version 4 is a CSP program's record; one that does not say its ops are actions is refused.
    with_version("4", false);
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("must say what its ops are"));
    // Version 3, an L0 run's record, is still read though nothing writes it any more: its ops
    // are TileOps, not actions.
    with_version("3", false);
    CHECK(read_tflow(a).op_space == "l0");
    with_version("2");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("records writeback as a DMA read"));
    with_version("1");
    CHECK_THROWS_WITH(read_tflow(a), ContainsSubstring("re-record"));
}

TEST_CASE("a corrupt or hostile bundle is refused, not trusted", "[program][record]") {
    // The reader is a trust boundary: everything after it indexes without checking.
    const Run r = run_matmul(32, 16, 8);
    const TileFlowRecord rec = r.record();
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

// ============================================================================
// Version 4: a CSP program's record (docs/plans/kpu-run-csp-programs.md step 4b)
// ============================================================================
TEST_CASE("v4: a CSP program's L-T1 run records its actions, its slots, and reads back", "[program][record][csp]") {
    for (const char* algo : {"matmul", "linear", "lu"}) {
        CAPTURE(algo);
        gen::Options o;
        o.algo = algo;
        o.size = 64;
        o.tile = 16;
        o.l3 = algo == std::string("lu") ? 16 : 12;
        if (o.algo == "linear") o.place = "unfused";       // a tile through two residencies
        const lang::Program ast = lang::parse(gen::generate(o));
        const TileProgram inputs = driver::csp_inputs(ast);
        const platform::DeploymentSpec spec = driver::make_deployment(driver::DeviceSpec{});
        const auto lt1 = driver::csp_run_level(driver::ExecutionLevel::BlockSequential, ast, &spec.device(0), inputs);
        REQUIRE_FALSE(lt1.skipped);

        const TileFlowRecord rec = build_csp_record(ast, lt1, inputs, spec);
        CHECK(rec.op_space == "csp-actions");
        CHECK(rec.ops.size() == lt1.actions);             // one op per action
        CHECK(rec.residency.size() == lt1.slots.size());   // the program's slots
        CHECK(rec.stations[1].capacity == static_cast<std::uint64_t>(o.l3));   // L3: the program's credits
        CHECK(peak_l3_occupancy(rec) == lt1.peak_l3);
        CHECK(peak_l3_occupancy(rec) <= static_cast<std::uint64_t>(o.l3));
        // A tile holds one L3 slot at a time, even across residencies (the unfused C).
        std::map<std::uint32_t, std::vector<std::pair<Cycle, Cycle>>> by_tile;
        for (const Residency& r : rec.residency) by_tile[r.tile].push_back({r.t0, r.t1});
        for (auto& [tile, ivs] : by_tile) {
            std::sort(ivs.begin(), ivs.end());
            for (std::size_t i = 1; i < ivs.size(); ++i) CHECK(ivs[i].first >= ivs[i - 1].second);
        }
        // Each Call is one compute; each Store two legs.
        std::size_t calls = 0, stores = 0;
        for (const Op& op : rec.ops) {
            calls += op.kind == static_cast<std::uint8_t>(csp::Action::Kind::Call);
            stores += op.kind == static_cast<std::uint8_t>(csp::Action::Kind::Store);
        }
        CHECK(rec.computes.size() == calls);
        std::size_t store_legs = 0;
        for (const Transit& t : rec.transits)
            store_legs += t.hop == static_cast<std::uint8_t>(Hop::BlockMoverL3ToDmaBuffer) ||
                          t.hop == static_cast<std::uint8_t>(Hop::DmaBufferToDram);
        CHECK(store_legs == 2 * stores);

        const auto dir = std::filesystem::temp_directory_path() / ("tflow_v4_" + std::string(algo));
        write_tflow(rec, dir.string());
        const TileFlowRecord back = read_tflow(dir.string());
        CHECK(back.op_space == "csp-actions");
        CHECK(back.ops.size() == rec.ops.size());
        CHECK(back.transits.size() == rec.transits.size());
        CHECK(peak_l3_occupancy(back) == peak_l3_occupancy(rec));
        std::filesystem::remove_all(dir);
    }
}

TEST_CASE("version 5: every residency is at the address L-T1 placed it, one tile per slot", "[program][record][placement]") {
    const Run r = run_text(slurp("tests/program/dflow/streamed_lowest.csp"));
    REQUIRE_FALSE(r.outcome.skipped);
    const TileFlowRecord rec = r.record();
    CHECK(rec.l3_banks == 1);
    REQUIRE(rec.residency.size() == r.outcome.slots.size());
    std::map<std::uint32_t, std::vector<std::pair<Cycle, Cycle>>> by_slot;
    for (std::size_t i = 0; i < rec.residency.size(); ++i) {
        const Residency& x = rec.residency[i];
        const auto& sl = r.outcome.slots[i];
        CHECK(x.slot == sl.place.slot);                 // the address L-T1 executed
        CHECK(x.bank == sl.place.bank);
        CHECK(x.l3_tile == sl.place.l3_tile);
        CHECK(x.slot < rec.stations[x.station].capacity);
        by_slot[x.slot].emplace_back(x.t0, x.t1);
    }
    // place = lowest on a stream reuses slots 0 and 1 only (step 5a).
    for (const auto& [slot, iv] : by_slot) CHECK(slot < 2);
    for (auto& [slot, iv] : by_slot) {
        std::sort(iv.begin(), iv.end());
        for (std::size_t i = 1; i < iv.size(); ++i) CHECK(iv[i].first >= iv[i - 1].second);
    }
    // The pyramid draws one row per slot, after the stations and movers.
    const Lod lod = build_lod(rec);
    std::size_t slot_rows = 0;
    for (const LodRow& row : lod.rows) slot_rows += row.kind == "l3_slot";
    CHECK(slot_rows == 5);
    const LodRow& last_slot = lod.rows[rec.stations.size() + rec.movers.size() + 4];
    CHECK(last_slot.kind == "l3_slot");
    CHECK(last_slot.name == rec.device + "/l3[*]/slot[4]");
}


TEST_CASE("version 6: the staged L2 copies and the fabric's chains, recorded and read back", "[program][record][l2]") {
    gen::Options o;
    o.size = 64;
    o.tile = 16;
    o.l3 = 9;
    o.stage_a = true;                                   // implies the row schedule
    o.orient = "row";
    Run r;
    r.ast = lang::parse(gen::generate(o));
    r.inputs = driver::csp_inputs(r.ast);
    r.spec = platform::read_spec_file("tests/program/deploy/kpu_s1.json");   // declares the fabric
    r.outcome = driver::csp_run_level(driver::ExecutionLevel::BlockSequential, r.ast, &r.spec.device(0), r.inputs);
    REQUIRE_FALSE(r.outcome.skipped);
    const TileFlowRecord rec = r.record();
    CHECK(rec.phases);
    CHECK(rec.l2_slots == 4);
    // Every staged copy, at the address L-T1 executed, with its reuse count: A[i,k] is fed once per j.
    REQUIRE(rec.l2_residency.size() == 16);
    REQUIRE(rec.l2_residency.size() == r.outcome.l2_slots.size());
    for (std::size_t i = 0; i < rec.l2_residency.size(); ++i) {
        const L2Residency& x = rec.l2_residency[i];
        CHECK(x.slot == r.outcome.l2_slots[i].place.slot);
        CHECK(x.feeds == 4);
        CHECK(rec.tiles[x.tile].operand == "A");
    }
    // L2 is modelled when the program stages: its capacity is the program's slots.
    const Station& l2 = rec.stations[2];
    CHECK(l2.kind == "l2");
    CHECK(l2.modelled);
    CHECK(l2.capacity == 4);
    CHECK(std::find(rec.unmodelled.begin(), rec.unmodelled.end(), "l2") == rec.unmodelled.end());
    // One chain per C tile, on the compute tile L-T1 ran it on.
    REQUIRE(rec.chains.size() == 16);
    REQUIRE(rec.chains.size() == r.outcome.chains.size());
    for (std::size_t i = 0; i < rec.chains.size(); ++i) {
        CHECK(rec.chains[i].calls == 4);
        CHECK(rec.chains[i].out1 == r.outcome.chains[i].out1);
        CHECK(rec.stations[rec.chains[i].station].kind == "cf");
    }

    const std::string dir = scratch("v6");
    write_tflow(rec, dir);
    CHECK_THAT(slurp(dir + "/manifest.json"), ContainsSubstring("\"version\": 6"));
    const TileFlowRecord back = read_tflow(dir);
    CHECK(back.phases);
    CHECK(back.l2_slots == rec.l2_slots);
    REQUIRE(back.l2_residency.size() == rec.l2_residency.size());
    for (std::size_t i = 0; i < rec.l2_residency.size(); ++i)
        CHECK((back.l2_residency[i].t0 == rec.l2_residency[i].t0 && back.l2_residency[i].t1 == rec.l2_residency[i].t1 &&
               back.l2_residency[i].feeds == rec.l2_residency[i].feeds && back.l2_residency[i].slot == rec.l2_residency[i].slot));
    REQUIRE(back.chains.size() == rec.chains.size());
    for (std::size_t i = 0; i < rec.chains.size(); ++i)
        CHECK((back.chains[i].in0 == rec.chains[i].in0 && back.chains[i].front == rec.chains[i].front &&
               back.chains[i].in1 == rec.chains[i].in1 && back.chains[i].last == rec.chains[i].last &&
               back.chains[i].out1 == rec.chains[i].out1 && back.chains[i].station == rec.chains[i].station));

    // The pyramid: after the L3 slot rows, one row per L2 slot, then each compute tile's in and out.
    const Lod lod = build_lod(rec);
    std::size_t l2_rows = 0, in_rows = 0, out_rows = 0;
    for (const LodRow& row : lod.rows) {
        l2_rows += row.kind == "l2_slot";
        in_rows += row.kind == "cf_in";
        out_rows += row.kind == "cf_out";
    }
    std::size_t n_cf = 0;
    for (const Station& st : rec.stations) n_cf += st.kind == "cf";
    CHECK(l2_rows == 4);
    CHECK(in_rows == n_cf);
    CHECK(out_rows == n_cf);
    CHECK(lod.rows.back().kind == "cf_out");

    // A staged copy beyond the program's slots is refused at the trust boundary.
    TileFlowRecord bad = rec;
    bad.l2_residency[0].slot = 9;
    const std::string bdir = scratch("v6bad");
    write_tflow(bad, bdir);
    CHECK_THROWS_WITH(read_tflow(bdir), ContainsSubstring("L2 slot 9 of 4"));
    // So are an inverted copy, a copy in no compute tile's L2, and a chain whose phases are out of order.
    bad = rec;
    bad.l2_residency[0].t0 = bad.l2_residency[0].t1 + 1;
    write_tflow(bad, bdir);
    CHECK_THROWS_WITH(read_tflow(bdir), ContainsSubstring("is inverted"));
    bad = rec;
    bad.l2_residency[0].cf = 7;
    write_tflow(bad, bdir);
    CHECK_THROWS_WITH(read_tflow(bdir), ContainsSubstring("compute tile 7"));
    bad = rec;
    bad.chains[0].out1 = bad.chains[0].in1 - 1;
    write_tflow(bad, bdir);
    CHECK_THROWS_WITH(read_tflow(bdir), ContainsSubstring("out of order"));
    // The writer refuses a phase an f64 column cannot hold exactly, whichever phase it is.
    bad = rec;
    bad.chains[0].front = (Cycle{1} << 53) + 1;
    CHECK_THROWS_WITH(write_tflow(bad, scratch("v6toolong")), ContainsSubstring("chain front"));
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(bdir);
}
