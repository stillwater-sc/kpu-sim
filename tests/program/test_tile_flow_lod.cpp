// ============================================================================
// tests/program/test_tile_flow_lod.cpp
// The level-of-detail pyramid (#286 step 3): every bin the exact merge of its children, the
// coarsest level equal to the raw totals, and the base level equal to a cycle-by-cycle count.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>

#include <sw/kpu/program/driver/program_spec.hpp>
#include <sw/kpu/program/record/tile_flow_lod.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace sw::kpu::program;
using namespace sw::kpu::program::record;

namespace {

TileFlowRecord record_of(unsigned n, unsigned t, Dim cap, std::size_t foreign = 0) {
    driver::ProgramSpec ps;
    ps.algo = "matmul";
    ps.size = n;
    ps.tile = t;
    TileProgram prog = driver::derive(ps);
    driver::DeviceSpec ds;
    ds.l3_tiles = cap;
    ds.dma_engines = 2;
    const auto spec = driver::make_deployment(ds);
    const auto dev = spec.device_view();
    const auto placement = Placement::single(dev.compute_tiles);
    const auto out = driver::run_at(driver::ExecutionLevel::BlockSequential, prog, dev, placement,
                                    nullptr, 0, {}, {}, foreign);
    return build_record(prog, out, spec, placement, 0, foreign);
}

// The occupancy of each row at each cycle, counted directly from the intervals.
std::vector<std::vector<std::uint32_t>> brute(const TileFlowRecord& rec, const Lod& lod) {
    std::vector<std::vector<std::uint32_t>> at(lod.rows.size(), std::vector<std::uint32_t>(rec.makespan, 0));
    for (Cycle c = 0; c < rec.makespan; ++c) {
        for (const Residency& x : rec.residency) if (x.t0 <= c && c < x.t1) ++at[x.station][c];
        for (const Compute& x : rec.computes) if (x.t0 <= c && c < x.t1) ++at[x.station][c];
        for (const Transit& x : rec.transits)
            if (x.t0 <= c && c < x.t1)
                for (std::size_t m = 0; m < rec.movers.size(); ++m)
                    if (rec.movers[m].name == to_string(static_cast<Mover>(x.mover)))
                        ++at[rec.stations.size() + m][c];
        at[rec.station(rec.device + "/l3[*]")][c] += static_cast<std::uint32_t>(rec.foreign_slots);
    }
    return at;
}

} // namespace

TEST_CASE("every pyramid bin is the exact merge of its two children", "[program][record][lod]") {
    const TileFlowRecord rec = record_of(64, 16, 0);
    const Lod lod = build_lod(rec, 16);           // a small base, so there are many levels
    REQUIRE(lod.levels.size() > 3);
    CHECK(lod.levels.front().bins <= 16);
    CHECK(lod.levels.back().bins == 1);
    const std::size_t rows = lod.rows.size();
    for (std::size_t i = 1; i < lod.levels.size(); ++i) {
        const LodLevel m = merge_level(lod.levels[i - 1], rows);
        CHECK(m.k == lod.levels[i].k);
        CHECK(m.occ == lod.levels[i].occ);
        CHECK(m.peak == lod.levels[i].peak);
        CHECK(m.starts == lod.levels[i].starts);
    }
}

TEST_CASE("the coarsest bin equals the raw totals", "[program][record][lod]") {
    const TileFlowRecord rec = record_of(64, 16, 0, 2);
    const Lod lod = build_lod(rec);
    const LodLevel& top = lod.levels.back();
    const std::uint32_t l3 = rec.station(rec.device + "/l3[*]");

    double l3_occ = static_cast<double>(rec.foreign_slots) * static_cast<double>(rec.makespan);
    for (const Residency& x : rec.residency) l3_occ += static_cast<double>(x.t1 - x.t0);
    CHECK(top.occ[l3] == l3_occ);
    CHECK(top.peak[l3] == peak_l3_occupancy(rec));
    CHECK(top.starts[l3] == rec.residency.size());

    double busy = 0;
    std::uint32_t computes = 0;
    for (std::size_t r = 0; r < rec.stations.size(); ++r)
        if (rec.stations[r].kind == "cf") { busy += top.occ[r]; computes += top.starts[r]; }
    double want = 0;
    for (const Compute& c : rec.computes) want += static_cast<double>(c.t1 - c.t0);
    CHECK(busy == want);
    CHECK(computes == rec.computes.size());

    double moving = 0, transit_time = 0;
    std::uint32_t moves = 0;
    for (std::size_t m = 0; m < rec.movers.size(); ++m) {
        moving += top.occ[rec.stations.size() + m];
        moves += top.starts[rec.stations.size() + m];
        CHECK(top.peak[rec.stations.size() + m] <= rec.movers[m].lanes);   // lanes are lanes
    }
    for (const Transit& t : rec.transits) transit_time += static_cast<double>(t.t1 - t.t0);
    CHECK(moving == transit_time);
    CHECK(moves == rec.transits.size());
}

TEST_CASE("at one cycle per bin the base level is a cycle-by-cycle count", "[program][record][lod]") {
    const TileFlowRecord rec = record_of(32, 16, 7, 1);   // 6 live + 1 foreign: tight, not wedged
    const Lod lod = build_lod(rec, rec.makespan);  // width 1
    const LodLevel& base = lod.levels.front();
    REQUIRE(base.k == 0);
    REQUIRE(base.bins == rec.makespan);
    const auto want = brute(rec, lod);
    for (std::size_t r = 0; r < lod.rows.size(); ++r)
        for (Cycle c = 0; c < rec.makespan; ++c) {
            if (base.peak[r * base.bins + c] != want[r][c] || base.occ[r * base.bins + c] != want[r][c]) {
                INFO("row " << lod.rows[r].name << " cycle " << c);
                CHECK(base.peak[r * base.bins + c] == want[r][c]);
                CHECK(base.occ[r * base.bins + c] == want[r][c]);
            }
        }
}

TEST_CASE("the pyramid is written with the bundle and reads back unchanged", "[program][record][lod]") {
    const TileFlowRecord rec = record_of(32, 16, 0);
    const auto dir = (std::filesystem::temp_directory_path() / "kpu-tflow-lod-test").string();
    std::filesystem::remove_all(dir);
    write_tflow(rec, dir);
    const Lod want = build_lod(rec);
    const Lod got = read_lod(dir);
    REQUIRE(got.rows.size() == want.rows.size());
    REQUIRE(got.levels.size() == want.levels.size());
    for (std::size_t i = 0; i < want.levels.size(); ++i) {
        CHECK(got.levels[i].occ == want.levels[i].occ);
        CHECK(got.levels[i].peak == want.levels[i].peak);
        CHECK(got.levels[i].starts == want.levels[i].starts);
    }
    // Rows that the level does not model say so.
    for (const LodRow& r : got.rows)
        if (r.kind == "l2" || r.kind == "l1") CHECK_FALSE(r.modelled);
}

TEST_CASE("a hostile pyramid file is refused, not allocated", "[program][record][lod]") {
    const TileFlowRecord rec = record_of(32, 16, 0);
    const auto dir = (std::filesystem::temp_directory_path() / "kpu-tflow-lod-hostile").string();
    std::filesystem::remove_all(dir);
    write_tflow(rec, dir);
    std::stringstream ss;
    ss << std::ifstream(dir + "/lod.json").rdbuf();
    const std::string good = ss.str();
    auto with_bins = [&](const std::string& bins) {
        std::string s = good;
        const auto p = s.find("\"bins\": ");
        const auto e = s.find_first_of(",\n}", p + 8);
        s.replace(p, e - p, "\"bins\": " + bins);
        std::ofstream(dir + "/lod.json", std::ios::trunc) << s;
    };
    with_bins("0");
    CHECK_THROWS_AS(read_lod(dir), RecordError);
    with_bins("18446744073709551615");
    CHECK_THROWS_AS(read_lod(dir), RecordError);
}
