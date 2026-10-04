// ============================================================================
// tests/program/test_floorplan.cpp
// The physical shape (#286 step 1): the array layout and its folded-torus NoC, and the
// floorplan generated from it -- every resource placed exactly once, nothing placed that the
// naming map cannot address, and a JSON form that round-trips and refuses what does not fit.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/floorplan.hpp>

#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>

using namespace sw::kpu::program::platform;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::Dim;

namespace {

const char* kDeploy = "tests/program/deploy/";

DeploymentSpec t64() { return read_spec_file(std::string(kDeploy) + "kpu_t64.json"); }
DeploymentSpec t4() { return read_spec_file(std::string(kDeploy) + "kpu_t4.json"); }

ArrayLayout t64_layout() {
    std::string why;
    auto L = ArrayLayout::of(t64().device(0), &why);
    REQUIRE(L.has_value());
    return *L;
}

std::map<BlockKind, std::size_t> kind_counts(const SocFloorplan& fp) {
    std::map<BlockKind, std::size_t> n;
    std::function<void(const std::vector<FloorplanBlock>&)> walk = [&](const auto& bs) {
        for (const FloorplanBlock& b : bs) {
            ++n[b.kind];
            walk(b.children);
        }
    };
    walk(fp.blocks);
    return n;
}

// Edit one block of a floorplan in place, by name.
void edit(std::vector<FloorplanBlock>& bs, const std::string& name,
          const std::function<void(FloorplanBlock&)>& f) {
    for (FloorplanBlock& b : bs) {
        if (b.name == name) f(b);
        edit(b.children, name, f);
    }
}

} // namespace

// ---- the layout -------------------------------------------------------------
TEST_CASE("the T64 is an 8x8 alternating board, and its BlockMovers sit on abutting edges",
          "[program][platform][layout]") {
    const ArrayLayout L = t64_layout();
    CHECK(L.rows() == 8);
    CHECK(L.cols() == 8);
    CHECK(L.l3_count() == 32);
    CHECK(L.cf_count() == 32);
    CHECK(L.cell(0, 0).kind == CellKind::L3);
    CHECK(L.cell(0, 1).kind == CellKind::Compute);

    // Every horizontal and vertical neighbour pair is one L3 and one compute tile, so the
    // BlockMover count is the number of such pairs: 2 x 8 x 7.
    CHECK(L.block_movers().size() == 112);

    // A corner tile abuts compute tiles only to the E and S; an interior tile on all four.
    std::map<Dim, std::set<Edge>> edges;
    for (const BlockMoverSite& m : L.block_movers()) edges[m.l3].insert(m.edge);
    CHECK(edges[L.cell(0, 0).index] == std::set<Edge>{Edge::E, Edge::S});
    CHECK(edges[L.cell(2, 2).index].size() == 4);
    // And the compute tile a mover feeds is the one across its edge.
    for (const BlockMoverSite& m : L.block_movers()) {
        const GridPos a = L.l3_pos(m.l3), b = L.cf_pos(m.cf);
        CHECK(std::abs(static_cast<int>(a.r) - static_cast<int>(b.r)) +
                  std::abs(static_cast<int>(a.c) - static_cast<int>(b.c)) == 1);
    }
}

TEST_CASE("the NoC is a folded 2D torus: 4x4 loops of 8 hubs, with fold-end ports",
          "[program][platform][layout][noc]") {
    const ArrayLayout L = t64_layout();
    REQUIRE(L.has_noc());
    REQUIRE(L.loops().size() == 8);      // 4 row loops + 4 column loops

    // The ring order the architect gave for rows 0-1.
    auto at = [&](Dim r, Dim c) { return L.cell(r, c).index; };
    const std::vector<Dim> row0 = {at(0, 0), at(0, 2), at(0, 4), at(0, 6),
                                   at(1, 7), at(1, 5), at(1, 3), at(1, 1)};
    CHECK(L.loops()[0].hubs == row0);
    const std::vector<Dim> col0 = {at(0, 0), at(2, 0), at(4, 0), at(6, 0),
                                   at(7, 1), at(5, 1), at(3, 1), at(1, 1)};
    CHECK(L.loops()[4].hubs == col0);

    // Every hub is on exactly one row loop and one column loop; every link is two cells
    // long except the fold links, which join diagonal neighbours.
    std::map<Dim, int> row_hits, col_hits;
    for (const NocLoop& loop : L.loops()) {
        REQUIRE(loop.hubs.size() == 8);
        for (std::size_t i = 0; i < loop.hubs.size(); ++i) {
            (loop.axis == NocLoop::Axis::Row ? row_hits : col_hits)[loop.hubs[i]]++;
            const GridPos a = L.l3_pos(loop.hubs[i]), b = L.l3_pos(loop.hubs[(i + 1) % 8]);
            const int dr = std::abs(static_cast<int>(a.r) - static_cast<int>(b.r));
            const int dc = std::abs(static_cast<int>(a.c) - static_cast<int>(b.c));
            const bool fold = i == 3 || i == 7;
            CHECK(((fold && dr == 1 && dc == 1) || (!fold && dr + dc == 2 && (dr == 0 || dc == 0))));
        }
    }
    CHECK(row_hits.size() == 32);
    CHECK(col_hits.size() == 32);
    for (const auto& [hub, n] : row_hits) CHECK(n == 1);
    for (const auto& [hub, n] : col_hits) CHECK(n == 1);

    // 16 fold-end ports, numbered row loops first.
    REQUIRE(L.ports().size() == 16);
    CHECK(L.ports()[0].label() == "row0.W");
    CHECK(L.ports()[1].label() == "row0.E");
    CHECK(L.ports()[8].label() == "col0.N");
    CHECK(L.ports()[15].label() == "col3.S");

    // At all FOUR corners a row loop and a column loop fold over the SAME link -- (0,0)-(1,1),
    // (0,6)-(1,7), (6,0)-(7,1), (6,6)-(7,7) -- so the 64 loop links are 60 distinct wires and
    // the eight corner hubs have three distinct neighbours, not four. Stated, because the
    // connectivity study has to know.
    CHECK(L.links().size() == 60);
    std::set<std::pair<Dim, Dim>> shared;
    for (auto [a, b] : {std::pair<Dim, Dim>{at(0, 0), at(1, 1)}, {at(0, 6), at(1, 7)},
                        {at(6, 0), at(7, 1)}, {at(6, 6), at(7, 7)}})
        shared.insert(a < b ? std::pair{a, b} : std::pair{b, a});
    std::map<std::pair<Dim, Dim>, int> uses;
    for (const NocLoop& loop : L.loops())
        for (std::size_t i = 0; i < 8; ++i) {
            Dim a = loop.hubs[i], b = loop.hubs[(i + 1) % 8];
            ++uses[a < b ? std::pair{a, b} : std::pair{b, a}];
        }
    for (const auto& [link, n] : uses) CHECK((n == 2) == (shared.count(link) == 1));

    // And every hub can reach every other.
    std::map<Dim, std::set<Dim>> adj;
    for (const auto& [a, b] : L.links()) { adj[a].insert(b); adj[b].insert(a); }
    std::set<Dim> seen{0};
    std::deque<Dim> q{0};
    while (!q.empty()) {
        const Dim h = q.front();
        q.pop_front();
        for (Dim n : adj[h])
            if (seen.insert(n).second) q.push_back(n);
    }
    CHECK(seen.size() == 32);
}

// ---- the T4: the reference SKU for evaluating schedules --------------------------
// Small enough to read every tile movement in the viewer: a 2x2 board, two 64x64-PE compute
// tiles (4096 MACs per cycle each), two L3 tiles, one memory controller with 8 DMA engines.
TEST_CASE("the T4 is a 2x2 alternating board whose L3 tiles each feed both compute tiles",
          "[program][platform][layout][t4]") {
    std::string why;
    const auto L = ArrayLayout::of(t4().device(0), &why);
    REQUIRE(L.has_value());
    CHECK(L->rows() == 2);
    CHECK(L->cols() == 2);
    CHECK(L->l3_count() == 2);
    CHECK(L->cf_count() == 2);
    CHECK(L->cell(0, 0).kind == CellKind::L3);
    CHECK(L->cell(1, 1).kind == CellKind::L3);
    CHECK(L->cell(0, 1).kind == CellKind::Compute);
    CHECK(L->cell(1, 0).kind == CellKind::Compute);

    // Each L3 tile abuts both compute tiles, so there are 2 x 2 BlockMovers and every
    // (L3, compute) pair has its own.
    REQUIRE(L->block_movers().size() == 4);
    std::map<Dim, std::set<Edge>> edges;
    std::set<std::pair<Dim, Dim>> pairs;
    for (const BlockMoverSite& m : L->block_movers()) {
        edges[m.l3].insert(m.edge);
        pairs.insert({m.l3, m.cf});
    }
    CHECK(edges[L->cell(0, 0).index] == std::set<Edge>{Edge::E, Edge::S});
    CHECK(edges[L->cell(1, 1).index] == std::set<Edge>{Edge::N, Edge::W});
    CHECK(pairs.size() == 4);

    // The torus degenerates: the row loop and the column loop are the same two-hub ring, so
    // the NoC is ONE wire between the two L3 hubs, still with four fold-end ports.
    REQUIRE(L->has_noc());
    REQUIRE(L->loops().size() == 2);
    const std::vector<Dim> ring = {L->cell(0, 0).index, L->cell(1, 1).index};
    for (const NocLoop& loop : L->loops()) CHECK(loop.hubs == ring);
    CHECK(L->links().size() == 1);
    REQUIRE(L->ports().size() == 4);
    CHECK(L->ports()[0].label() == "row0.W");
    CHECK(L->ports()[3].label() == "col0.S");
}

TEST_CASE("the generated T4 floorplan places every resource exactly once",
          "[program][platform][floorplan][t4]") {
    const DeploymentSpec spec = t4();
    REQUIRE(spec.validate().empty());
    const SocFloorplan fp = generate_floorplan(spec);
    CHECK(validate_floorplan(fp, spec).empty());

    const auto n = kind_counts(fp);
    CHECK(n.at(BlockKind::L3Tile) == 2);
    CHECK(n.at(BlockKind::ComputeTile) == 2);
    CHECK(n.at(BlockKind::BlockMover) == 4);
    CHECK(n.at(BlockKind::NocRouter) == 2);
    CHECK(n.at(BlockKind::NocPort) == 4);
    CHECK(n.at(BlockKind::MemoryController) == 1);
    CHECK(n.at(BlockKind::DmaEngine) == 8);
    CHECK(n.at(BlockKind::CpuHart) == 1);
    CHECK(n.at(BlockKind::L3Bank) == 2 * 4);

    std::map<NocLink::Kind, std::size_t> links;
    for (const NocLink& l : fp.noc) ++links[l.kind];
    CHECK(links[NocLink::Kind::Ring] == 1);
    CHECK(links[NocLink::Kind::Port] == 8);
    CHECK(links[NocLink::Kind::Attach] == 1);   // first pass: one per controller
}

TEST_CASE("a spec that describes no layout has none, and says why",
          "[program][platform][layout]") {
    // The checked-in 16-CF / 8-L3 fixture is a valid deployment with no alternating layout.
    const DeploymentSpec adr = read_spec_file(std::string(kDeploy) + "adr_checkerboard16.json");
    std::string why;
    CHECK_FALSE(ArrayLayout::of(adr.device(0), &why).has_value());
    CHECK_THAT(why, ContainsSubstring("as many L3 tiles as compute tiles"));

    // news: one compute tile fed by four L3 tiles, one BlockMover each, no NoC.
    DeviceSpecification news;
    news.topology = "news";
    news.compute_tiles = 1;
    news.l3.tiles = 4;
    const auto N = ArrayLayout::of(news);
    REQUIRE(N.has_value());
    CHECK(N->block_movers().size() == 4);
    CHECK_FALSE(N->has_noc());
    CHECK_THAT(N->noc_reason(), ContainsSubstring("no NoC"));

    // The plan's Q8: a declared pool that disagrees with the layout is reported, not refused.
    DeviceSpecification d = t64().device(0);
    d.movers.block_movers = 1;
    const auto notes = layout_notes(d);
    REQUIRE(notes.size() == 1);
    CHECK_THAT(notes[0], ContainsSubstring("112"));
    CHECK(layout_notes(t64().device(0)).empty());
}

// ---- the floorplan ------------------------------------------------------------
TEST_CASE("the generated T64 floorplan places every resource exactly once",
          "[program][platform][floorplan]") {
    const DeploymentSpec spec = t64();
    const SocFloorplan fp = generate_floorplan(spec);
    CHECK(validate_floorplan(fp, spec).empty());
    CHECK(fp.source == "generated:checkerboard");

    const auto n = kind_counts(fp);
    CHECK(n.at(BlockKind::L3Tile) == 32);
    CHECK(n.at(BlockKind::ComputeTile) == 32);
    CHECK(n.at(BlockKind::BlockMover) == 112);
    CHECK(n.at(BlockKind::NocRouter) == 32);
    CHECK(n.at(BlockKind::NocPort) == 16);
    CHECK(n.at(BlockKind::MemoryController) == 4);
    CHECK(n.at(BlockKind::DmaEngine) == 8);
    CHECK(n.at(BlockKind::CpuHart) == 4);
    CHECK(n.at(BlockKind::L3Bank) == 32 * 4);

    // The BlockMovers are children of their L3 tile, in the L3 clock domain.
    const FloorplanBlock* bm = fp.find("t64/l3[0]/bm[1]");
    REQUIRE(bm != nullptr);
    CHECK(bm->clock_domain == "l3");
    CHECK(fp.find("t64/l3[0]/bm[0]") == nullptr);           // a corner has no N mover

    // The NoC's wires: 60 ring links (64 loop links, four shared at the corners), two port links per port, and one DMA attachment per
    // N/S port in the first pass (W/E ports start unattached).
    std::map<NocLink::Kind, std::size_t> links;
    for (const NocLink& l : fp.noc) ++links[l.kind];
    CHECK(links[NocLink::Kind::Ring] == 60);
    CHECK(links[NocLink::Kind::Port] == 32);
    CHECK(links[NocLink::Kind::Attach] == 8);
}

TEST_CASE("the die holds everything placed, even when the CPU is taller than the array",
          "[program][platform][floorplan]") {
    // A 2x4 board (4 L3 + 4 compute tiles) is far shorter than an 8-hart CPU column and the IO
    // block below it -- by more than the die margin, which is what made the first version of
    // this test pass against the old sizing. The die must be sized from what was placed.
    DeploymentSpec spec = t64();
    DeviceSpecification& d = spec.device(0);
    d.compute_tiles = 4;
    d.l3.tiles = 4;
    d.array.rows = 2;
    d.array.cols = 4;
    d.movers.block_movers = 10;
    d.cpu.harts = 8;
    REQUIRE(spec.validate().empty());
    const SocFloorplan fp = generate_floorplan(spec);
    CHECK(validate_floorplan(fp, spec).empty());
    const FloorplanBlock* io = fp.find("t64/io");
    REQUIRE(io != nullptr);
    CHECK(io->rect.inside(fp.die));
    CHECK(fp.find("t64/cpu")->rect.inside(fp.die));
}

TEST_CASE("the floorplan JSON round-trips, byte for byte, and is deterministic",
          "[program][platform][floorplan]") {
    const DeploymentSpec spec = t64();
    const SocFloorplan fp = generate_floorplan(spec);
    const std::string text = floorplan_to_json(fp);
    CHECK(floorplan_to_json(generate_floorplan(spec)) == text);
    const SocFloorplan back = floorplan_from_json(text, spec);
    CHECK(floorplan_to_json(back) == text);
    CHECK(back.digest() == fp.digest());
    CHECK(back.block_count() == fp.block_count());
}

TEST_CASE("an imported floorplan that does not fit the deployment is refused, by name",
          "[program][platform][floorplan]") {
    const DeploymentSpec spec = t64();
    const SocFloorplan good = generate_floorplan(spec);
    auto refuse = [&](SocFloorplan fp) {
        try {
            floorplan_from_json(floorplan_to_json(fp), spec);
        } catch (const FloorplanError& e) {
            return std::string(e.what());
        }
        return std::string("accepted");
    };

    SECTION("a resource with no block") {
        SocFloorplan fp = good;
        edit(fp.blocks, "t64/l3[0]", [](FloorplanBlock& b) {
            std::erase_if(b.children, [](const FloorplanBlock& c) { return c.name == "t64/l3[0]/bm[1]"; });
        });
        CHECK_THAT(refuse(fp), ContainsSubstring("\"t64/l3[0]/bm[1]\" has no block"));
    }
    SECTION("a block naming a resource that does not exist") {
        SocFloorplan fp = good;
        edit(fp.blocks, "t64/l3[0]/bm[1]", [](FloorplanBlock& b) {
            b.resource->path = {0, 0};
            b.name = format(*b.resource);
        });
        CHECK_THAT(refuse(fp), ContainsSubstring("no compute tile on its N edge"));
    }
    SECTION("overlapping siblings") {
        SocFloorplan fp = good;
        const Rect bank = fp.find("t64/l3[0]/bank[0]")->rect;
        edit(fp.blocks, "t64/l3[0]/noc", [&](FloorplanBlock& b) { b.rect = bank; });
        CHECK_THAT(refuse(fp), ContainsSubstring("overlaps its sibling"));
    }
    SECTION("a child outside its parent") {
        SocFloorplan fp = good;
        edit(fp.blocks, "t64/l3[0]/noc", [](FloorplanBlock& b) { b.rect.x_um -= 5000; });
        CHECK_THAT(refuse(fp), ContainsSubstring("outside its parent"));
    }
    SECTION("a group that names a resource") {
        SocFloorplan fp = good;
        edit(fp.blocks, "t64/io", [](FloorplanBlock& b) { b.resource = parse_resource_name("t64/cpu/sram"); });
        CHECK_THAT(refuse(fp), ContainsSubstring("is a group"));
    }
    SECTION("a floorplan of another device") {
        SocFloorplan fp = good;
        fp.device = "t256";
        CHECK_THAT(refuse(fp), ContainsSubstring("does not declare"));
    }
    SECTION("a newer format") {
        std::string text = floorplan_to_json(good);
        text.replace(text.find("\"version\": 1"), 12, "\"version\": 2");
        CHECK_THROWS_WITH(floorplan_from_json(text, spec), ContainsSubstring("version 2"));
    }
}

TEST_CASE("a device with no layout has no floorplan, and the refusal says why",
          "[program][platform][floorplan]") {
    const DeploymentSpec adr = read_spec_file(std::string(kDeploy) + "adr_checkerboard16.json");
    CHECK_THROWS_WITH(generate_floorplan(adr), ContainsSubstring("no array layout") &&
                                                   ContainsSubstring("as many L3 tiles"));
}

TEST_CASE("the SVG draws every block and names it", "[program][platform][floorplan]") {
    const SocFloorplan fp = generate_floorplan(t64());
    const std::string svg = floorplan_to_svg(fp);
    std::size_t rects = 0;
    for (std::size_t p = svg.find("<rect"); p != std::string::npos; p = svg.find("<rect", p + 1)) ++rects;
    CHECK(rects == fp.block_count() + 1);                    // + the die
    CHECK_THAT(svg, ContainsSubstring("<title>t64/l3[0]/bm[1] (l3 clock)</title>"));
    CHECK_THAT(svg, ContainsSubstring("ring: t64/l3[0]/noc"));
}
