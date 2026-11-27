// ============================================================================
// tests/program/test_mesh_layout.cpp
// The DMM mesh as a deployment (#604; docs/plans/distributed-csp-programs.md §3.1, §4 step 1): the
// `dmm_mesh` topology and its periphery are validated and round-trip; MeshLayout derives the nodes,
// the one-hop links, the edge ports and the periphery ring's stops; the naming map names the eight
// BlockMovers per node and the edge ports; the floorplan places every resource once, and its
// structure record carries the mesh, the credit network and the periphery ring.
//
// Paths are relative to the project root; WORKING_DIRECTORY is CMAKE_SOURCE_DIR.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/deployment_json.hpp>
#include <sw/kpu/program/platform/floorplan.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>
#include <sw/kpu/program/platform/resource_map.hpp>

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace sw::kpu::program::platform;
using Catch::Matchers::ContainsSubstring;
using sw::kpu::program::Dim;

namespace {

const char* kDeploy = "tests/program/deploy/";
DeploymentSpec dmm2x2() { return read_spec_file(std::string(kDeploy) + "kpu_dmm_2x2.json"); }
DeploymentSpec dmm4x4() { return read_spec_file(std::string(kDeploy) + "kpu_dmm_4x4.json"); }

MeshLayout layout(const DeploymentSpec& s) {
    std::string why;
    const auto L = MeshLayout::of(s.device(0), &why);
    INFO(why);
    REQUIRE(L.has_value());
    return *L;
}

// A one-device dmm_mesh spec, edited by `f`, and its validation.
std::string problem(const std::function<void(DeviceSpecification&)>& f) {
    DeploymentSpec s = dmm2x2();
    f(s.device(0));
    return s.validate();
}

} // namespace

TEST_CASE("the DMM mesh samples load, validate and round-trip byte for byte", "[program][platform][mesh]") {
    for (const DeploymentSpec& s : {dmm2x2(), dmm4x4()}) {
        CHECK(s.validate().empty());
        CHECK(s.device(0).topology == "dmm_mesh");
        const std::string once = to_json(s);
        CHECK(to_json(from_json(once)) == once);
    }
    const DeviceSpecification d = dmm4x4().device(0);
    REQUIRE(d.periphery.has_value());
    CHECK(d.periphery->mc_after_port == std::vector<Dim>{1, 3, 5, 7, 9, 11, 13, 15});
    // The periphery is declared and laid out, and no level times it yet (#618).
    CHECK(declared(d, SpecField::Periphery));
    CHECK(declared_value(d, SpecField::Periphery) == "ring, 2 blocks per input, controllers after ports 1,3,5,7,9,11,13,15");
    for (auto l : {sw::kpu::program::driver::ExecutionLevel::Behavioral, sw::kpu::program::driver::ExecutionLevel::BlockSequential,
                   sw::kpu::program::driver::ExecutionLevel::CycleAccurate})
        CHECK_FALSE(sw::kpu::program::driver::level_models(l, SpecField::Periphery));
}

TEST_CASE("a dmm_mesh that cannot be built is refused, naming the field", "[program][platform][mesh]") {
    CHECK_THAT(problem([](auto& d) { d.array = {}; }), ContainsSubstring("declares its array.rows and array.cols"));
    // The flat counts are the hierarchy's views (#635): written to disagree, they are refused.
    CHECK_THAT(problem([](auto& d) { d.compute_tiles = 3; d.l3.tiles = 3; }),
               ContainsSubstring("compute_tiles disagrees with the cluster; on a dmm_mesh it is derived"));
    CHECK_THAT(problem([](auto& d) { d.l3.tiles.reset(); }), ContainsSubstring("l3 disagrees with the cluster"));
    CHECK_THAT(problem([](auto& d) { d.periphery->topology = "clos"; }),
               ContainsSubstring("periphery.topology 'clos' is not one of ring"));
    CHECK_THAT(problem([](auto& d) { d.periphery->queue_blocks = 1; }), ContainsSubstring("must be at least two"));
    CHECK_THAT(problem([](auto& d) { d.periphery->mc_after_port = {3}; }),
               ContainsSubstring("lists 1 position(s), one per memory controller, of which there are 2"));
    CHECK_THAT(problem([](auto& d) { d.periphery->mc_after_port = {5, 2}; }), ContainsSubstring("strictly increasing"));
    CHECK_THAT(problem([](auto& d) { d.periphery->mc_after_port = {2, 8}; }),
               ContainsSubstring("periphery.mc_after_port[1] is 8; the edge ports are 0..7"));
    CHECK_THAT(problem([](auto& d) { d.memory.controllers = 9; d.dma.engines = 9; }),
               ContainsSubstring("9 memory controllers, but the periphery has only 8 edge ports"));
    // The folded torus's hubs and ports are the checkerboard's; a dmm_mesh has none.
    CHECK_THAT(problem([](auto& d) { d.noc = DeviceSpecification::Noc{}; }),
               ContainsSubstring("the folded torus's hubs and ports are the checkerboard's"));
    // A periphery is the dmm_mesh's alone.
    DeploymentSpec t4 = read_spec_file(std::string(kDeploy) + "kpu_t4.json");
    t4.device(0).periphery = DeviceSpecification::Periphery{};
    CHECK_THAT(t4.validate(), ContainsSubstring("periphery applies to the dmm_mesh topology, not 'checkerboard'"));
    // The checkerboard's evenness rule is the folded torus's, not the mesh's: a 1x3 mesh is a mesh.
    CHECK(problem([](auto& d) {
              d.array.rows = 1; d.array.cols = 3;
              derive_mesh_views(d);
          }).empty());
}

TEST_CASE("the periphery's port indices are read as integers, and refused otherwise", "[program][platform][mesh]") {
    const std::string base = to_json(dmm2x2());
    auto with = [&](const std::string& value) {
        std::string t = base;
        const std::string key = "\"queue_blocks\": 2";
        const auto at = t.find(key);
        REQUIRE(at != std::string::npos);
        t.insert(at + key.size(), ", \"mc_after_port\": " + value);
        return t;
    };
    CHECK(from_json(with("[1, 5]")).device(0).periphery->mc_after_port == std::vector<Dim>{1, 5});
    CHECK_THROWS_WITH(from_json(with("[1, \"5\"]")), ContainsSubstring("periphery.mc_after_port[1] must be an integer"));
    CHECK_THROWS_WITH(from_json(with("[-1, 5]")), ContainsSubstring("must be an edge port index (got -1)"));
    CHECK_THROWS_WITH(from_json(with("3")), ContainsSubstring("mc_after_port must be an array"));
}

TEST_CASE("MeshLayout: nodes, one-hop links, clockwise edge ports and the ring's stops", "[program][platform][mesh]") {
    const MeshLayout L = layout(dmm4x4());
    CHECK(L.node_count() == 16);
    CHECK(L.node(2, 3) == 11);
    // Every neighbour pair, both ways: 2 x (4 x 3 + 3 x 4).
    CHECK(L.links().size() == 48);
    for (const MeshLink& k : L.links()) {
        const GridPos a = L.pos(k.from), b = L.pos(k.to);
        CHECK((a.r == b.r ? (a.c > b.c ? a.c - b.c : b.c - a.c) : (a.r > b.r ? a.r - b.r : b.r - a.r)) == 1);
        CHECK(L.neighbour(k.from, k.dir) == k.to);
    }
    CHECK_FALSE(L.neighbour(0, Edge::N).has_value());
    CHECK_FALSE(L.neighbour(15, Edge::E).has_value());
    // 2 x (rows + cols), clockwise from the NW corner.
    REQUIRE(L.edge_ports().size() == 16);
    const std::vector<std::string> labels = {"N0", "N1", "N2", "N3", "E0", "E1", "E2", "E3",
                                             "S3", "S2", "S1", "S0", "W3", "W2", "W1", "W0"};
    for (Dim k = 0; k < 16; ++k) CHECK(L.edge_ports()[k].label() == labels[k]);
    CHECK(L.edge_ports()[4].node == 3);              // E0: the NE corner
    CHECK(L.edge_ports()[8].node == 15);             // S3: the SE corner
    CHECK(L.edge_port(0, Edge::W) == Dim{15});       // the NW corner has two ports, N0 and W0
    CHECK_FALSE(L.edge_port(5, Edge::N).has_value());
    // The ring: every port, each controller after the port the spec names.
    REQUIRE(L.stops().size() == 24);
    CHECK(L.stops()[2].kind == PeripheryStop::Kind::Controller);
    CHECK(L.stops()[2].of == 0);
    CHECK(L.stops()[23].kind == PeripheryStop::Kind::Controller);
    CHECK(L.stops()[23].of == 7);
}

TEST_CASE("MeshLayout spreads undeclared controllers evenly around the ring", "[program][platform][mesh]") {
    const MeshLayout L = layout(dmm2x2());                // 8 ports, 2 controllers
    CHECK(L.controller_count() == 2);
    CHECK(L.mc_after_port(0) == 2);
    CHECK(L.mc_after_port(1) == 6);
    DeploymentSpec s = dmm2x2();
    s.device(0).memory.controllers = 8;
    s.device(0).dma.engines = 8;
    const MeshLayout full = layout(s);                    // one after every port
    for (Dim m = 0; m < 8; ++m) CHECK(full.mc_after_port(m) == m);
    CHECK(full.stops().size() == 16);
}

TEST_CASE("a dmm_mesh has no checkerboard layout, and a checkerboard no mesh layout", "[program][platform][mesh]") {
    std::string why;
    CHECK_FALSE(ArrayLayout::of(dmm2x2().device(0), &why).has_value());
    CHECK_THAT(why, ContainsSubstring("its layout is a MeshLayout"));
    CHECK_FALSE(MeshLayout::of(read_spec_file(std::string(kDeploy) + "kpu_t4.json").device(0), &why).has_value());
    CHECK_THAT(why, ContainsSubstring("is not a dmm_mesh"));
}

TEST_CASE("the naming map names eight BlockMovers per node and the edge ports, and no routers", "[program][platform][mesh]") {
    const ResourceMap map(dmm2x2());
    std::size_t movers = 0, ports = 0;
    for (const ResourceName& n : map.enumerate()) {
        movers += n.kind == ResourceKind::BlockMover;
        ports += n.kind == ResourceKind::NocPort;
        CHECK(n.kind != ResourceKind::NocRouter);
    }
    CHECK(movers == 4 * kMeshMovers);
    CHECK(ports == 8);
    CHECK_NOTHROW(map.require("dmm2x2/l3[3]/bm[7]"));
    CHECK_NOTHROW(map.require("dmm2x2/noc/port[7]"));
    CHECK_THROWS_WITH(map.require("dmm2x2/l3[0]/bm[8]"), ContainsSubstring("a mesh node has 8 BlockMovers"));
    CHECK_THROWS_WITH(map.require("dmm2x2/noc/port[8]"), ContainsSubstring("the mesh has 8 edge port(s)"));
    CHECK_THROWS_WITH(map.require("dmm2x2/l3[0]/noc"), ContainsSubstring("a dmm_mesh has no routers"));
    CHECK(std::string(to_string(MoverRole::FeedX)) == "x");
    CHECK(send_role(Edge::W) == MoverRole::SendW);
}

TEST_CASE("the mesh floorplan places every resource once, and draws the mesh, credit and periphery wires",
          "[program][platform][mesh][floorplan]") {
    for (const DeploymentSpec& s : {dmm2x2(), dmm4x4()}) {
        const SocFloorplan fp = generate_floorplan(s);
        INFO(s.device(0).name);
        CHECK(fp.source == "generated:dmm_mesh");
        CHECK(validate_floorplan(fp, s).empty());          // every resource exactly once, inside the die
        const MeshLayout L = layout(s);
        std::map<NocLink::Kind, std::size_t> n;
        for (const NocLink& l : fp.noc) ++n[l.kind];
        CHECK(n[NocLink::Kind::Mesh] == L.links().size() / 2);     // a pair carries both directions
        CHECK(n[NocLink::Kind::Credit] == L.links().size() / 2);
        CHECK(n[NocLink::Kind::Edge] == L.edge_ports().size());
        CHECK(n[NocLink::Kind::Periphery] == L.stops().size());     // a closed ring
        CHECK(n[NocLink::Kind::Ring] + n[NocLink::Kind::Port] + n[NocLink::Kind::Attach] == 0);
        // Round trip: the import validates it and writes the same bytes.
        CHECK(floorplan_to_json(floorplan_from_json(floorplan_to_json(fp), s)) == floorplan_to_json(fp));

        const nlohmann::json rec = nlohmann::json::parse(floorplan_to_structure(fp));
        std::map<std::string, std::string> nets;
        for (const auto& net : rec.at("networks")) nets[net.at("id")] = net.at("routing");
        CHECK(nets == std::map<std::string, std::string>{{"mesh", "one_hop"}, {"credit", "one_hop"}, {"periphery", "ring"}});
    }
}

TEST_CASE("an imported mesh floorplan whose ring joins the wrong blocks is refused", "[program][platform][mesh][floorplan]") {
    const DeploymentSpec s = dmm2x2();
    SocFloorplan fp = generate_floorplan(s);
    for (NocLink& l : fp.noc)
        if (l.kind == NocLink::Kind::Periphery) {
            l.b = "dmm2x2/l3[0]";                          // a ring stop is an edge port or a controller
            break;
        }
    CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("periphery link"));
    CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("joins the wrong kinds of block"));
}

// ---- regressions from the review of #630

TEST_CASE("MeshLayout refuses a mesh missing one of its dimensions, rather than reading it", "[program][platform][mesh]") {
    DeviceSpecification d = dmm2x2().device(0);
    d.array.cols.reset();                                  // rows declared, cols not: validate() is bypassed here
    std::string why;
    CHECK_FALSE(MeshLayout::of(d, &why).has_value());
    CHECK_THAT(why, ContainsSubstring("declares its array.rows and array.cols, both non-zero"));
    d.array.cols = 0;
    CHECK_FALSE(MeshLayout::of(d, &why).has_value());
}

TEST_CASE("a dmm_mesh declares its memory controllers, so every one it places has a name", "[program][platform][mesh]") {
    CHECK_THAT(problem([](auto& d) { d.memory.controllers.reset(); }), ContainsSubstring("declares memory.controllers"));
    DeviceSpecification d = dmm2x2().device(0);
    d.memory.controllers.reset();
    std::string why;
    CHECK_FALSE(MeshLayout::of(d, &why).has_value());
    CHECK_THAT(why, ContainsSubstring("declares memory.controllers"));
}

TEST_CASE("every DMA engine keeps a positive width, however many a controller has", "[program][platform][mesh][floorplan]") {
    DeploymentSpec s = dmm2x2();
    s.device(0).dma.engines = 64;                          // 32 per controller: wider than the style's gaps allow
    REQUIRE(s.validate().empty());
    const SocFloorplan fp = generate_floorplan(s);
    CHECK(validate_floorplan(fp, s).empty());
    CHECK_NOTHROW(floorplan_from_json(floorplan_to_json(fp), s));
    // The checkerboard's controllers share the rule.
    DeploymentSpec t4 = read_spec_file(std::string(kDeploy) + "kpu_t4.json");
    t4.device(0).dma.engines = 64;
    REQUIRE(t4.validate().empty());
    CHECK(validate_floorplan(generate_floorplan(t4), t4).empty());
}

TEST_CASE("an imported mesh floorplan must have exactly its layout's wires", "[program][platform][mesh][floorplan]") {
    const DeploymentSpec s = dmm4x4();
    const SocFloorplan base = generate_floorplan(s);
    auto first = [](SocFloorplan& fp, NocLink::Kind k) -> NocLink& {
        for (NocLink& l : fp.noc)
            if (l.kind == k) return l;
        FAIL("no link of that kind");
        return fp.noc.front();
    };
    {   // two L3 tiles, but not neighbours
        SocFloorplan fp = base;
        first(fp, NocLink::Kind::Mesh).b = "dmm4x4/l3[15]";
        CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("does not join one-hop neighbours of the mesh"));
    }
    {   // a ring segment from a stop to itself
        SocFloorplan fp = base;
        NocLink& l = first(fp, NocLink::Kind::Periphery);
        l.b = l.a;
        CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("does not join consecutive stops of the periphery ring"));
    }
    {   // an edge port wired to another node
        SocFloorplan fp = base;
        first(fp, NocLink::Kind::Edge).b = "dmm4x4/l3[5]";
        CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("does not join an edge port to its own node"));
    }
    {   // a wire drawn twice, and a wire left out
        SocFloorplan fp = base;
        fp.noc.push_back(first(fp, NocLink::Kind::Credit));
        CHECK_THAT(validate_floorplan(fp, s), ContainsSubstring("appears twice"));
        SocFloorplan gap = base;
        for (std::size_t i = 0; i < gap.noc.size(); ++i)
            if (gap.noc[i].kind == NocLink::Kind::Periphery) { gap.noc.erase(gap.noc.begin() + static_cast<long>(i)); break; }
        CHECK_THAT(validate_floorplan(gap, s), ContainsSubstring("is missing"));
    }
    {   // a mesh wire on a checkerboard
        const DeploymentSpec t4 = read_spec_file(std::string(kDeploy) + "kpu_t4.json");
        SocFloorplan fp = generate_floorplan(t4);
        fp.noc.push_back({NocLink::Kind::Mesh, "t4/l3[0]", "t4/l3[1]", {{0, 0}, {1, 1}}});
        CHECK_THAT(validate_floorplan(fp, t4), ContainsSubstring("is a dmm_mesh wire, and device \"t4\" is not a dmm_mesh"));
    }
    // The neighbour pair may be written either way round.
    SocFloorplan swapped = base;
    std::swap(first(swapped, NocLink::Kind::Mesh).a, first(swapped, NocLink::Kind::Mesh).b);
    CHECK(validate_floorplan(swapped, s).empty());
}

TEST_CASE("the structure record states both directions of every mesh, credit and edge wire",
          "[program][platform][mesh][floorplan]") {
    const nlohmann::json rec = nlohmann::json::parse(floorplan_to_structure(generate_floorplan(dmm2x2())));
    for (const char* net : {"mesh", "credit"}) {
        std::set<std::pair<std::string, std::string>> dirs;
        for (const auto& n : rec.at("networks"))
            if (n.at("id") == net)
                for (const auto& l : n.at("links")) dirs.insert({l.at("from").get<std::string>(), l.at("to").get<std::string>()});
        INFO(net);
        CHECK(dirs.count({"dmm2x2/l3[0]", "dmm2x2/l3[1]"}));
        CHECK(dirs.count({"dmm2x2/l3[1]", "dmm2x2/l3[0]"}));
        CHECK(dirs.count({"dmm2x2/l3[0]", "dmm2x2/l3[2]"}));
        CHECK(dirs.count({"dmm2x2/l3[2]", "dmm2x2/l3[0]"}));
    }
    std::set<std::pair<std::string, std::string>> edge;
    for (const auto& n : rec.at("networks"))
        if (n.at("id") == "mesh")
            for (const auto& l : n.at("links")) edge.insert({l.at("from").get<std::string>(), l.at("to").get<std::string>()});
    CHECK(edge.count({"dmm2x2/noc/port[0]", "dmm2x2/l3[0]"}));
    CHECK(edge.count({"dmm2x2/l3[0]", "dmm2x2/noc/port[0]"}));
}

TEST_CASE("the structure record's origin is the die's lower-left: north is up", "[program][platform][mesh][floorplan]") {
    const SocFloorplan fp = generate_floorplan(dmm4x4());
    const nlohmann::json rec = nlohmann::json::parse(floorplan_to_structure(fp));
    std::map<std::string, std::vector<double>> rect;
    for (const auto& b : rec.at("blocks")) rect[b.at("id")] = b.at("rect").get<std::vector<double>>();
    const auto& die = rect.at("dmm4x4/die");
    CHECK(die[0] == 0.0);
    CHECK(die[1] == 0.0);
    // Row 0 is the north row: in a lower-left frame it lies above row 3, and N0 above both.
    CHECK(rect.at("dmm4x4/l3[0]")[1] > rect.at("dmm4x4/l3[12]")[1]);
    CHECK(rect.at("dmm4x4/noc/port[0]")[1] > rect.at("dmm4x4/l3[0]")[1]);
    CHECK(rect.at("dmm4x4/noc/port[11]")[1] < rect.at("dmm4x4/l3[12]")[1]);   // S0, below the south row
    // Every block stays on the die.
    for (const auto& [id, r] : rect) {
        INFO(id);
        CHECK(r[1] >= -1e-9);
        CHECK(r[1] + r[3] <= die[3] + 1e-9);
    }
}

TEST_CASE("every cell of a tile's grid keeps an area, however many the tile declares", "[program][platform][mesh][floorplan]") {
    for (const char* f : {"kpu_dmm_2x2.json", "kpu_t4.json"}) {
        DeploymentSpec s = read_spec_file(std::string(kDeploy) + f);
        DeviceSpecification& dv = s.device(0);
        if (dv.cluster) {                                  // a mesh: its cluster, then its views
            dv.cluster->compute_tile.l1.vectors = 16;      // one column: more gaps than the style's height allows
            dv.cluster->l3_tile.banks = 64;
            derive_mesh_views(dv);
        } else {
            dv.l1.vectors = 16;
            dv.l3.banks = 64;
        }
        REQUIRE(s.validate().empty());
        const SocFloorplan fp = generate_floorplan(s);
        INFO(f);
        CHECK(validate_floorplan(fp, s).empty());
        CHECK_NOTHROW(floorplan_from_json(floorplan_to_json(fp), s));
    }
}

TEST_CASE("the structure record is die-local: a die that does not start at the origin moves nothing",
          "[program][platform][mesh][floorplan]") {
    const SocFloorplan fp = generate_floorplan(dmm2x2());
    SocFloorplan moved = fp;
    const double dx = 100, dy = 50;
    moved.die.x_um += dx;
    moved.die.y_um += dy;
    std::function<void(std::vector<FloorplanBlock>&)> shift = [&](std::vector<FloorplanBlock>& bs) {
        for (FloorplanBlock& b : bs) {
            b.rect.x_um += dx;
            b.rect.y_um += dy;
            shift(b.children);
        }
    };
    shift(moved.blocks);
    for (NocLink& l : moved.noc)
        for (auto& [x, y] : l.route_um) { x += dx; y += dy; }
    const nlohmann::json a = nlohmann::json::parse(floorplan_to_structure(fp));
    const nlohmann::json b = nlohmann::json::parse(floorplan_to_structure(moved));
    auto close = [](const nlohmann::json& x, const nlohmann::json& y) {
        if (x.is_number() && y.is_number()) return std::abs(x.get<double>() - y.get<double>()) < 1e-9;
        return x == y;
    };
    std::function<bool(const nlohmann::json&, const nlohmann::json&)> same = [&](const auto& x, const auto& y) {
        if (x.is_structured() != y.is_structured() || x.size() != y.size()) return false;
        if (!x.is_structured()) return close(x, y);
        if (x.is_array()) {
            for (std::size_t i = 0; i < x.size(); ++i) if (!same(x[i], y[i])) return false;
            return true;
        }
        for (auto it = x.begin(); it != x.end(); ++it) if (!y.contains(it.key()) || !same(*it, y.at(it.key()))) return false;
        return true;
    };
    CHECK(same(a.at("blocks"), b.at("blocks")));
    CHECK(same(a.at("networks"), b.at("networks")));
}

TEST_CASE("a mesh edge port's SVG label sits on the side its label names", "[program][platform][mesh][floorplan]") {
    const SocFloorplan fp = generate_floorplan(dmm2x2());
    const std::string svg = floorplan_to_svg(fp);
    const FloorplanBlock* n0 = fp.find("dmm2x2/noc/port[0]");
    const FloorplanBlock* w0 = fp.find("dmm2x2/noc/port[7]");
    REQUIRE(n0);
    REQUIRE(w0);
    REQUIRE(w0->label == "W0");
    auto text_attr = [&](const std::string& label, const std::string& attr) {
        const auto end = svg.find(">" + label + "</text>");
        REQUIRE(end != std::string::npos);
        const auto start = svg.rfind("<text", end);
        const auto at = svg.find(attr + "=\"", start);
        REQUIRE(at < end);
        return svg.substr(at + attr.size() + 2, svg.find('"', at + attr.size() + 2) - (at + attr.size() + 2));
    };
    CHECK(std::stod(text_attr("N0", "y")) < n0->rect.y_um);          // above a north port
    CHECK(text_attr("W0", "text-anchor") == "end");                    // left of a west port
}

// ---- the hierarchical SoC spec (#635)

TEST_CASE("a dmm_mesh is written as its hierarchy, and its flat fields are derived views", "[program][platform][mesh]") {
    const DeploymentSpec s = dmm2x2();
    const DeviceSpecification& d = s.device(0);
    REQUIRE(d.cluster.has_value());
    CHECK(d.compute_tiles == 4);
    CHECK(d.l3.tiles == Dim{4});
    CHECK(d.l3.capacity_tiles == 64);                      // 16 per tile
    CHECK(d.macs_per_cycle == 4096.0);                     // 64 x 64 PEs
    CHECK(d.movers.block_movers == 32);                    // 8 per node
    CHECK(d.movers.streamers == 4);
    CHECK(d.dma.engines == 2);                             // 1 per controller
    // The writer writes the hierarchy and nothing flat.
    const nlohmann::json j = nlohmann::json::parse(to_json(s));
    const nlohmann::json& dev = j.contains("devices") ? j.at("devices").at(0) : j;
    for (const char* flat : {"compute_tiles", "macs_per_cycle", "l3", "l2", "l1", "movers", "dma", "combiner", "fabric", "periphery"}) {
        INFO(flat);
        CHECK_FALSE(dev.contains(flat));
    }
    CHECK(dev.at("cluster").at("l3_tile").at("capacity_tiles") == 16);
    CHECK(dev.at("cluster").at("compute_tile").at("streamers").at("count") == 1);
    CHECK(dev.at("memory").at("dma").at("engines_per_controller") == 1);
    CHECK(dev.at("memory").at("periphery").at("topology") == "ring");
}

TEST_CASE("a flat field on a dmm_mesh is refused: it is derived, never written", "[program][platform][mesh]") {
    std::string t = to_json(dmm2x2());
    const std::string key = "\"topology\": \"dmm_mesh\",";
    const auto at = t.find(key);
    REQUIRE(at != std::string::npos);
    for (const char* flat : {"\"compute_tiles\": 4,", "\"l3\": {\"tiles\": 4},", "\"movers\": {\"streamers\": 4},",
                             "\"dma\": {\"engines\": 2},", "\"periphery\": {}," }) {
        std::string bad = t;
        bad.insert(at + key.size(), flat);
        INFO(flat);
        CHECK_THROWS_WITH(from_json(bad), ContainsSubstring("on a dmm_mesh this is derived from the hierarchy"));
    }
}

TEST_CASE("the cluster's resources are checked, by field", "[program][platform][mesh]") {
    CHECK_THAT(problem([](auto& d) { d.cluster.reset(); }), ContainsSubstring("declares its cluster"));
    CHECK_THAT(problem([](auto& d) { d.cluster->l3_tile.noc_ports = 3; }),
               ContainsSubstring("cluster.l3_tile.noc_ports is 3; a node has one per NEWS link, 4"));
    CHECK_THAT(problem([](auto& d) { d.cluster->l3_tile.l2_ports = 2; }), ContainsSubstring("the feeds a, b, x and the drain make 4"));
    CHECK_THAT(problem([](auto& d) { d.cluster->compute_tile.l2.banks = 3; }),
               ContainsSubstring("cluster.compute_tile.l2.banks must be a power of two"));
    CHECK_THAT(problem([](auto& d) { d.cluster->compute_tile.fabric.dataflow = "xs"; }), ContainsSubstring("must be os, ws or is"));
    CHECK_THAT(problem([](auto& d) { d.systolic = DeviceSpecification::Systolic{}; }),
               ContainsSubstring("a dmm_mesh's compute tile is cluster.compute_tile"));
    // A cluster belongs to a dmm_mesh.
    DeploymentSpec t4 = read_spec_file(std::string(kDeploy) + "kpu_t4.json");
    t4.device(0).cluster = DeviceSpecification::Cluster{};
    CHECK_THAT(t4.validate(), ContainsSubstring("a cluster and the mesh's NoC cluster apply to the dmm_mesh topology"));
}

TEST_CASE("the structure record nests each node's tiles in its cluster", "[program][platform][mesh][floorplan]") {
    const nlohmann::json rec = nlohmann::json::parse(floorplan_to_structure(generate_floorplan(dmm2x2())));
    std::map<std::string, std::string> parent, kind;
    for (const auto& b : rec.at("blocks")) {
        kind[b.at("id")] = b.at("kind");
        if (b.contains("parent")) parent[b.at("id")] = b.at("parent");
    }
    CHECK(kind.at("dmm2x2/cluster[1]") == "cluster");
    CHECK(parent.at("dmm2x2/l3[1]") == "dmm2x2/cluster[1]");
    CHECK(parent.at("dmm2x2/cf[1]") == "dmm2x2/cluster[1]");
    CHECK(parent.at("dmm2x2/l3[1]/bm[7]") == "dmm2x2/l3[1]");            // the movers: the L3 tile's
    CHECK(parent.at("dmm2x2/cf[1]/l2[0]") == "dmm2x2/cf[1]");            // the L2: the compute tile's
}

// ---- regressions from the review of #636

TEST_CASE("derived totals that would not fit 32 bits are refused, not wrapped", "[program][platform][mesh]") {
    CHECK_THAT(problem([](auto& d) {
                   d.cluster->l3_tile.capacity_tiles = 2147483648u;   // x4 nodes wraps to 0: "unbounded"
               }),
               ContainsSubstring("is more L3 slots than the spec can count"));
    std::string t = to_json(dmm2x2());
    const std::string from = "\"engines_per_controller\": 1";
    t.replace(t.find(from), from.size(), "\"engines_per_controller\": 2147483649");   // x2 controllers wraps to 2
    CHECK_THROWS_WITH(from_json(t), ContainsSubstring("is more engines than the spec can count"));
}

TEST_CASE("a vector unit's contents, not only its presence, are the hierarchy's", "[program][platform][mesh]") {
    DeploymentSpec s = dmm2x2();
    DeviceSpecification& d = s.device(0);
    d.cluster->l3_tile.block_movers.vector = DeviceSpecification::Movers::VectorUnit{16, 1.0, {"add"}};
    derive_mesh_views(d);
    REQUIRE(s.validate().empty());
    d.cluster->l3_tile.block_movers.vector->lanes = 32;    // the hierarchy changed, its view did not
    CHECK_THAT(s.validate(), ContainsSubstring("the vector units (movers.vector.bm, combiner) disagrees with the cluster"));
}

TEST_CASE("the writer refuses a mesh whose flat fields disagree with its hierarchy", "[program][platform][mesh]") {
    DeploymentSpec s = dmm2x2();
    s.device(0).movers.bm_bytes_per_cycle = 256;           // the cluster still says 128
    CHECK_THROWS_WITH(to_json(s), ContainsSubstring("movers disagrees with the cluster"));
}

TEST_CASE("adjacent clusters share their rounded edges at a fractional pitch", "[program][platform][mesh][floorplan]") {
    const DeploymentSpec s = dmm2x2();
    const SocFloorplan fp = generate_floorplan(s, 0, FloorplanStyle{1000.0004, 40, 400});
    INFO(validate_floorplan(fp, s));
    CHECK(validate_floorplan(fp, s).empty());
}
