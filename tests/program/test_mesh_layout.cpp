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
    CHECK_THAT(problem([](auto& d) { d.compute_tiles = 3; d.l3.tiles = 3; }),
               ContainsSubstring("has 4 nodes, each an L3 tile with its own compute tile, but compute_tiles is 3"));
    CHECK_THAT(problem([](auto& d) { d.l3.tiles.reset(); }), ContainsSubstring("l3.tiles = compute_tiles (4)"));
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
               ContainsSubstring("noc applies to the checkerboard's folded torus, not the 'dmm_mesh' topology"));
    // A periphery is the dmm_mesh's alone.
    DeploymentSpec t4 = read_spec_file(std::string(kDeploy) + "kpu_t4.json");
    t4.device(0).periphery = DeviceSpecification::Periphery{};
    CHECK_THAT(t4.validate(), ContainsSubstring("periphery applies to the dmm_mesh topology, not 'checkerboard'"));
    // The checkerboard's evenness rule is the folded torus's, not the mesh's: a 1x3 mesh is a mesh.
    CHECK(problem([](auto& d) {
              d.array.rows = 1; d.array.cols = 3; d.compute_tiles = 3; d.l3.tiles = 3;
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
        s.device(0).l1.vectors = 16;                       // one column: more gaps than the style's height allows
        s.device(0).l3.banks = 64;
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
