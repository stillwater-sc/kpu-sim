// ============================================================================
// src/program/floorplan.cpp
// The floorplan: generator, validator, JSON and SVG (#286 step 1). See floorplan.hpp for the
// rule every rectangle follows.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#include <sw/kpu/program/platform/floorplan.hpp>

#include <sw/kpu/program/platform/digest.hpp>
#include <sw/kpu/version.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace sw::kpu::program::platform {

namespace {
using json = nlohmann::ordered_json;

// Coordinates are rounded to a nanometre, so the canonical JSON is stable and two
// generators that agree on geometry agree on bytes.
double nm(double v) { return std::round(v * 1000.0) / 1000.0; }
Rect R(double x, double y, double w, double h) { return Rect{nm(x), nm(y), nm(w), nm(h)}; }

struct KindName {
    BlockKind kind;
    const char* name;
};
const std::vector<KindName>& kind_names() {
    static const std::vector<KindName> k = {
        {BlockKind::Array, "array"},         {BlockKind::Cpu, "cpu"},
        {BlockKind::DramPhy, "dram_phy"},    {BlockKind::Io, "io"},
        {BlockKind::L3Tile, "l3_tile"},      {BlockKind::L3Bank, "l3_bank"},
        {BlockKind::BlockMover, "block_mover"}, {BlockKind::NocRouter, "noc_router"},
        {BlockKind::NocPort, "noc_port"},    {BlockKind::ComputeTile, "compute_tile"},
        {BlockKind::L2Bank, "l2_bank"},      {BlockKind::L1Vector, "l1_vector"},
        {BlockKind::RegisterFile, "register_file"},
        {BlockKind::MemoryController, "memory_controller"},
        {BlockKind::DmaEngine, "dma_engine"}, {BlockKind::CpuHart, "cpu_hart"},
        {BlockKind::CpuSram, "cpu_sram"},    {BlockKind::DescriptorRing, "descriptor_ring"},
        {BlockKind::CompletionRing, "completion_ring"},
    };
    return k;
}
} // namespace

const char* to_string(BlockKind k) {
    for (const KindName& n : kind_names())
        if (n.kind == k) return n.name;
    return "?";
}

std::optional<BlockKind> block_kind_from(const std::string& s) {
    for (const KindName& n : kind_names())
        if (s == n.name) return n.kind;
    return std::nullopt;
}

std::optional<ResourceKind> resource_kind_of(BlockKind k) {
    switch (k) {
        case BlockKind::Array:
        case BlockKind::Cpu:
        case BlockKind::DramPhy:
        case BlockKind::Io:               return std::nullopt;
        case BlockKind::L3Tile:           return ResourceKind::L3Tile;
        case BlockKind::L3Bank:           return ResourceKind::L3Bank;
        case BlockKind::BlockMover:       return ResourceKind::BlockMover;
        case BlockKind::NocRouter:        return ResourceKind::NocRouter;
        case BlockKind::NocPort:          return ResourceKind::NocPort;
        case BlockKind::ComputeTile:      return ResourceKind::ComputeTile;
        case BlockKind::L2Bank:           return ResourceKind::L2Bank;
        case BlockKind::L1Vector:         return ResourceKind::L1Vector;
        case BlockKind::RegisterFile:     return ResourceKind::RegisterFile;
        case BlockKind::MemoryController: return ResourceKind::MemoryController;
        case BlockKind::DmaEngine:        return ResourceKind::DmaEngine;
        case BlockKind::CpuHart:          return ResourceKind::CpuHart;
        case BlockKind::CpuSram:          return ResourceKind::CpuSram;
        case BlockKind::DescriptorRing:   return ResourceKind::DescriptorRing;
        case BlockKind::CompletionRing:   return ResourceKind::CompletionRing;
    }
    return std::nullopt;
}

const char* to_string(NocLink::Kind k) {
    switch (k) {
        case NocLink::Kind::Ring:   return "ring";
        case NocLink::Kind::Port:   return "port";
        case NocLink::Kind::Attach: return "attach";
        case NocLink::Kind::Mesh:      return "mesh";
        case NocLink::Kind::Credit:    return "credit";
        case NocLink::Kind::Edge:      return "edge";
        case NocLink::Kind::Periphery: return "periphery";
    }
    return "?";
}

namespace {
constexpr double kEps = 1e-6;
}

bool Rect::inside(const Rect& o) const {
    return x_um >= o.x_um - kEps && y_um >= o.y_um - kEps &&
           x_um + w_um <= o.x_um + o.w_um + kEps && y_um + h_um <= o.y_um + o.h_um + kEps;
}

bool Rect::overlaps(const Rect& o) const {
    return x_um < o.x_um + o.w_um - kEps && o.x_um < x_um + w_um - kEps &&
           y_um < o.y_um + o.h_um - kEps && o.y_um < y_um + h_um - kEps;
}

namespace {
void walk(const std::vector<FloorplanBlock>& bs, const std::function<void(const FloorplanBlock&)>& f) {
    for (const FloorplanBlock& b : bs) {
        f(b);
        walk(b.children, f);
    }
}
} // namespace

const FloorplanBlock* SocFloorplan::find(const std::string& name) const {
    const FloorplanBlock* hit = nullptr;
    walk(blocks, [&](const FloorplanBlock& b) {
        if (!hit && b.name == name) hit = &b;
    });
    return hit;
}

std::size_t SocFloorplan::block_count() const {
    std::size_t n = 0;
    walk(blocks, [&](const FloorplanBlock&) { ++n; });
    return n;
}

std::string SocFloorplan::digest() const { return digest_of(floorplan_to_json(*this)); }

// ============================================================================
// The generator
// ============================================================================
namespace {

FloorplanBlock hw(const std::string& device, BlockKind kind, std::vector<Dim> path,
                  const std::string& clock, Rect rect, std::string label = {}) {
    FloorplanBlock b;
    b.kind = kind;
    b.resource = ResourceName{device, *resource_kind_of(kind), std::move(path), 0};
    b.name = format(*b.resource);
    b.clock_domain = clock;
    b.rect = rect;
    b.label = std::move(label);
    return b;
}

FloorplanBlock group(const std::string& name, BlockKind kind, Rect rect, std::string label = {}) {
    FloorplanBlock b;
    b.name = name;
    b.kind = kind;
    b.rect = rect;
    b.label = std::move(label);
    return b;
}

// A grid of `n` equal cells inside `area`, row-major, `cols` wide.
std::vector<Rect> grid(const Rect& area, Dim n, Dim cols, double gap) {
    std::vector<Rect> out;
    if (n == 0) return out;
    cols = std::max<Dim>(1, std::min(cols, n));
    const Dim rows = (n + cols - 1) / cols;
    const double w = (area.w_um - gap * (cols - 1)) / cols;
    const double h = (area.h_um - gap * (rows - 1)) / rows;
    for (Dim i = 0; i < n; ++i)
        out.push_back(R(area.x_um + static_cast<double>(i % cols) * (w + gap),
                        area.y_um + static_cast<double>(i / cols) * (h + gap), w, h));   // row index: integer by design
    return out;
}

Dim isqrt_ceil(Dim n) {
    Dim c = 1;
    while (c * c < n) ++c;
    return c;
}

// The gap between a controller's DMA engines, along a run of `len`: the style's gap, narrowed when
// `per` engines would not fit with it, so every engine keeps a positive width (len / 2per at least)
// for any count the spec accepts.
double engine_gap(double len, Dim per, double g) { return std::min(g, len / (2.0 * (per + 1))); }

// An L3 tile's banks, in the tile at (x, y) of side s.
void add_l3_banks(FloorplanBlock& tile, const DeviceSpecification& d, Dim t, double x, double y, double s) {
    if (!d.l3.banks) return;
    const double m = 0.1 * s;
    const Rect area = R(x + m, y + m, s - 2 * m, s - 2 * m - 0.2 * s);
    const auto cells = grid(area, *d.l3.banks, isqrt_ceil(*d.l3.banks), 0.02 * s);
    for (Dim b = 0; b < *d.l3.banks; ++b)
        tile.children.push_back(hw(d.name, BlockKind::L3Bank, {t, b}, "l3", cells[b]));
}

// A compute tile at (x, y) of side s: its register file, L2 banks and L1 vectors.
FloorplanBlock compute_tile_block(const DeviceSpecification& d, Dim cf, double x, double y, double s) {
    const double m = 0.1 * s;
    FloorplanBlock tile = hw(d.name, BlockKind::ComputeTile, {cf}, "cf", R(x, y, s, s),
                             "cf[" + std::to_string(cf) + "]");
    // The upper part is the processor array: the tile itself, not a sub-block.
    tile.children.push_back(hw(d.name, BlockKind::RegisterFile, {cf}, "cf",
                               R(x + m, y + m, 0.25 * s, 0.12 * s), "regs"));
    if (d.l2.banks_per_tile) {
        const Rect area = R(x + m, y + 0.6 * s, 0.5 * s, 0.3 * s);
        const auto cells = grid(area, *d.l2.banks_per_tile, isqrt_ceil(*d.l2.banks_per_tile), 0.02 * s);
        for (Dim b = 0; b < *d.l2.banks_per_tile; ++b)
            tile.children.push_back(hw(d.name, BlockKind::L2Bank, {cf, b}, "cf", cells[b]));
    }
    if (d.l1.vectors) {
        const Rect area = R(x + 0.65 * s, y + 0.6 * s, 0.25 * s, 0.3 * s);
        const auto cells = grid(area, *d.l1.vectors, 1, 0.02 * s);
        for (Dim v = 0; v < *d.l1.vectors; ++v)
            tile.children.push_back(hw(d.name, BlockKind::L1Vector, {cf, v}, "cf", cells[v]));
    }
    return tile;
}

// The attached CPU (cores, SRAM, descriptor/completion rings) in the column at x = M, from y = top,
// and IO below it.
void add_cpu_and_io(SocFloorplan& fp, const DeviceSpecification& d, const FloorplanStyle& st, double top,
                    double cpuW, double min_io_h) {
    const double P = st.pitch_um, g = st.gap_um, M = st.margin_um;
    const bool cpu = d.cpu.harts.has_value();
    double cpu_bottom = top;
    if (cpu) {
        const Dim harts = *d.cpu.harts;
        const double cx = M, cw = cpuW, inner = cw - 2 * g;
        const double hh = 0.7 * P;
        const double hart_rows = static_cast<double>((harts + 1) / 2);   // two harts per row
        const double sram_y = top + g + hart_rows * (hh + g);
        const double ring_y = sram_y + 0.6 * P + g;
        const double ch = ring_y + 0.4 * P + g - top;
        FloorplanBlock c = group(d.name + "/cpu", BlockKind::Cpu, R(cx, top, cw, ch), "CPU");
        const auto cells = grid(R(cx + g, top + g, inner, hart_rows * (hh + g) - g), harts, 2, g);
        for (Dim h = 0; h < harts; ++h)
            c.children.push_back(hw(d.name, BlockKind::CpuHart, {h}, "cpu", cells[h],
                                    "hart" + std::to_string(h)));
        c.children.push_back(hw(d.name, BlockKind::CpuSram, {}, "cpu", R(cx + g, sram_y, inner, 0.6 * P), "SRAM"));
        const double rw = (inner - g) / 2;
        c.children.push_back(hw(d.name, BlockKind::DescriptorRing, {}, "cpu",
                                R(cx + g, ring_y, rw, 0.4 * P), "desc ring"));
        c.children.push_back(hw(d.name, BlockKind::CompletionRing, {}, "cpu",
                                R(cx + g + rw + g, ring_y, rw, 0.4 * P), "cpl ring"));
        cpu_bottom = top + ch + g;
        fp.blocks.push_back(std::move(c));
    }
    fp.blocks.push_back(group(d.name + "/io", BlockKind::Io,
                              R(M, cpu_bottom + (cpu ? g : 0), cpuW, std::max(P, min_io_h)), "IO"));
}

// A dmm_mesh (mesh_layout.hpp). Node (r, c) is its L3 tile with its compute tile to the east. The
// edge ports sit in a band around the array, and the periphery ring runs through that band. Each
// memory controller sits outside the band, on the side of the edge port its ring stop follows,
// with its PHY outside it at the die's edge; its DMA engines are its children.
SocFloorplan generate_mesh_floorplan(const DeviceSpecification& d, const MeshLayout& L, const FloorplanStyle& st) {
    const double P = st.pitch_um, g = st.gap_um, M = st.margin_um, s = P - g;
    const double phyH = 0.25 * P, mcH = 0.6 * P, ring = 0.6 * P;
    const double mcBand = phyH + g + mcH + g;
    const double cpuW = 2.4 * P;
    const double nw = 2 * P;                                      // a node: two tiles side by side
    const double ax = M + cpuW + M + mcBand + ring, ay = M + mcBand + ring;
    const double aw = L.cols() * nw, ah = L.rows() * P;

    SocFloorplan fp;
    fp.source = "generated:" + d.topology;
    fp.device = d.name;

    // ---- the nodes: L3 tile (its eight BlockMovers, banks) and its compute tile
    FloorplanBlock array = group(d.name + "/array", BlockKind::Array, R(ax, ay, aw, ah), "array");
    std::map<Dim, Rect> l3_rect;
    for (Dim t = 0; t < L.node_count(); ++t) {
        const GridPos p = L.pos(t);
        const double x = ax + p.c * nw + g / 2, y = ay + p.r * P + g / 2;
        FloorplanBlock tile = hw(d.name, BlockKind::L3Tile, {t}, "l3", R(x, y, s, s), "l3[" + std::to_string(t) + "]");
        // The four send movers on the edges whose links they own; the feeds and the drain on the
        // edge that faces the tile's compute tile.
        const double th = 0.06 * s, len = 0.4 * s, f = 0.1 * s;
        const Rect mover[kMeshMovers] = {
            R(x + 0.3 * s, y, len, th),                      // send N
            R(x + s - th, y + 0.12 * s, th, 0.25 * s),       // send E
            R(x + 0.3 * s, y + s - th, len, th),             // send S
            R(x, y + 0.3 * s, th, len),                      // send W
            R(x + s - th, y + 0.45 * s, th, f),              // feed a
            R(x + s - th, y + 0.57 * s, th, f),              // feed b
            R(x + s - th, y + 0.69 * s, th, f),              // feed x
            R(x + s - th, y + 0.81 * s, th, f)};             // drain
        for (Dim k = 0; k < kMeshMovers; ++k)
            tile.children.push_back(hw(d.name, BlockKind::BlockMover, {t, k}, "l3", mover[k],
                                       to_string(static_cast<MoverRole>(k))));
        add_l3_banks(tile, d, t, x, y, s);
        l3_rect[t] = tile.rect;
        array.children.push_back(std::move(tile));
        array.children.push_back(compute_tile_block(d, t, x + P, y, s));
    }
    fp.blocks.push_back(std::move(array));

    // ---- the edge ports, in the band, on the side of the node they belong to
    const double ps = 0.2 * P;
    std::map<Dim, Rect> port_rect;
    for (const EdgePort& e : L.edge_ports()) {
        const GridPos p = L.pos(e.node);
        const double ocx = ax + p.c * nw + P, ocy = ay + p.r * P + P / 2;   // the node's centre
        Rect rr;
        switch (e.side) {
            case Edge::N: rr = R(ocx - ps / 2, ay - ring / 2 - ps / 2, ps, ps); break;
            case Edge::S: rr = R(ocx - ps / 2, ay + ah + ring / 2 - ps / 2, ps, ps); break;
            case Edge::W: rr = R(ax - ring / 2 - ps / 2, ocy - ps / 2, ps, ps); break;
            case Edge::E: rr = R(ax + aw + ring / 2 - ps / 2, ocy - ps / 2, ps, ps); break;
        }
        port_rect[e.index] = rr;
        fp.blocks.push_back(hw(d.name, BlockKind::NocPort, {e.index}, "l3", rr, e.label()));
    }

    // ---- the memory controllers, each past the port its ring stop follows, and their PHYs
    const auto& ports = L.edge_ports();
    const Dim nports = static_cast<Dim>(ports.size());
    const Dim per = d.dma.engines / L.controller_count();
    const double len = 0.8 * P;                                   // along the side: under the port pitch
    auto along = [&](Dim k) { return ports[k].side == Edge::N || ports[k].side == Edge::S ? port_rect[k].cx() : port_rect[k].cy(); };
    std::map<Dim, std::pair<Edge, double>> mc_at;                 // mc -> its side, its position along it
    for (Dim m = 0; m < L.controller_count(); ++m) {
        const Dim k = L.mc_after_port(m), kn = (k + 1) % nports;
        const Edge side = ports[k].side;
        const bool horiz = side == Edge::N || side == Edge::S;
        const double dir = side == Edge::N || side == Edge::E ? 1.0 : -1.0;   // clockwise
        const double pos = ports[kn].side == side ? (along(k) + along(kn)) / 2 : along(k) + dir * (horiz ? nw : P) / 2;
        mc_at[m] = {side, pos};
        Rect mr, pr;
        switch (side) {
            case Edge::N: mr = R(pos - len / 2, ay - ring - g - mcH, len, mcH); pr = R(pos - len / 2, M, len, phyH); break;
            case Edge::S: mr = R(pos - len / 2, ay + ah + ring + g, len, mcH);
                          pr = R(pos - len / 2, ay + ah + ring + g + mcH + g, len, phyH); break;
            case Edge::W: mr = R(ax - ring - g - mcH, pos - len / 2, mcH, len);
                          pr = R(ax - ring - g - mcH - g - phyH, pos - len / 2, phyH, len); break;
            case Edge::E: mr = R(ax + aw + ring + g, pos - len / 2, mcH, len);
                          pr = R(ax + aw + ring + g + mcH + g, pos - len / 2, phyH, len); break;
        }
        fp.blocks.push_back(group(d.name + "/phy[" + std::to_string(m) + "]", BlockKind::DramPhy, pr, "PHY"));
        FloorplanBlock mc = hw(d.name, BlockKind::MemoryController, {m}, "dram", mr, "MC" + std::to_string(m));
        const double ge = engine_gap(len, per, g), dl = (len - ge * (per + 1)) / per, dd = 0.35 * mcH;
        for (Dim e = 0; e < per; ++e) {
            const double o = ge + e * (dl + ge);
            const Rect er = horiz ? R(mr.x_um + o, mr.y_um + (mcH - dd) / 2, dl, dd)
                                  : R(mr.x_um + (mcH - dd) / 2, mr.y_um + o, dd, dl);
            mc.children.push_back(hw(d.name, BlockKind::DmaEngine, {m, e}, "dram", er, "DMA" + std::to_string(e)));
        }
        fp.blocks.push_back(std::move(mc));
    }

    add_cpu_and_io(fp, d, st, ay, cpuW, ah * 0.25);
    double bottom = ay + ah + ring + mcBand;
    for (const FloorplanBlock& b : fp.blocks) bottom = std::max(bottom, b.rect.y_um + b.rect.h_um);
    fp.die = R(0, 0, ax + aw + ring + mcBand + M, bottom + M);

    // ---- the wires: one-hop data links and their credit links, edge ports, the periphery ring
    auto l3n = [&](Dim t) { return format(ResourceName{d.name, ResourceKind::L3Tile, {t}, 0}); };
    auto portn = [&](Dim k) { return format(ResourceName{d.name, ResourceKind::NocPort, {k}, 0}); };
    auto mcn = [&](Dim m) { return format(ResourceName{d.name, ResourceKind::MemoryController, {m}, 0}); };
    const double off = 0.08 * s;                                  // the credit wire, beside the data wire
    for (const MeshLink& k : L.links()) {
        if (k.dir != Edge::E && k.dir != Edge::S) continue;       // each pair once; it carries both directions
        const Rect& a = l3_rect.at(k.from);
        const Rect& b = l3_rect.at(k.to);
        fp.noc.push_back({NocLink::Kind::Mesh, l3n(k.from), l3n(k.to), {{a.cx(), a.cy()}, {b.cx(), b.cy()}}});
        const double dx = k.dir == Edge::S ? off : 0, dy = k.dir == Edge::E ? off : 0;
        fp.noc.push_back({NocLink::Kind::Credit, l3n(k.from), l3n(k.to),
                          {{a.cx(), a.cy()}, {a.cx() + dx, a.cy() + dy}, {b.cx() + dx, b.cy() + dy}, {b.cx(), b.cy()}}});
    }
    for (const EdgePort& e : ports) {
        const Rect& a = port_rect.at(e.index);
        const Rect& b = l3_rect.at(e.node);
        fp.noc.push_back({NocLink::Kind::Edge, portn(e.index), l3n(e.node), {{a.cx(), a.cy()}, {b.cx(), b.cy()}}});
    }
    // The ring's track runs through the middle of the band; a segment turns at the band's corners.
    const double tl = ax - ring / 2, tr = ax + aw + ring / 2, tt = ay - ring / 2, tb = ay + ah + ring / 2;
    auto track = [&](Edge side, double pos) -> std::pair<double, double> {
        switch (side) {
            case Edge::N: return {pos, tt};
            case Edge::E: return {tr, pos};
            case Edge::S: return {pos, tb};
            case Edge::W: return {tl, pos};
        }
        return {pos, tt};
    };
    auto corner_after = [&](Edge side) -> std::pair<double, double> {   // clockwise, at the side's end
        switch (side) {
            case Edge::N: return {tr, tt};
            case Edge::E: return {tr, tb};
            case Edge::S: return {tl, tb};
            case Edge::W: return {tl, tt};
        }
        return {tl, tt};
    };
    // A controller's stop is where its wire meets the track; the segment starts or ends at the
    // controller itself.
    struct Stop { std::string name; Edge side; double pos; std::optional<std::pair<double, double>> at; };
    std::vector<Stop> stops;
    std::map<Dim, Rect> mc_rect;
    for (const FloorplanBlock& b : fp.blocks)
        if (b.kind == BlockKind::MemoryController) mc_rect[b.resource->path[0]] = b.rect;
    for (const PeripheryStop& st2 : L.stops()) {
        if (st2.kind == PeripheryStop::Kind::EdgePort)
            stops.push_back({portn(st2.of), ports[st2.of].side, along(st2.of), std::nullopt});
        else
            stops.push_back({mcn(st2.of), mc_at.at(st2.of).first, mc_at.at(st2.of).second,
                             std::make_pair(mc_rect.at(st2.of).cx(), mc_rect.at(st2.of).cy())});
    }
    for (std::size_t i = 0; i < stops.size(); ++i) {
        const Stop& a = stops[i];
        const Stop& b = stops[(i + 1) % stops.size()];
        std::vector<std::pair<double, double>> route;
        if (a.at) route.push_back(*a.at);
        route.push_back(track(a.side, a.pos));
        if (a.side != b.side) route.push_back(corner_after(a.side));
        route.push_back(track(b.side, b.pos));
        if (b.at) route.push_back(*b.at);
        fp.noc.push_back({NocLink::Kind::Periphery, a.name, b.name, std::move(route)});
    }
    return fp;
}

} // namespace

SocFloorplan generate_floorplan(const DeploymentSpec& spec, Dim device, const FloorplanStyle& st) {
    if (device >= spec.device_count())
        throw FloorplanError("floorplan: device " + std::to_string(device) + " of " +
                             std::to_string(spec.device_count()));
    const std::string bad = spec.validate();
    if (!bad.empty()) throw FloorplanError("floorplan: " + bad);
    const DeviceSpecification& d = spec.device(device);
    std::string why;
    if (d.topology == "dmm_mesh") {
        const auto ML = MeshLayout::of(d, &why);
        if (!ML) throw FloorplanError("floorplan: device \"" + d.name + "\" has no mesh layout, because " + why);
        return generate_mesh_floorplan(d, *ML, st);
    }
    const auto L = ArrayLayout::of(d, &why);
    if (!L)
        throw FloorplanError("floorplan: device \"" + d.name + "\" has no array layout, because " +
                             why + ". Every tile has to be placed, so there is no floorplan to draw");

    const double P = st.pitch_um, g = st.gap_um, M = st.margin_um, s = P - g;
    const double phyH = 0.25 * P, mcH = 0.6 * P, portBand = 0.4 * P;
    const double cpuW = 2.4 * P;
    const Dim mcs = d.memory.controllers.value_or(0);
    const Dim top_mcs = (mcs + 1) / 2, bottom_mcs = mcs - top_mcs;
    const double topBand = (top_mcs ? phyH + g + mcH + g : 0.0) + portBand;
    const double bottomBand = (bottom_mcs ? g + mcH + g + phyH : 0.0) + portBand;

    const double ax = M + cpuW + M + portBand;
    const double ay = M + topBand;
    const double aw = L->cols() * P, ah = L->rows() * P;

    SocFloorplan fp;
    fp.source = "generated:" + d.topology;
    fp.device = d.name;

    // ---- the array: L3 tiles (banks, BlockMovers, hub) and compute tiles (regs, L2, L1)
    FloorplanBlock array = group(d.name + "/array", BlockKind::Array, R(ax, ay, aw, ah), "array");
    std::map<Dim, Rect> hub_rect;
    for (Dim r = 0; r < L->rows(); ++r)
        for (Dim c = 0; c < L->cols(); ++c) {
            const Cell& cell = L->cell(r, c);
            if (cell.kind == CellKind::Empty) continue;
            const double x = ax + c * P + g / 2, y = ay + r * P + g / 2;
            const double m = 0.1 * s;
            if (cell.kind == CellKind::L3) {
                const Dim t = cell.index;
                FloorplanBlock tile = hw(d.name, BlockKind::L3Tile, {t}, "l3", R(x, y, s, s),
                                         "l3[" + std::to_string(t) + "]");
                // A BlockMover on each edge that abuts a compute tile, as a strip on that edge.
                const double th = 0.06 * s, len = 0.4 * s;
                for (Edge e : {Edge::N, Edge::E, Edge::S, Edge::W}) {
                    if (!L->abutting_cf(t, e)) continue;
                    Rect rr;
                    switch (e) {
                        case Edge::N: rr = R(x + 0.3 * s, y, len, th); break;
                        case Edge::S: rr = R(x + 0.3 * s, y + s - th, len, th); break;
                        case Edge::W: rr = R(x, y + 0.3 * s, th, len); break;
                        case Edge::E: rr = R(x + s - th, y + 0.3 * s, th, len); break;
                    }
                    tile.children.push_back(hw(d.name, BlockKind::BlockMover,
                                               {t, static_cast<Dim>(e)}, "l3", rr, to_string(e)));
                }
                if (L->has_noc()) {
                    const double hs = 0.16 * s;
                    const Rect hr = R(x + s - m - hs, y + s - m - hs, hs, hs);
                    hub_rect[t] = hr;
                    tile.children.push_back(hw(d.name, BlockKind::NocRouter, {t}, "l3", hr, "hub"));
                }
                add_l3_banks(tile, d, t, x, y, s);
                array.children.push_back(std::move(tile));
            } else {
                array.children.push_back(compute_tile_block(d, cell.index, x, y, s));
            }
        }
    fp.blocks.push_back(std::move(array));

    // ---- the fold-end ports, just outside the array edge they face
    std::map<Dim, Rect> port_rect;
    if (L->has_noc()) {
        const double ps = 0.2 * P;
        for (const NocPort& p : L->ports()) {
            const Rect a = hub_rect.at(p.hub_a), b = hub_rect.at(p.hub_b);
            const double mx = (a.cx() + b.cx()) / 2, my = (a.cy() + b.cy()) / 2;
            Rect rr;
            switch (p.side) {
                case Edge::N: rr = R(mx - ps / 2, ay - portBand / 2 - ps / 2, ps, ps); break;
                case Edge::S: rr = R(mx - ps / 2, ay + ah + portBand / 2 - ps / 2, ps, ps); break;
                case Edge::W: rr = R(ax - portBand / 2 - ps / 2, my - ps / 2, ps, ps); break;
                case Edge::E: rr = R(ax + aw + portBand / 2 - ps / 2, my - ps / 2, ps, ps); break;
            }
            port_rect[p.index] = rr;
            fp.blocks.push_back(hw(d.name, BlockKind::NocPort, {p.index}, "l3", rr, p.label()));
        }
    }

    // ---- memory controllers with their DMA engines, and their PHYs at the die edge
    std::vector<std::pair<Dim, bool>> mc_side;          // (mc, on top)
    std::map<Dim, Rect> mc_rect;
    if (mcs) {
        const Dim per = d.dma.engines / mcs;
        auto row = [&](Dim first, Dim count, bool top) {
            if (!count) return;
            const double slot = aw / count, w = std::min(2.0 * P, slot - 2 * g);
            for (Dim i = 0; i < count; ++i) {
                const Dim m = first + i;
                const double x = ax + (i + 0.5) * slot - w / 2;
                const double phy_y = top ? M : ay + ah + portBand + g + mcH + g;
                const double mc_y = top ? M + phyH + g : ay + ah + portBand + g;
                fp.blocks.push_back(group(d.name + "/phy[" + std::to_string(m) + "]",
                                          BlockKind::DramPhy, R(x, phy_y, w, phyH), "PHY"));
                FloorplanBlock mc = hw(d.name, BlockKind::MemoryController, {m}, "dram",
                                       R(x, mc_y, w, mcH), "MC" + std::to_string(m));
                const double ge = engine_gap(w, per, g);
                const double dh = 0.35 * mcH, dw = (w - ge * (per + 1)) / per;
                for (Dim e = 0; e < per; ++e)
                    mc.children.push_back(hw(d.name, BlockKind::DmaEngine, {m, e}, "dram",
                                             R(x + ge + e * (dw + ge), mc_y + mcH - dh - g, dw, dh),
                                             "DMA" + std::to_string(e)));
                mc_rect[m] = mc.rect;
                mc_side.emplace_back(m, top);
                fp.blocks.push_back(std::move(mc));
            }
        };
        row(0, top_mcs, true);
        row(top_mcs, bottom_mcs, false);
    }

    // ---- the attached CPU (cores, SRAM, descriptor/completion rings), and IO below it
    add_cpu_and_io(fp, d, st, ay, cpuW, ah * 0.25);

    // THE DIE, sized from what was placed rather than from the array alone: on a short array the
    // CPU column and IO below it reach lower than the array's bottom band, and a die sized to the
    // array would leave them outside it -- a generated floorplan that fails its own validation.
    double bottom = ay + ah + bottomBand;
    for (const FloorplanBlock& b : fp.blocks) bottom = std::max(bottom, b.rect.y_um + b.rect.h_um);
    fp.die = R(0, 0, ax + aw + portBand + M, bottom + M);

    // ---- the NoC's wires: ring links, port links, and DMA attachments
    if (L->has_noc()) {
        auto hub_name = [&](Dim t) { return format(ResourceName{d.name, ResourceKind::NocRouter, {t}, 0}); };
        auto port_name = [&](Dim k) { return format(ResourceName{d.name, ResourceKind::NocPort, {k}, 0}); };
        for (const auto& [a, b] : L->links())
            fp.noc.push_back({NocLink::Kind::Ring, hub_name(a), hub_name(b),
                              {{hub_rect[a].cx(), hub_rect[a].cy()}, {hub_rect[b].cx(), hub_rect[b].cy()}}});
        for (const NocPort& p : L->ports())
            for (Dim h : {p.hub_a, p.hub_b})
                fp.noc.push_back({NocLink::Kind::Port, port_name(p.index), hub_name(h),
                                  {{port_rect[p.index].cx(), port_rect[p.index].cy()},
                                   {hub_rect[h].cx(), hub_rect[h].cy()}}});
        // FIRST-PASS ATTACHMENT (§5.2.1: "DMA channels connect there"). Which DMA feeds which
        // port is a connectivity question the schedules will settle, so the default is the
        // simplest defensible one: each N port goes to the nearest controller on the top edge,
        // each S port to the nearest on the bottom, and every engine of a controller attaches to
        // one of that controller's ports, in turn. W and E ports start unattached, visibly.
        //
        // An attachment is QUEUE MEMBERSHIP, not bandwidth: the port controller holds one input
        // and one output queue per attached engine and moves ONE block per bus at a time
        // (docs/plans/noc-port-arbitration.md). It used to say that more engines per port let
        // more traffic through it; that read the engine count -- outstanding DRAM requests --
        // as concurrency through the port, which a port does not have.
        if (mcs) {
            const Dim per = d.dma.engines / mcs;
            std::map<Dim, std::vector<Dim>> ports_of;            // mc -> its ports, in index order
            auto nearest_mc = [&](const Rect& pr, bool top) {
                std::optional<Dim> best;
                double best_d = 0;
                for (const auto& [m, on_top] : mc_side) {
                    if (on_top != top) continue;
                    const double dist = std::abs(mc_rect[m].cx() - pr.cx());
                    if (!best || dist < best_d) { best = m; best_d = dist; }
                }
                return best;
            };
            for (const NocPort& p : L->ports()) {
                if (p.side != Edge::N && p.side != Edge::S) continue;
                if (const auto m = nearest_mc(port_rect[p.index], p.side == Edge::N))
                    ports_of[*m].push_back(p.index);
            }
            // A controller that won no port (more controllers than ports on its edge) shares the
            // port nearest it rather than being left with engines that reach nothing.
            for (const auto& [m, on_top] : mc_side) {
                if (!ports_of[m].empty()) continue;
                std::optional<Dim> best;
                double best_d = 0;
                for (const NocPort& p : L->ports()) {
                    if (p.side != (on_top ? Edge::N : Edge::S)) continue;
                    const double dist = std::abs(port_rect[p.index].cx() - mc_rect[m].cx());
                    if (!best || dist < best_d) { best = p.index; best_d = dist; }
                }
                if (best) ports_of[m].push_back(*best);
            }
            for (const auto& [m, on_top] : mc_side) {
                const auto& ports = ports_of[m];
                if (ports.empty()) continue;
                for (Dim e = 0; e < per; ++e) {
                    const Dim k = ports[e % ports.size()];
                    const ResourceName dma{d.name, ResourceKind::DmaEngine, {m, e}, 0};
                    const FloorplanBlock* db = nullptr;
                    for (const FloorplanBlock& b : fp.blocks)
                        for (const FloorplanBlock& ch : b.children)
                            if (ch.name == format(dma)) db = &ch;
                    const Rect pr = port_rect[k];
                    fp.noc.push_back({NocLink::Kind::Attach, format(dma), port_name(k),
                                      {{db->rect.cx(), db->rect.cy()}, {pr.cx(), pr.cy()}}});
                }
            }
        }
    }
    return fp;
}

std::vector<DmaPortAttachment> dma_port_attachment(const DeploymentSpec& spec, Dim device) {
    const SocFloorplan fp = generate_floorplan(spec, device);
    std::vector<DmaPortAttachment> out;
    for (const NocLink& l : fp.noc) {
        if (l.kind != NocLink::Kind::Attach) continue;
        const ResourceName dma = parse_resource_name(l.a);
        const ResourceName port = parse_resource_name(l.b);
        if (dma.kind != ResourceKind::DmaEngine || port.kind != ResourceKind::NocPort)
            throw FloorplanError("dma_port_attachment: attach link '" + l.a + "' -> '" + l.b +
                                 "' does not join a DMA engine to a NoC port");
        out.push_back({dma.path.at(0), dma.path.at(1), port.path.at(0)});
    }
    std::sort(out.begin(), out.end(), [](const DmaPortAttachment& x, const DmaPortAttachment& y) {
        return x.mc != y.mc ? x.mc < y.mc : x.engine < y.engine;
    });
    return out;
}

// ============================================================================
// Validation
// ============================================================================
std::string validate_floorplan(const SocFloorplan& fp, const DeploymentSpec& spec) {
    const std::size_t di = spec.index_of(fp.device);
    if (di == spec.devices.size())
        return "the floorplan lays out device \"" + fp.device + "\", which this deployment does not declare";
    const ResourceMap map(spec);

    // Every block: a hardware block names a resource of this device that exists; a group
    // names none; no name twice.
    std::set<std::string> names;
    std::map<std::string, int> covered;
    std::string err;
    std::function<void(const std::vector<FloorplanBlock>&, const Rect*, const std::string&)> check;
    check = [&](const std::vector<FloorplanBlock>& bs, const Rect* parent, const std::string& pname) {
        for (std::size_t i = 0; i < bs.size() && err.empty(); ++i) {
            const FloorplanBlock& b = bs[i];
            const std::string w = "block \"" + b.name + "\"";
            if (!names.insert(b.name).second) { err = w + " appears twice"; return; }
            const auto rk = resource_kind_of(b.kind);
            if (rk) {
                if (!b.resource) { err = w + " is a " + to_string(b.kind) + " and must name a resource"; return; }
                if (b.resource->kind != *rk) {
                    err = w + " is a " + std::string(to_string(b.kind)) + " but names a " +
                          to_string(b.resource->kind);
                    return;
                }
                if (b.resource->device != fp.device) { err = w + " names a resource of another device"; return; }
                if (b.name != format(*b.resource)) { err = w + " is named differently from its resource"; return; }
                if (!map.exists(*b.resource)) {
                    err = w + " does not resolve: " + map.why_not(*b.resource);
                    return;
                }
                ++covered[b.name];
            } else if (b.resource) {
                err = w + " is a group (" + to_string(b.kind) + ") and cannot name a resource";
                return;
            }
            if (b.rect.w_um <= 0 || b.rect.h_um <= 0) { err = w + " has no area"; return; }
            const Rect& outer = parent ? *parent : fp.die;
            if (!b.rect.inside(outer)) {
                err = w + " lies outside " + (parent ? "its parent \"" + pname + "\"" : std::string("the die"));
                return;
            }
            for (std::size_t j = 0; j < i; ++j)
                if (b.rect.overlaps(bs[j].rect)) {
                    err = w + " overlaps its sibling \"" + bs[j].name + "\"";
                    return;
                }
            check(b.children, &b.rect, b.name);
        }
    };
    check(fp.blocks, nullptr, "");
    if (!err.empty()) return err;

    // Coverage: every resource of this device has exactly one block (DRAM is off-die).
    for (const ResourceName& n : map.enumerate()) {
        if (n.device != fp.device || n.kind == ResourceKind::Dram) continue;
        if (!covered.count(format(n)))
            return "resource \"" + format(n) + "\" has no block; every resource the deployment "
                   "declares must be placed";
    }

    // The NoC's wires join blocks that exist, of the kinds each wire joins.
    auto kind_of = [&](const std::string& name) -> std::optional<BlockKind> {
        const FloorplanBlock* b = fp.find(name);
        return b ? std::optional<BlockKind>(b->kind) : std::nullopt;
    };
    for (const NocLink& l : fp.noc) {
        const std::string w = std::string(to_string(l.kind)) + " link " + l.a + " -- " + l.b;
        const auto ka = kind_of(l.a), kb = kind_of(l.b);
        if (!ka || !kb) return w + " joins a block that does not exist";
        const auto stop = [](BlockKind k) { return k == BlockKind::NocPort || k == BlockKind::MemoryController; };
        const bool ok = (l.kind == NocLink::Kind::Ring && *ka == BlockKind::NocRouter && *kb == BlockKind::NocRouter) ||
                        (l.kind == NocLink::Kind::Port && *ka == BlockKind::NocPort && *kb == BlockKind::NocRouter) ||
                        (l.kind == NocLink::Kind::Attach && *ka == BlockKind::DmaEngine && *kb == BlockKind::NocPort) ||
                        ((l.kind == NocLink::Kind::Mesh || l.kind == NocLink::Kind::Credit) &&
                         *ka == BlockKind::L3Tile && *kb == BlockKind::L3Tile) ||
                        (l.kind == NocLink::Kind::Edge && *ka == BlockKind::NocPort && *kb == BlockKind::L3Tile) ||
                        (l.kind == NocLink::Kind::Periphery && stop(*ka) && stop(*kb));
        if (!ok) return w + " joins the wrong kinds of block";
        if (l.route_um.size() < 2) return w + " has no route";
    }

    // A dmm_mesh's wires are exactly its layout's (mesh_layout.hpp): every neighbour pair once as a
    // mesh wire and once as a credit wire, every edge port to its own node, and the periphery ring's
    // consecutive stops, in ring order, closing. Any other device has none of them.
    const DeviceSpecification& dev = spec.device(static_cast<Dim>(di));
    const auto ML = MeshLayout::of(dev);
    const auto mesh_kind = [](NocLink::Kind k) {
        return k == NocLink::Kind::Mesh || k == NocLink::Kind::Credit || k == NocLink::Kind::Edge ||
               k == NocLink::Kind::Periphery;
    };
    if (!ML) {
        for (const NocLink& l : fp.noc)
            if (mesh_kind(l.kind))
                return std::string(to_string(l.kind)) + " link " + l.a + " -- " + l.b +
                       " is a dmm_mesh wire, and device \"" + dev.name + "\" is not a dmm_mesh";
        return {};
    }
    using Pair = std::pair<std::string, std::string>;
    const auto l3n = [&](Dim t) { return format(ResourceName{dev.name, ResourceKind::L3Tile, {t}, 0}); };
    const auto portn = [&](Dim k) { return format(ResourceName{dev.name, ResourceKind::NocPort, {k}, 0}); };
    const auto mcn = [&](Dim m) { return format(ResourceName{dev.name, ResourceKind::MemoryController, {m}, 0}); };
    const auto unordered = [](std::string a, std::string b) { return a < b ? Pair{a, b} : Pair{b, a}; };
    std::map<NocLink::Kind, std::map<Pair, int>> want;
    for (const MeshLink& k : ML->links())
        if (k.dir == Edge::E || k.dir == Edge::S) {
            want[NocLink::Kind::Mesh][unordered(l3n(k.from), l3n(k.to))] = 1;
            want[NocLink::Kind::Credit][unordered(l3n(k.from), l3n(k.to))] = 1;
        }
    for (const EdgePort& e : ML->edge_ports()) want[NocLink::Kind::Edge][{portn(e.index), l3n(e.node)}] = 1;
    const auto& stops = ML->stops();
    const auto stop_name = [&](const PeripheryStop& st) {
        return st.kind == PeripheryStop::Kind::EdgePort ? portn(st.of) : mcn(st.of);
    };
    for (std::size_t i = 0; i < stops.size(); ++i)
        want[NocLink::Kind::Periphery][{stop_name(stops[i]), stop_name(stops[(i + 1) % stops.size()])}] = 1;
    std::map<NocLink::Kind, std::map<Pair, int>> have;
    for (const NocLink& l : fp.noc) {
        const std::string w = std::string(to_string(l.kind)) + " link " + l.a + " -- " + l.b;
        if (!mesh_kind(l.kind))
            return w + ": a dmm_mesh has no torus; its DMA engines reach the edge ports over the periphery";
        const bool sym = l.kind == NocLink::Kind::Mesh || l.kind == NocLink::Kind::Credit;
        const Pair key = sym ? unordered(l.a, l.b) : Pair{l.a, l.b};
        if (!want[l.kind].count(key)) {
            if (sym) return w + " does not join one-hop neighbours of the mesh";
            if (l.kind == NocLink::Kind::Edge) return w + " does not join an edge port to its own node";
            return w + " does not join consecutive stops of the periphery ring, in ring order";
        }
        if (++have[l.kind][key] > 1) return w + " appears twice";
    }
    for (const auto& [kind, pairs] : want)
        for (const auto& [key, n] : pairs)
            if (!have[kind].count(key))
                return std::string("the ") + to_string(kind) + " link " + key.first + " -- " + key.second +
                       " is missing; a dmm_mesh's floorplan draws every wire its layout has";
    return {};
}

// ============================================================================
// JSON
// ============================================================================
namespace {

json rect_json(const Rect& r) { return json::array({r.x_um, r.y_um, r.w_um, r.h_um}); }

Rect rect_from(const json& j, const std::string& where) {
    if (!j.is_array() || j.size() != 4)
        throw FloorplanError("floorplan: " + where + ".rect must be [x, y, w, h]");
    for (const json& v : j)
        if (!v.is_number()) throw FloorplanError("floorplan: " + where + ".rect must hold numbers");
    return Rect{j[0].get<double>(), j[1].get<double>(), j[2].get<double>(), j[3].get<double>()};
}

json block_json(const FloorplanBlock& b) {
    json o = json::object();
    o["name"] = b.name;
    o["kind"] = to_string(b.kind);
    if (b.resource) o["resource"] = format(*b.resource);
    if (!b.clock_domain.empty()) o["clock"] = b.clock_domain;
    if (!b.label.empty()) o["label"] = b.label;
    o["rect"] = rect_json(b.rect);
    if (!b.children.empty()) {
        json ch = json::array();
        for (const FloorplanBlock& c : b.children) ch.push_back(block_json(c));
        o["children"] = ch;
    }
    return o;
}

std::string str_at(const json& o, const char* key, const std::string& where, bool required = true) {
    if (!o.contains(key)) {
        if (required) throw FloorplanError("floorplan: " + where + " has no \"" + key + "\"");
        return {};
    }
    if (!o.at(key).is_string()) throw FloorplanError("floorplan: " + where + "." + key + " must be a string");
    return o.at(key).get<std::string>();
}

FloorplanBlock block_from(const json& o, const std::string& where) {
    if (!o.is_object()) throw FloorplanError("floorplan: " + where + " must be an object");
    FloorplanBlock b;
    b.name = str_at(o, "name", where);
    const std::string w = where + " (\"" + b.name + "\")";
    const std::string k = str_at(o, "kind", w);
    const auto kind = block_kind_from(k);
    if (!kind) throw FloorplanError("floorplan: " + w + " has unknown kind \"" + k + "\"");
    b.kind = *kind;
    if (o.contains("resource")) {
        try {
            b.resource = parse_resource_name(str_at(o, "resource", w));
        } catch (const NameError& e) {
            throw FloorplanError("floorplan: " + w + ": " + e.what());
        }
    }
    b.clock_domain = str_at(o, "clock", w, false);
    b.label = str_at(o, "label", w, false);
    if (!o.contains("rect")) throw FloorplanError("floorplan: " + w + " has no \"rect\"");
    b.rect = rect_from(o.at("rect"), w);
    if (o.contains("children")) {
        const json& ch = o.at("children");
        if (!ch.is_array()) throw FloorplanError("floorplan: " + w + ".children must be an array");
        for (std::size_t i = 0; i < ch.size(); ++i)
            b.children.push_back(block_from(ch[i], w + ".children[" + std::to_string(i) + "]"));
    }
    return b;
}

} // namespace

std::string floorplan_to_json(const SocFloorplan& fp) {
    json root = json::object();
    root["format"] = "kpu-floorplan";
    root["version"] = 1;
    root["source"] = fp.source;
    root["device"] = fp.device;
    root["units"] = "um";
    root["die"] = rect_json(fp.die);
    json blocks = json::array();
    for (const FloorplanBlock& b : fp.blocks) blocks.push_back(block_json(b));
    root["blocks"] = blocks;
    json noc = json::array();
    for (const NocLink& l : fp.noc) {
        json o = json::object();
        o["kind"] = to_string(l.kind);
        o["a"] = l.a;
        o["b"] = l.b;
        json route = json::array();
        for (const auto& [x, y] : l.route_um) route.push_back(json::array({x, y}));
        o["route"] = route;
        noc.push_back(o);
    }
    root["noc"] = noc;
    return root.dump(1) + "\n";
}

std::string floorplan_to_structure(const SocFloorplan& fp) {
    constexpr double mm = 1e-3;                               // um -> mm
    // The floorplan is drawn from the top-left (SVG); the record's origin is the die's lower-left
    // (shared-structure-record.md §2.2), so every y flips about the die's top edge.
    const double top = fp.die.y_um + fp.die.h_um;
    const auto rect_mm = [&](const Rect& r) {
        return std::array<double, 4>{r.x_um * mm, (top - r.y_um - r.h_um) * mm, r.w_um * mm, r.h_um * mm};
    };
    const auto rect_out = [](const std::array<double, 4>& r) { return json::array({r[0], r[1], r[2], r[3]}); };

    json blocks = json::array();
    std::map<std::string, std::pair<std::string, std::array<double, 4>>> placed;   // id -> kind, rect
    const auto die = rect_mm(fp.die);
    const std::string root = fp.device + "/die";
    blocks.push_back(json{{"id", root}, {"kind", "die"}, {"rect", rect_out(die)}, {"area", die[2] * die[3]}});
    std::function<void(const FloorplanBlock&, const std::string&)> walk =
        [&](const FloorplanBlock& b, const std::string& parent) {
            // The die root's id is reserved here, and only here: a version-1 floorplan may name a
            // group `<device>/die`, and it still imports and draws (#453's review). Its record
            // would duplicate the id and make the group its own parent, so the record is refused.
            if (b.name == root)
                throw FloorplanError("structure: block \"" + b.name + "\" uses the id the die reserves in the "
                                     "structure record; rename the block to write one");
            const std::string kind = to_string(b.kind);
            json o = json::object();
            o["id"] = b.name;
            o["kind"] = kind;
            o["parent"] = parent;
            if (!b.label.empty()) o["label"] = b.label;
            if (b.resource) o["resource"] = format(*b.resource);
            if (!b.clock_domain.empty()) o["clock_domain"] = b.clock_domain;
            const auto r = rect_mm(b.rect);
            o["rect"] = rect_out(r);
            o["area"] = r[2] * r[3];
            static const std::map<std::string, std::string> role = {
                {"block_mover", "block_mover"}, {"dma_engine", "dma"}, {"compute_tile", "fabric"}};
            static const std::map<std::string, std::string> layer = {
                {"l3_tile", "l3"}, {"l3_bank", "l3"}, {"l2_bank", "l2"}, {"l1_vector", "l1"}, {"compute_tile", "fabric"}};
            if (auto it = role.find(kind); it != role.end()) o["role"] = it->second;
            if (auto it = layer.find(kind); it != layer.end()) o["layer"] = it->second;
            placed[b.name] = {kind, r};
            blocks.push_back(o);
            for (const FloorplanBlock& c : b.children) walk(c, b.name);
        };
    for (const FloorplanBlock& b : fp.blocks) walk(b, root);

    // The networks. A checkerboard's NoC is one network, `noc`. A dmm_mesh has three: `mesh` (the
    // one-hop data links and the edge ports), `credit` (the credit network) and `periphery` (the
    // ring). Each network's nodes are the blocks its links join, in first-mention order.
    struct Net { json nodes = json::array(), links = json::array(); std::set<std::string> seen; };
    std::map<std::string, Net> nets;
    const auto net_of = [](NocLink::Kind k) -> std::string {
        switch (k) {
            case NocLink::Kind::Mesh:
            case NocLink::Kind::Edge:      return "mesh";
            case NocLink::Kind::Credit:    return "credit";
            case NocLink::Kind::Periphery: return "periphery";
            default:                       return "noc";
        }
    };
    for (std::size_t i = 0; i < fp.noc.size(); ++i) {
        const NocLink& l = fp.noc[i];
        Net& net = nets[net_of(l.kind)];
        json& nodes = net.nodes;
        json& links = net.links;
        for (const std::string& end : {l.a, l.b}) {
            if (!net.seen.insert(end).second) continue;
            const auto it = placed.find(end);
            const std::string k = it == placed.end() ? "" : it->second.first;
            const std::string nk = k == "noc_router" ? "router" : k == "noc_port" ? "port"
                                 : k == "l3_tile" ? "node" : k == "memory_controller" ? "stop" : "attach";
            json n = json{{"id", end}, {"kind", nk}, {"block", end}};
            if (it != placed.end()) {
                const auto& r = it->second.second;
                n["at"] = json::array({r[0] + r[2] / 2, r[1] + r[3] / 2});
            }
            nodes.push_back(n);
        }
        json link = json{{"id", std::string(to_string(l.kind)) + "_" + std::to_string(i)}, {"from", l.a}, {"to", l.b}};
        json pts = json::array();
        double length = 0;
        if (!l.route_um.empty()) {
            for (std::size_t p = 0; p < l.route_um.size(); ++p) {
                const double x = l.route_um[p].first * mm, y = (top - l.route_um[p].second) * mm;
                pts.push_back(json::array({x, y}));
                if (p) length += std::abs(x - pts[p - 1][0].get<double>()) + std::abs(y - pts[p - 1][1].get<double>());
            }
            link["points"] = pts;
            link["length"] = length;
        }
        links.push_back(link);
        // A mesh, credit or edge wire carries both directions (mesh_layout.hpp); the record states each
        // as its own link, so a reader recovers every direction from from/to. The periphery ring and the
        // torus's wires stay as drawn.
        if (l.kind == NocLink::Kind::Mesh || l.kind == NocLink::Kind::Credit || l.kind == NocLink::Kind::Edge) {
            json back = json{{"id", std::string(to_string(l.kind)) + "_" + std::to_string(i) + "r"}, {"from", l.b}, {"to", l.a}};
            if (!pts.empty()) {
                json rev = json::array();
                for (std::size_t p = pts.size(); p-- > 0;) rev.push_back(pts[p]);
                back["points"] = rev;
                back["length"] = length;
            }
            links.push_back(back);
        }
    }
    json networks = json::array();
    struct NetInfo { const char* id; const char* kind; const char* routing; const char* description; };
    static const NetInfo kNets[] = {
        {"noc", "noc", "ring", "the NoC's ring, port and attach links (kpu-floorplan)"},
        {"mesh", "noc", "one_hop", "the dmm_mesh's one-hop data links, each pair both ways, and its edge ports (kpu-floorplan)"},
        {"credit", "credit", "one_hop", "the credit network: each data link's credits, the other way (kpu-floorplan)"},
        {"periphery", "ring", "ring", "the periphery ring, on the memory controllers' clock: edge ports and controllers, clockwise (kpu-floorplan)"}};
    for (const NetInfo& n : kNets) {
        const auto it = nets.find(n.id);
        if (it == nets.end() || it->second.links.empty()) continue;
        networks.push_back(json{{"id", n.id}, {"kind", n.kind}, {"routing", n.routing}, {"description", n.description},
                                {"nodes", it->second.nodes}, {"links", it->second.links}});
    }

    json root_doc = json::object();
    root_doc["format"] = "kpu-structure";
    root_doc["version"] = 1;
    root_doc["provenance"] = json{
        {"producer", "kpu-floorplan kpu-sim " + std::string(sw::kpu::VERSION_STRING)},
        {"source", "model"},
        {"configuration", fp.device},
        {"details", json{{"layout", fp.source}, {"floorplan_digest", fp.digest()}}}};
    root_doc["units"] = json{{"length", "mm"}, {"area", "mm2"}};
    root_doc["blocks"] = blocks;
    root_doc["networks"] = networks;
    // The model's pitch is illustrative, not silicon: every rect is an estimate.
    root_doc["floorplan"] = json{{"die", json{{"width", die[2]}, {"height", die[3]}, {"area", die[2] * die[3]}}},
                                 {"estimated", true}};
    return root_doc.dump(1) + "\n";
}

SocFloorplan floorplan_from_json(const std::string& text, const DeploymentSpec& spec) {
    json root;
    try {
        root = json::parse(text);
    } catch (const nlohmann::json::parse_error& e) {
        throw FloorplanError(std::string("floorplan: not valid JSON: ") + e.what());
    }
    if (!root.is_object()) throw FloorplanError("floorplan: the top level must be an object");
    if (str_at(root, "format", "the file") != "kpu-floorplan")
        throw FloorplanError("floorplan: not a kpu-floorplan file");
    if (!root.contains("version") || !root.at("version").is_number_integer())
        throw FloorplanError("floorplan: \"version\" must be an integer");
    if (root.at("version").get<int>() != 1)
        throw FloorplanError("floorplan: version " + std::to_string(root.at("version").get<int>()) +
                             " is not one this build reads (1)");
    if (root.contains("units") && str_at(root, "units", "the file") != "um")
        throw FloorplanError("floorplan: units must be \"um\"");

    SocFloorplan fp;
    fp.source = str_at(root, "source", "the file", false);
    fp.device = str_at(root, "device", "the file");
    if (!root.contains("die")) throw FloorplanError("floorplan: no \"die\"");
    fp.die = rect_from(root.at("die"), "die");
    if (!root.contains("blocks") || !root.at("blocks").is_array())
        throw FloorplanError("floorplan: \"blocks\" must be an array");
    const json& bs = root.at("blocks");
    for (std::size_t i = 0; i < bs.size(); ++i)
        fp.blocks.push_back(block_from(bs[i], "blocks[" + std::to_string(i) + "]"));
    if (root.contains("noc")) {
        const json& ns = root.at("noc");
        if (!ns.is_array()) throw FloorplanError("floorplan: \"noc\" must be an array");
        for (std::size_t i = 0; i < ns.size(); ++i) {
            const std::string w = "noc[" + std::to_string(i) + "]";
            const json& o = ns[i];
            if (!o.is_object()) throw FloorplanError("floorplan: " + w + " must be an object");
            NocLink l;
            const std::string k = str_at(o, "kind", w);
            if (k == "ring") l.kind = NocLink::Kind::Ring;
            else if (k == "port") l.kind = NocLink::Kind::Port;
            else if (k == "attach") l.kind = NocLink::Kind::Attach;
            else if (k == "mesh") l.kind = NocLink::Kind::Mesh;
            else if (k == "credit") l.kind = NocLink::Kind::Credit;
            else if (k == "edge") l.kind = NocLink::Kind::Edge;
            else if (k == "periphery") l.kind = NocLink::Kind::Periphery;
            else throw FloorplanError("floorplan: " + w + " has unknown kind \"" + k + "\"");
            l.a = str_at(o, "a", w);
            l.b = str_at(o, "b", w);
            if (o.contains("route")) {
                const json& r = o.at("route");
                if (!r.is_array()) throw FloorplanError("floorplan: " + w + ".route must be an array");
                for (const json& p : r) {
                    if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number())
                        throw FloorplanError("floorplan: " + w + ".route points must be [x, y]");
                    l.route_um.emplace_back(p[0].get<double>(), p[1].get<double>());
                }
            }
            fp.noc.push_back(std::move(l));
        }
    }
    const std::string bad = validate_floorplan(fp, spec);
    if (!bad.empty()) throw FloorplanError("floorplan: " + bad);
    return fp;
}

// ============================================================================
// SVG
// ============================================================================
namespace {
const char* fill_of(BlockKind k) {
    switch (k) {
        case BlockKind::Array:            return "none";
        case BlockKind::Cpu:              return "#dcdfe9";
        case BlockKind::DramPhy:          return "#e3e5e8";
        case BlockKind::Io:               return "#eceef0";
        case BlockKind::L3Tile:           return "#dfe3e8";
        case BlockKind::L3Bank:           return "#f4f6f8";
        case BlockKind::BlockMover:       return "#7b828c";
        case BlockKind::NocRouter:        return "#4a5059";
        case BlockKind::NocPort:          return "#2a5db0";
        case BlockKind::ComputeTile:      return "#e6e2da";
        case BlockKind::L2Bank:           return "#f7f4ee";
        case BlockKind::L1Vector:         return "#efe9de";
        case BlockKind::RegisterFile:     return "#d9d2c4";
        case BlockKind::MemoryController: return "#d6dbe4";
        case BlockKind::DmaEngine:        return "#9fb3d1";
        case BlockKind::CpuHart:          return "#c7cde0";
        case BlockKind::CpuSram:          return "#e9ebf3";
        case BlockKind::DescriptorRing:
        case BlockKind::CompletionRing:   return "#b8c2dc";
    }
    return "#ffffff";
}
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '&') o += "&amp;";
        else if (c == '<') o += "&lt;";
        else if (c == '>') o += "&gt;";
        else if (c == '"') o += "&quot;";
        else o += c;
    }
    return o;
}
void svg_block(std::ostringstream& o, const FloorplanBlock& b, int depth) {
    const Rect& r = b.rect;
    o << "<rect x=\"" << r.x_um << "\" y=\"" << r.y_um << "\" width=\"" << r.w_um << "\" height=\""
      << r.h_um << "\" fill=\"" << fill_of(b.kind) << "\" stroke=\"#9aa1ab\" stroke-width=\""
      << (depth == 0 ? 8 : 3) << "\"";
    if (b.kind == BlockKind::Io) o << " stroke-dasharray=\"20 12\"";
    o << "><title>" << esc(b.name) << (b.clock_domain.empty() ? "" : " (" + b.clock_domain + " clock)")
      << "</title></rect>\n";
    if (!b.label.empty() && (b.kind == BlockKind::L3Tile || b.kind == BlockKind::ComputeTile ||
                             b.kind == BlockKind::MemoryController || b.kind == BlockKind::Io))
        o << "<text x=\"" << r.x_um + 30 << "\" y=\"" << r.y_um + 110 << "\" font-size=\"90\" "
          << "font-family=\"monospace\" fill=\"#4a5059\">" << esc(b.label) << "</text>\n";
    // A port is small; its label goes beside it, on the side away from the array.
    if (b.kind == BlockKind::NocPort && !b.label.empty()) {
        const char side = b.label.back();
        const double tx = side == 'W' ? r.x_um - 30 : r.x_um + r.w_um + 30;
        const double ty = side == 'N' ? r.y_um - 30 : side == 'S' ? r.y_um + r.h_um + 100 : r.cy() + 30;
        o << "<text x=\"" << (side == 'N' || side == 'S' ? r.x_um : tx) << "\" y=\"" << ty
          << "\" font-size=\"80\" font-family=\"monospace\" fill=\"#2a5db0\" text-anchor=\""
          << (side == 'W' ? "end" : "start") << "\">" << esc(b.label) << "</text>\n";
    }
    for (const FloorplanBlock& c : b.children) svg_block(o, c, depth + 1);
}
} // namespace

std::string floorplan_to_svg(const SocFloorplan& fp) {
    std::ostringstream o;
    o << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"" << fp.die.x_um << " " << fp.die.y_um
      << " " << fp.die.w_um << " " << fp.die.h_um << "\" width=\"" << fp.die.w_um / 10 << "\" height=\""
      << fp.die.h_um / 10 << "\">\n";
    o << "<title>" << esc(fp.device) << " floorplan (" << esc(fp.source) << ")</title>\n";
    o << "<rect x=\"" << fp.die.x_um << "\" y=\"" << fp.die.y_um << "\" width=\"" << fp.die.w_um
      << "\" height=\"" << fp.die.h_um << "\" fill=\"#f4f5f6\" stroke=\"#15181c\" stroke-width=\"12\"/>\n";
    for (const FloorplanBlock& b : fp.blocks) svg_block(o, b, 0);
    for (const NocLink& l : fp.noc) {
        if (l.route_um.size() < 2) continue;
        o << "<polyline fill=\"none\" points=\"";
        for (const auto& [x, y] : l.route_um) o << x << "," << y << " ";
        const bool solid = l.kind == NocLink::Kind::Ring || l.kind == NocLink::Kind::Mesh ||
                           l.kind == NocLink::Kind::Periphery;
        const char* colour = l.kind == NocLink::Kind::Attach || l.kind == NocLink::Kind::Periphery ? "#2a5db0"
                           : l.kind == NocLink::Kind::Credit ? "#b0702a" : "#4a5059";
        o << "\" stroke=\"" << colour << "\" "
          << "stroke-width=\"" << (solid ? 10 : l.kind == NocLink::Kind::Credit ? 5 : 8) << "\" stroke-opacity=\"0.55\""
          << (solid ? "" : " stroke-dasharray=\"30 20\"") << "><title>"
          << to_string(l.kind) << ": " << esc(l.a) << " -- " << esc(l.b) << "</title></polyline>\n";
    }
    o << "</svg>\n";
    return o.str();
}

} // namespace sw::kpu::program::platform
