// ============================================================================
// include/sw/kpu/program/platform/floorplan.hpp
// Geometry for the naming map: where every resource is on the die (#286 step 1;
// docs/plans/tile-flow-debugger.md §3.1).
//
// ONE RULE: every rectangle that stands for hardware NAMES A NAMING-MAP RESOURCE, and every
// resource the map enumerates has exactly one rectangle. No geometry exists that an event
// record cannot address, and no record address lacks geometry -- which is what lets the
// tile-flow viewer put an event on the die without a lookup table of its own.
//
// The exceptions are named, not implied. Four block kinds are GROUPS, not hardware: the
// array outline, the CPU cluster outline, the DRAM PHYs and IO. They carry no resource, and
// no event is ever recorded against them. DRAM itself is off-die, so `dev/dram` is the one
// enumerated resource with no rectangle.
//
// TWO SOURCES, ONE TYPE:
//
//   generate_floorplan(spec)        parametric, from the deployment and its ArrayLayout. This
//                                   is the reference until a real SoC floorplan exists (plan
//                                   Q3). Units are µm on an illustrative pitch, not silicon.
//   floorplan_from_json(text, spec) a floorplan written by the physical-design flow, or by
//                                   floorplan_to_json. VALIDATED against the deployment before
//                                   it is returned; a mismatch is refused with the first
//                                   problem, the same shape as the loadable's capability check.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/resource_map.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::platform {

enum class BlockKind : std::uint8_t {
    // groups: carry no resource
    Array, Cpu, DramPhy, Io,
    // hardware: each names exactly one resource
    L3Tile, L3Bank, BlockMover, NocRouter, NocPort,
    ComputeTile, L2Bank, L1Vector, RegisterFile,
    MemoryController, DmaEngine,
    CpuHart, CpuSram, DescriptorRing, CompletionRing,
};

const char* to_string(BlockKind k);
std::optional<BlockKind> block_kind_from(const std::string& s);
// The resource kind a hardware block stands for; nullopt for a group.
std::optional<ResourceKind> resource_kind_of(BlockKind k);

struct Rect {
    double x_um = 0, y_um = 0, w_um = 0, h_um = 0;
    double cx() const { return x_um + w_um / 2; }
    double cy() const { return y_um + h_um / 2; }
    bool inside(const Rect& outer) const;     // with a sub-nanometre tolerance
    bool overlaps(const Rect& o) const;       // interiors intersect (shared edges are fine)
    bool operator==(const Rect& o) const {
        return x_um == o.x_um && y_um == o.y_um && w_um == o.w_um && h_um == o.h_um;
    }
};

struct FloorplanBlock {
    std::string name;                         // == format(*resource) when it has one
    std::optional<ResourceName> resource;
    BlockKind kind = BlockKind::Array;
    std::string clock_domain;                 // "l3", "cf", "dram", "cpu", or "" for a group
    Rect rect;
    std::string label;                        // a short human label: "row1.E", "N", "hart0"
    std::vector<FloorplanBlock> children;     // the hierarchy IS the level-of-detail ladder
};

// A wire of the NoC, so every burst has a drawable path.
struct NocLink {
    enum class Kind : std::uint8_t {
        Ring,       // hub to hub, along a torus loop
        Port,       // a fold-end port to one of the two hubs of its fold link
        Attach,     // a DMA engine to the port it injects into (first pass, see generator)
    };
    Kind kind = Kind::Ring;
    std::string a, b;                         // block names
    std::vector<std::pair<double, double>> route_um;
};
const char* to_string(NocLink::Kind k);

struct SocFloorplan {
    std::string source;                       // "generated:checkerboard", or the file's own
    std::string device;                       // the device it lays out, by name
    Rect die;
    std::vector<FloorplanBlock> blocks;
    std::vector<NocLink> noc;

    // Over the canonical JSON: a view names the floorplan it was drawn on, in provenance.
    std::string digest() const;
    // Depth-first search by name; nullptr when absent.
    const FloorplanBlock* find(const std::string& name) const;
    std::size_t block_count() const;
};

struct FloorplanStyle {
    double pitch_um = 1000.0;                 // one array cell, L3 or compute
    double gap_um = 40.0;                     // between cells
    double margin_um = 400.0;                 // around the die's contents
};

class FloorplanError : public std::runtime_error {
public:
    explicit FloorplanError(const std::string& what) : std::runtime_error(what) {}
};

// Generate device `device`'s floorplan. Throws FloorplanError naming what the spec lacks:
// a floorplan needs an array layout (array_layout.hpp), because every tile must be placed.
SocFloorplan generate_floorplan(const DeploymentSpec& spec, Dim device = 0,
                                const FloorplanStyle& style = {});

// Empty when `fp` is a floorplan of `spec`; otherwise the first problem, naming the block.
std::string validate_floorplan(const SocFloorplan& fp, const DeploymentSpec& spec);

std::string floorplan_to_json(const SocFloorplan& fp);
// Parses and VALIDATES; throws FloorplanError on malformed text or a floorplan that does
// not match the deployment.
SocFloorplan floorplan_from_json(const std::string& text, const DeploymentSpec& spec);

// A self-contained SVG of the floorplan and its NoC, for review and for the viewer's tests.
std::string floorplan_to_svg(const SocFloorplan& fp);

} // namespace sw::kpu::program::platform
