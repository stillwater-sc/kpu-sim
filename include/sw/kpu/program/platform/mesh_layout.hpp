#pragma once
// ============================================================================
// include/sw/kpu/program/platform/mesh_layout.hpp
// Where a `dmm_mesh`'s parts ARE (#604; docs/plans/distributed-csp-programs.md §2, §3.1): its nodes,
// their eight BlockMovers, the one-hop links between neighbours and the credit links back, the edge
// ports, and the periphery ring that joins the edge ports to the memory controllers' DMA engines.
//
// This is the ONE place a mesh is derived, as array_layout.hpp is for the checkerboard. The naming
// map reads it to name the movers and the edge ports, and the floorplan generator reads it to place
// them, so the two cannot disagree.
//
// ---------------------------------------------------------------------------
// THE NODES
// ---------------------------------------------------------------------------
// A rows x cols grid. Node (r, c) is index r * cols + c, and it is two tiles: l3[t] and its own
// compute tile cf[t], with the same index. It is the distributed-memory machine's node, so the DMM
// (MPI) algorithms apply to it.
//
// ---------------------------------------------------------------------------
// THE EIGHT BLOCKMOVERS (l3[t]/bm[k]), one CSP each, never shared
// ---------------------------------------------------------------------------
//   bm[0..3]  send N, E, S, W: each owns its outgoing link. On the mesh's boundary the link with
//             no neighbour is the node's EDGE PORT, onto the periphery.
//   bm[4]     feed a      L3 -> the compute tile's L2, the first input operand
//   bm[5]     feed b      the second input operand
//   bm[6]     feed x      the third input: bias, accumulator init, a gating operand (decided
//                         2026-11-27)
//   bm[7]     drain       the compute tile's L2 -> L3, the result
// A received packet lands in the L3 slot it names, under the credit its receiver granted. That is
// the L3 tile's receive port, not a mover, and not a CSP.
//
// ---------------------------------------------------------------------------
// THE LINKS: one hop, push only
// ---------------------------------------------------------------------------
// Every neighbour pair has a data link each way: from's send mover on that side pushes into the
// neighbour's L3. There are no routers and nothing passes through a node. Every data link has its
// CREDIT LINK on the separate credit network, the other way: the receiver returns a freed slot's
// credit to the one producer of that slot.
//
// ---------------------------------------------------------------------------
// THE EDGE PORTS AND THE PERIPHERY RING
// ---------------------------------------------------------------------------
// One edge port per boundary side of a boundary node (a corner has two), numbered CLOCKWISE from
// the NW corner: the N side left to right, the E side top to bottom, the S side right to left, the
// W side bottom to top. That makes 2 x (rows + cols).
//
// The periphery ring visits its stops in the same clockwise order: every edge port, and each
// memory controller after the port the spec names (periphery.mc_after_port), or spread evenly. A
// controller's DMA engines reach every edge node through the ring. The ring runs on the memory
// controllers' clock and carries DMA flows only, so a DMA flow crosses one clock domain, at its
// edge port.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/platform/array_layout.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::platform {

enum class MoverRole : std::uint8_t { SendN = 0, SendE, SendS, SendW, FeedA, FeedB, FeedX, Drain };
inline constexpr Dim kMeshMovers = 8;

inline const char* to_string(MoverRole m) {
    switch (m) {
        case MoverRole::SendN: return "N";
        case MoverRole::SendE: return "E";
        case MoverRole::SendS: return "S";
        case MoverRole::SendW: return "W";
        case MoverRole::FeedA: return "a";
        case MoverRole::FeedB: return "b";
        case MoverRole::FeedX: return "x";
        case MoverRole::Drain: return "c";
    }
    return "?";
}

// The send mover that owns the link on edge e.
inline MoverRole send_role(Edge e) { return static_cast<MoverRole>(static_cast<std::uint8_t>(e)); }

// One data link, one hop: node `from`'s send mover on `dir` pushes into node `to`'s L3. Its credit
// link is the reverse, on the credit network.
struct MeshLink {
    Dim from = 0, to = 0;
    Edge dir = Edge::N;
};

// One edge port: the node it belongs to and the side it faces.
struct EdgePort {
    Dim index = 0;                  // noc/port[index], clockwise from the NW corner
    Dim node = 0;
    Edge side = Edge::N;
    Dim along = 0;                  // the node's column (N, S) or row (E, W)
    std::string label() const { return std::string(to_string(side)) + std::to_string(along); }   // "N0", "E1"
};

// One stop of the periphery ring, in ring order.
struct PeripheryStop {
    enum class Kind : std::uint8_t { EdgePort, Controller } kind = Kind::EdgePort;
    Dim of = 0;                     // the edge port's index, or the memory controller's
};

class MeshLayout {
public:
    // The layout of device `d`, or nullopt with the reason in `why`.
    static std::optional<MeshLayout> of(const DeviceSpecification& d, std::string* why = nullptr);

    Dim rows() const { return rows_; }
    Dim cols() const { return cols_; }
    Dim node_count() const { return rows_ * cols_; }
    Dim node(Dim r, Dim c) const { return r * cols_ + c; }
    GridPos pos(Dim t) const { return GridPos{t / cols_, t % cols_}; }

    // The node on side e of node t, or nullopt on the boundary.
    std::optional<Dim> neighbour(Dim t, Edge e) const {
        const GridPos p = pos(t);
        switch (e) {
            case Edge::N: return p.r == 0 ? std::nullopt : std::optional<Dim>(node(p.r - 1, p.c));
            case Edge::S: return p.r + 1 == rows_ ? std::nullopt : std::optional<Dim>(node(p.r + 1, p.c));
            case Edge::W: return p.c == 0 ? std::nullopt : std::optional<Dim>(node(p.r, p.c - 1));
            case Edge::E: return p.c + 1 == cols_ ? std::nullopt : std::optional<Dim>(node(p.r, p.c + 1));
        }
        return std::nullopt;
    }

    // Every data link, by node, then N, E, S, W.
    const std::vector<MeshLink>& links() const { return links_; }
    const std::vector<EdgePort>& edge_ports() const { return ports_; }
    // The edge port on side e of node t, or nullopt when that side has a neighbour.
    std::optional<Dim> edge_port(Dim t, Edge e) const {
        for (const EdgePort& p : ports_)
            if (p.node == t && p.side == e) return p.index;
        return std::nullopt;
    }

    Dim controller_count() const { return static_cast<Dim>(mc_after_.size()); }
    Dim mc_after_port(Dim m) const { return mc_after_.at(m); }
    // The ring's stops, clockwise from edge port 0.
    const std::vector<PeripheryStop>& stops() const { return stops_; }

private:
    Dim rows_ = 0, cols_ = 0;
    std::vector<MeshLink> links_;
    std::vector<EdgePort> ports_;
    std::vector<Dim> mc_after_;
    std::vector<PeripheryStop> stops_;
};

inline std::optional<MeshLayout> MeshLayout::of(const DeviceSpecification& d, std::string* why) {
    auto fail = [&](std::string s) -> std::optional<MeshLayout> {
        if (why) *why = std::move(s);
        return std::nullopt;
    };
    if (d.topology != "dmm_mesh") return fail("topology '" + d.topology + "' is not a dmm_mesh");
    if (std::string bad = mesh_problem(d); !bad.empty()) return fail(bad);

    MeshLayout L;
    L.rows_ = *d.array.rows;
    L.cols_ = *d.array.cols;
    for (Dim t = 0; t < L.node_count(); ++t)
        for (Edge e : {Edge::N, Edge::E, Edge::S, Edge::W})
            if (const auto u = L.neighbour(t, e)) L.links_.push_back(MeshLink{t, *u, e});

    auto port = [&](Dim r, Dim c, Edge side, Dim along) {
        L.ports_.push_back(EdgePort{static_cast<Dim>(L.ports_.size()), L.node(r, c), side, along});
    };
    for (Dim c = 0; c < L.cols_; ++c) port(0, c, Edge::N, c);
    for (Dim r = 0; r < L.rows_; ++r) port(r, L.cols_ - 1, Edge::E, r);
    for (Dim c = L.cols_; c-- > 0;) port(L.rows_ - 1, c, Edge::S, c);
    for (Dim r = L.rows_; r-- > 0;) port(r, 0, Edge::W, r);

    // The controllers, after the ports the spec names, or spread evenly: controller m after port
    // floor((2m + 1) P / 2M). mesh_problem() has checked M <= P, so the positions are strictly
    // increasing and below P.
    const Dim P = static_cast<Dim>(L.ports_.size());
    const Dim M = *d.memory.controllers;          // declared: mesh_problem() requires it
    const auto& declared = d.periphery ? d.periphery->mc_after_port : std::vector<Dim>{};
    for (Dim m = 0; m < M; ++m)
        L.mc_after_.push_back(declared.empty()
                                  ? static_cast<Dim>((2ull * m + 1) * P / (2ull * M))
                                  : declared[m]);
    std::size_t next = 0;
    for (Dim k = 0; k < P; ++k) {
        L.stops_.push_back(PeripheryStop{PeripheryStop::Kind::EdgePort, k});
        while (next < L.mc_after_.size() && L.mc_after_[next] == k)
            L.stops_.push_back(PeripheryStop{PeripheryStop::Kind::Controller, static_cast<Dim>(next++)});
    }
    return L;
}

} // namespace sw::kpu::program::platform
