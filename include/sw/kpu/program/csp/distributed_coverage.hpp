#pragma once
// ============================================================================
// include/sw/kpu/program/csp/distributed_coverage.hpp
// The coverage measures (#617; docs/plans/distributed-csp-programs.md §3.5): what a level-2 program
// asks of the mesh it runs on, read off the program itself, so that the report on whether the
// 8-mover, one-hop allocation covers the pattern classes is measured, not written by hand.
//
// MESH. Movers and send movers a node runs (the most on one node), the L3 and L2 slots its inbound
// channels own (the most on one node), the credits per mesh flow (fewest and most), the hops a tile
// travels (for each tile at each node it reaches, the fewest pushes from where it was loaded or made,
// the most of those), and the pushes in transit: out of a node that neither feeds the tile nor made
// it, so the node only forwards it.
//
// PERIPHERY. Each DMA flow is a channel between a DMA engine and an edge node; its bytes are its tiles'
// elements at their type's width. An edge node's port is on the side its flow faces: a store's side is
// its sender's direction (bm.N leaves north), a load's is the node's first side of W, N, E, S that is on
// the edge. The ring model, a hypothesis stated with its numbers: N PHYs evenly spaced around the
// mesh's perimeter, each tensor interleaved over all of them, so each flow's bytes are split evenly
// over the N PHYs and each share takes the shorter way around the ring to its port. The busiest of the
// perimeter's unit segments (one per edge-node side) is the ring's load.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/distributed_behavioral.hpp>
#include <sw/kpu/program/csp/distributed_check.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp {

struct DistributedCoverage {
    Dim nodes = 0;
    std::size_t movers = 0, sends = 0;          // the most on one node
    std::uint32_t l3_slots = 0, l2_slots = 0;   // the most an inbound channel set owns on one node
    std::uint32_t l3_capacity = 0, l2_capacity = 0;
    std::uint32_t credits_min = 0, credits_max = 0;   // per mesh flow (a channel between two nodes)
    Dim hops = 0;                               // the farthest a tile travels, in pushes
    std::size_t pushes = 0, transit = 0;        // mesh pushes, and those a node only forwards
    std::size_t rounds = 0;                     // L-B's rounds to run it (a step count, not a timing)
    bool checks = false;                        // D0-D4 clean and D5 runs
    std::string failure;                        // the first problem, when not
    std::uint64_t dma_bytes = 0;                // every DMA flow's, in and out
    std::uint64_t port_bytes = 0;               // the busiest edge port's
    std::map<Dim, std::uint64_t> ring;          // N PHYs -> the busiest ring segment's bytes
};

// Bytes per element: the width of its number system's word.
inline std::uint64_t element_bytes(const std::string& type) {
    if (type == "int8" || type == "uint8" || type == "fp8" || type == "e4m3" || type == "e5m2") return 1;
    if (type == "int16" || type == "fp16" || type == "bf16") return 2;
    return 4;
}

inline DistributedCoverage measure_distributed(const DistributedProgram& p, const platform::DeviceSpecification& d,
                                               std::vector<Dim> ring_phys = {4, 8, 16, 32}) {
    DistributedCoverage m;
    m.nodes = p.node_count();
    m.l3_capacity = d.cluster ? d.cluster->l3_tile.capacity_tiles : 0;
    m.l2_capacity = d.cluster ? d.cluster->compute_tile.l2.slots : 0;
    // ---- checks: D0-D4, then D5 on L-B.
    if (std::string s = validate_structure(p); !s.empty()) m.failure = s;
    else if (const auto v = check_distributed(p, d); !v.empty()) m.failure = v.front().rule + " " + v.front().message;
    else {
        try {
            DistributedBehavioral lb;
            m.rounds = lb.run(p, distributed_inputs(p)).rounds;
            m.checks = true;
        } catch (const std::exception& e) {
            m.failure = e.what();
        }
    }
    // ---- movers per node.
    std::map<Dim, std::set<Role>> movers;
    for (const Process2& pr : p.processes)
        if (pr.id.role != Role::Dma && pr.id.role != Role::Compute) movers[pr.id.node].insert(pr.id.role);
    for (const auto& [t, roles] : movers) {
        m.movers = std::max(m.movers, roles.size());
        m.sends = std::max<std::size_t>(m.sends, static_cast<std::size_t>(std::count_if(roles.begin(), roles.end(), is_send)));
    }
    // ---- slots per node, credits per mesh flow.
    std::map<Dim, std::uint32_t> l3, l2;
    m.credits_min = std::numeric_limits<std::uint32_t>::max();
    for (const Channel2& c : p.channels) {
        if (c.to == Channel2::To::L3) l3[c.node] += static_cast<std::uint32_t>(c.slots.size());
        if (c.to == Channel2::To::L2) l2[c.node] += static_cast<std::uint32_t>(c.slots.size());
        if (c.to == Channel2::To::L3 && is_send(c.from.role)) {
            m.credits_min = std::min(m.credits_min, c.credit_count());
            m.credits_max = std::max(m.credits_max, c.credit_count());
        }
    }
    if (m.credits_min == std::numeric_limits<std::uint32_t>::max()) m.credits_min = 0;
    for (const auto& [t, n] : l3) m.l3_slots = std::max(m.l3_slots, n);
    for (const auto& [t, n] : l2) m.l2_slots = std::max(m.l2_slots, n);
    // ---- hops: the fewest pushes from a tile's origin (its load or its drain) to each node it reaches.
    using At = std::pair<std::string, Dim>;
    std::map<At, Dim> depth;
    std::set<At> used;                                         // fed there, or made there
    std::vector<std::pair<At, At>> edges;
    for (const Process2& pr : p.processes)
        for (std::size_t ai : pr.actions) {
            const Action2& a = p.actions[ai];
            const std::string k = a.tile.to_string();
            if (a.kind == Action2::Kind::Load) depth[{k, a.to->node}] = 0;
            if (a.kind == Action2::Kind::Drain) {
                depth[{k, a.to->node}] = 0;
                used.insert({k, a.to->node});
            }
            if (a.kind == Action2::Kind::Feed) used.insert({k, pr.id.node});
            if (a.kind == Action2::Kind::Push && a.to) {
                ++m.pushes;
                edges.push_back({{k, a.from->node}, {k, a.to->node}});
            }
        }
    for (bool changed = true; changed;) {                     // relax: each pass can only shorten
        changed = false;
        for (const auto& [from, to] : edges) {
            const auto f = depth.find(from);
            if (f == depth.end()) continue;
            const auto t = depth.find(to);
            if (t == depth.end() || t->second > f->second + 1) {
                depth[to] = f->second + 1;
                changed = true;
            }
        }
    }
    for (const auto& [at, h] : depth) m.hops = std::max(m.hops, h);
    for (const auto& [from, to] : edges)
        if (!used.count(from)) ++m.transit;
    // ---- periphery: DMA flows' bytes, by edge port.
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return m;
    const Dim R = L->rows(), C = L->cols();
    const auto tile_bytes = [&](const TileCoord& t) -> std::uint64_t {
        const Operand2* o = p.operand(t.operand);
        if (!o) return 0;
        const Dim r0 = t.ti * o->tile_rows, r1 = std::min(o->rows, r0 + o->tile_rows);
        const Dim c0 = o->is_vector ? 0 : t.tj * o->tile_cols, c1 = o->is_vector ? 1 : std::min(o->cols, c0 + o->tile_cols);
        return static_cast<std::uint64_t>(r1 - r0) * (c1 - c0) * element_bytes(o->type);
    };
    // A port: (node, side), side 0..3 = N, E, S, W. Its place on the perimeter, clockwise from the
    // north-west corner, one unit per edge-node side.
    const auto place = [&](Dim t, int side) -> double {
        const auto at = L->pos(t);
        switch (side) {
            case 0: return at.c + 0.5;
            case 1: return C + at.r + 0.5;
            case 2: return C + R + (C - 1 - at.c) + 0.5;
            default: return 2.0 * C + R + (R - 1 - at.r) + 0.5;
        }
    };
    const auto load_side = [&](Dim t) {
        const auto at = L->pos(t);
        if (at.c == 0) return 3;
        if (at.r == 0) return 0;
        if (at.c + 1 == C) return 1;
        return 2;
    };
    std::map<std::pair<Dim, int>, std::uint64_t> port;
    for (const Action2& a : p.actions) {
        if (a.kind == Action2::Kind::Load) port[{a.to->node, load_side(a.to->node)}] += tile_bytes(a.tile);
        if (a.kind == Action2::Kind::Push && !a.to) {
            const Channel2& c = p.channels.at(a.channel);
            const int side = c.from.role == Role::SendN ? 0 : c.from.role == Role::SendE ? 1 : c.from.role == Role::SendS ? 2 : 3;
            port[{c.from.node, side}] += tile_bytes(a.tile);
        }
    }
    for (const auto& [pt, b] : port) {
        m.dma_bytes += b;
        m.port_bytes = std::max(m.port_bytes, b);
    }
    const double perimeter = 2.0 * (R + C);
    const std::size_t segments = static_cast<std::size_t>(perimeter);
    for (Dim n : ring_phys) {
        std::vector<double> load(segments, 0.0);
        for (const auto& [pt, b] : port) {
            const double to = place(pt.first, pt.second), share = static_cast<double>(b) / n;
            for (Dim k = 0; k < n; ++k) {
                const double from = perimeter * k / n;
                double cw = to - from;                         // clockwise distance from the PHY to the port
                if (cw < 0) cw += perimeter;
                const bool clockwise = cw <= perimeter - cw;
                const double len = clockwise ? cw : perimeter - cw;
                // Every unit segment the route passes over carries the share.
                for (std::size_t s = 0; s < segments; ++s) {
                    const double mid = s + 0.5;
                    double off = clockwise ? mid - from : from - mid;
                    off = std::fmod(off + 2 * perimeter, perimeter);
                    if (off < len) load[s] += share;
                }
            }
        }
        m.ring[n] = static_cast<std::uint64_t>(std::llround(*std::max_element(load.begin(), load.end())));
    }
    return m;
}

} // namespace sw::kpu::program::csp
