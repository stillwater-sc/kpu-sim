#pragma once
// ============================================================================
// include/sw/kpu/program/csp/distributed_check.hpp
// The distributed checker (#607; docs/plans/distributed-csp-programs.md §3.3, §4 step 4): a level-2
// program against the machine it is written for, before any simulation.
//
//   D0 machine     the program's mesh is the deployment's (a dmm_mesh, rows x cols)
//   D1 resources   every process exists on the deployment (a node's movers, its compute tile, a
//                  controller's DMA engine); a mover or a DMA engine sends on at most one channel:
//                  one CSP per packet flow, so no flow waits behind another's credit
//   D2 one hop     a send mover reaches only the neighbour on its side, or, where it has none, a DMA
//                  engine through its edge port onto the periphery; a feed mover and the compute
//                  tile fill their own compute tile's L2, the drain mover its own L3; a DMA engine
//                  reaches the mesh only at an edge node. Edge to edge over the periphery is refused
//                  (it carries DMA flows only, decided 2026-11-27).
//   D3 credits     every channel is used: a channel nothing sends on holds credits no one returns
//   D4 placement   the slots fit the deployment (L3 slots per tile, the compute tile's L2 slots when
//                  declared); per slot, the reads its residencies declare are the reads the program
//                  makes, across every process, a call's operand counted once per mention; so are
//                  the merges; a held residency (reads 0) is its slot's last; and every tile read from
//                  a slot is one some write placed there
//
// D5 (liveness) and D6 (values) are L-B's: distributed_behavioral.hpp. Structure (names, roles,
// a channel's own slots) is validate_structure's, run first.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

struct DistributedViolation {
    std::string rule;                   // "D0" .. "D4"
    std::string message;
};

namespace detail {

inline std::optional<platform::Edge> send_edge(Role r) {
    switch (r) {
        case Role::SendN: return platform::Edge::N;
        case Role::SendE: return platform::Edge::E;
        case Role::SendS: return platform::Edge::S;
        case Role::SendW: return platform::Edge::W;
        default:          return std::nullopt;
    }
}

} // namespace detail

// Every violation, in rule order and then program order; empty when the program passes D0-D4.
inline std::vector<DistributedViolation> check_distributed(const DistributedProgram& p,
                                                           const platform::DeviceSpecification& d) {
    std::vector<DistributedViolation> out;
    const auto fail = [&](const char* rule, std::string m) { out.push_back({rule, std::move(m)}); };
    if (const std::string bad = validate_structure(p); !bad.empty()) {
        fail("D0", "the program's structure: " + bad);
        return out;
    }
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) {
        fail("D0", "device \"" + d.name + "\" is not a mesh the program can run on: " + why);
        return out;
    }
    if (L->rows() != p.rows || L->cols() != p.cols) {
        fail("D0", "the program is written for a " + std::to_string(p.rows) + "x" + std::to_string(p.cols) + " mesh; device \"" +
                       d.name + "\" is " + std::to_string(L->rows()) + "x" + std::to_string(L->cols()));
        return out;
    }
    const Dim mcs = d.memory.controllers.value_or(1);
    const Dim per_mc = d.dma.engines / mcs;

    // ---- D1 resources
    std::map<std::string, std::vector<std::string>> sends;   // process -> the channels it sends on
    for (const Channel2& c : p.channels) sends[c.from.to_string()].push_back(c.name);
    const auto exists = [&](const ProcessId& id) {
        return id.role == Role::Dma ? id.mc < mcs && id.engine < per_mc : id.node < L->node_count();
    };
    for (const Process2& pr : p.processes)
        if (!exists(pr.id))
            fail("D1", pr.id.to_string() + " does not exist on device \"" + d.name + "\" (" + std::to_string(mcs) +
                           " memory controller(s), " + std::to_string(per_mc) + " DMA engine(s) each)");
    for (const Channel2& c : p.channels) {
        if (!exists(c.from)) fail("D1", "channel " + c.name + ": its sender " + c.from.to_string() + " does not exist on the device");
        if (c.to == Channel2::To::Dma && (c.mc >= mcs || c.engine >= per_mc))
            fail("D1", "channel " + c.name + ": " + c.destination() + " does not exist on the device");
    }
    for (const auto& [proc, chans] : sends) {
        if ((proc.rfind("bm.", 0) != 0 && proc.rfind("dma[", 0) != 0) || chans.size() <= 1) continue;
        std::string list;
        for (const std::string& c : chans) list += (list.empty() ? "" : ", ") + c;
        fail("D1", proc + " sends on " + std::to_string(chans.size()) + " channels (" + list +
                       "): one CSP carries one flow, so neither waits behind the other's credit");
    }

    // ---- D2 one hop
    for (const Channel2& c : p.channels) {
        const std::string w = "channel " + c.name + " (" + c.from.to_string() + " -> " + c.destination() + ")";
        const Role r = c.from.role;
        const Dim t = c.from.node;
        if (const auto e = detail::send_edge(r)) {
            const auto nb = L->neighbour(t, *e);
            if (nb) {
                if (c.to != Channel2::To::L3 || c.node != *nb)
                    fail("D2", w + ": node " + std::to_string(t) + "'s " + platform::to_string(*e) + " link goes one hop, to l3[" +
                                   std::to_string(*nb) + "]");
            } else if (c.to != Channel2::To::Dma) {
                fail("D2", w + ": node " + std::to_string(t) + "'s " + platform::to_string(*e) +
                               " side is an edge port, onto the periphery, which carries DMA flows only");
            }
        } else if (is_feed(r) || r == Role::Compute) {
            if (c.to != Channel2::To::L2 || c.node != t)
                fail("D2", w + ": a " + std::string(role_name(r)) + " fills its own compute tile's L2, l2[" + std::to_string(t) + "]");
        } else if (r == Role::Drain) {
            if (c.to != Channel2::To::L3 || c.node != t)
                fail("D2", w + ": a drain fills its own node's L3, l3[" + std::to_string(t) + "]");
        } else {                                                  // a DMA engine
            if (c.to != Channel2::To::L3)
                fail("D2", w + ": a DMA engine pushes into an edge node's L3 over the periphery");
            else if (c.node < L->node_count()) {
                bool edge = false;
                for (platform::Edge e : {platform::Edge::N, platform::Edge::E, platform::Edge::S, platform::Edge::W})
                    edge |= L->edge_port(c.node, e).has_value();
                if (!edge)
                    fail("D2", w + ": node " + std::to_string(c.node) +
                                   " has no edge port; a DMA engine reaches the mesh only at an edge node");
            }
        }
    }

    // ---- D3 credits
    // A channel is used by its sender's actions; a Store receives from its channel and does not count.
    std::set<std::size_t> used;
    std::map<std::size_t, std::size_t> delivered, stored;     // a channel into a DMA engine: pushes, stores
    for (const Process2& pr : p.processes)                       // as executed: each process's own list
        for (std::size_t ai : pr.actions) {
            const Action2& a = p.actions[ai];
            if (a.channel == kNoChannel) continue;
            if (a.kind == Action2::Kind::Store) {
                ++stored[a.channel];
                continue;
            }
            used.insert(a.channel);
            if (!a.to) ++delivered[a.channel];
        }
    for (std::size_t i = 0; i < p.channels.size(); ++i) {
        const Channel2& c = p.channels[i];
        if (!used.count(i))
            fail("D3", "channel " + c.name + " is never used: its " + std::to_string(c.credit_count()) +
                           " credit(s) would be held by a flow that never sends");
        else if (c.to == Channel2::To::Dma && delivered[i] != stored[i])
            fail("D3", "channel " + c.name + " delivers " + std::to_string(delivered[i]) + " tile(s) to " + c.destination() +
                           ", which stores " + std::to_string(stored[i]) +
                           ": an unmatched store waits forever, an unmatched push holds a store-buffer credit");
    }

    // ---- D4 placement
    // Bounded when declared: capacity_tiles 0 is unbounded, and a declared capacity that gives a
    // tile no slot bounds it at zero (every slot is outside), not at "unbounded".
    const bool l3_bounded = d.l3.capacity_tiles != 0 && d.l3.tiles.has_value();
    const Dim l3_per_tile = l3_bounded ? d.l3.capacity_tiles / *d.l3.tiles : 0;
    const bool l2_bounded = d.systolic.has_value();
    const Dim l2_slots = l2_bounded ? d.systolic->l2_slots : 0;
    for (const Channel2& c : p.channels) {
        if (c.to == Channel2::To::Dma) continue;
        const bool bounded = c.to == Channel2::To::L3 ? l3_bounded : l2_bounded;
        const Dim cap = c.to == Channel2::To::L3 ? l3_per_tile : l2_slots;
        if (!bounded) continue;                                   // undeclared: unbounded
        for (std::uint32_t s : c.slots)
            if (s >= cap)
                fail("D4", "channel " + c.name + ": slot " + std::to_string(s) + " of " + c.destination() + " is outside its " +
                               std::to_string(cap) + " slot(s)");
    }
    struct Demand {
        std::uint64_t declared = 0, made = 0, merges_declared = 0, merges_made = 0;
        bool held = false, written_after_held = false;
        std::set<std::string> placed, read;
        std::size_t writes = 0;
    };
    std::map<SlotRef, Demand> slots;
    // Writes, in each writing process's own order (a slot has one writer: its channel's sender).
    for (const Process2& pr : p.processes)
        for (std::size_t ai : pr.actions) {
            const Action2& a = p.actions[ai];
            if (a.to && a.opens()) {
                Demand& s = slots[*a.to];
                if (s.held) s.written_after_held = true;
                ++s.writes;
                s.placed.insert(a.tile.to_string());
                if (a.reads == 0) s.held = true;
                else s.declared += a.reads;
                s.merges_declared += a.merges;
            }
            if (a.kind == Action2::Kind::Call && a.accumulate) ++slots[*a.to].merges_made;
            if (a.from) {
                Demand& s = slots[*a.from];
                ++s.made;
                s.read.insert(a.tile.to_string());
            }
            for (const SlotRef& r : a.args) ++slots[r].made;
        }
    for (const auto& [at, s] : slots) {
        const std::string w = "slot " + at.to_string();
        if (!s.writes && (s.made || s.merges_made)) {
            fail("D4", w + " is read but nothing writes it");
            continue;
        }
        if (s.written_after_held)
            fail("D4", w + ": a residency held to the end (reads 0) is followed by another write, which would wait forever");
        if (!s.held && s.made != s.declared)
            fail("D4", w + ": its residencies declare " + std::to_string(s.declared) + " read(s), and the program makes " +
                           std::to_string(s.made));
        // A held residency is the slot's last: the ones before it must have all their reads, or its
        // write never gets the credit (the held one may take any number of reads after them).
        if (s.held && s.made < s.declared)
            fail("D4", w + ": its residencies before the held one declare " + std::to_string(s.declared) +
                           " read(s), and the program makes " + std::to_string(s.made) + ", so the held write never gets its credit");
        if (s.merges_made != s.merges_declared)
            fail("D4", w + ": its results expect " + std::to_string(s.merges_declared) + " merge(s), and the program makes " +
                           std::to_string(s.merges_made));
        for (const std::string& t : s.read)
            if (!s.placed.count(t)) fail("D4", w + ": " + t + " is read from it, but no write places " + t + " there");
    }
    return out;
}

} // namespace sw::kpu::program::csp
