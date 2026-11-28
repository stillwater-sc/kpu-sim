#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_broadcast.hpp
// The P5 generator (#609; docs/plans/distributed-csp-programs.md §3.4): one small operand to every
// node of a DMM mesh, as a level-2 program -- a bias row, RoPE's tables, BatchNorm's scale and shift.
//
// THE PIPELINE, every hop one hop. The corner node (0, 0) loads each tile of the vector through its
// DMA engine. Row 0 passes it east (bm.E), and every node of row 0 passes it south down its column
// (bm.S). So the tile reaches node (r, c) after r + c hops, the least a one-hop mesh allows.
//
// HELD, NOT RE-READ. A node keeps its L3 copy held (`reads 0`): the copy serves the node's pushes
// onward and its feed, and stays to the end, as a broadcast operand is used by every call that needs
// it. The third input mover (bm.x, decided 2026-11-27) feeds it into the compute tile's L2, held
// too, where P6's elementwise and epilogue calls will read it.
//
// The slots: tile n of the vector in L3 slot n and L2 slot n of every node. Only the corner takes a
// stream in, so the program needs one DMA engine.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/gen/distributed_gemm.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::gen {

struct DistributedBroadcast {
    Dim length = 0, tile = 0;           // the vector: `length` elements in tiles of `tile`
    std::string type = "int32";
    std::string name = "broadcast";
};

inline DistributedProgram generate_distributed_broadcast(const DistributedBroadcast& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed broadcast of " + std::to_string(g.length) + " on \"" + d.name + "\": " + why);
    };
    if (g.length == 0 || g.tile == 0) return fail("the length and the tile must be non-zero");
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    const Dim n = g.length / g.tile + (g.length % g.tile != 0);
    if (n > d.cluster->l3_tile.capacity_tiles)
        return fail(std::to_string(n) + " tiles held in every L3 tile, which has " + std::to_string(d.cluster->l3_tile.capacity_tiles) + " slots");
    if (n > d.cluster->compute_tile.l2.slots)
        return fail(std::to_string(n) + " tiles held in every compute tile's L2, which has " +
                    std::to_string(d.cluster->compute_tile.l2.slots) + " slots");
    if (d.dma.engines == 0) return fail("the device has no DMA engine");

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"b", true, g.length, 1, g.tile, 1, g.type, "in"}};
    std::vector<std::uint32_t> slots;
    for (std::uint32_t s = 0; s < n; ++s) slots.push_back(s);
    std::map<std::string, std::size_t> proc;
    const auto act = [&](const ProcessId& id, Action2 x) {
        const auto [it, fresh] = proc.emplace(id.to_string(), p.processes.size());
        if (fresh) p.processes.push_back(Process2{id, {}});
        p.processes[it->second].actions.push_back(p.actions.size());
        p.actions.push_back(std::move(x));
    };
    const auto channel = [&](const std::string& name, const ProcessId& from, Channel2::To to, Dim t) {
        p.channels.push_back(Channel2{name, from, to, t, 0, 0, slots, 0});
        return p.channels.size() - 1;
    };
    const auto mover = [](Role r, Dim t) { return ProcessId{r, t, 0, 0}; };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };

    const ProcessId dma{Role::Dma, 0, 0, 0};
    // Every node's inbound channel: the corner's from the DMA engine, row 0's from the west, the
    // rest from the north.
    std::map<Dim, std::size_t> in;
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = L->node(r, c);
            if (r == 0 && c == 0) in[t] = channel("in0", dma, Channel2::To::L3, t);
            else if (r == 0) in[t] = channel("e" + std::to_string(L->node(0, c - 1)) + "to" + std::to_string(t), mover(Role::SendE, L->node(0, c - 1)), Channel2::To::L3, t);
            else in[t] = channel("s" + std::to_string(L->node(r - 1, c)) + "to" + std::to_string(t), mover(Role::SendS, L->node(r - 1, c)), Channel2::To::L3, t);
        }
    for (std::uint32_t s = 0; s < n; ++s) {
        Action2 x;
        x.kind = Action2::Kind::Load;
        x.tile = TileCoord{"b", s, 0};
        x.to = L3s(L->node(0, 0), s);
        x.reads = 0;                                          // held: forwarded and fed, never freed
        x.channel = in.at(L->node(0, 0));
        act(dma, x);
    }
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = L->node(r, c);
            const auto forward = [&](Role role, Dim to) {
                for (std::uint32_t s = 0; s < n; ++s) {
                    Action2 x;
                    x.kind = Action2::Kind::Push;
                    x.tile = TileCoord{"b", s, 0};
                    x.from = L3s(t, s);
                    x.to = L3s(to, s);
                    x.reads = 0;
                    x.channel = in.at(to);
                    act(mover(role, t), x);
                }
            };
            if (r == 0 && c + 1 < C) forward(Role::SendE, L->node(0, c + 1));
            if (r + 1 < R) forward(Role::SendS, L->node(r + 1, c));
            const std::size_t fx = p.channels.size();
            p.channels.push_back(Channel2{"fx" + std::to_string(t), mover(Role::FeedX, t), Channel2::To::L2, t, 0, 0, slots, 0});
            for (std::uint32_t s = 0; s < n; ++s) {
                Action2 x;
                x.kind = Action2::Kind::Feed;
                x.tile = TileCoord{"b", s, 0};
                x.from = L3s(t, s);
                x.to = L2s(t, s);
                x.reads = 0;                                  // held in L2 for the calls that use it
                x.channel = fx;
                act(mover(Role::FeedX, t), x);
            }
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
