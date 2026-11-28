#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_transpose.hpp
// The P7 generator (#613; docs/plans/distributed-csp-programs.md §3.4): a layout transform -- Y = X^T,
// a block-distributed matrix transposed -- as a level-2 program on a square DMM mesh, every hop one
// hop. It is the one-hop design's test case: every tile crosses the mesh by store-and-forward, and the
// program measures what that costs (DistributedTransposeCost).
//
// THE DISTRIBUTION. One tile per node: X[r, c] lives at node (r, c), and Y[c, r] = X[r, c]^T belongs to
// node (c, r). So the mesh is square.
//
// THE ROUTE, XY through the diagonal. X[r, c] is loaded at its row's west edge node and moves east to
// its home (r, c), as any row's input does. From there it moves along row r to the diagonal node
// (r, r), then along column r to (c, r), where the compute tile transposes it into Y[c, r]. Y then moves
// north to row 0 and out of its edge port to a DMA engine. Every hop lands in the next node's L3.
//
// NO CYCLE. Each sending mover carries every tile bound for its neighbour, in the order a synchronous
// schedule (one hop per step) would deliver them; the receiver gives each tile delivered into its L3 a
// slot of its own. So no delivery into an L3 slot waits for a credit, and a move waits only on an
// arrival from an earlier step. The one credit wait left is row 0's pushes out to the DMA engine: its
// store buffer has 2 credits and a column's channel may carry more Ys (4 on the 4x4), so a push can
// wait for the engine to store. The engine waits on nothing those pushes hold, so that wait ends: the
// program cannot deadlock. The slots this takes is the transform's L3 cost, checked against the tile's
// capacity.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/gen/distributed_gemm.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace sw::kpu::program::csp::gen {

struct DistributedTranspose {
    Dim m = 0, n = 0;                   // X is m x n, one tile per node
    std::string type = "int32";
    std::string name = "transpose";
};

// What the transform cost, as the program moves it.
struct DistributedTransposeCost {
    Dim max_hops = 0;                   // the longest route, home to destination
    Dim total_hops = 0;                 // every tile's route, summed
    Dim max_l3_slots = 0;               // the most L3 slots a node's channels and drain take
};

inline DistributedProgram generate_distributed_transpose(const DistributedTranspose& g, const platform::DeviceSpecification& d,
                                                         DistributedTransposeCost* cost = nullptr) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed transpose " + std::to_string(g.m) + "x" + std::to_string(g.n) + " on \"" + d.name + "\": " + why);
    };
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    if (R != C) return fail("a one-tile-per-node transpose maps node (r, c) to (c, r), so the mesh is square; this one is " +
                            std::to_string(R) + "x" + std::to_string(C));
    if (g.m == 0 || g.n == 0 || g.m % R != 0 || g.n % C != 0)
        return fail("X divides into one tile per node, so its rows and columns are multiples of the mesh's " + std::to_string(R));
    if (R > d.dma.engines)
        return fail(std::to_string(R) + " mesh rows take a stream in, one DMA engine each, and the device has " + std::to_string(d.dma.engines));
    if (d.cluster->compute_tile.l2.slots < 2) return fail("a compute tile needs 2 L2 slots");
    const Dim tr = g.m / R, tc = g.n / C;
    const Dim mcs = d.memory.controllers.value_or(1);

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"X", false, g.m, g.n, tr, tc, g.type, "in"}, Operand2{"Y", false, g.n, g.m, tc, tr, g.type, "out"}};
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };
    const auto engine = [&](Dim r) { return ProcessId{Role::Dma, 0, r % mcs, r / mcs}; };

    // ---- the schedule: every hop, with the step it leaves at.
    struct Hop {
        TileCoord tile;
        Dim from = 0, to = 0;
        platform::Edge dir = platform::Edge::E;
        int depart = 0;
        bool out = false;               // row 0 north: out of the edge port, to a DMA engine
    };
    std::vector<Hop> hops;
    std::map<std::string, std::pair<Dim, int>> arrival;     // tile -> its destination node, and the step it arrives
    const auto step = [&](Dim r, Dim c, platform::Edge e) -> std::pair<Dim, Dim> {
        switch (e) {
            case platform::Edge::E: return {r, c + 1};
            case platform::Edge::W: return {r, c - 1};
            case platform::Edge::S: return {r + 1, c};
            case platform::Edge::N: return {r - 1, c};
        }
        return {r, c};
    };
    Dim total = 0, longest = 0;
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const TileCoord X{"X", r, c};
            Dim at_r = r, at_c = 0;
            int t = static_cast<int>(c);                          // the k-th load of the row lands at step k
            const auto go = [&](platform::Edge e) {
                const auto [nr, nc] = step(at_r, at_c, e);
                hops.push_back(Hop{X, node(at_r, at_c), node(nr, nc), e, t, false});
                at_r = nr;
                at_c = nc;
                ++t;
            };
            while (at_c < c) go(platform::Edge::E);              // the load: west edge to home
            Dim moved = 0;
            while (at_c != r) { go(at_c < r ? platform::Edge::E : platform::Edge::W); ++moved; }   // along row r to (r, r)
            while (at_r != c) { go(at_r < c ? platform::Edge::S : platform::Edge::N); ++moved; }   // along column r to (c, r)
            arrival[X.to_string()] = {node(at_r, at_c), t};
            total += moved;
            longest = std::max(longest, moved);
        }
    // Y north from each destination, three steps after its X arrives (the feed, the call, the drain).
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const auto [dest, ta] = arrival.at(TileCoord{"X", r, c}.to_string());
            const TileCoord Y{"Y", c, r};
            Dim at_r = L->pos(dest).r;
            const Dim col = L->pos(dest).c;
            int t = ta + 3;
            while (true) {
                if (at_r == 0) {
                    hops.push_back(Hop{Y, node(0, col), node(0, col), platform::Edge::N, t, true});
                    break;
                }
                hops.push_back(Hop{Y, node(at_r, col), node(at_r - 1, col), platform::Edge::N, t, false});
                --at_r;
                ++t;
            }
        }
    // Each mover's sends in step order (ties by tile), and each channel's delivered tiles a slot each.
    std::stable_sort(hops.begin(), hops.end(), [](const Hop& a, const Hop& b) {
        return std::tie(a.depart, a.tile.operand, a.tile.ti, a.tile.tj) < std::tie(b.depart, b.tile.operand, b.tile.ti, b.tile.tj);
    });

    // ---- channels: one per (sender, direction) that sends, and each node's DMA load channel at column 0.
    std::map<std::pair<Dim, int>, std::size_t> chan;          // (from node, dir) -> channel
    std::map<Dim, std::uint32_t> next_slot;                   // receiver node -> its next free L3 slot
    std::map<std::pair<std::size_t, std::string>, std::uint32_t> slot_of;   // (channel, tile) -> its slot at the receiver
    const auto role = [](platform::Edge e) {
        switch (e) {
            case platform::Edge::N: return Role::SendN;
            case platform::Edge::E: return Role::SendE;
            case platform::Edge::S: return Role::SendS;
            case platform::Edge::W: return Role::SendW;
        }
        return Role::SendN;
    };
    std::map<Dim, std::size_t> load_chan;
    for (Dim r = 0; r < R; ++r) {
        const Dim t = node(r, 0);
        std::vector<std::uint32_t> slots;
        for (Dim c = 0; c < C; ++c) slots.push_back(next_slot[t]++);
        p.channels.push_back(Channel2{"in" + std::to_string(t), engine(r), Channel2::To::L3, t, 0, 0, slots, 0});
        load_chan[t] = p.channels.size() - 1;
        for (Dim c = 0; c < C; ++c) slot_of[{load_chan[t], TileCoord{"X", r, c}.to_string()}] = slots[c];
    }
    for (const Hop& h : hops) {
        const std::pair<Dim, int> key{h.from, static_cast<int>(h.dir)};
        if (!chan.count(key)) {
            const ProcessId from{role(h.dir), h.from, 0, 0};
            if (h.out)
                p.channels.push_back(Channel2{"st" + std::to_string(h.from), from, Channel2::To::Dma, 0, engine(0).mc, engine(0).engine, {}, 2});
            else
                p.channels.push_back(Channel2{std::string(1, platform::to_string(h.dir)[0]) + std::to_string(h.from) + "to" + std::to_string(h.to),
                                              from, Channel2::To::L3, h.to, 0, 0, {}, 0});
            chan[key] = p.channels.size() - 1;
        }
        Channel2& c = p.channels[chan[key]];
        if (!h.out) {
            const std::uint32_t s = next_slot[h.to]++;
            c.slots.push_back(s);
            slot_of[{chan[key], h.tile.to_string()}] = s;
        }
    }
    // Every node's drain slot.
    std::map<Dim, std::uint32_t> drain_slot;
    for (Dim t = 0; t < L->node_count(); ++t) drain_slot[t] = next_slot[t]++;
    Dim most = 0;
    for (const auto& [t, n] : next_slot) most = std::max<Dim>(most, n);
    if (most > d.cluster->l3_tile.capacity_tiles)
        return fail("a node needs " + std::to_string(most) + " L3 slots for the tiles it receives, and the tile has " +
                    std::to_string(d.cluster->l3_tile.capacity_tiles) + ": the transform's one-hop cost");
    if (cost) *cost = DistributedTransposeCost{longest, total, most};

    // ---- the actions.
    std::map<std::string, std::size_t> proc;
    const auto act = [&](const ProcessId& id, Action2 x) {
        const auto [it, fresh] = proc.emplace(id.to_string(), p.processes.size());
        if (fresh) p.processes.push_back(Process2{id, {}});
        p.processes[it->second].actions.push_back(p.actions.size());
        p.actions.push_back(std::move(x));
    };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
    // Where each tile sits at each node it reaches: the slot it arrived in.
    std::map<std::pair<Dim, std::string>, std::uint32_t> held;
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, 0);
            const TileCoord X{"X", r, c};
            const std::uint32_t s = slot_of.at({load_chan[t], X.to_string()});
            held[{t, X.to_string()}] = s;
            Action2 x;
            x.kind = Action2::Kind::Load;
            x.tile = X;
            x.to = L3s(t, s);
            x.reads = 1;
            x.channel = load_chan[t];
            act(engine(r), x);
        }
    // The hops are in step order, so a tile's arrival is recorded before it leaves again.
    std::map<Dim, std::vector<TileCoord>> stored;             // column -> the Ys row 0 sends out, in order
    for (const Hop& h : hops) {
        const std::size_t ch = chan.at({h.from, static_cast<int>(h.dir)});
        const std::string k = h.tile.to_string();
        const std::uint32_t from = h.tile.operand == "Y" && !held.count({h.from, k}) ? drain_slot[h.from] : held.at({h.from, k});
        Action2 x;
        x.kind = Action2::Kind::Push;
        x.tile = h.tile;
        x.from = L3s(h.from, from);
        x.channel = ch;
        if (!h.out) {
            const std::uint32_t s = slot_of.at({ch, k});
            x.to = L3s(h.to, s);
            x.reads = 1;
            held[{h.to, k}] = s;
        } else {
            stored[L->pos(h.from).c].push_back(h.tile);
        }
        act(ProcessId{role(h.dir), h.from, 0, 0}, x);
    }
    // At each destination: feed X, transpose it into Y, drain Y (its north hops are above).
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const TileCoord X{"X", r, c}, Y{"Y", c, r};
            const Dim dest = arrival.at(X.to_string()).first;
            const std::string sfx = std::to_string(dest);
            p.channels.push_back(Channel2{"fa" + sfx, ProcessId{Role::FeedA, dest, 0, 0}, Channel2::To::L2, dest, 0, 0, {0}, 0});
            const std::size_t fa = p.channels.size() - 1;
            p.channels.push_back(Channel2{"cc" + sfx, ProcessId{Role::Compute, dest, 0, 0}, Channel2::To::L2, dest, 0, 0, {1}, 0});
            const std::size_t cc = p.channels.size() - 1;
            p.channels.push_back(Channel2{"dc" + sfx, ProcessId{Role::Drain, dest, 0, 0}, Channel2::To::L3, dest, 0, 0, {drain_slot[dest]}, 0});
            const std::size_t dc = p.channels.size() - 1;
            Action2 f;
            f.kind = Action2::Kind::Feed;
            f.tile = X;
            f.from = L3s(dest, held.at({dest, X.to_string()}));
            f.to = L2s(dest, 0);
            f.reads = 1;
            f.channel = fa;
            act(ProcessId{Role::FeedA, dest, 0, 0}, f);
            Action2 k;
            k.kind = Action2::Kind::Call;
            k.op = "transpose";
            k.args = {L2s(dest, 0)};
            k.tile = Y;
            k.to = L2s(dest, 1);
            k.reads = 1;
            k.channel = cc;
            act(ProcessId{Role::Compute, dest, 0, 0}, k);
            Action2 dr;
            dr.kind = Action2::Kind::Drain;
            dr.tile = Y;
            dr.from = L2s(dest, 1);
            dr.to = L3s(dest, drain_slot[dest]);
            dr.reads = 1;
            dr.channel = dc;
            act(ProcessId{Role::Drain, dest, 0, 0}, dr);
        }
    for (Dim c = 0; c < C; ++c)
        for (const TileCoord& y : stored[c]) {
            Action2 x;
            x.kind = Action2::Kind::Store;
            x.tile = y;
            x.channel = chan.at({node(0, c), static_cast<int>(platform::Edge::N)});
            act(engine(0), x);
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
