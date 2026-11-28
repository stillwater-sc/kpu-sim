#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_window.hpp
// The P2 generator (#612; docs/plans/distributed-csp-programs.md §3.4): a sliding window -- a 3 x 3 box
// filter, Y = the sum of each element's 3 x 3 neighbourhood, zero past X's edges -- as a level-2
// program on a DMM mesh, every hop one hop. It is the plan's tightest case: every node uses its four
// sends, its three feeds and its drain.
//
// SEPARABLE, SO NO CORNERS. The filter is two passes: H = the 3-wide sums along each row (box3h), then
// Y = the 3-tall sums of H along each column (box3v). The row pass needs a node's west and east
// neighbours' tiles, the column pass its north and south neighbours' H tiles: a halo exchange with
// four neighbours, and none with a corner.
//
// ONE TILE PER NODE (Mt = rows, Nt = cols), for the L2's slots: X, its west and east halos, H, its
// north and south halos, Y. One flow per sending mover, each carrying everything bound for its
// neighbour:
//   bm.E   its own X tile to the east (the east's west halo), then the X tiles it forwards along the row
//   bm.W   its own X tile to the west (the west's east halo)
//   bm.S   its H tile to the south (the south's north halo)
//   bm.N   its H tile to the north (the north's south halo), then its Y, then the Ys from the south;
//          row 0 pushes the Ys out of its edge port to a DMA engine
//   bm.a   X into the compute tile;  bm.b the west halo, then the north;  bm.x the east, then the south
//   bm.c   H (to L3, for the halos) and Y (to L3, out)
// X enters each mesh row at its west edge node, through one DMA engine per row.
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
#include <optional>
#include <string>
#include <vector>

namespace sw::kpu::program::csp::gen {

struct DistributedWindow {
    Dim m = 0, n = 0;                   // X is m x n, one tile per node: (m / rows) x (n / cols)
    std::string type = "int32";
    std::string name = "box3x3";
};

inline DistributedProgram generate_distributed_box3x3(const DistributedWindow& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed box3x3 " + std::to_string(g.m) + "x" + std::to_string(g.n) + " on \"" + d.name + "\": " + why);
    };
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    if (g.m == 0 || g.n == 0 || g.m % R != 0 || g.n % C != 0)
        return fail("X divides into one tile per node, so its rows are a multiple of the mesh's " + std::to_string(R) +
                    " rows and its columns of its " + std::to_string(C) + " columns");
    const Dim tr = g.m / R, tc = g.n / C;
    // L3: the west stream 4, the east halo 2, the north halo 2, the south stream 4, H 1, Y 1.
    const std::uint32_t W0 = 0, E0 = 4, N0 = 6, S0 = 8, H3 = 12, Y3 = 13;
    if (Y3 + 1 > d.cluster->l3_tile.capacity_tiles)
        return fail("a node needs 14 L3 slots, and the tile has " + std::to_string(d.cluster->l3_tile.capacity_tiles));
    // L2: X, its west and east halos, H, its north and south halos, Y.
    const std::uint32_t X2 = 0, Wh = 1, Eh = 2, H2 = 3, Nh = 4, Sh = 5, Y2 = 6;
    if (Y2 + 1 > d.cluster->compute_tile.l2.slots)
        return fail("a compute tile needs 7 L2 slots, and it has " + std::to_string(d.cluster->compute_tile.l2.slots));
    const Dim mcs = d.memory.controllers.value_or(1);
    if (R > d.dma.engines)
        return fail(std::to_string(R) + " mesh rows take a stream in, one DMA engine each, and the device has " + std::to_string(d.dma.engines));

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"X", false, g.m, g.n, tr, tc, g.type, "in"},
                  Operand2{"H", false, g.m, g.n, tr, tc, g.type, "out"},
                  Operand2{"Y", false, g.m, g.n, tr, tc, g.type, "out"}};
    std::map<std::string, std::size_t> proc;
    const auto act = [&](const ProcessId& id, Action2 x) {
        const auto [it, fresh] = proc.emplace(id.to_string(), p.processes.size());
        if (fresh) p.processes.push_back(Process2{id, {}});
        p.processes[it->second].actions.push_back(p.actions.size());
        p.actions.push_back(std::move(x));
    };
    const auto pool = [](std::uint32_t base, std::uint32_t n) {
        std::vector<std::uint32_t> v;
        for (std::uint32_t i = 0; i < n; ++i) v.push_back(base + i);
        return v;
    };
    const auto channel = [&](const std::string& name, const ProcessId& from, Channel2::To to, Dim t, Dim mc, Dim e,
                             std::vector<std::uint32_t> slots, std::uint32_t credits) {
        p.channels.push_back(Channel2{name, from, to, t, mc, e, std::move(slots), credits});
        return p.channels.size() - 1;
    };
    const auto mover = [](Role r, Dim t) { return ProcessId{r, t, 0, 0}; };
    const auto cf = [](Dim t) { return ProcessId{Role::Compute, t, 0, 0}; };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
    const auto engine = [&](Dim r) { return ProcessId{Role::Dma, 0, r % mcs, r / mcs}; };
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };
    const auto push = [&](Role role, Dim t, const TileCoord& tile, SlotRef from, std::optional<SlotRef> to, std::uint32_t reads,
                          std::size_t ch) {
        Action2 x;
        x.kind = Action2::Kind::Push;
        x.tile = tile;
        x.from = from;
        x.to = to;
        x.reads = to ? reads : 0;
        x.channel = ch;
        act(mover(role, t), x);
    };
    const auto feed = [&](Role role, Dim t, const TileCoord& tile, SlotRef from, std::uint32_t l2, std::size_t ch) {
        Action2 x;
        x.kind = Action2::Kind::Feed;
        x.tile = tile;
        x.from = from;
        x.to = L2s(t, l2);
        x.reads = 1;
        x.channel = ch;
        act(mover(role, t), x);
    };

    // Every node's inbound channels.
    std::map<Dim, std::size_t> from_w, from_e, from_n, from_s;
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            from_w[t] = c == 0 ? channel("in" + sfx, engine(r), Channel2::To::L3, t, 0, 0, pool(W0, 4), 0)
                               : channel("e" + std::to_string(node(r, c - 1)) + "to" + sfx, mover(Role::SendE, node(r, c - 1)), Channel2::To::L3, t,
                                         0, 0, pool(W0, 4), 0);
            if (c + 1 < C)
                from_e[t] = channel("w" + std::to_string(node(r, c + 1)) + "to" + sfx, mover(Role::SendW, node(r, c + 1)), Channel2::To::L3, t,
                                    0, 0, pool(E0, 2), 0);
            if (r > 0)
                from_n[t] = channel("s" + std::to_string(node(r - 1, c)) + "to" + sfx, mover(Role::SendS, node(r - 1, c)), Channel2::To::L3, t,
                                    0, 0, pool(N0, 2), 0);
            if (r + 1 < R)
                from_s[t] = channel("n" + std::to_string(node(r + 1, c)) + "to" + sfx, mover(Role::SendN, node(r + 1, c)), Channel2::To::L3, t,
                                    0, 0, pool(S0, 4), 0);
        }
    std::map<Dim, std::size_t> store;                     // row 0's bm.N, out of the edge port
    for (Dim c = 0; c < C; ++c)
        store[node(0, c)] = channel("st" + std::to_string(node(0, c)), mover(Role::SendN, node(0, c)), Channel2::To::Dma, 0, engine(0).mc,
                                    engine(0).engine, {}, 2);

    for (Dim r = 0; r < R; ++r) {
        // The row's X, loaded at its west edge: its own tile first, then the ones it forwards.
        for (Dim c = 0; c < C; ++c) {
            Action2 x;
            x.kind = Action2::Kind::Load;
            x.tile = TileCoord{"X", r, c};
            x.to = L3s(node(r, 0), W0 + c % 4);
            x.reads = c == 0 ? 1 + (C > 1 ? 1 : 0) : 1;           // its own: the feed and the east halo
            x.channel = from_w[node(r, 0)];
            act(engine(r), x);
        }
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            const bool w = c > 0, e = c + 1 < C, n = r > 0, s = r + 1 < R;
            // The west stream: [the west halo,] its own X, then the X tiles it forwards east.
            const std::uint32_t own_at = W0 + (w ? 1 : 0);
            const std::size_t fa = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, {X2}, 0);
            // The halo feeds exist only where there are halos (a corner has one or neither): D3.
            const std::size_t fb = w || n ? channel("fb" + sfx, mover(Role::FeedB, t), Channel2::To::L2, t, 0, 0, {Wh, Nh}, 0) : kNoChannel;
            const std::size_t fx = e || s ? channel("fx" + sfx, mover(Role::FeedX, t), Channel2::To::L2, t, 0, 0, {Eh, Sh}, 0) : kNoChannel;
            const std::size_t cc = channel("cc" + sfx, cf(t), Channel2::To::L2, t, 0, 0, {H2, Y2}, 0);
            const std::size_t dc = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, {H3, Y3}, 0);
            const TileCoord X{"X", r, c}, H{"H", r, c}, Y{"Y", r, c};
            feed(Role::FeedA, t, X, L3s(t, own_at), X2, fa);
            if (e) {                                              // own X east first, then the forwards
                const std::uint32_t east_reads = 1;               // the east's feed of its west halo
                push(Role::SendE, t, X, L3s(t, own_at), L3s(node(r, c + 1), W0 + 0), east_reads, from_w[node(r, c + 1)]);
                for (Dim f = c + 1; f < C; ++f) {
                    const std::uint32_t k_here = (w ? 1 : 0) + 1 + (f - c - 1);       // its index in this node's stream
                    const std::uint32_t k_there = 1 + (f - c - 1);                   // in the east's: after the halo
                    const std::uint32_t reads = f == c + 1 ? 1 + 1 + (c + 2 < C ? 1 : 0) : 1;  // the east's own X, or a forward
                    push(Role::SendE, t, TileCoord{"X", r, f}, L3s(t, W0 + k_here % 4), L3s(node(r, c + 1), W0 + k_there % 4),
                         reads, from_w[node(r, c + 1)]);
                }
            }
            if (w) {
                push(Role::SendW, t, X, L3s(t, own_at), L3s(node(r, c - 1), E0 + 0), 1, from_e[node(r, c - 1)]);
                feed(Role::FeedB, t, TileCoord{"X", r, c - 1}, L3s(t, W0 + 0), Wh, fb);
            }
            if (e) feed(Role::FeedX, t, TileCoord{"X", r, c + 1}, L3s(t, E0 + 0), Eh, fx);
            // The row pass.
            {
                Action2 x;
                x.kind = Action2::Kind::Call;
                x.op = std::string("box3h") + (w || e ? "_" : "") + (w ? "w" : "") + (e ? "e" : "");
                if (w) x.args.push_back(L2s(t, Wh));
                x.args.push_back(L2s(t, X2));
                if (e) x.args.push_back(L2s(t, Eh));
                x.tile = H;
                x.to = L2s(t, H2);
                x.reads = 1 + (n || s ? 1 : 0);                   // the column pass, and its drain for the halos
                x.channel = cc;
                act(cf(t), x);
            }
            if (n || s) {
                Action2 x;
                x.kind = Action2::Kind::Drain;
                x.tile = H;
                x.from = L2s(t, H2);
                x.to = L3s(t, H3);
                x.reads = (n ? 1 : 0) + (s ? 1 : 0);
                x.channel = dc;
                act(mover(Role::Drain, t), x);
            }
            if (s) push(Role::SendS, t, H, L3s(t, H3), L3s(node(r + 1, c), N0 + 0), 1, from_n[node(r + 1, c)]);
            if (n) feed(Role::FeedB, t, TileCoord{"H", r - 1, c}, L3s(t, N0 + 0), Nh, fb);
            if (s) feed(Role::FeedX, t, TileCoord{"H", r + 1, c}, L3s(t, S0 + 0), Sh, fx);
            // The column pass.
            {
                Action2 x;
                x.kind = Action2::Kind::Call;
                x.op = std::string("box3v") + (n || s ? "_" : "") + (n ? "n" : "") + (s ? "s" : "");
                if (n) x.args.push_back(L2s(t, Nh));
                x.args.push_back(L2s(t, H2));
                if (s) x.args.push_back(L2s(t, Sh));
                x.tile = Y;
                x.to = L2s(t, Y2);
                x.reads = 1;
                x.channel = cc;
                act(cf(t), x);
            }
            {
                Action2 x;
                x.kind = Action2::Kind::Drain;
                x.tile = Y;
                x.from = L2s(t, Y2);
                x.to = L3s(t, Y3);
                x.reads = 1;
                x.channel = dc;
                act(mover(Role::Drain, t), x);
            }
            // North: [its H halo,] its Y, then the Ys from the south; row 0 out of its edge port.
            const std::size_t ch = n ? from_s[node(r - 1, c)] : store[t];
            std::uint32_t sent = 0;
            const auto north = [&](const TileCoord& tile, SlotRef from) {
                push(Role::SendN, t, tile, from, n ? std::optional<SlotRef>(L3s(node(r - 1, c), S0 + sent % 4)) : std::nullopt, 1, ch);
                ++sent;
            };
            if (n) north(H, L3s(t, H3));
            north(Y, L3s(t, Y3));
            for (Dim below = r + 1; below < R; ++below)           // the south stream: [H halo,] Y(r+1), Y(r+2), ...
                north(TileCoord{"Y", below, c}, L3s(t, S0 + (1 + below - (r + 1)) % 4));
        }
    }
    for (Dim c = 0; c < C; ++c)
        for (Dim r = 0; r < R; ++r) {
            Action2 x;
            x.kind = Action2::Kind::Store;
            x.tile = TileCoord{"Y", r, c};
            x.channel = store[node(0, c)];
            act(engine(0), x);
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
