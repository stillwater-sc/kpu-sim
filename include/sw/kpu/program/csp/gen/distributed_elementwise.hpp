#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_elementwise.hpp
// The P6 generator (#610; docs/plans/distributed-csp-programs.md §3.4): two operands streamed aligned
// into every compute tile, Z = X + Y -- a residual add, a gating product's operands -- as a level-2
// program on a DMM mesh, every hop one hop.
//
// THE DISTRIBUTION. X, Y and Z share one block distribution, as P1's C: node (r, c) owns an a x b
// block of tiles. So the operands are aligned by construction, and no tile moves between nodes for
// its partner. (Operands distributed differently need a realigning shift first: a layout transform,
// P7's.)
//
// THE STREAMS, every hop one hop:
//   in    each mesh row's west edge node loads the row's tiles through its DMA engine, column by
//         column and, within a node's block, X[i,j] then Y[i,j]: the two operands rate-matched, a
//         pair at a time. Every node keeps its own pairs and passes the rest east (bm.E), each tile
//         landing in the next node's L3 on the way.
//   add   bm.a feeds X, bm.b feeds Y into the compute tile's L2; `call add` writes Z into L2.
//   out   as P1's C: Z drains to L3 (bm.c) and moves north one hop at a time (bm.N); row 0 pushes it
//         out of its edge port to a DMA engine that stores it.
// One DMA engine per mesh row loads (D1: one flow each); row 0's stores go to the first.
//
// THE SLOTS, per node. L3: the row stream's pool 4, Z's drains 2, Z in transit from the south 2 (rows
// above the last). L2: X 2, Y 2, Z 2, each pair's slot q % 2.
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

struct DistributedElementwise {
    Dim m = 0, n = 0, tile = 0;
    std::string type = "int32";
    std::string name = "add";
};

inline DistributedProgram generate_distributed_add(const DistributedElementwise& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed add " + std::to_string(g.m) + "x" + std::to_string(g.n) + " on \"" + d.name +
                                  "\": " + why);
    };
    if (g.m == 0 || g.n == 0 || g.tile == 0) return fail("M, N and the tile must be non-zero");
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    const auto tiles = [&](Dim x) { return x / g.tile + (x % g.tile != 0); };
    const Dim Mt = tiles(g.m), Nt = tiles(g.n);
    if (Mt % R != 0 || Nt % C != 0)
        return fail("the " + std::to_string(Mt) + "x" + std::to_string(Nt) + " tiles do not divide over the " + std::to_string(R) +
                    "x" + std::to_string(C) + " mesh");
    const Dim a = Mt / R, b = Nt / C, blk = a * b;
    const std::uint32_t pin = 4, pc = 2, pt = 2;
    const std::uint32_t I0 = 0, C0 = pin, T0 = C0 + pc;
    if (T0 + (R > 1 ? pt : 0) > d.cluster->l3_tile.capacity_tiles)
        return fail("a node needs " + std::to_string(T0 + (R > 1 ? pt : 0)) + " L3 slots, and the tile has " +
                    std::to_string(d.cluster->l3_tile.capacity_tiles));
    if (6 > d.cluster->compute_tile.l2.slots)
        return fail("a compute tile needs 6 L2 slots, and it has " + std::to_string(d.cluster->compute_tile.l2.slots));
    const Dim mcs = d.memory.controllers.value_or(1);
    if (R > d.dma.engines)
        return fail(std::to_string(R) + " mesh rows take a stream in, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"X", false, g.m, g.n, g.tile, g.tile, g.type, "in"},
                  Operand2{"Y", false, g.m, g.n, g.tile, g.tile, g.type, "in"},
                  Operand2{"Z", false, g.m, g.n, g.tile, g.tile, g.type, "out"}};
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
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
    const auto engine = [&](Dim r) { return ProcessId{Role::Dma, 0, r % mcs, r / mcs}; };
    // The tile of the row stream's n-th element, for the node at column c0 onward.
    struct Item { TileCoord tile; Dim col; };
    const auto row_stream = [&](Dim r, Dim from_col) {
        std::vector<Item> v;
        for (Dim c = from_col; c < C; ++c)
            for (Dim q = 0; q < blk; ++q) {
                const Dim i = r * a + q / b, j = c * b + q % b;
                v.push_back({TileCoord{"X", i, j}, c});
                v.push_back({TileCoord{"Y", i, j}, c});
            }
        return v;
    };

    for (Dim r = 0; r < R; ++r) {
        // Each node's inbound channel: from the row's DMA engine at column 0, else from the west.
        std::vector<std::size_t> in(C);
        for (Dim c = 0; c < C; ++c) {
            const Dim t = L->node(r, c);
            in[c] = c == 0 ? channel("in" + std::to_string(t), engine(r), Channel2::To::L3, t, 0, 0, pool(I0, pin), 0)
                           : channel("e" + std::to_string(L->node(r, c - 1)) + "to" + std::to_string(t), mover(Role::SendE, L->node(r, c - 1)),
                                     Channel2::To::L3, t, 0, 0, pool(I0, pin), 0);
        }
        {
            const std::vector<Item> s = row_stream(r, 0);
            for (std::size_t k = 0; k < s.size(); ++k) {
                Action2 x;
                x.kind = Action2::Kind::Load;
                x.tile = s[k].tile;
                x.to = L3s(L->node(r, 0), I0 + static_cast<std::uint32_t>(k % pin));
                x.reads = 1;                                      // its feed, or its push east
                x.channel = in[0];
                act(engine(r), x);
            }
        }
        for (Dim c = 0; c < C; ++c) {
            const Dim t = L->node(r, c);
            const std::string sfx = std::to_string(t);
            const std::size_t fa = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, pool(0, 2), 0);
            const std::size_t fb = channel("fb" + sfx, mover(Role::FeedB, t), Channel2::To::L2, t, 0, 0, pool(2, 2), 0);
            const std::size_t cc = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0, pool(4, 2), 0);
            const std::size_t dc = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, pool(C0, pc), 0);
            const std::vector<Item> s = row_stream(r, c);
            std::uint32_t own = 0, fwd = 0;
            for (std::size_t k = 0; k < s.size(); ++k) {
                const SlotRef at = L3s(t, I0 + static_cast<std::uint32_t>(k % pin));
                Action2 x;
                if (s[k].col == c) {                              // ours: into the compute tile
                    const bool is_x = s[k].tile.operand == "X";
                    const Dim q = own++ / 2;
                    x.kind = Action2::Kind::Feed;
                    x.tile = s[k].tile;
                    x.from = at;
                    x.to = L2s(t, (is_x ? 0 : 2) + q % 2);
                    x.reads = 1;
                    x.channel = is_x ? fa : fb;
                    act(mover(is_x ? Role::FeedA : Role::FeedB, t), x);
                } else {                                          // further east
                    x.kind = Action2::Kind::Push;
                    x.tile = s[k].tile;
                    x.from = at;
                    x.to = L3s(L->node(r, c + 1), I0 + fwd++ % pin);
                    x.reads = 1;
                    x.channel = in[c + 1];
                    act(mover(Role::SendE, t), x);
                }
            }
            for (Dim q = 0; q < blk; ++q) {
                const Dim i = r * a + q / b, j = c * b + q % b;
                Action2 x;
                x.kind = Action2::Kind::Call;
                x.op = "add";
                x.args = {L2s(t, q % 2), L2s(t, 2 + q % 2)};
                x.tile = TileCoord{"Z", i, j};
                x.to = L2s(t, 4 + q % 2);
                x.reads = 1;
                x.channel = cc;
                act(ProcessId{Role::Compute, t, 0, 0}, x);
                Action2 dr;
                dr.kind = Action2::Kind::Drain;
                dr.tile = TileCoord{"Z", i, j};
                dr.from = L2s(t, 4 + q % 2);
                dr.to = L3s(t, C0 + q % pc);
                dr.reads = 1;
                dr.channel = dc;
                act(mover(Role::Drain, t), dr);
            }
        }
    }

    // Z north, as P1's C: each node its own tiles, then those from below; row 0 to the DMA engine.
    for (Dim c = 0; c < C; ++c)
        for (Dim r = 0; r < R; ++r) {
            const Dim t = L->node(r, c);
            const std::size_t ch = r == 0 ? channel("st" + std::to_string(t), mover(Role::SendN, t), Channel2::To::Dma, 0,
                                                    engine(0).mc, engine(0).engine, {}, 2)
                                          : channel("z" + std::to_string(t) + "to" + std::to_string(L->node(r - 1, c)),
                                                    mover(Role::SendN, t), Channel2::To::L3, L->node(r - 1, c), 0, 0, pool(T0, pt), 0);
            std::uint32_t sent = 0, got = 0;
            const auto push = [&](const TileCoord& tile, const SlotRef& from) {
                Action2 x;
                x.kind = Action2::Kind::Push;
                x.tile = tile;
                x.from = from;
                if (r > 0) {
                    x.to = L3s(L->node(r - 1, c), T0 + sent % pt);
                    x.reads = 1;
                }
                x.channel = ch;
                act(mover(Role::SendN, t), x);
                ++sent;
            };
            for (Dim q = 0; q < blk; ++q) push(TileCoord{"Z", r * a + q / b, c * b + q % b}, L3s(t, C0 + q % pc));
            for (Dim below = r + 1; below < R; ++below)
                for (Dim q = 0; q < blk; ++q) push(TileCoord{"Z", below * a + q / b, c * b + q % b}, L3s(t, T0 + got++ % pt));
            if (r == 0)
                for (Dim below = 0; below < R; ++below)
                    for (Dim q = 0; q < blk; ++q) {
                        Action2 x;
                        x.kind = Action2::Kind::Store;
                        x.tile = TileCoord{"Z", below * a + q / b, c * b + q % b};
                        x.channel = ch;
                        act(engine(0), x);
                    }
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
