#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_gemm.hpp
// The P1 generator (#608; docs/plans/distributed-csp-programs.md §3.4): C = A . B as a level-2
// program on a DMM mesh, edge-streamed and systolic, every hop one hop.
//
// THE DISTRIBUTION. C is stationary and block-distributed: node (r, c) owns an a x b block of C's
// tiles, rows [r*a, (r+1)*a), columns [c*b, (c+1)*b). C's tile grid must divide over the mesh.
//
// THE STREAMS, k outer:
//   A   enters each mesh row at its west edge node through a DMA engine, and moves east one hop at
//       a time: every node feeds it to its compute tile (bm.a) and pushes it on (bm.E). Node (r, c)
//       sees A[i, k] for its rows i, in the order k, then i.
//   B   enters each mesh column at its north edge node, and moves south the same way (bm.b, bm.S).
//   C   accumulates in the compute tile's L2 over k (a result expecting Kt - 1 merges), drains to L3
//       (bm.c), and moves north one hop at a time (bm.N), each node sending its own tiles and then
//       those from below; row 0 pushes them out of its north edge port to its DMA engine, which
//       stores them.
// Every edge node that takes a stream in (row 0 and column 0) has one DMA engine, on one channel
// (D1: one CSP per flow): the corner's carries A and B. A program needs that many engines.
//
// THE SLOTS, per node. L3: A in 2a, B in 2b, C's drains in 2, C in transit from the south in 2 (rows
// above the last). L2: a for A, b for B, a*b for the accumulators. A tile in a pool goes to slot
// n % pool, for the n-th tile of its stream; the reads it serves are its consumers: a feed, plus the
// push onward when there is a node beyond.
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp::gen {

struct DistributedGemm {
    Dim m = 0, n = 0, k = 0, tile = 0;
    std::string ab_type = "int8", c_type = "int32";
    std::string name = "gemm";
};

class DistributedGenError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

inline DistributedProgram generate_distributed_gemm(const DistributedGemm& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed gemm " + std::to_string(g.m) + "x" + std::to_string(g.n) + "x" +
                                  std::to_string(g.k) + " on \"" + d.name + "\": " + why);
    };
    if (g.m == 0 || g.n == 0 || g.k == 0 || g.tile == 0) return fail("M, N, K and the tile must be non-zero");
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    const auto tiles = [&](Dim x) { return x / g.tile + (x % g.tile != 0); };
    const Dim Mt = tiles(g.m), Nt = tiles(g.n), Kt = tiles(g.k);
    if (Mt % R != 0 || Nt % C != 0)
        return fail("C's " + std::to_string(Mt) + "x" + std::to_string(Nt) + " tiles do not divide over the " + std::to_string(R) +
                    "x" + std::to_string(C) + " mesh");
    const Dim a = Mt / R, b = Nt / C;
    // Pools, and the room the cluster gives them.
    const std::uint32_t pa = 2 * a, pb = 2 * b, pc = 2, pt = 2;
    const std::uint32_t A0 = 0, B0 = pa, C0 = pa + pb, T0 = C0 + pc;
    const Dim l3 = d.cluster->l3_tile.capacity_tiles, l2 = d.cluster->compute_tile.l2.slots;
    if (T0 + (R > 1 ? pt : 0) > l3)
        return fail("a node needs " + std::to_string(T0 + (R > 1 ? pt : 0)) + " L3 slots, and the tile has " + std::to_string(l3));
    if (a + b + a * b > l2)
        return fail("a compute tile needs " + std::to_string(a + b + a * b) + " L2 slots, and it has " + std::to_string(l2));
    const Dim mcs = d.memory.controllers.value_or(1), per_mc = d.dma.engines / mcs;
    const Dim inputs = R + C - 1;
    if (inputs > d.dma.engines)
        return fail(std::to_string(inputs) + " edge nodes take a stream in, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"A", false, g.m, g.k, g.tile, g.tile, g.ab_type, "in"},
                  Operand2{"B", false, g.k, g.n, g.tile, g.tile, g.ab_type, "in"},
                  Operand2{"C", false, g.m, g.n, g.tile, g.tile, g.c_type, "out"}};
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
    const auto pool = [](std::uint32_t base, std::uint32_t n) {
        std::vector<std::uint32_t> v;
        for (std::uint32_t i = 0; i < n; ++i) v.push_back(base + i);
        return v;
    };
    std::map<std::string, std::size_t> proc;                  // process -> index
    const auto process = [&](const ProcessId& id) -> Process2& {
        const auto [it, fresh] = proc.emplace(id.to_string(), p.processes.size());
        if (fresh) p.processes.push_back(Process2{id, {}});
        return p.processes[it->second];
    };
    const auto act = [&](const ProcessId& id, Action2 x) {
        process(id).actions.push_back(p.actions.size());
        p.actions.push_back(std::move(x));
    };
    const auto channel = [&](const std::string& name, const ProcessId& from, Channel2::To to, Dim t, Dim mc, Dim e,
                             std::vector<std::uint32_t> slots, std::uint32_t credits) {
        p.channels.push_back(Channel2{name, from, to, t, mc, e, std::move(slots), credits});
        return p.channels.size() - 1;
    };
    const auto mover = [&](Role r, Dim t) { return ProcessId{r, t, 0, 0}; };

    // The DMA engines, one per input edge node, spread over the controllers.
    std::map<Dim, ProcessId> engine_of;
    {
        Dim e = 0;
        for (Dim r = 0; r < R; ++r)
            for (Dim c = 0; c < C; ++c)
                if (r == 0 || c == 0) {
                    engine_of[node(r, c)] = ProcessId{Role::Dma, 0, e % mcs, e / mcs};
                    ++e;
                }
        (void)per_mc;
    }
    // The inbound channels: A from the west (the DMA engine at column 0), B from the north (the DMA
    // engine at row 0); one DMA channel per input node, owning its A and B pools.
    std::map<Dim, std::size_t> a_in, b_in;
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            if (r == 0 || c == 0) {
                std::vector<std::uint32_t> slots;
                if (c == 0) slots = pool(A0, pa);
                if (r == 0) for (std::uint32_t s : pool(B0, pb)) slots.push_back(s);
                const std::size_t ch = channel("in" + sfx, engine_of.at(t), Channel2::To::L3, t, 0, 0, slots, 0);
                if (c == 0) a_in[t] = ch;
                if (r == 0) b_in[t] = ch;
            }
            if (c > 0) a_in[t] = channel("a" + std::to_string(node(r, c - 1)) + "to" + sfx, mover(Role::SendE, node(r, c - 1)),
                                         Channel2::To::L3, t, 0, 0, pool(A0, pa), 0);
            if (r > 0) b_in[t] = channel("b" + std::to_string(node(r - 1, c)) + "to" + sfx, mover(Role::SendS, node(r - 1, c)),
                                         Channel2::To::L3, t, 0, 0, pool(B0, pb), 0);
        }

    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            const std::uint32_t a_reads = 1 + (c + 1 < C ? 1 : 0), b_reads = 1 + (r + 1 < R ? 1 : 0);
            // Loads at the edge, k outer: this k's A tiles, then its B tiles.
            if (r == 0 || c == 0)
                for (Dim k = 0; k < Kt; ++k) {
                    if (c == 0)
                        for (Dim ii = 0; ii < a; ++ii) {
                            Action2 x;
                            x.kind = Action2::Kind::Load;
                            x.tile = TileCoord{"A", r * a + ii, k};
                            x.to = L3s(t, A0 + (k * a + ii) % pa);
                            x.reads = a_reads;
                            x.channel = a_in.at(t);
                            act(engine_of.at(t), x);
                        }
                    if (r == 0)
                        for (Dim jj = 0; jj < b; ++jj) {
                            Action2 x;
                            x.kind = Action2::Kind::Load;
                            x.tile = TileCoord{"B", k, c * b + jj};
                            x.to = L3s(t, B0 + (k * b + jj) % pb);
                            x.reads = b_reads;
                            x.channel = b_in.at(t);
                            act(engine_of.at(t), x);
                        }
                }
            // A east and B south, one hop, into the neighbour's pool.
            if (c + 1 < C) {
                const Dim east = node(r, c + 1);
                const std::uint32_t reads = 1 + (c + 2 < C ? 1 : 0);
                for (Dim k = 0; k < Kt; ++k)
                    for (Dim ii = 0; ii < a; ++ii) {
                        const std::uint32_t s = A0 + (k * a + ii) % pa;
                        Action2 x;
                        x.kind = Action2::Kind::Push;
                        x.tile = TileCoord{"A", r * a + ii, k};
                        x.from = L3s(t, s);
                        x.to = L3s(east, s);
                        x.reads = reads;
                        x.channel = a_in.at(east);
                        act(mover(Role::SendE, t), x);
                    }
            }
            if (r + 1 < R) {
                const Dim south = node(r + 1, c);
                const std::uint32_t reads = 1 + (r + 2 < R ? 1 : 0);
                for (Dim k = 0; k < Kt; ++k)
                    for (Dim jj = 0; jj < b; ++jj) {
                        const std::uint32_t s = B0 + (k * b + jj) % pb;
                        Action2 x;
                        x.kind = Action2::Kind::Push;
                        x.tile = TileCoord{"B", k, c * b + jj};
                        x.from = L3s(t, s);
                        x.to = L3s(south, s);
                        x.reads = reads;
                        x.channel = b_in.at(south);
                        act(mover(Role::SendS, t), x);
                    }
            }
            // Feeds into the compute tile: A to L2 0..a-1 (read by b calls), B to a..a+b-1 (by a calls).
            const std::size_t fa = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, pool(0, a), 0);
            const std::size_t fb = channel("fb" + sfx, mover(Role::FeedB, t), Channel2::To::L2, t, 0, 0, pool(a, b), 0);
            const std::size_t cc = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0,
                                           pool(a + b, a * b), 0);
            const std::size_t dc = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, pool(C0, pc), 0);
            for (Dim k = 0; k < Kt; ++k) {
                for (Dim ii = 0; ii < a; ++ii) {
                    Action2 x;
                    x.kind = Action2::Kind::Feed;
                    x.tile = TileCoord{"A", r * a + ii, k};
                    x.from = L3s(t, A0 + (k * a + ii) % pa);
                    x.to = L2s(t, ii);
                    x.reads = b;
                    x.channel = fa;
                    act(mover(Role::FeedA, t), x);
                }
                for (Dim jj = 0; jj < b; ++jj) {
                    Action2 x;
                    x.kind = Action2::Kind::Feed;
                    x.tile = TileCoord{"B", k, c * b + jj};
                    x.from = L3s(t, B0 + (k * b + jj) % pb);
                    x.to = L2s(t, a + jj);
                    x.reads = a;
                    x.channel = fb;
                    act(mover(Role::FeedB, t), x);
                }
                for (Dim ii = 0; ii < a; ++ii)
                    for (Dim jj = 0; jj < b; ++jj) {
                        Action2 x;
                        x.kind = Action2::Kind::Call;
                        x.op = "gemm";
                        x.args = {L2s(t, ii), L2s(t, a + jj)};
                        x.to = L2s(t, a + b + ii * b + jj);
                        if (k == 0) {
                            x.tile = TileCoord{"C", r * a + ii, c * b + jj};
                            x.merges = Kt - 1;
                            x.reads = 1;
                            x.channel = cc;
                        } else {
                            x.accumulate = true;
                        }
                        act(ProcessId{Role::Compute, t, 0, 0}, x);
                    }
            }
            for (Dim q = 0; q < a * b; ++q) {
                Action2 x;
                x.kind = Action2::Kind::Drain;
                x.tile = TileCoord{"C", r * a + q / b, c * b + q % b};
                x.from = L2s(t, a + b + q);
                x.to = L3s(t, C0 + q % pc);
                x.reads = 1;
                x.channel = dc;
                act(mover(Role::Drain, t), x);
            }
        }

    // C north: each node sends its own tiles, then those from below; row 0 out of its edge port.
    for (Dim c = 0; c < C; ++c)
        for (Dim r = 0; r < R; ++r) {
            const Dim t = node(r, c);
            std::size_t ch;
            if (r == 0) ch = channel("st" + std::to_string(t), mover(Role::SendN, t), Channel2::To::Dma, 0, engine_of.at(t).mc,
                                     engine_of.at(t).engine, {}, 2);
            else ch = channel("c" + std::to_string(t) + "to" + std::to_string(node(r - 1, c)), mover(Role::SendN, t),
                              Channel2::To::L3, node(r - 1, c), 0, 0, pool(T0, pt), 0);
            std::uint32_t sent = 0;
            const auto push = [&](const TileCoord& tile, const SlotRef& from) {
                Action2 x;
                x.kind = Action2::Kind::Push;
                x.tile = tile;
                x.from = from;
                if (r > 0) {
                    x.to = L3s(node(r - 1, c), T0 + sent % pt);
                    x.reads = 1;
                }
                x.channel = ch;
                act(mover(Role::SendN, t), x);
                ++sent;
            };
            for (Dim q = 0; q < a * b; ++q) push(TileCoord{"C", r * a + q / b, c * b + q % b}, L3s(t, C0 + q % pc));
            std::uint32_t got = 0;                            // what arrived from below, in its order
            for (Dim below = r + 1; below < R; ++below)
                for (Dim q = 0; q < a * b; ++q) push(TileCoord{"C", below * a + q / b, c * b + q % b}, L3s(t, T0 + got++ % pt));
            if (r == 0)
                for (Dim below = 0; below < R; ++below)
                    for (Dim q = 0; q < a * b; ++q) {
                        Action2 x;
                        x.kind = Action2::Kind::Store;
                        x.tile = TileCoord{"C", below * a + q / b, c * b + q % b};
                        x.channel = ch;
                        act(engine_of.at(t), x);
                    }
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
