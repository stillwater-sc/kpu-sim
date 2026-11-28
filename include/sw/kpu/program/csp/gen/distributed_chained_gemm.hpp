#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_chained_gemm.hpp
// The P8 generator (#614; docs/plans/distributed-csp-programs.md §3.4): chained-GEMM fusion,
// E = (A . B) . D, as attention's (Q . K^T) . V is without its softmax. A level-2 program on a DMM mesh,
// every hop one hop. T = A . B is the fused intermediate: it is computed into the compute tiles' L2 and
// read there by the second GEMM, never drained, never stored (operand T is `temp`).
//
// THE DISTRIBUTION. T is stationary and block-distributed, as P1's C: node (r, c) owns an a x b block of
// T's tiles, rows [r*a, (r+1)*a), columns [c*b, (c+1)*b). E's tile row i belongs to mesh row r.
//
// THE FLOWS, one per mover:
//   A     enters each mesh row at its west edge node and moves east (bm.E), as P1's; fed by bm.a.
//   B, D  enter each mesh column at its north edge node and move south (bm.S) on one flow, B's tiles
//         then D's: the second GEMM's operand follows the first's down the column, a pipeline, not a
//         ring (§3.4). bm.b feeds both. Node (r, c) needs D's tile rows [c*b, (c+1)*b), all of them.
//   T     accumulates over k in L2 (P1), then stays: each tile is read by E's Pt tile columns.
//   E     each node's partial, E_c[i, p] = the sum over its jj of T[i, jj] . D[jj, p], is merged with the
//         partial arriving from the east (bm.x feeds it; `sum ... acc`), drained (bm.c) and pushed
//         west (bm.W): a chain along the row, P3's. The west end pushes the total out of its west edge
//         port to its DMA engine, which stores it.
//
// THE SLOTS, per node. L3: A in 2a, B and D in 2b, the partials from the east 2, the partials' drains
// 2. L2: a for A, b for B and D, a*b for T, a for E's partials, 1 for the partial from the east.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/csp/distributed.hpp>
#include <sw/kpu/program/csp/gen/distributed_gemm.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/platform/mesh_layout.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::csp::gen {

struct DistributedChainedGemm {
    Dim m = 0, k = 0, n = 0, p = 0, tile = 0;  // A m x k, B k x n, D n x p; T = A . B is m x n, E m x p
    std::string type = "fp32";                 // every operand's: the second GEMM reads T, so T's number system is D's
    std::string name = "chained_gemm";
};

inline DistributedProgram generate_distributed_chained_gemm(const DistributedChainedGemm& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed chained gemm " + std::to_string(g.m) + "x" + std::to_string(g.k) + "x" +
                                  std::to_string(g.n) + "x" + std::to_string(g.p) + " on \"" + d.name + "\": " + why);
    };
    if (g.m == 0 || g.k == 0 || g.n == 0 || g.p == 0 || g.tile == 0) return fail("M, K, N, P and the tile must be non-zero");
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols();
    const auto tiles = [&](Dim x) { return x / g.tile + (x % g.tile != 0); };
    const Dim Mt = tiles(g.m), Kt = tiles(g.k), Nt = tiles(g.n), Pt = tiles(g.p);
    if (Mt % R != 0 || Nt % C != 0)
        return fail("T's " + std::to_string(Mt) + "x" + std::to_string(Nt) + " tiles do not divide over the " + std::to_string(R) +
                    "x" + std::to_string(C) + " mesh");
    const Dim a = Mt / R, b = Nt / C;
    const std::uint32_t pa = 2 * a, pb = 2 * b, pe = 2, pp = 2;
    const std::uint32_t A0 = 0, B0 = pa, E0 = pa + pb, P0 = E0 + pe;
    const Dim l3 = d.cluster->l3_tile.capacity_tiles, l2 = d.cluster->compute_tile.l2.slots;
    if (P0 + pp > l3) return fail("a node needs " + std::to_string(P0 + pp) + " L3 slots, and the tile has " + std::to_string(l3));
    const std::uint32_t A2 = 0, B2 = a, T2 = a + b, Q2 = T2 + a * b, In2 = Q2 + a;
    if (In2 + 1 > l2)
        return fail("a compute tile needs " + std::to_string(In2 + 1) + " L2 slots (A, B and D, T resident, E's partials), and it has " +
                    std::to_string(l2));
    const Dim mcs = d.memory.controllers.value_or(1);
    const Dim inputs = R + C - 1;
    if (inputs > d.dma.engines)
        return fail(std::to_string(inputs) + " edge nodes take a stream in, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"A", false, g.m, g.k, g.tile, g.tile, g.type, "in"},
                  Operand2{"B", false, g.k, g.n, g.tile, g.tile, g.type, "in"},
                  Operand2{"D", false, g.n, g.p, g.tile, g.tile, g.type, "in"},
                  Operand2{"T", false, g.m, g.n, g.tile, g.tile, g.type, "temp"},
                  Operand2{"E", false, g.m, g.p, g.tile, g.tile, g.type, "out"}};
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
    const auto pool = [](std::uint32_t base, std::uint32_t n) {
        std::vector<std::uint32_t> v;
        for (std::uint32_t i = 0; i < n; ++i) v.push_back(base + i);
        return v;
    };
    std::map<std::string, std::size_t> proc;
    const auto act = [&](const ProcessId& id, Action2 x) {
        const auto [it, fresh] = proc.emplace(id.to_string(), p.processes.size());
        if (fresh) p.processes.push_back(Process2{id, {}});
        p.processes[it->second].actions.push_back(p.actions.size());
        p.actions.push_back(std::move(x));
    };
    const auto channel = [&](const std::string& name, const ProcessId& from, Channel2::To to, Dim t, Dim mc, Dim e,
                             std::vector<std::uint32_t> slots, std::uint32_t credits) {
        p.channels.push_back(Channel2{name, from, to, t, mc, e, std::move(slots), credits});
        return p.channels.size() - 1;
    };
    const auto mover = [](Role r, Dim t) { return ProcessId{r, t, 0, 0}; };

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
    }
    // The inbound channels, as P1's: A from the west, B and D from the north.
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
    // Column c's south flow, in its order: B[k, c*b + jj] (k, then jj), then D[c*b + jj, q] (q, then jj).
    const auto column_flow = [&](Dim c) {
        std::vector<TileCoord> v;
        for (Dim k = 0; k < Kt; ++k)
            for (Dim jj = 0; jj < b; ++jj) v.push_back(TileCoord{"B", k, c * b + jj});
        for (Dim q = 0; q < Pt; ++q)
            for (Dim jj = 0; jj < b; ++jj) v.push_back(TileCoord{"D", c * b + jj, q});
        return v;
    };

    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            const bool east = c + 1 < C;
            const std::vector<TileCoord> south_flow = column_flow(c);
            const std::uint32_t a_reads = 1 + (east ? 1 : 0), b_reads = 1 + (r + 1 < R ? 1 : 0);
            // Loads at the edge: k outer, this k's A tiles then its B tiles (P1's order); then D.
            if (r == 0 || c == 0) {
                const auto load = [&](const TileCoord& tile, std::uint32_t slot, std::uint32_t reads, std::size_t ch) {
                    Action2 x;
                    x.kind = Action2::Kind::Load;
                    x.tile = tile;
                    x.to = L3s(t, slot);
                    x.reads = reads;
                    x.channel = ch;
                    act(engine_of.at(t), x);
                };
                for (Dim k = 0; k < Kt; ++k) {
                    if (c == 0)
                        for (Dim ii = 0; ii < a; ++ii) load(TileCoord{"A", r * a + ii, k}, A0 + (k * a + ii) % pa, a_reads, a_in.at(t));
                    if (r == 0)
                        for (Dim jj = 0; jj < b; ++jj) load(south_flow[k * b + jj], B0 + (k * b + jj) % pb, b_reads, b_in.at(t));
                }
                if (r == 0)
                    for (std::size_t n = Kt * b; n < south_flow.size(); ++n)
                        load(south_flow[n], B0 + static_cast<std::uint32_t>(n % pb), b_reads, b_in.at(t));
            }
            // A east, and B then D south, one hop, into the neighbour's pool.
            if (east) {
                const Dim to = node(r, c + 1);
                const std::uint32_t reads = 1 + (c + 2 < C ? 1 : 0);
                for (Dim k = 0; k < Kt; ++k)
                    for (Dim ii = 0; ii < a; ++ii) {
                        const std::uint32_t s = A0 + (k * a + ii) % pa;
                        Action2 x;
                        x.kind = Action2::Kind::Push;
                        x.tile = TileCoord{"A", r * a + ii, k};
                        x.from = L3s(t, s);
                        x.to = L3s(to, s);
                        x.reads = reads;
                        x.channel = a_in.at(to);
                        act(mover(Role::SendE, t), x);
                    }
            }
            if (r + 1 < R) {
                const Dim to = node(r + 1, c);
                const std::uint32_t reads = 1 + (r + 2 < R ? 1 : 0);
                for (std::size_t n = 0; n < south_flow.size(); ++n) {
                    const std::uint32_t s = B0 + static_cast<std::uint32_t>(n % pb);
                    Action2 x;
                    x.kind = Action2::Kind::Push;
                    x.tile = south_flow[n];
                    x.from = L3s(t, s);
                    x.to = L3s(to, s);
                    x.reads = reads;
                    x.channel = b_in.at(to);
                    act(mover(Role::SendS, t), x);
                }
            }
            // The compute tile's channels: its feeds, its results (T and E's partials), its drain.
            const std::size_t fa = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, pool(A2, a), 0);
            const std::size_t fb = channel("fb" + sfx, mover(Role::FeedB, t), Channel2::To::L2, t, 0, 0, pool(B2, b), 0);
            const std::size_t cc = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0,
                                           pool(T2, a * b + a), 0);
            const std::size_t dc = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, pool(P0, pp), 0);
            const std::size_t fx = east ? channel("fx" + sfx, mover(Role::FeedX, t), Channel2::To::L2, t, 0, 0, pool(In2, 1), 0) : kNoChannel;
            const auto feed_b = [&](std::size_t n) {
                Action2 x;
                x.kind = Action2::Kind::Feed;
                x.tile = south_flow[n];
                x.from = L3s(t, B0 + static_cast<std::uint32_t>(n % pb));
                x.to = L2s(t, B2 + static_cast<std::uint32_t>(n % b));
                x.reads = a;                                      // the a calls of its column
                x.channel = fb;
                act(mover(Role::FeedB, t), x);
            };
            const auto call = [&](const char* op, std::vector<SlotRef> args, std::uint32_t to, const TileCoord* fresh, std::uint32_t merges,
                                  std::uint32_t reads) {
                Action2 x;
                x.kind = Action2::Kind::Call;
                x.op = op;
                x.args = std::move(args);
                x.to = L2s(t, to);
                if (fresh) {
                    x.tile = *fresh;
                    x.merges = merges;
                    x.reads = reads;
                    x.channel = cc;
                } else {
                    x.accumulate = true;
                }
                act(ProcessId{Role::Compute, t, 0, 0}, x);
            };
            // The first GEMM: T accumulates over k, then stays, read by E's Pt tile columns.
            for (Dim k = 0; k < Kt; ++k) {
                for (Dim ii = 0; ii < a; ++ii) {
                    Action2 x;
                    x.kind = Action2::Kind::Feed;
                    x.tile = TileCoord{"A", r * a + ii, k};
                    x.from = L3s(t, A0 + (k * a + ii) % pa);
                    x.to = L2s(t, A2 + ii);
                    x.reads = b;
                    x.channel = fa;
                    act(mover(Role::FeedA, t), x);
                }
                for (Dim jj = 0; jj < b; ++jj) feed_b(k * b + jj);
                for (Dim ii = 0; ii < a; ++ii)
                    for (Dim jj = 0; jj < b; ++jj) {
                        const TileCoord T{"T", r * a + ii, c * b + jj};
                        call("gemm", {L2s(t, A2 + ii), L2s(t, B2 + jj)}, T2 + ii * b + jj, k == 0 ? &T : nullptr, Kt - 1, Pt);
                    }
            }
            // The second GEMM, q outer: this node's partial of E[i, q], the east's merged in, west it goes.
            const std::size_t st = c == 0 ? channel("st" + sfx, mover(Role::SendW, t), Channel2::To::Dma, 0, engine_of.at(t).mc,
                                                    engine_of.at(t).engine, {}, 2)
                                          : kNoChannel;
            for (Dim q = 0; q < Pt; ++q) {
                for (Dim jj = 0; jj < b; ++jj) {
                    feed_b(Kt * b + q * b + jj);
                    for (Dim ii = 0; ii < a; ++ii) {
                        const TileCoord E{"E", r * a + ii, q};
                        call("gemm", {L2s(t, T2 + ii * b + jj), L2s(t, B2 + jj)}, Q2 + ii, jj == 0 ? &E : nullptr,
                             (b - 1) + (east ? 1 : 0), 1);
                    }
                }
                for (Dim ii = 0; ii < a; ++ii) {
                    const TileCoord E{"E", r * a + ii, q};
                    const std::uint32_t n = static_cast<std::uint32_t>(q * a + ii);
                    if (east) {
                        Action2 f;
                        f.kind = Action2::Kind::Feed;
                        f.tile = E;
                        f.from = L3s(t, E0 + n % pe);
                        f.to = L2s(t, In2);
                        f.reads = 1;
                        f.channel = fx;
                        act(mover(Role::FeedX, t), f);
                        call("sum", {L2s(t, In2)}, Q2 + ii, nullptr, 0, 0);
                    }
                    Action2 dr;
                    dr.kind = Action2::Kind::Drain;
                    dr.tile = E;
                    dr.from = L2s(t, Q2 + ii);
                    dr.to = L3s(t, P0 + n % pp);
                    dr.reads = 1;
                    dr.channel = dc;
                    act(mover(Role::Drain, t), dr);
                    Action2 w;
                    w.kind = Action2::Kind::Push;
                    w.tile = E;
                    w.from = L3s(t, P0 + n % pp);
                    if (c > 0) {
                        w.to = L3s(node(r, c - 1), E0 + n % pe);
                        w.reads = 1;
                        w.channel = kNoChannel;               // set below, once the west's inbound channel exists
                    } else {
                        w.channel = st;
                    }
                    act(mover(Role::SendW, t), w);
                    if (c == 0) {
                        Action2 s;
                        s.kind = Action2::Kind::Store;
                        s.tile = E;
                        s.channel = st;
                        act(engine_of.at(t), s);
                    }
                }
            }
        }
    // The partials' channels, east to west: node (r, c + 1)'s bm.W into node (r, c)'s pool.
    for (Dim r = 0; r < R; ++r)
        for (Dim c = 0; c + 1 < C; ++c) {
            const Dim to = node(r, c), from = node(r, c + 1);
            const std::size_t ch = channel("e" + std::to_string(from) + "to" + std::to_string(to), mover(Role::SendW, from), Channel2::To::L3,
                                           to, 0, 0, pool(E0, pe), 0);
            for (const Process2& pr : p.processes)
                if (pr.id == mover(Role::SendW, from))
                    for (std::size_t ai : pr.actions) p.actions[ai].channel = ch;
        }
    return p;
}

} // namespace sw::kpu::program::csp::gen
