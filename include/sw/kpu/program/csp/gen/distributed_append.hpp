#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_append.hpp
// The P9 generator (#615; docs/plans/distributed-csp-programs.md §3.4): append to a growing extent
// with a windowed re-read, as autoregressive decode appends each token's K to the KV cache and reads a
// window of it back. A level-2 program on a DMM mesh, every hop one hop.
//
// THE DISTRIBUTION, by head. Node t owns head t: its cache is K's tile row t, one tile per position,
// positions [0, L) already in DRAM. The decode runs S steps; step s appends position L + s.
//
// ONE STEP, at head h's node (r, c), whose DRAM is reached through its column's row 0:
//   in       X[h, s], the step's new token (in decode, the projection's output; here a stand-in), and
//            the cache's W - 1 positions before L + s, the window, come in from the column's DMA engine
//            and move south (bm.S), each node keeping its own and pushing the rest on; bm.a feeds them.
//   append   `call sum(X) to K[h, L + s]`: the new position, drained (bm.c) and pushed north (bm.N) to
//            row 0 and out of its edge port; the DMA engine stores it.
//   window   `call sum(K new) to O[h, s]`, then the window's tiles merged in (`sum ... acc`): O is the
//            sum over the window, the new position included. O goes north and is stored as K was.
// The column's DMA engine runs step s's loads, then its stores, then step s + 1's loads: so from step 1
// on, the window re-reads positions this program appended, read after their store. The order is the
// decode's own: a step needs the step before it.
//
// THE SLOTS, per node. L3: the inbound 4, the drains 2, in transit from the south 2. L2: the feeds 2,
// the new position and O 2.
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

struct DistributedAppend {
    Dim prefix = 0;                     // L: positions already in the cache
    Dim steps = 0;                      // S: positions appended, one a step
    Dim window = 0;                     // W: positions an output sums, the new one included
    Dim tile = 0;                       // a position is one tile x tile tile
    std::string type = "fp32";
    std::string name = "append";
};

inline DistributedProgram generate_distributed_append(const DistributedAppend& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed append L " + std::to_string(g.prefix) + ", S " + std::to_string(g.steps) + ", W " +
                                  std::to_string(g.window) + " on \"" + d.name + "\": " + why);
    };
    if (g.steps == 0 || g.window == 0 || g.tile == 0) return fail("S, W and the tile must be non-zero");
    if (g.window - 1 > g.prefix)
        return fail("the first step's window reads the " + std::to_string(g.window - 1) + " positions before it, and the cache has " +
                    std::to_string(g.prefix));
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols(), H = L->node_count();
    const std::uint32_t pi = 4, pd = 2, pn = 2;
    const std::uint32_t I0 = 0, D0 = pi, N0 = pi + pd;
    const Dim l3 = d.cluster->l3_tile.capacity_tiles, l2 = d.cluster->compute_tile.l2.slots;
    if (N0 + pn > l3) return fail("a node needs " + std::to_string(N0 + pn) + " L3 slots, and the tile has " + std::to_string(l3));
    const std::uint32_t F2 = 0, K2 = 2, O2 = 3;
    if (O2 + 1 > l2) return fail("a compute tile needs " + std::to_string(O2 + 1) + " L2 slots, and it has " + std::to_string(l2));
    if (C > d.dma.engines)
        return fail(std::to_string(C) + " columns reach DRAM through their row 0, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));
    const Dim mcs = d.memory.controllers.value_or(1);
    const Dim T = g.tile, positions = g.prefix + g.steps;

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"X", false, H * T, g.steps * T, T, T, g.type, "in"},
                  Operand2{"K", false, H * T, positions * T, T, T, g.type, "inout"},
                  Operand2{"O", false, H * T, g.steps * T, T, T, g.type, "out"}};
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
    // Head h's tiles for step s, in the order they come in: X, then the window, oldest first.
    const auto step_in = [&](Dim h, Dim s) {
        std::vector<TileCoord> v{TileCoord{"X", h, s}};
        for (Dim w = g.window - 1; w >= 1; --w) v.push_back(TileCoord{"K", h, g.prefix + s - w});
        return v;
    };

    // Every node's compute-tile channels: its feed, its results (the new position and O), its drain.
    std::vector<std::size_t> fa(H), cc(H), dc(H);
    for (Dim t = 0; t < H; ++t) {
        const std::string sfx = std::to_string(t);
        fa[t] = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, pool(F2, 2), 0);
        cc[t] = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0, {K2, O2}, 0);
        dc[t] = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, pool(D0, pd), 0);
    }
    for (Dim c = 0; c < C; ++c) {
        const ProcessId engine{Role::Dma, 0, c % mcs, c / mcs};
        // Each node's inbound channel (row 0's from the engine, the others' from the node above) and
        // its north channel (row 0's out of its edge port to the engine).
        std::vector<std::size_t> in(R), north(R);
        for (Dim r = 0; r < R; ++r) {
            const Dim t = node(r, c);
            in[r] = r == 0 ? channel("in" + std::to_string(t), engine, Channel2::To::L3, t, 0, 0, pool(I0, pi), 0)
                           : channel("s" + std::to_string(node(r - 1, c)) + "to" + std::to_string(t), mover(Role::SendS, node(r - 1, c)),
                                     Channel2::To::L3, t, 0, 0, pool(I0, pi), 0);
            north[r] = r == 0 ? channel("st" + std::to_string(t), mover(Role::SendN, t), Channel2::To::Dma, 0, engine.mc, engine.engine, {}, 2)
                              : channel("n" + std::to_string(t) + "to" + std::to_string(node(r - 1, c)), mover(Role::SendN, t),
                                        Channel2::To::L3, node(r - 1, c), 0, 0, pool(N0, pn), 0);
        }
        // Per row, how many tiles each of its pools has taken: the slot of the next is that count mod the pool.
        std::vector<std::uint32_t> got(R, 0), sent(R, 0), fed(R, 0), drained(R, 0), arrived(R, 0), forwarded(R, 0);
        for (Dim s = 0; s < g.steps; ++s) {
            // ---- in: the engine loads every head's step, row by row; each node keeps its own, forwards the rest.
            for (Dim r = 0; r < R; ++r) {
                const Dim h = node(r, c);
                for (const TileCoord& tile : step_in(h, s)) {
                    Action2 x;
                    x.kind = Action2::Kind::Load;
                    x.tile = tile;
                    x.to = L3s(node(0, c), I0 + got[0]++ % pi);
                    x.reads = 1;
                    x.channel = in[0];
                    act(engine, x);
                    for (Dim at = 0; at < r; ++at) {          // south, one hop at a time
                        Action2 f;
                        f.kind = Action2::Kind::Push;
                        f.tile = tile;
                        f.from = L3s(node(at, c), I0 + sent[at]++ % pi);
                        f.to = L3s(node(at + 1, c), I0 + got[at + 1]++ % pi);
                        f.reads = 1;
                        f.channel = in[at + 1];
                        act(mover(Role::SendS, node(at, c)), f);
                    }
                    Action2 fd;                                // home: into the compute tile
                    fd.kind = Action2::Kind::Feed;
                    fd.tile = tile;
                    fd.from = L3s(h, I0 + sent[r]++ % pi);
                    fd.to = L2s(h, F2 + fed[r]++ % 2);
                    fd.reads = 1;
                    fd.channel = fa[h];
                    act(mover(Role::FeedA, h), fd);
                }
            }
            // ---- append and window, at every head of the column.
            std::vector<std::uint32_t> mine(R);
            for (Dim r = 0; r < R; ++r) {
                const Dim h = node(r, c);
                const TileCoord Kn{"K", h, g.prefix + s}, O{"O", h, s};
                const std::uint32_t first = fed[r] - static_cast<std::uint32_t>(g.window);   // this step's X
                const auto call = [&](const SlotRef& arg, std::uint32_t to, const TileCoord* fresh, std::uint32_t merges,
                                      std::uint32_t reads) {
                    Action2 x;
                    x.kind = Action2::Kind::Call;
                    x.op = "sum";
                    x.args = {arg};
                    x.to = L2s(h, to);
                    if (fresh) {
                        x.tile = *fresh;
                        x.merges = merges;
                        x.reads = reads;
                        x.channel = cc[h];
                    } else {
                        x.accumulate = true;
                    }
                    act(ProcessId{Role::Compute, h, 0, 0}, x);
                };
                call(L2s(h, F2 + first % 2), K2, &Kn, 0, 2);   // the new position: read by its drain and by O
                call(L2s(h, K2), O2, &O, static_cast<std::uint32_t>(g.window - 1), 1);
                for (Dim w = 1; w < g.window; ++w) call(L2s(h, F2 + (first + static_cast<std::uint32_t>(w)) % 2), O2, nullptr, 0, 0);
                mine[r] = drained[r];
                for (const auto& [tile, from] : {std::pair{Kn, K2}, std::pair{O, O2}}) {
                    Action2 dr;
                    dr.kind = Action2::Kind::Drain;
                    dr.tile = tile;
                    dr.from = L2s(h, from);
                    dr.to = L3s(h, D0 + drained[r]++ % pd);
                    dr.reads = 1;
                    dr.channel = dc[h];
                    act(mover(Role::Drain, h), dr);
                }
            }
            // ---- north: each node sends its own K and O, then the rows below's, in the order they came;
            // row 0 pushes them out of its edge port, and the engine stores them in that order.
            for (Dim r = 0; r < R; ++r) {
                const Dim h = node(r, c);
                std::vector<TileCoord> flow;
                for (Dim rr = r; rr < R; ++rr) {
                    flow.push_back(TileCoord{"K", node(rr, c), g.prefix + s});
                    flow.push_back(TileCoord{"O", node(rr, c), s});
                }
                for (std::size_t i = 0; i < flow.size(); ++i) {
                    Action2 x;
                    x.kind = Action2::Kind::Push;
                    x.tile = flow[i];
                    x.from = i < 2 ? L3s(h, D0 + (mine[r] + static_cast<std::uint32_t>(i)) % pd) : L3s(h, N0 + forwarded[r]++ % pn);
                    if (r > 0) {
                        x.to = L3s(node(r - 1, c), N0 + arrived[r - 1]++ % pn);
                        x.reads = 1;
                    }
                    x.channel = north[r];
                    act(mover(Role::SendN, h), x);
                    if (r == 0) {
                        Action2 st;
                        st.kind = Action2::Kind::Store;
                        st.tile = flow[i];
                        st.channel = north[0];
                        act(engine, st);
                    }
                }
            }
        }
    }
    return p;
}

} // namespace sw::kpu::program::csp::gen
