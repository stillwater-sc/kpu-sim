#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_reduction.hpp
// The P3 generator (#611; docs/plans/distributed-csp-programs.md §3.4): a row reduction, s = the sums
// of X's rows, then the sums broadcast back along each mesh row, as softmax, LayerNorm and RMSNorm
// reduce and then normalize. A level-2 program on a DMM mesh, every hop one hop.
//
// THE DISTRIBUTION. X is block-distributed: node (r, c) owns an a x b block. s's tile i belongs to
// mesh row r, the row that owns X's tile row i.
//
// THREE FLOWS, one per mover (a mesh row has only its E and W links, so X does not use them):
//   in      X streams down each mesh column from its north edge node (a DMA engine each, bm.S), a node
//           keeping its own tiles, as P1's B; bm.a feeds them into the compute tile.
//   reduce  each node sums its own tiles row by row (`rowsum`), merges the partial arriving from the
//           east (bm.b feeds it; `sum ... acc`), drains its partial (bm.c), and pushes it west (bm.W):
//           a chain along the row, from its east end to its west.
//   back    the west end (column 0) holds the total: it stores it out of its west edge port (bm.W to
//           a DMA engine), and broadcasts it east (bm.E); every node holds it, in L3 and, fed by bm.x,
//           in its compute tile's L2, where the normalize that follows reads it.
//
// THE SLOTS, per node. L3: X's inbound 4, the partials' drains 2, the partials from the east 2, the
// broadcast's a (held). L2: X 2, the partials a, the incoming partial 1, the held sums a. A node's
// block may be at most 2 tiles tall (3 + 2a <= 8 L2 slots).
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

struct DistributedRowReduction {
    Dim m = 0, n = 0, tile = 0;
    std::string type = "int32";
    std::string name = "rowsum";
};

inline DistributedProgram generate_distributed_rowsum(const DistributedRowReduction& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed rowsum " + std::to_string(g.m) + "x" + std::to_string(g.n) + " on \"" + d.name +
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
        return fail("X's " + std::to_string(Mt) + "x" + std::to_string(Nt) + " tiles do not divide over the " + std::to_string(R) + "x" +
                    std::to_string(C) + " mesh");
    const Dim a = Mt / R, b = Nt / C;
    const std::uint32_t pin = 4, pp = 2, pe = 2;
    const std::uint32_t I0 = 0, P0 = pin, E0 = P0 + pp, B0 = E0 + pe;
    if (B0 + a > d.cluster->l3_tile.capacity_tiles)
        return fail("a node needs " + std::to_string(B0 + a) + " L3 slots, and the tile has " + std::to_string(d.cluster->l3_tile.capacity_tiles));
    const std::uint32_t X2 = 0, P2 = 2, In2 = 2 + a, S2 = 3 + a;
    if (S2 + a > d.cluster->compute_tile.l2.slots)
        return fail("a compute tile needs " + std::to_string(S2 + a) + " L2 slots (a block " + std::to_string(a) +
                    " tiles tall), and it has " + std::to_string(d.cluster->compute_tile.l2.slots));
    const Dim mcs = d.memory.controllers.value_or(1);
    if (C > d.dma.engines)
        return fail(std::to_string(C) + " mesh columns take a stream in, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"X", false, g.m, g.n, g.tile, g.tile, g.type, "in"},
                  Operand2{"s", true, g.m, 1, g.tile, 1, g.type, "out"}};
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
    const auto engine = [&](Dim c) { return ProcessId{Role::Dma, 0, c % mcs, c / mcs}; };
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };

    // ---- in: X down each column, a node keeping its own block.
    for (Dim c = 0; c < C; ++c) {
        std::vector<std::size_t> in(R);
        for (Dim r = 0; r < R; ++r) {
            const Dim t = node(r, c);
            in[r] = r == 0 ? channel("in" + std::to_string(t), engine(c), Channel2::To::L3, t, 0, 0, pool(I0, pin), 0)
                           : channel("x" + std::to_string(node(r - 1, c)) + "to" + std::to_string(t), mover(Role::SendS, node(r - 1, c)),
                                     Channel2::To::L3, t, 0, 0, pool(I0, pin), 0);
        }
        struct Item { TileCoord tile; Dim row; };
        const auto stream = [&](Dim from_row) {
            std::vector<Item> v;
            for (Dim r = from_row; r < R; ++r)
                for (Dim q = 0; q < a * b; ++q) v.push_back({TileCoord{"X", r * a + q / b, c * b + q % b}, r});
            return v;
        };
        const std::vector<Item> all = stream(0);
        for (std::size_t k = 0; k < all.size(); ++k) {
            Action2 x;
            x.kind = Action2::Kind::Load;
            x.tile = all[k].tile;
            x.to = L3s(node(0, c), I0 + static_cast<std::uint32_t>(k % pin));
            x.reads = 1;
            x.channel = in[0];
            act(engine(c), x);
        }
        for (Dim r = 0; r < R; ++r) {
            const Dim t = node(r, c);
            const std::size_t fa = channel("fa" + std::to_string(t), mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, pool(X2, 2), 0);
            const std::vector<Item> s = stream(r);
            std::uint32_t own = 0, fwd = 0;
            for (std::size_t k = 0; k < s.size(); ++k) {
                Action2 x;
                x.tile = s[k].tile;
                x.from = L3s(t, I0 + static_cast<std::uint32_t>(k % pin));
                x.reads = 1;
                if (s[k].row == r) {
                    x.kind = Action2::Kind::Feed;
                    x.to = L2s(t, X2 + own++ % 2);
                    x.channel = fa;
                    act(mover(Role::FeedA, t), x);
                } else {
                    x.kind = Action2::Kind::Push;
                    x.to = L3s(node(r + 1, c), I0 + fwd++ % pin);
                    x.channel = in[r + 1];
                    act(mover(Role::SendS, t), x);
                }
            }
        }
    }

    // ---- reduce along each row, east to west; then the total back east.
    for (Dim r = 0; r < R; ++r) {
        // Each node's inbound partials (from the east) and inbound broadcast (from the west).
        std::vector<std::size_t> from_east(C, kNoChannel), from_west(C, kNoChannel);
        for (Dim c = 0; c + 1 < C; ++c)
            from_east[c] = channel("p" + std::to_string(node(r, c + 1)) + "to" + std::to_string(node(r, c)), mover(Role::SendW, node(r, c + 1)),
                                   Channel2::To::L3, node(r, c), 0, 0, pool(E0, pe), 0);
        for (Dim c = 1; c < C; ++c)
            from_west[c] = channel("s" + std::to_string(node(r, c - 1)) + "to" + std::to_string(node(r, c)), mover(Role::SendE, node(r, c - 1)),
                                   Channel2::To::L3, node(r, c), 0, 0, pool(B0, a), 0);
        const ProcessId store_engine = engine(r % C);
        const std::size_t st = channel("st" + std::to_string(node(r, 0)), mover(Role::SendW, node(r, 0)), Channel2::To::Dma, 0,
                                       store_engine.mc, store_engine.engine, {}, 2);
        for (Dim c = 0; c < C; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            const bool east = c + 1 < C;
            const std::size_t cc = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0, pool(P2, a), 0);
            const std::size_t dc = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, pool(P0, pp), 0);
            const std::size_t fb = east ? channel("fb" + sfx, mover(Role::FeedB, t), Channel2::To::L2, t, 0, 0, pool(In2, 1), 0) : kNoChannel;
            const std::size_t fx = channel("fx" + sfx, mover(Role::FeedX, t), Channel2::To::L2, t, 0, 0, pool(S2, a), 0);
            for (Dim ii = 0; ii < a; ++ii) {
                const TileCoord si{"s", r * a + ii, 0};
                for (Dim jj = 0; jj < b; ++jj) {                   // own tiles, in stream order (ii, jj)
                    Action2 x;
                    x.kind = Action2::Kind::Call;
                    x.op = "rowsum";
                    x.args = {L2s(t, X2 + static_cast<std::uint32_t>((ii * b + jj) % 2))};
                    x.to = L2s(t, P2 + ii);
                    if (jj == 0) {
                        x.tile = si;
                        x.merges = (b - 1) + (east ? 1 : 0);
                        x.reads = 1;
                        x.channel = cc;
                    } else {
                        x.accumulate = true;
                    }
                    act(ProcessId{Role::Compute, t, 0, 0}, x);
                }
                if (east) {                                       // the partial from the east, merged in
                    Action2 f;
                    f.kind = Action2::Kind::Feed;
                    f.tile = si;
                    f.from = L3s(t, E0 + ii % pe);
                    f.to = L2s(t, In2);
                    f.reads = 1;
                    f.channel = fb;
                    act(mover(Role::FeedB, t), f);
                    Action2 x;
                    x.kind = Action2::Kind::Call;
                    x.op = "sum";
                    x.args = {L2s(t, In2)};
                    x.to = L2s(t, P2 + ii);
                    x.accumulate = true;
                    act(ProcessId{Role::Compute, t, 0, 0}, x);
                }
                // The partial (at column 0, the total) to L3.
                const std::uint32_t at = P0 + ii % pp;
                Action2 dr;
                dr.kind = Action2::Kind::Drain;
                dr.tile = si;
                dr.from = L2s(t, P2 + ii);
                dr.to = L3s(t, at);
                dr.channel = dc;
                if (c > 0) {
                    dr.reads = 1;                                 // west, as a partial
                    act(mover(Role::Drain, t), dr);
                    Action2 w;
                    w.kind = Action2::Kind::Push;
                    w.tile = si;
                    w.from = L3s(t, at);
                    w.to = L3s(node(r, c - 1), E0 + ii % pe);
                    w.reads = 1;
                    w.channel = from_east[c - 1];
                    act(mover(Role::SendW, t), w);
                } else {
                    dr.reads = 2 + (C > 1 ? 1 : 0);               // the store, its own feed, east
                    act(mover(Role::Drain, t), dr);
                    Action2 w;
                    w.kind = Action2::Kind::Push;
                    w.tile = si;
                    w.from = L3s(t, at);
                    w.channel = st;
                    act(mover(Role::SendW, t), w);
                    Action2 sto;
                    sto.kind = Action2::Kind::Store;
                    sto.tile = si;
                    sto.channel = st;
                    act(store_engine, sto);
                }
            }
            // The total back: column 0 from its drain, the others from the broadcast held in L3.
            for (Dim ii = 0; ii < a; ++ii) {
                const TileCoord si{"s", r * a + ii, 0};
                const SlotRef held = c == 0 ? L3s(t, P0 + ii % pp) : L3s(t, B0 + ii);
                if (east) {
                    Action2 e;
                    e.kind = Action2::Kind::Push;
                    e.tile = si;
                    e.from = held;
                    e.to = L3s(node(r, c + 1), B0 + ii);
                    e.reads = 0;                                  // held at the next node
                    e.channel = from_west[c + 1];
                    act(mover(Role::SendE, t), e);
                }
                Action2 f;
                f.kind = Action2::Kind::Feed;
                f.tile = si;
                f.from = held;
                f.to = L2s(t, S2 + ii);
                f.reads = 0;                                      // held, for the normalize
                f.channel = fx;
                act(mover(Role::FeedX, t), f);
            }
        }
    }
    return p;
}

} // namespace sw::kpu::program::csp::gen
