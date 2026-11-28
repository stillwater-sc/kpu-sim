#pragma once
// ============================================================================
// include/sw/kpu/program/csp/gen/distributed_gather.hpp
// The P4 generator (#616; docs/plans/distributed-csp-programs.md §3.4, §7 decision 2): gather and
// scatter, as an embedding lookup gathers a table's rows by token id and a scatter writes rows back by
// position. A level-2 program on a DMM mesh, every hop one hop.
//
// THE INDICES STAY AT THE DMA. They are run-time data, so no move on the mesh depends on them: a DMA
// engine's gather assembles tile G[i, j] from the rows of the table E that idx names, and its scatter
// writes tile Z[i, j]'s rows to the rows of W that jdx names. The mesh carries G and Z, which are `temp`
// (they never leave the nodes as themselves), in a pattern the program states.
//
// THE PATTERN. G's tile row i belongs to node i mod P. Each mesh row's DMA engine, at its west edge
// node, gathers the row's tiles in (i, then j) and they move east (bm.E) to their owner, which computes
// Z = G + G (`call add`); Z moves back west (bm.W), each node sending the row's tiles from itself
// eastward in the same order, and the engine scatters it. The engine runs two loads ahead of its
// stores, so the row's flows overlap without its store buffer (2 credits) closing a cycle.
//
// THE SLOTS, per node. L3: inbound for itself 2, inbound in transit 2, drains 2, in transit west 2. L2:
// the feeds 2, the results 2.
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

struct DistributedGather {
    Dim m = 0;                          // rows gathered (G's and Z's rows; idx's and jdx's length)
    Dim cols = 0;                       // every row's width
    Dim table_rows = 0;                 // E's rows, which idx indexes
    Dim out_rows = 0;                   // W's rows, which jdx indexes
    Dim tile = 0;
    std::string type = "fp32";
    std::string name = "gather";
};

inline DistributedProgram generate_distributed_gather(const DistributedGather& g, const platform::DeviceSpecification& d) {
    const auto fail = [&](const std::string& why) -> DistributedProgram {
        throw DistributedGenError("distributed gather " + std::to_string(g.m) + "x" + std::to_string(g.cols) + " on \"" + d.name + "\": " + why);
    };
    if (g.m == 0 || g.cols == 0 || g.table_rows == 0 || g.out_rows == 0 || g.tile == 0)
        return fail("the rows, columns, table, output and tile must be non-zero");
    std::string why;
    const auto L = platform::MeshLayout::of(d, &why);
    if (!L) return fail("not a mesh: " + why);
    const Dim R = L->rows(), C = L->cols(), P = L->node_count();
    const auto tiles = [&](Dim x) { return x / g.tile + (x % g.tile != 0); };
    const Dim Mt = tiles(g.m), Ct = tiles(g.cols);
    const std::uint32_t O0 = 0, T0 = 2, D0 = 4, W0 = 6;
    const Dim l3 = d.cluster->l3_tile.capacity_tiles, l2 = d.cluster->compute_tile.l2.slots;
    if (W0 + 2 > l3) return fail("a node needs " + std::to_string(W0 + 2) + " L3 slots, and the tile has " + std::to_string(l3));
    const std::uint32_t F2 = 0, Z2 = 2;
    if (Z2 + 2 > l2) return fail("a compute tile needs 4 L2 slots, and it has " + std::to_string(l2));
    if (R > d.dma.engines)
        return fail(std::to_string(R) + " mesh rows gather through their west edge node, one DMA engine each, and the device has " +
                    std::to_string(d.dma.engines));
    const Dim mcs = d.memory.controllers.value_or(1);

    DistributedProgram p;
    p.name = g.name;
    p.rows = R;
    p.cols = C;
    p.operands = {Operand2{"E", false, g.table_rows, g.cols, g.tile, g.tile, g.type, "in"},
                  Operand2{"idx", true, g.m, 1, g.tile, 1, "int32", "in"},
                  Operand2{"jdx", true, g.m, 1, g.tile, 1, "int32", "in"},
                  Operand2{"G", false, g.m, g.cols, g.tile, g.tile, g.type, "temp"},
                  Operand2{"Z", false, g.m, g.cols, g.tile, g.tile, g.type, "temp"},
                  Operand2{"W", false, g.out_rows, g.cols, g.tile, g.tile, g.type, "out"}};
    const auto node = [&](Dim r, Dim c) { return L->node(r, c); };
    const auto L3s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L3, t, s}; };
    const auto L2s = [](Dim t, std::uint32_t s) { return SlotRef{SlotRef::Pool::L2, t, s}; };
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

    for (Dim r = 0; r < R; ++r) {
        // The row's tiles, in their order, and each one's owner column.
        struct Item {
            Dim i, j, col;
        };
        std::vector<Item> items;
        Dim reach = 0;                                    // the easternmost owner, plus one
        for (Dim i = 0; i < Mt; ++i) {
            const auto at = L->pos(i % P);
            if (at.r != r) continue;
            for (Dim j = 0; j < Ct; ++j) items.push_back(Item{i, j, at.c});
            reach = std::max(reach, at.c + 1);
        }
        if (items.empty()) continue;
        const ProcessId engine{Role::Dma, 0, r % mcs, r / mcs};
        // Channels: east into each node up to the reach, west out of each; the owners' compute tiles.
        std::vector<std::size_t> in(reach), west(reach), fa(reach, kNoChannel), cc(reach, kNoChannel), dc(reach, kNoChannel);
        for (Dim c = 0; c < reach; ++c) {
            const Dim t = node(r, c);
            const std::string sfx = std::to_string(t);
            in[c] = c == 0 ? channel("in" + sfx, engine, Channel2::To::L3, t, 0, 0, {O0, O0 + 1, T0, T0 + 1}, 0)
                           : channel("g" + std::to_string(node(r, c - 1)) + "to" + sfx, mover(Role::SendE, node(r, c - 1)), Channel2::To::L3, t,
                                     0, 0, {O0, O0 + 1, T0, T0 + 1}, 0);
            west[c] = c == 0 ? channel("st" + sfx, mover(Role::SendW, t), Channel2::To::Dma, 0, engine.mc, engine.engine, {}, 2)
                             : channel("z" + sfx + "to" + std::to_string(node(r, c - 1)), mover(Role::SendW, t), Channel2::To::L3,
                                       node(r, c - 1), 0, 0, {W0, W0 + 1}, 0);
        }
        for (const Item& it : items)
            if (fa[it.col] == kNoChannel) {
                const Dim t = node(r, it.col);
                const std::string sfx = std::to_string(t);
                fa[it.col] = channel("fa" + sfx, mover(Role::FeedA, t), Channel2::To::L2, t, 0, 0, {F2, F2 + 1}, 0);
                cc[it.col] = channel("cc" + sfx, ProcessId{Role::Compute, t, 0, 0}, Channel2::To::L2, t, 0, 0, {Z2, Z2 + 1}, 0);
                dc[it.col] = channel("dc" + sfx, mover(Role::Drain, t), Channel2::To::L3, t, 0, 0, {D0, D0 + 1}, 0);
            }
        // Per node, how many tiles each of its pools has taken in and given out.
        std::vector<std::uint32_t> own_in(reach, 0), own_out(reach, 0), tr_in(reach, 0), tr_out(reach, 0), fed(reach, 0), made(reach, 0),
            drained(reach, 0), pushed_own(reach, 0), w_in(reach, 0), w_out(reach, 0);
        const auto store = [&](const Item& it) {
            Action2 s;
            s.kind = Action2::Kind::Store;
            s.tile = TileCoord{"Z", it.i, it.j};
            s.rows = "W";
            s.index = "jdx";
            s.channel = west[0];
            act(engine, s);
        };
        constexpr std::size_t kLag = 2;                   // loads ahead of stores at the engine
        for (std::size_t n = 0; n < items.size(); ++n) {
            const Item& it = items[n];
            const TileCoord G{"G", it.i, it.j}, Z{"Z", it.i, it.j};
            const auto slot_in = [&](Dim c) { return c == it.col ? O0 + own_in[c]++ % 2 : T0 + tr_in[c]++ % 2; };
            Action2 ld;                                   // the gather, at the west edge
            ld.kind = Action2::Kind::Load;
            ld.tile = G;
            ld.rows = "E";
            ld.index = "idx";
            ld.to = L3s(node(r, 0), slot_in(0));
            ld.reads = 1;
            ld.channel = in[0];
            act(engine, ld);
            if (n >= kLag) store(items[n - kLag]);
            for (Dim c = 0; c < it.col; ++c) {            // east, one hop at a time
                Action2 x;
                x.kind = Action2::Kind::Push;
                x.tile = G;
                x.from = L3s(node(r, c), T0 + tr_out[c]++ % 2);
                x.to = L3s(node(r, c + 1), slot_in(c + 1));
                x.reads = 1;
                x.channel = in[c + 1];
                act(mover(Role::SendE, node(r, c)), x);
            }
            const Dim t = node(r, it.col), oc = it.col;   // at its owner: Z = G + G
            Action2 f;
            f.kind = Action2::Kind::Feed;
            f.tile = G;
            f.from = L3s(t, O0 + own_out[oc]++ % 2);
            f.to = L2s(t, F2 + fed[oc] % 2);
            f.reads = 2;                                  // both of add's operands
            f.channel = fa[oc];
            act(mover(Role::FeedA, t), f);
            Action2 k;
            k.kind = Action2::Kind::Call;
            k.op = "add";
            k.args = {L2s(t, F2 + fed[oc] % 2), L2s(t, F2 + fed[oc] % 2)};
            ++fed[oc];
            k.tile = Z;
            k.to = L2s(t, Z2 + made[oc] % 2);
            k.reads = 1;
            k.channel = cc[oc];
            act(ProcessId{Role::Compute, t, 0, 0}, k);
            Action2 dr;
            dr.kind = Action2::Kind::Drain;
            dr.tile = Z;
            dr.from = L2s(t, Z2 + made[oc]++ % 2);
            dr.to = L3s(t, D0 + drained[oc]++ % 2);
            dr.reads = 1;
            dr.channel = dc[oc];
            act(mover(Role::Drain, t), dr);
        }
        for (std::size_t n = items.size() >= kLag ? items.size() - kLag : 0; n < items.size(); ++n) store(items[n]);
        // West: node c sends the row's tiles owned at c or east of it, in the row's order.
        for (Dim c = 0; c < reach; ++c) {
            const Dim t = node(r, c);
            for (const Item& it : items) {
                if (it.col < c) continue;
                Action2 x;
                x.kind = Action2::Kind::Push;
                x.tile = TileCoord{"Z", it.i, it.j};
                x.from = it.col == c ? L3s(t, D0 + pushed_own[c]++ % 2) : L3s(t, W0 + w_out[c]++ % 2);
                if (c > 0) {
                    x.to = L3s(node(r, c - 1), W0 + w_in[c - 1]++ % 2);
                    x.reads = 1;
                }
                x.channel = west[c];
                act(mover(Role::SendW, t), x);
            }
        }
    }
    return p;
}

} // namespace sw::kpu::program::csp::gen
