#pragma once
// ============================================================================
// include/sw/kpu/program/csp/distributed.hpp
// The LEVEL-2 CSP program: a distributed program for the DMM mesh (#605;
// docs/plans/distributed-csp-programs.md §3.2).
//
// Level 1 (csp_program.hpp) addresses one flat L3 and one fat fabric: one process of each kind,
// one global order. Level 2 addresses the machine as it is: a mesh of nodes, each an L3 tile with
// its own compute tile (platform/mesh_layout.hpp). Every process has a place and a role, and runs
// its own list of actions. There is no global order: processes meet only through the slots they
// fill and the credits those slots return.
//
// ---------------------------------------------------------------------------
// PROCESSES
// ---------------------------------------------------------------------------
//   dma[m, e]     DMA engine e of memory controller m: Loads from DRAM, Stores to DRAM
//   bm.N[t] ...   node t's send movers, bm.N bm.E bm.S bm.W: Pushes to a neighbour, or out of the
//                 edge port to a DMA engine
//   bm.a[t] bm.b[t] bm.x[t]   node t's feed movers: Feeds, L3 -> its compute tile's L2
//   bm.c[t]       node t's drain mover: Drains, its compute tile's L2 -> L3
//   cf[t]         node t's compute tile: Calls, on L2 slots
//
// ---------------------------------------------------------------------------
// CHANNELS: one sender, one receiving slot pool
// ---------------------------------------------------------------------------
// A channel is one packet flow. Exactly one process sends on it, and it owns its slots: the slots
// of the receiving pool (l3[t] or l2[t]) that it lists. A slot belongs to one channel, so its
// credit always goes back to the same sender. A channel into a DMA engine (a Store's path out of
// an edge port) has no slots, only `credits`: the engine's store buffer.
//
// ---------------------------------------------------------------------------
// ACTIONS, and the residencies they open
// ---------------------------------------------------------------------------
//   load T to l3[t]:s reads n via ch            DMA: DRAM's tile T into slot s
//   push T from l3[t]:s to l3[u]:s2 reads n via ch    send mover: one hop, into a neighbour's slot
//   push T from l3[t]:s to dma[m, e] via ch     send mover: out of the edge port, to a Store
//   feed T from l3[t]:s to l2[t]:s2 reads n via ch    feed mover
//   drain T from l2[t]:s to l3[t]:s2 reads n via ch   drain mover
//   store T via ch                              DMA: the pushed tile T to DRAM
//   call op(l2[t]:a, l2[t]:b) to T at l2[t]:s reads n via ch   compute: opens a result
//   call op(l2[t]:a, l2[t]:b) acc l2[t]:s       compute: merges into a live result
//
// Every write that fills a slot OPENS A RESIDENCY: it waits for its channel's credit for that
// slot, spends it, and states how many reads the residency serves (`reads n`). A read is a push,
// feed, drain or call that takes the slot as its source. When the n-th read is done the slot is
// freed and its credit returns to the channel's sender. `reads 0` holds the tile to the end of the
// program. So a residency's release is its last read, as level 1 records consumers per residency:
// the program states its reuse (reuse-is-deliberate), and no Release action is needed.
// An accumulating call (`acc`) writes into a residency its own channel opened, and neither spends
// a credit nor counts as a read.
//
// WHAT IS CHECKED HERE: structure (validate_structure): every name resolves, every action runs in
// a process of its role, on its own node, through its own channel, into one of that channel's
// slots, and every tile is an operand's. The machine's rules -- eight movers, one hop, credits,
// placement, liveness, values (D1-D6) -- are the distributed checker's (#607) and L-B's (#606).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

// The roles a process plays. The movers' order is the naming map's bm[k] (platform/mesh_layout.hpp).
enum class Role : std::uint8_t { SendN, SendE, SendS, SendW, FeedA, FeedB, FeedX, Drain, Dma, Compute };

inline const char* role_name(Role r) {
    switch (r) {
        case Role::SendN:   return "bm.N";
        case Role::SendE:   return "bm.E";
        case Role::SendS:   return "bm.S";
        case Role::SendW:   return "bm.W";
        case Role::FeedA:   return "bm.a";
        case Role::FeedB:   return "bm.b";
        case Role::FeedX:   return "bm.x";
        case Role::Drain:   return "bm.c";
        case Role::Dma:     return "dma";
        case Role::Compute: return "cf";
    }
    return "?";
}

inline std::optional<Role> role_from(const std::string& s) {
    for (Role r : {Role::SendN, Role::SendE, Role::SendS, Role::SendW, Role::FeedA, Role::FeedB, Role::FeedX,
                   Role::Drain, Role::Dma, Role::Compute})
        if (s == role_name(r)) return r;
    return std::nullopt;
}

inline bool is_send(Role r) { return r == Role::SendN || r == Role::SendE || r == Role::SendS || r == Role::SendW; }
inline bool is_feed(Role r) { return r == Role::FeedA || r == Role::FeedB || r == Role::FeedX; }

// Where a process is: a node of the mesh, or a memory controller's DMA engine.
struct ProcessId {
    Role role = Role::Dma;
    Dim node = 0;                       // a mover's or compute tile's node
    Dim mc = 0, engine = 0;             // a DMA engine's controller and index

    std::string to_string() const {
        if (role == Role::Dma) return "dma[" + std::to_string(mc) + ", " + std::to_string(engine) + "]";
        return std::string(role_name(role)) + "[" + std::to_string(node) + "]";
    }
    bool operator==(const ProcessId& o) const {
        return role == o.role && (role == Role::Dma ? mc == o.mc && engine == o.engine : node == o.node);
    }
    bool operator<(const ProcessId& o) const { return to_string() < o.to_string(); }
};

// One slot of a node's L3 tile or of its compute tile's L2.
struct SlotRef {
    enum class Pool : std::uint8_t { L3, L2 } pool = Pool::L3;
    Dim node = 0;
    std::uint32_t slot = 0;

    std::string to_string() const {
        return std::string(pool == Pool::L3 ? "l3" : "l2") + "[" + std::to_string(node) + "]:" + std::to_string(slot);
    }
    bool operator==(const SlotRef& o) const { return pool == o.pool && node == o.node && slot == o.slot; }
    bool operator<(const SlotRef& o) const {
        return pool != o.pool ? pool < o.pool : node != o.node ? node < o.node : slot < o.slot;
    }
};

// One packet flow: one sending process, one receiving pool.
struct Channel2 {
    std::string name;
    ProcessId from;
    enum class To : std::uint8_t { L3, L2, Dma } to = To::L3;
    Dim node = 0;                       // To::L3 / To::L2: the receiving node
    Dim mc = 0, engine = 0;             // To::Dma: the receiving engine
    std::vector<std::uint32_t> slots;   // To::L3 / To::L2: the slots it owns, one credit each
    std::uint32_t credits = 0;          // To::Dma: the store buffer's credits

    std::uint32_t credit_count() const { return to == To::Dma ? credits : static_cast<std::uint32_t>(slots.size()); }
    std::string destination() const {
        if (to == To::Dma) return "dma[" + std::to_string(mc) + ", " + std::to_string(engine) + "]";
        return std::string(to == To::L3 ? "l3" : "l2") + "[" + std::to_string(node) + "]";
    }
};

inline constexpr std::size_t kNoChannel = std::numeric_limits<std::size_t>::max();

struct Action2 {
    enum class Kind : std::uint8_t { Load, Push, Feed, Drain, Store, Call } kind = Kind::Load;
    TileCoord tile;                     // what moves; a Call: the result's tile (empty when it accumulates)
    std::optional<SlotRef> from;        // Push, Feed, Drain: the source slot
    std::optional<SlotRef> to;          // the slot it fills; a Call: its result slot; empty: a Push to a DMA engine, a Store
    std::size_t channel = kNoChannel;   // the channel it sends on, or a Store's receives from
    std::uint32_t reads = 0;            // a residency it opens: the reads it serves (0: held to the end)
    std::string op;                     // a Call: its tile function (gemm, ...)
    std::vector<SlotRef> args;          // a Call: its operand slots
    bool accumulate = false;            // a Call: merge into the live result at `to`
    int line = 0;                       // where it was written, for messages; 0 when generated

    // Does it open a residency (wait for a credit and spend it)?
    bool opens() const { return to.has_value() && !(kind == Kind::Call && accumulate); }
};

struct Process2 {
    ProcessId id;
    std::vector<std::size_t> actions;   // indices into DistributedProgram::actions, in its own order
};

// An operand, as level 1 declares it: its shape, its tiling and its number system.
struct Operand2 {
    std::string name;
    bool is_vector = false;
    Dim rows = 0, cols = 0, tile_rows = 0, tile_cols = 0;
    std::string type = "fp32";
    std::string io;                     // in | out | inout

    Dim tile_grid_rows() const { return tile_rows ? (rows + tile_rows - 1) / tile_rows : 0; }
    Dim tile_grid_cols() const { return is_vector ? 1 : tile_cols ? (cols + tile_cols - 1) / tile_cols : 0; }
};

struct DistributedProgram {
    std::string name;
    Dim rows = 0, cols = 0;             // the mesh it is written for
    std::vector<Operand2> operands;
    std::vector<Channel2> channels;
    std::vector<Process2> processes;
    std::vector<Action2> actions;

    Dim node_count() const { return rows * cols; }
    std::size_t channel_index(const std::string& n) const {
        for (std::size_t i = 0; i < channels.size(); ++i)
            if (channels[i].name == n) return i;
        return kNoChannel;
    }
    const Operand2* operand(const std::string& n) const {
        for (const Operand2& o : operands)
            if (o.name == n) return &o;
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// Structure (not the machine's rules: those are #607's)
// ---------------------------------------------------------------------------
// Empty when every name resolves and every action fits its process, channel and slots; otherwise
// the first problem, naming the action (and its line, when it was written).
inline std::string validate_structure(const DistributedProgram& p) {
    const auto node_ok = [&](Dim t) { return t < p.node_count(); };
    if (p.rows == 0 || p.cols == 0) return "the mesh has no nodes";
    std::set<std::string> names;
    for (const Operand2& o : p.operands) {
        if (!names.insert(o.name).second) return "operand " + o.name + " is declared twice";
        if (o.rows == 0 || o.tile_rows == 0 || (!o.is_vector && (o.cols == 0 || o.tile_cols == 0)))
            return "operand " + o.name + " has an empty shape or tile";
    }
    std::set<std::string> chans;
    std::map<SlotRef, std::string> owner;
    for (const Channel2& c : p.channels) {
        const std::string w = "channel " + c.name;
        if (!chans.insert(c.name).second) return w + " is declared twice";
        if (c.from.role != Role::Dma && !node_ok(c.from.node)) return w + ": its sender " + c.from.to_string() + " is off the mesh";
        if (c.to == Channel2::To::Dma) {
            if (c.credits == 0) return w + ": a channel into a DMA engine needs credits, its store buffer";
            if (!c.slots.empty()) return w + ": a channel into a DMA engine has no slots";
            continue;
        }
        if (!node_ok(c.node)) return w + ": " + c.destination() + " is off the mesh";
        if (c.slots.empty()) return w + " owns no slots, so it can never send";
        for (std::uint32_t s : c.slots) {
            const SlotRef r{c.to == Channel2::To::L3 ? SlotRef::Pool::L3 : SlotRef::Pool::L2, c.node, s};
            const auto [it, fresh] = owner.emplace(r, c.name);
            if (!fresh) return w + ": slot " + r.to_string() + " belongs to channel " + it->second + " already";
        }
    }
    std::set<std::string> procs;
    for (const Process2& pr : p.processes) {
        const std::string pw = "process " + pr.id.to_string();
        if (!procs.insert(pr.id.to_string()).second) return pw + " is declared twice";
        if (pr.id.role != Role::Dma && !node_ok(pr.id.node)) return pw + " is off the mesh";
        for (std::size_t ai : pr.actions) {
            const Action2& a = p.actions.at(ai);
            const std::string w = pw + (a.line ? ", line " + std::to_string(a.line) : std::string()) + ": ";
            const Role r = pr.id.role;
            // The role runs the action.
            const bool fits = (a.kind == Action2::Kind::Load && r == Role::Dma) ||
                              (a.kind == Action2::Kind::Store && r == Role::Dma) ||
                              (a.kind == Action2::Kind::Push && is_send(r)) ||
                              (a.kind == Action2::Kind::Feed && is_feed(r)) ||
                              (a.kind == Action2::Kind::Drain && r == Role::Drain) ||
                              (a.kind == Action2::Kind::Call && r == Role::Compute);
            static const char* kKind[] = {"load", "push", "feed", "drain", "store", "call"};
            const char* k = kKind[static_cast<int>(a.kind)];
            if (!fits) return w + "a " + k + " does not run in " + role_name(r);
            const bool shape = (a.kind == Action2::Kind::Load && !a.from && a.to) ||
                               (a.kind == Action2::Kind::Push && a.from) ||
                               ((a.kind == Action2::Kind::Feed || a.kind == Action2::Kind::Drain) && a.from && a.to) ||
                               (a.kind == Action2::Kind::Store && !a.from && !a.to) ||
                               (a.kind == Action2::Kind::Call && a.to && !a.from && !a.args.empty());
            if (!shape) return w + "a " + k + " is missing its source, its destination or its operands";
            // The tile is an operand's.
            if (!(a.kind == Action2::Kind::Call && a.accumulate)) {
                const Operand2* o = p.operand(a.tile.operand);
                if (!o) return w + "tile " + a.tile.to_string() + " is not a declared operand's";
                if (a.tile.ti >= o->tile_grid_rows() || a.tile.tj >= o->tile_grid_cols())
                    return w + "tile " + a.tile.to_string() + " is outside " + o->name + "'s tiles";
            }
            // Sources and operands are on the process's own node, in the pool its role reads.
            const auto own = [&](const SlotRef& s, SlotRef::Pool pool) {
                return s.pool == pool && s.node == pr.id.node;
            };
            if (a.from) {
                const SlotRef::Pool src = a.kind == Action2::Kind::Drain ? SlotRef::Pool::L2 : SlotRef::Pool::L3;
                if (!own(*a.from, src))
                    return w + "a " + k + " reads " + a.from->to_string() + ", not its own node's " +
                           (src == SlotRef::Pool::L3 ? "L3" : "L2");
            }
            for (const SlotRef& s : a.args)
                if (!own(s, SlotRef::Pool::L2)) return w + "a call reads " + s.to_string() + ", not its own compute tile's L2";
            // The accumulating call: into its own node's L2; no channel.
            if (a.kind == Action2::Kind::Call && a.accumulate) {
                if (!a.to || !own(*a.to, SlotRef::Pool::L2)) return w + "a call accumulates into its own compute tile's L2";
                if (a.channel != kNoChannel) return w + "an accumulating call spends no credit, so it names no channel";
                continue;
            }
            // Everything else names a channel this process sends on (a Store: one into its engine).
            if (a.channel >= p.channels.size()) return w + "its channel is not declared";
            const Channel2& c = p.channels[a.channel];
            if (a.kind == Action2::Kind::Store) {
                if (c.to != Channel2::To::Dma || c.mc != pr.id.mc || c.engine != pr.id.engine)
                    return w + "a store receives from a channel into its own engine, and " + c.name + " goes to " + c.destination();
                continue;
            }
            if (!(c.from == pr.id)) return w + "channel " + c.name + " is " + c.from.to_string() + "'s, not this process's";
            if (!a.to) {                                              // a push out to a DMA engine
                if (c.to != Channel2::To::Dma) return w + "channel " + c.name + " goes to " + c.destination() + ", not a DMA engine";
                continue;
            }
            const Channel2::To want = a.to->pool == SlotRef::Pool::L3 ? Channel2::To::L3 : Channel2::To::L2;
            if (c.to != want || c.node != a.to->node)
                return w + "it fills " + a.to->to_string() + " through channel " + c.name + ", which goes to " + c.destination();
            if (std::find(c.slots.begin(), c.slots.end(), a.to->slot) == c.slots.end())
                return w + "slot " + a.to->to_string() + " is not one of channel " + c.name + "'s";
            if ((a.kind == Action2::Kind::Feed || a.kind == Action2::Kind::Call) && a.to->node != pr.id.node)
                return w + "a " + std::string(k) + " fills its own compute tile's L2, not " + a.to->to_string();
            if (a.kind == Action2::Kind::Drain && a.to->node != pr.id.node)
                return w + "a drain fills its own node's L3, not " + a.to->to_string();
        }
    }
    return {};
}

} // namespace sw::kpu::program::csp
