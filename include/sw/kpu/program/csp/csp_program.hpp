// ============================================================================
// include/sw/kpu/program/csp/csp_program.hpp
// The CSP program (ADR 0002 §2.1; #281; docs/plans/csp-program-tile-sequencing.md §3.1).
//
// The sequencing layer every execution level interprets: PROCESSES, each with the ordered tile
// actions it performs, exchanging tiles over CHANNELS whose capacities are their credits. It is
// lowered from L0 (csp/lower.hpp), which today is derived from hand-coded loops and later from
// the Domain Flow Program -- never written by hand, so it has a disassembler and no syntax.
//
// LEVEL 1 addresses the machine as one flat L3 and one fat compute fabric (the plan's MPI
// hypothesis): one process of each kind, and inner loops collapsed into tile-function calls --
// a Call runs a whole L0 tile kernel (GEMM, GETRF, TRSM, ...) and never sequences elements.
// Level 2 (partitions over distributed L3 and compute tiles) is a later step.
//
// RESIDENCY IS THE PROGRAM'S. A tile occupies an L3 slot from the action that brings it in
// (Load, or a Writeback of a result) to its Release, which follows its last consumer (a Move
// out of L3, or the Store). Every residency records its consumer count. No tag-match at run
// time creates reuse the program did not decide; an interpreter that sees a read of a released
// tile has found a lowering bug.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/tile_program.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

inline constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

enum class ProcessKind : std::uint8_t { Dma, BlockMover, Streamer, Compute };

inline const char* to_string(ProcessKind k) {
    switch (k) {
        case ProcessKind::Dma:        return "dma";
        case ProcessKind::BlockMover: return "bm";
        case ProcessKind::Streamer:   return "str";
        case ProcessKind::Compute:    return "cf";
    }
    return "?";
}

// Level-1 channels, in their index order in CspProgram::channels.
enum class Chan : std::uint8_t { Dram, L3, L2, Cf };
inline constexpr std::size_t kChannels = 4;

inline const char* to_string(Chan c) {
    switch (c) {
        case Chan::Dram: return "dram";
        case Chan::L3:   return "l3";
        case Chan::L2:   return "l2";
        case Chan::Cf:   return "cf";
    }
    return "?";
}

// A tile context (docs/plans/csp-language.md §3.4): transforms bound to a tile as it crosses a
// boundary, each at the place it runs. A result leaves the fabric through `fabric` (on the
// accumulator, before the Drain), `str.drain` (the streamer's L1 -> L2 drain) and `bm.egress`
// (the BlockMover's L2 -> L3 writeback), in that order. `bm.ingress` (L3 -> L2) is an operand's
// way in.
enum class Place : std::uint8_t { Fabric, StrDrain, BmEgress, BmIngress };

inline const char* to_string(Place p) {
    switch (p) {
        case Place::Fabric:    return "fabric";
        case Place::StrDrain:  return "str.drain";
        case Place::BmEgress:  return "bm.egress";
        case Place::BmIngress: return "bm.ingress";
    }
    return "?";
}

inline bool parse_place(const std::string& s, Place& out) {
    for (Place p : {Place::Fabric, Place::StrDrain, Place::BmEgress, Place::BmIngress})
        if (s == to_string(p)) { out = p; return true; }
    return false;
}

// A vector operation: the epilogue (decision Q4). add takes a vector tile, broadcast down the
// rows; the rest are the activations.
enum class VeOp : std::uint8_t { Add, Relu, Gelu, Silu, Atan };
inline constexpr std::size_t kVeOps = 5;

inline const char* to_string(VeOp v) {
    switch (v) {
        case VeOp::Add:  return "add";
        case VeOp::Relu: return "relu";
        case VeOp::Gelu: return "gelu";
        case VeOp::Silu: return "silu";
        case VeOp::Atan: return "atan";
    }
    return "?";
}

inline bool parse_veop(const std::string& s, VeOp& out) {
    for (VeOp v : {VeOp::Add, VeOp::Relu, VeOp::Gelu, VeOp::Silu, VeOp::Atan})
        if (s == to_string(v)) { out = v; return true; }
    return false;
}

// The activation an op is, for an op that is one.
inline ActivationFn activation_of(VeOp v) {
    switch (v) {
        case VeOp::Gelu: return ActivationFn::Gelu;
        case VeOp::Silu: return ActivationFn::Silu;
        case VeOp::Atan: return ActivationFn::Atan;
        default:         return ActivationFn::Relu;
    }
}

struct Stage {
    VeOp op = VeOp::Relu;
    Place place = Place::Fabric;
    TileCoord arg;                      // add: the vector tile (resident in L3)
};

// One step of one process.
//   Load      dma  dram -> l3   brings a tile into an L3 slot (opens a residency)
//   Store     dma  l3 -> dram   writes a dirty tile back (an L3 consumer)
//   Move      bm   l3 -> l2     a copy out of L3 toward the fabric (an L3 consumer)
//   Writeback bm   l2 -> l3     a result back into L3 (opens or dirties a residency)
//   Feed      str  l2 -> cf     into the compute fabric
//   Drain     str  cf -> l2     a result out of the fabric
//   Call      cf                a tile function: the L0 op `l0_op`, on tiles the fabric holds
//   Release   (l3 credit)       the residency's last consumer is done: its slot is free
//   Inherit   (l3 credit)       the tile arrives resident, from the operator before this one: a
//                               residency opened with no Load, its slot held from the start
//   Retain    (l3 credit)       the residency ends without a Release: its slot, and the tile,
//                               outlive the program, for the operator after it
// (Inherit and Retain are kpu-run-csp-programs step 4d: residency across operators.)
struct Action {
    enum class Kind : std::uint8_t { Load, Store, Move, Writeback, Feed, Drain, Call, Release, Inherit, Retain };
    Kind kind = Kind::Call;
    TileCoord tile;                     // the tile moved, released, or the Call's first output
    std::size_t process = kNone;        // index into CspProgram::processes (Release: none)
    std::size_t l0_op = kNone;          // the L0 op this action implements (traceability)
    std::size_t residency = kNone;      // the L3 residency it opens, reads or releases
    bool accumulate = false;            // Call: adds to the fabric's accumulator for its output,
                                        // which starts at zero (an output-stationary chain)
    std::vector<Stage> context;         // applied to the tile as this action moves it, in order:
                                        // Drain (fabric, str.drain), Writeback (bm.egress),
                                        // Move (bm.ingress)
};

inline const char* to_string(Action::Kind k) {
    switch (k) {
        case Action::Kind::Load:      return "LOAD";
        case Action::Kind::Store:     return "STORE";
        case Action::Kind::Move:      return "MOVE";
        case Action::Kind::Writeback: return "WRITEBACK";
        case Action::Kind::Feed:      return "FEED";
        case Action::Kind::Drain:     return "DRAIN";
        case Action::Kind::Call:      return "CALL";
        case Action::Kind::Release:   return "RELEASE";
        case Action::Kind::Inherit:   return "INHERIT";
        case Action::Kind::Retain:    return "RETAIN";
    }
    return "?";
}

// Where an action moves a tile from and to (Call and Release move nothing).
inline Chan from_chan(Action::Kind k) {
    switch (k) {
        case Action::Kind::Load:      return Chan::Dram;
        case Action::Kind::Store:     return Chan::L3;
        case Action::Kind::Move:      return Chan::L3;
        case Action::Kind::Writeback: return Chan::L2;
        case Action::Kind::Feed:      return Chan::L2;
        case Action::Kind::Drain:     return Chan::Cf;
        default:                      return Chan::Cf;
    }
}
inline Chan to_chan(Action::Kind k) {
    switch (k) {
        case Action::Kind::Load:      return Chan::L3;
        case Action::Kind::Store:     return Chan::Dram;
        case Action::Kind::Move:      return Chan::L2;
        case Action::Kind::Writeback: return Chan::L3;
        case Action::Kind::Feed:      return Chan::Cf;
        case Action::Kind::Drain:     return Chan::L2;
        default:                      return Chan::Cf;
    }
}

struct Process {
    ProcessKind kind = ProcessKind::Dma;
    std::string name;
    std::vector<std::size_t> actions;   // indices into CspProgram::actions, in order
};

// A bounded buffer between processes. capacity = its credits, in tiles; 0 = unbounded (DRAM),
// or a transit buffer level 1 makes no residency decision about (L2, the fabric).
struct Channel {
    Chan id = Chan::Dram;
    std::string name;
    std::size_t capacity = 0;
};

// One tile's stay in L3: opened by a Load, a Writeback or an Inherit, closed by its Release --
// or by a Retain, when it outlives the program.
struct Residency {
    TileCoord tile;
    std::size_t open = kNone;           // action index of the Load / Writeback that opened it
    std::size_t release = kNone;        // action index of its Release
    std::size_t consumers = 0;          // Moves and Stores that read it
    bool loaded = false;                // opened by a Load (a DRAM read)
    bool dirty = false;                 // holds a result DRAM does not have yet
    bool inherited = false;             // opened by an Inherit (no DRAM read)
    bool retained = false;              // closed by a Retain: held when the program ends
};

// What the lowering decided about reuse, before any simulation.
struct ReuseReport {
    std::size_t loads = 0, stores = 0, moves = 0, calls = 0;
    std::size_t distinct_loaded = 0;    // tiles read from DRAM at least once
    std::size_t reloads = 0;            // loads - distinct_loaded: what capacity cost
    std::size_t peak_l3 = 0;            // most residencies open at once
};

class CspProgram {
public:
    std::string name;
    TileProgram source;                 // the L0 it was lowered from (operands, ops, values)
    std::vector<Process> processes;     // level 1: dma, bm, str, cf
    std::vector<Channel> channels;      // indexed by Chan
    std::vector<Action> actions;        // every action, in program order
    std::vector<Residency> residencies;

    const Channel& channel(Chan c) const { return channels.at(static_cast<std::size_t>(c)); }

    ReuseReport reuse() const {
        ReuseReport r;
        std::vector<std::string> seen;
        std::size_t open = 0;
        for (const Action& a : actions) {
            switch (a.kind) {
                case Action::Kind::Load:
                    ++r.loads;
                    ++open;
                    if (std::find(seen.begin(), seen.end(), a.tile.to_string()) == seen.end())
                        seen.push_back(a.tile.to_string());
                    break;
                case Action::Kind::Writeback:
                    if (residencies.at(a.residency).open == static_cast<std::size_t>(&a - actions.data())) ++open;
                    break;
                case Action::Kind::Store:   ++r.stores; break;
                case Action::Kind::Move:    ++r.moves; break;
                case Action::Kind::Call:    ++r.calls; break;
                case Action::Kind::Inherit: ++open; break;
                case Action::Kind::Release: --open; break;
                default: break;
            }
            r.peak_l3 = std::max(r.peak_l3, open);
        }
        r.distinct_loaded = seen.size();
        r.reloads = r.loads - r.distinct_loaded;
        return r;
    }

    // The disassembler (ADR 0002 §6: an in-memory IR needs one, and no syntax).
    std::string disassemble() const {
        std::ostringstream o;
        o << "csp program \"" << name << "\" (level 1)\n";
        for (const Channel& c : channels)
            o << "  channel " << c.name << "  capacity "
              << (c.capacity ? std::to_string(c.capacity) : std::string("unbounded")) << "\n";
        for (const Process& p : processes) o << "  process " << p.name << "  " << p.actions.size() << " actions\n";
        const ReuseReport r = reuse();
        o << "  reuse: " << r.loads << " loads of " << r.distinct_loaded << " tiles (" << r.reloads
          << " reloads), " << r.stores << " stores, " << r.moves << " moves, " << r.calls
          << " calls, peak L3 " << r.peak_l3 << "\n";
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const Action& a = actions[i];
            o << "  " << i << "  " << (a.process == kNone ? std::string("l3") : processes[a.process].name) << "  "
              << to_string(a.kind) << " " << a.tile.to_string();
            if (a.kind != Action::Kind::Call && a.kind != Action::Kind::Release && a.kind != Action::Kind::Inherit &&
                a.kind != Action::Kind::Retain)
                o << "  " << to_string(from_chan(a.kind)) << "->" << to_string(to_chan(a.kind));
            if (a.l0_op != kNone) o << "  [L0 " << a.l0_op << " " << to_string(source.ops()[a.l0_op].kind) << "]";
            for (std::size_t k = 0; k < a.context.size(); ++k) {
                const Stage& st = a.context[k];
                o << (k ? ", " : "  via ") << to_string(st.op);
                if (st.op == VeOp::Add) o << "(" << st.arg.to_string() << ")";
                o << " @ " << to_string(st.place);
            }
            if (a.residency != kNone && a.kind == Action::Kind::Release)
                o << "  (" << residencies[a.residency].consumers << " consumers)";
            o << "\n";
        }
        return o.str();
    }
};

}  // namespace sw::kpu::program::csp
