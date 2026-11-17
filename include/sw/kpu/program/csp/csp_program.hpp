// ============================================================================
// include/sw/kpu/program/csp/csp_program.hpp
// The CSP program (ADR 0002 §2.1; #281; docs/plans/csp-program-tile-sequencing.md §3.1).
//
// The sequencing layer every execution level interprets: PROCESSES, each with the ordered tile
// actions it performs, exchanging tiles over CHANNELS whose capacities are their credits. It is
// the compiled form of a written program (csp/lang/compile.hpp; ADR 0004), and its source is the
// program's L0 trace. An L0 is never turned back into a program: it is a trace, not a source
// (docs/plans/retire-l0-derivations.md §3.4).
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
// tile has found a bug in the program, or in the compiler that produced the actions.
//
// PLACEMENT IS THE PROGRAM'S TOO (ADR 0004 §5; docs/plans/l3-placement-at-lb.md). Every
// residency has an L3 address -- module, slot, bank -- fixed when it opens and freed when it is
// released, allocated once by the walker (csp/lang/place.hpp). Every level executes that address;
// none allocates a slot of its own.
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
#include <optional>
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
// accumulator, before the Drain), `combiner` (the Combiner on the L1 -> L2 result path,
// docs/01-architecture/combiner.md) and `bm.egress` (the BlockMover's L2 -> L3 writeback), in
// that order. `bm.ingress` (L3 -> L2) is an operand's way in. `str.drain` is the Combiner's
// name before #429, accepted as a deprecated alias for one release.
enum class Place : std::uint8_t { Fabric, Combiner, BmEgress, BmIngress };

inline const char* to_string(Place p) {
    switch (p) {
        case Place::Fabric:    return "fabric";
        case Place::Combiner:  return "combiner";
        case Place::BmEgress:  return "bm.egress";
        case Place::BmIngress: return "bm.ingress";
    }
    return "?";
}

inline bool parse_place(const std::string& s, Place& out) {
    for (Place p : {Place::Fabric, Place::Combiner, Place::BmEgress, Place::BmIngress})
        if (s == to_string(p)) { out = p; return true; }
    if (s == "str.drain") { out = Place::Combiner; return true; }   // deprecated (#429)
    return false;
}

// A vector operation: the epilogue (decision Q4). add takes a vector tile, broadcast down the
// rows; relu, gelu, silu and atan are the activations; mxint8 and mxfp8 are the block-format
// conversions into the OCP Microscaling formats (#427; numeric::block_convert), the last of a
// result's operations (the Combiner's order: bias, scale, activation, format).
// scale(m, shift) is the requantization (#427): an accumulator's values into its tensor's type,
// round(x * m / 2^shift), ties to even, saturated (numeric::requantize).
// Init is not written in a program: the walker puts it first on an L2 accumulator's first merge,
// `acc .. in l2 init X` (#427): the Combiner adds X, read from L3, to the merged partial.
enum class VeOp : std::uint8_t { Add, Relu, Gelu, Silu, Atan, MxInt8, MxFp8, Scale, Init };
inline constexpr std::size_t kVeOps = 8;

// A block-format conversion: its format's name (numeric.hpp); else empty.
inline const char* block_format_of(VeOp v) {
    return v == VeOp::MxInt8 ? "mxint8" : v == VeOp::MxFp8 ? "mxfp8" : "";
}
inline bool is_block_format(VeOp v) { return v == VeOp::MxInt8 || v == VeOp::MxFp8; }

inline const char* to_string(VeOp v) {
    switch (v) {
        case VeOp::Add:  return "add";
        case VeOp::Relu: return "relu";
        case VeOp::Gelu: return "gelu";
        case VeOp::Silu: return "silu";
        case VeOp::Atan: return "atan";
        case VeOp::MxInt8: return "mxint8";
        case VeOp::MxFp8: return "mxfp8";
        case VeOp::Scale: return "scale";
        case VeOp::Init: return "init";
    }
    return "?";
}

inline bool parse_veop(const std::string& s, VeOp& out) {
    for (VeOp v : {VeOp::Add, VeOp::Relu, VeOp::Gelu, VeOp::Silu, VeOp::Atan, VeOp::MxInt8, VeOp::MxFp8, VeOp::Scale})
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

// Where a residency lives in L3: as a register names where a CPU operand lives (tile_program.hpp).
using sw::kpu::program::L3Address;
using sw::kpu::program::L2Address;

// The L3 a program addresses: its slots and its interleaved banks. 0 leaves that bound unchecked
// (a caller with no program in hand); slots and banks out of the geometry are refused.
struct L3Geometry {
    std::uint32_t slots = 0;
    std::uint32_t banks = 0;
};

struct Stage {
    VeOp op = VeOp::Relu;
    Place place = Place::Fabric;
    TileCoord arg;                      // add: the vector tile (resident in L3); a per-channel scale: its multipliers'
    TileCoord arg2;                     // a per-channel scale: its shifts' vector tile
    bool channels = false;              // scale: per channel, from arg and arg2 (#427), not m and shift
    std::int64_t m = 1;                 // scale: the multiplier
    int shift = 0;                      //        and the right shift
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
//   L2Release (l2 credit)       a `stage` block ends: the staged tile's L2 slot is free
// (Inherit and Retain are kpu-run-csp-programs step 4d: residency across operators.)
//
// L2 staging (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 3): a Move that carries an
// L2 address opens a STAGING -- the tile's copy in that L2 slot, kept for the block -- and a
// Feed that carries one reads the staged copy and leaves it there. Unstaged Moves and Feeds
// carry none: the tile passes through L2 once, as before.
//
// L2 ACCUMULATORS (#429, #430; docs/01-architecture/combiner.md): `acc T in l2 { ... }` holds each
// result tile's partial in an L2 slot the walker allocates, and the Combiner merges every call's
// result into it. A call into one Drains a fresh partial from the fabric with `merge` set and the
// slot as its L2 address: Overwrite for the tile's first call (the slot opens; it is a staging
// whose `partial` is set), Accumulate for every later one (a read-modify-write in L2). The
// store's Writeback reads the slot (carrying its address), and its L2Release frees it.
enum class Merge : std::uint8_t { None, Overwrite, Accumulate };

// A reduction's dataflow on the systolic array (#395): which operand the PEs hold. Default: the
// deployment's `fabric.dataflow` (os when it declares none). Output-stationary holds C and drains
// it north when the reduction ends; weight- and input-stationary hold B or A (PRELOAD), stream the
// other, and each call's partial sums leave south, merged in L2 by the Combiner.
enum class Dataflow : std::uint8_t { Default, OS, WS, IS };

inline const char* to_string(Dataflow d) {
    switch (d) {
        case Dataflow::Default: return "default";
        case Dataflow::OS:      return "os";
        case Dataflow::WS:      return "ws";
        case Dataflow::IS:      return "is";
    }
    return "?";
}
inline bool parse_dataflow(const std::string& s, Dataflow& out) {
    for (Dataflow d : {Dataflow::OS, Dataflow::WS, Dataflow::IS})
        if (s == to_string(d)) { out = d; return true; }
    return false;
}

inline const char* to_string(Merge m) {
    switch (m) {
        case Merge::None:       return "none";
        case Merge::Overwrite:  return "overwrite";
        case Merge::Accumulate: return "accumulate";
    }
    return "?";
}

struct Action {
    enum class Kind : std::uint8_t { Load, Store, Move, Writeback, Feed, Drain, Call, Release, Inherit, Retain, L2Release };
    Kind kind = Kind::Call;
    TileCoord tile;                     // the tile moved, released, or the Call's first output
    std::size_t process = kNone;        // index into CspProgram::processes (Release: none)
    std::size_t l0_op = kNone;          // the L0 op this action implements (traceability)
    std::size_t residency = kNone;      // the L3 residency it opens, reads or releases
    bool accumulate = false;            // Call: adds to the fabric's accumulator for its output,
                                        // which starts at zero (an output-stationary chain)
    std::vector<Stage> context;         // applied to the tile as this action moves it, in order:
                                        // Drain (fabric, combiner), Writeback (bm.egress),
                                        // Move (bm.ingress)
    // The L3 slot the action writes (Load, Writeback, Inherit), reads (Move, Store) or frees
    // (Release, Retain): its residency's address. None for an action with no residency.
    std::optional<L3Address> place;
    // The L2 slot a staging Move writes, a staged Feed reads, or an L2Release frees; and which
    // staging it is (CspProgram::stagings). None on an unstaged action.
    std::optional<L2Address> l2;
    std::size_t staging = kNone;
    // Drain into an L2 accumulator: how the Combiner merges the partial into its slot.
    Merge merge = Merge::None;
    // A Call, and the Drain of its result: its accumulator's dataflow (`acc ... dataflow ws`; #395).
    Dataflow dataflow = Dataflow::Default;
    // Feed: the tile's values start this accumulator in the fabric (`acc Y in fabric init X`;
    // docs/plans/accumulator-init-stream.md): the init is one more edge stream, fed before the
    // accumulator's first call, and its values (a vector's broadcast down the rows) are the
    // accumulator's from then on. None on every other action.
    std::optional<TileCoord> seeds;
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
        case Action::Kind::L2Release: return "L2_RELEASE";
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
    L3Address place;                    // where it lives, from open to close
    bool pinned = false;                // the program named its slot (`@ slots(...)`)
};

// One tile's stay in an L2 slot: opened by a staging Move, read by staged Feeds, closed by its
// L2Release at the end of its `stage` block.
//
// A partial's stay in an L2 slot (an L2 accumulator, `partial`): opened by its first merge (a
// Drain, Overwrite), merged into by the rest, read by its store's Writeback, closed by its
// L2Release. It has no L3 residency.
struct Staging {
    TileCoord tile;
    std::size_t open = kNone;           // action index of the Move that staged it (a partial: its first Drain)
    std::size_t release = kNone;        // action index of its L2Release
    std::size_t feeds = 0;              // Feeds of the staged copy: its reuse count
    std::size_t residency = kNone;      // the L3 residency it was copied from (live throughout)
    L2Address place;
    bool partial = false;               // an L2 accumulator's partial, which the Combiner merges into
    std::size_t merges = 0;             // a partial: the Drains merged into it, the first included
};

// What the program decided about reuse, before any simulation.
struct ReuseReport {
    std::size_t loads = 0, stores = 0, moves = 0, calls = 0;
    std::size_t stagings = 0, staged_feeds = 0;   // L2: copies staged, and Feeds they served
    std::size_t partials = 0, merges = 0;         // L2 accumulators: partials held, Drains merged into them
    std::size_t peak_l2 = 0;            // most stagings open at once
    std::size_t distinct_loaded = 0;    // tiles read from DRAM at least once
    std::size_t reloads = 0;            // loads - distinct_loaded: what capacity cost
    std::size_t peak_l3 = 0;            // most residencies open at once
};

class CspProgram {
public:
    std::string name;
    TileProgram source;                 // its L0 trace (operands, ops, values)
    std::vector<Process> processes;     // level 1: dma, bm, str, cf
    std::vector<Channel> channels;      // indexed by Chan
    std::vector<Action> actions;        // every action, in program order
    std::vector<Residency> residencies;
    std::vector<Staging> stagings;
    std::uint32_t l2_slots = 0;         // the machine clause's `l2 = N`: slots per compute tile's L2
    // Placement (ADR 0004 §5): the L3's interleaved banks and the policy the walker placed by.
    std::uint32_t l3_banks = 1;
    std::string place = "fifo";

    const Channel& channel(Chan c) const { return channels.at(static_cast<std::size_t>(c)); }

    ReuseReport reuse() const {
        ReuseReport r;
        std::vector<std::string> seen;
        std::size_t open = 0, staged = 0;
        for (const Action& a : actions) {
            if (a.l2 && a.kind == Action::Kind::Move) { ++r.stagings; r.peak_l2 = std::max(r.peak_l2, ++staged); }
            if (a.merge == Merge::Overwrite) { ++r.partials; r.peak_l2 = std::max(r.peak_l2, ++staged); }
            if (a.merge != Merge::None) ++r.merges;
            if (a.l2 && a.kind == Action::Kind::Feed) ++r.staged_feeds;
            if (a.kind == Action::Kind::L2Release) --staged;
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
          << " calls, peak L3 " << r.peak_l3;
        if (r.stagings) o << ", " << r.stagings << " staged in L2 for " << r.staged_feeds << " feeds";
        if (r.partials) o << ", " << r.partials << " partials in L2 for " << r.merges << " merges";
        if (r.stagings || r.partials) o << ", peak L2 " << r.peak_l2;
        o << "\n";
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const Action& a = actions[i];
            o << "  " << i << "  " << (a.process == kNone ? std::string("l3") : processes[a.process].name) << "  "
              << to_string(a.kind) << " " << a.tile.to_string();
            if (a.kind != Action::Kind::Call && a.kind != Action::Kind::Release && a.kind != Action::Kind::Inherit &&
                a.kind != Action::Kind::Retain && a.kind != Action::Kind::L2Release)
                o << "  " << to_string(from_chan(a.kind)) << "->" << to_string(to_chan(a.kind));
            if (a.l0_op != kNone) o << "  [L0 " << a.l0_op << " " << to_string(source.ops()[a.l0_op].kind) << "]";
            for (std::size_t k = 0; k < a.context.size(); ++k) {
                const Stage& st = a.context[k];
                o << (k ? ", " : "  via ") << to_string(st.op);
                if (st.op == VeOp::Add || st.op == VeOp::Init) o << "(" << st.arg.to_string() << ")";
                if (st.op == VeOp::Scale && st.channels) o << "(" << st.arg.to_string() << ", " << st.arg2.to_string() << ")";
                else if (st.op == VeOp::Scale) o << "(" << st.m << ", " << st.shift << ")";
                o << " @ " << to_string(st.place);
            }
            if (a.place) o << "  @" << a.place->to_string();
            if (a.merge != Merge::None) o << "  merge " << to_string(a.merge) << " @ combiner";
            if (a.l2) o << "  @" << a.l2->to_string();
            if (a.residency != kNone && a.kind == Action::Kind::Release)
                o << "  (" << residencies[a.residency].consumers << " consumers)";
            o << "\n";
        }
        return o.str();
    }
};

}  // namespace sw::kpu::program::csp
