// ============================================================================
// include/sw/kpu/orchestration/descriptor.hpp
// What an orchestrator may SAY to the KPU, and what it hears back (#305 §6.2).
//
// THE RULE THIS FILE EXISTS TO MAKE CHECKABLE: no descriptor carries payload, and no
// completion returns any. An orchestrator issues descriptors and reads completions; it
// never reads or writes a tile's contents.
//
// A reviewer should be able to establish that by reading `Descriptor` below and finding
// no field that could hold a tensor element. That is deliberate — the plan's §3 states
// the rule, and a type is what enforces it. An MMIO window into buffer contents *is* a
// backdoor, and #284 owns the backdoor precisely so it stays flagged in a run's
// provenance; a second, unflagged one arriving through this ABI would silently
// invalidate every timing result that used it.
//
// HOPS DO NOT COLLAPSE. A `PLACE` names ONE leg, governed by one CSP process over one
// physical pathway. A tile reaching L1 from DRAM is three descriptors, because it is
// three moves. A single descriptor claiming DRAM→L1 would describe a machine that
// cannot be built.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/resource_map.hpp>
#include <sw/kpu/program/tile_program.hpp>
#include <sw/kpu/program/tile_transaction_executor.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

using program::Cycle;
using program::Dim;
using program::Hop;
using program::platform::ResourceName;

// ----------------------------------------------------------------------------
// A tile of a tensor
// ----------------------------------------------------------------------------
// Named the way the LOADABLE names things — a tensor name plus tile coordinates — rather
// than by the executor's internal key. `key()` converts, through the one public spelling,
// so an orchestrator never hand-builds that string. The first version of the residency
// test did hand-build it, spelled it "A[0,0]" instead of "A#0#0", matched nothing, and
// reported success.
struct TileRef {
    std::string tensor;
    Dim ti = 0, tj = 0;

    std::string key() const {
        return program::tile_key(program::TileCoord{tensor, ti, tj});
    }
    std::string str() const {
        return tensor + "[" + std::to_string(ti) + "," + std::to_string(tj) + "]";
    }
    bool operator==(const TileRef& o) const {
        return tensor == o.tensor && ti == o.ti && tj == o.tj;
    }
    bool operator<(const TileRef& o) const {
        if (tensor != o.tensor) return tensor < o.tensor;
        if (ti != o.ti) return ti < o.ti;
        return tj < o.tj;
    }
};

// ----------------------------------------------------------------------------
// The vocabulary
// ----------------------------------------------------------------------------
enum class DescriptorKind : std::uint8_t {
    Place,      // move one tile along ONE leg, into a named resource
    Release,    // return the credit for a tile
    Configure,  // load a domain-flow program into a programmable compute tile
    Launch,     // run an operator, given where its operands already are
    Fence,      // order a completion against a later descriptor
    Reserve,    // claim `slots` L3 credits for operator `target`, ALL OR NONE (#305 inc. 3)
};

// RELEASE flags. A release the device needs to know about AT LAUNCH TIME travels before the
// launch it governs: "this launch is the tile's last reader, do not keep it". Its completion
// is posted after that launch, because that is when the credit actually comes back -- so
// completion order is not issue order, deliberately.
inline constexpr std::uint32_t kReleaseAtLastRead = 1u;

const char* to_string(DescriptorKind k);

struct Descriptor {
    std::uint64_t id = 0;              // so a completion can name it
    DescriptorKind kind = DescriptorKind::Place;

    // PLACE / RELEASE
    TileRef tile;
    Hop leg = Hop::DmaDramToL3;        // ONE leg; see the header on why it cannot be two
    ResourceName resource;             // where it lands, in the #282 naming map's spelling

    // RESERVE / PLACE / LAUNCH: the operator. CONFIGURE: the domain-flow program.
    // A PLACE names its operator because a placement is made UNDER A RESERVATION, and the
    // device has to know which one (R3).
    std::string target;
    Dim compute_tile = 0;

    // FENCE
    std::uint64_t wait_for = 0;        // a descriptor id whose completion must precede this

    // RESERVE
    std::uint32_t slots = 0;           // L3 credits claimed, atomically

    // RELEASE
    std::uint32_t flags = 0;           // kReleaseAtLastRead

    // There is deliberately NO payload field, and there never will be. See the header.

    std::string str() const;
};

// ----------------------------------------------------------------------------
// What comes back
// ----------------------------------------------------------------------------
enum class CompletionStatus : std::uint8_t {
    Done,
    // THE ABI CAN SAY NO (#305 §6.4). For a static schedule, blocking on insufficient
    // credit is fine because the compiler proved the schedule fits. For a runtime
    // allocator a block is a HANG and a refusal is a DECISION POINT.
    RefusedInsufficientCredit,
    RefusedUnsupported,
};

const char* to_string(CompletionStatus s);

// WHY a descriptor was refused, as a number an orchestrator can branch on. The text in
// `Completion::diagnosis` is for a person; this is for the decider, which must tell "the
// machine is full, choose again" from "you asked out of order, which is your bug".
enum class RefusalCause : std::uint8_t {
    None,
    InsufficientCredit,        // R2: not enough free L3 slots; never queued
    ReservationOutOfOrder,     // R1/R4: an earlier operator holds no reservation / is not done
    NoReservation,             // R3/R4: a PLACE or LAUNCH for an operator holding no reservation
    ReservationExceeded,       // R3/R4: the operator needs more than it reserved
    Unsupported,               // the level, the program or the request cannot be served
};

const char* to_string(RefusalCause c);

inline constexpr std::uint32_t kNoOperator = 0xFFFFFFFFu;

struct Completion {
    std::uint64_t descriptor_id = 0;
    CompletionStatus status = CompletionStatus::Done;

    // Cycles this descriptor consumed, and whether that number means anything.
    //
    // `timed == false` is not the same as `cycles == 0`. At L-B nothing is timed; and at
    // L-T1 a PLACE has no latency of its own, because the executor schedules the legs
    // across the whole run rather than one at a time (#305 §6.2). Reporting 0 without
    // saying so would be a measurement invented out of an absence -- which is what
    // `RunOutcome::has_timing` exists to prevent one layer down.
    Cycle cycles = 0;
    bool timed = false;

    std::vector<TileRef> released;      // credits this completion returned
    std::string diagnosis;              // non-empty exactly when status != Done

    // THE ARITHMETIC OF A REFUSAL, as numbers. They cross the MMIO ABI as fixed-width fields,
    // so they are 32-bit here too: a type that held more than the wire would make the two
    // transports disagree on exactly the values a refusal is about.
    RefusalCause cause = RefusalCause::None;
    std::uint32_t needed = 0;           // slots asked for
    std::uint32_t available = 0;        // slots free when refused (0xFFFFFFFF = unbounded)
    std::uint32_t capacity = 0;         // L3 capacity in tiles (0 = unbounded)
    std::uint32_t blocking_op = kNoOperator;   // operator index that stands in the way

    std::string str() const;
};

// ----------------------------------------------------------------------------
// A recorded stream, for determinism
// ----------------------------------------------------------------------------
// ADR 0002 §3.5 says a run is a pure function of its inputs. A deciding orchestrator does
// not break that PROVIDED its decisions derive only from those inputs -- and the way to
// check a provided-that is to record what it decided and compare. Two runs of one loadable
// must produce identical bytes here; so must a host build and a cross-compiled one.
//
// This is a RECORDING, not a program. #305 §7.1 retired the idea of a static descriptor
// stream as an orchestration kind for exactly this reason: a list of descriptors is
// evidence of decisions, not the thing that makes them.
struct DescriptorTrace {
    std::vector<Descriptor> issued;
    std::vector<Completion> completions;

    std::string canonical_bytes() const;
    std::string digest() const;
};

} // namespace sw::kpu::orchestration
