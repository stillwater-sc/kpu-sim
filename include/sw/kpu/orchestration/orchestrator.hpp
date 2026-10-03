// ============================================================================
// include/sw/kpu/orchestration/orchestrator.hpp
// The thing that DECIDES (#305 increment 2).
//
// A loadable says which operators to run and where the tensors are. It does not say which
// tiles to keep in L3, when to give a slot back, or what to do when the machine is full.
// Those are runtime decisions (#305 §10 Q4), and this is what makes them.
//
// WHAT IT MAY TOUCH, and what it may not:
//
//   sees        a KpuPort -- status counts, manifests, completions. METADATA.
//   issues      Descriptors -- none of which carries payload.
//   never       a tensor element. Not one.
//
// The data plane belongs to the EXECUTOR below, which is the simulated machine's side of
// the ABI: it is the DMA that moves bytes, and modelling that is not the orchestrator's
// business. Splitting the two is how §3's rule survives contact with an implementation
// that has to move data somehow.
//
// ---------------------------------------------------------------------------
// ALLOCATION: PROGRAM-ORDER ACQUISITION, and why it is inherited rather than invented
// ---------------------------------------------------------------------------
// The L-T1 executor is deadlock-free because new slots are acquired in PROGRAM ORDER. That
// was not an aesthetic choice: the weaker rule -- "do not promote a ready op past an
// earlier ready one" -- wedges at every finite capacity below the unbounded run's greedy
// peak, measured at 29 tiles for the derived matmul, failing at 25 with a true live set of
// 21.
//
// A runtime allocator is exactly the thing that can reproduce that wedge. This one cannot,
// because it completes each operator before starting the next: there is never an earlier
// unfired operator holding slots while a later one waits. The existing proof therefore
// applies VERBATIM, which is the whole reason increment 2 uses this rule and increment 3
// buys more freedom deliberately, with a new argument, rather than by assumption.
//
// INCREMENT 3 makes the rule the DEVICE's to enforce rather than the orchestrator's to keep:
// every operator's slots are claimed by an atomic RESERVE, granted in operator order (R1-R4
// in kpu_device.hpp, with the proof). `ProgramOrder` reserves each operator just before
// placing for it, which is increment 2's behaviour restated as reservations.
// `ReserveThenLaunch` also reserves the NEXT operator when the machine can take it, and
// places for it BEFORE the current launch -- an order program-order acquisition forbids.
//
// THE ORCHESTRATOR NO LONGER HOLDS THE MACHINE. It reaches the device through a KpuPort and
// nothing else; the TensorStore and the VirtualPlatform are the device's (kpu_device.hpp).
//
// ---------------------------------------------------------------------------
// WHAT L-T1 LETS AN ORCHESTRATOR DECIDE, exactly
// ---------------------------------------------------------------------------
// **L3 residency, and nothing below it.** At L-T1 the unit of transaction is a tile move
// inside a run: the executor schedules the L3→L2 and L2→L1 legs itself, under credits,
// across the whole program. So the descriptors this orchestrator issues are `PLACE` on the
// DMA leg and `RELEASE`; the BlockMover and Streamer legs are not its to order.
//
// The vocabulary in descriptor.hpp is wider than that on purpose -- it is the ABI, and
// L-T2 (#283) makes the other legs real. Stating the gap here keeps increment 3 from
// inheriting it as an assumption, the same way the plan states that a `PLACE` has no
// latency of its own at this level.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/loadable/loadable.hpp>
#include <sw/kpu/orchestration/descriptor.hpp>
#include <sw/kpu/orchestration/kpu_device.hpp>
#include <sw/kpu/orchestration/port.hpp>
#include <sw/kpu/orchestration/status.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>

#include <map>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

using program::driver::ExecutionLevel;
using program::driver::RunOutcome;

// ----------------------------------------------------------------------------
// Options and result
// ----------------------------------------------------------------------------
enum class AllocationPolicy : std::uint8_t {
    ProgramOrder,        // reserve operator k just before placing for it (increment 2's rule)
    ReserveThenLaunch,   // also reserve k+1 when it fits, and place for it before LAUNCH k
    // TEST-ONLY: the WRONG shape of plan §3.3 -- place k+1's tiles, then k's, with no
    // reservations at all. Paired with DeviceOptions::enforce_reservations = false it shows
    // the hold-and-wait wedge, which the device must REFUSE rather than hang on.
    GreedyPrefetch,
};

const char* to_string(AllocationPolicy p);

enum class Transport : std::uint8_t { Direct, Mmio };

struct OrchestratorOptions {
    ExecutionLevel level = ExecutionLevel::BlockSequential;
    AllocationPolicy policy = AllocationPolicy::ProgramOrder;
    Transport transport = Transport::Direct;
    DeviceOptions device{};
    // Keep a tile resident when a later operator will read it. This is the decision that
    // makes the KPU stateful: with it off, every operator starts cold and the machine's
    // storage hierarchy is decoration.
    bool reuse_shared_inputs = true;
};

struct OrchestrationResult {
    DescriptorTrace trace;
    std::vector<RunOutcome> per_operator;
    std::vector<std::string> operator_names;

    // A REFUSAL IS A RESULT, not an exception. A runtime allocator that cannot satisfy a
    // reservation has made a discovery, and the orchestrator reports it rather than
    // hanging or throwing past its caller.
    bool refused = false;
    std::string diagnosis;

    // What no level here could represent, passed up rather than dropped.
    std::vector<std::string> unmodelled;

    std::size_t dma_transfers() const;     // summed over operators, for the reuse proof

    // MMIO transport only: every orchestrator-side bus access, digested. Two runs whose tensor
    // VALUES differ must produce the same digest -- the non-interference form of "no status
    // read carries payload" (plan §3.7). Empty and zero on the direct transport.
    std::string bus_log_digest;
    std::size_t bus_accesses = 0;
};

// THE DECIDER. Reaches the machine through `port` alone; it is handed no TensorStore and no
// platform. Fills `trace`, `operator_names`, `refused` and `diagnosis`; what the launches did
// (`per_operator`, `unmodelled`) is the device's evidence and is filled by the caller.
//
// The loadable is read for METADATA only -- operator names, which tensors each reads -- which
// is what an RV guest reading its own FlatBuffers tables in place will see. Tensor data is
// external to it by construction.
OrchestrationResult run_orchestrator(const loadable::Loadable& l, KpuPort& port,
                                     const OrchestratorOptions& opt = {});

// Run a loadable end to end: build the device, the chosen transport, and run the decider.
// Kept with increment 2's signature, so every caller from then is unchanged.
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,
                                const OrchestratorOptions& opt = {});

} // namespace sw::kpu::orchestration
