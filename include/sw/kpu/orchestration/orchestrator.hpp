// ============================================================================
// include/sw/kpu/orchestration/orchestrator.hpp
// The thing that DECIDES (#305 increment 2).
//
// A loadable says which operators to run and where the tensors are. Since step 4d.2 of
// kpu-run-csp-programs each operator is a CSP program, and the program says which tiles it
// loads, keeps, inherits from the operator before it and retains for the one after: residency
// is the program's. What is left to decide at run time is ADMISSION -- when to claim an
// operator's L3 and launch it, and what to do when the machine is full (#305 §10 Q4) -- and
// this is what decides it.
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
// launching it. `ReserveThenLaunch` also reserves the NEXT operator when the machine can take
// it, before the current launch -- so whether k+1 fits is decided while k runs.
//
// The PLACEs increment 3 issued ahead of a launch are gone with step 4d.2: a program loads its
// own tiles, so a PLACE would be a DMA no program sequences. The device refuses one.
//
// THE ORCHESTRATOR NO LONGER HOLDS THE MACHINE. It reaches the device through a KpuPort and
// nothing else; the TensorStore and the VirtualPlatform are the device's (kpu_device.hpp).
//
// ---------------------------------------------------------------------------
// WHAT AN ORCHESTRATOR DECIDES, exactly
// ---------------------------------------------------------------------------
// **Admission, and nothing below it.** Every move is sequenced by an operator's program, under
// its own L3 credits; the orchestrator issues RESERVE and LAUNCH. The vocabulary in
// descriptor.hpp is wider than that on purpose -- it is the ABI, and L-T2 (#283) gives PLACE
// back a meaning, ordering legs inside a run. Stating the gap here keeps the next increment
// from inheriting it as an assumption.
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

// ----------------------------------------------------------------------------
// Options and result
// ----------------------------------------------------------------------------
enum class AllocationPolicy : std::uint8_t {
    ProgramOrder,        // reserve operator k just before launching it (increment 2's rule)
    ReserveThenLaunch,   // also reserve k+1 when it fits, before LAUNCH k
};

const char* to_string(AllocationPolicy p);

enum class Transport : std::uint8_t { Direct, Mmio };

struct OrchestratorOptions {
    ExecutionLevel level = ExecutionLevel::BlockSequential;
    AllocationPolicy policy = AllocationPolicy::ProgramOrder;
    Transport transport = Transport::Direct;
};

struct OrchestrationResult {
    DescriptorTrace trace;
    std::vector<program::driver::CspLevelOutcome> per_operator;
    std::vector<std::string> operator_names;

    // A REFUSAL IS A RESULT, not an exception. A runtime allocator that cannot satisfy a
    // reservation has made a discovery, and the orchestrator reports it rather than
    // hanging or throwing past its caller.
    bool refused = false;
    std::string diagnosis;

    // What no level here could represent, passed up rather than dropped.
    std::vector<std::string> unmodelled;

    std::size_t dma_transfers() const;     // DRAM loads, summed over operators: the reuse proof

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
// The loadable is read for METADATA only -- operator names -- which
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
