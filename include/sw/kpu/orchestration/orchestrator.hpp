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
//   sees        a StatusView -- inventory, occupancy, credits, completions. METADATA.
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
// The data plane — the machine's side, not the orchestrator's
// ----------------------------------------------------------------------------
// Tensors live OUTSIDE the operator programs, which is what the loadable's separation
// means at run time. An operator's L0 program has its own operand arrays; this holds the
// tensors those arrays are views of, and copying between the two is the DMA's job modelled
// at this level.
//
// It is deliberately NOT reachable from the orchestrator (§3). The orchestrator decides
// which tiles move; this is what moving them does.
class TensorStore {
public:
    void declare(const loadable::TensorRef& t);
    bool has(const std::string& name) const;
    std::vector<float>& values(const std::string& name);
    const std::vector<float>& values(const std::string& name) const;

    // Copy a TENSOR into an operator program's OPERAND, and back out after the launch.
    //
    // The two are named separately on purpose. A loadable's tensors are the model's -- "A",
    // "layer3.weight" -- and an L0 program's operands are the kernel's, which for a derived
    // matmul are always A/B/C whatever the model calls them. Treating one name as the other
    // works only while they coincide, and the first version of this did exactly that: both
    // operators then read and wrote the SAME tensor, the second accumulated onto the first's
    // result, and C came out doubled.
    //
    // Shapes must agree: a mismatch means the loadable and the L0 program disagree about a
    // tensor, which is a refusal rather than a truncating copy.
    void load_into(program::TileProgram& prog, const std::string& operand,
                   const std::string& tensor) const;
    void store_from(const program::TileProgram& prog, const std::string& operand,
                    const std::string& tensor);

private:
    std::map<std::string, std::vector<float>> values_;
    std::map<std::string, std::vector<std::uint64_t>> shapes_;
};

// ----------------------------------------------------------------------------
// Options and result
// ----------------------------------------------------------------------------
struct OrchestratorOptions {
    ExecutionLevel level = ExecutionLevel::BlockSequential;
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
};

// Which TENSOR each of an operator's OPERANDS is, derived from the L0 program's structure
// and the loadable's declared order.
//
// Positional, as the schema says: the operator's `inputs` line up with the operands the
// program READS, in first-appearance order, and its `outputs` with the operands it only
// writes. Positional and therefore checkable -- a count mismatch is a refusal, because a
// loadable that names two inputs for a three-operand kernel does not describe a run.
std::map<std::string, std::string> operand_binding(const program::TileProgram& prog,
                                                   const loadable::Operator& op);

// Run a loadable. Decides placement as it goes; records every decision in the trace.
OrchestrationResult orchestrate(const loadable::Loadable& l,
                                program::platform::VirtualPlatform& platform,
                                TensorStore& tensors,
                                const OrchestratorOptions& opt = {});

} // namespace sw::kpu::orchestration
