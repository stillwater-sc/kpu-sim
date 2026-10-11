// ============================================================================
// include/sw/kpu/orchestration/kpu_device.hpp
// The MACHINE side of the call ABI (#305 increment 3), running CSP operators
// (kpu-run-csp-programs step 4d.2).
//
// Increment 2's orchestrator decided from the inside of the machine: it was handed the
// TensorStore and the VirtualPlatform, kept the credit ledger as its own bookkeeping, and
// promised in a comment not to read tensor contents. A guest under Renode (increment 4)
// cannot be held to a comment. So everything the orchestrator must not touch lives HERE,
// behind one entry point, and the orchestrator reaches it only through a `KpuPort`:
//
//   KpuDevice owns   the data plane (TensorStore = DRAM, and the values of the tiles held in
//                    L3 across launches), the platform, the credit ledger, the reservation
//                    table, and the per-operator manifests
//   the decider sees descriptors, completions, status counts, manifests -- METADATA
//
// ---------------------------------------------------------------------------
// RESIDENCY IS THE PROGRAM'S (step 4d.2)
// ---------------------------------------------------------------------------
// An operator is a CSP program. It loads its own tiles, and residency across operators is
// written in it: `inherit X;` opens it with X resident (no Load), `retain X;` ends a residency
// with no Release, leaving the slot and the value for a later operator. So the device holds a
// tile across launches exactly when a program retained it, until a program inherits it.
//
// The CHAIN is checked when the device is built, by tile in the LOADABLE's vocabulary (a tensor
// tile), through each operator's binding; the first operator it fails is marked invalid, and the
// orchestrator refuses when it reaches it:
//   - an `inherit` names a tile an earlier operator retains, and no operator in between claims
//     it, with the same tiling on both sides;
//   - every retained tile is inherited by a later operator: a retained tile need not be stored,
//     so one nothing claims would be a result that never reaches DRAM;
//   - no operator in between LOADS a retained tile that was never stored: DRAM does not hold
//     its value yet, so the Load would read a stale one.
//
// A PLACE for an operator is refused: the program loads its own tiles, and a PLACE would be a
// DMA no program sequences. The vocabulary keeps it for L-T2 (#283), where descriptors order
// the legs inside a run.
//
// ---------------------------------------------------------------------------
// RESERVE-THEN-LAUNCH, and the four rules that keep it deadlock-free (plan §3.3)
// ---------------------------------------------------------------------------
//   R1  A RESERVE for operator j is granted only if every uncompleted i < j already holds a
//       granted reservation. Otherwise: ReservationOutOfOrder, blocking_op = i.
//   R2  A RESERVE is granted only if slots <= free (capacity - held - reserved). Otherwise:
//       InsufficientCredit. It is NEVER queued -- a block is a hang, a refusal a decision.
//   R3  A PLACE is refused (above); it returns with L-T2.
//   R4  A LAUNCH of j needs j's reservation, every i < j complete, every tile j inherits held,
//       and bound(j) = l3(j) - |inherited| <= reserved(j): the program runs under its own L3
//       credits, and its inherited slots are already held. The unused remainder returns at
//       completion.
//
// Why that cannot deadlock: by R1 the earliest uncompleted operator e either holds a
// reservation or no later reservation exists. If it holds one, the reservation is a
// SUFFICIENT bound for its run (the program's L3 is what its own validator proved it runs in),
// so e completes and returns credits. If it holds none, nothing later does either, so either
// e's RESERVE fits or the machine is too small for e given what is held -- a refusal with a
// diagnosis. Nothing waits on an unbounded condition, so the only outcomes are progress or a
// refusal, by induction on e.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/loadable/loadable.hpp>
#include <sw/kpu/orchestration/descriptor.hpp>
#include <sw/kpu/orchestration/port.hpp>
#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/platform/virtual_platform.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

using program::driver::CspLevelOutcome;
using program::driver::ExecutionLevel;

// ----------------------------------------------------------------------------
// The data plane — the machine's side, not the orchestrator's
// ----------------------------------------------------------------------------
// Tensors live OUTSIDE the operator programs, which is what the loadable's separation
// means at run time. This is DRAM: an operator's program has its own operands, which are
// views of these tensors, and copying between the two is the DMA's job modelled at this
// level. Only a tile a program STORES changes it.
//
// It is reachable from KpuDevice and from the harness that fills inputs and reads outputs.
// It is NOT reachable from the orchestrator: as of increment 3 no orchestrator entry point
// takes one (§3 of the parent plan).
class TensorStore {
public:
    void declare(const loadable::TensorRef& t);
    bool has(const std::string& name) const;
    std::vector<float>& values(const std::string& name);
    const std::vector<float>& values(const std::string& name) const;

    // Copy a TENSOR into a program's OPERAND, and one TILE back out after the launch.
    //
    // The two are named separately on purpose. A loadable's tensors are the model's -- "X",
    // "layer3.weight" -- and a program's operands are the kernel's, which for a matmul are
    // A/B/C whatever the model calls them. Treating one name as the other works only while
    // they coincide, and the first version of this did exactly that: both operators then read
    // and wrote the SAME tensor, the second accumulated onto the first's result, and C came out
    // doubled.
    //
    // Sizes must agree: a mismatch means the loadable and the program disagree about a tensor,
    // which is a refusal rather than a truncating copy.
    void load_into(program::TileProgram& prog, const std::string& operand,
                   const std::string& tensor) const;
    void store_tile(const program::TileProgram& prog, const program::TileCoord& tile,
                    const std::string& tensor);

private:
    std::map<std::string, std::vector<float>> values_;
    std::map<std::string, std::vector<std::uint64_t>> shapes_;
};

// Which TENSOR each of a program's OPERANDS is: the operator's `inputs` line up with the
// operands declared `in` or `inout`, and its `outputs` with those declared `out`, each in
// declaration order. Declarations, not first use: the program states what each operand is,
// so nothing is inferred. A count mismatch is a refusal, because a loadable that names two
// inputs for a three-input program does not describe a run; so is an operand whose shape or
// tiling differs from its tensor's, because then a tile index would name a different region
// of the tensor in each program that touches it.
std::map<std::string, std::string> operand_binding(const program::csp::lang::Program& ast,
                                                   const loadable::Operator& op,
                                                   const loadable::Loadable& l);

// ----------------------------------------------------------------------------
// The device
// ----------------------------------------------------------------------------
class KpuDevice {
public:
    KpuDevice(const loadable::Loadable& l, program::platform::VirtualPlatform& platform,
              TensorStore& tensors, ExecutionLevel level);

    // THE one entry point. Every descriptor produces exactly one completion, posted when it is
    // serviced. Never a block.
    void submit(const Descriptor& d);
    bool pop_completion(Completion& out);
    bool completions_pending() const { return !completions_.empty(); }

    StatusSnapshot status() const;
    bool is_resident(const TileRef& t) const { return held_.count(t.key()) != 0; }
    std::uint32_t operator_count() const { return static_cast<std::uint32_t>(ops_.size()); }
    const OperatorManifest& manifest(std::uint32_t op) const { return ops_.at(op).manifest; }

    const loadable::Loadable& loadable() const { return l_; }
    const program::platform::DeploymentSpec& deployment() const { return platform_.deployment(); }

    // Host-side EVIDENCE of what the launches did -- like the descriptor trace, a recording,
    // not something the orchestrator reads. Indexed in launch order.
    const std::vector<CspLevelOutcome>& outcomes() const { return outcomes_; }

private:
    // A tile in both vocabularies: what the loadable calls it, and what the program does.
    struct BoundTile {
        TileRef tensor_tile;
        program::TileCoord operand_tile;
    };
    struct Op {
        OperatorManifest manifest;
        program::csp::lang::Program ast;
        std::map<std::string, std::string> binding;   // operand -> tensor
        std::vector<BoundTile> inherits, retains;
        std::vector<BoundTile> stores;                // every tile it stores, once
        bool reserved = false;
        std::uint32_t reservation = 0;
        bool completed = false;
    };

    void post(Completion c) { completions_.push_back(std::move(c)); }
    void refuse(const Descriptor& d, CompletionStatus st, RefusalCause cause, std::string why,
                std::uint32_t needed = 0, std::uint32_t available = 0,
                std::uint32_t blocking = kNoOperator);
    std::uint32_t free_slots() const;
    long earliest_uncompleted() const;
    bool find_op(const std::string& name, std::uint32_t& out) const;
    void check_chain();

    void do_reserve(const Descriptor& d);
    void do_place(const Descriptor& d);
    void do_release(const Descriptor& d);
    void do_launch(const Descriptor& d);

    const loadable::Loadable& l_;
    program::platform::VirtualPlatform& platform_;
    TensorStore& tensors_;
    ExecutionLevel level_;
    std::uint32_t capacity_ = 0;

    std::vector<Op> ops_;
    // Tiles held in L3 across launches -- retained by one program, not yet inherited by another
    // -- by tensor key, with their VALUES: a retained tile need not have been stored, so DRAM
    // may not hold what the next program inherits.
    std::set<std::string> held_;
    std::map<std::string, TileRef> held_refs_;
    std::map<std::string, std::vector<float>> held_values_;   // the tile's elements, row-major
    std::deque<Completion> completions_;
    std::vector<CspLevelOutcome> outcomes_;
};

} // namespace sw::kpu::orchestration
