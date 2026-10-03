// ============================================================================
// include/sw/kpu/orchestration/kpu_device.hpp
// The MACHINE side of the call ABI (#305 increment 3).
//
// Increment 2's orchestrator decided from the inside of the machine: it was handed the
// TensorStore and the VirtualPlatform, kept the credit ledger as its own bookkeeping, and
// promised in a comment not to read tensor contents. A guest under Renode (increment 4)
// cannot be held to a comment. So everything the orchestrator must not touch lives HERE,
// behind one entry point, and the orchestrator reaches it only through a `KpuPort`:
//
//   KpuDevice owns   the data plane (TensorStore), the platform, the credit ledger, the
//                    reservation table, and the per-operator manifests
//   the decider sees descriptors, completions, status counts, manifests -- METADATA
//
// ---------------------------------------------------------------------------
// RESERVE-THEN-LAUNCH, and the four rules that keep it deadlock-free (plan §3.3)
// ---------------------------------------------------------------------------
//   R1  A RESERVE for operator j is granted only if every uncompleted i < j already holds a
//       granted reservation. Otherwise: ReservationOutOfOrder, blocking_op = i.
//   R2  A RESERVE is granted only if slots <= free (capacity - held - reserved). Otherwise:
//       InsufficientCredit. It is NEVER queued -- a block is a hang, a refusal a decision.
//   R3  A PLACE must name an operator holding a granted reservation. A PLACE made AHEAD of an
//       earlier operator's launch (a prefetch) must also fit inside that reservation, because
//       those tiles occupy slots while the earlier operator runs.
//   R4  A LAUNCH of j needs j's reservation, every i < j complete, and
//       peak_live_tiles(j) + |retained| <= reserved(j). The unused remainder returns at
//       completion.
//
// Why that cannot deadlock: by R1 the earliest uncompleted operator e either holds a
// reservation or no later reservation exists. If it holds one, the reservation is a
// SUFFICIENT bound for its run (the executor's program-order proof, applied inside the run),
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
// It is reachable from KpuDevice and from the harness that fills inputs and reads outputs.
// It is NOT reachable from the orchestrator: as of increment 3 no orchestrator entry point
// takes one (§3 of the parent plan).
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

// Which TENSOR each of an operator's OPERANDS is, derived from the L0 program's structure
// and the loadable's declared order.
//
// Positional, as the schema says: the operator's `inputs` line up with the operands the
// program READS, in first-appearance order, and its `outputs` with the operands it only
// writes. Positional and therefore checkable -- a count mismatch is a refusal, because a
// loadable that names two inputs for a three-operand kernel does not describe a run.
std::map<std::string, std::string> operand_binding(const program::TileProgram& prog,
                                                   const loadable::Operator& op);

struct DeviceOptions {
    // TEST-ONLY ABLATION. With reservations off the device counts one slot per PLACE and
    // enforces nothing about order -- the greedy allocator the plan's §3.3 WRONG example
    // describes. It exists so a test can show that rule wedging, and so the device's diagnosis
    // is shown to stay accurate when the rule it exists to enforce is switched off: it must
    // REFUSE naming the operator holding the credits, never hang.
    bool enforce_reservations = true;
};

// ----------------------------------------------------------------------------
// The device
// ----------------------------------------------------------------------------
class KpuDevice {
public:
    KpuDevice(const loadable::Loadable& l, program::platform::VirtualPlatform& platform,
              TensorStore& tensors, ExecutionLevel level, DeviceOptions dopt = {});

    // THE one entry point. Every descriptor produces exactly one completion, now or -- for a
    // RELEASE at last read -- when the launch it governs completes. Never a block.
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
    const std::vector<RunOutcome>& outcomes() const { return outcomes_; }

private:
    struct ReadTile {
        TileRef tensor_tile;            // what the loadable calls it
        std::string operand_key;        // what the program calls it, in the executor's spelling
    };
    struct Op {
        OperatorManifest manifest;
        program::TileProgram prog;
        std::map<std::string, std::string> binding;   // operand -> tensor
        std::vector<ReadTile> reads;                  // by operand key, may repeat a tensor tile
        bool reserved = false;
        std::uint32_t reservation = 0;
        bool completed = false;
        std::set<std::string> pending;                // tensor keys PLACEd for it, pre-launch
        std::map<std::string, TileRef> pending_refs;
    };

    void post(Completion c) { completions_.push_back(std::move(c)); }
    void refuse(const Descriptor& d, CompletionStatus st, RefusalCause cause, std::string why,
                std::uint32_t needed = 0, std::uint32_t available = 0,
                std::uint32_t blocking = kNoOperator);
    std::uint32_t free_slots() const;                 // under reservations
    std::uint32_t occupied_ablation() const;          // one slot per PLACE, ablation only
    long earliest_uncompleted() const;
    bool find_op(const std::string& name, std::uint32_t& out) const;

    void do_reserve(const Descriptor& d);
    void do_place(const Descriptor& d);
    void do_release(const Descriptor& d);
    void do_launch(const Descriptor& d);

    const loadable::Loadable& l_;
    program::platform::VirtualPlatform& platform_;
    TensorStore& tensors_;
    ExecutionLevel level_;
    DeviceOptions dopt_;
    std::uint32_t capacity_ = 0;

    std::vector<Op> ops_;
    std::set<std::string> held_;                      // tensor keys held across launches
    std::map<std::string, TileRef> held_refs_;
    // RELEASEs at last read, waiting for the launch that reads them. Ordered by descriptor id
    // so their completions post deterministically.
    std::map<std::uint64_t, TileRef> pending_release_;
    std::deque<Completion> completions_;
    std::vector<RunOutcome> outcomes_;
};

} // namespace sw::kpu::orchestration
