// ============================================================================
// include/sw/kpu/orchestration/port.hpp
// The ONLY way an orchestrator reaches the machine (#305 increment 3).
//
// One orchestrator, one machine, two transports:
//
//   DirectPort   function calls into a KpuDevice -- the reference
//   MmioPort     descriptor ring, doorbell, completion ring and status registers over a
//                logged bus (mmio.hpp) -- what a RISC-V guest will drive in increment 4
//
// Both reach the SAME KpuDevice, so "identical through MMIO" is a differential test and not
// an assertion: any divergence between the two descriptor traces is a transport bug by
// definition.
//
// What crosses a port is METADATA: descriptors, completions, counts and tile names. Nothing
// here returns or accepts a tensor element, and the MMIO transport proves that dynamically
// (tensor DRAM faults on its bus, and two runs with different tensor values produce
// byte-identical bus logs).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/orchestration/descriptor.hpp>
#include <sw/kpu/program/platform/resource_map.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::orchestration {

// ----------------------------------------------------------------------------
// What the orchestrator may know about an operator
// ----------------------------------------------------------------------------
// Derived by the DEVICE from the L0 program at load time (plan Q2), so the orchestrator never
// parses L0 -- a freestanding RV64 guest should not carry a text parser to learn which tiles
// an operator reads. Every field is a tile NAME or a COUNT; none is a value.
struct OperatorManifest {
    std::uint32_t index = 0;
    bool valid = false;
    std::string error;                  // why it is not valid; empty when it is
    std::vector<TileRef> reads;         // distinct TENSOR tiles read, first-appearance order
    std::uint32_t peak_live_tiles = 0;  // characterize::peak_live_tiles(prog)

    // The SUFFICIENT reservation for a run that keeps `retained` of its read tiles past their
    // last reader (plan §3.4). Seeded tiles are live tiles of this program, so they are already
    // inside peak_live_tiles; retained tiles extend a lifetime past its last reader, so each can
    // raise the in-run live set by at most one.
    std::uint32_t bound(std::size_t retained) const {
        return peak_live_tiles + static_cast<std::uint32_t>(retained);
    }
};

// What the orchestrator may SEE of the machine's credit state. Counts, never contents.
struct StatusSnapshot {
    std::uint32_t l3_capacity = 0;      // 0 = unbounded, matching DeviceSpecification
    std::uint32_t held = 0;             // tiles held across launches (caller-owned)
    std::uint32_t reserved = 0;         // slots under outstanding reservations
    std::uint32_t operators = 0;
    std::uint32_t completed = 0;        // operators whose launch has completed
    bool unbounded() const { return l3_capacity == 0; }
    std::uint32_t credits_free() const {
        if (unbounded()) return 0xFFFFFFFFu;
        const std::uint32_t used = held + reserved;
        return l3_capacity > used ? l3_capacity - used : 0;
    }
};

// ----------------------------------------------------------------------------
// The port
// ----------------------------------------------------------------------------
class KpuPort {
public:
    virtual ~KpuPort() = default;

    // Hand the machine a descriptor. On the MMIO transport this writes a ring entry and rings
    // the doorbell; the device services it before the call returns (plan §3.6), which is
    // deterministic and needs no clock. Increment 4 replaces the trigger, not the protocol.
    virtual void submit(const Descriptor& d) = 0;

    // The next completion, if one is available. False means none is pending -- and since the
    // device never blocks, an orchestrator waiting on a descriptor that has no completion has
    // found a protocol bug, which it reports rather than spinning.
    virtual bool poll_completion(Completion& out) = 0;

    virtual StatusSnapshot read_status() = 0;
    virtual bool is_resident(const TileRef& t) = 0;
    virtual std::uint32_t operator_count() = 0;
    virtual OperatorManifest manifest(std::uint32_t op) = 0;
    // What exists: the #282 naming map's resources. Identity, not state.
    virtual std::vector<program::platform::ResourceName> inventory() = 0;
};

class KpuDevice;

class DirectPort final : public KpuPort {
public:
    explicit DirectPort(KpuDevice& dev) : dev_(dev) {}
    void submit(const Descriptor& d) override;
    bool poll_completion(Completion& out) override;
    StatusSnapshot read_status() override;
    bool is_resident(const TileRef& t) override;
    std::uint32_t operator_count() override;
    OperatorManifest manifest(std::uint32_t op) override;
    std::vector<program::platform::ResourceName> inventory() override;

private:
    KpuDevice& dev_;
};

} // namespace sw::kpu::orchestration
