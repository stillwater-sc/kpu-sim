// ============================================================================
// include/sw/kpu/timing/dram_bridge.hpp
// The cycle-accurate LPDDR5 controller, hosted on the executor's clock
// (docs/plans/dram-bank-model.md step 2).
//
// WHY A BRIDGE. The CSP executor's MemoryControllerProcess timed a whole tile as one request,
// never read its own data-bus state, and decoded addresses with a hard-coded mapping (§1.4).
// The repo already has a bank-faithful controller -- LPDDR5MemoryController, with bank groups,
// tRRD/tFAW, per-bank refresh and a per-channel data bus, validated by patterns/memory/lpddr5
// -- so rather than grow a second bank model that would drift from it, the process hosts it.
// The bridge is what makes that possible without editing the validated model:
//
//   address   the deployment's DramAddressMap decodes every burst; the bridge re-encodes the
//             coordinates into the controller's own fixed layout, so the spec's map (and its
//             XOR folds) is the one that decides banks
//   clock     the controller counts in its DRAM clock (data rate / 2); the executor in its
//             reference clock. A rate accumulator ticks the controller the right number of
//             times per executor cycle, so nanoseconds agree in both
//   timing    the controller's table is LPDDR5-6400. A different data rate scales the
//             nanosecond-fixed parameters to the faster clock and keeps the burst-relative
//             ones -- DERIVED, not a datasheet table, and named so in timing_note()
//
// The controller's scheduler is FCFS on the head of its queue, not FR-FCFS: a request waiting
// on its bank blocks the ones behind it. That is the hosted model's behavior, stated rather
// than hidden; FR-FCFS is a change to the controller, not to this bridge.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/dram_address_map.hpp>
#include <sw/kpu/timing/tile_descriptor.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sw::kpu::timing {

// What the executor needs to host a declared DRAM: the address map plus the two facts the map
// does not carry. Built from a deployment's memory.dram; refused (DramMapError) when the
// device declares none or one the hosted controller cannot model.
struct DramHosting {
    program::platform::DramAddressMap map;
    std::string technology;
    unsigned data_rate_mtps = 0;
    unsigned channel_width_bits = 0;

    static DramHosting of(const program::platform::DeviceSpecification& d);
};

class DramBridge {
public:
    struct Stats {
        std::uint64_t bursts = 0;           // submitted
        std::uint64_t page_hits = 0, page_empty = 0, page_conflicts = 0;
        std::uint64_t refreshes = 0;
        std::uint64_t stall_ticks = 0;      // controller ticks the head could not issue
        std::uint64_t misrouted = 0;        // bursts whose address names another controller
        std::uint64_t violations = 0;       // the controller's own invariant checker
    };

    // `controller_id` is this controller's index in the map's mc field. `queue_depth` bounds the
    // controller's request queue; submit() refuses past it (back-pressure, not loss).
    DramBridge(const DramHosting& h, unsigned controller_id, double executor_clock_ghz,
               std::uint32_t queue_depth = 32);
    ~DramBridge();
    DramBridge(const DramBridge&) = delete;
    DramBridge& operator=(const DramBridge&) = delete;

    // One burst at `address` (any byte inside it). False when the controller's queue is full.
    // Throws DramMapError for an address past the top of memory.
    bool submit(std::uint64_t address, bool is_load, std::uint64_t tag);

    // Bring the controller up to executor cycle `now` (inclusive), appending the tags of every
    // burst that completed. Cycles may be skipped; the accumulator is over elapsed time.
    void advance(Cycle now, std::vector<std::uint64_t>& completed);

    bool busy() const;
    void reset();

    double ticks_per_cycle() const { return ticks_per_cycle_; }
    const program::platform::DramAddressMap& map() const { return hosting_.map; }
    const std::string& timing_note() const { return timing_note_; }
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    DramHosting hosting_;
    unsigned controller_id_;
    double ticks_per_cycle_;
    std::string timing_note_;
};

} // namespace sw::kpu::timing
