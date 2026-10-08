// ============================================================================
// include/sw/kpu/timing/dram_bridge.hpp
// The cycle-accurate LPDDR5 controller, hosted on the executor's clock
// (docs/plans/dram-bank-model.md step 2).
//
// WHY A BRIDGE. The CSP executor's MemoryControllerProcess timed a whole tile as one request,
// never read its own data-bus state, and decoded addresses with a hard-coded mapping (§1.4).
// The repo already has a bank-level controller -- LPDDR5MemoryController, with bank groups,
// tRRD/tCCD/tFAW, per-bank refresh, per-channel data buses and FR-FCFS, invariant-checked by
// patterns/memory/lpddr5 -- so rather than grow a second bank model that would drift from it,
// the process hosts it. The bridge adapts it without making it know about deployments:
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
// Hosting it exposed defects in the controller itself -- reads that never pipelined, FCFS on
// the queue head, unchecked tRRD_S/tCCD_S, and refresh that skipped open banks -- fixed in the
// controller (docs/plans/dram-bank-model.md §1.4 correction), not worked around here.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/platform/dram_address_map.hpp>
#include <sw/kpu/timing/tile_descriptor.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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

    // One DRAM command, in executor cycles and the spec's coordinates
    // (docs/plans/memory-side-debugger.md §3.1). `tag` is the burst's submit() tag; a refresh has
    // none, and a precharge carries its page opener's tag while that burst is still in flight.
    struct Command {
        enum class Kind : std::uint8_t { Activate, Read, Write, Precharge, Refresh };
        Kind kind = Kind::Activate;
        unsigned mc = 0, channel = 0, bank_group = 0, bank = 0;
        std::uint64_t row = 0, col = 0;
        Cycle issue = 0, end = 0;
        Cycle data_start = 0, data_end = 0;     // Read/Write: the data-bus window
        std::optional<std::uint64_t> tag;
        bool activated = false, conflicted = false;     // Read/Write: the page outcome
    };
    // Called for every command the controller issues. Unset = no cost.
    void set_command_sink(std::function<void(const Command&)> sink);

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
