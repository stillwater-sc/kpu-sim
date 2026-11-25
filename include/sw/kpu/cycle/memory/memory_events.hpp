#pragma once
// ============================================================================
// include/sw/kpu/cycle/memory/memory_events.hpp
// The memory subsystem's events, for a recorder (docs/plans/cycle-mflow-and-debug-server.md §3.1,
// step 1): what the DMA engine and the LPDDR5 channel report, at the moment it happens.
//
// A component holds a MemoryEvents* that is null by default: with none attached, reporting costs
// one null test. A sink OBSERVES. It is told what happened and never asked anything, so attaching
// one cannot change what a run does (the recorder's test runs a program with and without one and
// compares everything).
//
// Times are picoseconds, the cycle core's time: the reporter's edge, Domain::edge(now). A DRAM
// command is reported as the controller's own record, in its clock's cycles (the channel's domain
// ticks the controller once a cycle, so they are the domain's cycles).
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#include <sw/kpu/cycle/memory/l3_program.hpp>
#include <sw/kpu/cycle/time.hpp>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>

#include <cstddef>
#include <cstdint>

namespace sw::kpu::cycle {

// Why a DMA engine did not advance something in a cycle; several can hold at once (dma_engine.hpp).
enum class DmaWait : std::uint8_t { SlotCredit, L3Credit, RequestCredit, Window, StoreBuffer, Order };
inline constexpr std::size_t kDmaWaits = 6;

inline const char* to_string(DmaWait w) {
    switch (w) {
        case DmaWait::SlotCredit: return "no slot credit";
        case DmaWait::L3Credit: return "no L3 link credit";
        case DmaWait::RequestCredit: return "no DRAM request credit";
        case DmaWait::Window: return "burst window full";
        case DmaWait::StoreBuffer: return "store buffer full";
        case DmaWait::Order: return "an earlier command's burst to the same address";
    }
    return "?";
}

struct MemoryEvents {
    using DramCommand = lpddr5::LPDDR5MemoryController::CommandRecord;

    virtual ~MemoryEvents() = default;

    // The run's clocks, before its first cycle: the KPU's (the DMA engine's) and DRAM's.
    virtual void clocks(const Domain& /*kpu*/, const Domain& /*dram*/) {}

    // ---- a DMA engine, `engine` ----
    // A program action (Load, Store or Fill) started: in the program's order, with its slot's
    // credit for a Load or a Fill. `ticket` is the action's index in the program.
    virtual void action_started(std::uint32_t /*engine*/, Picoseconds, std::size_t /*ticket*/, const L3Action&) {}
    // Its last burst (a Fill: its last beat) is done.
    virtual void action_completed(std::uint32_t /*engine*/, Picoseconds, std::size_t /*ticket*/) {}
    // A burst sent to DRAM: `request` is the DramRequest's id, unique per engine.
    virtual void burst_posted(std::uint32_t /*engine*/, Picoseconds, std::size_t /*ticket*/, std::uint64_t /*request*/,
                              std::uint64_t /*address*/, bool /*write*/) {}
    // A burst left the engine's window: a read's beat pushed to L3, a write's acknowledgment taken.
    virtual void burst_retired(std::uint32_t /*engine*/, Picoseconds, std::uint64_t /*request*/) {}
    // The causes that held something up this cycle, a bit per DmaWait; reported only when one did.
    virtual void dma_waits(std::uint32_t /*engine*/, Picoseconds, std::uint32_t /*mask*/) {}
    // The store buffer's occupancy after it changed, in beats.
    virtual void store_buffer(std::uint32_t /*engine*/, Picoseconds, std::size_t /*beats*/) {}

    // ---- an LPDDR5 channel, `channel` ----
    // A request accepted into the controller, which named it `controller_id`.
    virtual void dram_accepted(std::uint32_t /*channel*/, Picoseconds, std::uint64_t /*controller_id*/,
                               std::uint64_t /*request*/, bool /*write*/, std::uint64_t /*address*/) {}
    // A command the controller issued: its own record (controller cycles; request_id is the
    // controller's; a PRECHARGE names the request that opened the page, a REFRESH none).
    virtual void dram_command(std::uint32_t /*channel*/, const DramCommand&) {}
};

}  // namespace sw::kpu::cycle
