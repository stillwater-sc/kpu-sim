// ============================================================================
// include/sw/kpu/program/record/memory_flow_record.hpp
// The memory-flow record (.mflow): every burst, every DRAM command, every request, store-buffer
// occupancy and the port stubs' events of a run of the memory side
// (docs/plans/memory-side-debugger.md §3.3, step 3; it carries docs/plans/dram-bank-model.md
// step 4).
//
// A .mflow bundle is a directory in the .tflow container style (record/columnar.hpp):
//
//   manifest.json   format "kpu-mflow", version, source, device, makespan, window, write_window,
//                   timing note,
//                   ceiling_bytes_per_cycle, burst_bytes, dram_timing{ticks_per_cycle, params},
//                   stations, and tables{name: {file, rows, columns[{name, dtype, offset}]}}
//   bursts.bin      one row per DRAM burst
//   commands.bin    one row per DRAM command (ACT, RD, WR, PRE, REF)
//   requests.bin    one row per tile request (a DMA read, or an ejection written to DRAM)
//   buffers.bin     one row per store-buffer change
//   ports.bin       one row per port-stub event
//   waits.bin       one row per interval a cause held a DMA engine up (version 4)
//   lod.json/.bin   the pyramid (record/tile_flow_lod.hpp) over the stations below
//
// Times are executor cycles, as f64 (exact to 2^53). A command also carries its points in the
// controller's clock (k_issue, k_end, k_data0, k_data1; u64 ticks): an executor cycle is several
// ticks, too coarse to check DRAM timing against. An index that names nothing is kNone.
//
// STATIONS, the pyramid's rows, in this order: one per bank (kind dram_bank: its commands'
// windows), one per channel data bus (dram_bus: its bursts' data windows), one per DMA engine
// (dma: its bursts in flight, capacity = the window, or window + write_window when the engine has
// both), one per store buffer (dmabuf: its stores
// from slot to retirement, capacity = its depth), one per engine that records its waits (dmawait:
// the intervals it was held up, capacity = the number of causes), one per port stub (port: its
// loads from acceptance to landing).
//
// VERSIONS. read_mflow reads versions 3 and 4. A version-3 bundle has no waits, its requests'
// kinds follow is_load, and it names no ticket or slot.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/record/tile_flow_record.hpp>   // Cycle, RecordError

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sw::kpu::program::record {

// Version 2: commands carry their controller-clock ticks, and the manifest the timing table.
// Version 3: bursts carry t_posted (the engine posted it; t_submit is the controller's grant) and
// the manifest the controllers' arbitration.
// Version 4 (docs/plans/cycle-mflow-and-debug-server.md §3.3): the manifest names its source and
// a write window; a request carries its kind (Load, Store or Fill), its program action (ticket) and
// its L3 slot; and a waits table records why each DMA engine was held up, and when.
inline constexpr std::uint32_t kMflowVersion = 4;
inline constexpr std::uint32_t kMflowOldestVersion = 3;   // the oldest read_mflow reads
inline constexpr std::uint32_t kNone = 0xFFFFFFFFu;

struct MemoryFlowRecord {
    struct Station {
        std::string name, kind;             // dram_bank | dram_bus | dma | dmabuf | port
        std::uint64_t capacity = 0;
    };
    // Kinds, as stored: DRAM commands and page outcomes match MemoryControllerProcess.
    enum class CommandKind : std::uint8_t { Activate, Read, Write, Precharge, Refresh };
    enum class Outcome : std::uint8_t { Unknown, Hit, Empty, Conflict };
    enum class PortKind : std::uint8_t { Offer, Accept, Refuse, Land, EjectArrive, EjectWait, EjectDeliver };
    // A Fill puts a program's bytes in an L3 slot with no DRAM traffic: no bursts.
    enum class RequestKind : std::uint8_t { Load, Store, Fill };
    // Why a DMA engine did not advance something (the cycle core's DmaWait, in its order).
    enum class WaitReason : std::uint8_t { SlotCredit, L3Credit, RequestCredit, Window, StoreBuffer, Order };
    static constexpr std::uint8_t kWaitReasons = 6;

    struct Burst {
        std::uint32_t engine = 0, request = kNone;
        std::uint8_t mc = 0, channel = 0, rank = 0, bank_group = 0, bank = 0;
        std::uint32_t row = 0, col = 0;
        bool is_load = true;
        Outcome outcome = Outcome::Unknown;
        Cycle submitted = 0, first_command = 0, data_start = 0, data_end = 0, done = 0;
        Cycle posted = 0;                   // posted by its engine; `submitted` is the grant
    };
    struct Command {
        std::uint8_t mc = 0, channel = 0, bank_group = 0, bank = 0;
        std::uint32_t row = 0;
        CommandKind kind = CommandKind::Activate;
        std::uint32_t burst = kNone;        // index into bursts (a refresh has none)
        Cycle issue = 0, end = 0, data_start = 0, data_end = 0;
        // The same points in the controller's clock ticks (exact; DRAM timing is checked on them).
        std::uint64_t tick = 0, tick_end = 0, tick_data_start = 0, tick_data_end = 0;
    };
    struct Request {
        std::uint32_t engine = 0, port = 0;
        bool is_load = true;
        std::uint64_t address = 0;
        std::uint32_t bytes = 0;
        Cycle offered = 0, credit = 0, first_burst = 0, last_burst = 0, retired = 0;
        // Version 4. A Store is exactly a request that is not is_load; a Fill is_load (it lands in L3).
        RequestKind kind = RequestKind::Load;
        std::uint32_t ticket = kNone;       // the program action it executes
        std::uint32_t slot = kNone;         // the L3 slot it fills or drains
    };
    // One run of consecutive cycles [from, to) in which `reason` held an engine up. Several reasons
    // can hold at once, so two reasons' intervals may overlap; one reason's never do.
    struct Wait {
        std::uint32_t engine = 0;
        WaitReason reason = WaitReason::SlotCredit;
        Cycle from = 0, to = 0;
    };
    struct BufferSample { std::uint32_t engine = 0; Cycle t = 0; std::uint32_t held = 0, staged = 0; };
    struct PortEvent { std::uint32_t port = 0, engine = 0, request = kNone; PortKind kind{}; Cycle t = 0; };

    std::string source = "memory-side-harness";   // the model that ran: memory-side-harness | cycle-core
    std::string device;
    Cycle makespan = 0;
    std::uint32_t window = 0;               // bursts per engine (with write_window: its reads)
    std::uint32_t write_window = 0;         // its writes in flight; 0 = they share `window`
    std::string timing_note;                // the DRAM timing table's provenance
    double ceiling_bytes_per_cycle = 0;     // every controller's data buses at full rate (0 = unknown)
    std::uint32_t burst_bytes = 0;          // one DRAM burst
    std::string arbitration = "round_robin";    // how controllers grant engines: round_robin | fixed
    std::uint32_t grant_quantum = 1;        // round_robin: bursts per engine per turn
    double ticks_per_cycle = 0;             // controller clock ticks per executor cycle
    std::vector<std::pair<std::string, std::uint32_t>> dram_timing;   // name -> controller ticks
    std::vector<Station> stations;
    std::vector<Burst> bursts;
    std::vector<Command> commands;
    std::vector<Request> requests;
    std::vector<BufferSample> buffers;
    std::vector<PortEvent> ports;
    std::vector<Wait> waits;

    // The station index of a kind's i-th instance (banks by (mc, ch, bg, ba) in order).
    std::uint32_t station(const std::string& name) const;
};

const char* to_string(MemoryFlowRecord::CommandKind k);
const char* to_string(MemoryFlowRecord::Outcome o);
const char* to_string(MemoryFlowRecord::PortKind k);
const char* to_string(MemoryFlowRecord::RequestKind k);
const char* to_string(MemoryFlowRecord::WaitReason r);

// Write the bundle (and its pyramid) into `dir`, creating it. Throws RecordError.
void write_mflow(const MemoryFlowRecord& rec, const std::string& dir);
// Read a bundle back (version 3 or 4). Throws RecordError on a malformed or unknown-version bundle.
MemoryFlowRecord read_mflow(const std::string& dir);

} // namespace sw::kpu::program::record
