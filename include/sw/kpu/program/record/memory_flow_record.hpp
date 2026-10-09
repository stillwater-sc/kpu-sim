// ============================================================================
// include/sw/kpu/program/record/memory_flow_record.hpp
// The memory-flow record (.mflow): every burst, every DRAM command, every request, store-buffer
// occupancy and the port stubs' events of a run of the memory side
// (docs/plans/memory-side-debugger.md §3.3, step 3; it carries docs/plans/dram-bank-model.md
// step 4).
//
// A .mflow bundle is a directory in the .tflow container style (record/columnar.hpp):
//
//   manifest.json   format "kpu-mflow", version, device, makespan, window, timing note,
//                   ceiling_bytes_per_cycle, burst_bytes, dram_timing{ticks_per_cycle, params},
//                   stations, and tables{name: {file, rows, columns[{name, dtype, offset}]}}
//   bursts.bin      one row per DRAM burst
//   commands.bin    one row per DRAM command (ACT, RD, WR, PRE, REF)
//   requests.bin    one row per tile request (a DMA read, or an ejection written to DRAM)
//   buffers.bin     one row per store-buffer change
//   ports.bin       one row per port-stub event
//   lod.json/.bin   the pyramid (record/tile_flow_lod.hpp) over the stations below
//
// Times are executor cycles, as f64 (exact to 2^53). A command also carries its points in the
// controller's clock (k_issue, k_end, k_data0, k_data1; u64 ticks): an executor cycle is several
// ticks, too coarse to check DRAM timing against. An index that names nothing is kNone.
//
// STATIONS, the pyramid's rows, in this order: one per bank (kind dram_bank: its commands'
// windows), one per channel data bus (dram_bus: its bursts' data windows), one per DMA engine
// (dma: its bursts in flight, capacity = the window), one per store buffer (dmabuf: its stores
// from slot to retirement, capacity = its depth), one per port stub (port: its loads from
// acceptance to landing).
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
inline constexpr std::uint32_t kMflowVersion = 3;
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
    };
    struct BufferSample { std::uint32_t engine = 0; Cycle t = 0; std::uint32_t held = 0, staged = 0; };
    struct PortEvent { std::uint32_t port = 0, engine = 0, request = kNone; PortKind kind{}; Cycle t = 0; };

    std::string device;
    Cycle makespan = 0;
    std::uint32_t window = 0;               // bursts per engine
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

    // The station index of a kind's i-th instance (banks by (mc, ch, bg, ba) in order).
    std::uint32_t station(const std::string& name) const;
};

const char* to_string(MemoryFlowRecord::CommandKind k);
const char* to_string(MemoryFlowRecord::Outcome o);
const char* to_string(MemoryFlowRecord::PortKind k);

// Write the bundle (and its pyramid) into `dir`, creating it. Throws RecordError.
void write_mflow(const MemoryFlowRecord& rec, const std::string& dir);
// Read a bundle back. Throws RecordError on a malformed or unknown-version bundle.
MemoryFlowRecord read_mflow(const std::string& dir);

} // namespace sw::kpu::program::record
