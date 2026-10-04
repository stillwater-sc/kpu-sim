// ============================================================================
// include/sw/kpu/program/record/tile_flow_record.hpp
// The tile-flow RECORD (#286 step 2; docs/plans/tile-flow-debugger.md §3.2-3.3): what a run
// did, as events placed in space and time, written as a columnar `.tflow` bundle the viewer
// loads straight into typed arrays.
//
// THE RECORD IS NOT THE VIEW. It is built from a RunOutcome and nothing else, so a viewer, an
// invariant checker and a diff between runs all read the same thing, and none of them reads
// the executor.
//
// WHAT v0 CAN SAY, AND WHAT IT SAYS IT CANNOT (L-T1):
//
//   L3          a POOLED station. L-T1 models L3 as one capacity, so a residency interval says
//               a tile held A slot, never which L3 tile or which slot. The station is named
//               `<dev>/l3[*]` and flagged pooled; it is not a naming-map address (plan Q2).
//   L2, L1      UNMODELLED stations, pooled, present only as the endpoints of the BlockMover
//               and Streamer legs that cross them. The manifest lists them as unmodelled, and
//               the viewer must draw them so, never as empty (plan §3.6).
//   compute     one station per compute tile of the placement, `<dev>/cf[c]`: naming-map
//               addresses, so they land on the floorplan.
//   moves       one transit per hop, with its mover and lane. A lane in a pool is not yet a
//               mover in a place: binding a BlockMover hop to `l3[t]/bm[e]` needs L3 slot
//               binding first (plan step 6).
//
// Not in v0, by design and named so: cause edges (step 5) and the orchestrator's descriptor
// trace, which needs a time base across launches and arrives with the orchestrated record.
//
// THE FILE (`run.tflow/`):
//
//   manifest.json      format, version, level, device, deployment digest, makespan, unmodelled
//                      stations, foreign slots, the station and tile tables, and for each binary
//                      table its file, row count and columns (name, dtype, byte offset)
//   residency.bin      columns tile:u32 station:u32 t0:f64 t1:f64 flags:u8
//   transit.bin        columns tile:u32 op:u32 hop:u8 mover:u8 lane:u32 t0:f64 t1:f64 src:u32 dst:u32
//   compute.bin        columns op:u32 station:u32 t0:f64 t1:f64
//   op_tiles.bin       CSR of the tiles each op touches: kind:u8 offset:u32 (ops+1), tile:u32 (nnz)
//
// Little-endian, each column contiguous and 8-byte aligned. Cycles are written as f64, which is
// exact up to 2^53 -- the writer refuses a run longer than that rather than round it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/driver/execution_level.hpp>
#include <sw/kpu/program/placement.hpp>
#include <sw/kpu/program/platform/deployment_spec.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw::kpu::program::record {

class RecordError : public std::runtime_error {
public:
    explicit RecordError(const std::string& what) : std::runtime_error(what) {}
};

struct Station {
    std::string name;            // a naming-map address, or `<dev>/l3[*]` for a pooled station
    std::string kind;            // dram | l3 | l2 | l1 | cf
    std::uint64_t capacity = 0;  // tiles; 0 = unbounded or not modelled
    bool pooled = false;         // stands for every resource of its kind at once
    bool modelled = true;        // false: the level does not model it, draw as such
};

struct Tile {
    std::string operand;
    Dim ti = 0, tj = 0;
    std::string key() const;     // tile_key(): "A#0#1"
};

struct Op {
    std::uint8_t kind = 0;                 // TileOpKind
    std::vector<std::uint32_t> tiles;      // every tile it reads or writes, deduplicated
};

struct Residency {
    enum : std::uint8_t { Seeded = 1, Held = 2 };
    std::uint32_t tile = 0, station = 0;
    Cycle t0 = 0, t1 = 0;
    std::uint8_t flags = 0;
};

struct Transit {
    std::uint32_t tile = 0, op = 0;
    std::uint8_t hop = 0;        // Hop
    std::uint8_t mover = 0;      // Mover
    std::uint32_t lane = 0;
    Cycle t0 = 0, t1 = 0;
    std::uint32_t src = 0, dst = 0;   // stations
};

struct Compute {
    std::uint32_t op = 0, station = 0;
    Cycle t0 = 0, t1 = 0;
};

struct TileFlowRecord {
    std::string level;                 // short name of the level that ran
    std::string device;                // device name in the deployment
    std::string device_label;          // DeviceDescriptor::label()
    std::string deployment_digest;
    Cycle makespan = 0;
    std::uint64_t foreign_slots = 0;   // L3 slots the caller held that this program never names
    std::vector<std::string> unmodelled;   // station kinds this level does not model

    std::vector<Station> stations;
    std::vector<Tile> tiles;
    std::vector<Op> ops;
    std::vector<Residency> residency;
    std::vector<Transit> transits;
    std::vector<Compute> computes;

    // Index of a station by name; throws RecordError when absent.
    std::uint32_t station(const std::string& name) const;
};

// Build the record of one run of `prog` at a level that models resources. Throws RecordError
// for a level with no intervals (L-B), naming it. `foreign_slots` is the run's
// TileExecutionRequest::foreign_held_slots, which the outcome does not carry.
TileFlowRecord build_record(const TileProgram& prog, const driver::RunOutcome& outcome,
                            const platform::DeploymentSpec& spec, const Placement& placement,
                            Dim device = 0, std::uint64_t foreign_slots = 0);

// Write `dir` (created if needed) as a .tflow bundle; read one back. Byte-deterministic: the
// same record writes the same files.
void write_tflow(const TileFlowRecord& rec, const std::string& dir);
TileFlowRecord read_tflow(const std::string& dir);

// L3 occupancy at cycle t, counted from the residency intervals plus the foreign slots. What
// the invariant checker (step 3) compares against capacity, and what a test compares against
// the executor's own peak.
std::uint64_t l3_occupancy_at(const TileFlowRecord& rec, Cycle t);
// The maximum of l3_occupancy_at over the run, computed by a sweep over interval ends.
std::uint64_t peak_l3_occupancy(const TileFlowRecord& rec);

} // namespace sw::kpu::program::record
