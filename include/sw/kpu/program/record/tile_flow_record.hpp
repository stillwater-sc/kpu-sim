// ============================================================================
// include/sw/kpu/program/record/tile_flow_record.hpp
// The tile-flow RECORD (#286 step 2; docs/plans/tile-flow-debugger.md §3.2-3.3): what a run
// did, as events placed in space and time, written as a columnar `.tflow` bundle the viewer
// loads straight into typed arrays.
//
// THE RECORD IS NOT THE VIEW. It is built from a run's outcome and nothing else, so a viewer, an
// invariant checker and a diff between runs all read the same thing, and none of them reads
// the executor.
//
// WHAT v0 CAN SAY, AND WHAT IT SAYS IT CANNOT (L-T1):
//
//   L3          a POOLED station across the physical L3 tiles, named `<dev>/l3[*]` and flagged
//               pooled; it is not a naming-map address (plan Q2). From version 5 each residency
//               also carries the L3 ADDRESS it held (module, slot, bank): the program's
//               placement, executed by L-T1 (ADR 0004 §5; l3-placement-at-lb step 5). A slot is
//               an address in the program's L3, not a physical L3 tile, so the floorplan stays
//               pooled; the swimlanes draw one row per slot (tile_flow_lod.hpp).
//   L2, L1      UNMODELLED stations, pooled, present only as the endpoints of the BlockMover
//               and Streamer legs that cross them. The manifest lists them as unmodelled, and
//               the viewer must draw them so, never as empty (plan §3.6).
//   compute     one station per compute tile of the placement, `<dev>/cf[c]`: naming-map
//               addresses, so they land on the floorplan.
//   moves       one transit per hop, with its mover and lane. A lane in a pool is not yet a
//               mover in a place: binding a BlockMover hop to `l3[t]/bm[e]` needs L3 slot
//               binding first (plan step 6).
//
// VERSION 4 (docs/plans/kpu-run-csp-programs.md step 4b): the record of a CSP PROGRAM's run at
// L-T1. The columns are version 3's; what an "op" is changes, and the manifest says so
// (`"ops": "csp-actions"`): op i is the program's i-th ACTION, its kind an Action::Kind (Load,
// Store, Move, Writeback, Feed, Drain, Call, Release). Each movement action is one transit (a
// Store two: the ejection, then the DMA write), each Call one compute, and the residency
// intervals are the program's own slots, credit to Release. Versions 1-3 were the records of an
// L0 program's run at L-T1; the executor that made them retired (kpu-run-csp-programs step 4e),
// so they are read and no longer written.
//
// VERSION 5 (docs/plans/l3-placement-at-lb.md step 5b): a CSP program's record whose residencies
// carry their L3 address -- residency.bin gains l3_tile:u32 slot:u32 bank:u32, and the manifest
// `l3_banks`. Version 4 is still read; its residencies have no address (kNoSlot).
//
// VERSION 6 (docs/plans/l2-reuse-and-fabric-pipeline-view.md step 4): two tables more, and the
// manifest's `l2_slots` (the program's `l2 = N`; 0 when it stages nothing).
//   l2_residency  each STAGED copy's L2 slot, credit (its stage Move) to L2Release, with the
//                 Feeds that read it -- its reuse count, recorded, never computed by a reader.
//                 With l2_slots > 0 the L2 station is modelled: its capacity is the program's
//                 slots and its occupancy the staged copies. A transient Move's copy (an unstaged
//                 operand, fed once) holds no L2 credit at L-T1 and is not a residency.
//   chain         each accumulator's gemm chain on the systolic array (L-T1's fabric pipeline):
//                 its compute tile, its PEs, and its phases in0, front, in1, last, out1.
// The pyramid gains a row per L2 slot after the L3 slot rows, then two rows per compute tile:
// `<cf>/in`, the chains' operand streams [in0, in1), and `<cf>/out`, their results [in1, out1).
// Version 5 is still read; its records have neither table.
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
//                      (version 5) l3_tile:u32 slot:u32 bank:u32, the L3 address it held
//   transit.bin        columns tile:u32 op:u32 hop:u8 mover:u8 lane:u32 t0:f64 t1:f64 src:u32 dst:u32
//   compute.bin        columns op:u32 station:u32 t0:f64 t1:f64
//   l2_residency.bin   (version 6) columns tile:u32 cf:u32 slot:u32 t0:f64 t1:f64 feeds:u32
//   chain.bin          (version 6) columns tile:u32 station:u32 rows:u32 cols:u32 calls:u32
//                      in0:f64 front:f64 in1:f64 last:f64 out1:f64
//   op_tiles.bin       CSR of the tiles each op touches: kind:u8 offset:u32 (ops+1), tile:u32 and
//                      written:u8 (nnz; 1 when the op writes the tile, so a residency's filler
//                      can be told from a mere reader)
//
// Little-endian, each column contiguous and 8-byte aligned. Cycles are written as f64, which is
// exact up to 2^53 -- the writer refuses a run longer than that rather than round it.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/lang/parse.hpp>
#include <sw/kpu/program/driver/csp_run.hpp>
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
    std::vector<std::uint8_t> written;     // parallel to `tiles`: 1 if the op writes it
};

inline constexpr std::uint32_t kNoSlot = 0xFFFFFFFFu;   // a residency of a record with no addresses

struct Residency {
    enum : std::uint8_t { Seeded = 1, Held = 2 };
    std::uint32_t tile = 0, station = 0;
    Cycle t0 = 0, t1 = 0;
    std::uint8_t flags = 0;
    // The L3 address it held (version 5); kNoSlot in a record without placement.
    std::uint32_t l3_tile = kNoSlot, slot = kNoSlot, bank = kNoSlot;
};

// A staged copy's L2 slot (version 6): held from its stage Move's credit to its L2Release.
struct L2Residency {
    std::uint32_t tile = 0;
    std::uint32_t cf = 0, slot = 0;    // the L2 address: the compute tile's L2, and the slot in it
    Cycle t0 = 0, t1 = 0;
    std::uint32_t feeds = 0;           // the Feeds that read it: its reuse count
};

// An accumulator's gemm chain on the systolic array (version 6): L-T1's phases, uncalibrated.
struct Chain {
    std::uint32_t tile = 0;            // the accumulator
    std::uint32_t station = 0;         // the compute tile's station
    std::uint32_t rows = 0, cols = 0;  // the PEs its C tile occupies
    std::uint32_t calls = 0;
    Cycle in0 = 0, front = 0, in1 = 0, last = 0, out1 = 0;
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

// An operand's shape, so a consumer can lay the tensor out in an address space and say which
// bytes a tile is (the derived DRAM map of the viewer). The record states shape only; where
// a tensor lives in DRAM is not modelled at L-T1 and is not claimed here.
struct OperandShape {
    std::string name;
    Dim rows = 0, cols = 0, tile_rows = 0, tile_cols = 0;
};

struct MoverPool {
    std::string name;                  // to_string(Mover): dma | block-mover | streamer | noc
    Dim lanes = 0;
};

struct TileFlowRecord {
    std::string op_space = "l0";       // what an op is: "l0" (version 3) or "csp-actions" (4)
    std::string level;                 // short name of the level that ran
    std::string device;                // device name in the deployment
    std::string device_label;          // DeviceDescriptor::label()
    std::string deployment_digest;
    Cycle makespan = 0;
    std::uint64_t foreign_slots = 0;   // L3 slots the caller held that this program never names
    std::uint32_t l3_banks = 0;        // the program's L3 banks; 0: residencies carry no address
    std::vector<std::string> unmodelled;   // station kinds this level does not model
    bool phases = false;               // version 6: the l2_residency and chain tables are present
    std::uint32_t l2_slots = 0;        // version 6: the program's L2 slots; 0 when it stages nothing

    std::vector<Station> stations;
    std::vector<MoverPool> movers;     // lane count per pool: the capacity of a mover row
    std::vector<OperandShape> operands;    // program order
    Dim element_bytes = 0;
    std::vector<Tile> tiles;
    std::vector<Op> ops;
    std::vector<Residency> residency;
    std::vector<Transit> transits;
    std::vector<Compute> computes;
    std::vector<L2Residency> l2_residency;   // version 6
    std::vector<Chain> chains;               // version 6

    // Index of a station by name; throws RecordError when absent.
    std::uint32_t station(const std::string& name) const;
};

// Build the record of a CSP program's run at L-T1 (version 4): `outcome` is the L-T1 outcome of
// `ast` on `operands` (driver::csp_run_level), timed on device `device` of `spec`. Throws
// RecordError for an outcome that is not an L-T1 run.
TileFlowRecord build_csp_record(const csp::lang::Program& ast, const driver::CspLevelOutcome& outcome,
                                const TileProgram& operands, const platform::DeploymentSpec& spec, Dim device = 0);

// Write `dir` (created if needed) as a .tflow bundle, with its level-of-detail pyramid
// (tile_flow_lod.hpp) beside it; read the record back. Byte-deterministic: the same record
// writes the same files.
void write_tflow(const TileFlowRecord& rec, const std::string& dir);
TileFlowRecord read_tflow(const std::string& dir);

// L3 occupancy at cycle t, counted from the residency intervals plus the foreign slots. What
// the invariant checker (step 3) compares against capacity, and what a test compares against
// the executor's own peak.
std::uint64_t l3_occupancy_at(const TileFlowRecord& rec, Cycle t);
// The maximum of l3_occupancy_at over the run, computed by a sweep over interval ends.
std::uint64_t peak_l3_occupancy(const TileFlowRecord& rec);

} // namespace sw::kpu::program::record
