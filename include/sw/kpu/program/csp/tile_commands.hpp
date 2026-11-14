// ============================================================================
// include/sw/kpu/program/csp/tile_commands.hpp
// One program on both sides (#397, option 1): a CSP program lowered to the KPU RTL's tile command
// stream -- the LOAD / STORE / EXEC commands the systolic tile's controller executes, with their
// `sync` credits and tokens and their `chain` bits -- written as a versioned JSON document both
// sides read (`kpu-tile-cmds`, version 1; docs/plans/tile-command-stream.md).
//
// kpu-sim stays the one source of programs: the RTL's runtime executes the stream, and both sides
// write .dflow records whose results compare (shared-tile-flow-record.md §8.2, D3).
//
// What version 1 lowers: a matmul on ONE output-stationary systolic tile, int8 x int8 -> int32,
// every tile dim x dim (the array's size). A gemm Call is the tile's pair LOAD A (ACQUIRE the pair
// slot), LOAD B (TRANSPOSE; SIGNAL the pair slot), COMPUTE (WAIT and RELEASE the pair; ACQUIRE the
// accumulator slot on its reduction's first step, SIGNAL it on its last; chain bits across the
// reduction). A DRAIN and its WRITEBACK are one STORE from the accumulator slot into L3: the RTL's
// store reads the accumulator straight to L3 (.dflow 6.1). The program's Loads and Stores are the
// host's: it fills L3 before the tile runs (`l3_init`) and reads the results after (`results`).
// L3 is the RTL's flat memory cell: a residency lives at its walker slot times a fixed pitch, so the
// stream's L3 addresses are the program's placement (#399). Everything else is refused by name.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/csp/csp_program.hpp>
#include <sw/kpu/program/tile_program.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::program::csp {

namespace tilecmd {

// The RTL's command fields and encodings (kpugen src/kpugen/tiles/systolic/isa/command.py,
// util/commands.py, controller/sync.py). They change only with the format's version.
enum Type : std::uint8_t { kLoad = 0, kStore = 1, kExec = 2 };
enum DmaOpcode : std::uint8_t { kMemcpy = 0, kTranspose = 4 };
enum ExecOpcode : std::uint8_t { kConfig = 0x00, kCompute = 0x11 };
inline constexpr std::uint16_t kWait = 1u << 8, kRelease = 1u << 9, kAcquire = 1u << 10, kSignal = 1u << 11;
inline constexpr std::uint32_t kAccumulatorBit = 1u << 31;   // local_addr: the accumulator, not the scratchpad
inline constexpr std::uint32_t kDataflowOs = 0;               // EXEC CONFIG rs1
inline constexpr int kVersion = 1;

inline std::uint16_t sync(std::uint32_t src, std::uint32_t dst, std::uint16_t flags) {
    return static_cast<std::uint16_t>((src & 0xF) | ((dst & 0xF) << 4) | flags);
}

}  // namespace tilecmd

// The tile the stream is for: the RTL's SystolicConfig values it depends on.
struct TileTarget {
    std::uint32_t dim = 4;              // the array: dim x dim PEs; every tile is dim x dim
    std::uint32_t buffers = 2;          // L2 pair slots (A slot n, B slot buffers + n)
    std::uint32_t acc_slots = 2;        // accumulator slots
};

// One command: the RTL's `Command` fields, in its order, and where it came from.
struct TileCommand {
    std::uint8_t type = 0;
    std::uint8_t opcode = 0;
    std::uint8_t id = 0;
    std::uint64_t dram_addr = 0;        // an L3 byte address
    std::uint32_t local_addr = 0;
    std::uint16_t len = 0;
    std::uint32_t rs1 = 0, rd = 0;
    std::uint16_t k_dim = 0;
    std::uint32_t stride = 0, rs2 = 0;
    std::uint16_t row_bytes = 0;
    std::uint8_t activation = 0;
    std::uint16_t sync = 0;
    std::uint8_t chain = 0;
    // Not sent to the tile: the tile the command moves (the RTL's `tile` annotation) and the CSP
    // action it implements (the join to the program and its .dflow steps).
    TileCoord tile;
    std::size_t action = kNone;
};

// A tile in L3 at an address: what the host writes before the tile runs, or reads after.
struct L3Tile {
    TileCoord tile;
    std::uint64_t addr = 0;             // byte address in the tile's L3 memory cell
    std::uint32_t row_bytes = 0;        // = the L3 row stride
    std::string element;                // its number system
    std::vector<std::uint8_t> bytes;    // row-major, little-endian; empty when no values were given
};

struct TileCommandStream {
    std::string program;
    TileTarget target;
    std::uint64_t l3_pitch = 0;         // bytes per L3 slot
    std::uint64_t l3_bytes = 0;         // the L3 the stream addresses
    std::vector<L3Tile> l3_init;        // inputs, by the program's Loads
    std::vector<L3Tile> results;        // outputs, by the program's Stores, with the expected values
    std::vector<TileCommand> commands;
};

// Lower `p` for `t`. `inputs` gives the input tiles' values (l3_init's bytes) and `expected` the
// results' (L-B's); either may be null. Throws std::invalid_argument naming what version 1 does not
// lower.
TileCommandStream lower_tile_commands(const CspProgram& p, const TileTarget& t, const TileProgram* inputs = nullptr,
                                      const TileProgram* expected = nullptr);

// The stream as `kpu-tile-cmds` JSON, deterministic (fields in the RTL's order).
std::string to_json(const TileCommandStream& s, const std::string& producer);

}  // namespace sw::kpu::program::csp
