// ============================================================================
// include/sw/kpu/program/record/tile_flow_lod.hpp
// The level-of-detail pyramid over a tile-flow record (#286 step 3;
// docs/plans/tile-flow-debugger.md §3.4).
//
// The viewer cannot draw a million events per frame, so it draws a summary at the resolution
// of the screen: one value per row per time bin, at the finest power-of-two bin width that
// keeps about one bin per pixel. Zooming swaps the level the way a map swaps tiles.
//
// ROWS: one per station of the record, then one per mover pool.
//
//   l3 station    occupancy = resident tiles + the caller's foreign slots
//   cf station    occupancy = 1 while a compute runs on it
//   mover pool    occupancy = lanes busy
//   dram, l2, l1  no occupancy at L-T1 (DRAM is unbounded, L2 and L1 are not modelled);
//                 the row exists so the viewer can draw it, and says `modelled = false`
//
// METRICS, per row per bin, every one of them EXACTLY MERGEABLE:
//
//   occ     occupancy-time, sum of (occupancy x cycles) over the bin   -> merges by +
//   peak    the maximum instantaneous occupancy within the bin          -> merges by max
//   starts  intervals that begin in the bin                             -> merges by +
//
// Exactly mergeable is the point (plan §3.4, invariant 3): a bin at level k+1 IS the merge of
// its two children at level k, so a zoomed-out view never shows a value the zoomed-in view
// would contradict. A top-4 operator mix would not merge exactly and is not stored; v0 records
// one program, so there is no operator mix to show yet.
//
// Occupancy-time is stored as f64 and is exact while it stays below 2^53 tile-cycles.
//
// THE FILES: `lod.json` (rows, base level, per level its bin count and column offsets) and
// `lod.bin` (per level, occ f64[rows x bins], peak u32[rows x bins], starts u32[rows x bins],
// row-major, each column 8-byte aligned). They are derived from the record and written beside
// it; the record's own files never depend on them.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================
#pragma once

#include <sw/kpu/program/record/tile_flow_record.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace sw::kpu::program::record {

struct LodRow {
    std::string name;              // station name, or "mover:<pool>"
    std::string kind;              // dram | l3 | l2 | l1 | cf | mover
    std::uint64_t capacity = 0;    // tiles, lanes, 1 for a compute tile; 0 = unbounded / none
    bool modelled = true;
};

struct LodLevel {
    unsigned k = 0;                        // bin width = 2^k cycles
    std::uint64_t bins = 0;
    std::vector<double> occ;               // [row * bins + bin]
    std::vector<std::uint32_t> peak;
    std::vector<std::uint32_t> starts;
};

struct Lod {
    std::vector<LodRow> rows;
    std::vector<LodLevel> levels;          // finest first, coarsest (one bin) last
};

// Build the pyramid. The finest stored level is the smallest k whose bin count fits
// `max_base_bins`; finer zoom reads the raw events instead.
Lod build_lod(const TileFlowRecord& rec, std::uint64_t max_base_bins = 4096);

// The next level up: each bin the merge of two children (sum, max, sum).
LodLevel merge_level(const LodLevel& child, std::size_t rows);

void write_lod(const Lod& lod, const std::string& dir);
Lod read_lod(const std::string& dir);

} // namespace sw::kpu::program::record
