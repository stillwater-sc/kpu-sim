// patterns/memory/gddr7/common/workloads.hpp
//
// Standard workloads for GDDR7 pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <vector>
#include <cstdint>
#include "gddr7_configs.hpp"

namespace sw::kpu::patterns::gddr7 {

// ============================================================================
// Workload Description
// ============================================================================

struct WorkloadRequest {
    uint64_t address;
    uint32_t size;
    bool is_write;
};

struct Workload {
    std::string name;
    std::vector<WorkloadRequest> requests;
};

// ============================================================================
// Standard Workloads
// ============================================================================

/// Page hit workload: 16 reads to same row in bank 0
inline Workload make_page_hit_workload() {
    Workload w;
    w.name = "page_hits";
    for (int i = 0; i < 16; ++i) {
        w.requests.push_back({
            make_address(0, 100, i * 64),
            CACHE_LINE_BYTES,
            false
        });
    }
    return w;
}

/// Page conflict workload: alternating rows in same bank
inline Workload make_page_conflict_workload() {
    Workload w;
    w.name = "page_conflicts";
    for (int i = 0; i < 8; ++i) {
        // Even: row 100
        w.requests.push_back({
            make_address(0, 100, i * 64),
            CACHE_LINE_BYTES,
            false
        });
        // Odd: row 200 (conflict)
        w.requests.push_back({
            make_address(0, 200, i * 64),
            CACHE_LINE_BYTES,
            false
        });
    }
    return w;
}

/// Mixed read/write workload
inline Workload make_mixed_rw_workload() {
    Workload w;
    w.name = "mixed_rw";
    for (int i = 0; i < 8; ++i) {
        // Read
        w.requests.push_back({
            make_address(0, 100, i * 128),
            CACHE_LINE_BYTES,
            false
        });
        // Write
        w.requests.push_back({
            make_address(0, 100, i * 128 + 64),
            CACHE_LINE_BYTES,
            true
        });
    }
    return w;
}

/// Same bank group workload: banks 0, 1, 2, 3 (all in BG0)
inline Workload make_same_bank_group_workload() {
    Workload w;
    w.name = "same_bg";
    for (int bank = 0; bank < 4; ++bank) {
        for (int i = 0; i < 4; ++i) {
            w.requests.push_back({
                make_address(bank, 100, i * 64),
                CACHE_LINE_BYTES,
                false
            });
        }
    }
    return w;
}

/// Different bank groups workload: banks 0, 4, 8, 12 (one from each BG)
inline Workload make_diff_bank_groups_workload() {
    Workload w;
    w.name = "diff_bg";
    for (int bg = 0; bg < 4; ++bg) {
        uint8_t bank = bg * 4;  // First bank in each group
        for (int i = 0; i < 4; ++i) {
            w.requests.push_back({
                make_address(bank, 100, i * 64),
                CACHE_LINE_BYTES,
                false
            });
        }
    }
    return w;
}

/// Sequential streaming workload
inline Workload make_stream_workload(size_t num_requests = 64) {
    Workload w;
    w.name = "stream";
    for (size_t i = 0; i < num_requests; ++i) {
        // Spread across banks for parallelism
        uint8_t bank = (i % 16);
        w.requests.push_back({
            make_address(bank, 100, static_cast<uint32_t>((i / 16) * 64)),
            CACHE_LINE_BYTES,
            false
        });
    }
    return w;
}

/// Random access pattern
inline Workload make_random_workload(size_t num_requests = 32, uint32_t seed = 12345) {
    Workload w;
    w.name = "random";

    // Simple LCG for reproducible "random" pattern
    uint32_t state = seed;
    for (size_t i = 0; i < num_requests; ++i) {
        state = state * 1103515245 + 12345;
        uint8_t bank = (state >> 16) % 16;
        uint32_t row = (state >> 8) % 1024;
        uint32_t col = (state % 16) * 64;

        w.requests.push_back({
            make_address(bank, row, col),
            CACHE_LINE_BYTES,
            (state & 1) != 0  // Random read/write
        });
    }
    return w;
}

/// Tile load workload (4KB tile spread across banks)
inline Workload make_tile_load_workload() {
    Workload w;
    w.name = "tile_load";

    // 4KB tile = 64 cache lines
    for (int i = 0; i < 64; ++i) {
        uint8_t bank = i % 16;  // Spread across banks
        w.requests.push_back({
            make_address(bank, 100, (i / 16) * 64),
            CACHE_LINE_BYTES,
            false
        });
    }
    return w;
}

/// Maximum bandwidth workload: all 16 banks, page hits
inline Workload make_max_bandwidth_workload() {
    Workload w;
    w.name = "max_bandwidth";

    // First open all pages (page empty accesses)
    for (int bank = 0; bank < 16; ++bank) {
        w.requests.push_back({
            make_address(bank, 100, 0),
            CACHE_LINE_BYTES,
            false
        });
    }

    // Then sequential page hits across all banks
    for (int round = 1; round < 16; ++round) {
        for (int bank = 0; bank < 16; ++bank) {
            w.requests.push_back({
                make_address(bank, 100, round * 64),
                CACHE_LINE_BYTES,
                false
            });
        }
    }

    return w;
}

} // namespace sw::kpu::patterns::gddr7
