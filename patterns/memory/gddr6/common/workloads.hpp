// patterns/memory/gddr6/common/workloads.hpp
//
// Workload definitions for GDDR6 memory controller validation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include "gddr6_configs.hpp"

namespace sw::kpu::patterns::gddr6 {

// ============================================================================
// Memory Access Description
// ============================================================================

enum class AccessType : uint8_t {
    READ,
    WRITE
};

/// Single memory access in a workload
struct MemoryAccess {
    AccessType type = AccessType::READ;
    uint64_t address = 0;
    uint32_t size = 64;
    uint64_t relative_cycle = 0;  // When to submit relative to workload start

    // Convenience constructors
    static MemoryAccess read(uint64_t addr, uint32_t sz = 64, uint64_t cycle = 0) {
        return {AccessType::READ, addr, sz, cycle};
    }

    static MemoryAccess write(uint64_t addr, uint32_t sz = 64, uint64_t cycle = 0) {
        return {AccessType::WRITE, addr, sz, cycle};
    }
};

// ============================================================================
// Workload Definition
// ============================================================================

/// A sequence of memory accesses forming a test workload
struct Workload {
    std::string name;
    std::string description;
    std::vector<MemoryAccess> accesses;

    // Expected results for validation (from cycle-accurate reference)
    struct Expected {
        uint64_t page_hits = 0;
        uint64_t page_empty = 0;
        uint64_t page_conflicts = 0;
        uint64_t min_cycles = 0;    // Theoretical minimum
        uint64_t max_cycles = 0;    // With worst-case queuing
    } expected;

    size_t num_reads() const {
        size_t count = 0;
        for (const auto& a : accesses) {
            if (a.type == AccessType::READ) ++count;
        }
        return count;
    }

    size_t num_writes() const {
        size_t count = 0;
        for (const auto& a : accesses) {
            if (a.type == AccessType::WRITE) ++count;
        }
        return count;
    }
};

// ============================================================================
// Single-Bank Workloads
// ============================================================================

/// Sequential reads to same row (page hits)
inline Workload make_page_hit_workload(uint8_t channel = 0, uint8_t bank = 0,
                                        uint32_t row = 100, size_t count = 8) {
    Workload w;
    w.name = "page_hits";
    w.description = "Sequential reads to same row - maximizes page hits";

    for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(channel, bank, row, i)
        ));
    }

    w.expected.page_hits = count - 1;
    w.expected.page_empty = 1;
    w.expected.page_conflicts = 0;
    w.expected.min_cycles = tRCDRD + tRL + tBurst + (count - 1) * (tRL + tBurst);

    return w;
}

/// Reads to different rows (page conflicts)
inline Workload make_page_conflict_workload(uint8_t channel = 0, uint8_t bank = 0,
                                             size_t count = 4) {
    Workload w;
    w.name = "page_conflicts";
    w.description = "Reads to different rows - maximizes page conflicts";

    for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(channel, bank, i * 100, 0)
        ));
    }

    w.expected.page_hits = 0;
    w.expected.page_empty = 1;
    w.expected.page_conflicts = count - 1;
    w.expected.min_cycles = PAGE_EMPTY_READ_LATENCY + (count - 1) * PAGE_CONFLICT_READ_LATENCY;

    return w;
}

/// Alternating reads and writes (bus turnaround)
inline Workload make_mixed_rw_workload(uint8_t channel = 0, uint8_t bank = 0,
                                        uint32_t row = 100, size_t pairs = 4) {
    Workload w;
    w.name = "mixed_rw";
    w.description = "Alternating reads and writes - tests bus turnaround";

    for (uint32_t i = 0; i < static_cast<uint32_t>(pairs); ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(channel, bank, row, i * 2)
        ));
        w.accesses.push_back(MemoryAccess::write(
            make_address(channel, bank, row, i * 2 + 1)
        ));
    }

    w.expected.page_hits = pairs * 2 - 1;
    w.expected.page_empty = 1;
    w.expected.page_conflicts = 0;

    return w;
}

// ============================================================================
// Two-Bank Workloads
// ============================================================================

/// Two banks in same bank group (tRRD_L/tCCD_L constraint)
inline Workload make_two_banks_same_group_workload(uint8_t channel = 0, uint32_t row = 100,
                                                    size_t per_bank = 4) {
    Workload w;
    w.name = "two_banks_same_bg";
    w.description = "Two banks in same bank group - tests tRRD_L/tCCD_L";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_bank); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i)));  // Bank 0 (BG0)
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 1, row, i)));  // Bank 1 (BG0)
    }

    w.expected.page_empty = 2;
    w.expected.page_hits = per_bank * 2 - 2;

    return w;
}

/// Two banks in different bank groups (tRRD_S/tCCD_S - faster)
inline Workload make_two_banks_diff_groups_workload(uint8_t channel = 0, uint32_t row = 100,
                                                     size_t per_bank = 4) {
    Workload w;
    w.name = "two_banks_diff_bg";
    w.description = "Two banks in different bank groups - tests tRRD_S/tCCD_S";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_bank); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i)));  // Bank 0 (BG0)
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 4, row, i)));  // Bank 4 (BG1)
    }

    w.expected.page_empty = 2;
    w.expected.page_hits = per_bank * 2 - 2;

    return w;
}

// ============================================================================
// Three-Bank Workloads
// ============================================================================

/// Three banks in different groups (round-robin)
inline Workload make_three_banks_mixed_workload(uint8_t channel = 0, uint32_t row = 100,
                                                 size_t per_bank = 3) {
    Workload w;
    w.name = "three_banks_mixed_bg";
    w.description = "Three banks in different groups - tests multi-bank parallelism";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_bank); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i)));   // BG0
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 4, row, i)));   // BG1
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 8, row, i)));   // BG2
    }

    w.expected.page_empty = 3;
    w.expected.page_hits = per_bank * 3 - 3;

    return w;
}

/// Three banks in same group (tRRD_L limitation)
inline Workload make_three_banks_same_group_workload(uint8_t channel = 0, uint32_t row = 100,
                                                      size_t per_bank = 3) {
    Workload w;
    w.name = "three_banks_same_bg";
    w.description = "Three banks in same group - tests bank group limitations";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_bank); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i)));  // Bank 0
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 1, row, i)));  // Bank 1
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 2, row, i)));  // Bank 2
    }

    w.expected.page_empty = 3;
    w.expected.page_hits = per_bank * 3 - 3;

    return w;
}

// ============================================================================
// Four-Bank Workloads
// ============================================================================

/// Four banks in same group (tFAW constraint)
inline Workload make_four_banks_full_group_workload(uint8_t channel = 0, uint32_t row = 100,
                                                     size_t rounds = 2) {
    Workload w;
    w.name = "four_banks_full_bg";
    w.description = "Four banks in same bank group - tests tFAW constraint";

    for (uint32_t r = 0; r < static_cast<uint32_t>(rounds); ++r) {
        for (uint8_t b = 0; b < 4; ++b) {
            w.accesses.push_back(MemoryAccess::read(make_address(channel, b, row, r)));
        }
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = rounds * 4 - 4;

    return w;
}

/// Four banks across groups (maximum parallelism)
inline Workload make_four_banks_across_groups_workload(uint8_t channel = 0, uint32_t row = 100,
                                                        size_t per_bank = 2) {
    Workload w;
    w.name = "four_banks_across_bg";
    w.description = "Four banks, one per bank group - maximum parallelism";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_bank); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i)));   // BG0
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 4, row, i)));   // BG1
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 8, row, i)));   // BG2
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 12, row, i)));  // BG3
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = per_bank * 4 - 4;

    return w;
}

/// Sustained page hits across four banks (one per bank group)
inline Workload make_page_hit_burst_workload(uint8_t channel = 0, uint32_t row = 100,
                                              size_t per_bank = 4) {
    Workload w;
    w.name = "page_hit_burst";
    w.description = "Sustained page hits across four banks - tests peak throughput";

    // Open all four banks to same row, then burst page hits
    for (uint8_t b = 0; b < 4; ++b) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, b * 4, row, 0)));  // Initial
    }
    for (uint32_t i = 1; i < static_cast<uint32_t>(per_bank); ++i) {
        for (uint8_t b = 0; b < 4; ++b) {
            w.accesses.push_back(MemoryAccess::read(make_address(channel, b * 4, row, i)));
        }
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = 4 * per_bank - 4;

    return w;
}

// ============================================================================
// Dual-Channel Workloads (GDDR6-specific)
// ============================================================================

/// Independent dual-channel access
inline Workload make_dual_channel_independent_workload(uint32_t row = 100, size_t per_channel = 4) {
    Workload w;
    w.name = "dual_channel_independent";
    w.description = "Independent accesses to both channels - tests dual-channel parallelism";

    for (uint32_t i = 0; i < static_cast<uint32_t>(per_channel); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, 0, row, i)));  // Channel 0
        w.accesses.push_back(MemoryAccess::read(make_address(1, 0, row, i)));  // Channel 1
    }

    w.expected.page_empty = 2;  // One per channel
    w.expected.page_hits = per_channel * 2 - 2;

    return w;
}

/// Interleaved dual-channel access
inline Workload make_dual_channel_interleaved_workload(uint32_t row = 100, size_t count = 8) {
    Workload w;
    w.name = "dual_channel_interleaved";
    w.description = "Interleaved accesses across channels - tests bandwidth aggregation";

    for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
        uint8_t channel = i % 2;
        w.accesses.push_back(MemoryAccess::read(make_address(channel, 0, row, i / 2)));
    }

    w.expected.page_empty = 2;
    w.expected.page_hits = count - 2;

    return w;
}

// ============================================================================
// Complex Workloads
// ============================================================================

/// Strided access (models matrix column access)
inline Workload make_strided_workload(uint8_t channel = 0, uint8_t bank = 0,
                                       uint32_t stride_rows = 16, size_t count = 8) {
    Workload w;
    w.name = "strided";
    w.description = "Strided access pattern - models matrix column traversal";

    for (uint32_t i = 0; i < static_cast<uint32_t>(count); ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(channel, bank, i * stride_rows, 0)));
    }

    w.expected.page_empty = 1;
    w.expected.page_conflicts = count - 1;

    return w;
}

/// Random access pattern
inline Workload make_random_workload(uint32_t seed = 12345, size_t count = 16) {
    Workload w;
    w.name = "random";
    w.description = "Random bank/row access - worst-case locality";

    std::minstd_rand rng(seed);
    for (size_t i = 0; i < count; ++i) {
        uint8_t channel = rng() % 2;
        uint8_t bank = rng() % 16;
        uint32_t row = rng() % 1024;
        w.accesses.push_back(MemoryAccess::read(make_address(channel, bank, row, 0)));
    }

    // Expectations depend on random sequence
    return w;
}

/// Tile load pattern (models DMA tile fetch)
inline Workload make_tile_load_workload(uint8_t channel = 0, uint8_t bank = 0,
                                         uint32_t start_row = 0, size_t rows = 4) {
    Workload w;
    w.name = "tile_load";
    w.description = "Tile load pattern - models DMA tile fetch with row locality";

    // Load multiple columns from consecutive rows
    const uint32_t cols_per_row = 4;
    for (uint32_t r = 0; r < static_cast<uint32_t>(rows); ++r) {
        for (uint32_t c = 0; c < cols_per_row; ++c) {
            w.accesses.push_back(MemoryAccess::read(
                make_address(channel, bank, start_row + r, c)));
        }
    }

    // First access per row is page conflict (except very first)
    w.expected.page_empty = 1;
    w.expected.page_conflicts = rows - 1;
    w.expected.page_hits = rows * cols_per_row - rows;

    return w;
}

// ============================================================================
// Workload Collections
// ============================================================================

inline std::vector<Workload> single_bank_workloads() {
    return {
        make_page_hit_workload(),
        make_page_conflict_workload(),
        make_mixed_rw_workload()
    };
}

inline std::vector<Workload> two_bank_workloads() {
    return {
        make_two_banks_same_group_workload(),
        make_two_banks_diff_groups_workload()
    };
}

inline std::vector<Workload> three_bank_workloads() {
    return {
        make_three_banks_mixed_workload(),
        make_three_banks_same_group_workload()
    };
}

inline std::vector<Workload> four_bank_workloads() {
    return {
        make_four_banks_full_group_workload(),
        make_four_banks_across_groups_workload(),
        make_page_hit_burst_workload()
    };
}

inline std::vector<Workload> dual_channel_workloads() {
    return {
        make_dual_channel_independent_workload(),
        make_dual_channel_interleaved_workload()
    };
}

inline std::vector<Workload> complex_workloads() {
    return {
        make_strided_workload(),
        make_random_workload(),
        make_tile_load_workload()
    };
}

inline std::vector<Workload> all_workloads() {
    std::vector<Workload> all;
    auto sb = single_bank_workloads();
    auto tb = two_bank_workloads();
    auto thb = three_bank_workloads();
    auto fb = four_bank_workloads();
    auto dc = dual_channel_workloads();
    auto cx = complex_workloads();

    all.insert(all.end(), sb.begin(), sb.end());
    all.insert(all.end(), tb.begin(), tb.end());
    all.insert(all.end(), thb.begin(), thb.end());
    all.insert(all.end(), fb.begin(), fb.end());
    all.insert(all.end(), dc.begin(), dc.end());
    all.insert(all.end(), cx.begin(), cx.end());

    return all;
}

} // namespace sw::kpu::patterns::gddr6
