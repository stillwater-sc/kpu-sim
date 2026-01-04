// patterns/memory/lpddr5/common/workloads.hpp
//
// Workload definitions for LPDDR5 memory controller validation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include "lpddr5_configs.hpp"

namespace sw::kpu::patterns::lpddr5 {

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
inline Workload make_page_hit_workload(uint8_t bank = 0, uint32_t row = 100, size_t count = 8) {
    Workload w;
    w.name = "page_hits";
    w.description = "Sequential reads to same row - maximizes page hits";

    for (size_t i = 0; i < count; ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(bank, row, i * 64)
        ));
    }

    w.expected.page_hits = count - 1;
    w.expected.page_empty = 1;
    w.expected.page_conflicts = 0;
    w.expected.min_cycles = tRCD + tCL + tBurst + (count - 1) * (tCL + tBurst);

    return w;
}

/// Reads to different rows (page conflicts)
inline Workload make_page_conflict_workload(uint8_t bank = 0, size_t count = 4) {
    Workload w;
    w.name = "page_conflicts";
    w.description = "Reads to different rows - maximizes page conflicts";

    for (size_t i = 0; i < count; ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(bank, i * 100, 0)
        ));
    }

    w.expected.page_hits = 0;
    w.expected.page_empty = 1;
    w.expected.page_conflicts = count - 1;
    w.expected.min_cycles = PAGE_EMPTY_READ_LATENCY + (count - 1) * PAGE_CONFLICT_READ_LATENCY;

    return w;
}

/// Alternating reads and writes (bus turnaround)
inline Workload make_mixed_rw_workload(uint8_t bank = 0, uint32_t row = 100, size_t pairs = 4) {
    Workload w;
    w.name = "mixed_rw";
    w.description = "Alternating reads and writes - tests bus turnaround";

    for (size_t i = 0; i < pairs; ++i) {
        w.accesses.push_back(MemoryAccess::read(
            make_address(bank, row, i * 128)
        ));
        w.accesses.push_back(MemoryAccess::write(
            make_address(bank, row, i * 128 + 64)
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

/// Two banks in same bank group (tRRD_L constraint)
inline Workload make_two_banks_same_group_workload(uint32_t row = 100, size_t per_bank = 4) {
    Workload w;
    w.name = "two_banks_same_bg";
    w.description = "Two banks in same bank group - tests tRRD_L/tCCD_L";

    for (size_t i = 0; i < per_bank; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, row, i * 64)));  // Bank 0 (BG0)
        w.accesses.push_back(MemoryAccess::read(make_address(1, row, i * 64)));  // Bank 1 (BG0)
    }

    w.expected.page_empty = 2;
    w.expected.page_hits = per_bank * 2 - 2;

    return w;
}

/// Two banks in different bank groups (tRRD_S - faster)
inline Workload make_two_banks_diff_groups_workload(uint32_t row = 100, size_t per_bank = 4) {
    Workload w;
    w.name = "two_banks_diff_bg";
    w.description = "Two banks in different bank groups - tests tRRD_S/tCCD_S";

    for (size_t i = 0; i < per_bank; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, row, i * 64)));  // Bank 0 (BG0)
        w.accesses.push_back(MemoryAccess::read(make_address(4, row, i * 64)));  // Bank 4 (BG1)
    }

    w.expected.page_empty = 2;
    w.expected.page_hits = per_bank * 2 - 2;

    return w;
}

// ============================================================================
// Three-Bank Workloads
// ============================================================================

/// Three banks in different groups (round-robin)
inline Workload make_three_banks_mixed_workload(uint32_t row = 100, size_t per_bank = 3) {
    Workload w;
    w.name = "three_banks_mixed_bg";
    w.description = "Three banks in different groups - tests multi-bank parallelism";

    for (size_t i = 0; i < per_bank; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, row, i * 64)));  // BG0
        w.accesses.push_back(MemoryAccess::read(make_address(4, row, i * 64)));  // BG1
        w.accesses.push_back(MemoryAccess::read(make_address(8, row, i * 64)));  // BG2
    }

    w.expected.page_empty = 3;
    w.expected.page_hits = per_bank * 3 - 3;

    return w;
}

/// Three banks in same group (tRRD_L limitation)
inline Workload make_three_banks_same_group_workload(uint32_t row = 100, size_t per_bank = 3) {
    Workload w;
    w.name = "three_banks_same_bg";
    w.description = "Three banks in same group - tests bank group limitations";

    for (size_t i = 0; i < per_bank; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, row, i * 64)));  // Bank 0
        w.accesses.push_back(MemoryAccess::read(make_address(1, row, i * 64)));  // Bank 1
        w.accesses.push_back(MemoryAccess::read(make_address(2, row, i * 64)));  // Bank 2
    }

    w.expected.page_empty = 3;
    w.expected.page_hits = per_bank * 3 - 3;

    return w;
}

// ============================================================================
// Four-Bank Workloads
// ============================================================================

/// Four banks in same group (tFAW constraint)
inline Workload make_four_banks_full_group_workload(uint32_t row = 100, size_t rounds = 2) {
    Workload w;
    w.name = "four_banks_full_bg";
    w.description = "Four banks in same bank group - tests tFAW constraint";

    for (size_t r = 0; r < rounds; ++r) {
        for (uint8_t b = 0; b < 4; ++b) {
            w.accesses.push_back(MemoryAccess::read(make_address(b, row, r * 64)));
        }
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = rounds * 4 - 4;

    return w;
}

/// Four banks across groups (maximum parallelism)
inline Workload make_four_banks_across_groups_workload(uint32_t row = 100, size_t per_bank = 2) {
    Workload w;
    w.name = "four_banks_across_bg";
    w.description = "Four banks, one per bank group - maximum parallelism";

    for (size_t i = 0; i < per_bank; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(0, row, i * 64)));   // BG0
        w.accesses.push_back(MemoryAccess::read(make_address(4, row, i * 64)));   // BG1
        w.accesses.push_back(MemoryAccess::read(make_address(8, row, i * 64)));   // BG2
        w.accesses.push_back(MemoryAccess::read(make_address(12, row, i * 64)));  // BG3
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = per_bank * 4 - 4;

    return w;
}

/// Sustained page hits across four banks
inline Workload make_page_hit_burst_workload(uint32_t row = 100, size_t per_bank = 4) {
    Workload w;
    w.name = "page_hit_burst";
    w.description = "Sustained page hits across four banks - tests peak throughput";

    // Open all four banks to same row, then burst page hits
    for (uint8_t b = 0; b < 4; ++b) {
        w.accesses.push_back(MemoryAccess::read(make_address(b * 4, row, 0)));  // Initial access
    }
    for (size_t i = 1; i < per_bank; ++i) {
        for (uint8_t b = 0; b < 4; ++b) {
            w.accesses.push_back(MemoryAccess::read(make_address(b * 4, row, i * 64)));
        }
    }

    w.expected.page_empty = 4;
    w.expected.page_hits = 4 * per_bank - 4;

    return w;
}

// ============================================================================
// Complex Workloads
// ============================================================================

/// Strided access (models matrix column access)
inline Workload make_strided_workload(uint8_t bank = 0, uint32_t stride_rows = 16, size_t count = 8) {
    Workload w;
    w.name = "strided";
    w.description = "Strided access pattern - models matrix column traversal";

    for (size_t i = 0; i < count; ++i) {
        w.accesses.push_back(MemoryAccess::read(make_address(bank, i * stride_rows, 0)));
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
        uint8_t bank = rng() % 16;
        uint32_t row = rng() % 1024;
        w.accesses.push_back(MemoryAccess::read(make_address(bank, row, 0)));
    }

    // Expectations depend on random sequence
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

inline std::vector<Workload> four_bank_workloads() {
    return {
        make_four_banks_full_group_workload(),
        make_four_banks_across_groups_workload(),
        make_page_hit_burst_workload()
    };
}

inline std::vector<Workload> all_workloads() {
    std::vector<Workload> all;
    auto sb = single_bank_workloads();
    auto tb = two_bank_workloads();
    auto fb = four_bank_workloads();
    all.insert(all.end(), sb.begin(), sb.end());
    all.insert(all.end(), tb.begin(), tb.end());
    all.insert(all.end(), fb.begin(), fb.end());
    all.push_back(make_strided_workload());
    return all;
}

} // namespace sw::kpu::patterns::lpddr5
