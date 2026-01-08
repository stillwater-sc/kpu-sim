// patterns/memory/gddr6/bandwidth/page_burst.cpp
//
// Pattern: Maximum page hits per page open
// Tests: Optimal bandwidth with full row buffer utilization
//
// Strategy:
// - Open a page (one activation)
// - Access all 128 cache lines in the page sequentially
// - This amortizes the page miss across 127 page hits
//
// Page Size Analysis:
// - Row buffer: 8,192 bytes (8 KB)
// - Cache line: 64 bytes
// - Max accesses per page: 128 (1 miss + 127 hits)
//
// GDDR6 Bank Organization:
// - Bank Group 0: Banks 0, 1, 2, 3
// - Bank Group 1: Banks 4, 5, 6, 7
// - Bank Group 2: Banks 8, 9, 10, 11
// - Bank Group 3: Banks 12, 13, 14, 15
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>

#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

// Page geometry constants (use namespace CACHE_LINE_BYTES = 64)
constexpr int PAGE_SIZE_BYTES = 8192;       // 8 KB row buffer
constexpr int CACHE_LINES_PER_PAGE = PAGE_SIZE_BYTES / 64;  // 128

// Helper to create address on channel 0
inline uint64_t addr(uint8_t bank, uint32_t row, uint32_t col) {
    return make_address(0, bank, row, col);
}

/// Test full page burst - all 128 cache lines from one page
/// This is the optimal single-bank access pattern
bool test_full_page_burst() {
    std::cout << "\n=== Test: Full Page Burst (128 accesses) ===" << std::endl;
    std::cout << "Configuration: Single bank, single page" << std::endl;
    std::cout << "Pattern: Sequential access to all " << CACHE_LINES_PER_PAGE
              << " cache lines" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;

    // Access all 128 cache lines in the page
    for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
        harness.submit_read(addr(BANK, ROW, col * CACHE_LINE_BYTES));
    }

    if (!harness.run_until_complete(200000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    const auto& stats = harness.stats();
    uint64_t total_bytes = CACHE_LINES_PER_PAGE * CACHE_LINE_BYTES;  // 8 KB
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / cycles;
    double hit_rate = 100.0 * stats.page_hits / (stats.page_hits + stats.page_empty);

    std::cout << "\n--- Full Page Burst Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit rate: " << std::setprecision(1) << hit_rate << "%" << std::endl;
    std::cout << "Page utilization: 100% (all 128 cache lines accessed)" << std::endl;

    std::cout << "PASS: Full page burst completed" << std::endl;
    return true;
}

/// Test multi-bank page burst across all 16 banks
/// Open one page per bank and access all cache lines
bool test_multi_bank_page_burst() {
    std::cout << "\n=== Test: Multi-Bank Page Burst (16 banks × 128 accesses) ===" << std::endl;
    std::cout << "Pattern: Full page access across all 16 banks" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    constexpr std::array<uint8_t, 16> banks = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    constexpr uint32_t ROW = 100;

    // Round-robin across banks, accessing all cache lines
    for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
        for (uint8_t bank : banks) {
            harness.submit_read(addr(bank, ROW, col * CACHE_LINE_BYTES));
        }
    }

    if (!harness.run_until_complete(1000000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    const auto& stats = harness.stats();
    int total_accesses = 16 * CACHE_LINES_PER_PAGE;  // 2048 accesses
    uint64_t total_bytes = total_accesses * CACHE_LINE_BYTES;  // 128 KB
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- Multi-Bank Page Burst Analysis ---" << std::endl;
    std::cout << "Banks: 16, Pages: 16, Cache lines per page: " << CACHE_LINES_PER_PAGE << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    // Expected: 16 page misses + (2048-16) page hits = 99.2% hit rate
    std::cout << "Expected page hits: " << (total_accesses - 16) << std::endl;
    std::cout << "Expected page misses: 16" << std::endl;

    std::cout << "PASS: Multi-bank page burst completed" << std::endl;
    return true;
}

/// Compare partial page access vs full page access
/// Shows the benefit of maximizing page hits
bool test_page_utilization_comparison() {
    std::cout << "\n=== Test: Page Utilization Comparison ===" << std::endl;
    std::cout << "Compares 8, 32, 64, 128 accesses per page" << std::endl;

    struct Result {
        int accesses_per_page;
        uint64_t cycles;
        double bytes_per_cycle;
        double page_hit_rate;
    };
    std::array<Result, 4> results;
    std::array<int, 4> access_counts = {8, 32, 64, 128};

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;

    for (int i = 0; i < 4; ++i) {
        GDDR6Harness harness(gddr6_16000_config());
        int accesses = access_counts[i];

        for (int col = 0; col < accesses; ++col) {
            harness.submit_read(addr(BANK, ROW, col * CACHE_LINE_BYTES));
        }
        harness.run_until_complete();

        const auto& stats = harness.stats();
        results[i] = {
            accesses,
            harness.current_cycle(),
            static_cast<double>(accesses * CACHE_LINE_BYTES) / harness.current_cycle(),
            100.0 * stats.page_hits / accesses
        };
    }

    std::cout << "\n--- Page Utilization Results ---" << std::endl;
    std::cout << std::setw(12) << "Accesses"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle"
              << std::setw(14) << "Page Hit %"
              << std::setw(14) << "Utilization" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (const auto& r : results) {
        double utilization = 100.0 * r.accesses_per_page / CACHE_LINES_PER_PAGE;
        std::cout << std::setw(12) << r.accesses_per_page
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::setw(13) << std::setprecision(1) << r.page_hit_rate << "%"
                  << std::setw(13) << utilization << "%" << std::endl;
    }

    double speedup = results[3].bytes_per_cycle / results[0].bytes_per_cycle;
    std::cout << "\nBandwidth improvement (128 vs 8): " << std::setprecision(1)
              << speedup << "x" << std::endl;

    std::cout << "PASS: Page utilization comparison completed" << std::endl;
    return true;
}

/// Compare GDDR6 (16 banks) vs LPDDR5-equivalent (8 banks) page bursts
bool test_bank_scaling_page_burst() {
    std::cout << "\n=== Test: Bank Scaling with Full Page Bursts ===" << std::endl;
    std::cout << "Compares 4, 8, 16 banks with full page utilization" << std::endl;

    struct Result {
        int num_banks;
        uint64_t total_bytes;
        uint64_t cycles;
        double bytes_per_cycle;
    };
    std::array<Result, 3> results;

    constexpr uint32_t ROW = 100;

    // 4 banks (one per bank group)
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 4> banks = {0, 4, 8, 12};
        for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, col * CACHE_LINE_BYTES));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 4 * CACHE_LINES_PER_PAGE * CACHE_LINE_BYTES;
        results[0] = {4, bytes, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // 8 banks (two per bank group)
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 8> banks = {0, 1, 4, 5, 8, 9, 12, 13};
        for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, col * CACHE_LINE_BYTES));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 8 * CACHE_LINES_PER_PAGE * CACHE_LINE_BYTES;
        results[1] = {8, bytes, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // 16 banks (all)
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 16> banks = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, col * CACHE_LINE_BYTES));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 16 * CACHE_LINES_PER_PAGE * CACHE_LINE_BYTES;
        results[2] = {16, bytes, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    std::cout << "\n--- Bank Scaling Results (Full Page Bursts) ---" << std::endl;
    std::cout << std::setw(8) << "Banks"
              << std::setw(14) << "Total KB"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    double baseline = results[0].bytes_per_cycle;
    for (const auto& r : results) {
        std::cout << std::setw(8) << r.num_banks
                  << std::setw(14) << (r.total_bytes / 1024)
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::setw(11) << std::setprecision(1) << (r.bytes_per_cycle / baseline) << "x"
                  << std::endl;
    }

    std::cout << "PASS: Bank scaling with page bursts completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "GDDR6 Pattern: Page Burst (Max Page Hits)" << std::endl;
    std::cout << "Tests optimal bandwidth with full page utilization" << std::endl;
    std::cout << "Page size: " << PAGE_SIZE_BYTES << " bytes ("
              << CACHE_LINES_PER_PAGE << " cache lines)" << std::endl;
    std::cout << "================================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = true;
    pass &= test_full_page_burst();
    pass &= test_multi_bank_page_burst();
    pass &= test_page_utilization_comparison();
    pass &= test_bank_scaling_page_burst();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        // Generate trace with full page burst on all 16 banks
        constexpr std::array<uint8_t, 16> banks = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        constexpr uint32_t ROW = 100;

        // Access first 16 cache lines per bank for manageable trace size
        for (int col = 0; col < 16; ++col) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, col * CACHE_LINE_BYTES));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("bandwidth", "page_burst_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    if (pass) {
        std::cout << "=== PASS ===" << std::endl;
    } else {
        std::cout << "=== FAIL ===" << std::endl;
    }
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
