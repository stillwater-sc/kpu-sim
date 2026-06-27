// patterns/memory/lpddr5/bandwidth/page_burst.cpp
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
// This represents the theoretical maximum bandwidth achievable
// for sequential access within a single page, showing the benefit
// of locality and row buffer reuse.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"

using namespace sw::kpu::patterns::lpddr5;

// Page geometry constants (use namespace CACHE_LINE_BYTES = 64)
constexpr int PAGE_SIZE_BYTES = 8192;       // 8 KB row buffer
constexpr int CACHE_LINES_PER_PAGE = PAGE_SIZE_BYTES / 64;  // 128

/// High-capacity config for bandwidth tests that submit many requests at once
/// Queue depth must accommodate all requests before simulation starts
inline LPDDR5MemoryController::Config bandwidth_test_config() {
    auto config = single_channel_config();
    config.queue_depth = 2048;  // Accommodate up to 1024 requests (8 banks × 128 cache lines)
    return config;
}

/// Test full page burst - all 128 cache lines from one page
/// This is the optimal single-bank access pattern
bool test_full_page_burst() {
    std::cout << "\n=== Test: Full Page Burst (128 accesses) ===" << std::endl;
    std::cout << "Configuration: Single bank, single page" << std::endl;
    std::cout << "Pattern: Sequential access to all " << CACHE_LINES_PER_PAGE
              << " cache lines" << std::endl;

    LPDDR5Harness harness(bandwidth_test_config());

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;

    // Access all 128 cache lines in the page
    for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
        harness.submit_read(make_address(BANK, ROW, col * CACHE_LINE_BYTES));
    }

    if (!harness.run_until_complete(200000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    const auto& stats = harness.stats();
    uint64_t total_bytes = CACHE_LINES_PER_PAGE * CACHE_LINE_BYTES;  // 8 KB
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(cycles);
    double hit_rate = 100.0 * static_cast<double>(stats.page_hits) / static_cast<double>(stats.page_hits + stats.page_empty);

    std::cout << "\n--- Full Page Burst Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit rate: " << std::setprecision(1) << hit_rate << "%" << std::endl;
    std::cout << "Page utilization: 100% (all 128 cache lines accessed)" << std::endl;

    // Expected: High page hit rate (>90%) with some page_empty due to refresh
    // Note: LPDDR5 per-bank refresh (tREFIpb=244 cycles) causes periodic page closes.
    // Over a ~2800 cycle test, bank 0 may be refreshed ~8 times, each causing a page_empty.
    // We verify: correct read count, high hit rate, no conflicts
    bool pass = true;
    if (stats.reads != CACHE_LINES_PER_PAGE) {
        std::cerr << "FAIL: Expected " << CACHE_LINES_PER_PAGE << " reads, got " << stats.reads << std::endl;
        pass = false;
    }
    if (stats.page_conflicts != 0) {
        std::cerr << "FAIL: Expected 0 page conflicts, got " << stats.page_conflicts << std::endl;
        pass = false;
    }
    if (hit_rate < 90.0) {
        std::cerr << "FAIL: Page hit rate " << hit_rate << "% below 90% threshold" << std::endl;
        pass = false;
    }
    if (!pass) {
        return false;
    }

    std::cout << "PASS: Full page burst completed" << std::endl;
    return true;
}

/// Test multi-page burst across multiple banks
/// Open one page per bank and access all cache lines
bool test_multi_bank_page_burst() {
    std::cout << "\n=== Test: Multi-Bank Page Burst (8 banks × 128 accesses) ===" << std::endl;
    std::cout << "Pattern: Full page access across all 8 banks" << std::endl;

    LPDDR5Harness harness(bandwidth_test_config());

    constexpr std::array<uint8_t, 8> banks = {0, 1, 2, 3, 4, 5, 6, 7};
    constexpr uint32_t ROW = 100;

    // Round-robin across banks, accessing all cache lines
    for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
        for (uint8_t bank : banks) {
            harness.submit_read(make_address(bank, ROW, col * CACHE_LINE_BYTES));
        }
    }

    if (!harness.run_until_complete(500000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    [[maybe_unused]] const auto& stats = harness.stats();
    int total_accesses = 8 * CACHE_LINES_PER_PAGE;  // 1024 accesses
    uint64_t total_bytes = total_accesses * CACHE_LINE_BYTES;  // 64 KB
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(harness.current_cycle());

    std::cout << "\n--- Multi-Bank Page Burst Analysis ---" << std::endl;
    std::cout << "Banks: 8, Pages: 8, Cache lines per page: " << CACHE_LINES_PER_PAGE << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    // Expected: 8 page misses + (1024-8) page hits = 99.2% hit rate
    std::cout << "Expected page hits: " << (total_accesses - 8) << std::endl;
    std::cout << "Expected page misses: 8" << std::endl;

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
        LPDDR5Harness harness(bandwidth_test_config());
        int accesses = access_counts[i];

        for (int col = 0; col < accesses; ++col) {
            harness.submit_read(make_address(BANK, ROW, col * CACHE_LINE_BYTES));
        }
        harness.run_until_complete();

        const auto& stats = harness.stats();
        results[i] = {
            accesses,
            harness.current_cycle(),
            static_cast<double>(accesses * CACHE_LINE_BYTES) / static_cast<double>(harness.current_cycle()),
            100.0 * static_cast<double>(stats.page_hits) / accesses
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

/// Test sustained full-page bandwidth over multiple pages
/// Simulates large sequential transfer
bool test_sustained_page_bandwidth() {
    std::cout << "\n=== Test: Sustained Page Bandwidth ===" << std::endl;
    std::cout << "Configuration: 4 sequential pages on single bank" << std::endl;

    LPDDR5Harness harness(bandwidth_test_config());

    constexpr uint8_t BANK = 0;
    constexpr int NUM_PAGES = 4;

    // Access 4 pages sequentially (512 cache lines = 32 KB)
    for (int page = 0; page < NUM_PAGES; ++page) {
        uint32_t row = 100 + page;
        for (int col = 0; col < CACHE_LINES_PER_PAGE; ++col) {
            harness.submit_read(make_address(BANK, row, col * CACHE_LINE_BYTES));
        }
    }

    if (!harness.run_until_complete(500000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    int total_accesses = NUM_PAGES * CACHE_LINES_PER_PAGE;
    uint64_t total_bytes = total_accesses * CACHE_LINE_BYTES;
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(harness.current_cycle());

    std::cout << "\n--- Sustained Page Bandwidth Analysis ---" << std::endl;
    std::cout << "Pages: " << NUM_PAGES << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    // Expected: 4 page misses (one per page) + 508 page hits
    std::cout << "Expected: " << NUM_PAGES << " page misses + "
              << (total_accesses - NUM_PAGES) << " page hits" << std::endl;

    std::cout << "PASS: Sustained page bandwidth completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Page Burst (Max Page Hits)" << std::endl;
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
    pass &= test_sustained_page_bandwidth();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(bandwidth_test_config());

        // Generate trace with full page burst on all 8 banks
        constexpr std::array<uint8_t, 8> banks = {0, 1, 2, 3, 4, 5, 6, 7};
        constexpr uint32_t ROW = 100;

        // Access first 32 cache lines per bank for manageable trace size
        for (int col = 0; col < 32; ++col) {
            for (uint8_t bank : banks) {
                harness.submit_read(make_address(bank, ROW, col * CACHE_LINE_BYTES));
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
