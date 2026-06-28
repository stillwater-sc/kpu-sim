// patterns/memory/lpddr5/four-bank/page-hit-burst/main.cpp
//
// Pattern: Sustained page hits across four banks
// Tests: Peak throughput with optimal access patterns
//
// Strategy:
// - Open pages in 4 banks (one per bank group for speed)
// - Then burst page hits to maximize data bus utilization
// - Minimal activate overhead, maximum data transfer
//
// This represents the ideal case for tile data access!
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"
#include "../common/workloads.hpp"
#include "../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Test page hit burst after initial page opens
/// First access opens pages, then all subsequent are page hits
bool test_page_hit_burst_basic() {
    std::cout << "\n=== Test: Page Hit Burst (Basic) ===" << std::endl;
    std::cout << "Configuration: Banks 0,4,8,12 (one per group)" << std::endl;
    std::cout << "Pattern: Open all pages, then burst page hits" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Phase 1: Open pages (4 page empties)
    for (int b = 0; b < 4; ++b) {
        harness.submit_read(make_address(banks[b], ROW, 0));
    }

    // Phase 2: Burst page hits (16 page hits)
    for (int round = 1; round <= 4; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 20 reads (4 initial + 16 burst)
    // 4 page empties (initial opens)
    // 16 page hits (burst)
    if (!harness.verify_stats(20, 0, 16, 4, 0)) return false;

    std::cout << "PASS: Page hit burst basic works correctly" << std::endl;
    return true;
}

/// Test sustained page hits (long burst)
/// Maximizes page hit ratio
bool test_page_hit_burst_sustained() {
    std::cout << "\n=== Test: Page Hit Burst (Sustained) ===" << std::endl;
    std::cout << "Configuration: 60 page hits after 4 page opens" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // 16 rounds × 4 banks = 64 total
    // Round 0: 4 page empties
    // Rounds 1-15: 60 page hits
    for (int round = 0; round < 16; ++round) {
        for (int b = 0; b < 4; ++b) {
            // Stay within same row (column access varies)
            harness.submit_read(make_address(banks[b], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 64 reads: 4 page empty + 60 page hits
    if (!harness.verify_stats(64, 0, 60, 4, 0)) return false;

    // Check page hit ratio
    const auto& stats = harness.stats();
    double hit_ratio = 100.0 * static_cast<double>(stats.page_hits) / static_cast<double>(stats.page_hits + stats.page_empty + stats.page_conflicts);
    std::cout << "Page hit ratio: " << std::fixed << std::setprecision(1) << hit_ratio << "%" << std::endl;

    if (hit_ratio < 90.0) {
        std::cerr << "FAIL: Page hit ratio too low (expected >90%)" << std::endl;
        return false;
    }

    std::cout << "PASS: Page hit burst sustained works correctly" << std::endl;
    return true;
}

/// Test interleaved reads and writes with page hits
/// Models double-buffering: read from banks 0,4; write to banks 8,12
bool test_page_hit_double_buffer() {
    std::cout << "\n=== Test: Page Hit Double Buffer ===" << std::endl;
    std::cout << "Configuration: Read banks 0,4 (input); Write banks 8,12 (output)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // Simulate double-buffering pattern:
    // - Read input tiles from banks 0, 4
    // - Write output tiles to banks 8, 12
    for (int round = 0; round < 8; ++round) {
        // Read from input banks (page hits after first round)
        harness.submit_read(make_address(0, ROW, round * 64));
        harness.submit_read(make_address(4, ROW, round * 64));

        // Write to output banks (page hits after first round)
        harness.submit_write(make_address(8, ROW, round * 64));
        harness.submit_write(make_address(12, ROW, round * 64));
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads + 16 writes = 32 total
    // 4 banks × (1 page empty + 7 page hits) = 4 empty + 28 hits
    if (!harness.verify_stats(16, 16, 28, 4, 0)) return false;

    std::cout << "PASS: Page hit double buffer works correctly" << std::endl;
    return true;
}

/// Compare page hit burst vs page conflict workload
/// Shows the performance impact of access patterns
bool test_hit_vs_conflict_comparison() {
    std::cout << "\n=== Test: Page Hits vs Page Conflicts Comparison ===" << std::endl;

    const uint8_t banks[4] = {0, 4, 8, 12};

    // Page hit workload
    uint64_t hit_cycles;
    {
        std::cout << "\n--- Page Hit Workload ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint32_t ROW = 100;
        for (int round = 0; round < 4; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address(banks[b], ROW, round * 64));
            }
        }

        harness.run_until_complete(20000);
        hit_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << hit_cycles << std::endl;
        harness.print_stats();
    }

    // Page conflict workload
    uint64_t conflict_cycles;
    {
        std::cout << "\n--- Page Conflict Workload ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int round = 0; round < 4; ++round) {
            for (int b = 0; b < 4; ++b) {
                // Different row each round = page conflicts
                harness.submit_read(make_address(banks[b], 100 * (round + 1), 0));
            }
        }

        harness.run_until_complete(20000);
        conflict_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << conflict_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Page hits:     " << hit_cycles << " cycles" << std::endl;
    std::cout << "Page conflicts: " << conflict_cycles << " cycles" << std::endl;

    if (hit_cycles < conflict_cycles) {
        double speedup = 100.0 * static_cast<double>(conflict_cycles - hit_cycles) / static_cast<double>(conflict_cycles);
        std::cout << "Page hits are " << std::fixed << std::setprecision(1)
                  << speedup << "% faster" << std::endl;
        std::cout << "Cycles saved: " << (conflict_cycles - hit_cycles) << std::endl;
    }

    return true;
}

/// Test tile-like access pattern
/// Simulates loading 4 tiles (one per bank) with sequential column access
bool test_tile_access_pattern() {
    std::cout << "\n=== Test: Tile Access Pattern ===" << std::endl;
    std::cout << "Configuration: 4 tiles (256 bytes each) in 4 banks" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t TILE_ROWS[4] = {0, 1, 2, 3};  // Each tile starts at different row

    // Load 4 cache lines (64 bytes each) from each tile = 256 bytes per tile
    for (int line = 0; line < 4; ++line) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], TILE_ROWS[b], line * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads (4 banks × 4 cache lines)
    // Each bank: 1 page empty + 3 page hits
    if (!harness.verify_stats(16, 0, 12, 4, 0)) return false;

    // Calculate effective bandwidth
    uint64_t total_bytes = 16 * 64;  // 16 cache lines × 64 bytes
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(cycles);
    std::cout << "Effective throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: Tile access pattern works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    auto workload = make_page_hit_burst_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Page Hit Burst" << std::endl;
    std::cout << "Tests peak throughput with sustained page hits" << std::endl;
    std::cout << "================================================" << std::endl;

    bool run_fidelity = false;
    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = true;
    pass &= test_page_hit_burst_basic();
    pass &= test_page_hit_burst_sustained();
    pass &= test_page_hit_double_buffer();
    pass &= test_hit_vs_conflict_comparison();
    pass &= test_tile_access_pattern();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        const uint8_t banks[4] = {0, 4, 8, 12};
        const uint32_t ROW = 100;
        for (int round = 0; round < 4; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address(banks[b], ROW, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("four-bank", "page_hit_burst_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
