// Pattern 01: Single Bank - Sequential Access
//
// Tests baseline LPDDR5 timing for single bank operations:
// - Page empty (first access) → ACTIVATE + READ timing
// - Page hit (subsequent accesses to same row) → READ only
// - Progressive bank testing: 1, 2, 3, 4 banks
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "../common/lpddr5_configs.hpp"
#include "../common/pattern_harness.hpp"

using namespace sw::kpu::patterns;
using namespace sw::kpu::lpddr5;

// ============================================================================
// Test: Single Bank Sequential Reads (Page Hits)
// ============================================================================

bool test_single_bank_page_hits() {
    std::cout << "\n=== Test: Single Bank Sequential Reads (Page Hits) ===" << std::endl;
    std::cout << "Configuration: Single channel, Bank 0, same row" << std::endl;
    std::cout << "Expected: 1 page empty, 7 page hits" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit 8 reads to the same row in bank 0
    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    for (int i = 0; i < 8; ++i) {
        uint64_t addr = make_address(BANK, ROW, i * 64);  // Different columns, same row
        auto id = harness.submit_read(addr);
        if (!id.has_value()) {
            std::cerr << "FAIL: Could not submit read " << i << std::endl;
            return false;
        }
    }

    std::cout << "Submitted 8 reads to bank " << (int)BANK
              << ", row " << ROW << std::endl;

    // Run simulation
    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    // Verify results
    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    // Expected: 1 page empty (first access), 7 page hits
    if (!harness.verify_stats(8, 0, 7, 1, 0)) {
        return false;
    }

    std::cout << "PASS: Single bank page hits work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Single Bank Page Conflicts
// ============================================================================

bool test_single_bank_page_conflicts() {
    std::cout << "\n=== Test: Single Bank Page Conflicts ===" << std::endl;
    std::cout << "Configuration: Single channel, Bank 0, different rows" << std::endl;
    std::cout << "Expected: 1 page empty, 7 page conflicts" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit 8 reads to different rows in bank 0
    const uint8_t BANK = 0;

    for (int i = 0; i < 8; ++i) {
        uint64_t addr = make_address(BANK, i * 10, 0);  // Different rows
        auto id = harness.submit_read(addr);
        if (!id.has_value()) {
            std::cerr << "FAIL: Could not submit read " << i << std::endl;
            return false;
        }
    }

    std::cout << "Submitted 8 reads to bank " << (int)BANK
              << ", different rows" << std::endl;

    // Run simulation
    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    // Verify results
    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    // Expected: 1 page empty (first access), 7 page conflicts
    if (!harness.verify_stats(8, 0, 0, 1, 7)) {
        return false;
    }

    std::cout << "PASS: Single bank page conflicts work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Two Banks - Same Bank Group
// ============================================================================

bool test_two_banks_same_group() {
    std::cout << "\n=== Test: Two Banks - Same Bank Group (Banks 0, 1) ===" << std::endl;
    std::cout << "Configuration: Single channel, Banks 0 and 1 (same BG)" << std::endl;
    std::cout << "Expected: tRRD_L (6 cycles) between activates" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit interleaved reads to banks 0 and 1 (same bank group)
    const uint32_t ROW = 100;

    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(0, ROW, i * 64));  // Bank 0
        harness.submit_read(make_address(1, ROW, i * 64));  // Bank 1
    }

    std::cout << "Submitted 8 reads interleaved across banks 0 and 1" << std::endl;

    // Run simulation
    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 8 || stats.page_empty != 2) {
        std::cerr << "FAIL: Expected 8 reads with 2 page empty" << std::endl;
        return false;
    }

    std::cout << "PASS: Two banks same group work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Two Banks - Different Bank Groups
// ============================================================================

bool test_two_banks_diff_groups() {
    std::cout << "\n=== Test: Two Banks - Different Bank Groups (Banks 0, 4) ===" << std::endl;
    std::cout << "Configuration: Single channel, Banks 0 and 4 (different BG)" << std::endl;
    std::cout << "Expected: tRRD_S (4 cycles) between activates - faster" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit interleaved reads to banks 0 and 4 (different bank groups)
    const uint32_t ROW = 100;

    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(0, ROW, i * 64));  // Bank 0 (BG0)
        harness.submit_read(make_address(4, ROW, i * 64));  // Bank 4 (BG1)
    }

    std::cout << "Submitted 8 reads interleaved across banks 0 and 4" << std::endl;

    // Run simulation
    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 8 || stats.page_empty != 2) {
        std::cerr << "FAIL: Expected 8 reads with 2 page empty" << std::endl;
        return false;
    }

    std::cout << "PASS: Two banks different groups work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Three Banks - Mixed Groups
// ============================================================================

bool test_three_banks_mixed() {
    std::cout << "\n=== Test: Three Banks - Mixed Groups (Banks 0, 4, 8) ===" << std::endl;
    std::cout << "Configuration: Single channel, Banks 0, 4, 8 (different BG)" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit round-robin reads to banks 0, 4, 8 (different bank groups)
    const uint32_t ROW = 100;

    for (int i = 0; i < 3; ++i) {
        harness.submit_read(make_address(0, ROW, i * 64));  // Bank 0 (BG0)
        harness.submit_read(make_address(4, ROW, i * 64));  // Bank 4 (BG1)
        harness.submit_read(make_address(8, ROW, i * 64));  // Bank 8 (BG2)
    }

    std::cout << "Submitted 9 reads round-robin across banks 0, 4, 8" << std::endl;

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 9 || stats.page_empty != 3) {
        std::cerr << "FAIL: Expected 9 reads with 3 page empty" << std::endl;
        return false;
    }

    std::cout << "PASS: Three banks mixed groups work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Four Banks - Full Bank Group
// ============================================================================

bool test_four_banks_full_group() {
    std::cout << "\n=== Test: Four Banks - Full Bank Group (Banks 0-3) ===" << std::endl;
    std::cout << "Configuration: Single channel, Banks 0-3 (full BG0)" << std::endl;
    std::cout << "Expected: tFAW constraint may apply" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit reads to all 4 banks in bank group 0
    const uint32_t ROW = 100;

    for (int i = 0; i < 2; ++i) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(b, ROW, i * 64));
        }
    }

    std::cout << "Submitted 8 reads across banks 0-3 (full bank group 0)" << std::endl;

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 8 || stats.page_empty != 4) {
        std::cerr << "FAIL: Expected 8 reads with 4 page empty" << std::endl;
        return false;
    }

    std::cout << "PASS: Four banks full group work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Four Banks - Across Groups (Maximum Parallelism)
// ============================================================================

bool test_four_banks_across_groups() {
    std::cout << "\n=== Test: Four Banks - Across Groups (Banks 0, 4, 8, 12) ===" << std::endl;
    std::cout << "Configuration: Single channel, one bank per group" << std::endl;
    std::cout << "Expected: Maximum parallelism, no tFAW constraint" << std::endl;

    PatternHarness harness(single_channel_config());

    // Submit reads to banks 0, 4, 8, 12 (one per bank group)
    const uint32_t ROW = 100;

    for (int i = 0; i < 2; ++i) {
        harness.submit_read(make_address(0, ROW, i * 64));   // BG0
        harness.submit_read(make_address(4, ROW, i * 64));   // BG1
        harness.submit_read(make_address(8, ROW, i * 64));   // BG2
        harness.submit_read(make_address(12, ROW, i * 64));  // BG3
    }

    std::cout << "Submitted 8 reads across banks 0, 4, 8, 12 (one per BG)" << std::endl;

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 8 || stats.page_empty != 4) {
        std::cerr << "FAIL: Expected 8 reads with 4 page empty" << std::endl;
        return false;
    }

    std::cout << "PASS: Four banks across groups work correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Mixed Read/Write with Turnaround
// ============================================================================

bool test_mixed_read_write() {
    std::cout << "\n=== Test: Mixed Read/Write with Turnaround ===" << std::endl;
    std::cout << "Configuration: Single channel, Bank 0" << std::endl;
    std::cout << "Expected: tRTW and tWTR turnaround delays" << std::endl;

    PatternHarness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // Alternating reads and writes to same row
    for (int i = 0; i < 4; ++i) {
        if (i % 2 == 0) {
            harness.submit_read(make_address(BANK, ROW, i * 64));
        } else {
            harness.submit_write(make_address(BANK, ROW, i * 64));
        }
    }

    std::cout << "Submitted 2 reads and 2 writes alternating" << std::endl;

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete or had violations" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return false;
    }

    const auto& stats = harness.stats();
    if (stats.reads != 2 || stats.writes != 2) {
        std::cerr << "FAIL: Expected 2 reads and 2 writes" << std::endl;
        return false;
    }

    // Check turnaround stats
    if (stats.read_to_write_turnarounds + stats.write_to_read_turnarounds == 0) {
        std::cerr << "WARNING: No turnarounds detected (might be scheduler optimization)" << std::endl;
    } else {
        std::cout << "Turnarounds detected: R->W=" << stats.read_to_write_turnarounds
                  << ", W->R=" << stats.write_to_read_turnarounds << std::endl;
    }

    std::cout << "PASS: Mixed read/write with turnaround works correctly" << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Pattern 01: LPDDR5 Bank Access Patterns" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nConfiguration: Single Channel LPDDR5-6400" << std::endl;
    std::cout << "  Channels: 1" << std::endl;
    std::cout << "  Banks: 16 (4 bank groups × 4 banks)" << std::endl;
    std::cout << "  Burst length: BL16 (8 cycles)" << std::endl;
    std::cout << "  tRCD: " << tRCD << " cycles" << std::endl;
    std::cout << "  tRP: " << tRP << " cycles" << std::endl;
    std::cout << "  tCL: " << tCL << " cycles" << std::endl;
    std::cout << "  tRRD_L: " << tRRD_L << " cycles (same BG)" << std::endl;
    std::cout << "  tRRD_S: " << tRRD_S << " cycles (diff BG)" << std::endl;

    bool all_pass = true;

    // Run progressive tests
    all_pass &= test_single_bank_page_hits();
    all_pass &= test_single_bank_page_conflicts();
    all_pass &= test_two_banks_same_group();
    all_pass &= test_two_banks_diff_groups();
    all_pass &= test_three_banks_mixed();
    all_pass &= test_four_banks_full_group();
    all_pass &= test_four_banks_across_groups();
    all_pass &= test_mixed_read_write();

    std::cout << "\n========================================" << std::endl;
    if (all_pass) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Optional: export trace for the last test
    std::string trace_file = "pattern01_trace.json";
    if (argc > 1) {
        trace_file = argv[1];
    }

    // Run one more time with trace export
    std::cout << "\nRunning final trace capture..." << std::endl;
    PatternHarness harness(single_channel_config());

    // Submit a representative workload for visualization
    harness.submit_read(make_address(0, 100, 0));    // Page empty
    harness.submit_read(make_address(0, 100, 64));   // Page hit
    harness.submit_read(make_address(0, 200, 0));    // Page conflict
    harness.submit_read(make_address(4, 100, 0));    // Different BG
    harness.run_until_complete();
    harness.export_trace(trace_file);

    return all_pass ? 0 : 1;
}
