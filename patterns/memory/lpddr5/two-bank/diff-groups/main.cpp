// patterns/memory/lpddr5/two-bank/diff-groups/main.cpp
//
// Pattern: Two banks in different bank groups (tRRD_S constraint)
// Tests: Activate-to-activate timing across bank groups (faster)
//
// Bank Groups in LPDDR5 (4 groups × 4 banks):
//   Group 0: Banks 0, 1, 2, 3
//   Group 1: Banks 4, 5, 6, 7
//   Group 2: Banks 8, 9, 10, 11
//   Group 3: Banks 12, 13, 14, 15
//
// tRRD_S = 4 cycles (ACT-to-ACT across different bank groups)
// tRRD_L = 6 cycles (ACT-to-ACT within same bank group)
//
// Using different bank groups allows 33% faster activate pipelining!
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../../common/lpddr5_configs.hpp"
#include "../../common/lpddr5_harness.hpp"
#include "../../common/workloads.hpp"
#include "../../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Test interleaved reads to two banks in different bank groups
/// Banks 0 and 4 are in different bank groups (0 and 1)
/// Expected: tRRD_S (4 cycles) spacing between activates
bool test_diff_groups_interleaved() {
    std::cout << "\n=== Test: Two Banks Different Groups (Interleaved) ===" << std::endl;
    std::cout << "Configuration: Banks 0 (group 0) and 4 (group 1), interleaved" << std::endl;
    std::cout << "tRRD_S constraint: 4 cycles between ACT commands (faster)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;  // Bank group 0
    const uint8_t BANK_B = 4;  // Bank group 1 (different group!)
    const uint32_t ROW_A = 100;
    const uint32_t ROW_B = 200;

    // Verify bank groups are different
    std::cout << "Bank " << (int)BANK_A << " is in group " << (int)bank_group(BANK_A) << std::endl;
    std::cout << "Bank " << (int)BANK_B << " is in group " << (int)bank_group(BANK_B) << std::endl;
    std::cout << "Same group? " << (same_bank_group(BANK_A, BANK_B) ? "Yes" : "No") << std::endl;

    // Interleave requests: A, B, A, B, A, B, A, B (8 total)
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(BANK_A, ROW_A, i * 64));
        harness.submit_read(make_address(BANK_B, ROW_B, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads total
    // First access to each bank is page empty (2 page empties)
    // Subsequent accesses to each bank are page hits (6 page hits)
    if (!harness.verify_stats(8, 0, 6, 2, 0)) return false;

    std::cout << "PASS: Different bank group interleaved accesses work correctly" << std::endl;
    return true;
}

/// Test all four bank groups with one bank each
/// Banks 0, 4, 8, 12 (one from each group)
/// Maximum parallelism for activates
bool test_four_groups_round_robin() {
    std::cout << "\n=== Test: Four Bank Groups Round-Robin ===" << std::endl;
    std::cout << "Configuration: Banks 0,4,8,12 (one from each group)" << std::endl;
    std::cout << "Tests: Maximum activate parallelism across groups" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};  // One bank per group
    const uint32_t rows[4] = {100, 200, 300, 400};

    // Round-robin access pattern
    for (int round = 0; round < 4; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], rows[b], round * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads (4 banks × 4 accesses)
    // Each bank: 1 page empty + 3 page hits
    // Total: 16 reads, 0 writes, 12 page hits, 4 page empty, 0 conflicts
    if (!harness.verify_stats(16, 0, 12, 4, 0)) return false;

    std::cout << "PASS: Four bank groups round-robin works correctly" << std::endl;
    return true;
}

/// Compare timing between same-group and diff-group accesses
/// This demonstrates the performance benefit of spreading across groups
bool test_timing_comparison() {
    std::cout << "\n=== Test: Same Group vs Different Group Timing ===" << std::endl;

    // Same group test (Banks 0, 1)
    {
        std::cout << "\n--- Same Bank Group (Banks 0, 1) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(1, 200, i * 64));
        }

        harness.run_until_complete();
        auto same_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << same_cycles << std::endl;
        harness.print_stats();
    }

    // Different group test (Banks 0, 4)
    {
        std::cout << "\n--- Different Bank Groups (Banks 0, 4) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(4, 200, i * 64));
        }

        harness.run_until_complete();
        auto diff_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << diff_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\nExpected: Different groups should be faster due to tRRD_S < tRRD_L" << std::endl;
    std::cout << "  tRRD_S = " << tRRD_S << " cycles (different groups)" << std::endl;
    std::cout << "  tRRD_L = " << tRRD_L << " cycles (same group)" << std::endl;

    return true;
}

/// Test mixed read/write across different bank groups
bool test_diff_groups_mixed_rw() {
    std::cout << "\n=== Test: Different Groups Mixed Read/Write ===" << std::endl;
    std::cout << "Configuration: Banks 0 (reads) and 4 (writes)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t READ_BANK = 0;
    const uint8_t WRITE_BANK = 4;
    const uint32_t ROW = 100;

    // Interleave reads and writes across different bank groups
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(READ_BANK, ROW, i * 64));
        harness.submit_write(make_address(WRITE_BANK, ROW, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 4 reads + 4 writes
    // Each bank: 1 page empty + 3 page hits
    // Total: 4 reads, 4 writes, 6 page hits, 2 page empty, 0 conflicts
    if (!harness.verify_stats(4, 4, 6, 2, 0)) return false;

    std::cout << "PASS: Different groups mixed read/write works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    // Use the predefined two-bank different-groups workload
    auto workload = make_two_banks_diff_groups_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Two Banks Different Groups" << std::endl;
    std::cout << "Tests tRRD_S (4 cycles) constraint" << std::endl;
    std::cout << "============================================" << std::endl;

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
    pass &= test_diff_groups_interleaved();
    pass &= test_four_groups_round_robin();
    pass &= test_timing_comparison();
    pass &= test_diff_groups_mixed_rw();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(4, 200, i * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("two-bank", "diff_groups_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "============================================" << std::endl;

    return pass ? 0 : 1;
}
