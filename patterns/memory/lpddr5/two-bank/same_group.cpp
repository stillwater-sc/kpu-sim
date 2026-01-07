// patterns/memory/lpddr5/two-bank/same-group/main.cpp
//
// Pattern: Two banks in the same bank group (tRRD_L constraint)
// Tests: Activate-to-activate timing within same bank group
//
// Bank Groups in LPDDR5 (4 groups × 4 banks):
//   Group 0: Banks 0, 1, 2, 3
//   Group 1: Banks 4, 5, 6, 7
//   Group 2: Banks 8, 9, 10, 11
//   Group 3: Banks 12, 13, 14, 15
//
// tRRD_L = 6 cycles (ACT-to-ACT within same bank group)
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

/// Test interleaved reads to two banks in the same bank group
/// Banks 0 and 1 are both in bank group 0
/// Expected: tRRD_L (6 cycles) spacing between activates
bool test_same_group_interleaved() {
    std::cout << "\n=== Test: Two Banks Same Group (Interleaved) ===" << std::endl;
    std::cout << "Configuration: Banks 0,1 (both in group 0), interleaved reads" << std::endl;
    std::cout << "tRRD_L constraint: 6 cycles between ACT commands" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;  // Bank group 0
    const uint8_t BANK_B = 1;  // Bank group 0 (same group)
    const uint32_t ROW_A = 100;
    const uint32_t ROW_B = 200;

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
    // Expected: 8 reads, 0 writes, 6 page hits, 2 page empty, 0 conflicts
    if (!harness.verify_stats(8, 0, 6, 2, 0)) return false;

    std::cout << "PASS: Same bank group interleaved accesses work correctly" << std::endl;
    return true;
}

/// Test sequential bursts to two banks in same group
/// First 4 reads to Bank 0, then 4 reads to Bank 1
bool test_same_group_sequential() {
    std::cout << "\n=== Test: Two Banks Same Group (Sequential Bursts) ===" << std::endl;
    std::cout << "Configuration: Bank 0 burst, then Bank 1 burst" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;
    const uint8_t BANK_B = 1;
    const uint32_t ROW_A = 100;
    const uint32_t ROW_B = 200;

    // First burst: 4 reads to Bank 0
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(BANK_A, ROW_A, i * 64));
    }

    // Second burst: 4 reads to Bank 1
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(BANK_B, ROW_B, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads: Bank A gets 1 page empty + 3 page hits
    //          Bank B gets 1 page empty + 3 page hits
    // Total: 8 reads, 0 writes, 6 page hits, 2 page empty, 0 conflicts
    if (!harness.verify_stats(8, 0, 6, 2, 0)) return false;

    std::cout << "PASS: Same bank group sequential bursts work correctly" << std::endl;
    return true;
}

/// Test page conflicts across two banks in same group
/// Alternating between different rows in each bank
bool test_same_group_conflicts() {
    std::cout << "\n=== Test: Two Banks Same Group (Page Conflicts) ===" << std::endl;
    std::cout << "Configuration: Banks 0,1, alternating rows causing conflicts" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;
    const uint8_t BANK_B = 1;

    // Each read goes to a different row in alternating banks
    // This creates page conflicts within each bank
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(BANK_A, 100 + i, 0));  // Different rows
        harness.submit_read(make_address(BANK_B, 200 + i, 0));  // Different rows
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads total
    // Bank A: row 100 (page empty), row 101 (conflict), row 102 (conflict), row 103 (conflict)
    // Bank B: row 200 (page empty), row 201 (conflict), row 202 (conflict), row 203 (conflict)
    // Total: 8 reads, 0 writes, 0 page hits, 2 page empty, 6 page conflicts
    if (!harness.verify_stats(8, 0, 0, 2, 6)) return false;

    std::cout << "PASS: Same bank group with conflicts work correctly" << std::endl;
    return true;
}

/// Test all four banks in bank group 0
bool test_full_bank_group() {
    std::cout << "\n=== Test: All Four Banks in Group 0 ===" << std::endl;
    std::cout << "Configuration: Banks 0,1,2,3 (all in group 0)" << std::endl;
    std::cout << "Tests: tRRD_L between all activates, potential tFAW constraint" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Access all 4 banks in group 0, round-robin
    for (int round = 0; round < 4; ++round) {
        for (uint8_t bank = 0; bank < 4; ++bank) {
            harness.submit_read(make_address(bank, 100 + bank, round * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads total (4 banks × 4 accesses each)
    // Each bank: 1 page empty + 3 page hits
    // Total: 16 reads, 0 writes, 12 page hits, 4 page empty, 0 conflicts
    if (!harness.verify_stats(16, 0, 12, 4, 0)) return false;

    std::cout << "PASS: Full bank group (4 banks) works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    // Use the predefined two-bank same-group workload
    auto workload = make_two_banks_same_group_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Two Banks Same Group" << std::endl;
    std::cout << "Tests tRRD_L (6 cycles) constraint" << std::endl;
    std::cout << "==========================================" << std::endl;

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
    pass &= test_same_group_interleaved();
    pass &= test_same_group_sequential();
    pass &= test_same_group_conflicts();
    pass &= test_full_bank_group();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(1, 200, i * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("two-bank", "same_group_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n==========================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "==========================================" << std::endl;

    return pass ? 0 : 1;
}
