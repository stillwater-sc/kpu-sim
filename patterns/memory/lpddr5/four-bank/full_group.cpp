// patterns/memory/lpddr5/four-bank/full-group/main.cpp
//
// Pattern: All four banks in one bank group
// Tests: tFAW (Four Activate Window) constraint
//
// tFAW Constraint:
// - Maximum of 4 activates within any rolling 24-cycle window
// - Banks 0,1,2,3 are all in bank group 0
// - With tRRD_L = 6, activates at cycles 0,6,12,18 fit in tFAW
// - 5th activate must wait until cycle 24
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

/// Test round-robin across all 4 banks in group 0
/// This tests basic tFAW behavior with one activate per bank
bool test_full_group_round_robin() {
    std::cout << "\n=== Test: Four Banks Full Group (Round-Robin) ===" << std::endl;
    std::cout << "Configuration: Banks 0,1,2,3 (all in group 0)" << std::endl;
    std::cout << "tFAW = " << tFAW << " cycles (max 4 activates in window)" << std::endl;
    std::cout << "tRRD_L = " << tRRD_L << " cycles between activates" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Round-robin: 0,1,2,3, 0,1,2,3 (8 reads total, 2 per bank)
    for (int round = 0; round < 2; ++round) {
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

    // 8 reads (4 banks × 2 accesses)
    // Each bank: 1 page empty + 1 page hit
    // Total: 8 reads, 0 writes, 4 page hits, 4 page empty, 0 conflicts
    if (!harness.verify_stats(8, 0, 4, 4, 0)) return false;

    std::cout << "PASS: Four banks full group round-robin works correctly" << std::endl;
    return true;
}

/// Test sequential bursts to all 4 banks
/// Each bank gets multiple accesses before moving to next
bool test_full_group_sequential() {
    std::cout << "\n=== Test: Four Banks Full Group (Sequential) ===" << std::endl;
    std::cout << "Configuration: Burst to each bank in sequence" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Sequential bursts: 4 reads to each bank
    for (uint8_t bank = 0; bank < 4; ++bank) {
        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(bank, 100 + bank, i * 64));
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

    std::cout << "PASS: Four banks full group sequential works correctly" << std::endl;
    return true;
}

/// Test page conflicts across all 4 banks
/// Each access to a bank targets a different row
bool test_full_group_conflicts() {
    std::cout << "\n=== Test: Four Banks Full Group (Page Conflicts) ===" << std::endl;
    std::cout << "Configuration: Different row per access per bank" << std::endl;
    std::cout << "This stresses both tFAW and page conflict handling" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // 3 rounds of 4 banks, each access to different row
    for (int round = 0; round < 3; ++round) {
        for (uint8_t bank = 0; bank < 4; ++bank) {
            harness.submit_read(make_address(bank, 100 * (round + 1) + bank, 0));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 12 reads (4 banks × 3 accesses)
    // Each bank: 1 page empty + 2 page conflicts
    // Total: 12 reads, 0 writes, 0 page hits, 4 page empty, 8 page conflicts
    if (!harness.verify_stats(12, 0, 0, 4, 8)) return false;

    std::cout << "PASS: Four banks full group with conflicts works correctly" << std::endl;
    return true;
}

/// Test sustained load to stress tFAW
/// Many rounds of 4-bank access patterns
bool test_full_group_sustained() {
    std::cout << "\n=== Test: Four Banks Sustained tFAW Stress ===" << std::endl;
    std::cout << "Configuration: 8 rounds of 4-bank round-robin" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // 8 rounds × 4 banks = 32 accesses
    // First round opens pages, rest are page hits
    for (int round = 0; round < 8; ++round) {
        for (uint8_t bank = 0; bank < 4; ++bank) {
            harness.submit_read(make_address(bank, 100 + bank, round * 64));
        }
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 32 reads (4 banks × 8 accesses)
    // Each bank: 1 page empty + 7 page hits
    // Total: 32 reads, 0 writes, 28 page hits, 4 page empty, 0 conflicts
    if (!harness.verify_stats(32, 0, 28, 4, 0)) return false;

    std::cout << "PASS: Four banks sustained tFAW stress works correctly" << std::endl;
    return true;
}

/// Test mixed read/write across all 4 banks
bool test_full_group_mixed_rw() {
    std::cout << "\n=== Test: Four Banks Full Group (Mixed R/W) ===" << std::endl;
    std::cout << "Configuration: Banks 0,2 read; Banks 1,3 write" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // 4 rounds: R0, W1, R2, W3
    for (int round = 0; round < 4; ++round) {
        harness.submit_read(make_address(0, ROW, round * 64));
        harness.submit_write(make_address(1, ROW, round * 64));
        harness.submit_read(make_address(2, ROW, round * 64));
        harness.submit_write(make_address(3, ROW, round * 64));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads + 8 writes = 16 total
    // Each bank: 1 page empty + 3 page hits
    // Total: 8 reads, 8 writes, 12 page hits, 4 page empty, 0 conflicts
    if (!harness.verify_stats(8, 8, 12, 4, 0)) return false;

    std::cout << "PASS: Four banks full group mixed R/W works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    auto workload = make_four_banks_full_group_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Four Banks Full Group" << std::endl;
    std::cout << "Tests tFAW constraint (Four Activate Window)" << std::endl;
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
    pass &= test_full_group_round_robin();
    pass &= test_full_group_sequential();
    pass &= test_full_group_conflicts();
    pass &= test_full_group_sustained();
    pass &= test_full_group_mixed_rw();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        for (int round = 0; round < 2; ++round) {
            for (uint8_t bank = 0; bank < 4; ++bank) {
                harness.submit_read(make_address(bank, 100 + bank, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("four-bank", "full_group_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
