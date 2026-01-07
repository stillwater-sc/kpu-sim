// patterns/memory/lpddr5/four-bank/across-groups/main.cpp
//
// Pattern: Four banks across different bank groups
// Tests: Maximum parallelism with tRRD_S timing
//
// Configuration:
// - Banks 0, 4, 8, 12 (one from each bank group)
// - tRRD_S (4 cycles) between all activates
// - No tFAW concern since only one bank per group
//
// This is the optimal configuration for four-bank access!
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

/// Test round-robin across 4 banks from different groups
/// Banks 0, 4, 8, 12 provide maximum activate parallelism
bool test_across_groups_round_robin() {
    std::cout << "\n=== Test: Four Banks Across Groups (Round-Robin) ===" << std::endl;
    std::cout << "Configuration: Banks 0 (BG0), 4 (BG1), 8 (BG2), 12 (BG3)" << std::endl;
    std::cout << "tRRD_S = " << tRRD_S << " cycles between all activates" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};

    // Verify bank groups
    for (int i = 0; i < 4; ++i) {
        std::cout << "Bank " << (int)banks[i] << " is in group " << (int)bank_group(banks[i]) << std::endl;
    }

    // Round-robin: 0,4,8,12, 0,4,8,12 (8 reads, 2 per bank)
    for (int round = 0; round < 2; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], 100 + b, round * 64));
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
    if (!harness.verify_stats(8, 0, 4, 4, 0)) return false;

    std::cout << "PASS: Four banks across groups round-robin works correctly" << std::endl;
    return true;
}

/// Compare timing: full group vs across groups
/// This should show the benefit of cross-group access
bool test_timing_comparison() {
    std::cout << "\n=== Test: Full Group vs Across Groups Timing ===" << std::endl;

    // Full group (Banks 0,1,2,3) - tRRD_L + tFAW constraints
    uint64_t full_group_cycles;
    {
        std::cout << "\n--- Full Bank Group (Banks 0,1,2,3) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int round = 0; round < 2; ++round) {
            for (uint8_t bank = 0; bank < 4; ++bank) {
                harness.submit_read(make_address(bank, 100 + bank, round * 64));
            }
        }

        harness.run_until_complete(20000);
        full_group_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << full_group_cycles << std::endl;
        harness.print_stats();
    }

    // Across groups (Banks 0,4,8,12) - only tRRD_S constraints
    uint64_t across_groups_cycles;
    {
        std::cout << "\n--- Across Bank Groups (Banks 0,4,8,12) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint8_t banks[4] = {0, 4, 8, 12};
        for (int round = 0; round < 2; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address(banks[b], 100 + b, round * 64));
            }
        }

        harness.run_until_complete(20000);
        across_groups_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << across_groups_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Full group (tRRD_L + tFAW): " << full_group_cycles << " cycles" << std::endl;
    std::cout << "Across groups (tRRD_S):     " << across_groups_cycles << " cycles" << std::endl;

    if (across_groups_cycles < full_group_cycles) {
        std::cout << "Across groups is " << (full_group_cycles - across_groups_cycles)
                  << " cycles faster ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (full_group_cycles - across_groups_cycles) / full_group_cycles)
                  << "% improvement)" << std::endl;
    } else {
        std::cout << "No timing difference (may be bus-bound)" << std::endl;
    }

    return true;
}

/// Test sustained high-throughput access
/// Many rounds to show consistent behavior
bool test_across_groups_sustained() {
    std::cout << "\n=== Test: Four Banks Across Groups (Sustained) ===" << std::endl;
    std::cout << "Configuration: 8 rounds of 4-bank access" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};

    // 8 rounds × 4 banks = 32 accesses
    for (int round = 0; round < 8; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], 100 + b, round * 64));
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
    if (!harness.verify_stats(32, 0, 28, 4, 0)) return false;

    std::cout << "PASS: Four banks across groups sustained works correctly" << std::endl;
    return true;
}

/// Test mixed read/write across groups
bool test_across_groups_mixed_rw() {
    std::cout << "\n=== Test: Four Banks Across Groups (Mixed R/W) ===" << std::endl;
    std::cout << "Configuration: Banks 0,8 read; Banks 4,12 write" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // 4 rounds: R0, W4, R8, W12
    for (int round = 0; round < 4; ++round) {
        harness.submit_read(make_address(0, ROW, round * 64));
        harness.submit_write(make_address(4, ROW, round * 64));
        harness.submit_read(make_address(8, ROW, round * 64));
        harness.submit_write(make_address(12, ROW, round * 64));
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
    if (!harness.verify_stats(8, 8, 12, 4, 0)) return false;

    std::cout << "PASS: Four banks across groups mixed R/W works correctly" << std::endl;
    return true;
}

/// Test page conflicts across all 4 bank groups
bool test_across_groups_conflicts() {
    std::cout << "\n=== Test: Four Banks Across Groups (Page Conflicts) ===" << std::endl;
    std::cout << "Configuration: Different row per access per bank" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};

    // 3 rounds, each access to different row
    for (int round = 0; round < 3; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address(banks[b], 100 * (round + 1) + b, 0));
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
    if (!harness.verify_stats(12, 0, 0, 4, 8)) return false;

    std::cout << "PASS: Four banks across groups with conflicts works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    auto workload = make_four_banks_across_groups_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Four Banks Across Groups" << std::endl;
    std::cout << "Tests maximum parallelism with tRRD_S" << std::endl;
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
    pass &= test_across_groups_round_robin();
    pass &= test_timing_comparison();
    pass &= test_across_groups_sustained();
    pass &= test_across_groups_mixed_rw();
    pass &= test_across_groups_conflicts();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        const uint8_t banks[4] = {0, 4, 8, 12};
        for (int round = 0; round < 2; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address(banks[b], 100 + b, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("four-bank", "across_groups_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
