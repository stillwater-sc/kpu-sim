// patterns/memory/lpddr5/three-bank/same-group/main.cpp
//
// Pattern: Three banks in the same bank group
// Tests: Bank group limitations with tRRD_L timing
//
// Using banks from the same group demonstrates:
// - tRRD_L (6 cycles) constraint between all activates
// - Reduced parallelism compared to different groups
// - Bank group as a bottleneck
//
// Bank Group 0: Banks 0, 1, 2, 3
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

/// Test round-robin reads across three banks in the same group
/// Banks 0, 1, 2 are all in bank group 0
bool test_same_group_round_robin() {
    std::cout << "\n=== Test: Three Banks Same Group (Round-Robin) ===" << std::endl;
    std::cout << "Configuration: Banks 0, 1, 2 (all in group 0)" << std::endl;
    std::cout << "tRRD_L (6 cycles) between all activates" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 1, 2};  // All in bank group 0
    const uint32_t rows[3] = {100, 200, 300};

    // Verify all banks are in the same group
    std::cout << "All banks in group " << (int)bank_group(banks[0]) << std::endl;
    for (int i = 1; i < 3; ++i) {
        if (!same_bank_group(banks[0], banks[i])) {
            std::cerr << "ERROR: Banks not in same group!" << std::endl;
            return false;
        }
    }

    // Round-robin: 0, 1, 2, 0, 1, 2, 0, 1, 2 (9 reads total, 3 per bank)
    for (int round = 0; round < 3; ++round) {
        for (int b = 0; b < 3; ++b) {
            harness.submit_read(make_address(banks[b], rows[b], round * 64));
        }
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 9 reads total
    // Each bank: 1 page empty + 2 page hits
    // Total: 9 reads, 0 writes, 6 page hits, 3 page empty, 0 conflicts
    if (!harness.verify_stats(9, 0, 6, 3, 0)) return false;

    std::cout << "PASS: Three banks same group round-robin works correctly" << std::endl;
    return true;
}

/// Test sequential bursts to three banks in the same group
bool test_same_group_sequential() {
    std::cout << "\n=== Test: Three Banks Same Group (Sequential) ===" << std::endl;
    std::cout << "Configuration: Sequential bursts to banks 0, 1, 2" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 1, 2};
    const uint32_t rows[3] = {100, 200, 300};

    // Sequential bursts
    for (int b = 0; b < 3; ++b) {
        for (int i = 0; i < 4; ++i) {
            harness.submit_read(make_address(banks[b], rows[b], i * 64));
        }
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 12 reads (3 banks × 4 accesses)
    // Each bank: 1 page empty + 3 page hits
    // Total: 12 reads, 0 writes, 9 page hits, 3 page empty, 0 conflicts
    if (!harness.verify_stats(12, 0, 9, 3, 0)) return false;

    std::cout << "PASS: Three banks same group sequential works correctly" << std::endl;
    return true;
}

/// Test page conflicts across three banks in same group
bool test_same_group_conflicts() {
    std::cout << "\n=== Test: Three Banks Same Group (Page Conflicts) ===" << std::endl;
    std::cout << "Configuration: Banks 0, 1, 2 with different rows each access" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 1, 2};

    // Each access to a bank goes to a different row
    for (int round = 0; round < 3; ++round) {
        for (int b = 0; b < 3; ++b) {
            harness.submit_read(make_address(banks[b], 100 * (round + 1) + b, 0));
        }
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 9 reads total
    // Each bank: 1 page empty + 2 page conflicts
    // Total: 9 reads, 0 writes, 0 page hits, 3 page empty, 6 page conflicts
    if (!harness.verify_stats(9, 0, 0, 3, 6)) return false;

    std::cout << "PASS: Three banks same group with conflicts works correctly" << std::endl;
    return true;
}

/// Compare same-group vs mixed-groups timing
/// This demonstrates the performance impact of bank group selection
bool test_timing_comparison() {
    std::cout << "\n=== Test: Same Group vs Mixed Groups Timing ===" << std::endl;

    // Same group test (Banks 0, 1, 2)
    uint64_t same_group_cycles;
    {
        std::cout << "\n--- Same Bank Group (Banks 0, 1, 2) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int round = 0; round < 3; ++round) {
            harness.submit_read(make_address(0, 100, round * 64));
            harness.submit_read(make_address(1, 200, round * 64));
            harness.submit_read(make_address(2, 300, round * 64));
        }

        harness.run_until_complete();
        same_group_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << same_group_cycles << std::endl;
        harness.print_stats();
    }

    // Mixed groups test (Banks 0, 4, 8)
    uint64_t mixed_group_cycles;
    {
        std::cout << "\n--- Mixed Bank Groups (Banks 0, 4, 8) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int round = 0; round < 3; ++round) {
            harness.submit_read(make_address(0, 100, round * 64));
            harness.submit_read(make_address(4, 200, round * 64));
            harness.submit_read(make_address(8, 300, round * 64));
        }

        harness.run_until_complete();
        mixed_group_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << mixed_group_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Same group (tRRD_L):  " << same_group_cycles << " cycles" << std::endl;
    std::cout << "Mixed groups (tRRD_S): " << mixed_group_cycles << " cycles" << std::endl;
    std::cout << "tRRD_L = " << tRRD_L << " cycles, tRRD_S = " << tRRD_S << " cycles" << std::endl;

    if (mixed_group_cycles < same_group_cycles) {
        std::cout << "Mixed groups is " << (same_group_cycles - mixed_group_cycles)
                  << " cycles faster ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (same_group_cycles - mixed_group_cycles) / same_group_cycles)
                  << "% improvement)" << std::endl;
    } else if (mixed_group_cycles == same_group_cycles) {
        std::cout << "No timing difference observed (may be dominated by other constraints)" << std::endl;
    }

    return true;
}

/// Test mixed read/write across three banks in same group
bool test_same_group_mixed_rw() {
    std::cout << "\n=== Test: Three Banks Same Group (Mixed R/W) ===" << std::endl;
    std::cout << "Configuration: Banks 0 (read), 1 (write), 2 (read)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // Pattern: R0, W1, R2, R0, W1, R2, ...
    for (int round = 0; round < 3; ++round) {
        harness.submit_read(make_address(0, ROW, round * 64));
        harness.submit_write(make_address(1, ROW, round * 64));
        harness.submit_read(make_address(2, ROW, round * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 6 reads + 3 writes = 9 total
    // Each bank: 1 page empty + 2 page hits
    // Total: 6 reads, 3 writes, 6 page hits, 3 page empty, 0 conflicts
    if (!harness.verify_stats(6, 3, 6, 3, 0)) return false;

    std::cout << "PASS: Three banks same group mixed R/W works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    auto workload = make_three_banks_same_group_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Three Banks Same Group" << std::endl;
    std::cout << "Tests bank group limitations with tRRD_L" << std::endl;
    std::cout << "==============================================" << std::endl;

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
    pass &= test_same_group_round_robin();
    pass &= test_same_group_sequential();
    pass &= test_same_group_conflicts();
    pass &= test_timing_comparison();
    pass &= test_same_group_mixed_rw();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        for (int round = 0; round < 3; ++round) {
            harness.submit_read(make_address(0, 100, round * 64));
            harness.submit_read(make_address(1, 200, round * 64));
            harness.submit_read(make_address(2, 300, round * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("three-bank", "same_group_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "==============================================" << std::endl;

    return pass ? 0 : 1;
}
