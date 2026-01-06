// patterns/memory/lpddr5/three-bank/mixed-groups/main.cpp
//
// Pattern: Three banks across different bank groups
// Tests: Multi-bank parallelism with tRRD_S timing
//
// Using banks from different groups enables:
// - tRRD_S (4 cycles) between all activates
// - Better bank-level parallelism
// - Reduced queuing delays
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

/// Test round-robin reads across three banks in different groups
/// Banks 0, 4, 8 are in groups 0, 1, 2 respectively
bool test_three_groups_round_robin() {
    std::cout << "\n=== Test: Three Banks Different Groups (Round-Robin) ===" << std::endl;
    std::cout << "Configuration: Banks 0 (BG0), 4 (BG1), 8 (BG2)" << std::endl;
    std::cout << "tRRD_S between all activates (maximum parallelism)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 4, 8};  // One per bank group
    const uint32_t rows[3] = {100, 200, 300};

    // Verify bank groups
    for (int i = 0; i < 3; ++i) {
        std::cout << "Bank " << (int)banks[i] << " is in group " << (int)bank_group(banks[i]) << std::endl;
    }

    // Round-robin: 0, 4, 8, 0, 4, 8, 0, 4, 8 (9 reads total, 3 per bank)
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

    std::cout << "PASS: Three banks different groups round-robin works correctly" << std::endl;
    return true;
}

/// Test sequential bursts to three banks in different groups
/// First burst to bank 0, then bank 4, then bank 8
bool test_three_groups_sequential() {
    std::cout << "\n=== Test: Three Banks Different Groups (Sequential) ===" << std::endl;
    std::cout << "Configuration: Sequential bursts to banks 0, 4, 8" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 4, 8};
    const uint32_t rows[3] = {100, 200, 300};

    // Sequential bursts: all to bank 0, then all to bank 4, then all to bank 8
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

    std::cout << "PASS: Three banks different groups sequential works correctly" << std::endl;
    return true;
}

/// Test page conflicts across three banks
/// Each bank accesses different rows causing conflicts
bool test_three_groups_conflicts() {
    std::cout << "\n=== Test: Three Banks Different Groups (Page Conflicts) ===" << std::endl;
    std::cout << "Configuration: Banks 0, 4, 8 with different rows each access" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[3] = {0, 4, 8};

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

    std::cout << "PASS: Three banks different groups with conflicts works correctly" << std::endl;
    return true;
}

/// Test mixed read/write across three banks in different groups
bool test_three_groups_mixed_rw() {
    std::cout << "\n=== Test: Three Banks Different Groups (Mixed R/W) ===" << std::endl;
    std::cout << "Configuration: Banks 0 (read), 4 (write), 8 (read)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // Pattern: R0, W4, R8, R0, W4, R8, ... (2 reads + 1 write per round)
    for (int round = 0; round < 3; ++round) {
        harness.submit_read(make_address(0, ROW, round * 64));
        harness.submit_write(make_address(4, ROW, round * 64));
        harness.submit_read(make_address(8, ROW, round * 64));
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

    std::cout << "PASS: Three banks different groups mixed R/W works correctly" << std::endl;
    return true;
}

/// Test asymmetric load across three banks
/// Heavy load on bank 0, medium on bank 4, light on bank 8
bool test_three_groups_asymmetric() {
    std::cout << "\n=== Test: Three Banks Asymmetric Load ===" << std::endl;
    std::cout << "Configuration: Bank 0 (6 reads), Bank 4 (3 reads), Bank 8 (1 read)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint32_t ROW = 100;

    // Heavy load on bank 0
    for (int i = 0; i < 6; ++i) {
        harness.submit_read(make_address(0, ROW, i * 64));
    }

    // Medium load on bank 4
    for (int i = 0; i < 3; ++i) {
        harness.submit_read(make_address(4, ROW, i * 64));
    }

    // Light load on bank 8
    harness.submit_read(make_address(8, ROW, 0));

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 10 reads total
    // Bank 0: 1 page empty + 5 page hits
    // Bank 4: 1 page empty + 2 page hits
    // Bank 8: 1 page empty + 0 page hits
    // Total: 10 reads, 0 writes, 7 page hits, 3 page empty, 0 conflicts
    if (!harness.verify_stats(10, 0, 7, 3, 0)) return false;

    std::cout << "PASS: Three banks asymmetric load works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());

    auto workload = make_three_banks_mixed_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Three Banks Mixed Groups" << std::endl;
    std::cout << "Tests multi-bank parallelism with tRRD_S" << std::endl;
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
    pass &= test_three_groups_round_robin();
    pass &= test_three_groups_sequential();
    pass &= test_three_groups_conflicts();
    pass &= test_three_groups_mixed_rw();
    pass &= test_three_groups_asymmetric();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());
        for (int round = 0; round < 3; ++round) {
            harness.submit_read(make_address(0, 100, round * 64));
            harness.submit_read(make_address(4, 200, round * 64));
            harness.submit_read(make_address(8, 300, round * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("three-bank", "mixed_groups_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n==============================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "==============================================" << std::endl;

    return pass ? 0 : 1;
}
