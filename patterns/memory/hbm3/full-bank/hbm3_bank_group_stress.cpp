// patterns/memory/hbm3/full-bank/hbm3_bank_group_stress.cpp
//
// HBM3 Bank Group Stress Test
// Exercises bank group timing constraints:
// - tRRD_L: ACT to ACT within same bank group (4 cycles)
// - tRRD_S: ACT to ACT across different bank groups (2 cycles)
// - tFAW: Four Activate Window (16 cycles)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/hbm3_harness.hpp"

using namespace sw::kpu::patterns::hbm3;

/// Test rapid activates within same bank group (tRRD_L limited)
bool test_same_bank_group_stress(HBM3Harness& harness) {
    std::cout << "\n=== Test 1: Same Bank Group Stress (tRRD_L) ===" << std::endl;
    std::cout << "Rapidly activating banks 0,1,2,3 (all in Bank Group 0)" << std::endl;
    std::cout << "tRRD_L = " << tRRD_L << " cycles between ACTs" << std::endl;

    // Submit reads to banks 0-3 (all in BG0), different rows to force activates
    for (int round = 0; round < 4; ++round) {
        for (uint8_t bank = 0; bank < 4; ++bank) {
            uint64_t addr = make_address_single_pc(bank, 100 + round, 0);
            while (!harness.submit_read(addr)) {
                harness.run_cycles(1);
            }
        }
    }

    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: Same bank group stress did not complete" << std::endl;
        return false;
    }

    const auto& stats = harness.stats();
    std::cout << "  Reads completed: " << stats.reads << std::endl;
    std::cout << "  Page empty (activates): " << stats.page_empty << std::endl;
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(2)
              << stats.avg_latency() << " cycles" << std::endl;

    return harness.verify_no_violations();
}

/// Test rapid activates across different bank groups (tRRD_S limited)
bool test_diff_bank_group_stress(HBM3Harness& harness) {
    std::cout << "\n=== Test 2: Different Bank Group Stress (tRRD_S) ===" << std::endl;
    std::cout << "Rapidly activating banks 0,4,8,12 (one per Bank Group)" << std::endl;
    std::cout << "tRRD_S = " << tRRD_S << " cycles between ACTs" << std::endl;

    // Submit reads to banks 0,4,8,12 (one per BG), different rows
    uint8_t banks[] = {0, 4, 8, 12};
    for (int round = 0; round < 4; ++round) {
        for (uint8_t bank : banks) {
            uint64_t addr = make_address_single_pc(bank, 200 + round, 0);
            while (!harness.submit_read(addr)) {
                harness.run_cycles(1);
            }
        }
    }

    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: Different bank group stress did not complete" << std::endl;
        return false;
    }

    const auto& stats = harness.stats();
    std::cout << "  Reads completed: " << stats.reads << std::endl;
    std::cout << "  Page empty (activates): " << stats.page_empty << std::endl;
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(2)
              << stats.avg_latency() << " cycles" << std::endl;

    return harness.verify_no_violations();
}

/// Test tFAW constraint (Four Activate Window)
bool test_tfaw_constraint(HBM3Harness& harness) {
    std::cout << "\n=== Test 3: tFAW Stress Test ===" << std::endl;
    std::cout << "Submitting 16 activates rapidly to trigger tFAW" << std::endl;
    std::cout << "tFAW = " << tFAW << " cycles (max 4 ACTs in this window)" << std::endl;

    // Submit reads to all 16 banks, each to a different row (forces activate)
    for (uint8_t bank = 0; bank < 16; ++bank) {
        uint64_t addr = make_address_single_pc(bank, 300 + bank, 0);
        while (!harness.submit_read(addr)) {
            harness.run_cycles(1);
        }
    }

    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: tFAW stress did not complete" << std::endl;
        return false;
    }

    const auto& stats = harness.stats();
    std::cout << "  Reads completed: " << stats.reads << std::endl;
    std::cout << "  Page empty (activates): " << stats.page_empty << std::endl;

    // With tFAW=16 and 16 activates, minimum time is ~64 cycles (4 windows)
    uint64_t cycles = harness.current_cycle();
    std::cout << "  Total cycles: " << cycles << std::endl;

    // tFAW should limit throughput
    std::cout << "  Expected min activate time: ~64 cycles (tFAW limited)" << std::endl;

    return harness.verify_no_violations();
}

/// Test mixed bank group access pattern
bool test_mixed_bank_groups() {
    std::cout << "\n=== Test 4: Mixed Bank Group Pattern ===" << std::endl;
    std::cout << "Comparing same-group vs cross-group performance" << std::endl;

    // Test A: All accesses within Bank Group 0
    {
        auto config = single_pc_config();
        config.queue_depth = 128;
        HBM3Harness harness_bg0(config);

        std::cout << "\n  Sub-test A: Same Bank Group (BG0: banks 0-3)" << std::endl;

        for (int round = 0; round < 8; ++round) {
            for (uint8_t bank = 0; bank < 4; ++bank) {
                uint64_t addr = make_address_single_pc(bank, 400 + round, 0);
                while (!harness_bg0.submit_read(addr)) {
                    harness_bg0.run_cycles(1);
                }
            }
        }

        harness_bg0.run_until_complete(10000);
        const auto& stats_a = harness_bg0.stats();
        uint64_t cycles_a = harness_bg0.current_cycle();
        std::cout << "    Reads: " << stats_a.reads << ", Cycles: " << cycles_a << std::endl;
        std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
                  << (double)stats_a.reads / static_cast<double>(cycles_a) * 1000.0 << " reads/1000 cycles" << std::endl;

        // Export trace
        std::string trace_path = make_trace_path("full-bank", "hbm3_bank_group_same_trace.json");
        harness_bg0.export_trace(trace_path);
    }

    // Test B: Accesses across all 4 bank groups
    {
        auto config = single_pc_config();
        config.queue_depth = 128;
        HBM3Harness harness_all(config);

        std::cout << "\n  Sub-test B: Cross Bank Groups (banks 0,4,8,12)" << std::endl;

        uint8_t banks[] = {0, 4, 8, 12};
        for (int round = 0; round < 8; ++round) {
            for (uint8_t bank : banks) {
                uint64_t addr = make_address_single_pc(bank, 500 + round, 0);
                while (!harness_all.submit_read(addr)) {
                    harness_all.run_cycles(1);
                }
            }
        }

        harness_all.run_until_complete(10000);
        const auto& stats_b = harness_all.stats();
        uint64_t cycles_b = harness_all.current_cycle();
        std::cout << "    Reads: " << stats_b.reads << ", Cycles: " << cycles_b << std::endl;
        std::cout << "    Throughput: " << std::fixed << std::setprecision(2)
                  << (double)stats_b.reads / static_cast<double>(cycles_b) * 1000.0 << " reads/1000 cycles" << std::endl;

        // Export trace
        std::string trace_path = make_trace_path("full-bank", "hbm3_bank_group_cross_trace.json");
        harness_all.export_trace(trace_path);
    }

    std::cout << "\n  Note: Cross-group should be faster (tRRD_S < tRRD_L)" << std::endl;
    return true;
}

int main() {
    std::cout << "=== HBM3 Bank Group Stress Test ===" << std::endl;
    std::cout << "\nHBM3-5600 Bank Group Timing:" << std::endl;
    std::cout << "  tRRD_L (same group):  " << tRRD_L << " cycles" << std::endl;
    std::cout << "  tRRD_S (diff group):  " << tRRD_S << " cycles" << std::endl;
    std::cout << "  tCCD_L (same group):  " << tCCD_L << " cycles" << std::endl;
    std::cout << "  tCCD_S (diff group):  " << tCCD_S << " cycles" << std::endl;
    std::cout << "  tFAW (4-ACT window):  " << tFAW << " cycles" << std::endl;

    bool all_pass = true;

    // Test 1: Same bank group stress
    {
        auto config = single_pc_config();
        config.queue_depth = 128;
        HBM3Harness harness(config);
        if (!test_same_bank_group_stress(harness)) {
            all_pass = false;
        }
    }

    // Test 2: Different bank group stress
    {
        auto config = single_pc_config();
        config.queue_depth = 128;
        HBM3Harness harness(config);
        if (!test_diff_bank_group_stress(harness)) {
            all_pass = false;
        }
    }

    // Test 3: tFAW constraint
    {
        auto config = single_pc_config();
        config.queue_depth = 128;
        HBM3Harness harness(config);
        if (!test_tfaw_constraint(harness)) {
            all_pass = false;
        }
        // Export comprehensive trace
        std::string trace_path = make_trace_path("full-bank", "hbm3_bank_group_stress_trace.json");
        harness.export_trace(trace_path);
    }

    // Test 4: Mixed bank groups comparison
    if (!test_mixed_bank_groups()) {
        all_pass = false;
    }

    if (all_pass) {
        std::cout << "\nPASS: All bank group stress tests completed successfully" << std::endl;
        return 0;
    } else {
        std::cout << "\nFAIL: Some bank group stress tests had errors" << std::endl;
        return 1;
    }
}
