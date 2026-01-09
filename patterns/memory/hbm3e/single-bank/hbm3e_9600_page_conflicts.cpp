// patterns/memory/hbm3e/single-bank/hbm3e_9600_page_conflicts.cpp
//
// HBM3E-9600 single bank page conflict pattern - alternating rows
// Tests worst-case latency scenario
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm3e_harness.hpp"

using namespace sw::kpu::patterns::hbm3e;

int main() {
    std::cout << "=== HBM3E-9600 Single Bank Page Conflicts Pattern ===" << std::endl;
    std::cout << "Testing alternating row accesses (worst-case page conflict scenario)" << std::endl;

    // Use single pseudo-channel HBM3E-9600 config
    auto config = single_pc_config_9600();
    HBM3EHarness harness(config, HBM3EVariant::HBM3E_9600);

    std::cout << "Configuration: " << harness.variant_name() << " @ "
              << harness.clock_ghz() << " GHz" << std::endl;

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW_A = 100;
    constexpr uint32_t ROW_B = 200;
    constexpr int NUM_PAIRS = 4;

    // Submit reads alternating between two rows (page conflicts)
    std::cout << "\nSubmitting " << (NUM_PAIRS * 2) << " reads alternating between rows "
              << ROW_A << " and " << ROW_B << std::endl;

    for (int i = 0; i < NUM_PAIRS; ++i) {
        // Access row A
        uint64_t addr_a = make_address_single_pc(BANK, ROW_A, i);
        auto id_a = harness.submit_read(addr_a);
        if (!id_a) {
            std::cerr << "FAIL: Could not submit read to row A" << std::endl;
            return 1;
        }

        // Access row B (causes page conflict)
        uint64_t addr_b = make_address_single_pc(BANK, ROW_B, i);
        auto id_b = harness.submit_read(addr_b);
        if (!id_b) {
            std::cerr << "FAIL: Could not submit read to row B" << std::endl;
            return 1;
        }
    }

    // Run simulation
    if (!harness.run_until_complete(10000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    // Verify results
    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    if (stats.reads != NUM_PAIRS * 2) {
        std::cerr << "FAIL: Expected " << (NUM_PAIRS * 2) << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // First access is page_empty, then alternating conflicts
    // Pattern: empty, conflict, conflict, conflict, ...
    if (stats.page_empty != 1) {
        std::cerr << "FAIL: Expected 1 page empty, got " << stats.page_empty << std::endl;
        return 1;
    }

    if (stats.page_conflicts != (NUM_PAIRS * 2 - 1)) {
        std::cerr << "FAIL: Expected " << (NUM_PAIRS * 2 - 1) << " page conflicts, got "
                  << stats.page_conflicts << std::endl;
        return 1;
    }

    // Export trace
    std::string trace_path = make_trace_path("single-bank", "hbm3e_9600_page_conflicts_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM3E-9600 page conflicts pattern completed successfully" << std::endl;
    return 0;
}
