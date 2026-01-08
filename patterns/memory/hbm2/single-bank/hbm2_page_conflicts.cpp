// patterns/memory/hbm2/single-bank/hbm2_page_conflicts.cpp
//
// HBM2 single bank page conflict pattern - reads to different rows
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

int main() {
    std::cout << "=== HBM2 Single Bank Page Conflicts Pattern ===" << std::endl;
    std::cout << "Testing reads to different rows (page conflict scenario)" << std::endl;

    auto config = single_pc_config();
    HBM2Harness harness(config);

    constexpr uint8_t BANK = 0;
    constexpr int NUM_ACCESSES = 4;

    // Submit reads to different rows in same bank (page conflicts)
    std::cout << "\nSubmitting " << NUM_ACCESSES << " reads to different rows in bank "
              << (int)BANK << std::endl;

    for (int i = 0; i < NUM_ACCESSES; ++i) {
        uint32_t row = 100 + i * 10;  // Different row each time
        uint64_t addr = make_address_single_pc(BANK, row, 0);
        auto id = harness.submit_read(addr);
        if (!id) {
            std::cerr << "FAIL: Could not submit read " << i << std::endl;
            return 1;
        }
    }

    // Run simulation
    if (!harness.run_until_complete(5000)) {
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
    if (stats.reads != NUM_ACCESSES) {
        std::cerr << "FAIL: Expected " << NUM_ACCESSES << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // Page conflicts involve precharge (page_conflicts) + activate (page_empty)
    // So we get N page_empty (one for each row activated) and N-1 page_conflicts
    if (stats.page_empty != NUM_ACCESSES) {
        std::cerr << "FAIL: Expected " << NUM_ACCESSES << " page empty (activates), got " << stats.page_empty << std::endl;
        return 1;
    }

    if (stats.page_conflicts != NUM_ACCESSES - 1) {
        std::cerr << "FAIL: Expected " << (NUM_ACCESSES - 1) << " page conflicts (precharges), got " << stats.page_conflicts << std::endl;
        return 1;
    }

    // Export trace
    std::string trace_path = make_trace_path("single-bank", "hbm2_page_conflicts_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM2 page conflicts pattern completed successfully" << std::endl;
    return 0;
}
