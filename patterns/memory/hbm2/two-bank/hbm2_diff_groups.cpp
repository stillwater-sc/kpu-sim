// patterns/memory/hbm2/two-bank/hbm2_diff_groups.cpp
//
// HBM2 two-bank different bank groups pattern
// Tests tRRD_S and tCCD_S timing constraints
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

int main() {
    std::cout << "=== HBM2 Two Bank Different Groups Pattern ===" << std::endl;
    std::cout << "Testing accesses to two banks in different bank groups" << std::endl;
    std::cout << "Expects tRRD_S=" << tRRD_S << ", tCCD_S=" << tCCD_S << " constraints" << std::endl;

    auto config = single_pc_config();
    HBM2Harness harness(config);

    // Banks 0 and 4 are in different bank groups (BG0 and BG1)
    constexpr uint8_t BANK_A = 0;   // BG0
    constexpr uint8_t BANK_B = 4;   // BG1
    constexpr uint32_t ROW = 50;
    constexpr int ACCESSES_PER_BANK = 4;

    std::cout << "\nBank " << (int)BANK_A << " -> BG" << (int)bank_group(BANK_A) << std::endl;
    std::cout << "Bank " << (int)BANK_B << " -> BG" << (int)bank_group(BANK_B) << std::endl;
    std::cout << "Same bank group: " << (same_bank_group(BANK_A, BANK_B) ? "yes" : "no") << std::endl;

    // Interleave accesses between two banks in different groups
    for (int i = 0; i < ACCESSES_PER_BANK; ++i) {
        // Bank A
        uint64_t addr_a = make_address_single_pc(BANK_A, ROW, i);
        auto id_a = harness.submit_read(addr_a);
        if (!id_a) {
            std::cerr << "FAIL: Could not submit read to bank A" << std::endl;
            return 1;
        }

        // Bank B
        uint64_t addr_b = make_address_single_pc(BANK_B, ROW, i);
        auto id_b = harness.submit_read(addr_b);
        if (!id_b) {
            std::cerr << "FAIL: Could not submit read to bank B" << std::endl;
            return 1;
        }
    }

    // Run simulation
    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    if (stats.reads != ACCESSES_PER_BANK * 2) {
        std::cerr << "FAIL: Expected " << (ACCESSES_PER_BANK * 2) << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // Export trace
    std::string trace_path = make_trace_path("two-bank", "hbm2_diff_groups_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM2 different bank groups pattern completed successfully" << std::endl;
    return 0;
}
