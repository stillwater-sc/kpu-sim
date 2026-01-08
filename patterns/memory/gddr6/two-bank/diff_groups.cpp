// patterns/memory/gddr6/two-bank/diff_groups.cpp
//
// GDDR6 Two Banks Different Groups Pattern
// Tests tRRD_S and tCCD_S constraints (different bank group timing)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include "../common/gddr6_harness.hpp"
#include "../common/multi_fidelity.hpp"
#include "../common/workloads.hpp"

using namespace sw::kpu::patterns::gddr6;

int main(int argc, char* argv[]) {
    bool run_fidelity = false;
    bool export_trace = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    std::cout << "=== GDDR6 Two Banks Different Groups Pattern ===" << std::endl;
    std::cout << "Alternating accesses to banks 0 and 4 (different bank groups)" << std::endl;
    std::cout << "Tests tRRD_S (ACT-to-ACT) and tCCD_S (CAS-to-CAS) - faster than same group" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint8_t BANK_A = 0;  // Bank Group 0
    const uint8_t BANK_B = 4;  // Bank Group 1 (different group)
    const uint32_t ROW = 100;
    const uint32_t NUM_PER_BANK = 4;

    std::cout << "\nSubmitting " << (NUM_PER_BANK * 2) << " reads alternating between banks "
              << (int)BANK_A << " and " << (int)BANK_B << " (different bank groups)" << std::endl;

    for (uint32_t i = 0; i < NUM_PER_BANK; ++i) {
        harness.submit_read(make_address(CHANNEL, BANK_A, ROW, i));
        harness.submit_read(make_address(CHANNEL, BANK_B, ROW, i));
    }

    bool success = harness.run_until_complete(20000);
    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();

    // Expected: 2 page empty (one per bank), rest are page hits
    uint64_t expected_page_empty = 2;
    uint64_t expected_page_hits = NUM_PER_BANK * 2 - 2;

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Expected page_empty: " << expected_page_empty
              << ", Actual: " << stats.page_empty << std::endl;
    std::cout << "Expected page_hits: " << expected_page_hits
              << ", Actual: " << stats.page_hits << std::endl;

    bool pass = true;
    if (stats.page_empty != expected_page_empty) {
        std::cerr << "FAIL: page_empty mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_hits != expected_page_hits) {
        std::cerr << "FAIL: page_hits mismatch" << std::endl;
        pass = false;
    }

    // Verify different bank group constraint was exercised
    std::cout << "\nBank Group Analysis:" << std::endl;
    std::cout << "  Bank " << (int)BANK_A << " group: " << (int)bank_group(BANK_A) << std::endl;
    std::cout << "  Bank " << (int)BANK_B << " group: " << (int)bank_group(BANK_B) << std::endl;
    std::cout << "  Same group: " << (same_bank_group(BANK_A, BANK_B) ? "yes" : "no") << std::endl;

    if (!same_bank_group(BANK_A, BANK_B)) {
        std::cout << "  Benefit: tCCD_S (" << tCCD_S << ") < tCCD_L (" << tCCD_L << ")" << std::endl;
    }

    if (export_trace) {
        std::string trace_file = make_trace_path("two-bank", "diff_groups_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_two_banks_diff_groups_workload(CHANNEL, ROW, NUM_PER_BANK);

        std::cout << "\n--- Calibrated ---" << std::endl;
        mf_harness.run_calibrated(workload, true);
    }

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
