// patterns/memory/gddr6/three-bank/mixed_groups.cpp
//
// GDDR6 Three Banks Mixed Groups Pattern
// Tests cross-bank-group parallelism with tRRD_S/tCCD_S
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

    std::cout << "=== GDDR6 Three Banks Mixed Groups Pattern ===" << std::endl;
    std::cout << "Round-robin accesses to banks 0, 4, 8 (different bank groups)" << std::endl;
    std::cout << "Tests cross-group parallelism with tRRD_S/tCCD_S" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint32_t ROW = 100;
    const uint32_t NUM_PER_BANK = 3;

    // Banks 0, 4, 8 are in Bank Groups 0, 1, 2 respectively
    const uint8_t BANKS[] = {0, 4, 8};
    const size_t NUM_BANKS = 3;

    std::cout << "\nSubmitting " << (NUM_PER_BANK * NUM_BANKS) << " reads across "
              << NUM_BANKS << " banks in different groups" << std::endl;

    for (uint32_t i = 0; i < NUM_PER_BANK; ++i) {
        for (size_t b = 0; b < NUM_BANKS; ++b) {
            harness.submit_read(make_address(CHANNEL, BANKS[b], ROW, i));
        }
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

    // Expected: 3 page empty (one per bank), rest are page hits
    uint64_t expected_page_empty = NUM_BANKS;
    uint64_t expected_page_hits = NUM_PER_BANK * NUM_BANKS - NUM_BANKS;

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

    std::cout << "\nBank Group Analysis:" << std::endl;
    for (size_t b = 0; b < NUM_BANKS; ++b) {
        std::cout << "  Bank " << (int)BANKS[b] << " group: " << (int)bank_group(BANKS[b]) << std::endl;
    }
    std::cout << "  Benefit: tCCD_S (" << tCCD_S << ") < tCCD_L (" << tCCD_L << ")" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("three-bank", "mixed_groups_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_three_banks_mixed_workload(CHANNEL, ROW, NUM_PER_BANK);

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
