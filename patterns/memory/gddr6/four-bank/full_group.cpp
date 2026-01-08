// patterns/memory/gddr6/four-bank/full_group.cpp
//
// GDDR6 Four Banks Full Group Pattern
// Tests tFAW (Four Activate Window) constraint
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

    std::cout << "=== GDDR6 Four Banks Full Group Pattern ===" << std::endl;
    std::cout << "Accesses to banks 0, 1, 2, 3 (all in Bank Group 0)" << std::endl;
    std::cout << "Tests tFAW (Four Activate Window) constraint" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint32_t ROW = 100;
    const uint32_t NUM_ROUNDS = 2;

    // Banks 0, 1, 2, 3 are all in Bank Group 0
    const uint8_t BANKS[] = {0, 1, 2, 3};
    const size_t NUM_BANKS = 4;

    std::cout << "\nSubmitting " << (NUM_ROUNDS * NUM_BANKS) << " reads across "
              << NUM_BANKS << " banks in same group" << std::endl;

    for (uint32_t r = 0; r < NUM_ROUNDS; ++r) {
        for (size_t b = 0; b < NUM_BANKS; ++b) {
            harness.submit_read(make_address(CHANNEL, BANKS[b], ROW, r));
        }
    }

    bool success = harness.run_until_complete(30000);
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

    // Expected: 4 page empty (one per bank), rest are page hits
    uint64_t expected_page_empty = NUM_BANKS;
    uint64_t expected_page_hits = NUM_ROUNDS * NUM_BANKS - NUM_BANKS;

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
    std::cout << "  All same group: yes (constrained by tFAW=" << tFAW << ")" << std::endl;
    std::cout << "  tFAW limits rate of ACTIVATE commands within window" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("four-bank", "full_group_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_four_banks_full_group_workload(CHANNEL, ROW, NUM_ROUNDS);

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
