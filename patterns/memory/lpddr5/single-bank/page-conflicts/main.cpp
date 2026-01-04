// patterns/memory/lpddr5/single-bank/page-conflicts/main.cpp
//
// Pattern: Reads to different rows (page conflicts)
// Tests: tRP + tRCD + tCL + tBurst timing, precharge behavior
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

bool test_page_conflicts() {
    std::cout << "\n=== Test: Single Bank Page Conflicts ===" << std::endl;
    std::cout << "Configuration: Bank 0, different rows, 8 reads" << std::endl;
    std::cout << "Expected: 1 page empty, 7 page conflicts" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;

    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK, i * 100, 0));  // Different rows
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;
    if (!harness.verify_stats(8, 0, 0, 1, 7)) return false;

    std::cout << "PASS: Page conflicts work correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());
    auto workload = make_page_conflict_workload(0, 8);

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Single Bank Page Conflicts" << std::endl;
    std::cout << "==========================================" << std::endl;

    bool run_fidelity = false;
    std::string trace_file = "page_conflicts_trace.json";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        }
    }

    bool pass = test_page_conflicts();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace
    std::cout << "\n=== Trace Export ===" << std::endl;
    LPDDR5Harness harness(single_channel_config());
    for (int i = 0; i < 4; ++i) {
        harness.submit_read(make_address(0, i * 100, 0));
    }
    harness.run_until_complete();
    harness.export_trace(trace_file);

    return pass ? 0 : 1;
}
