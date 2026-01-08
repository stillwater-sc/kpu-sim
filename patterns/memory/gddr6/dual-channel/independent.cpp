// patterns/memory/gddr6/dual-channel/independent.cpp
//
// GDDR6 Dual Channel Independent Pattern
// Tests independent parallel access to both 16-bit channels
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

    std::cout << "=== GDDR6 Dual Channel Independent Pattern ===" << std::endl;
    std::cout << "Independent accesses to both 16-bit channels" << std::endl;
    std::cout << "Tests dual-channel parallelism (GDDR6 mandatory feature)" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;
    const uint32_t NUM_PER_CHANNEL = 4;

    std::cout << "\nSubmitting " << (NUM_PER_CHANNEL * 2) << " reads across "
              << "2 channels independently" << std::endl;

    // Alternate between channels - each channel operates independently
    for (uint32_t i = 0; i < NUM_PER_CHANNEL; ++i) {
        harness.submit_read(make_address(0, BANK, ROW, i));  // Channel 0
        harness.submit_read(make_address(1, BANK, ROW, i));  // Channel 1
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

    // Expected: 2 page empty (one per channel/bank), rest are page hits
    uint64_t expected_page_empty = 2;  // One per channel
    uint64_t expected_page_hits = NUM_PER_CHANNEL * 2 - 2;

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

    std::cout << "\nDual Channel Analysis:" << std::endl;
    std::cout << "  Channel 0: 16-bit independent channel" << std::endl;
    std::cout << "  Channel 1: 16-bit independent channel" << std::endl;
    std::cout << "  Combined: 32-bit data bus (2 x 16-bit)" << std::endl;
    std::cout << "  Each channel has independent timing constraints" << std::endl;
    std::cout << "  Parallel operation doubles effective bandwidth" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("dual-channel", "independent_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_dual_channel_independent_workload(ROW, NUM_PER_CHANNEL);

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
