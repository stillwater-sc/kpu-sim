// patterns/memory/gddr6/four-bank/page_hit_burst.cpp
//
// GDDR6 Four Banks Page Hit Burst Pattern
// Tests sustained page hits across multiple banks for peak throughput
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

    std::cout << "=== GDDR6 Four Banks Page Hit Burst Pattern ===" << std::endl;
    std::cout << "Sustained page hits across banks 0, 4, 8, 12 (one per group)" << std::endl;
    std::cout << "Tests peak throughput with maximum page hit rate" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint32_t ROW = 100;
    const uint32_t NUM_PER_BANK = 4;

    // Banks 0, 4, 8, 12 are in Bank Groups 0, 1, 2, 3 respectively
    const uint8_t BANKS[] = {0, 4, 8, 12};
    const size_t NUM_BANKS = 4;

    std::cout << "\nSubmitting " << (NUM_PER_BANK * NUM_BANKS) << " reads across "
              << NUM_BANKS << " banks for sustained page hits" << std::endl;

    // First round: open all banks to same row (page empty)
    for (size_t b = 0; b < NUM_BANKS; ++b) {
        harness.submit_read(make_address(CHANNEL, BANKS[b], ROW, 0));
    }

    // Subsequent rounds: burst page hits across all banks
    for (uint32_t i = 1; i < NUM_PER_BANK; ++i) {
        for (size_t b = 0; b < NUM_BANKS; ++b) {
            harness.submit_read(make_address(CHANNEL, BANKS[b], ROW, i));
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

    // Expected: 4 page empty (initial opens), rest are page hits
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

    // Calculate page hit rate
    size_t total_accesses = NUM_PER_BANK * NUM_BANKS;
    double page_hit_rate = 100.0 * static_cast<double>(stats.page_hits) / static_cast<double>(total_accesses);

    std::cout << "\nThroughput Analysis:" << std::endl;
    std::cout << "  Total accesses: " << total_accesses << std::endl;
    std::cout << "  Page hit rate: " << std::fixed << std::setprecision(1)
              << page_hit_rate << "%" << std::endl;
    std::cout << "  All banks in different groups: maximum parallelism" << std::endl;
    std::cout << "  This represents near-peak memory bandwidth utilization" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("four-bank", "page_hit_burst_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_page_hit_burst_workload(CHANNEL, ROW, NUM_PER_BANK);

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
