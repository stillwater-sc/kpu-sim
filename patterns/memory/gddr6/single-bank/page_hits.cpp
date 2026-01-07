// patterns/memory/gddr6/single-bank/page_hits.cpp
//
// GDDR6 Page Hit Pattern Test
// Sequential reads to the same row - demonstrates best-case latency
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
    // Parse command line arguments
    bool run_fidelity = false;
    bool export_trace = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }
    std::cout << "=== GDDR6 Page Hit Pattern ===" << std::endl;
    std::cout << "Sequential reads to the same row in a single bank" << std::endl;

    // Use GDDR6-16000 configuration (16 Gbps)
    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    // Generate addresses all hitting the same row in bank 0
    // Row 0, different columns
    const uint32_t NUM_READS = 8;
    const uint32_t ROW = 0;
    const uint8_t BANK = 0;
    const uint8_t CHANNEL = 0;

    std::cout << "\nSubmitting " << NUM_READS << " reads to row " << ROW
              << " in bank " << (int)BANK << std::endl;

    for (uint32_t col = 0; col < NUM_READS; ++col) {
        uint64_t addr = make_address(CHANNEL, BANK, ROW, col);
        auto id = harness.submit_read(addr);
        if (!id) {
            std::cerr << "ERROR: Failed to submit read " << col << std::endl;
            return 1;
        }
    }

    // Run until all requests complete
    bool success = harness.run_until_complete(10000);
    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    // Print statistics
    harness.print_stats();
    harness.print_calibration_data();

    // Verify invariants
    if (!harness.verify_no_violations()) {
        return 1;
    }

    // Verify expected outcomes
    const auto& stats = harness.stats();

    // First read opens the page (page_empty), subsequent reads are page hits
    uint64_t expected_page_empty = 1;
    uint64_t expected_page_hits = NUM_READS - 1;

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
    if (stats.page_conflicts != 0) {
        std::cerr << "FAIL: unexpected page conflicts: " << stats.page_conflicts << std::endl;
        pass = false;
    }

    // Verify latencies are reasonable
    // With conservative FCFS scheduling, each command waits for bus to be IDLE
    // This means latencies are serialized rather than pipelined
    // The key validation is that page hit/empty/conflict counts are correct
    double avg_latency = stats.avg_latency();
    std::cout << "Average latency: " << avg_latency << " cycles" << std::endl;
    std::cout << "Page hit latency: " << stats.avg_page_hit_latency() << " cycles" << std::endl;
    std::cout << "Page empty latency: " << stats.avg_page_empty_latency() << " cycles" << std::endl;

    // Basic sanity check - latency should be positive and bounded
    if (avg_latency < 20) {
        std::cerr << "FAIL: Average latency " << avg_latency
                  << " cycles is too low (below theoretical minimum)" << std::endl;
        pass = false;
    }

    // Export trace for visualization
    if (export_trace) {
        std::string trace_file = make_trace_path("single-bank", "page_hits_trace.json");
        harness.export_trace(trace_file, 2.0);  // 2.0 GHz for GDDR6-16000
    }

    // Run multi-fidelity comparison if requested
    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);

        // Create page hit workload matching our test
        auto workload = make_page_hit_workload(CHANNEL, BANK, ROW, NUM_READS);

        std::cout << "\n--- Uncalibrated ---" << std::endl;
        mf_harness.run_comparison(workload, true);

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
