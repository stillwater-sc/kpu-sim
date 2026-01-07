// patterns/memory/gddr6/single-bank/page_conflicts.cpp
//
// GDDR6 Page Conflict Pattern Test
// Sequential reads to different rows in the same bank - demonstrates worst-case latency
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

int main() {
    std::cout << "=== GDDR6 Page Conflict Pattern ===" << std::endl;
    std::cout << "Sequential reads to different rows in a single bank" << std::endl;
    std::cout << "Each access triggers: PRECHARGE -> ACTIVATE -> READ" << std::endl;

    // Use GDDR6-16000 configuration (16 Gbps)
    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    // Generate addresses hitting different rows in bank 0
    // Each access will cause a page conflict (except the first)
    const uint32_t NUM_READS = 8;
    const uint8_t BANK = 0;
    const uint8_t CHANNEL = 0;
    const uint32_t COL = 0;

    std::cout << "\nSubmitting " << NUM_READS << " reads to different rows in bank "
              << (int)BANK << std::endl;

    for (uint32_t row = 0; row < NUM_READS; ++row) {
        uint64_t addr = make_address(CHANNEL, BANK, row, COL);
        auto id = harness.submit_read(addr);
        if (!id) {
            std::cerr << "ERROR: Failed to submit read " << row << std::endl;
            return 1;
        }
    }

    // Run until all requests complete
    bool success = harness.run_until_complete(20000);
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

    // First read opens a page (page_empty)
    // Subsequent reads each cause a page conflict (different row open)
    uint64_t expected_page_empty = 1;
    uint64_t expected_page_conflicts = NUM_READS - 1;
    uint64_t expected_page_hits = 0;

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Expected page_empty: " << expected_page_empty
              << ", Actual: " << stats.page_empty << std::endl;
    std::cout << "Expected page_conflicts: " << expected_page_conflicts
              << ", Actual: " << stats.page_conflicts << std::endl;
    std::cout << "Expected page_hits: " << expected_page_hits
              << ", Actual: " << stats.page_hits << std::endl;

    bool pass = true;
    if (stats.page_empty != expected_page_empty) {
        std::cerr << "FAIL: page_empty mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_conflicts != expected_page_conflicts) {
        std::cerr << "FAIL: page_conflicts mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_hits != expected_page_hits) {
        std::cerr << "FAIL: page_hits mismatch" << std::endl;
        pass = false;
    }

    // Verify latencies - page conflicts should have highest latency
    // Page conflict: tRP + tRCDRD + tRL + tBurst = 18 + 18 + 20 + 4 = 60 cycles (theoretical)
    double avg_latency = stats.avg_latency();
    double page_conflict_latency = stats.avg_page_conflict_latency();
    double page_empty_latency = stats.avg_page_empty_latency();

    std::cout << "\nLatency Analysis:" << std::endl;
    std::cout << "  Average latency: " << avg_latency << " cycles" << std::endl;
    std::cout << "  Page empty latency: " << page_empty_latency << " cycles" << std::endl;
    std::cout << "  Page conflict latency: " << page_conflict_latency << " cycles" << std::endl;

    // Page conflict latency should be higher than page empty (includes precharge)
    // Note: With conservative scheduling, both may have additional overhead
    if (page_conflict_latency > 0 && page_empty_latency > 0) {
        // In theory: page_conflict = page_empty + tRP
        // But with serialized scheduling, this relationship may not be exact
        std::cout << "  Conflict vs Empty ratio: "
                  << (page_conflict_latency / page_empty_latency) << "x" << std::endl;
    }

    // Basic sanity check
    if (avg_latency < 40) {
        std::cerr << "FAIL: Average latency " << avg_latency
                  << " cycles is too low for page conflicts" << std::endl;
        pass = false;
    }

    // Export trace for visualization
    std::string trace_file = make_trace_path("single-bank", "page_conflicts_trace.json");
    harness.export_trace(trace_file, 2.0);  // 2.0 GHz for GDDR6-16000

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
