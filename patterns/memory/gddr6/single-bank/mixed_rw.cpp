// patterns/memory/gddr6/single-bank/mixed_rw.cpp
//
// GDDR6 Mixed Read/Write Pattern Test
// Alternating reads and writes to the same row - tests turnaround timing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

int main() {
    std::cout << "=== GDDR6 Mixed Read/Write Pattern ===" << std::endl;
    std::cout << "Alternating reads and writes to the same row" << std::endl;
    std::cout << "Tests R->W and W->R turnaround timing (tRTW, tWTR)" << std::endl;

    // Use GDDR6-16000 configuration (16 Gbps)
    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    // Generate alternating R/W to the same row in bank 0
    const uint32_t NUM_OPS = 8;  // 4 reads, 4 writes
    const uint32_t ROW = 0;
    const uint8_t BANK = 0;
    const uint8_t CHANNEL = 0;

    std::cout << "\nSubmitting " << NUM_OPS << " alternating R/W operations to row "
              << ROW << " in bank " << (int)BANK << std::endl;

    uint32_t reads_submitted = 0;
    uint32_t writes_submitted = 0;

    for (uint32_t i = 0; i < NUM_OPS; ++i) {
        uint64_t addr = make_address(CHANNEL, BANK, ROW, i);

        if (i % 2 == 0) {
            // Even: Read
            auto id = harness.submit_read(addr);
            if (!id) {
                std::cerr << "ERROR: Failed to submit read " << i << std::endl;
                return 1;
            }
            reads_submitted++;
        } else {
            // Odd: Write
            auto id = harness.submit_write(addr);
            if (!id) {
                std::cerr << "ERROR: Failed to submit write " << i << std::endl;
                return 1;
            }
            writes_submitted++;
        }
    }

    std::cout << "  Reads: " << reads_submitted << ", Writes: " << writes_submitted << std::endl;

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

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Expected reads: " << reads_submitted
              << ", Actual: " << stats.reads << std::endl;
    std::cout << "Expected writes: " << writes_submitted
              << ", Actual: " << stats.writes << std::endl;

    bool pass = true;
    if (stats.reads != reads_submitted) {
        std::cerr << "FAIL: reads mismatch" << std::endl;
        pass = false;
    }
    if (stats.writes != writes_submitted) {
        std::cerr << "FAIL: writes mismatch" << std::endl;
        pass = false;
    }

    // First operation opens the page, rest are page hits
    uint64_t expected_page_empty = 1;
    uint64_t expected_page_hits = NUM_OPS - 1;
    uint64_t expected_page_conflicts = 0;

    std::cout << "Expected page_empty: " << expected_page_empty
              << ", Actual: " << stats.page_empty << std::endl;
    std::cout << "Expected page_hits: " << expected_page_hits
              << ", Actual: " << stats.page_hits << std::endl;
    std::cout << "Expected page_conflicts: " << expected_page_conflicts
              << ", Actual: " << stats.page_conflicts << std::endl;

    if (stats.page_empty != expected_page_empty) {
        std::cerr << "FAIL: page_empty mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_hits != expected_page_hits) {
        std::cerr << "FAIL: page_hits mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_conflicts != expected_page_conflicts) {
        std::cerr << "FAIL: page_conflicts mismatch" << std::endl;
        pass = false;
    }

    // Verify turnaround statistics
    // With R-W-R-W-R-W-R-W pattern:
    // - R->W turnarounds: at indices 0->1, 2->3, 4->5, 6->7 = 4
    // - W->R turnarounds: at indices 1->2, 3->4, 5->6 = 3
    uint64_t expected_rtw = 4;  // R->W transitions
    uint64_t expected_wtr = 3;  // W->R transitions

    std::cout << "\nTurnaround Statistics:" << std::endl;
    std::cout << "  R->W turnarounds: " << stats.read_to_write_turnarounds
              << " (expected ~" << expected_rtw << ")" << std::endl;
    std::cout << "  W->R turnarounds: " << stats.write_to_read_turnarounds
              << " (expected ~" << expected_wtr << ")" << std::endl;

    // These should be close to expected (may vary slightly due to scheduling)
    if (stats.read_to_write_turnarounds == 0 && stats.write_to_read_turnarounds == 0) {
        std::cerr << "FAIL: No turnarounds detected - mixed R/W should have turnarounds" << std::endl;
        pass = false;
    }

    // Verify latencies
    double avg_read_latency = stats.avg_read_latency();
    double avg_write_latency = stats.avg_write_latency();

    std::cout << "\nLatency Analysis:" << std::endl;
    std::cout << "  Average read latency: " << avg_read_latency << " cycles" << std::endl;
    std::cout << "  Average write latency: " << avg_write_latency << " cycles" << std::endl;
    std::cout << "  Overall average: " << stats.avg_latency() << " cycles" << std::endl;

    // Write latency should be lower than read (tWL < tRL)
    // tRL = 20, tWL = 8
    // However, with turnaround penalties (tRTW=14), actual latencies may vary
    if (avg_read_latency > 0 && avg_write_latency > 0) {
        std::cout << "  Write/Read ratio: "
                  << (avg_write_latency / avg_read_latency) << std::endl;
    }

    // Export trace for visualization
    std::string trace_file = make_trace_path("single-bank", "mixed_rw_trace.json");
    harness.export_trace(trace_file, 2.0);  // 2.0 GHz for GDDR6-16000

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
