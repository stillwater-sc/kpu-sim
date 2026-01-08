// patterns/memory/hbm3/single-bank/hbm3_mixed_rw.cpp
//
// HBM3 single bank mixed read/write pattern
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm3_harness.hpp"

using namespace sw::kpu::patterns::hbm3;

int main() {
    std::cout << "=== HBM3 Single Bank Mixed R/W Pattern ===" << std::endl;
    std::cout << "Testing interleaved reads and writes with turnaround penalties" << std::endl;

    auto config = single_pc_config();
    HBM3Harness harness(config);

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 200;
    constexpr int NUM_PAIRS = 4;

    // Submit alternating reads and writes to same row
    std::cout << "\nSubmitting " << NUM_PAIRS << " read/write pairs to bank "
              << (int)BANK << ", row " << ROW << std::endl;

    for (int i = 0; i < NUM_PAIRS; ++i) {
        // Read
        uint64_t read_addr = make_address_single_pc(BANK, ROW, i * 2);
        auto read_id = harness.submit_read(read_addr);
        if (!read_id) {
            std::cerr << "FAIL: Could not submit read " << i << std::endl;
            return 1;
        }

        // Write
        uint64_t write_addr = make_address_single_pc(BANK, ROW, i * 2 + 1);
        auto write_id = harness.submit_write(write_addr);
        if (!write_id) {
            std::cerr << "FAIL: Could not submit write " << i << std::endl;
            return 1;
        }
    }

    // Run simulation
    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    // Verify results
    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    if (stats.reads != NUM_PAIRS) {
        std::cerr << "FAIL: Expected " << NUM_PAIRS << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    if (stats.writes != NUM_PAIRS) {
        std::cerr << "FAIL: Expected " << NUM_PAIRS << " writes, got " << stats.writes << std::endl;
        return 1;
    }

    // Should have turnaround penalties
    std::cout << "R->W turnarounds: " << stats.read_to_write_turnarounds << std::endl;
    std::cout << "W->R turnarounds: " << stats.write_to_read_turnarounds << std::endl;

    // Export trace
    std::string trace_path = make_trace_path("single-bank", "hbm3_mixed_rw_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM3 mixed R/W pattern completed successfully" << std::endl;
    return 0;
}
