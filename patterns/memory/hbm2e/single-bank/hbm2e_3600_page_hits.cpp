// patterns/memory/hbm2e/single-bank/hbm2e_3600_page_hits.cpp
//
// HBM2E-3600 single bank page hit pattern - sequential reads to same row
// Uses HBM2E-3600 timing @ 1.8 GHz CK (460.8 GB/s peak bandwidth)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm2e_harness.hpp"

using namespace sw::kpu::patterns::hbm2e;

int main() {
    std::cout << "=== HBM2E-3600 Single Bank Page Hits Pattern ===" << std::endl;
    std::cout << "Testing sequential reads to same row (page hit scenario)" << std::endl;

    // Use single pseudo-channel HBM2E-3600 config for focused testing
    auto config = single_pc_config_3600();
    HBM2EHarness harness(config, HBM2EVariant::HBM2E_3600);

    std::cout << "Configuration: " << harness.variant_name() << " @ "
              << harness.clock_ghz() << " GHz" << std::endl;
    std::cout << "Peak bandwidth: " << harness.peak_bandwidth() << " GB/s" << std::endl;

    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;
    constexpr int NUM_ACCESSES = 8;

    // Submit reads to different columns in same row (page hits)
    std::cout << "\nSubmitting " << NUM_ACCESSES << " reads to bank " << (int)BANK
              << ", row " << ROW << std::endl;

    for (int i = 0; i < NUM_ACCESSES; ++i) {
        uint64_t addr = make_address_single_pc(BANK, ROW, i);
        auto id = harness.submit_read(addr);
        if (!id) {
            std::cerr << "FAIL: Could not submit read " << i << std::endl;
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

    // First access is page empty, rest should be page hits
    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    if (stats.reads != NUM_ACCESSES) {
        std::cerr << "FAIL: Expected " << NUM_ACCESSES << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // First access opens page (page_empty), subsequent should be hits
    if (stats.page_hits != NUM_ACCESSES - 1) {
        std::cerr << "FAIL: Expected " << (NUM_ACCESSES - 1) << " page hits, got " << stats.page_hits << std::endl;
        return 1;
    }

    if (stats.page_empty != 1) {
        std::cerr << "FAIL: Expected 1 page empty, got " << stats.page_empty << std::endl;
        return 1;
    }

    // Export trace
    std::string trace_path = make_trace_path("single-bank", "hbm2e_3600_page_hits_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM2E-3600 page hits pattern completed successfully" << std::endl;
    return 0;
}
