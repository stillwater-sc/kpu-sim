// patterns/memory/gddr6/dual-channel/interleaved.cpp
//
// GDDR6 Dual Channel Interleaved Pattern
// Tests interleaved access across both channels for bandwidth aggregation
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

    std::cout << "=== GDDR6 Dual Channel Interleaved Pattern ===" << std::endl;
    std::cout << "Interleaved accesses across both channels" << std::endl;
    std::cout << "Tests bandwidth aggregation through address interleaving" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;
    const uint32_t NUM_ACCESSES = 8;

    std::cout << "\nSubmitting " << NUM_ACCESSES << " reads with channel interleaving" << std::endl;

    // Interleaved pattern: even addresses to channel 0, odd to channel 1
    // This simulates address-bit-based channel interleaving
    for (uint32_t i = 0; i < NUM_ACCESSES; ++i) {
        uint8_t channel = i % 2;  // Alternates: 0, 1, 0, 1, ...
        uint32_t col = i / 2;
        harness.submit_read(make_address(channel, BANK, ROW, col));
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

    // Expected: 2 page empty (first access to each channel), rest are page hits
    uint64_t expected_page_empty = 2;
    uint64_t expected_page_hits = NUM_ACCESSES - 2;

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

    std::cout << "\nInterleaving Analysis:" << std::endl;
    std::cout << "  Access pattern: CH0, CH1, CH0, CH1, ..." << std::endl;
    std::cout << "  Each channel receives sequential accesses" << std::endl;
    std::cout << "  Interleaving hides channel latency" << std::endl;
    std::cout << "  Effective bandwidth: 2x single channel bandwidth" << std::endl;
    std::cout << "  Common in GDDR6 memory controllers for graphics workloads" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("dual-channel", "interleaved_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_dual_channel_interleaved_workload(ROW, NUM_ACCESSES);

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
