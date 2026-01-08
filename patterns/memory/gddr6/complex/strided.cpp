// patterns/memory/gddr6/complex/strided.cpp
//
// GDDR6 Strided Access Pattern
// Tests strided access pattern modeling matrix column traversal
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

    std::cout << "=== GDDR6 Strided Access Pattern ===" << std::endl;
    std::cout << "Strided row access pattern (models matrix column traversal)" << std::endl;
    std::cout << "Worst case for page hit rate - every access is page conflict" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint8_t BANK = 0;
    const uint32_t STRIDE_ROWS = 16;  // Access every 16th row
    const uint32_t NUM_ACCESSES = 8;

    std::cout << "\nSubmitting " << NUM_ACCESSES << " reads with stride of "
              << STRIDE_ROWS << " rows" << std::endl;

    // Strided access - each access is to a different row
    for (uint32_t i = 0; i < NUM_ACCESSES; ++i) {
        uint32_t row = i * STRIDE_ROWS;
        harness.submit_read(make_address(CHANNEL, BANK, row, 0));
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

    // Expected: 1 page empty (first access), rest are page conflicts
    uint64_t expected_page_empty = 1;
    uint64_t expected_page_conflicts = NUM_ACCESSES - 1;

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Expected page_empty: " << expected_page_empty
              << ", Actual: " << stats.page_empty << std::endl;
    std::cout << "Expected page_conflicts: " << expected_page_conflicts
              << ", Actual: " << stats.page_conflicts << std::endl;

    bool pass = true;
    if (stats.page_empty != expected_page_empty) {
        std::cerr << "FAIL: page_empty mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_conflicts != expected_page_conflicts) {
        std::cerr << "FAIL: page_conflicts mismatch" << std::endl;
        pass = false;
    }

    std::cout << "\nStrided Access Analysis:" << std::endl;
    std::cout << "  Stride: " << STRIDE_ROWS << " rows" << std::endl;
    std::cout << "  This pattern models:" << std::endl;
    std::cout << "    - Matrix column access (row-major storage)" << std::endl;
    std::cout << "    - Transpose operations" << std::endl;
    std::cout << "    - Gather operations with large stride" << std::endl;
    std::cout << "  Performance impact:" << std::endl;
    std::cout << "    - 0% page hit rate (worst case)" << std::endl;
    std::cout << "    - Every access incurs PRE + ACT overhead" << std::endl;
    std::cout << "    - Latency: ~" << (tRP + tRCDRD + tRL + tBurst) << " cycles/access" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("complex", "strided_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_strided_workload(CHANNEL, BANK, STRIDE_ROWS, NUM_ACCESSES);

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
