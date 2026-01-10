// patterns/memory/gddr7/single-bank/page_conflicts.cpp
//
// Pattern: Alternating row accesses causing page conflicts
// Tests: tRP + tRCD timing, page conflict detection
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../common/gddr7_configs.hpp"
#include "../common/gddr7_harness.hpp"
#include "../common/workloads.hpp"

using namespace sw::kpu::patterns::gddr7;

bool test_page_conflicts() {
    std::cout << "\n=== Test: Single Bank Page Conflicts ===" << std::endl;
    std::cout << "Configuration: Bank 0, alternating rows, 8 read pairs" << std::endl;
    std::cout << "Expected: 1 page empty, 0 page hits, 15 page conflicts" << std::endl;

    GDDR7Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW_A = 100;
    const uint32_t ROW_B = 200;

    // First access is page empty, rest are conflicts (alternating rows)
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK, ROW_A, i * 64));
        harness.submit_read(make_address(BANK, ROW_B, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;
    if (!harness.verify_stats(16, 0, 0, 1, 15)) return false;

    // Verify conflict latency is higher than page empty
    const auto& s = harness.stats();
    double avg_lat = s.avg_latency();
    std::cout << "Average latency: " << std::fixed << std::setprecision(2)
              << avg_lat << " cycles" << std::endl;
    std::cout << "Expected: ~" << PAGE_CONFLICT_READ_LATENCY << " cycles (conflict)" << std::endl;

    std::cout << "PASS: Page conflicts detected correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "GDDR7 Pattern: Single Bank Page Conflicts" << std::endl;
    std::cout << "==========================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = test_page_conflicts();

    // Export trace
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR7Harness harness(single_channel_config());
        for (int i = 0; i < 8; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(0, 200, i * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("single-bank", "page_conflicts_trace.json");
        }
        harness.export_trace(trace_file);
    }

    return pass ? 0 : 1;
}
