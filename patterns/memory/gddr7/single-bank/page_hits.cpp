// patterns/memory/gddr7/single-bank/page_hits.cpp
//
// Pattern: Sequential reads to same row (page hits)
// Tests: tRL + tBurst timing, page hit behavior
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

bool test_page_hits() {
    std::cout << "\n=== Test: Single Bank Page Hits ===" << std::endl;
    std::cout << "Configuration: Bank 0, same row, 16 reads" << std::endl;
    std::cout << "Expected: 1 page empty, 15 page hits" << std::endl;

    GDDR7Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    for (int i = 0; i < 16; ++i) {
        harness.submit_read(make_address(BANK, ROW, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;
    if (!harness.verify_stats(16, 0, 15, 1, 0)) return false;

    std::cout << "PASS: Page hits work correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=====================================" << std::endl;
    std::cout << "GDDR7 Pattern: Single Bank Page Hits" << std::endl;
    std::cout << "=====================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = test_page_hits();

    // Export trace to organized directory
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR7Harness harness(single_channel_config());
        for (int i = 0; i < 16; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
        }
        harness.run_until_complete();

        // Use organized path if not specified
        if (trace_file.empty()) {
            trace_file = make_trace_path("single-bank", "page_hits_trace.json");
        }
        harness.export_trace(trace_file);
    }

    return pass ? 0 : 1;
}
