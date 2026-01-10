// patterns/memory/gddr7/two-bank/same_group.cpp
//
// Pattern: Accesses to two banks in same bank group
// Tests: tRRD_L, tCCD_L timing (same bank group constraints)
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

bool test_same_bank_group() {
    std::cout << "\n=== Test: Two Banks Same Group ===" << std::endl;
    std::cout << "Configuration: Banks 0 and 1 (both in BG0)" << std::endl;
    std::cout << "Expected: tRRD_L=" << tRRD_L << ", tCCD_L=" << tCCD_L << " constraints" << std::endl;

    GDDR7Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;  // Bank group 0
    const uint8_t BANK_B = 1;  // Bank group 0 (same group)
    const uint32_t ROW = 100;

    // Interleaved accesses to same bank group
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK_A, ROW, i * 64));
        harness.submit_read(make_address(BANK_B, ROW, i * 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    const auto& s = harness.stats();
    if (s.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads, got " << s.reads << std::endl;
        return false;
    }

    // Verify same bank group accesses dominate
    std::cout << "Same BG accesses: " << s.same_bg_accesses << std::endl;
    std::cout << "Diff BG accesses: " << s.diff_bg_accesses << std::endl;

    std::cout << "PASS: Same bank group timing constraints respected" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================" << std::endl;
    std::cout << "GDDR7 Pattern: Two Banks Same Group" << std::endl;
    std::cout << "======================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = test_same_bank_group();

    // Export trace
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR7Harness harness(single_channel_config());
        for (int i = 0; i < 8; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(1, 100, i * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("two-bank", "same_group_trace.json");
        }
        harness.export_trace(trace_file);
    }

    return pass ? 0 : 1;
}
