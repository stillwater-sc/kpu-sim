// patterns/memory/ddr5/two-bank/diff_groups.cpp
//
// Pattern: Accesses to two banks in different bank groups
// Tests: tRRD_S, tCCD_S timing (different bank group constraints)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../common/ddr5_configs.hpp"
#include "../common/ddr5_harness.hpp"
#include "../common/workloads.hpp"

using namespace sw::kpu::patterns::ddr5;

bool test_diff_bank_groups() {
    std::cout << "\n=== Test: Two Banks Different Groups ===" << std::endl;
    std::cout << "Configuration: Banks 0 (BG0) and 8 (BG1)" << std::endl;
    std::cout << "Expected: tRRD_S=" << tRRD_S << ", tCCD_S=" << tCCD_S << " constraints (faster)" << std::endl;

    DDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;   // Bank group 0
    const uint8_t BANK_B = 8;   // Bank group 1 (different group)
    const uint32_t ROW = 100;

    // Interleaved accesses to different bank groups
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

    // Verify different bank group accesses
    std::cout << "Same BG accesses: " << s.same_bg_accesses << std::endl;
    std::cout << "Diff BG accesses: " << s.diff_bg_accesses << std::endl;

    // Different bank groups should be faster due to shorter timing constraints
    std::cout << "Average latency: " << std::fixed << std::setprecision(2)
              << s.avg_latency() << " cycles" << std::endl;

    std::cout << "PASS: Different bank group timing constraints respected" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "DDR5 Pattern: Two Banks Different Groups" << std::endl;
    std::cout << "============================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = test_diff_bank_groups();

    // Export trace
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        DDR5Harness harness(single_channel_config());
        for (int i = 0; i < 8; ++i) {
            harness.submit_read(make_address(0, 100, i * 64));
            harness.submit_read(make_address(8, 100, i * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("two-bank", "diff_groups_trace.json");
        }
        harness.export_trace(trace_file);
    }

    return pass ? 0 : 1;
}
