// patterns/memory/lpddr5/single-bank/mixed-rw/main.cpp
//
// Pattern: Alternating reads and writes (bus turnaround)
// Tests: tRTW and tWTR timing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../../common/lpddr5_configs.hpp"
#include "../../common/lpddr5_harness.hpp"
#include "../../common/workloads.hpp"
#include "../../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

bool test_mixed_rw() {
    std::cout << "\n=== Test: Mixed Read/Write with Turnaround ===" << std::endl;
    std::cout << "Configuration: Bank 0, same row, alternating R/W" << std::endl;
    std::cout << "Expected: tRTW and tWTR turnaround delays" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    for (int i = 0; i < 4; ++i) {
        if (i % 2 == 0) {
            harness.submit_read(make_address(BANK, ROW, i * 64));
        } else {
            harness.submit_write(make_address(BANK, ROW, i * 64));
        }
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    const auto& stats = harness.stats();
    if (stats.reads != 2 || stats.writes != 2) {
        std::cerr << "FAIL: Expected 2 reads and 2 writes" << std::endl;
        return false;
    }

    if (stats.read_to_write_turnarounds + stats.write_to_read_turnarounds > 0) {
        std::cout << "Turnarounds: R->W=" << stats.read_to_write_turnarounds
                  << ", W->R=" << stats.write_to_read_turnarounds << std::endl;
    }

    std::cout << "PASS: Mixed read/write works correctly" << std::endl;
    return true;
}

void run_multi_fidelity() {
    std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;

    MultiFidelityHarness harness(single_channel_config());
    auto workload = make_mixed_rw_workload();

    std::cout << "\n--- Uncalibrated ---" << std::endl;
    harness.run_comparison(workload, true);

    std::cout << "\n--- Calibrated ---" << std::endl;
    harness.run_calibrated(workload, true);
}

int main(int argc, char* argv[]) {
    std::cout << "=======================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Mixed Read/Write" << std::endl;
    std::cout << "=======================================" << std::endl;

    bool run_fidelity = false;
    std::string trace_file = "mixed_rw_trace.json";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        }
    }

    bool pass = test_mixed_rw();

    if (run_fidelity) {
        run_multi_fidelity();
    }

    // Export trace
    std::cout << "\n=== Trace Export ===" << std::endl;
    LPDDR5Harness harness(single_channel_config());
    for (int i = 0; i < 4; ++i) {
        if (i % 2 == 0) {
            harness.submit_read(make_address(0, 100, i * 64));
        } else {
            harness.submit_write(make_address(0, 100, i * 64));
        }
    }
    harness.run_until_complete();
    harness.export_trace(trace_file);

    return pass ? 0 : 1;
}
