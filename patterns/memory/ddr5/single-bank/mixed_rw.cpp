// patterns/memory/ddr5/single-bank/mixed_rw.cpp
//
// Pattern: Mixed read/write operations to same row
// Tests: R->W and W->R turnaround timing (tRTW, tWTR)
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

bool test_mixed_rw() {
    std::cout << "\n=== Test: Mixed Read/Write ===" << std::endl;
    std::cout << "Configuration: Bank 0, same row, alternating R/W" << std::endl;
    std::cout << "Expected: 8 reads, 8 writes, page hits with turnarounds" << std::endl;

    DDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // Alternating read/write to same row
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK, ROW, i * 128));
        harness.submit_write(make_address(BANK, ROW, i * 128 + 64));
    }

    if (!harness.run_until_complete()) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    const auto& s = harness.stats();
    if (s.reads != 8 || s.writes != 8) {
        std::cerr << "FAIL: Expected 8 reads and 8 writes" << std::endl;
        return false;
    }

    // Check turnaround stats
    std::cout << "R->W turnarounds: " << s.read_to_write_turnarounds << std::endl;
    std::cout << "W->R turnarounds: " << s.write_to_read_turnarounds << std::endl;

    std::cout << "PASS: Mixed R/W operations completed correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================" << std::endl;
    std::cout << "DDR5 Pattern: Mixed Read/Write" << std::endl;
    std::cout << "================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = test_mixed_rw();

    // Export trace
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        DDR5Harness harness(single_channel_config());
        for (int i = 0; i < 8; ++i) {
            harness.submit_read(make_address(0, 100, i * 128));
            harness.submit_write(make_address(0, 100, i * 128 + 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("single-bank", "mixed_rw_trace.json");
        }
        harness.export_trace(trace_file);
    }

    return pass ? 0 : 1;
}
