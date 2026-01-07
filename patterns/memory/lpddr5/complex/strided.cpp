// patterns/memory/lpddr5/complex/strided/main.cpp
//
// Pattern: Strided memory access patterns
// Tests: Matrix transpose, column access, and irregular strides
//
// Strided access is common in:
// - Matrix column access (stride = row width)
// - Tensor dimension traversal
// - Image processing (e.g., accessing every Nth pixel)
//
// Challenge: Large strides may cross row boundaries, causing page conflicts
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"
#include "../common/workloads.hpp"
#include "../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Test unit stride (baseline: sequential access)
/// This is the optimal case for comparison
bool test_unit_stride() {
    std::cout << "\n=== Test: Unit Stride (Sequential) ===" << std::endl;
    std::cout << "Configuration: Access every element sequentially" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // 16 sequential cache line accesses
    for (int i = 0; i < 16; ++i) {
        harness.submit_read(make_address(BANK, ROW, i * 64));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // All accesses to same row = 1 page empty + 15 page hits
    if (!harness.verify_stats(16, 0, 15, 1, 0)) return false;

    std::cout << "PASS: Unit stride works correctly" << std::endl;
    return true;
}

/// Test small stride (2 cache lines)
/// Skipping every other cache line
bool test_stride_2() {
    std::cout << "\n=== Test: Stride 2 (Every Other Cache Line) ===" << std::endl;
    std::cout << "Configuration: Access pattern 0, 2, 4, 6, ... (stride=2)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // 16 accesses with stride 2
    for (int i = 0; i < 16; ++i) {
        harness.submit_read(make_address(BANK, ROW, (i * 2) * 64));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // Stride 2 pattern spans more column space, may cross internal boundaries
    // Expect some page empties and mostly page hits
    auto& stats = harness.stats();
    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }
    // Should have high page hit ratio (>80%)
    double hit_ratio = 100.0 * stats.page_hits / (stats.page_hits + stats.page_empty + stats.page_conflicts);
    if (hit_ratio < 80.0) {
        std::cerr << "FAIL: Page hit ratio too low: " << hit_ratio << "%" << std::endl;
        return false;
    }

    std::cout << "PASS: Stride 2 works correctly" << std::endl;
    return true;
}

/// Test row-crossing stride
/// Stride large enough to cross row boundaries
bool test_row_crossing_stride() {
    std::cout << "\n=== Test: Row-Crossing Stride ===" << std::endl;
    std::cout << "Configuration: Stride crosses row boundary each access" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;

    // Each access to a different row = all page conflicts after first
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK, i * 10, 0));  // Row 0, 10, 20, ...
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // Different row each time: 1 page empty + 7 page conflicts
    if (!harness.verify_stats(8, 0, 0, 1, 7)) return false;

    std::cout << "PASS: Row-crossing stride works correctly" << std::endl;
    return true;
}

/// Test matrix column access pattern
/// Simulates accessing a column in a row-major matrix
bool test_matrix_column_access() {
    std::cout << "\n=== Test: Matrix Column Access ===" << std::endl;
    std::cout << "Configuration: Column access in 64x64 matrix (row-major)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // 64x64 matrix of 4-byte elements = 16KB
    // Row stride = 64 * 4 = 256 bytes = 4 cache lines
    const uint8_t BANK = 0;
    const uint32_t BASE_ROW = 100;

    // Access column 0 of first 16 rows
    // Each row is 256 bytes, so stride = 4 cache lines
    for (int row = 0; row < 16; ++row) {
        // Column 0 is at offset 0 of each row
        // Row stride in cache lines = 4
        harness.submit_read(make_address(BANK, BASE_ROW, row * 256));
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // All in same DRAM row (different columns within 1KB row)
    // Should be mostly page hits
    auto& stats = harness.stats();
    std::cout << "Page hit ratio: " << std::fixed << std::setprecision(1)
              << (100.0 * stats.page_hits / (stats.page_hits + stats.page_empty + stats.page_conflicts))
              << "%" << std::endl;

    std::cout << "PASS: Matrix column access works correctly" << std::endl;
    return true;
}

/// Test bank-spreading stride
/// Stride that distributes accesses across multiple banks
bool test_bank_spreading_stride() {
    std::cout << "\n=== Test: Bank-Spreading Stride ===" << std::endl;
    std::cout << "Configuration: Stride distributes across 4 banks" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Alternate across banks with each access
    for (int i = 0; i < 16; ++i) {
        uint8_t bank = banks[i % 4];
        harness.submit_read(make_address(bank, ROW, (i / 4) * 64));
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 4 banks × 4 accesses = 16 total
    // Each bank: 1 page empty + 3 page hits
    // Total: 4 page empty + 12 page hits
    if (!harness.verify_stats(16, 0, 12, 4, 0)) return false;

    std::cout << "PASS: Bank-spreading stride works correctly" << std::endl;
    return true;
}

/// Compare stride performance vs sequential
bool test_stride_comparison() {
    std::cout << "\n=== Test: Stride Performance Comparison ===" << std::endl;

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // Sequential access (stride 1)
    uint64_t sequential_cycles;
    {
        std::cout << "\n--- Sequential (Stride 1) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 16; ++i) {
            harness.submit_read(make_address(BANK, ROW, i * 64));
        }

        harness.run_until_complete(20000);
        sequential_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << sequential_cycles << std::endl;
        harness.print_stats();
    }

    // Strided access (stride 4)
    uint64_t stride4_cycles;
    {
        std::cout << "\n--- Stride 4 ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 16; ++i) {
            harness.submit_read(make_address(BANK, ROW, (i * 4) * 64));
        }

        harness.run_until_complete(30000);
        stride4_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << stride4_cycles << std::endl;
        harness.print_stats();
    }

    // Row-crossing stride
    uint64_t row_crossing_cycles;
    {
        std::cout << "\n--- Row-Crossing Stride ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 16; ++i) {
            harness.submit_read(make_address(BANK, i * 10, 0));  // Different row each time
        }

        harness.run_until_complete(50000);
        row_crossing_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << row_crossing_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Sequential:    " << sequential_cycles << " cycles (baseline)" << std::endl;
    std::cout << "Stride 4:      " << stride4_cycles << " cycles";
    if (stride4_cycles > sequential_cycles) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * (stride4_cycles - sequential_cycles) / sequential_cycles)
                  << "% slower)";
    }
    std::cout << std::endl;

    std::cout << "Row-crossing:  " << row_crossing_cycles << " cycles";
    if (row_crossing_cycles > sequential_cycles) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * (row_crossing_cycles - sequential_cycles) / sequential_cycles)
                  << "% slower)";
    }
    std::cout << std::endl;

    return true;
}

/// Test power-of-2 strides
/// Common in FFT, image processing
bool test_power_of_2_strides() {
    std::cout << "\n=== Test: Power-of-2 Strides ===" << std::endl;

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    for (int power = 1; power <= 4; ++power) {
        int stride = 1 << power;  // 2, 4, 8, 16
        std::cout << "\n--- Stride " << stride << " ---" << std::endl;

        LPDDR5Harness harness(single_channel_config());

        // Adjust count to keep total access count reasonable
        int count = 16 / stride;
        if (count < 4) count = 4;

        for (int i = 0; i < count; ++i) {
            harness.submit_read(make_address(BANK, ROW, (i * stride) * 64));
        }

        harness.run_until_complete(20000);
        std::cout << "Total cycles: " << harness.current_cycle() << std::endl;
        harness.print_stats();
    }

    return true;
}

/// Test transpose-like access pattern
/// Simulates matrix transpose: read row-wise, write column-wise
bool test_transpose_pattern() {
    std::cout << "\n=== Test: Transpose-Like Pattern ===" << std::endl;
    std::cout << "Configuration: Read sequential, write strided" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;   // Source
    const uint8_t BANK_B = 4;   // Destination
    const uint32_t ROW = 100;

    // Simulated transpose: read 4 sequential elements, write with stride
    for (int i = 0; i < 4; ++i) {
        // Read sequential
        harness.submit_read(make_address(BANK_A, ROW, i * 64));
        // Write strided (simulating column-major output)
        harness.submit_write(make_address(BANK_B, ROW, i * 256));  // Stride of 4 cache lines
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 4 reads + 4 writes = 8 total
    // Reads: 1 empty + 3 hits
    // Writes: 1 empty + 3 hits
    // Total: 2 empty + 6 hits
    if (!harness.verify_stats(4, 4, 6, 2, 0)) return false;

    std::cout << "PASS: Transpose pattern works correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Strided Access" << std::endl;
    std::cout << "Tests non-sequential memory access patterns" << std::endl;
    std::cout << "================================================" << std::endl;

    bool export_trace = true;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    bool pass = true;
    pass &= test_unit_stride();
    pass &= test_stride_2();
    pass &= test_row_crossing_stride();
    pass &= test_matrix_column_access();
    pass &= test_bank_spreading_stride();
    pass &= test_stride_comparison();
    pass &= test_power_of_2_strides();
    pass &= test_transpose_pattern();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint8_t BANK = 0;
        const uint32_t ROW = 100;

        // Strided access pattern
        for (int i = 0; i < 8; ++i) {
            harness.submit_read(make_address(BANK, ROW, (i * 4) * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("complex", "strided_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
