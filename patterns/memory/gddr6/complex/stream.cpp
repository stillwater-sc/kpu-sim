// patterns/memory/gddr6/complex/stream.cpp
//
// Pattern: STREAM Benchmark Memory Access Patterns
// Tests: Classic memory bandwidth benchmark kernels
//
// STREAM Benchmark Kernels:
// - Copy:  a[i] = b[i]              (1 read, 1 write)
// - Scale: a[i] = q * b[i]          (1 read, 1 write)
// - Add:   a[i] = b[i] + c[i]       (2 reads, 1 write)
// - Triad: a[i] = b[i] + q * c[i]   (2 reads, 1 write)
//
// GDDR6 has 16 banks in 4 bank groups, allowing more parallelism
// than LPDDR5's 8 banks. This enables better STREAM performance.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>

#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

// STREAM configuration (use namespace CACHE_LINE_BYTES = 64)
constexpr int ELEMENTS_PER_CACHE_LINE = 64 / 8;  // 8 doubles per line
constexpr int STREAM_SIZE = 512;  // Number of cache lines per array (larger for GDDR6)

// Helper to create address on channel 0
inline uint64_t addr(uint8_t bank, uint32_t row, uint32_t col) {
    return make_address(0, bank, row, col);
}

/// STREAM Copy: a[i] = b[i]
/// 1 read from b, 1 write to a
bool test_stream_copy() {
    std::cout << "\n=== Test: STREAM Copy ===" << std::endl;
    std::cout << "Pattern: a[i] = b[i] (1 read, 1 write per element)" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    // Array layout using all 16 banks:
    // Banks 0-7:  array a (write destination)
    // Banks 8-15: array b (read source)
    constexpr std::array<uint8_t, 8> write_banks = {0, 1, 2, 3, 4, 5, 6, 7};
    constexpr std::array<uint8_t, 8> read_banks = {8, 9, 10, 11, 12, 13, 14, 15};

    for (int i = 0; i < STREAM_SIZE; ++i) {
        uint32_t row = 100 + (i / 64);
        uint32_t col = (i % 64) * CACHE_LINE_BYTES;
        uint8_t read_bank = read_banks[i % 8];
        uint8_t write_bank = write_banks[i % 8];

        // Read b[i], then write a[i]
        harness.submit_read(addr(read_bank, row, col));
        harness.submit_write(addr(write_bank, row, col));
    }

    if (!harness.run_until_complete(500000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_bytes = STREAM_SIZE * CACHE_LINE_BYTES * 2;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- STREAM Copy Analysis ---" << std::endl;
    std::cout << "Elements: " << (STREAM_SIZE * ELEMENTS_PER_CACHE_LINE) << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Active banks: 16" << std::endl;

    std::cout << "PASS: STREAM Copy completed" << std::endl;
    return true;
}

/// STREAM Add: a[i] = b[i] + c[i]
/// 2 reads (b, c), 1 write (a)
bool test_stream_add() {
    std::cout << "\n=== Test: STREAM Add ===" << std::endl;
    std::cout << "Pattern: a[i] = b[i] + c[i] (2 reads, 1 write per element)" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    // Array layout:
    // Banks 0-3:   array a (write destination)
    // Banks 4-7:   array b (read source 1)
    // Banks 8-11:  array c (read source 2)
    // Banks 12-15: unused

    for (int i = 0; i < STREAM_SIZE; ++i) {
        uint32_t row = 100 + (i / 64);
        uint32_t col = (i % 64) * CACHE_LINE_BYTES;

        uint8_t bank_a = (i % 4);           // Banks 0-3
        uint8_t bank_b = 4 + (i % 4);       // Banks 4-7
        uint8_t bank_c = 8 + (i % 4);       // Banks 8-11

        harness.submit_read(addr(bank_b, row, col));
        harness.submit_read(addr(bank_c, row, col));
        harness.submit_write(addr(bank_a, row, col));
    }

    if (!harness.run_until_complete(500000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_bytes = STREAM_SIZE * CACHE_LINE_BYTES * 3;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- STREAM Add Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "R:W ratio: 2:1" << std::endl;

    std::cout << "PASS: STREAM Add completed" << std::endl;
    return true;
}

/// STREAM Triad: a[i] = b[i] + q * c[i]
/// 2 reads (b, c), 1 write (a)
bool test_stream_triad() {
    std::cout << "\n=== Test: STREAM Triad ===" << std::endl;
    std::cout << "Pattern: a[i] = b[i] + q * c[i] (2 reads, 1 write per element)" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    for (int i = 0; i < STREAM_SIZE; ++i) {
        uint32_t row = 100 + (i / 64);
        uint32_t col = (i % 64) * CACHE_LINE_BYTES;

        uint8_t bank_a = (i % 4);
        uint8_t bank_b = 4 + (i % 4);
        uint8_t bank_c = 8 + (i % 4);

        harness.submit_read(addr(bank_b, row, col));
        harness.submit_read(addr(bank_c, row, col));
        harness.submit_write(addr(bank_a, row, col));
    }

    if (!harness.run_until_complete(500000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_bytes = STREAM_SIZE * CACHE_LINE_BYTES * 3;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- STREAM Triad Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: STREAM Triad completed" << std::endl;
    return true;
}

/// Compare all STREAM kernels on GDDR6
bool test_stream_comparison() {
    std::cout << "\n=== Test: STREAM Kernel Comparison (GDDR6) ===" << std::endl;

    struct Result {
        const char* name;
        int reads;
        int writes;
        uint64_t cycles;
        double bytes_per_cycle;
    };
    std::array<Result, 4> results;

    // Copy
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 8> write_banks = {0, 1, 2, 3, 4, 5, 6, 7};
        constexpr std::array<uint8_t, 8> read_banks = {8, 9, 10, 11, 12, 13, 14, 15};

        for (int i = 0; i < STREAM_SIZE; ++i) {
            uint32_t row = 100 + (i / 64);
            uint32_t col = (i % 64) * CACHE_LINE_BYTES;
            harness.submit_read(addr(read_banks[i % 8], row, col));
            harness.submit_write(addr(write_banks[i % 8], row, col));
        }
        harness.run_until_complete();
        uint64_t bytes = STREAM_SIZE * CACHE_LINE_BYTES * 2;
        results[0] = {"Copy", 1, 1, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Scale
    results[1] = results[0];
    results[1].name = "Scale";

    // Add
    {
        GDDR6Harness harness(gddr6_16000_config());
        for (int i = 0; i < STREAM_SIZE; ++i) {
            uint32_t row = 100 + (i / 64);
            uint32_t col = (i % 64) * CACHE_LINE_BYTES;
            harness.submit_read(addr(4 + (i % 4), row, col));
            harness.submit_read(addr(8 + (i % 4), row, col));
            harness.submit_write(addr(i % 4, row, col));
        }
        harness.run_until_complete();
        uint64_t bytes = STREAM_SIZE * CACHE_LINE_BYTES * 3;
        results[2] = {"Add", 2, 1, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Triad
    results[3] = results[2];
    results[3].name = "Triad";

    std::cout << "\n--- STREAM Comparison Results (GDDR6) ---" << std::endl;
    std::cout << std::setw(10) << "Kernel"
              << std::setw(8) << "Reads"
              << std::setw(8) << "Writes"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle" << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.name
                  << std::setw(8) << r.reads
                  << std::setw(8) << r.writes
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::endl;
    }

    std::cout << "\nPASS: STREAM comparison completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "GDDR6 Pattern: STREAM Benchmark" << std::endl;
    std::cout << "Tests classic memory bandwidth benchmark kernels" << std::endl;
    std::cout << "Array size: " << STREAM_SIZE << " cache lines ("
              << (STREAM_SIZE * CACHE_LINE_BYTES / 1024) << " KB per array)" << std::endl;
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
    pass &= test_stream_copy();
    pass &= test_stream_add();
    pass &= test_stream_triad();
    pass &= test_stream_comparison();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        for (int i = 0; i < 64; ++i) {
            uint32_t row = 100 + (i / 64);
            uint32_t col = (i % 64) * CACHE_LINE_BYTES;
            harness.submit_read(addr(4 + (i % 4), row, col));
            harness.submit_read(addr(8 + (i % 4), row, col));
            harness.submit_write(addr(i % 4, row, col));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("complex", "stream_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    if (pass) {
        std::cout << "=== PASS ===" << std::endl;
    } else {
        std::cout << "=== FAIL ===" << std::endl;
    }
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
