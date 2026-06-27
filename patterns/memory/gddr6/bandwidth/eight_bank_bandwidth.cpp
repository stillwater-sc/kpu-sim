// patterns/memory/gddr6/bandwidth/eight_bank_bandwidth.cpp
//
// Pattern: Bandwidth with 8 banks active (comparable to LPDDR5)
// Tests: Bandwidth scaling at LPDDR5-equivalent bank concurrency
//
// Strategy:
// - Use 8 banks distributed across all 4 bank groups (2 per group)
// - Provides comparison point with LPDDR5 max bandwidth
// - Shows GDDR6 performance at equivalent bank concurrency
//
// GDDR6 Bank Organization:
// - Bank Group 0: Banks 0, 1, 2, 3
// - Bank Group 1: Banks 4, 5, 6, 7
// - Bank Group 2: Banks 8, 9, 10, 11
// - Bank Group 3: Banks 12, 13, 14, 15
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>

#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

// Helper to create address on channel 0
inline uint64_t addr(uint8_t bank, uint32_t row, uint32_t col) {
    return make_address(0, bank, row, col);
}

/// Test 8-bank bandwidth (2 banks per group)
/// Comparable to LPDDR5 maximum configuration
bool test_eight_bank_bandwidth() {
    std::cout << "\n=== Test: 8-Bank Bandwidth ===" << std::endl;
    std::cout << "Configuration: 2 banks per group (0,1,4,5,8,9,12,13)" << std::endl;
    std::cout << "Pattern: Round-robin page hits across 8 banks" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    // 2 banks from each of the 4 bank groups
    constexpr std::array<uint8_t, 8> banks = {0, 1, 4, 5, 8, 9, 12, 13};
    constexpr uint32_t ROW = 100;
    constexpr int ROUNDS = 8;  // 8 rounds × 8 banks = 64 accesses

    for (int round = 0; round < ROUNDS; ++round) {
        for (uint8_t bank : banks) {
            harness.submit_read(addr(bank, ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    // Calculate effective bandwidth
    const auto& stats = harness.stats();
    uint64_t total_bytes = 64 * 64;  // 64 accesses × 64 bytes
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / cycles;
    double hit_ratio = 100.0 * stats.page_hits / (stats.page_hits + stats.page_empty);

    std::cout << "\n--- 8-Bank Bandwidth Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit ratio: " << std::setprecision(1) << hit_ratio << "%" << std::endl;

    std::cout << "PASS: 8-bank bandwidth test completed" << std::endl;
    return true;
}

/// Compare GDDR6 8-bank vs 4-bank bandwidth
/// Shows benefit of doubling bank concurrency
bool test_bank_concurrency_comparison() {
    std::cout << "\n=== Test: Bank Concurrency Comparison (4 vs 8 banks) ===" << std::endl;

    constexpr uint32_t ROW = 100;
    constexpr int ACCESSES_PER_BANK = 8;

    // 4-bank test (one per group)
    uint64_t cycles_4bank;
    double bw_4bank;
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 4> banks = {0, 4, 8, 12};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 4 * ACCESSES_PER_BANK * 64;
        cycles_4bank = harness.current_cycle();
        bw_4bank = static_cast<double>(bytes) / cycles_4bank;
    }

    // 8-bank test (two per group)
    uint64_t cycles_8bank;
    double bw_8bank;
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 8> banks = {0, 1, 4, 5, 8, 9, 12, 13};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 8 * ACCESSES_PER_BANK * 64;
        cycles_8bank = harness.current_cycle();
        bw_8bank = static_cast<double>(bytes) / cycles_8bank;
    }

    std::cout << "\n--- Comparison Results ---" << std::endl;
    std::cout << std::setw(12) << "Banks"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(12) << 4
              << std::setw(12) << cycles_4bank
              << std::setw(16) << std::fixed << std::setprecision(2) << bw_4bank << std::endl;
    std::cout << std::setw(12) << 8
              << std::setw(12) << cycles_8bank
              << std::setw(16) << bw_8bank << std::endl;

    double speedup = bw_8bank / bw_4bank;
    std::cout << "\nBandwidth improvement: " << std::setprecision(1) << speedup << "x" << std::endl;

    std::cout << "PASS: Bank concurrency comparison completed" << std::endl;
    return true;
}

/// Test mixed read/write with 8 banks
/// Simulates double-buffering with more banks
bool test_eight_bank_double_buffer() {
    std::cout << "\n=== Test: 8-Bank Double Buffer ===" << std::endl;
    std::cout << "Pattern: Read from 4 banks, write to 4 banks" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    // Read banks: 0, 1, 4, 5 (BG0, BG1)
    // Write banks: 8, 9, 12, 13 (BG2, BG3)
    constexpr std::array<uint8_t, 4> read_banks = {0, 1, 4, 5};
    constexpr std::array<uint8_t, 4> write_banks = {8, 9, 12, 13};
    constexpr uint32_t ROW = 100;

    for (int round = 0; round < 8; ++round) {
        // Interleave reads and writes
        for (size_t i = 0; i < 4; ++i) {
            harness.submit_read(addr(read_banks[i], ROW, round * 64));
            harness.submit_write(addr(write_banks[i], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    [[maybe_unused]] const auto& stats = harness.stats();
    uint64_t total_bytes = (32 + 32) * 64;  // 32 reads + 32 writes
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- Double Buffer Analysis ---" << std::endl;
    std::cout << "Reads: 32, Writes: 32" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: 8-bank double buffer test completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "GDDR6 Pattern: 8-Bank Bandwidth" << std::endl;
    std::cout << "Tests bandwidth with 8 banks (LPDDR5-comparable)" << std::endl;
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
    pass &= test_eight_bank_bandwidth();
    pass &= test_bank_concurrency_comparison();
    pass &= test_eight_bank_double_buffer();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        constexpr std::array<uint8_t, 8> banks = {0, 1, 4, 5, 8, 9, 12, 13};
        constexpr uint32_t ROW = 100;

        for (int round = 0; round < 8; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("bandwidth", "eight_bank_bandwidth_trace.json");
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
