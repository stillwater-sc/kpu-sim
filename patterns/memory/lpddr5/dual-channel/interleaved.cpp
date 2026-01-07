// patterns/memory/lpddr5/dual-channel/interleaved/main.cpp
//
// Pattern: Interleaved access across dual channels
// Tests: Address interleaving for bandwidth optimization
//
// Interleaving Strategy:
// - Sequential addresses alternate between channels
// - Maximizes channel utilization for streaming workloads
// - Simulates hardware address interleaving
//
// This is the ideal configuration for linear memory access patterns!
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"
#include "../common/workloads.hpp"
#include "../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Test basic interleaving: cache line granularity
/// Each consecutive cache line goes to alternate channel
bool test_cacheline_interleaving() {
    std::cout << "\n=== Test: Cache Line Interleaving ===" << std::endl;
    std::cout << "Configuration: Sequential cache lines alternate channels" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint32_t ROW = 100;
    const uint8_t BANK = 0;

    // 8 sequential cache lines, alternating channels
    // Simulates: addr 0->CH0, addr 64->CH1, addr 128->CH0, ...
    for (int i = 0; i < 8; ++i) {
        uint8_t channel = i % 2;
        harness.submit_read(make_address_dual(channel, BANK, ROW, (i / 2) * 64));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads: 2 page empty (one per channel), 6 page hits
    if (!harness.verify_stats(8, 0, 6, 2, 0)) return false;

    std::cout << "PASS: Cache line interleaving works correctly" << std::endl;
    return true;
}

/// Test row-granularity interleaving
/// Entire rows assigned to channels
bool test_row_interleaving() {
    std::cout << "\n=== Test: Row Interleaving ===" << std::endl;
    std::cout << "Configuration: Rows 0,2,4 -> CH0; Rows 1,3,5 -> CH1" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t BANK = 0;

    // Access rows 0-5, with channel based on row parity
    for (int row = 0; row < 6; ++row) {
        uint8_t channel = row % 2;
        harness.submit_read(make_address_dual(channel, BANK, row, 0));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 6 reads: all different rows = page empties (or conflicts)
    // Since each channel sees different rows, pattern depends on scheduler
    auto& stats = harness.stats();
    if (stats.reads != 6) {
        std::cerr << "FAIL: Expected 6 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Row interleaving works correctly" << std::endl;
    return true;
}

/// Test streaming read pattern
/// Simulates reading a contiguous buffer with interleaved addressing
bool test_streaming_read() {
    std::cout << "\n=== Test: Streaming Read ===" << std::endl;
    std::cout << "Configuration: 16KB streaming read with interleaving" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint32_t ROW = 100;
    const uint8_t banks[4] = {0, 4, 8, 12};

    // Stream 256 cache lines (16KB), interleaved across channels and banks
    // Optimized pattern: round-robin channels, then banks
    for (int line = 0; line < 64; ++line) {
        uint8_t channel = line % 2;
        uint8_t bank = banks[(line / 2) % 4];
        uint32_t col = ((line / 8) % 16) * 64;
        harness.submit_read(make_address_dual(channel, bank, ROW, col));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 64 reads total
    auto& stats = harness.stats();
    if (stats.reads != 64) {
        std::cerr << "FAIL: Expected 64 reads, got " << stats.reads << std::endl;
        return false;
    }

    // Calculate effective bandwidth
    uint64_t total_bytes = 64 * 64;  // 64 cache lines × 64 bytes
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / cycles;
    std::cout << "Effective throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: Streaming read works correctly" << std::endl;
    return true;
}

/// Test streaming write pattern
/// Simulates writing a contiguous buffer with interleaved addressing
bool test_streaming_write() {
    std::cout << "\n=== Test: Streaming Write ===" << std::endl;
    std::cout << "Configuration: 16KB streaming write with interleaving" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint32_t ROW = 200;
    const uint8_t banks[4] = {0, 4, 8, 12};

    // Stream 64 cache lines, interleaved
    for (int line = 0; line < 64; ++line) {
        uint8_t channel = line % 2;
        uint8_t bank = banks[(line / 2) % 4];
        uint32_t col = ((line / 8) % 16) * 64;
        harness.submit_write(make_address_dual(channel, bank, ROW, col));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.writes != 64) {
        std::cerr << "FAIL: Expected 64 writes, got " << stats.writes << std::endl;
        return false;
    }

    std::cout << "PASS: Streaming write works correctly" << std::endl;
    return true;
}

/// Test copy pattern: read interleaved, write interleaved
/// Simulates memcpy with dual-channel optimization
bool test_interleaved_copy() {
    std::cout << "\n=== Test: Interleaved Copy ===" << std::endl;
    std::cout << "Configuration: Read from rows 0-7, write to rows 100-107" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};

    // Copy pattern: alternate read/write, both interleaved
    for (int round = 0; round < 8; ++round) {
        uint8_t channel = round % 2;
        uint8_t bank = banks[(round / 2) % 4];

        // Read from source
        harness.submit_read(make_address_dual(channel, bank, round, round * 64));
        // Write to destination
        harness.submit_write(make_address_dual(channel, bank, 100 + round, round * 64));
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads + 8 writes = 16 total
    // Different rows per access = page conflicts after initial opens
    auto& stats = harness.stats();
    if (stats.reads != 8 || stats.writes != 8) {
        std::cerr << "FAIL: Expected 8 reads and 8 writes" << std::endl;
        return false;
    }

    std::cout << "PASS: Interleaved copy works correctly" << std::endl;
    return true;
}

/// Compare interleaved vs non-interleaved bandwidth
bool test_interleaving_benefit() {
    std::cout << "\n=== Test: Interleaving Benefit Comparison ===" << std::endl;

    const uint32_t ROW = 100;
    const uint8_t banks[4] = {0, 4, 8, 12};

    // Non-interleaved: all to channel 0
    uint64_t non_interleaved_cycles;
    {
        std::cout << "\n--- Non-Interleaved (Single Channel) ---" << std::endl;
        LPDDR5Harness harness(dual_channel_config());

        for (int round = 0; round < 16; ++round) {
            uint8_t bank = banks[round % 4];
            harness.submit_read(make_address_dual(0, bank, ROW, (round / 4) * 64));
        }

        harness.run_until_complete(30000);
        non_interleaved_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << non_interleaved_cycles << std::endl;
        harness.print_stats();
    }

    // Interleaved: alternate channels
    uint64_t interleaved_cycles;
    {
        std::cout << "\n--- Interleaved (Dual Channel) ---" << std::endl;
        LPDDR5Harness harness(dual_channel_config());

        for (int round = 0; round < 16; ++round) {
            uint8_t channel = round % 2;
            uint8_t bank = banks[(round / 2) % 4];
            harness.submit_read(make_address_dual(channel, bank, ROW, (round / 8) * 64));
        }

        harness.run_until_complete(30000);
        interleaved_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << interleaved_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Non-interleaved: " << non_interleaved_cycles << " cycles" << std::endl;
    std::cout << "Interleaved:     " << interleaved_cycles << " cycles" << std::endl;

    if (interleaved_cycles < non_interleaved_cycles) {
        double improvement = 100.0 * (non_interleaved_cycles - interleaved_cycles) / non_interleaved_cycles;
        std::cout << "Interleaving provides " << std::fixed << std::setprecision(1)
                  << improvement << "% improvement" << std::endl;
    } else if (interleaved_cycles == non_interleaved_cycles) {
        std::cout << "No difference (scheduler may already be optimal)" << std::endl;
    } else {
        std::cout << "Non-interleaved faster (unexpected)" << std::endl;
    }

    return true;
}

/// Test tile loading with interleaved channels
/// Simulates loading a tensor tile for KPU computation
bool test_tile_load_interleaved() {
    std::cout << "\n=== Test: Tile Load with Interleaving ===" << std::endl;
    std::cout << "Configuration: 4KB tile (32×32 int32) interleaved across channels" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t TILE_ROW = 0;

    // Load 64 cache lines (4KB), interleaved
    for (int line = 0; line < 64; ++line) {
        uint8_t channel = line % 2;
        uint8_t bank = banks[(line / 2) % 4];
        uint32_t col = ((line / 8) % 16) * 64;
        harness.submit_read(make_address_dual(channel, bank, TILE_ROW, col));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 64) {
        std::cerr << "FAIL: Expected 64 reads" << std::endl;
        return false;
    }

    // Calculate throughput
    uint64_t total_bytes = 64 * 64;  // 4KB
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / cycles;
    std::cout << "Tile load throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Cycles per 4KB tile: " << cycles << std::endl;

    std::cout << "PASS: Tile load interleaved works correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Dual Channel Interleaved" << std::endl;
    std::cout << "Tests address interleaving for bandwidth" << std::endl;
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
    pass &= test_cacheline_interleaving();
    pass &= test_row_interleaving();
    pass &= test_streaming_read();
    pass &= test_streaming_write();
    pass &= test_interleaved_copy();
    pass &= test_interleaving_benefit();
    pass &= test_tile_load_interleaved();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(dual_channel_config());
        const uint32_t ROW = 100;

        // Interleaved cache line pattern
        for (int i = 0; i < 8; ++i) {
            uint8_t channel = i % 2;
            harness.submit_read(make_address_dual(channel, 0, ROW, (i / 2) * 64));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("dual-channel", "interleaved_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
