// patterns/memory/lpddr5/complex/tile-load/main.cpp
//
// Pattern: KPU tile loading patterns
// Tests: Realistic tile data movement for matrix operations
//
// KPU Tile Format:
// - Standard tile: 32x32 elements = 4KB (int32)
// - Row-major layout
// - Aligned to cache line boundaries
//
// Tile Loading Scenarios:
// - Single tile load (matmul input)
// - Double-buffered tile streaming
// - Multi-tile parallel load
// - Tiled matrix multiply access patterns
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

/// Constants for tile operations
constexpr uint32_t TILE_DIM = 32;                          // 32x32 elements
constexpr uint32_t ELEMENT_SIZE = 4;                       // int32 = 4 bytes
constexpr uint32_t TILE_ROW_BYTES = TILE_DIM * ELEMENT_SIZE;  // 128 bytes
constexpr uint32_t TILE_TOTAL_BYTES = TILE_DIM * TILE_ROW_BYTES;  // 4KB
constexpr uint32_t TILE_CACHE_LINES = TILE_TOTAL_BYTES / CACHE_LINE_BYTES;  // 64

/// Test loading a single tile
/// Baseline for tile loading performance
bool test_single_tile_load() {
    std::cout << "\n=== Test: Single Tile Load ===" << std::endl;
    std::cout << "Configuration: Load 4KB tile (32x32 int32)" << std::endl;
    std::cout << "Tile cache lines: " << TILE_CACHE_LINES << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    const uint32_t ROW = 100;

    // Load tile row by row
    // Each row = 128 bytes = 2 cache lines
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
        uint32_t offset = row * TILE_ROW_BYTES;
        harness.submit_read(make_address(BANK, ROW, offset));
        harness.submit_read(make_address(BANK, ROW, offset + CACHE_LINE_BYTES));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != TILE_CACHE_LINES) {
        std::cerr << "FAIL: Expected " << TILE_CACHE_LINES << " reads" << std::endl;
        return false;
    }

    // Calculate throughput
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(TILE_TOTAL_BYTES) / static_cast<double>(cycles);
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Cycles per tile: " << cycles << std::endl;

    std::cout << "PASS: Single tile load works correctly" << std::endl;
    return true;
}

/// Test loading tile distributed across banks
/// Optimal for parallelism
bool test_multi_bank_tile_load() {
    std::cout << "\n=== Test: Multi-Bank Tile Load ===" << std::endl;
    std::cout << "Configuration: Tile distributed across 4 banks" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Distribute tile across banks: each bank gets 1KB
    // Round-robin bank assignment
    for (uint32_t i = 0; i < TILE_CACHE_LINES; ++i) {
        uint8_t bank = banks[i % 4];
        uint32_t offset = (i / 4) * CACHE_LINE_BYTES;
        harness.submit_read(make_address(bank, ROW, offset));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != TILE_CACHE_LINES) {
        std::cerr << "FAIL: Expected " << TILE_CACHE_LINES << " reads" << std::endl;
        return false;
    }

    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(TILE_TOTAL_BYTES) / static_cast<double>(cycles);
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Cycles per tile: " << cycles << std::endl;

    std::cout << "PASS: Multi-bank tile load works correctly" << std::endl;
    return true;
}

/// Test double-buffered tile loading
/// Common pattern: load next tile while processing current
bool test_double_buffer_tiles() {
    std::cout << "\n=== Test: Double-Buffered Tile Loading ===" << std::endl;
    std::cout << "Configuration: Alternate between two tile buffers" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;   // Buffer A
    const uint8_t BANK_B = 4;   // Buffer B
    const uint32_t ROW_A = 100;
    const uint32_t ROW_B = 200;

    // Simulate loading two tiles in double-buffer fashion
    // In practice, compute would happen between loads
    // Load 16 cache lines from each buffer (half tile for demo)
    for (int round = 0; round < 2; ++round) {
        for (int i = 0; i < 16; ++i) {
            if (round == 0) {
                harness.submit_read(make_address(BANK_A, ROW_A, i * CACHE_LINE_BYTES));
            } else {
                harness.submit_read(make_address(BANK_B, ROW_B, i * CACHE_LINE_BYTES));
            }
        }
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 32 reads total: 16 per buffer
    // Each buffer: 1 page empty + 15 page hits
    if (!harness.verify_stats(32, 0, 30, 2, 0)) return false;

    std::cout << "PASS: Double-buffered tiles work correctly" << std::endl;
    return true;
}

/// Test matmul tile access pattern
/// Load A tile rows, B tile columns, write C tile
bool test_matmul_tile_pattern() {
    std::cout << "\n=== Test: Matrix Multiply Tile Pattern ===" << std::endl;
    std::cout << "Configuration: C = A × B tile multiply" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK_A = 0;   // Input tile A
    const uint8_t BANK_B = 4;   // Input tile B
    const uint8_t BANK_C = 8;   // Output tile C
    const uint32_t ROW = 100;

    // Simplified pattern: load 8 rows from A, 8 cols from B, write 8 rows to C
    // In real matmul, this repeats for full tile

    // Load A tile (8 cache lines representing 4 rows)
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK_A, ROW, i * CACHE_LINE_BYTES));
    }

    // Load B tile (8 cache lines)
    for (int i = 0; i < 8; ++i) {
        harness.submit_read(make_address(BANK_B, ROW, i * CACHE_LINE_BYTES));
    }

    // Write C tile (8 cache lines)
    for (int i = 0; i < 8; ++i) {
        harness.submit_write(make_address(BANK_C, ROW, i * CACHE_LINE_BYTES));
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads + 8 writes = 24 total
    // Each bank: 1 page empty + 7 page hits
    // Total: 3 page empty + 21 page hits
    if (!harness.verify_stats(16, 8, 21, 3, 0)) return false;

    std::cout << "PASS: Matmul tile pattern works correctly" << std::endl;
    return true;
}

/// Test streaming multiple tiles sequentially
/// Simulates processing a large tensor tile by tile
bool test_tile_streaming() {
    std::cout << "\n=== Test: Tile Streaming ===" << std::endl;
    std::cout << "Configuration: Load 4 sequential tiles" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};

    // Load 4 tiles, each 16 cache lines (1KB for demo)
    for (int tile = 0; tile < 4; ++tile) {
        uint8_t bank = banks[tile];
        uint32_t base_row = 100 + tile;

        for (int line = 0; line < 16; ++line) {
            harness.submit_read(make_address(bank, base_row, line * CACHE_LINE_BYTES));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 64 reads (4 tiles × 16 cache lines)
    // Each tile: 1 page empty + 15 page hits
    // Total: 4 page empty + 60 page hits
    if (!harness.verify_stats(64, 0, 60, 4, 0)) return false;

    std::cout << "PASS: Tile streaming works correctly" << std::endl;
    return true;
}

/// Test dual-channel tile load
/// Maximum bandwidth configuration
bool test_dual_channel_tile_load() {
    std::cout << "\n=== Test: Dual-Channel Tile Load ===" << std::endl;
    std::cout << "Configuration: Tile interleaved across 2 channels" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Interleave tile load across channels
    for (uint32_t i = 0; i < TILE_CACHE_LINES; ++i) {
        uint8_t channel = i % 2;
        uint8_t bank = banks[(i / 2) % 4];
        uint32_t offset = (i / 8) * CACHE_LINE_BYTES;
        harness.submit_read(make_address_dual(channel, bank, ROW, offset));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != TILE_CACHE_LINES) {
        std::cerr << "FAIL: Expected " << TILE_CACHE_LINES << " reads" << std::endl;
        return false;
    }

    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(TILE_TOTAL_BYTES) / static_cast<double>(cycles);
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Cycles per tile: " << cycles << std::endl;

    std::cout << "PASS: Dual-channel tile load works correctly" << std::endl;
    return true;
}

/// Compare tile loading strategies
bool test_tile_load_comparison() {
    std::cout << "\n=== Test: Tile Load Strategy Comparison ===" << std::endl;

    const uint32_t ROW = 100;

    // Single bank
    uint64_t single_bank_cycles;
    {
        std::cout << "\n--- Single Bank ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (uint32_t i = 0; i < 32; ++i) {  // 32 cache lines for demo
            harness.submit_read(make_address(0, ROW, i * CACHE_LINE_BYTES));
        }

        harness.run_until_complete(50000);
        single_bank_cycles = harness.current_cycle();
        std::cout << "Cycles: " << single_bank_cycles << std::endl;
        harness.print_stats();
    }

    // Multi-bank (4 banks)
    uint64_t multi_bank_cycles;
    {
        std::cout << "\n--- Multi-Bank (4 banks) ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint8_t banks[4] = {0, 4, 8, 12};
        for (uint32_t i = 0; i < 32; ++i) {
            uint8_t bank = banks[i % 4];
            harness.submit_read(make_address(bank, ROW, (i / 4) * CACHE_LINE_BYTES));
        }

        harness.run_until_complete(50000);
        multi_bank_cycles = harness.current_cycle();
        std::cout << "Cycles: " << multi_bank_cycles << std::endl;
        harness.print_stats();
    }

    // Dual channel interleaved
    uint64_t dual_channel_cycles;
    {
        std::cout << "\n--- Dual Channel Interleaved ---" << std::endl;
        LPDDR5Harness harness(dual_channel_config());

        const uint8_t banks[4] = {0, 4, 8, 12};
        for (uint32_t i = 0; i < 32; ++i) {
            uint8_t channel = i % 2;
            uint8_t bank = banks[(i / 2) % 4];
            harness.submit_read(make_address_dual(channel, bank, ROW, (i / 8) * CACHE_LINE_BYTES));
        }

        harness.run_until_complete(50000);
        dual_channel_cycles = harness.current_cycle();
        std::cout << "Cycles: " << dual_channel_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison (32 cache lines = 2KB) ---" << std::endl;
    std::cout << "Single bank:          " << single_bank_cycles << " cycles (baseline)" << std::endl;
    std::cout << "Multi-bank (4):       " << multi_bank_cycles << " cycles";
    if (multi_bank_cycles < single_bank_cycles) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(single_bank_cycles - multi_bank_cycles) / static_cast<double>(single_bank_cycles))
                  << "% faster)";
    }
    std::cout << std::endl;
    std::cout << "Dual channel:         " << dual_channel_cycles << " cycles";
    if (dual_channel_cycles < single_bank_cycles) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(single_bank_cycles - dual_channel_cycles) / static_cast<double>(single_bank_cycles))
                  << "% faster)";
    }
    std::cout << std::endl;

    return true;
}

/// Test MLP-style tile loading
/// Multiple input tiles, weight tile, output tile
bool test_mlp_tile_pattern() {
    std::cout << "\n=== Test: MLP Tile Loading Pattern ===" << std::endl;
    std::cout << "Configuration: Input tiles + Weight tile + Output tile" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint32_t ROW = 100;

    // MLP pattern: load input, load weights, write output
    // Using different banks for each

    // Load input tile (16 cache lines, channel 0, bank 0)
    for (int i = 0; i < 16; ++i) {
        harness.submit_read(make_address_dual(0, 0, ROW, i * CACHE_LINE_BYTES));
    }

    // Load weight tile (16 cache lines, channel 1, bank 0)
    for (int i = 0; i < 16; ++i) {
        harness.submit_read(make_address_dual(1, 0, ROW + 10, i * CACHE_LINE_BYTES));
    }

    // Write output tile (16 cache lines, channel 0, bank 4)
    for (int i = 0; i < 16; ++i) {
        harness.submit_write(make_address_dual(0, 4, ROW + 20, i * CACHE_LINE_BYTES));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 32 reads + 16 writes = 48 total
    // Each tile: 1 page empty + 15 page hits
    // Total: 3 page empty + 45 page hits
    if (!harness.verify_stats(32, 16, 45, 3, 0)) return false;

    std::cout << "PASS: MLP tile pattern works correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Tile Loading" << std::endl;
    std::cout << "Tests KPU tile data movement patterns" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Tile dimensions: " << TILE_DIM << "x" << TILE_DIM << " elements" << std::endl;
    std::cout << "Tile size: " << TILE_TOTAL_BYTES << " bytes (" << TILE_CACHE_LINES << " cache lines)" << std::endl;
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
    pass &= test_single_tile_load();
    pass &= test_multi_bank_tile_load();
    pass &= test_double_buffer_tiles();
    pass &= test_matmul_tile_pattern();
    pass &= test_tile_streaming();
    pass &= test_dual_channel_tile_load();
    pass &= test_tile_load_comparison();
    pass &= test_mlp_tile_pattern();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint8_t banks[4] = {0, 4, 8, 12};
        const uint32_t ROW = 100;

        // Multi-bank tile load pattern
        for (uint32_t i = 0; i < 16; ++i) {
            uint8_t bank = banks[i % 4];
            harness.submit_read(make_address(bank, ROW, (i / 4) * CACHE_LINE_BYTES));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("complex", "tile_load_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
