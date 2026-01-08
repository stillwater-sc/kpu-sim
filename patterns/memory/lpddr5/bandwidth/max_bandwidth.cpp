// patterns/memory/lpddr5/bandwidth/max_bandwidth.cpp
//
// Pattern: Maximum bandwidth with all 8 banks active
// Tests: Peak sustained bandwidth with full bank parallelism
//
// Strategy:
// - Open pages in all 8 banks (one-time activation cost)
// - Interleave page-hit accesses across all banks
// - Measure sustained bandwidth with row buffers open
//
// This represents the theoretical maximum bandwidth achievable
// when workload has enough parallelism to keep all banks busy.
// Typical for GPU-style concurrent thread access patterns.
//
// LPDDR5 Bank Organization:
// - Bank Group 0: Banks 0, 1, 2, 3
// - Bank Group 1: Banks 4, 5, 6, 7
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Test maximum bandwidth with all 8 banks active
/// Interleaves accesses across all banks for maximum parallelism
bool test_max_bandwidth_all_banks() {
    std::cout << "\n=== Test: Maximum Bandwidth (All 8 Banks) ===" << std::endl;
    std::cout << "Configuration: Banks 0-7 (all banks in both groups)" << std::endl;
    std::cout << "Pattern: Round-robin page hits across all banks" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    constexpr std::array<uint8_t, 8> banks = {0, 1, 2, 3, 4, 5, 6, 7};
    constexpr uint32_t ROW = 100;
    constexpr int ROUNDS = 8;  // 8 rounds × 8 banks = 64 accesses

    // Round-robin across all 8 banks
    // First round: 8 page empties (open all pages)
    // Subsequent rounds: 56 page hits
    for (int round = 0; round < ROUNDS; ++round) {
        for (uint8_t bank : banks) {
            harness.submit_read(make_address(bank, ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 64 reads: 8 page empties + 56 page hits
    if (!harness.verify_stats(64, 0, 56, 8, 0)) return false;

    // Calculate effective bandwidth
    const auto& stats = harness.stats();
    uint64_t total_bytes = 64 * 64;  // 64 accesses × 64 bytes
    uint64_t cycles = harness.current_cycle();
    double bytes_per_cycle = static_cast<double>(total_bytes) / cycles;
    double hit_ratio = 100.0 * stats.page_hits / (stats.page_hits + stats.page_empty);

    std::cout << "\n--- Bandwidth Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit ratio: " << std::setprecision(1) << hit_ratio << "%" << std::endl;

    std::cout << "PASS: Maximum bandwidth test completed" << std::endl;
    return true;
}

/// Compare bandwidth scaling: 1, 2, 4, 8 banks
/// Shows how bandwidth scales with bank parallelism
bool test_bandwidth_scaling() {
    std::cout << "\n=== Test: Bandwidth Scaling (1/2/4/8 Banks) ===" << std::endl;
    std::cout << "Measures how bandwidth scales with bank concurrency" << std::endl;

    struct ScalingResult {
        int num_banks;
        uint64_t cycles;
        double bytes_per_cycle;
    };
    std::array<ScalingResult, 4> results;

    constexpr uint32_t ROW = 100;
    constexpr int ACCESSES_PER_BANK = 8;

    // Test with 1 bank
    {
        LPDDR5Harness harness(single_channel_config());
        for (int i = 0; i < ACCESSES_PER_BANK; ++i) {
            harness.submit_read(make_address(0, ROW, i * 64));
        }
        harness.run_until_complete();
        uint64_t bytes = ACCESSES_PER_BANK * 64;
        results[0] = {1, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 2 banks (different groups for max parallelism)
    {
        LPDDR5Harness harness(single_channel_config());
        constexpr std::array<uint8_t, 2> banks = {0, 4};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(make_address(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 2 * ACCESSES_PER_BANK * 64;
        results[1] = {2, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 4 banks (one per group slot)
    {
        LPDDR5Harness harness(single_channel_config());
        constexpr std::array<uint8_t, 4> banks = {0, 1, 4, 5};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(make_address(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 4 * ACCESSES_PER_BANK * 64;
        results[2] = {4, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 8 banks (all banks)
    {
        LPDDR5Harness harness(single_channel_config());
        constexpr std::array<uint8_t, 8> banks = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(make_address(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 8 * ACCESSES_PER_BANK * 64;
        results[3] = {8, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Print scaling results
    std::cout << "\n--- Bandwidth Scaling Results ---" << std::endl;
    std::cout << std::setw(8) << "Banks"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(48, '-') << std::endl;

    double baseline = results[0].bytes_per_cycle;
    for (const auto& r : results) {
        std::cout << std::setw(8) << r.num_banks
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::setw(11) << std::setprecision(1) << (r.bytes_per_cycle / baseline) << "x"
                  << std::endl;
    }

    std::cout << "\nPASS: Bandwidth scaling test completed" << std::endl;
    return true;
}

/// Test GPU-style access pattern
/// Many concurrent "threads" each accessing different banks
bool test_gpu_style_access() {
    std::cout << "\n=== Test: GPU-Style Concurrent Access ===" << std::endl;
    std::cout << "Simulates many threads accessing different banks concurrently" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Simulate 32 "threads" each doing 4 accesses
    // Threads are distributed across all 8 banks
    constexpr int NUM_THREADS = 32;
    constexpr int ACCESSES_PER_THREAD = 4;
    constexpr uint32_t BASE_ROW = 100;

    for (int access = 0; access < ACCESSES_PER_THREAD; ++access) {
        for (int thread = 0; thread < NUM_THREADS; ++thread) {
            uint8_t bank = thread % 8;
            uint32_t row = BASE_ROW + (thread / 8);  // Different rows per thread group
            harness.submit_read(make_address(bank, row, access * 64));
        }
    }

    if (!harness.run_until_complete(200000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    const auto& stats = harness.stats();
    uint64_t total_accesses = NUM_THREADS * ACCESSES_PER_THREAD;
    uint64_t total_bytes = total_accesses * 64;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- GPU-Style Access Analysis ---" << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Accesses per thread: " << ACCESSES_PER_THREAD << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: GPU-style access test completed" << std::endl;
    return true;
}

/// Test tile-based access pattern (accelerator style)
/// Sequential tile loads - fewer banks active at once
bool test_tile_sequential_access() {
    std::cout << "\n=== Test: Tile-Based Sequential Access ===" << std::endl;
    std::cout << "Simulates accelerator loading tiles sequentially" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Load 4 tiles sequentially, 2 banks per tile
    // This is more representative of how a tile-based accelerator works
    constexpr int NUM_TILES = 4;
    constexpr int CACHE_LINES_PER_TILE = 8;

    for (int tile = 0; tile < NUM_TILES; ++tile) {
        // Each tile uses 2 banks (one from each group)
        uint8_t bank0 = (tile % 4);      // BG0: 0, 1, 2, 3
        uint8_t bank1 = 4 + (tile % 4);  // BG1: 4, 5, 6, 7
        uint32_t row = 100 + tile;

        for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
            harness.submit_read(make_address(bank0, row, line * 64));
            harness.submit_read(make_address(bank1, row, line * 64));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_accesses = NUM_TILES * CACHE_LINES_PER_TILE * 2;
    uint64_t total_bytes = total_accesses * 64;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- Tile-Based Access Analysis ---" << std::endl;
    std::cout << "Tiles: " << NUM_TILES << std::endl;
    std::cout << "Cache lines per tile: " << CACHE_LINES_PER_TILE << " (× 2 banks)" << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: Tile-based access test completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Maximum Bandwidth (8 Banks)" << std::endl;
    std::cout << "Tests peak bandwidth with full bank parallelism" << std::endl;
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
    pass &= test_max_bandwidth_all_banks();
    pass &= test_bandwidth_scaling();
    pass &= test_gpu_style_access();
    pass &= test_tile_sequential_access();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        constexpr std::array<uint8_t, 8> banks = {0, 1, 2, 3, 4, 5, 6, 7};
        constexpr uint32_t ROW = 100;

        // Generate trace with all 8 banks active
        for (int round = 0; round < 8; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(make_address(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("bandwidth", "max_bandwidth_trace.json");
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
