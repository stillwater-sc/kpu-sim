// patterns/memory/gddr6/bandwidth/max_bandwidth.cpp
//
// Pattern: Maximum bandwidth with all 16 banks active
// Tests: Peak sustained bandwidth with full bank parallelism
//
// Strategy:
// - Open pages in all 16 banks (one-time activation cost)
// - Interleave page-hit accesses across all banks
// - Measure sustained bandwidth with row buffers open
//
// This represents the theoretical maximum bandwidth achievable
// when workload has enough parallelism to keep all banks busy.
// Typical for GPU-style concurrent thread access patterns.
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

/// Test maximum bandwidth with all 16 banks active
/// Interleaves accesses across all banks for maximum parallelism
bool test_max_bandwidth_all_banks() {
    std::cout << "\n=== Test: Maximum Bandwidth (All 16 Banks) ===" << std::endl;
    std::cout << "Configuration: Banks 0-15 (all banks in all 4 groups)" << std::endl;
    std::cout << "Pattern: Round-robin page hits across all banks" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    constexpr std::array<uint8_t, 16> banks = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    constexpr uint32_t ROW = 100;
    constexpr int ROUNDS = 4;  // 4 rounds × 16 banks = 64 accesses

    // Round-robin across all 16 banks
    // First round: 16 page empties (open all pages)
    // Subsequent rounds: 48 page hits
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

    std::cout << "\n--- Maximum Bandwidth Analysis ---" << std::endl;
    std::cout << "Total bytes: " << total_bytes << std::endl;
    std::cout << "Total cycles: " << cycles << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit ratio: " << std::setprecision(1) << hit_ratio << "%" << std::endl;

    std::cout << "PASS: Maximum bandwidth test completed" << std::endl;
    return true;
}

/// Compare bandwidth scaling: 1, 2, 4, 8, 16 banks
/// Shows how bandwidth scales with bank parallelism
bool test_bandwidth_scaling() {
    std::cout << "\n=== Test: Bandwidth Scaling (1/2/4/8/16 Banks) ===" << std::endl;
    std::cout << "Measures how bandwidth scales with bank concurrency" << std::endl;

    struct ScalingResult {
        int num_banks;
        uint64_t cycles;
        double bytes_per_cycle;
    };
    std::array<ScalingResult, 5> results;

    constexpr uint32_t ROW = 100;
    constexpr int ACCESSES_PER_BANK = 4;

    // Test with 1 bank
    {
        GDDR6Harness harness(gddr6_16000_config());
        for (int i = 0; i < ACCESSES_PER_BANK; ++i) {
            harness.submit_read(addr(0, ROW, i * 64));
        }
        harness.run_until_complete();
        uint64_t bytes = ACCESSES_PER_BANK * 64;
        results[0] = {1, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 2 banks (different groups)
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 2> banks = {0, 4};
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 2 * ACCESSES_PER_BANK * 64;
        results[1] = {2, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 4 banks (one per group)
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
        results[2] = {4, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 8 banks (two per group)
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
        results[3] = {8, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // Test with 16 banks (all banks)
    {
        GDDR6Harness harness(gddr6_16000_config());
        constexpr std::array<uint8_t, 16> banks = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        for (int round = 0; round < ACCESSES_PER_BANK; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
            }
        }
        harness.run_until_complete();
        uint64_t bytes = 16 * ACCESSES_PER_BANK * 64;
        results[4] = {16, harness.current_cycle(),
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

/// Test GPU-style access pattern with 16 banks
/// Many concurrent "threads" each accessing different banks
bool test_gpu_style_access() {
    std::cout << "\n=== Test: GPU-Style Concurrent Access (16 Banks) ===" << std::endl;
    std::cout << "Simulates many threads accessing all 16 banks concurrently" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    // Simulate 64 "threads" each doing 4 accesses
    // Threads are distributed across all 16 banks
    constexpr int NUM_THREADS = 64;
    constexpr int ACCESSES_PER_THREAD = 4;
    constexpr uint32_t BASE_ROW = 100;

    for (int access = 0; access < ACCESSES_PER_THREAD; ++access) {
        for (int thread = 0; thread < NUM_THREADS; ++thread) {
            uint8_t bank = thread % 16;
            uint32_t row = BASE_ROW + (thread / 16);  // Different rows per thread group
            harness.submit_read(addr(bank, row, access * 64));
        }
    }

    if (!harness.run_until_complete(200000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_accesses = NUM_THREADS * ACCESSES_PER_THREAD;
    uint64_t total_bytes = total_accesses * 64;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- GPU-Style Access Analysis ---" << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Accesses per thread: " << ACCESSES_PER_THREAD << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Banks utilized: 16" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: GPU-style access test completed" << std::endl;
    return true;
}

/// Compare GPU-style (high concurrency) vs Accelerator-style (sequential tiles)
bool test_gpu_vs_accelerator() {
    std::cout << "\n=== Test: GPU vs Accelerator Access Patterns ===" << std::endl;

    uint64_t gpu_cycles, accel_cycles;
    double gpu_bw, accel_bw;

    constexpr int TOTAL_ACCESSES = 64;

    // GPU-style: many threads, all banks active
    {
        std::cout << "\n--- GPU-Style (64 threads, 16 banks) ---" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        for (int i = 0; i < TOTAL_ACCESSES; ++i) {
            uint8_t bank = i % 16;
            harness.submit_read(addr(bank, 100, (i / 16) * 64));
        }
        harness.run_until_complete();
        harness.print_stats();

        gpu_cycles = harness.current_cycle();
        gpu_bw = static_cast<double>(TOTAL_ACCESSES * 64) / gpu_cycles;
    }

    // Accelerator-style: sequential tiles, 4 banks at a time
    {
        std::cout << "\n--- Accelerator-Style (16 tiles, 4 banks) ---" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        // Load 16 tiles, 4 accesses per tile, using only 4 banks
        constexpr std::array<uint8_t, 4> banks = {0, 4, 8, 12};
        for (int tile = 0; tile < 16; ++tile) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, 100 + tile, 0));
            }
        }
        harness.run_until_complete();
        harness.print_stats();

        accel_cycles = harness.current_cycle();
        accel_bw = static_cast<double>(TOTAL_ACCESSES * 64) / accel_cycles;
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "GPU-style:         " << std::fixed << std::setprecision(2) << gpu_bw
              << " bytes/cycle (" << gpu_cycles << " cycles)" << std::endl;
    std::cout << "Accelerator-style: " << accel_bw
              << " bytes/cycle (" << accel_cycles << " cycles)" << std::endl;
    std::cout << "GPU advantage: " << std::setprecision(1)
              << (gpu_bw / accel_bw) << "x bandwidth" << std::endl;

    std::cout << "\nPASS: GPU vs Accelerator comparison completed" << std::endl;
    return true;
}

/// Test sustained maximum bandwidth over long period
bool test_sustained_max_bandwidth() {
    std::cout << "\n=== Test: Sustained Maximum Bandwidth ===" << std::endl;
    std::cout << "Configuration: 256 accesses across all 16 banks" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    constexpr std::array<uint8_t, 16> banks = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    constexpr uint32_t ROW = 100;
    constexpr int ROUNDS = 16;  // 16 rounds × 16 banks = 256 accesses

    for (int round = 0; round < ROUNDS; ++round) {
        for (uint8_t bank : banks) {
            harness.submit_read(addr(bank, ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(300000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    const auto& stats = harness.stats();
    uint64_t total_bytes = 256 * 64;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();
    double hit_ratio = 100.0 * stats.page_hits / (stats.page_hits + stats.page_empty);

    std::cout << "\n--- Sustained Bandwidth Analysis ---" << std::endl;
    std::cout << "Total accesses: 256" << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Total cycles: " << harness.current_cycle() << std::endl;
    std::cout << "Sustained throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;
    std::cout << "Page hit ratio: " << std::setprecision(1) << hit_ratio << "%" << std::endl;

    std::cout << "PASS: Sustained maximum bandwidth test completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "GDDR6 Pattern: Maximum Bandwidth (16 Banks)" << std::endl;
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
    pass &= test_gpu_vs_accelerator();
    pass &= test_sustained_max_bandwidth();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        GDDR6Harness harness(gddr6_16000_config());

        constexpr std::array<uint8_t, 16> banks = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        };
        constexpr uint32_t ROW = 100;

        // Generate trace with all 16 banks active
        for (int round = 0; round < 4; ++round) {
            for (uint8_t bank : banks) {
                harness.submit_read(addr(bank, ROW, round * 64));
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
