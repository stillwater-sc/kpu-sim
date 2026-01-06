// patterns/memory/lpddr5/dual-channel/independent/main.cpp
//
// Pattern: Independent access to dual channels
// Tests: Channel isolation and parallel bandwidth
//
// Configuration:
// - Channel 0: Banks 0-15 (standard single-channel addressing)
// - Channel 1: Banks 0-15 (separate address space)
//
// Key insight: True dual-channel provides 2x theoretical bandwidth
// when both channels are accessed simultaneously
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

/// Test channel 0 only
/// Baseline for comparison with dual-channel
bool test_channel_0_only() {
    std::cout << "\n=== Test: Channel 0 Only (Baseline) ===" << std::endl;
    std::cout << "Configuration: 8 reads to channel 0, banks 0,4,8,12" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // All accesses to channel 0
    for (int round = 0; round < 2; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address_dual(0, banks[b], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads: 4 page empty + 4 page hits
    if (!harness.verify_stats(8, 0, 4, 4, 0)) return false;

    std::cout << "PASS: Channel 0 only works correctly" << std::endl;
    return true;
}

/// Test channel 1 only
/// Verifies channel 1 works independently
bool test_channel_1_only() {
    std::cout << "\n=== Test: Channel 1 Only ===" << std::endl;
    std::cout << "Configuration: 8 reads to channel 1, banks 0,4,8,12" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // All accesses to channel 1
    for (int round = 0; round < 2; ++round) {
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address_dual(1, banks[b], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads: 4 page empty + 4 page hits
    if (!harness.verify_stats(8, 0, 4, 4, 0)) return false;

    std::cout << "PASS: Channel 1 only works correctly" << std::endl;
    return true;
}

/// Test both channels independently
/// Alternating requests between channels
bool test_dual_channel_alternating() {
    std::cout << "\n=== Test: Dual Channel Alternating ===" << std::endl;
    std::cout << "Configuration: Alternate between channel 0 and channel 1" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint32_t ROW = 100;

    // Alternate: CH0-B0, CH1-B0, CH0-B4, CH1-B4, ...
    for (int round = 0; round < 4; ++round) {
        // Channel 0
        harness.submit_read(make_address_dual(0, round * 4, ROW, round * 64));
        // Channel 1
        harness.submit_read(make_address_dual(1, round * 4, ROW, round * 64));
    }

    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads total (4 per channel): all page empties
    if (!harness.verify_stats(8, 0, 0, 8, 0)) return false;

    std::cout << "PASS: Dual channel alternating works correctly" << std::endl;
    return true;
}

/// Test parallel load across both channels
/// Simulates loading data from both channels simultaneously
bool test_dual_channel_parallel_load() {
    std::cout << "\n=== Test: Dual Channel Parallel Load ===" << std::endl;
    std::cout << "Configuration: Load 4 banks from each channel in parallel" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Load from both channels, interleaved
    for (int round = 0; round < 4; ++round) {
        for (int b = 0; b < 4; ++b) {
            // Alternate channel each request for maximum parallelism
            uint8_t channel = (round + b) % 2;
            harness.submit_read(make_address_dual(channel, banks[b], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 16 reads total: distributed across channels
    // Each channel has 4 banks accessed twice
    // Channel 0: banks 0,8,4,12 (various patterns)
    // Channel 1: banks 4,12,0,8 (various patterns)
    // Some page hits expected due to same bank/row accesses
    auto& stats = harness.stats();
    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Dual channel parallel load works correctly" << std::endl;
    return true;
}

/// Test sustained dual-channel throughput
/// High-volume traffic to both channels
bool test_dual_channel_sustained() {
    std::cout << "\n=== Test: Dual Channel Sustained Throughput ===" << std::endl;
    std::cout << "Configuration: 32 reads split across both channels" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // 8 rounds × 4 banks = 32 accesses, split between channels
    for (int round = 0; round < 8; ++round) {
        uint8_t channel = round % 2;  // Alternate channels per round
        for (int b = 0; b < 4; ++b) {
            harness.submit_read(make_address_dual(channel, banks[b], ROW, (round / 2) * 64));
        }
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 32 reads total
    // Each channel: 4 rounds × 4 banks = 16 reads
    // Per channel: 4 page empty + 12 page hits = 16
    // Total: 8 page empty + 24 page hits
    if (!harness.verify_stats(32, 0, 24, 8, 0)) return false;

    std::cout << "PASS: Dual channel sustained works correctly" << std::endl;
    return true;
}

/// Compare single-channel vs dual-channel bandwidth
bool test_bandwidth_comparison() {
    std::cout << "\n=== Test: Single vs Dual Channel Bandwidth ===" << std::endl;

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Single channel workload
    uint64_t single_cycles;
    {
        std::cout << "\n--- Single Channel ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int round = 0; round < 8; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address(banks[b], ROW, round * 64));
            }
        }

        harness.run_until_complete(50000);
        single_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << single_cycles << std::endl;
        harness.print_stats();
    }

    // Dual channel workload (same total accesses)
    uint64_t dual_cycles;
    {
        std::cout << "\n--- Dual Channel ---" << std::endl;
        LPDDR5Harness harness(dual_channel_config());

        for (int round = 0; round < 8; ++round) {
            uint8_t channel = round % 2;
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address_dual(channel, banks[b], ROW, (round / 2) * 64));
            }
        }

        harness.run_until_complete(50000);
        dual_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << dual_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Single channel: " << single_cycles << " cycles" << std::endl;
    std::cout << "Dual channel:   " << dual_cycles << " cycles" << std::endl;

    if (dual_cycles < single_cycles) {
        double speedup = 100.0 * (single_cycles - dual_cycles) / single_cycles;
        std::cout << "Dual channel is " << std::fixed << std::setprecision(1)
                  << speedup << "% faster" << std::endl;
    } else {
        std::cout << "No speedup observed (may be scheduling-bound)" << std::endl;
    }

    return true;
}

/// Test read/write split across channels
/// Common pattern: read from one channel, write to another
bool test_read_write_channel_split() {
    std::cout << "\n=== Test: Read/Write Channel Split ===" << std::endl;
    std::cout << "Configuration: Read from CH0, Write to CH1" << std::endl;

    LPDDR5Harness harness(dual_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    const uint32_t ROW = 100;

    // Read from channel 0, write to channel 1
    for (int round = 0; round < 4; ++round) {
        for (int b = 0; b < 2; ++b) {
            harness.submit_read(make_address_dual(0, banks[b], ROW, round * 64));
            harness.submit_write(make_address_dual(1, banks[b + 2], ROW, round * 64));
        }
    }

    if (!harness.run_until_complete(30000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // 8 reads + 8 writes = 16 total
    // Each channel has 2 banks accessed 4 times
    // Per channel: 2 page empty + 6 page hits
    // Total: 4 page empty + 12 page hits
    if (!harness.verify_stats(8, 8, 12, 4, 0)) return false;

    std::cout << "PASS: Read/write channel split works correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Dual Channel Independent" << std::endl;
    std::cout << "Tests independent channel access patterns" << std::endl;
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
    pass &= test_channel_0_only();
    pass &= test_channel_1_only();
    pass &= test_dual_channel_alternating();
    pass &= test_dual_channel_parallel_load();
    pass &= test_dual_channel_sustained();
    pass &= test_bandwidth_comparison();
    pass &= test_read_write_channel_split();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(dual_channel_config());
        const uint8_t banks[4] = {0, 4, 8, 12};
        for (int round = 0; round < 2; ++round) {
            for (int b = 0; b < 4; ++b) {
                harness.submit_read(make_address_dual(round % 2, banks[b], 100, round * 64));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("dual-channel", "independent_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
