// patterns/memory/hbm2/full-bank/hbm2_bank_distribution.cpp
//
// HBM2 Bank Distribution Analysis
// Demonstrates and visualizes how addresses map to banks
// Shows different addressing patterns and their effects on bank utilization
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

/// Print bank distribution histogram
void print_bank_histogram(const std::vector<int>& bank_counts, int num_channels, int pcs_per_channel, int banks_per_pc) {
    int total_banks = num_channels * pcs_per_channel * banks_per_pc;
    int max_count = 0;
    for (int count : bank_counts) {
        max_count = std::max(max_count, count);
    }

    std::cout << "\n  Bank Access Distribution:" << std::endl;
    std::cout << "  ─────────────────────────" << std::endl;

    for (int ch = 0; ch < num_channels; ++ch) {
        std::cout << "  Ch" << ch << ": ";
        for (int pc = 0; pc < pcs_per_channel; ++pc) {
            std::cout << "PC" << pc << "[";
            for (int bank = 0; bank < banks_per_pc; ++bank) {
                int idx = ch * (pcs_per_channel * banks_per_pc) + pc * banks_per_pc + bank;
                int count = bank_counts[idx];
                if (count == 0) {
                    std::cout << ".";
                } else if (count < 5) {
                    std::cout << count;
                } else {
                    std::cout << "*";
                }
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n  Legend: . = 0 accesses, 1-4 = count, * = 5+ accesses" << std::endl;
}

/// Test sequential address access pattern
void test_sequential_addresses() {
    std::cout << "\n=== Test 1: Sequential Address Access ===" << std::endl;
    std::cout << "Accessing addresses 0, 64, 128, 192, ... (stride = 64 bytes)" << std::endl;
    std::cout << "This simulates sequential memory access (streaming)" << std::endl;

    auto config = hbm2_2000_config();
    config.queue_depth = 256;
    HBM2Harness harness(config);

    constexpr int NUM_ACCESSES = 256;
    constexpr int STRIDE = 64;  // Cache line stride
    std::vector<int> bank_counts(256, 0);

    for (int i = 0; i < NUM_ACCESSES; ++i) {
        uint64_t addr = i * STRIDE;

        // Decode address to see which bank it maps to
        // Address format: [row | bank | col | pc | channel | byte_offset]
        uint8_t channel = (addr >> 5) & 0x7;       // bits [7:5]
        uint8_t pc = (addr >> 8) & 0x1;            // bit [8]
        uint8_t bank = (addr >> 9) & 0xF;          // bits [12:9]

        int bank_id = channel * 32 + pc * 16 + bank;
        if (bank_id < 256) {
            bank_counts[bank_id]++;
        }

        while (!harness.submit_read(addr)) {
            harness.run_cycles(1);
        }
    }

    harness.run_until_complete(50000);

    const auto& stats = harness.stats();
    std::cout << "\n  Results:" << std::endl;
    std::cout << "  Reads: " << stats.reads << std::endl;
    std::cout << "  Total cycles: " << harness.current_cycle() << std::endl;

    print_bank_histogram(bank_counts, 8, 2, 16);

    std::string trace_path = make_trace_path("full-bank", "hbm2_sequential_distribution_trace.json");
    harness.export_trace(trace_path);
}

/// Test strided address access pattern (matrix column access)
void test_strided_addresses() {
    std::cout << "\n=== Test 2: Strided Address Access ===" << std::endl;
    std::cout << "Accessing addresses with stride = 4096 bytes (column access)" << std::endl;
    std::cout << "This simulates matrix column access or poor locality" << std::endl;

    auto config = hbm2_2000_config();
    config.queue_depth = 256;
    HBM2Harness harness(config);

    constexpr int NUM_ACCESSES = 256;
    constexpr int STRIDE = 4096;  // Row stride for 64x64 matrix of doubles
    std::vector<int> bank_counts(256, 0);

    for (int i = 0; i < NUM_ACCESSES; ++i) {
        uint64_t addr = i * STRIDE;

        // Decode address
        uint8_t channel = (addr >> 5) & 0x7;
        uint8_t pc = (addr >> 8) & 0x1;
        uint8_t bank = (addr >> 9) & 0xF;

        int bank_id = channel * 32 + pc * 16 + bank;
        if (bank_id < 256) {
            bank_counts[bank_id]++;
        }

        while (!harness.submit_read(addr)) {
            harness.run_cycles(1);
        }
    }

    harness.run_until_complete(50000);

    const auto& stats = harness.stats();
    std::cout << "\n  Results:" << std::endl;
    std::cout << "  Reads: " << stats.reads << std::endl;
    std::cout << "  Total cycles: " << harness.current_cycle() << std::endl;
    std::cout << "  Page conflicts: " << stats.page_conflicts << std::endl;

    print_bank_histogram(bank_counts, 8, 2, 16);

    std::string trace_path = make_trace_path("full-bank", "hbm2_strided_distribution_trace.json");
    harness.export_trace(trace_path);
}

/// Test explicit bank-spread access pattern
void test_bank_spread_pattern() {
    std::cout << "\n=== Test 3: Explicit Bank-Spread Pattern ===" << std::endl;
    std::cout << "Deliberately spreading accesses across all banks" << std::endl;
    std::cout << "This is optimal for maximum parallelism" << std::endl;

    auto config = hbm2_2000_config();
    config.queue_depth = 512;
    HBM2Harness harness(config);

    constexpr int ACCESSES_PER_BANK = 4;
    std::vector<int> bank_counts(256, 0);

    // Explicitly target each bank
    for (int access = 0; access < ACCESSES_PER_BANK; ++access) {
        for (uint8_t ch = 0; ch < 8; ++ch) {
            for (uint8_t pc = 0; pc < 2; ++pc) {
                for (uint8_t bank = 0; bank < 16; ++bank) {
                    uint64_t addr = make_address(ch, pc, bank, access, 0);

                    int bank_id = ch * 32 + pc * 16 + bank;
                    bank_counts[bank_id]++;

                    while (!harness.submit_read(addr)) {
                        harness.run_cycles(1);
                    }
                }
            }
        }
    }

    harness.run_until_complete(100000);

    const auto& stats = harness.stats();
    std::cout << "\n  Results:" << std::endl;
    std::cout << "  Reads: " << stats.reads << std::endl;
    std::cout << "  Total cycles: " << harness.current_cycle() << std::endl;
    std::cout << "  Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    print_bank_histogram(bank_counts, 8, 2, 16);

    std::string trace_path = make_trace_path("full-bank", "hbm2_bank_spread_distribution_trace.json");
    harness.export_trace(trace_path);
}

/// Test hot-bank pattern (all accesses to single bank)
void test_hot_bank_pattern() {
    std::cout << "\n=== Test 4: Hot Bank Pattern ===" << std::endl;
    std::cout << "All accesses to Bank 0 of Channel 0, PC 0" << std::endl;
    std::cout << "This is worst-case for parallelism (anti-pattern)" << std::endl;

    auto config = single_pc_config();
    config.queue_depth = 128;
    HBM2Harness harness(config);

    constexpr int NUM_ACCESSES = 64;
    std::vector<int> bank_counts(256, 0);

    // All accesses to bank 0, but different rows (page conflicts)
    for (int i = 0; i < NUM_ACCESSES; ++i) {
        uint64_t addr = make_address_single_pc(0, i, 0);  // Bank 0, row i
        bank_counts[0]++;

        while (!harness.submit_read(addr)) {
            harness.run_cycles(1);
        }
    }

    harness.run_until_complete(100000);

    const auto& stats = harness.stats();
    std::cout << "\n  Results:" << std::endl;
    std::cout << "  Reads: " << stats.reads << std::endl;
    std::cout << "  Total cycles: " << harness.current_cycle() << std::endl;
    std::cout << "  Page conflicts: " << stats.page_conflicts << std::endl;
    std::cout << "  Avg latency: " << std::fixed << std::setprecision(2)
              << stats.avg_latency() << " cycles" << std::endl;

    // Only show single PC for this test
    std::cout << "\n  Bank 0 received all " << bank_counts[0] << " accesses" << std::endl;
    std::cout << "  This serializes all operations - very inefficient!" << std::endl;

    std::string trace_path = make_trace_path("full-bank", "hbm2_hot_bank_distribution_trace.json");
    harness.export_trace(trace_path);
}

int main() {
    std::cout << "=== HBM2 Bank Distribution Analysis ===" << std::endl;
    std::cout << "\nThis test demonstrates how different access patterns" << std::endl;
    std::cout << "distribute across HBM2's 256 banks and the impact on" << std::endl;
    std::cout << "performance and parallelism." << std::endl;
    std::cout << "\nHBM2 Address Mapping:" << std::endl;
    std::cout << "  [row:14 | bank:4 | col:6 | pc:1 | channel:3 | offset:5]" << std::endl;
    std::cout << "  Total: 8 channels × 2 PCs × 16 banks = 256 banks" << std::endl;

    // Run all distribution tests
    test_sequential_addresses();
    test_strided_addresses();
    test_bank_spread_pattern();
    test_hot_bank_pattern();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "1. Sequential: Cycles through channels first (good locality)" << std::endl;
    std::cout << "2. Strided: May cause bank conflicts (depends on stride)" << std::endl;
    std::cout << "3. Bank-spread: Maximum parallelism (best throughput)" << std::endl;
    std::cout << "4. Hot-bank: Serialized access (worst case)" << std::endl;

    std::cout << "\nPASS: Bank distribution analysis completed" << std::endl;
    std::cout << "\nTraces exported to: traces/memory/hbm2/full-bank/" << std::endl;
    return 0;
}
