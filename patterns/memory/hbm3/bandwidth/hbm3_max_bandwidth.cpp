// patterns/memory/hbm3/bandwidth/hbm3_max_bandwidth.cpp
//
// HBM3 maximum bandwidth pattern
// Attempts to saturate all channels with streaming reads
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/hbm3_harness.hpp"

using namespace sw::kpu::patterns::hbm3;

int main() {
    std::cout << "=== HBM3 Maximum Bandwidth Pattern ===" << std::endl;
    std::cout << "Attempting to saturate bandwidth across all channels" << std::endl;
    std::cout << "Peak theoretical bandwidth: " << HBM3_5600_BANDWIDTH << " GB/s" << std::endl;

    auto config = hbm3_5600_config();
    HBM3Harness harness(config);

    constexpr int NUM_CHANNELS = 16;
    constexpr int NUM_PCS = 2;
    constexpr int BANKS_PER_PC = 4;  // Use 4 banks per PC to reduce total requests
    constexpr int ACCESSES_PER_BANK = 4;
    constexpr uint32_t BASE_ROW = 0;

    std::cout << "\nStreaming reads across resources:" << std::endl;
    std::cout << "  Channels: " << NUM_CHANNELS << std::endl;
    std::cout << "  Pseudo-channels per channel: " << NUM_PCS << std::endl;
    std::cout << "  Banks per pseudo-channel: " << BANKS_PER_PC << std::endl;
    std::cout << "  Accesses per bank: " << ACCESSES_PER_BANK << std::endl;

    int total_requests = 0;

    // Spread requests across channels, PCs, and banks for parallelism
    // Submit in waves, running simulation between waves to avoid queue overflow
    for (int access = 0; access < ACCESSES_PER_BANK; ++access) {
        for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
            for (uint8_t pc = 0; pc < NUM_PCS; ++pc) {
                for (uint8_t bank = 0; bank < BANKS_PER_PC; ++bank) {
                    uint64_t addr = make_address(ch, pc, bank, BASE_ROW, access);
                    // Try to submit, drain queue if full
                    while (!harness.submit_read(addr)) {
                        harness.run_cycles(1);
                    }
                    ++total_requests;
                }
            }
        }
    }

    std::cout << "\nTotal requests submitted: " << total_requests << std::endl;

    // Run simulation
    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    // Calculate achieved bandwidth
    const auto& stats = harness.stats();
    uint64_t total_bytes = stats.reads * CACHE_LINE_BYTES;
    uint64_t total_cycles = harness.current_cycle();
    double clock_ghz = 2.8;  // HBM3-5600 runs at 2.8 GHz CK
    double time_ns = static_cast<double>(total_cycles) / clock_ghz;
    double bandwidth_gbps = (static_cast<double>(total_bytes) / time_ns);

    std::cout << "\n=== Bandwidth Analysis ===" << std::endl;
    std::cout << "  Total bytes transferred: " << total_bytes << std::endl;
    std::cout << "  Total cycles: " << total_cycles << std::endl;
    std::cout << "  Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "  Efficiency: " << std::setprecision(1)
              << (bandwidth_gbps / HBM3_5600_BANDWIDTH * 100.0) << "%" << std::endl;

    // Export trace
    std::string trace_path = make_trace_path("bandwidth", "hbm3_max_bandwidth_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM3 max bandwidth pattern completed successfully" << std::endl;
    return 0;
}
