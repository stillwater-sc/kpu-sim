// patterns/memory/hbm3/multi-channel/hbm3_eight_channel.cpp
//
// HBM3 eight channel pattern
// Tests parallel operation across 8 channels
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm3_harness.hpp"

using namespace sw::kpu::patterns::hbm3;

int main() {
    std::cout << "=== HBM3 Eight Channel Pattern ===" << std::endl;
    std::cout << "Testing parallel operation across 8 channels" << std::endl;

    auto config = eight_channel_config();
    HBM3Harness harness(config);

    constexpr uint8_t PC = 0;
    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;
    constexpr int ACCESSES_PER_CHANNEL = 4;
    constexpr int NUM_CHANNELS = 8;

    std::cout << "\nSubmitting " << ACCESSES_PER_CHANNEL << " reads to each of "
              << NUM_CHANNELS << " channels" << std::endl;

    // Submit reads to all channels
    for (int i = 0; i < ACCESSES_PER_CHANNEL; ++i) {
        for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
            uint64_t addr = make_address(ch, PC, BANK, ROW, i);
            auto id = harness.submit_read(addr);
            if (!id) {
                std::cerr << "FAIL: Could not submit read to channel " << (int)ch << std::endl;
                return 1;
            }
        }
    }

    // Run simulation
    if (!harness.run_until_complete(5000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    uint64_t expected_reads = ACCESSES_PER_CHANNEL * NUM_CHANNELS;
    if (stats.reads != expected_reads) {
        std::cerr << "FAIL: Expected " << expected_reads << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // Verify channel distribution
    std::cout << "\nChannel access distribution:" << std::endl;
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        std::cout << "  Channel " << (int)ch << ": " << stats.channel_accesses[ch] << " accesses" << std::endl;
    }

    // Export trace
    std::string trace_path = make_trace_path("multi-channel", "hbm3_eight_channel_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM3 eight channel pattern completed successfully" << std::endl;
    return 0;
}
