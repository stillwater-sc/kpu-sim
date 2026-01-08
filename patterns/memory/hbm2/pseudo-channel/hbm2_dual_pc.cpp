// patterns/memory/hbm2/pseudo-channel/hbm2_dual_pc.cpp
//
// HBM2 dual pseudo-channel pattern
// Tests independent operation of two pseudo-channels within a channel
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

int main() {
    std::cout << "=== HBM2 Dual Pseudo-Channel Pattern ===" << std::endl;
    std::cout << "Testing parallel operation of two pseudo-channels" << std::endl;

    // Use single channel with 2 pseudo-channels
    auto config = single_channel_config();
    HBM2Harness harness(config);

    constexpr uint8_t CHANNEL = 0;
    constexpr uint8_t PC_0 = 0;
    constexpr uint8_t PC_1 = 1;
    constexpr uint8_t BANK = 0;
    constexpr uint32_t ROW = 100;
    constexpr int ACCESSES_PER_PC = 4;

    std::cout << "\nSubmitting " << ACCESSES_PER_PC << " reads to each pseudo-channel" << std::endl;

    // Interleave accesses to both pseudo-channels
    for (int i = 0; i < ACCESSES_PER_PC; ++i) {
        // PC0
        uint64_t addr_pc0 = make_address(CHANNEL, PC_0, BANK, ROW, i);
        auto id_0 = harness.submit_read(addr_pc0);
        if (!id_0) {
            std::cerr << "FAIL: Could not submit read to PC0" << std::endl;
            return 1;
        }

        // PC1
        uint64_t addr_pc1 = make_address(CHANNEL, PC_1, BANK, ROW, i);
        auto id_1 = harness.submit_read(addr_pc1);
        if (!id_1) {
            std::cerr << "FAIL: Could not submit read to PC1" << std::endl;
            return 1;
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
    if (stats.reads != ACCESSES_PER_PC * 2) {
        std::cerr << "FAIL: Expected " << (ACCESSES_PER_PC * 2) << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // Export trace
    std::string trace_path = make_trace_path("pseudo-channel", "hbm2_dual_pc_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM2 dual pseudo-channel pattern completed successfully" << std::endl;
    return 0;
}
