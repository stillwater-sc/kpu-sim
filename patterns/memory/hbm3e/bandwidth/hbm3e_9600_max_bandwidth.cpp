// patterns/memory/hbm3e/bandwidth/hbm3e_9600_max_bandwidth.cpp
//
// HBM3E-9600 maximum bandwidth test - all channels parallel
// Tests peak theoretical bandwidth of 1.229 TB/s
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/hbm3e_harness.hpp"

using namespace sw::kpu::patterns::hbm3e;

int main() {
    std::cout << "=== HBM3E-9600 Maximum Bandwidth Pattern ===" << std::endl;
    std::cout << "Testing peak bandwidth with all 16 channels active" << std::endl;

    // Use full stack HBM3E-9600 config (16 channels, 32 pseudo-channels)
    auto config = hbm3e_9600_config();
    HBM3EHarness harness(config, HBM3EVariant::HBM3E_9600);

    std::cout << "Configuration: " << harness.variant_name() << " @ "
              << harness.clock_ghz() << " GHz" << std::endl;
    std::cout << "Peak bandwidth: " << harness.peak_bandwidth() << " GB/s" << std::endl;
    std::cout << "Channels: " << (int)config.num_channels << std::endl;
    std::cout << "Pseudo-channels per channel: " << (int)config.pseudo_channels_per_channel << std::endl;

    // Strategy: Issue reads to all channels in parallel, to different banks
    // to maximize parallelism
    constexpr int ACCESSES_PER_PC = 4;
    constexpr uint32_t ROW = 100;

    int total_requests = 0;

    // Submit reads across all channels and pseudo-channels
    std::cout << "\nSubmitting reads across all channels..." << std::endl;

    for (uint8_t channel = 0; channel < config.num_channels; ++channel) {
        for (uint8_t pc = 0; pc < config.pseudo_channels_per_channel; ++pc) {
            for (int i = 0; i < ACCESSES_PER_PC; ++i) {
                // Use different banks for each access to enable parallelism
                uint8_t bank = i % 16;
                uint64_t addr = make_address(channel, pc, bank, ROW, 0);
                auto id = harness.submit_read(addr);
                if (!id) {
                    std::cerr << "FAIL: Could not submit read to channel " << (int)channel
                              << " pc " << (int)pc << std::endl;
                    return 1;
                }
                ++total_requests;
            }
        }
    }

    std::cout << "Total requests submitted: " << total_requests << std::endl;

    // Run simulation
    if (!harness.run_until_complete(20000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    // Verify results
    harness.print_stats();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();
    if (stats.reads != static_cast<uint64_t>(total_requests)) {
        std::cerr << "FAIL: Expected " << total_requests << " reads, got " << stats.reads << std::endl;
        return 1;
    }

    // Calculate achieved bandwidth
    uint64_t total_bytes = total_requests * CACHE_LINE_BYTES;
    uint64_t total_cycles = harness.current_cycle();
    double cycle_time_ns = 1.0 / harness.clock_ghz();  // ns per cycle
    double total_time_ns = total_cycles * cycle_time_ns;
    double achieved_bandwidth = (total_bytes / total_time_ns);  // GB/s

    std::cout << "\n=== Bandwidth Analysis ===" << std::endl;
    std::cout << "  Total bytes transferred: " << total_bytes << " bytes" << std::endl;
    std::cout << "  Total cycles: " << total_cycles << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2)
              << total_time_ns << " ns" << std::endl;
    std::cout << "  Achieved bandwidth: " << achieved_bandwidth << " GB/s" << std::endl;
    std::cout << "  Peak bandwidth: " << harness.peak_bandwidth() << " GB/s" << std::endl;
    std::cout << "  Efficiency: " << std::setprecision(1)
              << (achieved_bandwidth / harness.peak_bandwidth() * 100.0) << "%" << std::endl;

    // Export trace
    std::string trace_path = make_trace_path("bandwidth", "hbm3e_9600_max_bandwidth_trace.json");
    harness.export_trace(trace_path);

    std::cout << "\nPASS: HBM3E-9600 max bandwidth pattern completed successfully" << std::endl;
    return 0;
}
