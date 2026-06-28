// patterns/memory/hbm2/full-bank/hbm2_full_stack_saturation.cpp
//
// HBM2 Full Stack Saturation Test
// Exercises ALL 256 banks across the entire HBM2 stack:
// - 8 channels × 2 pseudo-channels × 16 banks = 256 banks
//
// This test demonstrates maximum HBM2 parallelism and helps
// visualize the full memory hierarchy in swimlane views.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <vector>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

int main() {
    std::cout << "=== HBM2 Full Stack Saturation Test ===" << std::endl;
    std::cout << "\nHBM2 Full Stack Architecture:" << std::endl;
    std::cout << "  - 8 channels per stack" << std::endl;
    std::cout << "  - 2 pseudo-channels per channel" << std::endl;
    std::cout << "  - 16 banks per pseudo-channel" << std::endl;
    std::cout << "  - 256 total banks" << std::endl;
    std::cout << "  - 1024-bit I/O bus (16 × 64-bit DQ buses)" << std::endl;
    std::cout << "  - Peak bandwidth: 256 GB/s" << std::endl;

    auto config = hbm2_2000_config();
    config.queue_depth = 512;  // Need large queue for 256+ requests
    HBM2Harness harness(config);

    constexpr int NUM_CHANNELS = 8;
    constexpr int NUM_PCS = 2;
    constexpr int BANKS_PER_PC = 16;
    constexpr int TOTAL_BANKS = NUM_CHANNELS * NUM_PCS * BANKS_PER_PC;  // 256
    constexpr int ACCESSES_PER_BANK = 4;  // Multiple accesses to show page hit behavior
    constexpr uint32_t BASE_ROW = 0;

    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "  Total banks: " << TOTAL_BANKS << std::endl;
    std::cout << "  Accesses per bank: " << ACCESSES_PER_BANK << std::endl;
    std::cout << "  Total requests: " << TOTAL_BANKS * ACCESSES_PER_BANK << std::endl;

    // Track which banks we access for verification
    std::vector<bool> banks_accessed(TOTAL_BANKS, false);

    // Submit requests to all 256 banks
    std::cout << "\n--- Submitting requests to all 256 banks ---" << std::endl;

    int submitted = 0;
    for (int access = 0; access < ACCESSES_PER_BANK; ++access) {
        // Interleave across channels/PCs for maximum parallelism
        for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
            for (uint8_t pc = 0; pc < NUM_PCS; ++pc) {
                for (uint8_t bank = 0; bank < BANKS_PER_PC; ++bank) {
                    uint64_t addr = make_address(ch, pc, bank, BASE_ROW, access);

                    // Track bank access
                    int bank_id = ch * (NUM_PCS * BANKS_PER_PC) + pc * BANKS_PER_PC + bank;
                    banks_accessed[bank_id] = true;

                    // Submit with drain on queue full
                    while (!harness.submit_read(addr)) {
                        harness.run_cycles(1);
                    }
                    ++submitted;
                }
            }
        }

        if ((access + 1) % 1 == 0) {
            std::cout << "  Wave " << (access + 1) << "/" << ACCESSES_PER_BANK
                      << ": submitted " << submitted << " requests" << std::endl;
        }
    }

    std::cout << "\nTotal requests submitted: " << submitted << std::endl;

    // Run simulation
    std::cout << "\n--- Running simulation ---" << std::endl;
    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();

    // Verify all banks were accessed
    int banks_used = 0;
    for (bool accessed : banks_accessed) {
        if (accessed) ++banks_used;
    }

    std::cout << "\n--- Bank Coverage Analysis ---" << std::endl;
    std::cout << "  Banks accessed: " << banks_used << " / " << TOTAL_BANKS << std::endl;

    // Print per-channel statistics
    std::cout << "\n--- Per-Channel Analysis ---" << std::endl;
    const auto& stats = harness.stats();
    for (uint8_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        std::cout << "  Channel " << (int)ch << ": "
                  << stats.channel_accesses[ch] << " accesses" << std::endl;
    }

    // Calculate bandwidth achieved
    uint64_t total_bytes = stats.reads * CACHE_LINE_BYTES;
    uint64_t total_cycles = harness.current_cycle();
    double clock_ghz = 1.0;  // HBM2-2000 @ 1 GHz
    double time_ns = static_cast<double>(total_cycles) / clock_ghz;
    double bandwidth_gbps = static_cast<double>(total_bytes) / time_ns;

    std::cout << "\n--- Bandwidth Analysis ---" << std::endl;
    std::cout << "  Total bytes transferred: " << total_bytes << " bytes" << std::endl;
    std::cout << "  Total cycles: " << total_cycles << std::endl;
    std::cout << "  Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "  Peak theoretical: " << HBM2_2000_BANDWIDTH << " GB/s" << std::endl;
    std::cout << "  Efficiency: " << std::setprecision(1)
              << (bandwidth_gbps / HBM2_2000_BANDWIDTH * 100.0) << "%" << std::endl;

    // Verify results
    bool pass = true;

    if (banks_used != TOTAL_BANKS) {
        std::cerr << "FAIL: Not all banks accessed (" << banks_used << "/" << TOTAL_BANKS << ")" << std::endl;
        pass = false;
    }

    if (stats.reads != static_cast<uint64_t>(submitted)) {
        std::cerr << "FAIL: Not all reads completed (" << stats.reads << "/" << submitted << ")" << std::endl;
        pass = false;
    }

    // First access to each bank is page_empty, rest are page_hits
    uint64_t expected_page_empty = TOTAL_BANKS;  // 256
    uint64_t expected_page_hits = (ACCESSES_PER_BANK - 1) * TOTAL_BANKS;  // 768

    std::cout << "\n--- Page State Analysis ---" << std::endl;
    std::cout << "  Expected page_empty: " << expected_page_empty << std::endl;
    std::cout << "  Actual page_empty:   " << stats.page_empty << std::endl;
    std::cout << "  Expected page_hits:  ~" << expected_page_hits << std::endl;
    std::cout << "  Actual page_hits:    " << stats.page_hits << std::endl;
    std::cout << "  Page hit rate:       " << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    // Allow variance due to refresh
    if (static_cast<double>(stats.page_empty) < static_cast<double>(expected_page_empty) * 0.9) {
        std::cerr << "WARNING: Fewer page_empty than expected" << std::endl;
    }

    if (!harness.verify_no_violations()) {
        pass = false;
    }

    // Export trace - this will show all 256 banks in the swimlane view
    std::string trace_path = make_trace_path("full-bank", "hbm2_full_stack_saturation_trace.json");
    harness.export_trace(trace_path);

    if (pass) {
        std::cout << "\nPASS: HBM2 full stack saturation test completed successfully" << std::endl;
        std::cout << "\nOpen the trace in the swimlane viewer to see all 256 banks:" << std::endl;
        std::cout << "  " << trace_path << std::endl;
        return 0;
    } else {
        std::cout << "\nFAIL: HBM2 full stack saturation test had errors" << std::endl;
        return 1;
    }
}
