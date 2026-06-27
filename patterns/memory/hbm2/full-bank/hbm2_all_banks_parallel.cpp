// patterns/memory/hbm2/full-bank/hbm2_all_banks_parallel.cpp
//
// HBM2 All Banks Parallel Pattern
// Exercises all 16 banks within a single pseudo-channel simultaneously
// to demonstrate maximum bank-level parallelism
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include "../common/hbm2_harness.hpp"

using namespace sw::kpu::patterns::hbm2;

int main() {
    std::cout << "=== HBM2 All Banks Parallel Pattern ===" << std::endl;
    std::cout << "Exercising all 16 banks within a single pseudo-channel" << std::endl;
    std::cout << "\nHBM2 Bank Architecture:" << std::endl;
    std::cout << "  - 16 banks per pseudo-channel" << std::endl;
    std::cout << "  - 4 bank groups (4 banks each)" << std::endl;
    std::cout << "  - Bank Group 0: Banks 0-3" << std::endl;
    std::cout << "  - Bank Group 1: Banks 4-7" << std::endl;
    std::cout << "  - Bank Group 2: Banks 8-11" << std::endl;
    std::cout << "  - Bank Group 3: Banks 12-15" << std::endl;

    // Use single pseudo-channel config to isolate bank behavior
    auto config = single_pc_config();
    config.queue_depth = 256;  // Accommodate all 16 banks × multiple accesses
    HBM2Harness harness(config);

    constexpr int BANKS_PER_PC = 16;
    constexpr int ACCESSES_PER_BANK = 8;  // Multiple accesses per bank
    constexpr uint32_t BASE_ROW = 100;

    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "  Banks: " << BANKS_PER_PC << std::endl;
    std::cout << "  Accesses per bank: " << ACCESSES_PER_BANK << std::endl;
    std::cout << "  Total requests: " << BANKS_PER_PC * ACCESSES_PER_BANK << std::endl;

    // Phase 1: Submit requests to all 16 banks in parallel (page empty)
    std::cout << "\n--- Phase 1: Page Empty Reads (all banks, same row) ---" << std::endl;

    int submitted = 0;
    for (int access = 0; access < ACCESSES_PER_BANK; ++access) {
        for (uint8_t bank = 0; bank < BANKS_PER_PC; ++bank) {
            // All accesses to same row (page hits after first)
            uint64_t addr = make_address_single_pc(bank, BASE_ROW, access);
            while (!harness.submit_read(addr)) {
                harness.run_cycles(1);
            }
            ++submitted;
        }
    }

    std::cout << "  Submitted " << submitted << " read requests" << std::endl;

    // Run simulation
    if (!harness.run_until_complete(10000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    std::cout << "\n--- Phase 1 Results ---" << std::endl;
    harness.print_stats();

    // Verify: 16 page_empty (first access to each bank) + rest are page_hits
    const auto& stats = harness.stats();

    bool pass = true;

    // First access to each bank is page_empty, rest are page_hits
    uint64_t expected_page_empty = BANKS_PER_PC;
    uint64_t expected_page_hits = (ACCESSES_PER_BANK - 1) * BANKS_PER_PC;

    std::cout << "\n--- Verification ---" << std::endl;
    std::cout << "  Expected page_empty: " << expected_page_empty << std::endl;
    std::cout << "  Actual page_empty:   " << stats.page_empty << std::endl;
    std::cout << "  Expected page_hits:  " << expected_page_hits << std::endl;
    std::cout << "  Actual page_hits:    " << stats.page_hits << std::endl;

    // Allow some variance due to refresh
    if (stats.page_empty < expected_page_empty) {
        std::cerr << "FAIL: Fewer page_empty than expected" << std::endl;
        pass = false;
    }

    // Page hit rate should be high (close to (ACCESSES-1)/ACCESSES = 87.5%)
    double hit_rate = stats.page_hit_rate() * 100.0;
    std::cout << "  Page hit rate: " << std::fixed << std::setprecision(1)
              << hit_rate << "%" << std::endl;

    if (hit_rate < 80.0) {
        std::cerr << "FAIL: Page hit rate below 80%" << std::endl;
        pass = false;
    }

    if (!harness.verify_no_violations()) {
        pass = false;
    }

    // Calculate bank utilization
    std::cout << "\n--- Bank Utilization ---" << std::endl;
    uint64_t total_cycles = harness.current_cycle();
    [[maybe_unused]] double theoretical_min_cycles =
        (BANKS_PER_PC * PAGE_EMPTY_READ_LATENCY) +  // First access to each bank
        ((ACCESSES_PER_BANK - 1) * BANKS_PER_PC * PAGE_HIT_READ_LATENCY / BANKS_PER_PC);  // Pipelined hits

    std::cout << "  Total cycles: " << total_cycles << std::endl;
    std::cout << "  Requests completed: " << stats.reads << std::endl;
    std::cout << "  Avg cycles per request: " << std::fixed << std::setprecision(2)
              << (double)total_cycles / static_cast<double>(stats.reads) << std::endl;

    // Export trace
    std::string trace_path = make_trace_path("full-bank", "hbm2_all_banks_parallel_trace.json");
    harness.export_trace(trace_path);

    if (pass) {
        std::cout << "\nPASS: HBM2 all banks parallel pattern completed successfully" << std::endl;
        return 0;
    } else {
        std::cout << "\nFAIL: HBM2 all banks parallel pattern had errors" << std::endl;
        return 1;
    }
}
