// patterns/memory/gddr6/complex/random.cpp
//
// GDDR6 Random Access Pattern
// Tests random bank/row access with worst-case locality
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <random>
#include "../common/gddr6_harness.hpp"
#include "../common/multi_fidelity.hpp"
#include "../common/workloads.hpp"

using namespace sw::kpu::patterns::gddr6;

int main(int argc, char* argv[]) {
    bool run_fidelity = false;
    bool export_trace = true;
    uint32_t seed = 12345;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        }
    }

    std::cout << "=== GDDR6 Random Access Pattern ===" << std::endl;
    std::cout << "Random bank/row access (worst-case locality)" << std::endl;
    std::cout << "Seed: " << seed << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint32_t NUM_ACCESSES = 16;

    std::cout << "\nSubmitting " << NUM_ACCESSES << " random reads" << std::endl;

    // Generate random accesses
    std::minstd_rand rng(seed);
    for (uint32_t i = 0; i < NUM_ACCESSES; ++i) {
        uint8_t channel = rng() % 2;
        uint8_t bank = rng() % 16;
        uint32_t row = rng() % 1024;
        harness.submit_read(make_address(channel, bank, row, 0));
    }

    bool success = harness.run_until_complete(50000);
    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        harness.print_violations();
        return 1;
    }

    harness.print_stats();
    harness.print_calibration_data();

    if (!harness.verify_no_violations()) {
        return 1;
    }

    const auto& stats = harness.stats();

    // Random access statistics are probabilistic
    uint64_t total = stats.page_hits + stats.page_empty + stats.page_conflicts;

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Total accesses: " << total << std::endl;

    bool pass = (total == NUM_ACCESSES);
    if (!pass) {
        std::cerr << "FAIL: Total accesses mismatch" << std::endl;
    }

    // Calculate hit rates
    double hit_rate = 100.0 * stats.page_hits / NUM_ACCESSES;
    double conflict_rate = 100.0 * stats.page_conflicts / NUM_ACCESSES;
    double empty_rate = 100.0 * stats.page_empty / NUM_ACCESSES;

    std::cout << "\nRandom Access Analysis:" << std::endl;
    std::cout << "  Page hits: " << stats.page_hits
              << " (" << std::fixed << std::setprecision(1) << hit_rate << "%)" << std::endl;
    std::cout << "  Page empty: " << stats.page_empty
              << " (" << empty_rate << "%)" << std::endl;
    std::cout << "  Page conflicts: " << stats.page_conflicts
              << " (" << conflict_rate << "%)" << std::endl;

    std::cout << "\nThis pattern models:" << std::endl;
    std::cout << "  - Sparse matrix operations" << std::endl;
    std::cout << "  - Graph traversal" << std::endl;
    std::cout << "  - Hash table lookups" << std::endl;
    std::cout << "  - Irregular memory access patterns" << std::endl;

    // Expected behavior: mostly page empty/conflicts, few hits
    std::cout << "\nExpected behavior:" << std::endl;
    std::cout << "  - Low page hit rate (random accesses)" << std::endl;
    std::cout << "  - Bank-level parallelism provides some mitigation" << std::endl;
    std::cout << "  - Dual-channel helps with random address distribution" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("complex", "random_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_random_workload(seed, NUM_ACCESSES);

        std::cout << "\n--- Calibrated ---" << std::endl;
        mf_harness.run_calibrated(workload, true);
    }

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
