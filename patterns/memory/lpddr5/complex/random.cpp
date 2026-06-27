// patterns/memory/lpddr5/complex/random/main.cpp
//
// Pattern: Random memory access patterns
// Tests: Worst-case locality scenarios
//
// Random access is common in:
// - Hash table lookups
// - Pointer chasing
// - Graph traversal
// - Sparse matrix operations
//
// Challenge: No spatial/temporal locality = worst-case page hit ratio
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <random>
#include <algorithm>
#include <vector>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"
#include "../common/workloads.hpp"
#include "../common/multi_fidelity.hpp"

using namespace sw::kpu::patterns::lpddr5;

/// Generate deterministic "random" sequence for reproducibility
std::vector<uint32_t> generate_random_rows(int count, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, 1023);  // 1K rows

    std::vector<uint32_t> rows;
    rows.reserve(count);
    for (int i = 0; i < count; ++i) {
        rows.push_back(dist(gen));
    }
    return rows;
}

/// Test random row access within single bank
/// Worst case for page locality
bool test_random_single_bank() {
    std::cout << "\n=== Test: Random Single Bank ===" << std::endl;
    std::cout << "Configuration: Random rows in bank 0" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;
    auto rows = generate_random_rows(16);

    for (uint32_t row : rows) {
        harness.submit_read(make_address(BANK, row, 0));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // Random rows = mostly page conflicts
    auto& stats = harness.stats();
    std::cout << "Page conflict ratio: " << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(stats.page_conflicts) / static_cast<double>(stats.page_hits + stats.page_empty + stats.page_conflicts))
              << "%" << std::endl;

    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Random single bank works correctly" << std::endl;
    return true;
}

/// Test random access across multiple banks
/// Better than single bank due to bank parallelism
bool test_random_multi_bank() {
    std::cout << "\n=== Test: Random Multi-Bank ===" << std::endl;
    std::cout << "Configuration: Random bank and row selection" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned int> bank_dist(0, 15);
    std::uniform_int_distribution<uint32_t> row_dist(0, 1023);

    for (int i = 0; i < 16; ++i) {
        uint8_t bank = static_cast<uint8_t>(bank_dist(gen));
        uint32_t row = row_dist(gen);
        harness.submit_read(make_address(bank, row, 0));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Random multi-bank works correctly" << std::endl;
    return true;
}

/// Test random with bank grouping optimization
/// Spread across bank groups for better parallelism
bool test_random_optimized_banks() {
    std::cout << "\n=== Test: Random with Bank Group Optimization ===" << std::endl;
    std::cout << "Configuration: Round-robin across bank groups, random rows" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};  // One per group
    auto rows = generate_random_rows(16);

    for (int i = 0; i < 16; ++i) {
        uint8_t bank = banks[i % 4];  // Round-robin groups
        harness.submit_read(make_address(bank, rows[i], 0));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Random with bank group optimization works correctly" << std::endl;
    return true;
}

/// Compare random vs sequential performance
bool test_random_vs_sequential() {
    std::cout << "\n=== Test: Random vs Sequential Comparison ===" << std::endl;

    const uint8_t BANK = 0;

    // Sequential access (best case)
    uint64_t sequential_cycles;
    {
        std::cout << "\n--- Sequential ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        for (int i = 0; i < 16; ++i) {
            harness.submit_read(make_address(BANK, 100, i * 64));
        }

        harness.run_until_complete(20000);
        sequential_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << sequential_cycles << std::endl;
        harness.print_stats();
    }

    // Random access (worst case)
    uint64_t random_cycles;
    {
        std::cout << "\n--- Random ---" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        auto rows = generate_random_rows(16);
        for (uint32_t row : rows) {
            harness.submit_read(make_address(BANK, row, 0));
        }

        harness.run_until_complete(100000);
        random_cycles = harness.current_cycle();
        std::cout << "Total cycles: " << random_cycles << std::endl;
        harness.print_stats();
    }

    std::cout << "\n--- Comparison ---" << std::endl;
    std::cout << "Sequential: " << sequential_cycles << " cycles (baseline)" << std::endl;
    std::cout << "Random:     " << random_cycles << " cycles";
    if (random_cycles > sequential_cycles) {
        std::cout << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(random_cycles - sequential_cycles) / static_cast<double>(sequential_cycles))
                  << "% slower)";
    }
    std::cout << std::endl;

    double slowdown = static_cast<double>(random_cycles) / static_cast<double>(sequential_cycles);
    std::cout << "Random access is " << std::fixed << std::setprecision(1)
              << slowdown << "x slower than sequential" << std::endl;

    return true;
}

/// Test pointer chasing pattern
/// Each access depends on result of previous (serialized)
bool test_pointer_chasing() {
    std::cout << "\n=== Test: Pointer Chasing Pattern ===" << std::endl;
    std::cout << "Configuration: Simulated linked list traversal" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t BANK = 0;

    // Simulate pointer chasing: predetermined "random" sequence
    // Each row points to next row in sequence
    std::vector<uint32_t> chain = {100, 357, 821, 42, 567, 199, 734, 88};

    for (uint32_t row : chain) {
        harness.submit_read(make_address(BANK, row, 0));
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    // All different rows = mostly page conflicts
    auto& stats = harness.stats();
    if (stats.reads != 8) {
        std::cerr << "FAIL: Expected 8 reads" << std::endl;
        return false;
    }

    // Should have 1 page empty + 7 page conflicts
    if (!harness.verify_stats(8, 0, 0, 1, 7)) return false;

    std::cout << "PASS: Pointer chasing works correctly" << std::endl;
    return true;
}

/// Test hash table access pattern
/// Uniform random with potential for collisions
bool test_hash_table_access() {
    std::cout << "\n=== Test: Hash Table Access Pattern ===" << std::endl;
    std::cout << "Configuration: Simulated hash table lookups" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Simulate hash table with 256 buckets across banks
    // Hash determines bank and row
    std::mt19937 gen(12345);
    std::uniform_int_distribution<uint32_t> key_dist(0, 0xFFFFFFFF);

    for (int i = 0; i < 16; ++i) {
        uint32_t key = key_dist(gen);
        uint8_t bank = static_cast<uint8_t>((key >> 8) & 0xF);
        uint32_t row = (key >> 16) & 0x3FF;  // 1K rows
        uint32_t col = (key & 0xFF) * 64;

        harness.submit_read(make_address(bank, row, col));
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 16) {
        std::cerr << "FAIL: Expected 16 reads" << std::endl;
        return false;
    }

    std::cout << "PASS: Hash table access works correctly" << std::endl;
    return true;
}

/// Test sparse matrix access pattern
/// Simulates CSR format sparse matrix access
bool test_sparse_matrix_access() {
    std::cout << "\n=== Test: Sparse Matrix Access ===" << std::endl;
    std::cout << "Configuration: Simulated sparse matrix row access" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Simulate accessing non-zero elements from a sparse matrix
    // Elements are scattered but within a row might have some locality
    struct SparseElement {
        uint8_t bank;
        uint32_t row;
        uint32_t col;
    };

    std::vector<SparseElement> elements = {
        {0, 100, 0}, {0, 100, 128}, {0, 100, 320},  // Cluster in row 100
        {4, 200, 0}, {4, 200, 64},                   // Cluster in row 200
        {8, 150, 256}, {8, 150, 512},               // Cluster in row 150
        {12, 300, 0}, {12, 300, 128}, {12, 300, 384},  // Cluster in row 300
        {0, 400, 0},  // Isolated
        {4, 500, 0},  // Isolated
    };

    for (const auto& elem : elements) {
        harness.submit_read(make_address(elem.bank, elem.row, elem.col));
    }

    if (!harness.run_until_complete(50000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 12) {
        std::cerr << "FAIL: Expected 12 reads" << std::endl;
        return false;
    }

    // Should have good page hit ratio due to clustering
    double hit_ratio = 100.0 * static_cast<double>(stats.page_hits) / static_cast<double>(stats.page_hits + stats.page_empty + stats.page_conflicts);
    std::cout << "Page hit ratio: " << std::fixed << std::setprecision(1)
              << hit_ratio << "%" << std::endl;

    std::cout << "PASS: Sparse matrix access works correctly" << std::endl;
    return true;
}

/// Test random with read/write mix
bool test_random_mixed_rw() {
    std::cout << "\n=== Test: Random Mixed Read/Write ===" << std::endl;
    std::cout << "Configuration: Random access with 50% read, 50% write" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    const uint8_t banks[4] = {0, 4, 8, 12};
    auto rows = generate_random_rows(16);

    for (int i = 0; i < 16; ++i) {
        uint8_t bank = banks[i % 4];
        if (i % 2 == 0) {
            harness.submit_read(make_address(bank, rows[i], 0));
        } else {
            harness.submit_write(make_address(bank, rows[i], 0));
        }
    }

    if (!harness.run_until_complete(100000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        harness.print_violations();
        return false;
    }

    harness.print_stats();

    if (!harness.verify_no_violations()) return false;

    auto& stats = harness.stats();
    if (stats.reads != 8 || stats.writes != 8) {
        std::cerr << "FAIL: Expected 8 reads and 8 writes" << std::endl;
        return false;
    }

    std::cout << "PASS: Random mixed R/W works correctly" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Random Access" << std::endl;
    std::cout << "Tests worst-case locality scenarios" << std::endl;
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
    pass &= test_random_single_bank();
    pass &= test_random_multi_bank();
    pass &= test_random_optimized_banks();
    pass &= test_random_vs_sequential();
    pass &= test_pointer_chasing();
    pass &= test_hash_table_access();
    pass &= test_sparse_matrix_access();
    pass &= test_random_mixed_rw();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;
        LPDDR5Harness harness(single_channel_config());

        const uint8_t BANK = 0;
        auto rows = generate_random_rows(8);

        for (uint32_t row : rows) {
            harness.submit_read(make_address(BANK, row, 0));
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("complex", "random_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << (pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
