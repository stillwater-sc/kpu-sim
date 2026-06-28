// patterns/memory/gddr6/complex/tile_load.cpp
//
// GDDR6 Tile Load Pattern
// Tests DMA-style tile fetch with row locality
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include "../common/gddr6_harness.hpp"
#include "../common/multi_fidelity.hpp"
#include "../common/workloads.hpp"

using namespace sw::kpu::patterns::gddr6;

int main(int argc, char* argv[]) {
    bool run_fidelity = false;
    bool export_trace = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fidelity") == 0) {
            run_fidelity = true;
        } else if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    std::cout << "=== GDDR6 Tile Load Pattern ===" << std::endl;
    std::cout << "DMA-style tile fetch with row locality" << std::endl;
    std::cout << "Common pattern in ML/graphics workloads" << std::endl;

    auto config = gddr6_16000_config();
    GDDR6Harness harness(config);

    const uint8_t CHANNEL = 0;
    const uint8_t BANK = 0;
    const uint32_t START_ROW = 0;
    const uint32_t TILE_ROWS = 4;
    const uint32_t COLS_PER_ROW = 4;

    std::cout << "\nSubmitting " << (TILE_ROWS * COLS_PER_ROW) << " reads ("
              << TILE_ROWS << " rows x " << COLS_PER_ROW << " columns)" << std::endl;

    // Load tile: multiple columns from consecutive rows
    for (uint32_t r = 0; r < TILE_ROWS; ++r) {
        for (uint32_t c = 0; c < COLS_PER_ROW; ++c) {
            harness.submit_read(make_address(CHANNEL, BANK, START_ROW + r, c));
        }
    }

    bool success = harness.run_until_complete(30000);
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

    // Expected: 1 page empty (first access), TILE_ROWS-1 page conflicts (row changes),
    //           rest are page hits (same row accesses)
    uint64_t expected_page_empty = 1;
    uint64_t expected_page_conflicts = TILE_ROWS - 1;
    uint64_t expected_page_hits = TILE_ROWS * COLS_PER_ROW - TILE_ROWS;

    std::cout << "=== Verification ===" << std::endl;
    std::cout << "Expected page_empty: " << expected_page_empty
              << ", Actual: " << stats.page_empty << std::endl;
    std::cout << "Expected page_conflicts: " << expected_page_conflicts
              << ", Actual: " << stats.page_conflicts << std::endl;
    std::cout << "Expected page_hits: " << expected_page_hits
              << ", Actual: " << stats.page_hits << std::endl;

    bool pass = true;
    if (stats.page_empty != expected_page_empty) {
        std::cerr << "FAIL: page_empty mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_conflicts != expected_page_conflicts) {
        std::cerr << "FAIL: page_conflicts mismatch" << std::endl;
        pass = false;
    }
    if (stats.page_hits != expected_page_hits) {
        std::cerr << "FAIL: page_hits mismatch" << std::endl;
        pass = false;
    }

    // Calculate page hit rate
    size_t total_accesses = TILE_ROWS * COLS_PER_ROW;
    double page_hit_rate = 100.0 * static_cast<double>(stats.page_hits) / static_cast<double>(total_accesses);

    std::cout << "\nTile Load Analysis:" << std::endl;
    std::cout << "  Tile size: " << TILE_ROWS << " x " << COLS_PER_ROW << std::endl;
    std::cout << "  Page hit rate: " << std::fixed << std::setprecision(1)
              << page_hit_rate << "%" << std::endl;
    std::cout << "  Row changes: " << expected_page_conflicts << std::endl;

    std::cout << "\nThis pattern models:" << std::endl;
    std::cout << "  - Matrix block/tile loading (GEMM)" << std::endl;
    std::cout << "  - Texture tile fetch (graphics)" << std::endl;
    std::cout << "  - DMA block transfers" << std::endl;
    std::cout << "  - Convolution input tile loading (ML)" << std::endl;

    std::cout << "\nOptimization insights:" << std::endl;
    std::cout << "  - More columns per row = higher page hit rate" << std::endl;
    std::cout << "  - Bank parallelism can hide row change latency" << std::endl;
    std::cout << "  - Larger tiles amortize page empty/conflict costs" << std::endl;

    if (export_trace) {
        std::string trace_file = make_trace_path("complex", "tile_load_trace.json");
        harness.export_trace(trace_file, 2.0);
    }

    if (run_fidelity) {
        std::cout << "\n=== Multi-Fidelity Comparison ===" << std::endl;
        MultiFidelityHarness mf_harness(config);
        auto workload = make_tile_load_workload(CHANNEL, BANK, START_ROW, TILE_ROWS);

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
