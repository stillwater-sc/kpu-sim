// patterns/dma/gemm/tile_pitched_narrow.cpp
//
// GEMM Tile Pitched Narrow Pattern
// Worst case: Small tile from wide matrix with large pitch
// Matrix: 4096x4096 (16KB pitch), Tile: 32x32
//
// This demonstrates the interplay between matrix pitch and tile size.
// When pitch >> tile_width * elem_size, each tile row causes a page conflict.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include "../common/dma_harness.hpp"

using namespace sw::kpu::patterns::dma;

int main(int argc, char* argv[]) {
    bool export_trace = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-trace") == 0) {
            export_trace = false;
        }
    }

    std::cout << "=== DMA GEMM Tile Pitched Narrow Pattern ===" << std::endl;
    std::cout << "Worst case: 32x32 tile from 4096-wide matrix" << std::endl;
    std::cout << "Demonstrates pitch vs tile size mismatch" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // Matrix configuration
    // 4096x4096 float matrix with full 16KB pitch per row
    constexpr size_t MATRIX_ROWS = 4096;
    constexpr size_t MATRIX_COLS = 4096;
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr size_t PITCH_BYTES = 4096 * SIZEOF_FLOAT;  // 16KB per row
    constexpr uint64_t MATRIX_BASE = 0x0;

    auto matrix = MatrixLayout::pitched(MATRIX_ROWS, MATRIX_COLS, PITCH_BYTES,
                                         ELEM_SIZE, MATRIX_BASE);
    harness.set_matrix_layout(matrix);

    // Tile configuration - NARROW tile from WIDE matrix
    constexpr size_t TILE_HEIGHT = 32;
    constexpr size_t TILE_WIDTH = 32;
    constexpr size_t TILE_ROW = 128;
    constexpr size_t TILE_COL = 256;

    // Calculate tile bytes per row
    size_t tile_row_bytes = TILE_WIDTH * ELEM_SIZE;  // 128 bytes

    std::cout << "\n--- Matrix Configuration ---" << std::endl;
    std::cout << "Matrix size: " << MATRIX_ROWS << " x " << MATRIX_COLS << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes (float)" << std::endl;
    std::cout << "Pitch: " << matrix.pitch_bytes << " bytes (" << (matrix.pitch_bytes / 1024)
              << " KB)" << std::endl;
    std::cout << "Row bytes: " << matrix.row_bytes() << " bytes" << std::endl;

    std::cout << "\n--- Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << TILE_HEIGHT << " x " << TILE_WIDTH << std::endl;
    std::cout << "Tile position: (" << TILE_ROW << ", " << TILE_COL << ")" << std::endl;
    std::cout << "Tile row bytes: " << tile_row_bytes << " bytes" << std::endl;
    std::cout << "Total tile bytes: " << (TILE_HEIGHT * tile_row_bytes) << " bytes" << std::endl;

    // Key insight: pitch vs tile width
    std::cout << "\n--- Pitch vs Tile Analysis ---" << std::endl;
    std::cout << "Matrix pitch: " << matrix.pitch_bytes << " bytes" << std::endl;
    std::cout << "Tile row size: " << tile_row_bytes << " bytes" << std::endl;
    std::cout << "Pitch / Tile row: " << (matrix.pitch_bytes / tile_row_bytes) << "x" << std::endl;

    // Page analysis
    // With 16KB pitch and 4KB pages, every row is a new page
    constexpr size_t PAGE_SIZE = 4096;
    size_t rows_per_page = PAGE_SIZE / matrix.pitch_bytes;
    std::cout << "\nRows per page: " << rows_per_page
              << (rows_per_page == 0 ? " (less than 1 - conflict every row!)" : "") << std::endl;

    size_t estimated_conflicts = matrix.estimate_page_conflicts(TILE_HEIGHT);
    double estimated_hit_rate = matrix.page_hit_rate(TILE_HEIGHT);
    std::cout << "Estimated page conflicts: " << estimated_conflicts << std::endl;
    std::cout << "Estimated page hit rate: " << std::fixed << std::setprecision(1)
              << (estimated_hit_rate * 100.0) << "%" << std::endl;

    // Submit tile read
    std::cout << "\n--- Submitting tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_ROW, TILE_COL, TILE_HEIGHT, TILE_WIDTH);
    std::cout << "Transfers submitted: " << submitted << " (one per tile row)" << std::endl;

    // Run simulation
    bool success = harness.run_until_complete(100000);

    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        return 1;
    }

    // Print statistics
    harness.print_stats(3.2);

    auto stats = harness.stats();

    // Analysis
    std::cout << "\n=== Pitch Mismatch Analysis ===" << std::endl;

    std::cout << "Actual page conflicts: " << stats.page_conflicts << std::endl;
    std::cout << "Actual page empty: " << stats.page_empty << std::endl;
    std::cout << "Actual page hits: " << stats.page_hits << std::endl;
    std::cout << "Actual page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    // Calculate bandwidth impact
    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    std::cout << "\nAchieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;

    // Memory efficiency
    // We're only transferring tile_row_bytes out of pitch_bytes per row
    double data_efficiency = 100.0 * static_cast<double>(tile_row_bytes) / static_cast<double>(matrix.pitch_bytes);
    std::cout << "Data efficiency per row: " << std::fixed << std::setprecision(1)
              << data_efficiency << "% (tile vs pitch)" << std::endl;

    // Key insight
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "When matrix pitch >> tile row size:" << std::endl;
    std::cout << "  - Every tile row accesses a different memory page" << std::endl;
    std::cout << "  - Page conflicts occur on nearly every row" << std::endl;
    std::cout << "  - Memory bandwidth is wasted due to row activations" << std::endl;
    std::cout << "\nThis is the WORST CASE for tile loading." << std::endl;
    std::cout << "\nMitigation strategies:" << std::endl;
    std::cout << "  1. Use wider tiles to amortize page open cost" << std::endl;
    std::cout << "  2. Allocate matrices with smaller pitch (tighter packing)" << std::endl;
    std::cout << "  3. Use bank-parallel access (multiple banks)" << std::endl;
    std::cout << "  4. Prefetch or overlap tile loads" << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("gemm/tile_pitched_narrow_trace.json", 3.2);
    }

    // Verification
    bool pass = true;

    // For this worst-case pattern, we expect LOW page hit rate
    if (stats.page_hit_rate() > 0.5) {
        std::cout << "INFO: Page hit rate higher than expected for narrow tiles" << std::endl;
    }

    // Check that we observed page conflicts
    if (stats.page_conflicts < TILE_HEIGHT / 2) {
        std::cout << "INFO: Fewer page conflicts than expected" << std::endl;
    }

    if (stats.dma_transfers_completed < submitted / 2) {
        std::cerr << "FAIL: Not enough transfers completed" << std::endl;
        pass = false;
    }

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
