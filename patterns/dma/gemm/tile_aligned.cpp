// patterns/dma/gemm/tile_aligned.cpp
//
// GEMM Tile Aligned Pattern
// Best case: Tile width matches cache-aligned pitch
// Matrix: 4096x4096 float (64KB rows), Tile: 64x64
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

    std::cout << "=== DMA GEMM Tile Aligned Pattern ===" << std::endl;
    std::cout << "Best case: 64x64 tile from cache-aligned matrix" << std::endl;
    std::cout << "Tests maximum page hit rate for tile loading" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // Matrix configuration
    // 4096x4096 float matrix with cache-aligned rows
    constexpr size_t MATRIX_ROWS = 4096;
    constexpr size_t MATRIX_COLS = 4096;
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr uint64_t MATRIX_BASE = 0x0;

    // Create cache-aligned layout (64-byte alignment)
    auto matrix = MatrixLayout::cache_aligned(MATRIX_ROWS, MATRIX_COLS, ELEM_SIZE,
                                               CACHE_LINE_BYTES, MATRIX_BASE);

    harness.set_matrix_layout(matrix);

    // Tile configuration
    constexpr size_t TILE_SIZE = 64;  // 64x64 tile
    constexpr size_t TILE_ROW = 128;  // Start at row 128
    constexpr size_t TILE_COL = 256;  // Start at column 256

    std::cout << "\n--- Matrix Configuration ---" << std::endl;
    std::cout << "Matrix size: " << MATRIX_ROWS << " x " << MATRIX_COLS << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes" << std::endl;
    std::cout << "Row bytes: " << matrix.row_bytes() << std::endl;
    std::cout << "Pitch bytes: " << matrix.pitch_bytes << std::endl;
    std::cout << "Contiguous: " << (matrix.is_contiguous() ? "yes" : "no") << std::endl;

    std::cout << "\n--- Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
    std::cout << "Tile position: (" << TILE_ROW << ", " << TILE_COL << ")" << std::endl;
    std::cout << "Tile bytes per row: " << (TILE_SIZE * ELEM_SIZE) << std::endl;
    std::cout << "Total tile bytes: " << (TILE_SIZE * TILE_SIZE * ELEM_SIZE) << std::endl;

    // Page analysis
    std::cout << "\n--- Page Analysis ---" << std::endl;
    size_t estimated_conflicts = matrix.estimate_page_conflicts(TILE_SIZE);
    double estimated_hit_rate = matrix.page_hit_rate(TILE_SIZE);
    std::cout << "Estimated page conflicts: " << estimated_conflicts << std::endl;
    std::cout << "Estimated page hit rate: " << std::fixed << std::setprecision(1)
              << (estimated_hit_rate * 100.0) << "%" << std::endl;

    // Submit tile read
    std::cout << "\n--- Submitting tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_ROW, TILE_COL, TILE_SIZE, TILE_SIZE);
    std::cout << "Transfers submitted: " << submitted << " (one per row)" << std::endl;

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
    std::cout << "\n=== Tile Alignment Analysis ===" << std::endl;

    // With cache-aligned pitch, consecutive tile rows should hit same page
    // until page boundary is crossed
    std::cout << "Actual page conflicts: " << stats.page_conflicts << std::endl;
    std::cout << "Actual page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    // Compare to theoretical
    std::cout << "\n--- Theoretical vs Actual ---" << std::endl;
    std::cout << "Estimated conflicts: " << estimated_conflicts << std::endl;
    std::cout << "Actual conflicts: " << stats.page_conflicts << std::endl;

    // Calculate bandwidth efficiency
    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    std::cout << "\nAchieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;

    // Key insight
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "Cache-aligned pitch allows multiple tile rows to" << std::endl;
    std::cout << "share the same memory page, maximizing page hits." << std::endl;
    std::cout << "This is the BEST CASE for tile loading performance." << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("gemm/tile_aligned_trace.json", 3.2);
    }

    // Verification
    bool pass = true;

    // Page hit rate should be high for aligned access
    if (stats.page_hit_rate() < 0.5) {
        std::cerr << "WARNING: Page hit rate lower than expected for aligned tiles" << std::endl;
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
