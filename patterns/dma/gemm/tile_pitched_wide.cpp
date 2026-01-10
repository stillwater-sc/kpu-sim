// patterns/dma/gemm/tile_pitched_wide.cpp
//
// GEMM Tile Pitched Wide Pattern
// Wide tile from narrow matrix - demonstrates good row locality
// Matrix: 256x256 (1KB pitch), Tile: 64x64
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

    std::cout << "=== DMA GEMM Tile Pitched Wide Pattern ===" << std::endl;
    std::cout << "Good case: Wide tile (64x64) from narrow matrix (256x256)" << std::endl;
    std::cout << "Tile covers significant portion of each row" << std::endl;

    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // Narrow matrix: 256x256 floats = 256KB total, 1KB per row
    constexpr size_t MATRIX_ROWS = 256;
    constexpr size_t MATRIX_COLS = 256;
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr uint64_t MATRIX_BASE = 0x0;

    auto matrix = MatrixLayout::row_major(MATRIX_ROWS, MATRIX_COLS, ELEM_SIZE, MATRIX_BASE);
    harness.set_matrix_layout(matrix);

    // Wide tile: 64x64 covers 25% of matrix width
    constexpr size_t TILE_SIZE = 64;
    constexpr size_t TILE_ROW = 32;
    constexpr size_t TILE_COL = 64;

    size_t tile_row_bytes = TILE_SIZE * ELEM_SIZE;  // 256 bytes

    std::cout << "\n--- Matrix Configuration ---" << std::endl;
    std::cout << "Matrix size: " << MATRIX_ROWS << " x " << MATRIX_COLS << std::endl;
    std::cout << "Pitch: " << matrix.pitch_bytes << " bytes (1 KB)" << std::endl;
    std::cout << "Total matrix: " << (matrix.total_bytes() / 1024) << " KB" << std::endl;

    std::cout << "\n--- Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
    std::cout << "Tile row bytes: " << tile_row_bytes << " bytes" << std::endl;
    std::cout << "Tile coverage: " << std::fixed << std::setprecision(0)
              << (100.0 * TILE_SIZE / MATRIX_COLS) << "% of row width" << std::endl;

    // With 1KB pitch and 4KB pages, 4 rows fit per page
    constexpr size_t PAGE_SIZE = 4096;
    size_t rows_per_page = PAGE_SIZE / matrix.pitch_bytes;
    std::cout << "\nRows per page: " << rows_per_page << std::endl;
    std::cout << "Expected page changes: " << (TILE_SIZE / rows_per_page) << std::endl;

    size_t estimated_conflicts = matrix.estimate_page_conflicts(TILE_SIZE);
    std::cout << "Estimated page conflicts: " << estimated_conflicts << std::endl;

    // Submit tile
    std::cout << "\n--- Submitting tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_ROW, TILE_COL, TILE_SIZE, TILE_SIZE);
    std::cout << "Transfers submitted: " << submitted << std::endl;

    bool success = harness.run_until_complete(100000);

    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        return 1;
    }

    harness.print_stats(3.2);
    auto stats = harness.stats();

    std::cout << "\n=== Wide Tile Analysis ===" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;

    // Key insight
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "With narrow matrices (small pitch):" << std::endl;
    std::cout << "  - Multiple rows fit in a single memory page" << std::endl;
    std::cout << "  - Page hit rate improves significantly" << std::endl;
    std::cout << "  - Less memory bandwidth wasted on row activations" << std::endl;

    if (export_trace) {
        harness.export_trace("gemm/tile_pitched_wide_trace.json", 3.2);
    }

    bool pass = stats.dma_transfers_completed >= submitted / 2;

    std::cout << (pass ? "\n=== PASS ===" : "\n=== FAIL ===") << std::endl;
    return pass ? 0 : 1;
}
