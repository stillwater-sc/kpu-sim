// patterns/dma/gemm/tile_page_boundary.cpp
//
// GEMM Tile Page Boundary Pattern
// Tile positioned to cross DRAM page boundaries
// Matrix: 4096x4096 (16KB pitch), Tile: 64x64 at strategic offset
//
// This demonstrates what happens when a tile straddles page boundaries,
// causing additional page activations within rows.
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

    std::cout << "=== DMA GEMM Tile Page Boundary Pattern ===" << std::endl;
    std::cout << "Tile positioned to cross DRAM page boundaries" << std::endl;
    std::cout << "Demonstrates intra-row page conflicts" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // Matrix configuration
    // 4096x4096 float matrix
    constexpr size_t MATRIX_ROWS = 4096;
    constexpr size_t MATRIX_COLS = 4096;
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr uint64_t MATRIX_BASE = 0x0;

    auto matrix = MatrixLayout::row_major(MATRIX_ROWS, MATRIX_COLS, ELEM_SIZE, MATRIX_BASE);
    harness.set_matrix_layout(matrix);

    // Tile configuration
    // Position tile to cross a page boundary within each row
    // Page size = 4KB = 4096 bytes
    // Row pitch = 4096 cols * 4 bytes = 16KB
    // We position the tile so that the tile columns span across a 4KB boundary
    constexpr size_t PAGE_SIZE = 4096;
    constexpr size_t TILE_SIZE = 64;  // 64x64 tile

    // Calculate column offset to cross page boundary
    // Tile at col 1008 will span bytes [4032, 4288) within each row (256B tile row)
    // This crosses the 4096-byte page boundary
    constexpr size_t TILE_ROW = 128;
    constexpr size_t TILE_COL = 1008;  // 1008 * 4 = 4032 bytes from row start

    size_t tile_row_bytes = TILE_SIZE * ELEM_SIZE;  // 256 bytes
    size_t tile_start_offset = TILE_COL * ELEM_SIZE;  // 4032 bytes
    size_t tile_end_offset = tile_start_offset + tile_row_bytes;  // 4288 bytes

    std::cout << "\n--- Matrix Configuration ---" << std::endl;
    std::cout << "Matrix size: " << MATRIX_ROWS << " x " << MATRIX_COLS << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes (float)" << std::endl;
    std::cout << "Row pitch: " << matrix.pitch_bytes << " bytes (" << (matrix.pitch_bytes / 1024)
              << " KB)" << std::endl;

    std::cout << "\n--- Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
    std::cout << "Tile position: (" << TILE_ROW << ", " << TILE_COL << ")" << std::endl;
    std::cout << "Tile row bytes: " << tile_row_bytes << " bytes" << std::endl;

    std::cout << "\n--- Page Boundary Analysis ---" << std::endl;
    std::cout << "Page size: " << PAGE_SIZE << " bytes" << std::endl;
    std::cout << "Tile byte offset in row: " << tile_start_offset << " - " << tile_end_offset << std::endl;
    std::cout << "Page boundary at: " << PAGE_SIZE << " bytes" << std::endl;

    bool crosses_boundary = (tile_start_offset < PAGE_SIZE) && (tile_end_offset > PAGE_SIZE);
    std::cout << "Crosses page boundary: " << (crosses_boundary ? "YES" : "NO") << std::endl;

    if (crosses_boundary) {
        size_t bytes_before = PAGE_SIZE - tile_start_offset;
        size_t bytes_after = tile_end_offset - PAGE_SIZE;
        std::cout << "  Bytes before boundary: " << bytes_before << std::endl;
        std::cout << "  Bytes after boundary: " << bytes_after << std::endl;
    }

    // Each row of the tile will cross a page boundary
    // This means each row access may require two page activations
    std::cout << "\nExpected behavior:" << std::endl;
    std::cout << "  - Each tile row access crosses a 4KB page boundary" << std::endl;
    std::cout << "  - May cause additional page activations within each row" << std::endl;
    std::cout << "  - Depends on memory controller's ability to handle split accesses" << std::endl;

    // Submit tile
    std::cout << "\n--- Submitting tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_ROW, TILE_COL, TILE_SIZE, TILE_SIZE);
    std::cout << "Transfers submitted: " << submitted << std::endl;

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
    std::cout << "\n=== Page Boundary Impact Analysis ===" << std::endl;
    std::cout << "Page hits: " << stats.page_hits << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;
    std::cout << "Page empty: " << stats.page_empty << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    // Compare to aligned case (no boundary crossing)
    // In the aligned case, we'd expect better page utilization
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "When tile data crosses page boundaries within a row:" << std::endl;
    std::cout << "  - Single row access may require 2 page activations" << std::endl;
    std::cout << "  - Memory controller may need to split the request" << std::endl;
    std::cout << "  - Overall bandwidth efficiency decreases" << std::endl;
    std::cout << "\nMitigation:" << std::endl;
    std::cout << "  - Align tile column positions to page boundaries" << std::endl;
    std::cout << "  - Use tile sizes that are multiples of page size" << std::endl;
    std::cout << "  - Allocate matrices with page-aligned rows" << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("gemm/tile_page_boundary_trace.json", 3.2);
    }

    // Verification
    bool pass = stats.dma_transfers_completed >= submitted / 2;

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
