// patterns/dma/gemm/a_tile_row_major.cpp
//
// GEMM A Matrix Tile Row-Major Pattern
// A matrix (M×K) in row-major format, extracting M×K_tile tiles
//
// In GEMM C = A × B:
//   A is M×K, accessed in M×K_tile blocks (row-sequential for each tile)
//   B is K×N, accessed in K_tile×N blocks (strided for column-major)
//
// This pattern demonstrates A matrix tile access, which has good locality
// when A is stored in row-major format.
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

    std::cout << "=== DMA GEMM A Matrix Row-Major Tile Pattern ===" << std::endl;
    std::cout << "A matrix tile access for GEMM: C[M×N] = A[M×K] × B[K×N]" << std::endl;
    std::cout << "Row-major A: good sequential access pattern" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // GEMM dimensions
    // Typical GEMM: C[1024×1024] = A[1024×512] × B[512×1024]
    constexpr size_t M = 1024;  // A rows = C rows
    constexpr size_t K = 512;   // A cols = B rows
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr uint64_t A_BASE = 0x0;

    // A matrix: M × K in row-major
    auto A_matrix = MatrixLayout::row_major(M, K, ELEM_SIZE, A_BASE);
    harness.set_matrix_layout(A_matrix);

    // Tile dimensions for systolic array feed
    // Typical tile: 64×64 or 128×128
    constexpr size_t M_TILE = 64;   // Rows of A tile
    constexpr size_t K_TILE = 64;   // Cols of A tile (shared dimension)

    // Load first tile of A at position (0, 0)
    constexpr size_t TILE_M = 0;
    constexpr size_t TILE_K = 0;

    std::cout << "\n--- GEMM Configuration ---" << std::endl;
    std::cout << "GEMM: C[" << M << "×N] = A[" << M << "×" << K << "] × B[" << K << "×N]" << std::endl;
    std::cout << "A matrix: " << M << " × " << K << " (row-major)" << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes (float)" << std::endl;

    std::cout << "\n--- A Matrix Layout ---" << std::endl;
    std::cout << "Total size: " << (A_matrix.total_bytes() / 1024) << " KB" << std::endl;
    std::cout << "Row pitch: " << A_matrix.pitch_bytes << " bytes" << std::endl;
    std::cout << "Contiguous: " << (A_matrix.is_contiguous() ? "yes" : "no") << std::endl;

    std::cout << "\n--- A Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << M_TILE << " × " << K_TILE << std::endl;
    std::cout << "Tile position: M=" << TILE_M << ", K=" << TILE_K << std::endl;
    std::cout << "Tile bytes per row: " << (K_TILE * ELEM_SIZE) << " bytes" << std::endl;
    std::cout << "Total tile bytes: " << (M_TILE * K_TILE * ELEM_SIZE / 1024) << " KB" << std::endl;

    // Number of tiles
    size_t num_m_tiles = (M + M_TILE - 1) / M_TILE;
    size_t num_k_tiles = (K + K_TILE - 1) / K_TILE;
    std::cout << "\nTotal A tiles: " << num_m_tiles << " × " << num_k_tiles
              << " = " << (num_m_tiles * num_k_tiles) << std::endl;

    // Access pattern analysis
    std::cout << "\n--- Row-Major Access Analysis ---" << std::endl;
    std::cout << "For A in row-major format:" << std::endl;
    std::cout << "  - Each tile row is K_TILE contiguous elements" << std::endl;
    std::cout << "  - Rows are separated by K×elem_size bytes (pitch)" << std::endl;

    // Calculate expected page behavior
    [[maybe_unused]] size_t tile_row_bytes = K_TILE * ELEM_SIZE;  // 256 bytes per tile row
    size_t rows_per_page = PAGE_SIZE / A_matrix.pitch_bytes;
    size_t estimated_conflicts = A_matrix.estimate_page_conflicts(M_TILE);

    std::cout << "\nPage behavior:" << std::endl;
    std::cout << "  Pitch: " << A_matrix.pitch_bytes << " bytes" << std::endl;
    std::cout << "  Page size: " << PAGE_SIZE << " bytes" << std::endl;
    std::cout << "  Rows per page: " << (rows_per_page > 0 ? std::to_string(rows_per_page) : "<1")
              << std::endl;
    std::cout << "  Estimated page conflicts: " << estimated_conflicts << std::endl;

    // Row-major advantage: consecutive tile rows may hit same page
    if (A_matrix.pitch_bytes < PAGE_SIZE) {
        std::cout << "\nRow-major advantage: pitch < page size" << std::endl;
        std::cout << "  Multiple tile rows can share a page" << std::endl;
    }

    // Submit tile read
    std::cout << "\n--- Submitting A tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_M, TILE_K, M_TILE, K_TILE);
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
    std::cout << "\n=== A Matrix Tile Access Analysis ===" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;

    // Bandwidth calculation
    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    std::cout << "Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;

    // Key insight
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "A matrix in row-major is GEMM-friendly because:" << std::endl;
    std::cout << "  1. Tile rows are contiguous in memory" << std::endl;
    std::cout << "  2. Sequential access pattern within each row" << std::endl;
    std::cout << "  3. Good cache line utilization" << std::endl;
    std::cout << "  4. DMA can issue long bursts for each tile row" << std::endl;
    std::cout << "\nThis is one half of the GEMM access pattern." << std::endl;
    std::cout << "B matrix (column-major or strided) is the harder case." << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("gemm/a_tile_row_major_trace.json", 3.2);
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
