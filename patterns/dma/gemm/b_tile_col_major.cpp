// patterns/dma/gemm/b_tile_col_major.cpp
//
// GEMM B Matrix Column-Major (Strided) Tile Pattern
// B matrix (K×N) accessed with strided pattern for column tiles
//
// In GEMM C = A × B:
//   A is M×K (row-major, good locality)
//   B is K×N, but we need K_tile×N_tile blocks
//   If B is row-major, accessing columns requires strided access
//
// This pattern demonstrates the challenging B matrix tile access pattern
// which has poor memory locality when B is stored in row-major format.
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

    std::cout << "=== DMA GEMM B Matrix Strided Tile Pattern ===" << std::endl;
    std::cout << "B matrix tile access for GEMM: C[M×N] = A[M×K] × B[K×N]" << std::endl;
    std::cout << "Row-major B: strided access pattern (poor locality)" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // GEMM dimensions
    // Typical GEMM: C[1024×1024] = A[1024×512] × B[512×1024]
    constexpr size_t K = 512;    // B rows = A cols (shared dimension)
    constexpr size_t N = 1024;   // B cols = C cols
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;
    constexpr uint64_t B_BASE = 0x10000000;  // Offset from A

    // B matrix: K × N in row-major
    // For efficient GEMM, we often want to access columns, not rows!
    auto B_matrix = MatrixLayout::row_major(K, N, ELEM_SIZE, B_BASE);
    harness.set_matrix_layout(B_matrix);

    // Tile dimensions
    // For systolic array: K_tile elements along K dimension, N_tile along N
    constexpr size_t K_TILE = 64;   // Rows of B tile
    constexpr size_t N_TILE = 64;   // Cols of B tile

    // Load first tile of B at position (0, 0)
    constexpr size_t TILE_K = 0;
    constexpr size_t TILE_N = 0;

    std::cout << "\n--- GEMM Configuration ---" << std::endl;
    std::cout << "GEMM: C[M×" << N << "] = A[M×" << K << "] × B[" << K << "×" << N << "]" << std::endl;
    std::cout << "B matrix: " << K << " × " << N << " (row-major)" << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes (float)" << std::endl;

    std::cout << "\n--- B Matrix Layout ---" << std::endl;
    std::cout << "Total size: " << (B_matrix.total_bytes() / 1024) << " KB" << std::endl;
    std::cout << "Row pitch: " << B_matrix.pitch_bytes << " bytes (" << (B_matrix.pitch_bytes / 1024)
              << " KB)" << std::endl;
    std::cout << "Contiguous: " << (B_matrix.is_contiguous() ? "yes" : "no") << std::endl;

    std::cout << "\n--- B Tile Configuration ---" << std::endl;
    std::cout << "Tile size: " << K_TILE << " × " << N_TILE << std::endl;
    std::cout << "Tile position: K=" << TILE_K << ", N=" << TILE_N << std::endl;
    std::cout << "Tile bytes per row: " << (N_TILE * ELEM_SIZE) << " bytes" << std::endl;
    std::cout << "Total tile bytes: " << (K_TILE * N_TILE * ELEM_SIZE / 1024) << " KB" << std::endl;

    // Number of tiles
    size_t num_k_tiles = (K + K_TILE - 1) / K_TILE;
    size_t num_n_tiles = (N + N_TILE - 1) / N_TILE;
    std::cout << "\nTotal B tiles: " << num_k_tiles << " × " << num_n_tiles
              << " = " << (num_k_tiles * num_n_tiles) << std::endl;

    // Strided access analysis - THE KEY ISSUE
    std::cout << "\n--- Strided Access Analysis (B Matrix Problem) ---" << std::endl;
    std::cout << "For row-major B accessed as K_tile × N_tile:" << std::endl;
    std::cout << "  - Each tile row is N_TILE contiguous elements (" << (N_TILE * ELEM_SIZE)
              << " bytes)" << std::endl;
    std::cout << "  - Rows are separated by N×elem_size = " << B_matrix.pitch_bytes
              << " bytes" << std::endl;

    // Calculate the striding problem
    size_t tile_row_bytes = N_TILE * ELEM_SIZE;  // 256 bytes
    double tile_row_fraction = static_cast<double>(tile_row_bytes) / static_cast<double>(B_matrix.pitch_bytes);

    std::cout << "\nTile row coverage: " << std::fixed << std::setprecision(1)
              << (tile_row_fraction * 100.0) << "% of matrix row" << std::endl;
    std::cout << "Gap between tile rows: " << (B_matrix.pitch_bytes - tile_row_bytes)
              << " bytes" << std::endl;

    // Page analysis for strided access
    std::cout << "\n--- Page Behavior for Strided B Access ---" << std::endl;
    size_t estimated_conflicts = B_matrix.estimate_page_conflicts(K_TILE);
    double estimated_hit_rate = B_matrix.page_hit_rate(K_TILE);

    std::cout << "Estimated page conflicts: " << estimated_conflicts << std::endl;
    std::cout << "Estimated page hit rate: " << std::fixed << std::setprecision(1)
              << (estimated_hit_rate * 100.0) << "%" << std::endl;

    // With 4KB pitch (N=1024, 4B elements), each row is a new page
    if (B_matrix.pitch_bytes >= PAGE_SIZE) {
        std::cout << "\nWARNING: Row pitch >= page size!" << std::endl;
        std::cout << "  Each tile row access will cause a page conflict" << std::endl;
        std::cout << "  This is the WORST CASE for B matrix access" << std::endl;
    }

    // Submit tile read
    std::cout << "\n--- Submitting B tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_K, TILE_N, K_TILE, N_TILE);
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
    std::cout << "\n=== B Matrix Strided Access Analysis ===" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;

    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    std::cout << "Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;

    // Compare to A matrix (expected better)
    std::cout << "\n--- Comparison to A Matrix Access ---" << std::endl;
    std::cout << "B matrix (row-major, accessed by columns) typically has:" << std::endl;
    std::cout << "  - Lower page hit rate than A matrix" << std::endl;
    std::cout << "  - More page conflicts per tile" << std::endl;
    std::cout << "  - Lower effective bandwidth" << std::endl;

    // Key insight and solutions
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "B matrix access pattern is the BOTTLENECK in GEMM because:" << std::endl;
    std::cout << "  1. Large stride between accessed elements" << std::endl;
    std::cout << "  2. Each tile row may hit a different page" << std::endl;
    std::cout << "  3. Poor cache line utilization (only use " << (N_TILE * ELEM_SIZE)
              << " of " << B_matrix.pitch_bytes << " bytes)" << std::endl;

    std::cout << "\nMitigation strategies:" << std::endl;
    std::cout << "  1. Store B in column-major (B^T) format" << std::endl;
    std::cout << "  2. Use tiled/blocked matrix layout (space-filling curve)" << std::endl;
    std::cout << "  3. Prefetch B tiles while computing with A tiles" << std::endl;
    std::cout << "  4. Use larger N_TILE to amortize page open cost" << std::endl;
    std::cout << "  5. Bank interleaving across multiple memory channels" << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("gemm/b_tile_col_major_trace.json", 3.2);
    }

    // Verification
    bool pass = stats.dma_transfers_completed >= submitted / 2;

    // Note: We expect poor page hit rate for this pattern!
    if (stats.page_hit_rate() < 0.3) {
        std::cout << "\nINFO: Low page hit rate confirms strided access pattern" << std::endl;
    }

    if (pass) {
        std::cout << "\n=== PASS ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== FAIL ===" << std::endl;
        return 1;
    }
}
