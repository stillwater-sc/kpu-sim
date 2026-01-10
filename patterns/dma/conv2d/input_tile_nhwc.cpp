// patterns/dma/conv2d/input_tile_nhwc.cpp
//
// Conv2D Input Tile NHWC Pattern
// Input activation tensor tile in NHWC (batch, height, width, channels) format
//
// Conv2D input tiles are loaded as spatial patches including all channels.
// NHWC format stores channels contiguously, making it efficient for:
//   - Loading all channels at once (vectorizable)
//   - Sequential access within a row
//   - Spatial locality within tile
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

    std::cout << "=== DMA Conv2D Input Tile NHWC Pattern ===" << std::endl;
    std::cout << "Input activation tile in NHWC format for convolution" << std::endl;
    std::cout << "Tests spatial+channel access patterns" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // Input tensor configuration in NHWC format
    // Typical ResNet/VGG layer: 56×56×256 (after first few convs)
    constexpr size_t BATCH = 1;
    constexpr size_t HEIGHT = 56;
    constexpr size_t WIDTH = 56;
    constexpr size_t CHANNELS = 256;
    constexpr size_t ELEM_SIZE = SIZEOF_FLOAT;  // FP32
    constexpr uint64_t INPUT_BASE = 0x0;

    // NHWC layout: row = H×W×C, treating as 2D matrix for DMA
    // Each spatial row (width × channels) is contiguous
    size_t row_channels = WIDTH * CHANNELS;  // 56 * 256 = 14336 elements per spatial row
    size_t pitch_bytes = row_channels * ELEM_SIZE;  // 57344 bytes (~56KB per row)

    auto input_layout = MatrixLayout::pitched(
        HEIGHT,             // rows = height
        row_channels,       // cols = width * channels
        pitch_bytes,        // pitch = exactly width * channels * elem_size
        ELEM_SIZE,
        INPUT_BASE
    );
    harness.set_matrix_layout(input_layout);

    // Tile configuration
    // For 3×3 conv with stride 1, we need (tile_h + 2) × (tile_w + 2) input pixels
    // Load 8×8 spatial tile (including halo for 3×3 kernel)
    constexpr size_t TILE_H = 8;  // Spatial rows
    constexpr size_t TILE_W = 8;  // Spatial columns
    constexpr size_t TILE_START_H = 16;
    constexpr size_t TILE_START_W = 16;

    // In NHWC, each spatial position has CHANNELS elements
    size_t tile_row_elems = TILE_W * CHANNELS;  // 8 * 256 = 2048 elements
    size_t tile_row_bytes = tile_row_elems * ELEM_SIZE;  // 8192 bytes (8KB per row)

    std::cout << "\n--- Input Tensor Configuration (NHWC) ---" << std::endl;
    std::cout << "Shape: [" << BATCH << ", " << HEIGHT << ", " << WIDTH << ", "
              << CHANNELS << "]" << std::endl;
    std::cout << "Element size: " << ELEM_SIZE << " bytes (float)" << std::endl;
    std::cout << "Total tensor size: " << (HEIGHT * WIDTH * CHANNELS * ELEM_SIZE / 1024 / 1024)
              << " MB" << std::endl;

    std::cout << "\n--- NHWC Memory Layout ---" << std::endl;
    std::cout << "Layout: channels are innermost (contiguous)" << std::endl;
    std::cout << "Row pitch (H stride): " << pitch_bytes << " bytes ("
              << (pitch_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Spatial row = " << WIDTH << " positions × " << CHANNELS
              << " channels = " << row_channels << " elements" << std::endl;

    std::cout << "\n--- Spatial Tile Configuration ---" << std::endl;
    std::cout << "Tile spatial size: " << TILE_H << " × " << TILE_W << " pixels" << std::endl;
    std::cout << "Tile position: H=" << TILE_START_H << ", W=" << TILE_START_W << std::endl;
    std::cout << "Tile elements per row: " << tile_row_elems << std::endl;
    std::cout << "Tile bytes per row: " << tile_row_bytes << " bytes ("
              << (tile_row_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Total tile bytes: " << (TILE_H * tile_row_bytes / 1024) << " KB" << std::endl;

    // Calculate starting column in our MatrixLayout representation
    size_t tile_start_col = TILE_START_W * CHANNELS;  // Convert spatial W to element column

    // Page analysis
    std::cout << "\n--- Page Behavior Analysis ---" << std::endl;
    std::cout << "Page size: " << PAGE_SIZE << " bytes" << std::endl;

    // Each spatial row is ~56KB, so each row is definitely a new page
    // But within a tile row (8KB), we're mostly contiguous
    size_t pages_per_tile_row = (tile_row_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    std::cout << "Pages per tile row: " << pages_per_tile_row << std::endl;
    std::cout << "Total pages for tile: ~" << (pages_per_tile_row * TILE_H) << std::endl;

    // Estimate conflicts based on the full pitch
    size_t estimated_conflicts = input_layout.estimate_page_conflicts(TILE_H);
    std::cout << "Estimated page conflicts: " << estimated_conflicts << std::endl;

    // NHWC advantage: channels are contiguous
    std::cout << "\n--- NHWC Format Advantages ---" << std::endl;
    std::cout << "1. All " << CHANNELS << " channels contiguous per spatial position" << std::endl;
    std::cout << "2. Good vectorization for channel-wise operations" << std::endl;
    std::cout << "3. Tile row access is sequential in memory" << std::endl;
    std::cout << "4. Cache-friendly for conv kernel computation" << std::endl;

    // Submit tile read - we read TILE_H rows, each with tile_row_elems elements
    std::cout << "\n--- Submitting input tile transfer ---" << std::endl;
    size_t submitted = harness.submit_tile_read(TILE_START_H, tile_start_col, TILE_H, tile_row_elems);
    std::cout << "Transfers submitted: " << submitted << std::endl;

    // Run simulation
    bool success = harness.run_until_complete(200000);

    if (!success) {
        std::cerr << "ERROR: Simulation did not complete" << std::endl;
        return 1;
    }

    // Print statistics
    harness.print_stats(3.2);

    auto stats = harness.stats();

    // Analysis
    std::cout << "\n=== Conv2D Input Tile Analysis ===" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;

    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    std::cout << "Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;

    // Bytes efficiency
    size_t tile_total_bytes = TILE_H * tile_row_bytes;
    double bytes_per_cycle = static_cast<double>(tile_total_bytes) / stats.total_cycles;
    std::cout << "Bytes per cycle: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << std::endl;

    // Key insight
    std::cout << "\n--- Key Insight ---" << std::endl;
    std::cout << "Conv2D input tiles in NHWC format have:" << std::endl;
    std::cout << "  + Good sequential access within tile rows" << std::endl;
    std::cout << "  + All channels per spatial position contiguous" << std::endl;
    std::cout << "  - Large pitch between spatial rows (causes page conflicts)" << std::endl;
    std::cout << "  - Each spatial row may span multiple pages" << std::endl;

    std::cout << "\nOptimization strategies:" << std::endl;
    std::cout << "  1. Load larger spatial tiles to amortize overhead" << std::endl;
    std::cout << "  2. Use multi-channel DMA for parallel row fetching" << std::endl;
    std::cout << "  3. Prefetch next tile while computing current" << std::endl;
    std::cout << "  4. Consider tiled memory layout for large tensors" << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("conv2d/input_tile_nhwc_trace.json", 3.2);
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
