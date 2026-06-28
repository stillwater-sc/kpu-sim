// patterns/memory/lpddr5/complex/multi_dma.cpp
//
// Pattern: Multi-DMA Concurrent Tile Loads
// Tests: Multiple DMA engines loading tiles simultaneously
//
// In a KPU with 16 or 32 DMA engines, each engine can independently
// load tiles from memory. This pattern simulates the memory access
// behavior when multiple DMA engines are active concurrently.
//
// Tile configurations tested:
// - 4 DMA engines (baseline accelerator)
// - 8 DMA engines (matches LPDDR5 bank count)
// - 16 DMA engines (high parallelism)
// - 32 DMA engines (maximum parallelism)
//
// Each tile is assumed to be a 2D block of data (e.g., for matrix operations)
// with multiple cache line accesses per tile.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>
#include <vector>

#include "../common/lpddr5_configs.hpp"
#include "../common/lpddr5_harness.hpp"

using namespace sw::kpu::patterns::lpddr5;

// Tile configuration (use namespace CACHE_LINE_BYTES = 64)
constexpr int TILE_ROWS = 8;                // 8 rows per tile
constexpr int TILE_COLS_BYTES = 512;        // 512 bytes per row = 8 cache lines
constexpr int CACHE_LINES_PER_TILE = TILE_ROWS * (TILE_COLS_BYTES / 64);  // 64

/// Test with N concurrent DMA engines loading tiles
bool test_concurrent_dma(int num_dmas, int tiles_per_dma) {
    std::cout << "\n=== Test: " << num_dmas << " Concurrent DMA Engines ===" << std::endl;
    std::cout << "Configuration: " << tiles_per_dma << " tiles per DMA, "
              << CACHE_LINES_PER_TILE << " cache lines per tile" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    // Each DMA engine loads tiles from different banks
    // Distribute DMAs across available banks (8 banks for LPDDR5)
    constexpr int NUM_BANKS = 8;

    int total_accesses = 0;

    // Simulate round-robin tile loading across all DMAs
    for (int tile = 0; tile < tiles_per_dma; ++tile) {
        for (int dma = 0; dma < num_dmas; ++dma) {
            // Each DMA uses a specific bank pattern
            uint8_t base_bank = dma % NUM_BANKS;
            uint32_t base_row = 100 + (dma / NUM_BANKS) * 100 + tile;

            // Load one tile (CACHE_LINES_PER_TILE cache lines)
            for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                // Distribute tile lines across 2 banks for parallelism
                uint8_t bank = (base_bank + (line % 2)) % NUM_BANKS;
                uint32_t col = (line / 2) * CACHE_LINE_BYTES;
                harness.submit_read(make_address(bank, base_row, col));
                total_accesses++;
            }
        }
    }

    if (!harness.run_until_complete(2000000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    uint64_t total_bytes = total_accesses * CACHE_LINE_BYTES;
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(harness.current_cycle());

    std::cout << "\n--- " << num_dmas << "-DMA Analysis ---" << std::endl;
    std::cout << "Total tiles: " << (num_dmas * tiles_per_dma) << std::endl;
    std::cout << "Total accesses: " << total_accesses << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: " << num_dmas << "-DMA test completed" << std::endl;
    return true;
}

/// Compare bandwidth scaling across different DMA counts
bool test_dma_scaling() {
    std::cout << "\n=== Test: DMA Count Scaling ===" << std::endl;
    std::cout << "Compares bandwidth with 4, 8, 16, 32 concurrent DMAs" << std::endl;

    struct Result {
        int num_dmas;
        int total_tiles;
        uint64_t total_bytes;
        uint64_t cycles;
        double bytes_per_cycle;
    };

    std::vector<Result> results;
    std::array<int, 4> dma_counts = {4, 8, 16, 32};
    constexpr int TILES_PER_DMA = 4;

    for (int num_dmas : dma_counts) {
        LPDDR5Harness harness(single_channel_config());
        constexpr int NUM_BANKS = 8;

        int total_accesses = 0;
        for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
            for (int dma = 0; dma < num_dmas; ++dma) {
                uint8_t base_bank = dma % NUM_BANKS;
                uint32_t base_row = 100 + (dma / NUM_BANKS) * 100 + tile;

                for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                    uint8_t bank = (base_bank + (line % 2)) % NUM_BANKS;
                    uint32_t col = (line / 2) * CACHE_LINE_BYTES;
                    harness.submit_read(make_address(bank, base_row, col));
                    total_accesses++;
                }
            }
        }
        harness.run_until_complete(2000000);

        uint64_t bytes = total_accesses * CACHE_LINE_BYTES;
        results.push_back({
            num_dmas,
            num_dmas * TILES_PER_DMA,
            bytes,
            harness.current_cycle(),
            static_cast<double>(bytes) / static_cast<double>(harness.current_cycle())
        });
    }

    std::cout << "\n--- DMA Scaling Results ---" << std::endl;
    std::cout << std::setw(8) << "DMAs"
              << std::setw(10) << "Tiles"
              << std::setw(12) << "KB"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    double baseline = results[0].bytes_per_cycle;
    for (const auto& r : results) {
        std::cout << std::setw(8) << r.num_dmas
                  << std::setw(10) << r.total_tiles
                  << std::setw(12) << (r.total_bytes / 1024)
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::setw(11) << std::setprecision(1) << (r.bytes_per_cycle / baseline) << "x"
                  << std::endl;
    }

    std::cout << "\nPASS: DMA scaling test completed" << std::endl;
    return true;
}

/// Test double-buffering with multiple DMAs
/// Half the DMAs read while the other half write
bool test_double_buffer_dma() {
    std::cout << "\n=== Test: Double-Buffer Multi-DMA ===" << std::endl;
    std::cout << "Pattern: 8 DMAs read, 8 DMAs write (ping-pong)" << std::endl;

    LPDDR5Harness harness(single_channel_config());

    constexpr int NUM_READ_DMAS = 4;  // Use 4 for LPDDR5's 8 banks
    constexpr int NUM_WRITE_DMAS = 4;
    constexpr int TILES_PER_DMA = 4;

    // Read DMAs use banks 0-3, write DMAs use banks 4-7
    for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
        // Interleave reads and writes
        for (int dma = 0; dma < NUM_READ_DMAS; ++dma) {
            uint8_t read_bank = static_cast<uint8_t>(dma);           // Banks 0-3
            uint8_t write_bank = static_cast<uint8_t>(4 + dma);      // Banks 4-7
            uint32_t row = 100 + tile;

            for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                uint32_t col = line * CACHE_LINE_BYTES;
                harness.submit_read(make_address(read_bank, row, col));
                harness.submit_write(make_address(write_bank, row, col));
            }
        }
    }

    if (!harness.run_until_complete(1000000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    int total_ops = NUM_READ_DMAS * TILES_PER_DMA * CACHE_LINES_PER_TILE * 2;  // reads + writes
    uint64_t total_bytes = total_ops * CACHE_LINE_BYTES;
    double bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(harness.current_cycle());

    std::cout << "\n--- Double-Buffer Analysis ---" << std::endl;
    std::cout << "Read DMAs: " << NUM_READ_DMAS << ", Write DMAs: " << NUM_WRITE_DMAS << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: Double-buffer multi-DMA completed" << std::endl;
    return true;
}

/// Test tile size impact on bandwidth
/// Larger tiles amortize page misses better
bool test_tile_size_impact() {
    std::cout << "\n=== Test: Tile Size Impact ===" << std::endl;
    std::cout << "Compares 16, 32, 64, 128 cache lines per tile" << std::endl;

    struct Result {
        int lines_per_tile;
        uint64_t total_bytes;
        uint64_t cycles;
        double bytes_per_cycle;
    };

    std::vector<Result> results;
    std::array<int, 4> tile_sizes = {16, 32, 64, 128};
    constexpr int NUM_DMAS = 8;
    constexpr int TILES_PER_DMA = 2;

    for (int lines_per_tile : tile_sizes) {
        LPDDR5Harness harness(single_channel_config());
        constexpr int NUM_BANKS = 8;

        int total_accesses = 0;
        for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
            for (int dma = 0; dma < NUM_DMAS; ++dma) {
                uint8_t bank = dma % NUM_BANKS;
                uint32_t row = 100 + (dma / NUM_BANKS) * 100 + tile;

                for (int line = 0; line < lines_per_tile; ++line) {
                    uint32_t col = line * CACHE_LINE_BYTES;
                    harness.submit_read(make_address(bank, row, col));
                    total_accesses++;
                }
            }
        }
        harness.run_until_complete(1000000);

        uint64_t bytes = total_accesses * CACHE_LINE_BYTES;
        results.push_back({
            lines_per_tile,
            bytes,
            harness.current_cycle(),
            static_cast<double>(bytes) / static_cast<double>(harness.current_cycle())
        });
    }

    std::cout << "\n--- Tile Size Results ---" << std::endl;
    std::cout << std::setw(14) << "Lines/Tile"
              << std::setw(12) << "KB"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(66, '-') << std::endl;

    double baseline = results[0].bytes_per_cycle;
    for (const auto& r : results) {
        std::cout << std::setw(14) << r.lines_per_tile
                  << std::setw(12) << (r.total_bytes / 1024)
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::setw(11) << std::setprecision(1) << (r.bytes_per_cycle / baseline) << "x"
                  << std::endl;
    }

    std::cout << "\nPASS: Tile size impact test completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "LPDDR5 Pattern: Multi-DMA Concurrent Tile Loads" << std::endl;
    std::cout << "Tests multiple DMA engines loading tiles" << std::endl;
    std::cout << "Default tile: " << TILE_ROWS << "x" << TILE_COLS_BYTES
              << " bytes (" << CACHE_LINES_PER_TILE << " cache lines)" << std::endl;
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
    pass &= test_concurrent_dma(4, 4);   // 4 DMAs, 4 tiles each
    pass &= test_concurrent_dma(8, 4);   // 8 DMAs, 4 tiles each
    pass &= test_concurrent_dma(16, 2);  // 16 DMAs, 2 tiles each
    pass &= test_dma_scaling();
    pass &= test_double_buffer_dma();
    pass &= test_tile_size_impact();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;

        // Use larger queue depth for trace export to fit all 8 banks x 16 lines = 128 requests
        auto trace_config = single_channel_config();
        trace_config.queue_depth = 256;
        LPDDR5Harness harness(trace_config);

        // Generate trace for 8 DMAs loading 1 tile each
        constexpr int NUM_DMAS = 8;
        constexpr int NUM_BANKS = 8;
        constexpr int LINES_TO_TRACE = 16;  // Limit for trace size

        for (int dma = 0; dma < NUM_DMAS; ++dma) {
            uint8_t bank = dma % NUM_BANKS;
            uint32_t row = 100 + (dma / NUM_BANKS);

            for (int line = 0; line < LINES_TO_TRACE; ++line) {
                harness.submit_read(make_address(bank, row, line * CACHE_LINE_BYTES));
            }
        }
        harness.run_until_complete();

        if (trace_file.empty()) {
            trace_file = make_trace_path("complex", "multi_dma_trace.json");
        }
        harness.export_trace(trace_file);
    }

    std::cout << "\n================================================" << std::endl;
    if (pass) {
        std::cout << "=== PASS ===" << std::endl;
    } else {
        std::cout << "=== FAIL ===" << std::endl;
    }
    std::cout << "================================================" << std::endl;

    return pass ? 0 : 1;
}
