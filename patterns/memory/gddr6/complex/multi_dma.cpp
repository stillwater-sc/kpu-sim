// patterns/memory/gddr6/complex/multi_dma.cpp
//
// Pattern: Multi-DMA Concurrent Tile Loads
// Tests: Multiple DMA engines loading tiles simultaneously
//
// GDDR6 with 16 banks enables more concurrent DMA operations than LPDDR5.
// This pattern simulates 16-32 DMA engines loading tiles concurrently.
//
// Tile configurations tested:
// - 4 DMA engines (baseline)
// - 8 DMA engines (LPDDR5-equivalent)
// - 16 DMA engines (matches GDDR6 bank count)
// - 32 DMA engines (2x bank count for maximum pressure)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>
#include <vector>

#include "../common/gddr6_harness.hpp"

using namespace sw::kpu::patterns::gddr6;

// Tile configuration (use namespace CACHE_LINE_BYTES)
constexpr int TILE_ROWS = 8;                // 8 rows per tile
constexpr int TILE_COLS_BYTES = 512;        // 512 bytes per row = 8 cache lines
constexpr int CACHE_LINES_PER_TILE = TILE_ROWS * (TILE_COLS_BYTES / 64);  // 64
constexpr int NUM_BANKS = 16;               // GDDR6 has 16 banks

// Helper to create address on channel 0
inline uint64_t addr(uint8_t bank, uint32_t row, uint32_t col) {
    return make_address(0, bank, row, col);
}

/// Test with N concurrent DMA engines loading tiles
bool test_concurrent_dma(int num_dmas, int tiles_per_dma) {
    std::cout << "\n=== Test: " << num_dmas << " Concurrent DMA Engines ===" << std::endl;
    std::cout << "Configuration: " << tiles_per_dma << " tiles per DMA, "
              << CACHE_LINES_PER_TILE << " cache lines per tile" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

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
                harness.submit_read(addr(bank, base_row, col));
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
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

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
    std::cout << "\n=== Test: DMA Count Scaling (GDDR6) ===" << std::endl;
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
        GDDR6Harness harness(gddr6_16000_config());

        int total_accesses = 0;
        for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
            for (int dma = 0; dma < num_dmas; ++dma) {
                uint8_t base_bank = dma % NUM_BANKS;
                uint32_t base_row = 100 + (dma / NUM_BANKS) * 100 + tile;

                for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                    uint8_t bank = (base_bank + (line % 2)) % NUM_BANKS;
                    uint32_t col = (line / 2) * CACHE_LINE_BYTES;
                    harness.submit_read(addr(bank, base_row, col));
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
            static_cast<double>(bytes) / harness.current_cycle()
        });
    }

    std::cout << "\n--- DMA Scaling Results (GDDR6) ---" << std::endl;
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

/// Test double-buffering with multiple DMAs using GDDR6's 16 banks
bool test_double_buffer_dma() {
    std::cout << "\n=== Test: Double-Buffer Multi-DMA (GDDR6) ===" << std::endl;
    std::cout << "Pattern: 8 DMAs read (banks 0-7), 8 DMAs write (banks 8-15)" << std::endl;

    GDDR6Harness harness(gddr6_16000_config());

    constexpr int NUM_READ_DMAS = 8;
    constexpr int NUM_WRITE_DMAS = 8;
    constexpr int TILES_PER_DMA = 4;

    // Read DMAs use banks 0-7, write DMAs use banks 8-15
    for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
        for (int dma = 0; dma < NUM_READ_DMAS; ++dma) {
            uint8_t read_bank = static_cast<uint8_t>(dma);           // Banks 0-7
            uint8_t write_bank = static_cast<uint8_t>(8 + dma);      // Banks 8-15
            uint32_t row = 100 + tile;

            for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                uint32_t col = line * CACHE_LINE_BYTES;
                harness.submit_read(addr(read_bank, row, col));
                harness.submit_write(addr(write_bank, row, col));
            }
        }
    }

    if (!harness.run_until_complete(2000000)) {
        std::cerr << "FAIL: Simulation did not complete" << std::endl;
        return false;
    }

    harness.print_stats();

    int total_ops = NUM_READ_DMAS * TILES_PER_DMA * CACHE_LINES_PER_TILE * 2;
    uint64_t total_bytes = total_ops * CACHE_LINE_BYTES;
    double bytes_per_cycle = static_cast<double>(total_bytes) / harness.current_cycle();

    std::cout << "\n--- Double-Buffer Analysis ---" << std::endl;
    std::cout << "Read DMAs: " << NUM_READ_DMAS << ", Write DMAs: " << NUM_WRITE_DMAS << std::endl;
    std::cout << "Total bytes: " << total_bytes << " (" << (total_bytes / 1024) << " KB)" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << bytes_per_cycle << " bytes/cycle" << std::endl;

    std::cout << "PASS: Double-buffer multi-DMA completed" << std::endl;
    return true;
}

/// Compare GDDR6 (16 banks) vs LPDDR5-equivalent (8 banks) DMA performance
bool test_gddr6_vs_lpddr5_dma() {
    std::cout << "\n=== Test: GDDR6 vs LPDDR5-equivalent DMA ===" << std::endl;
    std::cout << "Compares 8-bank and 16-bank DMA configurations" << std::endl;

    struct Result {
        const char* config;
        int banks_used;
        uint64_t total_bytes;
        uint64_t cycles;
        double bytes_per_cycle;
    };

    std::array<Result, 2> results;
    constexpr int NUM_DMAS = 16;
    constexpr int TILES_PER_DMA = 2;

    // LPDDR5-equivalent: only use 8 banks
    {
        GDDR6Harness harness(gddr6_16000_config());
        int total_accesses = 0;

        for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
            for (int dma = 0; dma < NUM_DMAS; ++dma) {
                uint8_t bank = dma % 8;  // Only 8 banks
                uint32_t row = 100 + (dma / 8) * 100 + tile;

                for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                    harness.submit_read(addr(bank, row, line * CACHE_LINE_BYTES));
                    total_accesses++;
                }
            }
        }
        harness.run_until_complete(2000000);

        uint64_t bytes = total_accesses * CACHE_LINE_BYTES;
        results[0] = {"LPDDR5-equiv", 8, bytes, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    // GDDR6: use all 16 banks
    {
        GDDR6Harness harness(gddr6_16000_config());
        int total_accesses = 0;

        for (int tile = 0; tile < TILES_PER_DMA; ++tile) {
            for (int dma = 0; dma < NUM_DMAS; ++dma) {
                uint8_t bank = dma % 16;  // All 16 banks
                uint32_t row = 100 + (dma / 16) * 100 + tile;

                for (int line = 0; line < CACHE_LINES_PER_TILE; ++line) {
                    harness.submit_read(addr(bank, row, line * CACHE_LINE_BYTES));
                    total_accesses++;
                }
            }
        }
        harness.run_until_complete(2000000);

        uint64_t bytes = total_accesses * CACHE_LINE_BYTES;
        results[1] = {"GDDR6-full", 16, bytes, harness.current_cycle(),
                      static_cast<double>(bytes) / harness.current_cycle()};
    }

    std::cout << "\n--- GDDR6 vs LPDDR5 DMA Results ---" << std::endl;
    std::cout << std::setw(14) << "Config"
              << std::setw(10) << "Banks"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Bytes/Cycle" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(14) << r.config
                  << std::setw(10) << r.banks_used
                  << std::setw(12) << r.cycles
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.bytes_per_cycle
                  << std::endl;
    }

    double speedup = results[1].bytes_per_cycle / results[0].bytes_per_cycle;
    std::cout << "\nGDDR6 advantage: " << std::setprecision(1) << speedup << "x" << std::endl;

    std::cout << "PASS: GDDR6 vs LPDDR5 comparison completed" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "GDDR6 Pattern: Multi-DMA Concurrent Tile Loads" << std::endl;
    std::cout << "Tests multiple DMA engines with 16 banks" << std::endl;
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
    pass &= test_concurrent_dma(4, 4);   // 4 DMAs
    pass &= test_concurrent_dma(8, 4);   // 8 DMAs (LPDDR5 equivalent)
    pass &= test_concurrent_dma(16, 4);  // 16 DMAs (matches banks)
    pass &= test_concurrent_dma(32, 2);  // 32 DMAs (max pressure)
    pass &= test_dma_scaling();
    pass &= test_double_buffer_dma();
    pass &= test_gddr6_vs_lpddr5_dma();

    // Export trace for visualization
    if (export_trace) {
        std::cout << "\n=== Trace Export ===" << std::endl;

        // Use larger queue depth for trace export to fit all 16 banks x 8 lines = 128 requests
        auto trace_config = gddr6_16000_config();
        trace_config.queue_depth = 256;
        GDDR6Harness harness(trace_config);

        // Generate trace for 16 DMAs loading 1 tile each
        constexpr int LINES_TO_TRACE = 8;

        for (int dma = 0; dma < 16; ++dma) {
            uint8_t bank = static_cast<uint8_t>(dma);
            uint32_t row = 100;

            for (int line = 0; line < LINES_TO_TRACE; ++line) {
                harness.submit_read(addr(bank, row, line * CACHE_LINE_BYTES));
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
