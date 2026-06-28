// patterns/dma/stream/stream_triad.cpp
//
// STREAM Triad Benchmark Pattern
// A[i] = B[i] + k*C[i] - Two reads + one write per element
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

    std::cout << "=== DMA STREAM Triad Pattern ===" << std::endl;
    std::cout << "A[i] = B[i] + k*C[i] - Classic memory bandwidth benchmark" << std::endl;
    std::cout << "Tests 2 read + 1 write pattern with interleaved access" << std::endl;

    // Configuration
    auto dma_cfg = high_throughput_dma_config();  // More channels for triad
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // STREAM parameters
    constexpr size_t VECTOR_SIZE = 512 * 1024;  // 512K elements (smaller for faster test)
    constexpr size_t ELEMENT_SIZE = SIZEOF_DOUBLE;  // 8 bytes (double)
    constexpr size_t TOTAL_BYTES = VECTOR_SIZE * ELEMENT_SIZE;
    constexpr size_t BURST_SIZE = 256;

    // Array base addresses (well separated to avoid address aliasing)
    constexpr uint64_t A_BASE = 0x00000000;   // Result array
    constexpr uint64_t B_BASE = 0x10000000;   // First input (256MB offset)
    constexpr uint64_t C_BASE = 0x20000000;   // Second input (512MB offset)

    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Vector size: " << VECTOR_SIZE << " elements" << std::endl;
    std::cout << "Element size: " << ELEMENT_SIZE << " bytes" << std::endl;
    std::cout << "Total bytes per array: " << (TOTAL_BYTES / 1024) << " KB" << std::endl;
    std::cout << "Burst size: " << BURST_SIZE << " bytes" << std::endl;
    std::cout << "Memory accesses per burst: 2 reads + 1 write = 3" << std::endl;

    std::cout << "\n--- Array Layout ---" << std::endl;
    std::cout << "A (result): 0x" << std::hex << A_BASE << std::dec << std::endl;
    std::cout << "B (input1): 0x" << std::hex << B_BASE << std::dec << std::endl;
    std::cout << "C (input2): 0x" << std::hex << C_BASE << std::dec << std::endl;

    std::cout << "\n--- Submitting transfers ---" << std::endl;

    size_t submitted = harness.submit_stream_triad(
        A_BASE, B_BASE, C_BASE, VECTOR_SIZE, ELEMENT_SIZE, BURST_SIZE);

    std::cout << "Transfers submitted: " << submitted << std::endl;
    std::cout << "Expected: " << ((TOTAL_BYTES + BURST_SIZE - 1) / BURST_SIZE) * 3
              << " (3 per burst)" << std::endl;

    // Run simulation
    std::cout << "\n--- Running simulation ---" << std::endl;
    bool success = harness.run_until_complete(10000000);

    if (!success) {
        std::cerr << "ERROR: Simulation did not complete within timeout" << std::endl;
        return 1;
    }

    // Print statistics
    harness.print_stats(3.2);

    auto stats = harness.stats();

    // STREAM Triad analysis
    std::cout << "\n=== STREAM Triad Analysis ===" << std::endl;

    // Triad moves 3 arrays worth of data (2 reads + 1 write)
    // But effective data is just 1 array (the result)
    double bytes_moved = static_cast<double>(stats.dma_bytes_transferred);
    double effective_bytes = bytes_moved / 3.0;  // Actual data written

    std::cout << "Total bytes moved: " << (bytes_moved / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Effective bytes (result): " << (effective_bytes / 1024 / 1024) << " MB" << std::endl;

    // Bandwidth calculation
    // STREAM Triad counts bytes moved, not effective data
    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    double triad_rate = effective_bytes / static_cast<double>(stats.total_cycles) * 3.2;  // GB/s of result data

    std::cout << "\nBandwidth metrics:" << std::endl;
    std::cout << "  Raw bandwidth (all transfers): " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;
    std::cout << "  Triad rate (result data): " << std::fixed << std::setprecision(2)
              << triad_rate << " GB/s" << std::endl;

    // Memory controller analysis
    std::cout << "\n--- Memory Access Pattern ---" << std::endl;
    std::cout << "Reads: " << stats.memory_reads << std::endl;
    std::cout << "Writes: " << stats.memory_writes << std::endl;
    double read_write_ratio = static_cast<double>(stats.memory_reads) /
                              static_cast<double>(std::max(uint64_t{1}, stats.memory_writes));
    std::cout << "Read/Write ratio: " << std::fixed << std::setprecision(2)
              << read_write_ratio << " (expected ~2.0)" << std::endl;

    // Page behavior
    std::cout << "\n--- Page Behavior ---" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;
    std::cout << "Page conflicts: " << stats.page_conflicts << std::endl;

    std::cout << "\nNote: Interleaved 3-stream access may reduce page hits" << std::endl;
    std::cout << "compared to pure sequential access due to bank conflicts." << std::endl;

    // Export trace
    if (export_trace) {
        harness.export_trace("stream/stream_triad_trace.json", 3.2);
    }

    // Verification
    bool pass = true;

    // Check read/write ratio (should be approximately 2:1)
    if (read_write_ratio < 1.5 || read_write_ratio > 2.5) {
        std::cerr << "WARNING: Read/write ratio outside expected range" << std::endl;
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
