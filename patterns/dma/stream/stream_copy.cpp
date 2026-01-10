// patterns/dma/stream/stream_copy.cpp
//
// STREAM Copy Benchmark Pattern
// A[i] = B[i] - Sequential read from source, sequential write to destination
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

    std::cout << "=== DMA STREAM Copy Pattern ===" << std::endl;
    std::cout << "A[i] = B[i] - Sequential copy benchmark" << std::endl;
    std::cout << "Tests sustained memory bandwidth for sequential access" << std::endl;

    // Configuration
    auto dma_cfg = standard_dma_config();
    auto mc_cfg = lpddr5_6400_config();

    DMAHarness harness(dma_cfg, mc_cfg);

    // STREAM parameters
    constexpr size_t VECTOR_SIZE = 1024 * 1024;  // 1M elements
    constexpr size_t ELEMENT_SIZE = SIZEOF_DOUBLE;  // 8 bytes (double)
    constexpr size_t TOTAL_BYTES = VECTOR_SIZE * ELEMENT_SIZE;
    constexpr size_t BURST_SIZE = 256;  // 256 byte bursts

    constexpr uint64_t SRC_BASE = 0x0;
    constexpr uint64_t DST_BASE = 0x10000000;  // 256MB offset

    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << "Vector size: " << VECTOR_SIZE << " elements" << std::endl;
    std::cout << "Element size: " << ELEMENT_SIZE << " bytes" << std::endl;
    std::cout << "Total bytes: " << (TOTAL_BYTES / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Burst size: " << BURST_SIZE << " bytes" << std::endl;
    std::cout << "Number of bursts: " << ((TOTAL_BYTES + BURST_SIZE - 1) / BURST_SIZE) << std::endl;

    std::cout << "\n--- Submitting transfers ---" << std::endl;

    size_t submitted = harness.submit_stream_copy(
        SRC_BASE, DST_BASE, VECTOR_SIZE, ELEMENT_SIZE, BURST_SIZE);

    std::cout << "Transfers submitted: " << submitted << std::endl;

    // Run simulation
    std::cout << "\n--- Running simulation ---" << std::endl;
    bool success = harness.run_until_complete(10000000);  // 10M cycle limit

    if (!success) {
        std::cerr << "ERROR: Simulation did not complete within timeout" << std::endl;
        return 1;
    }

    // Print statistics
    harness.print_stats(3.2);  // LPDDR5-6400 @ 3.2GHz

    auto stats = harness.stats();

    // STREAM Copy analysis
    std::cout << "\n=== STREAM Copy Analysis ===" << std::endl;

    // Calculate expected vs actual bandwidth
    // Theoretical: LPDDR5-6400 = 6.4 GT/s * 2 bytes/transfer * 2 channels = 25.6 GB/s
    double theoretical_bw = 25.6;
    double achieved_bw = stats.effective_bandwidth_gbps(3.2);
    double efficiency = (achieved_bw / theoretical_bw) * 100.0;

    std::cout << "Theoretical peak bandwidth: " << theoretical_bw << " GB/s" << std::endl;
    std::cout << "Achieved bandwidth: " << std::fixed << std::setprecision(2)
              << achieved_bw << " GB/s" << std::endl;
    std::cout << "Bandwidth efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "%" << std::endl;

    // Page behavior analysis
    std::cout << "\n--- Page Behavior ---" << std::endl;
    std::cout << "Sequential access should maximize page hits" << std::endl;
    std::cout << "Page hit rate: " << std::fixed << std::setprecision(1)
              << (stats.page_hit_rate() * 100.0) << "%" << std::endl;

    if (stats.page_hit_rate() < 0.5) {
        std::cout << "WARNING: Low page hit rate for sequential pattern" << std::endl;
    }

    // Export trace
    if (export_trace) {
        harness.export_trace("stream/stream_copy_trace.json", 3.2);
    }

    // Verification
    bool pass = true;
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
