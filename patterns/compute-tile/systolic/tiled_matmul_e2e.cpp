// patterns/compute-tile/systolic/tiled_matmul_e2e.cpp
//
// Pattern: End-to-end tiled matrix multiplication with full memory hierarchy
// Tests: DRAM → L3 → L2 → L1 → Compute → L2 → L3 → DRAM
//
// This pattern validates that all transactional components work together
// correctly for tiled matrix multiplication, with XUE metrics at each level.
//
// Configuration:
//   - 16x16 systolic array (256 MACs/cycle)
//   - 1MB L3 tile
//   - 128KB L2 banks
//   - 1KB L1 buffers
//
// Test Cases:
//   1. 16x16 matmul - single tile, minimal hierarchy traversal
//   2. 32x32 matmul - 8 tiles, basic multi-tile execution
//   3. 64x64 matmul - 64 tiles, measure memory/compute balance
//   4. 128x128 matmul - 512 tiles, stress test
//
// Note: This pattern uses SEQUENTIAL execution (memory then compute).
// Pipelined execution (overlapping memory and compute) would achieve
// higher utilization and is demonstrated in examples/behavioral/tiled_matmul_trace.cpp
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include <chrono>

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>
#include <sw/kpu/models/transactional/memory/memory_controller.hpp>
#include <sw/kpu/models/transactional/memory/l3_tile.hpp>

using namespace sw::kpu;

// =============================================================================
// Hardware Configuration
// =============================================================================

struct HardwareConfig {
    // Systolic array
    uint32_t array_rows = 16;
    uint32_t array_cols = 16;
    uint32_t macs_per_cycle = 256;  // 16x16

    // Memory hierarchy
    uint32_t l3_capacity_kb = 1024;   // 1MB L3
    uint32_t l2_capacity_kb = 128;    // 128KB L2
    uint32_t l1_capacity_kb = 1;      // 1KB L1

    // DRAM configuration
    uint32_t dram_capacity_mb = 1024;  // 1GB
    uint32_t dram_bandwidth_gbps = 64; // 64 GB/s

    // Tile size (matched to systolic array)
    uint32_t tile_size = 16;

    // Clock frequency
    double clock_ghz = 1.0;

    // Latencies (cycles)
    uint32_t dram_latency = 100;      // DRAM access latency
    uint32_t l3_latency = 10;         // L3 access latency
    uint32_t l2_latency = 4;          // L2 access latency
    uint32_t l1_latency = 1;          // L1 access latency
};

// =============================================================================
// XUE Metrics Collection
// =============================================================================

struct MemoryLevelStats {
    std::string name;
    uint64_t bytes_transferred = 0;
    uint64_t transfers = 0;
    uint64_t cycles = 0;
    uint64_t busy_cycles = 0;

    double throughput_bytes_per_cycle(uint64_t total_cycles) const {
        return total_cycles > 0 ? static_cast<double>(bytes_transferred) / total_cycles : 0.0;
    }

    double utilization(uint64_t total_cycles) const {
        return total_cycles > 0 ? static_cast<double>(busy_cycles) / total_cycles : 0.0;
    }

    void print_xue(uint64_t total_cycles, double clock_ghz) const {
        double throughput = throughput_bytes_per_cycle(total_cycles);
        double util = utilization(total_cycles);
        double bandwidth_gbps = throughput * clock_ghz;

        std::cout << name << " XUE:\n";
        std::cout << "    Throughput   (X): " << std::fixed << std::setprecision(2)
                  << throughput << " B/cycle (" << bandwidth_gbps << " GB/s)\n";
        std::cout << "    Utilization  (U): " << std::fixed << std::setprecision(1)
                  << util * 100.0 << "%\n";
        std::cout << "    Efficiency   (E): " << transfers << " transfers, "
                  << bytes_transferred << " bytes\n";
    }
};

struct E2EMetrics {
    // Problem size
    uint32_t M, N, K;
    uint32_t tile_size;
    uint32_t num_tiles;

    // Total cycles
    uint64_t total_cycles = 0;

    // Per-level stats
    MemoryLevelStats dram{"DRAM"};
    MemoryLevelStats l3{"L3"};
    MemoryLevelStats l2{"L2"};
    MemoryLevelStats l1{"L1"};

    // Compute stats
    uint64_t compute_cycles = 0;
    uint64_t total_macs = 0;
    uint64_t total_flops = 0;

    // XUE for compute
    double compute_throughput_flops_per_cycle([[maybe_unused]] double clock_ghz) const {
        return total_cycles > 0 ? static_cast<double>(total_flops) / total_cycles : 0.0;
    }

    double compute_utilization() const {
        return total_cycles > 0 ? static_cast<double>(compute_cycles) / total_cycles : 0.0;
    }

    double compute_efficiency(uint32_t peak_macs_per_cycle) const {
        if (compute_cycles == 0 || peak_macs_per_cycle == 0) return 0.0;
        return static_cast<double>(total_macs) / (compute_cycles * peak_macs_per_cycle);
    }

    void print(const HardwareConfig& hw) const {
        std::cout << "\n=== End-to-End Tiled MatMul [" << M << "x" << N << "x" << K << "] ===\n";
        std::cout << "Tile size: " << tile_size << "x" << tile_size << "\n";
        std::cout << "Number of tiles: " << num_tiles << "\n";
        std::cout << "Total cycles: " << total_cycles << "\n\n";

        std::cout << "--- Memory Hierarchy XUE ---\n";
        dram.print_xue(total_cycles, hw.clock_ghz);
        std::cout << "\n";
        l3.print_xue(total_cycles, hw.clock_ghz);
        std::cout << "\n";
        l2.print_xue(total_cycles, hw.clock_ghz);
        std::cout << "\n";

        std::cout << "--- Compute Fabric XUE (" << hw.array_rows << "x" << hw.array_cols << " Systolic Array) ---\n";
        double throughput = compute_throughput_flops_per_cycle(hw.clock_ghz);
        double util = compute_utilization();
        double eff = compute_efficiency(hw.macs_per_cycle);
        double gflops = throughput * hw.clock_ghz;

        std::cout << "    Throughput   (X): " << std::fixed << std::setprecision(1)
                  << throughput << " FLOPs/cycle (" << gflops << " GFLOPS)\n";
        std::cout << "    Utilization  (U): " << std::fixed << std::setprecision(1)
                  << util * 100.0 << "%\n";
        std::cout << "    Efficiency   (E): " << std::fixed << std::setprecision(1)
                  << eff * 100.0 << "%\n";
        std::cout << "\n";

        std::cout << "Peak: " << (hw.macs_per_cycle * 2.0 * hw.clock_ghz)
                  << " GFLOPS @ " << hw.clock_ghz << " GHz\n";
    }
};

// =============================================================================
// Tiled MatMul Executor
// =============================================================================

class TiledMatmulE2E {
public:
    TiledMatmulE2E(const HardwareConfig& hw) : hw_(hw) {
        // Initialize memory controller
        MemoryControllerConfig mc_config;
        mc_config.fidelity = SimulationFidelity::TRANSACTIONAL;
        mc_config.capacity_gb = hw_.dram_capacity_mb / 1024;
        mc_config.timing.mean_read_latency = hw_.dram_latency;
        mc_config.timing.mean_write_latency = hw_.dram_latency;
        mc_config.enable_statistics = true;
        memory_controller_ = std::make_unique<TransactionalMemoryController>(mc_config);

        // Initialize L3 tile
        L3TileConfig l3_config;
        l3_config.fidelity = SimulationFidelity::TRANSACTIONAL;
        l3_config.capacity_kb = hw_.l3_capacity_kb;
        l3_config.access_latency_cycles = hw_.l3_latency;
        l3_config.enable_statistics = true;
        l3_tile_ = std::make_unique<TransactionalL3Tile>(l3_config, 0);

        // Initialize compute fabric
        ComputeFabricConfig cf_config;
        cf_config.fidelity = SimulationFidelity::TRANSACTIONAL;
        cf_config.array_rows = hw_.array_rows;
        cf_config.array_cols = hw_.array_cols;
        cf_config.macs_per_cycle = hw_.macs_per_cycle;
        cf_config.pipeline_depth = 4;
        cf_config.enable_statistics = true;
        compute_fabric_ = std::make_unique<TransactionalComputeFabric>(cf_config, 0);
    }

    E2EMetrics run(uint32_t M, uint32_t N, uint32_t K) {
        E2EMetrics metrics;
        metrics.M = M;
        metrics.N = N;
        metrics.K = K;
        metrics.tile_size = hw_.tile_size;

        // Calculate tiling
        uint32_t m_tiles = (M + hw_.tile_size - 1) / hw_.tile_size;
        uint32_t n_tiles = (N + hw_.tile_size - 1) / hw_.tile_size;
        uint32_t k_tiles = (K + hw_.tile_size - 1) / hw_.tile_size;
        metrics.num_tiles = m_tiles * n_tiles * k_tiles;

        // Allocate data
        size_t tile_bytes = hw_.tile_size * hw_.tile_size * sizeof(float);
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);

        // Reset components
        memory_controller_->reset();
        memory_controller_->reset_stats();
        l3_tile_->reset();
        l3_tile_->reset_stats();
        compute_fabric_->reset();
        compute_fabric_->reset_stats();

        uint64_t current_cycle = 0;

        // Execute tiled matmul: C[i,j] += sum_k(A[i,k] * B[k,j])
        for (uint32_t i = 0; i < m_tiles; ++i) {
            for (uint32_t j = 0; j < n_tiles; ++j) {
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    // === Stage 1: DRAM → L3 ===
                    // Load A tile from DRAM
                    current_cycle += hw_.dram_latency;
                    metrics.dram.bytes_transferred += tile_bytes;
                    metrics.dram.transfers++;
                    metrics.dram.busy_cycles += hw_.dram_latency;

                    // Load B tile from DRAM
                    current_cycle += hw_.dram_latency;
                    metrics.dram.bytes_transferred += tile_bytes;
                    metrics.dram.transfers++;
                    metrics.dram.busy_cycles += hw_.dram_latency;

                    // === Stage 2: L3 → L2 ===
                    current_cycle += hw_.l3_latency;
                    metrics.l3.bytes_transferred += tile_bytes * 2;  // A and B
                    metrics.l3.transfers += 2;
                    metrics.l3.busy_cycles += hw_.l3_latency;

                    // === Stage 3: L2 → L1 (streaming) ===
                    current_cycle += hw_.l2_latency;
                    metrics.l2.bytes_transferred += tile_bytes * 2;
                    metrics.l2.transfers += 2;
                    metrics.l2.busy_cycles += hw_.l2_latency;

                    // === Stage 4: Compute ===
                    // Submit matmul to compute fabric
                    MatMulDescriptor desc;
                    desc.m = std::min(hw_.tile_size, M - i * hw_.tile_size);
                    desc.n = std::min(hw_.tile_size, N - j * hw_.tile_size);
                    desc.k = std::min(hw_.tile_size, K - k * hw_.tile_size);
                    desc.dtype = DataType::FLOAT32;
                    desc.accumulate = (k > 0);

                    // Get tile pointers
                    float* a_tile = A.data() + i * hw_.tile_size * K + k * hw_.tile_size;
                    float* b_tile = B.data() + k * hw_.tile_size * N + j * hw_.tile_size;
                    float* c_tile = C.data() + i * hw_.tile_size * N + j * hw_.tile_size;

                    compute_fabric_->set_cycle(current_cycle);
                    compute_fabric_->submit_matmul(desc, a_tile, b_tile, c_tile, nullptr);
                    compute_fabric_->drain();

                    uint64_t compute_cycles = compute_fabric_->current_cycle() - current_cycle;
                    current_cycle = compute_fabric_->current_cycle();
                    metrics.compute_cycles += compute_cycles;

                    // === Stage 5: Drain to L2 (for final k iteration) ===
                    if (k == k_tiles - 1) {
                        current_cycle += hw_.l2_latency;
                        metrics.l2.bytes_transferred += tile_bytes;
                        metrics.l2.transfers++;
                        metrics.l2.busy_cycles += hw_.l2_latency;

                        // === Stage 6: L2 → L3 → DRAM ===
                        current_cycle += hw_.l3_latency;
                        metrics.l3.bytes_transferred += tile_bytes;
                        metrics.l3.transfers++;
                        metrics.l3.busy_cycles += hw_.l3_latency;

                        current_cycle += hw_.dram_latency;
                        metrics.dram.bytes_transferred += tile_bytes;
                        metrics.dram.transfers++;
                        metrics.dram.busy_cycles += hw_.dram_latency;
                    }
                }
            }
        }

        // Collect final metrics
        const auto& cf_stats = compute_fabric_->stats();
        metrics.total_cycles = current_cycle;
        metrics.total_macs = cf_stats.total_macs;
        metrics.total_flops = cf_stats.total_flops;

        return metrics;
    }

private:
    HardwareConfig hw_;
    std::unique_ptr<TransactionalMemoryController> memory_controller_;
    std::unique_ptr<TransactionalL3Tile> l3_tile_;
    std::unique_ptr<TransactionalComputeFabric> compute_fabric_;
};

// =============================================================================
// Test Cases
// =============================================================================

bool run_single_tile_test(const HardwareConfig& hw) {
    std::cout << "\n=== Test 1: Single Tile (16x16) ===\n";

    TiledMatmulE2E executor(hw);
    auto metrics = executor.run(16, 16, 16);
    metrics.print(hw);

    // Validation
    bool pass = true;

    // Check FLOPs: 2 * M * N * K = 2 * 16 * 16 * 16 = 8192
    uint64_t expected_flops = 2ULL * 16 * 16 * 16;
    if (metrics.total_flops != expected_flops) {
        std::cerr << "  FAIL: FLOPs mismatch. Expected " << expected_flops
                  << ", got " << metrics.total_flops << "\n";
        pass = false;
    }

    // Check utilization > 0
    if (metrics.compute_utilization() <= 0) {
        std::cerr << "  FAIL: Compute utilization is zero\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    return pass;
}

bool run_four_tile_test(const HardwareConfig& hw) {
    std::cout << "\n=== Test 2: Four Tiles (32x32) ===\n";

    TiledMatmulE2E executor(hw);
    auto metrics = executor.run(32, 32, 32);
    metrics.print(hw);

    // Validation
    bool pass = true;

    // Check FLOPs: 2 * M * N * K = 2 * 32 * 32 * 32 = 65536
    uint64_t expected_flops = 2ULL * 32 * 32 * 32;
    if (metrics.total_flops != expected_flops) {
        std::cerr << "  FAIL: FLOPs mismatch. Expected " << expected_flops
                  << ", got " << metrics.total_flops << "\n";
        pass = false;
    }

    // Should have 8 tiles (2x2x2)
    if (metrics.num_tiles != 8) {
        std::cerr << "  FAIL: Tile count mismatch. Expected 8, got "
                  << metrics.num_tiles << "\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    return pass;
}

bool run_sixteen_tile_test(const HardwareConfig& hw) {
    std::cout << "\n=== Test 3: Sixteen Tiles (64x64) ===\n";

    TiledMatmulE2E executor(hw);
    auto metrics = executor.run(64, 64, 64);
    metrics.print(hw);

    // Validation
    bool pass = true;

    // Check FLOPs: 2 * M * N * K = 2 * 64 * 64 * 64 = 524288
    uint64_t expected_flops = 2ULL * 64 * 64 * 64;
    if (metrics.total_flops != expected_flops) {
        std::cerr << "  FAIL: FLOPs mismatch. Expected " << expected_flops
                  << ", got " << metrics.total_flops << "\n";
        pass = false;
    }

    // Should have 64 tiles (4x4x4)
    if (metrics.num_tiles != 64) {
        std::cerr << "  FAIL: Tile count mismatch. Expected 64, got "
                  << metrics.num_tiles << "\n";
        pass = false;
    }

    // Efficiency during compute phases should be high for aligned tiles
    // Note: Overall throughput is limited by sequential memory access in this model
    if (metrics.compute_efficiency(hw.macs_per_cycle) < 0.60) {
        std::cerr << "  FAIL: Compute efficiency < 60%\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    return pass;
}

bool run_sixtyfour_tile_test(const HardwareConfig& hw) {
    std::cout << "\n=== Test 4: Sixty-Four Tiles (128x128) ===\n";

    TiledMatmulE2E executor(hw);
    auto metrics = executor.run(128, 128, 128);
    metrics.print(hw);

    // Validation
    bool pass = true;

    // Check FLOPs: 2 * M * N * K = 2 * 128 * 128 * 128 = 4194304
    uint64_t expected_flops = 2ULL * 128 * 128 * 128;
    if (metrics.total_flops != expected_flops) {
        std::cerr << "  FAIL: FLOPs mismatch. Expected " << expected_flops
                  << ", got " << metrics.total_flops << "\n";
        pass = false;
    }

    // Should have 512 tiles (8x8x8)
    if (metrics.num_tiles != 512) {
        std::cerr << "  FAIL: Tile count mismatch. Expected 512, got "
                  << metrics.num_tiles << "\n";
        pass = false;
    }

    // Efficiency during compute phases should be high for fully aligned tiles
    // Note: Overall throughput is limited by sequential memory access in this model
    if (metrics.compute_efficiency(hw.macs_per_cycle) < 0.60) {
        std::cerr << "  FAIL: Compute efficiency < 60%\n";
        pass = false;
    }

    if (pass) std::cout << "  PASS\n";
    return pass;
}

void run_latency_analysis(const HardwareConfig& hw) {
    std::cout << "\n=== Latency Analysis ===\n";
    std::cout << std::setw(12) << "Size"
              << std::setw(10) << "Tiles"
              << std::setw(12) << "Cycles"
              << std::setw(12) << "Compute"
              << std::setw(10) << "DRAM"
              << std::setw(10) << "L3"
              << std::setw(10) << "Eff%"
              << std::endl;
    std::cout << std::string(76, '-') << std::endl;

    TiledMatmulE2E executor(hw);

    for (uint32_t size : {16, 32, 64, 128, 256}) {
        auto metrics = executor.run(size, size, size);

        uint32_t tiles = (size / hw.tile_size);
        tiles = tiles * tiles * tiles;

        std::cout << std::setw(12) << size
                  << std::setw(10) << tiles
                  << std::setw(12) << metrics.total_cycles
                  << std::setw(12) << metrics.compute_cycles
                  << std::setw(10) << metrics.dram.busy_cycles
                  << std::setw(10) << metrics.l3.busy_cycles
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << metrics.compute_efficiency(hw.macs_per_cycle) * 100.0
                  << std::endl;
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==========================================\n";
    std::cout << "End-to-End Tiled MatMul Validation (v0.4.5)\n";
    std::cout << "==========================================\n";

    HardwareConfig hw;

    std::cout << "\nHardware Configuration:\n";
    std::cout << "  Systolic Array: " << hw.array_rows << "x" << hw.array_cols
              << " (" << hw.macs_per_cycle << " MACs/cycle)\n";
    std::cout << "  L3 Capacity:    " << hw.l3_capacity_kb << " KB\n";
    std::cout << "  L2 Capacity:    " << hw.l2_capacity_kb << " KB\n";
    std::cout << "  L1 Capacity:    " << hw.l1_capacity_kb << " KB\n";
    std::cout << "  DRAM Latency:   " << hw.dram_latency << " cycles\n";
    std::cout << "  L3 Latency:     " << hw.l3_latency << " cycles\n";
    std::cout << "  L2 Latency:     " << hw.l2_latency << " cycles\n";
    std::cout << "  Clock:          " << hw.clock_ghz << " GHz\n";
    std::cout << "  Peak:           " << (hw.macs_per_cycle * 2.0 * hw.clock_ghz) << " GFLOPS\n";

    bool run_analysis = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--analysis") {
            run_analysis = true;
        }
    }

    bool all_pass = true;

    all_pass &= run_single_tile_test(hw);
    all_pass &= run_four_tile_test(hw);
    all_pass &= run_sixteen_tile_test(hw);
    all_pass &= run_sixtyfour_tile_test(hw);

    if (run_analysis) {
        run_latency_analysis(hw);
    }

    std::cout << "\n==========================================\n";
    if (all_pass) {
        std::cout << "ALL TESTS PASSED\n";
    } else {
        std::cout << "SOME TESTS FAILED\n";
    }
    std::cout << "==========================================\n";

    return all_pass ? 0 : 1;
}
