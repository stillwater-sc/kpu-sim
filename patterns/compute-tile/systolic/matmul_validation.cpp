// patterns/compute-tile/systolic/matmul_validation.cpp
//
// Pattern: MatMul performance validation for systolic array
// Tests: XUE metrics (throughput, utilization, efficiency)
//
// This pattern validates that the transactional compute fabric
// produces correct performance metrics for matrix multiplication.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>

using namespace sw::kpu;

// =============================================================================
// XUE Metrics Validation
// =============================================================================
// XUE methodology requires reporting metrics in order:
//   X = Throughput (work done per unit time)
//   U = Utilization (fraction of time resource is busy)
//   E = Efficiency (fraction of peak capability achieved)
// =============================================================================

struct XUEMetrics {
    // Configuration
    uint32_t array_size;      // Systolic array dimension (e.g., 16 for 16x16)
    uint32_t macs_per_cycle;  // Peak MACs/cycle
    double clock_ghz;         // Clock frequency

    // Problem size
    uint32_t M, N, K;         // MatMul dimensions

    // Measured
    uint64_t total_cycles;
    uint64_t busy_cycles;
    uint64_t idle_cycles;
    uint64_t total_macs;
    uint64_t total_flops;

    // XUE Metrics
    double throughput_flops_per_cycle() const {
        return total_cycles > 0 ? static_cast<double>(total_flops) / static_cast<double>(total_cycles) : 0.0;
    }

    double throughput_gflops() const {
        return throughput_flops_per_cycle() * clock_ghz;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    double efficiency() const {
        if (busy_cycles == 0 || macs_per_cycle == 0) return 0.0;
        // Efficiency = actual MACs / (busy_cycles * peak MACs/cycle)
        return static_cast<double>(total_macs) / static_cast<double>(busy_cycles * macs_per_cycle);
    }

    double peak_gflops() const {
        return macs_per_cycle * 2.0 * clock_ghz;  // 2 FLOPs per MAC
    }

    void print() const {
        std::cout << "\nMatMul [" << M << " x " << K << "] @ [" << K << " x " << N << "]:\n";
        std::cout << "  Total FLOPs:     " << total_flops << "\n";
        std::cout << "  Total Cycles:    " << total_cycles << "\n";
        std::cout << "  Busy Cycles:     " << busy_cycles << "\n";
        std::cout << "  Idle Cycles:     " << idle_cycles << "\n";
        std::cout << "\n";
        std::cout << "  Systolic Array XUE (" << array_size << "x" << array_size << "):\n";
        std::cout << "    Throughput   (X): " << std::fixed << std::setprecision(1)
                  << throughput_flops_per_cycle() << " FLOPs/cycle ("
                  << throughput_gflops() << " GFLOPS)\n";
        std::cout << "    Utilization  (U): " << std::fixed << std::setprecision(1)
                  << utilization() * 100.0 << "%\n";
        std::cout << "    Efficiency   (E): " << std::fixed << std::setprecision(1)
                  << efficiency() * 100.0 << "%\n";
        std::cout << "\n";
        std::cout << "  Peak: " << peak_gflops() << " GFLOPS @ " << clock_ghz << " GHz\n";
    }
};

// =============================================================================
// Test Functions
// =============================================================================

XUEMetrics run_matmul_test(uint32_t M, uint32_t N, uint32_t K,
                           uint32_t array_rows = 16, uint32_t array_cols = 16,
                           double clock_ghz = 1.0) {
    // Configure compute fabric
    ComputeFabricConfig config;
    config.fidelity = SimulationFidelity::TRANSACTIONAL;
    config.array_rows = array_rows;
    config.array_cols = array_cols;
    config.macs_per_cycle = array_rows * array_cols;  // Full utilization
    config.pipeline_depth = 4;
    config.enable_statistics = true;

    TransactionalComputeFabric fabric(config, 0);
    fabric.reset();
    fabric.reset_stats();

    // Allocate test data
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    // Submit matmul
    MatMulDescriptor desc;
    desc.m = M;
    desc.n = N;
    desc.k = K;
    desc.dtype = DataType::FLOAT32;

    fabric.submit_matmul(desc, A.data(), B.data(), C.data(), nullptr);
    fabric.drain();

    // Collect stats
    const auto& stats = fabric.stats();

    XUEMetrics metrics;
    metrics.array_size = array_rows;
    metrics.macs_per_cycle = config.macs_per_cycle;
    metrics.clock_ghz = clock_ghz;
    metrics.M = M;
    metrics.N = N;
    metrics.K = K;
    metrics.total_cycles = stats.total_compute_cycles;
    metrics.busy_cycles = stats.busy_cycles;
    metrics.idle_cycles = stats.idle_cycles;
    metrics.total_macs = stats.total_macs;
    metrics.total_flops = stats.total_flops;

    return metrics;
}

bool validate_xue_metrics(const XUEMetrics& metrics,
                          double min_utilization = 0.95,
                          double min_efficiency = 0.90) {
    bool pass = true;

    // Validate utilization
    if (metrics.utilization() < min_utilization) {
        std::cerr << "  FAIL: Utilization " << metrics.utilization() * 100
                  << "% < " << min_utilization * 100 << "%\n";
        pass = false;
    }

    // Validate efficiency
    if (metrics.efficiency() < min_efficiency) {
        std::cerr << "  FAIL: Efficiency " << metrics.efficiency() * 100
                  << "% < " << min_efficiency * 100 << "%\n";
        pass = false;
    }

    // Validate throughput is positive
    if (metrics.throughput_flops_per_cycle() <= 0) {
        std::cerr << "  FAIL: Throughput is zero or negative\n";
        pass = false;
    }

    // Validate FLOPs calculation: 2 * M * N * K
    uint64_t expected_flops = 2ULL * metrics.M * metrics.N * metrics.K;
    if (metrics.total_flops != expected_flops) {
        std::cerr << "  FAIL: FLOPs mismatch. Expected " << expected_flops
                  << ", got " << metrics.total_flops << "\n";
        pass = false;
    }

    // Validate MACs calculation: M * N * K
    uint64_t expected_macs = static_cast<uint64_t>(metrics.M) * metrics.N * metrics.K;
    if (metrics.total_macs != expected_macs) {
        std::cerr << "  FAIL: MACs mismatch. Expected " << expected_macs
                  << ", got " << metrics.total_macs << "\n";
        pass = false;
    }

    if (pass) {
        std::cout << "  PASS: XUE metrics valid\n";
    }

    return pass;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_small_matmul() {
    std::cout << "\n=== Test: Small MatMul (64x64x64) ===" << std::endl;

    auto metrics = run_matmul_test(64, 64, 64);
    metrics.print();

    return validate_xue_metrics(metrics, 0.95, 0.90);
}

bool test_medium_matmul() {
    std::cout << "\n=== Test: Medium MatMul (256x256x256) ===" << std::endl;

    auto metrics = run_matmul_test(256, 256, 256);
    metrics.print();

    return validate_xue_metrics(metrics, 0.98, 0.95);
}

bool test_large_matmul() {
    std::cout << "\n=== Test: Large MatMul (1024x1024x1024) ===" << std::endl;

    auto metrics = run_matmul_test(1024, 1024, 1024);
    metrics.print();

    // Large problems should have high utilization and efficiency
    return validate_xue_metrics(metrics, 0.99, 0.98);
}

bool test_mlp_like() {
    std::cout << "\n=== Test: MLP-like MatMul (32x784 @ 784x128) ===" << std::endl;

    auto metrics = run_matmul_test(32, 128, 784);
    metrics.print();

    return validate_xue_metrics(metrics, 0.95, 0.90);
}

bool test_transformer_like() {
    std::cout << "\n=== Test: Transformer-like MatMul (2048x768 @ 768x3072) ===" << std::endl;

    auto metrics = run_matmul_test(2048, 3072, 768);
    metrics.print();

    return validate_xue_metrics(metrics, 0.99, 0.98);
}

void run_size_sweep() {
    std::cout << "\n=== Size Sweep (XUE Analysis) ===" << std::endl;
    std::cout << std::setw(8) << "Size"
              << std::setw(12) << "Cycles"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "Util%"
              << std::setw(10) << "Eff%"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (uint32_t size : {32, 64, 128, 256, 512, 1024}) {
        auto metrics = run_matmul_test(size, size, size);
        std::cout << std::setw(8) << size
                  << std::setw(12) << metrics.total_cycles
                  << std::setw(12) << std::fixed << std::setprecision(1) << metrics.throughput_gflops()
                  << std::setw(10) << std::fixed << std::setprecision(1) << metrics.utilization() * 100
                  << std::setw(10) << std::fixed << std::setprecision(1) << metrics.efficiency() * 100
                  << std::endl;
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "Systolic Array MatMul Performance Validation" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "\nConfiguration: 16x16 systolic array (256 MACs/cycle)" << std::endl;
    std::cout << "Clock: 1.0 GHz" << std::endl;
    std::cout << "Peak: 512 GFLOPS" << std::endl;

    bool run_sweep = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--sweep") {
            run_sweep = true;
        }
    }

    bool all_pass = true;

    all_pass &= test_small_matmul();
    all_pass &= test_medium_matmul();
    all_pass &= test_large_matmul();
    all_pass &= test_mlp_like();
    all_pass &= test_transformer_like();

    if (run_sweep) {
        run_size_sweep();
    }

    std::cout << "\n==========================================" << std::endl;
    if (all_pass) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
    }
    std::cout << "==========================================" << std::endl;

    return all_pass ? 0 : 1;
}
