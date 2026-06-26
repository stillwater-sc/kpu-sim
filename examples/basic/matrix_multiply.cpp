/**
 * @file matrix_multiply.cpp
 * @brief Basic matrix multiplication example using KPU
 *
 * Demonstrates the direct L1 API: load operands into an L1 buffer at
 * known offsets, kick off compute with start_matmul(), wait, read the
 * result back. This is the same path exercised by
 * tests/compute/test_systolic_array.cpp and is the canonical way to
 * use the simulator for a single-tile matmul without going through
 * the full DRAM -> L3 -> L2 -> L1 -> Compute dataflow.
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void print_matrix(const std::vector<float>& matrix, size_t rows, size_t cols, const std::string& name) {
    std::cout << name << " [" << rows << "x" << cols << "]:\n";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " KPU Matrix Multiplication Example\n";
    std::cout << "===========================================\n\n";

    const size_t M = 4;
    const size_t N = 4;
    const size_t K = 4;

    std::cout << "Computing C = A * B where:\n";
    std::cout << "  A is " << M << "x" << K << "\n";
    std::cout << "  B is " << K << "x" << N << "\n";
    std::cout << "  C is " << M << "x" << N << "\n\n";

    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count       = 1;
    config.memory_bank_capacity_mb = 1024;
    config.memory_bandwidth_gbps   = 100;
    config.l1_buffer_count         = 1;
    config.l1_buffer_capacity_kb   = 64;
    config.compute_tile_count      = 1;
    config.dma_engine_count        = 1;

    sw::kpu::KPUSimulator kpu(config);

    // Build operand matrices.
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            A[i * K + j] = static_cast<float>(i + j + 1);
        }
    }
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    print_matrix(A, M, K, "Matrix A");
    print_matrix(B, K, N, "Matrix B");

    // L1 buffer layout: [ A | B | C ] back-to-back, byte offsets:
    const sw::kpu::Address a_offset = 0;
    const sw::kpu::Address b_offset = a_offset + M * K * sizeof(float);
    const sw::kpu::Address c_offset = b_offset + K * N * sizeof(float);

    const size_t bank_id      = 0;
    const size_t l1_buffer_id = 0;
    const size_t tile_id      = 0;

    // Stage operands through external memory bank 0, then copy into
    // L1 at our chosen offsets. (Going via the memory bank here is
    // illustrative; a real workload would DMA from host through the
    // full dataflow chain.)
    std::cout << "Staging operands through memory bank " << bank_id << "...\n";
    kpu.write_memory_bank(bank_id, 0,         A.data(), M * K * sizeof(float));
    kpu.write_memory_bank(bank_id, M * K * sizeof(float), B.data(), K * N * sizeof(float));

    std::vector<float> staged_a(M * K), staged_b(K * N);
    kpu.read_memory_bank(bank_id, 0,                       staged_a.data(), M * K * sizeof(float));
    kpu.read_memory_bank(bank_id, M * K * sizeof(float),   staged_b.data(), K * N * sizeof(float));
    kpu.write_l1_buffer(l1_buffer_id, a_offset, staged_a.data(), M * K * sizeof(float));
    kpu.write_l1_buffer(l1_buffer_id, b_offset, staged_b.data(), K * N * sizeof(float));

    std::cout << "Performing matrix multiplication on KPU...\n";
    bool compute_done = false;
    kpu.start_matmul(tile_id, l1_buffer_id, M, N, K,
                     a_offset, b_offset, c_offset,
                     [&compute_done]() { compute_done = true; });

    while (!compute_done) {
        kpu.step();
    }

    kpu.read_l1_buffer(l1_buffer_id, c_offset, C.data(), M * N * sizeof(float));
    std::cout << "Matrix multiplication completed.\n\n";
    print_matrix(C, M, N, "Result Matrix C");

    std::cout << "Performance metrics:\n";
    std::cout << "  Cycles: " << kpu.get_current_cycle() << "\n";
    std::cout << "  Elapsed time: " << kpu.get_elapsed_time_ms() << " ms\n";

    std::cout << "\n===========================================\n";
    std::cout << " Example completed successfully!\n";
    std::cout << "===========================================\n";

    return 0;
}
