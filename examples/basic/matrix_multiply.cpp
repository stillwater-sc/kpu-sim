/**
 * @file matrix_multiply.cpp
 * @brief Basic matrix multiplication example using KPU
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

    // Matrix dimensions (small for demonstration)
    const size_t M = 4;
    const size_t N = 4;
    const size_t K = 4;

    std::cout << "Computing C = A * B where:\n";
    std::cout << "  A is " << M << "x" << K << "\n";
    std::cout << "  B is " << K << "x" << N << "\n";
    std::cout << "  C is " << M << "x" << N << "\n\n";

    // Create KPU simulator
    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count = 1;
    config.memory_bank_capacity_mb = 1024;
    config.memory_bandwidth_gbps = 100;
    config.l1_buffer_count = 1;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count = 1;
    config.dma_engine_count = 1;

    sw::kpu::KPUSimulator kpu(config);

    // Create test matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    // Initialize A with simple values
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            A[i * K + j] = static_cast<float>(i + j + 1);
        }
    }

    // Initialize B with identity-like values
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    print_matrix(A, M, K, "Matrix A");
    print_matrix(B, K, N, "Matrix B");

    // Write matrices to KPU memory
    const size_t bank_id = 0;
    const size_t a_offset = 0;
    const size_t b_offset = M * K * sizeof(float);
    const size_t c_offset = b_offset + K * N * sizeof(float);

    std::cout << "Transferring matrices to KPU memory bank " << bank_id << "...\n";
    kpu.write_memory_bank(bank_id, a_offset, A.data(), M * K * sizeof(float));
    kpu.write_memory_bank(bank_id, b_offset, B.data(), K * N * sizeof(float));

    // Perform matrix multiplication using KPU
    std::cout << "Performing matrix multiplication on KPU...\n";

    sw::kpu::KPUSimulator::MatMulTest test;
    test.m = M;
    test.n = N;
    test.k = K;
    test.matrix_a = A;
    test.matrix_b = B;
    test.expected_c.resize(M * N);

    // Run the test
    bool success = kpu.run_matmul_test(test, bank_id, 0, 0);

    if (success) {
        std::cout << "Matrix multiplication completed successfully!\n\n";

        // Read result back
        kpu.read_memory_bank(bank_id, c_offset, C.data(), M * N * sizeof(float));
        print_matrix(C, M, N, "Result Matrix C");
    } else {
        std::cout << "Matrix multiplication failed!\n";
        return 1;
    }

    // Print statistics
    std::cout << "Performance metrics:\n";
    std::cout << "  Cycles: " << kpu.get_current_cycle() << "\n";
    std::cout << "  Elapsed time: " << kpu.get_elapsed_time_ms() << " ms\n";

    std::cout << "\n===========================================\n";
    std::cout << " Example completed successfully!\n";
    std::cout << "===========================================\n";

    return 0;
}
