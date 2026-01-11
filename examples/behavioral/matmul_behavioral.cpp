/**
 * @file matmul_behavioral.cpp
 * @brief Behavioral matmul test - validates matrix multiplication correctness
 *
 * This example tests the behavioral execution path:
 *   Host -> L3 -> BlockMover -> L2 -> Compute -> L2 -> BlockMover -> L3 -> Host
 *
 * Test cases:
 *   1. Small matmul: C[4,4] = A[4,8] @ B[8,4] with all ones
 *   2. Medium matmul: C[64,64] = A[64,128] @ B[128,64] with all ones
 *   3. Identity test: C = I @ A should equal A
 */

#include <sw/kpu/behavioral/orchestrator.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace sw::kpu::behavioral;

bool test_ones_matmul(uint32_t m, uint32_t k, uint32_t n) {
    std::cout << "Test: C[" << m << "," << n << "] = A[" << m << "," << k
              << "] @ B[" << k << "," << n << "] (all ones)" << std::endl;

    BehavioralOrchestrator orchestrator;
    auto& memory = orchestrator.memory();

    // Allocate host memory
    Size a_bytes = static_cast<Size>(m) * k * sizeof(float);
    Size b_bytes = static_cast<Size>(k) * n * sizeof(float);
    Size c_bytes = static_cast<Size>(m) * n * sizeof(float);

    auto a_alloc = memory.allocate_host(a_bytes, "A");
    auto b_alloc = memory.allocate_host(b_bytes, "B");
    auto c_alloc = memory.allocate_host(c_bytes, "C");

    // Initialize A and B with ones
    std::vector<float> ones_a(m * k, 1.0f);
    std::vector<float> ones_b(k * n, 1.0f);
    memory.write_floats(a_alloc.base_address, ones_a);
    memory.write_floats(b_alloc.base_address, ones_b);

    // Execute matmul
    orchestrator.execute_matmul(
        a_alloc.base_address, m, k,
        b_alloc.base_address, n,
        c_alloc.base_address);

    // Verify: all C elements should equal k
    auto c_data = memory.read_floats(c_alloc.base_address, m * n);
    float expected = static_cast<float>(k);
    int errors = 0;
    for (size_t i = 0; i < m * n; ++i) {
        if (std::abs(c_data[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::cout << "  Error at C[" << i << "]: " << c_data[i]
                          << " != " << expected << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "  PASS: All " << (m * n) << " elements = " << expected << std::endl;
    } else {
        std::cout << "  FAIL: " << errors << " errors" << std::endl;
    }

    // Print stats
    const auto& stats = orchestrator.stats();
    std::cout << "  Stats: " << stats.matmul_count << " matmuls, "
              << stats.total_flops << " FLOPs" << std::endl;

    // Cleanup
    memory.free(a_alloc);
    memory.free(b_alloc);
    memory.free(c_alloc);

    return errors == 0;
}

bool test_identity_matmul() {
    std::cout << "\nTest: C = I @ A (identity multiplication)" << std::endl;

    const uint32_t n = 4;
    BehavioralOrchestrator orchestrator;
    auto& memory = orchestrator.memory();

    // Allocate
    Size i_bytes = static_cast<Size>(n) * n * sizeof(float);
    Size a_bytes = static_cast<Size>(n) * n * sizeof(float);
    Size c_bytes = static_cast<Size>(n) * n * sizeof(float);

    auto i_alloc = memory.allocate_host(i_bytes, "I");
    auto a_alloc = memory.allocate_host(a_bytes, "A");
    auto c_alloc = memory.allocate_host(c_bytes, "C");

    // Initialize identity matrix I
    std::vector<float> identity(n * n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        identity[i * n + i] = 1.0f;
    }
    memory.write_floats(i_alloc.base_address, identity);

    // Initialize A with sequential values
    std::vector<float> a_data(n * n);
    for (uint32_t i = 0; i < n * n; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }
    memory.write_floats(a_alloc.base_address, a_data);

    // Execute: C = I @ A
    orchestrator.execute_matmul(
        i_alloc.base_address, n, n,
        a_alloc.base_address, n,
        c_alloc.base_address);

    // Verify: C should equal A
    auto c_data = memory.read_floats(c_alloc.base_address, n * n);
    int errors = 0;
    for (uint32_t i = 0; i < n * n; ++i) {
        if (std::abs(c_data[i] - a_data[i]) > 1e-5f) {
            if (errors < 5) {
                std::cout << "  Error at C[" << i << "]: " << c_data[i]
                          << " != " << a_data[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "  PASS: C = A" << std::endl;
    } else {
        std::cout << "  FAIL: " << errors << " errors" << std::endl;
    }

    // Cleanup
    memory.free(i_alloc);
    memory.free(a_alloc);
    memory.free(c_alloc);

    return errors == 0;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << " Behavioral Matmul Test" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::endl;

    bool all_passed = true;

    // Test 1: Small matmul
    all_passed &= test_ones_matmul(4, 8, 4);
    std::cout << std::endl;

    // Test 2: Medium matmul
    all_passed &= test_ones_matmul(32, 64, 32);
    std::cout << std::endl;

    // Test 3: Identity multiplication
    all_passed &= test_identity_matmul();
    std::cout << std::endl;

    std::cout << "==================================================" << std::endl;
    if (all_passed) {
        std::cout << " ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << " SOME TESTS FAILED" << std::endl;
    }
    std::cout << "==================================================" << std::endl;

    return all_passed ? 0 : 1;
}
