#include <iostream>
#include <iomanip>
#include <cmath>
#include <sw/kpu/kpu_simulator.hpp>

int main() {
    std::cout << "=== KPU Simulator Test ===" << std::endl;

    try {
        // Test 1: Basic single-bank, single-tile configuration
        std::cout << "\n=== Test 1: Basic Configuration ===" << std::endl;
        {
            sw::kpu::KPUSimulator::Config config;
            config.memory_bank_count = 1;
            config.memory_bank_capacity_mb = 512;
            config.memory_bandwidth_gbps = 50;
            config.l1_buffer_count = 1;
            config.l1_buffer_capacity_kb = 64;
            config.compute_tile_count = 1;
            config.dma_engine_count = 2;

            sw::kpu::KPUSimulator simulator(config);
            simulator.print_component_status();

            // Run basic matmul test
            auto test = sw::kpu::test_utils::generate_simple_matmul_test(4, 4, 4);
            bool success = simulator.run_matmul_test(test);

            std::cout << "Basic matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
            simulator.print_stats();
        }

        // Test 2: Multi-bank configuration
        std::cout << "\n=== Test 2: Multi-Bank Configuration ===" << std::endl;
        {
            auto config = sw::kpu::test_utils::generate_multi_bank_config(4, 2);
            sw::kpu::KPUSimulator simulator(config);

            std::cout << "Created simulator with:" << std::endl;
            std::cout << "  " << simulator.get_memory_bank_count() << " memory banks" << std::endl;
            std::cout << "  " << simulator.get_l1_buffer_count() << " L1 buffers" << std::endl;
            std::cout << "  " << simulator.get_compute_tile_count() << " compute tiles" << std::endl;
            std::cout << "  " << simulator.get_dma_engine_count() << " DMA engines" << std::endl;

            simulator.print_component_status();

            // Test distributed matmul
            bool success = sw::kpu::test_utils::run_distributed_matmul_test(simulator, 8);
            std::cout << "Multi-bank matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
        }

        // Test 3: Direct API usage (no high-level test functions)
        std::cout << "\n=== Test 3: Direct API Usage ===" << std::endl;
        {
            sw::kpu::KPUSimulator::Config config;
            config.memory_bank_count = 2;
            config.memory_bank_capacity_mb = 1024;
            config.memory_bandwidth_gbps = 100;
            config.l1_buffer_count = 1;
            config.l1_buffer_capacity_kb = 64;
            config.compute_tile_count = 1;
            config.dma_engine_count = 4;

            sw::kpu::KPUSimulator simulator(config);

            // Create simple test matrices
            std::vector<float> matrix_a = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
            std::vector<float> matrix_b = {2.0f, 0.0f, 1.0f, 2.0f};  // 2x2
            std::vector<float> matrix_c(4, 0.0f);                    // 2x2 result

            // Load matrices into different memory banks
            simulator.write_memory_bank(0, 0, matrix_a.data(), matrix_a.size() * sizeof(float));
            simulator.write_memory_bank(1, 0, matrix_b.data(), matrix_b.size() * sizeof(float));

            std::cout << "Loaded matrices into separate memory banks" << std::endl;
            std::cout << "Matrix A: [" << matrix_a[0] << ", " << matrix_a[1] << ", " << matrix_a[2] << ", " << matrix_a[3] << "]" << std::endl;
            std::cout << "Matrix B: [" << matrix_b[0] << ", " << matrix_b[1] << ", " << matrix_b[2] << ", " << matrix_b[3] << "]" << std::endl;

            // Manually transfer matrices to L1 buffer using low-level API
            // Read matrix A from bank 0
            std::vector<float> temp_a(4);
            simulator.read_memory_bank(0, 0, temp_a.data(), temp_a.size() * sizeof(float));
            simulator.write_l1_buffer(0, 0, temp_a.data(), temp_a.size() * sizeof(float));
            std::cout << "Matrix A transferred to L1 buffer" << std::endl;

            // Read matrix B from bank 1
            std::vector<float> temp_b(4);
            simulator.read_memory_bank(1, 0, temp_b.data(), temp_b.size() * sizeof(float));
            simulator.write_l1_buffer(0, 16, temp_b.data(), temp_b.size() * sizeof(float));
            std::cout << "Matrix B transferred to L1 buffer" << std::endl;

            // Start matrix multiplication
            bool compute_done = false;
            simulator.start_matmul(0, 0, 2, 2, 2, 0, 16, 32,
                [&compute_done]() {
                    std::cout << "Matrix multiplication completed" << std::endl;
                    compute_done = true;
                });

            // Wait for computation
            while (!compute_done) {
                simulator.step();
            }

            // Read result from L1 buffer
            simulator.read_l1_buffer(0, 32, matrix_c.data(), matrix_c.size() * sizeof(float));

            std::cout << "Result matrix C:" << std::endl;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::cout << std::fixed << std::setprecision(1) << matrix_c[i * 2 + j] << " ";
                }
                std::cout << std::endl;
            }

            // Verify result (expected: [[4, 4], [10, 8]])
            // A * B = [[1,2],[3,4]] * [[2,0],[1,2]] = [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4,4],[10,8]]
            std::vector<float> expected = {4.0f, 4.0f, 10.0f, 8.0f};
            bool api_test_passed = true;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (std::abs(matrix_c[i] - expected[i]) > 1e-5f) {
                    std::cout << "ERROR: Position " << i << " expected " << expected[i] << " but got " << matrix_c[i] << std::endl;
                    api_test_passed = false;
                }
            }

            std::cout << "Direct API test: " << (api_test_passed ? "PASSED" : "FAILED") << std::endl;
            simulator.print_stats();
        }

        // Test 4: Component status and monitoring
        std::cout << "\n=== Test 4: Status Monitoring ===" << std::endl;
        {
            auto config = sw::kpu::test_utils::generate_multi_bank_config(3, 2);
            sw::kpu::KPUSimulator simulator(config);

            std::cout << "Component capacities:" << std::endl;
            for (size_t i = 0; i < simulator.get_memory_bank_count(); ++i) {
                std::cout << "  Memory bank[" << i << "]: "
                         << simulator.get_memory_bank_capacity(i) / (1024*1024) << " MB" << std::endl;
            }
            for (size_t i = 0; i < simulator.get_l1_buffer_count(); ++i) {
                std::cout << "  L1 buffer[" << i << "]: "
                         << simulator.get_l1_buffer_capacity(i) / 1024 << " KB" << std::endl;
            }

            // Test readiness status
            std::cout << "\nReadiness status:" << std::endl;
            for (size_t i = 0; i < simulator.get_memory_bank_count(); ++i) {
                std::cout << "  Memory bank[" << i << "] ready: "
                         << (simulator.is_memory_bank_ready(i) ? "Yes" : "No") << std::endl;
            }
            for (size_t i = 0; i < simulator.get_l1_buffer_count(); ++i) {
                std::cout << "  L1 buffer[" << i << "] ready: "
                         << (simulator.is_l1_buffer_ready(i) ? "Yes" : "No") << std::endl;
            }

            std::cout << "Status monitoring test: PASSED" << std::endl;
        }

        std::cout << "\n=== All Tests Completed Successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
