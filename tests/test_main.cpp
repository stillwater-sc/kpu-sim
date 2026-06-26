#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <sw/kpu/kpu_simulator.hpp>

// Each regression sub-test returns the number of failures it observed
// (0 on success). main() sums those into nrOfFailedTests and returns
// EXIT_FAILURE if any failure was seen. This is the same pattern used
// in Universal and MTL5 — the previous version of this file printed
// "FAILED" but always returned 0, so failing sub-tests slipped past
// ctest as a green legacy_test_main.

namespace {

// Test 1: basic single-bank configuration exercising the
// memory_bank -> L3 -> compute -> L3 -> memory_bank pipeline.
int test_basic_configuration() {
    std::cout << "\n=== Test 1: Basic Configuration ===" << std::endl;

    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count    = 1;
    config.memory_bank_capacity_mb = 512;
    config.memory_bandwidth_gbps   = 50;
    // L3 / L2 are required by run_matmul_test's DMA pipeline. The old
    // version of this test left these at 0, which made get_l3_tile_base(0)
    // throw std::out_of_range and the matmul never actually executed.
    config.l3_tile_count        = 1;
    config.l3_tile_capacity_kb  = 1024;
    config.l2_bank_count        = 1;
    config.l2_bank_capacity_kb  = 256;
    config.l1_buffer_count      = 1;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count   = 1;
    config.dma_engine_count     = 2;

    sw::kpu::KPUSimulator simulator(config);
    simulator.print_component_status();

    auto test = sw::kpu::test_utils::generate_simple_matmul_test(4, 4, 4);
    bool success = simulator.run_matmul_test(test);

    std::cout << "Basic matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
    simulator.print_stats();
    return success ? 0 : 1;
}

// Test 2: multi-bank distributed matmul. Same caveat as Test 1 about
// needing L3 / L2 — generate_multi_bank_config does not set them, so
// override here.
int test_multi_bank_configuration() {
    std::cout << "\n=== Test 2: Multi-Bank Configuration ===" << std::endl;

    auto config = sw::kpu::test_utils::generate_multi_bank_config(4, 2);
    config.l3_tile_count       = 2;
    config.l3_tile_capacity_kb = 1024;
    config.l2_bank_count       = 2;
    config.l2_bank_capacity_kb = 256;
    sw::kpu::KPUSimulator simulator(config);

    std::cout << "Created simulator with:" << std::endl;
    std::cout << "  " << simulator.get_memory_bank_count() << " memory banks" << std::endl;
    std::cout << "  " << simulator.get_l1_buffer_count()  << " L1 buffers"   << std::endl;
    std::cout << "  " << simulator.get_compute_tile_count() << " compute tiles" << std::endl;
    std::cout << "  " << simulator.get_dma_engine_count() << " DMA engines"  << std::endl;
    simulator.print_component_status();

    bool success = sw::kpu::test_utils::run_distributed_matmul_test(simulator, 8);
    std::cout << "Multi-bank matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
    return success ? 0 : 1;
}

// Test 3: direct API exercise — bypasses L3/L2 entirely, the compute
// fabric reads operands and writes the result inside a single L1
// buffer using explicit byte offsets.
int test_direct_api() {
    std::cout << "\n=== Test 3: Direct API Usage ===" << std::endl;

    sw::kpu::KPUSimulator::Config config;
    config.memory_bank_count    = 2;
    config.memory_bank_capacity_mb = 1024;
    config.memory_bandwidth_gbps   = 100;
    config.l1_buffer_count      = 1;
    config.l1_buffer_capacity_kb = 64;
    config.compute_tile_count   = 1;
    config.dma_engine_count     = 4;

    sw::kpu::KPUSimulator simulator(config);

    std::vector<float> matrix_a = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
    std::vector<float> matrix_b = {2.0f, 0.0f, 1.0f, 2.0f};  // 2x2
    std::vector<float> matrix_c(4, 0.0f);                    // 2x2 result

    simulator.write_memory_bank(0, 0, matrix_a.data(), matrix_a.size() * sizeof(float));
    simulator.write_memory_bank(1, 0, matrix_b.data(), matrix_b.size() * sizeof(float));

    std::vector<float> temp_a(4);
    simulator.read_memory_bank(0, 0, temp_a.data(), temp_a.size() * sizeof(float));
    simulator.write_l1_buffer(0, 0, temp_a.data(), temp_a.size() * sizeof(float));

    std::vector<float> temp_b(4);
    simulator.read_memory_bank(1, 0, temp_b.data(), temp_b.size() * sizeof(float));
    simulator.write_l1_buffer(0, 16, temp_b.data(), temp_b.size() * sizeof(float));

    bool compute_done = false;
    simulator.start_matmul(0, 0, 2, 2, 2, 0, 16, 32,
        [&compute_done]() { compute_done = true; });

    while (!compute_done) {
        simulator.step();
    }

    simulator.read_l1_buffer(0, 32, matrix_c.data(), matrix_c.size() * sizeof(float));

    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << std::fixed << std::setprecision(1) << matrix_c[i * 2 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Expected: A * B = [[1,2],[3,4]] * [[2,0],[1,2]] = [[4,4],[10,8]]
    const std::vector<float> expected = {4.0f, 4.0f, 10.0f, 8.0f};
    int failures = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(matrix_c[i] - expected[i]) > 1e-5f) {
            std::cout << "ERROR: Position " << i << " expected " << expected[i]
                      << " but got " << matrix_c[i] << std::endl;
            ++failures;
        }
    }

    std::cout << "Direct API test: " << (failures == 0 ? "PASSED" : "FAILED") << std::endl;
    simulator.print_stats();
    return failures;
}

// Test 4: status monitoring — purely query-side, no compute.
int test_status_monitoring() {
    std::cout << "\n=== Test 4: Status Monitoring ===" << std::endl;

    auto config = sw::kpu::test_utils::generate_multi_bank_config(3, 2);
    sw::kpu::KPUSimulator simulator(config);

    std::cout << "Component capacities:" << std::endl;
    for (size_t i = 0; i < simulator.get_memory_bank_count(); ++i) {
        std::cout << "  Memory bank[" << i << "]: "
                  << simulator.get_memory_bank_capacity(i) / (1024 * 1024) << " MB" << std::endl;
    }
    for (size_t i = 0; i < simulator.get_l1_buffer_count(); ++i) {
        std::cout << "  L1 buffer[" << i << "]: "
                  << simulator.get_l1_buffer_capacity(i) / 1024 << " KB" << std::endl;
    }

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
    return 0;
}

// Wrap a sub-test so an exception escaping it counts as one failure
// (rather than aborting the entire run before later sub-tests get a
// chance).
int run_subtest(const char* name, int (*fn)()) {
    try {
        return fn();
    } catch (const std::exception& e) {
        std::cerr << name << " threw: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace

int main() {
    std::cout << "=== KPU Simulator Test ===" << std::endl;

    int nrOfFailedTests = 0;
    nrOfFailedTests += run_subtest("test_basic_configuration",   &test_basic_configuration);
    nrOfFailedTests += run_subtest("test_multi_bank_configuration", &test_multi_bank_configuration);
    nrOfFailedTests += run_subtest("test_direct_api",            &test_direct_api);
    nrOfFailedTests += run_subtest("test_status_monitoring",     &test_status_monitoring);

    std::cout << "\n=== Summary ===" << std::endl;
    if (nrOfFailedTests == 0) {
        std::cout << "All Tests Completed Successfully!" << std::endl;
    } else {
        std::cout << nrOfFailedTests << " sub-test(s) failed." << std::endl;
    }

    return nrOfFailedTests > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
