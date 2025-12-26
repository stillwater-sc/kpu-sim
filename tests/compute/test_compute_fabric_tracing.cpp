#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <set>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/compute_fabric.hpp>
#include <sw/kpu/components/l1_buffer.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>

#include "../test_utilities.hpp"

using namespace sw;
using namespace sw::kpu;
using namespace sw::trace;

// Test fixture for ComputeFabric tracing tests
class ComputeFabricTracingFixture {
public:
    std::vector<L1Buffer> l1_buffers;
    std::unique_ptr<ComputeFabric> compute_fabric_basic;
    std::unique_ptr<ComputeFabric> compute_fabric_systolic;
    TraceLogger& logger;

    ComputeFabricTracingFixture()
        : logger(TraceLogger::instance())
    {
        // Create 2 L1 buffers of 64KB each
        l1_buffers.emplace_back(0, 64);  // ID 0, 64 KB capacity
        l1_buffers.emplace_back(1, 64);  // ID 1, 64 KB capacity

        // Create ComputeFabric instances
        // Basic matmul: tile 0, BASIC_MATMUL, 1 GHz
        compute_fabric_basic = std::make_unique<ComputeFabric>(
            0, ComputeFabric::ComputeType::BASIC_MATMUL, 16, 16, 1.0);

        // Systolic array: tile 1, SYSTOLIC_ARRAY with 16x16 array, 1 GHz
        compute_fabric_systolic = std::make_unique<ComputeFabric>(
            1, ComputeFabric::ComputeType::SYSTOLIC_ARRAY, 16, 16, 1.0);

        // Reset and configure tracing
        logger.clear();
        logger.set_enabled(true);
        compute_fabric_basic->enable_tracing();
        compute_fabric_systolic->enable_tracing();
    }

    ~ComputeFabricTracingFixture() {
        // Leave logger enabled for inspection after tests
    }

    // Helper to generate float matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }

    // Helper to verify matrix multiply result C = A * B
    bool verify_matmul(const std::vector<float>& a, const std::vector<float>& b,
                      const std::vector<float>& c, size_t m, size_t n, size_t k,
                      float tolerance = 1e-3f) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float expected = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    expected += a[i * k + p] * b[p * n + j];
                }
                float actual = c[i * n + j];
                if (std::abs(expected - actual) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
};

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: ComputeFabric Single MatMul - BASIC_MATMUL", "[trace][compute][basic]") {
    const size_t m = 4, n = 4, k = 4;
    const Address a_addr = 0x0;
    const Address b_addr = a_addr + m * k * sizeof(float);
    const Address c_addr = b_addr + k * n * sizeof(float);

    // Generate and write test matrices to scratchpad
    auto matrix_a = generate_matrix(m, k, 1.0f);
    auto matrix_b = generate_matrix(k, n, 2.0f);

    l1_buffers[0].write(a_addr, matrix_a.data(), m * k * sizeof(float));
    l1_buffers[0].write(b_addr, matrix_b.data(), k * n * sizeof(float));

    // Set initial cycle
    compute_fabric_basic->set_cycle(1000);

    size_t initial_trace_count = logger.get_trace_count();

    // Start matmul operation
    bool operation_complete = false;
    ComputeFabric::MatMulConfig config{
        .m = m, .n = n, .k = k,
        .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
        .l1_buffer_id = 0,
        .completion_callback = [&operation_complete]() { operation_complete = true; }
    };

    compute_fabric_basic->start_matmul(config);

    // Should have logged the issue
    REQUIRE(logger.get_trace_count() == initial_trace_count + 1);

    // Process the operation - advance cycle each iteration
    while (!operation_complete) {
        compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
        compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
    }

    // Should have logged the completion
    REQUIRE(logger.get_trace_count() == initial_trace_count + 2);

    // Get traces for this ComputeFabric
    auto cf_traces = logger.get_component_traces(ComponentType::COMPUTE_FABRIC, 0);
    REQUIRE(cf_traces.size() >= 2);

    // Verify the last two traces (issue and completion)
    auto& issue_trace = cf_traces[cf_traces.size() - 2];
    auto& complete_trace = cf_traces[cf_traces.size() - 1];

    // Verify issue trace
    REQUIRE(issue_trace.component_type == ComponentType::COMPUTE_FABRIC);
    REQUIRE(issue_trace.component_id == 0);
    REQUIRE(issue_trace.transaction_type == TransactionType::MATMUL);
    REQUIRE(issue_trace.cycle_issue == 1000);
    REQUIRE(issue_trace.status == TransactionStatus::ISSUED);
    REQUIRE(issue_trace.description.find("BASIC_MATMUL") != std::string::npos);

    // Verify completion trace
    REQUIRE(complete_trace.component_type == ComponentType::COMPUTE_FABRIC);
    REQUIRE(complete_trace.component_id == 0);
    REQUIRE(complete_trace.transaction_type == TransactionType::MATMUL);
    REQUIRE(complete_trace.status == TransactionStatus::COMPLETED);
    REQUIRE(complete_trace.cycle_complete >= complete_trace.cycle_issue);

    // Verify payload data
    REQUIRE(std::holds_alternative<ComputePayload>(complete_trace.payload));
    const auto& payload = std::get<ComputePayload>(complete_trace.payload);
    REQUIRE(payload.num_operations == m * n * k);
    REQUIRE(payload.m == m);
    REQUIRE(payload.n == n);
    REQUIRE(payload.k == k);
    REQUIRE(payload.kernel_name == std::string("BASIC_MATMUL"));

    // Verify the computation result
    std::vector<float> result_c(m * n);
    l1_buffers[0].read(c_addr, result_c.data(), m * n * sizeof(float));
    REQUIRE(verify_matmul(matrix_a, matrix_b, result_c, m, n, k));

    std::cout << "\n=== ComputeFabric MatMul Trace (BASIC_MATMUL) ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Matrix dimensions: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Operations (MACs): " << payload.num_operations << std::endl;
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: ComputeFabric Single MatMul - SYSTOLIC_ARRAY", "[trace][compute][systolic]") {
    const size_t m = 8, n = 8, k = 8;
    const Address a_addr = 0x0;
    const Address b_addr = a_addr + m * k * sizeof(float);
    const Address c_addr = b_addr + k * n * sizeof(float);

    // Generate and write test matrices to scratchpad
    auto matrix_a = generate_matrix(m, k, 1.0f);
    auto matrix_b = generate_matrix(k, n, 2.0f);

    l1_buffers[1].write(a_addr, matrix_a.data(), m * k * sizeof(float));
    l1_buffers[1].write(b_addr, matrix_b.data(), k * n * sizeof(float));

    // Set initial cycle
    compute_fabric_systolic->set_cycle(2000);

    size_t initial_trace_count = logger.get_trace_count();

    // Start matmul operation
    bool operation_complete = false;
    ComputeFabric::MatMulConfig config{
        .m = m, .n = n, .k = k,
        .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
        .l1_buffer_id = 1,
        .completion_callback = [&operation_complete]() { operation_complete = true; }
    };

    compute_fabric_systolic->start_matmul(config);

    // Should have logged the issue
    REQUIRE(logger.get_trace_count() == initial_trace_count + 1);

    // Process the operation - advance cycle each iteration
    while (!operation_complete) {
        compute_fabric_systolic->set_cycle(compute_fabric_systolic->get_cycle() + 1);
        compute_fabric_systolic->update(compute_fabric_systolic->get_cycle(), l1_buffers);
    }

    // Should have logged the completion
    REQUIRE(logger.get_trace_count() == initial_trace_count + 2);

    // Get traces for this ComputeFabric
    auto cf_traces = logger.get_component_traces(ComponentType::COMPUTE_FABRIC, 1);
    REQUIRE(cf_traces.size() >= 2);

    // Verify the last two traces
    auto& issue_trace = cf_traces[cf_traces.size() - 2];
    auto& complete_trace = cf_traces[cf_traces.size() - 1];

    // Verify compute type is mentioned in description
    REQUIRE(issue_trace.description.find("SYSTOLIC_ARRAY") != std::string::npos);
    REQUIRE(complete_trace.description.find("SYSTOLIC_ARRAY") != std::string::npos);

    // Verify payload
    REQUIRE(std::holds_alternative<ComputePayload>(complete_trace.payload));
    const auto& payload = std::get<ComputePayload>(complete_trace.payload);
    REQUIRE(payload.kernel_name == std::string("SYSTOLIC_ARRAY"));

    std::cout << "\n=== ComputeFabric MatMul Trace (SYSTOLIC_ARRAY) ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Matrix dimensions: " << m << "x" << n << "x" << k << std::endl;
    std::cout << "Systolic array size: 16x16" << std::endl;
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: Multiple ComputeFabric Operations", "[trace][compute][queue]") {
    const size_t m = 4, n = 4, k = 4;
    const size_t matrix_size = m * k * sizeof(float);

    // Set initial cycle
    compute_fabric_basic->set_cycle(3000);

    size_t initial_trace_count = logger.get_trace_count();

    // Perform multiple matmul operations sequentially
    int num_operations = 3;
    int completed_count = 0;

    for (int i = 0; i < num_operations; i++) {
        Address a_addr = i * 3 * matrix_size;
        Address b_addr = a_addr + matrix_size;
        Address c_addr = b_addr + matrix_size;

        auto matrix_a = generate_matrix(m, k, static_cast<float>(i + 1));
        auto matrix_b = generate_matrix(k, n, static_cast<float>(i + 2));

        l1_buffers[0].write(a_addr, matrix_a.data(), matrix_size);
        l1_buffers[0].write(b_addr, matrix_b.data(), matrix_size);

        bool operation_complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
            .l1_buffer_id = 0,
            .completion_callback = [&operation_complete, &completed_count]() {
                operation_complete = true;
                completed_count++;
            }
        };

        compute_fabric_basic->start_matmul(config);

        // Process this operation
        while (!operation_complete) {
            compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
            compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
        }
    }

    // Should have logged issue + completion for each operation
    REQUIRE(logger.get_trace_count() == initial_trace_count + (num_operations * 2));
    REQUIRE(completed_count == num_operations);

    // Get all ComputeFabric traces
    auto cf_traces = logger.get_component_traces(ComponentType::COMPUTE_FABRIC, 0);

    // Verify all operations have both issue and complete
    int completed_trace_count = 0;
    for (const auto& trace : cf_traces) {
        if (trace.status == TransactionStatus::COMPLETED) {
            completed_trace_count++;
            REQUIRE(trace.cycle_complete >= trace.cycle_issue);
        }
    }

    REQUIRE(completed_trace_count >= num_operations);

    std::cout << "\n=== Multiple ComputeFabric Operations ===" << std::endl;
    std::cout << "Total traces logged: " << logger.get_trace_count() << std::endl;
    std::cout << "ComputeFabric 0 traces: " << cf_traces.size() << std::endl;
    std::cout << "Completed operations: " << completed_trace_count << std::endl;
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: Export ComputeFabric to CSV", "[trace][compute][export]") {
    const size_t m = 4, n = 4, k = 4;
    const size_t matrix_size = m * k * sizeof(float);

    // Clear previous traces for clean export
    logger.clear();

    // Generate some operations
    compute_fabric_basic->set_cycle(5000);

    for (int i = 0; i < 2; i++) {
        Address a_addr = i * 3 * matrix_size;
        Address b_addr = a_addr + matrix_size;
        Address c_addr = b_addr + matrix_size;

        auto matrix_a = generate_matrix(m, k);
        auto matrix_b = generate_matrix(k, n);

        l1_buffers[0].write(a_addr, matrix_a.data(), matrix_size);
        l1_buffers[0].write(b_addr, matrix_b.data(), matrix_size);

        bool complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
            .l1_buffer_id = 0,
            .completion_callback = [&complete]() { complete = true; }
        };

        compute_fabric_basic->start_matmul(config);

        while (!complete) {
            compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
            compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
        }
    }

    // Export traces to CSV
    auto csv_path = sw::test::get_test_output_path("compute_fabric_trace_test.csv");
    bool csv_export_success = export_logger_traces(csv_path, "csv", logger);
    REQUIRE(csv_export_success);

    std::cout << "\n=== ComputeFabric Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << csv_path << std::endl;
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: Export ComputeFabric to Chrome Trace Format", "[trace][compute][export][chrome]") {
    const size_t m = 8, n = 8, k = 8;
    const size_t matrix_size = m * k * sizeof(float);

    // Clear previous traces for cleaner visualization
    logger.clear();

    // Generate operations with clear cycle progression
    for (int i = 0; i < 5; i++) {
        CycleCount start_cycle = 10000 + i * 2000;
        compute_fabric_systolic->set_cycle(start_cycle);

        Address a_addr = i * 3 * matrix_size;
        Address b_addr = a_addr + matrix_size;
        Address c_addr = b_addr + matrix_size;

        auto matrix_a = generate_matrix(m, k);
        auto matrix_b = generate_matrix(k, n);

        l1_buffers[1].write(a_addr, matrix_a.data(), matrix_size);
        l1_buffers[1].write(b_addr, matrix_b.data(), matrix_size);

        bool complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
            .l1_buffer_id = 1,
            .completion_callback = [&complete]() { complete = true; }
        };

        compute_fabric_systolic->start_matmul(config);

        while (!complete) {
            compute_fabric_systolic->set_cycle(compute_fabric_systolic->get_cycle() + 1);
            compute_fabric_systolic->update(compute_fabric_systolic->get_cycle(), l1_buffers);
        }
    }

    // Export traces to Chrome trace format
    auto chrome_path = sw::test::get_test_output_path("compute_fabric_trace_test.trace");
    bool chrome_export_success = export_logger_traces(chrome_path, "chrome", logger);
    REQUIRE(chrome_export_success);

    std::cout << "\n=== Chrome Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to " << chrome_path << std::endl;
    std::cout << "Open in chrome://tracing for visualization" << std::endl;
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: Cycle Range Query for ComputeFabric", "[trace][compute][query]") {
    // Clear for clean test
    logger.clear();

    // Create operations at different cycle ranges
    std::vector<CycleCount> start_cycles = {1000, 5000, 10000, 15000};

    const size_t m = 4, n = 4, k = 4;
    const size_t matrix_size = m * k * sizeof(float);

    for (CycleCount start : start_cycles) {
        compute_fabric_basic->set_cycle(start);

        auto matrix_a = generate_matrix(m, k);
        auto matrix_b = generate_matrix(k, n);

        l1_buffers[0].write(0, matrix_a.data(), matrix_size);
        l1_buffers[0].write(matrix_size, matrix_b.data(), matrix_size);

        bool complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = 0, .b_addr = matrix_size, .c_addr = 2 * matrix_size,
            .l1_buffer_id = 0,
            .completion_callback = [&complete]() { complete = true; }
        };

        compute_fabric_basic->start_matmul(config);

        while (!complete) {
            compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
            compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
        }
    }

    // Query specific cycle ranges
    auto early_traces = logger.get_traces_in_range(0, 6000);
    auto late_traces = logger.get_traces_in_range(6000, 20000);

    std::cout << "\n=== Cycle Range Query ===" << std::endl;
    std::cout << "Early traces (0-6000): " << early_traces.size() << std::endl;
    std::cout << "Late traces (6000-20000): " << late_traces.size() << std::endl;

    // Should have captured traces in both ranges
    REQUIRE(early_traces.size() > 0);
    REQUIRE(late_traces.size() > 0);
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: ComputeFabric Throughput Analysis", "[trace][compute][analysis]") {
    // Clear for clean test
    logger.clear();

    std::vector<std::tuple<size_t, size_t, size_t>> matrix_configs = {
        {4, 4, 4},     // 64 MACs
        {8, 8, 8},     // 512 MACs
        {16, 16, 16}   // 4096 MACs
    };

    compute_fabric_basic->set_cycle(20000);

    for (const auto& [m, n, k] : matrix_configs) {
        size_t matrix_a_size = m * k * sizeof(float);
        size_t matrix_b_size = k * n * sizeof(float);
        size_t total_size = matrix_a_size + matrix_b_size + m * n * sizeof(float);

        if (total_size > l1_buffers[0].get_capacity()) continue;

        auto matrix_a = generate_matrix(m, k);
        auto matrix_b = generate_matrix(k, n);

        l1_buffers[0].write(0, matrix_a.data(), matrix_a_size);
        l1_buffers[0].write(matrix_a_size, matrix_b.data(), matrix_b_size);

        bool complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = 0,
            .b_addr = matrix_a_size,
            .c_addr = matrix_a_size + matrix_b_size,
            .l1_buffer_id = 0,
            .completion_callback = [&complete]() { complete = true; }
        };

        compute_fabric_basic->start_matmul(config);

        while (!complete) {
            compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
            compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
        }
    }

    // Analyze throughput from traces
    auto cf_traces = logger.get_component_traces(ComponentType::COMPUTE_FABRIC, 0);

    std::cout << "\n=== ComputeFabric Throughput Analysis ===" << std::endl;
    std::cout << "Matrix (MxNxK) | MACs | Duration (cycles) | GFLOPS" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (const auto& trace : cf_traces) {
        if (trace.status == TransactionStatus::COMPLETED && std::holds_alternative<ComputePayload>(trace.payload)) {
            const auto& payload = std::get<ComputePayload>(trace.payload);
            uint64_t duration = trace.get_duration_cycles();

            if (duration > 0 && trace.clock_freq_ghz.has_value()) {
                // GFLOPS = (2 * num_operations) / (duration_cycles / clock_freq_ghz)
                // Factor of 2 because each MAC is 2 operations (multiply + add)
                double ops_per_second = (2.0 * payload.num_operations * trace.clock_freq_ghz.value()) / duration;
                double gflops = ops_per_second; // Already in GFLOPS with 1 GHz clock

                std::cout << payload.m << "x" << payload.n << "x" << payload.k << " | "
                         << payload.num_operations << " | "
                         << duration << " | "
                         << gflops << std::endl;
            }
        }
    }
}

TEST_CASE_METHOD(ComputeFabricTracingFixture, "Trace: Verify Transaction ID Uniqueness", "[trace][compute][txn_id]") {
    // Clear for clean test
    logger.clear();

    // Create multiple operations
    const size_t num_operations = 5;
    const size_t m = 4, n = 4, k = 4;
    const size_t matrix_size = m * k * sizeof(float);

    compute_fabric_basic->set_cycle(30000);

    for (size_t i = 0; i < num_operations; i++) {
        Address a_addr = i * 3 * matrix_size;
        Address b_addr = a_addr + matrix_size;
        Address c_addr = b_addr + matrix_size;

        auto matrix_a = generate_matrix(m, k);
        auto matrix_b = generate_matrix(k, n);

        l1_buffers[0].write(a_addr, matrix_a.data(), matrix_size);
        l1_buffers[0].write(b_addr, matrix_b.data(), matrix_size);

        bool complete = false;
        ComputeFabric::MatMulConfig config{
            .m = m, .n = n, .k = k,
            .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
            .l1_buffer_id = 0,
            .completion_callback = [&complete]() { complete = true; }
        };

        compute_fabric_basic->start_matmul(config);

        while (!complete) {
            compute_fabric_basic->set_cycle(compute_fabric_basic->get_cycle() + 1);
            compute_fabric_basic->update(compute_fabric_basic->get_cycle(), l1_buffers);
        }
    }

    // Get all traces
    auto cf_traces = logger.get_component_traces(ComponentType::COMPUTE_FABRIC, 0);

    // Collect unique transaction IDs (each operation has 2 traces with same ID)
    std::set<uint64_t> unique_txn_ids;
    for (const auto& trace : cf_traces) {
        unique_txn_ids.insert(trace.transaction_id);
    }

    // We should have num_operations unique transaction IDs
    // (each operation generates 2 traces with the same ID: issue + completion)
    REQUIRE(unique_txn_ids.size() >= num_operations);

    std::cout << "\n=== Transaction ID Uniqueness ===" << std::endl;
    std::cout << "Total traces: " << cf_traces.size() << std::endl;
    std::cout << "Unique transaction IDs: " << unique_txn_ids.size() << std::endl;
    std::cout << "Expected unique IDs: " << num_operations << std::endl;
}
