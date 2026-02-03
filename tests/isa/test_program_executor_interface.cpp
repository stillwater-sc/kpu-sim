/**
 * @file test_program_executor_interface.cpp
 * @brief Tests for IProgramExecutor interface and fidelity switching
 *
 * Verifies that:
 * 1. Factory creates correct executor types
 * 2. Both fidelity levels produce correct results
 * 3. Interface methods work correctly
 */

#include <sw/kpu/isa/program_executor_interface.hpp>
#include <sw/kpu/dsl/schedule.hpp>
#include <sw/kpu/dsl/schedule_compiler.hpp>
#include <sw/kpu/schedules/matmul_schedule.hpp>

#include <iostream>
#include <cmath>

using namespace sw::kpu;
using namespace sw::kpu::isa;
using namespace sw::kpu::dsl;
using namespace sw::kpu::schedules;

// ============================================================================
// Test infrastructure
// ============================================================================

static int passed = 0;
static int failed = 0;

void check(bool condition, const std::string& msg) {
    if (condition) {
        std::cout << "  [PASS] " << msg << "\n";
        ++passed;
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
        ++failed;
    }
}

struct TestHardware {
    ExternalMemory ext_mem;
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;

    TestHardware() : ext_mem(16) {
        for (int i = 0; i < 4; ++i) l3_tiles.emplace_back(i, 256);
        for (int i = 0; i < 8; ++i) l2_banks.emplace_back(i, 128);
        for (int i = 0; i < 2; ++i) l1_buffers.emplace_back(i, 64);
    }

    HardwareContext context() {
        return {ext_mem, l3_tiles, l2_banks, l1_buffers};
    }
};

// ============================================================================
// Factory Tests
// ============================================================================

void test_factory_creates_behavioral() {
    std::cout << "test_factory_creates_behavioral:\n";

    TestHardware hw;
    auto exec = create_program_executor(SimulationFidelity::BEHAVIORAL, hw.context());

    check(exec != nullptr, "Factory returns non-null for BEHAVIORAL");
    check(exec->fidelity() == SimulationFidelity::BEHAVIORAL, "Fidelity is BEHAVIORAL");
    check(exec->total_cycles() == 0, "Behavioral has 0 cycles");
}

void test_factory_creates_transactional() {
    std::cout << "test_factory_creates_transactional:\n";

    TestHardware hw;
    auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw.context());

    check(exec != nullptr, "Factory returns non-null for TRANSACTIONAL");
    check(exec->fidelity() == SimulationFidelity::TRANSACTIONAL, "Fidelity is TRANSACTIONAL");
}

void test_factory_returns_null_for_cycle_accurate() {
    std::cout << "test_factory_returns_null_for_cycle_accurate:\n";

    TestHardware hw;
    auto exec = create_program_executor(SimulationFidelity::CYCLE_ACCURATE, hw.context());

    check(exec == nullptr, "Factory returns null for CYCLE_ACCURATE (not implemented)");
}

// ============================================================================
// Behavioral Execution Tests
// ============================================================================

void test_behavioral_matmul_correctness() {
    std::cout << "test_behavioral_matmul_correctness:\n";

    TestHardware hw;
    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    auto exec = create_program_executor(SimulationFidelity::BEHAVIORAL, hw.context());
    exec->load_program(prog, a_base, b_base, c_base);
    bool halted = exec->run();

    check(halted, "Program halted normally");
    check(exec->total_cycles() == 0, "Behavioral reports 0 cycles");

    // Verify result
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    bool all_correct = true;
    for (size_t i = 0; i < c_result.size(); ++i) {
        if (std::abs(c_result[i] - static_cast<float>(K)) > 1e-4f) {
            all_correct = false;
            break;
        }
    }
    check(all_correct, "All C elements equal K");
}

// ============================================================================
// Transactional Execution Tests
// ============================================================================

void test_transactional_matmul_correctness() {
    std::cout << "test_transactional_matmul_correctness:\n";

    TestHardware hw;
    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw.context());
    exec->load_program(prog, a_base, b_base, c_base);
    bool halted = exec->run();

    check(halted, "Program halted normally");
    check(exec->total_cycles() > 0, "Transactional reports positive cycles");

    // Verify result
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    bool all_correct = true;
    for (size_t i = 0; i < c_result.size(); ++i) {
        if (std::abs(c_result[i] - static_cast<float>(K)) > 1e-4f) {
            all_correct = false;
            break;
        }
    }
    check(all_correct, "All C elements equal K");
}

// ============================================================================
// Interface Method Tests
// ============================================================================

void test_statistics_string() {
    std::cout << "test_statistics_string:\n";

    TestHardware hw;
    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    // Test behavioral statistics
    {
        TestHardware hw2;
        hw2.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
        hw2.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
        hw2.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

        auto exec = create_program_executor(SimulationFidelity::BEHAVIORAL, hw2.context());
        exec->load_program(prog, a_base, b_base, c_base);
        exec->run();

        std::string stats = exec->statistics_string();
        check(!stats.empty(), "Behavioral statistics string not empty");
        check(stats.find("Instructions") != std::string::npos, "Contains Instructions count");
    }

    // Test transactional statistics
    {
        TestHardware hw3;
        hw3.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
        hw3.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
        hw3.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

        auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw3.context());
        exec->load_program(prog, a_base, b_base, c_base);
        exec->run();

        std::string stats = exec->statistics_string();
        check(!stats.empty(), "Transactional statistics string not empty");
        check(stats.find("cycles") != std::string::npos, "Contains cycle count");
    }
}

void test_export_trace() {
    std::cout << "test_export_trace:\n";

    TestHardware hw1, hw2;
    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    hw1.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw1.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw1.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    hw2.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw2.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw2.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    // Behavioral should return false
    {
        auto exec = create_program_executor(SimulationFidelity::BEHAVIORAL, hw1.context());
        exec->load_program(prog, a_base, b_base, c_base);
        exec->run();

        bool exported = exec->export_trace("/tmp/behavioral_trace.json");
        check(!exported, "Behavioral export_trace returns false");
    }

    // Transactional should return true
    {
        auto exec = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw2.context());
        exec->load_program(prog, a_base, b_base, c_base);
        exec->run();

        bool exported = exec->export_trace("/tmp/transactional_trace.json");
        check(exported, "Transactional export_trace returns true");
    }
}

// ============================================================================
// Fidelity Switching Test
// ============================================================================

void test_fidelity_switching_same_results() {
    std::cout << "test_fidelity_switching_same_results:\n";

    const Size M = 32, N = 32, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 2.0f);
    std::vector<float> b_data(K * N, 3.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    // Run behavioral
    TestHardware hw1;
    hw1.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw1.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw1.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto exec1 = create_program_executor(SimulationFidelity::BEHAVIORAL, hw1.context());
    exec1->load_program(prog, a_base, b_base, c_base);
    exec1->run();

    std::vector<float> c_behavioral(M * N);
    hw1.ext_mem.read(c_base, c_behavioral.data(), c_behavioral.size() * sizeof(float));

    // Run transactional
    TestHardware hw2;
    hw2.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw2.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw2.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto exec2 = create_program_executor(SimulationFidelity::TRANSACTIONAL, hw2.context());
    exec2->load_program(prog, a_base, b_base, c_base);
    exec2->run();

    std::vector<float> c_transactional(M * N);
    hw2.ext_mem.read(c_base, c_transactional.data(), c_transactional.size() * sizeof(float));

    // Compare results
    bool results_match = true;
    for (size_t i = 0; i < c_behavioral.size(); ++i) {
        if (std::abs(c_behavioral[i] - c_transactional[i]) > 1e-4f) {
            results_match = false;
            std::cout << "    Mismatch at " << i << ": behavioral=" << c_behavioral[i]
                      << " transactional=" << c_transactional[i] << "\n";
            break;
        }
    }
    check(results_match, "Behavioral and transactional produce identical results");

    // Verify expected value: 2 * 3 * K = 96
    float expected = 2.0f * 3.0f * K;
    bool values_correct = true;
    for (size_t i = 0; i < c_behavioral.size(); ++i) {
        if (std::abs(c_behavioral[i] - expected) > 1e-4f) {
            values_correct = false;
            break;
        }
    }
    check(values_correct, "Results equal expected value (2*3*K=96)");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== IProgramExecutor Interface Tests ===\n\n";

    // Factory tests
    test_factory_creates_behavioral();
    test_factory_creates_transactional();
    test_factory_returns_null_for_cycle_accurate();

    // Behavioral tests
    test_behavioral_matmul_correctness();

    // Transactional tests
    test_transactional_matmul_correctness();

    // Interface method tests
    test_statistics_string();
    test_export_trace();

    // Fidelity switching
    test_fidelity_switching_same_results();

    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}
