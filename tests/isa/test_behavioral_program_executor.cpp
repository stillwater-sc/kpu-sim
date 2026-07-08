/**
 * @file test_behavioral_program_executor.cpp
 * @brief End-to-end tests: DSL schedule → compile → behavioral execute → verify C = A × B
 *
 * These tests prove that the Schedule DSL produces correct matmul results
 * by executing the compiled DMProgram through real memory components with
 * actual float data.
 */

#include <sw/kpu/isa/behavioral_program_executor.hpp>
#include <sw/kpu/dsl/schedule.hpp>
#include <sw/kpu/dsl/schedule_compiler.hpp>
#include <sw/kpu/schedules/matmul_schedule.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>

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

/**
 * @brief Helper: create hardware context with specified L3/L2/L1 counts
 */
struct TestHardware {
    ExternalMemory ext_mem;
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;

    TestHardware(Size ext_mb = 16,
                 size_t num_l3 = 4, Size l3_kb = 256,
                 size_t num_l2 = 8, Size l2_kb = 128,
                 size_t num_l1 = 2, Size l1_kb = 64)
        : ext_mem(ext_mb)
    {
        l3_tiles.reserve(num_l3);
        for (size_t i = 0; i < num_l3; ++i)
            l3_tiles.emplace_back(i, l3_kb);

        l2_banks.reserve(num_l2);
        for (size_t i = 0; i < num_l2; ++i)
            l2_banks.emplace_back(i, l2_kb);

        l1_buffers.reserve(num_l1);
        for (size_t i = 0; i < num_l1; ++i)
            l1_buffers.emplace_back(i, l1_kb * 1024);
    }

    BehavioralProgramExecutor::HardwareContext context() {
        return {ext_mem, l3_tiles, l2_banks, l1_buffers};
    }
};

/**
 * @brief Reference matmul: C = A × B (naive triple loop)
 */
void reference_matmul(const float* a, const float* b, float* c,
                      Size m, Size n, Size k) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (Size kk = 0; kk < k; ++kk) {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// ============================================================================
// Test 1: Single-tile matmul (no tiling loops)
// ============================================================================

void test_single_tile_matmul() {
    std::cout << "\n=== Test: Single-Tile MatMul (16×16×16) ===\n";

    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    // Create DSL schedule
    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    check(prog.instructions.size() > 0, "Program has instructions");
    check(prog.instructions.back().opcode == DMOpcode::HALT, "Ends with HALT");

    // Set up hardware with external memory
    TestHardware hw;

    // Address layout in external memory
    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // Initialize A = all 1.0, B = all 1.0
    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));

    // Zero out C
    std::vector<float> c_zero(M * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // Execute
    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    bool completed = exec.run();

    check(completed, "Program completed (HALT reached)");

    // Read result
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    // Verify: ones × ones = K for every element
    float expected = static_cast<float>(K);
    int errors = 0;
    for (Size i = 0; i < M * N; ++i) {
        if (std::abs(c_result[i] - expected) > 1e-4f) {
            if (errors < 3) {
                std::cout << "    C[" << i << "] = " << c_result[i]
                          << ", expected " << expected << "\n";
            }
            errors++;
        }
    }
    check(errors == 0,
          "All " + std::to_string(M * N) + " elements correct (expected " +
          std::to_string(expected) + ")");

    // Print stats
    const auto& stats = exec.statistics();
    std::cout << "  Stats: " << stats.instructions_executed << " instrs, "
              << stats.dma_loads << " loads, "
              << stats.dma_stores << " stores, "
              << stats.compute_invocations << " computes\n";

    check(stats.dma_loads > 0, "DMA loads occurred");
    check(stats.dma_stores > 0, "DMA stores occurred");
    check(stats.compute_invocations > 0, "Compute fired");
}

// ============================================================================
// Test 2: Multi-tile matmul (4×4 output tile grid)
// ============================================================================

void test_tiled_matmul() {
    std::cout << "\n=== Test: Tiled MatMul (64×64×64, Ti=Tj=Tk=16) ===\n";

    const Size M = 64, N = 64, K = 64;
    const Size Ti = 16, Tj = 16, Tk = 16;

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    Size m_tiles = M / Ti;  // 4
    Size n_tiles = N / Tj;  // 4
    Size k_tiles = K / Tk;  // 4

    std::cout << "  Tile grid: " << m_tiles << "×" << n_tiles
              << ", K tiles: " << k_tiles << "\n";
    std::cout << "  Instructions: " << prog.instructions.size() << "\n";

    // Hardware
    TestHardware hw;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // Initialize A with row index, B with column index
    // C[i,j] = sum_k A[i,k]*B[k,j] = sum_k i * j = K * i * j
    // Actually simpler: A = ones, B = ones → C = K everywhere
    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));

    std::vector<float> c_zero(M * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // Execute
    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    bool completed = exec.run();

    check(completed, "Program completed");

    // Verify
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    float expected = static_cast<float>(K);
    int errors = 0;
    for (Size i = 0; i < M * N; ++i) {
        if (std::abs(c_result[i] - expected) > 1e-3f) {
            if (errors < 5) {
                std::cout << "    C[" << i / N << "," << i % N << "] = "
                          << c_result[i] << ", expected " << expected << "\n";
            }
            errors++;
        }
    }
    check(errors == 0,
          "All " + std::to_string(M * N) + " elements correct (" +
          std::to_string(errors) + " errors)");

    const auto& stats = exec.statistics();
    check(stats.compute_invocations == m_tiles * n_tiles * k_tiles,
          "Compute fired " + std::to_string(m_tiles * n_tiles * k_tiles) +
          " times (got " + std::to_string(stats.compute_invocations) + ")");
}

// ============================================================================
// Test 3: Identity matmul (C = I × A = A)
// ============================================================================

void test_identity_matmul() {
    std::cout << "\n=== Test: Identity MatMul (C = I × A, 16×16) ===\n";

    const Size N = 16;

    auto sched = matmul_output_stationary(N, N, N, N, N, N);
    DMProgram prog = compile_schedule(sched);

    TestHardware hw;

    Address i_base = 0;
    Address a_base = N * N * sizeof(float);
    Address c_base = a_base + N * N * sizeof(float);

    // Identity matrix
    std::vector<float> identity(N * N, 0.0f);
    for (Size i = 0; i < N; ++i) identity[i * N + i] = 1.0f;
    hw.ext_mem.write(i_base, identity.data(), identity.size() * sizeof(float));

    // A = sequential values
    std::vector<float> a_data(N * N);
    for (Size i = 0; i < N * N; ++i) a_data[i] = static_cast<float>(i + 1);
    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));

    std::vector<float> c_zero(N * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // Execute: C = I × A
    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, i_base, a_base, c_base);
    exec.run();

    // Verify C = A
    std::vector<float> c_result(N * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    int errors = 0;
    for (Size i = 0; i < N * N; ++i) {
        if (std::abs(c_result[i] - a_data[i]) > 1e-4f) {
            if (errors < 3) {
                std::cout << "    C[" << i << "] = " << c_result[i]
                          << ", expected " << a_data[i] << "\n";
            }
            errors++;
        }
    }
    check(errors == 0, "C = I × A = A verified");
}

// ============================================================================
// Test 4: Non-trivial matmul with reference comparison
// ============================================================================

void test_reference_matmul() {
    std::cout << "\n=== Test: Reference MatMul (32×48×24) ===\n";

    const Size M = 32, N = 48, K = 24;
    const Size Ti = 16, Tj = 16, Tk = 8;

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    TestHardware hw;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // A[i,k] = (i + k) * 0.1, B[k,j] = (k - j) * 0.1
    std::vector<float> a_data(M * K);
    for (Size i = 0; i < M; ++i)
        for (Size k = 0; k < K; ++k)
            a_data[i * K + k] = static_cast<float>(i + k) * 0.1f;

    std::vector<float> b_data(K * N);
    for (Size k = 0; k < K; ++k)
        for (Size j = 0; j < N; ++j)
            b_data[k * N + j] = static_cast<float>(k) * 0.05f - static_cast<float>(j) * 0.02f;

    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));

    std::vector<float> c_zero(M * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // Reference computation
    std::vector<float> c_ref(M * N, 0.0f);
    reference_matmul(a_data.data(), b_data.data(), c_ref.data(), M, N, K);

    // Execute DSL schedule
    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    // Read and compare
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    int errors = 0;
    float max_err = 0.0f;
    for (Size i = 0; i < M * N; ++i) {
        float err = std::abs(c_result[i] - c_ref[i]);
        max_err = std::max(max_err, err);
        // Use relative tolerance for larger values
        float tol = std::max(1e-3f, std::abs(c_ref[i]) * 1e-4f);
        if (err > tol) {
            if (errors < 3) {
                std::cout << "    C[" << i / N << "," << i % N << "] = "
                          << c_result[i] << ", ref = " << c_ref[i]
                          << ", err = " << err << "\n";
            }
            errors++;
        }
    }
    std::cout << "  Max absolute error: " << max_err << "\n";
    check(errors == 0,
          "All " + std::to_string(M * N) + " elements match reference (" +
          std::to_string(errors) + " errors)");

    const auto& stats = exec.statistics();
    Size expected_computes = (M / Ti) * (N / Tj) * (K / Tk);
    check(stats.compute_invocations == expected_computes,
          "Compute count: " + std::to_string(expected_computes));
}

// ============================================================================
// Test 5: Statistics verification
// ============================================================================

void test_execution_statistics() {
    std::cout << "\n=== Test: Execution Statistics ===\n";

    const Size M = 32, N = 32, K = 32;
    const Size Ti = 16, Tj = 16, Tk = 16;

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    TestHardware hw;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    std::vector<float> c_zero(M * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    const auto& stats = exec.statistics();

    Size m_tiles = M / Ti;  // 2
    Size n_tiles = N / Tj;  // 2
    Size k_tiles = K / Tk;  // 2

    // Per K iteration: 2 DMA loads (A + B)
    // Per output tile: 1 DMA store
    Size expected_loads = m_tiles * n_tiles * k_tiles * 2;
    Size expected_stores = m_tiles * n_tiles;
    Size expected_computes = m_tiles * n_tiles * k_tiles;
    Size expected_barriers = m_tiles * n_tiles * k_tiles;  // 1 barrier per K iter

    check(stats.dma_loads == expected_loads,
          "DMA loads: " + std::to_string(expected_loads) +
          " (got " + std::to_string(stats.dma_loads) + ")");
    check(stats.dma_stores == expected_stores,
          "DMA stores: " + std::to_string(expected_stores) +
          " (got " + std::to_string(stats.dma_stores) + ")");
    check(stats.compute_invocations == expected_computes,
          "Computes: " + std::to_string(expected_computes) +
          " (got " + std::to_string(stats.compute_invocations) + ")");
    check(stats.barriers == expected_barriers,
          "Barriers: " + std::to_string(expected_barriers) +
          " (got " + std::to_string(stats.barriers) + ")");
    check(stats.bytes_loaded > 0, "Bytes loaded > 0");
    check(stats.bytes_stored > 0, "Bytes stored > 0");
}

// ============================================================================
// Test 6: Loop machinery execution (LOOP_BEGIN/LOOP_END with AUTO addressing)
// ============================================================================

void test_loop_machinery() {
    std::cout << "\n=== Test: Loop Machinery (32×32×32, 2×2×2 tiles) ===\n";

    // Dimensions matching 2×2×2 tiles
    const Size M = 32, N = 32, K = 32;
    const Size Ti = 16, Tj = 16, Tk = 16;

    // Create program manually with loop instructions
    DMProgram prog;
    prog.name = "loop_matmul_32x32x32";
    prog.M = M;
    prog.N = N;
    prog.K = K;
    prog.Ti = Ti;
    prog.Tj = Tj;
    prog.Tk = Tk;
    prog.dataflow = DMProgram::Dataflow::OUTPUT_STATIONARY;

    // Configuration instructions using factory methods
    prog.instructions.push_back(DMInstruction::set_tile_dim(Ti, Tj, Tk, 4));

    // SET_STRIDE for A: row_stride=K*4, tile_i_stride=Ti*K*4, tile_j_stride=Tk*4
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::A, K*4, Ti*K*4, Tk*4));

    // SET_STRIDE for B: row_stride=N*4, tile_i_stride=Tk*N*4, tile_j_stride=Tj*4
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::B, N*4, Tk*N*4, Tj*4));

    // SET_STRIDE for C: row_stride=N*4, tile_i_stride=Ti*N*4, tile_j_stride=Tj*4
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::C, N*4, Ti*N*4, Tj*4));

    // Triple-nested loop: for ti, for tj, for tk
    Size m_tiles = M / Ti;  // 2
    Size n_tiles = N / Tj;  // 2
    Size k_tiles = K / Tk;  // 2

    // LOOP_BEGIN 0, 2, TI
    prog.instructions.push_back(DMInstruction::loop_begin(0, static_cast<uint16_t>(m_tiles), IndexRole::TI));

    // LOOP_BEGIN 1, 2, TJ
    prog.instructions.push_back(DMInstruction::loop_begin(1, static_cast<uint16_t>(n_tiles), IndexRole::TJ));

    // LOOP_BEGIN 2, 2, TK
    prog.instructions.push_back(DMInstruction::loop_begin(2, static_cast<uint16_t>(k_tiles), IndexRole::TK));

    // Inner loop body: DMA load A and B, move to L2, stream to L1
    prog.instructions.push_back(DMInstruction::dma_load_auto(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::dma_load_auto(MatrixID::B, 1));
    prog.instructions.push_back(DMInstruction::barrier());

    prog.instructions.push_back(DMInstruction::bm_move_auto(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::bm_move_auto(MatrixID::B, 0));
    prog.instructions.push_back(DMInstruction::barrier());

    prog.instructions.push_back(DMInstruction::str_feed_rows_auto(0));
    prog.instructions.push_back(DMInstruction::str_feed_cols_auto(0));
    prog.instructions.push_back(DMInstruction::barrier());

    // LOOP_END 2 (K loop)
    prog.instructions.push_back(DMInstruction::loop_end(2));

    // After K loop: drain, writeback, store
    prog.instructions.push_back(DMInstruction::str_drain_auto(0));
    prog.instructions.push_back(DMInstruction::barrier());

    prog.instructions.push_back(DMInstruction::bm_writeback_auto(MatrixID::C, 0));
    prog.instructions.push_back(DMInstruction::barrier());

    prog.instructions.push_back(DMInstruction::dma_store_auto(MatrixID::C, 0));
    prog.instructions.push_back(DMInstruction::barrier());

    // LOOP_END 1 (J loop)
    prog.instructions.push_back(DMInstruction::loop_end(1));

    // LOOP_END 0 (I loop)
    prog.instructions.push_back(DMInstruction::loop_end(0));

    // HALT
    prog.instructions.push_back(DMInstruction::halt());

    std::cout << "  Program has " << prog.instructions.size() << " instructions\n";
    std::cout << "  Loop nest: " << m_tiles << "×" << n_tiles << "×" << k_tiles
              << " = " << m_tiles * n_tiles * k_tiles << " tile ops\n";

    // Hardware setup
    TestHardware hw;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // Initialize A = all 1.0, B = all 1.0
    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));

    std::vector<float> c_zero(M * N, 0.0f);
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // Execute
    BehavioralProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    bool completed = exec.run();

    check(completed, "Program completed (HALT reached)");

    // Read result
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    // Verify: ones × ones = K for every element
    float expected = static_cast<float>(K);
    int errors = 0;
    for (Size i = 0; i < M * N; ++i) {
        if (std::abs(c_result[i] - expected) > 1e-4f) {
            if (errors < 3) {
                std::cout << "    C[" << i / N << "," << i % N << "] = "
                          << c_result[i] << ", expected " << expected << "\n";
            }
            errors++;
        }
    }
    check(errors == 0,
          "All " + std::to_string(M * N) + " elements correct (expected " +
          std::to_string(expected) + ")");

    // Verify statistics
    const auto& stats = exec.statistics();
    std::cout << "  Stats: " << stats.instructions_executed << " instrs executed\n";
    std::cout << "         " << stats.loop_iterations << " loop iterations\n";
    std::cout << "         " << stats.config_instructions << " config instrs\n";
    std::cout << "         " << stats.dma_loads << " DMA loads\n";
    std::cout << "         " << stats.compute_invocations << " computes\n";

    Size expected_computes = m_tiles * n_tiles * k_tiles;
    check(stats.compute_invocations == expected_computes,
          "Compute count: " + std::to_string(expected_computes) +
          " (got " + std::to_string(stats.compute_invocations) + ")");

    check(stats.loop_iterations > 0, "Loop iterations > 0");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Behavioral Program Executor Test Suite\n";
    std::cout << "(DSL Schedule → Compile → Execute → Verify)\n";
    std::cout << std::string(60, '*') << "\n";

    test_single_tile_matmul();
    test_tiled_matmul();
    test_identity_matmul();
    test_reference_matmul();
    test_execution_statistics();
    test_loop_machinery();

    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << std::string(60, '*') << "\n\n";

    return failed > 0 ? 1 : 0;
}
