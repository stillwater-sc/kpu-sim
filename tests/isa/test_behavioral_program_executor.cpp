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
#include <functional>
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
// Test: VE_ELEMENTWISE functional semantics (issue #100, epic E2)
// ============================================================================

void test_ve_elementwise() {
    std::cout << "\n=== Test: VE_ELEMENTWISE semantics (16x16 tiles) ===\n";

    const Size Ti = 16, Tj = 16;
    const Size elems = Ti * Tj;
    const Size bytes = elems * sizeof(float);

    // Three L1 buffers: src A (0), src B (1), dst (2)
    TestHardware hw(16, 4, 256, 8, 128, /*num_l1=*/3, 64);

    // Deterministic inputs (positive so SQRT/LOG are in-domain)
    std::vector<float> a(elems), b(elems);
    for (Size i = 0; i < elems; ++i) {
        a[i] = 0.5f + static_cast<float>(i % 17) * 0.25f;
        b[i] = 1.0f + static_cast<float>(i % 5) * 0.5f;
    }

    struct Case {
        DMInstruction instr;
        std::function<float(float, float)> ref;
        std::string name;
    };
    std::vector<Case> cases;
    cases.push_back({DMInstruction::ve_elementwise(VEOp::ADD, 0, 1, 2),
                     [](float x, float y) { return x + y; }, "ADD"});
    cases.push_back({DMInstruction::ve_elementwise(VEOp::MUL, 0, 1, 2),
                     [](float x, float y) { return x * y; }, "MUL"});
    cases.push_back({DMInstruction::ve_elementwise(VEOp::DIV, 0, 1, 2),
                     [](float x, float y) { return x / y; }, "DIV"});
    cases.push_back({DMInstruction::ve_elementwise_unary(VEOp::EXP, 0, 2),
                     [](float x, float) { return std::exp(x); }, "EXP"});
    cases.push_back({DMInstruction::ve_elementwise_unary(VEOp::SQRT, 0, 2),
                     [](float x, float) { return std::sqrt(x); }, "SQRT"});
    cases.push_back({DMInstruction::ve_elementwise_scalar(VEOp::MUL_S, 2.5f, 0, 2),
                     [](float x, float) { return x * 2.5f; }, "MUL_S 2.5"});

    for (auto& tc : cases) {
        // Write inputs directly into L1 and run a one-instruction program
        hw.l1_buffers[0].write(0, a.data(), bytes);
        hw.l1_buffers[1].write(0, b.data(), bytes);

        DMProgram program;
        program.name = "ve_test";
        program.Ti = Ti;
        program.Tj = Tj;
        program.Tk = 1;
        program.instructions.push_back(tc.instr);
        program.instructions.push_back(DMInstruction::halt());

        BehavioralProgramExecutor executor(hw.context());
        executor.load_program(program, 0, 0, 0);
        bool ran = executor.run();

        std::vector<float> out(elems, -1.0f);
        hw.l1_buffers[2].read(0, out.data(), bytes);

        float max_err = 0.0f;
        for (Size i = 0; i < elems; ++i) {
            float expected = tc.ref(a[i], b[i]);
            max_err = std::max(max_err, std::fabs(out[i] - expected));
        }
        check(ran && max_err == 0.0f,
              "VE " + tc.name + " exact vs host oracle (max err " +
              std::to_string(max_err) + ")");
    }
}

// ============================================================================
// Test: STR_BROADCAST resident-operand delivery + repeated VE consumption
// (issue #102, epic E2)
// ============================================================================

void test_str_broadcast_resident_operand() {
    std::cout << "\n=== Test: STR_BROADCAST delivers once, VE consumes many ===\n";

    const Size Ti = 16, Tj = 16;
    const Size elems = Ti * Tj;
    const Size bytes = elems * sizeof(float);

    // Three L1 buffers: streamed A (0), resident broadcast B (1), dst (2)
    TestHardware hw(16, 4, 256, 8, 128, /*num_l1=*/3, 64);

    // Resident bias tile staged in L2 bank 0
    std::vector<float> bias(elems);
    for (Size i = 0; i < elems; ++i) {
        bias[i] = -4.0f + static_cast<float>(i % 13) * 0.5f;
    }
    hw.l2_banks[0].write(0, bias.data(), bytes);

    // Delivery program: ONE broadcast, then halt. The operand must remain
    // resident in L1 for every later consumer without re-delivery.
    {
        DMProgram program;
        program.name = "broadcast_delivery";
        program.Ti = Ti;
        program.Tj = Tj;
        program.Tk = 1;
        program.instructions.push_back(DMInstruction::str_broadcast_col(
            MatrixID::B, TileCoord{0, 0, 0}, /*l2_bank=*/0, /*l1_buf=*/1,
            /*l2_addr=*/0, /*l1_addr=*/0, Ti, Tj, 16));
        program.instructions.push_back(DMInstruction::halt());

        BehavioralProgramExecutor executor(hw.context());
        executor.load_program(program, 0, 0, 0);
        bool ran = executor.run();
        check(ran, "broadcast delivery program runs");
        check(executor.statistics().str_feeds == 1, "broadcast counts one delivery");
    }

    // Three consumer tiles, each a separate program run: stream A into L1[0],
    // VE ADD against the resident B, read the result. B is NEVER re-delivered.
    for (int tile = 0; tile < 3; ++tile) {
        std::vector<float> a(elems);
        for (Size i = 0; i < elems; ++i) {
            a[i] = static_cast<float>(tile + 1) + static_cast<float>(i % 7) * 0.25f;
        }
        hw.l1_buffers[0].write(0, a.data(), bytes);

        DMProgram program;
        program.name = "broadcast_consume";
        program.Ti = Ti;
        program.Tj = Tj;
        program.Tk = 1;
        program.instructions.push_back(
            DMInstruction::ve_elementwise(VEOp::ADD, 0, 1, 2));
        program.instructions.push_back(DMInstruction::halt());

        BehavioralProgramExecutor executor(hw.context());
        executor.load_program(program, 0, 0, 0);
        bool ran = executor.run();

        std::vector<float> out(elems, -1.0f);
        hw.l1_buffers[2].read(0, out.data(), bytes);

        float max_err = 0.0f;
        for (Size i = 0; i < elems; ++i) {
            max_err = std::max(max_err, std::fabs(out[i] - (a[i] + bias[i])));
        }
        check(ran && max_err == 0.0f,
              "consumer tile " + std::to_string(tile) +
              " reads resident broadcast exactly (max err " +
              std::to_string(max_err) + ")");
    }
}

// ============================================================================
// Test: VE_REDUCE streaming semantics + phase flags + edge cases
// (issue #105, epic E3)
// ============================================================================

void test_ve_reduce() {
    std::cout << "\n=== Test: VE_REDUCE streaming semantics (3 tiles) ===\n";

    const Size Ti = 16, Tj = 16;
    const Size elems = Ti * Tj;
    const Size bytes = elems * sizeof(float);
    const int n_tiles = 3;

    // Deterministic multi-tile stream (both signs, non-trivial for VAR)
    std::vector<std::vector<float>> tiles(n_tiles, std::vector<float>(elems));
    for (int k = 0; k < n_tiles; ++k) {
        for (Size i = 0; i < elems; ++i) {
            tiles[k][i] = static_cast<float>(k * 100) - 128.0f
                        + static_cast<float>((i * 7 + k) % 251) * 0.5f;
        }
    }

    // Independent host oracle over the whole stream
    double total_sum = 0.0, total_sumsq = 0.0;
    double omax = -1e30, omin = 1e30;
    size_t total_count = 0;
    for (int k = 0; k < n_tiles; ++k) {
        for (Size i = 0; i < elems; ++i) {
            double v = tiles[k][i];
            total_sum += v; total_sumsq += v * v;
            omax = std::max(omax, v); omin = std::min(omin, v);
            ++total_count;
        }
    }
    const double omean = total_sum / total_count;
    const double ovar = std::max(0.0, total_sumsq / total_count - omean * omean);

    // Run one VE_REDUCE instruction against a persistent accumulator buffer.
    // Streaming = one instruction per tile (l1[0]=src, l1[1]=acc); the
    // accumulator persists across program runs because hw owns the buffers.
    auto run_reduce = [](TestHardware& hw, VEReduceOp op, uint8_t phase,
                         Size ti, Size tj) {
        DMProgram p; p.name = "reduce"; p.Ti = ti; p.Tj = tj; p.Tk = 1;
        p.instructions.push_back(DMInstruction::ve_reduce(op, 0, 1, phase));
        p.instructions.push_back(DMInstruction::halt());
        BehavioralProgramExecutor ex(hw.context());
        ex.load_program(p, 0, 0, 0);
        ex.run();
    };

    auto stream_reduce = [&](VEReduceOp op) {
        TestHardware hw(16, 4, 256, 8, 128, /*num_l1=*/3, 64);
        for (int k = 0; k < n_tiles; ++k) {
            hw.l1_buffers[0].write(0, tiles[k].data(), bytes);
            uint8_t phase = VEReducePhase::ACCUMULATE;
            if (k == 0) phase |= VEReducePhase::INIT;
            if (k == n_tiles - 1) phase |= VEReducePhase::FINALIZE;  // fuse finalize
            run_reduce(hw, op, phase, Ti, Tj);
        }
        std::vector<float> acc(3, -1.0f);
        hw.l1_buffers[1].read(0, acc.data(), 3 * sizeof(float));
        return acc;
    };

    auto approx = [](double a, double b) {
        return std::fabs(a - b) <= 1e-3 * (1.0 + std::fabs(b));
    };

    {
        auto acc = stream_reduce(VEReduceOp::MAX);
        check(static_cast<double>(acc[0]) == omax, "VE_REDUCE MAX over 3 tiles");
    }
    {
        auto acc = stream_reduce(VEReduceOp::MIN);
        check(static_cast<double>(acc[0]) == omin, "VE_REDUCE MIN over 3 tiles");
    }
    {
        auto acc = stream_reduce(VEReduceOp::SUM);
        check(approx(acc[0], total_sum), "VE_REDUCE SUM over 3 tiles");
    }
    {
        // Finalized MEAN layout: [mean, count, sumsq]
        auto acc = stream_reduce(VEReduceOp::MEAN);
        check(approx(acc[0], omean) && acc[1] == static_cast<float>(total_count),
              "VE_REDUCE MEAN over 3 tiles (mean + count exposed)");
    }
    {
        // Finalized VAR layout: [var, mean, count]
        auto acc = stream_reduce(VEReduceOp::VAR);
        check(approx(acc[0], ovar) && approx(acc[1], omean),
              "VE_REDUCE VAR over 3 tiles (var + mean exposed)");
    }

    // Single-shot: one tile, INIT|ACCUMULATE|FINALIZE fused
    {
        TestHardware hw(16, 4, 256, 8, 128, 3, 64);
        hw.l1_buffers[0].write(0, tiles[0].data(), bytes);
        run_reduce(hw, VEReduceOp::SUM,
                   VEReducePhase::INIT | VEReducePhase::ACCUMULATE |
                   VEReducePhase::FINALIZE, Ti, Tj);
        std::vector<float> acc(3);
        hw.l1_buffers[1].read(0, acc.data(), 3 * sizeof(float));
        double t0 = 0.0; for (Size i = 0; i < elems; ++i) t0 += tiles[0][i];
        check(approx(acc[0], t0), "VE_REDUCE single-shot SUM (all phases fused)");
    }

    // Edge: empty reduction (INIT then FINALIZE, no ACCUMULATE) -> NaN
    {
        TestHardware hw(16, 4, 256, 8, 128, 3, 64);
        run_reduce(hw, VEReduceOp::MEAN, VEReducePhase::INIT, Ti, Tj);
        run_reduce(hw, VEReduceOp::MEAN, VEReducePhase::FINALIZE, Ti, Tj);
        std::vector<float> acc(3);
        hw.l1_buffers[1].read(0, acc.data(), 3 * sizeof(float));
        check(std::isnan(acc[0]), "VE_REDUCE MEAN of empty stream -> NaN");
    }
    {
        TestHardware hw(16, 4, 256, 8, 128, 3, 64);
        run_reduce(hw, VEReduceOp::VAR, VEReducePhase::INIT, Ti, Tj);
        run_reduce(hw, VEReduceOp::VAR, VEReducePhase::FINALIZE, Ti, Tj);
        std::vector<float> acc(3);
        hw.l1_buffers[1].read(0, acc.data(), 3 * sizeof(float));
        check(std::isnan(acc[0]), "VE_REDUCE VAR of empty stream -> NaN");
    }

    // Edge: single sample -> variance exactly 0
    {
        TestHardware hw(16, 4, 256, 8, 128, 3, 64);
        float one = 42.5f;
        hw.l1_buffers[0].write(0, &one, sizeof(float));
        run_reduce(hw, VEReduceOp::VAR,
                   VEReducePhase::INIT | VEReducePhase::ACCUMULATE |
                   VEReducePhase::FINALIZE, 1, 1);
        std::vector<float> acc(3);
        hw.l1_buffers[1].read(0, acc.data(), 3 * sizeof(float));
        check(acc[0] == 0.0f && acc[1] == 42.5f,
              "VE_REDUCE VAR of single sample -> var 0, mean = sample");
    }
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
    test_ve_elementwise();
    test_str_broadcast_resident_operand();
    test_ve_reduce();

    std::cout << "\n" << std::string(60, '*') << "\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << std::string(60, '*') << "\n\n";

    return failed > 0 ? 1 : 0;
}
