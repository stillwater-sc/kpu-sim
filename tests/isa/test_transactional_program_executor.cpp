/**
 * @file test_transactional_program_executor.cpp
 * @brief Tests for TransactionalProgramExecutor
 *
 * Verifies that the transactional executor:
 * 1. Produces the same correct results as the behavioral executor
 * 2. Generates reasonable timing estimates
 * 3. Exports valid Chrome Trace format
 */

#include <sw/kpu/isa/transactional_program_executor.hpp>
#include <sw/kpu/dsl/schedule.hpp>
#include <sw/kpu/dsl/schedule_compiler.hpp>
#include <sw/kpu/schedules/matmul_schedule.hpp>
#include <sw/kpu/models/temporal/memory/l3_tile.hpp>
#include <sw/kpu/models/temporal/memory/l2_bank.hpp>
#include <sw/kpu/models/temporal/memory/l1_buffer.hpp>
#include <sw/memory/external_memory.hpp>

#include <iostream>
#include <cmath>
#include <filesystem>
#include <fstream>

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
 * @brief Helper: create hardware context
 */
struct TestHardware {
    ExternalMemory ext_mem;
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;

    TestHardware() : ext_mem(16) {  // 16 KB
        for (int i = 0; i < 4; ++i) l3_tiles.emplace_back(i, 256);  // 256 KB
        for (int i = 0; i < 8; ++i) l2_banks.emplace_back(i, 128);  // 128 KB
        for (int i = 0; i < 2; ++i) l1_buffers.emplace_back(i, 64); // 64 KB
    }

    BehavioralProgramExecutor::HardwareContext context() {
        return {ext_mem, l3_tiles, l2_banks, l1_buffers};
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

void test_constructs_with_defaults() {
    std::cout << "test_constructs_with_defaults:\n";

    TestHardware hw;
    TransactionalProgramExecutor exec(hw.context());

    check(exec.makespan() == 0, "Initial makespan is 0");
    check(exec.timing_events().empty(), "Initial events are empty");
}

void test_constructs_with_custom_timing() {
    std::cout << "test_constructs_with_custom_timing:\n";

    TestHardware hw;
    TimingConfig timing;
    timing.dma_clock_mhz = 500.0;
    timing.block_mover_clock_mhz = 1000.0;

    TransactionalProgramExecutor exec(hw.context(), timing);

    check(exec.makespan() == 0, "Initial makespan with custom timing is 0");
}

// ============================================================================
// Single-Tile Matmul Tests
// ============================================================================

void test_single_tile_matmul_correctness() {
    std::cout << "test_single_tile_matmul_correctness:\n";

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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    bool halted = exec.run();

    check(halted, "Program halted normally");

    // Verify: C[i,j] = K * 1 * 1 = K
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    bool all_correct = true;
    for (size_t i = 0; i < c_result.size(); ++i) {
        if (std::abs(c_result[i] - static_cast<float>(K)) > 1e-4f) {
            all_correct = false;
            std::cout << "    Mismatch at " << i << ": " << c_result[i]
                      << " (expected " << K << ")\n";
            break;
        }
    }
    check(all_correct, "All C elements equal K");
}

void test_single_tile_generates_timing() {
    std::cout << "test_single_tile_generates_timing:\n";

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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    check(!exec.timing_events().empty(), "Generated timing events");
    check(exec.makespan() > 0, "Makespan is positive");

    // Check event categories
    bool has_dma = false, has_bm = false, has_str = false;
    for (const auto& event : exec.timing_events()) {
        if (event.category == "dma") has_dma = true;
        if (event.category == "block_mover") has_bm = true;
        if (event.category == "streamer") has_str = true;
    }

    check(has_dma, "Has DMA events");
    check(has_bm, "Has BlockMover events");
    check(has_str, "Has Streamer events");
}

// ============================================================================
// Multi-Tile Matmul Tests
// ============================================================================

void test_multi_tile_matmul_correctness() {
    std::cout << "test_multi_tile_matmul_correctness:\n";

    TestHardware hw;
    const Size M = 32, N = 32, K = 16;
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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    bool halted = exec.run();

    check(halted, "Program halted normally");

    // Verify each tile
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    bool all_tiles_ok = true;
    for (Size ti = 0; ti < M / Ti; ++ti) {
        for (Size tj = 0; tj < N / Tj; ++tj) {
            Size first_row = ti * Ti;
            Size first_col = tj * Tj;
            float first = c_result[first_row * N + first_col];
            float last = c_result[(first_row + Ti - 1) * N + first_col + Tj - 1];

            if (std::abs(first - K) > 1e-4f || std::abs(last - K) > 1e-4f) {
                all_tiles_ok = false;
                std::cout << "    Tile[" << ti << "," << tj << "] first="
                          << first << " last=" << last << " (expected " << K << ")\n";
            }
        }
    }
    check(all_tiles_ok, "All tiles correct");
}

void test_multi_tile_timing_reasonable() {
    std::cout << "test_multi_tile_timing_reasonable:\n";

    TestHardware hw;
    const Size M = 32, N = 32, K = 32;
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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    auto stats = exec.get_timing_stats();

    check(stats.total_cycles > 0, "Total cycles is positive");
    check(stats.total_cycles < 1000000, "Total cycles is reasonable (< 1M)");
    check(stats.dma_cycles > 0, "DMA cycles is positive");
    check(stats.block_mover_cycles > 0, "BlockMover cycles is positive");
    check(stats.streamer_cycles > 0, "Streamer cycles is positive");

    std::cout << "    Total cycles: " << stats.total_cycles << "\n";
    std::cout << "    DMA: " << stats.dma_cycles
              << ", BM: " << stats.block_mover_cycles
              << ", STR: " << stats.streamer_cycles << "\n";
}

// ============================================================================
// Chrome Trace Export Tests
// ============================================================================

void test_export_chrome_trace() {
    std::cout << "test_export_chrome_trace:\n";

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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    std::string trace_file = (std::filesystem::temp_directory_path() / "test_transactional_trace.json").string();
    exec.export_chrome_trace(trace_file);

    check(std::filesystem::exists(trace_file), "Trace file created");

    auto file_size = std::filesystem::file_size(trace_file);
    check(file_size > 0, "Trace file is not empty");

    // Check JSON format
    std::ifstream in(trace_file);
    char first;
    in >> first;
    check(first == '{', "Trace file starts with '{'");

    std::filesystem::remove(trace_file);
}

// ============================================================================
// Timeline Generation Tests
// ============================================================================

void test_generate_timeline() {
    std::cout << "test_generate_timeline:\n";

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

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    std::string timeline = exec.generate_timeline(80);

    check(!timeline.empty(), "Timeline is not empty");
    check(timeline.find("dma") != std::string::npos, "Timeline contains dma");
    check(timeline.find("block_mover") != std::string::npos, "Timeline contains block_mover");
    check(timeline.find("streamer") != std::string::npos, "Timeline contains streamer");
    check(timeline.find("cycles") != std::string::npos, "Timeline contains cycles");
}

// ============================================================================
// Timing Configuration Tests
// ============================================================================

void test_different_timing_configs() {
    std::cout << "test_different_timing_configs:\n";

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

    // Default timing
    TransactionalProgramExecutor exec_default(hw1.context());
    exec_default.load_program(prog, a_base, b_base, c_base);
    exec_default.run();
    Cycle cycles_default = exec_default.makespan();

    // Fast timing (wider bus)
    TimingConfig fast_timing;
    fast_timing.dma_bus_width_bytes = 128;

    TransactionalProgramExecutor exec_fast(hw2.context(), fast_timing);
    exec_fast.load_program(prog, a_base, b_base, c_base);
    exec_fast.run();
    Cycle cycles_fast = exec_fast.makespan();

    check(cycles_fast <= cycles_default, "Wider bus gives fewer or equal cycles");
    std::cout << "    Default: " << cycles_default << " cycles\n";
    std::cout << "    Fast:    " << cycles_fast << " cycles\n";
}

// ============================================================================
// Identity Matrix Test
// ============================================================================

void test_identity_matrix_multiplication() {
    std::cout << "test_identity_matrix_multiplication:\n";

    TestHardware hw;
    const Size M = 16, N = 16, K = 16;
    const Size Ti = 16, Tj = 16, Tk = 16;

    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // A = identity
    std::vector<float> a_data(M * K, 0.0f);
    for (Size i = 0; i < std::min(M, K); ++i) {
        a_data[i * K + i] = 1.0f;
    }

    // B = test pattern
    std::vector<float> b_data(K * N);
    for (Size i = 0; i < K; ++i) {
        for (Size j = 0; j < N; ++j) {
            b_data[i * N + j] = static_cast<float>(i + j);
        }
    }

    std::vector<float> c_zero(M * N, 0.0f);

    hw.ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    hw.ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    hw.ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    auto sched = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram prog = compile_schedule(sched);

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    // C should equal B
    std::vector<float> c_result(M * N);
    hw.ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

    bool all_correct = true;
    for (Size i = 0; i < M; ++i) {
        for (Size j = 0; j < N; ++j) {
            float expected = static_cast<float>(i + j);
            if (std::abs(c_result[i * N + j] - expected) > 1e-4f) {
                all_correct = false;
                std::cout << "    Mismatch at (" << i << "," << j << "): "
                          << c_result[i * N + j] << " (expected " << expected << ")\n";
                break;
            }
        }
        if (!all_correct) break;
    }
    check(all_correct, "C = I × B = B");
}

// ============================================================================
// Loop Timing Tests
// ============================================================================

void test_loop_timing_overhead() {
    std::cout << "test_loop_timing_overhead:\n";

    TestHardware hw;

    // Create a simple program with a loop
    DMProgram prog;
    prog.name = "loop_test";
    prog.M = 32; prog.N = 16; prog.K = 16;
    prog.Ti = 16; prog.Tj = 16; prog.Tk = 16;

    // Configuration
    prog.instructions.push_back(DMInstruction::set_tile_dim(16, 16, 16, 4));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::B, 2048));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::C, 4096));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::A, 64, 1024, 64));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::B, 64, 1024, 64));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::C, 64, 1024, 64));

    // Loop with 2 iterations bound to TI
    prog.instructions.push_back(DMInstruction::loop_begin(0, 2, IndexRole::TI));

    // Loop body: load, move, stream, drain
    prog.instructions.push_back(DMInstruction::dma_load_auto(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::dma_load_auto(MatrixID::B, 1));
    prog.instructions.push_back(DMInstruction::barrier());
    prog.instructions.push_back(DMInstruction::bm_move_auto(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::bm_move_auto(MatrixID::B, 1));
    prog.instructions.push_back(DMInstruction::barrier());

    prog.instructions.push_back(DMInstruction::loop_end(0));

    prog.instructions.push_back(DMInstruction::halt());

    // Initialize memory
    Address a_base = 0;
    Address b_base = 2048;
    Address c_base = 4096;
    std::vector<float> data(1024, 1.0f);
    hw.ext_mem.write(a_base, data.data(), data.size() * sizeof(float));
    hw.ext_mem.write(b_base, data.data(), data.size() * sizeof(float));

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    auto stats = exec.get_timing_stats();

    // Verify loop overhead is tracked
    check(stats.loop_overhead_cycles > 0, "Loop overhead cycles tracked");
    check(stats.loop_iterations >= 2, "Loop iterations counted");

    std::cout << "    Loop overhead: " << stats.loop_overhead_cycles << " cycles\n";
    std::cout << "    Loop iterations: " << stats.loop_iterations << "\n";
    std::cout << "    Total cycles: " << stats.total_cycles << "\n";
}

void test_nested_loop_timing() {
    std::cout << "test_nested_loop_timing:\n";

    TestHardware hw;

    // Create a program with nested loops (like output-stationary matmul)
    DMProgram prog;
    prog.name = "nested_loop_test";
    prog.M = 32; prog.N = 32; prog.K = 16;
    prog.Ti = 16; prog.Tj = 16; prog.Tk = 16;

    // Configuration
    prog.instructions.push_back(DMInstruction::set_tile_dim(16, 16, 16, 4));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::A, 0));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::B, 2048));
    prog.instructions.push_back(DMInstruction::set_base(MatrixID::C, 4096));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::A, 64, 1024, 64));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::B, 64, 1024, 64));
    prog.instructions.push_back(DMInstruction::set_stride(MatrixID::C, 128, 2048, 64));

    // Outer loop: ti = 0..1 (2 iterations)
    prog.instructions.push_back(DMInstruction::loop_begin(0, 2, IndexRole::TI));

      // Inner loop: tj = 0..1 (2 iterations)
      prog.instructions.push_back(DMInstruction::loop_begin(1, 2, IndexRole::TJ));

        // Simple body with auto addressing
        prog.instructions.push_back(DMInstruction::dma_load_auto(MatrixID::A, 0));
        prog.instructions.push_back(DMInstruction::barrier());

      prog.instructions.push_back(DMInstruction::loop_end(1));

    prog.instructions.push_back(DMInstruction::loop_end(0));

    prog.instructions.push_back(DMInstruction::halt());

    // Initialize memory
    Address a_base = 0;
    Address b_base = 2048;
    Address c_base = 4096;
    std::vector<float> data(1024, 1.0f);
    hw.ext_mem.write(a_base, data.data(), data.size() * sizeof(float));

    TransactionalProgramExecutor exec(hw.context());
    exec.load_program(prog, a_base, b_base, c_base);
    exec.run();

    auto stats = exec.get_timing_stats();

    // With 2x2 nested loops, we should have 4 total iterations
    // plus loop setup/teardown overhead
    check(stats.loop_overhead_cycles > 0, "Nested loop overhead tracked");
    check(stats.loop_iterations >= 4, "Nested loop iterations counted (2x2=4+)");

    // Check that events contain loop markers
    bool has_loop_events = false;
    for (const auto& event : exec.timing_events()) {
        if (event.category == "loop") {
            has_loop_events = true;
            break;
        }
    }
    check(has_loop_events, "Loop events recorded in trace");

    std::cout << "    Loop overhead: " << stats.loop_overhead_cycles << " cycles\n";
    std::cout << "    Loop iterations: " << stats.loop_iterations << "\n";
    std::cout << "    Total cycles: " << stats.total_cycles << "\n";
}

void test_loop_timing_config() {
    std::cout << "test_loop_timing_config:\n";

    TestHardware hw1, hw2;

    // Create a program with a loop
    DMProgram prog;
    prog.name = "loop_timing_config_test";
    prog.M = 16; prog.N = 16; prog.K = 16;
    prog.Ti = 16; prog.Tj = 16; prog.Tk = 16;

    prog.instructions.push_back(DMInstruction::set_tile_dim(16, 16, 16, 4));
    prog.instructions.push_back(DMInstruction::loop_begin(0, 4, IndexRole::TI));
    prog.instructions.push_back(DMInstruction::barrier());  // Minimal body
    prog.instructions.push_back(DMInstruction::loop_end(0));
    prog.instructions.push_back(DMInstruction::halt());

    // Default timing
    TransactionalProgramExecutor exec1(hw1.context());
    exec1.load_program(prog, 0, 0, 0);
    exec1.run();
    auto stats1 = exec1.get_timing_stats();

    // Custom timing with higher loop latency
    TimingConfig slow_loops;
    slow_loops.loop_begin_latency = 10;
    slow_loops.loop_end_latency = 5;
    slow_loops.loop_branch_taken_latency = 5;

    TransactionalProgramExecutor exec2(hw2.context(), slow_loops);
    exec2.load_program(prog, 0, 0, 0);
    exec2.run();
    auto stats2 = exec2.get_timing_stats();

    check(stats2.loop_overhead_cycles > stats1.loop_overhead_cycles,
          "Higher loop latency config increases overhead");

    std::cout << "    Default loop overhead: " << stats1.loop_overhead_cycles << " cycles\n";
    std::cout << "    Slow loop overhead: " << stats2.loop_overhead_cycles << " cycles\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TransactionalProgramExecutor Tests ===\n\n";

    // Basic tests
    test_constructs_with_defaults();
    test_constructs_with_custom_timing();

    // Single-tile tests
    test_single_tile_matmul_correctness();
    test_single_tile_generates_timing();

    // Multi-tile tests
    test_multi_tile_matmul_correctness();
    test_multi_tile_timing_reasonable();

    // Export tests
    test_export_chrome_trace();
    test_generate_timeline();

    // Configuration tests
    test_different_timing_configs();

    // Special cases
    test_identity_matrix_multiplication();

    // Loop timing tests
    test_loop_timing_overhead();
    test_nested_loop_timing();
    test_loop_timing_config();

    // Summary
    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}
