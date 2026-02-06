/**
 * @file run_matmul.cpp
 * @brief Run a matmul on behavioral or transactional fidelity
 *
 * Build:
 *   cmake --build --preset release --target run_matmul
 *
 * Usage:
 *   ./build/examples/run_matmul [--transactional] [--trace output.json]
 *
 * Examples:
 *   ./build/examples/run_matmul                          # Behavioral only
 *   ./build/examples/run_matmul --transactional          # With timing
 *   ./build/examples/run_matmul --transactional --trace trace.json
 */

#include <sw/kpu/isa/behavioral_program_executor.hpp>
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
#include <cstring>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::isa;
using namespace sw::kpu::dsl;
using namespace sw::kpu::schedules;

int main(int argc, char* argv[]) {
    // ========================================
    // Parse arguments
    // ========================================
    bool use_transactional = false;
    std::string trace_file;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--transactional") == 0 || strcmp(argv[i], "-t") == 0) {
            use_transactional = true;
        } else if ((strcmp(argv[i], "--trace") == 0) && i + 1 < argc) {
            trace_file = argv[++i];
            use_transactional = true;  // Trace implies transactional
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [--transactional] [--trace output.json]\n";
            return 0;
        }
    }

    // ========================================
    // Configure matmul dimensions
    // ========================================
    // MODIFY THESE to change the problem size:
    const Size M = 64;   // A rows, C rows
    const Size N = 64;   // B cols, C cols
    const Size K = 64;   // A cols, B rows (reduction dimension)

    // Tile sizes (must evenly divide dimensions)
    const Size Ti = 16;  // Tile rows
    const Size Tj = 16;  // Tile cols
    const Size Tk = 16;  // Tile reduction

    std::cout << "=== MatMul: C[" << M << "," << N << "] = A[" << M << "," << K
              << "] x B[" << K << "," << N << "] ===\n";
    std::cout << "Tiles: " << (M/Ti) << " x " << (N/Tj) << " x " << (K/Tk)
              << " = " << (M/Ti)*(N/Tj)*(K/Tk) << " tile ops\n";
    std::cout << "FLOPs: " << 2ULL*M*N*K << "\n";
    std::cout << "Fidelity: " << (use_transactional ? "TRANSACTIONAL" : "BEHAVIORAL") << "\n\n";

    // ========================================
    // Create hardware context
    // ========================================
    size_t mem_size = (M*K + K*N + M*N) * sizeof(float) / 1024 + 16;  // KB
    ExternalMemory ext_mem(mem_size);

    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;

    for (int i = 0; i < 4; ++i) l3_tiles.emplace_back(i, 256);   // 256 KB each
    for (int i = 0; i < 8; ++i) l2_banks.emplace_back(i, 128);   // 128 KB each
    for (int i = 0; i < 2; ++i) l1_buffers.emplace_back(i, 64);  // 64 KB each

    BehavioralProgramExecutor::HardwareContext hw{ext_mem, l3_tiles, l2_banks, l1_buffers};

    // ========================================
    // Initialize matrices
    // ========================================
    Address a_base = 0;
    Address b_base = M * K * sizeof(float);
    Address c_base = b_base + K * N * sizeof(float);

    // A = all 1s, B = all 1s => C should be all K
    std::vector<float> a_data(M * K, 1.0f);
    std::vector<float> b_data(K * N, 1.0f);
    std::vector<float> c_zero(M * N, 0.0f);

    ext_mem.write(a_base, a_data.data(), a_data.size() * sizeof(float));
    ext_mem.write(b_base, b_data.data(), b_data.size() * sizeof(float));
    ext_mem.write(c_base, c_zero.data(), c_zero.size() * sizeof(float));

    // ========================================
    // Generate schedule and compile to program
    // ========================================
    auto schedule = matmul_output_stationary(M, N, K, Ti, Tj, Tk);
    DMProgram program = compile_schedule(schedule);

    std::cout << "Program: " << program.name << "\n";
    std::cout << "Instructions: " << program.instructions.size() << "\n\n";

    // ========================================
    // Execute
    // ========================================
    if (use_transactional) {
        // Transactional: behavioral correctness + timing model
        TransactionalProgramExecutor exec(hw);
        exec.load_program(program, a_base, b_base, c_base);

        std::cout << "Running transactional execution...\n";
        bool halted = exec.run();

        // Get results
        auto& beh_stats = exec.behavioral_stats();
        auto timing = exec.get_timing_stats();

        std::cout << "\n--- Behavioral Stats ---\n";
        std::cout << "Halted: " << (halted ? "yes" : "no") << "\n";
        std::cout << "Instructions: " << beh_stats.instructions_executed << "\n";
        std::cout << "DMA loads: " << beh_stats.dma_loads << "\n";
        std::cout << "DMA stores: " << beh_stats.dma_stores << "\n";
        std::cout << "BM moves: " << beh_stats.bm_moves << "\n";
        std::cout << "Compute invocations: " << beh_stats.compute_invocations << "\n";

        std::cout << "\n--- Timing Model ---\n";
        std::cout << "Total cycles: " << timing.total_cycles << "\n";
        std::cout << "DMA cycles: " << timing.dma_cycles
                  << " (" << std::fixed << std::setprecision(1)
                  << (timing.dma_utilization * 100) << "% util)\n";
        std::cout << "BlockMover cycles: " << timing.block_mover_cycles
                  << " (" << (timing.block_mover_utilization * 100) << "% util)\n";
        std::cout << "Streamer cycles: " << timing.streamer_cycles
                  << " (" << (timing.streamer_utilization * 100) << "% util)\n";
        std::cout << "Loop overhead: " << timing.loop_overhead_cycles << " cycles\n";
        std::cout << "Loop iterations: " << timing.loop_iterations << "\n";

        // Export trace if requested
        if (!trace_file.empty()) {
            exec.export_chrome_trace(trace_file);
            std::cout << "\nTrace exported to: " << trace_file << "\n";
            std::cout << "View at: https://ui.perfetto.dev\n";
        }

        // Verify result
        std::vector<float> c_result(M * N);
        ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

        bool correct = true;
        float expected = static_cast<float>(K);
        for (size_t i = 0; i < c_result.size(); ++i) {
            if (std::abs(c_result[i] - expected) > 1e-4f) {
                correct = false;
                std::cout << "\nERROR: C[" << i << "] = " << c_result[i]
                          << ", expected " << expected << "\n";
                break;
            }
        }
        std::cout << "\nResult: " << (correct ? "CORRECT" : "INCORRECT") << "\n";

    } else {
        // Behavioral only: functional correctness
        BehavioralProgramExecutor exec(hw);
        exec.load_program(program, a_base, b_base, c_base);

        std::cout << "Running behavioral execution...\n";
        bool halted = exec.run();

        auto& stats = exec.statistics();

        std::cout << "\n--- Behavioral Stats ---\n";
        std::cout << "Halted: " << (halted ? "yes" : "no") << "\n";
        std::cout << "Instructions: " << stats.instructions_executed << "\n";
        std::cout << "DMA loads: " << stats.dma_loads << "\n";
        std::cout << "DMA stores: " << stats.dma_stores << "\n";
        std::cout << "BM moves: " << stats.bm_moves << "\n";
        std::cout << "BM writebacks: " << stats.bm_writebacks << "\n";
        std::cout << "Streamer feeds: " << stats.str_feeds << "\n";
        std::cout << "Streamer drains: " << stats.str_drains << "\n";
        std::cout << "Compute invocations: " << stats.compute_invocations << "\n";
        std::cout << "Bytes loaded: " << stats.bytes_loaded << "\n";
        std::cout << "Bytes stored: " << stats.bytes_stored << "\n";

        // Verify result
        std::vector<float> c_result(M * N);
        ext_mem.read(c_base, c_result.data(), c_result.size() * sizeof(float));

        bool correct = true;
        float expected = static_cast<float>(K);
        for (size_t i = 0; i < c_result.size(); ++i) {
            if (std::abs(c_result[i] - expected) > 1e-4f) {
                correct = false;
                std::cout << "\nERROR: C[" << i << "] = " << c_result[i]
                          << ", expected " << expected << "\n";
                break;
            }
        }
        std::cout << "\nResult: " << (correct ? "CORRECT" : "INCORRECT") << "\n";
    }

    return 0;
}
