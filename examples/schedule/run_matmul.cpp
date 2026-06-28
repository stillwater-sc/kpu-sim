/**
 * @file run_matmul.cpp
 * @brief Run matmul using CSP concurrent timing model
 *
 * This is a kernel performance test for optimizing CSP process concurrency.
 * It uses the ConcurrentTimingExecutor with credit-based dataflow to model
 * true concurrency between DMA, BlockMover, and Streamer processes.
 *
 * Build:
 *   cmake --build --preset release --target run_matmul
 *
 * Usage:
 *   ./build/examples/schedule/run_matmul [options]
 *
 * Options:
 *   -M <size>      Matrix M dimension (default: 64)
 *   -N <size>      Matrix N dimension (default: 64)
 *   -K <size>      Matrix K dimension (default: 64)
 *   --Ti <size>    Tile i dimension (default: 16)
 *   --Tj <size>    Tile j dimension (default: 16)
 *   --Tk <size>    Tile k dimension (default: 16)
 *   --strategy <s> Scheduling strategy: interleaved (default), blocked, output_stationary
 *   --trace <file> Export Chrome trace to file
 *   --validate     Run schedule validation before execution
 *   -h, --help     Show this help
 *
 * Examples:
 *   ./build/examples/schedule/run_matmul
 *   ./build/examples/schedule/run_matmul -M 128 -N 128 -K 128
 *   ./build/examples/schedule/run_matmul --strategy blocked --validate
 *   ./build/examples/schedule/run_matmul --trace trace.json
 */

#include <sw/kpu/timing/concurrent_timing_executor.hpp>
#include <sw/kpu/timing/schedule/matmul_schedule_generator.hpp>
#include <sw/kpu/timing/schedule/schedule_executor.hpp>
#include <sw/kpu/timing/schedule/schedule_validator.hpp>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>

using namespace sw::kpu::timing;
using namespace sw::kpu::timing::schedule;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -M <size>      Matrix M dimension (default: 64)\n";
    std::cout << "  -N <size>      Matrix N dimension (default: 64)\n";
    std::cout << "  -K <size>      Matrix K dimension (default: 64)\n";
    std::cout << "  --Ti <size>    Tile i dimension (default: 16)\n";
    std::cout << "  --Tj <size>    Tile j dimension (default: 16)\n";
    std::cout << "  --Tk <size>    Tile k dimension (default: 16)\n";
    std::cout << "  --strategy <s> Scheduling strategy:\n";
    std::cout << "                   interleaved (default) - A-B-A-B ordering, livelock-safe\n";
    std::cout << "                   blocked - All A then all B (can livelock)\n";
    std::cout << "                   output_stationary - C stays in accumulators\n";
    std::cout << "  --trace <file> Export Chrome trace to file\n";
    std::cout << "  --validate     Run schedule validation before execution\n";
    std::cout << "  --l3-buffers <n>  L3 buffer count (default: 32)\n";
    std::cout << "  --livelock <n>    Livelock threshold cycles (default: 10000)\n";
    std::cout << "  -h, --help     Show this help\n";
}

int main(int argc, char* argv[]) {
    // ========================================
    // Default configuration
    // ========================================
    size_t M = 64, N = 64, K = 64;
    size_t Ti = 16, Tj = 16, Tk = 16;
    MatMulScheduleGenerator::Strategy strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;
    std::string strategy_name = "INTERLEAVED_AB";
    std::string trace_file;
    bool validate = false;
    size_t l3_buffers = 32;
    size_t livelock_threshold = 10000;

    // ========================================
    // Parse arguments
    // ========================================
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            M = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            K = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--Ti") == 0 && i + 1 < argc) {
            Ti = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--Tj") == 0 && i + 1 < argc) {
            Tj = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--Tk") == 0 && i + 1 < argc) {
            Tk = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "interleaved") {
                strategy = MatMulScheduleGenerator::Strategy::INTERLEAVED_AB;
                strategy_name = "INTERLEAVED_AB";
            } else if (s == "blocked") {
                strategy = MatMulScheduleGenerator::Strategy::BLOCKED_AB;
                strategy_name = "BLOCKED_AB";
            } else if (s == "output_stationary") {
                strategy = MatMulScheduleGenerator::Strategy::OUTPUT_STATIONARY;
                strategy_name = "OUTPUT_STATIONARY";
            } else {
                std::cerr << "Unknown strategy: " << s << "\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_file = argv[++i];
        } else if (strcmp(argv[i], "--validate") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "--l3-buffers") == 0 && i + 1 < argc) {
            l3_buffers = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "--livelock") == 0 && i + 1 < argc) {
            livelock_threshold = std::stoull(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate dimensions
    if (M % Ti != 0 || N % Tj != 0 || K % Tk != 0) {
        std::cerr << "Error: Dimensions must be evenly divisible by tile sizes\n";
        std::cerr << "  M=" << M << " % Ti=" << Ti << " = " << (M % Ti) << "\n";
        std::cerr << "  N=" << N << " % Tj=" << Tj << " = " << (N % Tj) << "\n";
        std::cerr << "  K=" << K << " % Tk=" << Tk << " = " << (K % Tk) << "\n";
        return 1;
    }

    // ========================================
    // Print configuration
    // ========================================
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     CSP Concurrent Timing Model - MatMul Performance Test            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Problem: C[" << M << "," << N << "] = A[" << M << "," << K
              << "] x B[" << K << "," << N << "]\n";
    std::cout << "Tiles:   " << (M/Ti) << " x " << (N/Tj) << " x " << (K/Tk)
              << " = " << (M/Ti)*(N/Tj)*(K/Tk) << " tile operations\n";
    std::cout << "Tile size: " << Ti << " x " << Tj << " x " << Tk << "\n";
    std::cout << "Strategy: " << strategy_name << "\n";
    std::cout << "FLOPs:   " << 2ULL * M * N * K << "\n\n";

    // ========================================
    // Generate CSP schedule
    // ========================================
    std::cout << "Generating CSP schedule...\n";

    MatMulScheduleGenerator::Config config;
    config.M = static_cast<Size>(M);
    config.N = static_cast<Size>(N);
    config.K = static_cast<Size>(K);
    config.Ti = static_cast<Size>(Ti);
    config.Tj = static_cast<Size>(Tj);
    config.Tk = static_cast<Size>(Tk);
    config.strategy = strategy;

    // Set matrix base addresses in DRAM (for trace display)
    // A matrix at 0x0000'1000, B at 0x0010'0000, C at 0x0020'0000
    config.a_base = 0x00001000;
    config.b_base = 0x00100000;
    config.c_base = 0x00200000;

    MatMulScheduleGenerator generator(config);
    auto schedule = generator.generate();

    std::cout << "  Schedule: " << schedule.metadata.name << "\n";
    std::cout << "  Total operations: " << schedule.size() << "\n";
    std::cout << "  Operation breakdown:\n";
    std::cout << "    LOAD:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::LOAD) << "\n";
    std::cout << "    MOVE:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::MOVE) << "\n";
    std::cout << "    FEED:      " << std::setw(6) << schedule.count_ops(ScheduleOpType::FEED) << "\n";
    std::cout << "    COMPUTE:   " << std::setw(6) << schedule.count_ops(ScheduleOpType::COMPUTE) << "\n";
    std::cout << "    DRAIN:     " << std::setw(6) << schedule.count_ops(ScheduleOpType::DRAIN) << "\n";
    std::cout << "    WRITEBACK: " << std::setw(6) << schedule.count_ops(ScheduleOpType::WRITEBACK) << "\n";
    std::cout << "    STORE:     " << std::setw(6) << schedule.count_ops(ScheduleOpType::STORE) << "\n";

    // ========================================
    // Analyze schedule for livelock safety
    // ========================================
    std::cout << "\nLivelock analysis:\n";
    auto analysis = ScheduleAnalysis::analyze(schedule);
    std::cout << "  Max consecutive A ops: " << analysis.max_consecutive_a << "\n";
    std::cout << "  Max consecutive B ops: " << analysis.max_consecutive_b << "\n";
    std::cout << "  Interleaved: " << (analysis.is_interleaved ? "YES" : "NO") << "\n";
    std::cout << "  Livelock-safe: " << (is_livelock_safe(schedule) ? "YES" : "NO") << "\n";

    // ========================================
    // Validate schedule (optional)
    // ========================================
    if (validate) {
        std::cout << "\nValidating schedule...\n";
        auto validation = validate_schedule(schedule);
        std::cout << "  Result: " << (validation.valid ? "PASSED" : "FAILED") << "\n";
        std::cout << "  Errors: " << validation.count_errors() << "\n";
        std::cout << "  Warnings: " << validation.count_warnings() << "\n";

        if (!validation.valid) {
            std::cout << "\nValidation issues:\n";
            for (const auto& issue : validation.issues) {
                std::cout << "  " << issue.to_string() << "\n";
            }
            return 1;
        }
    }

    // ========================================
    // Configure ConcurrentTimingExecutor
    // ========================================
    std::cout << "\nConfiguring ConcurrentTimingExecutor...\n";

    ConcurrentTimingExecutor::Config exec_config;
    exec_config.num_memory_controllers = 1;  // Single MC with correct command bus constraint
    exec_config.l3_buffer_count = l3_buffers;
    exec_config.num_block_movers = 4;
    exec_config.l2_bank_count = 64;
    exec_config.num_row_streamers = 2;
    exec_config.num_col_streamers = 2;
    exec_config.max_cycles = 10000000;  // 10M cycles max
    exec_config.enable_livelock_detection = true;
    exec_config.livelock_threshold = livelock_threshold;

    std::cout << "  Memory controllers: " << exec_config.num_memory_controllers << "\n";
    std::cout << "  L3 buffers: " << exec_config.l3_buffer_count << "\n";
    std::cout << "  BlockMovers: " << exec_config.num_block_movers << "\n";
    std::cout << "  L2 banks: " << exec_config.l2_bank_count << "\n";
    std::cout << "  Row streamers: " << exec_config.num_row_streamers << "\n";
    std::cout << "  Col streamers: " << exec_config.num_col_streamers << "\n";

    ConcurrentTimingExecutor executor(exec_config);

    // ========================================
    // Execute schedule
    // ========================================
    std::cout << "\nExecuting CSP schedule...\n";

    ScheduleExecutor sched_exec(executor);
    auto result = sched_exec.execute(schedule);

    std::cout << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Execution Results\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "  Status: " << (result.success ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Total cycles: " << result.total_cycles << "\n";
    std::cout << "  Ops completed: " << result.ops_completed << " / " << result.ops_total << "\n";

    if (result.livelock_detected) {
        std::cout << "  WARNING: Livelock detected!\n";
    }

    // ========================================
    // Print statistics
    // ========================================
    auto stats = executor.get_statistics();
    std::cout << "\nResource Utilization:\n";
    std::cout << "  DMA utilization:     " << std::fixed << std::setprecision(1)
              << (stats.dma_utilization() * 100) << "%\n";
    std::cout << "  BlockMover util:     " << (stats.bm_utilization() * 100) << "%\n";
    std::cout << "  Streamer util:       " << (stats.str_utilization() * 100) << "%\n";

    std::cout << "\nTile Throughput:\n";
    double total = static_cast<double>(stats.total_cycles);
    if (total > 0) {
        std::cout << "  DMA tiles/cycle:     " << std::setprecision(4)
                  << static_cast<double>(stats.tiles_loaded + stats.tiles_stored) / total << "\n";
        std::cout << "  BM tiles/cycle:      "
                  << static_cast<double>(stats.tiles_moved + stats.tiles_writeback) / total << "\n";
        std::cout << "  STR tiles/cycle:     "
                  << static_cast<double>(stats.tiles_fed + stats.tiles_drained) / total << "\n";
    }

    std::cout << "\nStall Analysis:\n";
    std::cout << "  DMA credit stalls:   " << stats.dma_credit_stalls << " cycles\n";
    std::cout << "  BM tag stalls:       " << stats.bm_tag_stalls << " cycles\n";
    std::cout << "  BM credit stalls:    " << stats.bm_credit_stalls << " cycles\n";
    std::cout << "  STR tag stalls:      " << stats.str_tag_stalls << " cycles\n";
    std::cout << "  STR credit stalls:   " << stats.str_credit_stalls << " cycles\n";

    // ========================================
    // Export trace (optional)
    // ========================================
    if (!trace_file.empty()) {
        std::cout << "\nExporting Chrome trace to: " << trace_file << "\n";
        executor.export_chrome_trace(trace_file);
        std::cout << "View at: https://ui.perfetto.dev\n";
    }

    // ========================================
    // Performance summary
    // ========================================
    std::cout << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Performance Summary\n";
    std::cout << std::string(70, '=') << "\n";

    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double cycles = static_cast<double>(result.total_cycles);
    double flops_per_cycle = flops / cycles;

    std::cout << "  FLOPs: " << std::scientific << std::setprecision(2) << flops << "\n";
    std::cout << "  Cycles: " << std::fixed << std::setprecision(0) << cycles << "\n";
    std::cout << "  FLOP/cycle: " << std::setprecision(2) << flops_per_cycle << "\n";

    // Assuming 1 GHz clock for illustrative purposes
    double clock_ghz = 1.0;
    double gflops = flops_per_cycle * clock_ghz;
    std::cout << "  GFLOP/s @ " << clock_ghz << " GHz: " << gflops << "\n";

    std::cout << std::string(70, '=') << "\n\n";

    return result.success ? 0 : 1;
}
