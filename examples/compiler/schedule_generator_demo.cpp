/**
 * @file schedule_generator_demo.cpp
 * @brief Demonstration of ScheduleGenerator for matrix multiplication
 *
 * Shows how to:
 * - Optimize tile sizes with TileOptimizer
 * - Generate execution schedules with ScheduleGenerator
 * - Compare different scheduling strategies
 * - Analyze performance characteristics
 */

#include <sw/compiler/tile_optimizer.hpp>
#include <sw/compiler/schedule_generator.hpp>
#include <iostream>
#include <iomanip>
#include <string>

using namespace sw::kpu::compiler;

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void demonstrate_basic_schedule(Size M, Size N, Size K) {
    print_header("Basic Schedule Generation: C[" + std::to_string(M) + "," +
                 std::to_string(N) + "] = A[" + std::to_string(M) + "," +
                 std::to_string(K) + "] × B[" + std::to_string(K) + "," +
                 std::to_string(N) + "]");

    // Step 1: Optimize tile sizes
    std::cout << "\nStep 1: Optimizing tile sizes...\n";
    TileOptimizer optimizer;
    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "  Tile configuration: " << config.Ti << " × "
              << config.Tj << " × " << config.Tk << "\n";
    std::cout << "  Reuse factors: A=" << config.reuse_A << "x, B="
              << config.reuse_B << "x, C=" << config.reuse_C << "x\n";
    std::cout << "  DRAM traffic: " << (static_cast<double>(config.dram_accesses) / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  Arithmetic intensity: " << config.arithmetic_intensity
              << " FLOPs/byte\n";

    // Step 2: Generate execution schedule
    std::cout << "\nStep 2: Generating execution schedule...\n";
    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config,
                                       ScheduleGenerator::Strategy::SEQUENTIAL);

    // Step 3: Display schedule statistics
    generator.print_schedule(schedule, false);
}

void demonstrate_detailed_schedule(Size M, Size N, Size K) {
    print_header("Detailed Schedule Analysis");

    TileOptimizer optimizer;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    // Print verbose schedule with command timeline
    generator.print_schedule(schedule, true);
}

void compare_strategies(Size M, Size N, Size K) {
    print_header("Scheduling Strategy Comparison");

    std::cout << "\nMatrix: C[" << M << "," << N << "] = "
              << "A[" << M << "," << K << "] × B[" << K << "," << N << "]\n";

    // Optimize tiles once
    TileOptimizer optimizer;
    auto config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "\nTile configuration: " << config.Ti << " × "
              << config.Tj << " × " << config.Tk << "\n";

    // Compare different scheduling strategies
    ScheduleGenerator generator;

    std::cout << "\nGenerating schedules with different strategies:\n";

    auto seq_schedule = generator.generate(M, N, K, config,
                                           ScheduleGenerator::Strategy::SEQUENTIAL);
    std::cout << "  Sequential strategy:      done\n";

    auto db_schedule = generator.generate(M, N, K, config,
                                          ScheduleGenerator::Strategy::DOUBLE_BUFFERED);
    std::cout << "  Double-buffered strategy: done\n";

    auto pipe_schedule = generator.generate(M, N, K, config,
                                            ScheduleGenerator::Strategy::FULLY_PIPELINED);
    std::cout << "  Fully pipelined strategy: done\n";

    // Compare results
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << std::setw(20) << "Strategy"
              << std::setw(12) << "Cycles"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "DRAM (MB)"
              << std::setw(11) << "AI (F/B)\n";
    std::cout << std::string(70, '-') << "\n";

    auto print_row = [](const char* name, const ScheduleGenerator::Schedule& s) {
        std::cout << std::setw(20) << std::left << name
                  << std::setw(12) << std::right << s.total_cycles
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << s.estimated_time_ms
                  << std::setw(15) << std::setprecision(2)
                  << (static_cast<double>(s.total_dram_bytes) / (1024.0 * 1024.0))
                  << std::setw(11) << std::setprecision(1)
                  << s.arithmetic_intensity << "\n";
    };

    print_row("Sequential", seq_schedule);
    print_row("Double-buffered", db_schedule);
    print_row("Fully pipelined", pipe_schedule);

    std::cout << std::string(70, '-') << "\n";

    // Analyze speedup
    double speedup = static_cast<double>(seq_schedule.total_cycles) / static_cast<double>(pipe_schedule.total_cycles);
    std::cout << "\nPipelining speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x (" << seq_schedule.total_cycles << " → "
              << pipe_schedule.total_cycles << " cycles)\n";

    std::cout << "\nOptimization benefits:\n";
    std::cout << "  • Sequential: All operations execute in strict order\n";
    std::cout << "  • Double-buffered: Load next tile while computing current tile\n";
    std::cout << "  • Fully pipelined: Multiple tiles in flight simultaneously\n";
    std::cout << "    - BlockMove, Stream, and Compute overlap across tiles\n";
    std::cout << "    - Achieves " << std::setprecision(1) << speedup
              << "x improvement for this workload\n";

    // Show command timelines to visualize the difference
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "DETAILED COMMAND TIMELINES\n";
    std::cout << std::string(80, '=') << "\n";

    const char* type_names[] = {
        "DMA", "BlockMove", "Stream→L1", "Stream→L2", "Compute", "Barrier"
    };

    auto print_timeline = [&](const char* strategy_name, const ScheduleGenerator::Schedule& s) {
        std::cout << "\n" << strategy_name << " (" << s.commands.size() << " commands, "
                  << s.total_cycles << " cycles):\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << " #  | Type       | Label                      | Start  → End    (Dur) | Buf\n";
        std::cout << std::string(80, '-') << "\n";

        for (size_t i = 0; i < s.commands.size(); ++i) {
            const auto& cmd = s.commands[i];
            std::string label = cmd.tile_label;
            if (label.length() > 26) label = label.substr(0, 23) + "...";

            std::cout << std::setw(3) << i << " | "
                      << std::setw(10) << std::left << type_names[static_cast<int>(cmd.type)]
                      << " | " << std::setw(26) << label
                      << " | " << std::setw(6) << std::right << cmd.start_cycle
                      << " → " << std::setw(6) << cmd.end_cycle
                      << " (" << std::setw(4) << cmd.latency_cycles << ")";
            if (cmd.buffer_id >= 0) {
                std::cout << " | " << cmd.buffer_id;
            } else {
                std::cout << " | -";
            }
            std::cout << "\n";
        }
    };

    print_timeline("Sequential", seq_schedule);
    print_timeline("Double-buffered", db_schedule);
    print_timeline("Fully pipelined", pipe_schedule);

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "KEY OBSERVATION:\n";
    std::cout << "In Sequential:   Commands execute one after another\n";
    std::cout << "In Pipelined:    Multiple commands overlap (notice same start cycles)\n";
    std::cout << "                 → This is why pipelined finishes in " << pipe_schedule.total_cycles
              << " vs " << seq_schedule.total_cycles << " cycles\n";
    std::cout << std::string(70, '=') << "\n";
}

void analyze_different_shapes() {
    print_header("Analysis Across Different Matrix Shapes");

    struct TestCase {
        Size M, N, K;
        const char* description;
    };

    TestCase cases[] = {
        {128, 128, 128, "Small square"},
        {512, 512, 512, "Medium square"},
        {1024, 1024, 1024, "Large square"},
        {2048, 128, 512, "Tall skinny"},
        {128, 2048, 512, "Short wide"},
        {512, 512, 4096, "Large K dimension"}
    };

    TileOptimizer optimizer;
    ScheduleGenerator generator;

    std::cout << "\n" << std::string(90, '-') << "\n";
    std::cout << std::setw(20) << "Shape"
              << std::setw(15) << "Dimensions"
              << std::setw(15) << "Tiles"
              << std::setw(12) << "Commands"
              << std::setw(12) << "Cycles"
              << std::setw(16) << "Time (ms)\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& tc : cases) {
        auto config = optimizer.optimize(tc.M, tc.N, tc.K);
        auto schedule = generator.generate(tc.M, tc.N, tc.K, config);

        std::string dims = std::to_string(tc.M) + "×" +
                          std::to_string(tc.N) + "×" +
                          std::to_string(tc.K);

        std::string tiles = std::to_string(config.Ti) + "×" +
                           std::to_string(config.Tj) + "×" +
                           std::to_string(config.Tk);

        std::cout << std::setw(20) << std::left << tc.description
                  << std::setw(15) << dims
                  << std::setw(15) << tiles
                  << std::setw(12) << std::right << schedule.commands.size()
                  << std::setw(12) << schedule.total_cycles
                  << std::setw(16) << std::fixed << std::setprecision(3)
                  << schedule.estimated_time_ms << "\n";
    }

    std::cout << std::string(90, '-') << "\n";
}

void demonstrate_memory_hierarchy() {
    print_header("Memory Hierarchy Breakdown");

    Size M = 512, N = 512, K = 512;

    TileOptimizer optimizer;
    auto config = optimizer.optimize(M, N, K);

    ScheduleGenerator generator;
    auto schedule = generator.generate(M, N, K, config);

    std::cout << "\nMatrix: " << M << "×" << N << "×" << K << "\n";
    std::cout << "Tiles:  " << config.Ti << "×" << config.Tj << "×" << config.Tk << "\n";

    std::cout << "\nMemory Traffic by Level:\n";
    std::cout << "  DRAM (GDDR6): " << (static_cast<double>(schedule.total_dram_bytes) / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  L3 Cache:     " << (static_cast<double>(schedule.total_l3_bytes) / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  L2 Cache:     " << (static_cast<double>(schedule.total_l2_bytes) / (1024.0 * 1024.0))
              << " MB\n";

    std::cout << "\nCommand Breakdown:\n";
    std::cout << "  DMA transfers (DRAM↔L3):  " << schedule.num_dma_transfers << "\n";
    std::cout << "  Block moves (L3↔L2):      " << schedule.num_block_moves << "\n";
    std::cout << "  Streams (L2↔L1):          " << schedule.num_streams << "\n";
    std::cout << "  Compute operations:       " << schedule.num_computes << "\n";

    std::cout << "\nMemory Allocations:\n";
    for (const auto& alloc : schedule.allocations) {
        const char* level_names[] = {
            "HOST_DDR", "GDDR6", "L3", "L2", "L1", "PE"
        };
        std::cout << "  " << std::setw(12) << std::left << alloc.label
                  << ": " << std::setw(8) << level_names[static_cast<int>(alloc.level)]
                  << " @ 0x" << std::hex << std::setfill('0') << std::setw(8)
                  << alloc.base_addr << std::dec << std::setfill(' ')
                  << " (" << (static_cast<double>(alloc.size_bytes) / 1024.0) << " KB)\n";
    }
}

int main() {
    std::cout << "ScheduleGenerator Demonstration\n";
    std::cout << "================================\n\n";
    std::cout << "This demo shows how to generate hardware execution schedules\n";
    std::cout << "for tiled matrix multiplication on the KPU architecture.\n";

    // Demo 1: Basic schedule for small matrix
    demonstrate_basic_schedule(256, 256, 256);

    // Demo 2: Detailed schedule with command timeline
    std::cout << "\n\nPress Enter to see detailed schedule...";
    std::cin.get();
    demonstrate_detailed_schedule(128, 128, 128);

    // Demo 3: Strategy comparison (use small matrix for readable timeline)
    std::cout << "\n\nPress Enter to compare scheduling strategies...";
    std::cin.get();
    compare_strategies(128, 128, 128);

    // Demo 4: Different matrix shapes
    std::cout << "\n\nPress Enter to analyze different matrix shapes...";
    std::cin.get();
    analyze_different_shapes();

    // Demo 5: Memory hierarchy analysis
    std::cout << "\n\nPress Enter to see memory hierarchy breakdown...";
    std::cin.get();
    demonstrate_memory_hierarchy();

    std::cout << "\n\n";
    print_header("Demonstration Complete");
    std::cout << "\nKey Takeaways:\n";
    std::cout << "1. TileOptimizer finds optimal tile sizes for cache reuse\n";
    std::cout << "2. ScheduleGenerator converts tiles into hardware commands\n";
    std::cout << "3. Schedules include DMA, BlockMove, Stream, and Compute operations\n";
    std::cout << "4. Memory hierarchy is explicitly managed across 5 levels\n";
    std::cout << "5. Pipelining achieves 4-5x speedup by overlapping operations\n";
    std::cout << "6. Tile notation: A_tile[ti,tk], B_tile[tk,tj], C_tile[ti,tj]\n";
    std::cout << "   where ti=M-tile, tj=N-tile, tk=K-tile indices\n";

    return 0;
}
