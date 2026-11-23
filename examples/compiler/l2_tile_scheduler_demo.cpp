/**
 * @file l2_tile_scheduler_demo.cpp
 * @brief Demonstration of L2 tile scheduler for KPU matrix multiplication
 *
 * This demo shows how the L2 tile scheduler manages tile allocations and
 * orders the L3 -> L2 tile moves to minimize DRAM traffic.
 */

#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace sw::kpu::compiler;

void demo_small_matmul() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                              DEMO 1                                ║\n";
    std::cout << "║            Small Domain of Computation (256x256x256)               ║\n";
    std::cout << "║      Matrix (256x256) @ Matrix (256x256) => Matrix (256x256)       ║\n";        
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Matrix dimensions
    Size M = 256, N = 256, K = 256;

    // Create tile optimizer
    TileOptimizer optimizer;
    auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti = " << tile_config.Ti
              << ", Tj = " << tile_config.Tj
              << ", Tk = " << tile_config.Tk << "\n";
    std::cout << "  Reuse A: " << tile_config.reuse_A
              << ", Reuse B: " << tile_config.reuse_B
              << ", Reuse C: " << tile_config.reuse_C << "\n";
    std::cout << "\n";

    // Create L2 tile scheduler
    L2TileScheduler scheduler;
    auto schedule = scheduler.generate_schedule(
        M, N, K, tile_config,
        L2TileScheduler::ReplacementPolicy::LRU,
        L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

    // Print schedule
    scheduler.print_schedule(schedule, true);
}

void demo_medium_matmul() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                               DEMO 2                               ║\n";
    std::cout << "║             Medium Domain of Computation (512x512x512)             ║\n";
    std::cout << "║      Matrix (512x512) @ Matrix (512x512) => Matrix (512x512)       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    Size M = 512, N = 512, K = 512;

    TileOptimizer optimizer;
    auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti = " << tile_config.Ti
              << ", Tj = " << tile_config.Tj
              << ", Tk = " << tile_config.Tk << "\n";
    std::cout << "\n";

    L2TileScheduler scheduler;
    auto schedule = scheduler.generate_schedule(
        M, N, K, tile_config,
        L2TileScheduler::ReplacementPolicy::LRU,
        L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

    scheduler.print_schedule(schedule, true);
}

void demo_large_matmul() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                DEMO 3                              ║\n";
    std::cout << "║           Large Domain of computation (1024x1024x1024)             ║\n";
    std::cout << "║   Matrix (1024x1024) @ Matrix (1024x1024) => Matrix (1024x1024)    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    Size M = 1024, N = 1024, K = 1024;

    TileOptimizer optimizer;
    auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti = " << tile_config.Ti
              << ", Tj = " << tile_config.Tj
              << ", Tk = " << tile_config.Tk << "\n";
    std::cout << "\n";

    L2TileScheduler scheduler;
    auto schedule = scheduler.generate_schedule(
        M, N, K, tile_config,
        L2TileScheduler::ReplacementPolicy::LRU,
        L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

    scheduler.print_schedule(schedule, false);

    std::cout << "\nDetailed L2 State:\n";
    scheduler.print_l2_state(schedule);

    std::cout << "\nLoad Sequence (first 30 entries):\n";
    scheduler.print_load_sequence(schedule, 30);

    std::cout << "\nReuse Statistics:\n";
    scheduler.print_reuse_stats(schedule);
}

void demo_rectangular_matmul() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                               DEMO 4                               ║\n";
    std::cout << "║          Rectangular Domain of Computation (2048x128x512)          ║\n";
    std::cout << "║     Matrix (2048x512) @ Matrix (512x128) => Matrix (2048x128)      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    Size M = 2048, N = 128, K = 512;

    TileOptimizer optimizer;
    auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::HEURISTIC_HYBRID);

    std::cout << "Tile Configuration:\n";
    std::cout << "  Ti = " << tile_config.Ti
              << ", Tj = " << tile_config.Tj
              << ", Tk = " << tile_config.Tk << "\n";
    std::cout << "\n";

    L2TileScheduler scheduler;
    auto schedule = scheduler.generate_schedule(
        M, N, K, tile_config,
        L2TileScheduler::ReplacementPolicy::LRU,
        L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

    scheduler.print_schedule(schedule, false);
}

void demo_comparison() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          DEMO 5: L2 Capacity Comparison (Various Sizes)            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    Size M = 1024, N = 1024, K = 1024;

    TileOptimizer optimizer;
    auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

    std::cout << "Comparing L2 schedules for " << M << "x" << N << "x" << K << " matrix:\n";
    std::cout << "\n";

    std::vector<Size> l2_capacities = {128, 256, 512};

    std::cout << "  ┌────────────────┬───────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "  │  L2 Capacity   │  Total Loads  │ Repeat Moves │   L2 Reuse   │   L3 Reuse   │\n";
    std::cout << "  ├────────────────┼───────────────┼──────────────┼──────────────┼──────────────┤\n";

    for (Size capacity : l2_capacities) {
        // Modify memory hierarchy
        auto mem = optimizer.memory_hierarchy();
        // Note: We can't directly change max_l2_slots, but we can increase L2 size
        mem.L2_size = capacity * 1024; // Increase L2 bank size

        TileOptimizer opt_custom(mem);
        L2TileScheduler scheduler(mem);

        auto schedule = scheduler.generate_schedule(
            M, N, K, tile_config,
            L2TileScheduler::ReplacementPolicy::LRU,
            L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

        std::cout << "  │ " << std::setw(11) << capacity << " KB │ "
                  << std::setw(13) << schedule.total_loads << " │ "
                  << std::setw(12) << schedule.reloads << " │ "
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << schedule.l2_hit_rate << "% │ "
                  << std::setw(11) << schedule.l3_hit_rate << "% │\n";
    }

    std::cout << "  └────────────────┴───────────────┴──────────────┴──────────────┴──────────────┘\n";
    std::cout << "\n";
}

void demo_systolic_array_sizes() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       DEMO 6: Impact of Systolic Array Size (512x512x512)          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    Size M = 512, N = 512, K = 512;

    std::vector<Size> systolic_sizes = {8, 16, 32, 64};

    std::cout << "Comparing different systolic array sizes:\n";
    std::cout << "\n";

    std::cout << "  ┌──────────────┬───────────────┬────────────┬───────────────┬──────────────┐\n";
    std::cout << "  │  Array Size  │ Ti x Tj x Tk  │ Tile Count │  Total Loads  │   L2 Reuse   │\n";
    std::cout << "  ├──────────────┼───────────────┼────────────┼───────────────┼──────────────┤\n";

    for (Size sys_size : systolic_sizes) {
        TileOptimizer::MemoryHierarchy mem;
        mem.systolic_rows = sys_size;
        mem.systolic_cols = sys_size;

        TileOptimizer optimizer(mem);
        auto tile_config = optimizer.optimize(M, N, K, TileOptimizer::Strategy::ANALYTICAL);

        L2TileScheduler scheduler(mem, sys_size);
        auto schedule = scheduler.generate_schedule(
            M, N, K, tile_config,
            L2TileScheduler::ReplacementPolicy::LRU,
            L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);

        Size total_tiles = schedule.num_tile_rows_A * schedule.num_tile_cols_A +
                          schedule.num_tile_rows_B * schedule.num_tile_cols_B +
                          schedule.num_tile_rows_C * schedule.num_tile_cols_C;

        std::cout << "  │ " << std::setw(4) << sys_size << " x" << std::setw(4) << sys_size
                  << "   │ "
                  << std::setw(3) << tile_config.Ti << " x"
                  << std::setw(3) << tile_config.Tj << " x"
                  << std::setw(3) << tile_config.Tk << " │ "
                  << std::setw(10) << total_tiles << " │ "
                  << std::setw(13) << schedule.total_loads << " │ "
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << schedule.l2_hit_rate << "% │\n";
    }

    std::cout << "  └──────────────┴───────────────┴────────────┴───────────────┴──────────────┘\n";
    std::cout << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█                    L2 TILE SCHEDULER DEMONSTRATION                       █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  This demo showcases the L2 tile scheduler for KPU matrix multiply.      █\n";
    std::cout << "█  The scheduler manages tile allocations in the L2 banks and              █\n";
    std::cout << "█  orders the L3 -> L2 tile moves to minimize DRAM traffic.                █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";

    try {
        demo_small_matmul();
        demo_medium_matmul();
        demo_large_matmul();
        demo_rectangular_matmul();
        demo_comparison();
        demo_systolic_array_sizes();

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                        ALL DEMOS COMPLETED                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
