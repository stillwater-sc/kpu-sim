// Efficiency Diagnostic Test
// Analyze why efficiency is lower than expected

#include <catch2/catch_test_macros.hpp>

#include <sw/benchmark/benchmark.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <sw/compiler/kernel_compiler.hpp>

#include <iostream>
#include <iomanip>

using namespace sw::benchmark;
using namespace sw::kpu;
using namespace sw::kpu::isa;
using namespace sw::kpu::compiler;

TEST_CASE("64x64x64 Matmul Efficiency Diagnostic", "[diagnostic][efficiency]") {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "64x64x64 MATMUL EFFICIENCY DIAGNOSTIC\n";
    std::cout << "============================================================\n\n";

    // Compile the kernel
    KernelCompiler compiler;
    Kernel kernel = compiler.compile_matmul(64, 64, 64);
    const DMProgram& program = kernel.program();

    std::cout << "=== KERNEL INFO ===\n";
    std::cout << "Problem: " << program.M << "x" << program.N << "x" << program.K << "\n";
    std::cout << "Tiles:   " << program.Ti << "x" << program.Tj << "x" << program.Tk << "\n";
    std::cout << "L1_Ki:   " << program.L1_Ki << "\n";
    std::cout << "Instructions: " << program.instructions.size() << "\n\n";

    // Expected compute cycles (ideal)
    Size systolic_size = 16;  // 16x16 array
    Size ops_per_cycle = systolic_size * systolic_size * 2;  // 2 FMA ops
    uint64_t total_flops = 2ULL * program.M * program.N * program.K;
    Cycle ideal_compute_cycles = (total_flops + ops_per_cycle - 1) / ops_per_cycle;

    std::cout << "=== THEORETICAL ANALYSIS ===\n";
    std::cout << "Total FLOPs: " << total_flops << "\n";
    std::cout << "Systolic array: " << systolic_size << "x" << systolic_size << "\n";
    std::cout << "Ops per cycle: " << ops_per_cycle << "\n";
    std::cout << "Ideal compute cycles: " << ideal_compute_cycles << "\n\n";

    // Memory requirements
    Size elem_size = 4;  // float32
    Size a_bytes = program.M * program.K * elem_size;
    Size b_bytes = program.K * program.N * elem_size;
    Size c_bytes = program.M * program.N * elem_size;
    Size total_bytes = a_bytes + b_bytes + c_bytes;

    std::cout << "=== MEMORY REQUIREMENTS ===\n";
    std::cout << "A matrix: " << a_bytes / 1024.0 << " KB\n";
    std::cout << "B matrix: " << b_bytes / 1024.0 << " KB\n";
    std::cout << "C matrix: " << c_bytes / 1024.0 << " KB\n";
    std::cout << "Total external: " << total_bytes / 1024.0 << " KB\n";
    std::cout << "Arithmetic Intensity: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(total_flops) / total_bytes) << " FLOP/byte\n\n";

    // Execute with ConcurrentExecutor
    ResourceConfig config;  // Default config
    ConcurrentExecutor executor(config);
    Cycle cycles = executor.execute(program);

    std::cout << "=== EXECUTION RESULTS ===\n";
    std::cout << "Total cycles (DMA timebase): " << cycles << "\n";
    std::cout << "Ideal compute cycles: " << ideal_compute_cycles << "\n";
    std::cout << "Overhead: " << ((static_cast<double>(cycles) / ideal_compute_cycles) - 1.0) * 100
              << "%\n\n";

    // Utilization stats
    auto util = executor.get_utilization();
    std::cout << "=== UTILIZATION ===\n";
    std::cout << "DMA:         " << std::fixed << std::setprecision(1)
              << (util.dma_utilization * 100) << "%\n";
    std::cout << "Block Mover: " << (util.block_mover_utilization * 100) << "%\n";
    std::cout << "Streamer:    " << (util.streamer_utilization * 100) << "%\n";
    std::cout << "Compute:     " << (util.compute_utilization * 100) << "%\n";
    std::cout << "Makespan:    " << util.makespan << " cycles\n\n";

    // Timeline visualization
    std::cout << "=== TIMELINE (first 120 chars) ===\n";
    std::cout << executor.generate_timeline(120) << "\n";

    // Cycle report
    std::cout << "=== OPERATION DETAILS ===\n";
    auto& ops = executor.get_all_operations();
    std::cout << "Total operations: " << ops.size() << "\n\n";

    // Group operations by type
    Cycle dma_total = 0, bm_total = 0, str_total = 0, comp_total = 0;
    size_t dma_count = 0, bm_count = 0, str_count = 0, comp_count = 0;

    for (const auto& op : ops) {
        Cycle dur = op.end_cycle - op.start_cycle;
        switch (op.resource.type) {
            case ResourceType::DMA_ENGINE:
                dma_total += dur; dma_count++; break;
            case ResourceType::BLOCK_MOVER:
                bm_total += dur; bm_count++; break;
            case ResourceType::STREAMER:
                str_total += dur; str_count++; break;
            case ResourceType::COMPUTE_FABRIC:
                comp_total += dur; comp_count++; break;
            default: break;
        }
    }

    std::cout << "Operation Breakdown:\n";
    std::cout << "  DMA:         " << dma_count << " ops, " << dma_total << " total cycles\n";
    std::cout << "  Block Mover: " << bm_count << " ops, " << bm_total << " total cycles\n";
    std::cout << "  Streamer:    " << str_count << " ops, " << str_total << " total cycles\n";
    std::cout << "  Compute:     " << comp_count << " ops, " << comp_total << " total cycles\n\n";

    // Print first few operations
    std::cout << "First 15 operations (sorted by start cycle):\n";
    std::cout << std::setw(6) << "Start" << std::setw(6) << "End"
              << std::setw(6) << "Dur" << "  " << "Resource" << "\n";
    std::cout << std::string(60, '-') << "\n";

    size_t limit = std::min(ops.size(), size_t(15));
    for (size_t i = 0; i < limit; ++i) {
        const auto& op = ops[i];
        std::cout << std::setw(6) << op.start_cycle
                  << std::setw(6) << op.end_cycle
                  << std::setw(6) << (op.end_cycle - op.start_cycle)
                  << "  " << op.resource.to_string()
                  << " " << op.label << "\n";
    }

    std::cout << "\n";

    // Analyze pipeline bubbles
    std::cout << "=== PIPELINE ANALYSIS ===\n";

    // Find compute operations and check for gaps
    std::vector<const ScheduledOp*> compute_ops;
    for (const auto& op : ops) {
        if (op.resource.type == ResourceType::COMPUTE_FABRIC) {
            compute_ops.push_back(&op);
        }
    }

    if (compute_ops.empty()) {
        std::cout << "WARNING: No compute operations found!\n";
    } else {
        std::cout << "Compute operations: " << compute_ops.size() << "\n";
        Cycle first_compute = compute_ops.front()->start_cycle;
        Cycle last_compute_end = compute_ops.back()->end_cycle;
        std::cout << "First compute starts: cycle " << first_compute << "\n";
        std::cout << "Last compute ends:    cycle " << last_compute_end << "\n";
        std::cout << "Compute span:         " << (last_compute_end - first_compute) << " cycles\n";

        // Calculate pipeline fill/drain
        std::cout << "Pipeline startup:     " << first_compute << " cycles (before first compute)\n";
        std::cout << "Pipeline drain:       " << (cycles - last_compute_end) << " cycles (after last compute)\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "END DIAGNOSTIC\n";
    std::cout << "============================================================\n\n";

    REQUIRE(cycles > 0);
}

TEST_CASE("Compare efficiency across sizes", "[diagnostic][efficiency]") {
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "EFFICIENCY COMPARISON ACROSS SIZES\n";
    std::cout << "============================================================\n\n";

    KernelCompiler compiler;
    ResourceConfig config;

    std::vector<Size> sizes = {64, 128, 256, 512, 1024};

    std::cout << std::setw(8) << "Size" << std::setw(12) << "Cycles"
              << std::setw(12) << "Ideal" << std::setw(10) << "Overhead"
              << std::setw(12) << "Comp Util" << std::setw(12) << "DMA Util" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (Size size : sizes) {
        Kernel kernel = compiler.compile_matmul(size, size, size);
        ConcurrentExecutor executor(config);
        Cycle cycles = executor.execute(kernel.program());

        // Ideal compute cycles
        Size systolic_size = 16;
        Size ops_per_cycle = systolic_size * systolic_size * 2;
        uint64_t total_flops = 2ULL * size * size * size;
        Cycle ideal = (total_flops + ops_per_cycle - 1) / ops_per_cycle;

        auto util = executor.get_utilization();

        double overhead = ((static_cast<double>(cycles) / ideal) - 1.0) * 100;

        std::cout << std::setw(8) << size
                  << std::setw(12) << cycles
                  << std::setw(12) << ideal
                  << std::setw(9) << std::fixed << std::setprecision(1) << overhead << "%"
                  << std::setw(11) << (util.compute_utilization * 100) << "%"
                  << std::setw(11) << (util.dma_utilization * 100) << "%" << "\n";
    }

    std::cout << "\n";
    REQUIRE(true);
}
