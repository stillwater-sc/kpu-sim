// Matrix Multiplication Kernel Demo
// Demonstrates the Kernel and KernelCompiler APIs for matmul operations
//
// This example shows how to:
// - Create matmul kernels with factory methods (simplest API)
// - Compile matmul kernels with custom options (more control)
// - Access kernel metadata and arguments
// - Inspect compilation statistics
// - Execute kernels using ConcurrentExecutor

/*
 Matrix Multiplication Kernel Demo (examples/kernel/matmul_kernel.cpp)

 The demo demonstrates the key capabilities of the Kernel API for matmul:

  | Section            | Functionality                                          |
  |--------------------|--------------------------------------------------------|
  | 1. Simple Creation | Create kernel with Kernel::create_matmul() one-liner   |
  | 2. Custom Compiler | Use KernelCompiler with custom tile sizes and options  |
  | 3. Metadata Access | Inspect kernel arguments, dimensions, and properties   |
  | 4. Statistics      | View compilation stats (time, tiles, instructions)     |
  | 5. Execution       | Run kernel on ConcurrentExecutor                       |
  | 6. Size Comparison | Compare kernels of different sizes                     |

 Running the Demo

  ./build/examples/kernel/matmul_kernel

 Key Output Highlights

  - Simple API: Kernel::create_matmul(M, N, K) handles all complexity
  - Auto-tiling: TileOptimizer selects optimal tile sizes automatically
  - Detailed stats: Instruction counts, memory estimates, arithmetic intensity
  - Execution: Integration with existing ConcurrentExecutor infrastructure
*/

#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>

#include <iostream>
#include <iomanip>
#include <vector>

using namespace sw::kpu;
using namespace sw::kpu::compiler;

// Print a separator line
void separator(const std::string& title = "") {
    if (title.empty()) {
        std::cout << std::string(70, '-') << "\n";
    } else {
        std::cout << "\n=== " << title << " " << std::string(65 - title.length(), '=') << "\n";
    }
}

// Format bytes nicely
std::string format_bytes(Size bytes) {
    if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

// Format large numbers with K/M suffix
std::string format_count(Size count) {
    if (count >= 1000000) {
        return std::to_string(count / 1000000) + "M";
    } else if (count >= 1000) {
        return std::to_string(count / 1000) + "K";
    }
    return std::to_string(count);
}

int main() {
    std::cout << "KPU Simulator - Matrix Multiplication Kernel Demo\n";
    separator();

    // =========================================================================
    // 1. Simple Kernel Creation (One-liner)
    // =========================================================================
    separator("1. Simple Kernel Creation");

    std::cout << "\nCreating a 1024x1024x1024 matrix multiplication kernel...\n";
    std::cout << "  Kernel kernel = Kernel::create_matmul(1024, 1024, 1024);\n\n";

    Kernel kernel = Kernel::create_matmul(1024, 1024, 1024);

    std::cout << "Kernel created successfully!\n";
    std::cout << "  Valid:        " << (kernel.is_valid() ? "yes" : "no") << "\n";
    std::cout << "  Operation:    " << kernel_op_type_name(kernel.op_type()) << "\n";
    std::cout << "  Data Type:    " << dtype_name(kernel.dtype()) << "\n";
    std::cout << "  Dimensions:   M=" << kernel.M() << ", N=" << kernel.N()
              << ", K=" << kernel.K() << "\n";
    std::cout << "  Tile Sizes:   Ti=" << kernel.Ti() << ", Tj=" << kernel.Tj()
              << ", Tk=" << kernel.Tk() << "\n";
    std::cout << "  Program Size: " << kernel.instruction_count() << " operations\n";

    // =========================================================================
    // 2. Kernel Creation with Compiler Options
    // =========================================================================
    separator("2. Custom Compilation with KernelCompiler");

    KernelCompiler compiler;

    std::cout << "\nUsing KernelCompiler for more control...\n";

    // 2a. Auto-optimized compilation
    std::cout << "\n[2a] Auto-optimized compilation:\n";
    Kernel kernel_auto = compiler.compile_matmul(512, 512, 512);
    std::cout << "  Tiles (auto): Ti=" << kernel_auto.Ti()
              << ", Tj=" << kernel_auto.Tj()
              << ", Tk=" << kernel_auto.Tk() << "\n";

    // 2b. Explicit tile sizes
    std::cout << "\n[2b] Explicit tile sizes:\n";
    Kernel kernel_explicit = compiler.compile_matmul(512, 512, 512, 64, 64, 128);
    std::cout << "  Tiles (explicit): Ti=" << kernel_explicit.Ti()
              << ", Tj=" << kernel_explicit.Tj()
              << ", Tk=" << kernel_explicit.Tk() << "\n";

    // 2c. Custom options
    std::cout << "\n[2c] Custom compile options:\n";
    CompileOptions opts;
    opts.Ti = 32;
    opts.Tj = 32;
    opts.Tk = 64;
    opts.dtype = DataType::FLOAT16;
    opts.double_buffer = true;

    Kernel kernel_custom = compiler.compile_matmul(256, 256, 256, opts);
    std::cout << "  Data type: " << dtype_name(kernel_custom.dtype()) << "\n";
    std::cout << "  Tiles: Ti=" << kernel_custom.Ti()
              << ", Tj=" << kernel_custom.Tj()
              << ", Tk=" << kernel_custom.Tk() << "\n";

    // =========================================================================
    // 3. Kernel Metadata and Arguments
    // =========================================================================
    separator("3. Kernel Metadata and Arguments");

    std::cout << "\nKernel Arguments for 512x1024x768 matmul:\n";
    Kernel kernel_md = Kernel::create_matmul(512, 1024, 768);

    std::cout << std::left;
    std::cout << std::setw(10) << "Name"
              << std::setw(12) << "Type"
              << std::setw(20) << "Shape"
              << std::setw(12) << "Size"
              << std::setw(10) << "I/O" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (const auto& arg : kernel_md.arguments()) {
        std::string shape_str;
        for (size_t i = 0; i < arg.shape.size(); ++i) {
            if (i > 0) shape_str += " x ";
            shape_str += std::to_string(arg.shape[i]);
        }

        std::cout << std::setw(10) << arg.name
                  << std::setw(12) << dtype_name(arg.dtype)
                  << std::setw(20) << shape_str
                  << std::setw(12) << format_bytes(arg.size_bytes)
                  << std::setw(10) << (arg.is_output ? "Output" : "Input")
                  << "\n";
    }

    std::cout << "\nMemory Summary:\n";
    std::cout << "  Total Input:  " << format_bytes(kernel_md.total_input_bytes()) << "\n";
    std::cout << "  Total Output: " << format_bytes(kernel_md.total_output_bytes()) << "\n";
    std::cout << "  Total FLOPs:  " << format_count(kernel_md.total_flops()) << "\n";
    std::cout << "  Arithmetic Intensity: " << std::fixed << std::setprecision(2)
              << kernel_md.arithmetic_intensity() << " FLOPs/byte\n";

    // =========================================================================
    // 4. Compilation Statistics
    // =========================================================================
    separator("4. Compilation Statistics");

    std::cout << "\nCompiling 1024x1024x1024 matmul and examining stats...\n\n";

    Kernel kernel_stats = compiler.compile_matmul(1024, 1024, 1024);
    const CompilationStats& stats = compiler.last_stats();

    std::cout << "Compile Time: " << std::fixed << std::setprecision(1)
              << stats.compile_time_us << " microseconds\n";

    std::cout << "\nTile Configuration:\n";
    std::cout << "  Auto-optimized: " << (stats.used_auto_tiling ? "yes" : "no") << "\n";
    std::cout << "  Selected: Ti=" << stats.selected_Ti
              << ", Tj=" << stats.selected_Tj
              << ", Tk=" << stats.selected_Tk
              << ", L1_Ki=" << stats.selected_L1_Ki << "\n";
    std::cout << "  Tile Grid: " << stats.num_m_tiles << " x "
              << stats.num_n_tiles << " x " << stats.num_k_tiles
              << " = " << stats.total_tiles << " tiles\n";

    // Display the new Operation Breakdown
    std::cout << "\n" << stats.operations.summary();

    std::cout << "\nMemory Traffic Estimates:\n";
    std::cout << "  External (DRAM): " << format_bytes(stats.estimated_external_bytes) << "\n";
    std::cout << "  L3 Cache:        " << format_bytes(stats.estimated_l3_bytes) << "\n";
    std::cout << "  L2 Cache:        " << format_bytes(stats.estimated_l2_bytes) << "\n";
    std::cout << "  Arithmetic Intensity: " << std::fixed << std::setprecision(2)
              << stats.estimated_arithmetic_intensity << " FLOPs/byte\n";

    std::cout << "\nDataflow Strategy: " << dataflow_strategy_name(stats.dataflow_used) << "\n";

    // =========================================================================
    // 5. Kernel Execution
    // =========================================================================
    separator("5. Kernel Execution");

    std::cout << "\nExecuting kernel on ConcurrentExecutor...\n";

    // Create executor configuration
    isa::ResourceConfig resource_config;
    resource_config.num_memory_channels = 4;
    resource_config.num_block_movers = 8;
    resource_config.num_streamers = 16;

    isa::ConcurrentExecutor executor(resource_config);

    // Execute the kernel's program
    Cycle cycles = executor.execute(kernel.program());

    std::cout << "Execution complete!\n";
    std::cout << "  Simulated Cycles: " << cycles << "\n";

    // Calculate throughput estimates (assuming 1GHz clock)
    double time_ms = cycles / 1e6;  // cycles / (1e9 cycles/sec) * 1e3 ms/sec
    double gflops = (static_cast<double>(kernel.total_flops()) / 1e9) / (time_ms / 1000.0);

    std::cout << "  Estimated Time (@ 1GHz): " << std::fixed << std::setprecision(3)
              << time_ms << " ms\n";
    std::cout << "  Estimated Throughput: " << std::fixed << std::setprecision(1)
              << gflops << " GFLOPS\n";

    // =========================================================================
    // 6. Size Comparison
    // =========================================================================
    separator("6. Size Comparison");

    std::cout << "\nComparing matmul kernels of different sizes:\n\n";

    std::cout << std::setw(12) << "Size"
              << std::setw(10) << "DMA Ops"
              << std::setw(10) << "BM Ops"
              << std::setw(10) << "STR Ops"
              << std::setw(12) << "Volume"
              << std::setw(10) << "AI"
              << std::setw(12) << "Cycles" << "\n";
    std::cout << std::string(76, '-') << "\n";

    std::vector<Size> sizes = {128, 256, 512, 1024, 2048};

    for (Size size : sizes) {
        Kernel k = compiler.compile_matmul(size, size, size);
        const CompilationStats& s = compiler.last_stats();
        Cycle c = executor.execute(k.program());

        std::string size_str = std::to_string(size) + "x" + std::to_string(size);
        std::cout << std::setw(12) << size_str
                  << std::setw(10) << s.operations.external_memory.count
                  << std::setw(10) << s.operations.l3_l2.count
                  << std::setw(10) << s.operations.l2_l1.count
                  << std::setw(12) << format_bytes(s.estimated_external_bytes)
                  << std::setw(10) << std::fixed << std::setprecision(1) << k.arithmetic_intensity()
                  << std::setw(12) << c << "\n";
    }

    separator();
    std::cout << "\nMatmul kernel demo complete!\n";

    return 0;
}
