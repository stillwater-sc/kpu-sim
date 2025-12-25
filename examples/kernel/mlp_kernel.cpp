// MLP Kernel Demo
// Demonstrates the MLP Kernel with fused matmul + bias + activation
//
// This example shows how to:
// - Create MLP kernels with various activation functions
// - Configure bias handling
// - Compare different activation types
// - Inspect MLP kernel metadata and arguments
// - Execute MLP kernels using ConcurrentExecutor

/*
 MLP Kernel Demo (examples/kernel/mlp_kernel.cpp)

 The MLP kernel implements a fused operation: C = activation(A @ B + bias)

 This fusion provides significant performance benefits:
  - Single memory pass instead of 3 separate operations
  - 4x reduction in L2 memory traffic
  - Zero-copy inline processing during L1->L2 transfer

 Supported Activation Functions:
  - NONE:      Pass-through (just matmul + bias)
  - RELU:      max(0, x)
  - GELU:      x * 0.5 * (1 + erf(x/sqrt(2)))
  - SIGMOID:   1 / (1 + exp(-x))
  - TANH:      (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  - SILU:      x * sigmoid(x)
  - SOFTPLUS:  log(1 + exp(x))

  | Section            | Functionality                                          |
  |--------------------|--------------------------------------------------------|
  | 1. Simple Creation | Create kernel with Kernel::create_mlp() one-liner      |
  | 2. Activations     | Compare different activation functions                 |
  | 3. Bias Options    | MLP with and without bias                              |
  | 4. Metadata        | Inspect MLP kernel arguments                           |
  | 5. Execution       | Run MLP kernel on ConcurrentExecutor                   |
  | 6. Performance     | Compare activation types                               |

 Running the Demo

  ./build/examples/kernel/mlp_kernel

 Key Output Highlights

  - Fused operation: matmul + bias + activation in single pass
  - Vector Engine (VE) inline processing for zero additional latency
  - SFU-based activation with LUT + interpolation for accuracy
*/

#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <sw/kpu/components/sfu.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

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

// Get activation type name
std::string activation_name(ActivationType act) {
    switch (act) {
        case ActivationType::NONE: return "NONE";
        case ActivationType::RELU: return "RELU";
        case ActivationType::GELU: return "GELU";
        case ActivationType::SIGMOID: return "SIGMOID";
        case ActivationType::TANH: return "TANH";
        case ActivationType::SILU: return "SILU";
        case ActivationType::SOFTPLUS: return "SOFTPLUS";
        case ActivationType::LEAKY_RELU: return "LEAKY_RELU";
        default: return "UNKNOWN";
    }
}

int main() {
    std::cout << "KPU Simulator - MLP Kernel Demo\n";
    std::cout << "Fused Operation: C = activation(A @ B + bias)\n";
    separator();

    // =========================================================================
    // 1. Simple MLP Kernel Creation
    // =========================================================================
    separator("1. Simple MLP Kernel Creation");

    std::cout << "\nCreating a 512x1024x768 MLP kernel with GELU activation...\n";
    std::cout << "  Kernel kernel = Kernel::create_mlp(512, 1024, 768,\n";
    std::cout << "                                      ActivationType::GELU,\n";
    std::cout << "                                      true);  // has_bias\n\n";

    Kernel mlp_kernel = Kernel::create_mlp(512, 1024, 768,
                                            ActivationType::GELU,
                                            true);  // has_bias

    std::cout << "MLP Kernel created successfully!\n";
    std::cout << "  Valid:        " << (mlp_kernel.is_valid() ? "yes" : "no") << "\n";
    std::cout << "  Operation:    " << kernel_op_type_name(mlp_kernel.op_type()) << "\n";
    std::cout << "  Data Type:    " << dtype_name(mlp_kernel.dtype()) << "\n";
    std::cout << "  Dimensions:   M=" << mlp_kernel.M() << ", N=" << mlp_kernel.N()
              << ", K=" << mlp_kernel.K() << "\n";
    std::cout << "  Activation:   " << activation_name(mlp_kernel.activation()) << "\n";
    std::cout << "  Has Bias:     " << (mlp_kernel.has_bias() ? "yes" : "no") << "\n";
    std::cout << "  Program Size: " << mlp_kernel.instruction_count() << " operations\n";

    // =========================================================================
    // 2. Activation Function Comparison
    // =========================================================================
    separator("2. Activation Function Comparison");

    std::cout << "\nCreating MLP kernels with different activation functions:\n\n";

    std::vector<ActivationType> activations = {
        ActivationType::NONE,
        ActivationType::RELU,
        ActivationType::GELU,
        ActivationType::SIGMOID,
        ActivationType::TANH,
        ActivationType::SILU
    };

    std::cout << std::left;
    std::cout << std::setw(12) << "Activation"
              << std::setw(15) << "Total FLOPs"
              << std::setw(15) << "Input Bytes"
              << std::setw(15) << "Output Bytes"
              << std::setw(12) << "Valid" << "\n";
    std::cout << std::string(69, '-') << "\n";

    for (ActivationType act : activations) {
        Kernel k = Kernel::create_mlp(256, 256, 256, act, true);

        std::cout << std::setw(12) << activation_name(act)
                  << std::setw(15) << format_count(k.total_flops())
                  << std::setw(15) << format_bytes(k.total_input_bytes())
                  << std::setw(15) << format_bytes(k.total_output_bytes())
                  << std::setw(12) << (k.is_valid() ? "yes" : "no") << "\n";
    }

    // =========================================================================
    // 3. Bias Options
    // =========================================================================
    separator("3. Bias Options");

    std::cout << "\nComparing MLP with and without bias:\n\n";

    Kernel mlp_with_bias = Kernel::create_mlp(256, 512, 128,
                                               ActivationType::RELU,
                                               true);  // has_bias

    Kernel mlp_no_bias = Kernel::create_mlp(256, 512, 128,
                                             ActivationType::RELU,
                                             false);  // no bias

    std::cout << "MLP with bias (4 arguments: A, B, bias, C):\n";
    std::cout << "  Arguments: " << mlp_with_bias.arguments().size() << "\n";
    std::cout << "  Input bytes: " << format_bytes(mlp_with_bias.total_input_bytes()) << "\n";
    for (const auto& arg : mlp_with_bias.arguments()) {
        std::string io = arg.is_output ? "output" : "input";
        std::cout << "    - " << arg.name << ": " << io << "\n";
    }

    std::cout << "\nMLP without bias (3 arguments: A, B, C):\n";
    std::cout << "  Arguments: " << mlp_no_bias.arguments().size() << "\n";
    std::cout << "  Input bytes: " << format_bytes(mlp_no_bias.total_input_bytes()) << "\n";
    for (const auto& arg : mlp_no_bias.arguments()) {
        std::string io = arg.is_output ? "output" : "input";
        std::cout << "    - " << arg.name << ": " << io << "\n";
    }

    // =========================================================================
    // 4. MLP Kernel Metadata and Arguments
    // =========================================================================
    separator("4. MLP Kernel Metadata and Arguments");

    std::cout << "\nDetailed argument inspection for 512x1024x768 MLP with bias:\n\n";

    Kernel kernel_md = Kernel::create_mlp(512, 1024, 768,
                                           ActivationType::GELU,
                                           true);

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

    std::cout << "\nKernel Summary:\n";
    std::cout << kernel_md.summary() << "\n";

    // =========================================================================
    // 5. MLP Kernel Execution
    // =========================================================================
    separator("5. MLP Kernel Execution");

    std::cout << "\nExecuting MLP kernel on ConcurrentExecutor...\n";

    // Create executor configuration
    isa::ResourceConfig resource_config;
    resource_config.num_memory_channels = 4;
    resource_config.num_block_movers = 8;
    resource_config.num_streamers = 16;

    isa::ConcurrentExecutor executor(resource_config);

    // Execute the MLP kernel's program
    Cycle cycles = executor.execute(mlp_kernel.program());

    std::cout << "Execution complete!\n";
    std::cout << "  Simulated Cycles: " << cycles << "\n";

    // Calculate throughput estimates (assuming 1GHz clock)
    double time_ms = cycles / 1e6;
    double gflops = (static_cast<double>(mlp_kernel.total_flops()) / 1e9) / (time_ms / 1000.0);

    std::cout << "  Estimated Time (@ 1GHz): " << std::fixed << std::setprecision(3)
              << time_ms << " ms\n";
    std::cout << "  Estimated Throughput: " << std::fixed << std::setprecision(1)
              << gflops << " GFLOPS\n";

    // =========================================================================
    // 6. Performance Comparison by Activation
    // =========================================================================
    separator("6. Performance Comparison by Activation");

    std::cout << "\nComparing MLP kernel performance with different activations:\n";
    std::cout << "(Fixed size: 1024x1024x1024, with bias)\n\n";

    std::cout << std::setw(12) << "Activation"
              << std::setw(10) << "DMA Ops"
              << std::setw(10) << "BM Ops"
              << std::setw(12) << "Volume"
              << std::setw(10) << "AI"
              << std::setw(12) << "Cycles" << "\n";
    std::cout << std::string(66, '-') << "\n";

    KernelCompiler compiler;

    for (ActivationType act : activations) {
        Kernel k = compiler.compile_mlp(1024, 1024, 1024, act, true);
        const CompilationStats& s = compiler.last_stats();
        Cycle c = executor.execute(k.program());

        std::cout << std::setw(12) << activation_name(act)
                  << std::setw(10) << s.operations.external_memory.count
                  << std::setw(10) << s.operations.l3_l2.count
                  << std::setw(12) << format_bytes(s.estimated_external_bytes)
                  << std::setw(10) << std::fixed << std::setprecision(1) << k.arithmetic_intensity()
                  << std::setw(12) << c << "\n";
    }

    // =========================================================================
    // 7. Fusion Benefits
    // =========================================================================
    separator("7. Fusion Benefits");

    std::cout << "\nMemory Traffic Comparison (1024x1024x1024):\n\n";

    Size M = 1024, N = 1024;
    Size elem_size = 4;  // FLOAT32

    Size matmul_output = M * N * elem_size;

    // Without fusion: 3 separate passes
    Size unfused_l2_traffic = matmul_output * 5;  // write + read(bias) + write + read(act) + write

    // With fusion: single pass through Vector Engine
    Size fused_l2_traffic = matmul_output;  // single write with inline bias+activation

    std::cout << "Without MLP fusion (3 separate operations):\n";
    std::cout << "  1. Matmul: A @ B -> temp1              (" << format_bytes(matmul_output) << " write)\n";
    std::cout << "  2. Bias:   temp1 + bias -> temp2       (" << format_bytes(matmul_output * 2) << " read+write)\n";
    std::cout << "  3. Activ:  activation(temp2) -> C      (" << format_bytes(matmul_output * 2) << " read+write)\n";
    std::cout << "  Total L2 traffic: " << format_bytes(unfused_l2_traffic) << "\n";

    std::cout << "\nWith MLP fusion (Vector Engine inline processing):\n";
    std::cout << "  Single pass: activation(A @ B + bias) -> C\n";
    std::cout << "  Total L2 traffic: " << format_bytes(fused_l2_traffic) << "\n";

    std::cout << "\nMemory traffic reduction: " << std::fixed << std::setprecision(1)
              << (static_cast<double>(unfused_l2_traffic) / fused_l2_traffic) << "x\n";

    separator();
    std::cout << "\nMLP kernel demo complete!\n";

    return 0;
}
