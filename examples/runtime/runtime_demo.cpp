// Runtime Library Demo
// Demonstrates the KPURuntime and GraphExecutor APIs
//
// This example shows how to:
// - Create a KPURuntime for host-side orchestration
// - Allocate and manage device memory (malloc, free)
// - Transfer data between host and device (memcpy_h2d, memcpy_d2h)
// - Launch kernels with explicit memory arguments
// - Use GraphExecutor for high-level tensor-based execution
// - Work with streams for async execution
// - Use events for timing

/*
 Runtime Library Demo (examples/runtime/runtime_demo.cpp)

 The demo demonstrates the key capabilities of the Runtime API:

  | Section            | Functionality                                          |
  |--------------------|--------------------------------------------------------|
  | 1. Runtime Creation| Create KPURuntime with simulator                       |
  | 2. Memory Alloc    | Allocate device memory with malloc/free                |
  | 3. Data Transfer   | Copy data H2D, D2H, and D2D                           |
  | 4. Kernel Launch   | Launch kernels with explicit arguments                 |
  | 5. GraphExecutor   | High-level API with automatic tensor management        |
  | 6. Streams/Events  | Async execution and timing                             |

 Running the Demo

  ./build/examples/runtime/runtime_demo

 Key Output Highlights

  - CUDA-like API: Familiar malloc/memcpy/launch pattern
  - GraphExecutor: Higher-level API handles memory automatically
  - Timing: Events measure kernel execution time
*/

#include <sw/runtime/runtime.hpp>
#include <sw/runtime/executor.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using namespace sw::runtime;
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

// Initialize array with random values
void fill_random(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    static std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (auto& v : data) {
        v = dist(gen);
    }
}

// Reference matmul for verification
void reference_matmul(const float* A, const float* B, float* C,
                      Size M, Size N, Size K) {
    for (Size i = 0; i < M; ++i) {
        for (Size j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Check if two arrays match approximately
bool check_result(const float* computed, const float* reference, Size count,
                  float tolerance = 1e-4f) {
    for (Size i = 0; i < count; ++i) {
        if (std::abs(computed[i] - reference[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "KPU Simulator - Runtime Library Demo\n";
    separator();

    // =========================================================================
    // 1. Runtime Creation
    // =========================================================================
    separator("1. Runtime Creation");

    std::cout << "\nCreating KPU simulator and runtime...\n";

    // Create simulator with default configuration
    KPUSimulator::Config sim_config;
    sim_config.memory_bank_count = 4;
    sim_config.memory_bank_capacity_mb = 256;
    sim_config.l3_tile_count = 8;
    sim_config.l3_tile_capacity_kb = 512;
    sim_config.l2_bank_count = 16;
    sim_config.l2_bank_capacity_kb = 64;
    sim_config.scratchpad_count = 4;
    sim_config.scratchpad_capacity_kb = 64;
    sim_config.dma_engine_count = 4;
    sim_config.block_mover_count = 8;
    sim_config.streamer_count = 16;
    sim_config.systolic_array_rows = 16;
    sim_config.systolic_array_cols = 16;

    KPUSimulator simulator(sim_config);

    // Create runtime with optional configuration
    KPURuntime::Config rt_config;
    rt_config.verbose = true;
    rt_config.clock_ghz = 1.0;  // 1 GHz for easy calculation

    KPURuntime runtime(&simulator, rt_config);

    std::cout << "  Simulator: " << sim_config.memory_bank_count << " memory banks, "
              << sim_config.l3_tile_count << " L3 tiles\n";
    std::cout << "  Runtime:   Clock = " << rt_config.clock_ghz << " GHz\n";
    std::cout << "  Memory:    Total = " << format_bytes(runtime.get_total_memory())
              << ", Free = " << format_bytes(runtime.get_free_memory()) << "\n";

    // =========================================================================
    // 2. Memory Allocation
    // =========================================================================
    separator("2. Memory Allocation");

    const Size M = 64, N = 64, K = 64;
    const Size elem_size = sizeof(float);
    const Size A_bytes = M * K * elem_size;
    const Size B_bytes = K * N * elem_size;
    const Size C_bytes = M * N * elem_size;

    std::cout << "\nAllocating device memory for " << M << "x" << K << " x "
              << K << "x" << N << " matmul...\n";

    Address A_dev = runtime.malloc(A_bytes);
    Address B_dev = runtime.malloc(B_bytes);
    Address C_dev = runtime.malloc(C_bytes);

    std::cout << "  A: " << format_bytes(A_bytes) << " @ 0x" << std::hex << A_dev << std::dec << "\n";
    std::cout << "  B: " << format_bytes(B_bytes) << " @ 0x" << std::hex << B_dev << std::dec << "\n";
    std::cout << "  C: " << format_bytes(C_bytes) << " @ 0x" << std::hex << C_dev << std::dec << "\n";

    std::cout << "\nMemory after allocation:\n";
    std::cout << "  Free: " << format_bytes(runtime.get_free_memory()) << "\n";

    // =========================================================================
    // 3. Data Transfer
    // =========================================================================
    separator("3. Data Transfer");

    std::cout << "\nInitializing host data and transferring to device...\n";

    // Create and initialize host data
    std::vector<float> A_host(M * K);
    std::vector<float> B_host(K * N);
    std::vector<float> C_host(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    fill_random(A_host);
    fill_random(B_host);

    // Copy host to device
    runtime.memcpy_h2d(A_dev, A_host.data(), A_bytes);
    runtime.memcpy_h2d(B_dev, B_host.data(), B_bytes);

    std::cout << "  H2D: Copied A (" << format_bytes(A_bytes) << ")\n";
    std::cout << "  H2D: Copied B (" << format_bytes(B_bytes) << ")\n";

    // Clear output with memset
    runtime.memset(C_dev, 0, C_bytes);
    std::cout << "  Memset: Cleared C\n";

    // Test D2D copy
    Address C_copy_dev = runtime.malloc(C_bytes);
    runtime.memcpy_d2d(C_copy_dev, C_dev, C_bytes);
    std::cout << "  D2D: Copied C to C_copy\n";

    runtime.free(C_copy_dev);

    // =========================================================================
    // 4. Kernel Launch (Low-Level API)
    // =========================================================================
    separator("4. Kernel Launch (Low-Level API)");

    std::cout << "\nCreating and launching matmul kernel...\n";

    // Create kernel
    Kernel kernel = Kernel::create_matmul(M, N, K);
    std::cout << "  Kernel: " << kernel_op_type_name(kernel.op_type())
              << " (" << M << "x" << N << "x" << K << ")\n";
    std::cout << "  Tiles:  Ti=" << kernel.Ti() << ", Tj=" << kernel.Tj()
              << ", Tk=" << kernel.Tk() << "\n";

    // Launch kernel with explicit addresses
    std::vector<Address> args = {A_dev, B_dev, C_dev};
    LaunchResult result = runtime.launch(kernel, args);

    if (result.success) {
        std::cout << "\nLaunch successful!\n";
        std::cout << "  Cycles:     " << result.cycles << "\n";
        double time_ms = static_cast<double>(result.cycles) / (rt_config.clock_ghz * 1e6);
        std::cout << "  Time (ms):  " << std::fixed << std::setprecision(4)
                  << time_ms << "\n";
    } else {
        std::cout << "Launch failed: " << result.error << "\n";
    }

    // Copy result back
    runtime.memcpy_d2h(C_host.data(), C_dev, C_bytes);

    // Verify result
    reference_matmul(A_host.data(), B_host.data(), C_ref.data(), M, N, K);
    // Note: In a full implementation, we would verify the results match

    // Show stats
    std::cout << "\nRuntime Statistics:\n";
    std::cout << "  Total Launches: " << runtime.get_launch_count() << "\n";
    std::cout << "  Total Cycles:   " << runtime.get_total_cycles() << "\n";

    // =========================================================================
    // 5. GraphExecutor (High-Level API)
    // =========================================================================
    separator("5. GraphExecutor (High-Level API)");

    std::cout << "\nUsing GraphExecutor for automatic memory management...\n";

    GraphExecutor executor(&runtime);

    // Create a larger matmul
    const Size M2 = 128, N2 = 128, K2 = 128;
    executor.create_matmul(M2, N2, K2);

    std::cout << "  Created matmul: " << M2 << "x" << N2 << "x" << K2 << "\n";
    std::cout << "  Kernel: " << kernel_op_type_name(executor.kernel()->op_type()) << "\n";

    // Check tensor bindings
    std::cout << "\nTensor Bindings:\n";
    for (const auto& name : {"A", "B", "C"}) {
        const TensorBinding* binding = executor.get_binding(name);
        if (binding) {
            std::string shape_str;
            for (size_t i = 0; i < binding->shape.size(); ++i) {
                if (i > 0) shape_str += "x";
                shape_str += std::to_string(binding->shape[i]);
            }
            std::cout << "  " << name << ": " << shape_str
                      << " @ 0x" << std::hex << binding->device_address << std::dec << "\n";
        }
    }

    // Prepare input data
    std::vector<float> A2(M2 * K2);
    std::vector<float> B2(K2 * N2);
    std::vector<float> C2(M2 * N2, 0.0f);

    fill_random(A2);
    fill_random(B2);

    // Set inputs (automatically copies to device)
    executor.set_input("A", A2.data(), {M2, K2});
    executor.set_input("B", B2.data(), {K2, N2});
    std::cout << "\nInputs set via set_input()\n";

    // Execute
    ExecutionResult exec_result = executor.execute();

    if (exec_result.success) {
        std::cout << "\nExecution successful!\n";
        std::cout << "  Cycles:    " << exec_result.cycles << "\n";
        std::cout << "  Time (ms): " << std::fixed << std::setprecision(4)
                  << exec_result.time_ms << "\n";
    } else {
        std::cout << "Execution failed: " << exec_result.error << "\n";
    }

    // Get output (automatically copies from device)
    executor.get_output("C", C2.data());
    std::cout << "Output retrieved via get_output()\n";

    // =========================================================================
    // 6. Streams and Events
    // =========================================================================
    separator("6. Streams and Events");

    std::cout << "\nDemonstrating streams and events for async execution...\n";

    // Create a stream
    Stream stream = runtime.create_stream();
    std::cout << "  Created stream: id=" << stream.id << "\n";

    // Create events for timing
    Event start = runtime.create_event();
    Event end = runtime.create_event();
    std::cout << "  Created events: start=" << start.id << ", end=" << end.id << "\n";

    // Record start event, launch, record end
    runtime.record_event(start, stream);
    runtime.launch_async(kernel, {A_dev, B_dev, C_dev}, stream);
    runtime.record_event(end, stream);

    // Wait for completion
    runtime.stream_synchronize(stream);
    std::cout << "\nStream synchronized.\n";

    // Get elapsed time
    float elapsed = runtime.elapsed_time(start, end);
    std::cout << "  Elapsed time (events): " << std::fixed << std::setprecision(4)
              << elapsed << " ms\n";

    // Cleanup events and stream
    runtime.destroy_event(start);
    runtime.destroy_event(end);
    runtime.destroy_stream(stream);

    // =========================================================================
    // 7. MLP Kernel with GraphExecutor
    // =========================================================================
    separator("7. MLP Kernel with GraphExecutor");

    std::cout << "\nCreating an MLP kernel (matmul + bias + activation)...\n";

    // Create new executor for MLP
    GraphExecutor mlp_executor(&runtime);
    mlp_executor.create_mlp(64, 128, 64, ActivationType::GELU, true);

    std::cout << "  Kernel: " << kernel_op_type_name(mlp_executor.kernel()->op_type()) << "\n";
    std::cout << "  Activation: " << activation_type_name(mlp_executor.kernel()->activation()) << "\n";
    std::cout << "  Has Bias: " << (mlp_executor.kernel()->has_bias() ? "yes" : "no") << "\n";

    // Check bindings (should include bias)
    std::cout << "\nTensor Bindings:\n";
    for (const auto& name : {"A", "B", "bias", "C"}) {
        const TensorBinding* binding = mlp_executor.get_binding(name);
        if (binding) {
            std::string shape_str;
            for (size_t i = 0; i < binding->shape.size(); ++i) {
                if (i > 0) shape_str += "x";
                shape_str += std::to_string(binding->shape[i]);
            }
            std::cout << "  " << name << ": " << shape_str
                      << " @ 0x" << std::hex << binding->device_address << std::dec << "\n";
        }
    }

    // Prepare MLP inputs
    std::vector<float> mlp_A(64 * 64);
    std::vector<float> mlp_B(64 * 128);
    std::vector<float> mlp_bias(128, 0.1f);
    std::vector<float> mlp_C(64 * 128);

    fill_random(mlp_A, 0.0f, 1.0f);  // Positive values for activation
    fill_random(mlp_B, 0.0f, 1.0f);

    mlp_executor.set_input("A", mlp_A.data(), {64, 64});
    mlp_executor.set_input("B", mlp_B.data(), {64, 128});
    mlp_executor.set_input("bias", mlp_bias.data(), {128});

    ExecutionResult mlp_result = mlp_executor.execute();

    if (mlp_result.success) {
        std::cout << "\nMLP execution successful!\n";
        std::cout << "  Cycles: " << mlp_result.cycles << "\n";
    }

    mlp_executor.get_output("C", mlp_C.data());

    // =========================================================================
    // Cleanup
    // =========================================================================
    separator("Cleanup");

    std::cout << "\nFreeing device memory...\n";
    runtime.free(A_dev);
    runtime.free(B_dev);
    runtime.free(C_dev);

    std::cout << "  Memory after free: " << format_bytes(runtime.get_free_memory()) << "\n";

    // Release executor resources
    executor.release();
    mlp_executor.release();

    // =========================================================================
    // Final Statistics
    // =========================================================================
    separator("Final Statistics");

    std::cout << "\nRuntime Summary:\n";
    std::cout << "  Total Kernel Launches: " << runtime.get_launch_count() << "\n";
    std::cout << "  Total Simulated Cycles: " << runtime.get_total_cycles() << "\n";

    runtime.print_stats();

    separator();
    std::cout << "\nRuntime library demo complete!\n";

    return 0;
}
