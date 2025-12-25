#pragma once
// Graph Executor for KPU Runtime
// High-level execution API for computational graphs

#include <sw/runtime/runtime.hpp>
#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace sw::runtime {

/**
 * @brief Tensor descriptor for input/output binding
 */
struct TensorBinding {
    std::string name;
    std::vector<kpu::Size> shape;
    kpu::DataType dtype = kpu::DataType::FLOAT32;
    kpu::Address device_address = 0;
    kpu::Size size_bytes = 0;

    TensorBinding() = default;
    TensorBinding(const std::string& n, const std::vector<kpu::Size>& s,
                  kpu::DataType dt = kpu::DataType::FLOAT32)
        : name(n), shape(s), dtype(dt) {
        compute_size();
    }

    void compute_size() {
        size_bytes = kpu::dtype_size(dtype);
        for (auto dim : shape) {
            size_bytes *= dim;
        }
    }
};

/**
 * @brief Execution result with timing information
 */
struct ExecutionResult {
    bool success = false;
    kpu::Cycle cycles = 0;
    double time_ms = 0.0;
    std::string error;

    ExecutionResult() = default;
    ExecutionResult(bool ok, kpu::Cycle c, double t, const std::string& err = "")
        : success(ok), cycles(c), time_ms(t), error(err) {}
};

/**
 * @brief Graph Executor - High-level execution API
 *
 * Provides a simple interface for executing kernels without
 * manually managing memory addresses. Handles:
 * - Automatic memory allocation for tensors
 * - Input data staging
 * - Output data retrieval
 * - Memory cleanup
 *
 * Usage:
 * @code
 * KPUSimulator sim(config);
 * KPURuntime runtime(&sim);
 * GraphExecutor executor(&runtime);
 *
 * // Create a matmul kernel
 * Kernel kernel = Kernel::create_matmul(1024, 1024, 1024);
 * executor.set_kernel(kernel);
 *
 * // Prepare input tensors
 * std::vector<float> A(1024 * 1024), B(1024 * 1024);
 * // ... fill A and B with data ...
 *
 * executor.set_input("A", A.data(), {1024, 1024});
 * executor.set_input("B", B.data(), {1024, 1024});
 *
 * // Execute
 * auto result = executor.execute();
 *
 * // Get output
 * std::vector<float> C(1024 * 1024);
 * executor.get_output("C", C.data());
 * @endcode
 */
class GraphExecutor {
public:
    /**
     * @brief Construct executor with runtime
     * @param runtime Pointer to KPURuntime (must outlive executor)
     */
    explicit GraphExecutor(KPURuntime* runtime);

    ~GraphExecutor();

    // Non-copyable
    GraphExecutor(const GraphExecutor&) = delete;
    GraphExecutor& operator=(const GraphExecutor&) = delete;

    // Movable
    GraphExecutor(GraphExecutor&&) noexcept;
    GraphExecutor& operator=(GraphExecutor&&) noexcept;

    // =========================================
    // Kernel Setup
    // =========================================

    /**
     * @brief Set the kernel to execute
     * @param kernel The kernel
     *
     * This allocates device memory for all kernel arguments.
     */
    void set_kernel(const kpu::Kernel& kernel);

    /**
     * @brief Create and set a matmul kernel
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param dtype Data type (default FLOAT32)
     */
    void create_matmul(kpu::Size M, kpu::Size N, kpu::Size K,
                       kpu::DataType dtype = kpu::DataType::FLOAT32);

    /**
     * @brief Create and set an MLP kernel
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param activation Activation function
     * @param has_bias Whether to include bias
     * @param dtype Data type
     */
    void create_mlp(kpu::Size M, kpu::Size N, kpu::Size K,
                    kpu::ActivationType activation,
                    bool has_bias = true,
                    kpu::DataType dtype = kpu::DataType::FLOAT32);

    /**
     * @brief Get the current kernel
     */
    const kpu::Kernel* kernel() const { return kernel_.get(); }

    // =========================================
    // Input/Output Binding
    // =========================================

    /**
     * @brief Set input tensor data
     * @param name Tensor name (must match kernel argument)
     * @param data Host data pointer
     * @param shape Tensor shape (for validation)
     *
     * Copies data from host to device.
     */
    void set_input(const std::string& name, const void* data,
                   const std::vector<kpu::Size>& shape);

    /**
     * @brief Set input tensor data (without shape check)
     * @param name Tensor name
     * @param data Host data pointer
     * @param size_bytes Number of bytes to copy
     */
    void set_input(const std::string& name, const void* data, kpu::Size size_bytes);

    /**
     * @brief Get output tensor data
     * @param name Tensor name (must match kernel argument)
     * @param data Host destination pointer
     *
     * Copies data from device to host.
     */
    void get_output(const std::string& name, void* data);

    /**
     * @brief Get output tensor data with size
     * @param name Tensor name
     * @param data Host destination pointer
     * @param size_bytes Number of bytes to copy
     */
    void get_output(const std::string& name, void* data, kpu::Size size_bytes);

    /**
     * @brief Get binding for a tensor
     * @param name Tensor name
     * @return TensorBinding, or nullptr if not found
     */
    const TensorBinding* get_binding(const std::string& name) const;

    // =========================================
    // Execution
    // =========================================

    /**
     * @brief Execute the kernel
     * @return Execution result with timing
     */
    ExecutionResult execute();

    /**
     * @brief Get the last execution result
     */
    const ExecutionResult& last_result() const { return last_result_; }

    /**
     * @brief Get the last execution time in milliseconds
     */
    double get_last_execution_time_ms() const { return last_result_.time_ms; }

    /**
     * @brief Get the last execution cycles
     */
    kpu::Cycle get_last_execution_cycles() const { return last_result_.cycles; }

    // =========================================
    // Cleanup
    // =========================================

    /**
     * @brief Free all allocated device memory
     */
    void release();

    /**
     * @brief Check if executor has a kernel set
     */
    bool has_kernel() const { return kernel_ != nullptr; }

    /**
     * @brief Get the runtime
     */
    KPURuntime* runtime() const { return runtime_; }

private:
    KPURuntime* runtime_;
    std::unique_ptr<kpu::Kernel> kernel_;
    std::unordered_map<std::string, TensorBinding> bindings_;
    std::vector<kpu::Address> arg_addresses_;
    ExecutionResult last_result_;

    void allocate_tensors();
    void free_tensors();
    const kpu::KernelArgument* find_argument(const std::string& name) const;
};

} // namespace sw::runtime
