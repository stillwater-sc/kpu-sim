#pragma once
// Kernel Abstraction Layer for KPU simulator
// Provides a high-level interface for creating and managing executable kernels

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/data_types.hpp>
#include <sw/concepts.hpp>

#include <string>
#include <vector>
#include <memory>

namespace sw::kpu {

/**
 * @brief Type of kernel operation
 */
enum class KernelOpType : uint8_t {
    MATMUL = 0,         // Matrix multiplication C = A x B
    BATCH_MATMUL = 1,   // Batched matrix multiplication
    CONV2D = 2,         // 2D convolution (future)
    ELEMENTWISE = 3,    // Elementwise operations (future)
    MLP = 4,            // Fused matmul + bias + activation: C = activation(A x B + bias)
    CUSTOM = 255        // Custom/user-defined
};

/**
 * @brief Get string name for kernel operation type
 */
inline const char* kernel_op_type_name(KernelOpType op) {
    switch (op) {
        case KernelOpType::MATMUL: return "matmul";
        case KernelOpType::BATCH_MATMUL: return "batch_matmul";
        case KernelOpType::CONV2D: return "conv2d";
        case KernelOpType::ELEMENTWISE: return "elementwise";
        case KernelOpType::MLP: return "mlp";
        case KernelOpType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

/**
 * @brief Kernel argument descriptor
 *
 * Describes an input or output argument to a kernel, including
 * its name, data type, shape, and size.
 */
struct KernelArgument {
    std::string name;           // Argument name (e.g., "A", "B", "C")
    DataType dtype;             // Data type
    std::vector<Size> shape;    // Shape (e.g., {M, K} for matrix A)
    bool is_output;             // True if this is an output argument
    Size size_bytes;            // Total size in bytes

    KernelArgument()
        : dtype(DataType::FLOAT32), is_output(false), size_bytes(0) {}

    KernelArgument(const std::string& n, DataType dt,
                   std::vector<Size> s, bool output = false)
        : name(n), dtype(dt), shape(std::move(s)), is_output(output) {
        size_bytes = compute_size();
    }

    /**
     * @brief Compute total size in bytes based on shape and dtype
     */
    Size compute_size() const {
        Size elements = 1;
        for (Size d : shape) elements *= d;
        return elements * dtype_size(dtype);
    }
};

/**
 * @brief Kernel - High-level abstraction for executable programs
 *
 * A Kernel encapsulates a DMProgram with metadata about the operation,
 * its arguments, and provides convenient methods for inspection.
 *
 * Usage:
 *   // Create via factory method (simplest)
 *   auto kernel = Kernel::create_matmul(1024, 1024, 1024);
 *
 *   // Or via KernelCompiler for more control
 *   KernelCompiler compiler;
 *   auto kernel = compiler.compile_matmul(1024, 1024, 1024);
 *
 *   // Access underlying program for execution
 *   const DMProgram& program = kernel.program();
 *   ConcurrentExecutor executor(config);
 *   Cycle cycles = executor.execute(program);
 */
class Kernel {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Default constructor - creates invalid kernel
     */
    Kernel();

    /**
     * @brief Construct kernel from existing DMProgram
     * @param program The compiled program (moved)
     * @param op_type Operation type
     * @param dtype Data type of elements
     */
    Kernel(isa::DMProgram program, KernelOpType op_type,
           DataType dtype = DataType::FLOAT32);

    /**
     * @brief Construct MLP kernel from existing DMProgram
     * @param program The compiled program (moved)
     * @param dtype Data type of elements
     * @param activation Activation function type
     * @param has_bias Whether bias addition is enabled
     */
    Kernel(isa::DMProgram program, DataType dtype,
           ActivationType activation, bool has_bias);

    // Move semantics (efficient, default)
    Kernel(Kernel&&) = default;
    Kernel& operator=(Kernel&&) = default;

    // Copy semantics (programs can be large, but allowed)
    Kernel(const Kernel&) = default;
    Kernel& operator=(const Kernel&) = default;

    ~Kernel() = default;

    // =========================================
    // Factory Methods
    // =========================================

    /**
     * @brief Create a matrix multiplication kernel with default settings
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Uses automatic tile optimization and output-stationary dataflow.
     * This is the simplest way to create a kernel.
     */
    static Kernel create_matmul(Size M, Size N, Size K,
                                DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create kernel from explicit program builder config
     * @param config The OutputStationaryProgramBuilder::Config
     * @param dtype Data type of elements
     * @return Compiled kernel
     *
     * For users who want full control over tiling and configuration.
     */
    static Kernel create_from_config(
        const isa::OutputStationaryProgramBuilder::Config& config,
        DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create a fused MLP kernel (matmul + bias + activation)
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param activation Activation function type
     * @param has_bias Whether to apply bias addition
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Creates C = activation(A @ B + bias) in a single fused operation.
     * The Vector Engine applies bias and activation inline during
     * the output drain phase, avoiding extra memory passes.
     *
     * Arguments:
     *   - A: [M, K] input matrix
     *   - B: [K, N] weight matrix
     *   - bias: [N] bias vector (if has_bias=true)
     *   - C: [M, N] output matrix
     */
    static Kernel create_mlp(Size M, Size N, Size K,
                             ActivationType activation,
                             bool has_bias = true,
                             DataType dtype = DataType::FLOAT32);

    // =========================================
    // Metadata Accessors
    // =========================================

    /**
     * @brief Check if kernel is valid (has program with instructions)
     */
    bool is_valid() const { return !program_.instructions.empty(); }

    /**
     * @brief Get kernel name (from underlying program)
     */
    const std::string& name() const { return program_.name; }

    /**
     * @brief Get operation type
     */
    KernelOpType op_type() const { return op_type_; }

    /**
     * @brief Get data type
     */
    DataType dtype() const { return dtype_; }

    /**
     * @brief Get kernel arguments
     */
    const std::vector<KernelArgument>& arguments() const { return arguments_; }

    /**
     * @brief Get input arguments only
     */
    std::vector<KernelArgument> input_arguments() const;

    /**
     * @brief Get output arguments only
     */
    std::vector<KernelArgument> output_arguments() const;

    /**
     * @brief Get total input size in bytes
     */
    Size total_input_bytes() const;

    /**
     * @brief Get total output size in bytes
     */
    Size total_output_bytes() const;

    // =========================================
    // Matrix Dimension Accessors (for MATMUL)
    // =========================================

    /**
     * @brief Get M dimension (rows of A and C)
     */
    Size M() const { return program_.M; }

    /**
     * @brief Get N dimension (columns of B and C)
     */
    Size N() const { return program_.N; }

    /**
     * @brief Get K dimension (columns of A, rows of B)
     */
    Size K() const { return program_.K; }

    /**
     * @brief Get Ti tile size (M-dimension)
     */
    Size Ti() const { return program_.Ti; }

    /**
     * @brief Get Tj tile size (N-dimension)
     */
    Size Tj() const { return program_.Tj; }

    /**
     * @brief Get Tk tile size (K-dimension)
     */
    Size Tk() const { return program_.Tk; }

    // =========================================
    // MLP Accessors (for MLP kernels)
    // =========================================

    /**
     * @brief Get activation function type (for MLP kernels)
     */
    ActivationType activation() const { return activation_; }

    /**
     * @brief Check if kernel uses bias (for MLP kernels)
     */
    bool has_bias() const { return has_bias_; }

    // =========================================
    // Program Access
    // =========================================

    /**
     * @brief Get underlying DMProgram (const)
     *
     * Use this to pass the program to ProgramExecutor or ConcurrentExecutor.
     */
    const isa::DMProgram& program() const { return program_; }

    /**
     * @brief Get underlying DMProgram (mutable)
     *
     * Use this if you need to modify the program (e.g., bind addresses).
     */
    isa::DMProgram& program() { return program_; }

    /**
     * @brief Get performance estimates from program
     */
    const isa::DMProgram::Estimates& estimates() const {
        return program_.estimates;
    }

    /**
     * @brief Get instruction count
     */
    size_t instruction_count() const { return program_.instructions.size(); }

    // =========================================
    // Utility Methods
    // =========================================

    /**
     * @brief Get human-readable summary string
     */
    std::string summary() const;

    /**
     * @brief Validate kernel for execution
     * @param error Output error message if invalid
     * @return true if valid
     */
    bool validate(std::string& error) const;

    /**
     * @brief Calculate arithmetic intensity (FLOPs per byte from DRAM)
     */
    double arithmetic_intensity() const;

    /**
     * @brief Calculate total FLOPs for this kernel
     */
    Size total_flops() const;

private:
    isa::DMProgram program_;
    KernelOpType op_type_ = KernelOpType::CUSTOM;
    DataType dtype_ = DataType::FLOAT32;
    std::vector<KernelArgument> arguments_;

    // MLP-specific members
    ActivationType activation_ = ActivationType::NONE;
    bool has_bias_ = false;

    /**
     * @brief Set up arguments for MATMUL operation
     */
    void setup_matmul_arguments();

    /**
     * @brief Set up arguments for MLP operation
     */
    void setup_mlp_arguments();
};

} // namespace sw::kpu
