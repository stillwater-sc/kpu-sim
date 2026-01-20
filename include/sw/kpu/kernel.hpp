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
    CONV2D = 2,         // 2D convolution via im2col + GEMM
    ELEMENTWISE = 3,    // Elementwise operations (future)
    MLP = 4,            // Fused matmul + bias + activation: C = activation(A x B + bias)
    POOL2D = 5,         // 2D pooling (max, avg)
    SOFTMAX = 6,        // Softmax normalization
    LAYERNORM = 7,      // Layer normalization
    CUSTOM = 255        // Custom/user-defined
};

/**
 * @brief Pooling operation type
 */
enum class PoolType : uint8_t {
    MAX = 0,
    AVG = 1,
    GLOBAL_AVG = 2
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
        case KernelOpType::POOL2D: return "pool2d";
        case KernelOpType::SOFTMAX: return "softmax";
        case KernelOpType::LAYERNORM: return "layernorm";
        case KernelOpType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

inline const char* pool_type_name(PoolType pt) {
    switch (pt) {
        case PoolType::MAX: return "max";
        case PoolType::AVG: return "avg";
        case PoolType::GLOBAL_AVG: return "global_avg";
        default: return "unknown";
    }
}

/**
 * @brief Conv2D configuration parameters
 */
struct Conv2DConfig {
    Size batch_size;       // N: batch size
    Size in_channels;      // C_in: input channels
    Size out_channels;     // C_out: output channels (num filters)
    Size input_height;     // H: input height
    Size input_width;      // W: input width
    Size kernel_height;    // K_h: kernel height
    Size kernel_width;     // K_w: kernel width
    Size stride_h;         // Stride in height dimension
    Size stride_w;         // Stride in width dimension
    Size padding_h;        // Padding in height dimension
    Size padding_w;        // Padding in width dimension
    Size dilation_h;       // Dilation in height (default 1)
    Size dilation_w;       // Dilation in width (default 1)
    Size groups;           // Number of groups (default 1)

    Conv2DConfig()
        : batch_size(1), in_channels(1), out_channels(1)
        , input_height(1), input_width(1)
        , kernel_height(1), kernel_width(1)
        , stride_h(1), stride_w(1)
        , padding_h(0), padding_w(0)
        , dilation_h(1), dilation_w(1)
        , groups(1) {}

    // Compute output dimensions
    Size output_height() const {
        return (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    }

    Size output_width() const {
        return (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    }

    // Compute equivalent GEMM dimensions for im2col approach
    // im2col converts conv to: [N * H_out * W_out, C_in * K_h * K_w] @ [C_out, C_in * K_h * K_w].T
    Size gemm_M() const { return batch_size * output_height() * output_width(); }
    Size gemm_N() const { return out_channels; }
    Size gemm_K() const { return (in_channels / groups) * kernel_height * kernel_width; }

    // Total FLOPs for conv2d
    Size total_flops() const {
        // 2 * (N * H_out * W_out) * C_out * (C_in/groups * K_h * K_w)
        return 2 * gemm_M() * gemm_N() * gemm_K();
    }
};

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

    /**
     * @brief Create a 2D convolution kernel using im2col + GEMM
     * @param config Conv2D configuration parameters
     * @param has_bias Whether to apply bias addition
     * @param activation Activation function (default NONE)
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Implements convolution via the im2col approach:
     * 1. im2col transforms input patches into columns
     * 2. GEMM: im2col_matrix @ weight_matrix.T
     * 3. Optional bias and activation
     *
     * Arguments:
     *   - input: [N, C_in, H, W] input tensor
     *   - weight: [C_out, C_in, K_h, K_w] weight tensor
     *   - bias: [C_out] bias vector (if has_bias=true)
     *   - output: [N, C_out, H_out, W_out] output tensor
     */
    static Kernel create_conv2d(const Conv2DConfig& config,
                                bool has_bias = true,
                                ActivationType activation = ActivationType::NONE,
                                DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create a 2D convolution kernel with explicit parameters
     * @param batch_size N: batch size
     * @param in_channels C_in: input channels
     * @param out_channels C_out: output channels
     * @param input_height H: input height
     * @param input_width W: input width
     * @param kernel_size K: kernel size (square kernel)
     * @param stride Stride (same for H and W)
     * @param padding Padding (same for H and W)
     * @param has_bias Whether to apply bias
     * @param activation Activation function
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_conv2d(Size batch_size, Size in_channels, Size out_channels,
                                Size input_height, Size input_width,
                                Size kernel_size, Size stride = 1, Size padding = 0,
                                bool has_bias = true,
                                ActivationType activation = ActivationType::NONE,
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
    // Conv2D Accessors (for CONV2D kernels)
    // =========================================

    /**
     * @brief Get Conv2D configuration (for CONV2D kernels)
     */
    const Conv2DConfig& conv2d_config() const { return conv2d_config_; }

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
     * @brief Set the underlying DMProgram
     *
     * Used by serialization to load a pre-compiled program.
     * @param program The program to set
     */
    void set_program(isa::DMProgram program) { program_ = std::move(program); }

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

    // Conv2D-specific members
    Conv2DConfig conv2d_config_;

    /**
     * @brief Set up arguments for MATMUL operation
     */
    void setup_matmul_arguments();

    /**
     * @brief Set up arguments for MLP operation
     */
    void setup_mlp_arguments();

    /**
     * @brief Set up arguments for CONV2D operation
     */
    void setup_conv2d_arguments();
};

} // namespace sw::kpu
