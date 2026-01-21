#pragma once
// Kernel Abstraction Layer for KPU simulator
// Provides a high-level interface for creating and managing executable kernels

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/data_types.hpp>
#include <sw/kpu/resource_api.hpp>  // For ElementwiseOp enum
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
    ATTENTION = 8,      // Scaled dot-product attention
    RMSNORM = 9,        // RMS normalization (LLaMA-style)
    BATCHNORM = 10,     // Batch normalization (CNN inference)
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
        case KernelOpType::ATTENTION: return "attention";
        case KernelOpType::RMSNORM: return "rmsnorm";
        case KernelOpType::BATCHNORM: return "batchnorm";
        case KernelOpType::CUSTOM: return "custom";
        default: return "unknown";
    }
}

// Note: ElementwiseOp enum and elementwise_op_name() are defined in resource_api.hpp

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
 * @brief Attention configuration parameters for transformer self-attention
 *
 * Implements scaled dot-product attention:
 *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * With optional QKV and output projections:
 *   Q, K, V = X @ W_Q, X @ W_K, X @ W_V
 *   output = Attention(Q, K, V) @ W_O
 */
struct AttentionConfig {
    Size batch_size;       // B: batch size
    Size seq_len;          // S: sequence length
    Size d_model;          // D: model/embedding dimension
    Size num_heads;        // H: number of attention heads
    bool causal_mask;      // Apply causal mask for decoder attention
    bool include_qkv_projection;    // Include Q, K, V projections (default true)
    bool include_output_projection; // Include output projection (default true)

    AttentionConfig()
        : batch_size(1), seq_len(1), d_model(64), num_heads(1)
        , causal_mask(false)
        , include_qkv_projection(true)
        , include_output_projection(true) {}

    // Head dimension (d_k = d_v = d_model / num_heads)
    Size head_dim() const { return d_model / num_heads; }

    // Compute total FLOPs for attention operation
    Size total_flops() const {
        Size B = batch_size;
        Size S = seq_len;
        Size D = d_model;
        Size H = num_heads;
        Size d_k = head_dim();

        Size flops = 0;

        // QKV Projection: 3 * (2 * B * S * D * D) = 6 * B * S * D^2
        if (include_qkv_projection) {
            flops += 6 * B * S * D * D;
        }

        // Q @ K^T: 2 * B * H * S * S * d_k (per head matmul)
        flops += 2 * B * H * S * S * d_k;

        // Scale by 1/sqrt(d_k): B * H * S * S (multiply)
        flops += B * H * S * S;

        // Softmax: ~5 ops per element (max, sub, exp, sum, div)
        flops += 5 * B * H * S * S;

        // Attention @ V: 2 * B * H * S * d_k * S (per head matmul)
        flops += 2 * B * H * S * d_k * S;

        // Output Projection: 2 * B * S * D * D
        if (include_output_projection) {
            flops += 2 * B * S * D * D;
        }

        return flops;
    }
};

/**
 * @brief Layer normalization configuration parameters
 *
 * Implements LayerNorm:
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * Where mean and var are computed over the normalized_shape dimensions.
 * For transformers, typically normalized_shape = [d_model].
 */
struct LayerNormConfig {
    Size batch_size;           // B: batch size
    Size seq_len;              // S: sequence length (or spatial dims)
    Size normalized_dim;       // D: dimension to normalize over (e.g., d_model)
    float eps;                 // Epsilon for numerical stability
    bool has_weight;           // Apply learned scale (gamma)
    bool has_bias;             // Apply learned bias (beta)

    LayerNormConfig()
        : batch_size(1), seq_len(1), normalized_dim(64)
        , eps(1e-5f), has_weight(true), has_bias(true) {}

    // Total elements in input tensor
    Size total_elements() const { return batch_size * seq_len * normalized_dim; }

    // Number of normalization groups (each normalized independently)
    Size num_groups() const { return batch_size * seq_len; }

    // Compute total FLOPs for layer normalization
    // Per group: mean (D adds), var (D muls + D adds), normalize (D subs + D divs),
    //            scale (D muls), bias (D adds)
    // Simplified: ~7 ops per element
    Size total_flops() const {
        Size D = normalized_dim;
        Size groups = num_groups();

        Size flops = 0;

        // Mean computation: D additions + 1 division per group
        flops += groups * (D + 1);

        // Variance computation: D subtractions + D multiplications + D additions + 1 division
        flops += groups * (3 * D + 1);

        // Normalization: D subtractions + D divisions (or muls with rsqrt)
        flops += groups * (2 * D);

        // Scale (gamma): D multiplications
        if (has_weight) {
            flops += groups * D;
        }

        // Bias (beta): D additions
        if (has_bias) {
            flops += groups * D;
        }

        return flops;
    }
};

/**
 * @brief RMS normalization configuration parameters
 *
 * Implements RMSNorm (used in LLaMA, Gemma, etc.):
 *   y = x / sqrt(mean(x^2) + eps) * gamma
 *
 * RMSNorm is a simplified LayerNorm without mean centering or bias.
 * It's computationally cheaper and works well for transformer architectures.
 */
struct RMSNormConfig {
    Size batch_size;           // B: batch size
    Size seq_len;              // S: sequence length (or spatial dims)
    Size normalized_dim;       // D: dimension to normalize over (e.g., d_model)
    float eps;                 // Epsilon for numerical stability

    RMSNormConfig()
        : batch_size(1), seq_len(1), normalized_dim(64)
        , eps(1e-6f) {}  // RMSNorm typically uses smaller eps (1e-6)

    // Total elements in input tensor
    Size total_elements() const { return batch_size * seq_len * normalized_dim; }

    // Number of normalization groups (each normalized independently)
    Size num_groups() const { return batch_size * seq_len; }

    // Compute total FLOPs for RMS normalization
    // Per group: square (D muls), mean (D adds + 1 div), rsqrt (1 op),
    //            normalize (D muls), scale (D muls)
    Size total_flops() const {
        Size D = normalized_dim;
        Size groups = num_groups();

        Size flops = 0;

        // Square each element: D multiplications per group
        flops += groups * D;

        // Mean of squares: D additions + 1 division per group
        flops += groups * (D + 1);

        // Reciprocal sqrt: 1 operation per group (approximated)
        flops += groups;

        // Normalize: D multiplications per group (x * rsqrt)
        flops += groups * D;

        // Scale (gamma): D multiplications per group
        flops += groups * D;

        return flops;
    }
};

/**
 * @brief Batch normalization configuration parameters
 *
 * Implements BatchNorm2D for CNN inference:
 *   y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
 *
 * During inference, uses precomputed running statistics.
 * Can be folded into preceding convolution by adjusting weights.
 * Input shape: [N, C, H, W], normalizes over C dimension.
 */
struct BatchNormConfig {
    Size batch_size;           // N: batch size
    Size num_features;         // C: number of channels/features
    Size height;               // H: spatial height
    Size width;                // W: spatial width
    float eps;                 // Epsilon for numerical stability
    bool affine;               // Use learnable gamma (weight) and beta (bias)

    BatchNormConfig()
        : batch_size(1), num_features(64), height(1), width(1)
        , eps(1e-5f), affine(true) {}

    // Total elements in input tensor
    Size total_elements() const { return batch_size * num_features * height * width; }

    // Spatial size per channel
    Size spatial_size() const { return height * width; }

    // Total number of points to normalize per channel
    Size points_per_channel() const { return batch_size * height * width; }

    // Compute total FLOPs for batch normalization (inference mode)
    // Per element: subtract mean (1), multiply by 1/sqrt(var+eps) (1), scale (1), bias (1)
    // The 1/sqrt(var+eps) is precomputed per channel, so just a multiply per element
    Size total_flops() const {
        Size total = total_elements();
        Size flops = 0;

        // Subtract running_mean: 1 op per element
        flops += total;

        // Multiply by 1/sqrt(running_var + eps): 1 op per element
        // (The sqrt and reciprocal are precomputed per channel, not counted here)
        flops += total;

        // Scale by gamma: 1 op per element
        if (affine) {
            flops += total;
        }

        // Add beta: 1 op per element
        if (affine) {
            flops += total;
        }

        return flops;
    }
};

/**
 * @brief Elementwise operation configuration
 *
 * Implements elementwise operations between tensors:
 *   C = op(A, B) or C = op(A) for unary operations
 *
 * Supports binary ops (add, sub, mul, div, max, min),
 * unary ops (neg, abs, sqrt, exp, log),
 * and scalar ops (add_scalar, mul_scalar, pow_scalar).
 *
 * Tensors must be broadcastable (same shape or scalar).
 */
struct ElementwiseConfig {
    ElementwiseOp op;              // Operation type
    std::vector<Size> shape;       // Output tensor shape
    bool is_unary;                 // True for unary operations (only A input)
    bool is_scalar_b;              // True if B is a scalar (broadcast)
    float scalar_value;            // Scalar value for scalar operations

    ElementwiseConfig()
        : op(ElementwiseOp::ADD), shape{1}, is_unary(false)
        , is_scalar_b(false), scalar_value(0.0f) {}

    // Total elements in output tensor
    Size total_elements() const {
        Size total = 1;
        for (Size dim : shape) {
            total *= dim;
        }
        return total;
    }

    // Compute total FLOPs for elementwise operation
    // Most ops are 1 FLOP per element, some (like exp, log, sqrt) count as more
    Size total_flops() const {
        Size total = total_elements();

        switch (op) {
            case ElementwiseOp::ADD:
            case ElementwiseOp::SUB:
            case ElementwiseOp::MUL:
            case ElementwiseOp::DIV:
            case ElementwiseOp::MAX:
            case ElementwiseOp::MIN:
            case ElementwiseOp::NEG:
            case ElementwiseOp::ABS:
            case ElementwiseOp::ADD_SCALAR:
            case ElementwiseOp::MUL_SCALAR:
                return total;  // 1 op per element

            case ElementwiseOp::SQRT:
            case ElementwiseOp::EXP:
            case ElementwiseOp::LOG:
            case ElementwiseOp::POW_SCALAR:
                return total * 4;  // Approximated as 4 ops per element

            default:
                return total;
        }
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

    /**
     * @brief Create a scaled dot-product attention kernel
     * @param config Attention configuration parameters
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Implements transformer self-attention:
     *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
     *
     * With optional projections:
     *   Q, K, V = X @ W_Q, X @ W_K, X @ W_V
     *   output = Attention(Q, K, V) @ W_O
     *
     * Arguments (when include_qkv_projection=true):
     *   - X: [B, S, D] input tensor
     *   - W_Q: [D, D] query projection weights
     *   - W_K: [D, D] key projection weights
     *   - W_V: [D, D] value projection weights
     *   - W_O: [D, D] output projection weights (if include_output_projection)
     *   - output: [B, S, D] output tensor
     *
     * Arguments (when include_qkv_projection=false):
     *   - Q: [B, H, S, d_k] query tensor
     *   - K: [B, H, S, d_k] key tensor
     *   - V: [B, H, S, d_k] value tensor
     *   - output: [B, H, S, d_k] output tensor
     */
    static Kernel create_attention(const AttentionConfig& config,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create attention kernel with explicit parameters
     * @param batch_size B: batch size
     * @param seq_len S: sequence length
     * @param d_model D: model dimension
     * @param num_heads H: number of attention heads
     * @param causal_mask Apply causal masking (for decoder)
     * @param include_qkv_projection Include Q/K/V projections
     * @param include_output_projection Include output projection
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_attention(Size batch_size, Size seq_len, Size d_model,
                                   Size num_heads, bool causal_mask = false,
                                   bool include_qkv_projection = true,
                                   bool include_output_projection = true,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create a layer normalization kernel
     * @param config LayerNorm configuration parameters
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Implements layer normalization:
     *   y = (x - mean) / sqrt(var + eps) * gamma + beta
     *
     * Arguments:
     *   - input: [B, S, D] input tensor
     *   - weight: [D] scale parameter (gamma), if has_weight
     *   - bias: [D] bias parameter (beta), if has_bias
     *   - output: [B, S, D] output tensor
     */
    static Kernel create_layernorm(const LayerNormConfig& config,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create layer normalization kernel with explicit parameters
     * @param batch_size B: batch size
     * @param seq_len S: sequence length
     * @param normalized_dim D: dimension to normalize over
     * @param eps Epsilon for numerical stability
     * @param has_weight Apply learned scale (gamma)
     * @param has_bias Apply learned bias (beta)
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_layernorm(Size batch_size, Size seq_len, Size normalized_dim,
                                   float eps = 1e-5f,
                                   bool has_weight = true, bool has_bias = true,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create an RMS normalization kernel
     * @param config RMSNorm configuration parameters
     * @param dtype Data type (default FLOAT32)
     * @return Compiled kernel
     *
     * Implements RMS normalization (used in LLaMA, Gemma, etc.):
     *   y = x / sqrt(mean(x^2) + eps) * gamma
     *
     * RMSNorm is simpler than LayerNorm - no mean centering, no bias.
     *
     * Arguments:
     *   - input: [B, S, D] input tensor
     *   - weight: [D] scale parameter (gamma)
     *   - output: [B, S, D] output tensor
     */
    static Kernel create_rmsnorm(const RMSNormConfig& config,
                                 DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create RMS normalization kernel with explicit parameters
     * @param batch_size B: batch size
     * @param seq_len S: sequence length
     * @param normalized_dim D: dimension to normalize over
     * @param eps Epsilon for numerical stability (default 1e-6)
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_rmsnorm(Size batch_size, Size seq_len, Size normalized_dim,
                                 float eps = 1e-6f,
                                 DataType dtype = DataType::FLOAT32);

    // -----------------------------------------
    // BatchNorm Kernel Creation (v0.5.4)
    // -----------------------------------------

    /**
     * @brief Create batch normalization kernel for CNN inference
     * @param config BatchNorm configuration
     * @param dtype Data type
     * @return Compiled kernel
     *
     * BatchNorm normalizes over the channel dimension using precomputed
     * running statistics. Can be folded into preceding convolution.
     *
     * Input shape: [N, C, H, W]
     * Formula: y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
     *
     * Arguments:
     *   - input: [N, C, H, W] input tensor
     *   - running_mean: [C] running mean per channel
     *   - running_var: [C] running variance per channel
     *   - weight: [C] scale parameter (gamma), if affine
     *   - bias: [C] bias parameter (beta), if affine
     *   - output: [N, C, H, W] output tensor
     */
    static Kernel create_batchnorm(const BatchNormConfig& config,
                                   DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create batch normalization kernel with explicit parameters
     * @param batch_size N: batch size
     * @param num_features C: number of channels
     * @param height H: spatial height
     * @param width W: spatial width
     * @param eps Epsilon for numerical stability (default 1e-5)
     * @param affine Use learnable gamma and beta (default true)
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_batchnorm(Size batch_size, Size num_features,
                                   Size height, Size width,
                                   float eps = 1e-5f, bool affine = true,
                                   DataType dtype = DataType::FLOAT32);

    // -----------------------------------------
    // Elementwise Kernel Creation (v0.5.5)
    // -----------------------------------------

    /**
     * @brief Create elementwise operation kernel
     * @param config Elementwise configuration
     * @param dtype Data type
     * @return Compiled kernel
     *
     * Elementwise operations apply an operation to each element:
     *   Binary: C = op(A, B) where A, B have same shape
     *   Unary: C = op(A)
     *   Scalar: C = op(A, scalar) where scalar is broadcast
     *
     * Arguments (binary):
     *   - A: input tensor
     *   - B: input tensor (same shape as A)
     *   - C: output tensor
     *
     * Arguments (unary):
     *   - A: input tensor
     *   - C: output tensor
     */
    static Kernel create_elementwise(const ElementwiseConfig& config,
                                     DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create binary elementwise operation kernel
     * @param op Operation type (ADD, SUB, MUL, DIV, MAX, MIN)
     * @param shape Tensor shape
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_elementwise(ElementwiseOp op,
                                     const std::vector<Size>& shape,
                                     DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create unary elementwise operation kernel
     * @param op Operation type (NEG, ABS, SQRT, EXP, LOG)
     * @param shape Tensor shape
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_elementwise_unary(ElementwiseOp op,
                                           const std::vector<Size>& shape,
                                           DataType dtype = DataType::FLOAT32);

    /**
     * @brief Create scalar elementwise operation kernel
     * @param op Operation type (ADD_SCALAR, MUL_SCALAR, POW_SCALAR)
     * @param shape Tensor shape
     * @param scalar Scalar value to apply
     * @param dtype Data type
     * @return Compiled kernel
     */
    static Kernel create_elementwise_scalar(ElementwiseOp op,
                                            const std::vector<Size>& shape,
                                            float scalar,
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
    // Attention Accessors (for ATTENTION kernels)
    // =========================================

    /**
     * @brief Get Attention configuration (for ATTENTION kernels)
     */
    const AttentionConfig& attention_config() const { return attention_config_; }

    // =========================================
    // LayerNorm Accessors (for LAYERNORM kernels)
    // =========================================

    /**
     * @brief Get LayerNorm configuration (for LAYERNORM kernels)
     */
    const LayerNormConfig& layernorm_config() const { return layernorm_config_; }

    // =========================================
    // RMSNorm Accessors (for RMSNORM kernels)
    // =========================================

    /**
     * @brief Get RMSNorm configuration (for RMSNORM kernels)
     */
    const RMSNormConfig& rmsnorm_config() const { return rmsnorm_config_; }

    // =========================================
    // BatchNorm Accessors (for BATCHNORM kernels)
    // =========================================

    /**
     * @brief Get BatchNorm configuration (for BATCHNORM kernels)
     */
    const BatchNormConfig& batchnorm_config() const { return batchnorm_config_; }

    // =========================================
    // Elementwise Accessors (for ELEMENTWISE kernels)
    // =========================================

    /**
     * @brief Get Elementwise configuration (for ELEMENTWISE kernels)
     */
    const ElementwiseConfig& elementwise_config() const { return elementwise_config_; }

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

    // Attention-specific members
    AttentionConfig attention_config_;

    // LayerNorm-specific members
    LayerNormConfig layernorm_config_;

    // RMSNorm-specific members
    RMSNormConfig rmsnorm_config_;

    // BatchNorm-specific members
    BatchNormConfig batchnorm_config_;

    // Elementwise-specific members
    ElementwiseConfig elementwise_config_;

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

    /**
     * @brief Set up arguments for ATTENTION operation
     */
    void setup_attention_arguments();

    /**
     * @brief Set up arguments for LAYERNORM operation
     */
    void setup_layernorm_arguments();

    /**
     * @brief Set up arguments for RMSNORM operation
     */
    void setup_rmsnorm_arguments();

    /**
     * @brief Set up arguments for BATCHNORM operation
     */
    void setup_batchnorm_arguments();

    /**
     * @brief Set up arguments for ELEMENTWISE operation
     */
    void setup_elementwise_arguments();
};

} // namespace sw::kpu
