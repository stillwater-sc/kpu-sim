// Kernel implementation for KPU simulator
// Provides high-level kernel abstraction for executable programs

#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>
#include <sstream>
#include <iomanip>

namespace sw::kpu {

// ============================================================================
// Constructors
// ============================================================================

Kernel::Kernel()
    : op_type_(KernelOpType::CUSTOM)
    , dtype_(DataType::FLOAT32) {
}

Kernel::Kernel(isa::DMProgram program, KernelOpType op_type, DataType dtype)
    : program_(std::move(program))
    , op_type_(op_type)
    , dtype_(dtype) {

    // Set up arguments based on operation type
    if (op_type == KernelOpType::MATMUL || op_type == KernelOpType::BATCH_MATMUL) {
        setup_matmul_arguments();
    } else if (op_type == KernelOpType::MLP) {
        setup_mlp_arguments();
    }
}

Kernel::Kernel(isa::DMProgram program, DataType dtype,
               ActivationType activation, bool has_bias)
    : program_(std::move(program))
    , op_type_(KernelOpType::MLP)
    , dtype_(dtype)
    , activation_(activation)
    , has_bias_(has_bias) {

    setup_mlp_arguments();
}

// ============================================================================
// Factory Methods
// ============================================================================

Kernel Kernel::create_matmul(Size M, Size N, Size K, DataType dtype) {
    // Use KernelCompiler with default options
    compiler::KernelCompiler compiler;
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;

    return compiler.compile_matmul(M, N, K, opts);
}

Kernel Kernel::create_from_config(
    const isa::OutputStationaryProgramBuilder::Config& config,
    DataType dtype) {

    // Build program directly from config
    isa::OutputStationaryProgramBuilder builder(config);
    isa::DMProgram program = builder.build();

    // Create kernel with matmul type
    return Kernel(std::move(program), KernelOpType::MATMUL, dtype);
}

Kernel Kernel::create_mlp(Size M, Size N, Size K,
                          ActivationType activation,
                          bool has_bias,
                          DataType dtype) {
    // Use KernelCompiler with MLP options
    compiler::KernelCompiler compiler;
    return compiler.compile_mlp(M, N, K, activation, has_bias, dtype);
}

Kernel Kernel::create_conv2d(const Conv2DConfig& config,
                             bool has_bias,
                             ActivationType activation,
                             DataType dtype) {
    // Conv2D uses im2col + GEMM approach
    // GEMM dimensions: [M, K] @ [K, N] -> [M, N]
    // where M = batch * H_out * W_out
    //       K = C_in * kernel_h * kernel_w
    //       N = C_out

    compiler::KernelCompiler compiler;

    // Compile as MLP (matmul + optional bias + activation)
    Kernel kernel = compiler.compile_mlp(
        config.gemm_M(),
        config.gemm_N(),
        config.gemm_K(),
        activation,
        has_bias,
        dtype
    );

    // Override operation type to CONV2D
    kernel.op_type_ = KernelOpType::CONV2D;
    kernel.conv2d_config_ = config;
    kernel.has_bias_ = has_bias;
    kernel.activation_ = activation;

    // Rename the program
    std::ostringstream name;
    name << "conv2d_" << config.batch_size << "x"
         << config.in_channels << "x"
         << config.input_height << "x" << config.input_width
         << "_k" << config.kernel_height << "x" << config.kernel_width
         << "_s" << config.stride_h << "x" << config.stride_w;
    kernel.program_.name = name.str();

    // Set up Conv2D-specific arguments
    kernel.setup_conv2d_arguments();

    return kernel;
}

Kernel Kernel::create_conv2d(Size batch_size, Size in_channels, Size out_channels,
                             Size input_height, Size input_width,
                             Size kernel_size, Size stride, Size padding,
                             bool has_bias,
                             ActivationType activation,
                             DataType dtype) {
    Conv2DConfig config;
    config.batch_size = batch_size;
    config.in_channels = in_channels;
    config.out_channels = out_channels;
    config.input_height = input_height;
    config.input_width = input_width;
    config.kernel_height = kernel_size;
    config.kernel_width = kernel_size;
    config.stride_h = stride;
    config.stride_w = stride;
    config.padding_h = padding;
    config.padding_w = padding;
    config.dilation_h = 1;
    config.dilation_w = 1;
    config.groups = 1;

    return create_conv2d(config, has_bias, activation, dtype);
}

Kernel Kernel::create_attention(const AttentionConfig& config, DataType dtype) {
    // Attention is composed from matmul operations:
    // 1. QKV projections (if enabled): Q = X @ W_Q, K = X @ W_K, V = X @ W_V
    // 2. Attention scores: scores = Q @ K^T / sqrt(d_k)
    // 3. Softmax: attention_weights = softmax(scores)
    // 4. Context: context = attention_weights @ V
    // 5. Output projection (if enabled): output = context @ W_O

    compiler::KernelCompiler compiler;

    // For the underlying program, we use the QK matmul dimensions as the primary
    // This is: [B*H, S, d_k] @ [B*H, d_k, S] -> [B*H, S, S]
    Size B = config.batch_size;
    Size S = config.seq_len;
    Size D = config.d_model;
    Size H = config.num_heads;
    Size d_k = config.head_dim();

    // Create a placeholder program using the largest matmul (QKV projection)
    // The actual execution will compose multiple operations
    Size M, N, K;
    if (config.include_qkv_projection) {
        // QKV projection: [B*S, D] @ [D, D]
        M = B * S;
        N = D;
        K = D;
    } else {
        // Attention core: [B*H, S, d_k] @ [B*H, d_k, S]
        M = S;
        N = S;
        K = d_k;
    }

    // Compile the base matmul kernel
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;
    Kernel kernel = compiler.compile_matmul(M, N, K, opts);

    // Override operation type to ATTENTION
    kernel.op_type_ = KernelOpType::ATTENTION;
    kernel.attention_config_ = config;
    kernel.dtype_ = dtype;

    // Rename the program
    std::ostringstream name;
    name << "attention_b" << B << "_s" << S << "_d" << D << "_h" << H;
    if (config.causal_mask) {
        name << "_causal";
    }
    kernel.program_.name = name.str();

    // Set up attention-specific arguments
    kernel.setup_attention_arguments();

    return kernel;
}

Kernel Kernel::create_attention(Size batch_size, Size seq_len, Size d_model,
                                Size num_heads, bool causal_mask,
                                bool include_qkv_projection,
                                bool include_output_projection,
                                DataType dtype) {
    AttentionConfig config;
    config.batch_size = batch_size;
    config.seq_len = seq_len;
    config.d_model = d_model;
    config.num_heads = num_heads;
    config.causal_mask = causal_mask;
    config.include_qkv_projection = include_qkv_projection;
    config.include_output_projection = include_output_projection;

    return create_attention(config, dtype);
}

Kernel Kernel::create_layernorm(const LayerNormConfig& config, DataType dtype) {
    // LayerNorm computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
    // This is primarily a reduction + elementwise operation
    // For the underlying program, we model it as operating on [B*S, D] data

    compiler::KernelCompiler compiler;

    Size B = config.batch_size;
    Size S = config.seq_len;
    Size D = config.normalized_dim;

    // LayerNorm is not a matmul, but we create a placeholder program
    // using a simple matmul structure for resource estimation
    // In practice, layernorm would use the Vector Engine for reductions
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;

    // Use a 1xD matmul as placeholder for the reduction operations
    // This gives reasonable resource estimates for the D-dimension operations
    Kernel kernel = compiler.compile_matmul(1, D, D, opts);

    // Override operation type to LAYERNORM
    kernel.op_type_ = KernelOpType::LAYERNORM;
    kernel.layernorm_config_ = config;
    kernel.dtype_ = dtype;
    kernel.has_bias_ = config.has_bias;

    // Rename the program
    std::ostringstream name;
    name << "layernorm_b" << B << "_s" << S << "_d" << D;
    kernel.program_.name = name.str();

    // Set up layernorm-specific arguments
    kernel.setup_layernorm_arguments();

    return kernel;
}

Kernel Kernel::create_layernorm(Size batch_size, Size seq_len, Size normalized_dim,
                                float eps, bool has_weight, bool has_bias,
                                DataType dtype) {
    LayerNormConfig config;
    config.batch_size = batch_size;
    config.seq_len = seq_len;
    config.normalized_dim = normalized_dim;
    config.eps = eps;
    config.has_weight = has_weight;
    config.has_bias = has_bias;

    return create_layernorm(config, dtype);
}

Kernel Kernel::create_rmsnorm(const RMSNormConfig& config, DataType dtype) {
    // RMSNorm computes: y = x / sqrt(mean(x^2) + eps) * gamma
    // This is primarily a reduction + elementwise operation

    compiler::KernelCompiler compiler;

    Size B = config.batch_size;
    Size S = config.seq_len;
    Size D = config.normalized_dim;

    // RMSNorm is not a matmul, but we create a placeholder program
    // using a simple matmul structure for resource estimation
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;

    // Use a 1xD matmul as placeholder for the reduction operations
    Kernel kernel = compiler.compile_matmul(1, D, D, opts);

    // Override operation type to RMSNORM
    kernel.op_type_ = KernelOpType::RMSNORM;
    kernel.rmsnorm_config_ = config;
    kernel.dtype_ = dtype;

    // Rename the program
    std::ostringstream name;
    name << "rmsnorm_b" << B << "_s" << S << "_d" << D;
    kernel.program_.name = name.str();

    // Set up rmsnorm-specific arguments
    kernel.setup_rmsnorm_arguments();

    return kernel;
}

Kernel Kernel::create_rmsnorm(Size batch_size, Size seq_len, Size normalized_dim,
                              float eps, DataType dtype) {
    RMSNormConfig config;
    config.batch_size = batch_size;
    config.seq_len = seq_len;
    config.normalized_dim = normalized_dim;
    config.eps = eps;

    return create_rmsnorm(config, dtype);
}

Kernel Kernel::create_batchnorm(const BatchNormConfig& config, DataType dtype) {
    compiler::KernelCompiler compiler;
    Size N = config.batch_size;
    Size C = config.num_features;
    Size H = config.height;
    Size W = config.width;

    // BatchNorm is elementwise, but we compile a small matmul as placeholder
    // for instruction generation (actual compute is per-element)
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;
    Kernel kernel = compiler.compile_matmul(1, C, C, opts);

    // Override operation type and config
    kernel.op_type_ = KernelOpType::BATCHNORM;
    kernel.batchnorm_config_ = config;
    kernel.dtype_ = dtype;

    // Generate descriptive name
    std::ostringstream name;
    name << "batchnorm_n" << N << "_c" << C << "_h" << H << "_w" << W;
    kernel.program_.name = name.str();

    // Set up batchnorm-specific arguments
    kernel.setup_batchnorm_arguments();

    return kernel;
}

Kernel Kernel::create_batchnorm(Size batch_size, Size num_features,
                                Size height, Size width,
                                float eps, bool affine, DataType dtype) {
    BatchNormConfig config;
    config.batch_size = batch_size;
    config.num_features = num_features;
    config.height = height;
    config.width = width;
    config.eps = eps;
    config.affine = affine;

    return create_batchnorm(config, dtype);
}

Kernel Kernel::create_elementwise(const ElementwiseConfig& config, DataType dtype) {
    compiler::KernelCompiler compiler;
    Size total = config.total_elements();

    // Elementwise is simple, compile a small matmul as placeholder
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;
    Kernel kernel = compiler.compile_matmul(1, total, 1, opts);

    // Override operation type and config
    kernel.op_type_ = KernelOpType::ELEMENTWISE;
    kernel.elementwise_config_ = config;
    kernel.dtype_ = dtype;

    // Generate descriptive name
    std::ostringstream name;
    name << "elementwise_" << elementwise_op_name(config.op);
    for (Size dim : config.shape) {
        name << "_" << dim;
    }
    kernel.program_.name = name.str();

    // Set up elementwise-specific arguments
    kernel.setup_elementwise_arguments();

    return kernel;
}

Kernel Kernel::create_elementwise(ElementwiseOp op,
                                  const std::vector<Size>& shape,
                                  DataType dtype) {
    ElementwiseConfig config;
    config.op = op;
    config.shape = shape;
    config.is_unary = false;
    config.is_scalar_b = false;

    return create_elementwise(config, dtype);
}

Kernel Kernel::create_elementwise_unary(ElementwiseOp op,
                                        const std::vector<Size>& shape,
                                        DataType dtype) {
    ElementwiseConfig config;
    config.op = op;
    config.shape = shape;
    config.is_unary = true;
    config.is_scalar_b = false;

    return create_elementwise(config, dtype);
}

Kernel Kernel::create_elementwise_scalar(ElementwiseOp op,
                                         const std::vector<Size>& shape,
                                         float scalar,
                                         DataType dtype) {
    ElementwiseConfig config;
    config.op = op;
    config.shape = shape;
    config.is_unary = false;
    config.is_scalar_b = true;
    config.scalar_value = scalar;

    return create_elementwise(config, dtype);
}

// ============================================================================
// Pool2D Kernel Creation (v0.5.6)
// ============================================================================

Kernel Kernel::create_pool2d(const Pool2DConfig& config, DataType dtype) {
    compiler::KernelCompiler compiler;

    // Pool2D is a spatial reduction operation
    // Use a small matmul as placeholder for the program
    Size out_elems = config.output_elements();
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;
    Kernel kernel = compiler.compile_matmul(1, out_elems, 1, opts);

    // Override operation type and config
    kernel.op_type_ = KernelOpType::POOL2D;
    kernel.pool2d_config_ = config;
    kernel.dtype_ = dtype;

    // Generate descriptive name
    std::ostringstream name;
    name << pool_type_name(config.pool_type) << "_pool2d_"
         << config.batch_size << "x" << config.channels << "x"
         << config.input_height << "x" << config.input_width;
    if (config.pool_type != PoolType::GLOBAL_AVG) {
        name << "_k" << config.kernel_height << "x" << config.kernel_width
             << "_s" << config.stride_h << "_p" << config.padding_h;
    }
    kernel.program_.name = name.str();

    // Set up pool2d-specific arguments
    kernel.setup_pool2d_arguments();

    return kernel;
}

Kernel Kernel::create_max_pool2d(Size batch_size, Size channels,
                                 Size input_height, Size input_width,
                                 Size kernel_size, Size stride, Size padding,
                                 DataType dtype) {
    Pool2DConfig config;
    config.pool_type = PoolType::MAX;
    config.batch_size = batch_size;
    config.channels = channels;
    config.input_height = input_height;
    config.input_width = input_width;
    config.kernel_height = kernel_size;
    config.kernel_width = kernel_size;
    config.stride_h = (stride == 0) ? kernel_size : stride;  // Default: stride = kernel_size
    config.stride_w = (stride == 0) ? kernel_size : stride;
    config.padding_h = padding;
    config.padding_w = padding;

    return create_pool2d(config, dtype);
}

Kernel Kernel::create_avg_pool2d(Size batch_size, Size channels,
                                 Size input_height, Size input_width,
                                 Size kernel_size, Size stride, Size padding,
                                 DataType dtype) {
    Pool2DConfig config;
    config.pool_type = PoolType::AVG;
    config.batch_size = batch_size;
    config.channels = channels;
    config.input_height = input_height;
    config.input_width = input_width;
    config.kernel_height = kernel_size;
    config.kernel_width = kernel_size;
    config.stride_h = (stride == 0) ? kernel_size : stride;  // Default: stride = kernel_size
    config.stride_w = (stride == 0) ? kernel_size : stride;
    config.padding_h = padding;
    config.padding_w = padding;

    return create_pool2d(config, dtype);
}

Kernel Kernel::create_global_avg_pool2d(Size batch_size, Size channels,
                                        Size input_height, Size input_width,
                                        DataType dtype) {
    Pool2DConfig config;
    config.pool_type = PoolType::GLOBAL_AVG;
    config.batch_size = batch_size;
    config.channels = channels;
    config.input_height = input_height;
    config.input_width = input_width;
    // For global avg pool, kernel covers entire spatial dimension
    config.kernel_height = input_height;
    config.kernel_width = input_width;
    config.stride_h = 1;
    config.stride_w = 1;
    config.padding_h = 0;
    config.padding_w = 0;

    return create_pool2d(config, dtype);
}

// ============================================================================
// Softmax Kernel Creation (v0.5.7)
// ============================================================================

Kernel Kernel::create_softmax(const SoftmaxConfig& config, DataType dtype) {
    compiler::KernelCompiler compiler;

    // Softmax is a reduction operation - use small matmul as placeholder
    Size total = config.total_elements();
    compiler::CompileOptions opts = compiler::CompileOptions::defaults();
    opts.dtype = dtype;
    Kernel kernel = compiler.compile_matmul(1, total, 1, opts);

    // Override operation type and config
    kernel.op_type_ = KernelOpType::SOFTMAX;
    kernel.softmax_config_ = config;
    kernel.dtype_ = dtype;

    // Generate descriptive name
    std::ostringstream name;
    name << "softmax_";
    for (size_t i = 0; i < config.shape.size(); ++i) {
        if (i > 0) name << "x";
        name << config.shape[i];
    }
    name << "_axis" << config.axis;
    kernel.program_.name = name.str();

    // Set up softmax-specific arguments
    kernel.setup_softmax_arguments();

    return kernel;
}

Kernel Kernel::create_softmax(const std::vector<Size>& shape,
                              int axis,
                              DataType dtype) {
    SoftmaxConfig config(shape, axis);
    return create_softmax(config, dtype);
}

// ============================================================================
// Argument Accessors
// ============================================================================

std::vector<KernelArgument> Kernel::input_arguments() const {
    std::vector<KernelArgument> inputs;
    for (const auto& arg : arguments_) {
        if (!arg.is_output) {
            inputs.push_back(arg);
        }
    }
    return inputs;
}

std::vector<KernelArgument> Kernel::output_arguments() const {
    std::vector<KernelArgument> outputs;
    for (const auto& arg : arguments_) {
        if (arg.is_output) {
            outputs.push_back(arg);
        }
    }
    return outputs;
}

Size Kernel::total_input_bytes() const {
    Size total = 0;
    for (const auto& arg : arguments_) {
        if (!arg.is_output) {
            total += arg.size_bytes;
        }
    }
    return total;
}

Size Kernel::total_output_bytes() const {
    Size total = 0;
    for (const auto& arg : arguments_) {
        if (arg.is_output) {
            total += arg.size_bytes;
        }
    }
    return total;
}

// ============================================================================
// Utility Methods
// ============================================================================

std::string Kernel::summary() const {
    std::ostringstream ss;

    ss << "Kernel: " << (program_.name.empty() ? "(unnamed)" : program_.name) << "\n";
    ss << "  Operation: " << kernel_op_type_name(op_type_) << "\n";
    ss << "  Data Type: " << dtype_name(dtype_) << "\n";

    if (op_type_ == KernelOpType::MATMUL || op_type_ == KernelOpType::BATCH_MATMUL ||
        op_type_ == KernelOpType::MLP) {
        ss << "  Dimensions: M=" << M() << ", N=" << N() << ", K=" << K() << "\n";
        ss << "  Tile Sizes: Ti=" << Ti() << ", Tj=" << Tj() << ", Tk=" << Tk() << "\n";
    }

    if (op_type_ == KernelOpType::MLP) {
        ss << "  Activation: " << activation_type_name(activation_) << "\n";
        ss << "  Has Bias: " << (has_bias_ ? "yes" : "no") << "\n";
    }

    if (op_type_ == KernelOpType::CONV2D) {
        const auto& cfg = conv2d_config_;
        ss << "  Input: [" << cfg.batch_size << ", " << cfg.in_channels
           << ", " << cfg.input_height << ", " << cfg.input_width << "]\n";
        ss << "  Kernel: " << cfg.kernel_height << "x" << cfg.kernel_width << "\n";
        ss << "  Stride: " << cfg.stride_h << "x" << cfg.stride_w
           << ", Padding: " << cfg.padding_h << "x" << cfg.padding_w << "\n";
        ss << "  Output: [" << cfg.batch_size << ", " << cfg.out_channels
           << ", " << cfg.output_height() << ", " << cfg.output_width() << "]\n";
        ss << "  GEMM Dims: M=" << cfg.gemm_M() << ", N=" << cfg.gemm_N()
           << ", K=" << cfg.gemm_K() << "\n";
        ss << "  Activation: " << activation_type_name(activation_) << "\n";
        ss << "  Has Bias: " << (has_bias_ ? "yes" : "no") << "\n";
    }

    if (op_type_ == KernelOpType::ATTENTION) {
        const auto& cfg = attention_config_;
        ss << "  Batch Size: " << cfg.batch_size << "\n";
        ss << "  Seq Length: " << cfg.seq_len << "\n";
        ss << "  Model Dim: " << cfg.d_model << "\n";
        ss << "  Num Heads: " << cfg.num_heads << "\n";
        ss << "  Head Dim: " << cfg.head_dim() << "\n";
        ss << "  Causal Mask: " << (cfg.causal_mask ? "yes" : "no") << "\n";
        ss << "  QKV Projection: " << (cfg.include_qkv_projection ? "yes" : "no") << "\n";
        ss << "  Output Projection: " << (cfg.include_output_projection ? "yes" : "no") << "\n";
    }

    if (op_type_ == KernelOpType::LAYERNORM) {
        const auto& cfg = layernorm_config_;
        ss << "  Batch Size: " << cfg.batch_size << "\n";
        ss << "  Seq Length: " << cfg.seq_len << "\n";
        ss << "  Normalized Dim: " << cfg.normalized_dim << "\n";
        ss << "  Epsilon: " << cfg.eps << "\n";
        ss << "  Has Weight: " << (cfg.has_weight ? "yes" : "no") << "\n";
        ss << "  Has Bias: " << (cfg.has_bias ? "yes" : "no") << "\n";
    }

    if (op_type_ == KernelOpType::RMSNORM) {
        const auto& cfg = rmsnorm_config_;
        ss << "  Batch Size: " << cfg.batch_size << "\n";
        ss << "  Seq Length: " << cfg.seq_len << "\n";
        ss << "  Normalized Dim: " << cfg.normalized_dim << "\n";
        ss << "  Epsilon: " << cfg.eps << "\n";
    }

    if (op_type_ == KernelOpType::BATCHNORM) {
        const auto& cfg = batchnorm_config_;
        ss << "  Batch Size: " << cfg.batch_size << "\n";
        ss << "  Num Features: " << cfg.num_features << "\n";
        ss << "  Height: " << cfg.height << "\n";
        ss << "  Width: " << cfg.width << "\n";
        ss << "  Epsilon: " << cfg.eps << "\n";
        ss << "  Affine: " << (cfg.affine ? "yes" : "no") << "\n";
    }

    if (op_type_ == KernelOpType::ELEMENTWISE) {
        const auto& cfg = elementwise_config_;
        ss << "  Operation: " << elementwise_op_name(cfg.op) << "\n";
        ss << "  Shape: [";
        for (size_t i = 0; i < cfg.shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << cfg.shape[i];
        }
        ss << "]\n";
        ss << "  Total Elements: " << cfg.total_elements() << "\n";
        ss << "  Is Unary: " << (cfg.is_unary ? "yes" : "no") << "\n";
        if (cfg.is_scalar_b) {
            ss << "  Scalar Value: " << cfg.scalar_value << "\n";
        }
    }

    if (op_type_ == KernelOpType::POOL2D) {
        const auto& cfg = pool2d_config_;
        ss << "  Pool Type: " << pool_type_name(cfg.pool_type) << "\n";
        ss << "  Input: [" << cfg.batch_size << ", " << cfg.channels
           << ", " << cfg.input_height << ", " << cfg.input_width << "]\n";
        if (cfg.pool_type != PoolType::GLOBAL_AVG) {
            ss << "  Kernel: " << cfg.kernel_height << "x" << cfg.kernel_width << "\n";
            ss << "  Stride: " << cfg.stride_h << "x" << cfg.stride_w
               << ", Padding: " << cfg.padding_h << "x" << cfg.padding_w << "\n";
        }
        ss << "  Output: [" << cfg.batch_size << ", " << cfg.channels
           << ", " << cfg.output_height() << ", " << cfg.output_width() << "]\n";
        ss << "  Window Size: " << cfg.window_size() << " elements\n";
    }

    if (op_type_ == KernelOpType::SOFTMAX) {
        const auto& cfg = softmax_config_;
        ss << "  Shape: [";
        for (size_t i = 0; i < cfg.shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << cfg.shape[i];
        }
        ss << "]\n";
        ss << "  Axis: " << cfg.axis << " (normalized: " << cfg.normalized_axis() << ")\n";
        ss << "  Reduction Size: " << cfg.reduction_size() << "\n";
        ss << "  Num Softmax Ops: " << cfg.num_softmax_ops() << "\n";
    }

    ss << "  Program Size: " << instruction_count() << " operations\n";
    ss << "  Arguments:\n";
    for (const auto& arg : arguments_) {
        ss << "    " << arg.name << ": "
           << dtype_name(arg.dtype) << "[";
        for (size_t i = 0; i < arg.shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << arg.shape[i];
        }
        ss << "] = " << arg.size_bytes << " bytes"
           << (arg.is_output ? " (output)" : " (input)") << "\n";
    }

    ss << "  FLOPs: " << total_flops() << "\n";
    ss << "  Arithmetic Intensity: " << std::fixed << std::setprecision(2)
       << arithmetic_intensity() << " FLOPs/byte\n";

    return ss.str();
}

bool Kernel::validate(std::string& error) const {
    if (program_.instructions.empty()) {
        error = "Kernel program is empty";
        return false;
    }

    if (op_type_ == KernelOpType::MATMUL || op_type_ == KernelOpType::BATCH_MATMUL) {
        if (M() == 0 || N() == 0 || K() == 0) {
            error = "Matrix dimensions must be non-zero";
            return false;
        }
        if (Ti() == 0 || Tj() == 0 || Tk() == 0) {
            error = "Tile sizes must be non-zero";
            return false;
        }
        if (arguments_.size() < 3) {
            error = "MATMUL kernel must have at least 3 arguments (A, B, C)";
            return false;
        }
    }

    if (op_type_ == KernelOpType::MLP) {
        if (M() == 0 || N() == 0 || K() == 0) {
            error = "Matrix dimensions must be non-zero";
            return false;
        }
        if (Ti() == 0 || Tj() == 0 || Tk() == 0) {
            error = "Tile sizes must be non-zero";
            return false;
        }
        // MLP: A, B, [bias,] C
        size_t expected_args = has_bias_ ? 4 : 3;
        if (arguments_.size() < expected_args) {
            error = "MLP kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    if (op_type_ == KernelOpType::CONV2D) {
        const auto& cfg = conv2d_config_;

        if (cfg.batch_size == 0) {
            error = "Conv2D batch size must be non-zero";
            return false;
        }
        if (cfg.in_channels == 0 || cfg.out_channels == 0) {
            error = "Conv2D channel dimensions must be non-zero";
            return false;
        }
        if (cfg.input_height == 0 || cfg.input_width == 0) {
            error = "Conv2D input dimensions must be non-zero";
            return false;
        }
        if (cfg.kernel_height == 0 || cfg.kernel_width == 0) {
            error = "Conv2D kernel dimensions must be non-zero";
            return false;
        }
        if (cfg.stride_h == 0 || cfg.stride_w == 0) {
            error = "Conv2D stride must be non-zero";
            return false;
        }
        if (cfg.output_height() == 0 || cfg.output_width() == 0) {
            error = "Conv2D output dimensions would be zero (check padding/stride)";
            return false;
        }
        if (cfg.groups == 0 || cfg.in_channels % cfg.groups != 0) {
            error = "Conv2D groups must divide input channels evenly";
            return false;
        }

        // CONV2D: input, weight, [bias,] output
        size_t expected_args = has_bias_ ? 4 : 3;
        if (arguments_.size() < expected_args) {
            error = "CONV2D kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    if (op_type_ == KernelOpType::ATTENTION) {
        const auto& cfg = attention_config_;

        if (cfg.batch_size == 0) {
            error = "Attention batch size must be non-zero";
            return false;
        }
        if (cfg.seq_len == 0) {
            error = "Attention sequence length must be non-zero";
            return false;
        }
        if (cfg.d_model == 0) {
            error = "Attention model dimension must be non-zero";
            return false;
        }
        if (cfg.num_heads == 0) {
            error = "Attention num_heads must be non-zero";
            return false;
        }
        if (cfg.d_model % cfg.num_heads != 0) {
            error = "Attention d_model must be divisible by num_heads";
            return false;
        }

        // Validate argument count based on configuration
        size_t expected_args;
        if (cfg.include_qkv_projection) {
            // X, W_Q, W_K, W_V, [W_O,] output
            expected_args = cfg.include_output_projection ? 6 : 5;
        } else {
            // Q, K, V, output
            expected_args = 4;
        }
        if (arguments_.size() < expected_args) {
            error = "ATTENTION kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    if (op_type_ == KernelOpType::LAYERNORM) {
        const auto& cfg = layernorm_config_;

        if (cfg.batch_size == 0) {
            error = "LayerNorm batch size must be non-zero";
            return false;
        }
        if (cfg.seq_len == 0) {
            error = "LayerNorm sequence length must be non-zero";
            return false;
        }
        if (cfg.normalized_dim == 0) {
            error = "LayerNorm normalized dimension must be non-zero";
            return false;
        }
        if (cfg.eps <= 0) {
            error = "LayerNorm epsilon must be positive";
            return false;
        }

        // Validate argument count: input, [weight,] [bias,] output
        size_t expected_args = 2;  // input + output
        if (cfg.has_weight) expected_args++;
        if (cfg.has_bias) expected_args++;
        if (arguments_.size() < expected_args) {
            error = "LAYERNORM kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    // RMSNORM validation
    if (op_type_ == KernelOpType::RMSNORM) {
        const auto& cfg = rmsnorm_config_;
        if (cfg.batch_size == 0) {
            error = "RMSNorm batch size must be non-zero";
            return false;
        }
        if (cfg.seq_len == 0) {
            error = "RMSNorm sequence length must be non-zero";
            return false;
        }
        if (cfg.normalized_dim == 0) {
            error = "RMSNorm normalized dimension must be non-zero";
            return false;
        }
        if (cfg.eps <= 0) {
            error = "RMSNorm epsilon must be positive";
            return false;
        }

        // Validate argument count: input, weight (gamma), output
        // RMSNorm always has weight (gamma) but no bias
        size_t expected_args = 3;  // input + weight + output
        if (arguments_.size() < expected_args) {
            error = "RMSNORM kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    // BATCHNORM validation
    if (op_type_ == KernelOpType::BATCHNORM) {
        const auto& cfg = batchnorm_config_;
        if (cfg.batch_size == 0) {
            error = "BatchNorm batch size must be non-zero";
            return false;
        }
        if (cfg.num_features == 0) {
            error = "BatchNorm num_features must be non-zero";
            return false;
        }
        if (cfg.height == 0) {
            error = "BatchNorm height must be non-zero";
            return false;
        }
        if (cfg.width == 0) {
            error = "BatchNorm width must be non-zero";
            return false;
        }
        if (cfg.eps <= 0) {
            error = "BatchNorm epsilon must be positive";
            return false;
        }

        // Validate argument count:
        // input, running_mean, running_var, [weight,] [bias,] output
        size_t expected_args = 4;  // input + running_mean + running_var + output
        if (cfg.affine) expected_args += 2;  // weight + bias
        if (arguments_.size() < expected_args) {
            error = "BATCHNORM kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    // ELEMENTWISE validation
    if (op_type_ == KernelOpType::ELEMENTWISE) {
        const auto& cfg = elementwise_config_;
        if (cfg.shape.empty()) {
            error = "Elementwise shape must not be empty";
            return false;
        }
        for (Size dim : cfg.shape) {
            if (dim == 0) {
                error = "Elementwise shape dimensions must be non-zero";
                return false;
            }
        }

        // Validate argument count:
        // Binary: A, B, C (3)
        // Unary: A, C (2)
        // Scalar: A, C (2) - scalar is embedded in config
        size_t expected_args = cfg.is_unary || cfg.is_scalar_b ? 2 : 3;
        if (arguments_.size() < expected_args) {
            error = "ELEMENTWISE kernel must have at least " + std::to_string(expected_args) + " arguments";
            return false;
        }
    }

    // POOL2D validation
    if (op_type_ == KernelOpType::POOL2D) {
        const auto& cfg = pool2d_config_;
        if (cfg.batch_size == 0) {
            error = "Pool2D batch size must be non-zero";
            return false;
        }
        if (cfg.channels == 0) {
            error = "Pool2D channels must be non-zero";
            return false;
        }
        if (cfg.input_height == 0 || cfg.input_width == 0) {
            error = "Pool2D input dimensions must be non-zero";
            return false;
        }
        if (cfg.pool_type != PoolType::GLOBAL_AVG) {
            if (cfg.kernel_height == 0 || cfg.kernel_width == 0) {
                error = "Pool2D kernel size must be non-zero";
                return false;
            }
            if (cfg.stride_h == 0 || cfg.stride_w == 0) {
                error = "Pool2D stride must be non-zero";
                return false;
            }
        }
        // Validate output dimensions are valid
        if (cfg.output_height() == 0 || cfg.output_width() == 0) {
            error = "Pool2D output dimensions would be zero - check kernel/stride/padding";
            return false;
        }
        // Validate argument count: input, output (2)
        if (arguments_.size() < 2) {
            error = "POOL2D kernel must have at least 2 arguments (input, output)";
            return false;
        }
    }

    // SOFTMAX validation
    if (op_type_ == KernelOpType::SOFTMAX) {
        const auto& cfg = softmax_config_;
        if (cfg.shape.empty()) {
            error = "Softmax shape must not be empty";
            return false;
        }
        for (Size dim : cfg.shape) {
            if (dim == 0) {
                error = "Softmax shape dimensions must be non-zero";
                return false;
            }
        }
        // Validate axis is in range
        int ndim = static_cast<int>(cfg.shape.size());
        if (cfg.axis < -ndim || cfg.axis >= ndim) {
            error = "Softmax axis out of range";
            return false;
        }
        // Validate argument count: input, output (2)
        if (arguments_.size() < 2) {
            error = "SOFTMAX kernel must have at least 2 arguments (input, output)";
            return false;
        }
    }

    error.clear();
    return true;
}

double Kernel::arithmetic_intensity() const {
    Size total_bytes = total_input_bytes() + total_output_bytes();
    if (total_bytes == 0) return 0.0;

    return static_cast<double>(total_flops()) / static_cast<double>(total_bytes);
}

Size Kernel::total_flops() const {
    if (op_type_ == KernelOpType::MATMUL || op_type_ == KernelOpType::BATCH_MATMUL) {
        // Matrix multiplication: 2*M*N*K (multiply-add per element)
        return 2 * M() * N() * K();
    }
    if (op_type_ == KernelOpType::MLP) {
        // MLP = matmul + bias + activation
        // Matmul: 2*M*N*K
        Size flops = 2 * M() * N() * K();

        // Bias addition: M*N additions
        if (has_bias_) {
            flops += M() * N();
        }

        // Activation: M*N operations (varies by type, count as 1 op each)
        if (activation_ != ActivationType::NONE) {
            flops += M() * N();
        }

        return flops;
    }
    if (op_type_ == KernelOpType::CONV2D) {
        // Conv2D FLOPS from config
        Size flops = conv2d_config_.total_flops();

        // Output elements for bias/activation
        Size output_elements = conv2d_config_.batch_size *
                               conv2d_config_.out_channels *
                               conv2d_config_.output_height() *
                               conv2d_config_.output_width();

        // Bias addition
        if (has_bias_) {
            flops += output_elements;
        }

        // Activation
        if (activation_ != ActivationType::NONE) {
            flops += output_elements;
        }

        return flops;
    }
    if (op_type_ == KernelOpType::ATTENTION) {
        return attention_config_.total_flops();
    }
    if (op_type_ == KernelOpType::LAYERNORM) {
        return layernorm_config_.total_flops();
    }
    if (op_type_ == KernelOpType::RMSNORM) {
        return rmsnorm_config_.total_flops();
    }
    if (op_type_ == KernelOpType::BATCHNORM) {
        return batchnorm_config_.total_flops();
    }
    if (op_type_ == KernelOpType::ELEMENTWISE) {
        return elementwise_config_.total_flops();
    }
    if (op_type_ == KernelOpType::POOL2D) {
        return pool2d_config_.total_flops();
    }
    if (op_type_ == KernelOpType::SOFTMAX) {
        return softmax_config_.total_flops();
    }
    // For other operation types, could return estimates or 0
    return 0;
}

// ============================================================================
// Private Methods
// ============================================================================

void Kernel::setup_matmul_arguments() {
    arguments_.clear();

    // Input A: [M, K]
    arguments_.emplace_back(
        "A", dtype_,
        std::vector<Size>{M(), K()},
        false  // is_output = false
    );

    // Input B: [K, N]
    arguments_.emplace_back(
        "B", dtype_,
        std::vector<Size>{K(), N()},
        false
    );

    // Output C: [M, N]
    arguments_.emplace_back(
        "C", dtype_,
        std::vector<Size>{M(), N()},
        true   // is_output = true
    );
}

void Kernel::setup_mlp_arguments() {
    arguments_.clear();

    // Input A: [M, K]
    arguments_.emplace_back(
        "A", dtype_,
        std::vector<Size>{M(), K()},
        false
    );

    // Input B (weights): [K, N]
    arguments_.emplace_back(
        "B", dtype_,
        std::vector<Size>{K(), N()},
        false
    );

    // Bias: [N] (if enabled)
    if (has_bias_) {
        arguments_.emplace_back(
            "bias", dtype_,
            std::vector<Size>{N()},
            false
        );
    }

    // Output C: [M, N]
    arguments_.emplace_back(
        "C", dtype_,
        std::vector<Size>{M(), N()},
        true
    );
}

void Kernel::setup_conv2d_arguments() {
    arguments_.clear();

    const auto& cfg = conv2d_config_;

    // Input: [N, C_in, H, W] (NCHW format)
    arguments_.emplace_back(
        "input", dtype_,
        std::vector<Size>{cfg.batch_size, cfg.in_channels,
                          cfg.input_height, cfg.input_width},
        false
    );

    // Weight: [C_out, C_in/groups, K_h, K_w]
    arguments_.emplace_back(
        "weight", dtype_,
        std::vector<Size>{cfg.out_channels, cfg.in_channels / cfg.groups,
                          cfg.kernel_height, cfg.kernel_width},
        false
    );

    // Bias: [C_out] (if enabled)
    if (has_bias_) {
        arguments_.emplace_back(
            "bias", dtype_,
            std::vector<Size>{cfg.out_channels},
            false
        );
    }

    // Output: [N, C_out, H_out, W_out]
    arguments_.emplace_back(
        "output", dtype_,
        std::vector<Size>{cfg.batch_size, cfg.out_channels,
                          cfg.output_height(), cfg.output_width()},
        true
    );
}

void Kernel::setup_attention_arguments() {
    arguments_.clear();

    const auto& cfg = attention_config_;
    Size B = cfg.batch_size;
    Size S = cfg.seq_len;
    Size D = cfg.d_model;
    Size H = cfg.num_heads;
    Size d_k = cfg.head_dim();

    if (cfg.include_qkv_projection) {
        // Full attention with projections
        // Input: X [B, S, D]
        arguments_.emplace_back(
            "X", dtype_,
            std::vector<Size>{B, S, D},
            false
        );

        // Query projection: W_Q [D, D]
        arguments_.emplace_back(
            "W_Q", dtype_,
            std::vector<Size>{D, D},
            false
        );

        // Key projection: W_K [D, D]
        arguments_.emplace_back(
            "W_K", dtype_,
            std::vector<Size>{D, D},
            false
        );

        // Value projection: W_V [D, D]
        arguments_.emplace_back(
            "W_V", dtype_,
            std::vector<Size>{D, D},
            false
        );

        // Output projection: W_O [D, D] (if enabled)
        if (cfg.include_output_projection) {
            arguments_.emplace_back(
                "W_O", dtype_,
                std::vector<Size>{D, D},
                false
            );
        }

        // Output: [B, S, D]
        arguments_.emplace_back(
            "output", dtype_,
            std::vector<Size>{B, S, D},
            true
        );
    } else {
        // Attention core only (Q, K, V already computed)
        // Query: Q [B, H, S, d_k]
        arguments_.emplace_back(
            "Q", dtype_,
            std::vector<Size>{B, H, S, d_k},
            false
        );

        // Key: K [B, H, S, d_k]
        arguments_.emplace_back(
            "K", dtype_,
            std::vector<Size>{B, H, S, d_k},
            false
        );

        // Value: V [B, H, S, d_k]
        arguments_.emplace_back(
            "V", dtype_,
            std::vector<Size>{B, H, S, d_k},
            false
        );

        // Output: [B, H, S, d_k]
        arguments_.emplace_back(
            "output", dtype_,
            std::vector<Size>{B, H, S, d_k},
            true
        );
    }
}

void Kernel::setup_layernorm_arguments() {
    arguments_.clear();

    const auto& cfg = layernorm_config_;
    Size B = cfg.batch_size;
    Size S = cfg.seq_len;
    Size D = cfg.normalized_dim;

    // Input: [B, S, D]
    arguments_.emplace_back(
        "input", dtype_,
        std::vector<Size>{B, S, D},
        false
    );

    // Weight (gamma): [D] - scale parameter
    if (cfg.has_weight) {
        arguments_.emplace_back(
            "weight", dtype_,
            std::vector<Size>{D},
            false
        );
    }

    // Bias (beta): [D] - shift parameter
    if (cfg.has_bias) {
        arguments_.emplace_back(
            "bias", dtype_,
            std::vector<Size>{D},
            false
        );
    }

    // Output: [B, S, D]
    arguments_.emplace_back(
        "output", dtype_,
        std::vector<Size>{B, S, D},
        true
    );
}

void Kernel::setup_rmsnorm_arguments() {
    arguments_.clear();

    const auto& cfg = rmsnorm_config_;
    Size B = cfg.batch_size;
    Size S = cfg.seq_len;
    Size D = cfg.normalized_dim;

    // Input: [B, S, D]
    arguments_.emplace_back(
        "input", dtype_,
        std::vector<Size>{B, S, D},
        false
    );

    // Weight (gamma): [D] - scale parameter
    // RMSNorm always has weight, no bias (unlike LayerNorm)
    arguments_.emplace_back(
        "weight", dtype_,
        std::vector<Size>{D},
        false
    );

    // Output: [B, S, D]
    arguments_.emplace_back(
        "output", dtype_,
        std::vector<Size>{B, S, D},
        true
    );
}

void Kernel::setup_batchnorm_arguments() {
    arguments_.clear();

    const auto& cfg = batchnorm_config_;
    Size N = cfg.batch_size;
    Size C = cfg.num_features;
    Size H = cfg.height;
    Size W = cfg.width;

    // Input: [N, C, H, W]
    arguments_.emplace_back(
        "input", dtype_,
        std::vector<Size>{N, C, H, W},
        false
    );

    // Running mean: [C] - precomputed mean per channel
    arguments_.emplace_back(
        "running_mean", dtype_,
        std::vector<Size>{C},
        false
    );

    // Running variance: [C] - precomputed variance per channel
    arguments_.emplace_back(
        "running_var", dtype_,
        std::vector<Size>{C},
        false
    );

    // Weight (gamma): [C] - scale parameter (if affine)
    if (cfg.affine) {
        arguments_.emplace_back(
            "weight", dtype_,
            std::vector<Size>{C},
            false
        );
    }

    // Bias (beta): [C] - shift parameter (if affine)
    if (cfg.affine) {
        arguments_.emplace_back(
            "bias", dtype_,
            std::vector<Size>{C},
            false
        );
    }

    // Output: [N, C, H, W]
    arguments_.emplace_back(
        "output", dtype_,
        std::vector<Size>{N, C, H, W},
        true
    );
}

void Kernel::setup_elementwise_arguments() {
    arguments_.clear();

    const auto& cfg = elementwise_config_;

    // Input A: shape from config
    arguments_.emplace_back(
        "A", dtype_,
        cfg.shape,
        false
    );

    // Input B: only for binary operations (not unary, not scalar)
    if (!cfg.is_unary && !cfg.is_scalar_b) {
        arguments_.emplace_back(
            "B", dtype_,
            cfg.shape,
            false
        );
    }

    // Output C: same shape as input
    arguments_.emplace_back(
        "C", dtype_,
        cfg.shape,
        true
    );
}

void Kernel::setup_pool2d_arguments() {
    arguments_.clear();

    const auto& cfg = pool2d_config_;

    // Input: [N, C, H, W]
    arguments_.emplace_back(
        "input", dtype_,
        std::vector<Size>{cfg.batch_size, cfg.channels, cfg.input_height, cfg.input_width},
        false
    );

    // Output: [N, C, H_out, W_out]
    arguments_.emplace_back(
        "output", dtype_,
        std::vector<Size>{cfg.batch_size, cfg.channels, cfg.output_height(), cfg.output_width()},
        true
    );
}

void Kernel::setup_softmax_arguments() {
    arguments_.clear();

    const auto& cfg = softmax_config_;

    // Input: shape from config
    arguments_.emplace_back(
        "input", dtype_,
        cfg.shape,
        false
    );

    // Output: same shape as input
    arguments_.emplace_back(
        "output", dtype_,
        cfg.shape,
        true
    );
}

} // namespace sw::kpu
