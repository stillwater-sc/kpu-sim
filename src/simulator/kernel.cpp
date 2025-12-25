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

    ss << "  Instructions: " << instruction_count() << "\n";
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
        error = "Kernel has no instructions";
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

} // namespace sw::kpu
