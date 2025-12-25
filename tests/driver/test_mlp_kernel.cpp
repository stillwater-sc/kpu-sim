// Test suite for MLP Kernel abstraction
// Tests MLP kernel creation, compilation, and metadata

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>

using namespace sw::kpu;
using namespace sw::kpu::compiler;

// ============================================================================
// MLP Kernel Factory Tests
// ============================================================================

TEST_CASE("Kernel::create_mlp factory method", "[kernel][mlp]") {
    SECTION("Basic MLP creation with RELU") {
        Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                            ActivationType::RELU,
                                            true);  // has_bias

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.op_type() == KernelOpType::MLP);
        REQUIRE(kernel.dtype() == DataType::FLOAT32);
        REQUIRE(kernel.activation() == ActivationType::RELU);
        REQUIRE(kernel.has_bias() == true);
        REQUIRE(kernel.M() == 256);
        REQUIRE(kernel.N() == 256);
        REQUIRE(kernel.K() == 256);
    }

    SECTION("MLP with GELU activation") {
        Kernel kernel = Kernel::create_mlp(512, 1024, 768,
                                            ActivationType::GELU,
                                            true);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.activation() == ActivationType::GELU);
    }

    SECTION("MLP without bias") {
        Kernel kernel = Kernel::create_mlp(128, 128, 128,
                                            ActivationType::SIGMOID,
                                            false);  // no bias

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.has_bias() == false);
    }

    SECTION("MLP with different data types") {
        Kernel kernel_f16 = Kernel::create_mlp(256, 256, 256,
                                                ActivationType::TANH,
                                                true,
                                                DataType::FLOAT16);

        REQUIRE(kernel_f16.is_valid());
        REQUIRE(kernel_f16.dtype() == DataType::FLOAT16);
    }
}

// ============================================================================
// MLP Kernel Argument Tests
// ============================================================================

TEST_CASE("MLP kernel arguments", "[kernel][mlp]") {
    SECTION("MLP with bias has 4 arguments") {
        Kernel kernel = Kernel::create_mlp(256, 512, 128,
                                            ActivationType::RELU,
                                            true);

        REQUIRE(kernel.arguments().size() == 4);  // A, B, bias, C

        auto inputs = kernel.input_arguments();
        auto outputs = kernel.output_arguments();

        REQUIRE(inputs.size() == 3);  // A, B, bias
        REQUIRE(outputs.size() == 1);  // C

        // Check argument shapes
        bool found_A = false, found_B = false, found_bias = false, found_C = false;
        for (const auto& arg : kernel.arguments()) {
            if (arg.name == "A") {
                found_A = true;
                REQUIRE(arg.shape.size() == 2);
                REQUIRE(arg.shape[0] == 256);  // M
                REQUIRE(arg.shape[1] == 128);  // K
                REQUIRE(arg.is_output == false);
            }
            if (arg.name == "B") {
                found_B = true;
                REQUIRE(arg.shape[0] == 128);  // K
                REQUIRE(arg.shape[1] == 512);  // N
            }
            if (arg.name == "bias") {
                found_bias = true;
                REQUIRE(arg.shape.size() == 1);
                REQUIRE(arg.shape[0] == 512);  // N
            }
            if (arg.name == "C") {
                found_C = true;
                REQUIRE(arg.shape[0] == 256);  // M
                REQUIRE(arg.shape[1] == 512);  // N
                REQUIRE(arg.is_output == true);
            }
        }
        REQUIRE(found_A);
        REQUIRE(found_B);
        REQUIRE(found_bias);
        REQUIRE(found_C);
    }

    SECTION("MLP without bias has 3 arguments") {
        Kernel kernel = Kernel::create_mlp(256, 512, 128,
                                            ActivationType::RELU,
                                            false);  // no bias

        REQUIRE(kernel.arguments().size() == 3);  // A, B, C

        bool has_bias_arg = false;
        for (const auto& arg : kernel.arguments()) {
            if (arg.name == "bias") {
                has_bias_arg = true;
            }
        }
        REQUIRE(has_bias_arg == false);
    }
}

// ============================================================================
// MLP Kernel Size Calculations
// ============================================================================

TEST_CASE("MLP kernel byte size calculations", "[kernel][mlp]") {
    // M=256, N=512, K=128, Float32, with bias
    Kernel kernel = Kernel::create_mlp(256, 512, 128,
                                        ActivationType::GELU,
                                        true);

    SECTION("Total input bytes") {
        // A: 256 * 128 * 4 = 131072
        // B: 128 * 512 * 4 = 262144
        // bias: 512 * 4 = 2048
        // Total: 395264
        Size expected = (256 * 128 + 128 * 512 + 512) * 4;
        REQUIRE(kernel.total_input_bytes() == expected);
    }

    SECTION("Total output bytes") {
        // C: 256 * 512 * 4 = 524288
        Size expected = 256 * 512 * 4;
        REQUIRE(kernel.total_output_bytes() == expected);
    }
}

// ============================================================================
// MLP Kernel FLOPs Calculation
// ============================================================================

TEST_CASE("MLP kernel total_flops calculation", "[kernel][mlp]") {
    SECTION("MLP with bias and activation") {
        Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                            ActivationType::GELU,
                                            true);

        // FLOPs = 2*M*N*K (matmul) + M*N (bias) + M*N (activation)
        Size expected_flops = 2 * 256 * 256 * 256  // matmul
                            + 256 * 256             // bias
                            + 256 * 256;            // activation
        REQUIRE(kernel.total_flops() == expected_flops);
    }

    SECTION("MLP without bias") {
        Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                            ActivationType::RELU,
                                            false);

        Size expected_flops = 2 * 256 * 256 * 256  // matmul
                            + 256 * 256;            // activation only
        REQUIRE(kernel.total_flops() == expected_flops);
    }

    SECTION("MLP with NONE activation (just bias)") {
        Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                            ActivationType::NONE,
                                            true);

        Size expected_flops = 2 * 256 * 256 * 256  // matmul
                            + 256 * 256;            // bias only
        REQUIRE(kernel.total_flops() == expected_flops);
    }
}

// ============================================================================
// MLP Kernel Validation Tests
// ============================================================================

TEST_CASE("MLP kernel validation", "[kernel][mlp]") {
    SECTION("Valid MLP kernel passes validation") {
        Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                            ActivationType::RELU,
                                            true);
        std::string error;
        REQUIRE(kernel.validate(error) == true);
        REQUIRE(error.empty());
    }
}

// ============================================================================
// MLP Kernel Summary Tests
// ============================================================================

TEST_CASE("MLP kernel summary string", "[kernel][mlp]") {
    Kernel kernel = Kernel::create_mlp(256, 512, 128,
                                        ActivationType::GELU,
                                        true);

    std::string summary = kernel.summary();

    // Check that summary contains expected information
    REQUIRE(summary.find("mlp") != std::string::npos);
    REQUIRE(summary.find("256") != std::string::npos);  // M dimension
    REQUIRE(summary.find("512") != std::string::npos);  // N dimension
    REQUIRE(summary.find("128") != std::string::npos);  // K dimension
    REQUIRE(summary.find("gelu") != std::string::npos);  // activation
    REQUIRE(summary.find("bias") != std::string::npos);  // bias present
    REQUIRE(summary.find("FLOPs") != std::string::npos);
}

// ============================================================================
// KernelCompiler MLP Tests
// ============================================================================

TEST_CASE("KernelCompiler compile_mlp", "[kernel_compiler][mlp]") {
    KernelCompiler compiler;

    SECTION("Compile MLP with default options") {
        Kernel kernel = compiler.compile_mlp(256, 256, 256,
                                              ActivationType::RELU,
                                              true);

        REQUIRE(compiler.last_succeeded());
        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.op_type() == KernelOpType::MLP);
    }

    SECTION("Compile MLP with various activations") {
        std::vector<ActivationType> activations = {
            ActivationType::RELU,
            ActivationType::GELU,
            ActivationType::SIGMOID,
            ActivationType::TANH,
            ActivationType::SILU
        };

        for (ActivationType act : activations) {
            Kernel kernel = compiler.compile_mlp(128, 128, 128, act, true);
            REQUIRE(kernel.is_valid());
            REQUIRE(kernel.activation() == act);
        }
    }

    SECTION("Compilation statistics include activation info") {
        Kernel kernel = compiler.compile_mlp(512, 512, 512,
                                              ActivationType::GELU,
                                              true);

        const CompilationStats& stats = compiler.last_stats();
        REQUIRE(stats.instruction_count > 0);
        REQUIRE(stats.estimated_arithmetic_intensity > 0);
    }
}

// ============================================================================
// MLP Kernel Program Tests
// ============================================================================

TEST_CASE("MLP kernel program access", "[kernel][mlp]") {
    Kernel kernel = Kernel::create_mlp(256, 256, 256,
                                        ActivationType::RELU,
                                        true);

    SECTION("Program has instructions") {
        REQUIRE(kernel.instruction_count() > 0);
    }

    SECTION("Program name includes mlp identifier") {
        const isa::DMProgram& prog = kernel.program();
        REQUIRE(prog.name.find("mlp") != std::string::npos);
    }
}

// ============================================================================
// MLP Kernel Edge Cases
// ============================================================================

TEST_CASE("MLP kernel edge cases", "[kernel][mlp]") {
    SECTION("Small MLP (single tile)") {
        Kernel kernel = Kernel::create_mlp(16, 16, 16,
                                            ActivationType::RELU,
                                            true);
        REQUIRE(kernel.is_valid());
    }

    SECTION("Large MLP") {
        Kernel kernel = Kernel::create_mlp(2048, 2048, 2048,
                                            ActivationType::GELU,
                                            true);
        REQUIRE(kernel.is_valid());
    }

    SECTION("Non-square MLP") {
        Kernel kernel = Kernel::create_mlp(64, 4096, 768,
                                            ActivationType::SILU,
                                            true);
        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.M() == 64);
        REQUIRE(kernel.N() == 4096);
        REQUIRE(kernel.K() == 768);
    }
}
