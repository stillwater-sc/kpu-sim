// Test suite for Kernel abstraction and KernelCompiler
// Tests kernel creation, metadata, compilation, and statistics

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/kernel.hpp>
#include <sw/compiler/kernel_compiler.hpp>

using namespace sw::kpu;
using namespace sw::kpu::compiler;

// ============================================================================
// KernelOpType Tests
// ============================================================================

TEST_CASE("KernelOpType enumeration", "[kernel]") {
    SECTION("Operation type names") {
        REQUIRE(std::string(kernel_op_type_name(KernelOpType::MATMUL)) == "matmul");
        REQUIRE(std::string(kernel_op_type_name(KernelOpType::BATCH_MATMUL)) == "batch_matmul");
        REQUIRE(std::string(kernel_op_type_name(KernelOpType::CONV2D)) == "conv2d");
        REQUIRE(std::string(kernel_op_type_name(KernelOpType::ELEMENTWISE)) == "elementwise");
        REQUIRE(std::string(kernel_op_type_name(KernelOpType::CUSTOM)) == "custom");
    }
}

// ============================================================================
// KernelArgument Tests
// ============================================================================

TEST_CASE("KernelArgument construction", "[kernel]") {
    SECTION("Default constructor") {
        KernelArgument arg;
        REQUIRE(arg.name.empty());
        REQUIRE(arg.dtype == DataType::FLOAT32);
        REQUIRE(arg.is_output == false);
        REQUIRE(arg.size_bytes == 0);
    }

    SECTION("Parameterized constructor") {
        KernelArgument arg("A", DataType::FLOAT32, {1024, 512}, false);
        REQUIRE(arg.name == "A");
        REQUIRE(arg.dtype == DataType::FLOAT32);
        REQUIRE(arg.shape.size() == 2);
        REQUIRE(arg.shape[0] == 1024);
        REQUIRE(arg.shape[1] == 512);
        REQUIRE(arg.is_output == false);
        REQUIRE(arg.size_bytes == 1024 * 512 * 4);  // Float32 = 4 bytes
    }

    SECTION("Size computation for different data types") {
        KernelArgument f32("A", DataType::FLOAT32, {100, 100}, false);
        REQUIRE(f32.size_bytes == 100 * 100 * 4);

        KernelArgument f16("A", DataType::FLOAT16, {100, 100}, false);
        REQUIRE(f16.size_bytes == 100 * 100 * 2);

        KernelArgument i8("A", DataType::INT8, {100, 100}, false);
        REQUIRE(i8.size_bytes == 100 * 100 * 1);
    }
}

// ============================================================================
// Kernel Factory Method Tests
// ============================================================================

TEST_CASE("Kernel::create_matmul factory method", "[kernel]") {
    SECTION("Basic matmul creation") {
        Kernel kernel = Kernel::create_matmul(256, 256, 256);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.op_type() == KernelOpType::MATMUL);
        REQUIRE(kernel.dtype() == DataType::FLOAT32);
        REQUIRE(kernel.M() == 256);
        REQUIRE(kernel.N() == 256);
        REQUIRE(kernel.K() == 256);
    }

    SECTION("Matmul with custom data type") {
        Kernel kernel = Kernel::create_matmul(128, 128, 128, DataType::FLOAT16);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.dtype() == DataType::FLOAT16);
    }

    SECTION("Non-square matrices") {
        Kernel kernel = Kernel::create_matmul(512, 1024, 768);

        REQUIRE(kernel.M() == 512);
        REQUIRE(kernel.N() == 1024);
        REQUIRE(kernel.K() == 768);
    }

    SECTION("Small matrices") {
        Kernel kernel = Kernel::create_matmul(32, 32, 32);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.M() == 32);
    }

    SECTION("Large matrices") {
        Kernel kernel = Kernel::create_matmul(2048, 2048, 2048);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.instruction_count() > 0);
    }
}

// ============================================================================
// Kernel Metadata Tests
// ============================================================================

TEST_CASE("Kernel arguments for matmul", "[kernel]") {
    Kernel kernel = Kernel::create_matmul(256, 512, 128);

    SECTION("Total argument count") {
        REQUIRE(kernel.arguments().size() == 3);  // A, B, C
    }

    SECTION("Input arguments") {
        auto inputs = kernel.input_arguments();
        REQUIRE(inputs.size() == 2);  // A and B

        // Find A argument
        bool found_A = false;
        bool found_B = false;
        for (const auto& arg : inputs) {
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
                REQUIRE(arg.is_output == false);
            }
        }
        REQUIRE(found_A);
        REQUIRE(found_B);
    }

    SECTION("Output arguments") {
        auto outputs = kernel.output_arguments();
        REQUIRE(outputs.size() == 1);  // C only

        const auto& C = outputs[0];
        REQUIRE(C.name == "C");
        REQUIRE(C.shape[0] == 256);  // M
        REQUIRE(C.shape[1] == 512);  // N
        REQUIRE(C.is_output == true);
    }
}

TEST_CASE("Kernel byte size calculations", "[kernel]") {
    // M=256, N=512, K=128, Float32
    Kernel kernel = Kernel::create_matmul(256, 512, 128);

    SECTION("Total input bytes") {
        // A: 256 * 128 * 4 = 131072
        // B: 128 * 512 * 4 = 262144
        // Total: 393216
        Size expected = (256 * 128 + 128 * 512) * 4;
        REQUIRE(kernel.total_input_bytes() == expected);
    }

    SECTION("Total output bytes") {
        // C: 256 * 512 * 4 = 524288
        Size expected = 256 * 512 * 4;
        REQUIRE(kernel.total_output_bytes() == expected);
    }
}

// ============================================================================
// Kernel Utility Method Tests
// ============================================================================

TEST_CASE("Kernel total_flops calculation", "[kernel]") {
    Kernel kernel = Kernel::create_matmul(256, 256, 256);

    // FLOPs = 2 * M * N * K (multiply-add per element)
    Size expected_flops = 2 * 256 * 256 * 256;
    REQUIRE(kernel.total_flops() == expected_flops);
}

TEST_CASE("Kernel arithmetic_intensity calculation", "[kernel]") {
    Kernel kernel = Kernel::create_matmul(1024, 1024, 1024);

    // FLOPs = 2 * M * N * K = 2 * 1024^3
    // Bytes = (M*K + K*N + M*N) * 4 = 3 * 1024^2 * 4
    // Intensity = (2 * 1024^3) / (3 * 1024^2 * 4) = 2*1024 / 12 = 170.67

    double intensity = kernel.arithmetic_intensity();
    REQUIRE(intensity > 100.0);  // Should be relatively high
    REQUIRE(intensity < 200.0);  // But bounded
}

TEST_CASE("Kernel validation", "[kernel]") {
    SECTION("Valid kernel passes validation") {
        Kernel kernel = Kernel::create_matmul(256, 256, 256);
        std::string error;
        REQUIRE(kernel.validate(error) == true);
        REQUIRE(error.empty());
    }

    SECTION("Invalid (empty) kernel fails validation") {
        Kernel kernel;  // Default constructor creates invalid kernel
        std::string error;
        REQUIRE(kernel.validate(error) == false);
        REQUIRE(!error.empty());
    }
}

TEST_CASE("Kernel summary string", "[kernel]") {
    Kernel kernel = Kernel::create_matmul(256, 512, 128);

    std::string summary = kernel.summary();

    // Check that summary contains expected information
    REQUIRE(summary.find("matmul") != std::string::npos);
    REQUIRE(summary.find("256") != std::string::npos);  // M dimension
    REQUIRE(summary.find("512") != std::string::npos);  // N dimension
    REQUIRE(summary.find("128") != std::string::npos);  // K dimension
    REQUIRE(summary.find("FLOPs") != std::string::npos);
}

// ============================================================================
// Kernel Program Access Tests
// ============================================================================

TEST_CASE("Kernel program access", "[kernel]") {
    Kernel kernel = Kernel::create_matmul(256, 256, 256);

    SECTION("Program has instructions") {
        REQUIRE(kernel.instruction_count() > 0);
    }

    SECTION("Const program access") {
        const isa::DMProgram& prog = kernel.program();
        REQUIRE(prog.instructions.size() == kernel.instruction_count());
    }

    SECTION("Mutable program access") {
        isa::DMProgram& prog = kernel.program();
        // Should be able to access (though we don't modify in test)
        REQUIRE(prog.M == 256);
    }
}

// ============================================================================
// DataflowStrategy Tests
// ============================================================================

TEST_CASE("DataflowStrategy enumeration", "[kernel_compiler]") {
    SECTION("Strategy names") {
        REQUIRE(std::string(dataflow_strategy_name(DataflowStrategy::OUTPUT_STATIONARY))
                == "output_stationary");
        REQUIRE(std::string(dataflow_strategy_name(DataflowStrategy::WEIGHT_STATIONARY))
                == "weight_stationary");
        REQUIRE(std::string(dataflow_strategy_name(DataflowStrategy::INPUT_STATIONARY))
                == "input_stationary");
        REQUIRE(std::string(dataflow_strategy_name(DataflowStrategy::AUTO))
                == "auto");
    }
}

// ============================================================================
// CompileOptions Tests
// ============================================================================

TEST_CASE("CompileOptions construction", "[kernel_compiler]") {
    SECTION("Default options") {
        CompileOptions opts = CompileOptions::defaults();
        REQUIRE(opts.dataflow == DataflowStrategy::AUTO);
        REQUIRE(opts.is_auto_tiling() == true);
        REQUIRE(opts.double_buffer == true);
        REQUIRE(opts.enable_tile_caching == true);
        REQUIRE(opts.systolic_size == 16);
    }

    SECTION("Options with explicit tiles") {
        CompileOptions opts = CompileOptions::with_tiles(64, 64, 128);
        REQUIRE(opts.Ti == 64);
        REQUIRE(opts.Tj == 64);
        REQUIRE(opts.Tk == 128);
        REQUIRE(opts.is_auto_tiling() == false);
        REQUIRE(opts.dataflow == DataflowStrategy::OUTPUT_STATIONARY);
    }

    SECTION("Inference options") {
        CompileOptions opts = CompileOptions::for_inference();
        REQUIRE(opts.dataflow == DataflowStrategy::WEIGHT_STATIONARY);
    }
}

// ============================================================================
// KernelCompiler Tests
// ============================================================================

TEST_CASE("KernelCompiler basic compilation", "[kernel_compiler]") {
    KernelCompiler compiler;

    SECTION("Compile with default options") {
        Kernel kernel = compiler.compile_matmul(256, 256, 256);

        REQUIRE(compiler.last_succeeded());
        REQUIRE(kernel.is_valid());
    }

    SECTION("Compile with explicit tiles") {
        Kernel kernel = compiler.compile_matmul(256, 256, 256, 64, 64, 64);

        REQUIRE(compiler.last_succeeded());
        REQUIRE(kernel.Ti() == 64);
        REQUIRE(kernel.Tj() == 64);
        REQUIRE(kernel.Tk() == 64);
    }

    SECTION("Compile with options") {
        CompileOptions opts;
        opts.Ti = 32;
        opts.Tj = 32;
        opts.Tk = 64;
        opts.dtype = DataType::FLOAT16;

        Kernel kernel = compiler.compile_matmul(256, 256, 256, opts);

        REQUIRE(compiler.last_succeeded());
        REQUIRE(kernel.dtype() == DataType::FLOAT16);
    }
}

TEST_CASE("KernelCompiler compilation statistics", "[kernel_compiler]") {
    KernelCompiler compiler;

    Kernel kernel = compiler.compile_matmul(512, 512, 512);

    const CompilationStats& stats = compiler.last_stats();

    SECTION("Compile time recorded") {
        REQUIRE(stats.compile_time_us > 0);
    }

    SECTION("Tile sizes recorded") {
        REQUIRE(stats.selected_Ti > 0);
        REQUIRE(stats.selected_Tj > 0);
        REQUIRE(stats.selected_Tk > 0);
    }

    SECTION("Instruction counts") {
        REQUIRE(stats.instruction_count > 0);
        // At least some DMA and streamer ops for matmul
        // Note: dma_ops and streamer_ops are size_t (unsigned), so just verify
        // they are reasonable (not checking >= 0 as that's always true)
        REQUIRE(stats.instruction_count >= stats.dma_ops + stats.streamer_ops);
    }

    SECTION("Tile loop counts") {
        REQUIRE(stats.num_m_tiles >= 1);
        REQUIRE(stats.num_n_tiles >= 1);
        REQUIRE(stats.num_k_tiles >= 1);
        REQUIRE(stats.total_tiles == stats.num_m_tiles * stats.num_n_tiles * stats.num_k_tiles);
    }

    SECTION("Memory estimates") {
        REQUIRE(stats.estimated_external_bytes > 0);
        REQUIRE(stats.estimated_arithmetic_intensity > 0);
    }

    SECTION("Summary string") {
        std::string summary = stats.summary();
        REQUIRE(summary.find("Compile Time") != std::string::npos);
        REQUIRE(summary.find("Tile Configuration") != std::string::npos);
    }
}

TEST_CASE("KernelCompiler tile optimization", "[kernel_compiler]") {
    KernelCompiler compiler;

    SECTION("Optimize tiles directly") {
        auto config = compiler.optimize_tiles(1024, 1024, 1024);

        REQUIRE(config.valid);
        REQUIRE(config.Ti > 0);
        REQUIRE(config.Tj > 0);
        REQUIRE(config.Tk > 0);
        // Tiles should be multiples of systolic size (16)
        REQUIRE(config.Ti % 16 == 0);
        REQUIRE(config.Tj % 16 == 0);
    }

    SECTION("Auto-tiling produces valid tiles") {
        Kernel kernel = compiler.compile_matmul(768, 512, 256);

        // Tiles should be set
        REQUIRE(kernel.Ti() > 0);
        REQUIRE(kernel.Tj() > 0);
        REQUIRE(kernel.Tk() > 0);

        // Tiles should not exceed dimensions
        REQUIRE(kernel.Ti() <= 768);
        REQUIRE(kernel.Tj() <= 512);
        REQUIRE(kernel.Tk() <= 256);
    }
}

TEST_CASE("KernelCompiler memory hierarchy configuration", "[kernel_compiler]") {
    SECTION("Custom memory hierarchy") {
        TileOptimizer::MemoryHierarchy mem;
        mem.L2_size = 128 * 1024;  // 128 KB
        mem.L3_size = 256 * 1024;  // 256 KB

        KernelCompiler compiler(mem);

        REQUIRE(compiler.memory_hierarchy().L2_size == 128 * 1024);
        REQUIRE(compiler.memory_hierarchy().L3_size == 256 * 1024);
    }

    SECTION("Update memory hierarchy") {
        KernelCompiler compiler;

        TileOptimizer::MemoryHierarchy mem;
        mem.L2_size = 256 * 1024;
        compiler.set_memory_hierarchy(mem);

        REQUIRE(compiler.memory_hierarchy().L2_size == 256 * 1024);
    }
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_CASE("Kernel edge cases", "[kernel]") {
    SECTION("Minimum valid dimensions") {
        // Smallest valid matmul (single systolic pass)
        Kernel kernel = Kernel::create_matmul(16, 16, 16);
        REQUIRE(kernel.is_valid());
    }

    SECTION("Dimensions not aligned to systolic size") {
        // 100 is not a multiple of 16
        Kernel kernel = Kernel::create_matmul(100, 100, 100);
        REQUIRE(kernel.is_valid());
        // Should still work (with padding or partial tiles)
    }

    SECTION("Very asymmetric dimensions") {
        // Very tall/thin matrix
        Kernel kernel = Kernel::create_matmul(4096, 32, 64);
        REQUIRE(kernel.is_valid());

        // Very wide matrix
        Kernel kernel2 = Kernel::create_matmul(32, 4096, 64);
        REQUIRE(kernel2.is_valid());
    }
}

// ============================================================================
// Conv2D Kernel Tests
// ============================================================================

TEST_CASE("Conv2DConfig structure", "[kernel][conv2d]") {
    SECTION("Default constructor") {
        Conv2DConfig cfg;
        REQUIRE(cfg.batch_size == 1);
        REQUIRE(cfg.in_channels == 1);
        REQUIRE(cfg.out_channels == 1);
        REQUIRE(cfg.stride_h == 1);
        REQUIRE(cfg.stride_w == 1);
        REQUIRE(cfg.padding_h == 0);
        REQUIRE(cfg.padding_w == 0);
        REQUIRE(cfg.dilation_h == 1);
        REQUIRE(cfg.dilation_w == 1);
        REQUIRE(cfg.groups == 1);
    }

    SECTION("Output dimension calculation - no padding") {
        Conv2DConfig cfg;
        cfg.batch_size = 1;
        cfg.in_channels = 3;
        cfg.out_channels = 64;
        cfg.input_height = 32;
        cfg.input_width = 32;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 0;
        cfg.padding_w = 0;

        // Output = (32 - 3 + 2*0) / 1 + 1 = 30
        REQUIRE(cfg.output_height() == 30);
        REQUIRE(cfg.output_width() == 30);
    }

    SECTION("Output dimension calculation - with padding") {
        Conv2DConfig cfg;
        cfg.input_height = 32;
        cfg.input_width = 32;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 1;
        cfg.padding_w = 1;

        // Output = (32 - 3 + 2*1) / 1 + 1 = 32 (same as input)
        REQUIRE(cfg.output_height() == 32);
        REQUIRE(cfg.output_width() == 32);
    }

    SECTION("Output dimension calculation - with stride") {
        Conv2DConfig cfg;
        cfg.input_height = 32;
        cfg.input_width = 32;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 2;
        cfg.stride_w = 2;
        cfg.padding_h = 1;
        cfg.padding_w = 1;

        // Output = (32 - 3 + 2*1) / 2 + 1 = 16
        REQUIRE(cfg.output_height() == 16);
        REQUIRE(cfg.output_width() == 16);
    }

    SECTION("GEMM dimension calculation") {
        Conv2DConfig cfg;
        cfg.batch_size = 2;
        cfg.in_channels = 3;
        cfg.out_channels = 64;
        cfg.input_height = 28;
        cfg.input_width = 28;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 0;
        cfg.padding_w = 0;
        cfg.groups = 1;

        // H_out = W_out = 26
        // M = batch * H_out * W_out = 2 * 26 * 26 = 1352
        // N = out_channels = 64
        // K = in_channels * kernel_h * kernel_w = 3 * 3 * 3 = 27
        REQUIRE(cfg.gemm_M() == 2 * 26 * 26);
        REQUIRE(cfg.gemm_N() == 64);
        REQUIRE(cfg.gemm_K() == 3 * 3 * 3);
    }

    SECTION("Total FLOPs calculation") {
        Conv2DConfig cfg;
        cfg.batch_size = 1;
        cfg.in_channels = 3;
        cfg.out_channels = 64;
        cfg.input_height = 32;
        cfg.input_width = 32;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 1;
        cfg.padding_w = 1;
        cfg.groups = 1;

        // 2 * M * N * K where M = 1*32*32 = 1024, N = 64, K = 3*3*3 = 27
        Size expected_flops = 2 * cfg.gemm_M() * cfg.gemm_N() * cfg.gemm_K();
        REQUIRE(cfg.total_flops() == expected_flops);
    }
}

TEST_CASE("PoolType enumeration", "[kernel][conv2d]") {
    SECTION("Pool type names") {
        REQUIRE(std::string(pool_type_name(PoolType::MAX)) == "max");
        REQUIRE(std::string(pool_type_name(PoolType::AVG)) == "avg");
        REQUIRE(std::string(pool_type_name(PoolType::GLOBAL_AVG)) == "global_avg");
    }
}

TEST_CASE("Kernel::create_conv2d factory method", "[kernel][conv2d]") {
    SECTION("Basic Conv2D creation with Config") {
        Conv2DConfig cfg;
        cfg.batch_size = 1;
        cfg.in_channels = 3;
        cfg.out_channels = 64;
        cfg.input_height = 32;
        cfg.input_width = 32;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 1;
        cfg.padding_w = 1;

        Kernel kernel = Kernel::create_conv2d(cfg);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.op_type() == KernelOpType::CONV2D);
        REQUIRE(kernel.dtype() == DataType::FLOAT32);

        // Verify config was stored
        REQUIRE(kernel.conv2d_config().batch_size == 1);
        REQUIRE(kernel.conv2d_config().in_channels == 3);
        REQUIRE(kernel.conv2d_config().out_channels == 64);
    }

    SECTION("Conv2D with explicit parameters") {
        // batch=2, in_ch=3, out_ch=64, H=28, W=28, kernel=3, stride=1, pad=1
        Kernel kernel = Kernel::create_conv2d(2, 3, 64, 28, 28, 3, 1, 1);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.op_type() == KernelOpType::CONV2D);
        REQUIRE(kernel.conv2d_config().batch_size == 2);
        REQUIRE(kernel.conv2d_config().in_channels == 3);
        REQUIRE(kernel.conv2d_config().out_channels == 64);
        REQUIRE(kernel.conv2d_config().kernel_height == 3);
        REQUIRE(kernel.conv2d_config().kernel_width == 3);
    }

    SECTION("Conv2D with bias and activation") {
        Conv2DConfig cfg;
        cfg.batch_size = 1;
        cfg.in_channels = 64;
        cfg.out_channels = 128;
        cfg.input_height = 16;
        cfg.input_width = 16;
        cfg.kernel_height = 3;
        cfg.kernel_width = 3;
        cfg.stride_h = 1;
        cfg.stride_w = 1;
        cfg.padding_h = 1;
        cfg.padding_w = 1;

        Kernel kernel = Kernel::create_conv2d(cfg, true, ActivationType::RELU);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.has_bias() == true);
        REQUIRE(kernel.activation() == ActivationType::RELU);
    }

    SECTION("Conv2D with stride=2 (downsampling)") {
        Kernel kernel = Kernel::create_conv2d(1, 64, 128, 32, 32, 3, 2, 1);

        REQUIRE(kernel.is_valid());
        REQUIRE(kernel.conv2d_config().stride_h == 2);
        REQUIRE(kernel.conv2d_config().stride_w == 2);
        // Output should be 16x16
        REQUIRE(kernel.conv2d_config().output_height() == 16);
        REQUIRE(kernel.conv2d_config().output_width() == 16);
    }
}

TEST_CASE("Conv2D kernel arguments", "[kernel][conv2d]") {
    Kernel kernel = Kernel::create_conv2d(2, 3, 64, 28, 28, 3, 1, 0, true);

    SECTION("Number of arguments") {
        // input, weight, bias, output = 4
        REQUIRE(kernel.arguments().size() == 4);
    }

    SECTION("Input argument") {
        auto inputs = kernel.input_arguments();
        REQUIRE(inputs.size() == 3);  // input, weight, bias

        const auto& input = inputs[0];
        REQUIRE(input.name == "input");
        REQUIRE(input.shape.size() == 4);
        REQUIRE(input.shape[0] == 2);   // batch
        REQUIRE(input.shape[1] == 3);   // in_channels
        REQUIRE(input.shape[2] == 28);  // height
        REQUIRE(input.shape[3] == 28);  // width
    }

    SECTION("Weight argument") {
        auto inputs = kernel.input_arguments();
        const auto& weight = inputs[1];
        REQUIRE(weight.name == "weight");
        REQUIRE(weight.shape.size() == 4);
        REQUIRE(weight.shape[0] == 64);  // out_channels
        REQUIRE(weight.shape[1] == 3);   // in_channels
        REQUIRE(weight.shape[2] == 3);   // kernel_h
        REQUIRE(weight.shape[3] == 3);   // kernel_w
    }

    SECTION("Bias argument") {
        auto inputs = kernel.input_arguments();
        const auto& bias = inputs[2];
        REQUIRE(bias.name == "bias");
        REQUIRE(bias.shape.size() == 1);
        REQUIRE(bias.shape[0] == 64);  // out_channels
    }

    SECTION("Output argument") {
        auto outputs = kernel.output_arguments();
        REQUIRE(outputs.size() == 1);

        const auto& output = outputs[0];
        REQUIRE(output.name == "output");
        REQUIRE(output.shape.size() == 4);
        REQUIRE(output.shape[0] == 2);   // batch
        REQUIRE(output.shape[1] == 64);  // out_channels
        REQUIRE(output.shape[2] == 26);  // output_height
        REQUIRE(output.shape[3] == 26);  // output_width
    }
}

TEST_CASE("Conv2D kernel validation", "[kernel][conv2d]") {
    SECTION("Valid Conv2D passes validation") {
        Kernel kernel = Kernel::create_conv2d(1, 3, 64, 32, 32, 3, 1, 1);
        std::string error;
        REQUIRE(kernel.validate(error) == true);
        REQUIRE(error.empty());
    }
}

TEST_CASE("Conv2D kernel summary", "[kernel][conv2d]") {
    Kernel kernel = Kernel::create_conv2d(2, 3, 64, 28, 28, 3, 1, 0, true, ActivationType::RELU);

    std::string summary = kernel.summary();

    // Check that summary contains expected Conv2D information
    REQUIRE(summary.find("conv2d") != std::string::npos);
    REQUIRE(summary.find("Input:") != std::string::npos);
    REQUIRE(summary.find("Kernel:") != std::string::npos);
    REQUIRE(summary.find("GEMM Dims:") != std::string::npos);
    REQUIRE(summary.find("relu") != std::string::npos);
    REQUIRE(summary.find("FLOPs") != std::string::npos);
}

TEST_CASE("Conv2D kernel FLOPs calculation", "[kernel][conv2d]") {
    Conv2DConfig cfg;
    cfg.batch_size = 1;
    cfg.in_channels = 3;
    cfg.out_channels = 64;
    cfg.input_height = 32;
    cfg.input_width = 32;
    cfg.kernel_height = 3;
    cfg.kernel_width = 3;
    cfg.stride_h = 1;
    cfg.stride_w = 1;
    cfg.padding_h = 1;
    cfg.padding_w = 1;

    SECTION("Without bias or activation") {
        Kernel kernel = Kernel::create_conv2d(cfg, false, ActivationType::NONE);
        REQUIRE(kernel.total_flops() == cfg.total_flops());
    }

    SECTION("With bias") {
        Kernel kernel = Kernel::create_conv2d(cfg, true, ActivationType::NONE);
        Size output_elements = cfg.batch_size * cfg.out_channels *
                               cfg.output_height() * cfg.output_width();
        REQUIRE(kernel.total_flops() == cfg.total_flops() + output_elements);
    }

    SECTION("With bias and activation") {
        Kernel kernel = Kernel::create_conv2d(cfg, true, ActivationType::RELU);
        Size output_elements = cfg.batch_size * cfg.out_channels *
                               cfg.output_height() * cfg.output_width();
        // bias + activation = 2 * output_elements
        REQUIRE(kernel.total_flops() == cfg.total_flops() + 2 * output_elements);
    }
}
