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
