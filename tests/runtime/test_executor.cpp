// Test suite for GraphExecutor
// Tests high-level execution API for kernels

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/runtime/executor.hpp>
#include <sw/kpu/kernel.hpp>

#include <vector>
#include <cmath>

using namespace sw::runtime;
using namespace sw::kpu;

// ============================================================================
// Test Fixture
// ============================================================================

class ExecutorTestFixture {
protected:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> simulator;
    std::unique_ptr<KPURuntime> runtime;
    std::unique_ptr<GraphExecutor> executor;

    ExecutorTestFixture() {
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.page_buffer_count = 2;
        config.page_buffer_capacity_kb = 64;
        config.l1_buffer_count = 4;
        config.l1_buffer_capacity_kb = 64;
        config.dma_engine_count = 2;
        config.block_mover_count = 4;
        config.streamer_count = 8;
        config.processor_array_rows = 16;
        config.processor_array_cols = 16;
        config.use_systolic_array_mode = true;

        simulator = std::make_unique<KPUSimulator>(config);
        runtime = std::make_unique<KPURuntime>(simulator.get());
        executor = std::make_unique<GraphExecutor>(runtime.get());
    }
};

// ============================================================================
// GraphExecutor Construction Tests
// ============================================================================

TEST_CASE("GraphExecutor construction", "[executor]") {
    SECTION("Basic construction") {
        KPUSimulator::Config config;
        KPUSimulator sim(config);
        KPURuntime runtime(&sim);
        GraphExecutor executor(&runtime);

        REQUIRE(executor.runtime() == &runtime);
        REQUIRE_FALSE(executor.has_kernel());
    }

    SECTION("Null runtime throws") {
        REQUIRE_THROWS_AS(GraphExecutor(nullptr), std::invalid_argument);
    }
}

// ============================================================================
// Kernel Setup Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor kernel setup", "[executor][kernel]") {
    SECTION("Create matmul kernel") {
        executor->create_matmul(64, 64, 64);

        REQUIRE(executor->has_kernel());
        REQUIRE(executor->kernel() != nullptr);
        REQUIRE(executor->kernel()->op_type() == KernelOpType::MATMUL);
    }

    SECTION("Create MLP kernel") {
        executor->create_mlp(64, 128, 64, ActivationType::GELU, true);

        REQUIRE(executor->has_kernel());
        REQUIRE(executor->kernel()->op_type() == KernelOpType::MLP);
        REQUIRE(executor->kernel()->activation() == ActivationType::GELU);
        REQUIRE(executor->kernel()->has_bias());
    }

    SECTION("Set external kernel") {
        Kernel kernel = Kernel::create_matmul(128, 128, 128);
        executor->set_kernel(kernel);

        REQUIRE(executor->has_kernel());
        REQUIRE(executor->kernel()->M() == 128);
    }

    SECTION("Tensor bindings created") {
        executor->create_matmul(64, 64, 64);

        const TensorBinding* A = executor->get_binding("A");
        const TensorBinding* B = executor->get_binding("B");
        const TensorBinding* C = executor->get_binding("C");

        REQUIRE(A != nullptr);
        REQUIRE(B != nullptr);
        REQUIRE(C != nullptr);

        REQUIRE(A->shape == std::vector<Size>{64, 64});
        REQUIRE(B->shape == std::vector<Size>{64, 64});
        REQUIRE(C->shape == std::vector<Size>{64, 64});

        REQUIRE(A->device_address != 0);
        REQUIRE(B->device_address != 0);
        REQUIRE(C->device_address != 0);
    }

    SECTION("MLP with bias has correct bindings") {
        executor->create_mlp(64, 128, 32, ActivationType::RELU, true);

        REQUIRE(executor->get_binding("A") != nullptr);
        REQUIRE(executor->get_binding("B") != nullptr);
        REQUIRE(executor->get_binding("bias") != nullptr);
        REQUIRE(executor->get_binding("C") != nullptr);

        auto bias = executor->get_binding("bias");
        REQUIRE(bias->shape == std::vector<Size>{128});
    }
}

// ============================================================================
// Input/Output Binding Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor input/output", "[executor][io]") {
    SECTION("Set and get input") {
        const Size M = 32, N = 32, K = 32;
        executor->create_matmul(M, N, K);

        // Create input data
        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 2.0f);

        // Set inputs
        REQUIRE_NOTHROW(executor->set_input("A", A.data(), {M, K}));
        REQUIRE_NOTHROW(executor->set_input("B", B.data(), {K, N}));
    }

    SECTION("Get output after execution") {
        const Size M = 32, N = 32, K = 32;
        executor->create_matmul(M, N, K);

        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);

        executor->set_input("A", A.data(), {M, K});
        executor->set_input("B", B.data(), {K, N});

        executor->execute();

        REQUIRE_NOTHROW(executor->get_output("C", C.data()));
    }

    SECTION("Invalid tensor name throws") {
        executor->create_matmul(32, 32, 32);

        std::vector<float> data(32 * 32);
        REQUIRE_THROWS_AS(executor->set_input("X", data.data(), {32, 32}), std::invalid_argument);
        REQUIRE_THROWS_AS(executor->get_output("Y", data.data()), std::invalid_argument);
    }

    SECTION("Shape mismatch throws") {
        executor->create_matmul(32, 32, 32);

        std::vector<float> data(64 * 64);
        // Wrong shape
        REQUIRE_THROWS_AS(executor->set_input("A", data.data(), {64, 64}), std::invalid_argument);
    }
}

// ============================================================================
// Execution Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor execution", "[executor][exec]") {
    SECTION("Execute matmul") {
        const Size M = 64, N = 64, K = 64;
        executor->create_matmul(M, N, K);

        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);

        executor->set_input("A", A.data(), {M, K});
        executor->set_input("B", B.data(), {K, N});

        auto result = executor->execute();

        REQUIRE(result.success);
        REQUIRE(result.cycles > 0);
        REQUIRE(result.time_ms > 0.0);
    }

    SECTION("Execute MLP") {
        const Size M = 32, N = 64, K = 32;
        executor->create_mlp(M, N, K, ActivationType::RELU, true);

        std::vector<float> A(M * K, 0.5f);
        std::vector<float> B(K * N, 0.5f);
        std::vector<float> bias(N, 0.1f);

        executor->set_input("A", A.data(), {M, K});
        executor->set_input("B", B.data(), {K, N});
        executor->set_input("bias", bias.data(), {N});

        auto result = executor->execute();

        REQUIRE(result.success);
    }

    SECTION("Last result tracking") {
        executor->create_matmul(32, 32, 32);

        std::vector<float> A(32 * 32, 1.0f);
        std::vector<float> B(32 * 32, 1.0f);

        executor->set_input("A", A.data(), {32, 32});
        executor->set_input("B", B.data(), {32, 32});

        executor->execute();

        REQUIRE(executor->last_result().success);
        REQUIRE(executor->get_last_execution_cycles() > 0);
        REQUIRE(executor->get_last_execution_time_ms() > 0.0);
    }

    SECTION("Execute without kernel fails") {
        auto result = executor->execute();
        REQUIRE_FALSE(result.success);
        REQUIRE(result.error.find("No kernel") != std::string::npos);
    }
}

// ============================================================================
// Release Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor release", "[executor][release]") {
    SECTION("Release frees memory") {
        executor->create_matmul(256, 256, 256);

        Size free_before = runtime->get_free_memory();

        executor->release();

        Size free_after = runtime->get_free_memory();

        REQUIRE_FALSE(executor->has_kernel());
        REQUIRE(free_after >= free_before);
    }

    SECTION("Setting new kernel releases old") {
        executor->create_matmul(128, 128, 128);
        auto* old_kernel = executor->kernel();

        executor->create_matmul(64, 64, 64);
        auto* new_kernel = executor->kernel();

        REQUIRE(new_kernel != old_kernel);
        REQUIRE(executor->kernel()->M() == 64);
    }
}

// ============================================================================
// Multiple Execution Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor multiple executions", "[executor][multi]") {
    SECTION("Execute same kernel multiple times") {
        const Size M = 32, N = 32, K = 32;
        executor->create_matmul(M, N, K);

        std::vector<float> A(M * K, 1.0f);
        std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N);

        executor->set_input("A", A.data(), {M, K});
        executor->set_input("B", B.data(), {K, N});

        // Execute multiple times
        for (int i = 0; i < 3; ++i) {
            auto result = executor->execute();
            REQUIRE(result.success);
        }

        executor->get_output("C", C.data());
    }

    SECTION("Update inputs between executions") {
        const Size M = 32, N = 32, K = 32;
        executor->create_matmul(M, N, K);

        std::vector<float> A1(M * K, 1.0f);
        std::vector<float> A2(M * K, 2.0f);
        std::vector<float> B(K * N, 1.0f);

        // First execution
        executor->set_input("A", A1.data(), {M, K});
        executor->set_input("B", B.data(), {K, N});
        executor->execute();

        // Update input and execute again
        executor->set_input("A", A2.data(), {M, K});
        auto result = executor->execute();

        REQUIRE(result.success);
    }
}

// ============================================================================
// Different Kernel Types Tests
// ============================================================================

TEST_CASE_METHOD(ExecutorTestFixture, "GraphExecutor different kernels", "[executor][kernels]") {
    SECTION("Different activation functions") {
        std::vector<ActivationType> activations = {
            ActivationType::RELU,
            ActivationType::GELU,
            ActivationType::SIGMOID,
            ActivationType::TANH
        };

        for (auto act : activations) {
            executor->create_mlp(32, 32, 32, act, true);

            std::vector<float> A(32 * 32, 0.5f);
            std::vector<float> B(32 * 32, 0.5f);
            std::vector<float> bias(32, 0.1f);

            executor->set_input("A", A.data(), {32, 32});
            executor->set_input("B", B.data(), {32, 32});
            executor->set_input("bias", bias.data(), {32});

            auto result = executor->execute();
            REQUIRE(result.success);
        }
    }

    SECTION("MLP without bias") {
        executor->create_mlp(32, 32, 32, ActivationType::RELU, false);

        // Should only have A, B, C (no bias)
        REQUIRE(executor->get_binding("A") != nullptr);
        REQUIRE(executor->get_binding("B") != nullptr);
        REQUIRE(executor->get_binding("C") != nullptr);
        REQUIRE(executor->get_binding("bias") == nullptr);

        std::vector<float> A(32 * 32, 0.5f);
        std::vector<float> B(32 * 32, 0.5f);

        executor->set_input("A", A.data(), {32, 32});
        executor->set_input("B", B.data(), {32, 32});

        auto result = executor->execute();
        REQUIRE(result.success);
    }
}
