// Test suite for Vector Engine (VE)
// Tests inline bias+activation processing

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/vector_engine.hpp>
#include <cmath>
#include <vector>
#include <cstring>

using namespace sw::kpu;
using Catch::Approx;

// ============================================================================
// VectorEngine Configuration Tests
// ============================================================================

TEST_CASE("VectorEngine construction", "[ve]") {
    SECTION("Default construction") {
        VectorEngine ve(0);
        REQUIRE(ve.id() == 0);
        REQUIRE(ve.is_enabled() == true);
        REQUIRE(ve.is_busy() == false);
    }

    SECTION("Construction with custom config") {
        VectorEngineConfig config;
        config.vector_width = 16;
        config.bias_buffer_size = 2048;
        config.enabled = true;

        VectorEngine ve(1, config);
        REQUIRE(ve.id() == 1);
        REQUIRE(ve.config().vector_width == 16);
        REQUIRE(ve.config().bias_buffer_size == 2048);
    }
}

TEST_CASE("VectorEngine configuration", "[ve]") {
    VectorEngine ve(0);

    SECTION("Enable/disable") {
        ve.disable();
        REQUIRE(ve.is_enabled() == false);

        ve.enable();
        REQUIRE(ve.is_enabled() == true);
    }

    SECTION("Set activation") {
        ve.set_activation(ActivationType::GELU);
        REQUIRE(ve.sfu().activation() == ActivationType::GELU);
    }

    SECTION("Reset clears state") {
        ve.reset();
        REQUIRE(ve.is_busy() == false);
        REQUIRE(ve.has_pending_operations() == false);
    }
}

// ============================================================================
// Bias Preloading Tests
// ============================================================================

TEST_CASE("VectorEngine bias preloading", "[ve][bias]") {
    VectorEngine ve(0);

    SECTION("Preload bias vector") {
        std::vector<float> bias = {0.1f, 0.2f, 0.3f, 0.4f};
        ve.preload_bias(bias.data(), bias.size());
        // Bias should be loaded without error
    }

    SECTION("Large bias vector") {
        VectorEngineConfig config;
        config.bias_buffer_size = 1024;
        VectorEngine ve2(0, config);

        std::vector<float> bias(1024, 0.5f);
        ve2.preload_bias(bias.data(), bias.size());
    }
}

// ============================================================================
// Immediate Processing Tests (Synchronous)
// ============================================================================

TEST_CASE("VectorEngine immediate processing - RELU only", "[ve][immediate]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::RELU);

    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(input.size());

    ve.process_row_immediate(input.data(), output.data(), input.size());

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 0.0f);
    REQUIRE(output[2] == 0.0f);
    REQUIRE(output[3] == 1.0f);
    REQUIRE(output[4] == 2.0f);
}

TEST_CASE("VectorEngine immediate processing - bias + RELU", "[ve][immediate]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::RELU);

    // Preload bias
    std::vector<float> bias = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};  // Add 1 to all
    ve.preload_bias(bias.data(), bias.size());

    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(input.size());

    ve.process_row_immediate(input.data(), output.data(), input.size());

    // After bias: -1, 0, 1, 2, 3
    // After RELU: 0, 0, 1, 2, 3
    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 0.0f);
    REQUIRE(output[2] == 1.0f);
    REQUIRE(output[3] == 2.0f);
    REQUIRE(output[4] == 3.0f);
}

TEST_CASE("VectorEngine immediate processing - GELU", "[ve][immediate]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::GELU);

    std::vector<float> input = {-2.0f, 0.0f, 2.0f};
    std::vector<float> output(input.size());

    ve.process_row_immediate(input.data(), output.data(), input.size());

    // Compare with SFU reference
    for (size_t i = 0; i < input.size(); ++i) {
        float ref = SFU::reference_gelu(input[i]);
        REQUIRE(output[i] == Approx(ref).margin(0.1f));
    }
}

TEST_CASE("VectorEngine tile processing", "[ve][tile]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::SIGMOID);

    // 3x4 tile
    Size height = 3;
    Size width = 4;
    std::vector<float> input = {
        -2.0f, -1.0f, 0.0f, 1.0f,
        0.0f, 0.5f, 1.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> output(input.size());

    ve.process_tile_immediate(input.data(), output.data(), height, width);

    // Verify all outputs are valid sigmoid values
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output[i] >= 0.0f);
        REQUIRE(output[i] <= 1.0f);
    }

    // Check a few specific values
    REQUIRE(output[2] == Approx(0.5f));  // sigmoid(0) = 0.5
}

// ============================================================================
// Operation Queue Tests
// ============================================================================

TEST_CASE("VectorEngine operation queue", "[ve][queue]") {
    VectorEngine ve(0);

    SECTION("Initially empty") {
        REQUIRE(ve.has_pending_operations() == false);
        REQUIRE(ve.pending_operation_count() == 0);
    }

    SECTION("Enqueue operation") {
        VEOperation op;
        op.height = 16;
        op.width = 32;
        op.activation = ActivationType::RELU;

        ve.enqueue_operation(op);

        // Operation starts immediately if idle
        REQUIRE(ve.is_busy() == true);
    }

    SECTION("Multiple operations queue up") {
        VEOperation op1, op2;
        op1.height = 16;
        op1.width = 32;
        op2.height = 8;
        op2.width = 16;

        ve.enqueue_operation(op1);
        ve.enqueue_operation(op2);

        // One is executing, one is pending
        REQUIRE(ve.pending_operation_count() == 1);
    }
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_CASE("VectorEngine statistics", "[ve][stats]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::RELU);

    // Process some data
    std::vector<float> input = {-1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(input.size());

    ve.process_row_immediate(input.data(), output.data(), input.size());

    const VEStats& stats = ve.stats();
    // Stats track operations (may be 0 for immediate mode depending on impl)

    SECTION("Reset statistics") {
        ve.reset_stats();
        const VEStats& new_stats = ve.stats();
        REQUIRE(new_stats.elements_processed == 0);
        REQUIRE(new_stats.operations_completed == 0);
    }
}

// ============================================================================
// Timing Estimation Tests
// ============================================================================

TEST_CASE("VectorEngine timing", "[ve][timing]") {
    VectorEngine ve(0);

    SECTION("Latency") {
        REQUIRE(ve.get_latency_cycles() == 3);  // Default pipeline depth
    }

    SECTION("Throughput") {
        REQUIRE(ve.get_throughput() == 8);  // Default vector width
    }

    SECTION("Cycle estimate for tile") {
        // 32x64 tile with vector_width=8
        // Rows: 32, chunks per row: ceil(64/8) = 8
        // Total chunks: 32 * 8 = 256
        // Cycles: 256 + pipeline_depth
        Cycle estimated = ve.estimate_cycles(32, 64);
        REQUIRE(estimated == 256 + 3);
    }
}

// ============================================================================
// Cycle-Accurate Update Tests
// ============================================================================

TEST_CASE("VectorEngine cycle-accurate update", "[ve][cycle]") {
    VectorEngine ve(0);
    ve.set_activation(ActivationType::RELU);

    // Mock memory for testing
    std::vector<float> l1_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> l2_data(l1_data.size(), 0.0f);

    // Memory access functions
    auto l1_read = [&](size_t /*id*/, Address addr, void* data, Size size) {
        std::memcpy(data, reinterpret_cast<uint8_t*>(l1_data.data()) + addr, size);
    };

    auto l2_write = [&](size_t /*id*/, Address addr, const void* data, Size size) {
        std::memcpy(reinterpret_cast<uint8_t*>(l2_data.data()) + addr, data, size);
    };

    SECTION("Operation completes after cycles") {
        VEOperation op;
        op.l1_scratchpad_id = 0;
        op.l1_base_addr = 0;
        op.l2_bank_id = 0;
        op.l2_base_addr = 0;
        op.height = 1;
        op.width = 8;
        op.element_size = sizeof(float);
        op.activation = ActivationType::RELU;

        ve.enqueue_operation(op);

        // Run until completion
        Cycle cycle = 0;
        bool completed = false;
        while (cycle < 100 && !completed) {
            completed = ve.update(cycle, l1_read, l2_write);
            cycle++;
        }

        REQUIRE(completed == true);
        REQUIRE(cycle < 50);  // Should complete reasonably quickly
    }
}

// ============================================================================
// SFU Access Tests
// ============================================================================

TEST_CASE("VectorEngine SFU access", "[ve][sfu]") {
    VectorEngine ve(0);

    SECTION("SFU is accessible") {
        SFU& sfu = ve.sfu();
        sfu.configure(ActivationType::TANH);
        REQUIRE(sfu.activation() == ActivationType::TANH);
    }

    SECTION("Const SFU access") {
        const VectorEngine& const_ve = ve;
        const SFU& sfu = const_ve.sfu();
        REQUIRE(sfu.activation() == ActivationType::NONE);  // Default
    }
}
