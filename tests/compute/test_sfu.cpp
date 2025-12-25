// Test suite for Special Function Unit (SFU)
// Tests LUT-based activation functions and accuracy

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/sfu.hpp>
#include <cmath>
#include <vector>

using namespace sw::kpu;
using Catch::Approx;

// ============================================================================
// ActivationType Tests
// ============================================================================

TEST_CASE("ActivationType enumeration", "[sfu]") {
    SECTION("Activation type names") {
        REQUIRE(std::string(activation_type_name(ActivationType::NONE)) == "none");
        REQUIRE(std::string(activation_type_name(ActivationType::RELU)) == "relu");
        REQUIRE(std::string(activation_type_name(ActivationType::GELU)) == "gelu");
        REQUIRE(std::string(activation_type_name(ActivationType::SIGMOID)) == "sigmoid");
        REQUIRE(std::string(activation_type_name(ActivationType::TANH)) == "tanh");
        REQUIRE(std::string(activation_type_name(ActivationType::SILU)) == "silu");
        REQUIRE(std::string(activation_type_name(ActivationType::SOFTPLUS)) == "softplus");
        REQUIRE(std::string(activation_type_name(ActivationType::LEAKY_RELU)) == "leaky_relu");
    }
}

// ============================================================================
// SFU Reference Implementation Tests
// ============================================================================

TEST_CASE("SFU reference implementations", "[sfu]") {
    SECTION("Reference RELU") {
        REQUIRE(SFU::reference_relu(2.0f) == 2.0f);
        REQUIRE(SFU::reference_relu(-2.0f) == 0.0f);
        REQUIRE(SFU::reference_relu(0.0f) == 0.0f);
    }

    SECTION("Reference Leaky RELU") {
        REQUIRE(SFU::reference_leaky_relu(2.0f, 0.01f) == 2.0f);
        REQUIRE(SFU::reference_leaky_relu(-2.0f, 0.01f) == Approx(-0.02f));
        REQUIRE(SFU::reference_leaky_relu(0.0f, 0.01f) == 0.0f);
    }

    SECTION("Reference Sigmoid") {
        REQUIRE(SFU::reference_sigmoid(0.0f) == Approx(0.5f));
        REQUIRE(SFU::reference_sigmoid(-10.0f) == Approx(0.0f).margin(0.001));
        REQUIRE(SFU::reference_sigmoid(10.0f) == Approx(1.0f).margin(0.001));
    }

    SECTION("Reference Tanh") {
        REQUIRE(SFU::reference_tanh(0.0f) == Approx(0.0f));
        REQUIRE(SFU::reference_tanh(-5.0f) == Approx(-1.0f).margin(0.01));
        REQUIRE(SFU::reference_tanh(5.0f) == Approx(1.0f).margin(0.01));
    }

    SECTION("Reference GELU") {
        REQUIRE(SFU::reference_gelu(0.0f) == Approx(0.0f).margin(0.01));
        // GELU(-x) = -GELU(x) approximately for large |x|
        REQUIRE(SFU::reference_gelu(2.0f) == Approx(1.954f).margin(0.1));
        REQUIRE(SFU::reference_gelu(-2.0f) == Approx(-0.046f).margin(0.1));
    }

    SECTION("Reference SILU") {
        REQUIRE(SFU::reference_silu(0.0f) == Approx(0.0f));
        // SILU(x) = x * sigmoid(x)
        float x = 1.0f;
        float expected = x * SFU::reference_sigmoid(x);
        REQUIRE(SFU::reference_silu(x) == Approx(expected));
    }

    SECTION("Reference Softplus") {
        REQUIRE(SFU::reference_softplus(0.0f) == Approx(std::log(2.0f)));
        // Softplus(x) ≈ x for large x
        REQUIRE(SFU::reference_softplus(25.0f) == Approx(25.0f).margin(0.1));
    }
}

// ============================================================================
// SFU Configuration Tests
// ============================================================================

TEST_CASE("SFU construction and configuration", "[sfu]") {
    SECTION("Default construction") {
        SFU sfu;
        REQUIRE(sfu.activation() == ActivationType::NONE);
        REQUIRE(sfu.get_table_size() == 256);
    }

    SECTION("Configuration with explicit values") {
        SFUConfig config;
        config.activation = ActivationType::SIGMOID;
        config.table_size = 512;
        config.input_range_min = -10.0f;
        config.input_range_max = 10.0f;

        SFU sfu(config);
        REQUIRE(sfu.activation() == ActivationType::SIGMOID);
        REQUIRE(sfu.get_table_size() == 512);

        auto [min_val, max_val] = sfu.get_input_range();
        REQUIRE(min_val == -10.0f);
        REQUIRE(max_val == 10.0f);
    }

    SECTION("Reconfiguration") {
        SFU sfu;
        sfu.configure(ActivationType::GELU, 256);
        REQUIRE(sfu.activation() == ActivationType::GELU);

        sfu.configure(ActivationType::TANH, 128);
        REQUIRE(sfu.activation() == ActivationType::TANH);
        REQUIRE(sfu.get_table_size() == 128);
    }

    SECTION("Set input range") {
        SFU sfu;
        sfu.set_input_range(-4.0f, 4.0f);

        auto [min_val, max_val] = sfu.get_input_range();
        REQUIRE(min_val == -4.0f);
        REQUIRE(max_val == 4.0f);
    }
}

// ============================================================================
// SFU Evaluation Tests (RELU - no LUT)
// ============================================================================

TEST_CASE("SFU RELU evaluation (direct, no LUT)", "[sfu][relu]") {
    SFU sfu;
    sfu.configure(ActivationType::RELU);

    SECTION("Positive values pass through") {
        REQUIRE(sfu.evaluate(1.0f) == 1.0f);
        REQUIRE(sfu.evaluate(100.0f) == 100.0f);
        REQUIRE(sfu.evaluate(0.001f) == 0.001f);
    }

    SECTION("Negative values clipped to zero") {
        REQUIRE(sfu.evaluate(-1.0f) == 0.0f);
        REQUIRE(sfu.evaluate(-100.0f) == 0.0f);
        REQUIRE(sfu.evaluate(-0.001f) == 0.0f);
    }

    SECTION("Zero returns zero") {
        REQUIRE(sfu.evaluate(0.0f) == 0.0f);
    }
}

// ============================================================================
// SFU Evaluation Tests (LUT-based functions)
// ============================================================================

TEST_CASE("SFU Sigmoid LUT evaluation", "[sfu][sigmoid]") {
    SFU sfu;
    sfu.configure(ActivationType::SIGMOID, 256);

    SECTION("Accuracy within tolerance") {
        std::vector<float> test_values = {-6.0f, -4.0f, -2.0f, -1.0f, 0.0f,
                                           1.0f, 2.0f, 4.0f, 6.0f};
        for (float x : test_values) {
            float lut_result = sfu.evaluate(x);
            float ref_result = SFU::reference_sigmoid(x);
            // Allow 1% relative error or 0.01 absolute for values near 0 or 1
            float tolerance = std::max(0.01f, std::abs(ref_result) * 0.01f);
            REQUIRE(lut_result == Approx(ref_result).margin(tolerance));
        }
    }
}

TEST_CASE("SFU Tanh LUT evaluation", "[sfu][tanh]") {
    SFU sfu;
    sfu.configure(ActivationType::TANH, 256);

    SECTION("Accuracy within tolerance") {
        std::vector<float> test_values = {-4.0f, -2.0f, -1.0f, 0.0f,
                                           1.0f, 2.0f, 4.0f};
        for (float x : test_values) {
            float lut_result = sfu.evaluate(x);
            float ref_result = SFU::reference_tanh(x);
            float tolerance = std::max(0.01f, std::abs(ref_result) * 0.01f);
            REQUIRE(lut_result == Approx(ref_result).margin(tolerance));
        }
    }
}

TEST_CASE("SFU GELU LUT evaluation", "[sfu][gelu]") {
    SFU sfu;
    sfu.configure(ActivationType::GELU, 256);

    SECTION("Accuracy within tolerance") {
        std::vector<float> test_values = {-3.0f, -2.0f, -1.0f, 0.0f,
                                           1.0f, 2.0f, 3.0f};
        for (float x : test_values) {
            float lut_result = sfu.evaluate(x);
            float ref_result = SFU::reference_gelu(x);
            // GELU has more complex shape, allow 2% error
            float tolerance = std::max(0.02f, std::abs(ref_result) * 0.02f);
            REQUIRE(lut_result == Approx(ref_result).margin(tolerance));
        }
    }
}

TEST_CASE("SFU SILU LUT evaluation", "[sfu][silu]") {
    SFU sfu;
    sfu.configure(ActivationType::SILU, 256);

    SECTION("Accuracy within tolerance") {
        std::vector<float> test_values = {-4.0f, -2.0f, -1.0f, 0.0f,
                                           1.0f, 2.0f, 4.0f};
        for (float x : test_values) {
            float lut_result = sfu.evaluate(x);
            float ref_result = SFU::reference_silu(x);
            float tolerance = std::max(0.02f, std::abs(ref_result) * 0.02f);
            REQUIRE(lut_result == Approx(ref_result).margin(tolerance));
        }
    }
}

// ============================================================================
// Vector Evaluation Tests
// ============================================================================

TEST_CASE("SFU vector evaluation", "[sfu][vector]") {
    SFU sfu;
    sfu.configure(ActivationType::RELU);

    SECTION("Vector RELU") {
        std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        std::vector<float> output(input.size());

        sfu.evaluate_vector(input.data(), output.data(), input.size());

        REQUIRE(output[0] == 0.0f);
        REQUIRE(output[1] == 0.0f);
        REQUIRE(output[2] == 0.0f);
        REQUIRE(output[3] == 1.0f);
        REQUIRE(output[4] == 2.0f);
    }

    SECTION("In-place evaluation") {
        std::vector<float> data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

        sfu.evaluate_inplace(data.data(), data.size());

        REQUIRE(data[0] == 0.0f);
        REQUIRE(data[1] == 0.0f);
        REQUIRE(data[2] == 0.0f);
        REQUIRE(data[3] == 1.0f);
        REQUIRE(data[4] == 2.0f);
    }
}

TEST_CASE("SFU vector evaluation with LUT", "[sfu][vector]") {
    SFU sfu;
    sfu.configure(ActivationType::SIGMOID);

    std::vector<float> input = {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f};
    std::vector<float> output(input.size());

    sfu.evaluate_vector(input.data(), output.data(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        float ref = SFU::reference_sigmoid(input[i]);
        REQUIRE(output[i] == Approx(ref).margin(0.01f));
    }
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

TEST_CASE("SFU edge cases", "[sfu][edge]") {
    SFU sfu;
    sfu.configure(ActivationType::SIGMOID, 256);

    SECTION("Values outside LUT range are clamped") {
        // Values beyond the LUT range should return edge values
        float far_negative = sfu.evaluate(-100.0f);
        float far_positive = sfu.evaluate(100.0f);

        REQUIRE(far_negative == Approx(0.0f).margin(0.01f));
        REQUIRE(far_positive == Approx(1.0f).margin(0.01f));
    }

    SECTION("Pass-through with NONE activation") {
        sfu.configure(ActivationType::NONE);
        REQUIRE(sfu.evaluate(5.0f) == 5.0f);
        REQUIRE(sfu.evaluate(-5.0f) == -5.0f);
        REQUIRE(sfu.evaluate(0.0f) == 0.0f);
    }
}

// ============================================================================
// Timing Tests
// ============================================================================

TEST_CASE("SFU timing characteristics", "[sfu][timing]") {
    SFU sfu;
    sfu.configure(ActivationType::GELU);

    SECTION("Latency is 2 cycles by default") {
        REQUIRE(sfu.get_latency_cycles() == 2);
    }

    SECTION("Throughput is 1 element per cycle") {
        REQUIRE(sfu.get_throughput() == 1);
    }
}

// ============================================================================
// LUT Access Tests
// ============================================================================

TEST_CASE("SFU LUT access", "[sfu][lut]") {
    SFU sfu;
    sfu.configure(ActivationType::SIGMOID, 256);

    SECTION("LUT has correct size") {
        const auto& lut = sfu.get_lut();
        REQUIRE(lut.size() == 256);
    }

    SECTION("LUT values are valid sigmoid outputs") {
        const auto& lut = sfu.get_lut();
        for (float val : lut) {
            REQUIRE(val >= 0.0f);
            REQUIRE(val <= 1.0f);
        }
    }

    SECTION("LUT is monotonically increasing for sigmoid") {
        const auto& lut = sfu.get_lut();
        for (size_t i = 1; i < lut.size(); ++i) {
            REQUIRE(lut[i] >= lut[i-1]);
        }
    }
}
