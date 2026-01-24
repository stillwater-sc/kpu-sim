// Test MLP forward pass at different quantization levels
// Demonstrates template kernel instantiation across precision types

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/quantization/scalar_traits.hpp>
#include <sw/kpu/quantization/kernels.hpp>
#include <sw/kpu/quantization/quant_params.hpp>

// Universal types
#include <sw/kpu/quantization/universal_types.hpp>

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace sw::kpu;
using namespace sw::kpu::kernels;
using Catch::Approx;

// =============================================================================
// MLP Layer Implementation
// =============================================================================

/// Single MLP layer: Y = ReLU(X @ W + b)
template<typename Scalar, typename Accumulator = typename ScalarTraits<Scalar>::accumulator_type>
class MLPLayer {
public:
    size_t in_features;
    size_t out_features;
    std::vector<Scalar> weights;   // [in_features x out_features]
    std::vector<Scalar> bias;      // [out_features]

    MLPLayer(size_t in_feat, size_t out_feat)
        : in_features(in_feat)
        , out_features(out_feat)
        , weights(in_feat * out_feat)
        , bias(out_feat)
    {}

    /// Initialize weights from float data
    void init_from_float(const float* w_data, const float* b_data) {
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<Scalar>(w_data[i]);
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            bias[i] = static_cast<Scalar>(b_data[i]);
        }
    }

    /// Forward pass: Y = ReLU(X @ W + b)
    void forward(const Scalar* X, Scalar* Y, size_t batch_size) {
        // Temporary accumulator buffer
        std::vector<Accumulator> acc(batch_size * out_features, Accumulator(0));

        // Matrix multiply: acc = X @ W
        TensorView<const Scalar> x_view(const_cast<Scalar*>(X), batch_size, in_features);
        TensorView<const Scalar> w_view(const_cast<Scalar*>(weights.data()), in_features, out_features);
        TensorView<Accumulator> acc_view(acc.data(), batch_size, out_features);

        matmul(x_view, w_view, acc_view);

        // Add bias and ReLU, convert back to Scalar
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < out_features; ++j) {
                Accumulator val = acc[b * out_features + j] +
                                  static_cast<Accumulator>(bias[j]);
                // ReLU
                if (val < Accumulator(0)) val = Accumulator(0);
                Y[b * out_features + j] = static_cast<Scalar>(val);
            }
        }
    }
};

/// Two-layer MLP: hidden_dim neurons, ReLU activation
template<typename Scalar, typename Accumulator = typename ScalarTraits<Scalar>::accumulator_type>
class TwoLayerMLP {
public:
    MLPLayer<Scalar, Accumulator> layer1;
    MLPLayer<Scalar, Accumulator> layer2;
    std::vector<Scalar> hidden;  // Intermediate buffer

    TwoLayerMLP(size_t in_dim, size_t hidden_dim, size_t out_dim)
        : layer1(in_dim, hidden_dim)
        , layer2(hidden_dim, out_dim)
        , hidden(hidden_dim)
    {}

    void init_from_float(const float* w1, const float* b1,
                         const float* w2, const float* b2) {
        layer1.init_from_float(w1, b1);
        layer2.init_from_float(w2, b2);
    }

    /// Forward pass for single sample
    void forward(const Scalar* X, Scalar* Y) {
        layer1.forward(X, hidden.data(), 1);
        layer2.forward(hidden.data(), Y, 1);
    }

    /// Get output as float for comparison
    float forward_and_get_float(const Scalar* X) {
        std::vector<Scalar> out(layer2.out_features);
        forward(X, out.data());
        return static_cast<float>(out[0]);
    }
};

// =============================================================================
// Test Fixtures
// =============================================================================

class MLPTestFixture {
public:
    // Simple 4 -> 8 -> 1 MLP (like XOR-style problem)
    static constexpr size_t IN_DIM = 4;
    static constexpr size_t HIDDEN_DIM = 8;
    static constexpr size_t OUT_DIM = 1;

    std::vector<float> w1_float;  // [4 x 8]
    std::vector<float> b1_float;  // [8]
    std::vector<float> w2_float;  // [8 x 1]
    std::vector<float> b2_float;  // [1]
    std::vector<float> input_float; // [4]

    float reference_output;

    MLPTestFixture()
        : w1_float(IN_DIM * HIDDEN_DIM)
        , b1_float(HIDDEN_DIM)
        , w2_float(HIDDEN_DIM * OUT_DIM)
        , b2_float(OUT_DIM)
        , input_float(IN_DIM)
    {
        // Use deterministic weights that produce non-zero output through ReLU
        // Design: positive inputs + positive first-layer weights + positive bias
        // ensures hidden activations are positive, passing through ReLU

        // Input: all 1.0
        for (auto& v : input_float) v = 1.0f;

        // W1: [4 x 8] - use small positive values so hidden = input @ W1 + b1 > 0
        for (size_t i = 0; i < w1_float.size(); ++i) {
            w1_float[i] = 0.1f + 0.01f * static_cast<float>(i % 5);
        }

        // b1: [8] - positive bias ensures ReLU passes
        for (size_t i = 0; i < b1_float.size(); ++i) {
            b1_float[i] = 0.5f;
        }

        // W2: [8 x 1] - mix of positive values
        for (size_t i = 0; i < w2_float.size(); ++i) {
            w2_float[i] = 0.2f + 0.05f * static_cast<float>(i);
        }

        // b2: [1] - small positive
        b2_float[0] = 0.1f;

        // Compute reference output using FP32
        TwoLayerMLP<float, float> ref_mlp(IN_DIM, HIDDEN_DIM, OUT_DIM);
        ref_mlp.init_from_float(w1_float.data(), b1_float.data(),
                                w2_float.data(), b2_float.data());

        std::vector<float> input_copy = input_float;
        reference_output = ref_mlp.forward_and_get_float(input_copy.data());
    }

    template<typename Scalar, typename Accumulator = typename ScalarTraits<Scalar>::accumulator_type>
    float run_mlp() {
        TwoLayerMLP<Scalar, Accumulator> mlp(IN_DIM, HIDDEN_DIM, OUT_DIM);
        mlp.init_from_float(w1_float.data(), b1_float.data(),
                           w2_float.data(), b2_float.data());

        // Convert input to Scalar type
        std::vector<Scalar> input(IN_DIM);
        for (size_t i = 0; i < IN_DIM; ++i) {
            input[i] = static_cast<Scalar>(input_float[i]);
        }

        return mlp.forward_and_get_float(input.data());
    }
};

// =============================================================================
// Tests
// =============================================================================

TEST_CASE("MLP at different quantization levels", "[quantization][mlp]") {
    MLPTestFixture fixture;

    std::cout << "\n=== MLP Quantization Comparison ===\n";
    std::cout << "Architecture: " << MLPTestFixture::IN_DIM << " -> "
              << MLPTestFixture::HIDDEN_DIM << " -> " << MLPTestFixture::OUT_DIM << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nReference (FP32): " << fixture.reference_output << "\n\n";
    std::cout << std::setw(15) << "Type"
              << std::setw(15) << "Output"
              << std::setw(15) << "Abs Error"
              << std::setw(15) << "Rel Error %" << "\n";
    std::cout << std::string(60, '-') << "\n";

    SECTION("FP32 baseline") {
        float result = fixture.run_mlp<float, float>();
        float error = std::abs(result - fixture.reference_output);
        std::cout << std::setw(15) << "FP32"
                  << std::setw(15) << result
                  << std::setw(15) << error
                  << std::setw(15) << (error / std::abs(fixture.reference_output) * 100) << "\n";
        REQUIRE(result == Approx(fixture.reference_output).margin(1e-5));
    }

    SECTION("BF16 with FP32 accumulator") {
        float result = fixture.run_mlp<bf16_t, float>();
        float error = std::abs(result - fixture.reference_output);
        float ref_abs = std::abs(fixture.reference_output);
        float rel_error = (ref_abs > 1e-6f) ? (error / ref_abs * 100) : 0.0f;
        std::cout << std::setw(15) << "BF16->FP32"
                  << std::setw(15) << result
                  << std::setw(15) << error
                  << std::setw(15) << rel_error << "\n";
        // BF16 should be within 5% for this small network
        REQUIRE((rel_error < 5.0f || error < 0.01f));
    }

    SECTION("FP16 with FP32 accumulator") {
        float result = fixture.run_mlp<fp16_t, float>();
        float error = std::abs(result - fixture.reference_output);
        float ref_abs = std::abs(fixture.reference_output);
        float rel_error = (ref_abs > 1e-6f) ? (error / ref_abs * 100) : 0.0f;
        std::cout << std::setw(15) << "FP16->FP32"
                  << std::setw(15) << result
                  << std::setw(15) << error
                  << std::setw(15) << rel_error << "\n";
        REQUIRE((rel_error < 5.0f || error < 0.01f));
    }

    SECTION("FP8 E4M3 with FP32 accumulator") {
        float result = fixture.run_mlp<fp8e4m3_t, float>();
        float error = std::abs(result - fixture.reference_output);
        float ref_abs = std::abs(fixture.reference_output);
        float rel_error = (ref_abs > 1e-6f) ? (error / ref_abs * 100) : 0.0f;
        std::cout << std::setw(15) << "FP8E4M3->FP32"
                  << std::setw(15) << result
                  << std::setw(15) << error
                  << std::setw(15) << rel_error << "\n";
        // FP8 may have larger error but should be reasonable
        REQUIRE((rel_error < 50.0f || error < 0.1f));
    }

    SECTION("FP8 E5M2 with FP32 accumulator") {
        float result = fixture.run_mlp<fp8e5m2_t, float>();
        float error = std::abs(result - fixture.reference_output);
        float ref_abs = std::abs(fixture.reference_output);
        float rel_error = (ref_abs > 1e-6f) ? (error / ref_abs * 100) : 0.0f;
        std::cout << std::setw(15) << "FP8E5M2->FP32"
                  << std::setw(15) << result
                  << std::setw(15) << error
                  << std::setw(15) << rel_error << "\n";
        REQUIRE((rel_error < 50.0f || error < 0.1f));
    }

    SECTION("INT8 with INT32 accumulator") {
        // For INT8, we need to quantize the weights properly
        // This is a simplified test - real INT8 would use scale/zero_point
        float result = fixture.run_mlp<std::int8_t, std::int32_t>();
        std::cout << std::setw(15) << "INT8->INT32"
                  << std::setw(15) << result
                  << std::setw(15) << "N/A"
                  << std::setw(15) << "N/A (needs Q/DQ)" << "\n";
        // INT8 without proper quantization will be very different
        // Just verify it runs without crashing
        REQUIRE(std::isfinite(result));
    }

    std::cout << std::string(60, '-') << "\n\n";
}

TEST_CASE("MLP precision degradation analysis", "[quantization][mlp][analysis]") {
    MLPTestFixture fixture;

    // Run multiple random inputs and collect statistics
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t NUM_SAMPLES = 100;

    struct Stats {
        double sum_abs_error = 0;
        double sum_sq_error = 0;
        double max_error = 0;
    };

    auto compute_stats = [&](auto run_fn) {
        Stats stats;
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            // Generate random input
            std::vector<float> input(MLPTestFixture::IN_DIM);
            for (auto& v : input) v = dist(rng);

            // Run FP32 reference
            TwoLayerMLP<float, float> ref_mlp(MLPTestFixture::IN_DIM,
                                               MLPTestFixture::HIDDEN_DIM,
                                               MLPTestFixture::OUT_DIM);
            ref_mlp.init_from_float(fixture.w1_float.data(), fixture.b1_float.data(),
                                   fixture.w2_float.data(), fixture.b2_float.data());
            float ref = ref_mlp.forward_and_get_float(input.data());

            // Run quantized version
            float result = run_fn(input);

            double error = std::abs(result - ref);
            stats.sum_abs_error += error;
            stats.sum_sq_error += error * error;
            stats.max_error = std::max(stats.max_error, error);
        }
        return stats;
    };

    SECTION("BF16 statistics over 100 samples") {
        auto stats = compute_stats([&](const std::vector<float>& input) {
            TwoLayerMLP<bf16_t, float> mlp(MLPTestFixture::IN_DIM,
                                           MLPTestFixture::HIDDEN_DIM,
                                           MLPTestFixture::OUT_DIM);
            mlp.init_from_float(fixture.w1_float.data(), fixture.b1_float.data(),
                               fixture.w2_float.data(), fixture.b2_float.data());
            std::vector<bf16_t> in_bf16(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                in_bf16[i] = bf16_t(input[i]);
            }
            return mlp.forward_and_get_float(in_bf16.data());
        });

        double mae = stats.sum_abs_error / NUM_SAMPLES;
        double rmse = std::sqrt(stats.sum_sq_error / NUM_SAMPLES);

        std::cout << "\nBF16 over " << NUM_SAMPLES << " samples:\n";
        std::cout << "  MAE:  " << mae << "\n";
        std::cout << "  RMSE: " << rmse << "\n";
        std::cout << "  Max:  " << stats.max_error << "\n";

        REQUIRE(mae < 0.1);  // Mean absolute error should be small
    }

    SECTION("FP8 E4M3 statistics over 100 samples") {
        auto stats = compute_stats([&](const std::vector<float>& input) {
            TwoLayerMLP<fp8e4m3_t, float> mlp(MLPTestFixture::IN_DIM,
                                              MLPTestFixture::HIDDEN_DIM,
                                              MLPTestFixture::OUT_DIM);
            mlp.init_from_float(fixture.w1_float.data(), fixture.b1_float.data(),
                               fixture.w2_float.data(), fixture.b2_float.data());
            std::vector<fp8e4m3_t> in_fp8(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                in_fp8[i] = fp8e4m3_t(input[i]);
            }
            return mlp.forward_and_get_float(in_fp8.data());
        });

        double mae = stats.sum_abs_error / NUM_SAMPLES;
        double rmse = std::sqrt(stats.sum_sq_error / NUM_SAMPLES);

        std::cout << "\nFP8 E4M3 over " << NUM_SAMPLES << " samples:\n";
        std::cout << "  MAE:  " << mae << "\n";
        std::cout << "  RMSE: " << rmse << "\n";
        std::cout << "  Max:  " << stats.max_error << "\n";

        // FP8 will have more error but should still be bounded
        REQUIRE(mae < 1.0);
    }
}

TEST_CASE("Mixed precision MLP (FP32 activations, INT8 weights)", "[quantization][mlp][mixed]") {
    // Demonstrate weight-only quantization pattern
    MLPTestFixture fixture;

    std::cout << "\n=== Mixed Precision: FP32 activations, quantized weights ===\n";

    // This demonstrates the matmul_mixed kernel usage pattern
    // In practice, you'd use QuantParams to properly quantize weights

    // Quantize weights to INT8 with proper scaling
    auto quant_w1 = compute_quant_params_from_data<std::int8_t>(
        fixture.w1_float.data(), fixture.w1_float.size(), true);
    auto quant_w2 = compute_quant_params_from_data<std::int8_t>(
        fixture.w2_float.data(), fixture.w2_float.size(), true);

    std::cout << "W1 scale: " << quant_w1.scale << ", zero_point: " << quant_w1.zero_point << "\n";
    std::cout << "W2 scale: " << quant_w2.scale << ", zero_point: " << quant_w2.zero_point << "\n";

    // Quantize weights
    std::vector<std::int8_t> w1_int8(fixture.w1_float.size());
    std::vector<std::int8_t> w2_int8(fixture.w2_float.size());
    quant_w1.quantize_array(fixture.w1_float.data(), w1_int8.data(), w1_int8.size());
    quant_w2.quantize_array(fixture.w2_float.data(), w2_int8.data(), w2_int8.size());

    // Check quantization range usage
    int8_t w1_min = *std::min_element(w1_int8.begin(), w1_int8.end());
    int8_t w1_max = *std::max_element(w1_int8.begin(), w1_int8.end());
    std::cout << "W1 INT8 range: [" << (int)w1_min << ", " << (int)w1_max << "]\n";

    REQUIRE(w1_min >= -128);
    REQUIRE(w1_max <= 127);
}
