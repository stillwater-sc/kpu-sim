// Tests for the quantization type infrastructure
// Validates scalar traits, type conversions, and kernel instantiation

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/quantization/scalar_traits.hpp>
#include <sw/kpu/quantization/packed_types.hpp>
#include <sw/kpu/quantization/quant_params.hpp>
#include <sw/kpu/quantization/kernels.hpp>

// Universal types - only if available
#if __has_include(<universal/number/bfloat16/bfloat16.hpp>)
#include <sw/kpu/quantization/universal_types.hpp>
#define HAS_UNIVERSAL 1
#else
#define HAS_UNIVERSAL 0
#endif

using namespace sw::kpu;
using Catch::Approx;

// =============================================================================
// Scalar Traits Tests
// =============================================================================

TEST_CASE("ScalarTraits for native types", "[quantization][traits]") {
    SECTION("float traits") {
        REQUIRE(ScalarTraits<float>::dtype == DataType::FLOAT32);
        REQUIRE(ScalarTraits<float>::bits == 32);
        REQUIRE(ScalarTraits<float>::is_packed == false);
        REQUIRE(ScalarTraits<float>::requires_qdq == false);
        REQUIRE(std::is_same_v<ScalarTraits<float>::accumulator_type, float>);
    }

    SECTION("int32 traits") {
        REQUIRE(ScalarTraits<std::int32_t>::dtype == DataType::INT32);
        REQUIRE(ScalarTraits<std::int32_t>::bits == 32);
        REQUIRE(ScalarTraits<std::int32_t>::is_packed == false);
        REQUIRE(std::is_same_v<ScalarTraits<std::int32_t>::accumulator_type, std::int32_t>);
    }

    SECTION("int16 traits") {
        REQUIRE(ScalarTraits<std::int16_t>::dtype == DataType::INT16);
        REQUIRE(ScalarTraits<std::int16_t>::bits == 16);
        REQUIRE(ScalarTraits<std::int16_t>::requires_qdq == true);
        REQUIRE(std::is_same_v<ScalarTraits<std::int16_t>::accumulator_type, std::int32_t>);
    }

    SECTION("int8 traits") {
        REQUIRE(ScalarTraits<std::int8_t>::dtype == DataType::INT8);
        REQUIRE(ScalarTraits<std::int8_t>::bits == 8);
        REQUIRE(ScalarTraits<std::int8_t>::requires_qdq == true);
        REQUIRE(std::is_same_v<ScalarTraits<std::int8_t>::accumulator_type, std::int32_t>);
    }

    SECTION("uint8 traits") {
        REQUIRE(ScalarTraits<std::uint8_t>::dtype == DataType::UINT8);
        REQUIRE(ScalarTraits<std::uint8_t>::bits == 8);
        REQUIRE(ScalarTraits<std::uint8_t>::max_value() == 255);
        REQUIRE(ScalarTraits<std::uint8_t>::min_value() == 0);
    }
}

TEST_CASE("AccumulatorType alias", "[quantization][traits]") {
    REQUIRE(std::is_same_v<AccumulatorType<float>, float>);
    REQUIRE(std::is_same_v<AccumulatorType<std::int8_t>, std::int32_t>);
    REQUIRE(std::is_same_v<AccumulatorType<std::int16_t>, std::int32_t>);
}

// =============================================================================
// Packed INT4 Tests
// =============================================================================

TEST_CASE("PackedInt4 storage", "[quantization][packed]") {
    SECTION("pack and unpack") {
        PackedInt4 packed;
        packed.set_high(5);   // Index 0
        packed.set_low(-3);   // Index 1

        REQUIRE(packed.high() == 5);
        REQUIRE(packed.low() == -3);
        REQUIRE(packed.get(0) == 5);
        REQUIRE(packed.get(1) == -3);
    }

    SECTION("sign extension") {
        PackedInt4 packed;
        packed.set(0, -8);  // Min value for 4-bit signed
        packed.set(1, 7);   // Max value for 4-bit signed

        REQUIRE(packed.get(0) == -8);
        REQUIRE(packed.get(1) == 7);
    }

    SECTION("value clamping in Int4Unpacked") {
        Int4Unpacked a(10);   // Should clamp to 7
        Int4Unpacked b(-10);  // Should clamp to -8

        REQUIRE(a.value == 7);
        REQUIRE(b.value == -8);
    }
}

TEST_CASE("INT4 Q/DQ conversion", "[quantization][packed]") {
    SECTION("dequantize and quantize roundtrip") {
        PackedInt4 packed_src[2];
        packed_src[0].set(0, 3);
        packed_src[0].set(1, -2);
        packed_src[1].set(0, 7);
        packed_src[1].set(1, -8);

        // Dequantize
        Int4Unpacked unpacked[4];
        dequantize_int4(packed_src, unpacked, 4);

        REQUIRE(unpacked[0].value == 3);
        REQUIRE(unpacked[1].value == -2);
        REQUIRE(unpacked[2].value == 7);
        REQUIRE(unpacked[3].value == -8);

        // Quantize back
        PackedInt4 packed_dst[2];
        quantize_int4(unpacked, packed_dst, 4);

        REQUIRE(packed_dst[0].get(0) == 3);
        REQUIRE(packed_dst[0].get(1) == -2);
        REQUIRE(packed_dst[1].get(0) == 7);
        REQUIRE(packed_dst[1].get(1) == -8);
    }
}

// =============================================================================
// Quantization Parameters Tests
// =============================================================================

TEST_CASE("QuantParams computation", "[quantization][params]") {
    SECTION("symmetric quantization for int8") {
        auto params = compute_quant_params<std::int8_t>(-1.0f, 1.0f, true);

        REQUIRE(params.zero_point == 0);
        REQUIRE(params.scale == Approx(1.0f / 127.0f).epsilon(0.01));

        // Test quantize/dequantize
        std::int8_t q = params.quantize(0.5f);
        float dq = params.dequantize(q);
        REQUIRE(dq == Approx(0.5f).margin(0.01));
    }

    SECTION("asymmetric quantization for uint8") {
        auto params = compute_quant_params<std::uint8_t>(0.0f, 1.0f, false);

        // scale should be 1/255, zero_point should be 0
        REQUIRE(params.scale == Approx(1.0f / 255.0f).epsilon(0.01));
        REQUIRE(params.zero_point == 0);

        // Test values
        REQUIRE(params.quantize(0.0f) == 0);
        REQUIRE(params.quantize(1.0f) == 255);
    }

    SECTION("per-channel quantization") {
        float data[] = {
            // Channel 0: range [-2, 2]
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            // Channel 1: range [0, 10]
            0.0f, 2.5f, 5.0f, 7.5f, 10.0f
        };

        auto params = compute_per_channel_quant_params<std::int8_t>(
            data, 2, 5, true);

        REQUIRE(params.num_channels == 2);
        REQUIRE(params.scales[0] != params.scales[1]);  // Different ranges
    }
}

// =============================================================================
// Kernel Tests
// =============================================================================

TEST_CASE("Template matmul kernel", "[quantization][kernels]") {
    SECTION("float32 matmul") {
        // A: 2x3, B: 3x2, C: 2x2
        float A_data[] = {1, 2, 3, 4, 5, 6};
        float B_data[] = {1, 2, 3, 4, 5, 6};
        float C_data[] = {0, 0, 0, 0};

        kernels::TensorView<const float> A(A_data, 2, 3);
        kernels::TensorView<const float> B(B_data, 3, 2);
        kernels::TensorView<float> C(C_data, 2, 2);

        kernels::matmul(A, B, C);

        // Expected: [[22, 28], [49, 64]]
        REQUIRE(C(0, 0) == Approx(22.0f));
        REQUIRE(C(0, 1) == Approx(28.0f));
        REQUIRE(C(1, 0) == Approx(49.0f));
        REQUIRE(C(1, 1) == Approx(64.0f));
    }

    SECTION("int8 matmul with int32 accumulator") {
        std::int8_t A_data[] = {1, 2, 3, 4};  // 2x2
        std::int8_t B_data[] = {5, 6, 7, 8};  // 2x2
        std::int32_t C_data[] = {0, 0, 0, 0}; // 2x2

        kernels::TensorView<const std::int8_t> A(A_data, 2, 2);
        kernels::TensorView<const std::int8_t> B(B_data, 2, 2);
        kernels::TensorView<std::int32_t> C(C_data, 2, 2);

        kernels::matmul(A, B, C);

        // Expected: [[19, 22], [43, 50]]
        REQUIRE(C(0, 0) == 19);
        REQUIRE(C(0, 1) == 22);
        REQUIRE(C(1, 0) == 43);
        REQUIRE(C(1, 1) == 50);
    }
}

TEST_CASE("Mixed precision matmul", "[quantization][kernels]") {
    // float activations, int8 weights
    float A_data[] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
    std::int8_t B_data[] = {1, 2, 3, 4};         // 2x2
    float C_data[] = {0, 0, 0, 0};               // 2x2

    kernels::TensorView<const float> A(A_data, 2, 2);
    kernels::TensorView<const std::int8_t> B(B_data, 2, 2);
    kernels::TensorView<float> C(C_data, 2, 2);

    kernels::matmul_mixed(A, B, C);

    // Expected: [[7, 10], [15, 22]]
    REQUIRE(C(0, 0) == Approx(7.0f));
    REQUIRE(C(0, 1) == Approx(10.0f));
    REQUIRE(C(1, 0) == Approx(15.0f));
    REQUIRE(C(1, 1) == Approx(22.0f));
}

TEST_CASE("Activation functions", "[quantization][kernels]") {
    SECTION("relu") {
        float X[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float Y[5];

        kernels::relu(X, Y, 5);

        REQUIRE(Y[0] == 0.0f);
        REQUIRE(Y[1] == 0.0f);
        REQUIRE(Y[2] == 0.0f);
        REQUIRE(Y[3] == 1.0f);
        REQUIRE(Y[4] == 2.0f);
    }

    SECTION("softmax") {
        float X[] = {1.0f, 2.0f, 3.0f};
        float Y[3];

        kernels::softmax(X, Y, 1, 3);

        // Should sum to 1
        float sum = Y[0] + Y[1] + Y[2];
        REQUIRE(sum == Approx(1.0f));

        // Should be ordered: Y[0] < Y[1] < Y[2]
        REQUIRE(Y[0] < Y[1]);
        REQUIRE(Y[1] < Y[2]);
    }
}

// =============================================================================
// Universal Types Tests (if available)
// =============================================================================

#if HAS_UNIVERSAL
TEST_CASE("Universal type traits", "[quantization][universal]") {
    SECTION("bf16 traits") {
        REQUIRE(ScalarTraits<bf16_t>::dtype == DataType::BFLOAT16);
        REQUIRE(ScalarTraits<bf16_t>::bits == 16);
        REQUIRE(std::is_same_v<ScalarTraits<bf16_t>::accumulator_type, float>);
    }

    SECTION("fp16 traits") {
        REQUIRE(ScalarTraits<fp16_t>::dtype == DataType::FLOAT16);
        REQUIRE(ScalarTraits<fp16_t>::bits == 16);
        REQUIRE(std::is_same_v<ScalarTraits<fp16_t>::accumulator_type, float>);
    }

    SECTION("fp8e4m3 traits") {
        REQUIRE(ScalarTraits<fp8e4m3_t>::dtype == DataType::FP8_E4M3);
        REQUIRE(ScalarTraits<fp8e4m3_t>::bits == 8);
        REQUIRE(std::is_same_v<ScalarTraits<fp8e4m3_t>::accumulator_type, float>);
    }

    SECTION("fp4 traits") {
        REQUIRE(ScalarTraits<fp4_t>::dtype == DataType::FP4);
        REQUIRE(ScalarTraits<fp4_t>::bits == 4);
        REQUIRE(ScalarTraits<fp4_t>::storage_bits == 8);  // Unpacked
        REQUIRE(std::is_same_v<ScalarTraits<fp4_t>::accumulator_type, float>);
    }
}

TEST_CASE("Universal type conversions", "[quantization][universal]") {
    SECTION("bf16 to float") {
        bf16_t val(3.14f);
        float f = to_float(val);
        REQUIRE(f == Approx(3.14f).margin(0.01));
    }

    SECTION("float to bf16") {
        float f = 2.718f;
        bf16_t val = from_float<bf16_t>(f);
        REQUIRE(to_float(val) == Approx(2.718f).margin(0.01));
    }

    SECTION("bf16 matmul with float accumulator") {
        bf16_t A_data[] = {bf16_t(1), bf16_t(2), bf16_t(3), bf16_t(4)};
        bf16_t B_data[] = {bf16_t(1), bf16_t(2), bf16_t(3), bf16_t(4)};
        float C_data[] = {0, 0, 0, 0};

        kernels::TensorView<const bf16_t> A(A_data, 2, 2);
        kernels::TensorView<const bf16_t> B(B_data, 2, 2);
        kernels::TensorView<float> C(C_data, 2, 2);

        kernels::matmul(A, B, C);

        REQUIRE(C(0, 0) == Approx(7.0f).margin(0.1));
        REQUIRE(C(0, 1) == Approx(10.0f).margin(0.1));
        REQUIRE(C(1, 0) == Approx(15.0f).margin(0.1));
        REQUIRE(C(1, 1) == Approx(22.0f).margin(0.1));
    }
}
#endif

// =============================================================================
// DataType Extension Tests
// =============================================================================

TEST_CASE("DataType enum extensions", "[quantization][datatypes]") {
    SECTION("FP8 types exist") {
        REQUIRE(dtype_name(DataType::FP8_E4M3) == "fp8_e4m3");
        REQUIRE(dtype_name(DataType::FP8_E5M2) == "fp8_e5m2");
        REQUIRE(dtype_name(DataType::FP8_E3M4) == "fp8_e3m4");
        REQUIRE(dtype_name(DataType::FP8_E2M5) == "fp8_e2m5");
    }

    SECTION("FP4 type exists") {
        REQUIRE(dtype_name(DataType::FP4) == "fp4");
        REQUIRE(dtype_bits(DataType::FP4) == 4);
    }

    SECTION("INT16 type exists") {
        REQUIRE(dtype_name(DataType::INT16) == "int16");
        REQUIRE(dtype_bits(DataType::INT16) == 16);
        REQUIRE(dtype_size(DataType::INT16) == 2);
    }

    SECTION("accumulator types") {
        REQUIRE(accumulator_type(DataType::FP8_E4M3) == DataType::FLOAT32);
        REQUIRE(accumulator_type(DataType::FP4) == DataType::FLOAT32);
        REQUIRE(accumulator_type(DataType::INT16) == DataType::INT32);
    }

    SECTION("is_floating for FP8/FP4") {
        REQUIRE(dtype_is_floating(DataType::FP8_E4M3));
        REQUIRE(dtype_is_floating(DataType::FP8_E5M2));
        REQUIRE(dtype_is_floating(DataType::FP4));
    }

    SECTION("parse from string") {
        REQUIRE(dtype_from_name("fp8_e4m3") == DataType::FP8_E4M3);
        REQUIRE(dtype_from_name("fp8e5m2") == DataType::FP8_E5M2);
        REQUIRE(dtype_from_name("fp4") == DataType::FP4);
        REQUIRE(dtype_from_name("int16") == DataType::INT16);
    }
}
