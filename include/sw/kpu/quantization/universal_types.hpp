#pragma once
// Universal library type aliases and ScalarTraits specializations
// Provides FP16, BF16, FP8 variants, and FP4 types via sw::universal

// Universal library headers for scalar types
// These are header-only and provide software emulation of hardware number formats
#include <universal/number/cfloat/cfloat.hpp>    // Includes half (fp16) typedef
#include <universal/number/bfloat16/bfloat16.hpp>  // bfloat16 type

#include <sw/kpu/quantization/scalar_traits.hpp>

namespace sw::kpu {

// =============================================================================
// Type Aliases for Universal Library Types
// =============================================================================

/// IEEE 754 half-precision float (FP16)
/// Defined in cfloat.hpp as cfloat<16, 5, uint16_t, true, false, false>
using fp16_t = sw::universal::half;

/// Brain floating point (BF16) - same exponent range as FP32
/// Native bfloat16 class from Universal
using bf16_t = sw::universal::bfloat16;

/// FP8 E4M3 - NVIDIA format (4-bit exponent, 3-bit mantissa)
/// Total: 8 bits = 1 sign + 4 exp + 3 mantissa
using fp8e4m3_t = sw::universal::cfloat<8, 4, std::uint8_t, true, true, false>;

/// FP8 E5M2 - IEEE-like format (5-bit exponent, 2-bit mantissa)
/// Total: 8 bits = 1 sign + 5 exp + 2 mantissa
using fp8e5m2_t = sw::universal::cfloat<8, 5, std::uint8_t, true, true, false>;

/// FP8 E3M4 - Extended mantissa format (3-bit exponent, 4-bit mantissa)
/// Total: 8 bits = 1 sign + 3 exp + 4 mantissa
using fp8e3m4_t = sw::universal::cfloat<8, 3, std::uint8_t, true, true, false>;

/// FP8 E2M5 - High-precision format (2-bit exponent, 5-bit mantissa)
/// Total: 8 bits = 1 sign + 2 exp + 5 mantissa
using fp8e2m5_t = sw::universal::cfloat<8, 2, std::uint8_t, true, true, false>;

/// FP4 - Ultra-low precision (1-bit exponent, 2-bit mantissa)
/// Total: 4 bits = 1 sign + 1 exp + 2 mantissa
/// Storage: unpacked, 1 byte per element (for computation efficiency)
using fp4_t = sw::universal::cfloat<4, 1, std::uint8_t, true, true, false>;

// =============================================================================
// ScalarTraits Specializations for Universal Types
// =============================================================================

/// FP16 (half) specialization
template<>
struct ScalarTraits<fp16_t> {
    using type = fp16_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp16_t;

    static constexpr DataType dtype = DataType::FLOAT16;
    static constexpr size_t bits = 16;
    static constexpr size_t storage_bits = 16;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic

    static fp16_t max_value() { return fp16_t(65504.0f); }
    static fp16_t min_value() { return fp16_t(-65504.0f); }
    static fp16_t epsilon() { return fp16_t(0.0009765625f); }  // 2^-10
};

/// BF16 (bfloat16) specialization
template<>
struct ScalarTraits<bf16_t> {
    using type = bf16_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = bf16_t;

    static constexpr DataType dtype = DataType::BFLOAT16;
    static constexpr size_t bits = 16;
    static constexpr size_t storage_bits = 16;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic

    static bf16_t max_value() { return bf16_t(3.38953139e+38f); }
    static bf16_t min_value() { return bf16_t(-3.38953139e+38f); }
    static bf16_t epsilon() { return bf16_t(0.0078125f); }  // 2^-7
};

/// FP8 E4M3 specialization (NVIDIA format)
template<>
struct ScalarTraits<fp8e4m3_t> {
    using type = fp8e4m3_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp8e4m3_t;

    static constexpr DataType dtype = DataType::FP8_E4M3;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic via Universal

    static fp8e4m3_t max_value() { return fp8e4m3_t(448.0f); }
    static fp8e4m3_t min_value() { return fp8e4m3_t(-448.0f); }
    static fp8e4m3_t epsilon() { return fp8e4m3_t(0.125f); }  // 2^-3
};

/// FP8 E5M2 specialization (IEEE-like format)
template<>
struct ScalarTraits<fp8e5m2_t> {
    using type = fp8e5m2_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp8e5m2_t;

    static constexpr DataType dtype = DataType::FP8_E5M2;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic via Universal

    static fp8e5m2_t max_value() { return fp8e5m2_t(57344.0f); }
    static fp8e5m2_t min_value() { return fp8e5m2_t(-57344.0f); }
    static fp8e5m2_t epsilon() { return fp8e5m2_t(0.25f); }  // 2^-2
};

/// FP8 E3M4 specialization
template<>
struct ScalarTraits<fp8e3m4_t> {
    using type = fp8e3m4_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp8e3m4_t;

    static constexpr DataType dtype = DataType::FP8_E3M4;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic via Universal

    static fp8e3m4_t max_value() { return fp8e3m4_t(15.5f); }
    static fp8e3m4_t min_value() { return fp8e3m4_t(-15.5f); }
    static fp8e3m4_t epsilon() { return fp8e3m4_t(0.0625f); }  // 2^-4
};

/// FP8 E2M5 specialization
template<>
struct ScalarTraits<fp8e2m5_t> {
    using type = fp8e2m5_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp8e2m5_t;

    static constexpr DataType dtype = DataType::FP8_E2M5;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;  // Native arithmetic via Universal

    static fp8e2m5_t max_value() { return fp8e2m5_t(3.875f); }
    static fp8e2m5_t min_value() { return fp8e2m5_t(-3.875f); }
    static fp8e2m5_t epsilon() { return fp8e2m5_t(0.03125f); }  // 2^-5
};

/// FP4 specialization (1-bit exponent, 2-bit mantissa)
template<>
struct ScalarTraits<fp4_t> {
    using type = fp4_t;
    using accumulator_type = float;  // Accumulate in FP32
    using storage_type = fp4_t;      // Unpacked storage (1 byte per element)

    static constexpr DataType dtype = DataType::FP4;
    static constexpr size_t bits = 4;
    static constexpr size_t storage_bits = 8;  // Unpacked: 1 byte per element
    static constexpr bool is_packed = false;   // Not packed for computation
    static constexpr bool requires_qdq = false;  // Native arithmetic via Universal

    static fp4_t max_value() { return fp4_t(6.0f); }
    static fp4_t min_value() { return fp4_t(-6.0f); }
    static fp4_t epsilon() { return fp4_t(0.5f); }  // 2^-1 (2-bit mantissa)
};

// =============================================================================
// Type Conversion Utilities
// =============================================================================

/// Convert from any scalar type to float
template<typename Scalar>
inline float to_float(const Scalar& value) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return value;
    } else if constexpr (std::is_same_v<Scalar, double>) {
        return static_cast<float>(value);
    } else if constexpr (std::is_integral_v<Scalar>) {
        return static_cast<float>(value);
    } else {
        // Universal types have implicit conversion to float
        return static_cast<float>(value);
    }
}

/// Convert from float to any scalar type
template<typename Scalar>
inline Scalar from_float(float value) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return value;
    } else if constexpr (std::is_same_v<Scalar, double>) {
        return static_cast<double>(value);
    } else if constexpr (std::is_integral_v<Scalar>) {
        return static_cast<Scalar>(value);
    } else {
        // Universal types have constructor from float
        return Scalar(value);
    }
}

/// Promote value to accumulator type
template<typename Scalar>
inline typename ScalarTraits<Scalar>::accumulator_type
to_accumulator(const Scalar& value) {
    using Acc = typename ScalarTraits<Scalar>::accumulator_type;
    return static_cast<Acc>(value);
}

/// Convert accumulator back to scalar type
template<typename Scalar>
inline Scalar from_accumulator(
    const typename ScalarTraits<Scalar>::accumulator_type& value) {
    return static_cast<Scalar>(value);
}

} // namespace sw::kpu
