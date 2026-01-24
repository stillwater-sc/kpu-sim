#pragma once
// Scalar type traits for quantization support
// Maps C++ types to KPU DataType enum and accumulator types

#include <cstdint>
#include <type_traits>
#include <limits>
#include <sw/kpu/data_types.hpp>

namespace sw::kpu {

/// Primary template - must be specialized for each supported type
template<typename Scalar>
struct ScalarTraits {
    static_assert(sizeof(Scalar) == 0,
        "ScalarTraits must be specialized for this type. "
        "See include/sw/kpu/quantization/scalar_traits.hpp");
};

// =============================================================================
// Native Float Types
// =============================================================================

/// Float32 specialization
template<>
struct ScalarTraits<float> {
    using type = float;
    using accumulator_type = float;
    using storage_type = float;

    static constexpr DataType dtype = DataType::FLOAT32;
    static constexpr size_t bits = 32;
    static constexpr size_t storage_bits = 32;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;

    static constexpr float max_value() { return std::numeric_limits<float>::max(); }
    static constexpr float min_value() { return std::numeric_limits<float>::lowest(); }
    static constexpr float epsilon() { return std::numeric_limits<float>::epsilon(); }
};

/// Double specialization (typically converted to float for KPU)
template<>
struct ScalarTraits<double> {
    using type = double;
    using accumulator_type = double;
    using storage_type = double;

    static constexpr DataType dtype = DataType::FLOAT32;  // Maps to float32 for KPU
    static constexpr size_t bits = 64;
    static constexpr size_t storage_bits = 64;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;

    static constexpr double max_value() { return std::numeric_limits<double>::max(); }
    static constexpr double min_value() { return std::numeric_limits<double>::lowest(); }
    static constexpr double epsilon() { return std::numeric_limits<double>::epsilon(); }
};

// =============================================================================
// Native Integer Types
// =============================================================================

/// INT32 specialization
template<>
struct ScalarTraits<std::int32_t> {
    using type = std::int32_t;
    using accumulator_type = std::int32_t;
    using storage_type = std::int32_t;

    static constexpr DataType dtype = DataType::INT32;
    static constexpr size_t bits = 32;
    static constexpr size_t storage_bits = 32;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = false;

    static constexpr std::int32_t max_value() { return std::numeric_limits<std::int32_t>::max(); }
    static constexpr std::int32_t min_value() { return std::numeric_limits<std::int32_t>::min(); }
};

/// INT16 specialization
template<>
struct ScalarTraits<std::int16_t> {
    using type = std::int16_t;
    using accumulator_type = std::int32_t;  // Promote to int32 for accumulation
    using storage_type = std::int16_t;

    static constexpr DataType dtype = DataType::INT16;
    static constexpr size_t bits = 16;
    static constexpr size_t storage_bits = 16;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = true;  // Quantized type needs scale/zero_point

    static constexpr std::int16_t max_value() { return std::numeric_limits<std::int16_t>::max(); }
    static constexpr std::int16_t min_value() { return std::numeric_limits<std::int16_t>::min(); }
};

/// INT8 specialization
template<>
struct ScalarTraits<std::int8_t> {
    using type = std::int8_t;
    using accumulator_type = std::int32_t;  // Promote to int32 for accumulation
    using storage_type = std::int8_t;

    static constexpr DataType dtype = DataType::INT8;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = true;  // Quantized type needs scale/zero_point

    static constexpr std::int8_t max_value() { return 127; }
    static constexpr std::int8_t min_value() { return -128; }
};

/// UINT8 specialization
template<>
struct ScalarTraits<std::uint8_t> {
    using type = std::uint8_t;
    using accumulator_type = std::int32_t;  // Promote to int32 for accumulation
    using storage_type = std::uint8_t;

    static constexpr DataType dtype = DataType::UINT8;
    static constexpr size_t bits = 8;
    static constexpr size_t storage_bits = 8;
    static constexpr bool is_packed = false;
    static constexpr bool requires_qdq = true;  // Quantized type needs scale/zero_point

    static constexpr std::uint8_t max_value() { return 255; }
    static constexpr std::uint8_t min_value() { return 0; }
};

// =============================================================================
// Helper Concepts (C++20)
// =============================================================================

/// Concept for types with ScalarTraits specialization
template<typename T>
concept HasScalarTraits = requires {
    typename ScalarTraits<T>::type;
    typename ScalarTraits<T>::accumulator_type;
    { ScalarTraits<T>::dtype } -> std::convertible_to<DataType>;
    { ScalarTraits<T>::bits } -> std::convertible_to<size_t>;
};

/// Concept for types that require Q/DQ conversion
template<typename T>
concept RequiresQDQ = HasScalarTraits<T> && ScalarTraits<T>::requires_qdq;

/// Concept for packed types (multiple elements per storage unit)
template<typename T>
concept IsPackedType = HasScalarTraits<T> && ScalarTraits<T>::is_packed;

/// Concept for floating-point scalar types
template<typename T>
concept IsFloatingScalar = HasScalarTraits<T> &&
    (ScalarTraits<T>::dtype == DataType::FLOAT32 ||
     ScalarTraits<T>::dtype == DataType::FLOAT16 ||
     ScalarTraits<T>::dtype == DataType::BFLOAT16 ||
     ScalarTraits<T>::dtype == DataType::FP8_E4M3 ||
     ScalarTraits<T>::dtype == DataType::FP8_E5M2 ||
     ScalarTraits<T>::dtype == DataType::FP8_E3M4 ||
     ScalarTraits<T>::dtype == DataType::FP8_E2M5 ||
     ScalarTraits<T>::dtype == DataType::FP4);

/// Concept for integer scalar types
template<typename T>
concept IsIntegerScalar = HasScalarTraits<T> &&
    (ScalarTraits<T>::dtype == DataType::INT32 ||
     ScalarTraits<T>::dtype == DataType::INT16 ||
     ScalarTraits<T>::dtype == DataType::INT8 ||
     ScalarTraits<T>::dtype == DataType::UINT8 ||
     ScalarTraits<T>::dtype == DataType::INT4);

// =============================================================================
// Type Selection Utilities
// =============================================================================

/// Select accumulator type for given scalar type
template<typename Scalar>
using AccumulatorType = typename ScalarTraits<Scalar>::accumulator_type;

/// Select storage type for given scalar type
template<typename Scalar>
using StorageType = typename ScalarTraits<Scalar>::storage_type;

/// Get DataType enum for a C++ type
template<HasScalarTraits Scalar>
constexpr DataType dtype_of = ScalarTraits<Scalar>::dtype;

/// Get bit width for a C++ type
template<HasScalarTraits Scalar>
constexpr size_t bits_of = ScalarTraits<Scalar>::bits;

// =============================================================================
// Mixed-Precision Accumulator Selection
// =============================================================================

/// Select common accumulator type for two scalar types
/// Used for mixed-precision operations
template<typename ScalarA, typename ScalarB>
struct CommonAccumulator {
    // Default: use larger accumulator
    using type = std::conditional_t<
        std::is_floating_point_v<typename ScalarTraits<ScalarA>::accumulator_type>,
        float,  // Float accumulator takes precedence
        std::conditional_t<
            sizeof(typename ScalarTraits<ScalarA>::accumulator_type) >=
            sizeof(typename ScalarTraits<ScalarB>::accumulator_type),
            typename ScalarTraits<ScalarA>::accumulator_type,
            typename ScalarTraits<ScalarB>::accumulator_type
        >
    >;
};

template<typename ScalarA, typename ScalarB>
using CommonAccumulatorType = typename CommonAccumulator<ScalarA, ScalarB>::type;

} // namespace sw::kpu
