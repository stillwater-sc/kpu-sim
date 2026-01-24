#pragma once
// Data type definitions for KPU simulator
// Supports multiple numeric formats for compute operations

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <sw/concepts.hpp>

namespace sw::kpu {

/**
 * @brief Numeric data types supported by the KPU compute fabric
 *
 * The KPU supports various data types for different precision/performance tradeoffs:
 * - FLOAT32: Standard IEEE 754 single precision (4 bytes)
 * - FLOAT16: IEEE 754 half precision (2 bytes)
 * - BFLOAT16: Brain floating point - same range as float32 with less precision (2 bytes)
 * - INT32: Signed 32-bit integer, typically used for accumulators (4 bytes)
 * - INT16: Signed 16-bit integer (2 bytes)
 * - INT8: Signed 8-bit integer for quantized inference (1 byte)
 * - UINT8: Unsigned 8-bit integer (1 byte)
 * - INT4: Signed 4-bit integer for aggressive quantization (packed, 0.5 bytes)
 * - FP8_E4M3: 8-bit float with 4-bit exponent, 3-bit mantissa (NVIDIA format)
 * - FP8_E5M2: 8-bit float with 5-bit exponent, 2-bit mantissa (IEEE-like)
 * - FP8_E3M4: 8-bit float with 3-bit exponent, 4-bit mantissa
 * - FP8_E2M5: 8-bit float with 2-bit exponent, 5-bit mantissa
 * - FP4: 4-bit float with 1-bit exponent, 2-bit mantissa (unpacked storage)
 */
enum class DataType : uint8_t {
    FLOAT32 = 0,    // 4 bytes, IEEE 754 single precision
    FLOAT16 = 1,    // 2 bytes, IEEE 754 half precision
    BFLOAT16 = 2,   // 2 bytes, brain float (bf16)
    INT32 = 3,      // 4 bytes, signed 32-bit integer (accumulators)
    INT16 = 4,      // 2 bytes, signed 16-bit integer
    INT8 = 5,       // 1 byte, signed 8-bit integer
    UINT8 = 6,      // 1 byte, unsigned 8-bit integer
    INT4 = 7,       // 0.5 bytes (packed), signed 4-bit integer

    // FP8 variants (1 byte each)
    FP8_E4M3 = 8,   // 4-bit exponent, 3-bit mantissa (NVIDIA)
    FP8_E5M2 = 9,   // 5-bit exponent, 2-bit mantissa (IEEE-like)
    FP8_E3M4 = 10,  // 3-bit exponent, 4-bit mantissa
    FP8_E2M5 = 11,  // 2-bit exponent, 5-bit mantissa

    // FP4 (1 byte unpacked storage)
    FP4 = 12,       // 1-bit exponent, 2-bit mantissa

    // Sentinel for iteration
    COUNT = 13
};

/**
 * @brief Get the size of a data type in bytes
 * @param dt The data type
 * @return Size in bytes (for INT4, returns 1 as minimum addressable unit)
 */
constexpr Size dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::BFLOAT16: return 2;
        case DataType::INT32: return 4;
        case DataType::INT16: return 2;
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        case DataType::INT4: return 1;  // Minimum addressable unit (2 elements packed)
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        case DataType::FP8_E3M4: return 1;
        case DataType::FP8_E2M5: return 1;
        case DataType::FP4: return 1;   // Unpacked storage (1 byte per element)
        default: return 0;
    }
}

/**
 * @brief Get the size of a data type in bits
 * @param dt The data type
 * @return Size in bits
 */
constexpr Size dtype_bits(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 32;
        case DataType::FLOAT16: return 16;
        case DataType::BFLOAT16: return 16;
        case DataType::INT32: return 32;
        case DataType::INT16: return 16;
        case DataType::INT8: return 8;
        case DataType::UINT8: return 8;
        case DataType::INT4: return 4;
        case DataType::FP8_E4M3: return 8;
        case DataType::FP8_E5M2: return 8;
        case DataType::FP8_E3M4: return 8;
        case DataType::FP8_E2M5: return 8;
        case DataType::FP4: return 4;
        default: return 0;
    }
}

/**
 * @brief Check if a data type is an integer type
 * @param dt The data type
 * @return true if integer type (signed or unsigned)
 */
constexpr bool dtype_is_integer(DataType dt) {
    switch (dt) {
        case DataType::INT32:
        case DataType::INT16:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::INT4:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a data type is a floating point type
 * @param dt The data type
 * @return true if floating point type
 */
constexpr bool dtype_is_floating(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP8_E3M4:
        case DataType::FP8_E2M5:
        case DataType::FP4:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a data type is signed
 * @param dt The data type
 * @return true if signed type
 */
constexpr bool dtype_is_signed(DataType dt) {
    switch (dt) {
        case DataType::UINT8:
            return false;
        default:
            return true;  // All other types are signed
    }
}

/**
 * @brief Check if a data type requires packing (sub-byte types)
 * @param dt The data type
 * @return true if type requires packing
 */
constexpr bool dtype_is_packed(DataType dt) {
    // INT4 is packed (2 elements per byte), FP4 is unpacked (1 byte per element)
    return dt == DataType::INT4;
}

/**
 * @brief Get the number of elements that pack into one byte
 * @param dt The data type
 * @return Elements per byte (1 for most types, 2 for INT4)
 */
constexpr Size dtype_elements_per_byte(DataType dt) {
    if (dt == DataType::INT4) {
        return 2;
    }
    return 8 / dtype_bits(dt);  // Will be >= 1 for all valid types
}

/**
 * @brief Get the appropriate accumulator type for a given input type
 *
 * For quantized computations, accumulators need higher precision to
 * avoid overflow during matrix multiplication.
 *
 * @param dt The input data type
 * @return The accumulator data type
 */
constexpr DataType accumulator_type(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:
            return DataType::FLOAT32;  // FP32 accumulates to FP32
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP8_E3M4:
        case DataType::FP8_E2M5:
        case DataType::FP4:
            return DataType::FLOAT32;  // All reduced-precision floats accumulate to FP32
        case DataType::INT16:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::INT4:
            return DataType::INT32;    // Integer types accumulate to INT32
        case DataType::INT32:
            return DataType::INT32;    // INT32 stays INT32
        default:
            return DataType::FLOAT32;
    }
}

/**
 * @brief Calculate bytes needed for a given number of elements
 * @param dt The data type
 * @param num_elements Number of elements
 * @return Total bytes needed (rounded up for packed types)
 */
constexpr Size dtype_bytes_for_elements(DataType dt, Size num_elements) {
    if (dt == DataType::INT4) {
        // INT4: 2 elements per byte, round up
        return (num_elements + 1) / 2;
    }
    return num_elements * dtype_size(dt);
}

/**
 * @brief Get string name of a data type
 * @param dt The data type
 * @return String representation
 */
inline std::string dtype_name(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::BFLOAT16: return "bfloat16";
        case DataType::INT32: return "int32";
        case DataType::INT16: return "int16";
        case DataType::INT8: return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::INT4: return "int4";
        case DataType::FP8_E4M3: return "fp8_e4m3";
        case DataType::FP8_E5M2: return "fp8_e5m2";
        case DataType::FP8_E3M4: return "fp8_e3m4";
        case DataType::FP8_E2M5: return "fp8_e2m5";
        case DataType::FP4: return "fp4";
        default: return "unknown";
    }
}

/**
 * @brief Parse data type from string name
 * @param name The string name (case-insensitive)
 * @return The data type
 * @throws std::invalid_argument if name is not recognized
 */
inline DataType dtype_from_name(const std::string& name) {
    // Simple case-insensitive comparison
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (lower == "float32" || lower == "f32" || lower == "float") return DataType::FLOAT32;
    if (lower == "float16" || lower == "f16" || lower == "half") return DataType::FLOAT16;
    if (lower == "bfloat16" || lower == "bf16") return DataType::BFLOAT16;
    if (lower == "int32" || lower == "i32") return DataType::INT32;
    if (lower == "int16" || lower == "i16") return DataType::INT16;
    if (lower == "int8" || lower == "i8") return DataType::INT8;
    if (lower == "uint8" || lower == "u8") return DataType::UINT8;
    if (lower == "int4" || lower == "i4") return DataType::INT4;
    if (lower == "fp8_e4m3" || lower == "fp8e4m3") return DataType::FP8_E4M3;
    if (lower == "fp8_e5m2" || lower == "fp8e5m2") return DataType::FP8_E5M2;
    if (lower == "fp8_e3m4" || lower == "fp8e3m4") return DataType::FP8_E3M4;
    if (lower == "fp8_e2m5" || lower == "fp8e2m5") return DataType::FP8_E2M5;
    if (lower == "fp4") return DataType::FP4;

    throw std::invalid_argument("Unknown data type: " + name);
}

/**
 * @brief Get the maximum value representable by a data type
 * @param dt The data type
 * @return Maximum value as double
 */
constexpr double dtype_max_value(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 3.402823466e+38;  // FLT_MAX
        case DataType::FLOAT16: return 65504.0;          // Max finite float16
        case DataType::BFLOAT16: return 3.38953139e+38;  // Approx BF16 max
        case DataType::INT32: return 2147483647.0;       // INT32_MAX
        case DataType::INT16: return 32767.0;            // INT16_MAX
        case DataType::INT8: return 127.0;               // INT8_MAX
        case DataType::UINT8: return 255.0;              // UINT8_MAX
        case DataType::INT4: return 7.0;                 // 4-bit signed max
        case DataType::FP8_E4M3: return 448.0;           // Max value for E4M3
        case DataType::FP8_E5M2: return 57344.0;         // Max value for E5M2
        case DataType::FP8_E3M4: return 15.5;            // Max value for E3M4
        case DataType::FP8_E2M5: return 3.875;           // Max value for E2M5
        case DataType::FP4: return 6.0;                  // Max value for FP4 (1 exp bit)
        default: return 0.0;
    }
}

/**
 * @brief Get the minimum value representable by a data type
 * @param dt The data type
 * @return Minimum value as double
 */
constexpr double dtype_min_value(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return -3.402823466e+38;
        case DataType::FLOAT16: return -65504.0;
        case DataType::BFLOAT16: return -3.38953139e+38;
        case DataType::INT32: return -2147483648.0;
        case DataType::INT16: return -32768.0;
        case DataType::INT8: return -128.0;
        case DataType::UINT8: return 0.0;
        case DataType::INT4: return -8.0;
        case DataType::FP8_E4M3: return -448.0;
        case DataType::FP8_E5M2: return -57344.0;
        case DataType::FP8_E3M4: return -15.5;
        case DataType::FP8_E2M5: return -3.875;
        case DataType::FP4: return -6.0;
        default: return 0.0;
    }
}

} // namespace sw::kpu
