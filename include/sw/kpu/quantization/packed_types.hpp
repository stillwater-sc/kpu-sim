#pragma once
// Packed storage types for sub-byte quantization (INT4)
// Provides Q/DQ conversion between packed storage and unpacked computation

#include <cstdint>
#include <cstddef>
#include <sw/kpu/quantization/scalar_traits.hpp>

namespace sw::kpu {

// =============================================================================
// Packed INT4 Storage
// =============================================================================

/**
 * @brief Packed INT4 storage: 2 elements per byte
 *
 * Layout: High nibble = even index, Low nibble = odd index
 * This matches the memory layout used by common accelerators.
 *
 * Example: For elements [a, b, c, d, e, f]:
 *   byte 0: (a << 4) | (b & 0x0F)
 *   byte 1: (c << 4) | (d & 0x0F)
 *   byte 2: (e << 4) | (f & 0x0F)
 */
struct PackedInt4 {
    std::uint8_t data;

    /// Default constructor
    constexpr PackedInt4() : data(0) {}

    /// Construct from raw byte
    constexpr explicit PackedInt4(std::uint8_t d) : data(d) {}

    /// Extract element at index (0 or 1)
    /// Index 0 = high nibble, Index 1 = low nibble
    constexpr std::int8_t get(size_t idx) const {
        std::int8_t val = (idx == 0)
            ? static_cast<std::int8_t>((data >> 4) & 0x0F)
            : static_cast<std::int8_t>(data & 0x0F);
        // Sign extend from 4 bits (if bit 3 is set, the value is negative)
        return (val & 0x08) ? (val | static_cast<std::int8_t>(0xF0)) : val;
    }

    /// Set element at index (0 or 1)
    constexpr void set(size_t idx, std::int8_t val) {
        std::uint8_t masked = static_cast<std::uint8_t>(val) & 0x0F;
        if (idx == 0) {
            data = static_cast<std::uint8_t>((data & 0x0F) | (masked << 4));
        } else {
            data = (data & 0xF0) | masked;
        }
    }

    /// Get high nibble (even index)
    constexpr std::int8_t high() const { return get(0); }

    /// Get low nibble (odd index)
    constexpr std::int8_t low() const { return get(1); }

    /// Set high nibble (even index)
    constexpr void set_high(std::int8_t val) { set(0, val); }

    /// Set low nibble (odd index)
    constexpr void set_low(std::int8_t val) { set(1, val); }
};

// =============================================================================
// Unpacked INT4 for Computation
// =============================================================================

/**
 * @brief Unpacked INT4 for kernel computation
 *
 * Uses 1 byte per element for simple arithmetic.
 * Values are sign-extended from 4 bits.
 * Range: [-8, 7]
 */
struct Int4Unpacked {
    std::int8_t value;

    constexpr Int4Unpacked() : value(0) {}
    constexpr explicit Int4Unpacked(std::int8_t v) : value(clamp(v)) {}

    /// Clamp value to 4-bit range [-8, 7]
    static constexpr std::int8_t clamp(std::int8_t v) {
        if (v < -8) return -8;
        if (v > 7) return 7;
        return v;
    }

    /// Arithmetic operations (accumulate in int8 internally)
    constexpr Int4Unpacked operator+(Int4Unpacked rhs) const {
        return Int4Unpacked(static_cast<std::int8_t>(value + rhs.value));
    }

    constexpr Int4Unpacked operator-(Int4Unpacked rhs) const {
        return Int4Unpacked(static_cast<std::int8_t>(value - rhs.value));
    }

    constexpr Int4Unpacked operator*(Int4Unpacked rhs) const {
        return Int4Unpacked(static_cast<std::int8_t>(value * rhs.value));
    }

    constexpr Int4Unpacked& operator+=(Int4Unpacked rhs) {
        value = clamp(static_cast<std::int8_t>(value + rhs.value));
        return *this;
    }

    constexpr Int4Unpacked& operator-=(Int4Unpacked rhs) {
        value = clamp(static_cast<std::int8_t>(value - rhs.value));
        return *this;
    }

    /// Comparison operators
    constexpr bool operator<(Int4Unpacked rhs) const { return value < rhs.value; }
    constexpr bool operator>(Int4Unpacked rhs) const { return value > rhs.value; }
    constexpr bool operator==(Int4Unpacked rhs) const { return value == rhs.value; }
    constexpr bool operator!=(Int4Unpacked rhs) const { return value != rhs.value; }

    /// Conversion to/from wider types
    constexpr explicit operator std::int8_t() const { return value; }
    constexpr explicit operator std::int32_t() const { return static_cast<std::int32_t>(value); }
    constexpr explicit operator float() const { return static_cast<float>(value); }
};

/// ScalarTraits for Int4Unpacked
template<>
struct ScalarTraits<Int4Unpacked> {
    using type = Int4Unpacked;
    using accumulator_type = std::int32_t;  // Accumulate in INT32
    using storage_type = PackedInt4;        // Packed storage

    static constexpr DataType dtype = DataType::INT4;
    static constexpr size_t bits = 4;
    static constexpr size_t storage_bits = 4;  // 2 elements per byte
    static constexpr bool is_packed = true;
    static constexpr bool requires_qdq = true;

    static constexpr Int4Unpacked max_value() { return Int4Unpacked(7); }
    static constexpr Int4Unpacked min_value() { return Int4Unpacked(-8); }
};

// =============================================================================
// Q/DQ Conversion Functions
// =============================================================================

/**
 * @brief Dequantize packed INT4 to unpacked for computation
 *
 * @param src Source packed data (count/2 bytes)
 * @param dst Destination unpacked data (count elements)
 * @param count Number of elements (must be even)
 */
inline void dequantize_int4(
    const PackedInt4* src,
    Int4Unpacked* dst,
    size_t count
) {
    size_t packed_count = (count + 1) / 2;
    size_t out_idx = 0;

    for (size_t i = 0; i < packed_count && out_idx < count; ++i) {
        dst[out_idx++] = Int4Unpacked(src[i].high());
        if (out_idx < count) {
            dst[out_idx++] = Int4Unpacked(src[i].low());
        }
    }
}

/**
 * @brief Quantize unpacked INT4 to packed storage
 *
 * @param src Source unpacked data (count elements)
 * @param dst Destination packed data (count/2 bytes)
 * @param count Number of elements (must be even)
 */
inline void quantize_int4(
    const Int4Unpacked* src,
    PackedInt4* dst,
    size_t count
) {
    size_t packed_count = (count + 1) / 2;

    for (size_t i = 0; i < packed_count; ++i) {
        size_t src_idx = i * 2;
        std::int8_t high = (src_idx < count) ? src[src_idx].value : 0;
        std::int8_t low = (src_idx + 1 < count) ? src[src_idx + 1].value : 0;

        dst[i] = PackedInt4();
        dst[i].set_high(high);
        dst[i].set_low(low);
    }
}

/**
 * @brief Dequantize packed INT4 directly to int8 buffer
 *
 * Useful for immediate kernel use without intermediate type
 */
inline void dequantize_int4_to_int8(
    const PackedInt4* src,
    std::int8_t* dst,
    size_t count
) {
    size_t packed_count = (count + 1) / 2;
    size_t out_idx = 0;

    for (size_t i = 0; i < packed_count && out_idx < count; ++i) {
        dst[out_idx++] = src[i].high();
        if (out_idx < count) {
            dst[out_idx++] = src[i].low();
        }
    }
}

/**
 * @brief Quantize int8 buffer to packed INT4 storage
 *
 * Values outside [-8, 7] are clamped
 */
inline void quantize_int8_to_int4(
    const std::int8_t* src,
    PackedInt4* dst,
    size_t count
) {
    size_t packed_count = (count + 1) / 2;

    for (size_t i = 0; i < packed_count; ++i) {
        size_t src_idx = i * 2;
        std::int8_t high = (src_idx < count) ? Int4Unpacked::clamp(src[src_idx]) : 0;
        std::int8_t low = (src_idx + 1 < count) ? Int4Unpacked::clamp(src[src_idx + 1]) : 0;

        dst[i] = PackedInt4();
        dst[i].set_high(high);
        dst[i].set_low(low);
    }
}

// =============================================================================
// Packed Buffer Utilities
// =============================================================================

/// Calculate packed buffer size for given element count
inline constexpr size_t packed_int4_size(size_t element_count) {
    return (element_count + 1) / 2;
}

/// Check if element count is valid for packed storage (should be even)
inline constexpr bool is_valid_int4_count(size_t element_count) {
    return (element_count % 2) == 0;
}

} // namespace sw::kpu
