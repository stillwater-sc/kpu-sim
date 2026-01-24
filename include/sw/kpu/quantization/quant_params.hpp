#pragma once
// Quantization parameters for affine quantization
// Supports per-tensor and per-channel quantization schemes

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sw/kpu/quantization/scalar_traits.hpp>

namespace sw::kpu {

// =============================================================================
// Per-Tensor Quantization Parameters
// =============================================================================

/**
 * @brief Affine quantization parameters
 *
 * Quantization formula: x_q = round(x / scale) + zero_point
 * Dequantization formula: x = (x_q - zero_point) * scale
 *
 * @tparam Scalar The quantized scalar type (e.g., int8_t, Int4Unpacked)
 */
template<typename Scalar>
struct QuantParams {
    float scale;         // Scale factor
    std::int32_t zero_point;  // Zero point offset

    /// Default constructor (identity for floats)
    QuantParams() : scale(1.0f), zero_point(0) {}

    /// Construct with explicit parameters
    QuantParams(float s, std::int32_t zp) : scale(s), zero_point(zp) {}

    /// Quantize a float value to Scalar
    Scalar quantize(float x) const {
        float q = std::round(x / scale) + static_cast<float>(zero_point);
        // Clamp to valid range
        float min_val = static_cast<float>(dtype_min_value(ScalarTraits<Scalar>::dtype));
        float max_val = static_cast<float>(dtype_max_value(ScalarTraits<Scalar>::dtype));
        q = std::clamp(q, min_val, max_val);
        return static_cast<Scalar>(static_cast<std::int32_t>(q));
    }

    /// Dequantize a Scalar value to float
    float dequantize(Scalar x_q) const {
        return (static_cast<float>(static_cast<std::int32_t>(x_q)) -
                static_cast<float>(zero_point)) * scale;
    }

    /// Quantize an array
    void quantize_array(const float* src, Scalar* dst, size_t count) const {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = quantize(src[i]);
        }
    }

    /// Dequantize an array
    void dequantize_array(const Scalar* src, float* dst, size_t count) const {
        for (size_t i = 0; i < count; ++i) {
            dst[i] = dequantize(src[i]);
        }
    }
};

// =============================================================================
// Per-Channel Quantization Parameters
// =============================================================================

/**
 * @brief Per-channel quantization parameters
 *
 * Each channel has its own scale and zero_point.
 * Common for weight quantization where each output channel
 * has different statistics.
 *
 * @tparam Scalar The quantized scalar type
 */
template<typename Scalar>
struct PerChannelQuantParams {
    std::vector<float> scales;
    std::vector<std::int32_t> zero_points;
    size_t num_channels;
    int channel_axis;  // Which axis contains channels (typically 0 for weights)

    /// Default constructor
    PerChannelQuantParams() : num_channels(0), channel_axis(0) {}

    /// Construct with channel count
    explicit PerChannelQuantParams(size_t n_channels, int axis = 0)
        : scales(n_channels, 1.0f)
        , zero_points(n_channels, 0)
        , num_channels(n_channels)
        , channel_axis(axis)
    {}

    /// Quantize a single value for a specific channel
    Scalar quantize(float x, size_t channel) const {
        float q = std::round(x / scales[channel]) +
                  static_cast<float>(zero_points[channel]);
        float min_val = static_cast<float>(dtype_min_value(ScalarTraits<Scalar>::dtype));
        float max_val = static_cast<float>(dtype_max_value(ScalarTraits<Scalar>::dtype));
        q = std::clamp(q, min_val, max_val);
        return static_cast<Scalar>(static_cast<std::int32_t>(q));
    }

    /// Dequantize a single value for a specific channel
    float dequantize(Scalar x_q, size_t channel) const {
        return (static_cast<float>(static_cast<std::int32_t>(x_q)) -
                static_cast<float>(zero_points[channel])) * scales[channel];
    }
};

// =============================================================================
// Quantization Parameter Computation
// =============================================================================

/**
 * @brief Compute quantization parameters from data range
 *
 * @tparam Scalar Target quantized type
 * @param min_val Minimum value in the data
 * @param max_val Maximum value in the data
 * @param symmetric If true, use symmetric quantization (zero_point = 0)
 * @return QuantParams for the specified range
 */
template<typename Scalar>
QuantParams<Scalar> compute_quant_params(
    float min_val,
    float max_val,
    bool symmetric = false
) {
    QuantParams<Scalar> params;

    const float qmin = static_cast<float>(dtype_min_value(ScalarTraits<Scalar>::dtype));
    const float qmax = static_cast<float>(dtype_max_value(ScalarTraits<Scalar>::dtype));

    if (symmetric) {
        // Symmetric quantization: zero_point = 0
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        // Handle edge case where data is all zeros
        if (abs_max < 1e-10f) abs_max = 1e-10f;
        params.scale = abs_max / std::max(std::abs(qmin), qmax);
        params.zero_point = 0;
    } else {
        // Asymmetric quantization
        float range = max_val - min_val;
        if (range < 1e-10f) range = 1e-10f;  // Handle constant data

        params.scale = range / (qmax - qmin);
        params.zero_point = static_cast<std::int32_t>(
            std::round(qmin - min_val / params.scale)
        );

        // Clamp zero_point to valid range
        params.zero_point = std::clamp(
            params.zero_point,
            static_cast<std::int32_t>(qmin),
            static_cast<std::int32_t>(qmax)
        );
    }

    return params;
}

/**
 * @brief Compute quantization parameters from data statistics
 *
 * Uses min/max from the data array.
 *
 * @tparam Scalar Target quantized type
 * @param data Input data array
 * @param count Number of elements
 * @param symmetric If true, use symmetric quantization
 * @return QuantParams for the data
 */
template<typename Scalar>
QuantParams<Scalar> compute_quant_params_from_data(
    const float* data,
    size_t count,
    bool symmetric = false
) {
    if (count == 0) {
        return QuantParams<Scalar>(1.0f, 0);
    }

    float min_val = data[0];
    float max_val = data[0];
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    return compute_quant_params<Scalar>(min_val, max_val, symmetric);
}

/**
 * @brief Compute per-channel quantization parameters
 *
 * @tparam Scalar Target quantized type
 * @param data Input data array [num_channels x elements_per_channel]
 * @param num_channels Number of channels
 * @param elements_per_channel Elements in each channel
 * @param symmetric If true, use symmetric quantization
 * @return PerChannelQuantParams for the data
 */
template<typename Scalar>
PerChannelQuantParams<Scalar> compute_per_channel_quant_params(
    const float* data,
    size_t num_channels,
    size_t elements_per_channel,
    bool symmetric = false
) {
    PerChannelQuantParams<Scalar> params(num_channels);

    for (size_t c = 0; c < num_channels; ++c) {
        const float* channel_data = data + c * elements_per_channel;
        auto channel_params = compute_quant_params_from_data<Scalar>(
            channel_data, elements_per_channel, symmetric
        );
        params.scales[c] = channel_params.scale;
        params.zero_points[c] = channel_params.zero_point;
    }

    return params;
}

// =============================================================================
// Requantization (Output Scaling)
// =============================================================================

/**
 * @brief Requantization parameters for fused matmul output
 *
 * After matmul: acc = A @ B (in accumulator type)
 * Requantize: output = round(acc * output_scale) + output_zero_point
 *
 * output_scale = (scale_A * scale_B) / scale_output
 */
struct RequantParams {
    float output_scale;           // Combined scale factor
    std::int32_t output_zero_point;  // Output zero point

    RequantParams() : output_scale(1.0f), output_zero_point(0) {}

    RequantParams(float scale, std::int32_t zp)
        : output_scale(scale), output_zero_point(zp) {}

    /// Compute requant params from input/output quant params
    template<typename ScalarA, typename ScalarB, typename ScalarOut>
    static RequantParams compute(
        const QuantParams<ScalarA>& params_a,
        const QuantParams<ScalarB>& params_b,
        const QuantParams<ScalarOut>& params_out
    ) {
        RequantParams rq;
        rq.output_scale = (params_a.scale * params_b.scale) / params_out.scale;
        rq.output_zero_point = params_out.zero_point;
        return rq;
    }

    /// Apply requantization to accumulator value
    template<typename Accumulator, typename ScalarOut>
    ScalarOut apply(Accumulator acc) const {
        float result = static_cast<float>(acc) * output_scale +
                       static_cast<float>(output_zero_point);
        float min_val = static_cast<float>(dtype_min_value(ScalarTraits<ScalarOut>::dtype));
        float max_val = static_cast<float>(dtype_max_value(ScalarTraits<ScalarOut>::dtype));
        result = std::clamp(result, min_val, max_val);
        return static_cast<ScalarOut>(static_cast<std::int32_t>(std::round(result)));
    }
};

} // namespace sw::kpu
