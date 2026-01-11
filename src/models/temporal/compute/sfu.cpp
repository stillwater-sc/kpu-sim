// Special Function Unit (SFU) Implementation
// LUT-based activation functions with linear interpolation

#include <sw/kpu/components/sfu.hpp>
#include <stdexcept>
#include <cmath>

namespace sw::kpu {

// ============================================================================
// Constructors
// ============================================================================

SFU::SFU() : SFU(SFUConfig{}) {}

SFU::SFU(const SFUConfig& config) : config_(config) {
    build_lut();
}

// ============================================================================
// Configuration
// ============================================================================

void SFU::configure(ActivationType activation, Size table_size) {
    config_.activation = activation;
    config_.table_size = table_size;
    build_lut();
}

void SFU::set_input_range(float min_val, float max_val) {
    if (min_val >= max_val) {
        throw std::invalid_argument("SFU: input_range_min must be less than input_range_max");
    }
    config_.input_range_min = min_val;
    config_.input_range_max = max_val;
    build_lut();
}

// ============================================================================
// LUT Building
// ============================================================================

void SFU::build_lut() {
    if (config_.table_size < 2) {
        throw std::invalid_argument("SFU: table_size must be at least 2");
    }

    // Calculate scaling factors
    float range = config_.input_range_max - config_.input_range_min;
    scale_ = static_cast<float>(config_.table_size - 1) / range;
    inv_scale_ = range / static_cast<float>(config_.table_size - 1);

    // Resize and populate LUT
    lut_.resize(config_.table_size);

    for (Size i = 0; i < config_.table_size; ++i) {
        float x = config_.input_range_min + static_cast<float>(i) * inv_scale_;
        lut_[i] = reference_evaluate(config_.activation, x, config_.leaky_alpha);
    }
}

// ============================================================================
// Evaluation
// ============================================================================

float SFU::evaluate(float x) const {
    // RELU and LEAKY_RELU are simple enough to compute directly
    // (no LUT needed, deterministic single-cycle)
    if (config_.activation == ActivationType::NONE) {
        return x;
    }
    if (config_.activation == ActivationType::RELU) {
        return x > 0.0f ? x : 0.0f;
    }
    if (config_.activation == ActivationType::LEAKY_RELU) {
        return x > 0.0f ? x : config_.leaky_alpha * x;
    }

    // Use LUT + interpolation for transcendental functions
    return lookup_interpolate(x);
}

void SFU::evaluate_vector(const float* input, float* output, Size count) const {
    for (Size i = 0; i < count; ++i) {
        output[i] = evaluate(input[i]);
    }
}

void SFU::evaluate_inplace(float* data, Size count) const {
    for (Size i = 0; i < count; ++i) {
        data[i] = evaluate(data[i]);
    }
}

// ============================================================================
// LUT Lookup with Linear Interpolation
// ============================================================================

float SFU::lookup_interpolate(float x) const {
    // Clamp input to LUT range
    if (x <= config_.input_range_min) {
        return lut_.front();
    }
    if (x >= config_.input_range_max) {
        return lut_.back();
    }

    // Calculate fractional index
    float normalized = (x - config_.input_range_min) * scale_;
    Size idx = static_cast<Size>(normalized);
    float frac = normalized - static_cast<float>(idx);

    // Ensure we don't go out of bounds
    if (idx >= config_.table_size - 1) {
        return lut_.back();
    }

    // Linear interpolation: y = y0 + frac * (y1 - y0)
    float y0 = lut_[idx];
    float y1 = lut_[idx + 1];
    return y0 + frac * (y1 - y0);
}

// ============================================================================
// Reference Implementations (for validation and LUT generation)
// ============================================================================

float SFU::reference_evaluate(ActivationType type, float x, float alpha) {
    switch (type) {
        case ActivationType::NONE:
            return x;

        case ActivationType::RELU:
            return reference_relu(x);

        case ActivationType::LEAKY_RELU:
            return reference_leaky_relu(x, alpha);

        case ActivationType::SIGMOID:
            return reference_sigmoid(x);

        case ActivationType::TANH:
            return reference_tanh(x);

        case ActivationType::GELU:
            return reference_gelu(x);

        case ActivationType::SILU:
            return reference_silu(x);

        case ActivationType::SOFTPLUS:
            return reference_softplus(x);

        default:
            return x;
    }
}

} // namespace sw::kpu
