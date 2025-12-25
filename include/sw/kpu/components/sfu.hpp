#pragma once
// Special Function Unit (SFU) for activation functions
// Uses LUT + linear interpolation for fast, deterministic evaluation

#include <sw/concepts.hpp>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::kpu {

/**
 * @brief Activation function types supported by the SFU
 */
enum class ActivationType : uint8_t {
    NONE = 0,           ///< Pass-through (no activation)
    RELU = 1,           ///< max(0, x)
    GELU = 2,           ///< x * 0.5 * (1 + erf(x/sqrt(2)))
    SIGMOID = 3,        ///< 1 / (1 + exp(-x))
    TANH = 4,           ///< (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    SILU = 5,           ///< x * sigmoid(x), aka Swish
    SOFTPLUS = 6,       ///< ln(1 + exp(x))
    LEAKY_RELU = 7,     ///< max(alpha*x, x), alpha typically 0.01
};

/**
 * @brief Get string name for activation type
 */
inline const char* activation_type_name(ActivationType type) {
    switch (type) {
        case ActivationType::NONE: return "none";
        case ActivationType::RELU: return "relu";
        case ActivationType::GELU: return "gelu";
        case ActivationType::SIGMOID: return "sigmoid";
        case ActivationType::TANH: return "tanh";
        case ActivationType::SILU: return "silu";
        case ActivationType::SOFTPLUS: return "softplus";
        case ActivationType::LEAKY_RELU: return "leaky_relu";
        default: return "unknown";
    }
}

/**
 * @brief SFU Configuration
 */
struct SFUConfig {
    ActivationType activation = ActivationType::NONE;
    Size table_size = 256;              ///< LUT entries (256-1024 typical)
    float input_range_min = -8.0f;      ///< Input domain minimum
    float input_range_max = 8.0f;       ///< Input domain maximum
    float leaky_alpha = 0.01f;          ///< For LEAKY_RELU
    Size pipeline_depth = 2;            ///< 2 cycles: lookup + interpolate
};

/**
 * @brief Special Function Unit - LUT + Linear Interpolation
 *
 * Implements transcendental activation functions using piecewise linear
 * approximation with configurable table size. This approach provides:
 * - Deterministic latency (2 cycles)
 * - High throughput (one result per cycle after pipeline fills)
 * - Configurable accuracy/area tradeoff via table size
 *
 * Pipeline stages:
 *   Cycle 1: Address calculation + LUT lookup (two adjacent entries)
 *   Cycle 2: Linear interpolation + output
 *
 * For RELU, the LUT is bypassed entirely (combinatorial logic).
 *
 * Accuracy characteristics (256-entry LUT, [-8, 8] range):
 *   - RELU: Exact (no LUT needed)
 *   - SIGMOID: < 0.1% max error
 *   - TANH: < 0.1% max error
 *   - GELU: < 0.5% max error (more complex shape)
 *   - SILU: < 0.3% max error
 */
class KPU_API SFU {
public:
    // =========================================
    // Constructors
    // =========================================

    /**
     * @brief Default constructor with NONE activation
     */
    SFU();

    /**
     * @brief Construct with specific configuration
     */
    explicit SFU(const SFUConfig& config);

    ~SFU() = default;

    // =========================================
    // Configuration
    // =========================================

    /**
     * @brief Configure the SFU for a specific activation
     * @param activation Activation type
     * @param table_size Number of LUT entries (default 256)
     *
     * Rebuilds the LUT for the new activation function.
     */
    void configure(ActivationType activation, Size table_size = 256);

    /**
     * @brief Set input range for LUT coverage
     * @param min_val Minimum input value
     * @param max_val Maximum input value
     *
     * Rebuilds the LUT with new range. Default [-8, 8] covers
     * typical neural network activation ranges.
     */
    void set_input_range(float min_val, float max_val);

    /**
     * @brief Set leaky ReLU alpha parameter
     */
    void set_leaky_alpha(float alpha) { config_.leaky_alpha = alpha; }

    /**
     * @brief Get current configuration
     */
    const SFUConfig& config() const { return config_; }

    /**
     * @brief Get current activation type
     */
    ActivationType activation() const { return config_.activation; }

    // =========================================
    // Evaluation
    // =========================================

    /**
     * @brief Evaluate activation for a single element
     * @param x Input value
     * @return Activated value
     *
     * Uses LUT + linear interpolation for transcendental functions.
     * For testing and validation; vectorized version is more efficient.
     */
    float evaluate(float x) const;

    /**
     * @brief Evaluate activation for a vector of elements
     * @param input Input array
     * @param output Output array (must be pre-allocated)
     * @param count Number of elements
     *
     * Processes elements in pipeline fashion. Caller should account
     * for pipeline_depth cycles of latency for first result.
     */
    void evaluate_vector(const float* input, float* output, Size count) const;

    /**
     * @brief In-place activation (input == output allowed)
     */
    void evaluate_inplace(float* data, Size count) const;

    // =========================================
    // Timing
    // =========================================

    /**
     * @brief Get pipeline latency in cycles
     */
    Size get_latency_cycles() const { return config_.pipeline_depth; }

    /**
     * @brief Get throughput in elements per cycle
     * @return 1 (fully pipelined)
     */
    Size get_throughput() const { return 1; }

    // =========================================
    // LUT Access (for debugging/analysis)
    // =========================================

    /**
     * @brief Get the lookup table
     */
    const std::vector<float>& get_lut() const { return lut_; }

    /**
     * @brief Get table size
     */
    Size get_table_size() const { return config_.table_size; }

    /**
     * @brief Get input range
     */
    std::pair<float, float> get_input_range() const {
        return {config_.input_range_min, config_.input_range_max};
    }

    // =========================================
    // Reference Implementations (for validation)
    // =========================================

    /**
     * @brief Reference RELU implementation
     */
    static float reference_relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }

    /**
     * @brief Reference Leaky RELU implementation
     */
    static float reference_leaky_relu(float x, float alpha = 0.01f) {
        return x > 0.0f ? x : alpha * x;
    }

    /**
     * @brief Reference Sigmoid implementation
     */
    static float reference_sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief Reference Tanh implementation
     */
    static float reference_tanh(float x) {
        return std::tanh(x);
    }

    /**
     * @brief Reference GELU implementation
     * GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
     */
    static float reference_gelu(float x) {
        constexpr float sqrt2_inv = 0.7071067811865476f;  // 1/sqrt(2)
        return x * 0.5f * (1.0f + std::erf(x * sqrt2_inv));
    }

    /**
     * @brief Reference SILU (Swish) implementation
     * SILU(x) = x * sigmoid(x)
     */
    static float reference_silu(float x) {
        return x * reference_sigmoid(x);
    }

    /**
     * @brief Reference Softplus implementation
     * Softplus(x) = ln(1 + exp(x))
     */
    static float reference_softplus(float x) {
        // Numerically stable version
        if (x > 20.0f) return x;  // Avoid overflow
        return std::log1p(std::exp(x));
    }

    /**
     * @brief Get reference implementation for any activation type
     */
    static float reference_evaluate(ActivationType type, float x, float alpha = 0.01f);

private:
    SFUConfig config_;
    std::vector<float> lut_;    ///< Lookup table
    float scale_;               ///< (table_size - 1) / (max - min)
    float inv_scale_;           ///< 1 / scale for interpolation delta

    /**
     * @brief Build lookup table for current activation
     */
    void build_lut();

    /**
     * @brief Lookup with linear interpolation
     */
    float lookup_interpolate(float x) const;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
