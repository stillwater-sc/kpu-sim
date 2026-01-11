// ============================================================================
// include/sw/kpu/models/behavioral/compute/vector_engine.hpp
// Behavioral (functional) Vector Engine - bias addition and activation
//
// The Vector Engine applies elementwise operations to matmul results:
// - Bias addition: output[i,j] += bias[j] (broadcast across rows)
// - Activation functions: output = activation(output)
//
// In behavioral mode, these operations are instant.
// ============================================================================

#pragma once

#include <sw/kpu/behavioral/memory_model.hpp>
#include <sw/kpu/components/sfu.hpp>
#include <cstdint>
#include <vector>

namespace sw::kpu::behavioral {

/// Behavioral Vector Engine for bias and activation
///
/// This is a functional simulation that performs actual computation:
/// - Bias addition (row-broadcast)
/// - Activation functions via SFU
///
/// Used for:
/// - Functional verification of MLP layers
/// - Executing complete neural network inference
class BehavioralVectorEngine {
public:
    /// Statistics tracking
    struct Stats {
        uint64_t activations_applied = 0;
        uint64_t bias_additions = 0;
        uint64_t elements_processed = 0;

        void reset() {
            activations_applied = 0;
            bias_additions = 0;
            elements_processed = 0;
        }
    };

    // ========================================================================
    // Constructors
    // ========================================================================

    BehavioralVectorEngine();
    ~BehavioralVectorEngine() = default;

    // Non-copyable, movable
    BehavioralVectorEngine(const BehavioralVectorEngine&) = delete;
    BehavioralVectorEngine& operator=(const BehavioralVectorEngine&) = delete;
    BehavioralVectorEngine(BehavioralVectorEngine&&) = default;
    BehavioralVectorEngine& operator=(BehavioralVectorEngine&&) = default;

    // ========================================================================
    // In-Place Operations on Memory Model
    // ========================================================================

    /// Apply bias and activation to a 2D tile in memory
    ///
    /// @param memory       Memory model containing the data
    /// @param data_addr    Address of tile data [height, width]
    /// @param height       Number of rows (batch dimension)
    /// @param width        Number of columns (output dimension)
    /// @param bias_addr    Address of bias vector [width], or 0 for no bias
    /// @param activation   Activation function to apply
    ///
    /// Operation: data[i,j] = activation(data[i,j] + bias[j])
    void apply_bias_activation(
        BehavioralMemoryModel* memory,
        Address data_addr,
        uint32_t height,
        uint32_t width,
        Address bias_addr,
        ActivationType activation);

    /// Apply only activation (no bias) to a 2D tile in memory
    void apply_activation(
        BehavioralMemoryModel* memory,
        Address data_addr,
        uint32_t height,
        uint32_t width,
        ActivationType activation);

    /// Apply only bias (no activation) to a 2D tile in memory
    void apply_bias(
        BehavioralMemoryModel* memory,
        Address data_addr,
        uint32_t height,
        uint32_t width,
        Address bias_addr);

    // ========================================================================
    // Direct Memory Operations (for testing)
    // ========================================================================

    /// Apply bias and activation to raw float arrays
    ///
    /// @param data         Data array [height * width], modified in place
    /// @param height       Number of rows
    /// @param width        Number of columns
    /// @param bias         Bias vector [width], or nullptr for no bias
    /// @param activation   Activation function
    void process_inplace(
        float* data,
        uint32_t height,
        uint32_t width,
        const float* bias,
        ActivationType activation);

    /// Apply activation only to a 1D vector
    void apply_activation_1d(
        float* data,
        uint32_t count,
        ActivationType activation);

    // ========================================================================
    // Statistics
    // ========================================================================

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }

    // ========================================================================
    // SFU Access
    // ========================================================================

    /// Get the underlying SFU (for configuration or testing)
    SFU& sfu() { return sfu_; }
    const SFU& sfu() const { return sfu_; }

private:
    SFU sfu_;
    Stats stats_;
};

} // namespace sw::kpu::behavioral
