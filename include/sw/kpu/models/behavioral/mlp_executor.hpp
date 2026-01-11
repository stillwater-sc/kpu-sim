// ============================================================================
// include/sw/kpu/models/behavioral/mlp_executor.hpp
// Behavioral MLP Executor - multi-layer perceptron execution
//
// The MLP Executor manages complete neural network inference:
// - Loads network weights and biases into memory
// - Executes forward pass through all layers
// - Tracks per-layer and total statistics
//
// In behavioral mode, all operations are instant (no timing simulation).
// ============================================================================

#pragma once

#include <sw/kpu/models/behavioral/orchestrator.hpp>
#include <sw/kpu/components/sfu.hpp>
#include <cstdint>
#include <vector>
#include <string>

namespace sw::kpu::behavioral {

/// Behavioral MLP executor for multi-layer perceptron inference
///
/// This executor provides a high-level interface for running MLP networks:
/// - Define network architecture with layers
/// - Load pre-trained weights
/// - Execute forward pass
/// - Retrieve results
///
/// Usage example:
///   BehavioralMLPExecutor executor;
///   executor.add_layer(784, 256, ActivationType::RELU, true);
///   executor.add_layer(256, 10, ActivationType::NONE, true);
///   executor.load_weights({...}, {...});
///   executor.forward(input, output);
class BehavioralMLPExecutor {
public:
    /// Layer definition
    struct Layer {
        uint32_t input_dim;         ///< Input dimension
        uint32_t output_dim;        ///< Output dimension
        ActivationType activation;  ///< Activation function
        bool use_bias;              ///< Whether layer has bias
        std::string name;           ///< Optional layer name
    };

    /// Network definition (list of layers)
    struct Network {
        std::vector<Layer> layers;
        uint32_t batch_size = 1;    ///< Default batch size
    };

    /// Statistics for execution
    struct Stats {
        uint64_t total_flops = 0;
        uint64_t total_bytes_transferred = 0;
        std::vector<uint64_t> layer_flops;
        uint32_t forward_passes = 0;

        void reset() {
            total_flops = total_bytes_transferred = 0;
            layer_flops.clear();
            forward_passes = 0;
        }
    };

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Default constructor with default orchestrator config
    BehavioralMLPExecutor();

    /// Constructor with custom orchestrator config
    explicit BehavioralMLPExecutor(const BehavioralOrchestrator::Config& config);

    ~BehavioralMLPExecutor() = default;

    // Non-copyable, movable
    BehavioralMLPExecutor(const BehavioralMLPExecutor&) = delete;
    BehavioralMLPExecutor& operator=(const BehavioralMLPExecutor&) = delete;
    BehavioralMLPExecutor(BehavioralMLPExecutor&&) = default;
    BehavioralMLPExecutor& operator=(BehavioralMLPExecutor&&) = default;

    // ========================================================================
    // Network Definition
    // ========================================================================

    /// Add a layer to the network
    void add_layer(
        uint32_t input_dim,
        uint32_t output_dim,
        ActivationType activation = ActivationType::NONE,
        bool use_bias = true,
        const std::string& name = "");

    /// Clear all layers
    void clear_layers();

    /// Get network definition
    const Network& network() const { return network_; }

    /// Set batch size for inference
    void set_batch_size(uint32_t batch_size) { network_.batch_size = batch_size; }

    // ========================================================================
    // Weight Loading
    // ========================================================================

    /// Load weights for all layers from separate vectors
    ///
    /// @param weights Vector of weight matrices, one per layer
    ///               weights[i] has size input_dim[i] * output_dim[i]
    /// @param biases  Vector of bias vectors, one per layer (may have empty vectors for no bias)
    ///               biases[i] has size output_dim[i] or 0
    void load_weights(
        const std::vector<std::vector<float>>& weights,
        const std::vector<std::vector<float>>& biases);

    /// Load weights for a single layer
    void load_layer_weights(
        size_t layer_idx,
        const std::vector<float>& weights,
        const std::vector<float>& bias = {});

    /// Check if weights are loaded
    bool weights_loaded() const { return weights_loaded_; }

    // ========================================================================
    // Forward Pass
    // ========================================================================

    /// Execute forward pass with std::vector input/output
    ///
    /// @param input  Input data [batch_size, input_dim_of_first_layer]
    /// @param output Output data [batch_size, output_dim_of_last_layer]
    void forward(const std::vector<float>& input, std::vector<float>& output);

    /// Execute forward pass with raw pointers (data already in memory)
    ///
    /// @param input_addr  Host memory address of input
    /// @param output_addr Host memory address of output
    void forward(Address input_addr, Address output_addr);

    // ========================================================================
    // Component Access
    // ========================================================================

    /// Get the orchestrator
    BehavioralOrchestrator& orchestrator() { return orchestrator_; }
    const BehavioralOrchestrator& orchestrator() const { return orchestrator_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }

    /// Compute FLOPs for the network (single forward pass)
    uint64_t compute_network_flops() const;

private:
    BehavioralOrchestrator orchestrator_;
    Network network_;
    Stats stats_;

    // Weight storage in host memory
    bool weights_loaded_ = false;
    std::vector<Address> weight_addrs_;     // Host addresses for weights
    std::vector<Address> bias_addrs_;       // Host addresses for biases (0 = no bias)
    std::vector<Address> activation_addrs_; // Host addresses for intermediate activations

    // Memory allocations (stored for cleanup)
    std::vector<Allocation> allocations_;

    /// Allocate memory for weights, biases, and activations
    void allocate_buffers();

    /// Free all allocated buffers
    void free_buffers();
};

} // namespace sw::kpu::behavioral
