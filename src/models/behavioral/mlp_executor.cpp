// ============================================================================
// src/models/behavioral/mlp_executor.cpp
// Behavioral MLP Executor implementation
// ============================================================================

#include <sw/kpu/behavioral/mlp_executor.hpp>
#include <stdexcept>
#include <algorithm>

namespace sw::kpu::behavioral {

BehavioralMLPExecutor::BehavioralMLPExecutor()
    : orchestrator_()
{
}

BehavioralMLPExecutor::BehavioralMLPExecutor(const BehavioralOrchestrator::Config& config)
    : orchestrator_(config)
{
}

void BehavioralMLPExecutor::add_layer(
    uint32_t input_dim,
    uint32_t output_dim,
    ActivationType activation,
    bool use_bias,
    const std::string& name)
{
    // Validate dimensions
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument("BehavioralMLPExecutor: layer dimensions must be > 0");
    }

    // If not first layer, check dimension compatibility
    if (!network_.layers.empty()) {
        const auto& prev_layer = network_.layers.back();
        if (prev_layer.output_dim != input_dim) {
            throw std::invalid_argument(
                "BehavioralMLPExecutor: layer input_dim (" + std::to_string(input_dim) +
                ") must match previous layer output_dim (" +
                std::to_string(prev_layer.output_dim) + ")");
        }
    }

    Layer layer;
    layer.input_dim = input_dim;
    layer.output_dim = output_dim;
    layer.activation = activation;
    layer.use_bias = use_bias;
    layer.name = name.empty() ? ("layer_" + std::to_string(network_.layers.size())) : name;

    network_.layers.push_back(layer);

    // Invalidate loaded weights
    weights_loaded_ = false;
}

void BehavioralMLPExecutor::clear_layers() {
    network_.layers.clear();
    free_buffers();
    weights_loaded_ = false;
}

void BehavioralMLPExecutor::allocate_buffers() {
    free_buffers();

    auto& memory = orchestrator_.memory();

    weight_addrs_.clear();
    bias_addrs_.clear();
    activation_addrs_.clear();
    allocations_.clear();

    // Allocate weight and bias buffers for each layer
    for (size_t i = 0; i < network_.layers.size(); ++i) {
        const auto& layer = network_.layers[i];

        // Weights: [input_dim, output_dim]
        Size weight_bytes = static_cast<Size>(layer.input_dim) * layer.output_dim * sizeof(float);
        auto weight_alloc = memory.allocate_host(weight_bytes, layer.name + "_weights");
        weight_addrs_.push_back(weight_alloc.base_address);
        allocations_.push_back(weight_alloc);

        // Bias: [output_dim] or 0 if no bias
        if (layer.use_bias) {
            Size bias_bytes = static_cast<Size>(layer.output_dim) * sizeof(float);
            auto bias_alloc = memory.allocate_host(bias_bytes, layer.name + "_bias");
            bias_addrs_.push_back(bias_alloc.base_address);
            allocations_.push_back(bias_alloc);
        } else {
            bias_addrs_.push_back(0);
        }
    }

    // Allocate intermediate activation buffers (one per layer, except output)
    // We need buffers for layer outputs that become inputs to next layer
    for (size_t i = 0; i < network_.layers.size() - 1; ++i) {
        const auto& layer = network_.layers[i];
        Size activation_bytes = static_cast<Size>(network_.batch_size) * layer.output_dim * sizeof(float);
        auto act_alloc = memory.allocate_host(activation_bytes, layer.name + "_output");
        activation_addrs_.push_back(act_alloc.base_address);
        allocations_.push_back(act_alloc);
    }
}

void BehavioralMLPExecutor::free_buffers() {
    auto& memory = orchestrator_.memory();
    for (const auto& alloc : allocations_) {
        memory.free(alloc);
    }
    allocations_.clear();
    weight_addrs_.clear();
    bias_addrs_.clear();
    activation_addrs_.clear();
}

void BehavioralMLPExecutor::load_weights(
    const std::vector<std::vector<float>>& weights,
    const std::vector<std::vector<float>>& biases)
{
    if (network_.layers.empty()) {
        throw std::runtime_error("BehavioralMLPExecutor: no layers defined");
    }

    if (weights.size() != network_.layers.size()) {
        throw std::invalid_argument(
            "BehavioralMLPExecutor: weights vector size (" + std::to_string(weights.size()) +
            ") must match number of layers (" + std::to_string(network_.layers.size()) + ")");
    }

    if (biases.size() != network_.layers.size()) {
        throw std::invalid_argument(
            "BehavioralMLPExecutor: biases vector size (" + std::to_string(biases.size()) +
            ") must match number of layers (" + std::to_string(network_.layers.size()) + ")");
    }

    // Allocate buffers
    allocate_buffers();

    auto& memory = orchestrator_.memory();

    // Load weights and biases for each layer
    for (size_t i = 0; i < network_.layers.size(); ++i) {
        const auto& layer = network_.layers[i];

        // Validate weight size
        Size expected_weights = static_cast<Size>(layer.input_dim) * layer.output_dim;
        if (weights[i].size() != expected_weights) {
            throw std::invalid_argument(
                "BehavioralMLPExecutor: layer " + std::to_string(i) +
                " weight size (" + std::to_string(weights[i].size()) +
                ") must be " + std::to_string(expected_weights));
        }

        // Write weights
        memory.write_floats(weight_addrs_[i], weights[i]);

        // Write bias if present
        if (layer.use_bias) {
            if (biases[i].size() != layer.output_dim) {
                throw std::invalid_argument(
                    "BehavioralMLPExecutor: layer " + std::to_string(i) +
                    " bias size (" + std::to_string(biases[i].size()) +
                    ") must be " + std::to_string(layer.output_dim));
            }
            memory.write_floats(bias_addrs_[i], biases[i]);
        }
    }

    weights_loaded_ = true;
}

void BehavioralMLPExecutor::load_layer_weights(
    size_t layer_idx,
    const std::vector<float>& weights,
    const std::vector<float>& bias)
{
    if (layer_idx >= network_.layers.size()) {
        throw std::out_of_range("BehavioralMLPExecutor: layer index out of range");
    }

    // Ensure buffers are allocated
    if (weight_addrs_.empty()) {
        allocate_buffers();
    }

    const auto& layer = network_.layers[layer_idx];
    auto& memory = orchestrator_.memory();

    // Validate and write weights
    Size expected_weights = static_cast<Size>(layer.input_dim) * layer.output_dim;
    if (weights.size() != expected_weights) {
        throw std::invalid_argument(
            "BehavioralMLPExecutor: weight size mismatch for layer " + std::to_string(layer_idx));
    }
    memory.write_floats(weight_addrs_[layer_idx], weights);

    // Write bias if provided and layer uses bias
    if (layer.use_bias && !bias.empty()) {
        if (bias.size() != layer.output_dim) {
            throw std::invalid_argument(
                "BehavioralMLPExecutor: bias size mismatch for layer " + std::to_string(layer_idx));
        }
        memory.write_floats(bias_addrs_[layer_idx], bias);
    }

    // Mark weights as loaded if all layers have been loaded
    // (This is a simple check - in production you'd track per-layer)
    weights_loaded_ = true;
}

void BehavioralMLPExecutor::forward(const std::vector<float>& input, std::vector<float>& output) {
    if (network_.layers.empty()) {
        throw std::runtime_error("BehavioralMLPExecutor: no layers defined");
    }
    if (!weights_loaded_) {
        throw std::runtime_error("BehavioralMLPExecutor: weights not loaded");
    }

    const auto& first_layer = network_.layers.front();
    const auto& last_layer = network_.layers.back();

    // Validate input size
    Size expected_input_size = static_cast<Size>(network_.batch_size) * first_layer.input_dim;
    if (input.size() != expected_input_size) {
        throw std::invalid_argument(
            "BehavioralMLPExecutor: input size (" + std::to_string(input.size()) +
            ") must be batch_size * input_dim (" + std::to_string(expected_input_size) + ")");
    }

    // Allocate and write input
    auto& memory = orchestrator_.memory();
    Size input_bytes = input.size() * sizeof(float);
    auto input_alloc = memory.allocate_host(input_bytes, "forward_input");
    memory.write_floats(input_alloc.base_address, input);

    // Allocate output buffer
    Size output_size = static_cast<Size>(network_.batch_size) * last_layer.output_dim;
    Size output_bytes = output_size * sizeof(float);
    auto output_alloc = memory.allocate_host(output_bytes, "forward_output");

    // Execute forward pass
    forward(input_alloc.base_address, output_alloc.base_address);

    // Read output
    output = memory.read_floats(output_alloc.base_address, output_size);

    // Free temporary allocations
    memory.free(input_alloc);
    memory.free(output_alloc);
}

void BehavioralMLPExecutor::forward(Address input_addr, Address output_addr) {
    if (network_.layers.empty()) {
        throw std::runtime_error("BehavioralMLPExecutor: no layers defined");
    }
    if (!weights_loaded_) {
        throw std::runtime_error("BehavioralMLPExecutor: weights not loaded");
    }

    stats_.layer_flops.clear();

    Address current_input = input_addr;

    for (size_t i = 0; i < network_.layers.size(); ++i) {
        const auto& layer = network_.layers[i];

        // Determine output address
        Address current_output;
        if (i == network_.layers.size() - 1) {
            // Last layer outputs to final output
            current_output = output_addr;
        } else {
            // Intermediate layers output to activation buffers
            current_output = activation_addrs_[i];
        }

        // Execute layer: output = activation(input @ weights + bias)
        orchestrator_.execute_mlp_layer(
            current_input, network_.batch_size, layer.input_dim,
            weight_addrs_[i], layer.output_dim,
            bias_addrs_[i],  // 0 if no bias
            current_output,
            layer.activation);

        // Track layer FLOPs: 2*batch*in*out for matmul
        uint64_t layer_flops = 2ULL * network_.batch_size * layer.input_dim * layer.output_dim;
        stats_.layer_flops.push_back(layer_flops);
        stats_.total_flops += layer_flops;

        // Output becomes input for next layer
        current_input = current_output;
    }

    stats_.forward_passes++;

    // Update total bytes from orchestrator stats
    const auto& orch_stats = orchestrator_.stats();
    stats_.total_bytes_transferred = orch_stats.host_to_l3_bytes + orch_stats.l3_to_host_bytes +
                                     orch_stats.l3_to_l2_bytes + orch_stats.l2_to_l3_bytes;
}

uint64_t BehavioralMLPExecutor::compute_network_flops() const {
    uint64_t total = 0;
    for (const auto& layer : network_.layers) {
        // FLOPs = 2 * batch * input_dim * output_dim (matmul)
        // Plus bias add if present: batch * output_dim
        // Plus activation: batch * output_dim (approximate)
        total += 2ULL * network_.batch_size * layer.input_dim * layer.output_dim;
        if (layer.use_bias) {
            total += static_cast<uint64_t>(network_.batch_size) * layer.output_dim;
        }
        if (layer.activation != ActivationType::NONE) {
            total += static_cast<uint64_t>(network_.batch_size) * layer.output_dim;
        }
    }
    return total;
}

} // namespace sw::kpu::behavioral
