// ============================================================================
// src/models/behavioral/compute/vector_engine.cpp
// Behavioral Vector Engine implementation
// ============================================================================

#include <sw/kpu/models/behavioral/compute/vector_engine.hpp>
#include <stdexcept>

namespace sw::kpu::behavioral {

BehavioralVectorEngine::BehavioralVectorEngine()
    : sfu_()
{
    // Default SFU configuration
    sfu_.configure(ActivationType::NONE);
}

void BehavioralVectorEngine::apply_bias_activation(
    BehavioralMemoryModel* memory,
    Address data_addr,
    uint32_t height,
    uint32_t width,
    Address bias_addr,
    ActivationType activation)
{
    if (!memory) {
        throw std::invalid_argument("BehavioralVectorEngine: memory is null");
    }

    // Get data pointer
    float* data = memory->get_ptr<float>(data_addr);

    // Get bias pointer (may be null if bias_addr == 0)
    const float* bias = nullptr;
    if (bias_addr != 0) {
        bias = memory->get_ptr<float>(bias_addr);
    }

    // Process
    process_inplace(data, height, width, bias, activation);
}

void BehavioralVectorEngine::apply_activation(
    BehavioralMemoryModel* memory,
    Address data_addr,
    uint32_t height,
    uint32_t width,
    ActivationType activation)
{
    apply_bias_activation(memory, data_addr, height, width, 0, activation);
}

void BehavioralVectorEngine::apply_bias(
    BehavioralMemoryModel* memory,
    Address data_addr,
    uint32_t height,
    uint32_t width,
    Address bias_addr)
{
    apply_bias_activation(memory, data_addr, height, width, bias_addr, ActivationType::NONE);
}

void BehavioralVectorEngine::process_inplace(
    float* data,
    uint32_t height,
    uint32_t width,
    const float* bias,
    ActivationType activation)
{
    if (!data) {
        throw std::invalid_argument("BehavioralVectorEngine: data is null");
    }

    // Configure SFU for the activation
    if (activation != ActivationType::NONE) {
        sfu_.configure(activation);
    }

    // Process row by row
    for (uint32_t row = 0; row < height; ++row) {
        float* row_data = data + static_cast<size_t>(row) * width;

        // Apply bias (broadcast along row)
        if (bias != nullptr) {
            for (uint32_t col = 0; col < width; ++col) {
                row_data[col] += bias[col];
            }
            stats_.bias_additions++;
        }

        // Apply activation
        if (activation != ActivationType::NONE) {
            sfu_.evaluate_inplace(row_data, width);
            stats_.activations_applied++;
        }
    }

    stats_.elements_processed += static_cast<uint64_t>(height) * width;
}

void BehavioralVectorEngine::apply_activation_1d(
    float* data,
    uint32_t count,
    ActivationType activation)
{
    if (!data) {
        throw std::invalid_argument("BehavioralVectorEngine: data is null");
    }

    if (activation == ActivationType::NONE) {
        return;  // Nothing to do
    }

    sfu_.configure(activation);
    sfu_.evaluate_inplace(data, count);

    stats_.activations_applied++;
    stats_.elements_processed += count;
}

} // namespace sw::kpu::behavioral
