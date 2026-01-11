// ============================================================================
// src/components/compute/behavioral_compute_fabric.cpp
// Behavioral (functional) compute fabric implementation
// ============================================================================

#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>

#include <algorithm>
#include <cstring>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

BehavioralComputeFabric::BehavioralComputeFabric(
    const ComputeFabricConfig& config, uint32_t tile_id)
    : config_(config)
    , tile_id_(tile_id)
{
}

void BehavioralComputeFabric::reset() {
    current_cycle_ = 0;
    next_op_id_ = 1;

    // Clear pending callbacks
    while (!pending_callbacks_.empty()) {
        pending_callbacks_.pop();
    }

    stats_.reset();
}

// ============================================================================
// Compute Operations
// ============================================================================

std::optional<uint64_t> BehavioralComputeFabric::submit_matmul(
    const MatMulDescriptor& desc,
    const void* a_data,
    const void* b_data,
    void* c_data,
    std::function<void()> callback)
{
    // Execute matmul immediately
    if (desc.element_size == 4) {
        execute_matmul_fp32(desc,
                           static_cast<const float*>(a_data),
                           static_cast<const float*>(b_data),
                           static_cast<float*>(c_data));
    }
    // TODO: Support other data types (INT8, FP16, BF16)

    uint64_t op_id = next_op_id_++;

    // Calculate MACs
    uint64_t macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    // Update statistics
    stats_.matmuls++;
    stats_.total_macs += macs;
    stats_.total_flops += macs * 2;  // mul + add per MAC

    // Estimate latency and schedule callback
    uint32_t latency = estimate_latency(desc);

    if (callback) {
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
    }

    stats_.total_compute_cycles += latency;
    stats_.min_latency = std::min(stats_.min_latency, static_cast<uint64_t>(latency));
    stats_.max_latency = std::max(stats_.max_latency, static_cast<uint64_t>(latency));

    return op_id;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void BehavioralComputeFabric::tick() {
    current_cycle_++;

    // Process completed callbacks
    while (!pending_callbacks_.empty() &&
           pending_callbacks_.top().completion_cycle <= current_cycle_) {

        PendingCallback cb = pending_callbacks_.top();
        pending_callbacks_.pop();

        if (cb.callback) {
            cb.callback();
        }
    }

    // Update utilization stats
    if (!pending_callbacks_.empty()) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void BehavioralComputeFabric::drain() {
    while (has_pending()) {
        tick();
    }
}

// ============================================================================
// Helpers
// ============================================================================

void BehavioralComputeFabric::execute_matmul_fp32(
    const MatMulDescriptor& desc,
    const float* a, const float* b, float* c)
{
    const uint32_t m = desc.m;
    const uint32_t n = desc.n;
    const uint32_t k = desc.k;

    // Simple triple-loop matmul
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float sum = desc.accumulate ? c[i * n + j] : 0.0f;
            for (uint32_t p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

uint32_t BehavioralComputeFabric::estimate_latency(const MatMulDescriptor& desc) const {
    // Estimate based on throughput
    uint64_t total_macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    if (config_.macs_per_cycle == 0) {
        return 1;  // Minimum latency
    }

    uint32_t cycles = static_cast<uint32_t>(total_macs / config_.macs_per_cycle);
    return std::max(cycles, 1u);
}

} // namespace sw::kpu
