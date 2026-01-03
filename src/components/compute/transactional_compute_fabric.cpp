// ============================================================================
// src/components/compute/transactional_compute_fabric.cpp
// Transactional compute fabric implementation
// ============================================================================

#include <sw/kpu/components/compute/transactional_compute_fabric.hpp>

#include <algorithm>

namespace sw::kpu {

// ============================================================================
// Constructor / Reset
// ============================================================================

TransactionalComputeFabric::TransactionalComputeFabric(
    const ComputeFabricConfig& config, uint32_t tile_id)
    : config_(config)
    , tile_id_(tile_id)
{
}

void TransactionalComputeFabric::reset() {
    current_cycle_ = 0;
    next_op_id_ = 1;
    pending_count_ = 0;
    current_state_ = PipelineState::IDLE;
    state_end_cycle_ = 0;
    has_current_op_ = false;

    while (!completion_queue_.empty()) {
        completion_queue_.pop();
    }

    stats_.reset();
}

// ============================================================================
// Compute Operations
// ============================================================================

std::optional<uint64_t> TransactionalComputeFabric::submit_matmul(
    const MatMulDescriptor& desc,
    const void* a_data,
    const void* b_data,
    void* c_data,
    std::function<void()> callback)
{
    // Execute matmul immediately (data is computed now, timing is simulated)
    if (desc.element_size == 4) {
        execute_matmul_fp32(desc,
                           static_cast<const float*>(a_data),
                           static_cast<const float*>(b_data),
                           static_cast<float*>(c_data));
    }

    uint64_t op_id = next_op_id_++;

    // Calculate duration based on throughput
    uint32_t duration = compute_duration(desc);

    // Calculate when this can start (after current work)
    uint64_t start_cycle = std::max(current_cycle_, state_end_cycle_);
    uint64_t completion_cycle = start_cycle + duration;

    // Queue the operation
    PendingOp op{
        .op_id = op_id,
        .submit_cycle = current_cycle_,
        .completion_cycle = completion_cycle,
        .m = desc.m,
        .n = desc.n,
        .k = desc.k,
        .callback = callback
    };

    if (callback) {
        completion_queue_.push(op);
        pending_count_++;
    }

    // Update state tracking
    state_end_cycle_ = completion_cycle;
    if (current_state_ == PipelineState::IDLE) {
        current_state_ = PipelineState::COMPUTING;
        has_current_op_ = true;
        current_op_ = op;
    }

    // Calculate MACs
    uint64_t macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    // Update statistics
    stats_.matmuls++;
    stats_.total_macs += macs;
    stats_.total_flops += macs * 2;
    stats_.total_compute_cycles += duration;
    stats_.min_latency = std::min(stats_.min_latency, static_cast<uint64_t>(duration));
    stats_.max_latency = std::max(stats_.max_latency, static_cast<uint64_t>(duration));

    return op_id;
}

bool TransactionalComputeFabric::can_accept() const {
    // Always accept in transactional mode (queue is unlimited)
    return true;
}

// ============================================================================
// Simulation Interface
// ============================================================================

void TransactionalComputeFabric::tick() {
    current_cycle_++;

    // Process completed operations
    while (!completion_queue_.empty() &&
           completion_queue_.top().completion_cycle <= current_cycle_) {

        PendingOp op = completion_queue_.top();
        completion_queue_.pop();
        pending_count_--;

        if (op.callback) {
            op.callback();
        }
    }

    // Advance state machine
    advance_state();

    // Update utilization stats
    if (current_state_ != PipelineState::IDLE) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }
}

void TransactionalComputeFabric::drain() {
    while (has_pending()) {
        tick();
    }
}

uint8_t TransactionalComputeFabric::pipeline_progress() const {
    if (current_state_ == PipelineState::IDLE) {
        return 100;
    }

    if (!has_current_op_ || state_end_cycle_ <= current_cycle_) {
        return 100;
    }

    uint64_t total = state_end_cycle_ - current_op_.submit_cycle;
    uint64_t elapsed = current_cycle_ - current_op_.submit_cycle;

    if (total == 0) return 100;
    return static_cast<uint8_t>(std::min(100UL, (elapsed * 100) / total));
}

// ============================================================================
// Helpers
// ============================================================================

void TransactionalComputeFabric::execute_matmul_fp32(
    const MatMulDescriptor& desc,
    const float* a, const float* b, float* c)
{
    const uint32_t m = desc.m;
    const uint32_t n = desc.n;
    const uint32_t k = desc.k;

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

uint32_t TransactionalComputeFabric::compute_duration(const MatMulDescriptor& desc) const {
    uint64_t total_macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    if (config_.macs_per_cycle == 0) {
        return 1;
    }

    // Account for pipeline depth (startup + steady state + drain)
    uint32_t compute_cycles = static_cast<uint32_t>(total_macs / config_.macs_per_cycle);
    uint32_t pipeline_overhead = config_.pipeline_depth * 2;  // Start + drain

    return std::max(compute_cycles + pipeline_overhead, 1u);
}

void TransactionalComputeFabric::advance_state() {
    if (current_state_ == PipelineState::IDLE) {
        return;
    }

    // Check if current operation is complete
    if (current_cycle_ >= state_end_cycle_) {
        current_state_ = PipelineState::IDLE;
        has_current_op_ = false;
    }
}

} // namespace sw::kpu
