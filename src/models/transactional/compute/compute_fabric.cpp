// ============================================================================
// src/components/compute/transactional_compute_fabric.cpp
// Transactional compute fabric implementation
// ============================================================================

#include <sw/kpu/models/transactional/compute/compute_fabric.hpp>
#include <sw/xue/event_collector.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace sw::kpu {

// Constants for activation functions
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
constexpr float GELU_COEFF = 0.044715f;

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
    // Skip computation if data pointers are null (timing-only submission)
    if (desc.dtype == DataType::FLOAT32 && a_data && b_data && c_data) {
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

    // Record XUE event (tile-level for hardware-accurate counters)
    // Skip XUE when called with null data pointers (timing-only submission
    // for fused ops where behavioral fabric already recorded XUE events)
    if (a_data) {
        sw::xue::xue().set_cycle(current_cycle_);
        sw::xue::xue().record_matmul(desc.m, desc.n, desc.k, duration);
    }

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_conv2d(
    const Conv2DDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    void* output_data,
    std::function<void()> callback)
{
    // Execute conv2d immediately
    if (desc.dtype == DataType::FLOAT32) {
        execute_conv2d_fp32(desc,
                           static_cast<const float*>(input_data),
                           static_cast<const float*>(weight_data),
                           static_cast<const float*>(bias_data),
                           static_cast<float*>(output_data));
    }

    // Schedule timing
    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    // Update statistics
    uint64_t macs = desc.compute_macs();
    stats_.conv2ds++;
    stats_.total_macs += macs;
    stats_.total_flops += macs * 2;
    stats_.total_compute_cycles += duration;
    stats_.min_latency = std::min(stats_.min_latency, static_cast<uint64_t>(duration));
    stats_.max_latency = std::max(stats_.max_latency, static_cast<uint64_t>(duration));

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record(sw::xue::EventType::OP_IM2COL,
                          sw::xue::EventMetadata::compute(macs * 2, 0, 0, 0));

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_elementwise(
    const ElementwiseDescriptor& desc,
    const void* a_data,
    const void* b_data,
    void* output_data,
    std::function<void()> callback)
{
    if (desc.dtype == DataType::FLOAT32) {
        execute_elementwise_fp32(desc,
                                 static_cast<const float*>(a_data),
                                 static_cast<const float*>(b_data),
                                 static_cast<float*>(output_data));
    }

    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    stats_.elementwise_ops++;
    stats_.total_flops += desc.count;
    stats_.total_compute_cycles += duration;

    // Record XUE event based on operation type
    sw::xue::xue().set_cycle(current_cycle_);
    switch (desc.op) {
        case ElementwiseOp::RELU:
            sw::xue::xue().record_relu(desc.count, duration);
            break;
        case ElementwiseOp::ADD:
        case ElementwiseOp::ADD_SCALAR:
            sw::xue::xue().record_add(desc.count, duration);
            break;
        case ElementwiseOp::MUL:
        case ElementwiseOp::MUL_SCALAR:
            sw::xue::xue().record_mul(desc.count, duration);
            break;
        case ElementwiseOp::SIGMOID:
            sw::xue::xue().record_sigmoid(desc.count, duration);
            break;
        case ElementwiseOp::TANH:
            sw::xue::xue().record_tanh(desc.count, duration);
            break;
        case ElementwiseOp::GELU:
            sw::xue::xue().record_gelu(desc.count, duration);
            break;
        default:
            sw::xue::xue().record_add(desc.count, duration);
            break;
    }

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_pool2d(
    const Pool2DDescriptor& desc,
    const void* input_data,
    void* output_data,
    std::function<void()> callback)
{
    if (desc.dtype == DataType::FLOAT32) {
        execute_pool2d_fp32(desc,
                           static_cast<const float*>(input_data),
                           static_cast<float*>(output_data));
    }

    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    stats_.pool2ds++;
    stats_.total_compute_cycles += duration;

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    uint64_t output_elements = static_cast<uint64_t>(desc.batch_size) * desc.channels *
                                desc.out_height() * desc.out_width();
    if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
        sw::xue::xue().record(sw::xue::EventType::OP_POOL_MAX,
                              sw::xue::EventMetadata::compute(output_elements, 0, 0, 0));
    } else {
        sw::xue::xue().record(sw::xue::EventType::OP_POOL_AVG,
                              sw::xue::EventMetadata::compute(output_elements, 0, 0, 0));
    }

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_softmax(
    const SoftmaxDescriptor& desc,
    const void* input_data,
    void* output_data,
    std::function<void()> callback)
{
    if (desc.dtype == DataType::FLOAT32) {
        execute_softmax_fp32(desc,
                            static_cast<const float*>(input_data),
                            static_cast<float*>(output_data));
    }

    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    stats_.softmaxes++;
    stats_.total_flops += desc.total_elements() * 5;
    stats_.total_compute_cycles += duration;

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_softmax(desc.total_elements(), duration);

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_layernorm(
    const LayerNormDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    void* output_data,
    std::function<void()> callback)
{
    if (desc.dtype == DataType::FLOAT32) {
        execute_layernorm_fp32(desc,
                              static_cast<const float*>(input_data),
                              static_cast<const float*>(weight_data),
                              static_cast<const float*>(bias_data),
                              static_cast<float*>(output_data));
    }

    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    stats_.layernorms++;
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.normalized_size;
    stats_.total_flops += total * 8;
    stats_.total_compute_cycles += duration;

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_layernorm(total, duration);

    return op_id;
}

std::optional<uint64_t> TransactionalComputeFabric::submit_batchnorm(
    const BatchNormDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    const void* running_mean,
    const void* running_var,
    void* output_data,
    std::function<void()> callback)
{
    if (desc.dtype == DataType::FLOAT32) {
        execute_batchnorm_fp32(desc,
                              static_cast<const float*>(input_data),
                              static_cast<const float*>(weight_data),
                              static_cast<const float*>(bias_data),
                              static_cast<const float*>(running_mean),
                              static_cast<const float*>(running_var),
                              static_cast<float*>(output_data));
    }

    uint32_t duration = compute_duration(desc);
    uint64_t op_id = schedule_operation(duration, callback);

    stats_.batchnorms++;
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.num_features * desc.spatial_size;
    stats_.total_flops += total * 4;
    stats_.total_compute_cycles += duration;

    // Record XUE event (batchnorm as layernorm for now, similar operation structure)
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_layernorm(total, duration);

    return op_id;
}

uint64_t TransactionalComputeFabric::schedule_operation(uint32_t duration, std::function<void()> callback) {
    uint64_t op_id = next_op_id_++;

    uint64_t start_cycle = std::max(current_cycle_, state_end_cycle_);
    uint64_t completion_cycle = start_cycle + duration;

    if (callback) {
        PendingOp op{
            .op_id = op_id,
            .submit_cycle = current_cycle_,
            .completion_cycle = completion_cycle,
            .m = 0, .n = 0, .k = 0,
            .callback = callback
        };
        completion_queue_.push(op);
        pending_count_++;
    }

    state_end_cycle_ = completion_cycle;
    if (current_state_ == PipelineState::IDLE) {
        current_state_ = PipelineState::COMPUTING;
    }

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

    // Update utilization stats BEFORE advancing state machine
    // This ensures the final compute cycle is counted as busy, not idle
    if (current_state_ != PipelineState::IDLE) {
        stats_.busy_cycles++;
    } else {
        stats_.idle_cycles++;
    }

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
}

void TransactionalComputeFabric::drain() {
    // Process pending operations with callbacks
    while (has_pending()) {
        tick();
    }

    // Also tick through remaining scheduled cycles for timing stats
    // This ensures busy/idle cycles are tracked even without callbacks
    while (current_cycle_ < state_end_cycle_) {
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
    return static_cast<uint8_t>(std::min(uint64_t{100}, (elapsed * 100) / total));
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

// ============================================================================
// Duration Computation for Various Operations
// ============================================================================

uint32_t TransactionalComputeFabric::compute_duration(const Conv2DDescriptor& desc) const {
    uint64_t total_macs = desc.compute_macs();
    if (config_.macs_per_cycle == 0) return 1;
    uint32_t compute_cycles = static_cast<uint32_t>(total_macs / config_.macs_per_cycle);
    uint32_t pipeline_overhead = config_.pipeline_depth * 2;
    return std::max(compute_cycles + pipeline_overhead, 1u);
}

uint32_t TransactionalComputeFabric::compute_duration(const ElementwiseDescriptor& desc) const {
    // Elementwise ops are memory-bound, ~64 elements per cycle
    return std::max(static_cast<uint32_t>(desc.count / 64), 1u);
}

uint32_t TransactionalComputeFabric::compute_duration(const Pool2DDescriptor& desc) const {
    uint64_t output_size = static_cast<uint64_t>(desc.batch_size) * desc.channels *
                           desc.out_height() * desc.out_width();
    return std::max(static_cast<uint32_t>(output_size / 64), 1u);
}

uint32_t TransactionalComputeFabric::compute_duration(const SoftmaxDescriptor& desc) const {
    return std::max(static_cast<uint32_t>(desc.total_elements() / 64), 1u);
}

uint32_t TransactionalComputeFabric::compute_duration(const LayerNormDescriptor& desc) const {
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.normalized_size;
    return std::max(static_cast<uint32_t>(total / 64), 1u);
}

uint32_t TransactionalComputeFabric::compute_duration(const BatchNormDescriptor& desc) const {
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.num_features * desc.spatial_size;
    return std::max(static_cast<uint32_t>(total / 64), 1u);
}

// ============================================================================
// Execute Functions - Same as Behavioral for Functional Correctness
// ============================================================================

void TransactionalComputeFabric::execute_conv2d_fp32(
    const Conv2DDescriptor& desc,
    const float* input, const float* weight, const float* bias, float* output)
{
    const uint32_t N = desc.batch_size;
    const uint32_t C_in = desc.in_channels;
    const uint32_t H_in = desc.in_height;
    const uint32_t W_in = desc.in_width;
    const uint32_t C_out = desc.out_channels;
    const uint32_t K_h = desc.kernel_height;
    const uint32_t K_w = desc.kernel_width;
    const uint32_t H_out = desc.out_height();
    const uint32_t W_out = desc.out_width();
    const uint32_t stride_h = desc.stride_h;
    const uint32_t stride_w = desc.stride_w;
    const uint32_t pad_h = desc.padding_h;
    const uint32_t pad_w = desc.padding_w;
    const uint32_t dilation_h = desc.dilation_h;
    const uint32_t dilation_w = desc.dilation_w;
    const uint32_t groups = desc.groups;
    const uint32_t C_in_per_group = C_in / groups;
    const uint32_t C_out_per_group = C_out / groups;

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < groups; ++g) {
            for (uint32_t c_out = 0; c_out < C_out_per_group; ++c_out) {
                uint32_t oc = g * C_out_per_group + c_out;
                for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
                    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
                        float sum = 0.0f;
                        for (uint32_t c_in = 0; c_in < C_in_per_group; ++c_in) {
                            uint32_t ic = g * C_in_per_group + c_in;
                            for (uint32_t kh = 0; kh < K_h; ++kh) {
                                for (uint32_t kw = 0; kw < K_w; ++kw) {
                                    int32_t h_in = static_cast<int32_t>(h_out * stride_h + kh * dilation_h) - static_cast<int32_t>(pad_h);
                                    int32_t w_in = static_cast<int32_t>(w_out * stride_w + kw * dilation_w) - static_cast<int32_t>(pad_w);

                                    if (h_in >= 0 && h_in < static_cast<int32_t>(H_in) &&
                                        w_in >= 0 && w_in < static_cast<int32_t>(W_in)) {
                                        uint64_t in_idx = ((n * C_in + ic) * H_in + h_in) * W_in + w_in;
                                        uint64_t w_idx = ((oc * C_in_per_group + c_in) * K_h + kh) * K_w + kw;
                                        sum += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                        if (bias != nullptr) {
                            sum += bias[oc];
                        }
                        uint64_t out_idx = ((n * C_out + oc) * H_out + h_out) * W_out + w_out;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
}

void TransactionalComputeFabric::execute_elementwise_fp32(
    const ElementwiseDescriptor& desc,
    const float* a, const float* b, float* output)
{
    const uint64_t count = desc.count;

    switch (desc.op) {
        case ElementwiseOp::ADD:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] + b[i];
            break;
        case ElementwiseOp::SUB:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] - b[i];
            break;
        case ElementwiseOp::MUL:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] * b[i];
            break;
        case ElementwiseOp::DIV:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] / b[i];
            break;
        case ElementwiseOp::MAX:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::max(a[i], b[i]);
            break;
        case ElementwiseOp::MIN:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::min(a[i], b[i]);
            break;
        case ElementwiseOp::NEG:
            for (uint64_t i = 0; i < count; ++i) output[i] = -a[i];
            break;
        case ElementwiseOp::ABS:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::abs(a[i]);
            break;
        case ElementwiseOp::SQRT:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::sqrt(a[i]);
            break;
        case ElementwiseOp::EXP:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::exp(a[i]);
            break;
        case ElementwiseOp::LOG:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::log(a[i]);
            break;
        case ElementwiseOp::TANH:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::tanh(a[i]);
            break;
        case ElementwiseOp::SIGMOID:
            for (uint64_t i = 0; i < count; ++i) output[i] = 1.0f / (1.0f + std::exp(-a[i]));
            break;
        case ElementwiseOp::RELU:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::max(0.0f, a[i]);
            break;
        case ElementwiseOp::GELU:
            for (uint64_t i = 0; i < count; ++i) {
                float x = a[i];
                float x3 = x * x * x;
                float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
                output[i] = x * 0.5f * (1.0f + std::tanh(inner));
            }
            break;
        case ElementwiseOp::SILU:
            for (uint64_t i = 0; i < count; ++i) {
                float x = a[i];
                output[i] = x / (1.0f + std::exp(-x));
            }
            break;
        case ElementwiseOp::ADD_SCALAR:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] + b[0];
            break;
        case ElementwiseOp::MUL_SCALAR:
            for (uint64_t i = 0; i < count; ++i) output[i] = a[i] * b[0];
            break;
        case ElementwiseOp::POW_SCALAR:
            for (uint64_t i = 0; i < count; ++i) output[i] = std::pow(a[i], b[0]);
            break;
    }
}

void TransactionalComputeFabric::execute_pool2d_fp32(
    const Pool2DDescriptor& desc,
    const float* input, float* output)
{
    const uint32_t N = desc.batch_size;
    const uint32_t C = desc.channels;
    const uint32_t H_in = desc.in_height;
    const uint32_t W_in = desc.in_width;
    const uint32_t H_out = desc.out_height();
    const uint32_t W_out = desc.out_width();

    if (desc.pool_type == Pool2DDescriptor::PoolType::ADAPTIVE_AVG) {
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
                    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
                        uint32_t h_start = (h_out * H_in) / H_out;
                        uint32_t h_end = ((h_out + 1) * H_in) / H_out;
                        uint32_t w_start = (w_out * W_in) / W_out;
                        uint32_t w_end = ((w_out + 1) * W_in) / W_out;

                        float sum = 0.0f;
                        uint32_t cnt = 0;
                        for (uint32_t h = h_start; h < h_end; ++h) {
                            for (uint32_t w = w_start; w < w_end; ++w) {
                                sum += input[((n * C + c) * H_in + h) * W_in + w];
                                cnt++;
                            }
                        }
                        output[((n * C + c) * H_out + h_out) * W_out + w_out] = sum / static_cast<float>(cnt);
                    }
                }
            }
        }
    } else {
        const uint32_t K_h = desc.kernel_height;
        const uint32_t K_w = desc.kernel_width;
        const uint32_t stride_h = desc.stride_h > 0 ? desc.stride_h : K_h;
        const uint32_t stride_w = desc.stride_w > 0 ? desc.stride_w : K_w;
        const uint32_t pad_h = desc.padding_h;
        const uint32_t pad_w = desc.padding_w;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t ho = 0; ho < H_out; ++ho) {
                    for (uint32_t wo = 0; wo < W_out; ++wo) {
                        float result = (desc.pool_type == Pool2DDescriptor::PoolType::MAX)
                            ? -std::numeric_limits<float>::infinity() : 0.0f;
                        uint32_t cnt = 0;

                        for (uint32_t kh = 0; kh < K_h; ++kh) {
                            for (uint32_t kw = 0; kw < K_w; ++kw) {
                                int32_t hi = static_cast<int32_t>(ho * stride_h + kh) - static_cast<int32_t>(pad_h);
                                int32_t wi = static_cast<int32_t>(wo * stride_w + kw) - static_cast<int32_t>(pad_w);

                                if (hi >= 0 && hi < static_cast<int32_t>(H_in) &&
                                    wi >= 0 && wi < static_cast<int32_t>(W_in)) {
                                    float val = input[((n * C + c) * H_in + hi) * W_in + wi];
                                    if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
                                        result = std::max(result, val);
                                    } else {
                                        result += val;
                                    }
                                    cnt++;
                                }
                            }
                        }
                        if (desc.pool_type == Pool2DDescriptor::PoolType::AVG && cnt > 0) {
                            result /= static_cast<float>(cnt);
                        }
                        output[((n * C + c) * H_out + ho) * W_out + wo] = result;
                    }
                }
            }
        }
    }
}

void TransactionalComputeFabric::execute_softmax_fp32(
    const SoftmaxDescriptor& desc,
    const float* input, float* output)
{
    const uint32_t batch = desc.batch_size;
    const uint32_t dim = desc.dim_size;
    const uint32_t inner = desc.inner_size;

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < inner; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                max_val = std::max(max_val, input[idx]);
            }

            float sum = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                float exp_val = std::exp(input[idx] - max_val);
                output[idx] = exp_val;
                sum += exp_val;
            }

            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                output[idx] /= sum;
            }
        }
    }
}

void TransactionalComputeFabric::execute_layernorm_fp32(
    const LayerNormDescriptor& desc,
    const float* input, const float* weight, const float* bias, float* output)
{
    const uint32_t batch = desc.batch_size;
    const uint32_t norm_size = desc.normalized_size;
    const float eps = desc.eps;

    for (uint32_t b = 0; b < batch; ++b) {
        const float* x = input + b * norm_size;
        float* y = output + b * norm_size;

        float mean = 0.0f;
        for (uint32_t i = 0; i < norm_size; ++i) mean += x[i];
        mean /= static_cast<float>(norm_size);

        float var = 0.0f;
        for (uint32_t i = 0; i < norm_size; ++i) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(norm_size);

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (uint32_t i = 0; i < norm_size; ++i) {
            float normalized = (x[i] - mean) * inv_std;
            if (weight && desc.has_weight) normalized *= weight[i];
            if (bias && desc.has_bias) normalized += bias[i];
            y[i] = normalized;
        }
    }
}

void TransactionalComputeFabric::execute_batchnorm_fp32(
    const BatchNormDescriptor& desc,
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float* output)
{
    const uint32_t N = desc.batch_size;
    const uint32_t C = desc.num_features;
    const uint32_t spatial = desc.spatial_size;
    const float eps = desc.eps;

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            float mean = running_mean[c];
            float var = running_var[c];
            float inv_std = 1.0f / std::sqrt(var + eps);
            float gamma = weight ? weight[c] : 1.0f;
            float beta = bias ? bias[c] : 0.0f;

            for (uint32_t s = 0; s < spatial; ++s) {
                uint64_t idx = (n * C + c) * spatial + s;
                output[idx] = (input[idx] - mean) * inv_std * gamma + beta;
            }
        }
    }
}

} // namespace sw::kpu
