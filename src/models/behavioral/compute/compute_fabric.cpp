// ============================================================================
// src/components/compute/behavioral_compute_fabric.cpp
// Behavioral (functional) compute fabric implementation
// ============================================================================

#include <sw/kpu/models/behavioral/compute/compute_fabric.hpp>
#include <sw/kpu/quantization/type_dispatch.hpp>
#include <sw/xue/event_collector.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace sw::kpu {

// Constants for activation functions
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_COEFF = 0.044715f;

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
    // ==========================================================================
    // TYPE-DISPATCHED MATMUL EXECUTION
    // ==========================================================================
    // The dispatch_matmul() function routes to the appropriate template kernel
    // based on desc.dtype. Supported types include:
    //   - FP32: Native float computation
    //   - FP16, BF16: Universal library 16-bit floats
    //   - FP8 variants: E4M3, E5M2, E3M4, E2M5 via Universal cfloat
    //   - FP4: Ultra-low precision via Universal cfloat<4,1>
    //   - INT8, INT4: Integer quantized with INT32 accumulator
    //
    // Type information flow to KPU resources:
    //   - DMA/BlockMover: Use desc.element_size() for byte calculations
    //   - Streamer: Use desc.dtype for pack/unpack of sub-byte types (INT4)
    //   - Compute: Template kernel selected by dtype
    // ==========================================================================
    dispatch_matmul(desc.dtype, desc.m, desc.n, desc.k,
                    a_data, b_data, c_data, desc.accumulate);

    uint64_t op_id = next_op_id_++;

    // Calculate MACs
    uint64_t macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    // Update statistics
    stats_.matmuls++;
    stats_.total_macs += macs;
    stats_.total_flops += macs * 2;  // mul + add per MAC

    // Estimate latency and schedule callback
    uint32_t latency = estimate_latency(desc);

    // Record XUE events (tile-level for hardware-accurate counters)
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_matmul(desc.m, desc.n, desc.k, latency);

    // Record memory events for roofline analysis
    // C[M,N] = A[M,K] @ B[K,N]
    uint64_t elem_size = dtype_size(desc.dtype);
    uint64_t a_bytes = static_cast<uint64_t>(desc.m) * desc.k * elem_size;
    uint64_t b_bytes = static_cast<uint64_t>(desc.k) * desc.n * elem_size;
    uint64_t c_bytes = static_cast<uint64_t>(desc.m) * desc.n * elem_size;
    sw::xue::xue().record_dram_read(a_bytes);
    sw::xue::xue().record_dram_read(b_bytes);
    sw::xue::xue().record_dram_write(c_bytes);

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

std::optional<uint64_t> BehavioralComputeFabric::submit_conv2d(
    const Conv2DDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    void* output_data,
    std::function<void()> callback)
{
    // Execute conv2d immediately (FP32 only for now, typed dispatch future work)
    if (desc.dtype == DataType::FLOAT32) {
        execute_conv2d_fp32(desc,
                           static_cast<const float*>(input_data),
                           static_cast<const float*>(weight_data),
                           static_cast<const float*>(bias_data),
                           static_cast<float*>(output_data));
    }

    uint64_t op_id = next_op_id_++;

    // Calculate MACs
    uint64_t macs = desc.compute_macs();

    // Update statistics
    stats_.conv2ds++;
    stats_.total_macs += macs;
    stats_.total_flops += macs * 2;

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record(sw::xue::EventType::OP_IM2COL,
                          sw::xue::EventMetadata::compute(macs * 2, 0, 0, 0));

    stats_.total_compute_cycles += latency;
    stats_.min_latency = std::min(stats_.min_latency, static_cast<uint64_t>(latency));
    stats_.max_latency = std::max(stats_.max_latency, static_cast<uint64_t>(latency));

    return op_id;
}

std::optional<uint64_t> BehavioralComputeFabric::submit_elementwise(
    const ElementwiseDescriptor& desc,
    const void* a_data,
    const void* b_data,
    void* output_data,
    std::function<void()> callback)
{
    // ==========================================================================
    // TYPE-DISPATCHED ELEMENTWISE EXECUTION
    // ==========================================================================
    // Common activations (ReLU, GELU, SiLU) and binary ops (ADD, MUL) use
    // type dispatch. Other operations fall back to FP32-only implementation.
    // ==========================================================================
    bool dispatched = false;
    switch (desc.op) {
        case ElementwiseOp::RELU:
            dispatch_relu(desc.dtype, a_data, output_data, desc.count);
            dispatched = true;
            break;
        case ElementwiseOp::RELU6:
            dispatch_relu6(desc.dtype, a_data, output_data, desc.count);
            dispatched = true;
            break;
        case ElementwiseOp::GELU:
            dispatch_gelu(desc.dtype, a_data, output_data, desc.count);
            dispatched = true;
            break;
        case ElementwiseOp::SILU:
            dispatch_silu(desc.dtype, a_data, output_data, desc.count);
            dispatched = true;
            break;
        case ElementwiseOp::ADD:
            dispatch_add(desc.dtype, a_data, b_data, output_data, desc.count);
            dispatched = true;
            break;
        case ElementwiseOp::MUL:
            dispatch_mul(desc.dtype, a_data, b_data, output_data, desc.count);
            dispatched = true;
            break;
        default:
            // Fall back to FP32 implementation for other ops
            if (desc.dtype == DataType::FLOAT32) {
                execute_elementwise_fp32(desc,
                                         static_cast<const float*>(a_data),
                                         static_cast<const float*>(b_data),
                                         static_cast<float*>(output_data));
                dispatched = true;
            }
            break;
    }
    (void)dispatched;  // Silence unused variable warning

    uint64_t op_id = next_op_id_++;

    // Update statistics
    stats_.elementwise_ops++;
    stats_.total_flops += desc.count;  // Approximate: 1 FLOP per element

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    stats_.total_compute_cycles += latency;

    // Record XUE event based on operation type
    sw::xue::xue().set_cycle(current_cycle_);
    switch (desc.op) {
        case ElementwiseOp::RELU:
            sw::xue::xue().record_relu(desc.count, latency);
            break;
        case ElementwiseOp::ADD:
        case ElementwiseOp::ADD_SCALAR:
            sw::xue::xue().record_add(desc.count, latency);
            break;
        case ElementwiseOp::MUL:
        case ElementwiseOp::MUL_SCALAR:
            sw::xue::xue().record_mul(desc.count, latency);
            break;
        case ElementwiseOp::SIGMOID:
            sw::xue::xue().record_sigmoid(desc.count, latency);
            break;
        case ElementwiseOp::TANH:
            sw::xue::xue().record_tanh(desc.count, latency);
            break;
        case ElementwiseOp::GELU:
            sw::xue::xue().record_gelu(desc.count, latency);
            break;
        default:
            sw::xue::xue().record_add(desc.count, latency);
            break;
    }

    // Record memory events for roofline analysis
    uint64_t elem_size = dtype_size(desc.dtype);
    uint64_t input_bytes = static_cast<uint64_t>(desc.count) * elem_size;
    bool is_binary = (desc.op == ElementwiseOp::ADD || desc.op == ElementwiseOp::ADD_SCALAR ||
                      desc.op == ElementwiseOp::MUL || desc.op == ElementwiseOp::MUL_SCALAR ||
                      desc.op == ElementwiseOp::SUB || desc.op == ElementwiseOp::DIV);
    if (is_binary && b_data != nullptr) {
        sw::xue::xue().record_dram_read(input_bytes);  // A
        sw::xue::xue().record_dram_read(input_bytes);  // B
    } else {
        sw::xue::xue().record_dram_read(input_bytes);  // A (unary ops)
    }
    sw::xue::xue().record_dram_write(input_bytes);     // Output

    return op_id;
}

std::optional<uint64_t> BehavioralComputeFabric::submit_pool2d(
    const Pool2DDescriptor& desc,
    const void* input_data,
    void* output_data,
    std::function<void()> callback)
{
    // Execute pool2d immediately (FP32 only for now)
    if (desc.dtype == DataType::FLOAT32) {
        execute_pool2d_fp32(desc,
                           static_cast<const float*>(input_data),
                           static_cast<float*>(output_data));
    }

    uint64_t op_id = next_op_id_++;

    // Update statistics
    stats_.pool2ds++;

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    stats_.total_compute_cycles += latency;

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

std::optional<uint64_t> BehavioralComputeFabric::submit_softmax(
    const SoftmaxDescriptor& desc,
    const void* input_data,
    void* output_data,
    std::function<void()> callback)
{
    // ==========================================================================
    // TYPE-DISPATCHED SOFTMAX EXECUTION
    // ==========================================================================
    dispatch_softmax(desc.dtype, input_data, output_data,
                     desc.batch_size, desc.dim_size);

    uint64_t op_id = next_op_id_++;

    // Update statistics
    stats_.softmaxes++;
    // Softmax: exp + sum + div per element ≈ 5 FLOPs per element
    stats_.total_flops += desc.total_elements() * 5;

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    stats_.total_compute_cycles += latency;

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_softmax(desc.total_elements(), latency);

    return op_id;
}

std::optional<uint64_t> BehavioralComputeFabric::submit_layernorm(
    const LayerNormDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    void* output_data,
    std::function<void()> callback)
{
    // ==========================================================================
    // TYPE-DISPATCHED LAYERNORM EXECUTION
    // ==========================================================================
    dispatch_layernorm(desc.dtype, input_data, output_data,
                       weight_data, bias_data,
                       desc.batch_size, desc.normalized_size, desc.eps);

    uint64_t op_id = next_op_id_++;

    // Update statistics
    stats_.layernorms++;
    // LayerNorm: mean + var + normalize ≈ 8 FLOPs per element
    uint64_t total_elements = static_cast<uint64_t>(desc.batch_size) * desc.normalized_size;
    stats_.total_flops += total_elements * 8;

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    stats_.total_compute_cycles += latency;

    // Record XUE event
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_layernorm(total_elements, latency);

    return op_id;
}

std::optional<uint64_t> BehavioralComputeFabric::submit_batchnorm(
    const BatchNormDescriptor& desc,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    const void* running_mean,
    const void* running_var,
    void* output_data,
    std::function<void()> callback)
{
    // Execute batchnorm immediately (FP32 only for now, typed dispatch future work)
    if (desc.dtype == DataType::FLOAT32) {
        execute_batchnorm_fp32(desc,
                              static_cast<const float*>(input_data),
                              static_cast<const float*>(weight_data),
                              static_cast<const float*>(bias_data),
                              static_cast<const float*>(running_mean),
                              static_cast<const float*>(running_var),
                              static_cast<float*>(output_data));
    }

    uint64_t op_id = next_op_id_++;

    // Update statistics
    stats_.batchnorms++;
    uint64_t total_elements = static_cast<uint64_t>(desc.batch_size) *
                              desc.num_features * desc.spatial_size;
    stats_.total_flops += total_elements * 4;  // sub, div, mul, add

    // Schedule callback
    uint32_t latency = estimate_latency(desc);
    schedule_callback(latency, callback);

    stats_.total_compute_cycles += latency;

    // Record XUE event (batchnorm as layernorm for now, similar operation structure)
    sw::xue::xue().set_cycle(current_cycle_);
    sw::xue::xue().record_layernorm(total_elements, latency);

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

// Note: execute_matmul is now handled by dispatch_matmul() in type_dispatch.hpp
// The typed kernels in quantization/kernels.hpp provide the actual computation.

uint32_t BehavioralComputeFabric::estimate_latency(const MatMulDescriptor& desc) const {
    // Estimate based on throughput
    uint64_t total_macs = static_cast<uint64_t>(desc.m) * desc.n * desc.k;

    if (config_.macs_per_cycle == 0) {
        return 1;  // Minimum latency
    }

    uint32_t cycles = static_cast<uint32_t>(total_macs / config_.macs_per_cycle);
    return std::max(cycles, 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const Conv2DDescriptor& desc) const {
    uint64_t total_macs = desc.compute_macs();
    if (config_.macs_per_cycle == 0) return 1;
    return std::max(static_cast<uint32_t>(total_macs / config_.macs_per_cycle), 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const ElementwiseDescriptor& desc) const {
    // Elementwise ops are typically memory-bound, estimate ~1 element per cycle
    return std::max(static_cast<uint32_t>(desc.count / 64), 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const Pool2DDescriptor& desc) const {
    uint64_t output_size = static_cast<uint64_t>(desc.batch_size) * desc.channels *
                           desc.out_height() * desc.out_width();
    return std::max(static_cast<uint32_t>(output_size / 64), 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const SoftmaxDescriptor& desc) const {
    return std::max(static_cast<uint32_t>(desc.total_elements() / 64), 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const LayerNormDescriptor& desc) const {
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.normalized_size;
    return std::max(static_cast<uint32_t>(total / 64), 1u);
}

uint32_t BehavioralComputeFabric::estimate_latency(const BatchNormDescriptor& desc) const {
    uint64_t total = static_cast<uint64_t>(desc.batch_size) * desc.num_features * desc.spatial_size;
    return std::max(static_cast<uint32_t>(total / 64), 1u);
}

void BehavioralComputeFabric::schedule_callback(uint32_t latency, std::function<void()> callback) {
    if (callback) {
        pending_callbacks_.push(PendingCallback{
            .completion_cycle = current_cycle_ + latency,
            .callback = callback
        });
    }
}

// ============================================================================
// Execute Functions - Actual Compute Implementations
// ============================================================================

void BehavioralComputeFabric::execute_conv2d_fp32(
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

    // Standard convolution with groups support
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

void BehavioralComputeFabric::execute_elementwise_fp32(
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
        case ElementwiseOp::RELU6:
            // ReLU6(x) = min(max(x, 0), 6) - MobileNetV2 activation.
            for (uint64_t i = 0; i < count; ++i)
                output[i] = std::min(std::max(0.0f, a[i]), 6.0f);
            break;
        case ElementwiseOp::GELU:
            // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            for (uint64_t i = 0; i < count; ++i) {
                float x = a[i];
                float x3 = x * x * x;
                float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
                output[i] = x * 0.5f * (1.0f + std::tanh(inner));
            }
            break;
        case ElementwiseOp::SILU:
            // SiLU(x) = x * sigmoid(x)
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

void BehavioralComputeFabric::execute_pool2d_fp32(
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
        // Adaptive average pooling
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
                    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
                        uint32_t h_start = (h_out * H_in) / H_out;
                        uint32_t h_end = ((h_out + 1) * H_in) / H_out;
                        uint32_t w_start = (w_out * W_in) / W_out;
                        uint32_t w_end = ((w_out + 1) * W_in) / W_out;

                        float sum = 0.0f;
                        uint32_t count = 0;
                        for (uint32_t h = h_start; h < h_end; ++h) {
                            for (uint32_t w = w_start; w < w_end; ++w) {
                                sum += input[((n * C + c) * H_in + h) * W_in + w];
                                count++;
                            }
                        }
                        output[((n * C + c) * H_out + h_out) * W_out + w_out] = sum / static_cast<float>(count);
                    }
                }
            }
        }
    } else {
        // Standard max/avg pooling
        const uint32_t K_h = desc.kernel_height;
        const uint32_t K_w = desc.kernel_width;
        const uint32_t stride_h = desc.stride_h > 0 ? desc.stride_h : K_h;
        const uint32_t stride_w = desc.stride_w > 0 ? desc.stride_w : K_w;
        const uint32_t pad_h = desc.padding_h;
        const uint32_t pad_w = desc.padding_w;

        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
                    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
                        float result;
                        if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
                            result = -std::numeric_limits<float>::infinity();
                        } else {
                            result = 0.0f;
                        }
                        uint32_t count = 0;

                        for (uint32_t kh = 0; kh < K_h; ++kh) {
                            for (uint32_t kw = 0; kw < K_w; ++kw) {
                                int32_t h_in = static_cast<int32_t>(h_out * stride_h + kh) - static_cast<int32_t>(pad_h);
                                int32_t w_in = static_cast<int32_t>(w_out * stride_w + kw) - static_cast<int32_t>(pad_w);

                                if (h_in >= 0 && h_in < static_cast<int32_t>(H_in) &&
                                    w_in >= 0 && w_in < static_cast<int32_t>(W_in)) {
                                    float val = input[((n * C + c) * H_in + h_in) * W_in + w_in];
                                    if (desc.pool_type == Pool2DDescriptor::PoolType::MAX) {
                                        result = std::max(result, val);
                                    } else {
                                        result += val;
                                    }
                                    count++;
                                }
                            }
                        }

                        if (desc.pool_type == Pool2DDescriptor::PoolType::AVG && count > 0) {
                            result /= static_cast<float>(count);
                        }
                        output[((n * C + c) * H_out + h_out) * W_out + w_out] = result;
                    }
                }
            }
        }
    }
}

void BehavioralComputeFabric::execute_softmax_fp32(
    const SoftmaxDescriptor& desc,
    const float* input, float* output)
{
    const uint32_t batch = desc.batch_size;
    const uint32_t dim = desc.dim_size;
    const uint32_t inner = desc.inner_size;

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < inner; ++i) {
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                max_val = std::max(max_val, input[idx]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                float exp_val = std::exp(input[idx] - max_val);
                output[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (uint32_t d = 0; d < dim; ++d) {
                uint64_t idx = (b * dim + d) * inner + i;
                output[idx] /= sum;
            }
        }
    }
}

void BehavioralComputeFabric::execute_layernorm_fp32(
    const LayerNormDescriptor& desc,
    const float* input, const float* weight, const float* bias, float* output)
{
    const uint32_t batch = desc.batch_size;
    const uint32_t norm_size = desc.normalized_size;
    const float eps = desc.eps;

    for (uint32_t b = 0; b < batch; ++b) {
        const float* x = input + b * norm_size;
        float* y = output + b * norm_size;

        // Compute mean
        float mean = 0.0f;
        for (uint32_t i = 0; i < norm_size; ++i) {
            mean += x[i];
        }
        mean /= static_cast<float>(norm_size);

        // Compute variance
        float var = 0.0f;
        for (uint32_t i = 0; i < norm_size; ++i) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(norm_size);

        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (uint32_t i = 0; i < norm_size; ++i) {
            float normalized = (x[i] - mean) * inv_std;
            if (weight != nullptr && desc.has_weight) {
                normalized *= weight[i];
            }
            if (bias != nullptr && desc.has_bias) {
                normalized += bias[i];
            }
            y[i] = normalized;
        }
    }
}

void BehavioralComputeFabric::execute_batchnorm_fp32(
    const BatchNormDescriptor& desc,
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float* output)
{
    const uint32_t N = desc.batch_size;
    const uint32_t C = desc.num_features;
    const uint32_t spatial = desc.spatial_size;
    const float eps = desc.eps;

    // Process each channel
    for (uint32_t c = 0; c < C; ++c) {
        float mean, var;

        if (running_mean != nullptr && running_var != nullptr) {
            // Use provided running statistics (inference mode)
            mean = running_mean[c];
            var = running_var[c];
        } else {
            // Compute batch statistics (training mode or missing stats)
            mean = 0.0f;
            uint64_t count = 0;
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t s = 0; s < spatial; ++s) {
                    uint64_t idx = (n * C + c) * spatial + s;
                    mean += input[idx];
                    count++;
                }
            }
            // Guard against count == 0 (N * spatial == 0). MSVC C4723 fires
            // otherwise; in practice valid BatchNorm input has count > 0.
            if (count > 0) {
                mean /= static_cast<float>(count);
            }

            var = 0.0f;
            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t s = 0; s < spatial; ++s) {
                    uint64_t idx = (n * C + c) * spatial + s;
                    float diff = input[idx] - mean;
                    var += diff * diff;
                }
            }
            if (count > 0) {
                var /= static_cast<float>(count);
            }
        }

        float inv_std = 1.0f / std::sqrt(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;

        // Apply normalization
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t s = 0; s < spatial; ++s) {
                uint64_t idx = (n * C + c) * spatial + s;
                output[idx] = (input[idx] - mean) * inv_std * gamma + beta;
            }
        }
    }
}

} // namespace sw::kpu
