// ============================================================================
// include/sw/kpu/models/behavioral/compute/compute_fabric.hpp
// Behavioral (functional) compute fabric - instant computation
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================
//
// TYPE DISPATCH ARCHITECTURE:
// ───────────────────────────
// The BehavioralComputeFabric uses a type dispatch mechanism to route compute
// operations to the appropriate typed kernel instantiation. The flow is:
//
// 1. User specifies dtype in descriptor (e.g., MatMulDescriptor.dtype = FP16)
// 2. submit_matmul() calls dispatch_matmul() from type_dispatch.hpp
// 3. dispatch_matmul() switches on dtype and calls execute_typed_matmul<T, Acc>()
// 4. execute_typed_matmul uses the template kernels from quantization/kernels.hpp
//
// This design separates concerns:
// - Interface (this file): Operation submission and statistics
// - Dispatch (type_dispatch.hpp): Runtime-to-compile-time type mapping
// - Kernels (kernels.hpp): Type-parameterized computation
//
// For type propagation to KPU resources, see the documentation in
// compute_fabric_interface.hpp (MatMulDescriptor section).
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

#include <sw/kpu/models/interfaces/compute_fabric_interface.hpp>
// Note: type_dispatch.hpp is included in the .cpp file to avoid propagating
// Universal library dependencies to all code that includes this header.

namespace sw::kpu {

/// Behavioral compute fabric with instant computation
///
/// This is the fastest simulation mode, providing:
/// - Functional correctness (correct computation results)
/// - Optional fixed latency modeling
/// - No pipeline stages, no data movement delays
///
/// Use for:
/// - Software bring-up and functional verification
/// - Unit testing
/// - CI/CD pipelines where speed matters more than timing accuracy
class BehavioralComputeFabric : public IComputeFabric {
public:
    explicit BehavioralComputeFabric(const ComputeFabricConfig& config, uint32_t tile_id);
    ~BehavioralComputeFabric() override = default;

    // Disable copying, allow moving
    BehavioralComputeFabric(const BehavioralComputeFabric&) = delete;
    BehavioralComputeFabric& operator=(const BehavioralComputeFabric&) = delete;
    BehavioralComputeFabric(BehavioralComputeFabric&&) = default;
    BehavioralComputeFabric& operator=(BehavioralComputeFabric&&) = default;

    // ========================================================================
    // Compute Operations
    // ========================================================================

    std::optional<uint64_t> submit_matmul(
        const MatMulDescriptor& desc,
        const void* a_data,
        const void* b_data,
        void* c_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_conv2d(
        const Conv2DDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_elementwise(
        const ElementwiseDescriptor& desc,
        const void* a_data,
        const void* b_data,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_pool2d(
        const Pool2DDescriptor& desc,
        const void* input_data,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_softmax(
        const SoftmaxDescriptor& desc,
        const void* input_data,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_layernorm(
        const LayerNormDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_batchnorm(
        const BatchNormDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        const void* running_mean,
        const void* running_var,
        void* output_data,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override { return true; }  // Always accepts
    bool has_pending() const override { return !pending_callbacks_.empty(); }
    size_t pending_count() const override { return pending_callbacks_.size(); }
    bool is_busy() const override { return !pending_callbacks_.empty(); }

    // ========================================================================
    // Simulation Interface
    // ========================================================================

    void tick() override;
    void drain() override;
    void reset() override;

    uint64_t current_cycle() const override { return current_cycle_; }
    void set_cycle(uint64_t cycle) override { current_cycle_ = cycle; }

    // ========================================================================
    // Configuration Queries
    // ========================================================================

    SimulationFidelity fidelity() const override { return SimulationFidelity::BEHAVIORAL; }
    ComputeTechnology technology() const override { return config_.technology; }
    const ComputeFabricConfig& config() const override { return config_; }
    uint32_t tile_id() const override { return tile_id_; }
    uint32_t array_rows() const override { return config_.array_rows; }
    uint32_t array_cols() const override { return config_.array_cols; }
    uint32_t peak_macs_per_cycle() const override { return config_.macs_per_cycle; }

    // ========================================================================
    // Pipeline State
    // ========================================================================

    PipelineState get_pipeline_state() const override {
        return PipelineState::IDLE;  // Always idle in behavioral
    }

    uint8_t pipeline_progress() const override { return 100; }

    // ========================================================================
    // Statistics
    // ========================================================================

    const ComputeFabricStatistics& stats() const override { return stats_; }
    void reset_stats() override { stats_.reset(); }

    // ========================================================================
    // Observability
    // ========================================================================

    void enable_tracing(bool enable) override { tracing_enabled_ = enable; }
    bool tracing_enabled() const override { return tracing_enabled_; }
    void set_resource_tracker(sw::trace::ResourceTracker* tracker) override {
        resource_tracker_ = tracker;
    }

private:
    // Configuration
    ComputeFabricConfig config_;
    uint32_t tile_id_;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_op_id_ = 1;

    // Pending callbacks with completion cycle
    struct PendingCallback {
        uint64_t completion_cycle;
        std::function<void()> callback;

        bool operator>(const PendingCallback& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<PendingCallback,
                       std::vector<PendingCallback>,
                       std::greater<PendingCallback>> pending_callbacks_;

    // Statistics
    ComputeFabricStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // ========================================================================
    // Execution Functions (Type-Dispatched and FP32 Fallbacks)
    // ========================================================================
    // For matmul, the type dispatch mechanism in type_dispatch.hpp handles:
    // - Runtime DataType to compile-time template instantiation
    // - Universal library types (fp16_t, bf16_t, fp8eXmY_t, fp4_t)
    // - Accumulator type selection via ScalarTraits
    //
    // For elementwise ops, common activations (ReLU, GELU, SiLU) and binary
    // ops (ADD, MUL) use type dispatch; others fall back to FP32.
    //
    // Conv2D, Pool2D, and BatchNorm currently use FP32 implementations.
    // ========================================================================

    void execute_conv2d_fp32(const Conv2DDescriptor& desc,
                             const float* input, const float* weight,
                             const float* bias, float* output);

    void execute_elementwise_fp32(const ElementwiseDescriptor& desc,
                                  const float* a, const float* b, float* output);

    void execute_pool2d_fp32(const Pool2DDescriptor& desc,
                             const float* input, float* output);

    void execute_softmax_fp32(const SoftmaxDescriptor& desc,
                              const float* input, float* output);

    void execute_layernorm_fp32(const LayerNormDescriptor& desc,
                                const float* input, const float* weight,
                                const float* bias, float* output);

    void execute_batchnorm_fp32(const BatchNormDescriptor& desc,
                                const float* input, const float* weight,
                                const float* bias, const float* running_mean,
                                const float* running_var, float* output);

    // Latency estimation
    uint32_t estimate_latency(const MatMulDescriptor& desc) const;
    uint32_t estimate_latency(const Conv2DDescriptor& desc) const;
    uint32_t estimate_latency(const ElementwiseDescriptor& desc) const;
    uint32_t estimate_latency(const Pool2DDescriptor& desc) const;
    uint32_t estimate_latency(const SoftmaxDescriptor& desc) const;
    uint32_t estimate_latency(const LayerNormDescriptor& desc) const;
    uint32_t estimate_latency(const BatchNormDescriptor& desc) const;

    // Common callback scheduling
    void schedule_callback(uint32_t latency, std::function<void()> callback);
};

} // namespace sw::kpu
