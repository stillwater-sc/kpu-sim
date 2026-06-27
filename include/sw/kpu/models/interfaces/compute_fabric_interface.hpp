// ============================================================================
// include/sw/kpu/models/interfaces/compute/compute_fabric_interface.hpp
// Abstract interface for compute fabrics at all fidelity levels
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/kpu/resource_api.hpp>  // For ElementwiseOp enum
#include <sw/kpu/data_types.hpp>    // For DataType enum

// Forward declarations
namespace sw::trace {
    class ResourceTracker;
}

namespace sw::kpu {

// ============================================================================
// Compute Operation Types
// ============================================================================

/// Type of compute operation
enum class ComputeOpType : uint8_t {
    MATMUL,         // Matrix multiplication
    CONV2D,         // 2D convolution
    ELEMENTWISE,    // Element-wise operations (add, mul, relu, etc.)
    REDUCE,         // Reduction operations (sum, max, etc.)
    POOL2D,         // 2D pooling (max, avg, adaptive)
    SOFTMAX,        // Softmax
    LAYERNORM,      // Layer normalization
    BATCHNORM       // Batch normalization
};

constexpr std::string_view to_string(ComputeOpType op) {
    switch (op) {
        case ComputeOpType::MATMUL:      return "MATMUL";
        case ComputeOpType::CONV2D:      return "CONV2D";
        case ComputeOpType::ELEMENTWISE: return "ELEMENTWISE";
        case ComputeOpType::REDUCE:      return "REDUCE";
        case ComputeOpType::POOL2D:      return "POOL2D";
        case ComputeOpType::SOFTMAX:     return "SOFTMAX";
        case ComputeOpType::LAYERNORM:   return "LAYERNORM";
        case ComputeOpType::BATCHNORM:   return "BATCHNORM";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// TYPE DISPATCH DESIGN - How DataType Propagates Through KPU Resources
// =============================================================================
//
// This section documents how data type information flows from the compute
// descriptors through the KPU resource hierarchy. Different components need
// different levels of type awareness:
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                        TYPE INFORMATION FLOW                            │
// │                                                                         │
// │   User Code                                                             │
// │       │                                                                 │
// │       ▼                                                                 │
// │   MatMulDescriptor { dtype = FP16, ... }                                │
// │       │                                                                 │
// │       ├───────────────────┬────────────────────┬─────────────────────┐  │
// │       │                   │                    │                     │  │
// │       ▼                   ▼                    ▼                     ▼  │
// │   Compute Fabric      DMA Engine         BlockMover            Streamer │
// │   ───────────────     ──────────         ──────────            ──────── │
// │   Needs: Full         Needs: Size        Needs: Size        Needs: Full │
// │   DataType for        only (bytes        only (bytes        DataType    │
// │   template dispatch   per element)       per element)       for pack/   │
// │   to typed kernels    for burst          for tile           unpack      │
// │                       calculation        alignment          operations  │
// │                                                                         │
// │   FP16 → matmul<fp16>  FP16 → 2 bytes    FP16 → 2 bytes    FP16 → unpack│
// │   INT8 → matmul<int8>  INT8 → 1 byte     INT8 → 1 byte     INT4 → unpack│
// │   INT4 → Q/DQ path     INT4 → 0.5 byte*  INT4 → aligned    from packed  │
// │                        (packed)          to byte           storage      │
// │                                                                         │
// └─────────────────────────────────────────────────────────────────────────┘
//
// KEY INSIGHT: Storage vs. Compute Type
// ─────────────────────────────────────
// - Storage path (DMA, BlockMover): Only needs element_size in bytes
// - Compute path (Fabric): Needs full dtype for kernel template instantiation
// - Stream path (Streamer): Needs dtype for packing/unpacking sub-byte types
//
// PACKING CONSIDERATIONS:
// ───────────────────────
// - INT4: Packed storage (2 elements per byte), requires unpacking to compute
// - FP4:  Unpacked storage (1 byte per element), uses Universal cfloat<4,1>
// - All other types: Natural alignment, no packing needed
//
// The Streamer must know the semantic DataType (not just byte size) to:
// 1. Unpack INT4 pairs into separate values for compute
// 2. Pack compute results back into INT4 storage format
// 3. Handle Universal library type conversions for FP8/FP4
//
// ACCUMULATOR TYPE:
// ─────────────────
// The compute fabric automatically selects the appropriate accumulator:
// - FP16/BF16/FP8/FP4 → FP32 accumulator (prevents overflow)
// - INT8/INT4 → INT32 accumulator (prevents overflow)
// - FP32 → FP32 accumulator (native precision)
//
// This is handled by ScalarTraits<T>::accumulator_type in quantization/scalar_traits.hpp
// =============================================================================

/// Matrix multiplication descriptor
///
/// Contains all information needed to dispatch a matmul operation:
/// - Matrix dimensions (M, N, K)
/// - Data type for compute dispatch
/// - Memory addresses for data location
///
/// The dtype field drives two critical paths:
/// 1. Compute dispatch: Selects template instantiation (matmul<fp16>, matmul<int8>, etc.)
/// 2. Resource allocation: Derives element_size for DMA/BlockMover byte calculations
struct MatMulDescriptor {
    uint32_t m = 0;           // Output rows
    uint32_t n = 0;           // Output columns
    uint32_t k = 0;           // Inner dimension

    // Data locations (addresses in local memory)
    uint64_t a_addr = 0;      // A[m, k]
    uint64_t b_addr = 0;      // B[k, n]
    uint64_t c_addr = 0;      // C[m, n] (output)

    // ==========================================================================
    // Data Type Specification
    // ==========================================================================
    //
    // dtype: Semantic data type for compute dispatch
    //   - Used by BehavioralComputeFabric to select template kernel instantiation
    //   - Used by Streamers to determine packing/unpacking requirements
    //   - Automatically derives element_size via dtype_size()
    //
    // element_size: Physical size in bytes (derived from dtype for consistency)
    //   - Used by DMA for burst size calculation
    //   - Used by BlockMover for tile alignment
    //   - For INT4, this is the STORAGE size (0.5 bytes), not compute size
    //
    // Example:
    //   MatMulDescriptor desc;
    //   desc.dtype = DataType::FP16;
    //   // element_size automatically = 2 (from dtype_size(FP16))
    //   // Compute fabric dispatches to matmul<fp16_t, float>
    //   // DMA calculates burst as m * k * 2 bytes
    // ==========================================================================
    DataType dtype = DataType::FLOAT32;  // Input/output data type

    /// Get element size in bytes (derived from dtype)
    /// For packed types like INT4, returns storage bytes (1 for 2 elements)
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }

    bool accumulate = false;   // Add to existing C vs overwrite

    // User tag for identification
    uint64_t user_tag = 0;
};

/// 2D Convolution descriptor
struct Conv2DDescriptor {
    // Input dimensions [N, C_in, H_in, W_in]
    uint32_t batch_size = 1;
    uint32_t in_channels = 0;
    uint32_t in_height = 0;
    uint32_t in_width = 0;

    // Weight dimensions [C_out, C_in/groups, K_h, K_w]
    uint32_t out_channels = 0;
    uint32_t kernel_height = 0;
    uint32_t kernel_width = 0;

    // Convolution parameters
    uint32_t stride_h = 1;
    uint32_t stride_w = 1;
    uint32_t padding_h = 0;
    uint32_t padding_w = 0;
    uint32_t dilation_h = 1;
    uint32_t dilation_w = 1;
    uint32_t groups = 1;

    // Data format (see MatMulDescriptor for type dispatch documentation)
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    bool has_bias = false;

    // User tag for identification
    uint64_t user_tag = 0;

    // Computed output dimensions
    uint32_t out_height() const {
        return (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    }
    uint32_t out_width() const {
        return (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    }

    // Compute MACs for this convolution
    uint64_t compute_macs() const {
        uint64_t out_h = out_height();
        uint64_t out_w = out_width();
        uint64_t in_ch_per_group = in_channels / groups;
        return batch_size * out_channels * out_h * out_w * in_ch_per_group * kernel_height * kernel_width;
    }
};

// Note: ElementwiseOp enum is defined in <sw/kpu/resource_api.hpp>
// Helper functions for element-wise operations:

/// Check if operation is unary (single input)
constexpr bool is_unary_op(ElementwiseOp op) {
    // NEG=10, ABS=11, SQRT=12, EXP=13, LOG=14, SILU=15 (and activations RELU=6, GELU=7, SIGMOID=8, TANH=9)
    uint8_t v = static_cast<uint8_t>(op);
    return (v >= 6 && v <= 15);  // All activation and unary ops
}

/// Check if operation uses a scalar second operand
constexpr bool is_scalar_op(ElementwiseOp op) {
    return op >= ElementwiseOp::ADD_SCALAR;
}

/// Element-wise operation descriptor
struct ElementwiseDescriptor {
    ElementwiseOp op = ElementwiseOp::ADD;
    uint64_t count = 0;        // Number of elements
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    uint64_t user_tag = 0;
};

/// 2D Pooling descriptor
struct Pool2DDescriptor {
    enum class PoolType : uint8_t { MAX, AVG, ADAPTIVE_AVG };

    PoolType pool_type = PoolType::MAX;

    // Input dimensions [N, C, H_in, W_in]
    uint32_t batch_size = 1;
    uint32_t channels = 0;
    uint32_t in_height = 0;
    uint32_t in_width = 0;

    // Pooling window
    uint32_t kernel_height = 0;
    uint32_t kernel_width = 0;
    uint32_t stride_h = 0;      // 0 means use kernel size
    uint32_t stride_w = 0;
    uint32_t padding_h = 0;
    uint32_t padding_w = 0;

    // For adaptive pooling: target output size
    uint32_t target_out_height = 0;
    uint32_t target_out_width = 0;

    // Data format (see MatMulDescriptor for type dispatch documentation)
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    uint64_t user_tag = 0;

    // Computed output dimensions
    uint32_t out_height() const {
        if (pool_type == PoolType::ADAPTIVE_AVG) return target_out_height;
        uint32_t s_h = stride_h > 0 ? stride_h : kernel_height;
        return (in_height + 2 * padding_h - kernel_height) / s_h + 1;
    }
    uint32_t out_width() const {
        if (pool_type == PoolType::ADAPTIVE_AVG) return target_out_width;
        uint32_t s_w = stride_w > 0 ? stride_w : kernel_width;
        return (in_width + 2 * padding_w - kernel_width) / s_w + 1;
    }
};

constexpr std::string_view to_string(Pool2DDescriptor::PoolType pt) {
    switch (pt) {
        case Pool2DDescriptor::PoolType::MAX: return "MAX";
        case Pool2DDescriptor::PoolType::AVG: return "AVG";
        case Pool2DDescriptor::PoolType::ADAPTIVE_AVG: return "ADAPTIVE_AVG";
        default: return "UNKNOWN";
    }
}

/// Softmax descriptor
struct SoftmaxDescriptor {
    uint32_t batch_size = 0;   // Product of all dims before softmax dim
    uint32_t dim_size = 0;     // Size of softmax dimension
    uint32_t inner_size = 1;   // Product of all dims after softmax dim
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    uint64_t user_tag = 0;

    uint64_t total_elements() const {
        return static_cast<uint64_t>(batch_size) * dim_size * inner_size;
    }
};

/// Layer normalization descriptor
struct LayerNormDescriptor {
    uint32_t batch_size = 0;       // Batch dimension
    uint32_t normalized_size = 0;  // Size of normalized shape (product)
    float eps = 1e-5f;
    bool has_weight = true;
    bool has_bias = true;
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    uint64_t user_tag = 0;
};

/// Batch normalization descriptor
struct BatchNormDescriptor {
    uint32_t batch_size = 0;
    uint32_t num_features = 0;  // C dimension
    uint32_t spatial_size = 0;  // H * W
    float eps = 1e-5f;
    float momentum = 0.1f;
    bool training = false;      // If true, update running stats
    DataType dtype = DataType::FLOAT32;
    uint8_t element_size() const { return static_cast<uint8_t>(dtype_size(dtype)); }
    uint64_t user_tag = 0;
};

// ============================================================================
// Compute Fabric Statistics
// ============================================================================

/// Common statistics interface for all compute fabric implementations
struct ComputeFabricStatistics {
    // Operation counts
    uint64_t matmuls = 0;
    uint64_t conv2ds = 0;
    uint64_t elementwise_ops = 0;
    uint64_t reductions = 0;
    uint64_t pool2ds = 0;
    uint64_t softmaxes = 0;
    uint64_t layernorms = 0;
    uint64_t batchnorms = 0;

    // Compute volume
    uint64_t total_macs = 0;    // Multiply-accumulate operations
    uint64_t total_flops = 0;   // Floating point operations

    // Latency statistics
    uint64_t total_compute_cycles = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t max_latency = 0;

    // Utilization
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;
    uint64_t stall_cycles = 0;  // Waiting for data

    // Derived metrics
    uint64_t total_ops() const {
        return matmuls + conv2ds + elementwise_ops + reductions +
               pool2ds + softmaxes + layernorms + batchnorms;
    }

    double avg_latency() const {
        uint64_t ops = total_ops();
        return ops > 0 ? static_cast<double>(total_compute_cycles) / static_cast<double>(ops) : 0.0;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    double mac_efficiency(uint64_t peak_macs_per_cycle) const {
        if (busy_cycles == 0 || peak_macs_per_cycle == 0) return 0.0;
        return static_cast<double>(total_macs) / static_cast<double>(busy_cycles * peak_macs_per_cycle);
    }

    void reset() {
        matmuls = conv2ds = elementwise_ops = reductions = 0;
        pool2ds = softmaxes = layernorms = batchnorms = 0;
        total_macs = total_flops = 0;
        total_compute_cycles = 0;
        min_latency = UINT64_MAX;
        max_latency = 0;
        busy_cycles = idle_cycles = stall_cycles = 0;
    }
};

// ============================================================================
// Compute Fabric Interface
// ============================================================================

/// Abstract interface for compute fabrics at all fidelity levels
///
/// This interface provides a common API that is implemented by:
/// - BehavioralComputeFabric (instant compute)
/// - TransactionalComputeFabric (throughput-based timing)
/// - CycleAccurateSystolicArray (full pipeline timing)
///
/// All implementations guarantee:
/// 1. Functional correctness (correct computation results)
/// 2. Callback semantics (callbacks invoked when operation completes)
/// 3. Statistics collection (if enabled)
/// 4. Tracing support (if enabled)
class IComputeFabric {
public:
    virtual ~IComputeFabric() = default;

    // ========================================================================
    // Compute Operations
    // ========================================================================

    /// Submit a matrix multiplication
    ///
    /// @param desc MatMul descriptor
    /// @param a_data Pointer to A matrix data
    /// @param b_data Pointer to B matrix data
    /// @param c_data Pointer to C matrix output buffer
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    ///
    /// Behavior by fidelity:
    /// - BEHAVIORAL: Completes immediately
    /// - TRANSACTIONAL: Throughput-based delay
    /// - CYCLE_ACCURATE: Full systolic pipeline timing
    virtual std::optional<uint64_t> submit_matmul(
        const MatMulDescriptor& desc,
        const void* a_data,
        const void* b_data,
        void* c_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a 2D convolution
    ///
    /// @param desc Conv2D descriptor
    /// @param input_data Input tensor [N, C_in, H, W]
    /// @param weight_data Weight tensor [C_out, C_in/groups, K_h, K_w]
    /// @param bias_data Optional bias tensor [C_out], nullptr if no bias
    /// @param output_data Output tensor [N, C_out, H_out, W_out]
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_conv2d(
        const Conv2DDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit an element-wise operation
    ///
    /// @param desc Elementwise descriptor
    /// @param a_data First input tensor
    /// @param b_data Second input tensor (ignored for unary ops)
    /// @param output_data Output tensor
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_elementwise(
        const ElementwiseDescriptor& desc,
        const void* a_data,
        const void* b_data,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a 2D pooling operation
    ///
    /// @param desc Pool2D descriptor
    /// @param input_data Input tensor [N, C, H, W]
    /// @param output_data Output tensor [N, C, H_out, W_out]
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_pool2d(
        const Pool2DDescriptor& desc,
        const void* input_data,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a softmax operation
    ///
    /// @param desc Softmax descriptor
    /// @param input_data Input tensor
    /// @param output_data Output tensor
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_softmax(
        const SoftmaxDescriptor& desc,
        const void* input_data,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a layer normalization operation
    ///
    /// @param desc LayerNorm descriptor
    /// @param input_data Input tensor
    /// @param weight_data Optional gamma (scale) tensor
    /// @param bias_data Optional beta (shift) tensor
    /// @param output_data Output tensor
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_layernorm(
        const LayerNormDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a batch normalization operation
    ///
    /// @param desc BatchNorm descriptor
    /// @param input_data Input tensor [N, C, H, W]
    /// @param weight_data Gamma (scale) tensor [C]
    /// @param bias_data Beta (shift) tensor [C]
    /// @param running_mean Running mean tensor [C]
    /// @param running_var Running variance tensor [C]
    /// @param output_data Output tensor [N, C, H, W]
    /// @param callback Optional callback when operation completes
    /// @return Operation ID if accepted, nullopt if busy
    virtual std::optional<uint64_t> submit_batchnorm(
        const BatchNormDescriptor& desc,
        const void* input_data,
        const void* weight_data,
        const void* bias_data,
        const void* running_mean,
        const void* running_var,
        void* output_data,
        std::function<void()> callback = nullptr) = 0;

    /// Check if fabric can accept more operations
    virtual bool can_accept() const = 0;

    /// Check if there are pending operations
    virtual bool has_pending() const = 0;

    /// Get number of pending operations
    virtual size_t pending_count() const = 0;

    /// Check if fabric is currently computing
    virtual bool is_busy() const = 0;

    // ========================================================================
    // Simulation Interface
    // ========================================================================

    /// Advance simulation by one cycle
    virtual void tick() = 0;

    /// Process until all pending operations complete
    virtual void drain() = 0;

    /// Reset fabric to initial state
    virtual void reset() = 0;

    /// Get current simulation cycle
    virtual uint64_t current_cycle() const = 0;

    /// Set current simulation cycle
    virtual void set_cycle(uint64_t cycle) = 0;

    // ========================================================================
    // Configuration Queries
    // ========================================================================

    /// Get simulation fidelity level
    virtual SimulationFidelity fidelity() const = 0;

    /// Get compute technology
    virtual ComputeTechnology technology() const = 0;

    /// Get full configuration
    virtual const ComputeFabricConfig& config() const = 0;

    /// Get tile ID
    virtual uint32_t tile_id() const = 0;

    /// Get array dimensions
    virtual uint32_t array_rows() const = 0;
    virtual uint32_t array_cols() const = 0;

    /// Get peak MACs per cycle
    virtual uint32_t peak_macs_per_cycle() const = 0;

    // ========================================================================
    // Pipeline State (for CYCLE_ACCURATE)
    // ========================================================================

    /// Pipeline stage state
    enum class PipelineState : uint8_t {
        IDLE,
        LOADING,    // Loading data into array
        COMPUTING,  // Systolic computation
        DRAINING,   // Draining results
        STORING     // Storing results
    };

    /// Get current pipeline state
    virtual PipelineState get_pipeline_state() const = 0;

    /// Get pipeline progress (0-100%)
    virtual uint8_t pipeline_progress() const = 0;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get current statistics
    virtual const ComputeFabricStatistics& stats() const = 0;

    /// Reset statistics
    virtual void reset_stats() = 0;

    // ========================================================================
    // Observability
    // ========================================================================

    /// Enable or disable tracing
    virtual void enable_tracing(bool enable) = 0;

    /// Check if tracing is enabled
    virtual bool tracing_enabled() const = 0;

    /// Set resource tracker for trace export
    virtual void set_resource_tracker(sw::trace::ResourceTracker* tracker) = 0;
};

// ============================================================================
// Compute Fabric Factory
// ============================================================================

/// Create a compute fabric based on configuration
///
/// @param config Compute fabric configuration
/// @param tile_id Unique tile identifier
/// @return Unique pointer to compute fabric implementation
std::unique_ptr<IComputeFabric> create_compute_fabric(
    const ComputeFabricConfig& config, uint32_t tile_id);

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert pipeline state to string
constexpr std::string_view to_string(IComputeFabric::PipelineState state) {
    switch (state) {
        case IComputeFabric::PipelineState::IDLE:      return "IDLE";
        case IComputeFabric::PipelineState::LOADING:   return "LOADING";
        case IComputeFabric::PipelineState::COMPUTING: return "COMPUTING";
        case IComputeFabric::PipelineState::DRAINING:  return "DRAINING";
        case IComputeFabric::PipelineState::STORING:   return "STORING";
        default: return "UNKNOWN";
    }
}

} // namespace sw::kpu
