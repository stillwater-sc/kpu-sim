// ============================================================================
// include/sw/kpu/components/compute/compute_fabric_interface.hpp
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
    SOFTMAX,        // Softmax
    LAYERNORM       // Layer normalization
};

constexpr std::string_view to_string(ComputeOpType op) {
    switch (op) {
        case ComputeOpType::MATMUL:      return "MATMUL";
        case ComputeOpType::CONV2D:      return "CONV2D";
        case ComputeOpType::ELEMENTWISE: return "ELEMENTWISE";
        case ComputeOpType::REDUCE:      return "REDUCE";
        case ComputeOpType::SOFTMAX:     return "SOFTMAX";
        case ComputeOpType::LAYERNORM:   return "LAYERNORM";
        default: return "UNKNOWN";
    }
}

/// Matrix multiplication descriptor
struct MatMulDescriptor {
    uint32_t m = 0;           // Output rows
    uint32_t n = 0;           // Output columns
    uint32_t k = 0;           // Inner dimension

    // Data locations (addresses in local memory)
    uint64_t a_addr = 0;      // A[m, k]
    uint64_t b_addr = 0;      // B[k, n]
    uint64_t c_addr = 0;      // C[m, n] (output)

    // Data format
    uint8_t element_size = 4;  // Bytes per element
    bool accumulate = false;   // Add to existing C vs overwrite

    // User tag for identification
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
    double avg_latency() const {
        uint64_t ops = matmuls + conv2ds + elementwise_ops + reductions;
        return ops > 0 ? static_cast<double>(total_compute_cycles) / ops : 0.0;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / total : 0.0;
    }

    double mac_efficiency(uint64_t peak_macs_per_cycle) const {
        if (busy_cycles == 0 || peak_macs_per_cycle == 0) return 0.0;
        return static_cast<double>(total_macs) / (busy_cycles * peak_macs_per_cycle);
    }

    void reset() {
        matmuls = conv2ds = elementwise_ops = reductions = 0;
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
