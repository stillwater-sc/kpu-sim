// ============================================================================
// include/sw/kpu/models/transactional/compute/compute_fabric.hpp
// Transactional compute fabric - throughput-based timing
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <queue>
#include <vector>

#include <sw/kpu/components/compute/compute_fabric_interface.hpp>

namespace sw::kpu {

/// Transactional compute fabric with throughput-based timing
///
/// This is a middle-ground simulation mode, providing:
/// - Throughput-based timing (MACs per cycle)
/// - Queue-based operation scheduling
/// - Basic pipeline modeling
/// - ~10-100x faster than cycle-accurate
///
/// Use for:
/// - Architecture exploration
/// - Workload characterization
/// - Performance estimation
/// - Bottleneck identification
class TransactionalComputeFabric : public IComputeFabric {
public:
    explicit TransactionalComputeFabric(const ComputeFabricConfig& config, uint32_t tile_id);
    ~TransactionalComputeFabric() override = default;

    // Disable copying, allow moving
    TransactionalComputeFabric(const TransactionalComputeFabric&) = delete;
    TransactionalComputeFabric& operator=(const TransactionalComputeFabric&) = delete;
    TransactionalComputeFabric(TransactionalComputeFabric&&) = default;
    TransactionalComputeFabric& operator=(TransactionalComputeFabric&&) = default;

    // ========================================================================
    // Compute Operations
    // ========================================================================

    std::optional<uint64_t> submit_matmul(
        const MatMulDescriptor& desc,
        const void* a_data,
        const void* b_data,
        void* c_data,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override;
    bool has_pending() const override { return pending_count_ > 0; }
    size_t pending_count() const override { return pending_count_; }
    bool is_busy() const override { return current_state_ != PipelineState::IDLE; }

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

    SimulationFidelity fidelity() const override { return SimulationFidelity::TRANSACTIONAL; }
    ComputeTechnology technology() const override { return config_.technology; }
    const ComputeFabricConfig& config() const override { return config_; }
    uint32_t tile_id() const override { return tile_id_; }
    uint32_t array_rows() const override { return config_.array_rows; }
    uint32_t array_cols() const override { return config_.array_cols; }
    uint32_t peak_macs_per_cycle() const override { return config_.macs_per_cycle; }

    // ========================================================================
    // Pipeline State
    // ========================================================================

    PipelineState get_pipeline_state() const override { return current_state_; }
    uint8_t pipeline_progress() const override;

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
    size_t pending_count_ = 0;
    PipelineState current_state_ = PipelineState::IDLE;
    uint64_t state_end_cycle_ = 0;

    // Queue of pending operations
    struct PendingOp {
        uint64_t op_id;
        uint64_t submit_cycle;
        uint64_t completion_cycle;
        uint32_t m, n, k;
        std::function<void()> callback;

        bool operator>(const PendingOp& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<PendingOp,
                       std::vector<PendingOp>,
                       std::greater<PendingOp>> completion_queue_;

    // Current operation being processed
    PendingOp current_op_{};
    bool has_current_op_ = false;

    // Statistics
    ComputeFabricStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Helpers
    void execute_matmul_fp32(const MatMulDescriptor& desc,
                             const float* a, const float* b, float* c);
    uint32_t compute_duration(const MatMulDescriptor& desc) const;
    void advance_state();
};

} // namespace sw::kpu
