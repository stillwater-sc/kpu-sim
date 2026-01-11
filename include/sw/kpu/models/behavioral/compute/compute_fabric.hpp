// ============================================================================
// include/sw/kpu/models/behavioral/compute/compute_fabric.hpp
// Behavioral (functional) compute fabric - instant computation
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

#include <sw/kpu/components/compute/compute_fabric_interface.hpp>

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

    // Helpers
    void execute_matmul_fp32(const MatMulDescriptor& desc,
                             const float* a, const float* b, float* c);
    uint32_t estimate_latency(const MatMulDescriptor& desc) const;
};

} // namespace sw::kpu
