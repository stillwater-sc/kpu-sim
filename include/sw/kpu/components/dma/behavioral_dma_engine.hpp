// ============================================================================
// include/sw/kpu/components/dma/behavioral_dma_engine.hpp
// Behavioral (functional) DMA engine - instant/fixed latency
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <vector>

#include <sw/kpu/components/dma/dma_engine_interface.hpp>

namespace sw::kpu {

/// Behavioral DMA engine with instant or fixed latency
///
/// This is the fastest simulation mode, providing:
/// - Functional correctness (data is transferred correctly)
/// - Optional fixed latency modeling
/// - No channel arbitration, no bandwidth limits
///
/// Use for:
/// - Software bring-up and functional verification
/// - Unit testing
/// - CI/CD pipelines where speed matters more than timing accuracy
class BehavioralDMAEngine : public IDMAEngine {
public:
    explicit BehavioralDMAEngine(const DMAEngineConfig& config);
    ~BehavioralDMAEngine() override = default;

    // Disable copying, allow moving
    BehavioralDMAEngine(const BehavioralDMAEngine&) = delete;
    BehavioralDMAEngine& operator=(const BehavioralDMAEngine&) = delete;
    BehavioralDMAEngine(BehavioralDMAEngine&&) = default;
    BehavioralDMAEngine& operator=(BehavioralDMAEngine&&) = default;

    // ========================================================================
    // Request Interface
    // ========================================================================

    std::optional<uint64_t> submit(
        const DMATransfer& transfer,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override { return true; }  // Always accepts
    bool has_pending() const override { return !pending_callbacks_.empty(); }
    size_t pending_count() const override { return pending_callbacks_.size(); }

    bool wait_for(uint64_t transfer_id) override;
    bool cancel(uint64_t transfer_id) override;

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
    const DMAEngineConfig& config() const override { return config_; }
    uint32_t num_channels() const override { return config_.num_channels; }

    ChannelState get_channel_state(uint32_t channel) const override {
        (void)channel;
        return ChannelState::IDLE;  // Always idle in behavioral
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    const DMAEngineStatistics& stats() const override { return stats_; }
    void reset_stats() override { stats_.reset(); }

    // ========================================================================
    // Observability
    // ========================================================================

    void enable_tracing(bool enable) override { tracing_enabled_ = enable; }
    bool tracing_enabled() const override { return tracing_enabled_; }
    void set_resource_tracker(sw::trace::ResourceTracker* tracker) override {
        resource_tracker_ = tracker;
    }

    // ========================================================================
    // Memory Interface
    // ========================================================================

    void set_read_callback(DMAMemoryType type, ReadCallback callback) override {
        read_callbacks_[static_cast<size_t>(type)] = callback;
    }

    void set_write_callback(DMAMemoryType type, WriteCallback callback) override {
        write_callbacks_[static_cast<size_t>(type)] = callback;
    }

private:
    // Configuration
    DMAEngineConfig config_;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_transfer_id_ = 1;

    // Pending callbacks with completion cycle
    struct PendingCallback {
        uint64_t transfer_id;
        uint64_t completion_cycle;
        std::function<void()> callback;

        bool operator>(const PendingCallback& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<PendingCallback,
                       std::vector<PendingCallback>,
                       std::greater<PendingCallback>> pending_callbacks_;

    // Track active transfer IDs for wait_for/cancel
    std::map<uint64_t, bool> active_transfers_;  // id -> completed

    // Statistics
    DMAEngineStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Memory callbacks
    static constexpr size_t NUM_MEMORY_TYPES = 5;
    std::array<ReadCallback, NUM_MEMORY_TYPES> read_callbacks_;
    std::array<WriteCallback, NUM_MEMORY_TYPES> write_callbacks_;

    // Helper to compute transfer latency
    uint32_t compute_latency(const DMATransfer& transfer) const;

    // Helper to update statistics
    void update_stats(const DMATransfer& transfer, uint32_t latency);
};

} // namespace sw::kpu
