// ============================================================================
// include/sw/kpu/models/transactional/datamovement/dma_engine.hpp
// Transactional (bandwidth-limited) DMA engine - queue-based timing
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <queue>
#include <vector>

#include <sw/kpu/models/interfaces/dma_engine_interface.hpp>

namespace sw::kpu {

/// Transactional DMA engine with bandwidth-limited timing
///
/// This is a middle-ground simulation mode, providing:
/// - Bandwidth-limited transfer modeling
/// - Per-channel queue tracking
/// - Basic contention modeling
/// - ~10-100x faster than cycle-accurate
///
/// Use for:
/// - Architecture exploration
/// - Workload characterization
/// - Performance estimation
/// - Bottleneck identification
class TransactionalDMAEngine : public IDMAEngine {
public:
    explicit TransactionalDMAEngine(const DMAEngineConfig& config);
    ~TransactionalDMAEngine() override = default;

    // Disable copying, allow moving
    TransactionalDMAEngine(const TransactionalDMAEngine&) = delete;
    TransactionalDMAEngine& operator=(const TransactionalDMAEngine&) = delete;
    TransactionalDMAEngine(TransactionalDMAEngine&&) = default;
    TransactionalDMAEngine& operator=(TransactionalDMAEngine&&) = default;

    // ========================================================================
    // Request Interface
    // ========================================================================

    std::optional<uint64_t> submit(
        const DMATransfer& transfer,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override;
    bool has_pending() const override { return pending_count_ > 0; }
    size_t pending_count() const override { return pending_count_; }

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

    SimulationFidelity fidelity() const override { return SimulationFidelity::TRANSACTIONAL; }
    const DMAEngineConfig& config() const override { return config_; }
    uint32_t num_channels() const override { return config_.num_channels; }

    ChannelState get_channel_state(uint32_t channel) const override;

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
    size_t pending_count_ = 0;

    // Bytes per cycle (derived from bandwidth)
    double bytes_per_cycle_;

    // Per-channel state
    struct ChannelInfo {
        uint64_t busy_until_cycle = 0;
        std::deque<uint64_t> pending_transfer_ids;
        bool is_idle(uint64_t cycle) const { return cycle >= busy_until_cycle; }
    };
    std::vector<ChannelInfo> channels_;

    // Transfer tracking
    struct TransferInfo {
        uint64_t transfer_id;
        uint64_t submit_cycle;
        uint64_t completion_cycle;
        uint32_t channel;
        uint32_t size;
        DMATransfer transfer;
        std::function<void()> callback;
        bool completed = false;
        bool cancelled = false;
    };
    std::map<uint64_t, TransferInfo> transfers_;

    // Completion queue (min-heap by completion cycle)
    struct CompletionEvent {
        uint64_t transfer_id;
        uint64_t completion_cycle;

        bool operator>(const CompletionEvent& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<CompletionEvent,
                       std::vector<CompletionEvent>,
                       std::greater<CompletionEvent>> completion_queue_;

    // Statistics
    DMAEngineStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Memory callbacks
    static constexpr size_t NUM_MEMORY_TYPES = 5;
    std::array<ReadCallback, NUM_MEMORY_TYPES> read_callbacks_;
    std::array<WriteCallback, NUM_MEMORY_TYPES> write_callbacks_;

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Select best channel for new transfer (round-robin or least-loaded)
    uint32_t select_channel(const DMATransfer& transfer);

    /// Compute transfer duration based on size and bandwidth
    uint32_t compute_duration(const DMATransfer& transfer) const;

    /// Update statistics for completed transfer
    void update_stats(const TransferInfo& info);
};

} // namespace sw::kpu
