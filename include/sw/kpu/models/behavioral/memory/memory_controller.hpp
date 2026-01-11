// ============================================================================
// include/sw/kpu/models/behavioral/memory/memory_controller.hpp
// Behavioral (functional) memory controller - instant/fixed latency
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <vector>

#include <sw/kpu/components/memory/memory_controller_interface.hpp>

namespace sw::kpu {

/// Behavioral memory controller with instant or fixed latency
///
/// This is the fastest simulation mode, providing:
/// - Functional correctness (data is stored and retrieved correctly)
/// - Optional fixed latency modeling
/// - No state machine, no queuing delays, no contention
///
/// Use for:
/// - Software bring-up and functional verification
/// - Unit testing
/// - CI/CD pipelines where speed matters more than timing accuracy
class BehavioralMemoryController : public IMemoryController {
public:
    explicit BehavioralMemoryController(const MemoryControllerConfig& config);
    ~BehavioralMemoryController() override = default;

    // Disable copying, allow moving
    BehavioralMemoryController(const BehavioralMemoryController&) = delete;
    BehavioralMemoryController& operator=(const BehavioralMemoryController&) = delete;
    BehavioralMemoryController(BehavioralMemoryController&&) = default;
    BehavioralMemoryController& operator=(BehavioralMemoryController&&) = default;

    // ========================================================================
    // Request Interface
    // ========================================================================

    std::optional<uint64_t> submit_read(
        uint64_t address,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_write(
        uint64_t address,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override { return true; }  // Always accepts
    bool has_pending() const override { return !pending_callbacks_.empty(); }
    size_t pending_count() const override { return pending_callbacks_.size(); }

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
    MemoryTechnology technology() const override { return config_.technology; }
    VerificationLevel verification_level() const override { return config_.verification; }
    const MemoryControllerConfig& config() const override { return config_; }

    // ========================================================================
    // Bank State Queries
    // ========================================================================

    // Behavioral model always reports banks as ACTIVE (no real state)
    BankState get_bank_state(uint8_t /*channel*/, uint8_t /*bank*/) const override {
        return BankState::ACTIVE;
    }

    bool is_row_open(uint8_t /*channel*/, uint8_t /*bank*/, uint32_t /*row*/) const override {
        return true;  // All rows are "open" in behavioral model
    }

    uint8_t num_channels() const override { return config_.num_channels; }
    uint8_t banks_per_channel() const override { return config_.banks_per_channel; }

    // ========================================================================
    // Statistics
    // ========================================================================

    const MemoryControllerStatistics& stats() const override { return stats_; }
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
    // Invariant Checking (not applicable for behavioral)
    // ========================================================================

    void enable_invariants(bool /*enable*/) override { }  // No-op
    bool invariants_enabled() const override { return false; }
    const std::vector<InvariantViolation>& violations() const override {
        return empty_violations_;
    }
    bool has_violations() const override { return false; }
    void clear_violations() override { }

    // ========================================================================
    // Direct Memory Access (for testing/initialization)
    // ========================================================================

    /// Read data directly from backing store
    void read_direct(uint64_t address, void* data, uint32_t size) const;

    /// Write data directly to backing store
    void write_direct(uint64_t address, const void* data, uint32_t size);

    /// Get total memory capacity
    uint64_t capacity() const { return capacity_; }

private:
    // Configuration
    MemoryControllerConfig config_;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 1;
    uint64_t capacity_ = 0;

    // Sparse backing store (only allocate pages that are written)
    static constexpr size_t PAGE_SIZE = 4096;
    mutable std::map<uint64_t, std::vector<uint8_t>> pages_;

    // Pending callbacks with completion cycle
    struct PendingCallback {
        uint64_t completion_cycle;
        std::function<void()> callback;

        // Comparison for priority queue (min-heap by completion cycle)
        bool operator>(const PendingCallback& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<PendingCallback,
                       std::vector<PendingCallback>,
                       std::greater<PendingCallback>> pending_callbacks_;

    // Statistics
    MemoryControllerStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Empty violations list (behavioral never has violations)
    static const std::vector<InvariantViolation> empty_violations_;

    // Helper to get/create page
    std::vector<uint8_t>& get_page(uint64_t page_addr);
    const std::vector<uint8_t>* find_page(uint64_t page_addr) const;
};

} // namespace sw::kpu
