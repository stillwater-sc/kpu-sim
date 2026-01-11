// ============================================================================
// include/sw/kpu/models/transactional/memory/memory_controller.hpp
// Transactional (queue-based) memory controller - statistical timing
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <queue>
#include <random>
#include <vector>

#include <sw/kpu/components/memory/memory_controller_interface.hpp>

namespace sw::kpu {

/// Transactional memory controller with queue-based timing
///
/// This is a middle-ground simulation mode, providing:
/// - Queue-based contention modeling
/// - Statistical latency (mean + variance)
/// - Per-bank queue tracking
/// - Basic row buffer modeling (hit vs miss)
/// - ~10-100x faster than cycle-accurate
///
/// Use for:
/// - Architecture exploration
/// - Workload characterization
/// - Performance estimation
/// - Bottleneck identification
class TransactionalMemoryController : public IMemoryController {
public:
    explicit TransactionalMemoryController(const MemoryControllerConfig& config);
    ~TransactionalMemoryController() override = default;

    // Disable copying, allow moving
    TransactionalMemoryController(const TransactionalMemoryController&) = delete;
    TransactionalMemoryController& operator=(const TransactionalMemoryController&) = delete;
    TransactionalMemoryController(TransactionalMemoryController&&) = default;
    TransactionalMemoryController& operator=(TransactionalMemoryController&&) = default;

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

    bool can_accept() const override;
    bool has_pending() const override { return pending_count_ > 0; }
    size_t pending_count() const override { return pending_count_; }

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
    MemoryTechnology technology() const override { return config_.technology; }
    VerificationLevel verification_level() const override { return config_.verification; }
    const MemoryControllerConfig& config() const override { return config_; }

    // ========================================================================
    // Bank State Queries
    // ========================================================================

    // Transactional model tracks aggregate bank state (busy/idle)
    BankState get_bank_state(uint8_t channel, uint8_t bank) const override;
    bool is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const override;

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
    // Invariant Checking (limited for transactional)
    // ========================================================================

    void enable_invariants(bool enable) override { invariants_enabled_ = enable; }
    bool invariants_enabled() const override { return invariants_enabled_; }
    const std::vector<InvariantViolation>& violations() const override {
        return violations_;
    }
    bool has_violations() const override { return !violations_.empty(); }
    void clear_violations() override { violations_.clear(); }

    // ========================================================================
    // Direct Memory Access (for testing/initialization)
    // ========================================================================

    /// Read data directly from backing store
    void read_direct(uint64_t address, void* data, uint32_t size) const;

    /// Write data directly to backing store
    void write_direct(uint64_t address, const void* data, uint32_t size);

    /// Get total memory capacity
    uint64_t capacity() const { return capacity_; }

    // ========================================================================
    // Transactional-Specific Configuration
    // ========================================================================

    /// Set latency variance (0 = fixed latency, 1 = high variance)
    void set_latency_variance(double variance) { latency_variance_ = variance; }

    /// Get current latency variance
    double latency_variance() const { return latency_variance_; }

private:
    // Configuration
    MemoryControllerConfig config_;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 1;
    uint64_t capacity_ = 0;
    size_t pending_count_ = 0;

    // Sparse backing store (only allocate pages that are written)
    static constexpr size_t PAGE_SIZE = 4096;
    mutable std::map<uint64_t, std::vector<uint8_t>> pages_;

    // Per-request tracking
    struct PendingRequest {
        uint64_t request_id;
        uint64_t completion_cycle;
        uint64_t submit_cycle;
        uint8_t channel;
        uint8_t bank;
        MemoryRequestType type;
        std::function<void()> callback;

        // Comparison for priority queue (min-heap by completion cycle)
        bool operator>(const PendingRequest& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };

    // Priority queue for completion events
    std::priority_queue<PendingRequest,
                       std::vector<PendingRequest>,
                       std::greater<PendingRequest>> completion_queue_;

    // Per-bank state for contention modeling
    struct BankInfo {
        uint64_t busy_until_cycle = 0;     // When bank becomes free
        uint32_t open_row = UINT32_MAX;    // Currently open row (UINT32_MAX = none)
        std::deque<uint64_t> queue;        // Request IDs waiting for this bank

        bool is_idle(uint64_t cycle) const { return cycle >= busy_until_cycle; }
        bool has_row_open() const { return open_row != UINT32_MAX; }
    };

    std::vector<std::vector<BankInfo>> banks_;  // [channel][bank]

    // Latency modeling
    double latency_variance_ = 0.1;  // 10% variance by default
    std::mt19937 rng_;

    // Last operation type for turnaround tracking
    MemoryRequestType last_op_type_ = MemoryRequestType::READ;
    uint64_t last_op_cycle_ = 0;

    // Statistics
    MemoryControllerStatistics stats_;

    // Tracing and verification
    bool tracing_enabled_ = false;
    bool invariants_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;
    std::vector<InvariantViolation> violations_;

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Decode address to channel, bank, row
    void decode_address(uint64_t address, uint8_t& channel, uint8_t& bank, uint32_t& row) const;

    /// Calculate latency for a request
    uint32_t calculate_latency(uint8_t channel, uint8_t bank, uint32_t row,
                               MemoryRequestType type);

    /// Sample from statistical latency distribution
    uint32_t sample_latency(uint32_t base_latency);

    /// Get/create page in backing store
    std::vector<uint8_t>& get_page(uint64_t page_addr);
    const std::vector<uint8_t>* find_page(uint64_t page_addr) const;

    /// Submit a request (common implementation)
    std::optional<uint64_t> submit_request(
        uint64_t address,
        uint32_t size,
        MemoryRequestType type,
        const void* data,  // null for reads
        std::function<void()> callback);

    /// Report invariant violation
    void report_violation(const std::string& id, const std::string& msg,
                         uint8_t channel, uint8_t bank);
};

} // namespace sw::kpu
