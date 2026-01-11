// ============================================================================
// include/sw/kpu/models/transactional/memory/l3_tile.hpp
// Transactional L3 tile - bank contention and access latency
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <queue>
#include <vector>

#include <sw/kpu/models/interfaces/l3_tile_interface.hpp>

namespace sw::kpu {

/// Transactional L3 tile with bank contention modeling
///
/// This is a middle-ground simulation mode, providing:
/// - Bank-level contention modeling
/// - Configurable access latency
/// - Basic port arbitration
/// - ~10-100x faster than cycle-accurate
///
/// Use for:
/// - Architecture exploration
/// - Workload characterization
/// - Performance estimation
/// - Bottleneck identification
class TransactionalL3Tile : public IL3Tile {
public:
    explicit TransactionalL3Tile(const L3TileConfig& config, uint32_t tile_id);
    ~TransactionalL3Tile() override = default;

    // Disable copying, allow moving
    TransactionalL3Tile(const TransactionalL3Tile&) = delete;
    TransactionalL3Tile& operator=(const TransactionalL3Tile&) = delete;
    TransactionalL3Tile(TransactionalL3Tile&&) = default;
    TransactionalL3Tile& operator=(TransactionalL3Tile&&) = default;

    // ========================================================================
    // Basic Memory Operations
    // ========================================================================

    std::optional<uint64_t> read(
        uint64_t addr,
        void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> write(
        uint64_t addr,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    // ========================================================================
    // Block Operations
    // ========================================================================

    std::optional<uint64_t> read_block(
        uint64_t base_addr,
        void* data,
        uint32_t height,
        uint32_t width,
        uint32_t element_size,
        uint32_t stride = 0,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> write_block(
        uint64_t base_addr,
        const void* data,
        uint32_t height,
        uint32_t width,
        uint32_t element_size,
        uint32_t stride = 0,
        std::function<void()> callback = nullptr) override;

    // ========================================================================
    // Status Queries
    // ========================================================================

    bool can_accept() const override;
    bool has_pending() const override { return pending_count_ > 0; }
    size_t pending_count() const override { return pending_count_; }
    bool is_ready() const override { return true; }

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
    const L3TileConfig& config() const override { return config_; }
    uint32_t tile_id() const override { return tile_id_; }
    uint64_t capacity() const override { return capacity_; }
    uint8_t num_banks() const override { return config_.num_banks; }
    uint8_t num_ports() const override { return config_.num_ports; }

    // ========================================================================
    // Bank State Queries
    // ========================================================================

    BankState get_bank_state(uint8_t bank) const override;
    bool is_bank_busy(uint8_t bank) const override;

    // ========================================================================
    // Statistics
    // ========================================================================

    const L3TileStatistics& stats() const override { return stats_; }
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
    // Direct Memory Access
    // ========================================================================

    void read_direct(uint64_t addr, void* data, uint32_t size) const override;
    void write_direct(uint64_t addr, const void* data, uint32_t size) override;

private:
    // Configuration
    L3TileConfig config_;
    uint32_t tile_id_;
    uint64_t capacity_;
    size_t pending_count_ = 0;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 1;

    // Backing store
    std::vector<uint8_t> memory_;

    // Per-bank state
    struct BankInfo {
        uint64_t busy_until_cycle = 0;
        BankState state = BankState::IDLE;

        bool is_idle(uint64_t cycle) const { return cycle >= busy_until_cycle; }
    };
    std::vector<BankInfo> banks_;

    // Request tracking
    struct PendingRequest {
        uint64_t request_id;
        uint64_t completion_cycle;
        uint64_t submit_cycle;
        uint8_t bank;
        bool is_write;
        std::function<void()> callback;

        bool operator>(const PendingRequest& other) const {
            return completion_cycle > other.completion_cycle;
        }
    };
    std::priority_queue<PendingRequest,
                       std::vector<PendingRequest>,
                       std::greater<PendingRequest>> completion_queue_;

    // Statistics
    L3TileStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Helpers
    uint8_t address_to_bank(uint64_t addr) const;
    bool validate_address(uint64_t addr, uint32_t size) const;
    uint64_t schedule_bank_access(uint8_t bank, bool is_write);
};

} // namespace sw::kpu
