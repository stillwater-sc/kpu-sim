// ============================================================================
// include/sw/kpu/models/behavioral/memory/l3_tile.hpp
// Behavioral (functional) L3 tile - instant access
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

#include <sw/kpu/models/interfaces/l3_tile_interface.hpp>

namespace sw::kpu {

/// Behavioral L3 tile with instant or fixed latency
///
/// This is the fastest simulation mode, providing:
/// - Functional correctness (data is stored and retrieved correctly)
/// - Optional fixed latency modeling
/// - No bank conflicts, no port arbitration
///
/// Use for:
/// - Software bring-up and functional verification
/// - Unit testing
/// - CI/CD pipelines where speed matters more than timing accuracy
class BehavioralL3Tile : public IL3Tile {
public:
    explicit BehavioralL3Tile(const L3TileConfig& config, uint32_t tile_id);
    ~BehavioralL3Tile() override = default;

    // Disable copying, allow moving
    BehavioralL3Tile(const BehavioralL3Tile&) = delete;
    BehavioralL3Tile& operator=(const BehavioralL3Tile&) = delete;
    BehavioralL3Tile(BehavioralL3Tile&&) = default;
    BehavioralL3Tile& operator=(BehavioralL3Tile&&) = default;

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

    bool can_accept() const override { return true; }  // Always accepts
    bool has_pending() const override { return !pending_callbacks_.empty(); }
    size_t pending_count() const override { return pending_callbacks_.size(); }
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

    SimulationFidelity fidelity() const override { return SimulationFidelity::BEHAVIORAL; }
    const L3TileConfig& config() const override { return config_; }
    uint32_t tile_id() const override { return tile_id_; }
    uint64_t capacity() const override { return capacity_; }
    uint8_t num_banks() const override { return config_.num_banks; }
    uint8_t num_ports() const override { return config_.num_ports; }

    // ========================================================================
    // Bank State Queries
    // ========================================================================

    BankState get_bank_state(uint8_t bank) const override {
        (void)bank;
        return BankState::IDLE;  // Always idle in behavioral
    }

    bool is_bank_busy(uint8_t bank) const override {
        (void)bank;
        return false;
    }

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

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 1;

    // Backing store
    std::vector<uint8_t> memory_;

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
    L3TileStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Helper to validate address
    bool validate_address(uint64_t addr, uint32_t size) const;
};

} // namespace sw::kpu
