// ============================================================================
// include/sw/kpu/harness/dma_harness.hpp
// Test harness for DMA engine validation
//
// Tests DMA engines with credit-based flow to L3 buffers:
// - Single tile load/store operations
// - Burst transfers
// - Multi-engine concurrent access
// - Bank conflict detection
// - Credit stall cycle analysis
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/pattern_harness_base.hpp>
#include <sw/kpu/harness/tile_journey_tracker.hpp>
#include <sw/kpu/models/interfaces/dma_engine_interface.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <queue>

namespace sw::kpu::harness {

// ============================================================================
// DMA Transfer Request
// ============================================================================

/// Simplified request structure for harness use
struct DMARequest {
    TileID tile_id;                     // Tile being transferred
    uint64_t dram_address = 0;          // DRAM source/destination address
    uint32_t l3_buffer_id = 0;          // L3 buffer to use
    uint32_t size = 0;                  // Transfer size in bytes
    bool is_load = true;                // true=DRAM->L3, false=L3->DRAM
    std::function<void()> callback;     // Completion callback
};

// ============================================================================
// L3 Buffer Pool (Simulated)
// ============================================================================

/// Simple L3 buffer pool simulation for credit tracking
class L3BufferPool {
public:
    explicit L3BufferPool(uint32_t num_buffers, uint64_t buffer_size)
        : num_buffers_(num_buffers)
        , buffer_size_(buffer_size)
        , buffer_occupied_(num_buffers, false)
        , buffer_tile_ids_(num_buffers)
    {}

    /// Get number of available (free) buffers
    uint32_t available_credits() const {
        uint32_t count = 0;
        for (bool occupied : buffer_occupied_) {
            if (!occupied) count++;
        }
        return count;
    }

    /// Allocate a buffer, returns buffer ID or UINT32_MAX if none available
    uint32_t allocate() {
        for (uint32_t i = 0; i < num_buffers_; ++i) {
            if (!buffer_occupied_[i]) {
                buffer_occupied_[i] = true;
                return i;
            }
        }
        return UINT32_MAX;
    }

    /// Reserve a specific buffer
    bool reserve(uint32_t buffer_id, const TileID& tile_id) {
        if (buffer_id >= num_buffers_) return false;
        if (buffer_occupied_[buffer_id]) return false;
        buffer_occupied_[buffer_id] = true;
        buffer_tile_ids_[buffer_id] = tile_id;
        return true;
    }

    /// Release a buffer (return credit upstream)
    void release(uint32_t buffer_id) {
        if (buffer_id < num_buffers_) {
            buffer_occupied_[buffer_id] = false;
            buffer_tile_ids_[buffer_id] = TileID{};
        }
    }

    /// Check if a buffer is occupied
    bool is_occupied(uint32_t buffer_id) const {
        return buffer_id < num_buffers_ && buffer_occupied_[buffer_id];
    }

    /// Get tile in buffer
    TileID get_tile(uint32_t buffer_id) const {
        if (buffer_id < num_buffers_) {
            return buffer_tile_ids_[buffer_id];
        }
        return TileID{};
    }

    /// Get buffer size
    uint64_t buffer_size() const { return buffer_size_; }

    /// Get total number of buffers
    uint32_t num_buffers() const { return num_buffers_; }

    /// Reset all buffers
    void reset() {
        std::fill(buffer_occupied_.begin(), buffer_occupied_.end(), false);
        std::fill(buffer_tile_ids_.begin(), buffer_tile_ids_.end(), TileID{});
    }

private:
    uint32_t num_buffers_;
    uint64_t buffer_size_;
    std::vector<bool> buffer_occupied_;
    std::vector<TileID> buffer_tile_ids_;
};

// ============================================================================
// DMA Harness Statistics
// ============================================================================

/// DMA-specific statistics
struct DMAHarnessStats {
    // Transfer counts
    uint64_t tiles_loaded = 0;          // DRAM -> L3
    uint64_t tiles_stored = 0;          // L3 -> DRAM
    uint64_t transfers_completed = 0;
    uint64_t transfers_failed = 0;

    // Data volume
    uint64_t bytes_transferred = 0;
    uint64_t bytes_loaded = 0;          // DRAM -> L3
    uint64_t bytes_stored = 0;          // L3 -> DRAM

    // Timing
    Cycle total_transfer_cycles = 0;
    Cycle min_transfer_latency = UINT64_MAX;
    Cycle max_transfer_latency = 0;

    // Stalls
    Cycle credit_stall_cycles = 0;      // Waiting for L3 buffer credit
    Cycle memory_stall_cycles = 0;      // Waiting for memory controller

    // Utilization
    Cycle busy_cycles = 0;
    Cycle idle_cycles = 0;

    // Derived metrics
    double avg_transfer_latency() const {
        return transfers_completed > 0
            ? static_cast<double>(total_transfer_cycles) / static_cast<double>(transfers_completed)
            : 0.0;
    }

    double bandwidth_utilization(double peak_gbps, double clock_ghz, Cycle total_cycles) const {
        if (total_cycles == 0 || peak_gbps <= 0) return 0.0;
        double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        double actual_gbps = (static_cast<double>(bytes_transferred) / 1e9) / seconds;
        return actual_gbps / peak_gbps;
    }

    double utilization() const {
        Cycle total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    /// Reset all statistics
    void reset() {
        tiles_loaded = tiles_stored = 0;
        transfers_completed = transfers_failed = 0;
        bytes_transferred = bytes_loaded = bytes_stored = 0;
        total_transfer_cycles = 0;
        min_transfer_latency = UINT64_MAX;
        max_transfer_latency = 0;
        credit_stall_cycles = memory_stall_cycles = 0;
        busy_cycles = idle_cycles = 0;
    }

    /// Generate summary string
    std::string to_string() const;
};

// ============================================================================
// DMA Harness
// ============================================================================

/// Test harness for DMA engines
///
/// Provides isolated testing of DMA functionality including:
/// - Credit-based flow control with L3 buffer pool
/// - Multi-engine operation
/// - Statistics and tracing
///
class DMAHarness : public PatternHarnessBase<DMAHarnessConfig> {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    explicit DMAHarness(const DMAHarnessConfig& config);
    ~DMAHarness() override = default;

    // ========================================================================
    // Simulation Control (PatternHarnessBase interface)
    // ========================================================================

    /// Run simulation until all pending transfers complete
    bool run() override;

    /// Advance simulation by one cycle
    void tick() override;

    /// Reset harness to initial state
    void reset() override;

    /// Check if there are pending transfers
    bool has_pending() const override;

    // ========================================================================
    // Request Interface
    // ========================================================================

    /// Request a tile load (DRAM -> L3)
    /// @param tile_id Tile identifier
    /// @param dram_address Source address in DRAM
    /// @param l3_buffer_id Target L3 buffer (or UINT32_MAX for auto-allocate)
    /// @param callback Optional completion callback
    /// @return true if request was accepted
    bool request_tile_load(
        const TileID& tile_id,
        uint64_t dram_address,
        uint32_t l3_buffer_id = UINT32_MAX,
        std::function<void()> callback = nullptr);

    /// Request a tile store (L3 -> DRAM)
    bool request_tile_store(
        const TileID& tile_id,
        uint32_t l3_buffer_id,
        uint64_t dram_address,
        std::function<void()> callback = nullptr);

    /// Request a batch of tile loads
    size_t request_tile_loads(
        const std::vector<std::pair<TileID, uint64_t>>& requests);

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Get number of tiles transferred (loaded + stored)
    uint64_t tiles_transferred() const {
        return dma_stats_.tiles_loaded + dma_stats_.tiles_stored;
    }

    /// Get available credits (free L3 buffers)
    uint32_t credits_available() const {
        return l3_buffers_.available_credits();
    }

    /// Get L3 buffer pool
    const L3BufferPool& l3_buffers() const { return l3_buffers_; }
    L3BufferPool& l3_buffers() { return l3_buffers_; }

    /// Get number of pending requests
    size_t pending_requests() const { return pending_queue_.size(); }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get DMA-specific statistics
    const DMAHarnessStats& dma_stats() const { return dma_stats_; }

    /// Print DMA statistics
    void print_stats(std::ostream& os = std::cout) const override;

    // ========================================================================
    // Tile Journey Tracking
    // ========================================================================

    /// Get tile journey tracker
    const TileJourneyTracker& journey_tracker() const { return journey_tracker_; }
    TileJourneyTracker& journey_tracker() { return journey_tracker_; }

    // ========================================================================
    // Validation
    // ========================================================================

    /// Validate simulation results
    ValidationResult validate() const override;

    /// Verify expected statistics
    bool verify_stats(uint64_t expected_loads, uint64_t expected_stores) const;

protected:
    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /// Process pending requests that can be issued
    void process_pending_requests();

    /// Update statistics from DMA engine
    void update_stats_from_engine(size_t engine_id);

    /// Handle transfer completion
    void on_transfer_complete(const TileID& tile_id, bool is_load, uint32_t size);

private:
    // Component instances
    std::vector<std::unique_ptr<IDMAEngine>> dma_engines_;
    L3BufferPool l3_buffers_;

    // Request tracking
    std::queue<DMARequest> pending_queue_;
    std::unordered_map<uint64_t, DMARequest> in_flight_requests_;
    uint64_t next_transfer_id_ = 1;

    // Statistics
    DMAHarnessStats dma_stats_;

    // Tile tracking
    TileJourneyTracker journey_tracker_;

    // State tracking
    bool was_busy_last_cycle_ = false;
};

// ============================================================================
// DMAHarnessStats Implementation
// ============================================================================

inline std::string DMAHarnessStats::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "DMA Harness Statistics\n";
    ss << "======================\n";
    ss << "Transfers:\n";
    ss << "  Tiles loaded:  " << tiles_loaded << "\n";
    ss << "  Tiles stored:  " << tiles_stored << "\n";
    ss << "  Completed:     " << transfers_completed << "\n";
    ss << "  Failed:        " << transfers_failed << "\n";
    ss << "\nData Volume:\n";
    ss << "  Bytes loaded:  " << (static_cast<double>(bytes_loaded) / 1024.0) << " KB\n";
    ss << "  Bytes stored:  " << (static_cast<double>(bytes_stored) / 1024.0) << " KB\n";
    ss << "  Total:         " << (static_cast<double>(bytes_transferred) / 1024.0) << " KB\n";
    ss << "\nLatency (cycles):\n";
    ss << "  Average: " << avg_transfer_latency() << "\n";
    ss << "  Min:     " << (min_transfer_latency < UINT64_MAX ? min_transfer_latency : 0) << "\n";
    ss << "  Max:     " << max_transfer_latency << "\n";
    ss << "\nStalls:\n";
    ss << "  Credit stalls: " << credit_stall_cycles << " cycles\n";
    ss << "  Memory stalls: " << memory_stall_cycles << " cycles\n";
    ss << "\nUtilization: " << (utilization() * 100) << "%\n";
    return ss.str();
}

} // namespace sw::kpu::harness
