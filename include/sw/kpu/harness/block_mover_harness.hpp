// ============================================================================
// include/sw/kpu/harness/block_mover_harness.hpp
// Test harness for BlockMover validation
//
// Tests L3->L2 tile movement with optional transformations:
// - Tile movement correctness
// - Transform operations (transpose, swizzle)
// - Credit flow L3->L2
// - Multi-BlockMover scheduling
// - Bank conflict handling
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/pattern_harness_base.hpp>
#include <sw/kpu/harness/tile_journey_tracker.hpp>
#include <sw/kpu/models/behavioral/datamovement/block_mover.hpp>
#include <sw/kpu/models/behavioral/memory/memory_model.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <queue>
#include <unordered_map>

namespace sw::kpu::harness {

// ============================================================================
// L2 Bank Array (Simulated)
// ============================================================================

/// Simple L2 bank array for testing
class L2BankArray {
public:
    explicit L2BankArray(uint32_t num_banks, uint64_t bank_size)
        : num_banks_(num_banks)
        , bank_size_(bank_size)
        , bank_occupied_(num_banks, false)
        , bank_tile_ids_(num_banks)
        , bank_data_(num_banks)
    {
        for (auto& bank : bank_data_) {
            bank.resize(bank_size, 0);
        }
    }

    /// Get number of free banks
    uint32_t available_credits() const {
        uint32_t count = 0;
        for (bool occupied : bank_occupied_) {
            if (!occupied) count++;
        }
        return count;
    }

    /// Allocate a bank
    uint32_t allocate() {
        for (uint32_t i = 0; i < num_banks_; ++i) {
            if (!bank_occupied_[i]) {
                bank_occupied_[i] = true;
                return i;
            }
        }
        return UINT32_MAX;
    }

    /// Reserve specific bank
    bool reserve(uint32_t bank_id, const TileID& tile_id) {
        if (bank_id >= num_banks_) return false;
        if (bank_occupied_[bank_id]) return false;
        bank_occupied_[bank_id] = true;
        bank_tile_ids_[bank_id] = tile_id;
        return true;
    }

    /// Release a bank
    void release(uint32_t bank_id) {
        if (bank_id < num_banks_) {
            bank_occupied_[bank_id] = false;
            bank_tile_ids_[bank_id] = TileID{};
        }
    }

    /// Check if bank is occupied
    bool is_occupied(uint32_t bank_id) const {
        return bank_id < num_banks_ && bank_occupied_[bank_id];
    }

    /// Get tile in bank
    TileID get_tile(uint32_t bank_id) const {
        if (bank_id < num_banks_) {
            return bank_tile_ids_[bank_id];
        }
        return TileID{};
    }

    /// Set tile ID for a bank (used after allocate())
    void set_tile(uint32_t bank_id, const TileID& tile_id) {
        if (bank_id < num_banks_) {
            bank_tile_ids_[bank_id] = tile_id;
        }
    }

    /// Write data to bank
    void write(uint32_t bank_id, uint64_t offset, const void* data, size_t size) {
        if (bank_id < num_banks_ && offset + size <= bank_size_) {
            std::memcpy(bank_data_[bank_id].data() + offset, data, size);
        }
    }

    /// Read data from bank
    void read(uint32_t bank_id, uint64_t offset, void* data, size_t size) const {
        if (bank_id < num_banks_ && offset + size <= bank_size_) {
            std::memcpy(data, bank_data_[bank_id].data() + offset, size);
        }
    }

    /// Get bank data for inspection
    const std::vector<uint8_t>& get_bank_data(uint32_t bank_id) const {
        static const std::vector<uint8_t> empty;
        if (bank_id < num_banks_) {
            return bank_data_[bank_id];
        }
        return empty;
    }

    uint64_t bank_size() const { return bank_size_; }
    uint32_t num_banks() const { return num_banks_; }

    void reset() {
        std::fill(bank_occupied_.begin(), bank_occupied_.end(), false);
        std::fill(bank_tile_ids_.begin(), bank_tile_ids_.end(), TileID{});
        for (auto& bank : bank_data_) {
            std::fill(bank.begin(), bank.end(), 0);
        }
    }

private:
    uint32_t num_banks_;
    uint64_t bank_size_;
    std::vector<bool> bank_occupied_;
    std::vector<TileID> bank_tile_ids_;
    std::vector<std::vector<uint8_t>> bank_data_;
};

// ============================================================================
// BlockMover Transfer Request
// ============================================================================

/// Request for L3->L2 or L2->L3 transfer
struct BlockMoverRequest {
    TileID tile_id;
    uint32_t l3_buffer_id = UINT32_MAX;
    uint32_t l2_bank_id = UINT32_MAX;
    uint64_t l3_offset = 0;
    uint64_t l2_offset = 0;
    uint32_t height = 16;
    uint32_t width = 16;
    uint32_t element_size = 2;  // FP16 default
    Transform transform = Transform::IDENTITY;
    bool l3_to_l2 = true;  // Direction: true=L3->L2, false=L2->L3
    std::function<void()> callback;
};

// ============================================================================
// BlockMover Harness Statistics
// ============================================================================

/// BlockMover-specific statistics
struct BlockMoverHarnessStats {
    // Transfer counts
    uint64_t tiles_moved = 0;
    uint64_t l3_to_l2_transfers = 0;
    uint64_t l2_to_l3_transfers = 0;

    // Data volume
    uint64_t bytes_moved = 0;

    // Transform counts
    std::unordered_map<Transform, uint64_t> transform_counts;

    // Timing
    Cycle total_transfer_cycles = 0;
    Cycle min_transfer_latency = UINT64_MAX;
    Cycle max_transfer_latency = 0;

    // Stalls
    Cycle credit_stall_cycles = 0;  // Waiting for L2 bank credit
    Cycle data_stall_cycles = 0;    // Waiting for tile in L3

    // Utilization
    Cycle busy_cycles = 0;
    Cycle idle_cycles = 0;

    // Bandwidth
    double l3_bandwidth_utilization = 0.0;
    double l2_bandwidth_utilization = 0.0;

    // Derived metrics
    double avg_transfer_latency() const {
        return tiles_moved > 0
            ? static_cast<double>(total_transfer_cycles) / tiles_moved
            : 0.0;
    }

    double utilization() const {
        Cycle total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / total : 0.0;
    }

    uint64_t get_transform_count(Transform xform) const {
        auto it = transform_counts.find(xform);
        return it != transform_counts.end() ? it->second : 0;
    }

    void reset() {
        tiles_moved = l3_to_l2_transfers = l2_to_l3_transfers = 0;
        bytes_moved = 0;
        transform_counts.clear();
        total_transfer_cycles = 0;
        min_transfer_latency = UINT64_MAX;
        max_transfer_latency = 0;
        credit_stall_cycles = data_stall_cycles = 0;
        busy_cycles = idle_cycles = 0;
        l3_bandwidth_utilization = l2_bandwidth_utilization = 0.0;
    }

    std::string to_string() const;
};

// ============================================================================
// BlockMover Harness
// ============================================================================

/// Test harness for BlockMover validation
class BlockMoverHarness : public PatternHarnessBase<BlockMoverHarnessConfig> {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    explicit BlockMoverHarness(const BlockMoverHarnessConfig& config);
    ~BlockMoverHarness() override = default;

    // ========================================================================
    // Simulation Control
    // ========================================================================

    bool run() override;
    void tick() override;
    void reset() override;
    bool has_pending() const override;

    // ========================================================================
    // L3 Preloading
    // ========================================================================

    /// Preload L3 buffer with tile data
    void preload_l3(uint32_t buffer_id, const TileID& tile_id,
                    const void* data, size_t size);

    /// Preload L3 buffer with pattern-generated data
    void preload_l3_pattern(uint32_t buffer_id, const TileID& tile_id,
                            uint32_t height, uint32_t width,
                            uint32_t element_size);

    // ========================================================================
    // Transfer Operations
    // ========================================================================

    /// Move tile from L3 to L2
    bool move_l3_to_l2(
        const TileID& tile_id,
        uint32_t l3_buffer_id,
        uint32_t l2_bank_id = UINT32_MAX,
        Transform transform = Transform::IDENTITY,
        std::function<void()> callback = nullptr);

    /// Move tile from L2 to L3
    bool move_l2_to_l3(
        const TileID& tile_id,
        uint32_t l2_bank_id,
        uint32_t l3_buffer_id,
        std::function<void()> callback = nullptr);

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Get L2 bank array
    const L2BankArray& l2_banks() const { return l2_banks_; }
    L2BankArray& l2_banks() { return l2_banks_; }

    /// Get credits available (free L2 banks)
    uint32_t credits_available() const {
        return l2_banks_.available_credits();
    }

    /// Check if tile is resident in L2
    bool tile_resident_l2(const TileID& tile_id) const;

    /// Read data from L2 bank
    std::vector<uint8_t> read_l2(uint32_t bank_id, uint64_t offset, size_t size) const;

    // ========================================================================
    // Statistics
    // ========================================================================

    const BlockMoverHarnessStats& block_mover_stats() const { return bm_stats_; }
    void print_stats(std::ostream& os = std::cout) const override;

    // ========================================================================
    // Tile Journey Tracking
    // ========================================================================

    const TileJourneyTracker& journey_tracker() const { return journey_tracker_; }
    TileJourneyTracker& journey_tracker() { return journey_tracker_; }

    // ========================================================================
    // Validation
    // ========================================================================

    ValidationResult validate() const override;

protected:
    void process_pending_requests();
    void execute_transfer(const BlockMoverRequest& request);

private:
    // Components
    std::vector<std::unique_ptr<behavioral::BehavioralBlockMover>> block_movers_;
    std::unique_ptr<behavioral::BehavioralMemoryModel> memory_model_;
    L2BankArray l2_banks_;

    // L3 buffer simulation (simplified)
    std::vector<std::vector<uint8_t>> l3_buffer_data_;
    std::vector<bool> l3_buffer_valid_;
    std::vector<TileID> l3_buffer_tiles_;

    // Request tracking
    std::queue<BlockMoverRequest> pending_queue_;

    // Statistics
    BlockMoverHarnessStats bm_stats_;

    // Journey tracking
    TileJourneyTracker journey_tracker_;
};

// ============================================================================
// BlockMoverHarnessStats Implementation
// ============================================================================

inline std::string BlockMoverHarnessStats::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "BlockMover Harness Statistics\n";
    ss << "==============================\n";
    ss << "Transfers:\n";
    ss << "  Tiles moved:   " << tiles_moved << "\n";
    ss << "  L3->L2:        " << l3_to_l2_transfers << "\n";
    ss << "  L2->L3:        " << l2_to_l3_transfers << "\n";
    ss << "\nData Volume:\n";
    ss << "  Bytes moved:   " << (bytes_moved / 1024.0) << " KB\n";
    ss << "\nTransforms:\n";
    ss << "  Transpose:     " << get_transform_count(Transform::TRANSPOSE) << "\n";
    ss << "  Swizzle 4x4:   " << get_transform_count(Transform::SWIZZLE_4x4) << "\n";
    ss << "  Swizzle 8x8:   " << get_transform_count(Transform::SWIZZLE_8x8) << "\n";
    ss << "\nLatency (cycles):\n";
    ss << "  Average:       " << avg_transfer_latency() << "\n";
    ss << "  Min:           " << (min_transfer_latency < UINT64_MAX ? min_transfer_latency : 0) << "\n";
    ss << "  Max:           " << max_transfer_latency << "\n";
    ss << "\nStalls:\n";
    ss << "  Credit stalls: " << credit_stall_cycles << " cycles\n";
    ss << "  Data stalls:   " << data_stall_cycles << " cycles\n";
    ss << "\nUtilization:     " << (utilization() * 100) << "%\n";
    return ss.str();
}

} // namespace sw::kpu::harness
