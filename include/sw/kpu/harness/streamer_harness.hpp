// ============================================================================
// include/sw/kpu/harness/streamer_harness.hpp
// Test harness for Streamer validation
//
// Tests L2->L1 streaming and compute integration:
// - Row/column streaming correctness
// - Compute integration (matmul fire on data arrival)
// - Drain and writeback
// - Vector engine operations
// - Double-buffering
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/pattern_harness_base.hpp>
#include <sw/kpu/harness/tile_journey_tracker.hpp>
#include <sw/kpu/harness/block_mover_harness.hpp>  // For L2BankArray

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <queue>

namespace sw::kpu::harness {

// ============================================================================
// L1 Buffer Array (Simulated)
// ============================================================================

/// Simple L1 buffer array for streaming
class L1BufferArray {
public:
    explicit L1BufferArray(uint32_t depth, uint32_t buffer_size)
        : depth_(depth)
        , buffer_size_(buffer_size)
        , buffers_(depth)
        , buffer_valid_(depth, false)
    {
        for (auto& buf : buffers_) {
            buf.resize(buffer_size, 0);
        }
    }

    /// Get buffer depth (pipeline stages)
    uint32_t depth() const { return depth_; }

    /// Get buffer size
    uint32_t buffer_size() const { return buffer_size_; }

    /// Write to buffer
    void write(uint32_t buffer_id, const void* data, size_t size) {
        if (buffer_id < depth_ && size <= buffer_size_) {
            std::memcpy(buffers_[buffer_id].data(), data, size);
            buffer_valid_[buffer_id] = true;
        }
    }

    /// Read from buffer
    void read(uint32_t buffer_id, void* data, size_t size) const {
        if (buffer_id < depth_ && size <= buffer_size_) {
            std::memcpy(data, buffers_[buffer_id].data(), size);
        }
    }

    /// Check if buffer has valid data
    bool is_valid(uint32_t buffer_id) const {
        return buffer_id < depth_ && buffer_valid_[buffer_id];
    }

    /// Invalidate buffer (consumed by compute)
    void invalidate(uint32_t buffer_id) {
        if (buffer_id < depth_) {
            buffer_valid_[buffer_id] = false;
        }
    }

    /// Get buffer data
    const std::vector<uint8_t>& get_data(uint32_t buffer_id) const {
        static const std::vector<uint8_t> empty;
        if (buffer_id < depth_) return buffers_[buffer_id];
        return empty;
    }

    /// Reset all buffers
    void reset() {
        std::fill(buffer_valid_.begin(), buffer_valid_.end(), false);
        for (auto& buf : buffers_) {
            std::fill(buf.begin(), buf.end(), 0);
        }
    }

private:
    uint32_t depth_;
    uint32_t buffer_size_;
    std::vector<std::vector<uint8_t>> buffers_;
    std::vector<bool> buffer_valid_;
};

// ============================================================================
// Accumulator (Simulated Compute Result)
// ============================================================================

/// Simple accumulator for compute results
class Accumulator {
public:
    explicit Accumulator(uint32_t size = 16 * 16)
        : size_(size)
        , data_(size, 0.0f)
        , valid_(false)
    {}

    /// Accumulate values (MAC operation)
    void accumulate(const float* a, const float* b, uint32_t count) {
        for (uint32_t i = 0; i < std::min(count, size_); ++i) {
            data_[i] += a[i] * b[i];
        }
        valid_ = true;
    }

    /// Get accumulated data
    const std::vector<float>& data() const { return data_; }

    /// Check if accumulator has valid data
    bool is_valid() const { return valid_; }

    /// Clear accumulator
    void clear() {
        std::fill(data_.begin(), data_.end(), 0.0f);
        valid_ = false;
    }

    /// Get size
    uint32_t size() const { return size_; }

private:
    uint32_t size_;
    std::vector<float> data_;
    bool valid_;
};

// ============================================================================
// Stream Request
// ============================================================================

/// Streamer operation request
struct StreamRequest {
    enum class Type {
        ROW_STREAM,     // Stream rows (A operand)
        COL_STREAM,     // Stream columns (B operand)
        BROADCAST,      // Broadcast value (bias/scalar)
        DRAIN           // Drain accumulator to L2
    };

    TileID tile_id;
    Type type = Type::ROW_STREAM;
    uint32_t l2_bank_id = UINT32_MAX;
    uint32_t l1_buffer_id = 0;
    uint64_t l2_offset = 0;
    uint32_t height = 16;
    uint32_t width = 16;
    uint32_t element_size = 2;
    std::function<void()> callback;
};

// ============================================================================
// Streamer Harness Statistics
// ============================================================================

/// Streamer-specific statistics
struct StreamerHarnessStats {
    // Operation counts
    uint64_t rows_streamed = 0;
    uint64_t cols_streamed = 0;
    uint64_t broadcasts = 0;
    uint64_t drains = 0;

    // Data volume
    uint64_t bytes_l2_to_l1 = 0;
    uint64_t bytes_l1_to_l2 = 0;

    // Compute operations (triggered by streaming)
    uint64_t compute_operations = 0;

    // Timing
    Cycle total_stream_cycles = 0;
    Cycle compute_cycles = 0;

    // Stalls
    Cycle credit_stall_cycles = 0;
    Cycle data_stall_cycles = 0;

    // Utilization
    Cycle busy_cycles = 0;
    Cycle idle_cycles = 0;

    // Bandwidth
    double l2_read_bandwidth = 0.0;
    double l1_write_bandwidth = 0.0;

    double utilization() const {
        Cycle total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    void reset() {
        rows_streamed = cols_streamed = broadcasts = drains = 0;
        bytes_l2_to_l1 = bytes_l1_to_l2 = 0;
        compute_operations = 0;
        total_stream_cycles = compute_cycles = 0;
        credit_stall_cycles = data_stall_cycles = 0;
        busy_cycles = idle_cycles = 0;
        l2_read_bandwidth = l1_write_bandwidth = 0.0;
    }

    std::string to_string() const;
};

// ============================================================================
// Streamer Harness
// ============================================================================

/// Test harness for Streamer validation
class StreamerHarness : public PatternHarnessBase<StreamerHarnessConfig> {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    explicit StreamerHarness(const StreamerHarnessConfig& config);
    ~StreamerHarness() override = default;

    // ========================================================================
    // Simulation Control
    // ========================================================================

    bool run() override;
    void tick() override;
    void reset() override;
    bool has_pending() const override;

    // ========================================================================
    // L2 Preloading
    // ========================================================================

    /// Preload L2 bank with tile data
    void preload_l2(uint32_t bank_id, const TileID& tile_id,
                    const void* data, size_t size);

    /// Preload L2 bank with pattern
    void preload_l2_pattern(uint32_t bank_id, const TileID& tile_id,
                            uint32_t height, uint32_t width,
                            uint32_t element_size);

    // ========================================================================
    // Stream Operations
    // ========================================================================

    /// Stream rows from L2 to L1 (A operand)
    bool stream_rows(const TileID& tile_id, uint32_t l2_bank_id,
                     std::function<void()> callback = nullptr);

    /// Stream columns from L2 to L1 (B operand)
    bool stream_cols(const TileID& tile_id, uint32_t l2_bank_id,
                     std::function<void()> callback = nullptr);

    /// Broadcast value from L2 (bias/scalar)
    bool broadcast(const TileID& tile_id, uint32_t l2_bank_id,
                   std::function<void()> callback = nullptr);

    /// Drain accumulator to L2
    bool drain(const TileID& tile_id, uint32_t l2_bank_id,
               std::function<void()> callback = nullptr);

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Get L2 bank array
    const L2BankArray& l2_banks() const { return l2_banks_; }
    L2BankArray& l2_banks() { return l2_banks_; }

    /// Get L1 buffer array
    const L1BufferArray& l1_buffers() const { return l1_buffers_; }
    L1BufferArray& l1_buffers() { return l1_buffers_; }

    /// Get accumulator
    const Accumulator& accumulator() const { return accumulator_; }
    Accumulator& accumulator() { return accumulator_; }

    /// Read drain buffer
    std::vector<float> read_drain_buffer() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    const StreamerHarnessStats& streamer_stats() const { return str_stats_; }
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
    void execute_stream(const StreamRequest& request);

private:
    // Memory
    L2BankArray l2_banks_;
    L1BufferArray l1_buffers_;

    // Compute
    Accumulator accumulator_;
    std::vector<float> drain_buffer_;

    // Request tracking
    std::queue<StreamRequest> pending_queue_;

    // Statistics
    StreamerHarnessStats str_stats_;

    // Journey tracking
    TileJourneyTracker journey_tracker_;
};

// ============================================================================
// StreamerHarnessStats Implementation
// ============================================================================

inline std::string StreamerHarnessStats::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Streamer Harness Statistics\n";
    ss << "============================\n";
    ss << "Operations:\n";
    ss << "  Rows streamed: " << rows_streamed << "\n";
    ss << "  Cols streamed: " << cols_streamed << "\n";
    ss << "  Broadcasts:    " << broadcasts << "\n";
    ss << "  Drains:        " << drains << "\n";
    ss << "\nData Volume:\n";
    ss << "  L2->L1:        " << (static_cast<double>(bytes_l2_to_l1) / 1024.0) << " KB\n";
    ss << "  L1->L2:        " << (static_cast<double>(bytes_l1_to_l2) / 1024.0) << " KB\n";
    ss << "\nCompute:\n";
    ss << "  Operations:    " << compute_operations << "\n";
    ss << "  Cycles:        " << compute_cycles << "\n";
    ss << "\nStalls:\n";
    ss << "  Credit stalls: " << credit_stall_cycles << " cycles\n";
    ss << "  Data stalls:   " << data_stall_cycles << " cycles\n";
    ss << "\nUtilization:     " << (utilization() * 100) << "%\n";
    return ss.str();
}

} // namespace sw::kpu::harness
