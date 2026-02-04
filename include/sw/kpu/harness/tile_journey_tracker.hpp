// ============================================================================
// include/sw/kpu/harness/tile_journey_tracker.hpp
// Track tile movement timing through the DRAM->L3->L2->L1->Compute pipeline
//
// Records precise timing for each tile as it moves through the memory hierarchy,
// enabling detailed latency analysis and bottleneck identification.
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/harness_config.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace sw::kpu::harness {

using Cycle = uint64_t;

// ============================================================================
// Tile Journey Record
// ============================================================================

/// Complete timing record for a single tile's journey through the pipeline
struct TileJourney {
    // === Tile Identification ===
    TileID tile_id;

    // === DRAM Phase Timing ===
    Cycle dram_request_cycle = 0;       // When DMA request was issued
    Cycle dram_fetch_start = 0;         // When DRAM access started
    Cycle dram_fetch_complete = 0;      // When DRAM read completed

    // === L3 Phase Timing ===
    Cycle l3_arrival = 0;               // When tile arrived in L3 buffer
    Cycle l3_departure = 0;             // When tile left L3 for L2

    // === L2 Phase Timing ===
    Cycle l2_arrival = 0;               // When tile arrived in L2 bank
    Cycle l2_departure = 0;             // When tile left L2 for L1

    // === L1 Phase Timing ===
    Cycle l1_arrival = 0;               // When tile arrived in L1 buffer
    Cycle l1_departure = 0;             // When tile was consumed by compute

    // === Compute Phase Timing ===
    Cycle compute_start = 0;            // When compute operation started
    Cycle compute_end = 0;              // When compute operation completed

    // === Drain Phase Timing (for output tiles) ===
    Cycle drain_start = 0;              // When drain to L2 started
    Cycle drain_complete = 0;           // When drain completed
    Cycle dram_writeback_start = 0;     // When DRAM write started
    Cycle dram_writeback_complete = 0;  // When DRAM write completed

    // === Buffer/Location Tracking ===
    uint32_t l3_buffer_id = UINT32_MAX;
    uint32_t l2_bank_id = UINT32_MAX;
    uint32_t l1_buffer_id = UINT32_MAX;

    // === Stall Tracking ===
    Cycle credit_stall_cycles = 0;      // Cycles stalled waiting for credit
    Cycle data_stall_cycles = 0;        // Cycles stalled waiting for data
    Cycle memory_stall_cycles = 0;      // Cycles stalled in memory

    // ========================================================================
    // Computed Metrics
    // ========================================================================

    /// Total end-to-end latency (request to compute complete)
    Cycle total_latency() const {
        if (compute_end > 0 && dram_request_cycle > 0) {
            return compute_end - dram_request_cycle;
        }
        return 0;
    }

    /// Total data movement latency (DRAM to L1)
    Cycle data_movement_latency() const {
        if (l1_arrival > 0 && dram_request_cycle > 0) {
            return l1_arrival - dram_request_cycle;
        }
        return 0;
    }

    /// Compute latency only
    Cycle compute_latency() const {
        if (compute_end > 0 && compute_start > 0) {
            return compute_end - compute_start;
        }
        return 0;
    }

    /// DRAM fetch latency
    Cycle dram_latency() const {
        if (dram_fetch_complete > 0 && dram_fetch_start > 0) {
            return dram_fetch_complete - dram_fetch_start;
        }
        return 0;
    }

    /// L3 residence time
    Cycle l3_residence_time() const {
        if (l3_departure > 0 && l3_arrival > 0) {
            return l3_departure - l3_arrival;
        }
        return 0;
    }

    /// L2 residence time
    Cycle l2_residence_time() const {
        if (l2_departure > 0 && l2_arrival > 0) {
            return l2_departure - l2_arrival;
        }
        return 0;
    }

    /// Total stall time
    Cycle total_stall_cycles() const {
        return credit_stall_cycles + data_stall_cycles + memory_stall_cycles;
    }

    /// Check if journey is complete (reached compute)
    bool is_complete() const {
        return compute_end > 0;
    }

    /// Check if this is an output tile (has drain phase)
    bool is_output_tile() const {
        return drain_start > 0 || dram_writeback_start > 0;
    }

    /// Generate summary string
    std::string summary() const {
        std::ostringstream ss;
        ss << tile_id.to_string() << ": ";
        ss << "total=" << total_latency() << "cyc, ";
        ss << "data_move=" << data_movement_latency() << "cyc, ";
        ss << "compute=" << compute_latency() << "cyc, ";
        ss << "stalls=" << total_stall_cycles() << "cyc";
        return ss.str();
    }
};

// ============================================================================
// Tile Journey Statistics
// ============================================================================

/// Aggregate statistics across all tracked tile journeys
struct TileJourneyStats {
    // === Counts ===
    uint64_t total_tiles = 0;
    uint64_t completed_tiles = 0;
    uint64_t input_tiles = 0;
    uint64_t output_tiles = 0;

    // === Latency Statistics ===
    Cycle min_total_latency = UINT64_MAX;
    Cycle max_total_latency = 0;
    Cycle sum_total_latency = 0;

    Cycle min_data_movement_latency = UINT64_MAX;
    Cycle max_data_movement_latency = 0;
    Cycle sum_data_movement_latency = 0;

    Cycle min_compute_latency = UINT64_MAX;
    Cycle max_compute_latency = 0;
    Cycle sum_compute_latency = 0;

    // === Stall Statistics ===
    Cycle total_credit_stalls = 0;
    Cycle total_data_stalls = 0;
    Cycle total_memory_stalls = 0;

    // === Per-Stage Latency Sums ===
    Cycle sum_dram_latency = 0;
    Cycle sum_l3_residence = 0;
    Cycle sum_l2_residence = 0;

    // === Computed Averages ===
    double avg_total_latency() const {
        return completed_tiles > 0 ? static_cast<double>(sum_total_latency) / completed_tiles : 0.0;
    }

    double avg_data_movement_latency() const {
        return completed_tiles > 0 ? static_cast<double>(sum_data_movement_latency) / completed_tiles : 0.0;
    }

    double avg_compute_latency() const {
        return completed_tiles > 0 ? static_cast<double>(sum_compute_latency) / completed_tiles : 0.0;
    }

    double avg_dram_latency() const {
        return completed_tiles > 0 ? static_cast<double>(sum_dram_latency) / completed_tiles : 0.0;
    }

    /// Generate summary string
    std::string to_string() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Tile Journey Statistics\n";
        ss << "========================\n";
        ss << "Tiles: " << completed_tiles << "/" << total_tiles << " completed\n";
        ss << "  Input tiles:  " << input_tiles << "\n";
        ss << "  Output tiles: " << output_tiles << "\n";
        ss << "\nLatency (cycles):\n";
        ss << "  Total:     avg=" << avg_total_latency()
           << " min=" << (min_total_latency < UINT64_MAX ? min_total_latency : 0)
           << " max=" << max_total_latency << "\n";
        ss << "  DataMove:  avg=" << avg_data_movement_latency()
           << " min=" << (min_data_movement_latency < UINT64_MAX ? min_data_movement_latency : 0)
           << " max=" << max_data_movement_latency << "\n";
        ss << "  Compute:   avg=" << avg_compute_latency()
           << " min=" << (min_compute_latency < UINT64_MAX ? min_compute_latency : 0)
           << " max=" << max_compute_latency << "\n";
        ss << "\nStalls (total cycles):\n";
        ss << "  Credit: " << total_credit_stalls << "\n";
        ss << "  Data:   " << total_data_stalls << "\n";
        ss << "  Memory: " << total_memory_stalls << "\n";
        return ss.str();
    }
};

// ============================================================================
// Tile Journey Tracker
// ============================================================================

/// Tracks timing for tiles moving through the pipeline
///
/// Usage:
///   1. Call begin_journey() when a tile is first requested
///   2. Call record_*() methods as tile reaches each stage
///   3. Call end_journey() or mark_compute_complete() when done
///   4. Use get_journey() or get_statistics() for analysis
///
class TileJourneyTracker {
public:
    TileJourneyTracker() = default;
    ~TileJourneyTracker() = default;

    // Non-copyable, movable
    TileJourneyTracker(const TileJourneyTracker&) = delete;
    TileJourneyTracker& operator=(const TileJourneyTracker&) = delete;
    TileJourneyTracker(TileJourneyTracker&&) = default;
    TileJourneyTracker& operator=(TileJourneyTracker&&) = default;

    // ========================================================================
    // Journey Lifecycle
    // ========================================================================

    /// Begin tracking a new tile journey
    /// @param tile_id Unique tile identifier
    /// @param cycle Current simulation cycle
    /// @return Reference to the journey record
    TileJourney& begin_journey(const TileID& tile_id, Cycle cycle) {
        auto& journey = journeys_[tile_id];
        journey.tile_id = tile_id;
        journey.dram_request_cycle = cycle;
        return journey;
    }

    /// Check if a journey exists for a tile
    bool has_journey(const TileID& tile_id) const {
        return journeys_.find(tile_id) != journeys_.end();
    }

    /// Get a journey record (creates if not exists)
    TileJourney& get_or_create_journey(const TileID& tile_id) {
        return journeys_[tile_id];
    }

    /// Get a journey record (optional, returns nullopt if not found)
    std::optional<std::reference_wrapper<const TileJourney>>
    get_journey(const TileID& tile_id) const {
        auto it = journeys_.find(tile_id);
        if (it != journeys_.end()) {
            return std::cref(it->second);
        }
        return std::nullopt;
    }

    // ========================================================================
    // DRAM Phase Recording
    // ========================================================================

    void record_dram_fetch_start(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.dram_fetch_start = cycle;
        }
    }

    void record_dram_fetch_complete(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.dram_fetch_complete = cycle;
        }
    }

    // ========================================================================
    // L3 Phase Recording
    // ========================================================================

    void record_l3_arrival(const TileID& tile_id, Cycle cycle, uint32_t buffer_id = UINT32_MAX) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l3_arrival = cycle;
            if (buffer_id != UINT32_MAX) {
                it->second.l3_buffer_id = buffer_id;
            }
        }
    }

    void record_l3_departure(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l3_departure = cycle;
        }
    }

    // ========================================================================
    // L2 Phase Recording
    // ========================================================================

    void record_l2_arrival(const TileID& tile_id, Cycle cycle, uint32_t bank_id = UINT32_MAX) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l2_arrival = cycle;
            if (bank_id != UINT32_MAX) {
                it->second.l2_bank_id = bank_id;
            }
        }
    }

    void record_l2_departure(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l2_departure = cycle;
        }
    }

    // ========================================================================
    // L1 Phase Recording
    // ========================================================================

    void record_l1_arrival(const TileID& tile_id, Cycle cycle, uint32_t buffer_id = UINT32_MAX) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l1_arrival = cycle;
            if (buffer_id != UINT32_MAX) {
                it->second.l1_buffer_id = buffer_id;
            }
        }
    }

    void record_l1_departure(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.l1_departure = cycle;
        }
    }

    // ========================================================================
    // Compute Phase Recording
    // ========================================================================

    void record_compute_start(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.compute_start = cycle;
        }
    }

    void record_compute_end(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.compute_end = cycle;
        }
    }

    // ========================================================================
    // Drain Phase Recording (for output tiles)
    // ========================================================================

    void record_drain_start(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.drain_start = cycle;
        }
    }

    void record_drain_complete(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.drain_complete = cycle;
        }
    }

    void record_dram_writeback_start(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.dram_writeback_start = cycle;
        }
    }

    void record_dram_writeback_complete(const TileID& tile_id, Cycle cycle) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.dram_writeback_complete = cycle;
        }
    }

    // ========================================================================
    // Stall Recording
    // ========================================================================

    void record_credit_stall(const TileID& tile_id, Cycle cycles) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.credit_stall_cycles += cycles;
        }
    }

    void record_data_stall(const TileID& tile_id, Cycle cycles) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.data_stall_cycles += cycles;
        }
    }

    void record_memory_stall(const TileID& tile_id, Cycle cycles) {
        if (auto it = journeys_.find(tile_id); it != journeys_.end()) {
            it->second.memory_stall_cycles += cycles;
        }
    }

    // ========================================================================
    // Analysis
    // ========================================================================

    /// Get all tracked journeys
    const std::unordered_map<TileID, TileJourney, TileIDHash>& journeys() const {
        return journeys_;
    }

    /// Get journeys as a sorted vector (by request cycle)
    std::vector<TileJourney> get_sorted_journeys() const {
        std::vector<TileJourney> result;
        result.reserve(journeys_.size());
        for (const auto& [id, journey] : journeys_) {
            result.push_back(journey);
        }
        std::sort(result.begin(), result.end(),
            [](const TileJourney& a, const TileJourney& b) {
                return a.dram_request_cycle < b.dram_request_cycle;
            });
        return result;
    }

    /// Calculate aggregate statistics
    TileJourneyStats calculate_statistics() const {
        TileJourneyStats stats;
        stats.total_tiles = journeys_.size();

        for (const auto& [id, j] : journeys_) {
            if (j.is_complete()) {
                stats.completed_tiles++;

                // Total latency
                Cycle total = j.total_latency();
                if (total > 0) {
                    stats.min_total_latency = std::min(stats.min_total_latency, total);
                    stats.max_total_latency = std::max(stats.max_total_latency, total);
                    stats.sum_total_latency += total;
                }

                // Data movement latency
                Cycle dm = j.data_movement_latency();
                if (dm > 0) {
                    stats.min_data_movement_latency = std::min(stats.min_data_movement_latency, dm);
                    stats.max_data_movement_latency = std::max(stats.max_data_movement_latency, dm);
                    stats.sum_data_movement_latency += dm;
                }

                // Compute latency
                Cycle comp = j.compute_latency();
                if (comp > 0) {
                    stats.min_compute_latency = std::min(stats.min_compute_latency, comp);
                    stats.max_compute_latency = std::max(stats.max_compute_latency, comp);
                    stats.sum_compute_latency += comp;
                }

                // Stage latencies
                stats.sum_dram_latency += j.dram_latency();
                stats.sum_l3_residence += j.l3_residence_time();
                stats.sum_l2_residence += j.l2_residence_time();

                // Stalls
                stats.total_credit_stalls += j.credit_stall_cycles;
                stats.total_data_stalls += j.data_stall_cycles;
                stats.total_memory_stalls += j.memory_stall_cycles;
            }

            // Count input vs output tiles
            if (j.is_output_tile()) {
                stats.output_tiles++;
            } else {
                stats.input_tiles++;
            }
        }

        return stats;
    }

    /// Get number of tracked tiles
    size_t size() const { return journeys_.size(); }

    /// Clear all tracked journeys
    void clear() { journeys_.clear(); }

    /// Reset tracker
    void reset() { clear(); }

private:
    std::unordered_map<TileID, TileJourney, TileIDHash> journeys_;
};

} // namespace sw::kpu::harness
