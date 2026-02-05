// ============================================================================
// include/sw/kpu/timing/livelock_detector.hpp
// Livelock detection for concurrent timing simulation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>
#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>

#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Livelock detection and diagnostic reporting
 *
 * In credit-based dataflow systems, livelock can occur when:
 * - All L3 buffers are full with A tiles waiting for B tiles (or vice versa)
 * - Circular dependencies where each component waits for another
 * - Resources are held without making progress
 *
 * The LivelockDetector monitors:
 * - Progress (tiles completed over time)
 * - Stall duration (cycles without progress)
 * - Resource utilization patterns
 *
 * When potential livelock is detected, it can:
 * - Report diagnostic information
 * - Trigger system state dump
 * - Invoke a user-provided callback
 */
class LivelockDetector {
public:
    /**
     * @brief Configuration for livelock detection
     */
    struct Config {
        /// Cycles without progress before triggering alarm
        Cycle stall_threshold = 1000;

        /// Maximum time a tile can be in TagCAM before being "stuck"
        Cycle tile_stuck_threshold = 5000;

        /// Enable automatic state dump on detection
        bool dump_state_on_detect = true;

        /// Enable automatic abort on detection (for testing)
        bool abort_on_detect = false;
    };

    /**
     * @brief Progress metrics tracked by the detector
     */
    struct ProgressMetrics {
        size_t tiles_dma_completed = 0;      ///< Tiles loaded/stored by DMA
        size_t tiles_moved = 0;              ///< Tiles moved by BlockMover
        size_t tiles_streamed = 0;           ///< Tiles streamed to compute
        size_t compute_ops_completed = 0;    ///< Compute operations finished

        /// Total progress (sum of all metrics)
        size_t total() const {
            return tiles_dma_completed + tiles_moved +
                   tiles_streamed + compute_ops_completed;
        }

        /// Check if any progress made
        bool operator>(const ProgressMetrics& other) const {
            return total() > other.total();
        }
    };

    /**
     * @brief Snapshot of system state for diagnostics
     */
    struct SystemState {
        Cycle current_cycle = 0;

        // Credit state
        size_t l3_credits_available = 0;
        size_t l3_credits_capacity = 0;
        size_t l2_credits_available = 0;
        size_t l2_credits_capacity = 0;

        // TagCAM state
        size_t l3_cam_entries = 0;
        size_t l2_cam_entries = 0;

        // Queue state
        size_t dma_queue_depth = 0;
        size_t bm_queue_depth = 0;
        size_t str_queue_depth = 0;

        // Component state
        bool dma_idle = true;
        bool bm_idle = true;
        bool str_idle = true;
        bool compute_idle = true;

        /// Generate human-readable diagnostic string
        std::string to_string() const {
            std::ostringstream ss;
            ss << "=== System State at Cycle " << current_cycle << " ===\n";
            ss << "L3 Credits: " << l3_credits_available << "/" << l3_credits_capacity << "\n";
            ss << "L2 Credits: " << l2_credits_available << "/" << l2_credits_capacity << "\n";
            ss << "L3 CAM Entries: " << l3_cam_entries << "\n";
            ss << "L2 CAM Entries: " << l2_cam_entries << "\n";
            ss << "Queue Depths - DMA: " << dma_queue_depth
               << ", BM: " << bm_queue_depth
               << ", STR: " << str_queue_depth << "\n";
            ss << "Component State - DMA: " << (dma_idle ? "idle" : "busy")
               << ", BM: " << (bm_idle ? "idle" : "busy")
               << ", STR: " << (str_idle ? "idle" : "busy")
               << ", Compute: " << (compute_idle ? "idle" : "busy") << "\n";
            return ss.str();
        }
    };

    /**
     * @brief Livelock detection result
     */
    struct DetectionResult {
        bool livelock_detected = false;
        Cycle stall_duration = 0;
        std::string message;
        SystemState state;
    };

    /**
     * @brief Callback type for livelock notification
     */
    using LivelockCallback = std::function<void(const DetectionResult&)>;

    /**
     * @brief Construct with default configuration
     */
    LivelockDetector() : config_() {}

    /**
     * @brief Construct with custom configuration
     */
    explicit LivelockDetector(const Config& config) : config_(config) {}

    /**
     * @brief Set callback for livelock detection
     */
    void set_callback(LivelockCallback callback) {
        callback_ = std::move(callback);
    }

    /**
     * @brief Check for livelock conditions
     * @param current_cycle Current simulation cycle
     * @param metrics Current progress metrics
     * @return Detection result (livelock_detected will be true if stalled)
     *
     * Call this every N cycles (e.g., every 100) to check for livelock.
     */
    DetectionResult check(Cycle current_cycle, const ProgressMetrics& metrics) {
        DetectionResult result;
        result.state.current_cycle = current_cycle;

        // Check if progress was made
        if (metrics > last_metrics_) {
            // Progress made - reset stall tracking
            last_progress_cycle_ = current_cycle;
            last_metrics_ = metrics;
            result.livelock_detected = false;
            return result;
        }

        // No progress - check stall duration
        Cycle stall_duration = current_cycle - last_progress_cycle_;
        result.stall_duration = stall_duration;

        if (stall_duration > config_.stall_threshold) {
            // Potential livelock detected
            result.livelock_detected = true;
            result.message = "No progress for " + std::to_string(stall_duration) +
                           " cycles (threshold: " + std::to_string(config_.stall_threshold) + ")";

            // Invoke callback if set
            if (callback_) {
                callback_(result);
            }

            // Abort if configured
            if (config_.abort_on_detect) {
                throw std::runtime_error("Livelock detected: " + result.message);
            }

            livelock_count_++;
        }

        return result;
    }

    /**
     * @brief Check with system state for detailed diagnostics
     */
    DetectionResult check_with_state(Cycle current_cycle,
                                      const ProgressMetrics& metrics,
                                      const SystemState& state) {
        DetectionResult result = check(current_cycle, metrics);
        result.state = state;

        if (result.livelock_detected && config_.dump_state_on_detect) {
            // Append system state to message
            result.message += "\n" + state.to_string();
        }

        return result;
    }

    /**
     * @brief Check if any tile is stuck in TagCAM too long
     * @param cam The TagCAM to check
     * @param current_cycle Current simulation cycle
     * @return Vector of stuck tile entries (empty if none)
     */
    std::vector<TagCAMEntry> find_stuck_tiles(const TagCAM& cam, Cycle current_cycle) const {
        std::vector<TagCAMEntry> stuck;
        for (const auto& entry : cam.all_entries()) {
            if (entry.valid &&
                (current_cycle - entry.arrival_cycle > config_.tile_stuck_threshold)) {
                stuck.push_back(entry);
            }
        }
        return stuck;
    }

    /**
     * @brief Check for circular credit dependency (potential deadlock)
     * @param l3_credits L3 credit pool
     * @param l2_credits L2 credit pool
     * @param l3_cam L3 TagCAM
     * @param bm_waiting_for_credit Number of BlockMovers waiting for L2 credit
     * @return true if circular dependency detected
     *
     * Circular dependency occurs when:
     * - L3 is full (no credits)
     * - L2 is full (no credits)
     * - BlockMovers are waiting for L2 credits to free L3
     */
    bool check_circular_dependency(const CreditPool& l3_credits,
                                   const CreditPool& l2_credits,
                                   const TagCAM& l3_cam,
                                   size_t bm_waiting_for_credit) const {
        // L3 full AND L2 full AND BlockMovers blocked
        return l3_credits.empty() &&
               l2_credits.empty() &&
               bm_waiting_for_credit > 0 &&
               !l3_cam.empty();
    }

    /**
     * @brief Verify invariants that prevent livelock
     * @return true if all invariants hold
     *
     * Call periodically to catch livelock conditions early.
     */
    bool verify_invariants(const CreditPool& l3_credits,
                          const CreditPool& l2_credits,
                          const TagCAM& l3_cam,
                          const TagCAM& l2_cam,
                          Cycle current_cycle,
                          bool any_component_progressing) const {
        // Invariant 1: At least one component making progress (unless idle)
        if (!any_component_progressing && (!l3_cam.empty() || !l2_cam.empty())) {
            return false;
        }

        // Invariant 2: No tile stuck too long
        for (const auto& entry : l3_cam.all_entries()) {
            if (entry.valid &&
                (current_cycle - entry.arrival_cycle > config_.tile_stuck_threshold)) {
                return false;
            }
        }
        for (const auto& entry : l2_cam.all_entries()) {
            if (entry.valid &&
                (current_cycle - entry.arrival_cycle > config_.tile_stuck_threshold)) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Reset detector state (for new simulation run)
     */
    void reset() {
        last_progress_cycle_ = 0;
        last_metrics_ = ProgressMetrics{};
        livelock_count_ = 0;
    }

    /**
     * @brief Get number of livelock detections so far
     */
    size_t livelock_count() const {
        return livelock_count_;
    }

    /**
     * @brief Get current configuration
     */
    const Config& config() const {
        return config_;
    }

    /**
     * @brief Update configuration
     */
    void set_config(const Config& config) {
        config_ = config;
    }

private:
    Config config_;
    Cycle last_progress_cycle_ = 0;
    ProgressMetrics last_metrics_;
    size_t livelock_count_ = 0;
    LivelockCallback callback_;
};

} // namespace sw::kpu::timing
