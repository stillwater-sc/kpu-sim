/**
 * @file event_counter.hpp
 * @brief XUE Observation Architecture - Event Counters
 * @version 0.4.0
 *
 * Provides efficient event counting with hierarchical aggregation
 * and occupancy integration for Operational Analysis.
 *
 * Per Intel Patent 6,023,759 (Observation Architecture), event counters track:
 *   - Occurrences: Number of times event occurred
 *   - Occupancy Integral: Σ(occupancy_t) over observation period
 *   - Total bytes/FLOPs/cycles
 *   - Min/max/avg payload sizes
 *
 * Operational Analysis metrics (Little's Law):
 *   - Average Delay = (Occupancy Integral / T) / Arrival Rate
 *   - Where T = total observation cycles
 *
 * XUE Metrics derivation:
 *   - X (Throughput): completions / T
 *   - U (Utilization): busy_cycles / T
 *   - E (Efficiency): achieved_throughput / peak_throughput
 *
 * Design goals:
 *   - O(1) event recording (atomic increment only)
 *   - Lock-free where possible
 *   - Hierarchical rollup for analysis
 *   - JSON serialization for tooling
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
 */

#pragma once

#include <sw/xue/event_hierarchy.hpp>
#include <atomic>
#include <array>
#include <map>
#include <string>
#include <sstream>
#include <mutex>
#include <cstdint>
#include <cmath>

namespace sw::xue {

// ============================================================================
// EventStats - Per-Event Statistics with Occupancy Tracking
// ============================================================================

/**
 * @brief Statistics for a single event type with occupancy integration
 *
 * Implements the Intel OA patent's approach to Operational Analysis:
 * - Track current_occupancy (items in queue/buffer)
 * - Accumulate occupancy_integral each clock cycle
 * - Derive average delay via Little's Law: (occupancy_integral/T) / arrival_rate
 */
struct EventStats {
    // Occurrence counts (atomically updated)
    std::atomic<uint64_t> count{0};              // Number of occurrences
    std::atomic<uint64_t> completions{0};        // Number of completions (for queues)
    std::atomic<uint64_t> total_bytes{0};        // Total bytes transferred
    std::atomic<uint64_t> total_flops{0};        // Total FLOPs performed
    std::atomic<uint64_t> total_cycles{0};       // Total cycles spent on this event type
    std::atomic<uint64_t> total_elements{0};     // Total elements (for L1 tracking)

    // Occupancy tracking for Operational Analysis
    // These require per-cycle accumulation via accumulate_occupancy()
    std::atomic<uint64_t> occupancy_integral{0}; // Σ(occupancy_t) over observation
    std::atomic<int64_t> current_occupancy{0};   // Current queue/buffer occupancy
    std::atomic<uint64_t> max_occupancy{0};      // Peak occupancy observed

    // Latency tracking
    std::atomic<uint64_t> total_latency{0};      // Sum of all latencies (for avg)
    std::atomic<uint64_t> min_latency{UINT64_MAX};  // Minimum latency
    std::atomic<uint64_t> max_latency{0};        // Maximum latency

    // Payload size distribution (non-atomic, protected by mutex in collector)
    uint64_t min_bytes = UINT64_MAX;
    uint64_t max_bytes = 0;

    EventStats() = default;

    // Copy constructor for snapshot
    EventStats(const EventStats& other)
        : count(other.count.load(std::memory_order_relaxed)),
          completions(other.completions.load(std::memory_order_relaxed)),
          total_bytes(other.total_bytes.load(std::memory_order_relaxed)),
          total_flops(other.total_flops.load(std::memory_order_relaxed)),
          total_cycles(other.total_cycles.load(std::memory_order_relaxed)),
          total_elements(other.total_elements.load(std::memory_order_relaxed)),
          occupancy_integral(other.occupancy_integral.load(std::memory_order_relaxed)),
          current_occupancy(other.current_occupancy.load(std::memory_order_relaxed)),
          max_occupancy(other.max_occupancy.load(std::memory_order_relaxed)),
          total_latency(other.total_latency.load(std::memory_order_relaxed)),
          min_latency(other.min_latency.load(std::memory_order_relaxed)),
          max_latency(other.max_latency.load(std::memory_order_relaxed)),
          min_bytes(other.min_bytes),
          max_bytes(other.max_bytes) {}

    EventStats& operator=(const EventStats& other) {
        count.store(other.count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        completions.store(other.completions.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_bytes.store(other.total_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_flops.store(other.total_flops.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_cycles.store(other.total_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_elements.store(other.total_elements.load(std::memory_order_relaxed), std::memory_order_relaxed);
        occupancy_integral.store(other.occupancy_integral.load(std::memory_order_relaxed), std::memory_order_relaxed);
        current_occupancy.store(other.current_occupancy.load(std::memory_order_relaxed), std::memory_order_relaxed);
        max_occupancy.store(other.max_occupancy.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_latency.store(other.total_latency.load(std::memory_order_relaxed), std::memory_order_relaxed);
        min_latency.store(other.min_latency.load(std::memory_order_relaxed), std::memory_order_relaxed);
        max_latency.store(other.max_latency.load(std::memory_order_relaxed), std::memory_order_relaxed);
        min_bytes = other.min_bytes;
        max_bytes = other.max_bytes;
        return *this;
    }

    // ========================================================================
    // Event Recording (Fast Path - atomic increments only)
    // ========================================================================

    /**
     * @brief Record an event arrival (fast path)
     */
    void record_arrival(const EventMetadata& meta = EventMetadata{}, uint64_t cycles = 0) {
        count.fetch_add(1, std::memory_order_relaxed);
        total_bytes.fetch_add(meta.bytes, std::memory_order_relaxed);
        total_flops.fetch_add(meta.flops, std::memory_order_relaxed);
        total_cycles.fetch_add(cycles, std::memory_order_relaxed);
        total_elements.fetch_add(meta.elements, std::memory_order_relaxed);
        current_occupancy.fetch_add(1, std::memory_order_relaxed);

        // Update max occupancy (relaxed memory order is fine for statistics)
        uint64_t curr = static_cast<uint64_t>(current_occupancy.load(std::memory_order_relaxed));
        uint64_t prev_max = max_occupancy.load(std::memory_order_relaxed);
        while (curr > prev_max && !max_occupancy.compare_exchange_weak(
                   prev_max, curr, std::memory_order_relaxed)) {}
    }

    /**
     * @brief Record an event completion (fast path)
     */
    void record_completion(uint64_t latency = 0) {
        completions.fetch_add(1, std::memory_order_relaxed);
        current_occupancy.fetch_sub(1, std::memory_order_relaxed);

        if (latency > 0) {
            total_latency.fetch_add(latency, std::memory_order_relaxed);

            // Update min latency
            uint64_t prev_min = min_latency.load(std::memory_order_relaxed);
            while (latency < prev_min && !min_latency.compare_exchange_weak(
                       prev_min, latency, std::memory_order_relaxed)) {}

            // Update max latency
            uint64_t prev_max = max_latency.load(std::memory_order_relaxed);
            while (latency > prev_max && !max_latency.compare_exchange_weak(
                       prev_max, latency, std::memory_order_relaxed)) {}
        }
    }

    /**
     * @brief Legacy record method for backward compatibility
     */
    void record(const EventMetadata& meta, uint64_t cycles = 0) {
        record_arrival(meta, cycles);
        record_completion();  // Immediate completion
    }

    // ========================================================================
    // Occupancy Integration (called each clock cycle)
    // ========================================================================

    /**
     * @brief Accumulate current occupancy into the integral
     *
     * This MUST be called every clock cycle for accurate Operational Analysis.
     * Per Intel OA patent, the occupancy integral is the sum of occupancy
     * at each clock cycle over the observation period.
     *
     * Average delay via Little's Law:
     *   avg_delay = (occupancy_integral / T) / arrival_rate
     * where:
     *   T = total observation cycles
     *   arrival_rate = count / T
     */
    void accumulate_occupancy() {
        int64_t occ = current_occupancy.load(std::memory_order_relaxed);
        if (occ > 0) {
            occupancy_integral.fetch_add(static_cast<uint64_t>(occ), std::memory_order_relaxed);
        }
    }

    // ========================================================================
    // Derived Metrics
    // ========================================================================

    /**
     * @brief Calculate average delay using Little's Law
     *
     * avg_delay = (occupancy_integral / T) / arrival_rate
     *           = occupancy_integral / count
     *
     * @param observation_cycles Total cycles of observation (T)
     * @return Average delay in cycles
     */
    double average_delay([[maybe_unused]] uint64_t observation_cycles = 0) const {
        uint64_t c = count.load(std::memory_order_relaxed);
        if (c == 0) return 0.0;

        uint64_t occ_int = occupancy_integral.load(std::memory_order_relaxed);

        // Little's Law: avg_delay = avg_occupancy / arrival_rate
        // avg_occupancy = occ_int / T
        // arrival_rate = c / T
        // avg_delay = (occ_int / T) / (c / T) = occ_int / c
        return static_cast<double>(occ_int) / static_cast<double>(c);
    }

    /**
     * @brief Calculate average occupancy over observation period
     */
    double average_occupancy(uint64_t observation_cycles) const {
        if (observation_cycles == 0) return 0.0;
        return static_cast<double>(occupancy_integral.load(std::memory_order_relaxed)) /
               static_cast<double>(observation_cycles);
    }

    /**
     * @brief Calculate throughput (completions per cycle)
     */
    double throughput(uint64_t observation_cycles) const {
        if (observation_cycles == 0) return 0.0;
        return static_cast<double>(completions.load(std::memory_order_relaxed)) /
               static_cast<double>(observation_cycles);
    }

    /**
     * @brief Calculate utilization (fraction of time with non-zero occupancy)
     *
     * This is approximated as occupancy_integral / (T * max_occupancy)
     * A more accurate method would track busy_cycles separately.
     */
    double utilization(uint64_t observation_cycles) const {
        if (observation_cycles == 0) return 0.0;
        uint64_t max_occ = max_occupancy.load(std::memory_order_relaxed);
        if (max_occ == 0) return 0.0;
        uint64_t occ_int = occupancy_integral.load(std::memory_order_relaxed);
        return static_cast<double>(occ_int) / static_cast<double>(observation_cycles * max_occ);
    }

    double avg_bytes() const {
        uint64_t c = count.load(std::memory_order_relaxed);
        return c > 0 ? static_cast<double>(total_bytes.load(std::memory_order_relaxed)) / static_cast<double>(c) : 0.0;
    }

    double avg_flops() const {
        uint64_t c = count.load(std::memory_order_relaxed);
        return c > 0 ? static_cast<double>(total_flops.load(std::memory_order_relaxed)) / static_cast<double>(c) : 0.0;
    }

    double avg_cycles() const {
        uint64_t c = count.load(std::memory_order_relaxed);
        return c > 0 ? static_cast<double>(total_cycles.load(std::memory_order_relaxed)) / static_cast<double>(c) : 0.0;
    }

    double avg_latency() const {
        uint64_t comp = completions.load(std::memory_order_relaxed);
        return comp > 0 ? static_cast<double>(total_latency.load(std::memory_order_relaxed)) / static_cast<double>(comp) : 0.0;
    }

    void update_payload_range(uint64_t bytes) {
        // Called under lock
        if (bytes < min_bytes) min_bytes = bytes;
        if (bytes > max_bytes) max_bytes = bytes;
    }

    void reset() {
        count.store(0, std::memory_order_relaxed);
        completions.store(0, std::memory_order_relaxed);
        total_bytes.store(0, std::memory_order_relaxed);
        total_flops.store(0, std::memory_order_relaxed);
        total_cycles.store(0, std::memory_order_relaxed);
        total_elements.store(0, std::memory_order_relaxed);
        occupancy_integral.store(0, std::memory_order_relaxed);
        current_occupancy.store(0, std::memory_order_relaxed);
        max_occupancy.store(0, std::memory_order_relaxed);
        total_latency.store(0, std::memory_order_relaxed);
        min_latency.store(UINT64_MAX, std::memory_order_relaxed);
        max_latency.store(0, std::memory_order_relaxed);
        min_bytes = UINT64_MAX;
        max_bytes = 0;
    }

    void merge(const EventStats& other) {
        count.fetch_add(other.count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        completions.fetch_add(other.completions.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_bytes.fetch_add(other.total_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_flops.fetch_add(other.total_flops.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_cycles.fetch_add(other.total_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_elements.fetch_add(other.total_elements.load(std::memory_order_relaxed), std::memory_order_relaxed);
        occupancy_integral.fetch_add(other.occupancy_integral.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_latency.fetch_add(other.total_latency.load(std::memory_order_relaxed), std::memory_order_relaxed);

        // Update min/max
        uint64_t other_max = other.max_occupancy.load(std::memory_order_relaxed);
        uint64_t prev_max = max_occupancy.load(std::memory_order_relaxed);
        while (other_max > prev_max && !max_occupancy.compare_exchange_weak(
                   prev_max, other_max, std::memory_order_relaxed)) {}

        uint64_t other_min_lat = other.min_latency.load(std::memory_order_relaxed);
        uint64_t prev_min_lat = min_latency.load(std::memory_order_relaxed);
        while (other_min_lat < prev_min_lat && !min_latency.compare_exchange_weak(
                   prev_min_lat, other_min_lat, std::memory_order_relaxed)) {}

        uint64_t other_max_lat = other.max_latency.load(std::memory_order_relaxed);
        uint64_t prev_max_lat = max_latency.load(std::memory_order_relaxed);
        while (other_max_lat > prev_max_lat && !max_latency.compare_exchange_weak(
                   prev_max_lat, other_max_lat, std::memory_order_relaxed)) {}

        if (other.min_bytes < min_bytes) min_bytes = other.min_bytes;
        if (other.max_bytes > max_bytes) max_bytes = other.max_bytes;
    }
};

// ============================================================================
// AggregateStats - Hierarchical Aggregation
// ============================================================================

/**
 * @brief Aggregated statistics at a hierarchy level
 */
struct AggregateStats {
    uint64_t total_events = 0;
    uint64_t total_completions = 0;
    uint64_t total_bytes = 0;
    uint64_t total_flops = 0;
    uint64_t total_cycles = 0;
    uint64_t total_elements = 0;
    uint64_t total_occupancy_integral = 0;
    std::map<std::string, uint64_t> event_counts;
    std::map<std::string, const EventStats*> event_stats;

    void add(const std::string& event_name, const EventStats& stats) {
        uint64_t cnt = stats.count.load(std::memory_order_relaxed);
        total_events += cnt;
        total_completions += stats.completions.load(std::memory_order_relaxed);
        total_bytes += stats.total_bytes.load(std::memory_order_relaxed);
        total_flops += stats.total_flops.load(std::memory_order_relaxed);
        total_cycles += stats.total_cycles.load(std::memory_order_relaxed);
        total_elements += stats.total_elements.load(std::memory_order_relaxed);
        total_occupancy_integral += stats.occupancy_integral.load(std::memory_order_relaxed);
        event_counts[event_name] = cnt;
        event_stats[event_name] = &stats;
    }

    double average_delay() const {
        if (total_events == 0) return 0.0;
        return static_cast<double>(total_occupancy_integral) / static_cast<double>(total_events);
    }

    double throughput(uint64_t observation_cycles) const {
        if (observation_cycles == 0) return 0.0;
        return static_cast<double>(total_completions) / static_cast<double>(observation_cycles);
    }
};

// ============================================================================
// XUEMetrics - Throughput/Utilization/Efficiency Summary
// ============================================================================

/**
 * @brief XUE Metrics summary structure
 */
struct XUEMetrics {
    // Throughput (X)
    double throughput_flops_per_cycle = 0.0;
    double throughput_bytes_per_cycle = 0.0;
    double throughput_gflops = 0.0;

    // Utilization (U)
    double compute_utilization = 0.0;
    double memory_utilization = 0.0;
    double overall_utilization = 0.0;

    // Efficiency (E)
    double compute_efficiency = 0.0;  // achieved / peak
    double memory_efficiency = 0.0;
    double arithmetic_intensity = 0.0;  // FLOP/byte

    // Operational Analysis
    double avg_compute_delay = 0.0;
    double avg_memory_delay = 0.0;

    std::string to_string() const {
        std::ostringstream ss;
        ss << "XUE Metrics Summary:\n";
        ss << "  Throughput (X):\n";
        ss << "    FLOPs/cycle:  " << throughput_flops_per_cycle << "\n";
        ss << "    Bytes/cycle:  " << throughput_bytes_per_cycle << "\n";
        ss << "    GFLOPS:       " << throughput_gflops << "\n";
        ss << "  Utilization (U):\n";
        ss << "    Compute:      " << (compute_utilization * 100.0) << "%\n";
        ss << "    Memory:       " << (memory_utilization * 100.0) << "%\n";
        ss << "    Overall:      " << (overall_utilization * 100.0) << "%\n";
        ss << "  Efficiency (E):\n";
        ss << "    Compute:      " << (compute_efficiency * 100.0) << "%\n";
        ss << "    Memory:       " << (memory_efficiency * 100.0) << "%\n";
        ss << "    Arith. Int.:  " << arithmetic_intensity << " FLOP/byte\n";
        ss << "  Operational Analysis:\n";
        ss << "    Avg Compute Delay: " << avg_compute_delay << " cycles\n";
        ss << "    Avg Memory Delay:  " << avg_memory_delay << " cycles\n";
        return ss.str();
    }
};

// ============================================================================
// EventCounter - Central Counter Collection
// ============================================================================

/**
 * @brief Central event counter collection
 *
 * Maintains counters for all event types and provides
 * hierarchical aggregation for analysis.
 */
class EventCounter {
public:
    static constexpr size_t MAX_EVENT_TYPES = static_cast<size_t>(EventType::_COUNT);

    EventCounter() = default;

    // ========================================================================
    // Event Recording (Fast Path)
    // ========================================================================

    /**
     * @brief Record an event occurrence (fast path)
     */
    void record(EventType type, const EventMetadata& meta = EventMetadata{},
                uint64_t cycles = 0) {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            counters_[idx].record(meta, cycles);

            // Update payload range under lock if bytes > 0
            if (meta.bytes > 0) {
                std::lock_guard<std::mutex> lock(range_mutex_);
                counters_[idx].update_payload_range(meta.bytes);
            }
        }
    }

    /**
     * @brief Record an event arrival (for queued events)
     */
    void record_arrival(EventType type, const EventMetadata& meta = EventMetadata{},
                       uint64_t cycles = 0) {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            counters_[idx].record_arrival(meta, cycles);

            if (meta.bytes > 0) {
                std::lock_guard<std::mutex> lock(range_mutex_);
                counters_[idx].update_payload_range(meta.bytes);
            }
        }
    }

    /**
     * @brief Record an event completion
     */
    void record_completion(EventType type, uint64_t latency = 0) {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            counters_[idx].record_completion(latency);
        }
    }

    /**
     * @brief Convenience method for compute events
     */
    void record_compute(EventType type, uint64_t flops,
                       uint32_t m = 0, uint32_t n = 0, uint32_t k = 0,
                       uint64_t cycles = 0) {
        record(type, EventMetadata::compute(flops, m, n, k), cycles);
    }

    /**
     * @brief Convenience method for memory events
     */
    void record_memory(EventType type, uint64_t bytes,
                      uint16_t src = 0, uint16_t dst = 0,
                      uint64_t cycles = 0) {
        record(type, EventMetadata::memory(bytes, src, dst), cycles);
    }

    /**
     * @brief Convenience method for element transfer events
     */
    void record_elements(EventType type, uint64_t elements, uint64_t cycles = 0) {
        record(type, EventMetadata::element_transfer(elements), cycles);
    }

    // ========================================================================
    // Occupancy Integration
    // ========================================================================

    /**
     * @brief Accumulate occupancy for all event types (call each cycle)
     *
     * This MUST be called every simulation clock cycle for accurate
     * Operational Analysis.
     */
    void accumulate_all_occupancy() {
        for (size_t i = 0; i < MAX_EVENT_TYPES; ++i) {
            counters_[i].accumulate_occupancy();
        }
        total_observation_cycles_++;
    }

    /**
     * @brief Accumulate occupancy for specific event type
     */
    void accumulate_occupancy(EventType type) {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            counters_[idx].accumulate_occupancy();
        }
    }

    /**
     * @brief Get total observation cycles
     */
    uint64_t observation_cycles() const {
        return total_observation_cycles_;
    }

    // ========================================================================
    // Statistics Retrieval
    // ========================================================================

    /**
     * @brief Get stats for a specific event type
     */
    EventStats get_stats(EventType type) const {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            return counters_[idx];
        }
        return EventStats{};
    }

    /**
     * @brief Get reference to stats for a specific event type
     */
    const EventStats& stats(EventType type) const {
        size_t idx = static_cast<size_t>(type);
        if (idx < MAX_EVENT_TYPES) {
            return counters_[idx];
        }
        static EventStats empty;
        return empty;
    }

    /**
     * @brief Get aggregate stats for a category
     */
    AggregateStats get_category_stats(EventCategory cat) const {
        AggregateStats agg;

        for (size_t i = 0; i < MAX_EVENT_TYPES; ++i) {
            EventType type = static_cast<EventType>(i);
            if (get_category(type) == cat && counters_[i].count.load(std::memory_order_relaxed) > 0) {
                agg.add(std::string(to_string(type)), counters_[i]);
            }
        }

        return agg;
    }

    /**
     * @brief Get aggregate stats for compute subcategory
     */
    AggregateStats get_compute_subcategory_stats(ComputeSubcategory sub) const {
        AggregateStats agg;

        for (size_t i = 0x0100; i < 0x0200 && i < MAX_EVENT_TYPES; ++i) {
            EventType type = static_cast<EventType>(i);
            if (get_compute_subcategory(type) == sub &&
                counters_[i].count.load(std::memory_order_relaxed) > 0) {
                agg.add(std::string(to_string(type)), counters_[i]);
            }
        }

        return agg;
    }

    /**
     * @brief Get aggregate stats for memory subcategory
     */
    AggregateStats get_memory_subcategory_stats(MemorySubcategory sub) const {
        AggregateStats agg;

        for (size_t i = 0x0200; i < 0x0300 && i < MAX_EVENT_TYPES; ++i) {
            EventType type = static_cast<EventType>(i);
            if (get_memory_subcategory(type) == sub &&
                counters_[i].count.load(std::memory_order_relaxed) > 0) {
                agg.add(std::string(to_string(type)), counters_[i]);
            }
        }

        return agg;
    }

    /**
     * @brief Get aggregate stats for data movement subcategory
     */
    AggregateStats get_data_movement_subcategory_stats(DataMovementSubcategory sub) const {
        AggregateStats agg;

        for (size_t i = 0x0300; i < 0x0400 && i < MAX_EVENT_TYPES; ++i) {
            EventType type = static_cast<EventType>(i);
            if (get_data_movement_subcategory(type) == sub &&
                counters_[i].count.load(std::memory_order_relaxed) > 0) {
                agg.add(std::string(to_string(type)), counters_[i]);
            }
        }

        return agg;
    }

    // ========================================================================
    // Aggregate Metrics
    // ========================================================================

    /**
     * @brief Get total FLOPs across all compute events
     */
    uint64_t total_flops() const {
        auto compute_stats = get_category_stats(EventCategory::COMPUTE);
        return compute_stats.total_flops;
    }

    /**
     * @brief Get total bytes moved across all memory/data movement events
     */
    uint64_t total_bytes_moved() const {
        auto mem_stats = get_category_stats(EventCategory::MEMORY);
        auto dm_stats = get_category_stats(EventCategory::DATA_MOVEMENT);
        return mem_stats.total_bytes + dm_stats.total_bytes;
    }

    /**
     * @brief Get DRAM traffic (external memory only)
     */
    uint64_t dram_bytes() const {
        auto dram_stats = get_memory_subcategory_stats(MemorySubcategory::DRAM);
        return dram_stats.total_bytes;
    }

    /**
     * @brief Calculate arithmetic intensity (FLOPs / DRAM bytes)
     */
    double arithmetic_intensity() const {
        uint64_t bytes = dram_bytes();
        if (bytes == 0) return 0.0;
        return static_cast<double>(total_flops()) / static_cast<double>(bytes);
    }

    /**
     * @brief Get total ALU operations (primitives)
     */
    uint64_t total_alu_ops() const {
        auto alu_stats = get_compute_subcategory_stats(ComputeSubcategory::ALU_PRIMITIVE);
        return alu_stats.total_events;
    }

    /**
     * @brief Get total named operations
     */
    uint64_t total_named_ops() const {
        auto named_stats = get_compute_subcategory_stats(ComputeSubcategory::NAMED_OP);
        return named_stats.total_events;
    }

    // ========================================================================
    // XUE Metrics Calculation
    // ========================================================================

    /**
     * @brief Calculate XUE metrics
     *
     * @param clock_ghz Clock frequency in GHz (for GFLOPS calculation)
     * @param peak_flops_per_cycle Peak FLOPs/cycle capability
     * @param peak_bytes_per_cycle Peak bytes/cycle capability
     */
    XUEMetrics calculate_xue_metrics(double clock_ghz = 1.0,
                                      double peak_flops_per_cycle = 512.0,
                                      double peak_bytes_per_cycle = 64.0) const {
        XUEMetrics metrics;
        uint64_t T = total_observation_cycles_;
        if (T == 0) T = 1;  // Avoid division by zero

        // Throughput (X)
        metrics.throughput_flops_per_cycle = static_cast<double>(total_flops()) / static_cast<double>(T);
        metrics.throughput_bytes_per_cycle = static_cast<double>(total_bytes_moved()) / static_cast<double>(T);
        metrics.throughput_gflops = metrics.throughput_flops_per_cycle * clock_ghz;

        // Utilization (U) - based on occupancy integral
        auto compute_stats = get_category_stats(EventCategory::COMPUTE);
        auto memory_stats = get_category_stats(EventCategory::MEMORY);

        if (compute_stats.total_events > 0) {
            double avg_compute_occ = static_cast<double>(compute_stats.total_occupancy_integral) / static_cast<double>(T);
            metrics.compute_utilization = std::min(1.0, avg_compute_occ);
            metrics.avg_compute_delay = compute_stats.average_delay();
        }

        if (memory_stats.total_events > 0) {
            double avg_memory_occ = static_cast<double>(memory_stats.total_occupancy_integral) / static_cast<double>(T);
            metrics.memory_utilization = std::min(1.0, avg_memory_occ);
            metrics.avg_memory_delay = memory_stats.average_delay();
        }

        metrics.overall_utilization = (metrics.compute_utilization + metrics.memory_utilization) / 2.0;

        // Efficiency (E)
        metrics.compute_efficiency = metrics.throughput_flops_per_cycle / peak_flops_per_cycle;
        metrics.memory_efficiency = metrics.throughput_bytes_per_cycle / peak_bytes_per_cycle;
        metrics.arithmetic_intensity = arithmetic_intensity();

        return metrics;
    }

    // ========================================================================
    // Reset and Merge
    // ========================================================================

    /**
     * @brief Reset all counters
     */
    void reset() {
        std::lock_guard<std::mutex> lock(range_mutex_);
        for (auto& counter : counters_) {
            counter.reset();
        }
        total_observation_cycles_ = 0;
    }

    /**
     * @brief Merge counters from another EventCounter
     */
    void merge(const EventCounter& other) {
        std::lock_guard<std::mutex> lock(range_mutex_);
        for (size_t i = 0; i < MAX_EVENT_TYPES; ++i) {
            counters_[i].merge(other.counters_[i]);
        }
        total_observation_cycles_ += other.total_observation_cycles_;
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /**
     * @brief Generate JSON representation of all counters
     */
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"version\": \"0.4.0\",\n";
        ss << "  \"observation_cycles\": " << total_observation_cycles_ << ",\n";
        ss << "  \"summary\": {\n";
        ss << "    \"total_flops\": " << total_flops() << ",\n";
        ss << "    \"total_bytes_moved\": " << total_bytes_moved() << ",\n";
        ss << "    \"dram_bytes\": " << dram_bytes() << ",\n";
        ss << "    \"arithmetic_intensity\": " << arithmetic_intensity() << ",\n";
        ss << "    \"total_alu_ops\": " << total_alu_ops() << ",\n";
        ss << "    \"total_named_ops\": " << total_named_ops() << "\n";
        ss << "  },\n";

        // Categories
        ss << "  \"categories\": {\n";
        bool first_cat = true;
        for (size_t c = 0; c < static_cast<size_t>(EventCategory::_COUNT); ++c) {
            EventCategory cat = static_cast<EventCategory>(c);
            auto stats = get_category_stats(cat);
            if (stats.total_events > 0) {
                if (!first_cat) ss << ",\n";
                first_cat = false;
                ss << "    \"" << to_string(cat) << "\": {\n";
                ss << "      \"total_events\": " << stats.total_events << ",\n";
                ss << "      \"total_completions\": " << stats.total_completions << ",\n";
                ss << "      \"total_bytes\": " << stats.total_bytes << ",\n";
                ss << "      \"total_flops\": " << stats.total_flops << ",\n";
                ss << "      \"occupancy_integral\": " << stats.total_occupancy_integral << ",\n";
                ss << "      \"avg_delay\": " << stats.average_delay() << ",\n";
                ss << "      \"events\": {";
                bool first_evt = true;
                for (const auto& [name, count] : stats.event_counts) {
                    if (!first_evt) ss << ",";
                    first_evt = false;
                    ss << "\n        \"" << name << "\": " << count;
                }
                ss << "\n      }\n";
                ss << "    }";
            }
        }
        ss << "\n  }\n";
        ss << "}\n";

        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary() const {
        std::ostringstream ss;
        ss << "=== XUE Observation Architecture Summary (v0.4.0) ===\n\n";

        ss << "Observation Period: " << total_observation_cycles_ << " cycles\n\n";

        ss << "Overall Metrics:\n";
        ss << "  Total FLOPs:          " << total_flops() << "\n";
        ss << "  Total Bytes Moved:    " << total_bytes_moved() << "\n";
        ss << "  DRAM Traffic:         " << dram_bytes() << " bytes\n";
        ss << "  Arithmetic Intensity: " << arithmetic_intensity() << " FLOP/byte\n";
        ss << "  Total ALU Ops:        " << total_alu_ops() << "\n";
        ss << "  Total Named Ops:      " << total_named_ops() << "\n\n";

        for (size_t c = 0; c < static_cast<size_t>(EventCategory::_COUNT); ++c) {
            EventCategory cat = static_cast<EventCategory>(c);
            auto stats = get_category_stats(cat);
            if (stats.total_events > 0) {
                ss << to_string(cat) << " Events:\n";
                ss << "  Total: " << stats.total_events;
                ss << ", Avg Delay: " << stats.average_delay() << " cycles\n";
                for (const auto& [name, count] : stats.event_counts) {
                    ss << "    " << name << ": " << count << "\n";
                }
                ss << "\n";
            }
        }

        return ss.str();
    }

private:
    std::array<EventStats, MAX_EVENT_TYPES> counters_;
    mutable std::mutex range_mutex_;  // Only for payload range updates
    uint64_t total_observation_cycles_ = 0;
};

// ============================================================================
// ThreadLocalEventCounter - High-Performance Recording
// ============================================================================

/**
 * @brief Thread-local event counter for high-performance recording
 *
 * Each thread maintains its own counters, which are periodically
 * merged into a global counter.
 */
class ThreadLocalEventCounter {
public:
    explicit ThreadLocalEventCounter(EventCounter& global)
        : global_(global) {}

    ~ThreadLocalEventCounter() {
        flush();
    }

    void record(EventType type, const EventMetadata& meta = EventMetadata{},
                uint64_t cycles = 0) {
        local_.record(type, meta, cycles);
        ++record_count_;

        // Flush periodically to global
        if (record_count_ >= FLUSH_THRESHOLD) {
            flush();
        }
    }

    void record_arrival(EventType type, const EventMetadata& meta = EventMetadata{},
                       uint64_t cycles = 0) {
        local_.record_arrival(type, meta, cycles);
        ++record_count_;

        if (record_count_ >= FLUSH_THRESHOLD) {
            flush();
        }
    }

    void record_completion(EventType type, uint64_t latency = 0) {
        local_.record_completion(type, latency);
    }

    void accumulate_all_occupancy() {
        local_.accumulate_all_occupancy();
    }

    void flush() {
        global_.merge(local_);
        local_.reset();
        record_count_ = 0;
    }

private:
    static constexpr size_t FLUSH_THRESHOLD = 10000;

    EventCounter& global_;
    EventCounter local_;
    size_t record_count_ = 0;
};

} // namespace sw::xue
