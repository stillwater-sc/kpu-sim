/**
 * @file event_counter.hpp
 * @brief XUE Event Counters for Operational Analysis
 * @version 0.3.3
 *
 * Provides efficient event counting with hierarchical aggregation.
 * Counters track:
 *   - Event occurrence count
 *   - Total bytes (for memory/data movement)
 *   - Total FLOPs (for compute)
 *   - Min/max/avg payload sizes
 *
 * Design goals:
 *   - O(1) event recording
 *   - Lock-free where possible (atomic counters)
 *   - Hierarchical rollup for analysis
 *   - JSON serialization for tooling
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

namespace sw::xue {

/**
 * @brief Statistics for a single event type
 */
struct EventStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<uint64_t> total_flops{0};
    std::atomic<uint64_t> total_cycles{0};  // Cycles spent on this event type

    // For payload size distribution (non-atomic, protected by mutex in collector)
    uint64_t min_bytes = UINT64_MAX;
    uint64_t max_bytes = 0;

    EventStats() = default;

    // Copy constructor for snapshot
    EventStats(const EventStats& other)
        : count(other.count.load()),
          total_bytes(other.total_bytes.load()),
          total_flops(other.total_flops.load()),
          total_cycles(other.total_cycles.load()),
          min_bytes(other.min_bytes),
          max_bytes(other.max_bytes) {}

    EventStats& operator=(const EventStats& other) {
        count.store(other.count.load());
        total_bytes.store(other.total_bytes.load());
        total_flops.store(other.total_flops.load());
        total_cycles.store(other.total_cycles.load());
        min_bytes = other.min_bytes;
        max_bytes = other.max_bytes;
        return *this;
    }

    void record(const EventMetadata& meta, uint64_t cycles = 0) {
        count.fetch_add(1, std::memory_order_relaxed);
        total_bytes.fetch_add(meta.bytes, std::memory_order_relaxed);
        total_flops.fetch_add(meta.flops, std::memory_order_relaxed);
        total_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    void update_payload_range(uint64_t bytes) {
        // Called under lock
        if (bytes < min_bytes) min_bytes = bytes;
        if (bytes > max_bytes) max_bytes = bytes;
    }

    double avg_bytes() const {
        uint64_t c = count.load();
        return c > 0 ? static_cast<double>(total_bytes.load()) / c : 0.0;
    }

    double avg_flops() const {
        uint64_t c = count.load();
        return c > 0 ? static_cast<double>(total_flops.load()) / c : 0.0;
    }

    double avg_cycles() const {
        uint64_t c = count.load();
        return c > 0 ? static_cast<double>(total_cycles.load()) / c : 0.0;
    }

    void reset() {
        count.store(0);
        total_bytes.store(0);
        total_flops.store(0);
        total_cycles.store(0);
        min_bytes = UINT64_MAX;
        max_bytes = 0;
    }

    void merge(const EventStats& other) {
        count.fetch_add(other.count.load(), std::memory_order_relaxed);
        total_bytes.fetch_add(other.total_bytes.load(), std::memory_order_relaxed);
        total_flops.fetch_add(other.total_flops.load(), std::memory_order_relaxed);
        total_cycles.fetch_add(other.total_cycles.load(), std::memory_order_relaxed);
        if (other.min_bytes < min_bytes) min_bytes = other.min_bytes;
        if (other.max_bytes > max_bytes) max_bytes = other.max_bytes;
    }
};

/**
 * @brief Aggregated statistics at a hierarchy level
 */
struct AggregateStats {
    uint64_t total_events = 0;
    uint64_t total_bytes = 0;
    uint64_t total_flops = 0;
    uint64_t total_cycles = 0;
    std::map<std::string, uint64_t> event_counts;

    void add(const std::string& event_name, const EventStats& stats) {
        uint64_t cnt = stats.count.load();
        total_events += cnt;
        total_bytes += stats.total_bytes.load();
        total_flops += stats.total_flops.load();
        total_cycles += stats.total_cycles.load();
        event_counts[event_name] = cnt;
    }
};

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
     * @brief Get aggregate stats for a category
     */
    AggregateStats get_category_stats(EventCategory cat) const {
        AggregateStats agg;

        for (size_t i = 0; i < MAX_EVENT_TYPES; ++i) {
            EventType type = static_cast<EventType>(i);
            if (get_category(type) == cat && counters_[i].count.load() > 0) {
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
            if (get_compute_subcategory(type) == sub && counters_[i].count.load() > 0) {
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
            if (get_memory_subcategory(type) == sub && counters_[i].count.load() > 0) {
                agg.add(std::string(to_string(type)), counters_[i]);
            }
        }

        return agg;
    }

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
        auto dram_stats = get_memory_subcategory_stats(MemorySubcategory::EXTERNAL);
        return dram_stats.total_bytes;
    }

    /**
     * @brief Calculate arithmetic intensity (FLOPs / DRAM bytes)
     */
    double arithmetic_intensity() const {
        uint64_t bytes = dram_bytes();
        if (bytes == 0) return 0.0;
        return static_cast<double>(total_flops()) / bytes;
    }

    /**
     * @brief Reset all counters
     */
    void reset() {
        std::lock_guard<std::mutex> lock(range_mutex_);
        for (auto& counter : counters_) {
            counter.reset();
        }
    }

    /**
     * @brief Merge counters from another EventCounter
     */
    void merge(const EventCounter& other) {
        std::lock_guard<std::mutex> lock(range_mutex_);
        for (size_t i = 0; i < MAX_EVENT_TYPES; ++i) {
            counters_[i].merge(other.counters_[i]);
        }
    }

    /**
     * @brief Generate JSON representation of all counters
     */
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"version\": \"0.3.3\",\n";
        ss << "  \"summary\": {\n";
        ss << "    \"total_flops\": " << total_flops() << ",\n";
        ss << "    \"total_bytes_moved\": " << total_bytes_moved() << ",\n";
        ss << "    \"dram_bytes\": " << dram_bytes() << ",\n";
        ss << "    \"arithmetic_intensity\": " << arithmetic_intensity() << "\n";
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
                ss << "      \"total_bytes\": " << stats.total_bytes << ",\n";
                ss << "      \"total_flops\": " << stats.total_flops << ",\n";
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
        ss << "=== XUE Event Summary (v0.3.3) ===\n\n";

        ss << "Overall Metrics:\n";
        ss << "  Total FLOPs:          " << total_flops() << "\n";
        ss << "  Total Bytes Moved:    " << total_bytes_moved() << "\n";
        ss << "  DRAM Traffic:         " << dram_bytes() << " bytes\n";
        ss << "  Arithmetic Intensity: " << arithmetic_intensity() << " FLOP/byte\n\n";

        for (size_t c = 0; c < static_cast<size_t>(EventCategory::_COUNT); ++c) {
            EventCategory cat = static_cast<EventCategory>(c);
            auto stats = get_category_stats(cat);
            if (stats.total_events > 0) {
                ss << to_string(cat) << " Events:\n";
                for (const auto& [name, count] : stats.event_counts) {
                    ss << "  " << name << ": " << count << "\n";
                }
                ss << "\n";
            }
        }

        return ss.str();
    }

private:
    std::array<EventStats, MAX_EVENT_TYPES> counters_;
    mutable std::mutex range_mutex_;  // Only for payload range updates
};

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
