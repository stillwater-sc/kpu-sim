/**
 * @file cycle_breakdown.hpp
 * @brief Cycle Breakdown Statistics for KPU Simulator
 * @version 0.3.4
 *
 * Provides detailed breakdown of where cycles are spent during execution:
 *   - Compute cycles (active processing)
 *   - Memory cycles (waiting for data)
 *   - Stall cycles (blocked on dependencies)
 *   - Idle cycles (no work available)
 *
 * This enables identification of performance bottlenecks and guides
 * optimization efforts.
 */

#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <array>
#include <atomic>

namespace sw::kpu::stats {

/**
 * @brief Categories for cycle breakdown
 */
enum class CycleCategory : uint8_t {
    COMPUTE = 0,        // Active computation (matmul, elementwise, etc.)
    MEMORY_ACCESS = 1,  // Memory operations (load/store)
    MEMORY_STALL = 2,   // Waiting for memory (latency)
    CREDIT_STALL = 3,   // Waiting for downstream buffer credit
    DATA_STALL = 4,     // Waiting for upstream data
    SYNC_STALL = 5,     // Barrier/synchronization
    IDLE = 6,           // No work to do
    OVERHEAD = 7,       // Control/scheduling overhead
    _COUNT = 8
};

constexpr const char* to_string(CycleCategory cat) {
    switch (cat) {
        case CycleCategory::COMPUTE: return "compute";
        case CycleCategory::MEMORY_ACCESS: return "memory_access";
        case CycleCategory::MEMORY_STALL: return "memory_stall";
        case CycleCategory::CREDIT_STALL: return "credit_stall";
        case CycleCategory::DATA_STALL: return "data_stall";
        case CycleCategory::SYNC_STALL: return "sync_stall";
        case CycleCategory::IDLE: return "idle";
        case CycleCategory::OVERHEAD: return "overhead";
        default: return "unknown";
    }
}

/**
 * @brief Cycle breakdown statistics
 *
 * Tracks cycles spent in each category for performance analysis.
 * Thread-safe through atomic counters.
 */
class CycleBreakdown {
public:
    static constexpr size_t NUM_CATEGORIES = static_cast<size_t>(CycleCategory::_COUNT);

    CycleBreakdown() = default;

    // Copy constructor
    CycleBreakdown(const CycleBreakdown& other) {
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            cycles_[i].store(other.cycles_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }

    // Move constructor
    CycleBreakdown(CycleBreakdown&& other) noexcept {
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            cycles_[i].store(other.cycles_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }

    // Copy assignment
    CycleBreakdown& operator=(const CycleBreakdown& other) {
        if (this != &other) {
            for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
                cycles_[i].store(other.cycles_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        return *this;
    }

    // Move assignment
    CycleBreakdown& operator=(CycleBreakdown&& other) noexcept {
        if (this != &other) {
            for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
                cycles_[i].store(other.cycles_[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }
        return *this;
    }

    /**
     * @brief Record cycles in a category
     */
    void record(CycleCategory cat, uint64_t cycles) {
        size_t idx = static_cast<size_t>(cat);
        if (idx < NUM_CATEGORIES) {
            cycles_[idx].fetch_add(cycles, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Get cycles for a category
     */
    uint64_t get(CycleCategory cat) const {
        size_t idx = static_cast<size_t>(cat);
        return idx < NUM_CATEGORIES ? cycles_[idx].load(std::memory_order_relaxed) : 0;
    }

    /**
     * @brief Get total cycles across all categories
     */
    uint64_t total_cycles() const {
        uint64_t total = 0;
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            total += cycles_[i].load(std::memory_order_relaxed);
        }
        return total;
    }

    /**
     * @brief Get compute cycles (active processing)
     */
    uint64_t compute_cycles() const {
        return get(CycleCategory::COMPUTE);
    }

    /**
     * @brief Get all stall cycles combined
     */
    uint64_t stall_cycles() const {
        return get(CycleCategory::MEMORY_STALL) +
               get(CycleCategory::CREDIT_STALL) +
               get(CycleCategory::DATA_STALL) +
               get(CycleCategory::SYNC_STALL);
    }

    /**
     * @brief Get memory-related cycles (access + stall)
     */
    uint64_t memory_cycles() const {
        return get(CycleCategory::MEMORY_ACCESS) +
               get(CycleCategory::MEMORY_STALL);
    }

    /**
     * @brief Get percentage for a category
     */
    double percentage(CycleCategory cat) const {
        uint64_t total = total_cycles();
        return total > 0 ? 100.0 * static_cast<double>(get(cat)) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Get compute efficiency (compute / (compute + stall))
     */
    double compute_efficiency() const {
        uint64_t compute = compute_cycles();
        uint64_t stall = stall_cycles();
        return (compute + stall) > 0 ?
            static_cast<double>(compute) / static_cast<double>(compute + stall) : 0.0;
    }

    /**
     * @brief Get overall utilization (non-idle / total)
     */
    double utilization() const {
        uint64_t total = total_cycles();
        uint64_t idle = get(CycleCategory::IDLE);
        return total > 0 ? static_cast<double>(total - idle) / static_cast<double>(total) : 0.0;
    }

    /**
     * @brief Reset all counters
     */
    void reset() {
        for (auto& counter : cycles_) {
            counter.store(0, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Merge another breakdown into this one
     */
    void merge(const CycleBreakdown& other) {
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            cycles_[i].fetch_add(other.cycles_[i].load(), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Generate JSON representation
     */
    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"total_cycles\": " << total_cycles() << ",\n";
        ss << "  \"compute_efficiency\": " << compute_efficiency() << ",\n";
        ss << "  \"utilization\": " << utilization() << ",\n";
        ss << "  \"breakdown\": {\n";
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            CycleCategory cat = static_cast<CycleCategory>(i);
            uint64_t cycles = get(cat);
            if (i > 0) ss << ",\n";
            ss << "    \"" << to_string(cat) << "\": {\n";
            ss << "      \"cycles\": " << cycles << ",\n";
            ss << "      \"percent\": " << percentage(cat) << "\n";
            ss << "    }";
        }
        ss << "\n  }\n";
        ss << "}";
        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1);
        ss << "Cycle Breakdown (total: " << total_cycles() << " cycles)\n";
        ss << "  Compute Efficiency: " << (compute_efficiency() * 100) << "%\n";
        ss << "  Utilization:        " << (utilization() * 100) << "%\n\n";
        ss << "  Category          Cycles        Percent\n";
        ss << "  ----------------------------------------\n";
        for (size_t i = 0; i < NUM_CATEGORIES; ++i) {
            CycleCategory cat = static_cast<CycleCategory>(i);
            ss << "  " << std::left << std::setw(16) << to_string(cat)
               << std::right << std::setw(12) << get(cat)
               << std::setw(12) << percentage(cat) << "%\n";
        }
        return ss.str();
    }

private:
    std::array<std::atomic<uint64_t>, NUM_CATEGORIES> cycles_{};
};

/**
 * @brief RAII cycle recorder for automatic timing
 */
class ScopedCycleRecorder {
public:
    ScopedCycleRecorder(CycleBreakdown& breakdown, CycleCategory cat,
                        uint64_t& current_cycle)
        : breakdown_(breakdown), category_(cat), cycle_ref_(current_cycle),
          start_cycle_(current_cycle) {}

    ~ScopedCycleRecorder() {
        uint64_t elapsed = cycle_ref_ - start_cycle_;
        breakdown_.record(category_, elapsed);
    }

    ScopedCycleRecorder(const ScopedCycleRecorder&) = delete;
    ScopedCycleRecorder& operator=(const ScopedCycleRecorder&) = delete;

private:
    CycleBreakdown& breakdown_;
    CycleCategory category_;
    uint64_t& cycle_ref_;
    uint64_t start_cycle_;
};

#define RECORD_CYCLES(breakdown, category, cycle_var) \
    sw::kpu::stats::ScopedCycleRecorder _cycle_recorder_##__LINE__(breakdown, category, cycle_var)

} // namespace sw::kpu::stats
