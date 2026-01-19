/**
 * @file utilization.hpp
 * @brief Resource Utilization Statistics for KPU Simulator
 * @version 0.3.4
 *
 * Tracks utilization of hardware resources:
 *   - Compute units (systolic array, vector engine)
 *   - Memory controllers
 *   - Data movement units (DMA, BlockMover, Streamer)
 *   - NoC links
 *
 * Utilization is measured as:
 *   U = Busy_cycles / Total_cycles
 *
 * Follows the XUE (X, U, E) methodology from the Observation Architecture.
 */

#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <atomic>

namespace sw::kpu::stats {

/**
 * @brief Resource types for utilization tracking
 */
enum class ResourceType : uint8_t {
    SYSTOLIC_ARRAY = 0,
    VECTOR_ENGINE = 1,
    MEMORY_CONTROLLER = 2,
    DMA_ENGINE = 3,
    BLOCK_MOVER = 4,
    STREAMER = 5,
    NOC_LINK = 6,
    L3_BUFFER = 7,
    L2_BUFFER = 8,
    L1_BUFFER = 9,
    _COUNT = 10
};

constexpr const char* to_string(ResourceType type) {
    switch (type) {
        case ResourceType::SYSTOLIC_ARRAY: return "systolic_array";
        case ResourceType::VECTOR_ENGINE: return "vector_engine";
        case ResourceType::MEMORY_CONTROLLER: return "memory_controller";
        case ResourceType::DMA_ENGINE: return "dma_engine";
        case ResourceType::BLOCK_MOVER: return "block_mover";
        case ResourceType::STREAMER: return "streamer";
        case ResourceType::NOC_LINK: return "noc_link";
        case ResourceType::L3_BUFFER: return "l3_buffer";
        case ResourceType::L2_BUFFER: return "l2_buffer";
        case ResourceType::L1_BUFFER: return "l1_buffer";
        default: return "unknown";
    }
}

/**
 * @brief Utilization statistics for a single resource instance
 */
struct ResourceUtilization {
    std::atomic<uint64_t> busy_cycles{0};
    std::atomic<uint64_t> idle_cycles{0};
    std::atomic<uint64_t> stall_cycles{0};
    std::atomic<uint64_t> operations{0};
    std::atomic<uint64_t> work_units{0};  // FLOPs, bytes, etc.

    // Default constructor
    ResourceUtilization() = default;

    // Copy constructor (load values from atomics)
    ResourceUtilization(const ResourceUtilization& other)
        : busy_cycles(other.busy_cycles.load(std::memory_order_relaxed))
        , idle_cycles(other.idle_cycles.load(std::memory_order_relaxed))
        , stall_cycles(other.stall_cycles.load(std::memory_order_relaxed))
        , operations(other.operations.load(std::memory_order_relaxed))
        , work_units(other.work_units.load(std::memory_order_relaxed)) {}

    // Move constructor
    ResourceUtilization(ResourceUtilization&& other) noexcept
        : busy_cycles(other.busy_cycles.load(std::memory_order_relaxed))
        , idle_cycles(other.idle_cycles.load(std::memory_order_relaxed))
        , stall_cycles(other.stall_cycles.load(std::memory_order_relaxed))
        , operations(other.operations.load(std::memory_order_relaxed))
        , work_units(other.work_units.load(std::memory_order_relaxed)) {}

    // Copy assignment
    ResourceUtilization& operator=(const ResourceUtilization& other) {
        if (this != &other) {
            busy_cycles.store(other.busy_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            idle_cycles.store(other.idle_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stall_cycles.store(other.stall_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            operations.store(other.operations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            work_units.store(other.work_units.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // Move assignment
    ResourceUtilization& operator=(ResourceUtilization&& other) noexcept {
        if (this != &other) {
            busy_cycles.store(other.busy_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            idle_cycles.store(other.idle_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stall_cycles.store(other.stall_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            operations.store(other.operations.load(std::memory_order_relaxed), std::memory_order_relaxed);
            work_units.store(other.work_units.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    void record_busy(uint64_t cycles) {
        busy_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    void record_idle(uint64_t cycles) {
        idle_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    void record_stall(uint64_t cycles) {
        stall_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    void record_operation(uint64_t work = 1) {
        operations.fetch_add(1, std::memory_order_relaxed);
        work_units.fetch_add(work, std::memory_order_relaxed);
    }

    uint64_t total_cycles() const {
        return busy_cycles.load() + idle_cycles.load() + stall_cycles.load();
    }

    /**
     * @brief Get utilization (U = busy / total)
     */
    double utilization() const {
        uint64_t total = total_cycles();
        return total > 0 ? static_cast<double>(busy_cycles.load()) / total : 0.0;
    }

    /**
     * @brief Get efficiency (E = busy / (busy + stall))
     */
    double efficiency() const {
        uint64_t busy = busy_cycles.load();
        uint64_t stall = stall_cycles.load();
        return (busy + stall) > 0 ? static_cast<double>(busy) / (busy + stall) : 1.0;
    }

    /**
     * @brief Get throughput (X = operations / total_cycles)
     */
    double throughput() const {
        uint64_t total = total_cycles();
        return total > 0 ? static_cast<double>(operations.load()) / total : 0.0;
    }

    void reset() {
        busy_cycles.store(0);
        idle_cycles.store(0);
        stall_cycles.store(0);
        operations.store(0);
        work_units.store(0);
    }

    void merge(const ResourceUtilization& other) {
        busy_cycles.fetch_add(other.busy_cycles.load());
        idle_cycles.fetch_add(other.idle_cycles.load());
        stall_cycles.fetch_add(other.stall_cycles.load());
        operations.fetch_add(other.operations.load());
        work_units.fetch_add(other.work_units.load());
    }
};

/**
 * @brief Unique identifier for a resource instance
 */
struct ResourceId {
    ResourceType type;
    uint32_t instance;

    bool operator<(const ResourceId& other) const {
        if (type != other.type) return type < other.type;
        return instance < other.instance;
    }

    bool operator==(const ResourceId& other) const {
        return type == other.type && instance == other.instance;
    }

    std::string to_string() const {
        return std::string(sw::kpu::stats::to_string(type)) +
               "[" + std::to_string(instance) + "]";
    }
};

/**
 * @brief Utilization tracker for all resources
 */
class UtilizationTracker {
public:
    static constexpr size_t NUM_TYPES = static_cast<size_t>(ResourceType::_COUNT);

    UtilizationTracker() = default;

    // Copy constructor - creates fresh mutex, copies resources
    UtilizationTracker(const UtilizationTracker& other) {
        std::lock_guard<std::mutex> lock(other.mutex_);
        resources_ = other.resources_;
    }

    // Move constructor - creates fresh mutex, moves resources
    UtilizationTracker(UtilizationTracker&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        resources_ = std::move(other.resources_);
    }

    // Copy assignment - fresh mutex, copy resources
    UtilizationTracker& operator=(const UtilizationTracker& other) {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(mutex_);
            std::lock_guard<std::mutex> lock2(other.mutex_);
            resources_ = other.resources_;
        }
        return *this;
    }

    // Move assignment - fresh mutex, move resources
    UtilizationTracker& operator=(UtilizationTracker&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock1(mutex_);
            std::lock_guard<std::mutex> lock2(other.mutex_);
            resources_ = std::move(other.resources_);
        }
        return *this;
    }

    /**
     * @brief Register a resource for tracking
     */
    void register_resource(ResourceType type, uint32_t instance) {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceId id{type, instance};
        resources_[id] = ResourceUtilization{};
    }

    /**
     * @brief Get or create utilization tracker for a resource
     */
    ResourceUtilization& get(ResourceType type, uint32_t instance = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceId id{type, instance};
        return resources_[id];
    }

    /**
     * @brief Record busy cycles for a resource
     */
    void record_busy(ResourceType type, uint32_t instance, uint64_t cycles) {
        get(type, instance).record_busy(cycles);
    }

    /**
     * @brief Record idle cycles for a resource
     */
    void record_idle(ResourceType type, uint32_t instance, uint64_t cycles) {
        get(type, instance).record_idle(cycles);
    }

    /**
     * @brief Record stall cycles for a resource
     */
    void record_stall(ResourceType type, uint32_t instance, uint64_t cycles) {
        get(type, instance).record_stall(cycles);
    }

    /**
     * @brief Record an operation on a resource
     */
    void record_operation(ResourceType type, uint32_t instance, uint64_t work = 1) {
        get(type, instance).record_operation(work);
    }

    /**
     * @brief Get aggregate utilization for a resource type
     */
    ResourceUtilization get_aggregate(ResourceType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceUtilization agg;
        for (const auto& [id, util] : resources_) {
            if (id.type == type) {
                agg.merge(util);
            }
        }
        return agg;
    }

    /**
     * @brief Get average utilization for a resource type
     */
    double average_utilization(ResourceType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        double sum = 0.0;
        size_t count = 0;
        for (const auto& [id, util] : resources_) {
            if (id.type == type) {
                sum += util.utilization();
                ++count;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    /**
     * @brief Get overall system utilization
     */
    double overall_utilization() const {
        std::lock_guard<std::mutex> lock(mutex_);
        double sum = 0.0;
        size_t count = 0;
        for (const auto& [id, util] : resources_) {
            sum += util.utilization();
            ++count;
        }
        return count > 0 ? sum / count : 0.0;
    }

    /**
     * @brief Reset all utilization counters
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, util] : resources_) {
            util.reset();
        }
    }

    /**
     * @brief Generate JSON representation
     */
    std::string to_json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{\n";
        ss << "  \"overall_utilization\": " << overall_utilization_unlocked() << ",\n";
        ss << "  \"resources\": {\n";

        bool first = true;
        for (const auto& [id, util] : resources_) {
            if (!first) ss << ",\n";
            first = false;
            ss << "    \"" << id.to_string() << "\": {\n";
            ss << "      \"busy_cycles\": " << util.busy_cycles.load() << ",\n";
            ss << "      \"idle_cycles\": " << util.idle_cycles.load() << ",\n";
            ss << "      \"stall_cycles\": " << util.stall_cycles.load() << ",\n";
            ss << "      \"operations\": " << util.operations.load() << ",\n";
            ss << "      \"work_units\": " << util.work_units.load() << ",\n";
            ss << "      \"utilization\": " << util.utilization() << ",\n";
            ss << "      \"efficiency\": " << util.efficiency() << ",\n";
            ss << "      \"throughput\": " << util.throughput() << "\n";
            ss << "    }";
        }

        ss << "\n  },\n";
        ss << "  \"by_type\": {\n";

        first = true;
        for (size_t i = 0; i < NUM_TYPES; ++i) {
            ResourceType type = static_cast<ResourceType>(i);
            double avg = average_utilization_unlocked(type);
            if (avg > 0 || has_type_unlocked(type)) {
                if (!first) ss << ",\n";
                first = false;
                ss << "    \"" << to_string(type) << "\": " << avg;
            }
        }
        ss << "\n  }\n";
        ss << "}";
        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1);
        ss << "Resource Utilization Summary\n";
        ss << "  Overall Utilization: " << (overall_utilization_unlocked() * 100) << "%\n\n";

        ss << "  Resource Type       Util%    Eff%     Ops       X (ops/cyc)\n";
        ss << "  -------------------------------------------------------------\n";

        for (size_t i = 0; i < NUM_TYPES; ++i) {
            ResourceType type = static_cast<ResourceType>(i);
            auto agg = get_aggregate_unlocked(type);
            if (agg.total_cycles() > 0) {
                ss << "  " << std::left << std::setw(18) << to_string(type)
                   << std::right << std::setw(8) << (agg.utilization() * 100)
                   << std::setw(8) << (agg.efficiency() * 100)
                   << std::setw(10) << agg.operations.load()
                   << std::setw(14) << agg.throughput() << "\n";
            }
        }
        return ss.str();
    }

private:
    // Unlocked versions for internal use
    double overall_utilization_unlocked() const {
        double sum = 0.0;
        size_t count = 0;
        for (const auto& [id, util] : resources_) {
            sum += util.utilization();
            ++count;
        }
        return count > 0 ? sum / count : 0.0;
    }

    double average_utilization_unlocked(ResourceType type) const {
        double sum = 0.0;
        size_t count = 0;
        for (const auto& [id, util] : resources_) {
            if (id.type == type) {
                sum += util.utilization();
                ++count;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    ResourceUtilization get_aggregate_unlocked(ResourceType type) const {
        ResourceUtilization agg;
        for (const auto& [id, util] : resources_) {
            if (id.type == type) {
                agg.merge(util);
            }
        }
        return agg;
    }

    bool has_type_unlocked(ResourceType type) const {
        for (const auto& [id, util] : resources_) {
            if (id.type == type) return true;
        }
        return false;
    }

    mutable std::mutex mutex_;
    std::map<ResourceId, ResourceUtilization> resources_;
};

} // namespace sw::kpu::stats
