/**
 * @file memory_traffic.hpp
 * @brief Memory Traffic Statistics for KPU Simulator
 * @version 0.3.4
 *
 * Tracks data movement across the memory hierarchy:
 *   - External (DRAM)
 *   - L3 Buffer
 *   - L2 Buffer
 *   - L1 Stream
 *
 * Enables analysis of:
 *   - Bandwidth utilization at each level
 *   - Data reuse (traffic amplification)
 *   - Bottleneck identification
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
 * @brief Memory hierarchy levels
 */
enum class MemoryLevel : uint8_t {
    EXTERNAL = 0,   // DRAM / off-chip
    L3 = 1,         // L3 buffer (closest to external)
    L2 = 2,         // L2 buffer
    L1 = 3,         // L1 stream (closest to compute)
    REGISTER = 4,   // Register file / accumulator
    _COUNT = 5
};

constexpr const char* to_string(MemoryLevel level) {
    switch (level) {
        case MemoryLevel::EXTERNAL: return "external";
        case MemoryLevel::L3: return "l3";
        case MemoryLevel::L2: return "l2";
        case MemoryLevel::L1: return "l1";
        case MemoryLevel::REGISTER: return "register";
        default: return "unknown";
    }
}

/**
 * @brief Traffic statistics for a single memory level
 */
struct LevelTraffic {
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> write_bytes{0};
    std::atomic<uint64_t> read_count{0};
    std::atomic<uint64_t> write_count{0};
    std::atomic<uint64_t> read_cycles{0};
    std::atomic<uint64_t> write_cycles{0};

    // Default constructor
    LevelTraffic() = default;

    // Copy constructor
    LevelTraffic(const LevelTraffic& other)
        : read_bytes(other.read_bytes.load(std::memory_order_relaxed))
        , write_bytes(other.write_bytes.load(std::memory_order_relaxed))
        , read_count(other.read_count.load(std::memory_order_relaxed))
        , write_count(other.write_count.load(std::memory_order_relaxed))
        , read_cycles(other.read_cycles.load(std::memory_order_relaxed))
        , write_cycles(other.write_cycles.load(std::memory_order_relaxed)) {}

    // Move constructor
    LevelTraffic(LevelTraffic&& other) noexcept
        : read_bytes(other.read_bytes.load(std::memory_order_relaxed))
        , write_bytes(other.write_bytes.load(std::memory_order_relaxed))
        , read_count(other.read_count.load(std::memory_order_relaxed))
        , write_count(other.write_count.load(std::memory_order_relaxed))
        , read_cycles(other.read_cycles.load(std::memory_order_relaxed))
        , write_cycles(other.write_cycles.load(std::memory_order_relaxed)) {}

    // Copy assignment
    LevelTraffic& operator=(const LevelTraffic& other) {
        if (this != &other) {
            read_bytes.store(other.read_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_bytes.store(other.write_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            read_count.store(other.read_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_count.store(other.write_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            read_cycles.store(other.read_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_cycles.store(other.write_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    // Move assignment
    LevelTraffic& operator=(LevelTraffic&& other) noexcept {
        if (this != &other) {
            read_bytes.store(other.read_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_bytes.store(other.write_bytes.load(std::memory_order_relaxed), std::memory_order_relaxed);
            read_count.store(other.read_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_count.store(other.write_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            read_cycles.store(other.read_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
            write_cycles.store(other.write_cycles.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    void record_read(uint64_t bytes, uint64_t cycles = 0) {
        read_bytes.fetch_add(bytes, std::memory_order_relaxed);
        read_count.fetch_add(1, std::memory_order_relaxed);
        read_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    void record_write(uint64_t bytes, uint64_t cycles = 0) {
        write_bytes.fetch_add(bytes, std::memory_order_relaxed);
        write_count.fetch_add(1, std::memory_order_relaxed);
        write_cycles.fetch_add(cycles, std::memory_order_relaxed);
    }

    uint64_t total_bytes() const {
        return read_bytes.load(std::memory_order_relaxed) +
               write_bytes.load(std::memory_order_relaxed);
    }

    uint64_t total_count() const {
        return read_count.load(std::memory_order_relaxed) +
               write_count.load(std::memory_order_relaxed);
    }

    uint64_t total_cycles() const {
        return read_cycles.load(std::memory_order_relaxed) +
               write_cycles.load(std::memory_order_relaxed);
    }

    double avg_read_size() const {
        uint64_t count = read_count.load();
        return count > 0 ? static_cast<double>(read_bytes.load()) / static_cast<double>(count) : 0.0;
    }

    double avg_write_size() const {
        uint64_t count = write_count.load();
        return count > 0 ? static_cast<double>(write_bytes.load()) / static_cast<double>(count) : 0.0;
    }

    void reset() {
        read_bytes.store(0);
        write_bytes.store(0);
        read_count.store(0);
        write_count.store(0);
        read_cycles.store(0);
        write_cycles.store(0);
    }

    void merge(const LevelTraffic& other) {
        read_bytes.fetch_add(other.read_bytes.load());
        write_bytes.fetch_add(other.write_bytes.load());
        read_count.fetch_add(other.read_count.load());
        write_count.fetch_add(other.write_count.load());
        read_cycles.fetch_add(other.read_cycles.load());
        write_cycles.fetch_add(other.write_cycles.load());
    }
};

/**
 * @brief Memory traffic statistics across all hierarchy levels
 */
class MemoryTraffic {
public:
    static constexpr size_t NUM_LEVELS = static_cast<size_t>(MemoryLevel::_COUNT);

    // Default bandwidths (GB/s) for each level
    static constexpr double DEFAULT_EXTERNAL_BW = 64.0;
    static constexpr double DEFAULT_L3_BW = 128.0;
    static constexpr double DEFAULT_L2_BW = 256.0;
    static constexpr double DEFAULT_L1_BW = 512.0;
    static constexpr double DEFAULT_REG_BW = 1024.0;

    MemoryTraffic() {
        peak_bandwidth_[0] = DEFAULT_EXTERNAL_BW;
        peak_bandwidth_[1] = DEFAULT_L3_BW;
        peak_bandwidth_[2] = DEFAULT_L2_BW;
        peak_bandwidth_[3] = DEFAULT_L1_BW;
        peak_bandwidth_[4] = DEFAULT_REG_BW;
    }

    // Copy constructor
    MemoryTraffic(const MemoryTraffic& other)
        : peak_bandwidth_(other.peak_bandwidth_) {
        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            levels_[i] = other.levels_[i];
        }
    }

    // Move constructor
    MemoryTraffic(MemoryTraffic&& other) noexcept
        : peak_bandwidth_(std::move(other.peak_bandwidth_)) {
        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            levels_[i] = std::move(other.levels_[i]);
        }
    }

    // Copy assignment
    MemoryTraffic& operator=(const MemoryTraffic& other) {
        if (this != &other) {
            peak_bandwidth_ = other.peak_bandwidth_;
            for (size_t i = 0; i < NUM_LEVELS; ++i) {
                levels_[i] = other.levels_[i];
            }
        }
        return *this;
    }

    // Move assignment
    MemoryTraffic& operator=(MemoryTraffic&& other) noexcept {
        if (this != &other) {
            peak_bandwidth_ = std::move(other.peak_bandwidth_);
            for (size_t i = 0; i < NUM_LEVELS; ++i) {
                levels_[i] = std::move(other.levels_[i]);
            }
        }
        return *this;
    }

    /**
     * @brief Set peak bandwidth for a level
     */
    void set_peak_bandwidth(MemoryLevel level, double bw_gbs) {
        size_t idx = static_cast<size_t>(level);
        if (idx < NUM_LEVELS) {
            peak_bandwidth_[idx] = bw_gbs;
        }
    }

    /**
     * @brief Record a read at a memory level
     */
    void record_read(MemoryLevel level, uint64_t bytes, uint64_t cycles = 0) {
        size_t idx = static_cast<size_t>(level);
        if (idx < NUM_LEVELS) {
            levels_[idx].record_read(bytes, cycles);
        }
    }

    /**
     * @brief Record a write at a memory level
     */
    void record_write(MemoryLevel level, uint64_t bytes, uint64_t cycles = 0) {
        size_t idx = static_cast<size_t>(level);
        if (idx < NUM_LEVELS) {
            levels_[idx].record_write(bytes, cycles);
        }
    }

    /**
     * @brief Get traffic for a specific level
     */
    const LevelTraffic& get_level(MemoryLevel level) const {
        size_t idx = static_cast<size_t>(level);
        return levels_[idx < NUM_LEVELS ? idx : 0];
    }

    /**
     * @brief Get total bytes at a level
     */
    uint64_t total_bytes(MemoryLevel level) const {
        return get_level(level).total_bytes();
    }

    /**
     * @brief Get external (DRAM) traffic
     */
    uint64_t external_bytes() const {
        return total_bytes(MemoryLevel::EXTERNAL);
    }

    /**
     * @brief Get total traffic across all levels
     */
    uint64_t total_traffic() const {
        uint64_t total = 0;
        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            total += levels_[i].total_bytes();
        }
        return total;
    }

    /**
     * @brief Calculate traffic amplification (internal / external)
     *
     * A value > 1 indicates data reuse in the hierarchy.
     */
    double traffic_amplification() const {
        uint64_t external = external_bytes();
        uint64_t internal = total_traffic() - external;
        return external > 0 ? static_cast<double>(internal) / static_cast<double>(external) : 0.0;
    }

    /**
     * @brief Calculate achieved bandwidth at a level
     */
    double achieved_bandwidth(MemoryLevel level, double clock_ghz, uint64_t total_cycles) const {
        if (total_cycles == 0) return 0.0;
        uint64_t bytes = total_bytes(level);
        double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        return (static_cast<double>(bytes) / 1e9) / seconds;  // GB/s
    }

    /**
     * @brief Calculate bandwidth utilization at a level
     */
    double bandwidth_utilization(MemoryLevel level, double clock_ghz, uint64_t total_cycles) const {
        size_t idx = static_cast<size_t>(level);
        if (idx >= NUM_LEVELS) return 0.0;
        double achieved = achieved_bandwidth(level, clock_ghz, total_cycles);
        return peak_bandwidth_[idx] > 0 ? achieved / peak_bandwidth_[idx] : 0.0;
    }

    /**
     * @brief Reset all counters
     */
    void reset() {
        for (auto& level : levels_) {
            level.reset();
        }
    }

    /**
     * @brief Merge another MemoryTraffic into this one
     */
    void merge(const MemoryTraffic& other) {
        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            levels_[i].merge(other.levels_[i]);
        }
    }

    /**
     * @brief Generate JSON representation
     */
    std::string to_json(double clock_ghz = 1.0, uint64_t total_cycles = 0) const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"total_traffic_bytes\": " << total_traffic() << ",\n";
        ss << "  \"external_traffic_bytes\": " << external_bytes() << ",\n";
        ss << "  \"traffic_amplification\": " << traffic_amplification() << ",\n";
        ss << "  \"levels\": {\n";

        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            MemoryLevel level = static_cast<MemoryLevel>(i);
            const auto& lt = levels_[i];
            if (i > 0) ss << ",\n";
            ss << "    \"" << to_string(level) << "\": {\n";
            ss << "      \"read_bytes\": " << lt.read_bytes.load() << ",\n";
            ss << "      \"write_bytes\": " << lt.write_bytes.load() << ",\n";
            ss << "      \"read_count\": " << lt.read_count.load() << ",\n";
            ss << "      \"write_count\": " << lt.write_count.load() << ",\n";
            ss << "      \"total_bytes\": " << lt.total_bytes() << ",\n";
            ss << "      \"peak_bandwidth_gbs\": " << peak_bandwidth_[i] << ",\n";
            ss << "      \"achieved_bandwidth_gbs\": "
               << achieved_bandwidth(level, clock_ghz, total_cycles) << ",\n";
            ss << "      \"bandwidth_utilization\": "
               << bandwidth_utilization(level, clock_ghz, total_cycles) << "\n";
            ss << "    }";
        }
        ss << "\n  }\n";
        ss << "}";
        return ss.str();
    }

    /**
     * @brief Generate human-readable summary
     */
    std::string summary(double clock_ghz = 1.0, uint64_t total_cycles = 0) const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Memory Traffic Summary\n";
        ss << "  Total Traffic:       " << (static_cast<double>(total_traffic()) / 1e6) << " MB\n";
        ss << "  External (DRAM):     " << (static_cast<double>(external_bytes()) / 1e6) << " MB\n";
        ss << "  Traffic Amplification: " << traffic_amplification() << "x\n\n";

        ss << "  Level      Read (MB)   Write (MB)   BW (GB/s)   Util%\n";
        ss << "  -------------------------------------------------------\n";

        for (size_t i = 0; i < NUM_LEVELS; ++i) {
            MemoryLevel level = static_cast<MemoryLevel>(i);
            const auto& lt = levels_[i];
            double achieved = achieved_bandwidth(level, clock_ghz, total_cycles);
            double util = bandwidth_utilization(level, clock_ghz, total_cycles) * 100;
            ss << "  " << std::left << std::setw(10) << to_string(level)
               << std::right << std::setw(10) << (static_cast<double>(lt.read_bytes.load()) / 1e6)
               << std::setw(12) << (static_cast<double>(lt.write_bytes.load()) / 1e6)
               << std::setw(12) << achieved
               << std::setw(10) << util << "%\n";
        }
        return ss.str();
    }

private:
    std::array<LevelTraffic, NUM_LEVELS> levels_;
    std::array<double, NUM_LEVELS> peak_bandwidth_;
};

} // namespace sw::kpu::stats
