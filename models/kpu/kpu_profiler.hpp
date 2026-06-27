/**
 * @file kpu_profiler.hpp
 * @brief Profiling and instrumentation utilities for KPU execution
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sw/concepts.hpp>

namespace sw::kpu {

/**
 * @brief Tracks timing and performance metrics for a single operation
 */
struct ProfileEvent {
    std::string name;
    Cycle start_cycle;
    Cycle end_cycle;
    size_t bytes_transferred;
    std::string stage_type;  // "memory_transfer", "compute", etc.

    ProfileEvent() : name(""), start_cycle(0), end_cycle(0), bytes_transferred(0), stage_type("generic") {}

    ProfileEvent(const std::string& n, Cycle start = 0)
        : name(n), start_cycle(start), end_cycle(0), bytes_transferred(0), stage_type("generic") {}

    Cycle duration() const { return end_cycle - start_cycle; }

    double bandwidth_gbps(double cycle_time_ns = 1.0) const {
        if (duration() == 0) return 0.0;
        double time_sec = static_cast<double>(duration()) * cycle_time_ns * 1e-9;
        return (static_cast<double>(bytes_transferred) / 1e9) / time_sec;
    }
};

/**
 * @brief Profiler for tracking KPU execution performance
 */
class KPUProfiler {
public:
    explicit KPUProfiler(bool enabled = true) : enabled_(enabled), current_cycle_(0) {}

    // Event tracking
    void start_event(const std::string& name, Cycle cycle) {
        if (!enabled_) return;
        ProfileEvent event(name, cycle);
        active_events_[name] = event;
    }

    void end_event(const std::string& name, Cycle cycle, size_t bytes = 0) {
        if (!enabled_) return;
        auto it = active_events_.find(name);
        if (it != active_events_.end()) {
            it->second.end_cycle = cycle;
            it->second.bytes_transferred = bytes;
            completed_events_.push_back(it->second);
            active_events_.erase(it);
        }
    }

    void record_event(const std::string& name, const std::string& stage_type,
                      Cycle start, Cycle end, size_t bytes = 0) {
        if (!enabled_) return;
        ProfileEvent event(name, start);
        event.end_cycle = end;
        event.bytes_transferred = bytes;
        event.stage_type = stage_type;
        completed_events_.push_back(event);
    }

    // Tile-level tracking
    struct TileMetrics {
        size_t tile_id;
        size_t m_idx, n_idx, k_idx;  // Tile coordinates
        Cycle start_cycle;
        Cycle end_cycle;
        Cycle load_a_cycles;
        Cycle load_b_cycles;
        Cycle compute_cycles;
        Cycle store_c_cycles;

        TileMetrics() : tile_id(0), m_idx(0), n_idx(0), k_idx(0),
                       start_cycle(0), end_cycle(0), load_a_cycles(0),
                       load_b_cycles(0), compute_cycles(0), store_c_cycles(0) {}
    };

    void start_tile(size_t tile_id, size_t m_idx, size_t n_idx, size_t k_idx, Cycle cycle) {
        if (!enabled_) return;
        TileMetrics metrics;
        metrics.tile_id = tile_id;
        metrics.m_idx = m_idx;
        metrics.n_idx = n_idx;
        metrics.k_idx = k_idx;
        metrics.start_cycle = cycle;
        current_tile_ = metrics;
    }

    void end_tile(Cycle cycle, Cycle load_a, Cycle load_b, Cycle compute, Cycle store_c) {
        if (!enabled_) return;
        current_tile_.end_cycle = cycle;
        current_tile_.load_a_cycles = load_a;
        current_tile_.load_b_cycles = load_b;
        current_tile_.compute_cycles = compute;
        current_tile_.store_c_cycles = store_c;
        tile_metrics_.push_back(current_tile_);
    }

    // Component utilization tracking
    void record_component_usage(const std::string& component, Cycle cycles) {
        component_cycles_[component] += cycles;
    }

    // Memory bandwidth tracking
    void record_memory_transfer(const std::string& source, const std::string& dest,
                                size_t bytes, Cycle cycles) {
        std::string path = source + "->" + dest;
        bandwidth_stats_[path].bytes += bytes;
        bandwidth_stats_[path].cycles += cycles;
    }

    // Report generation
    void print_summary(Cycle total_cycles, double cycle_time_ns = 1.0) const {
        if (!enabled_) return;

        std::cout << "\n========================================\n";
        std::cout << "  KPU Profiler Summary\n";
        std::cout << "========================================\n";

        // Pipeline stages summary
        std::map<std::string, std::vector<Cycle>> stage_durations;
        size_t total_bytes = 0;

        for (const auto& event : completed_events_) {
            stage_durations[event.stage_type].push_back(event.duration());
            total_bytes += event.bytes_transferred;
        }

        std::cout << "\nPipeline Stages:\n";
        std::cout << std::left << std::setw(25) << "Stage"
                  << std::right << std::setw(10) << "Count"
                  << std::setw(12) << "Avg Cycles"
                  << std::setw(12) << "Total Cycles" << "\n";
        std::cout << std::string(59, '-') << "\n";

        for (const auto& [stage, durations] : stage_durations) {
            Cycle total = 0;
            for (auto d : durations) total += d;
            Cycle avg = durations.empty() ? 0 : total / durations.size();

            std::cout << std::left << std::setw(25) << stage
                      << std::right << std::setw(10) << durations.size()
                      << std::setw(12) << avg
                      << std::setw(12) << total << "\n";
        }

        // Component utilization
        if (!component_cycles_.empty()) {
            std::cout << "\nComponent Utilization:\n";
            std::cout << std::left << std::setw(25) << "Component"
                      << std::right << std::setw(12) << "Cycles"
                      << std::setw(12) << "Utilization" << "\n";
            std::cout << std::string(49, '-') << "\n";

            for (const auto& [component, cycles] : component_cycles_) {
                double util = (100.0 * static_cast<double>(cycles)) / static_cast<double>(total_cycles);
                std::cout << std::left << std::setw(25) << component
                          << std::right << std::setw(12) << cycles
                          << std::setw(11) << std::fixed << std::setprecision(1) << util << "%\n";
            }
        }

        // Memory bandwidth
        if (!bandwidth_stats_.empty()) {
            std::cout << "\nMemory Bandwidth:\n";
            std::cout << std::left << std::setw(20) << "Path"
                      << std::right << std::setw(12) << "Bytes"
                      << std::setw(10) << "Cycles"
                      << std::setw(15) << "BW (GB/s)" << "\n";
            std::cout << std::string(57, '-') << "\n";

            for (const auto& [path, stats] : bandwidth_stats_) {
                double time_sec = static_cast<double>(stats.cycles) * cycle_time_ns * 1e-9;
                double bw_gbps = time_sec > 0 ? (static_cast<double>(stats.bytes) / 1e9) / time_sec : 0.0;

                std::cout << std::left << std::setw(20) << path
                          << std::right << std::setw(12) << stats.bytes
                          << std::setw(10) << stats.cycles
                          << std::setw(15) << std::fixed << std::setprecision(2) << bw_gbps << "\n";
            }
        }

        // Tile breakdown
        if (!tile_metrics_.empty()) {
            std::cout << "\nTile Execution Breakdown:\n";
            std::cout << std::left << std::setw(8) << "Tile"
                      << std::right << std::setw(10) << "Load A"
                      << std::setw(10) << "Load B"
                      << std::setw(10) << "Compute"
                      << std::setw(10) << "Store C"
                      << std::setw(12) << "Total" << "\n";
            std::cout << std::string(60, '-') << "\n";

            for (const auto& tile : tile_metrics_) {
                Cycle total = tile.end_cycle - tile.start_cycle;
                std::cout << std::left << std::setw(8) << tile.tile_id
                          << std::right << std::setw(10) << tile.load_a_cycles
                          << std::setw(10) << tile.load_b_cycles
                          << std::setw(10) << tile.compute_cycles
                          << std::setw(10) << tile.store_c_cycles
                          << std::setw(12) << total << "\n";
            }
        }

        // Overall summary
        std::cout << "\nOverall:\n";
        std::cout << "  Total cycles: " << total_cycles << "\n";
        std::cout << "  Total data transferred: " << (static_cast<double>(total_bytes) / 1024.0) << " KB\n";
        std::cout << "  Events recorded: " << completed_events_.size() << "\n";
        std::cout << "========================================\n";
    }

    void print_detailed_timeline() const {
        if (!enabled_ || completed_events_.empty()) return;

        std::cout << "\n========================================\n";
        std::cout << "  Detailed Event Timeline\n";
        std::cout << "========================================\n";

        std::cout << std::left << std::setw(35) << "Event"
                  << std::right << std::setw(10) << "Start"
                  << std::setw(10) << "End"
                  << std::setw(10) << "Duration"
                  << std::setw(12) << "Bytes" << "\n";
        std::cout << std::string(77, '-') << "\n";

        for (const auto& event : completed_events_) {
            std::cout << std::left << std::setw(35) << event.name
                      << std::right << std::setw(10) << event.start_cycle
                      << std::setw(10) << event.end_cycle
                      << std::setw(10) << event.duration()
                      << std::setw(12) << event.bytes_transferred << "\n";
        }
        std::cout << "========================================\n";
    }

    void reset() {
        completed_events_.clear();
        active_events_.clear();
        tile_metrics_.clear();
        component_cycles_.clear();
        bandwidth_stats_.clear();
        current_cycle_ = 0;
    }

    bool is_enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    const std::vector<ProfileEvent>& get_events() const { return completed_events_; }
    const std::vector<TileMetrics>& get_tile_metrics() const { return tile_metrics_; }

private:
    bool enabled_;
    Cycle current_cycle_;

    std::vector<ProfileEvent> completed_events_;
    std::map<std::string, ProfileEvent> active_events_;

    std::vector<TileMetrics> tile_metrics_;
    TileMetrics current_tile_;

    std::map<std::string, Cycle> component_cycles_;

    struct BandwidthStats {
        size_t bytes = 0;
        Cycle cycles = 0;
    };
    std::map<std::string, BandwidthStats> bandwidth_stats_;
};

} // namespace sw::kpu
