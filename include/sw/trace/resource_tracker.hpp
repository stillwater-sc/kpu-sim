#pragma once

#include <sw/trace/trace_entry.hpp>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <optional>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace sw::trace {

// Resource states for tracking occupancy
enum class ResourceState : uint8_t {
    IDLE = 0,       // Resource is available
    BUSY = 1,       // Resource is actively processing
    STALLED = 2,    // Resource is waiting for dependency
    DRAINING = 3,   // Resource is completing pending work
    DISABLED = 4    // Resource is not available
};

inline const char* to_string(ResourceState state) {
    switch (state) {
        case ResourceState::IDLE: return "IDLE";
        case ResourceState::BUSY: return "BUSY";
        case ResourceState::STALLED: return "STALLED";
        case ResourceState::DRAINING: return "DRAINING";
        case ResourceState::DISABLED: return "DISABLED";
        default: return "UNKNOWN";
    }
}

// Unique identifier for a resource instance
struct ResourceId {
    ComponentType type;
    uint32_t id;

    bool operator<(const ResourceId& other) const {
        if (type != other.type) return static_cast<uint8_t>(type) < static_cast<uint8_t>(other.type);
        return id < other.id;
    }

    bool operator==(const ResourceId& other) const {
        return type == other.type && id == other.id;
    }

    std::string to_string() const {
        return std::string(sw::trace::to_string(type)) + "[" + std::to_string(id) + "]";
    }
};

// State transition event
struct StateTransition {
    CycleCount cycle;
    ResourceState from_state;
    ResourceState to_state;
    uint64_t operation_id;       // Associated operation (0 if none)
    std::string reason;          // Why the transition occurred
};

// Per-resource tracking data
struct ResourceTrack {
    ResourceId resource;
    ResourceState current_state = ResourceState::IDLE;
    CycleCount last_transition_cycle = 0;

    // State duration accumulators (in cycles)
    std::map<ResourceState, CycleCount> state_durations;

    // State transition history
    std::vector<StateTransition> transitions;

    // Operation tracking
    uint64_t current_operation_id = 0;
    CycleCount current_operation_start = 0;
    std::vector<std::pair<CycleCount, CycleCount>> operation_intervals;  // (start, end)

    ResourceTrack() = default;
    explicit ResourceTrack(ResourceId res) : resource(res) {}

    // Get total cycles in a given state
    CycleCount get_state_duration(ResourceState state) const {
        auto it = state_durations.find(state);
        return it != state_durations.end() ? it->second : 0;
    }

    // Get utilization as fraction (0.0 to 1.0)
    double get_utilization(CycleCount total_cycles) const {
        if (total_cycles == 0) return 0.0;
        CycleCount busy = get_state_duration(ResourceState::BUSY);
        return static_cast<double>(busy) / static_cast<double>(total_cycles);
    }

    // Get efficiency (busy / (busy + stalled))
    double get_efficiency() const {
        CycleCount busy = get_state_duration(ResourceState::BUSY);
        CycleCount stalled = get_state_duration(ResourceState::STALLED);
        if (busy + stalled == 0) return 1.0;
        return static_cast<double>(busy) / static_cast<double>(busy + stalled);
    }
};

// Snapshot of all resource states at a given cycle
struct ResourceSnapshot {
    CycleCount cycle;
    std::map<ResourceId, ResourceState> states;
    size_t active_count = 0;      // Resources in BUSY state
    size_t stalled_count = 0;     // Resources in STALLED state
    size_t idle_count = 0;        // Resources in IDLE state
};

// Main resource tracker class
class ResourceTracker {
public:
    ResourceTracker() = default;

    // Register a resource for tracking
    void register_resource(ComponentType type, uint32_t id) {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceId res{type, id};
        if (resources_.find(res) == resources_.end()) {
            resources_[res] = ResourceTrack(res);
        }
    }

    // Record a state transition
    void transition(ComponentType type, uint32_t id, ResourceState new_state,
                   CycleCount cycle, uint64_t operation_id = 0,
                   const std::string& reason = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceId res{type, id};

        auto it = resources_.find(res);
        if (it == resources_.end()) {
            resources_[res] = ResourceTrack(res);
            it = resources_.find(res);
        }

        auto& track = it->second;
        ResourceState old_state = track.current_state;

        // Accumulate duration in previous state
        if (cycle > track.last_transition_cycle) {
            track.state_durations[old_state] += (cycle - track.last_transition_cycle);
        }

        // Record transition
        StateTransition trans{cycle, old_state, new_state, operation_id, reason};
        track.transitions.push_back(trans);

        // Update current state
        track.current_state = new_state;
        track.last_transition_cycle = cycle;

        // Track operation intervals
        if (new_state == ResourceState::BUSY && old_state != ResourceState::BUSY) {
            track.current_operation_id = operation_id;
            track.current_operation_start = cycle;
        } else if (old_state == ResourceState::BUSY && new_state != ResourceState::BUSY) {
            if (track.current_operation_start > 0) {
                track.operation_intervals.emplace_back(track.current_operation_start, cycle);
            }
            track.current_operation_id = 0;
            track.current_operation_start = 0;
        }

        // Update max cycle
        if (cycle > max_cycle_) max_cycle_ = cycle;
        if (min_cycle_ == 0 || cycle < min_cycle_) min_cycle_ = cycle;
    }

    // Convenience methods for common transitions
    void mark_busy(ComponentType type, uint32_t id, CycleCount cycle,
                  uint64_t operation_id = 0, const std::string& reason = "") {
        transition(type, id, ResourceState::BUSY, cycle, operation_id, reason);
    }

    void mark_idle(ComponentType type, uint32_t id, CycleCount cycle,
                  const std::string& reason = "") {
        transition(type, id, ResourceState::IDLE, cycle, 0, reason);
    }

    void mark_stalled(ComponentType type, uint32_t id, CycleCount cycle,
                     uint64_t waiting_for = 0, const std::string& reason = "") {
        transition(type, id, ResourceState::STALLED, cycle, waiting_for, reason);
    }

    // Finalize tracking at end of simulation
    void finalize(CycleCount end_cycle) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [res_id, track] : resources_) {
            if (end_cycle > track.last_transition_cycle) {
                track.state_durations[track.current_state] +=
                    (end_cycle - track.last_transition_cycle);
                track.last_transition_cycle = end_cycle;
            }
        }
        max_cycle_ = end_cycle;
    }

    // Get snapshot of all resources at a given cycle
    ResourceSnapshot get_snapshot(CycleCount cycle) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceSnapshot snap;
        snap.cycle = cycle;

        for (const auto& [res_id, track] : resources_) {
            // Find state at this cycle by walking transitions
            ResourceState state = ResourceState::IDLE;
            for (const auto& trans : track.transitions) {
                if (trans.cycle <= cycle) {
                    state = trans.to_state;
                } else {
                    break;
                }
            }
            snap.states[res_id] = state;

            switch (state) {
                case ResourceState::BUSY: snap.active_count++; break;
                case ResourceState::STALLED: snap.stalled_count++; break;
                case ResourceState::IDLE: snap.idle_count++; break;
                default: break;
            }
        }
        return snap;
    }

    // Get concurrency over time (number of BUSY resources at each cycle)
    // Uses sweep-line algorithm for O(n log n) instead of O(samples * transitions)
    std::vector<std::pair<CycleCount, size_t>> get_concurrency_timeline(
            CycleCount sample_interval = 100) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_concurrency_timeline_unlocked(sample_interval);
    }

private:
    // Internal version without locking (for use when lock is already held)
    std::vector<std::pair<CycleCount, size_t>> get_concurrency_timeline_unlocked(
            CycleCount sample_interval) const {
        std::vector<std::pair<CycleCount, size_t>> timeline;

        if (min_cycle_ >= max_cycle_ || resources_.empty()) return timeline;

        // Build sorted list of (cycle, delta) events
        // delta = +1 when entering BUSY, -1 when leaving BUSY
        std::vector<std::pair<CycleCount, int>> events;
        events.reserve(resources_.size() * 4);  // Rough estimate

        for (const auto& [res_id, track] : resources_) {
            ResourceState prev_state = ResourceState::IDLE;
            for (const auto& trans : track.transitions) {
                bool was_busy = (prev_state == ResourceState::BUSY);
                bool is_busy = (trans.to_state == ResourceState::BUSY);
                if (!was_busy && is_busy) {
                    events.emplace_back(trans.cycle, +1);
                } else if (was_busy && !is_busy) {
                    events.emplace_back(trans.cycle, -1);
                }
                prev_state = trans.to_state;
            }
        }

        // Sort events by cycle
        std::sort(events.begin(), events.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // Process events to build timeline at sample points
        int current_busy = 0;
        size_t event_idx = 0;

        for (CycleCount c = min_cycle_; c <= max_cycle_; c += sample_interval) {
            // Process all events up to and including this cycle
            while (event_idx < events.size() && events[event_idx].first <= c) {
                current_busy += events[event_idx].second;
                event_idx++;
            }
            timeline.emplace_back(c, static_cast<size_t>(std::max(0, current_busy)));
        }

        return timeline;
    }

public:

    // Get resource track for a specific resource
    std::optional<ResourceTrack> get_resource_track(ComponentType type, uint32_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        ResourceId res{type, id};
        auto it = resources_.find(res);
        if (it != resources_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Get all resource tracks
    std::map<ResourceId, ResourceTrack> get_all_tracks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return resources_;
    }

    // Get aggregate statistics
    struct AggregateStats {
        size_t total_resources = 0;
        CycleCount total_cycles = 0;

        // Per-resource-type stats
        std::map<ComponentType, size_t> resource_counts;
        std::map<ComponentType, double> avg_utilization;
        std::map<ComponentType, double> avg_efficiency;

        // Concurrency stats
        double avg_concurrency = 0.0;
        size_t max_concurrency = 0;
        size_t min_concurrency = 0;

        // Overall
        double overall_utilization = 0.0;
    };

    AggregateStats get_aggregate_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        AggregateStats stats;

        CycleCount total_cycles = max_cycle_ - min_cycle_;
        stats.total_cycles = total_cycles;
        stats.total_resources = resources_.size();

        if (total_cycles == 0 || resources_.empty()) return stats;

        // Accumulate per-type stats
        std::map<ComponentType, double> utilization_sums;
        std::map<ComponentType, double> efficiency_sums;

        for (const auto& [res_id, track] : resources_) {
            ComponentType type = res_id.type;
            stats.resource_counts[type]++;
            utilization_sums[type] += track.get_utilization(total_cycles);
            efficiency_sums[type] += track.get_efficiency();
        }

        // Calculate averages
        double total_util = 0.0;
        for (const auto& [type, count] : stats.resource_counts) {
            if (count > 0) {
                stats.avg_utilization[type] = utilization_sums[type] / count;
                stats.avg_efficiency[type] = efficiency_sums[type] / count;
                total_util += utilization_sums[type];
            }
        }
        stats.overall_utilization = total_util / static_cast<double>(resources_.size());

        // Concurrency stats from timeline (use unlocked version since we hold the lock)
        auto timeline = get_concurrency_timeline_unlocked(100);
        if (!timeline.empty()) {
            size_t sum = 0;
            stats.max_concurrency = 0;
            stats.min_concurrency = resources_.size();
            for (const auto& [cycle, count] : timeline) {
                sum += count;
                stats.max_concurrency = std::max(stats.max_concurrency, count);
                stats.min_concurrency = std::min(stats.min_concurrency, count);
            }
            stats.avg_concurrency = static_cast<double>(sum) / static_cast<double>(timeline.size());
        }

        return stats;
    }

    // Clear all tracking data
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        resources_.clear();
        min_cycle_ = 0;
        max_cycle_ = 0;
    }

    // Get cycle range
    CycleCount get_min_cycle() const { return min_cycle_; }
    CycleCount get_max_cycle() const { return max_cycle_; }

private:
    mutable std::mutex mutex_;
    std::map<ResourceId, ResourceTrack> resources_;
    CycleCount min_cycle_ = 0;
    CycleCount max_cycle_ = 0;
};

} // namespace sw::trace
