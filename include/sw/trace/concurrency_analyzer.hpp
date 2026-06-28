#pragma once

#include <sw/trace/trace_logger.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>

namespace sw::trace {

// Operation overlap analysis
struct OperationOverlap {
    uint64_t op1_id;
    uint64_t op2_id;
    ComponentType op1_type;
    ComponentType op2_type;
    CycleCount overlap_start;
    CycleCount overlap_end;
    CycleCount overlap_duration;
};

// Bottleneck detection result
struct Bottleneck {
    ComponentType resource_type;
    uint32_t resource_id;
    CycleCount stall_duration;
    CycleCount busy_duration;
    double contention_ratio;  // stall / (stall + busy)
    std::string description;
};

// Per-wave statistics for wave-based execution
struct WaveStats {
    size_t wave_id;
    CycleCount start_cycle;
    CycleCount end_cycle;
    CycleCount duration;

    size_t operations_started;
    size_t operations_completed;

    size_t max_concurrent;
    double avg_concurrent;

    double compute_utilization;
    double memory_utilization;
};

// Comprehensive concurrency analysis report
struct ConcurrencyReport {
    // Time range
    CycleCount start_cycle;
    CycleCount end_cycle;
    CycleCount total_duration;

    // Operation counts
    size_t total_operations;
    size_t completed_operations;
    size_t failed_operations;

    // Concurrency metrics
    size_t peak_concurrency;
    double avg_concurrency;
    CycleCount cycles_at_peak;

    // Resource utilization
    std::map<ComponentType, double> utilization_by_type;
    std::map<ComponentType, size_t> operation_count_by_type;

    // Bottlenecks
    std::vector<Bottleneck> bottlenecks;

    // Wave statistics (if applicable)
    std::vector<WaveStats> wave_stats;

    // Overlap analysis
    size_t total_overlaps;
    CycleCount total_overlap_cycles;
    double overlap_efficiency;  // overlap_cycles / (sum of all op durations)
};

// Main analyzer class
class ConcurrencyAnalyzer {
public:
    ConcurrencyAnalyzer() = default;

    // Set data sources
    void set_traces(const std::vector<TraceEntry>& traces) {
        traces_ = traces;
    }

    void set_resource_tracker(const ResourceTracker& tracker) {
        resource_stats_ = tracker.get_aggregate_stats();
        resource_tracks_ = tracker.get_all_tracks();
    }

    // Analyze operation overlaps using sweep-line algorithm (O(n log n) instead of O(n²))
    std::vector<OperationOverlap> analyze_overlaps(size_t max_overlaps = 10000) const {
        std::vector<OperationOverlap> overlaps;

        // Build list of completed operations with time ranges
        struct OpRange {
            uint64_t id;
            ComponentType type;
            uint32_t component_id;
            CycleCount start;
            CycleCount end;
        };
        std::vector<OpRange> ops;

        for (const auto& trace : traces_) {
            if (trace.status == TransactionStatus::COMPLETED && trace.cycle_complete > 0) {
                ops.push_back({
                    trace.transaction_id,
                    trace.component_type,
                    trace.component_id,
                    trace.cycle_issue,
                    trace.cycle_complete
                });
            }
        }

        if (ops.empty()) return overlaps;

        // Sort by start time for efficient sweep-line processing
        std::sort(ops.begin(), ops.end(), [](const OpRange& a, const OpRange& b) {
            return a.start < b.start;
        });

        // Sweep-line: maintain active operations (operations that haven't ended yet)
        // Only check overlaps with active operations - this is O(n * k) where k is avg concurrency
        std::vector<size_t> active_indices;
        active_indices.reserve(64);  // Typical max concurrency

        for (size_t i = 0; i < ops.size() && overlaps.size() < max_overlaps; ++i) {
            const auto& current = ops[i];

            // Remove operations that have ended before current starts
            active_indices.erase(
                std::remove_if(active_indices.begin(), active_indices.end(),
                    [&](size_t idx) { return ops[idx].end <= current.start; }),
                active_indices.end());

            // Check overlaps with all still-active operations
            for (size_t active_idx : active_indices) {
                const auto& active = ops[active_idx];
                // active.start <= current.start (due to sorting)
                // active.end > current.start (still active)
                // So overlap exists from current.start to min(active.end, current.end)
                CycleCount overlap_end = std::min(active.end, current.end);
                if (current.start < overlap_end) {
                    overlaps.push_back({
                        active.id, current.id,
                        active.type, current.type,
                        current.start, overlap_end,
                        overlap_end - current.start
                    });
                    if (overlaps.size() >= max_overlaps) break;
                }
            }

            // Add current operation to active set
            active_indices.push_back(i);
        }

        return overlaps;
    }

    // Detect bottlenecks (resources with high contention)
    std::vector<Bottleneck> detect_bottlenecks(double contention_threshold = 0.2) const {
        std::vector<Bottleneck> bottlenecks;

        for (const auto& [res_id, track] : resource_tracks_) {
            CycleCount busy = track.get_state_duration(ResourceState::BUSY);
            CycleCount stalled = track.get_state_duration(ResourceState::STALLED);

            if (busy + stalled > 0) {
                double contention = static_cast<double>(stalled) / static_cast<double>(busy + stalled);
                if (contention >= contention_threshold) {
                    Bottleneck bn;
                    bn.resource_type = res_id.type;
                    bn.resource_id = res_id.id;
                    bn.stall_duration = stalled;
                    bn.busy_duration = busy;
                    bn.contention_ratio = contention;

                    std::ostringstream desc;
                    desc << to_string(res_id.type) << "[" << res_id.id << "] "
                         << "stalled " << std::fixed << std::setprecision(1)
                         << (contention * 100) << "% of active time";
                    bn.description = desc.str();

                    bottlenecks.push_back(bn);
                }
            }
        }

        // Sort by contention ratio (highest first)
        std::sort(bottlenecks.begin(), bottlenecks.end(),
            [](const Bottleneck& a, const Bottleneck& b) {
                return a.contention_ratio > b.contention_ratio;
            });

        return bottlenecks;
    }

    // Analyze wave-based execution patterns
    std::vector<WaveStats> analyze_waves(const std::vector<std::pair<CycleCount, CycleCount>>& wave_ranges) const {
        std::vector<WaveStats> stats;

        for (size_t w = 0; w < wave_ranges.size(); ++w) {
            WaveStats ws;
            ws.wave_id = w;
            ws.start_cycle = wave_ranges[w].first;
            ws.end_cycle = wave_ranges[w].second;
            ws.duration = ws.end_cycle - ws.start_cycle;

            // Count operations in this wave
            ws.operations_started = 0;
            ws.operations_completed = 0;

            std::vector<std::pair<CycleCount, int>> events;  // (cycle, +1 start / -1 end)

            for (const auto& trace : traces_) {
                if (trace.cycle_issue >= ws.start_cycle && trace.cycle_issue < ws.end_cycle) {
                    ws.operations_started++;
                    events.emplace_back(trace.cycle_issue, +1);
                }
                if (trace.cycle_complete > 0 &&
                    trace.cycle_complete >= ws.start_cycle &&
                    trace.cycle_complete <= ws.end_cycle) {
                    ws.operations_completed++;
                    events.emplace_back(trace.cycle_complete, -1);
                }
            }

            // Calculate concurrency
            std::sort(events.begin(), events.end());
            int current = 0;
            ws.max_concurrent = 0;
            CycleCount weighted_sum = 0;
            CycleCount last_cycle = ws.start_cycle;

            for (const auto& [cycle, delta] : events) {
                if (cycle > last_cycle) {
                    weighted_sum += current * (cycle - last_cycle);
                }
                current += delta;
                ws.max_concurrent = std::max(ws.max_concurrent, static_cast<size_t>(current));
                last_cycle = cycle;
            }

            ws.avg_concurrent = ws.duration > 0 ?
                static_cast<double>(weighted_sum) / static_cast<double>(ws.duration) : 0.0;

            // Calculate utilization for this wave
            CycleCount compute_busy = 0;
            CycleCount memory_busy = 0;

            for (const auto& [res_id, track] : resource_tracks_) {
                for (const auto& [start, end] : track.operation_intervals) {
                    CycleCount overlap_start = std::max(start, ws.start_cycle);
                    CycleCount overlap_end = std::min(end, ws.end_cycle);
                    if (overlap_start < overlap_end) {
                        CycleCount overlap = overlap_end - overlap_start;
                        if (res_id.type == ComponentType::SYSTOLIC_ARRAY ||
                            res_id.type == ComponentType::COMPUTE_FABRIC) {
                            compute_busy += overlap;
                        } else if (res_id.type == ComponentType::DMA_ENGINE ||
                                   res_id.type == ComponentType::BLOCK_MOVER ||
                                   res_id.type == ComponentType::STREAMER) {
                            memory_busy += overlap;
                        }
                    }
                }
            }

            // Count resources of each type
            size_t compute_resources = 0;
            size_t memory_resources = 0;
            for (const auto& [res_id, _] : resource_tracks_) {
                if (res_id.type == ComponentType::SYSTOLIC_ARRAY ||
                    res_id.type == ComponentType::COMPUTE_FABRIC) {
                    compute_resources++;
                } else if (res_id.type == ComponentType::DMA_ENGINE ||
                           res_id.type == ComponentType::BLOCK_MOVER ||
                           res_id.type == ComponentType::STREAMER) {
                    memory_resources++;
                }
            }

            CycleCount max_compute = compute_resources * ws.duration;
            CycleCount max_memory = memory_resources * ws.duration;

            ws.compute_utilization = max_compute > 0 ?
                static_cast<double>(compute_busy) / static_cast<double>(max_compute) : 0.0;
            ws.memory_utilization = max_memory > 0 ?
                static_cast<double>(memory_busy) / static_cast<double>(max_memory) : 0.0;

            stats.push_back(ws);
        }

        return stats;
    }

    // Generate comprehensive report
    ConcurrencyReport generate_report() const {
        ConcurrencyReport report;

        if (traces_.empty()) return report;

        // Time range
        report.start_cycle = traces_.front().cycle_issue;
        report.end_cycle = 0;
        for (const auto& t : traces_) {
            report.start_cycle = std::min(report.start_cycle, t.cycle_issue);
            report.end_cycle = std::max(report.end_cycle,
                t.cycle_complete > 0 ? t.cycle_complete : t.cycle_issue);
        }
        report.total_duration = report.end_cycle - report.start_cycle;

        // Operation counts
        report.total_operations = traces_.size();
        report.completed_operations = 0;
        report.failed_operations = 0;
        for (const auto& t : traces_) {
            if (t.status == TransactionStatus::COMPLETED) report.completed_operations++;
            if (t.status == TransactionStatus::FAILED) report.failed_operations++;
        }

        // Concurrency metrics
        std::vector<std::pair<CycleCount, int>> events;
        for (const auto& t : traces_) {
            events.emplace_back(t.cycle_issue, +1);
            if (t.cycle_complete > 0) {
                events.emplace_back(t.cycle_complete, -1);
            }
        }
        std::sort(events.begin(), events.end());

        int current = 0;
        report.peak_concurrency = 0;
        report.cycles_at_peak = 0;
        CycleCount weighted_sum = 0;
        CycleCount last_cycle = report.start_cycle;

        for (const auto& [cycle, delta] : events) {
            if (cycle > last_cycle) {
                weighted_sum += current * (cycle - last_cycle);
                if (static_cast<size_t>(current) == report.peak_concurrency) {
                    report.cycles_at_peak += (cycle - last_cycle);
                }
            }
            current += delta;
            if (static_cast<size_t>(current) > report.peak_concurrency) {
                report.peak_concurrency = current;
                report.cycles_at_peak = 0;
            }
            last_cycle = cycle;
        }

        report.avg_concurrency = report.total_duration > 0 ?
            static_cast<double>(weighted_sum) / static_cast<double>(report.total_duration) : 0.0;

        // Utilization by type
        for (const auto& [type, util] : resource_stats_.avg_utilization) {
            report.utilization_by_type[type] = util;
        }

        // Operation count by type
        for (const auto& t : traces_) {
            report.operation_count_by_type[t.component_type]++;
        }

        // Bottlenecks
        report.bottlenecks = detect_bottlenecks();

        // Overlap analysis
        auto overlaps = analyze_overlaps();
        report.total_overlaps = overlaps.size();
        report.total_overlap_cycles = 0;
        for (const auto& o : overlaps) {
            report.total_overlap_cycles += o.overlap_duration;
        }

        CycleCount total_op_duration = 0;
        for (const auto& t : traces_) {
            total_op_duration += t.get_duration_cycles();
        }
        report.overlap_efficiency = total_op_duration > 0 ?
            static_cast<double>(report.total_overlap_cycles) / static_cast<double>(total_op_duration) : 0.0;

        return report;
    }

    // Format report as text
    static std::string format_report(const ConcurrencyReport& report) {
        std::ostringstream oss;

        oss << "\n";
        oss << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        oss << "║                      CONCURRENCY ANALYSIS REPORT                             ║\n";
        oss << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

        oss << "\n┌─ Time Range ─────────────────────────────────────────────────────────────────┐\n";
        oss << "│ Start Cycle:    " << std::setw(12) << report.start_cycle << "\n";
        oss << "│ End Cycle:      " << std::setw(12) << report.end_cycle << "\n";
        oss << "│ Total Duration: " << std::setw(12) << report.total_duration << " cycles\n";
        oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";

        oss << "\n┌─ Operation Counts ───────────────────────────────────────────────────────────┐\n";
        oss << "│ Total:     " << std::setw(8) << report.total_operations << "\n";
        oss << "│ Completed: " << std::setw(8) << report.completed_operations << "\n";
        oss << "│ Failed:    " << std::setw(8) << report.failed_operations << "\n";
        oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";

        oss << "\n┌─ Concurrency Metrics ────────────────────────────────────────────────────────┐\n";
        oss << "│ Peak Concurrency:    " << std::setw(8) << report.peak_concurrency << " operations\n";
        oss << "│ Average Concurrency: " << std::setw(8) << std::fixed << std::setprecision(2)
            << report.avg_concurrency << " operations\n";
        oss << "│ Cycles at Peak:      " << std::setw(8) << report.cycles_at_peak << " cycles\n";
        oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";

        if (!report.utilization_by_type.empty()) {
            oss << "\n┌─ Resource Utilization ───────────────────────────────────────────────────────┐\n";
            for (const auto& [type, util] : report.utilization_by_type) {
                oss << "│ " << std::setw(20) << std::left << to_string(type)
                    << std::right << std::setw(6) << std::fixed << std::setprecision(1)
                    << (util * 100) << "% utilization\n";
            }
            oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";
        }

        if (!report.bottlenecks.empty()) {
            oss << "\n┌─ Bottlenecks (sorted by severity) ──────────────────────────────────────────┐\n";
            for (const auto& bn : report.bottlenecks) {
                oss << "│ " << bn.description << "\n";
            }
            oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";
        }

        if (!report.wave_stats.empty()) {
            oss << "\n┌─ Wave Statistics ────────────────────────────────────────────────────────────┐\n";
            oss << "│ Wave   Duration   Max Conc.   Avg Conc.   Compute%   Memory%\n";
            oss << "│ ────   ────────   ─────────   ─────────   ────────   ───────\n";
            for (const auto& ws : report.wave_stats) {
                oss << "│ " << std::setw(4) << ws.wave_id
                    << "   " << std::setw(8) << ws.duration
                    << "   " << std::setw(9) << ws.max_concurrent
                    << "   " << std::setw(9) << std::fixed << std::setprecision(1) << ws.avg_concurrent
                    << "   " << std::setw(7) << (ws.compute_utilization * 100) << "%"
                    << "   " << std::setw(6) << (ws.memory_utilization * 100) << "%\n";
            }
            oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";
        }

        oss << "\n┌─ Overlap Analysis ───────────────────────────────────────────────────────────┐\n";
        oss << "│ Total Overlapping Pairs: " << std::setw(8) << report.total_overlaps << "\n";
        oss << "│ Total Overlap Cycles:    " << std::setw(8) << report.total_overlap_cycles << "\n";
        oss << "│ Overlap Efficiency:      " << std::setw(7) << std::fixed << std::setprecision(1)
            << (report.overlap_efficiency * 100) << "%\n";
        oss << "└──────────────────────────────────────────────────────────────────────────────┘\n";

        return oss.str();
    }

    // Export to CSV for external analysis
    bool export_timeline_csv(const std::string& filename, CycleCount sample_interval = 100) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;

        // Header
        file << "Cycle,ActiveOperations";
        std::set<ComponentType> types;
        for (const auto& [res_id, _] : resource_tracks_) {
            types.insert(res_id.type);
        }
        for (auto type : types) {
            file << "," << to_string(type) << "_Busy";
        }
        file << "\n";

        // Build event list
        std::vector<std::pair<CycleCount, int>> events;
        for (const auto& t : traces_) {
            events.emplace_back(t.cycle_issue, +1);
            if (t.cycle_complete > 0) {
                events.emplace_back(t.cycle_complete, -1);
            }
        }
        std::sort(events.begin(), events.end());

        CycleCount start = traces_.empty() ? 0 : traces_.front().cycle_issue;
        CycleCount end = 0;
        for (const auto& t : traces_) {
            end = std::max(end, t.cycle_complete > 0 ? t.cycle_complete : t.cycle_issue);
        }

        for (CycleCount cycle = start; cycle <= end; cycle += sample_interval) {
            // Count active operations at this cycle
            int active = 0;
            for (const auto& t : traces_) {
                if (t.cycle_issue <= cycle &&
                    (t.cycle_complete == 0 || t.cycle_complete > cycle)) {
                    active++;
                }
            }

            file << cycle << "," << active;

            // Count busy resources of each type
            for (auto type : types) {
                int busy = 0;
                for (const auto& [res_id, track] : resource_tracks_) {
                    if (res_id.type == type) {
                        for (const auto& [op_start, op_end] : track.operation_intervals) {
                            if (op_start <= cycle && op_end > cycle) {
                                busy++;
                                break;
                            }
                        }
                    }
                }
                file << "," << busy;
            }
            file << "\n";
        }

        file.close();
        return true;
    }

private:
    std::vector<TraceEntry> traces_;
    ResourceTracker::AggregateStats resource_stats_;
    std::map<ResourceId, ResourceTrack> resource_tracks_;
};

} // namespace sw::trace
