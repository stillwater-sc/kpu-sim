// ============================================================================
// include/sw/kpu/harness/pattern_harness_base.hpp
// Abstract base class for KPU component test harnesses
//
// This provides the common interface and functionality for all harnesses:
// - Simulation control (run, step, reset)
// - Statistics collection
// - Tracing support
// - Validation utilities
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/harness_config.hpp>
#include <sw/kpu/stats/stats_collector.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/resource_tracker.hpp>

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>

namespace sw::kpu::harness {

// ============================================================================
// Validation Result
// ============================================================================

/// Result of validation operations
struct ValidationResult {
    bool passed = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    /// Add an error
    void add_error(const std::string& msg) {
        passed = false;
        errors.push_back(msg);
    }

    /// Add a warning
    void add_warning(const std::string& msg) {
        warnings.push_back(msg);
    }

    /// Check if validation passed
    explicit operator bool() const { return passed; }

    /// Get summary string
    std::string summary() const {
        std::string result;
        if (passed) {
            result = "PASSED";
        } else {
            result = "FAILED";
        }
        if (!errors.empty()) {
            result += " (" + std::to_string(errors.size()) + " errors)";
        }
        if (!warnings.empty()) {
            result += " (" + std::to_string(warnings.size()) + " warnings)";
        }
        return result;
    }

    /// Print detailed results
    void print(std::ostream& os = std::cerr) const {
        os << "Validation " << summary() << "\n";
        for (const auto& err : errors) {
            os << "  ERROR: " << err << "\n";
        }
        for (const auto& warn : warnings) {
            os << "  WARNING: " << warn << "\n";
        }
    }
};

// ============================================================================
// Pattern Harness Base
// ============================================================================

/// Abstract base class for all component test harnesses
///
/// Provides common functionality for:
/// - Simulation lifecycle (run, step, reset)
/// - Statistics collection and reporting
/// - Event tracing and export
/// - Validation infrastructure
///
/// @tparam ConfigT Configuration type (must derive from HarnessConfig)
template<typename ConfigT>
class PatternHarnessBase {
    static_assert(std::is_base_of_v<HarnessConfig, ConfigT>,
                  "ConfigT must derive from HarnessConfig");

public:
    using Config = ConfigT;
    using Cycle = uint64_t;

    // ========================================================================
    // Construction
    // ========================================================================

    explicit PatternHarnessBase(const ConfigT& config)
        : config_(config)
        , current_cycle_(0)
        , stats_collector_(make_stats_config(config))
        , resource_tracker_(std::make_unique<sw::trace::ResourceTracker>())
    {
        if (config.enable_tracing) {
            sw::trace::TraceLogger::instance().set_enabled(true);
        }
    }

    virtual ~PatternHarnessBase() = default;

    // Non-copyable
    PatternHarnessBase(const PatternHarnessBase&) = delete;
    PatternHarnessBase& operator=(const PatternHarnessBase&) = delete;

    // Movable
    PatternHarnessBase(PatternHarnessBase&&) = default;
    PatternHarnessBase& operator=(PatternHarnessBase&&) = default;

    // ========================================================================
    // Simulation Control (Virtual Interface)
    // ========================================================================

    /// Run simulation until completion or timeout
    /// @return true if simulation completed successfully
    virtual bool run() = 0;

    /// Advance simulation by one cycle
    virtual void tick() {
        ++current_cycle_;
    }

    /// Advance simulation by specified number of cycles
    virtual void step(Cycle cycles) {
        for (Cycle i = 0; i < cycles; ++i) {
            tick();
        }
    }

    /// Reset harness to initial state
    virtual void reset() {
        current_cycle_ = 0;
        stats_collector_.reset();
        resource_tracker_ = std::make_unique<sw::trace::ResourceTracker>();
        sw::trace::TraceLogger::instance().clear();
    }

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Get current simulation cycle
    Cycle current_cycle() const { return current_cycle_; }

    /// Get configuration
    const ConfigT& config() const { return config_; }

    /// Check if simulation has work pending
    virtual bool has_pending() const = 0;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get statistics collector
    const stats::StatsCollector& stats() const { return stats_collector_; }
    stats::StatsCollector& stats() { return stats_collector_; }

    /// Generate statistics summary
    stats::StatsSummary summarize_stats() const {
        return stats_collector_.summarize();
    }

    /// Print statistics to stream
    virtual void print_stats(std::ostream& os = std::cout) const {
        os << stats_collector_.summary();
    }

    /// Export statistics to file
    bool export_stats(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) return false;
        ofs << stats_collector_.to_json();
        return true;
    }

    // ========================================================================
    // Tracing
    // ========================================================================

    /// Get resource tracker
    sw::trace::ResourceTracker& resource_tracker() { return *resource_tracker_; }
    const sw::trace::ResourceTracker& resource_tracker() const { return *resource_tracker_; }

    /// Enable or disable tracing
    void enable_tracing(bool enable) {
        sw::trace::TraceLogger::instance().set_enabled(enable);
    }

    /// Check if tracing is enabled
    bool tracing_enabled() const {
        return sw::trace::TraceLogger::instance().is_enabled();
    }

    /// Export trace to Chrome trace format
    bool export_trace(const std::string& path) const {
        return export_trace_impl(path, config_.clock_frequency_ghz);
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /// Validate simulation results
    /// Override in derived classes to add specific validation
    virtual ValidationResult validate() const {
        ValidationResult result;
        // Base class does minimal validation
        if (current_cycle_ > config_.max_cycles) {
            result.add_warning("Simulation exceeded max_cycles limit");
        }
        return result;
    }

protected:
    // ========================================================================
    // Protected State
    // ========================================================================

    ConfigT config_;
    Cycle current_cycle_;
    stats::StatsCollector stats_collector_;
    std::unique_ptr<sw::trace::ResourceTracker> resource_tracker_;

    // ========================================================================
    // Protected Helpers
    // ========================================================================

    /// Create stats config from harness config
    static stats::StatsConfig make_stats_config(const HarnessConfig& config) {
        stats::StatsConfig sc;
        sc.clock_frequency_ghz = config.clock_frequency_ghz;
        sc.enable_detailed_tracking = config.enable_stats;
        return sc;
    }

    /// Export trace implementation
    bool export_trace_impl(const std::string& path, double clock_ghz) const;

    /// Record component state transition
    void record_component_transition(
        HarnessComponent comp, uint32_t id,
        sw::trace::ResourceState state,
        Cycle cycle,
        const std::string& reason = "");

    /// Check if cycle limit exceeded
    bool check_cycle_limit() const {
        return current_cycle_ < config_.max_cycles;
    }
};

// ============================================================================
// Template Implementation
// ============================================================================

template<typename ConfigT>
bool PatternHarnessBase<ConfigT>::export_trace_impl(
    const std::string& path, double clock_ghz) const
{
    // Finalize resource tracker
    const_cast<sw::trace::ResourceTracker*>(resource_tracker_.get())->finalize(current_cycle_);

    // Get all traces
    auto traces = sw::trace::TraceLogger::instance().get_all_traces();
    if (traces.empty()) {
        return false;
    }

    // Export using trace exporter
    std::ofstream ofs(path);
    if (!ofs) return false;

    // Write Chrome trace format
    ofs << "{\n\"traceEvents\": [\n";

    bool first = true;
    for (const auto& entry : traces) {
        if (!first) ofs << ",\n";
        first = false;

        // Calculate timestamps in microseconds
        double start_us = (entry.cycle_issue / clock_ghz) / 1000.0;
        double duration_us = 0.0;
        if (entry.cycle_complete > entry.cycle_issue) {
            duration_us = ((entry.cycle_complete - entry.cycle_issue) / clock_ghz) / 1000.0;
        }

        ofs << "  {\"name\": \"" << entry.description << "\", "
            << "\"cat\": \"" << sw::trace::to_string(entry.component_type) << "\", "
            << "\"ph\": \"X\", "
            << "\"ts\": " << start_us << ", "
            << "\"dur\": " << duration_us << ", "
            << "\"pid\": 1, "
            << "\"tid\": " << static_cast<int>(entry.component_type) << "}";
    }

    ofs << "\n],\n\"displayTimeUnit\": \"ns\"\n}\n";
    return true;
}

template<typename ConfigT>
void PatternHarnessBase<ConfigT>::record_component_transition(
    HarnessComponent comp, uint32_t id,
    sw::trace::ResourceState state,
    Cycle cycle,
    const std::string& reason)
{
    // Map harness component to trace component type
    sw::trace::ComponentType trace_comp;
    switch (comp) {
        case HarnessComponent::DMA_ENGINE:        trace_comp = sw::trace::ComponentType::DMA_ENGINE; break;
        case HarnessComponent::BLOCK_MOVER:       trace_comp = sw::trace::ComponentType::BLOCK_MOVER; break;
        case HarnessComponent::STREAMER:          trace_comp = sw::trace::ComponentType::STREAMER; break;
        case HarnessComponent::COMPUTE_FABRIC:    trace_comp = sw::trace::ComponentType::COMPUTE_FABRIC; break;
        case HarnessComponent::MEMORY_CONTROLLER: trace_comp = sw::trace::ComponentType::LPDDR5_CHANNEL; break;
        case HarnessComponent::L3_BUFFER:         trace_comp = sw::trace::ComponentType::L3_TILE; break;
        case HarnessComponent::L2_BANK:           trace_comp = sw::trace::ComponentType::L2_BANK; break;
        case HarnessComponent::L1_BUFFER:         trace_comp = sw::trace::ComponentType::L1; break;
        default:                                  trace_comp = sw::trace::ComponentType::UNKNOWN; break;
    }

    resource_tracker_->transition(trace_comp, id, state, cycle, 0, reason);
}

} // namespace sw::kpu::harness
