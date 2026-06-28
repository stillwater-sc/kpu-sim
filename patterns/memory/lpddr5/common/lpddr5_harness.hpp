// patterns/memory/lpddr5/common/lpddr5_harness.hpp
//
// LPDDR5-specific test harness for pattern validation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>

#include <filesystem>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/trace_exporter.hpp>
#include "lpddr5_configs.hpp"

namespace fs = std::filesystem;

namespace sw::kpu::patterns::lpddr5 {

using namespace sw::kpu::lpddr5;

// ============================================================================
// Trace Directory Management
// ============================================================================

/// Get project root by finding CMakeLists.txt
inline fs::path find_project_root() {
    fs::path current = fs::current_path();
    while (!current.empty() && current != current.root_path()) {
        if (fs::exists(current / "CMakeLists.txt") &&
            fs::exists(current / "patterns")) {
            return current;
        }
        current = current.parent_path();
    }
    // Fallback to current directory
    return fs::current_path();
}

/// Get trace directory for a specific pattern category
/// Creates the directory if it doesn't exist
inline fs::path get_trace_dir(const std::string& category) {
    fs::path root = find_project_root();
    fs::path trace_dir = root / "traces" / "memory" / "lpddr5" / category;
    fs::create_directories(trace_dir);
    return trace_dir;
}

/// Build full trace path
inline std::string make_trace_path(const std::string& category,
                                    const std::string& filename) {
    return (get_trace_dir(category) / filename).string();
}

/// LPDDR5-specific test harness for memory controller patterns
///
/// Provides:
/// - Memory controller setup with tracing
/// - Request submission helpers
/// - Simulation execution
/// - Statistics reporting
/// - Trace export
class LPDDR5Harness {
public:
    explicit LPDDR5Harness(const LPDDR5MemoryController::Config& config)
        : config_(config)
        , mc_(std::make_unique<LPDDR5MemoryController>(config))
        , tracker_(std::make_unique<sw::trace::ResourceTracker>())
    {
        mc_->enable_tracing(true);
        mc_->set_resource_tracker(tracker_.get());
    }

    virtual ~LPDDR5Harness() = default;

    // ========================================================================
    // Simulation Control
    // ========================================================================

    /// Advance simulation by one cycle
    void tick() {
        mc_->tick();
    }

    /// Run until all pending requests complete or timeout
    bool run_until_complete(uint64_t max_cycles = 10000) {
        uint64_t start = mc_->current_cycle();
        while (mc_->has_pending() && (mc_->current_cycle() - start) < max_cycles) {
            mc_->tick();
            if (mc_->has_violations()) {
                return false;
            }
        }
        return !mc_->has_pending();
    }

    /// Run for a specific number of cycles
    void run_cycles(uint64_t cycles) {
        for (uint64_t i = 0; i < cycles; ++i) {
            mc_->tick();
        }
    }

    // ========================================================================
    // Request Submission
    // ========================================================================

    /// Submit a read request
    std::optional<uint64_t> submit_read(uint64_t address, uint32_t size = CACHE_LINE_BYTES) {
        return mc_->submit_read(address, size);
    }

    /// Submit a read request with callback
    std::optional<uint64_t> submit_read(uint64_t address, uint32_t size,
                                        std::function<void()> callback) {
        return mc_->submit_read(address, size, callback);
    }

    /// Submit a write request
    std::optional<uint64_t> submit_write(uint64_t address, const std::vector<uint8_t>& data) {
        return mc_->submit_write(address, data.data(), static_cast<uint32_t>(data.size()));
    }

    /// Submit a write request (size only, no actual data)
    std::optional<uint64_t> submit_write(uint64_t address, uint32_t size = CACHE_LINE_BYTES) {
        static std::vector<uint8_t> dummy_data(256, 0xAB);
        if (size > dummy_data.size()) {
            dummy_data.resize(size, 0xAB);
        }
        return mc_->submit_write(address, dummy_data.data(), size);
    }

    // ========================================================================
    // State Queries
    // ========================================================================

    uint64_t current_cycle() const { return mc_->current_cycle(); }
    bool has_pending() const { return mc_->has_pending(); }
    bool has_violations() const { return mc_->has_violations(); }

    /// Get statistics
    const Statistics& stats() const { return mc_->lpddr5_stats(); }

    /// Get violations
    const std::vector<InvariantViolation>& violations() const {
        return mc_->lpddr5_violations();
    }

    // ========================================================================
    // Reporting
    // ========================================================================

    /// Print statistics summary
    void print_stats(std::ostream& os = std::cout) const {
        const auto& s = mc_->lpddr5_stats();
        os << "\n=== Memory Controller Statistics ===" << std::endl;
        os << "  Reads:              " << s.reads << std::endl;
        os << "  Writes:             " << s.writes << std::endl;
        os << "  Page hits:          " << s.page_hits << std::endl;
        os << "  Page empty:         " << s.page_empty << std::endl;
        os << "  Page conflicts:     " << s.page_conflicts << std::endl;
        os << "  Avg latency:        " << std::fixed << std::setprecision(2)
           << s.avg_latency() << " cycles" << std::endl;
        os << "  Stall cycles:       " << s.stall_cycles << std::endl;
        os << "  R->W turnarounds:   " << s.read_to_write_turnarounds << std::endl;
        os << "  W->R turnarounds:   " << s.write_to_read_turnarounds << std::endl;
        os << "  Total cycles:       " << mc_->current_cycle() << std::endl;
        os << "====================================\n" << std::endl;
    }

    /// Print violations if any
    void print_violations(std::ostream& os = std::cerr) const {
        if (mc_->has_violations()) {
            os << "\n=== INVARIANT VIOLATIONS ===" << std::endl;
            for (const auto& v : mc_->lpddr5_violations()) {
                os << "  Cycle " << v.cycle
                   << ": [" << v.invariant_id << "] "
                   << v.message
                   << " (channel=" << (int)v.channel
                   << ", bank=" << (int)v.bank << ")"
                   << std::endl;
            }
            os << "============================\n" << std::endl;
        }
    }

    // ========================================================================
    // Trace Export
    // ========================================================================

    /// Export trace to Chrome Trace JSON format
    bool export_trace(const std::string& filename, double clock_ghz = 3.2) {
        // Finalize resource tracker
        tracker_->finalize(mc_->current_cycle());

        // Export memory controller trace entries
        bool success = sw::trace::ChromeTraceExporter::export_traces(
            filename,
            mc_->trace_entries(),
            clock_ghz
        );

        if (success) {
            std::cout << "Trace exported to: " << filename << std::endl;
            std::cout << "  Events: " << mc_->trace_entries().size() << std::endl;
            std::cout << "  Open with: https://ui.perfetto.dev" << std::endl;
        }

        return success;
    }

    /// Export resource utilization to separate file
    bool export_resources(const std::string& filename, double clock_ghz = 3.2) {
        tracker_->finalize(mc_->current_cycle());
        auto tracks = tracker_->get_all_tracks();
        return sw::trace::ResourceTrackerExporter::export_to_chrome_trace(
            filename, tracks, clock_ghz
        );
    }

    // ========================================================================
    // Verification Helpers
    // ========================================================================

    /// Verify no violations occurred
    bool verify_no_violations() const {
        if (mc_->has_violations()) {
            print_violations();
            return false;
        }
        return true;
    }

    /// Verify expected statistics
    bool verify_stats(uint64_t expected_reads, uint64_t expected_writes,
                      uint64_t expected_page_hits, uint64_t expected_page_empty,
                      uint64_t expected_page_conflicts) const {
        const auto& s = mc_->lpddr5_stats();
        bool pass = true;

        if (s.reads != expected_reads) {
            std::cerr << "FAIL: Expected " << expected_reads << " reads, got " << s.reads << std::endl;
            pass = false;
        }
        if (s.writes != expected_writes) {
            std::cerr << "FAIL: Expected " << expected_writes << " writes, got " << s.writes << std::endl;
            pass = false;
        }
        if (s.page_hits != expected_page_hits) {
            std::cerr << "FAIL: Expected " << expected_page_hits << " page hits, got " << s.page_hits << std::endl;
            pass = false;
        }
        if (s.page_empty != expected_page_empty) {
            std::cerr << "FAIL: Expected " << expected_page_empty << " page empty, got " << s.page_empty << std::endl;
            pass = false;
        }
        if (s.page_conflicts != expected_page_conflicts) {
            std::cerr << "FAIL: Expected " << expected_page_conflicts << " page conflicts, got " << s.page_conflicts << std::endl;
            pass = false;
        }

        return pass;
    }

    /// Access underlying memory controller
    LPDDR5MemoryController& memory_controller() { return *mc_; }
    const LPDDR5MemoryController& memory_controller() const { return *mc_; }

protected:
    LPDDR5MemoryController::Config config_;
    std::unique_ptr<LPDDR5MemoryController> mc_;
    std::unique_ptr<sw::trace::ResourceTracker> tracker_;
};

} // namespace sw::kpu::patterns::lpddr5
