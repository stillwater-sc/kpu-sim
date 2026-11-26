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
#include <optional>
#include <unordered_map>
#include <sw/kpu/models/temporal/memory/controllers/lpddr5_controller.hpp>
#include <sw/kpu/program/record/memory_flow_record.hpp>
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
/// - A .mflow record of the run (docs/plans/cycle-mflow-and-debug-server.md step 10): every
///   request, its burst and the commands that served it, from the controller's command observer
class LPDDR5Harness {
public:
    using MFR = sw::kpu::program::record::MemoryFlowRecord;

    explicit LPDDR5Harness(const LPDDR5MemoryController::Config& config)
        : config_(config)
        , mc_(std::make_unique<LPDDR5MemoryController>(config))
        , tracker_(std::make_unique<sw::trace::ResourceTracker>())
    {
        mc_->enable_tracing(true);
        mc_->set_resource_tracker(tracker_.get());
        mc_->set_command_observer([this](const LPDDR5MemoryController::CommandRecord& r) { observe(r); });
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
        return submit_read(address, size, nullptr);
    }

    /// Submit a read request with callback
    std::optional<uint64_t> submit_read(uint64_t address, uint32_t size,
                                        std::function<void()> callback) {
        return recorded(true, address, size, std::move(callback), [&](std::function<void()> cb) {
            return mc_->submit_read(address, size, std::move(cb));
        });
    }

    /// Submit a write request
    std::optional<uint64_t> submit_write(uint64_t address, const std::vector<uint8_t>& data) {
        const auto size = static_cast<uint32_t>(data.size());
        return recorded(false, address, size, nullptr, [&](std::function<void()> cb) {
            return mc_->submit_write(address, data.data(), size, std::move(cb));
        });
    }

    /// Submit a write request (size only, no actual data)
    std::optional<uint64_t> submit_write(uint64_t address, uint32_t size = CACHE_LINE_BYTES) {
        static std::vector<uint8_t> dummy_data(256, 0xAB);
        if (size > dummy_data.size()) {
            dummy_data.resize(size, 0xAB);
        }
        return recorded(false, address, size, nullptr, [&](std::function<void()> cb) {
            return mc_->submit_write(address, dummy_data.data(), size, std::move(cb));
        });
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
    // .mflow record (docs/plans/cycle-mflow-and-debug-server.md §3.5, step 10)
    // ========================================================================
    //
    // A pattern drives the controller directly: no DMA engine, no store buffer, no wait and no
    // port, so the record has none (engine kNone). Each request is one burst (the controller
    // times one CAS per request), and its commands name it by the controller's request id; a
    // PRECHARGE names the request that opened the page it closes. Cycles are the controller's
    // own clock ticks (ticks_per_cycle 1).

    /// The run so far, as a .mflow record of source "lpddr5-pattern".
    MFR record(const std::string& device = "lpddr5") const {
        MFR rec = rec_;
        rec.source = "lpddr5-pattern";
        rec.device = device;
        rec.makespan = mc_->current_cycle();
        rec.window = config_.queue_depth;        // no engine reads it: the controller's queue
        rec.burst_bytes = CACHE_LINE_BYTES;
        rec.arbitration = "fixed";               // no engines: nothing to arbitrate
        rec.ticks_per_cycle = 1.0;
        rec.timing_note = "LPDDR5 pattern: the controller's own timing table, driven directly (one CAS per "
                          "request); cycles are the controller's clock ticks";
        const TimingParams& t = config_.timing;
        const uint32_t burst = config_.burst_length == BurstLength::BL32 ? t.tBurst_BL32 : t.tBurst_BL16;
        rec.dram_timing = {{"tRCD", t.tRCD},     {"tRP", t.tRP},       {"tRAS", t.tRAS},     {"tRC", t.tRC},
                           {"tCL", t.tCL},       {"tWL", t.tWL},       {"tWR", t.tWR},       {"tRTP", t.tRTP},
                           {"tRRD_L", t.tRRD_L}, {"tRRD_S", t.tRRD_S}, {"tCCD_L", t.tCCD_L}, {"tCCD_S", t.tCCD_S},
                           {"tWTR_L", t.tWTR_L}, {"tWTR_S", t.tWTR_S}, {"tRTW", t.tRTW},     {"tBurst", burst},
                           {"tRFCpb", t.tRFCpb}, {"tRFCab", t.tRFCab}, {"tREFIpb", t.tREFIpb}, {"tFAW", t.tFAW}};
        // The model's own ceiling: each channel's data bus carries one request's CAS per tBurst.
        rec.ceiling_bytes_per_cycle = burst ? static_cast<double>(config_.num_channels) * CACHE_LINE_BYTES / burst : 0.0;
        const unsigned per_group = config_.banks_per_channel / config_.bank_groups;
        for (unsigned ch = 0; ch < config_.num_channels; ++ch)
            for (unsigned bg = 0; bg < config_.bank_groups; ++bg)
                for (unsigned ba = 0; ba < per_group; ++ba)
                    rec.stations.push_back({device + "/mc[0]/ch[" + std::to_string(ch) + "]/bg[" + std::to_string(bg) +
                                                "]/ba[" + std::to_string(ba) + "]", "dram_bank", 1});
        for (unsigned ch = 0; ch < config_.num_channels; ++ch)
            rec.stations.push_back({device + "/mc[0]/ch[" + std::to_string(ch) + "]/bus", "dram_bus", 1});
        return rec;
    }

    /// Write the run's record as a .mflow bundle at `dir` (a directory, written whole). A
    /// directory that is there already is replaced only when it is a .mflow bundle or empty.
    bool export_mflow(const std::string& dir, const std::string& device = "lpddr5") const {
        std::error_code ec;
        if (fs::exists(dir, ec)) {
            const bool bundle = fs::is_regular_file(fs::path(dir) / "manifest.json", ec);
            if (!fs::is_directory(dir, ec) || (!bundle && !fs::is_empty(dir, ec))) {
                std::cerr << "export_mflow: " << dir << " is there and is not a .mflow bundle; not replacing it\n";
                return false;
            }
            fs::remove_all(dir, ec);
        }
        try {
            sw::kpu::program::record::write_mflow(record(device), dir);
        } catch (const std::exception& e) {
            std::cerr << "export_mflow: " << e.what() << "\n";
            return false;
        }
        std::cout << "Record written to: " << dir << " (" << rec_.requests.size() << " requests, "
                  << rec_.commands.size() << " commands)" << std::endl;
        std::cout << "  Check it: python3 tools/trace/mflow_check.py " << dir << std::endl;
        return true;
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

private:
    MFR rec_;                                              // requests, bursts and commands so far
    std::unordered_map<uint64_t, uint32_t> burst_of_;      // controller request id -> burst index
    std::vector<bool> commanded_;                          // per burst: a command has served it

    /// Submit through `submit`, recording the request and its burst when the controller takes it.
    template <class Submit>
    std::optional<uint64_t> recorded(bool is_load, uint64_t address, uint32_t size,
                                      std::function<void()> callback, Submit submit) {
        const auto q = static_cast<uint32_t>(rec_.requests.size());
        const uint64_t now = mc_->current_cycle();
        auto id = submit([this, q, callback = std::move(callback)] {
            const uint64_t t = mc_->current_cycle();
            rec_.requests[q].retired = t;
            rec_.bursts[q].done = t;
            if (callback) callback();
        });
        if (!id) return id;
        MFR::Request r;
        r.engine = sw::kpu::program::record::kNone;
        r.port = sw::kpu::program::record::kNone;
        r.is_load = is_load;
        r.address = address;
        r.bytes = size;
        r.offered = r.credit = r.first_burst = r.last_burst = now;
        r.kind = is_load ? MFR::RequestKind::Load : MFR::RequestKind::Store;
        rec_.requests.push_back(r);
        MFR::Burst b;
        b.engine = sw::kpu::program::record::kNone;
        b.request = q;
        b.is_load = is_load;
        b.posted = b.submitted = now;
        rec_.bursts.push_back(b);
        commanded_.push_back(false);
        burst_of_[*id] = q;
        return id;
    }

    /// One command, at its issue: recorded, and its burst's coordinates, outcome and times.
    void observe(const LPDDR5MemoryController::CommandRecord& r) {
        using K = LPDDR5MemoryController::CommandRecord::Kind;
        const unsigned per_group = config_.banks_per_channel / config_.bank_groups;
        MFR::Command c;
        c.mc = 0;
        c.channel = r.channel;
        c.bank_group = static_cast<uint8_t>(r.bank / per_group);
        c.bank = static_cast<uint8_t>(r.bank % per_group);
        c.row = r.row;
        c.kind = r.kind == K::Activate ? MFR::CommandKind::Activate
               : r.kind == K::Read     ? MFR::CommandKind::Read
               : r.kind == K::Write    ? MFR::CommandKind::Write
               : r.kind == K::Precharge ? MFR::CommandKind::Precharge
                                        : MFR::CommandKind::Refresh;
        c.issue = c.tick = r.issue;
        c.end = c.tick_end = r.end;
        const bool cas = r.kind == K::Read || r.kind == K::Write;
        if (cas) {
            c.data_start = c.tick_data_start = r.data_start;
            c.data_end = c.tick_data_end = r.data_end;
        }
        auto it = r.request_id ? burst_of_.find(r.request_id) : burst_of_.end();
        if (it != burst_of_.end()) {
            c.burst = it->second;
            MFR::Burst& b = rec_.bursts[it->second];
            if (r.kind == K::Activate || cas) {
                // Its first command: the ACT that opens its row, or its CAS on a hit.
                if (!commanded_[it->second]) { b.first_command = r.issue; commanded_[it->second] = true; }
                b.channel = r.channel;
                b.bank_group = c.bank_group;
                b.bank = c.bank;
                b.row = r.row;
            }
            if (cas) {
                b.col = r.col;
                b.data_start = r.data_start;
                b.data_end = r.data_end;
                b.outcome = !r.activated ? MFR::Outcome::Hit : r.conflicted ? MFR::Outcome::Conflict : MFR::Outcome::Empty;
            }
        }
        rec_.commands.push_back(c);
    }
};

} // namespace sw::kpu::patterns::lpddr5
