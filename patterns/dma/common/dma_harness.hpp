// patterns/dma/common/dma_harness.hpp
//
// DMA test harness for pattern validation
// Integrates DMA engine with memory controller and NoC
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <functional>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/components/dma/dma_placement.hpp>
#include <sw/kpu/components/memory/memory_controller_interface.hpp>
#include <sw/kpu/noc/noc_interface.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/trace_exporter.hpp>
#include <sw/trace/trace_entry.hpp>

#include "dma_configs.hpp"
#include "matrix_layouts.hpp"

namespace fs = std::filesystem;

namespace sw::kpu::patterns::dma {

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
    return fs::current_path();
}

/// Get trace directory for a specific pattern category
inline fs::path get_trace_dir(const std::string& category) {
    fs::path root = find_project_root();
    fs::path trace_dir = root / "traces" / "dma" / category;
    fs::create_directories(trace_dir);
    return trace_dir;
}

/// Build full trace path
inline std::string make_trace_path(const std::string& category,
                                    const std::string& filename) {
    return (get_trace_dir(category) / filename).string();
}

// ============================================================================
// DMA Transfer Event Tracking
// ============================================================================

/// Represents a single DMA transfer event for tracing
struct DMATransferEvent {
    uint64_t transfer_id;        // Unique transfer ID
    uint64_t submit_cycle;       // Cycle when transfer was submitted
    uint64_t complete_cycle;     // Cycle when transfer completed (0 if pending)
    uint64_t src_addr;           // Source address
    uint64_t dst_addr;           // Destination address
    uint32_t size;               // Transfer size in bytes
    bool is_read;                // True if reading from device memory
    uint32_t channel;            // DMA channel used

    DMATransferEvent()
        : transfer_id(0), submit_cycle(0), complete_cycle(0)
        , src_addr(0), dst_addr(0), size(0), is_read(true), channel(0) {}
};

/// Memory controller command event for tracing
struct MemoryCommandEvent {
    uint64_t cycle;              // Cycle when command was issued
    std::string command;         // Command type (ACT, READ, WRITE, PRE)
    uint8_t channel;             // Memory channel
    uint8_t bank;                // Bank number
    uint64_t address;            // Row/column address
    uint64_t txn_id;             // Transaction ID

    MemoryCommandEvent()
        : cycle(0), channel(0), bank(0), address(0), txn_id(0) {}
};

// ============================================================================
// DMA Harness Statistics
// ============================================================================

/// Combined statistics from DMA, memory controller, and NoC
struct DMAHarnessStats {
    // DMA statistics
    uint64_t dma_transfers_completed = 0;
    uint64_t dma_transfers_pending = 0;
    uint64_t dma_bytes_transferred = 0;

    // Memory statistics
    uint64_t memory_reads = 0;
    uint64_t memory_writes = 0;
    uint64_t page_hits = 0;
    uint64_t page_empty = 0;
    uint64_t page_conflicts = 0;

    // Timing
    uint64_t total_cycles = 0;
    uint64_t dma_stall_cycles = 0;
    uint64_t noc_stall_cycles = 0;

    // Derived metrics
    double page_hit_rate() const {
        uint64_t total = page_hits + page_empty + page_conflicts;
        return total > 0 ? static_cast<double>(page_hits) / total : 0.0;
    }

    double effective_bandwidth_gbps(double clock_ghz = 2.0) const {
        if (total_cycles == 0) return 0.0;
        double bytes_per_cycle = static_cast<double>(dma_bytes_transferred) / total_cycles;
        return bytes_per_cycle * clock_ghz;
    }

    double memory_utilization() const {
        uint64_t active = memory_reads + memory_writes;
        if (total_cycles == 0) return 0.0;
        return static_cast<double>(active) / total_cycles;
    }
};

// ============================================================================
// DMA Harness
// ============================================================================

/// Test harness for DMA patterns
///
/// Integrates:
/// - CycleAccurateDMAEngine for data movement
/// - IMemoryController for DRAM access
/// - INoC for on-chip transport (optional)
///
/// Provides:
/// - Request submission helpers
/// - Matrix/tile operations
/// - Simulation control
/// - Statistics and trace export
class DMAHarness {
public:
    /// Construct with DMA and memory controller configurations
    DMAHarness(const CycleAccurateDMAConfig& dma_config,
               const MemoryControllerConfig& mc_config,
               bool include_noc = false)
        : dma_config_(dma_config)
        , mc_config_(mc_config)
        , include_noc_(include_noc)
    {
        // Create memory controller
        mc_ = create_memory_controller(mc_config);
        mc_->enable_tracing(true);

        // Create DMA engine
        dma_ = std::make_unique<CycleAccurateDMAEngine>(dma_config);
        dma_->enable_tracing(true);
        dma_->bind_memory_controller(mc_.get());

        // Optionally create NoC
        if (include_noc) {
            noc::NoCConfig noc_config;
            noc_config.rows = 4;
            noc_config.cols = 4;
            noc_config.buffer_depth = 8;
            noc_ = noc::create_noc(noc::NoCType::DATAFLOW, noc_config);
            dma_->bind_noc(noc_.get(), 0);  // Bind to router 0
        }

        // Create resource tracker
        tracker_ = std::make_unique<sw::trace::ResourceTracker>();
        dma_->set_resource_tracker(tracker_.get());
    }

    virtual ~DMAHarness() = default;

    // ========================================================================
    // Matrix Layout Configuration
    // ========================================================================

    /// Set the current matrix layout for tile operations
    void set_matrix_layout(const MatrixLayout& layout) {
        matrix_layout_ = layout;
    }

    /// Get current matrix layout
    const MatrixLayout& matrix_layout() const { return matrix_layout_; }

    // ========================================================================
    // Basic DMA Operations
    // ========================================================================

    /// Submit a single DMA transfer
    /// @return Transfer ID if accepted, nullopt if queue full
    std::optional<uint64_t> submit_transfer(const DMATransfer& transfer,
                                             std::function<void()> callback = nullptr) {
        auto result = dma_->submit(transfer, callback);
        if (result.has_value()) {
            record_transfer_submit(result.value(), transfer);
        }
        return result;
    }

    /// Submit read from device memory to L3 tile
    std::optional<uint64_t> submit_read(uint64_t src_addr, uint32_t size,
                                         uint64_t dst_addr = 0) {
        DMATransfer xfer;
        xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
        xfer.src_addr = src_addr;
        xfer.dst_type = DMAMemoryType::L3_TILE;
        xfer.dst_addr = dst_addr;
        xfer.size = size;

        auto id = dma_->submit(xfer, [this, id = next_transfer_id_]() {
            record_transfer_complete(id);
            transfers_completed_++;
        });
        if (id.has_value()) {
            record_transfer_submit(id.value(), xfer);
        }
        return id;
    }

    /// Submit write from L3 tile to device memory
    std::optional<uint64_t> submit_write(uint64_t dst_addr, uint32_t size,
                                          uint64_t src_addr = 0) {
        DMATransfer xfer;
        xfer.src_type = DMAMemoryType::L3_TILE;
        xfer.src_addr = src_addr;
        xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
        xfer.dst_addr = dst_addr;
        xfer.size = size;

        auto id = dma_->submit(xfer, [this, id = next_transfer_id_]() {
            record_transfer_complete(id);
            transfers_completed_++;
        });
        if (id.has_value()) {
            record_transfer_submit(id.value(), xfer);
        }
        return id;
    }

    // ========================================================================
    // Tile Operations
    // ========================================================================

    /// Submit DMA transfers to read a tile from the matrix
    /// Uses current matrix layout
    size_t submit_tile_read(size_t tile_row, size_t tile_col,
                            size_t tile_height, size_t tile_width,
                            uint64_t dst_base = 0) {
        auto transfers = matrix_layout_.tile_read_transfers(
            tile_row, tile_col, tile_height, tile_width, dst_base);

        size_t submitted = 0;
        for (auto& xfer : transfers) {
            uint64_t expected_id = next_transfer_id_;
            auto id = dma_->submit(xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), xfer);
                submitted++;
            }
        }
        return submitted;
    }

    /// Submit DMA transfers to write a tile to the matrix
    size_t submit_tile_write(size_t tile_row, size_t tile_col,
                             size_t tile_height, size_t tile_width,
                             uint64_t src_base = 0) {
        auto transfers = matrix_layout_.tile_write_transfers(
            tile_row, tile_col, tile_height, tile_width, src_base);

        size_t submitted = 0;
        for (auto& xfer : transfers) {
            uint64_t expected_id = next_transfer_id_;
            auto id = dma_->submit(xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), xfer);
                submitted++;
            }
        }
        return submitted;
    }

    // ========================================================================
    // STREAM Operations
    // ========================================================================

    /// Submit STREAM copy pattern: A[i] = B[i]
    /// @param src_base Source array base address
    /// @param dst_base Destination array base address
    /// @param count Number of elements
    /// @param elem_size Element size in bytes
    /// @param burst_size Transfer burst size
    size_t submit_stream_copy(uint64_t src_base, uint64_t dst_base,
                               size_t count, size_t elem_size = 8,
                               size_t burst_size = 256) {
        size_t total_bytes = count * elem_size;
        size_t num_bursts = (total_bytes + burst_size - 1) / burst_size;
        size_t submitted = 0;

        for (size_t i = 0; i < num_bursts; ++i) {
            size_t offset = i * burst_size;
            size_t size = std::min(burst_size, total_bytes - offset);

            // Read from source
            DMATransfer read_xfer;
            read_xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            read_xfer.src_addr = src_base + offset;
            read_xfer.dst_type = DMAMemoryType::L3_TILE;
            read_xfer.dst_addr = 0;
            read_xfer.size = static_cast<uint32_t>(size);

            uint64_t expected_id = next_transfer_id_;
            auto id = dma_->submit(read_xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), read_xfer);
                submitted++;
            }

            // Write to destination
            DMATransfer write_xfer;
            write_xfer.src_type = DMAMemoryType::L3_TILE;
            write_xfer.src_addr = 0;
            write_xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            write_xfer.dst_addr = dst_base + offset;
            write_xfer.size = static_cast<uint32_t>(size);

            expected_id = next_transfer_id_;
            id = dma_->submit(write_xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), write_xfer);
                submitted++;
            }
        }

        return submitted;
    }

    /// Submit STREAM triad pattern: A[i] = B[i] + k*C[i]
    /// Requires 2 reads + 1 write per burst
    size_t submit_stream_triad(uint64_t a_base, uint64_t b_base, uint64_t c_base,
                                size_t count, size_t elem_size = 8,
                                size_t burst_size = 256) {
        size_t total_bytes = count * elem_size;
        size_t num_bursts = (total_bytes + burst_size - 1) / burst_size;
        size_t submitted = 0;

        for (size_t i = 0; i < num_bursts; ++i) {
            size_t offset = i * burst_size;
            size_t size = std::min(burst_size, total_bytes - offset);

            // Read from B
            DMATransfer b_xfer;
            b_xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            b_xfer.src_addr = b_base + offset;
            b_xfer.dst_type = DMAMemoryType::L3_TILE;
            b_xfer.dst_addr = 0;
            b_xfer.size = static_cast<uint32_t>(size);

            uint64_t expected_id = next_transfer_id_;
            auto id = dma_->submit(b_xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), b_xfer);
                submitted++;
            }

            // Read from C
            DMATransfer c_xfer;
            c_xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            c_xfer.src_addr = c_base + offset;
            c_xfer.dst_type = DMAMemoryType::L3_TILE;
            c_xfer.dst_addr = TILE_L3_SIZE_BYTES / 2;  // Different tile region
            c_xfer.size = static_cast<uint32_t>(size);

            expected_id = next_transfer_id_;
            id = dma_->submit(c_xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), c_xfer);
                submitted++;
            }

            // Write to A
            DMATransfer a_xfer;
            a_xfer.src_type = DMAMemoryType::L3_TILE;
            a_xfer.src_addr = 0;
            a_xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            a_xfer.dst_addr = a_base + offset;
            a_xfer.size = static_cast<uint32_t>(size);

            expected_id = next_transfer_id_;
            id = dma_->submit(a_xfer, [this, expected_id]() {
                record_transfer_complete(expected_id);
                transfers_completed_++;
            });
            if (id.has_value()) {
                record_transfer_submit(id.value(), a_xfer);
                submitted++;
            }
        }

        return submitted;
    }

    // ========================================================================
    // Simulation Control
    // ========================================================================

    /// Advance simulation by one cycle
    void tick() {
        dma_->set_cycle(cycle_);
        mc_->set_cycle(cycle_);
        dma_->tick();
        mc_->tick();
        if (noc_) {
            noc_->step(cycle_);
        }
        cycle_++;
    }

    /// Run until all pending requests complete or timeout
    bool run_until_complete(uint64_t max_cycles = 100000) {
        uint64_t start = cycle_;
        while ((dma_->has_pending() || mc_->has_pending() ||
                (noc_ && !noc_->is_idle())) &&
               (cycle_ - start) < max_cycles) {
            tick();
        }
        return !dma_->has_pending() && !mc_->has_pending();
    }

    /// Run for a fixed number of cycles
    void run_cycles(uint64_t cycles) {
        for (uint64_t i = 0; i < cycles; ++i) {
            tick();
        }
    }

    /// Get current simulation cycle
    uint64_t current_cycle() const { return cycle_; }

    /// Check if DMA has pending transfers
    bool has_pending() const {
        return dma_->has_pending() || mc_->has_pending();
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get combined statistics
    DMAHarnessStats stats() const {
        DMAHarnessStats s;

        // DMA stats
        const auto& dma_stats = dma_->stats();
        s.dma_transfers_completed = dma_stats.transfers_completed;
        s.dma_transfers_pending = dma_stats.transfers_pending;
        s.dma_bytes_transferred = dma_stats.bytes_transferred;
        s.dma_stall_cycles = dma_stats.stall_cycles;

        // Memory stats
        const auto& mc_stats = mc_->stats();
        s.memory_reads = mc_stats.reads;
        s.memory_writes = mc_stats.writes;
        s.page_hits = mc_stats.page_hits;
        s.page_empty = mc_stats.page_empty;
        s.page_conflicts = mc_stats.page_conflicts;

        s.total_cycles = cycle_;

        return s;
    }

    /// Print statistics to stdout
    void print_stats(double clock_ghz = 2.0) const {
        auto s = stats();

        std::cout << "\n=== DMA Harness Statistics ===" << std::endl;

        std::cout << "\n--- DMA Engine ---" << std::endl;
        std::cout << "  Transfers completed: " << s.dma_transfers_completed << std::endl;
        std::cout << "  Bytes transferred: " << s.dma_bytes_transferred << std::endl;
        std::cout << "  Stall cycles: " << s.dma_stall_cycles << std::endl;

        std::cout << "\n--- Memory Controller ---" << std::endl;
        std::cout << "  Reads: " << s.memory_reads << std::endl;
        std::cout << "  Writes: " << s.memory_writes << std::endl;
        std::cout << "  Page hits: " << s.page_hits << std::endl;
        std::cout << "  Page empty: " << s.page_empty << std::endl;
        std::cout << "  Page conflicts: " << s.page_conflicts << std::endl;
        std::cout << "  Page hit rate: " << std::fixed << std::setprecision(1)
                  << (s.page_hit_rate() * 100.0) << "%" << std::endl;

        std::cout << "\n--- Timing ---" << std::endl;
        std::cout << "  Total cycles: " << s.total_cycles << std::endl;
        std::cout << "  Effective bandwidth: " << std::fixed << std::setprecision(2)
                  << s.effective_bandwidth_gbps(clock_ghz) << " GB/s" << std::endl;
    }

    // ========================================================================
    // Trace Export
    // ========================================================================

    /// Export DMA trace to JSON file with per-transfer events
    bool export_trace(const std::string& filename, double clock_ghz = 2.0) {
        std::string full_path = make_trace_path("", filename);

        // Create parent directories
        fs::path path(full_path);
        fs::create_directories(path.parent_path());

        std::ofstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open trace file: " << full_path << std::endl;
            return false;
        }

        // Convert cycles to microseconds for Chrome trace format
        auto cycle_to_us = [clock_ghz](uint64_t cycle) -> double {
            return static_cast<double>(cycle) / (clock_ghz * 1000.0);
        };

        // Export as Chrome trace format
        file << "[\n";

        // Add process/thread metadata
        file << "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
             << "\"args\": {\"name\": \"DMA_ENGINE\"}},\n";
        file << "  {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": 0, "
             << "\"args\": {\"name\": \"DMA Transfers\"}},\n";
        file << "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 2, "
             << "\"args\": {\"name\": \"MEMORY_CONTROLLER\"}},\n";
        file << "  {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 2, \"tid\": 0, "
             << "\"args\": {\"name\": \"Memory Operations\"}},\n";

        // Add summary event
        auto s = stats();
        file << "  {\"name\": \"Summary\", \"cat\": \"info\", \"ph\": \"i\", "
             << "\"ts\": 0, \"pid\": 0, \"s\": \"g\", \"args\": {"
             << "\"transfers\": " << s.dma_transfers_completed << ", "
             << "\"bytes\": " << s.dma_bytes_transferred << ", "
             << "\"page_hit_rate\": " << std::fixed << std::setprecision(2)
             << (s.page_hit_rate() * 100.0) << ", "
             << "\"bandwidth_gbps\": " << s.effective_bandwidth_gbps(clock_ghz)
             << "}},\n";

        // Get memory controller trace entries
        auto mc_trace_entries = mc_->trace_entries();

        // Compute actual start cycles for DMA transfers from MC commands
        // Strategy: First compute MC-to-transfer associations, then for each transfer,
        // find the earliest MC command that is associated with it.
        std::unordered_map<uint64_t, uint64_t> transfer_start_cycles;

        // Build a mapping from MC command index to DMA transfer ID
        // (same logic as find_dma_transfer, but precomputed)
        std::vector<int64_t> mc_to_transfer(mc_trace_entries.size(), -1);
        for (size_t i = 0; i < mc_trace_entries.size(); ++i) {
            const auto& entry = mc_trace_entries[i];
            bool is_read_cmd = (entry.transaction_type == sw::trace::TransactionType::READ ||
                                entry.transaction_type == sw::trace::TransactionType::BURST_READ);

            // Find the transfer with smallest complete_cycle >= MC complete_cycle
            int64_t best_id = -1;
            uint64_t best_complete = UINT64_MAX;
            for (const auto& evt : transfer_events_) {
                if (evt.complete_cycle >= entry.cycle_complete) {
                    if (evt.is_read == is_read_cmd || !is_read_cmd) {  // ACT/PRE can match any
                        if (evt.complete_cycle < best_complete) {
                            best_complete = evt.complete_cycle;
                            best_id = static_cast<int64_t>(evt.transfer_id);
                        }
                    }
                }
            }
            mc_to_transfer[i] = best_id;
        }

        // Now for each transfer, find the earliest MC command associated with it
        for (const auto& evt : transfer_events_) {
            uint64_t earliest_issue = UINT64_MAX;
            for (size_t i = 0; i < mc_trace_entries.size(); ++i) {
                if (mc_to_transfer[i] == static_cast<int64_t>(evt.transfer_id)) {
                    if (mc_trace_entries[i].cycle_issue < earliest_issue) {
                        earliest_issue = mc_trace_entries[i].cycle_issue;
                    }
                }
            }
            if (earliest_issue != UINT64_MAX) {
                transfer_start_cycles[evt.transfer_id] = earliest_issue;
            }
        }

        // Export per-transfer events
        bool has_mc_entries = !mc_trace_entries.empty();
        {
            std::lock_guard<std::mutex> lock(events_mutex_);
            for (size_t i = 0; i < transfer_events_.size(); ++i) {
                const auto& evt = transfer_events_[i];

                // Use computed start cycle if available, otherwise use submit_cycle
                uint64_t actual_start = evt.submit_cycle;
                auto it = transfer_start_cycles.find(evt.transfer_id);
                if (it != transfer_start_cycles.end()) {
                    actual_start = it->second;
                }

                // Calculate duration from actual start to completion
                uint64_t duration_cycles = (evt.complete_cycle > actual_start)
                    ? (evt.complete_cycle - actual_start) : 1;

                // Transfer type name
                std::string name = evt.is_read ? "DMA_READ" : "DMA_WRITE";

                // Start time and duration in microseconds
                double ts = cycle_to_us(actual_start);
                double dur = cycle_to_us(duration_cycles);

                // Ensure minimum visible duration
                if (dur < 0.001) dur = 0.001;

                file << "  {\"name\": \"" << name << "\", \"cat\": \"dma\", \"ph\": \"X\", "
                     << "\"ts\": " << std::fixed << std::setprecision(3) << ts << ", "
                     << "\"dur\": " << dur << ", "
                     << "\"pid\": 1, \"tid\": " << (evt.channel % 4) << ", "
                     << "\"args\": {"
                     << "\"id\": " << evt.transfer_id << ", "
                     << "\"src\": \"0x" << std::hex << evt.src_addr << std::dec << "\", "
                     << "\"dst\": \"0x" << std::hex << evt.dst_addr << std::dec << "\", "
                     << "\"size\": " << evt.size << ", "
                     << "\"channel\": " << evt.channel << ", "
                     << "\"submit_cycle\": " << actual_start << ", "
                     << "\"complete_cycle\": " << evt.complete_cycle
                     << "}}";

                // Add comma unless last event
                if (i < transfer_events_.size() - 1 || !memory_events_.empty() || has_mc_entries) {
                    file << ",";
                }
                file << "\n";
            }

            // Export local memory events (if any)
            for (size_t i = 0; i < memory_events_.size(); ++i) {
                const auto& evt = memory_events_[i];

                double ts = cycle_to_us(evt.cycle);

                file << "  {\"name\": \"" << evt.command << "\", \"cat\": \"memory\", \"ph\": \"i\", "
                     << "\"ts\": " << std::fixed << std::setprecision(3) << ts << ", "
                     << "\"s\": \"t\", "
                     << "\"pid\": 2, \"tid\": " << static_cast<int>(evt.bank % 8) << ", "
                     << "\"args\": {"
                     << "\"channel\": " << static_cast<int>(evt.channel) << ", "
                     << "\"bank\": " << static_cast<int>(evt.bank) << ", "
                     << "\"addr\": \"0x" << std::hex << evt.address << std::dec << "\", "
                     << "\"txn_id\": " << evt.txn_id
                     << "}}";

                if (i < memory_events_.size() - 1 || has_mc_entries) {
                    file << ",";
                }
                file << "\n";
            }
        }

        // Helper: find DMA transfer that this MC command serves
        // Strategy: Find the transfer with the smallest complete_cycle that is >= the MC command's complete_cycle
        // This correctly associates each MC command with the transfer it's completing.
        auto find_dma_transfer = [this](uint64_t complete_cycle, bool is_read) -> int64_t {
            std::lock_guard<std::mutex> lock(events_mutex_);
            int64_t best_id = -1;
            uint64_t best_complete = UINT64_MAX;

            for (const auto& evt : transfer_events_) {
                // The transfer's complete_cycle should be >= the MC command's complete_cycle
                // (the MC command must finish before or when the transfer completes)
                if (evt.complete_cycle >= complete_cycle) {
                    // Match read/write type
                    if (evt.is_read == is_read || !is_read) {  // ACT/PRE can match any
                        // Take the transfer with smallest complete_cycle (earliest completion)
                        if (evt.complete_cycle < best_complete) {
                            best_complete = evt.complete_cycle;
                            best_id = static_cast<int64_t>(evt.transfer_id);
                        }
                    }
                }
            }
            return best_id;
        };

        // Export memory controller trace entries with DMA linkage
        for (size_t i = 0; i < mc_trace_entries.size(); ++i) {
            const auto& entry = mc_trace_entries[i];

            // Convert transaction type to command name
            std::string cmd_name;
            bool is_read_cmd = false;
            switch (entry.transaction_type) {
                case sw::trace::TransactionType::ACTIVATE:    cmd_name = "ACT"; break;
                case sw::trace::TransactionType::PRECHARGE:   cmd_name = "PRE"; break;
                case sw::trace::TransactionType::REFRESH:     cmd_name = "REF"; break;
                case sw::trace::TransactionType::BURST_READ:  cmd_name = "READ"; is_read_cmd = true; break;
                case sw::trace::TransactionType::BURST_WRITE: cmd_name = "WRITE"; break;
                case sw::trace::TransactionType::TURNAROUND:  cmd_name = "TURNAROUND"; break;
                case sw::trace::TransactionType::READ:        cmd_name = "READ"; is_read_cmd = true; break;
                case sw::trace::TransactionType::WRITE:       cmd_name = "WRITE"; break;
                default: cmd_name = "CMD"; break;
            }

            // Find associated DMA transfer (using complete_cycle for proper association)
            int64_t dma_xfer_id = find_dma_transfer(entry.cycle_complete, is_read_cmd);

            // Calculate timing
            double ts = cycle_to_us(entry.cycle_issue);

            // For duration events (X), use issue to complete
            // For instant events (i), just use issue time
            bool has_duration = (entry.cycle_complete > entry.cycle_issue);

            if (has_duration) {
                double dur = cycle_to_us(entry.cycle_complete - entry.cycle_issue);
                if (dur < 0.001) dur = 0.001;

                file << "  {\"name\": \"" << cmd_name << "\", \"cat\": \"mc\", \"ph\": \"X\", "
                     << "\"ts\": " << std::fixed << std::setprecision(3) << ts << ", "
                     << "\"dur\": " << dur << ", "
                     << "\"pid\": 2, \"tid\": " << (entry.component_id % 8) << ", "
                     << "\"args\": {"
                     << "\"dma_transfer_id\": " << dma_xfer_id << ", "
                     << "\"txn_id\": " << entry.transaction_id << ", "
                     << "\"component\": " << static_cast<int>(entry.component_id) << ", "
                     << "\"issue_cycle\": " << entry.cycle_issue << ", "
                     << "\"complete_cycle\": " << entry.cycle_complete;
            } else {
                file << "  {\"name\": \"" << cmd_name << "\", \"cat\": \"mc\", \"ph\": \"i\", "
                     << "\"ts\": " << std::fixed << std::setprecision(3) << ts << ", "
                     << "\"s\": \"t\", "
                     << "\"pid\": 2, \"tid\": " << (entry.component_id % 8) << ", "
                     << "\"args\": {"
                     << "\"dma_transfer_id\": " << dma_xfer_id << ", "
                     << "\"txn_id\": " << entry.transaction_id << ", "
                     << "\"component\": " << static_cast<int>(entry.component_id) << ", "
                     << "\"cycle\": " << entry.cycle_issue;
            }

            // Add description if present
            if (!entry.description.empty()) {
                file << ", \"desc\": \"" << entry.description << "\"";
            }

            file << "}}";

            if (i < mc_trace_entries.size() - 1) {
                file << ",";
            }
            file << "\n";
        }

        file << "]\n";
        file.close();

        std::cout << "Trace exported to: " << full_path << std::endl;
        std::cout << "  DMA transfers: " << transfer_events_.size() << std::endl;
        std::cout << "  Memory events: " << memory_events_.size() << std::endl;
        std::cout << "  MC trace entries: " << mc_trace_entries.size() << std::endl;
        std::cout << "  Open with: https://ui.perfetto.dev" << std::endl;
        return true;
    }

    /// Get transfer events for external analysis
    const std::vector<DMATransferEvent>& transfer_events() const {
        return transfer_events_;
    }

    /// Clear recorded events
    void clear_events() {
        std::lock_guard<std::mutex> lock(events_mutex_);
        transfer_events_.clear();
        memory_events_.clear();
    }

    // ========================================================================
    // Component Access
    // ========================================================================

    /// Get DMA engine reference
    CycleAccurateDMAEngine& dma() { return *dma_; }
    const CycleAccurateDMAEngine& dma() const { return *dma_; }

    /// Get memory controller reference
    IMemoryController& memory_controller() { return *mc_; }
    const IMemoryController& memory_controller() const { return *mc_; }

    /// Get NoC reference (may be null)
    noc::INoC* noc() { return noc_.get(); }

private:
    // Configuration
    CycleAccurateDMAConfig dma_config_;
    MemoryControllerConfig mc_config_;
    bool include_noc_;

    // Components
    std::unique_ptr<CycleAccurateDMAEngine> dma_;
    std::unique_ptr<IMemoryController> mc_;
    std::unique_ptr<noc::INoC> noc_;
    std::unique_ptr<sw::trace::ResourceTracker> tracker_;

    // State
    uint64_t cycle_ = 0;
    std::atomic<uint64_t> transfers_completed_{0};

    // Current matrix layout
    MatrixLayout matrix_layout_;

    // Event tracking
    mutable std::mutex events_mutex_;
    std::vector<DMATransferEvent> transfer_events_;
    std::vector<MemoryCommandEvent> memory_events_;
    uint64_t next_transfer_id_ = 0;

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    /// Record a transfer submission
    void record_transfer_submit(uint64_t id, const DMATransfer& xfer) {
        std::lock_guard<std::mutex> lock(events_mutex_);

        DMATransferEvent evt;
        evt.transfer_id = next_transfer_id_++;
        evt.submit_cycle = cycle_;
        evt.complete_cycle = 0;  // Not yet complete
        evt.src_addr = xfer.src_addr;
        evt.dst_addr = xfer.dst_addr;
        evt.size = xfer.size;
        evt.is_read = (xfer.src_type == DMAMemoryType::DEVICE_MEMORY);
        evt.channel = static_cast<uint32_t>(id % dma_config_.num_channels);

        transfer_events_.push_back(evt);
    }

    /// Record a transfer completion
    void record_transfer_complete(uint64_t id) {
        std::lock_guard<std::mutex> lock(events_mutex_);

        // Find the transfer event and update completion time
        for (auto& evt : transfer_events_) {
            if (evt.transfer_id == id && evt.complete_cycle == 0) {
                evt.complete_cycle = cycle_;
                break;
            }
        }
    }

    /// Record a memory controller command
    void record_memory_command(const std::string& cmd, uint8_t channel,
                                uint8_t bank, uint64_t addr, uint64_t txn_id) {
        std::lock_guard<std::mutex> lock(events_mutex_);

        MemoryCommandEvent evt;
        evt.cycle = cycle_;
        evt.command = cmd;
        evt.channel = channel;
        evt.bank = bank;
        evt.address = addr;
        evt.txn_id = txn_id;

        memory_events_.push_back(evt);
    }
};

} // namespace sw::kpu::patterns::dma
