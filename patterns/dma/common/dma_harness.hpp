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
#include <functional>
#include <filesystem>
#include <atomic>

#include <sw/kpu/components/dma/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/components/dma/dma_placement.hpp>
#include <sw/kpu/components/memory/memory_controller_interface.hpp>
#include <sw/kpu/noc/noc_interface.hpp>
#include <sw/trace/resource_tracker.hpp>
#include <sw/trace/trace_exporter.hpp>

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
        return dma_->submit(transfer, callback);
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
        return dma_->submit(xfer, [this]() { transfers_completed_++; });
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
        return dma_->submit(xfer, [this]() { transfers_completed_++; });
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
            auto id = dma_->submit(xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) {
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
            auto id = dma_->submit(xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) {
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

            auto id = dma_->submit(read_xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) submitted++;

            // Write to destination
            DMATransfer write_xfer;
            write_xfer.src_type = DMAMemoryType::L3_TILE;
            write_xfer.src_addr = 0;
            write_xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            write_xfer.dst_addr = dst_base + offset;
            write_xfer.size = static_cast<uint32_t>(size);

            id = dma_->submit(write_xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) submitted++;
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
            auto id = dma_->submit(b_xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) submitted++;

            // Read from C
            DMATransfer c_xfer;
            c_xfer.src_type = DMAMemoryType::DEVICE_MEMORY;
            c_xfer.src_addr = c_base + offset;
            c_xfer.dst_type = DMAMemoryType::L3_TILE;
            c_xfer.dst_addr = TILE_L3_SIZE_BYTES / 2;  // Different tile region
            c_xfer.size = static_cast<uint32_t>(size);
            id = dma_->submit(c_xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) submitted++;

            // Write to A
            DMATransfer a_xfer;
            a_xfer.src_type = DMAMemoryType::L3_TILE;
            a_xfer.src_addr = 0;
            a_xfer.dst_type = DMAMemoryType::DEVICE_MEMORY;
            a_xfer.dst_addr = a_base + offset;
            a_xfer.size = static_cast<uint32_t>(size);
            id = dma_->submit(a_xfer, [this]() { transfers_completed_++; });
            if (id.has_value()) submitted++;
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

    /// Export DMA trace to JSON file
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

        // Export as Chrome trace format
        file << "[\n";

        // Add process name
        file << "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 4, "
             << "\"args\": {\"name\": \"DMA_ENGINE\"}},\n";
        file << "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 20, "
             << "\"args\": {\"name\": \"MEMORY_CONTROLLER\"}},\n";

        // Add summary event
        auto s = stats();
        file << "  {\"name\": \"DMA_SUMMARY\", \"cat\": \"summary\", \"ph\": \"i\", "
             << "\"ts\": 0, \"pid\": 0, \"s\": \"g\", \"args\": {"
             << "\"transfers\": " << s.dma_transfers_completed << ", "
             << "\"bytes\": " << s.dma_bytes_transferred << ", "
             << "\"page_hit_rate\": " << std::fixed << std::setprecision(2)
             << (s.page_hit_rate() * 100.0) << ", "
             << "\"bandwidth_gbps\": " << s.effective_bandwidth_gbps(clock_ghz)
             << "}}\n";

        file << "]\n";
        file.close();

        std::cout << "Trace exported to: " << full_path << std::endl;
        return true;
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
};

} // namespace sw::kpu::patterns::dma
