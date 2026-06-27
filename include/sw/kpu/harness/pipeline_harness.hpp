// ============================================================================
// include/sw/kpu/harness/pipeline_harness.hpp
// Full data movement pipeline harness
//
// Tests the complete DRAM->L3->L2->L1->Compute->L2->L3->DRAM pipeline:
// - End-to-end data flow validation
// - Pipeline efficiency analysis
// - Bottleneck identification
// - Double-buffering verification
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/harness/pattern_harness_base.hpp>
#include <sw/kpu/harness/dma_harness.hpp>
#include <sw/kpu/harness/block_mover_harness.hpp>
#include <sw/kpu/harness/streamer_harness.hpp>
#include <sw/kpu/harness/tile_journey_tracker.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace sw::kpu::harness {

// ============================================================================
// Pipeline Statistics
// ============================================================================

/// Aggregate statistics for the full pipeline
struct PipelineStats {
    // Component stats
    DMAHarnessStats dma_stats;
    BlockMoverHarnessStats block_mover_stats;
    StreamerHarnessStats streamer_stats;

    // Pipeline metrics
    Cycle total_cycles = 0;
    Cycle stall_cycles = 0;
    Cycle compute_cycles = 0;

    // Efficiency metrics
    double pipeline_efficiency = 0.0;       // Compute / total cycles
    double compute_utilization = 0.0;       // Actual vs peak compute
    double memory_bandwidth_utilization = 0.0;

    // Bottleneck analysis
    std::string bottleneck_component;
    double bottleneck_utilization = 0.0;

    /// Calculate derived metrics
    void finalize(double peak_gflops, double peak_bw_gbps, double clock_ghz);

    /// Generate summary string
    std::string to_string() const;

    /// Reset all statistics
    void reset();
};

// ============================================================================
// Schedule Representation
// ============================================================================

/// Operation in a data movement schedule
struct ScheduleOperation {
    enum class Type {
        DMA_LOAD,       // DRAM -> L3
        DMA_STORE,      // L3 -> DRAM
        BM_L3_TO_L2,    // L3 -> L2
        BM_L2_TO_L3,    // L2 -> L3
        STREAM_ROWS,    // L2 -> L1 (A operand)
        STREAM_COLS,    // L2 -> L1 (B operand)
        COMPUTE,        // Trigger matmul
        DRAIN,          // L1 -> L2 (result)
        BARRIER         // Synchronization point
    };

    Type type;
    TileID tile_id;
    uint64_t address = 0;       // DRAM address for DMA operations
    uint32_t src_buffer = 0;    // Source buffer/bank ID
    uint32_t dst_buffer = 0;    // Destination buffer/bank ID
    Transform transform = Transform::IDENTITY;
    std::vector<TileID> dependencies;  // Tiles that must complete first

    /// Create DMA load operation
    static ScheduleOperation dma_load(const TileID& tile, uint64_t addr, uint32_t l3_buf = UINT32_MAX) {
        ScheduleOperation op;
        op.type = Type::DMA_LOAD;
        op.tile_id = tile;
        op.address = addr;
        op.dst_buffer = l3_buf;
        return op;
    }

    /// Create BlockMover L3->L2 operation
    static ScheduleOperation bm_l3_to_l2(const TileID& tile, uint32_t l3_buf, uint32_t l2_bank = UINT32_MAX,
                                          Transform xform = Transform::IDENTITY) {
        ScheduleOperation op;
        op.type = Type::BM_L3_TO_L2;
        op.tile_id = tile;
        op.src_buffer = l3_buf;
        op.dst_buffer = l2_bank;
        op.transform = xform;
        return op;
    }

    /// Create stream rows operation
    static ScheduleOperation stream_rows(const TileID& tile, uint32_t l2_bank) {
        ScheduleOperation op;
        op.type = Type::STREAM_ROWS;
        op.tile_id = tile;
        op.src_buffer = l2_bank;
        return op;
    }

    /// Create stream cols operation
    static ScheduleOperation stream_cols(const TileID& tile, uint32_t l2_bank) {
        ScheduleOperation op;
        op.type = Type::STREAM_COLS;
        op.tile_id = tile;
        op.src_buffer = l2_bank;
        return op;
    }

    /// Create drain operation
    static ScheduleOperation drain(const TileID& tile, uint32_t l2_bank) {
        ScheduleOperation op;
        op.type = Type::DRAIN;
        op.tile_id = tile;
        op.dst_buffer = l2_bank;
        return op;
    }
};

/// Complete schedule for pipeline execution
struct PipelineSchedule {
    std::vector<ScheduleOperation> operations;

    /// Add operation
    void add(ScheduleOperation op) {
        operations.push_back(std::move(op));
    }

    /// Clear schedule
    void clear() {
        operations.clear();
    }

    /// Get operation count
    size_t size() const { return operations.size(); }
};

// ============================================================================
// Data Movement Pipeline Harness
// ============================================================================

/// Full integrated pipeline test harness
///
/// Combines DMA, BlockMover, and Streamer harnesses to test
/// complete data movement schedules.
///
class DataMovementPipelineHarness : public PatternHarnessBase<PipelineHarnessConfig> {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    explicit DataMovementPipelineHarness(const PipelineHarnessConfig& config);
    ~DataMovementPipelineHarness() override = default;

    // ========================================================================
    // Simulation Control
    // ========================================================================

    bool run() override;
    void tick() override;
    void reset() override;
    bool has_pending() const override;

    /// Step simulation by specified cycles
    void step(Cycle cycles) override;

    // ========================================================================
    // Memory Initialization
    // ========================================================================

    /// Initialize DRAM with matrix data
    /// @param matrix_id Matrix identifier
    /// @param data Matrix data (row-major)
    /// @param rows Number of rows
    /// @param cols Number of columns
    /// @param element_size Bytes per element
    void init_dram(uint32_t matrix_id, const void* data,
                   uint32_t rows, uint32_t cols, uint32_t element_size);

    /// Initialize DRAM with pattern data
    void init_dram_pattern(uint32_t matrix_id, uint32_t rows, uint32_t cols,
                           uint32_t element_size);

    /// Read matrix from DRAM
    std::vector<uint8_t> read_dram(uint32_t matrix_id);

    // ========================================================================
    // Schedule Execution
    // ========================================================================

    /// Load and execute a schedule
    void load_schedule(const PipelineSchedule& schedule);

    /// Execute next operation in schedule
    bool execute_next_operation();

    /// Get current operation index
    size_t current_operation_index() const { return current_op_index_; }

    /// Get total operations in schedule
    size_t total_operations() const { return schedule_.size(); }

    // ========================================================================
    // Component Access
    // ========================================================================

    /// Get DMA harness
    DMAHarness& dma_harness() { return *dma_harness_; }
    const DMAHarness& dma_harness() const { return *dma_harness_; }

    /// Get BlockMover harness
    BlockMoverHarness& block_mover_harness() { return *block_mover_harness_; }
    const BlockMoverHarness& block_mover_harness() const { return *block_mover_harness_; }

    /// Get Streamer harness
    StreamerHarness& streamer_harness() { return *streamer_harness_; }
    const StreamerHarness& streamer_harness() const { return *streamer_harness_; }

    // ========================================================================
    // Tile Journey Tracking
    // ========================================================================

    /// Get tile journeys from all components
    std::vector<TileJourney> get_tile_journeys() const;

    /// Get journey for a specific tile
    std::optional<TileJourney> get_tile_journey(const TileID& tile_id) const;

    /// Get tile journey statistics
    TileJourneyStats get_journey_stats() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get aggregate pipeline statistics
    PipelineStats get_pipeline_stats() const;

    /// Print pipeline statistics
    void print_stats(std::ostream& os = std::cout) const override;

    // ========================================================================
    // Validation
    // ========================================================================

    ValidationResult validate() const override;

    /// Validate output against expected data
    bool validate_output(uint32_t matrix_id, const void* expected,
                         size_t size, double tolerance = 1e-5);

protected:
    /// Execute a single schedule operation
    bool execute_operation(const ScheduleOperation& op);

    /// Check if operation dependencies are satisfied
    bool dependencies_satisfied(const ScheduleOperation& op) const;

    /// Collect aggregate statistics
    void collect_stats();

private:
    // Component harnesses
    std::unique_ptr<DMAHarness> dma_harness_;
    std::unique_ptr<BlockMoverHarness> block_mover_harness_;
    std::unique_ptr<StreamerHarness> streamer_harness_;

    // Simulated DRAM
    std::unordered_map<uint32_t, std::vector<uint8_t>> dram_data_;
    std::unordered_map<uint32_t, uint64_t> dram_base_addresses_;

    // Schedule execution
    PipelineSchedule schedule_;
    size_t current_op_index_ = 0;

    // Completed tile tracking
    std::unordered_map<TileID, Cycle, TileIDHash> completed_tiles_;

    // Buffer allocation tracking
    std::unordered_map<TileID, uint32_t, TileIDHash> tile_l3_buffers_;
    std::unordered_map<TileID, uint32_t, TileIDHash> tile_l2_banks_;

    // Aggregate statistics
    PipelineStats pipeline_stats_;
};

// ============================================================================
// PipelineStats Implementation
// ============================================================================

inline void PipelineStats::finalize(double peak_gflops, double peak_bw_gbps, double clock_ghz) {
    total_cycles = std::max({
        dma_stats.busy_cycles + dma_stats.idle_cycles,
        block_mover_stats.busy_cycles + block_mover_stats.idle_cycles,
        streamer_stats.busy_cycles + streamer_stats.idle_cycles
    });

    compute_cycles = streamer_stats.compute_cycles;
    stall_cycles = dma_stats.credit_stall_cycles +
                   block_mover_stats.credit_stall_cycles +
                   streamer_stats.credit_stall_cycles;

    // Pipeline efficiency
    if (total_cycles > 0) {
        pipeline_efficiency = static_cast<double>(compute_cycles) / static_cast<double>(total_cycles);
    }

    // Compute utilization
    if (peak_gflops > 0 && total_cycles > 0) {
        double achieved_gflops = (static_cast<double>(streamer_stats.compute_operations) /
                                  static_cast<double>(total_cycles)) * clock_ghz;
        compute_utilization = achieved_gflops / peak_gflops;
    }

    // Memory bandwidth utilization
    if (peak_bw_gbps > 0 && total_cycles > 0) {
        double bytes = static_cast<double>(dma_stats.bytes_transferred);
        double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        double achieved_bw = (bytes / 1e9) / seconds;
        memory_bandwidth_utilization = achieved_bw / peak_bw_gbps;
    }

    // Bottleneck analysis
    double dma_util = dma_stats.utilization();
    double bm_util = block_mover_stats.utilization();
    double str_util = streamer_stats.utilization();

    bottleneck_utilization = std::max({dma_util, bm_util, str_util});
    if (bottleneck_utilization == dma_util) {
        bottleneck_component = "DMA";
    } else if (bottleneck_utilization == bm_util) {
        bottleneck_component = "BlockMover";
    } else {
        bottleneck_component = "Streamer";
    }
}

inline std::string PipelineStats::to_string() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Pipeline Statistics\n";
    ss << "===================\n";
    ss << "\nTiming:\n";
    ss << "  Total cycles:   " << total_cycles << "\n";
    ss << "  Compute cycles: " << compute_cycles << "\n";
    ss << "  Stall cycles:   " << stall_cycles << "\n";
    ss << "\nEfficiency:\n";
    ss << "  Pipeline:       " << (pipeline_efficiency * 100) << "%\n";
    ss << "  Compute:        " << (compute_utilization * 100) << "%\n";
    ss << "  Memory BW:      " << (memory_bandwidth_utilization * 100) << "%\n";
    ss << "\nBottleneck:\n";
    ss << "  Component:      " << bottleneck_component << "\n";
    ss << "  Utilization:    " << (bottleneck_utilization * 100) << "%\n";
    ss << "\n--- DMA ---\n" << dma_stats.to_string();
    ss << "\n--- BlockMover ---\n" << block_mover_stats.to_string();
    ss << "\n--- Streamer ---\n" << streamer_stats.to_string();
    return ss.str();
}

inline void PipelineStats::reset() {
    dma_stats.reset();
    block_mover_stats.reset();
    streamer_stats.reset();
    total_cycles = stall_cycles = compute_cycles = 0;
    pipeline_efficiency = compute_utilization = memory_bandwidth_utilization = 0.0;
    bottleneck_component.clear();
    bottleneck_utilization = 0.0;
}

} // namespace sw::kpu::harness
