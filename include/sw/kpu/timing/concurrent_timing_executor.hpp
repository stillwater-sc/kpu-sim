// ============================================================================
// include/sw/kpu/timing/concurrent_timing_executor.hpp
// Concurrent Timing Executor for CSP-style simulation
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <sw/kpu/timing/tile_descriptor.hpp>
#include <sw/kpu/timing/credit_pool.hpp>
#include <sw/kpu/timing/tag_cam.hpp>
#include <sw/kpu/timing/dma_engine_process.hpp>
#include <sw/kpu/timing/block_mover_process.hpp>
#include <sw/kpu/timing/streamer_process.hpp>
#include <sw/kpu/timing/livelock_detector.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Concurrent Timing Executor for KPU data movement simulation
 *
 * The executor orchestrates multiple concurrent component processes:
 * - DMA engines: DRAM ↔ L3 transfers
 * - BlockMovers: L3 ↔ L2 transfers
 * - Streamers: L2 ↔ L1/Compute transfers
 *
 * All components operate concurrently with credit-based flow control.
 * The executor advances simulation cycle-by-cycle, collecting timing events
 * for trace visualization and performance analysis.
 */
class ConcurrentTimingExecutor {
public:
    /**
     * @brief Executor configuration
     */
    struct Config {
        // DMA configuration
        size_t num_dma_engines = 4;       ///< Number of DMA engines
        size_t dma_queue_depth = 8;       ///< Max in-flight per DMA engine
        double dma_bandwidth_gbps = 25.6; ///< DMA bandwidth per engine
        Cycle dma_startup_latency = 10;   ///< DMA startup latency

        // L3 configuration
        size_t l3_buffer_count = 32;      ///< Number of L3 buffers
        size_t l3_buffer_size = 64 * 1024; ///< Size of each L3 buffer (64KB)

        // BlockMover configuration
        size_t num_block_movers = 4;      ///< Number of BlockMovers
        double bm_bandwidth_gbps = 51.2;  ///< BlockMover bandwidth
        Cycle bm_startup_latency = 4;     ///< BlockMover startup latency

        // L2 configuration
        size_t l2_bank_count = 64;        ///< Number of L2 banks
        size_t l2_bank_size = 64 * 1024;  ///< Size of each L2 bank (64KB)

        // Streamer configuration
        size_t num_row_streamers = 2;     ///< Row streamers (West edge)
        size_t num_col_streamers = 2;     ///< Column streamers (North edge)
        double str_bandwidth_gbps = 102.4; ///< Streamer bandwidth
        Cycle str_startup_latency = 2;    ///< Streamer startup latency

        // Timing parameters
        double clock_ghz = 1.0;           ///< Reference clock in GHz
        Cycle max_cycles = 10'000'000;    ///< Maximum simulation cycles

        // Livelock detection
        bool enable_livelock_detection = true;
        Cycle livelock_threshold = 10000; ///< Cycles without progress

        // Work-conserving and priority aging
        bool enable_work_conserving = true;
        bool enable_priority_aging = false;
    };

    /**
     * @brief Simulation statistics
     */
    struct Statistics {
        Cycle total_cycles = 0;

        // Component busy cycles
        Cycle dma_busy_cycles = 0;
        Cycle bm_busy_cycles = 0;
        Cycle str_busy_cycles = 0;

        // Stall breakdown
        Cycle dma_credit_stalls = 0;
        Cycle bm_tag_stalls = 0;
        Cycle bm_credit_stalls = 0;
        Cycle str_tag_stalls = 0;
        Cycle str_credit_stalls = 0;

        // Throughput
        size_t tiles_loaded = 0;
        size_t tiles_stored = 0;
        size_t tiles_moved = 0;
        size_t tiles_writeback = 0;
        size_t tiles_fed = 0;
        size_t tiles_drained = 0;

        // Bytes transferred
        size_t bytes_loaded = 0;
        size_t bytes_stored = 0;

        // Utilization (0.0 - 1.0)
        [[nodiscard]] double dma_utilization() const {
            return total_cycles > 0 ? static_cast<double>(dma_busy_cycles) / total_cycles : 0.0;
        }

        [[nodiscard]] double bm_utilization() const {
            return total_cycles > 0 ? static_cast<double>(bm_busy_cycles) / total_cycles : 0.0;
        }

        [[nodiscard]] double str_utilization() const {
            return total_cycles > 0 ? static_cast<double>(str_busy_cycles) / total_cycles : 0.0;
        }

        // Bandwidth (GB/s)
        [[nodiscard]] double effective_load_bandwidth(double clock_ghz) const {
            if (total_cycles == 0) return 0.0;
            double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
            return static_cast<double>(bytes_loaded) / (seconds * 1e9);
        }

        [[nodiscard]] double effective_store_bandwidth(double clock_ghz) const {
            if (total_cycles == 0) return 0.0;
            double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
            return static_cast<double>(bytes_stored) / (seconds * 1e9);
        }
    };

    /**
     * @brief Construct executor with configuration
     */
    explicit ConcurrentTimingExecutor(const Config& config);

    // ========================================================================
    // Tile Scheduling API
    // ========================================================================

    /**
     * @brief Schedule a tile load from DRAM to L3
     * @param tile Tile descriptor
     * @param engine_id Optional specific DMA engine (-1 for auto-select)
     */
    void schedule_load(const TileDescriptor& tile, int engine_id = -1);

    /**
     * @brief Schedule a tile store from L3 to DRAM
     * @param tile Tile descriptor
     * @param engine_id Optional specific DMA engine (-1 for auto-select)
     */
    void schedule_store(const TileDescriptor& tile, int engine_id = -1);

    /**
     * @brief Schedule a tile move from L3 to L2
     * @param tile Tile descriptor
     * @param transpose Whether to transpose during transfer
     * @param mover_id Optional specific BlockMover (-1 for auto-select)
     */
    void schedule_move(const TileDescriptor& tile, bool transpose = false, int mover_id = -1);

    /**
     * @brief Schedule a tile writeback from L2 to L3
     * @param tile Tile descriptor
     * @param mover_id Optional specific BlockMover (-1 for auto-select)
     */
    void schedule_writeback(const TileDescriptor& tile, int mover_id = -1);

    /**
     * @brief Schedule a tile feed from L2 to compute
     * @param tile Tile descriptor
     * @param streamer_id Optional specific Streamer (-1 for auto-select)
     */
    void schedule_feed(const TileDescriptor& tile, int streamer_id = -1);

    /**
     * @brief Schedule a result drain from compute to L2
     * @param tile Tile descriptor
     * @param streamer_id Optional specific Streamer (-1 for auto-select)
     */
    void schedule_drain(const TileDescriptor& tile, int streamer_id = -1);

    // ========================================================================
    // Simulation Control
    // ========================================================================

    /**
     * @brief Run simulation to completion
     * @return true if completed normally, false if hit max_cycles or livelock
     */
    bool run();

    /**
     * @brief Step simulation by one cycle
     * @return true if simulation is complete
     */
    bool step();

    /**
     * @brief Check if simulation is complete
     */
    [[nodiscard]] bool is_complete() const;

    /**
     * @brief Reset simulation state
     */
    void reset();

    // ========================================================================
    // Results and Statistics
    // ========================================================================

    /**
     * @brief Get current cycle count
     */
    [[nodiscard]] Cycle current_cycle() const { return current_cycle_; }

    /**
     * @brief Get all timing events
     */
    [[nodiscard]] const std::vector<TimingEvent>& events() const { return events_; }

    /**
     * @brief Get simulation statistics
     */
    [[nodiscard]] Statistics get_statistics() const;

    /**
     * @brief Get configuration
     */
    [[nodiscard]] const Config& config() const { return config_; }

    // ========================================================================
    // Trace Export
    // ========================================================================

    /**
     * @brief Export events to Chrome trace format (JSON)
     * @param filename Output file path
     */
    void export_chrome_trace(const std::string& filename) const;

    /**
     * @brief Export events to CSV format
     * @param filename Output file path
     */
    void export_csv(const std::string& filename) const;

    // ========================================================================
    // Component Access (for testing/debugging)
    // ========================================================================

    [[nodiscard]] size_t num_dma_engines() const { return dma_engines_.size(); }
    [[nodiscard]] size_t num_block_movers() const { return block_movers_.size(); }
    [[nodiscard]] size_t num_row_streamers() const { return row_streamers_.size(); }
    [[nodiscard]] size_t num_col_streamers() const { return col_streamers_.size(); }

    [[nodiscard]] CreditPool& l3_credits() { return l3_credits_; }
    [[nodiscard]] CreditPool& l2_credits() { return l2_credits_; }
    [[nodiscard]] TagCAM& l3_tag_cam() { return l3_tag_cam_; }
    [[nodiscard]] TagCAM& l2_tag_cam() { return l2_tag_cam_; }

private:
    Config config_;
    Cycle current_cycle_ = 0;
    std::vector<TimingEvent> events_;

    // Credit pools
    CreditPool l3_credits_;
    CreditPool l2_credits_;

    // Tag CAMs
    TagCAM l3_tag_cam_;
    TagCAM l2_tag_cam_;

    // Component processes
    std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines_;
    std::vector<std::unique_ptr<BlockMoverProcess>> block_movers_;
    std::vector<std::unique_ptr<StreamerProcess>> row_streamers_;
    std::vector<std::unique_ptr<StreamerProcess>> col_streamers_;

    // Livelock detection
    std::unique_ptr<LivelockDetector> livelock_detector_;

    // Round-robin counters for work distribution
    mutable uint32_t next_dma_engine_ = 0;
    mutable uint32_t next_block_mover_ = 0;
    mutable uint32_t next_row_streamer_ = 0;
    mutable uint32_t next_col_streamer_ = 0;

    // ========================================================================
    // Work Assignment
    // ========================================================================

    /**
     * @brief Select DMA engine for a tile operation
     */
    [[nodiscard]] uint32_t select_dma_engine(const TileDescriptor& tile) const;

    /**
     * @brief Select BlockMover for a tile operation
     */
    [[nodiscard]] uint32_t select_block_mover(const TileDescriptor& tile) const;

    /**
     * @brief Select Streamer for a tile feed/drain
     */
    [[nodiscard]] uint32_t select_streamer(const TileDescriptor& tile, bool is_row) const;

    // ========================================================================
    // Internal Helpers
    // ========================================================================

    /**
     * @brief Create component processes based on configuration
     */
    void create_components();

    /**
     * @brief Collect statistics from all components
     */
    void collect_statistics(Statistics& stats) const;

    /**
     * @brief Count progress for livelock detection
     */
    [[nodiscard]] size_t count_completed_tiles() const;
};

// ============================================================================
// Implementation
// ============================================================================

inline ConcurrentTimingExecutor::ConcurrentTimingExecutor(const Config& config)
    : config_(config),
      l3_credits_(config.l3_buffer_count),
      l2_credits_(config.l2_bank_count),
      l3_tag_cam_(config.l3_buffer_count),
      l2_tag_cam_(config.l2_bank_count) {
    create_components();

    if (config_.enable_livelock_detection) {
        LivelockDetector::Config ld_config;
        ld_config.stall_threshold = config_.livelock_threshold;
        livelock_detector_ = std::make_unique<LivelockDetector>(ld_config);
    }
}

inline void ConcurrentTimingExecutor::create_components() {
    // Create DMA engines
    for (size_t i = 0; i < config_.num_dma_engines; ++i) {
        DMAEngineProcess::Config dma_config;
        dma_config.engine_id = static_cast<uint32_t>(i);
        dma_config.queue_depth = config_.dma_queue_depth;
        dma_config.bandwidth_gbps = config_.dma_bandwidth_gbps;
        dma_config.startup_latency = config_.dma_startup_latency;
        dma_config.clock_ghz = config_.clock_ghz;
        dma_config.name = "DMA";

        dma_engines_.push_back(std::make_unique<DMAEngineProcess>(
            dma_config, l3_credits_, l3_tag_cam_, l2_tag_cam_));
    }

    // Create BlockMovers (IDs 100+)
    for (size_t i = 0; i < config_.num_block_movers; ++i) {
        BlockMoverProcess::Config bm_config;
        bm_config.mover_id = static_cast<uint32_t>(100 + i);  // 100, 101, 102, ...
        bm_config.bandwidth_gbps = config_.bm_bandwidth_gbps;
        bm_config.startup_latency = config_.bm_startup_latency;
        bm_config.clock_ghz = config_.clock_ghz;
        bm_config.priority_aging = config_.enable_priority_aging;
        bm_config.name = "BM";

        block_movers_.push_back(std::make_unique<BlockMoverProcess>(
            bm_config, l3_tag_cam_, l3_credits_, l2_credits_, l2_tag_cam_));
    }

    // Create Row Streamers (West edge - for A matrix, IDs 200+)
    for (size_t i = 0; i < config_.num_row_streamers; ++i) {
        StreamerProcess::Config str_config;
        str_config.streamer_id = static_cast<uint32_t>(200 + i);  // 200, 201, ...
        str_config.type = StreamerType::ROW_STREAMER;
        str_config.bandwidth_gbps = config_.str_bandwidth_gbps;
        str_config.startup_latency = config_.str_startup_latency;
        str_config.clock_ghz = config_.clock_ghz;
        str_config.priority_aging = config_.enable_priority_aging;
        str_config.name = "RowSTR";

        row_streamers_.push_back(std::make_unique<StreamerProcess>(
            str_config, l2_tag_cam_, l2_credits_));
    }

    // Create Column Streamers (North edge - for B matrix, IDs 210+)
    for (size_t i = 0; i < config_.num_col_streamers; ++i) {
        StreamerProcess::Config str_config;
        str_config.streamer_id = static_cast<uint32_t>(210 + i);  // 210, 211, ...
        str_config.type = StreamerType::COL_STREAMER;
        str_config.bandwidth_gbps = config_.str_bandwidth_gbps;
        str_config.startup_latency = config_.str_startup_latency;
        str_config.clock_ghz = config_.clock_ghz;
        str_config.priority_aging = config_.enable_priority_aging;
        str_config.name = "ColSTR";

        col_streamers_.push_back(std::make_unique<StreamerProcess>(
            str_config, l2_tag_cam_, l2_credits_));
    }
}

inline void ConcurrentTimingExecutor::schedule_load(const TileDescriptor& tile, int engine_id) {
    uint32_t engine = (engine_id >= 0)
        ? static_cast<uint32_t>(engine_id)
        : select_dma_engine(tile);
    dma_engines_[engine % dma_engines_.size()]->schedule_load(tile);
}

inline void ConcurrentTimingExecutor::schedule_store(const TileDescriptor& tile, int engine_id) {
    uint32_t engine = (engine_id >= 0)
        ? static_cast<uint32_t>(engine_id)
        : select_dma_engine(tile);
    dma_engines_[engine % dma_engines_.size()]->schedule_store(tile);
}

inline void ConcurrentTimingExecutor::schedule_move(const TileDescriptor& tile, bool transpose, int mover_id) {
    uint32_t mover = (mover_id >= 0)
        ? static_cast<uint32_t>(mover_id)
        : select_block_mover(tile);
    block_movers_[mover % block_movers_.size()]->schedule_move(tile, transpose);
}

inline void ConcurrentTimingExecutor::schedule_writeback(const TileDescriptor& tile, int mover_id) {
    uint32_t mover = (mover_id >= 0)
        ? static_cast<uint32_t>(mover_id)
        : select_block_mover(tile);
    block_movers_[mover % block_movers_.size()]->schedule_writeback(tile);
}

inline void ConcurrentTimingExecutor::schedule_feed(const TileDescriptor& tile, int streamer_id) {
    // Determine if this is a row (A) or column (B) tile
    bool is_row = (tile.tile_id.matrix == isa::MatrixID::A);
    uint32_t streamer = (streamer_id >= 0)
        ? static_cast<uint32_t>(streamer_id)
        : select_streamer(tile, is_row);

    if (is_row) {
        row_streamers_[streamer % row_streamers_.size()]->schedule_feed(tile);
    } else {
        col_streamers_[streamer % col_streamers_.size()]->schedule_feed(tile);
    }
}

inline void ConcurrentTimingExecutor::schedule_drain(const TileDescriptor& tile, int streamer_id) {
    // Drains typically go through row streamers (result tiles)
    uint32_t streamer = (streamer_id >= 0)
        ? static_cast<uint32_t>(streamer_id)
        : select_streamer(tile, true);
    row_streamers_[streamer % row_streamers_.size()]->schedule_drain(tile);
}

inline bool ConcurrentTimingExecutor::run() {
    while (!is_complete() && current_cycle_ < config_.max_cycles) {
        step();

        // Check for livelock (every 100 cycles to avoid overhead)
        if (livelock_detector_ && (current_cycle_ % 100 == 0)) {
            LivelockDetector::ProgressMetrics metrics;
            metrics.tiles_dma_completed = 0;
            metrics.tiles_moved = 0;
            metrics.tiles_streamed = 0;
            for (const auto& engine : dma_engines_) {
                metrics.tiles_dma_completed += engine->total_bytes_loaded() / 1024;
            }
            for (const auto& mover : block_movers_) {
                metrics.tiles_moved += mover->total_tiles_moved();
            }
            for (const auto& streamer : row_streamers_) {
                metrics.tiles_streamed += streamer->total_tiles_fed();
            }
            auto result = livelock_detector_->check(current_cycle_, metrics);
            if (result.livelock_detected) {
                // Livelock detected - could log or throw
                return false;
            }
        }
    }
    return is_complete();
}

inline bool ConcurrentTimingExecutor::step() {
    // Tick all components (order doesn't matter within same cycle)

    // 1. Tick DMA engines
    for (auto& engine : dma_engines_) {
        auto engine_events = engine->tick(current_cycle_);
        events_.insert(events_.end(), engine_events.begin(), engine_events.end());
    }

    // 2. Tick BlockMovers
    for (auto& mover : block_movers_) {
        auto mover_events = mover->tick(current_cycle_);
        events_.insert(events_.end(), mover_events.begin(), mover_events.end());
    }

    // 3. Tick Row Streamers
    for (auto& streamer : row_streamers_) {
        auto str_events = streamer->tick(current_cycle_);
        events_.insert(events_.end(), str_events.begin(), str_events.end());
    }

    // 4. Tick Column Streamers
    for (auto& streamer : col_streamers_) {
        auto str_events = streamer->tick(current_cycle_);
        events_.insert(events_.end(), str_events.begin(), str_events.end());
    }

    // Advance clock
    ++current_cycle_;

    return is_complete();
}

inline bool ConcurrentTimingExecutor::is_complete() const {
    // Complete when all queues are empty and no in-flight work
    for (const auto& engine : dma_engines_) {
        if (!engine->is_idle() || engine->has_pending_work()) return false;
    }
    for (const auto& mover : block_movers_) {
        if (!mover->is_idle() || mover->has_pending_work()) return false;
    }
    for (const auto& streamer : row_streamers_) {
        if (!streamer->is_idle() || streamer->has_pending_work()) return false;
    }
    for (const auto& streamer : col_streamers_) {
        if (!streamer->is_idle() || streamer->has_pending_work()) return false;
    }
    return true;
}

inline void ConcurrentTimingExecutor::reset() {
    current_cycle_ = 0;
    events_.clear();

    l3_credits_.reset();
    l2_credits_.reset();
    l3_tag_cam_.reset();
    l2_tag_cam_.reset();

    for (auto& engine : dma_engines_) {
        engine->reset();
    }
    for (auto& mover : block_movers_) {
        mover->reset();
    }
    for (auto& streamer : row_streamers_) {
        streamer->reset();
    }
    for (auto& streamer : col_streamers_) {
        streamer->reset();
    }

    if (livelock_detector_) {
        livelock_detector_->reset();
    }

    next_dma_engine_ = 0;
    next_block_mover_ = 0;
    next_row_streamer_ = 0;
    next_col_streamer_ = 0;
}

inline ConcurrentTimingExecutor::Statistics ConcurrentTimingExecutor::get_statistics() const {
    Statistics stats;
    stats.total_cycles = current_cycle_;
    collect_statistics(stats);
    return stats;
}

inline void ConcurrentTimingExecutor::collect_statistics(Statistics& stats) const {
    // Aggregate from DMA engines
    for (const auto& engine : dma_engines_) {
        stats.dma_credit_stalls += engine->stall_cycles();
        stats.bytes_loaded += engine->total_bytes_loaded();
        stats.bytes_stored += engine->total_bytes_stored();
    }

    // Aggregate from BlockMovers
    for (const auto& mover : block_movers_) {
        stats.bm_tag_stalls += mover->stall_cycles_tag();
        stats.bm_credit_stalls += mover->stall_cycles_credit();
        stats.tiles_moved += mover->total_tiles_moved();
        stats.tiles_writeback += mover->total_tiles_writeback();
    }

    // Aggregate from Streamers
    for (const auto& streamer : row_streamers_) {
        stats.str_tag_stalls += streamer->stall_cycles_tag();
        stats.str_credit_stalls += streamer->stall_cycles_credit();
        stats.tiles_fed += streamer->total_tiles_fed();
        stats.tiles_drained += streamer->total_tiles_drained();
    }
    for (const auto& streamer : col_streamers_) {
        stats.str_tag_stalls += streamer->stall_cycles_tag();
        stats.str_credit_stalls += streamer->stall_cycles_credit();
        stats.tiles_fed += streamer->total_tiles_fed();
    }

    // Count events for loaded/stored tiles
    for (const auto& event : events_) {
        if (event.type == EventType::TILE_ARRIVED_L3) {
            stats.tiles_loaded++;
        } else if (event.type == EventType::DMA_STORE_COMPLETE) {
            stats.tiles_stored++;
        }
    }

    // Compute average busy cycles per component type
    // Stalls are aggregated across N parallel components, so divide by N to get
    // average stalls per component, then subtract from total_cycles.
    // This gives utilization as average activity across all components of that type.
    size_t n_dma = dma_engines_.size();
    size_t n_bm = block_movers_.size();
    size_t n_str = row_streamers_.size() + col_streamers_.size();

    // Compute average stalls per component, capped to prevent underflow
    Cycle avg_dma_stalls = n_dma > 0 ? stats.dma_credit_stalls / n_dma : 0;
    Cycle avg_bm_stalls = n_bm > 0 ? (stats.bm_tag_stalls + stats.bm_credit_stalls) / n_bm : 0;
    Cycle avg_str_stalls = n_str > 0 ? (stats.str_tag_stalls + stats.str_credit_stalls) / n_str : 0;

    // Busy = total - average stalls, capped at 0 to prevent underflow
    stats.dma_busy_cycles = avg_dma_stalls < stats.total_cycles ? stats.total_cycles - avg_dma_stalls : 0;
    stats.bm_busy_cycles = avg_bm_stalls < stats.total_cycles ? stats.total_cycles - avg_bm_stalls : 0;
    stats.str_busy_cycles = avg_str_stalls < stats.total_cycles ? stats.total_cycles - avg_str_stalls : 0;
}

inline size_t ConcurrentTimingExecutor::count_completed_tiles() const {
    size_t count = 0;
    for (const auto& engine : dma_engines_) {
        count += engine->total_bytes_loaded() / 1024;  // Approximate tile count
    }
    for (const auto& mover : block_movers_) {
        count += mover->total_tiles_moved();
    }
    for (const auto& streamer : row_streamers_) {
        count += streamer->total_tiles_fed();
    }
    return count;
}

inline uint32_t ConcurrentTimingExecutor::select_dma_engine(const TileDescriptor& tile) const {
    // Strategy: Round-robin for load balancing
    uint32_t engine = next_dma_engine_;
    next_dma_engine_ = (next_dma_engine_ + 1) % static_cast<uint32_t>(dma_engines_.size());
    (void)tile;  // Could use tile info for smarter selection
    return engine;
}

inline uint32_t ConcurrentTimingExecutor::select_block_mover(const TileDescriptor& tile) const {
    // Strategy: Round-robin for load balancing
    uint32_t mover = next_block_mover_;
    next_block_mover_ = (next_block_mover_ + 1) % static_cast<uint32_t>(block_movers_.size());
    (void)tile;
    return mover;
}

inline uint32_t ConcurrentTimingExecutor::select_streamer(const TileDescriptor& tile, bool is_row) const {
    if (is_row) {
        uint32_t streamer = next_row_streamer_;
        next_row_streamer_ = (next_row_streamer_ + 1) % static_cast<uint32_t>(row_streamers_.size());
        (void)tile;
        return streamer;
    } else {
        uint32_t streamer = next_col_streamer_;
        next_col_streamer_ = (next_col_streamer_ + 1) % static_cast<uint32_t>(col_streamers_.size());
        (void)tile;
        return streamer;
    }
}

inline void ConcurrentTimingExecutor::export_chrome_trace(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "{\"traceEvents\":[\n";

    // ========================================================================
    // Emit process and thread metadata first for human-readable trace display
    // Sort order follows dataflow: DMA → BlockMover → Streamer (top to bottom)
    // ========================================================================

    // Process name
    file << R"({"name":"process_name","ph":"M","pid":1,"tid":0,"args":{"name":"CSP Concurrent Timing Executor"}})";

    // DMA Channel threads (IDs 0-N, sort_index 0+)
    for (size_t i = 0; i < config_.num_dma_engines; ++i) {
        file << ",\n";
        file << R"({"name":"thread_name","ph":"M","pid":1,"tid":)" << i
             << R"(,"args":{"name":"DMA Channel )" << i << R"("}})";
        file << ",\n";
        file << R"({"name":"thread_sort_index","ph":"M","pid":1,"tid":)" << i
             << R"(,"args":{"sort_index":)" << i << R"(}})";
    }

    // BlockMover threads (IDs 100+, sort_index 10+)
    for (size_t i = 0; i < config_.num_block_movers; ++i) {
        file << ",\n";
        file << R"({"name":"thread_name","ph":"M","pid":1,"tid":)" << (100 + i)
             << R"(,"args":{"name":"BlockMover )" << i << R"("}})";
        file << ",\n";
        file << R"({"name":"thread_sort_index","ph":"M","pid":1,"tid":)" << (100 + i)
             << R"(,"args":{"sort_index":)" << (10 + i) << R"(}})";
    }

    // Row Streamer threads (IDs 200+, sort_index 20+)
    for (size_t i = 0; i < config_.num_row_streamers; ++i) {
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (200 + i)
             << ",\"args\":{\"name\":\"Row Streamer " << i << " - A matrix\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (200 + i)
             << ",\"args\":{\"sort_index\":" << (20 + i) << "}}";
    }

    // Column Streamer threads (IDs 210+, sort_index 30+)
    for (size_t i = 0; i < config_.num_col_streamers; ++i) {
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (210 + i)
             << ",\"args\":{\"name\":\"Col Streamer " << i << " - B matrix\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (210 + i)
             << ",\"args\":{\"sort_index\":" << (30 + i) << "}}";
    }

    // ========================================================================
    // Emit timing events
    // ========================================================================
    for (const auto& event : events_) {
        file << ",\n";
        file << event.to_chrome_trace_json();
    }

    file << "\n]}\n";
}

inline void ConcurrentTimingExecutor::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    // Header
    file << "cycle,type,component,tile_id,duration,slot_id\n";

    for (const auto& event : events_) {
        file << event.cycle << ","
             << to_string(event.type) << ","
             << event.component_name << ","
             << static_cast<int>(event.tile_id.matrix) << "_"
             << event.tile_id.ti << "_"
             << event.tile_id.tj << "_"
             << event.tile_id.tk << ","
             << event.duration << ","
             << event.slot_id << "\n";
    }
}

} // namespace sw::kpu::timing
