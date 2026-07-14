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
#include <sw/kpu/timing/memory_controller_process.hpp>
#include <sw/kpu/timing/dma_engine_process.hpp>
#include <sw/kpu/timing/block_mover_process.hpp>
#include <sw/kpu/timing/streamer_process.hpp>
#include <sw/kpu/timing/livelock_detector.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw::kpu::timing {

/**
 * @brief Concurrent Timing Executor for KPU data movement simulation
 *
 * The executor orchestrates multiple concurrent component processes:
 * - Memory Controllers: DRAM access contention (command bus, bank states)
 * - DMA engines: DRAM ↔ L3 transfers (use MCs for DRAM access)
 * - BlockMovers: L3 ↔ L2 transfers
 * - Streamers: L2 ↔ L1/Compute transfers
 *
 * Architecture:
 *   DMA Engine (CSP Process) --uses--> Memory Controller (Resource)
 *
 * All components operate concurrently with credit-based flow control.
 * The executor advances simulation cycle-by-cycle, collecting timing events
 * for trace visualization and performance analysis.
 */
class ConcurrentTimingExecutor {
public:
    enum class FunctionalActivation {
        NONE,
        RELU
    };

    struct MatMulComputeSpec {
        std::vector<TileID> a_tiles;
        std::vector<TileID> b_tiles;
        // Inputs produced by an earlier compute may remain in the compute
        // fabric and feed this operation without a drain/reload round trip.
        std::vector<TileID> resident_tiles;
        std::vector<float> bias;
        FunctionalActivation activation = FunctionalActivation::NONE;
    };
    struct FunctionalComputeSpec {
        std::vector<TileID> input_tiles;
        std::vector<TileID> resident_tiles;
        std::function<TilePayload(const std::vector<TilePayload>&)> operation;
    };
    /**
     * @brief Executor configuration
     */
    struct Config {
        // Grid topology configuration
        size_t num_memory_controllers = 1;  ///< Number of memory controllers (1 per DRAM channel)
        size_t num_dma_engines = 1;         ///< Number of DMA engines
        size_t l3_tile_rows = 2;            ///< L3 tile grid rows (memory tiles)
        size_t l3_tile_cols = 2;            ///< L3 tile grid columns
        size_t compute_tile_rows = 2;       ///< Compute tile grid rows
        size_t compute_tile_cols = 2;       ///< Compute tile grid columns

        // Memory Controller configuration
        size_t mc_request_queue_depth = 32; ///< MC request queue depth
        size_t mc_num_banks = 16;           ///< Banks per MC (LPDDR5: 4 BG × 4 banks)
        double mc_bandwidth_gbps = 25.6;    ///< MC bandwidth per channel
        Cycle mc_startup_latency = 5;       ///< MC command startup latency
        Cycle mc_t_cl = 10;                 ///< CAS latency
        Cycle mc_t_rcd = 15;                ///< RAS to CAS delay
        Cycle mc_t_rp = 15;                 ///< Row precharge time
        Cycle mc_t_burst = 4;               ///< Burst transfer duration

        // DMA Engine configuration
        size_t dma_queue_depth = 32;        ///< DMA request queue depth

        // L3 configuration
        size_t l3_buffer_count = 32;      ///< Number of L3 buffers
        size_t l3_buffer_size = 64 * 1024; ///< Size of each L3 buffer (64KB)

        // BlockMover configuration
        size_t num_block_movers = 4;      ///< Number of BlockMovers (= l3_tile_rows * l3_tile_cols)
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

        // Compute configuration
        Cycle compute_latency = 32;       ///< Base cycles for tile computation (after all inputs fed)
        Cycle compute_cycles_per_k_slice = 32;  ///< Additional cycles per K slice beyond the
                                                ///< first (accumulation depth; K-slice count is
                                                ///< derived from the COMPUTE dependency set)

        // Timing parameters
        double clock_ghz = 1.0;           ///< Reference clock in GHz
        Cycle max_cycles = 10'000'000;    ///< Maximum simulation cycles

        // Livelock detection
        bool enable_livelock_detection = true;
        Cycle livelock_threshold = 10000; ///< Cycles without progress

        // Work-conserving and priority aging
        bool enable_work_conserving = true;
        bool enable_priority_aging = false;

        // Per-matrix credit partitioning (issue #89): partitions the L3/L2
        // credit pools by matrix class (A/B/C, equal split with remainder
        // to C) via PartitionedCreditPool, restoring the original v0.9
        // design's structural livelock prevention - no matrix class can
        // monopolize the buffers another needs, regardless of schedule
        // ordering. Defense-in-depth for hand-written or third-party
        // schedules; the shipped generators are constructively safe (#67)
        // even without it. Off by default because partitioning
        // intentionally forbids single-matrix workloads (e.g. pure DMA
        // streaming tests) from filling the whole pool.
        bool partition_l3_credits = false;
        bool partition_l2_credits = false;

        // Credit reserves (optional guard against upstream producers
        // starving downstream completion paths - see issue #61):
        // - L3 reserve: credits DMA loads may not consume, kept for C-tile
        //   writebacks (BlockMover L2->L3)
        // - L2 reserve: credits BlockMover moves may not consume, kept for
        //   compute-result drains (Streamer)
        // Default 0: tile-reuse deduplication (DMA in-flight dedup, BlockMover
        // move dedup, tile-affine work assignment) removes the credit leaks
        // that caused livelock, and a reserve of 0 keeps load-only workloads
        // able to fill the entire pool. Set >0 for schedules with extreme
        // prefetch depth if writeback/drain starvation is observed.
        // Effective values are clamped to (pool_size - 1) / 2 so small pools
        // keep at least half their credits usable by the primary producer.
        size_t l3_writeback_credit_reserve = 0;
        size_t l2_drain_credit_reserve = 0;

        /// Reserve clamp: never reserve more than half of (pool - 1)
        [[nodiscard]] static size_t clamp_reserve(size_t reserve, size_t pool_size) {
            size_t max_reserve = pool_size > 0 ? (pool_size - 1) / 2 : 0;
            return reserve < max_reserve ? reserve : max_reserve;
        }
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
            return total_cycles > 0 ? static_cast<double>(dma_busy_cycles) / static_cast<double>(total_cycles) : 0.0;
        }

        [[nodiscard]] double bm_utilization() const {
            return total_cycles > 0 ? static_cast<double>(bm_busy_cycles) / static_cast<double>(total_cycles) : 0.0;
        }

        [[nodiscard]] double str_utilization() const {
            return total_cycles > 0 ? static_cast<double>(str_busy_cycles) / static_cast<double>(total_cycles) : 0.0;
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

    /**
     * @brief Schedule a compute completion (signals result tile is ready)
     * @param tile Result tile descriptor (C matrix tile)
     * @param dependency_tile Single input tile that must be FED before
     *        compute starts (legacy single-dependency form)
     *
     * This must be called after all FEED operations for the input tiles
     * that contribute to this result. DRAIN waits for this before proceeding.
     */
    void schedule_compute(const TileDescriptor& tile, const TileID& dependency_tile);

    /**
     * @brief Schedule a compute with the full dependency set
     * @param tile Result tile descriptor (C matrix tile)
     * @param dependencies ALL input tiles (every A and B K-slice) that must
     *        be FED before compute starts
     *
     * Compute starts only when every dependency has been FED, and its
     * latency scales with the K-slice count (dependencies.size() / 2):
     * latency = compute_latency + (k_slices - 1) * compute_cycles_per_k_slice.
     *
     * Feed tracking is per-instance: each dependency requires as many
     * completed feeds as were scheduled for that tile when this compute was
     * scheduled, so a feed consumed by an earlier output iteration cannot
     * satisfy a later iteration's compute.
     */
    void schedule_compute(const TileDescriptor& tile, std::vector<TileID> dependencies);

    /**
     * @brief Schedule a compute with both fed and compute-resident inputs
     * @param tile Result tile descriptor
     * @param feed_dependencies input tiles that must be FED before compute
     * @param resident_dependencies input tiles produced by a PRIOR compute
     *        that stay in the compute fabric (no fresh feed) - each requires
     *        that many completed computes of the producing tile
     *
     * The timing-tier counterpart of MatMulComputeSpec/FunctionalComputeSpec
     * resident_tiles, for schedules (e.g. online softmax, issue #155) whose
     * apply computes consume a running-stat tile resident rather than via a
     * DRAM round-trip. A resident dependency cannot be satisfied until the
     * producing compute completes, so it orders the chain without a race.
     */
    void schedule_compute(const TileDescriptor& tile,
                          std::vector<TileID> feed_dependencies,
                          std::vector<TileID> resident_dependencies);

    /**
     * @brief Schedule a compute completion (no explicit dependency)
     * @param tile Result tile descriptor (C matrix tile)
     *
     * This version auto-generates a dependency based on the tile ID.
     */
    void schedule_compute(const TileDescriptor& tile);

    /**
     * @brief Schedule a value-producing tiled matmul under CSP ordering.
     *
     * Every listed A/B tile must complete the feed occurrence that precedes
     * this call. The numeric result is produced at compute completion and is
     * then made visible to DRAIN through the normal compute-result TagCAM.
     */
    void schedule_matmul_compute(const TileDescriptor& tile,
                                 const MatMulComputeSpec& spec);
    void schedule_functional_compute(const TileDescriptor& tile,
                                     const FunctionalComputeSpec& spec);

    // ========================================================================
    // Functional Payload API
    // ========================================================================

    void set_tile_payload(const TileID& tile_id, TilePayload payload) {
        if (!payload.valid()) {
            throw std::invalid_argument("Tile payload dimensions do not match value count");
        }
        functional_payloads_enabled_ = true;
        write_payload(MemoryLevel::DRAM, tile_id, std::move(payload));
    }

    [[nodiscard]] bool has_tile_payload(const TileID& tile_id) const {
        return find_payload(tile_id) != nullptr;
    }

    [[nodiscard]] const TilePayload& tile_payload(const TileID& tile_id) const {
        const auto* payload = find_payload(tile_id);
        if (payload == nullptr) {
            throw std::out_of_range("No numeric payload for " + tile_id.to_string());
        }
        return *payload;
    }

    [[nodiscard]] bool has_tile_payload_at(MemoryLevel level, const TileID& tile_id) const {
        return payload_store(level).find(tile_id) != payload_store(level).end();
    }

    [[nodiscard]] const TilePayload& tile_payload_at(MemoryLevel level,
                                                      const TileID& tile_id) const {
        const auto& store = payload_store(level);
        auto it = store.find(tile_id);
        if (it == store.end()) {
            throw std::out_of_range(std::string("No payload at ") + to_string(level) +
                                    " for " + tile_id.to_string());
        }
        return it->second.payload;
    }

    void clear_tile_payloads() {
        dram_payloads_.clear(); l3_payloads_.clear(); l2_payloads_.clear();
        l1_payloads_.clear(); compute_payloads_.clear();
    }

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

    [[nodiscard]] size_t num_memory_controllers() const { return memory_controllers_.size(); }
    [[nodiscard]] size_t num_dma_engines() const { return dma_engines_.size(); }
    [[nodiscard]] size_t num_block_movers() const { return block_movers_.size(); }
    [[nodiscard]] size_t num_row_streamers() const { return row_streamers_.size(); }
    [[nodiscard]] size_t num_col_streamers() const { return col_streamers_.size(); }

    [[nodiscard]] CreditPool& l3_credits() { return l3_credits_; }
    [[nodiscard]] CreditPool& l2_credits() { return l2_credits_; }
    [[nodiscard]] TagCAM& l3_tag_cam() { return l3_tag_cam_; }
    [[nodiscard]] TagCAM& l2_tag_cam() { return l2_tag_cam_; }
    [[nodiscard]] TagCAM& compute_result_tag_cam() { return compute_result_tag_cam_; }

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
    TagCAM compute_result_tag_cam_;  ///< Tracks result tiles ready for DRAIN

    // Pending compute operations (tile + dependencies + state)
    struct PendingCompute {
        TileDescriptor tile;         ///< Result tile (C)
        std::vector<std::pair<TileID, size_t>> dependencies;          ///< (tile, required completed-feed count)
        std::vector<std::pair<TileID, size_t>> resident_dependencies; ///< (tile, required completed-compute count)
        std::unique_ptr<MatMulComputeSpec> matmul;
        std::unique_ptr<FunctionalComputeSpec> functional;
        Cycle schedule_cycle = 0;    ///< When scheduled
        Cycle complete_cycle = 0;    ///< When computation will complete (set when started)
        Cycle latency = 0;           ///< Computed latency (set when started)
        bool started = false;        ///< Has computation started?
    };
    std::vector<PendingCompute> pending_computes_;

    // Count scheduled and completed feed occurrences. Counts are required:
    // a boolean "ever fed" flag lets a reused tile start a later compute early.
    std::unordered_map<TileID, size_t, TileIDHash> scheduled_feed_counts_;
    std::unordered_map<TileID, size_t, TileIDHash> completed_feed_counts_;
    std::unordered_map<TileID, size_t, TileIDHash> scheduled_compute_counts_;
    std::unordered_map<TileID, size_t, TileIDHash> completed_compute_counts_;

    struct StoredPayload {
        TilePayload payload;
        std::vector<std::byte> bytes;
        uint32_t slot_id = 0;
        Cycle arrival_cycle = 0;
    };
    using PayloadStore = std::unordered_map<TileID, StoredPayload, TileIDHash>;
    PayloadStore dram_payloads_, l3_payloads_, l2_payloads_, l1_payloads_, compute_payloads_;
    bool functional_payloads_enabled_ = false;

    // Component processes
    std::vector<std::unique_ptr<MemoryControllerProcess>> memory_controllers_;
    std::vector<std::unique_ptr<DMAEngineProcess>> dma_engines_;
    std::vector<std::unique_ptr<BlockMoverProcess>> block_movers_;
    std::vector<std::unique_ptr<StreamerProcess>> row_streamers_;
    std::vector<std::unique_ptr<StreamerProcess>> col_streamers_;

    // Livelock detection
    std::unique_ptr<LivelockDetector> livelock_detector_;

    // Slot counter for compute results (instance-local, reset with executor)
    uint32_t next_compute_slot_ = 0;

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

    [[nodiscard]] bool dependencies_satisfied(const PendingCompute& pc) const;
    void execute_matmul(const PendingCompute& pc);
    void execute_functional(const PendingCompute& pc);
    void apply_payload_event(const TimingEvent& event);
    void write_payload(MemoryLevel level, const TileID& id, TilePayload payload,
                       uint32_t slot_id = 0);
    void copy_payload(MemoryLevel source, MemoryLevel destination, const TileID& id,
                      uint32_t slot_id);
    [[nodiscard]] const TilePayload* find_payload(const TileID& id) const;
    [[nodiscard]] PayloadStore& payload_store(MemoryLevel level);
    [[nodiscard]] const PayloadStore& payload_store(MemoryLevel level) const;
};

// ============================================================================
// Implementation
// ============================================================================

inline ConcurrentTimingExecutor::ConcurrentTimingExecutor(const Config& config)
    : config_(config),
      l3_credits_(config.l3_buffer_count),
      l2_credits_(config.l2_bank_count),
      l3_tag_cam_(config.l3_buffer_count),
      l2_tag_cam_(config.l2_bank_count),
      compute_result_tag_cam_(256) {  // 256 pending compute results max
    if (config_.partition_l3_credits) {
        l3_credits_.partition_equal();
    }
    if (config_.partition_l2_credits) {
        l2_credits_.partition_equal();
    }
    create_components();

    if (config_.enable_livelock_detection) {
        LivelockDetector::Config ld_config;
        ld_config.stall_threshold = config_.livelock_threshold;
        livelock_detector_ = std::make_unique<LivelockDetector>(ld_config);
    }
}

inline void ConcurrentTimingExecutor::create_components() {
    // ========================================================================
    // Step 1: Create Memory Controllers (DRAM access contention resource)
    // ========================================================================
    // Each MC models a single LPDDR5 channel with correct resource contention:
    // - Command bus: 1 command per cycle (shared across all banks)
    // - Bank state machines: Track open row per bank
    // - Data bus: Occupied during burst transfers
    for (size_t mc = 0; mc < config_.num_memory_controllers; ++mc) {
        MemoryControllerProcess::Config mc_config;
        mc_config.controller_id = static_cast<uint32_t>(mc);
        mc_config.num_banks = config_.mc_num_banks;
        mc_config.request_queue_depth = config_.mc_request_queue_depth;
        mc_config.bandwidth_gbps = config_.mc_bandwidth_gbps;
        mc_config.startup_latency = config_.mc_startup_latency;
        mc_config.t_cl = config_.mc_t_cl;
        mc_config.t_rcd = config_.mc_t_rcd;
        mc_config.t_rp = config_.mc_t_rp;
        mc_config.t_burst = config_.mc_t_burst;
        mc_config.clock_ghz = config_.clock_ghz;
        mc_config.name = mc_config.display_name();

        memory_controllers_.push_back(std::make_unique<MemoryControllerProcess>(mc_config));
    }

    // ========================================================================
    // Step 2: Create DMA Engines (CSP processes that use MCs)
    // ========================================================================
    // DMA engines are the programmable ISA-driven processes.
    // They handle L3 credit/tag management and use MCs for DRAM access.
    for (size_t dma = 0; dma < config_.num_dma_engines; ++dma) {
        DMAEngineProcess::Config dma_config;
        dma_config.engine_id = static_cast<uint32_t>(dma);
        dma_config.queue_depth = config_.dma_queue_depth;
        // The writeback reserve is a heuristic guard that per-matrix
        // partitioning supersedes structurally (the C partition IS the
        // writeback protection). A pool-clamped reserve can also exceed a
        // partition's share and block it permanently, so it is disabled in
        // partition mode.
        dma_config.l3_credit_reserve = config_.partition_l3_credits
            ? 0
            : Config::clamp_reserve(
                  config_.l3_writeback_credit_reserve, config_.l3_buffer_count);
        dma_config.name = dma_config.display_name();

        // Assign DMA to MC (round-robin if more DMAs than MCs)
        size_t mc_id = dma % memory_controllers_.size();

        dma_engines_.push_back(std::make_unique<DMAEngineProcess>(
            dma_config, *memory_controllers_[mc_id], l3_credits_, l3_tag_cam_));
    }

    // ========================================================================
    // Step 3: Create BlockMovers (L3 ↔ L2)
    // ========================================================================
    // BlockMovers handle L3 tile grid positions
    // Map to 2D grid: row = i / cols, col = i % cols
    for (size_t i = 0; i < config_.num_block_movers; ++i) {
        BlockMoverProcess::Config bm_config;
        bm_config.mover_id = static_cast<uint32_t>(100 + i);  // 100, 101, 102, ...
        bm_config.l3_tile_pos = GridPosition(
            static_cast<uint32_t>(i / config_.l3_tile_cols),
            static_cast<uint32_t>(i % config_.l3_tile_cols)
        );
        bm_config.bandwidth_gbps = config_.bm_bandwidth_gbps;
        bm_config.startup_latency = config_.bm_startup_latency;
        bm_config.clock_ghz = config_.clock_ghz;
        bm_config.priority_aging = config_.enable_priority_aging;
        // Drain reserve disabled in partition mode for the same reason as
        // the DMA writeback reserve: the C partition supersedes it, and a
        // pool-clamped reserve can exceed a partition's share
        bm_config.l2_credit_reserve = config_.partition_l2_credits
            ? 0
            : Config::clamp_reserve(
                  config_.l2_drain_credit_reserve, config_.l2_bank_count);
        bm_config.name = bm_config.display_name();

        block_movers_.push_back(std::make_unique<BlockMoverProcess>(
            bm_config, l3_tag_cam_, l3_credits_, l2_credits_, l2_tag_cam_));
    }

    // ========================================================================
    // Step 4: Create Row Streamers (West edge - for A matrix)
    // ========================================================================
    // Row streamers are positioned along the West edge of the compute grid
    for (size_t i = 0; i < config_.num_row_streamers; ++i) {
        StreamerProcess::Config str_config;
        str_config.streamer_id = static_cast<uint32_t>(200 + i);  // 200, 201, ...
        str_config.type = StreamerType::ROW_STREAMER;
        str_config.compute_tile_pos = GridPosition(
            static_cast<uint32_t>(i),  // Row index
            0                          // West edge (column 0)
        );
        str_config.bandwidth_gbps = config_.str_bandwidth_gbps;
        str_config.startup_latency = config_.str_startup_latency;
        str_config.clock_ghz = config_.clock_ghz;
        str_config.priority_aging = config_.enable_priority_aging;
        str_config.name = str_config.display_name();

        row_streamers_.push_back(std::make_unique<StreamerProcess>(
            str_config, l2_tag_cam_, l2_credits_, compute_result_tag_cam_));
    }

    // ========================================================================
    // Step 5: Create Column Streamers (North edge - for B matrix)
    // ========================================================================
    // Column streamers are positioned along the North edge of the compute grid
    for (size_t i = 0; i < config_.num_col_streamers; ++i) {
        StreamerProcess::Config str_config;
        str_config.streamer_id = static_cast<uint32_t>(210 + i);  // 210, 211, ...
        str_config.type = StreamerType::COL_STREAMER;
        str_config.compute_tile_pos = GridPosition(
            0,                          // North edge (row 0)
            static_cast<uint32_t>(i)    // Column index
        );
        str_config.bandwidth_gbps = config_.str_bandwidth_gbps;
        str_config.startup_latency = config_.str_startup_latency;
        str_config.clock_ghz = config_.clock_ghz;
        str_config.priority_aging = config_.enable_priority_aging;
        str_config.name = str_config.display_name();

        col_streamers_.push_back(std::make_unique<StreamerProcess>(
            str_config, l2_tag_cam_, l2_credits_, compute_result_tag_cam_));
    }
}

inline void ConcurrentTimingExecutor::schedule_load(const TileDescriptor& tile, int engine_id) {
    uint32_t dma = (engine_id >= 0)
        ? static_cast<uint32_t>(engine_id)
        : select_dma_engine(tile);
    dma_engines_[dma % dma_engines_.size()]->schedule_load(tile);
}

inline void ConcurrentTimingExecutor::schedule_store(const TileDescriptor& tile, int engine_id) {
    uint32_t dma = (engine_id >= 0)
        ? static_cast<uint32_t>(engine_id)
        : select_dma_engine(tile);
    dma_engines_[dma % dma_engines_.size()]->schedule_store(tile);
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
    ++scheduled_feed_counts_[tile.tile_id];
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

inline void ConcurrentTimingExecutor::schedule_compute(
    const TileDescriptor& tile, const TileID& dependency_tile) {
    schedule_compute(tile, std::vector<TileID>{dependency_tile});
}

inline void ConcurrentTimingExecutor::schedule_compute(
    const TileDescriptor& tile, std::vector<TileID> dependencies) {
    PendingCompute pc;
    pc.tile = tile;
    // Per-instance feed accounting: each dependency requires as many
    // completed feeds as have been scheduled for that tile so far, so a
    // feed consumed by an earlier output iteration cannot satisfy a later
    // iteration's compute.
    for (const auto& dep : dependencies) {
        const size_t required = std::max<size_t>(1, scheduled_feed_counts_[dep]);
        pc.dependencies.push_back({dep, required});
    }
    pc.schedule_cycle = current_cycle_;
    pc.complete_cycle = 0;  // Set when started
    pc.started = false;
    pending_computes_.push_back(std::move(pc));
}

inline void ConcurrentTimingExecutor::schedule_compute(
    const TileDescriptor& tile, std::vector<TileID> feed_dependencies,
    std::vector<TileID> resident_dependencies) {
    PendingCompute pc;
    pc.tile = tile;
    for (const auto& dep : feed_dependencies) {
        const size_t required = std::max<size_t>(1, scheduled_feed_counts_[dep]);
        pc.dependencies.push_back({dep, required});
    }
    // Resident inputs are gated on the producing compute having completed,
    // not on a fresh feed (the #66 completed-compute accounting)
    for (const auto& dep : resident_dependencies) {
        const size_t required = std::max<size_t>(1, scheduled_compute_counts_[dep]);
        pc.resident_dependencies.push_back({dep, required});
    }
    pc.schedule_cycle = current_cycle_;
    pc.complete_cycle = 0;
    pc.started = false;
    // This compute may itself be a resident source for a later compute
    ++scheduled_compute_counts_[tile.tile_id];
    pending_computes_.push_back(std::move(pc));
}

inline void ConcurrentTimingExecutor::schedule_matmul_compute(
    const TileDescriptor& tile, const MatMulComputeSpec& spec) {
    if (spec.a_tiles.empty() || spec.a_tiles.size() != spec.b_tiles.size()) {
        throw std::invalid_argument("Matmul compute requires paired, non-empty A/B tiles");
    }

    PendingCompute pc;
    pc.tile = tile;
    pc.schedule_cycle = current_cycle_;
    pc.complete_cycle = 0;
    pc.started = false;
    pc.matmul = std::make_unique<MatMulComputeSpec>(spec);

    auto is_resident = [&spec](const TileID& id) {
        return std::find(spec.resident_tiles.begin(), spec.resident_tiles.end(), id) !=
               spec.resident_tiles.end();
    };

    for (const auto& id : spec.a_tiles) {
        if (is_resident(id)) {
            const size_t required = std::max<size_t>(1, scheduled_compute_counts_[id]);
            pc.resident_dependencies.push_back({id, required});
            continue;
        }
        if (scheduled_feed_counts_[id] == 0) {
            throw std::invalid_argument("Matmul A tile has no scheduled feed: " + id.to_string());
        }
        pc.dependencies.push_back({id, scheduled_feed_counts_[id]});
    }
    for (const auto& id : spec.b_tiles) {
        if (is_resident(id)) {
            const size_t required = std::max<size_t>(1, scheduled_compute_counts_[id]);
            pc.resident_dependencies.push_back({id, required});
            continue;
        }
        if (scheduled_feed_counts_[id] == 0) {
            throw std::invalid_argument("Matmul B tile has no scheduled feed: " + id.to_string());
        }
        pc.dependencies.push_back({id, scheduled_feed_counts_[id]});
    }
    ++scheduled_compute_counts_[tile.tile_id];
    pending_computes_.push_back(std::move(pc));
}

inline void ConcurrentTimingExecutor::schedule_functional_compute(
    const TileDescriptor& tile, const FunctionalComputeSpec& spec) {
    if (spec.input_tiles.empty() || !spec.operation) {
        throw std::invalid_argument("Functional compute requires inputs and an operation");
    }
    PendingCompute pc;
    pc.tile = tile;
    pc.schedule_cycle = current_cycle_;
    pc.complete_cycle = 0;
    pc.started = false;
    pc.functional = std::make_unique<FunctionalComputeSpec>(spec);
    for (const auto& id : spec.input_tiles) {
        const bool resident = std::find(spec.resident_tiles.begin(), spec.resident_tiles.end(), id) !=
                              spec.resident_tiles.end();
        if (resident) {
            pc.resident_dependencies.push_back(
                {id, std::max<size_t>(1, scheduled_compute_counts_[id])});
        } else {
            if (scheduled_feed_counts_[id] == 0) {
                throw std::invalid_argument("Functional input has no scheduled feed: " +
                                            id.to_string());
            }
            pc.dependencies.push_back({id, scheduled_feed_counts_[id]});
        }
    }
    ++scheduled_compute_counts_[tile.tile_id];
    pending_computes_.push_back(std::move(pc));
}

inline void ConcurrentTimingExecutor::schedule_compute(const TileDescriptor& tile) {
    // Auto-generate dependency: last B tile for this output position
    // C[ti,tj] depends on B[k-1,tj] being fed (or A[ti,k-1])
    // We use B[*,tj,k_max-1] as dependency (approximation)
    TileID dep;
    dep.matrix = isa::MatrixID::B;
    dep.ti = 0;  // B tiles use tk for first dimension
    dep.tj = tile.tile_id.tj;
    dep.tk = tile.tile_id.tk;  // tk=0 for C tiles, so we use the C position

    schedule_compute(tile, dep);
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
            for (const auto& dma : dma_engines_) {
                metrics.tiles_dma_completed += dma->total_bytes_loaded() / 1024;
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
    // Step 0a: Check pending computes for dependency satisfaction and start them
    for (auto& pc : pending_computes_) {
        if (!pc.started) {
            // Compute starts only when every dependency's required
            // completed-feed (or completed-compute, for resident inputs)
            // count has been reached - per-instance accounting, not a
            // was-ever-fed set
            if (dependencies_satisfied(pc)) {
                // Latency scales with accumulation depth: the K-slice count
                // is dependencies/2 (one A and one B tile per K slice)
                size_t k_slices = pc.dependencies.size() >= 2
                    ? pc.dependencies.size() / 2 : 1;
                pc.latency = config_.compute_latency +
                    static_cast<Cycle>(k_slices - 1) * config_.compute_cycles_per_k_slice;
                pc.started = true;
                pc.complete_cycle = current_cycle_ + pc.latency;

                // Emit COMPUTE_START event
                TimingEvent event(EventType::COMPUTE_START, current_cycle_, 0,
                                  pc.tile.tile_id, "Compute");
                event.matrix_base_address = pc.tile.matrix_base_address;
                events_.push_back(event);
            }
        }
    }

    // Step 0b: Process completed computes (insert results into compute_result_tag_cam)
    auto it = pending_computes_.begin();
    while (it != pending_computes_.end()) {
        if (it->started && current_cycle_ >= it->complete_cycle) {
            if (it->matmul) {
                execute_matmul(*it);
            } else if (it->functional) {
                execute_functional(*it);
            }
            ++completed_compute_counts_[it->tile.tile_id];
            // Compute completed - result tile is now ready for DRAIN
            uint32_t slot = next_compute_slot_++;
            compute_result_tag_cam_.insert(it->tile.tile_id, slot, current_cycle_);

            // Emit COMPUTE_COMPLETE event
            TimingEvent event = TimingEvent::duration_event(
                EventType::COMPUTE_COMPLETE,
                it->complete_cycle - it->latency,
                it->latency,
                0, it->tile.tile_id, "Compute");
            event.matrix_base_address = it->tile.matrix_base_address;
            events_.push_back(event);

            it = pending_computes_.erase(it);
        } else {
            ++it;
        }
    }

    // ========================================================================
    // Tick all components (order matters for proper MC→DMA completion flow)
    // ========================================================================

    // 1. Tick Memory Controllers FIRST (process DRAM commands, generate completions)
    for (auto& mc : memory_controllers_) {
        auto mc_events = mc->tick(current_cycle_);
        for (const auto& event : mc_events) apply_payload_event(event);
        events_.insert(events_.end(), mc_events.begin(), mc_events.end());
    }

    // 2. Tick DMA Engines (poll MC completions, submit new requests)
    for (auto& dma : dma_engines_) {
        auto dma_events = dma->tick(current_cycle_);
        for (const auto& event : dma_events) apply_payload_event(event);
        events_.insert(events_.end(), dma_events.begin(), dma_events.end());
    }

    // 3. Tick BlockMovers
    for (auto& mover : block_movers_) {
        auto mover_events = mover->tick(current_cycle_);
        for (const auto& event : mover_events) apply_payload_event(event);
        events_.insert(events_.end(), mover_events.begin(), mover_events.end());
    }

    // 4. Tick Row Streamers
    for (auto& streamer : row_streamers_) {
        auto str_events = streamer->tick(current_cycle_);
        // Track tiles that have been fed to compute
        for (const auto& event : str_events) {
            apply_payload_event(event);
            if (event.type == EventType::TILE_FED_TO_COMPUTE) {
                ++completed_feed_counts_[event.tile_id];
            }
        }
        events_.insert(events_.end(), str_events.begin(), str_events.end());
    }

    // 5. Tick Column Streamers
    for (auto& streamer : col_streamers_) {
        auto str_events = streamer->tick(current_cycle_);
        // Track tiles that have been fed to compute
        for (const auto& event : str_events) {
            apply_payload_event(event);
            if (event.type == EventType::TILE_FED_TO_COMPUTE) {
                ++completed_feed_counts_[event.tile_id];
            }
        }
        events_.insert(events_.end(), str_events.begin(), str_events.end());
    }

    // Advance clock
    ++current_cycle_;

    return is_complete();
}

inline bool ConcurrentTimingExecutor::is_complete() const {
    // Complete when all queues are empty and no in-flight work

    // Check pending computes (started or waiting on dependencies)
    if (!pending_computes_.empty()) return false;

    // Check MCs (should be idle when DMA is complete)
    for (const auto& mc : memory_controllers_) {
        if (!mc->is_idle() || mc->has_pending_work()) return false;
    }

    // Check DMA engines
    for (const auto& dma : dma_engines_) {
        if (!dma->is_idle() || dma->has_pending_work()) return false;
    }

    // Check BlockMovers
    for (const auto& mover : block_movers_) {
        if (!mover->is_idle() || mover->has_pending_work()) return false;
    }

    // Check Streamers
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
    compute_result_tag_cam_.reset();
    pending_computes_.clear();
    scheduled_feed_counts_.clear();
    completed_feed_counts_.clear();
    scheduled_compute_counts_.clear();
    completed_compute_counts_.clear();

    for (auto& mc : memory_controllers_) {
        mc->reset();
    }
    for (auto& dma : dma_engines_) {
        dma->reset();
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

    next_compute_slot_ = 0;
}

inline bool ConcurrentTimingExecutor::dependencies_satisfied(const PendingCompute& pc) const {
    for (const auto& [tile_id, required_count] : pc.dependencies) {
        auto it = completed_feed_counts_.find(tile_id);
        size_t completed = it == completed_feed_counts_.end() ? 0 : it->second;
        if (completed < required_count) return false;
    }
    for (const auto& [tile_id, required_count] : pc.resident_dependencies) {
        auto it = completed_compute_counts_.find(tile_id);
        const size_t completed = it == completed_compute_counts_.end() ? 0 : it->second;
        if (completed < required_count) return false;
    }
    return true;
}

inline void ConcurrentTimingExecutor::execute_matmul(const PendingCompute& pc) {
    const auto& spec = *pc.matmul;
    const auto& first_a = tile_payload(spec.a_tiles.front());
    const auto& first_b = tile_payload(spec.b_tiles.front());
    const Size m = first_a.rows;
    const Size n = first_b.cols;

    if (spec.bias.size() != 0 && spec.bias.size() != n) {
        throw std::runtime_error("Matmul bias width does not match output tile");
    }

    TilePayload output;
    output.rows = m;
    output.cols = n;
    output.values.assign(static_cast<size_t>(m) * n, 0.0f);

    for (size_t tile_index = 0; tile_index < spec.a_tiles.size(); ++tile_index) {
        const auto& a = tile_payload(spec.a_tiles[tile_index]);
        const auto& b = tile_payload(spec.b_tiles[tile_index]);
        if (!a.valid() || !b.valid() || a.rows != m || b.cols != n || a.cols != b.rows) {
            throw std::runtime_error("Incompatible functional matmul tile payloads");
        }
        for (Size i = 0; i < m; ++i) {
            for (Size k = 0; k < a.cols; ++k) {
                const float av = a.values[static_cast<size_t>(i) * a.cols + k];
                for (Size j = 0; j < n; ++j) {
                    output.values[static_cast<size_t>(i) * n + j] +=
                        av * b.values[static_cast<size_t>(k) * n + j];
                }
            }
        }
    }

    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float& value = output.values[static_cast<size_t>(i) * n + j];
            if (!spec.bias.empty()) value += spec.bias[j];
            if (spec.activation == FunctionalActivation::RELU && value < 0.0f) value = 0.0f;
        }
    }
    write_payload(MemoryLevel::COMPUTE, pc.tile.tile_id, std::move(output),
                  next_compute_slot_);
}

inline void ConcurrentTimingExecutor::execute_functional(const PendingCompute& pc) {
    std::vector<TilePayload> inputs;
    inputs.reserve(pc.functional->input_tiles.size());
    for (const auto& id : pc.functional->input_tiles) inputs.push_back(tile_payload(id));
    TilePayload output = pc.functional->operation(inputs);
    if (!output.valid()) throw std::runtime_error("Functional operation returned invalid payload");
    write_payload(MemoryLevel::COMPUTE, pc.tile.tile_id, std::move(output), next_compute_slot_);
}

inline ConcurrentTimingExecutor::PayloadStore&
ConcurrentTimingExecutor::payload_store(MemoryLevel level) {
    switch (level) {
        case MemoryLevel::DRAM: return dram_payloads_;
        case MemoryLevel::L3: return l3_payloads_;
        case MemoryLevel::L2: return l2_payloads_;
        case MemoryLevel::L1: return l1_payloads_;
        case MemoryLevel::COMPUTE: return compute_payloads_;
    }
    throw std::invalid_argument("Unknown memory level");
}

inline const ConcurrentTimingExecutor::PayloadStore&
ConcurrentTimingExecutor::payload_store(MemoryLevel level) const {
    return const_cast<ConcurrentTimingExecutor*>(this)->payload_store(level);
}

inline void ConcurrentTimingExecutor::write_payload(MemoryLevel level, const TileID& id,
                                                     TilePayload payload, uint32_t slot_id) {
    if (!payload.valid()) throw std::invalid_argument("Invalid tile payload");
    StoredPayload stored;
    stored.bytes.resize(payload.values.size() * sizeof(float));
    std::memcpy(stored.bytes.data(), payload.values.data(), stored.bytes.size());
    stored.payload = std::move(payload);
    stored.slot_id = slot_id;
    stored.arrival_cycle = current_cycle_;
    payload_store(level)[id] = std::move(stored);
}

inline void ConcurrentTimingExecutor::copy_payload(MemoryLevel source, MemoryLevel destination,
                                                    const TileID& id, uint32_t slot_id) {
    const auto& sources = payload_store(source);
    auto it = sources.find(id);
    if (it == sources.end()) {
        if (!functional_payloads_enabled_) return;
        throw std::runtime_error(std::string("Payload transfer completed without source bytes at ") +
                                 to_string(source) + " for " + id.to_string());
    }
    StoredPayload copy = it->second;
    copy.slot_id = slot_id;
    copy.arrival_cycle = current_cycle_;
    payload_store(destination)[id] = std::move(copy);
}

inline const TilePayload* ConcurrentTimingExecutor::find_payload(const TileID& id) const {
    for (MemoryLevel level : {MemoryLevel::COMPUTE, MemoryLevel::L1, MemoryLevel::L2,
                              MemoryLevel::L3, MemoryLevel::DRAM}) {
        const auto& store = payload_store(level);
        auto it = store.find(id);
        if (it != store.end()) return &it->second.payload;
    }
    return nullptr;
}

inline void ConcurrentTimingExecutor::apply_payload_event(const TimingEvent& event) {
    switch (event.type) {
        case EventType::TILE_ARRIVED_L3:
            if (event.component_name.find("DMA") != std::string::npos) {
                copy_payload(MemoryLevel::DRAM, MemoryLevel::L3, event.tile_id, event.slot_id);
            }
            break;
        case EventType::BM_MOVE_COMPLETE:
            copy_payload(MemoryLevel::L3, MemoryLevel::L2, event.tile_id, event.slot_id);
            break;
        case EventType::TILE_FED_TO_COMPUTE:
            copy_payload(MemoryLevel::L2, MemoryLevel::L1, event.tile_id, event.component_id);
            copy_payload(MemoryLevel::L1, MemoryLevel::COMPUTE, event.tile_id, event.component_id);
            break;
        case EventType::TILE_DRAINED:
            copy_payload(MemoryLevel::COMPUTE, MemoryLevel::L1, event.tile_id, event.component_id);
            copy_payload(MemoryLevel::L1, MemoryLevel::L2, event.tile_id, event.slot_id);
            break;
        case EventType::BM_WRITEBACK_COMPLETE:
            copy_payload(MemoryLevel::L2, MemoryLevel::L3, event.tile_id, event.slot_id);
            break;
        case EventType::DMA_STORE_COMPLETE:
            copy_payload(MemoryLevel::L3, MemoryLevel::DRAM, event.tile_id, event.slot_id);
            break;
        case EventType::CREDIT_RELEASED:
            // Component processes update TagCAM ref-counts before emitting the
            // event. Retire bytes only when the final reference disappeared.
            if (!l3_tag_cam_.lookup(event.tile_id)) l3_payloads_.erase(event.tile_id);
            if (!l2_tag_cam_.lookup(event.tile_id)) l2_payloads_.erase(event.tile_id);
            break;
        default:
            break;
    }
}

inline ConcurrentTimingExecutor::Statistics ConcurrentTimingExecutor::get_statistics() const {
    Statistics stats;
    stats.total_cycles = current_cycle_;
    collect_statistics(stats);
    return stats;
}

inline void ConcurrentTimingExecutor::collect_statistics(Statistics& stats) const {
    // Aggregate from DMA Engines
    for (const auto& dma : dma_engines_) {
        stats.dma_credit_stalls += dma->stall_cycles_credit();
        stats.bytes_loaded += dma->total_bytes_loaded();
        stats.bytes_stored += dma->total_bytes_stored();
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
    for (const auto& dma : dma_engines_) {
        count += dma->total_bytes_loaded() / 1024;  // Approximate tile count
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
    // Strategy: Tile-affine assignment - all operations for the same tile go
    // to the same engine. Round-robin spreads a reused tile's operations
    // across engines, which allows concurrent duplicate transfers of the
    // same tile: each acquires a credit but the ref-counted TagCAM entry
    // releases only one, leaking credits until the pipeline wedges (#61).
    return static_cast<uint32_t>(
        TileIDHash{}(tile.tile_id) % dma_engines_.size());
}

inline uint32_t ConcurrentTimingExecutor::select_block_mover(const TileDescriptor& tile) const {
    // Tile-affine assignment (see select_dma_engine for rationale)
    return static_cast<uint32_t>(
        TileIDHash{}(tile.tile_id) % block_movers_.size());
}

inline uint32_t ConcurrentTimingExecutor::select_streamer(const TileDescriptor& tile, bool is_row) const {
    // Tile-affine assignment (see select_dma_engine for rationale)
    size_t n = is_row ? row_streamers_.size() : col_streamers_.size();
    return static_cast<uint32_t>(TileIDHash{}(tile.tile_id) % n);
}

inline void ConcurrentTimingExecutor::export_chrome_trace(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "{\"traceEvents\":[\n";

    // ========================================================================
    // Emit process and thread metadata first for human-readable trace display
    // Sort order follows dataflow: MC → DMA → BlockMover → Streamer (top to bottom)
    // ========================================================================

    // Process name with grid topology info
    file << "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"KPU CSP Executor ("
         << config_.num_memory_controllers << " MCs, "
         << config_.num_dma_engines << " DMAs, "
         << config_.l3_tile_rows << "x" << config_.l3_tile_cols << " L3 tiles, "
         << config_.compute_tile_rows << "x" << config_.compute_tile_cols << " CTs)\"}}";

    // Memory Controller threads
    for (size_t i = 0; i < memory_controllers_.size(); ++i) {
        const auto& mc = memory_controllers_[i];
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << i
             << ",\"args\":{\"name\":\"" << mc->name() << " (banks=" << config_.mc_num_banks << ")\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << i
             << ",\"args\":{\"sort_index\":" << i << "}}";
    }

    // DMA Engine threads
    for (size_t i = 0; i < dma_engines_.size(); ++i) {
        const auto& dma = dma_engines_[i];
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (50 + i)
             << ",\"args\":{\"name\":\"" << dma->name() << "\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (50 + i)
             << ",\"args\":{\"sort_index\":" << (5 + i) << "}}";
    }

    // BlockMover threads - use component names with L3(row,col):BM format
    for (size_t i = 0; i < block_movers_.size(); ++i) {
        const auto& mover = block_movers_[i];
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (100 + i)
             << ",\"args\":{\"name\":\"" << mover->name() << "\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (100 + i)
             << ",\"args\":{\"sort_index\":" << (10 + i) << "}}";
    }

    // Row Streamer threads - use component names with CT(row,col):RowSTR format
    for (size_t i = 0; i < row_streamers_.size(); ++i) {
        const auto& streamer = row_streamers_[i];
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (200 + i)
             << ",\"args\":{\"name\":\"" << streamer->name() << " - A matrix\"}}";
        file << ",\n";
        file << "{\"name\":\"thread_sort_index\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (200 + i)
             << ",\"args\":{\"sort_index\":" << (20 + i) << "}}";
    }

    // Column Streamer threads - use component names with CT(row,col):ColSTR format
    for (size_t i = 0; i < col_streamers_.size(); ++i) {
        const auto& streamer = col_streamers_[i];
        file << ",\n";
        file << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << (210 + i)
             << ",\"args\":{\"name\":\"" << streamer->name() << " - B matrix\"}}";
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
