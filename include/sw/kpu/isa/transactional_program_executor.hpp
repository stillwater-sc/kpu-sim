/**
 * @file transactional_program_executor.hpp
 * @brief Transactional executor combining behavioral data movement with timing overlay
 *
 * The TransactionalProgramExecutor provides:
 * 1. Functional correctness via BehavioralProgramExecutor (actual data movement)
 * 2. Timing accuracy via analytical timing models (cycle counting)
 * 3. Visualization via Chrome Trace output (Perfetto-compatible)
 *
 * Architecture:
 *
 *   TransactionalProgramExecutor
 *        │
 *        ├── BehavioralProgramExecutor  (for actual data)
 *        │
 *        └── TimingModel               (for cycle counts)
 *             ├── DMA: bytes / bandwidth = cycles
 *             ├── BM:  block_size / bus_width = cycles
 *             ├── STR: elements / bus_width = cycles
 *             └── Compute: 2*M*N*K / throughput = cycles
 *
 * Each instruction is dispatched behaviorally (moving real data) and then
 * timed analytically (computing when resources become available). The result
 * is correct numerical output AND an accurate execution timeline.
 */

#pragma once

#include <sw/kpu/isa/behavioral_program_executor.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <sw/kpu/isa/data_movement_isa.hpp>

#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sw::kpu::isa {

// ============================================================================
// Timing Event for Chrome Trace
// ============================================================================

/**
 * @brief A timing event for Chrome trace visualization
 *
 * Events follow the Chrome Trace format:
 * - name: event name (e.g., "DMA_LOAD A[0,0]")
 * - cat: category (e.g., "dma", "bm", "str", "compute")
 * - ph: phase ('B' for begin, 'E' for end, 'X' for complete)
 * - ts: timestamp in microseconds
 * - dur: duration in microseconds (for 'X' phase)
 * - pid: process ID (we use 1)
 * - tid: thread ID (we use resource type * 100 + resource index)
 */
struct TimingEvent {
    std::string name;
    std::string category;
    char phase;  // 'B', 'E', or 'X'
    double timestamp_us;
    double duration_us;
    int process_id;
    int thread_id;
    MatrixID matrix;
    TileCoord tile;
};

// ============================================================================
// Transactional Timing Model Configuration
// ============================================================================

/**
 * @brief Configuration for timing model
 *
 * These parameters determine cycle costs for each operation type.
 * Derived from ResourceConfig but expressed in terms the timing
 * model needs: cycles per operation based on transfer sizes.
 */
struct TimingConfig {
    // Reference clock for time conversion (MHz)
    double reference_clock_mhz = 1000.0;

    // DMA timing (L3 domain)
    double dma_clock_mhz = 250.0;
    Size dma_bus_width_bytes = 64;
    Cycle dma_startup_latency = 10;  // Cycles to initiate transfer

    // BlockMover timing (L2 domain)
    double block_mover_clock_mhz = 500.0;
    Size block_mover_bus_width_bytes = 64;
    Cycle block_mover_startup_latency = 4;

    // Streamer timing (L1 domain)
    double streamer_clock_mhz = 500.0;
    Size streamer_bus_width_bytes = 64;
    Cycle streamer_startup_latency = 2;

    // Compute timing
    double compute_clock_mhz = 2000.0;
    Size systolic_size = 16;  // 16x16 array
    // One systolic pass: array_dim cycles to fill + array_dim to drain
    // For 16x16: 16 cycles to fill west edge, then results flow

    // Construct from ResourceConfig
    static TimingConfig from_resource_config(const ResourceConfig& rc) {
        TimingConfig tc;
        tc.dma_clock_mhz = rc.dma_clock_mhz;
        tc.dma_bus_width_bytes = rc.dma_bus_width_bytes;
        tc.block_mover_clock_mhz = rc.block_mover_clock_mhz;
        tc.block_mover_bus_width_bytes = rc.block_mover_bus_width_bytes;
        tc.streamer_clock_mhz = rc.streamer_clock_mhz;
        tc.streamer_bus_width_bytes = rc.streamer_bus_width_bytes;
        tc.compute_clock_mhz = rc.compute_clock_mhz;
        tc.systolic_size = rc.systolic_size;
        return tc;
    }
};

// ============================================================================
// Resource Timeline
// ============================================================================

/**
 * @brief Tracks when each resource becomes available
 */
class ResourceTimeline {
public:
    explicit ResourceTimeline(uint8_t num_dma = 4, uint8_t num_bm = 4,
                               uint8_t num_str = 4);

    // Schedule an operation on a resource, returns end cycle
    Cycle schedule_dma(uint8_t channel, Cycle earliest, Cycle duration);
    Cycle schedule_block_mover(uint8_t id, Cycle earliest, Cycle duration);
    Cycle schedule_streamer(uint8_t id, Cycle earliest, Cycle duration);
    Cycle schedule_compute(Cycle earliest, Cycle duration);

    // Get next available time for a resource
    Cycle dma_available(uint8_t channel) const;
    Cycle block_mover_available(uint8_t id) const;
    Cycle streamer_available(uint8_t id) const;
    Cycle compute_available() const;

    // Get the makespan (when everything finishes)
    Cycle makespan() const;

    void reset();

private:
    std::vector<Cycle> dma_available_;
    std::vector<Cycle> bm_available_;
    std::vector<Cycle> str_available_;
    Cycle compute_available_ = 0;
    Cycle makespan_ = 0;
};

// ============================================================================
// Transactional Program Executor
// ============================================================================

/**
 * @brief Executor combining behavioral correctness with transactional timing
 *
 * For each instruction:
 * 1. Dispatch to BehavioralProgramExecutor (moves actual data)
 * 2. Compute timing cost based on transfer size and bandwidth
 * 3. Update resource timeline
 * 4. Record timing event for visualization
 */
class TransactionalProgramExecutor {
public:
    /**
     * @brief Construct with hardware context and optional timing config
     */
    explicit TransactionalProgramExecutor(
        BehavioralProgramExecutor::HardwareContext hw,
        const TimingConfig& timing = TimingConfig());

    /**
     * @brief Load a DMProgram for execution
     */
    void load_program(const DMProgram& program,
                      Address a_base, Address b_base, Address c_base);

    /**
     * @brief Execute the loaded program to completion
     * @return True if HALT reached
     */
    bool run();

    /**
     * @brief Get behavioral execution statistics
     */
    const BehavioralProgramExecutor::Statistics& behavioral_stats() const {
        return behavioral_.statistics();
    }

    /**
     * @brief Get timing statistics
     */
    struct TimingStats {
        Cycle total_cycles = 0;
        Cycle dma_cycles = 0;
        Cycle block_mover_cycles = 0;
        Cycle streamer_cycles = 0;
        Cycle compute_cycles = 0;
        double dma_utilization = 0.0;
        double block_mover_utilization = 0.0;
        double streamer_utilization = 0.0;
        double compute_utilization = 0.0;
    };
    TimingStats get_timing_stats() const;

    /**
     * @brief Get all timing events for visualization
     */
    const std::vector<TimingEvent>& timing_events() const { return events_; }

    /**
     * @brief Export timing events to Chrome Trace JSON format
     * @param filename Output file path (should end in .json)
     */
    void export_chrome_trace(const std::string& filename) const;

    /**
     * @brief Generate ASCII timeline visualization
     * @param width Character width of timeline
     */
    std::string generate_timeline(size_t width = 80) const;

    /**
     * @brief Access external memory (for reading results)
     */
    ExternalMemory& memory() { return behavioral_.memory(); }

    /**
     * @brief Get the makespan in cycles
     */
    Cycle makespan() const { return timeline_.makespan(); }

    /**
     * @brief Convert cycles to microseconds using reference clock
     */
    double cycles_to_us(Cycle cycles) const {
        return static_cast<double>(cycles) / timing_.reference_clock_mhz;
    }

private:
    BehavioralProgramExecutor behavioral_;
    TimingConfig timing_;
    ResourceTimeline timeline_;
    std::vector<TimingEvent> events_;

    const DMProgram* program_ = nullptr;

    // Tile dimension tracking for cycle computation
    Size tile_rows_ = 16;
    Size tile_cols_ = 16;
    Size element_size_ = sizeof(float);

    // Current cycle tracking
    Cycle current_cycle_ = 0;

    // Barrier tracking - all ops before barrier must complete
    Cycle last_barrier_cycle_ = 0;

    // Cumulative timing stats
    Cycle total_dma_cycles_ = 0;
    Cycle total_bm_cycles_ = 0;
    Cycle total_str_cycles_ = 0;
    Cycle total_compute_cycles_ = 0;

    // Dispatch with timing
    void dispatch_with_timing(const DMInstruction& instr);

    // Timing calculation helpers
    Cycle compute_dma_cycles(Size bytes) const;
    Cycle compute_block_mover_cycles(Size bytes) const;
    Cycle compute_streamer_cycles(Size bytes) const;
    Cycle compute_matmul_cycles(Size m, Size n, Size k) const;

    // Resource selection (round-robin or based on tile layout)
    uint8_t select_dma_channel(MatrixID matrix, const TileCoord& tile) const;
    uint8_t select_block_mover(MatrixID matrix, const TileCoord& tile) const;
    uint8_t select_streamer(MatrixID matrix, const TileCoord& tile) const;

    // Event recording
    void record_event(const std::string& name, const std::string& category,
                     Cycle start, Cycle end, int resource_id,
                     MatrixID matrix, TileCoord tile);

    // Get tile size from instruction operands
    Size get_tile_bytes(const DMInstruction& instr) const;
};

} // namespace sw::kpu::isa
