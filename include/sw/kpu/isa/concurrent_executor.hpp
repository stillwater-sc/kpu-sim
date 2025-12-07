/**
 * @file concurrent_executor.hpp
 * @brief Concurrent execution model for Data Movement ISA
 *
 * The KPU has multiple hardware resources that execute concurrently:
 * - Multiple DMA engines (one per memory channel)
 * - Multiple BlockMovers (L3→L2)
 * - Multiple Streamers (L2→L1)
 * - Compute fabric (systolic array)
 *
 * This executor models the true concurrent nature of the architecture,
 * scheduling operations onto resources and tracking occupancy over time.
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/tile_layout.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <memory>

namespace sw::kpu::isa {

// ============================================================================
// Resource Types
// ============================================================================

/**
 * @brief Resource type enumeration
 */
enum class ResourceType {
    DMA_ENGINE,
    BLOCK_MOVER,
    STREAMER,
    COMPUTE_FABRIC
};

/**
 * @brief Resource identifier (type + index)
 */
struct ResourceId {
    ResourceType type;
    uint8_t index;

    bool operator==(const ResourceId& other) const {
        return type == other.type && index == other.index;
    }

    bool operator<(const ResourceId& other) const {
        if (type != other.type) return type < other.type;
        return index < other.index;
    }

    std::string to_string() const {
        std::string type_str;
        switch (type) {
            case ResourceType::DMA_ENGINE: type_str = "DMA"; break;
            case ResourceType::BLOCK_MOVER: type_str = "BM"; break;
            case ResourceType::STREAMER: type_str = "STR"; break;
            case ResourceType::COMPUTE_FABRIC: type_str = "COMP"; break;
        }
        return type_str + "[" + std::to_string(index) + "]";
    }
};

// ============================================================================
// Operation in Flight
// ============================================================================

/**
 * @brief Represents an operation scheduled on a resource
 */
struct ScheduledOp {
    uint32_t instruction_id;
    ResourceId resource;
    Cycle start_cycle;
    Cycle end_cycle;
    std::string label;
    MatrixID matrix;
    TileCoord tile;

    Cycle duration() const { return end_cycle - start_cycle; }
};

// ============================================================================
// Hardware Resource Model
// ============================================================================

/**
 * @brief Models a hardware resource with timing and occupancy
 *
 * Timing model: Each resource has a bus width (bytes per cycle) that determines
 * how many cycles a transfer takes. For example:
 * - DMA engine with 64-byte bus @ 250 MHz: 64 bytes per DMA cycle
 * - BlockMover with 64-byte bus @ 500 MHz: 64 bytes per BM cycle
 *
 * The cycle count is simply: ceil(transfer_bytes / bus_width_bytes)
 * This gives the number of cycles in the resource's clock domain.
 */
class HardwareResource {
public:
    ResourceId id;
    double bandwidth_gb_s;      // Bandwidth for this resource (for reporting)
    Size bus_width_bytes;       // Bytes transferred per cycle
    Cycle next_available_cycle; // When resource becomes free

    // Occupancy tracking
    std::vector<ScheduledOp> completed_ops;
    ScheduledOp* current_op;

    HardwareResource(ResourceType type, uint8_t index, double bandwidth = 16.0,
                     Size bus_width = 64)
        : id{type, index}, bandwidth_gb_s(bandwidth), bus_width_bytes(bus_width),
          next_available_cycle(0), current_op(nullptr) {}

    bool is_busy(Cycle at_cycle) const {
        return at_cycle < next_available_cycle;
    }

    Cycle schedule_op(Cycle earliest, Size bytes, uint32_t instr_id,
                      const std::string& label, MatrixID mat, TileCoord tile) {
        Cycle start = std::max(earliest, next_available_cycle);
        // Calculate cycles: ceiling division of bytes by bus width
        // This is the number of bus transactions needed
        Cycle cycles = (bytes + bus_width_bytes - 1) / bus_width_bytes;
        if (cycles == 0) cycles = 1;  // Minimum 1 cycle

        ScheduledOp op;
        op.instruction_id = instr_id;
        op.resource = id;
        op.start_cycle = start;
        op.end_cycle = start + cycles;
        op.label = label;
        op.matrix = mat;
        op.tile = tile;

        completed_ops.push_back(op);
        next_available_cycle = op.end_cycle;

        return op.end_cycle;
    }
};

// ============================================================================
// Memory Channel Model
// ============================================================================

/**
 * @brief Models a memory channel with its associated DMA engine
 *
 * Each memory channel has:
 * - One DMA engine for external memory transfers
 * - Associated bandwidth constraints (derived from bus width × clock)
 * - Queue of pending transfers
 */
struct MemoryChannel {
    uint8_t channel_id;
    double bandwidth_gb_s;
    Size bus_width_bytes;
    HardwareResource dma_engine;

    MemoryChannel(uint8_t id, double bw = 16.0, Size bus_width = 64)
        : channel_id(id), bandwidth_gb_s(bw), bus_width_bytes(bus_width),
          dma_engine(ResourceType::DMA_ENGINE, id, bw, bus_width) {}
};

// ============================================================================
// System Resource Configuration
// ============================================================================

/**
 * @brief Configuration for system resources
 *
 * Clock Domain Hierarchy:
 * - Compute fabric (ALUs):     2.0 GHz (500 ps cycle time)
 * - L1/L2/Streamer/BlockMover: 500 MHz (2 ns cycle time)
 * - L3/DMA engines:            250 MHz (4 ns cycle time)
 *
 * Bandwidth Analysis:
 * - Systolic 16x16 array ingress: 32 elements × 4 bytes × 2 GHz = 256 GB/s
 * - DMA engine: 64-byte burst per cycle @ 250 MHz = 16 GB/s per channel
 * - BlockMover: 64-byte per cycle @ 500 MHz = 32 GB/s per mover
 * - Streamer: 64-byte per cycle @ 500 MHz = 32 GB/s per streamer
 * - L2 banks: 8 banks × 32 GB/s = 256 GB/s aggregate (matches systolic demand)
 *
 * Memory Interface:
 * - DMA uses 512-bit (64-byte) bus to L3, burst-oriented
 * - Ring bus topology for contention-free tile movement
 */
struct ResourceConfig {
    uint8_t num_memory_channels = 4;    // DMA engines (one per memory channel)
    uint8_t num_block_movers = 4;       // L3→L2 movers
    uint8_t num_streamers = 4;          // L2→L1 streamers

    // Clock frequencies (MHz)
    double dma_clock_mhz = 250.0;           // L3/DMA domain
    double block_mover_clock_mhz = 500.0;   // L2 domain
    double streamer_clock_mhz = 500.0;      // L1/L2 domain
    double compute_clock_mhz = 2000.0;      // ALU domain

    // Bus widths (bytes per cycle)
    Size dma_bus_width_bytes = 64;          // 512-bit bus, 64-byte cache line
    Size block_mover_bus_width_bytes = 64;  // 512-bit internal bus
    Size streamer_bus_width_bytes = 64;     // 512-bit to L1 buffers

    // Derived bandwidths (GB/s) - computed from clock × bus width
    // DMA: 64 bytes × 250 MHz = 16 GB/s per channel
    // BM:  64 bytes × 500 MHz = 32 GB/s per mover
    // STR: 64 bytes × 500 MHz = 32 GB/s per streamer
    double dma_bandwidth_gb_s = 16.0;           // Per channel
    double block_mover_bandwidth_gb_s = 32.0;   // Per mover
    double streamer_bandwidth_gb_s = 32.0;      // Per streamer

    // Compute fabric
    Size systolic_size = 16;                // 16x16 systolic array
    double compute_throughput_gflops = 1024.0;  // 16×16×2×2GHz = 1024 GFLOPS
};

// ============================================================================
// Concurrent Executor
// ============================================================================

/**
 * @brief Executes Data Movement programs with true concurrency model
 *
 * This executor:
 * 1. Schedules operations onto available resources using a configurable tile layout
 * 2. Respects data dependencies
 * 3. Tracks resource occupancy over time
 * 4. Generates timeline visualizations
 *
 * The tile layout policy determines how tiles are mapped to memory channels,
 * which directly affects bandwidth utilization and potential conflicts.
 */
class ConcurrentExecutor {
public:
    /**
     * @brief Construct executor with resource config (uses MATRIX_PARTITIONED layout)
     */
    explicit ConcurrentExecutor(const ResourceConfig& config);

    /**
     * @brief Construct executor with explicit tile layout
     * @param config Hardware resource configuration
     * @param layout Tile layout policy (takes ownership)
     */
    ConcurrentExecutor(const ResourceConfig& config, std::unique_ptr<TileLayout> layout);

    /**
     * @brief Set the tile layout policy
     * @param layout New layout policy (takes ownership)
     */
    void set_tile_layout(std::unique_ptr<TileLayout> layout);

    /**
     * @brief Get the current layout policy
     */
    LayoutPolicy get_layout_policy() const;

    /**
     * @brief Execute a program and collect timing information
     * @param program The Data Movement program to execute
     * @return Total execution cycles
     */
    Cycle execute(const DMProgram& program);

    /**
     * @brief Get resource utilization statistics
     */
    struct UtilizationStats {
        double dma_utilization;
        double block_mover_utilization;
        double streamer_utilization;
        double compute_utilization;
        Cycle total_cycles;
        Cycle makespan;  // Wall-clock cycles from start to finish
    };
    UtilizationStats get_utilization() const;

    /**
     * @brief Generate ASCII timeline visualization
     * @param width Character width of timeline
     * @return Multi-line string showing resource occupancy
     */
    std::string generate_timeline(size_t width = 80) const;

    /**
     * @brief Generate detailed cycle-by-cycle report
     */
    std::string generate_cycle_report() const;

    /**
     * @brief Get all scheduled operations for analysis
     */
    const std::vector<ScheduledOp>& get_all_operations() const { return all_ops_; }

private:
    ResourceConfig config_;

    // Hardware resources
    std::vector<MemoryChannel> memory_channels_;
    std::vector<HardwareResource> block_movers_;
    std::vector<HardwareResource> streamers_;
    HardwareResource compute_fabric_;

    // Execution state
    std::vector<ScheduledOp> all_ops_;
    std::map<uint32_t, Cycle> instruction_completion_; // instr_id -> completion cycle
    Cycle current_cycle_;
    Cycle makespan_;

    // Barrier tracking
    Cycle last_barrier_cycle_;

    // Tile layout policy
    std::unique_ptr<TileLayout> tile_layout_;

    // Schedule an instruction onto appropriate resource
    void schedule_instruction(const DMInstruction& instr);

    // Find least-loaded resource of given type
    HardwareResource* find_available_resource(ResourceType type, Cycle at_cycle);

    // Calculate transfer size from instruction
    Size get_transfer_size(const DMInstruction& instr) const;

    // Get dependency completion cycle
    Cycle get_dependency_cycle(const DMInstruction& instr) const;

    // Resource allocation using tile layout
    uint8_t select_dma_channel(MatrixID matrix, TileCoord tile) const;
    uint8_t select_block_mover(MatrixID matrix, TileCoord tile) const;
    uint8_t select_streamer(MatrixID matrix, TileCoord tile) const;

    // Create default layout config from program dimensions
    void initialize_layout_for_program(const DMProgram& program);
};

// ============================================================================
// Timeline Formatter
// ============================================================================

/**
 * @brief Formats execution timeline for display
 */
class TimelineFormatter {
public:
    /**
     * @brief Generate ASCII Gantt chart
     */
    static std::string format_gantt(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle total_cycles,
        size_t width = 80);

    /**
     * @brief Generate resource occupancy table
     */
    static std::string format_occupancy_table(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle total_cycles);

    /**
     * @brief Generate cycle-by-cycle activity view
     */
    static std::string format_cycle_view(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle start_cycle,
        Cycle end_cycle);
};

} // namespace sw::kpu::isa
