#pragma once
// Resource Statistics and Performance Counters for KPU simulator
// Provides observability into resource utilization and operational status

#include <cstdint>
#include <cstddef>
#include <string>
#include <atomic>

#include <sw/concepts.hpp>
#include <sw/kpu/resource_handle.hpp>

namespace sw::kpu {

/**
 * @brief Operational state of a resource
 */
enum class ResourceState : uint8_t {
    UNINITIALIZED = 0,  // Resource not yet configured
    IDLE = 1,           // Ready, not processing
    BUSY = 2,           // Currently processing an operation
    STALLED = 3,        // Waiting on dependency
    ERROR = 4,          // Error state, needs reset
    DISABLED = 5        // Administratively disabled
};

inline const char* resource_state_name(ResourceState state) {
    switch (state) {
        case ResourceState::UNINITIALIZED: return "uninitialized";
        case ResourceState::IDLE: return "idle";
        case ResourceState::BUSY: return "busy";
        case ResourceState::STALLED: return "stalled";
        case ResourceState::ERROR: return "error";
        case ResourceState::DISABLED: return "disabled";
        default: return "unknown";
    }
}

/**
 * @brief Statistics for memory resources
 */
struct MemoryResourceStats {
    // Capacity and utilization
    Size capacity_bytes;            // Total capacity
    Size allocated_bytes;           // Currently allocated
    Size peak_allocated_bytes;      // High watermark
    Size available_bytes;           // Free space

    // Access statistics
    uint64_t read_count;            // Number of read operations
    uint64_t write_count;           // Number of write operations
    uint64_t bytes_read;            // Total bytes read
    uint64_t bytes_written;         // Total bytes written

    // Performance metrics
    uint64_t read_cycles;           // Cycles spent reading
    uint64_t write_cycles;          // Cycles spent writing
    uint64_t stall_cycles;          // Cycles stalled waiting

    MemoryResourceStats()
        : capacity_bytes(0), allocated_bytes(0), peak_allocated_bytes(0),
          available_bytes(0), read_count(0), write_count(0),
          bytes_read(0), bytes_written(0), read_cycles(0),
          write_cycles(0), stall_cycles(0) {}

    // Derived metrics
    double utilization_percent() const {
        return capacity_bytes > 0 ?
            100.0 * allocated_bytes / capacity_bytes : 0.0;
    }

    double read_bandwidth_gb_s(double clock_ghz) const {
        return read_cycles > 0 ?
            (bytes_read / 1e9) / (read_cycles / (clock_ghz * 1e9)) : 0.0;
    }

    double write_bandwidth_gb_s(double clock_ghz) const {
        return write_cycles > 0 ?
            (bytes_written / 1e9) / (write_cycles / (clock_ghz * 1e9)) : 0.0;
    }

    void reset_counters() {
        read_count = 0;
        write_count = 0;
        bytes_read = 0;
        bytes_written = 0;
        read_cycles = 0;
        write_cycles = 0;
        stall_cycles = 0;
        // Note: capacity/allocated/peak are NOT reset
    }
};

/**
 * @brief Statistics for compute resources
 */
struct ComputeResourceStats {
    // Operation counts
    uint64_t matmul_count;          // Matrix multiply operations
    uint64_t elementwise_count;     // Elementwise operations
    uint64_t total_ops;             // Total operations

    // Compute metrics
    uint64_t total_flops;           // Floating-point operations performed
    uint64_t compute_cycles;        // Cycles actively computing
    uint64_t idle_cycles;           // Cycles idle
    uint64_t stall_cycles;          // Cycles stalled on data

    // Dimension statistics (for analysis)
    Size max_M, max_N, max_K;       // Maximum dimensions seen

    ComputeResourceStats()
        : matmul_count(0), elementwise_count(0), total_ops(0),
          total_flops(0), compute_cycles(0), idle_cycles(0),
          stall_cycles(0), max_M(0), max_N(0), max_K(0) {}

    // Derived metrics
    double utilization_percent() const {
        uint64_t total = compute_cycles + idle_cycles + stall_cycles;
        return total > 0 ? 100.0 * compute_cycles / total : 0.0;
    }

    double flops_rate(double clock_ghz) const {
        return compute_cycles > 0 ?
            (total_flops * clock_ghz * 1e9) / compute_cycles : 0.0;
    }

    void reset_counters() {
        matmul_count = 0;
        elementwise_count = 0;
        total_ops = 0;
        total_flops = 0;
        compute_cycles = 0;
        idle_cycles = 0;
        stall_cycles = 0;
        max_M = 0;
        max_N = 0;
        max_K = 0;
    }
};

/**
 * @brief Statistics for data movement resources (DMA, BlockMover, Streamer)
 */
struct DataMovementStats {
    // Transfer counts
    uint64_t transfer_count;        // Total transfers
    uint64_t bytes_transferred;     // Total bytes moved

    // Queue statistics
    Size current_queue_depth;       // Current pending transfers
    Size max_queue_depth;           // High watermark
    uint64_t queue_full_count;      // Times queue was full

    // Timing
    uint64_t active_cycles;         // Cycles transferring
    uint64_t idle_cycles;           // Cycles idle
    uint64_t stall_cycles;          // Cycles stalled

    // Latency tracking
    uint64_t total_latency_cycles;  // Sum of all transfer latencies
    uint64_t min_latency_cycles;    // Minimum observed latency
    uint64_t max_latency_cycles;    // Maximum observed latency

    DataMovementStats()
        : transfer_count(0), bytes_transferred(0),
          current_queue_depth(0), max_queue_depth(0), queue_full_count(0),
          active_cycles(0), idle_cycles(0), stall_cycles(0),
          total_latency_cycles(0), min_latency_cycles(UINT64_MAX),
          max_latency_cycles(0) {}

    // Derived metrics
    double utilization_percent() const {
        uint64_t total = active_cycles + idle_cycles + stall_cycles;
        return total > 0 ? 100.0 * active_cycles / total : 0.0;
    }

    double bandwidth_gb_s(double clock_ghz) const {
        return active_cycles > 0 ?
            (bytes_transferred / 1e9) / (active_cycles / (clock_ghz * 1e9)) : 0.0;
    }

    double avg_latency_cycles() const {
        return transfer_count > 0 ?
            static_cast<double>(total_latency_cycles) / transfer_count : 0.0;
    }

    void record_transfer(Size bytes, uint64_t latency_cycles) {
        transfer_count++;
        bytes_transferred += bytes;
        total_latency_cycles += latency_cycles;
        if (latency_cycles < min_latency_cycles) min_latency_cycles = latency_cycles;
        if (latency_cycles > max_latency_cycles) max_latency_cycles = latency_cycles;
    }

    void reset_counters() {
        transfer_count = 0;
        bytes_transferred = 0;
        current_queue_depth = 0;
        max_queue_depth = 0;
        queue_full_count = 0;
        active_cycles = 0;
        idle_cycles = 0;
        stall_cycles = 0;
        total_latency_cycles = 0;
        min_latency_cycles = UINT64_MAX;
        max_latency_cycles = 0;
    }
};

/**
 * @brief Unified resource status combining state and type-specific stats
 */
struct ResourceStatus {
    ResourceHandle handle;          // Which resource
    ResourceState state;            // Current operational state
    std::string error_message;      // If state == ERROR

    // Type-specific statistics (only one is valid based on handle.type)
    MemoryResourceStats memory_stats;
    ComputeResourceStats compute_stats;
    DataMovementStats data_movement_stats;

    ResourceStatus() : state(ResourceState::UNINITIALIZED) {}

    bool is_healthy() const {
        return state != ResourceState::ERROR &&
               state != ResourceState::UNINITIALIZED;
    }

    bool is_available() const {
        return state == ResourceState::IDLE;
    }
};

/**
 * @brief System-wide statistics aggregated across all resources
 */
struct SystemStats {
    // Cycle count
    uint64_t current_cycle;

    // Aggregate memory stats
    Size total_memory_capacity;
    Size total_memory_allocated;
    uint64_t total_memory_read_bytes;
    uint64_t total_memory_write_bytes;

    // Aggregate compute stats
    uint64_t total_compute_ops;
    uint64_t total_flops;

    // Aggregate data movement stats
    uint64_t total_transfers;
    uint64_t total_bytes_moved;

    SystemStats()
        : current_cycle(0), total_memory_capacity(0), total_memory_allocated(0),
          total_memory_read_bytes(0), total_memory_write_bytes(0),
          total_compute_ops(0), total_flops(0),
          total_transfers(0), total_bytes_moved(0) {}
};

} // namespace sw::kpu
