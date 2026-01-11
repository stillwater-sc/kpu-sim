#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <queue>
#include <array>
#include <mutex>
#include <atomic>
#include <unordered_map>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>

namespace sw::kpu {

// Forward declarations
class BlockMover;
class Streamer;

/**
 * StorageScheduler: Autonomous multi-bank storage engine that executes scheduled command sequences
 *
 * This component provides explicitly managed storage with autonomous command execution for:
 * - Coordinated data movement between storage hierarchy levels
 * - Pipelined fetch/store operations with dependency management
 * - Integration with upstream/downstream storage components via DMA/streaming
 *
 * The scheduler supports both autonomous operation (scheduled commands) and direct access
 * for different use cases within the memory hierarchy.
 *
 */
class KPU_API StorageScheduler {
public:
    // Storage Operation Types
    enum class StorageOperation {
        FETCH_UPSTREAM,      // Fetch data from higher-level storage (e.g., main memory)
        FETCH_DOWNSTREAM,    // Fetch data from lower-level storage (e.g., L1 cache)
        WRITEBACK_UPSTREAM,  // Write data back to higher-level storage
        WRITEBACK_DOWNSTREAM,// Write data back to lower-level storage
        YIELD,               // Allow external access to bank data
        BARRIER              // Synchronization barrier between operations
    };

    // Memory Bank Access Patterns for Storage Operations
    enum class AccessPattern {
        SEQUENTIAL,    // Sequential access within bank
        STRIDED,       // Strided access pattern
        RANDOM,        // Random access (worst case for caching)
        BROADCAST      // One-to-many distribution
    };

    // Memory Bank Configuration
    struct BankConfig {
        Size bank_size_kb;          // Size per bank in KB
        Size cache_line_size;       // Cache line size (typically 64 bytes)
        Size num_ports;             // Number of concurrent access ports
        AccessPattern access_pattern; // Expected access pattern for optimization
        bool enable_prefetch;       // Enable hardware prefetching
    };

    // Storage Scheduling Command
    struct StorageCommand {
        StorageOperation operation;
        size_t bank_id;

        // Memory addressing
        Address source_addr;        // Source address (for PREFETCH/WRITEBACK)
        Address dest_addr;          // Destination address
        Size transfer_size;         // Size in bytes

        // Orchestration control
        size_t sequence_id;         // For ordering dependencies
        std::vector<size_t> dependencies; // Commands that must complete first

        // Integration with data movers
        size_t block_mover_id;      // Which BlockMover to use (-1 if none)
        size_t streamer_id;         // Which Streamer to use (-1 if none)

        // Completion notification
        std::function<void(const StorageCommand&)> completion_callback;
    };

    // Per-bank state tracking
    struct BankState {
        std::vector<std::uint8_t> data;     // Bank memory storage
        Size capacity;                      // Total capacity in bytes
        Size current_occupancy;             // Current data occupancy

        // Access tracking
        std::atomic<bool> is_reading;       // Currently serving read requests
        std::atomic<bool> is_writing;       // Currently serving write requests

        // Storage operation tracking
        StorageOperation current_operation;
        size_t active_sequence_id;

        // Performance counters
        std::atomic<size_t> read_accesses;
        std::atomic<size_t> write_accesses;
        std::atomic<size_t> cache_hits;
        std::atomic<size_t> cache_misses;
    };

private:
    // Configuration
    size_t num_banks;
    std::vector<BankConfig> bank_configs;
    std::vector<std::unique_ptr<BankState>> bank_states;
    size_t scheduler_id;

    // Storage command scheduling
    std::queue<StorageCommand> command_queue;
    std::unordered_map<size_t, StorageCommand> active_commands;  // sequence_id -> command
    std::unordered_map<size_t, std::vector<size_t>> dependency_graph; // sequence_id -> dependents

    // Thread synchronization (for thread-safe operation)
    mutable std::mutex bank_mutex;
    mutable std::mutex command_mutex;

    // Integration with data movers
    std::vector<BlockMover*> block_movers;
    std::vector<Streamer*> streamers;

    // Sequence ID generator
    size_t next_sequence_id;

    // Internal scheduling methods
    bool can_execute_command(const StorageCommand& cmd) const;
    void execute_fetch_upstream_command(const StorageCommand& cmd);
    void execute_fetch_downstream_command(const StorageCommand& cmd);
    void execute_writeback_upstream_command(const StorageCommand& cmd);
    void execute_writeback_downstream_command(const StorageCommand& cmd);
    void execute_yield_command(const StorageCommand& cmd);
    void execute_barrier_command(const StorageCommand& cmd);
    void complete_command(const StorageCommand& cmd);
    void update_dependencies(size_t completed_sequence_id);

    // Bank management
    bool is_bank_available(size_t bank_id, StorageOperation operation) const;
    void transition_bank_operation(size_t bank_id, StorageOperation new_operation, size_t sequence_id);

    // Address validation and mapping
    bool validate_bank_access(size_t bank_id, Address addr, Size size) const;
    Address map_to_bank_address(size_t bank_id, Address global_addr) const;

public:
    explicit StorageScheduler(size_t scheduler_id, size_t num_banks = 4,
                   const BankConfig& default_config = {64, 64, 2, AccessPattern::SEQUENTIAL, true});
    ~StorageScheduler() = default;

    // Custom copy/move for vector compatibility
    StorageScheduler(const StorageScheduler& other);
    StorageScheduler& operator=(const StorageScheduler& other);
    StorageScheduler(StorageScheduler&&) = delete;
    StorageScheduler& operator=(StorageScheduler&&) = delete;

    // Configuration and initialization
    void configure_bank(size_t bank_id, const BankConfig& config);
    void register_block_mover(BlockMover* mover);
    void register_streamer(Streamer* streamer);

    // Direct access interface (bypass scheduling)
    void direct_read(size_t bank_id, Address addr, void* data, Size size);
    void direct_write(size_t bank_id, Address addr, const void* data, Size size);
    bool is_ready(size_t bank_id) const;

    // Autonomous storage scheduling interface
    void schedule_operation(const StorageCommand& cmd);
    bool execute_pending_operations();  // Process one cycle of scheduled operations
    size_t get_pending_operations() const;

    // Advanced storage patterns
    void schedule_double_buffer(size_t bank_a, size_t bank_b,
                               Address src_addr, Size transfer_size);
    void schedule_pipeline_stage(size_t input_bank, size_t output_bank,
                                const std::function<void()>& yield_func);

    // Status and performance monitoring
    bool is_busy() const;
    bool is_bank_busy(size_t bank_id) const;
    StorageOperation get_bank_operation(size_t bank_id) const;

    // Performance metrics
    struct PerformanceMetrics {
        size_t total_read_accesses;
        size_t total_write_accesses;
        size_t total_cache_hits;
        size_t total_cache_misses;
        double average_bank_utilization;
        size_t completed_storage_operations;
    };
    PerformanceMetrics get_performance_metrics() const;

    // Configuration queries
    size_t get_scheduler_id() const { return scheduler_id; }
    size_t get_num_banks() const { return num_banks; }
    Size get_bank_capacity(size_t bank_id) const;
    Size get_bank_occupancy(size_t bank_id) const;

    // Reset and cleanup
    void reset();
    void flush_all_banks();
    void abort_pending_operations();
};

// Helper class for storage workflow construction
class KPU_API StorageWorkflowBuilder {
private:
    std::vector<StorageScheduler::StorageCommand> commands;
    size_t next_sequence_id;

public:
    StorageWorkflowBuilder() : next_sequence_id(0) {}

    // Workflow construction methods
    StorageWorkflowBuilder& fetch_upstream(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    StorageWorkflowBuilder& fetch_downstream(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    StorageWorkflowBuilder& writeback_upstream(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    StorageWorkflowBuilder& writeback_downstream(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    StorageWorkflowBuilder& yield(size_t bank_id, const std::function<void()>& yield_func);
    StorageWorkflowBuilder& barrier();
    StorageWorkflowBuilder& depend_on(size_t dependency_sequence_id);

    // Build and execute
    std::vector<StorageScheduler::StorageCommand> build();
    void execute_on(StorageScheduler& scheduler);
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif