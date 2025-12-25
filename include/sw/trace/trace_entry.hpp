#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <optional>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::trace {

// Fundamental time unit for the simulator
using CycleCount = uint64_t;

// Component types in the KPU architecture
enum class ComponentType : uint8_t {
    // Host system components
    HOST_MEMORY = 0,           // Host DDR/system memory
    HOST_CPU = 1,              // Host CPU/controller
    PCIE_BUS = 2,              // PCIe interconnect (shared resource)

    // KPU data movement components
    DMA_ENGINE = 3,            // PCIe bus master for host<->KPU transfers
    BLOCK_MOVER = 4,           // L3->L2 block transfers
    STREAMER = 5,              // L2<->L1 streaming transfers

    // KPU memory hierarchy
    KPU_MEMORY = 6,            // KPU main memory banks (GDDR6)
    L3_TILE = 7,               // L3 cache tiles
    L2_BANK = 8,               // L2 cache banks
    L1 = 9,                    // L1 streaming buffers (compute fabric)

    // Memory controller components
    PAGE_BUFFER = 12,           // Page buffers (memory controller aggregation)

    // KPU compute components
    COMPUTE_FABRIC = 10,       // Compute orchestrator
    SYSTOLIC_ARRAY = 11,       // Systolic array compute engine

    // System orchestration
    STORAGE_SCHEDULER = 20,
    MEMORY_ORCHESTRATOR = 21,

    UNKNOWN = 255
};

// Transaction types across different components
enum class TransactionType : uint8_t {
    // Data movement transactions
    READ = 0,
    WRITE = 1,
    TRANSFER = 2,
    COPY = 3,

    // Compute transactions
    COMPUTE = 10,
    MATMUL = 11,
    DOT_PRODUCT = 12,

    // Control transactions
    CONFIGURE = 20,
    SYNC = 21,
    FENCE = 22,

    // Memory management
    ALLOCATE = 30,
    DEALLOCATE = 31,

    UNKNOWN = 255
};

// Transaction status
enum class TransactionStatus : uint8_t {
    ISSUED = 0,      // Transaction has been issued
    IN_PROGRESS = 1, // Transaction is being processed
    COMPLETED = 2,   // Transaction completed successfully
    FAILED = 3,      // Transaction failed
    CANCELLED = 4    // Transaction was cancelled
};

// Memory location descriptor
struct MemoryLocation {
    uint64_t address;        // Memory address
    uint64_t size;           // Data size in bytes
    uint32_t bank_id;        // Bank/tile/scratchpad ID
    ComponentType type;      // Type of memory component

    MemoryLocation() : address(0), size(0), bank_id(0), type(ComponentType::UNKNOWN) {}
    MemoryLocation(uint64_t addr, uint64_t sz, uint32_t id, ComponentType t)
        : address(addr), size(sz), bank_id(id), type(t) {}
};

// Transaction-specific payload data structures

// DMA transfer payload - captures source, destination, and data movement details
struct DMAPayload {
    MemoryLocation source;
    MemoryLocation destination;
    uint64_t bytes_transferred;  // Actual data size
    double bandwidth_gb_s;       // Theoretical bandwidth in GB/s

    DMAPayload() : bytes_transferred(0), bandwidth_gb_s(0.0) {}
};

// Compute operation payload - captures computation details
struct ComputePayload {
    uint64_t num_operations;  // Number of operations (MACs, FLOPs, etc.)
    uint64_t m, n, k;         // Matrix dimensions for GEMM (if applicable)
    std::string kernel_name;  // Name of compute kernel

    ComputePayload() : num_operations(0), m(0), n(0), k(0) {}
};

// Control/synchronization payload
struct ControlPayload {
    std::string command;      // Control command string
    uint64_t parameter;       // Generic parameter

    ControlPayload() : parameter(0) {}
};

// Memory operation payload
struct MemoryPayload {
    MemoryLocation location;
    bool is_hit;              // Cache hit/miss (if applicable)
    uint32_t latency_cycles;  // Additional latency

    MemoryPayload() : is_hit(true), latency_cycles(0) {}
};

// Variant to hold different payload types
using PayloadData = std::variant<
    std::monostate,          // No payload
    DMAPayload,
    ComputePayload,
    ControlPayload,
    MemoryPayload
>;

// Main trace entry structure - cycle-based timestamping
struct KPU_API TraceEntry {
    // Cycle-based timing (deterministic)
    CycleCount cycle_issue;      // Cycle when transaction was issued
    CycleCount cycle_complete;   // Cycle when transaction completed (0 if not completed)

    // Component identification
    ComponentType component_type;   // Type of component (DMA, Streamer, etc.)
    uint32_t component_id;          // Instance ID of the component

    // Transaction details
    TransactionType transaction_type;
    TransactionStatus status;
    uint64_t transaction_id;        // Unique transaction ID

    // Optional payload
    PayloadData payload;

    // Optional metadata
    std::string description;        // Human-readable description

    // Clock frequency for this component (GHz) - optional, for time conversion
    std::optional<double> clock_freq_ghz;

    // Constructor for issued transactions
    TraceEntry(CycleCount cycle, ComponentType comp_type, uint32_t comp_id,
               TransactionType trans_type, uint64_t trans_id)
        : cycle_issue(cycle)
        , cycle_complete(0)
        , component_type(comp_type)
        , component_id(comp_id)
        , transaction_type(trans_type)
        , status(TransactionStatus::ISSUED)
        , transaction_id(trans_id)
        , payload()
        , description()
        , clock_freq_ghz()
    {}

    // Mark transaction as completed
    void complete(CycleCount completion_cycle, TransactionStatus final_status = TransactionStatus::COMPLETED) {
        cycle_complete = completion_cycle;
        status = final_status;
    }

    // Get duration in cycles (returns 0 if not completed)
    CycleCount get_duration_cycles() const {
        if (status == TransactionStatus::ISSUED || status == TransactionStatus::IN_PROGRESS) {
            return 0;
        }
        return cycle_complete - cycle_issue;
    }

    // Convert cycle counts to time (nanoseconds) if clock frequency is available
    double get_issue_time_ns() const {
        if (!clock_freq_ghz.has_value()) return -1.0;
        return static_cast<double>(cycle_issue) / clock_freq_ghz.value();
    }

    double get_complete_time_ns() const {
        if (!clock_freq_ghz.has_value() || cycle_complete == 0) return -1.0;
        return static_cast<double>(cycle_complete) / clock_freq_ghz.value();
    }

    double get_duration_ns() const {
        if (!clock_freq_ghz.has_value() || cycle_complete == 0) return -1.0;
        return static_cast<double>(get_duration_cycles()) / clock_freq_ghz.value();
    }

    // Check if transaction overlaps with a given cycle range (for conflict detection)
    bool overlaps_with(CycleCount start_cycle, CycleCount end_cycle) const {
        if (cycle_complete == 0) return false; // Not completed yet
        return !(cycle_complete < start_cycle || cycle_issue > end_cycle);
    }
};

// Helper functions to convert enums to strings for export/debugging
KPU_API const char* to_string(ComponentType type);
KPU_API const char* to_string(TransactionType type);
KPU_API const char* to_string(TransactionStatus status);

} // namespace sw::trace

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
