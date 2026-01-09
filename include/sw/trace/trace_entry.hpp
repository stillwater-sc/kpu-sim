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

    // LPDDR5 Memory Controller components
    LPDDR5_BANK = 13,           // Individual LPDDR5 bank
    LPDDR5_BANK_GROUP = 14,     // LPDDR5 bank group (4 banks)
    LPDDR5_CHANNEL = 15,        // LPDDR5 channel
    LPDDR5_DATA_BUS = 16,       // LPDDR5 data bus
    LPDDR5_CMD_BUS = 17,        // LPDDR5 command bus

    // GDDR6 Memory Controller components
    GDDR6_BANK = 22,            // Individual GDDR6 bank (16 banks total)
    GDDR6_BANK_GROUP = 23,      // GDDR6 bank group (4 groups, 4 banks each)
    GDDR6_CHANNEL = 24,         // GDDR6 channel (dual 16-bit channels)
    GDDR6_DATA_BUS = 25,        // GDDR6 data bus (DQ)
    GDDR6_CMD_BUS = 26,         // GDDR6 command/address bus (CA)

    // HBM2 Memory Controller components
    HBM2_BANK = 30,             // Individual HBM2 bank (16 banks per pseudo-channel)
    HBM2_BANK_GROUP = 31,       // HBM2 bank group (4 groups per pseudo-channel)
    HBM2_PSEUDO_CHANNEL = 32,   // HBM2 pseudo-channel (2 per channel)
    HBM2_CHANNEL = 33,          // HBM2 channel (8 channels per stack)
    HBM2_DATA_BUS = 34,         // HBM2 data bus (64-bit per pseudo-channel)
    HBM2_CMD_BUS = 35,          // HBM2 command/address bus

    // HBM3 Memory Controller components
    HBM3_BANK = 40,             // Individual HBM3 bank (16-32 banks per pseudo-channel)
    HBM3_BANK_GROUP = 41,       // HBM3 bank group (4-8 groups per pseudo-channel)
    HBM3_PSEUDO_CHANNEL = 42,   // HBM3 pseudo-channel (2 per channel)
    HBM3_CHANNEL = 43,          // HBM3 channel (16 channels per stack)
    HBM3_DATA_BUS = 44,         // HBM3 data bus (32-bit per pseudo-channel)
    HBM3_CMD_BUS = 45,          // HBM3 command/address bus

    // DDR5 Memory Controller components
    DDR5_BANK = 46,             // Individual DDR5 bank (32 banks per channel)
    DDR5_BANK_GROUP = 47,       // DDR5 bank group (4 groups, 8 banks each)
    DDR5_CHANNEL = 48,          // DDR5 channel (dual-channel)
    DDR5_DATA_BUS = 49,         // DDR5 data bus (64-bit per channel)
    DDR5_CMD_BUS = 50,          // DDR5 command/address bus

    // GDDR7 Memory Controller components
    GDDR7_BANK = 51,            // Individual GDDR7 bank (16 banks total)
    GDDR7_BANK_GROUP = 52,      // GDDR7 bank group (4 groups, 4 banks each)
    GDDR7_CHANNEL = 53,         // GDDR7 channel (dual 16-bit channels, PAM3)
    GDDR7_DATA_BUS = 54,        // GDDR7 data bus (DQ)
    GDDR7_CMD_BUS = 55,         // GDDR7 command/address bus (CA)

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

    // LPDDR5 Memory Controller operations
    ACTIVATE = 40,       // Row activation (ACT command)
    PRECHARGE = 41,      // Row precharge (PRE command)
    REFRESH = 42,        // Per-bank or all-bank refresh
    BURST_READ = 43,     // Read burst (CAS read)
    BURST_WRITE = 44,    // Write burst (CAS write)
    TURNAROUND = 45,     // Bus turnaround (R->W or W->R)

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
