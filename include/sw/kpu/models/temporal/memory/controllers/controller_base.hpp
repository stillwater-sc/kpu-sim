#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <queue>
#include <optional>
#include <functional>

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
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/dataflow/tile_flow_tracer.hpp>

namespace sw::kpu {

// ============================================================================
// DRAM Timing Parameters
// ============================================================================

/// DRAM timing configuration (GDDR6/HBM2e typical values)
struct DRAMTiming {
    // Row timing (cycles at memory clock)
    uint32_t tRCD = 14;     // Row-to-Column Delay: ACT to READ/WRITE
    uint32_t tRP = 14;      // Row Precharge: PRE to ACT
    uint32_t tRAS = 32;     // Row Active Time: ACT to PRE minimum
    uint32_t tRC = 46;      // Row Cycle: ACT to ACT (tRAS + tRP)
    uint32_t tRRD = 4;      // Row-to-Row Delay: ACT to ACT (different banks)

    // Column timing
    uint32_t tCL = 14;      // CAS Latency: READ to data
    uint32_t tWL = 4;       // Write Latency: WRITE to data
    uint32_t tBurst = 4;    // Burst length (cycles to transfer burst)

    // Refresh timing
    uint32_t tRFC = 280;    // Refresh Cycle: REF to ACT
    uint32_t tREFI = 3900;  // Refresh Interval: time between refreshes

    // Bank group timing (for GDDR6/DDR5)
    uint32_t tCCD_L = 4;    // CAS-to-CAS delay (same bank group)
    uint32_t tCCD_S = 2;    // CAS-to-CAS delay (different bank group)

    // Data bus timing
    uint32_t data_bus_width = 256;  // bits (GDDR6: 256, HBM2: 1024)
    double clock_freq_ghz = 2.0;    // Memory clock frequency

    /// Get bytes per burst
    uint32_t bytes_per_burst() const {
        return (data_bus_width / 8) * tBurst;
    }

    /// Get bandwidth in GB/s
    double bandwidth_gb_s() const {
        return (data_bus_width / 8.0) * clock_freq_ghz * 2.0;  // DDR = 2x
    }
};

// ============================================================================
// Bank State
// ============================================================================

/// State of a single DRAM bank
enum class BankState : uint8_t {
    IDLE,           // Bank is closed (precharged)
    ACTIVATING,     // Row activation in progress
    ACTIVE,         // Row is open
    PRECHARGING,    // Precharge in progress
    REFRESHING      // Refresh in progress
};

inline const char* to_string(BankState state) {
    switch (state) {
        case BankState::IDLE: return "IDLE";
        case BankState::ACTIVATING: return "ACTIVATING";
        case BankState::ACTIVE: return "ACTIVE";
        case BankState::PRECHARGING: return "PRECHARGING";
        case BankState::REFRESHING: return "REFRESHING";
        default: return "UNKNOWN";
    }
}

/// Per-bank state tracking
struct Bank {
    BankState state = BankState::IDLE;
    uint32_t open_row = 0;          // Currently open row (if ACTIVE)
    uint64_t state_until = 0;       // Cycle when state transition completes
    uint64_t last_access = 0;       // Last access cycle (for timing checks)
};

// ============================================================================
// Memory Request
// ============================================================================

/// Type of memory request
enum class MemoryRequestType : uint8_t {
    READ,
    WRITE
};

/// A memory request in the controller queue
struct MemoryRequest {
    uint64_t id;                    // Unique request ID
    MemoryRequestType type;         // READ or WRITE
    uint64_t address;               // Physical address
    uint32_t size;                  // Bytes to transfer
    uint8_t bank_id;                // Target bank
    uint32_t row;                   // Target row
    uint32_t col;                   // Target column
    uint64_t issue_cycle;           // When request was issued
    uint64_t completion_cycle;      // When data will be available
    std::vector<uint8_t> data;      // Data buffer
    std::function<void()> callback; // Completion callback

    // For tracing correlation
    uint8_t dma_channel = 0;
    uint8_t l3_id = 0;
};

// ============================================================================
// Memory Controller
// ============================================================================

/// Memory Controller with cycle-accurate DRAM timing simulation
///
/// Implements FR-FCFS (First-Ready First-Come-First-Serve) scheduling:
/// 1. Prioritize requests to already-open rows (page hits)
/// 2. Among same-priority, prefer older requests
///
/// Traces bank state transitions for visualization.
class KPU_API MemoryController {
public:
    /// Configuration
    struct Config {
        uint32_t num_banks;             // Number of banks
        uint32_t num_bank_groups;       // Bank groups (0 = no grouping)
        uint32_t rows_per_bank;         // Rows per bank
        uint32_t row_size_bytes;        // Bytes per row (page size)
        uint32_t queue_depth;           // Max pending requests
        DRAMTiming timing;              // Timing parameters

        Config()
            : num_banks(16)
            , num_bank_groups(4)
            , rows_per_bank(32768)
            , row_size_bytes(2048)
            , queue_depth(32)
            , timing()
        {}
    };

    /// Statistics
    struct Stats {
        uint64_t reads = 0;
        uint64_t writes = 0;
        uint64_t page_hits = 0;
        uint64_t page_conflicts = 0;
        uint64_t page_empty = 0;        // Access to closed bank
        uint64_t refreshes = 0;
        uint64_t stall_cycles = 0;      // Queue full stalls
        uint64_t total_latency = 0;     // Sum of all request latencies
        uint64_t min_latency = UINT64_MAX;
        uint64_t max_latency = 0;

        double hit_rate() const {
            uint64_t total = page_hits + page_conflicts + page_empty;
            return total > 0 ? static_cast<double>(page_hits) / total : 0.0;
        }

        double avg_latency() const {
            uint64_t total = reads + writes;
            return total > 0 ? static_cast<double>(total_latency) / total : 0.0;
        }
    };

private:
    Config config_;
    std::vector<Bank> banks_;
    std::queue<MemoryRequest> pending_queue_;
    std::vector<MemoryRequest> active_requests_;  // Currently processing
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 0;
    uint64_t next_refresh_cycle_ = 0;

    Stats stats_;

    // Tracing
    dataflow::TileFlowTracer* tracer_ = nullptr;

    // Backing storage
    ExternalMemory* memory_ = nullptr;

public:
    explicit MemoryController(const Config& config = Config());
    ~MemoryController() = default;

    // ========== Configuration ==========

    void set_tracer(dataflow::TileFlowTracer* tracer) { tracer_ = tracer; }
    void set_memory(ExternalMemory* memory) { memory_ = memory; }

    const Config& config() const { return config_; }
    const Stats& stats() const { return stats_; }

    // ========== Address Decoding ==========

    /// Decode physical address to bank/row/column
    void decode_address(uint64_t address, uint8_t& bank, uint32_t& row, uint32_t& col) const {
        // Simple interleaved mapping:
        // [row][bank][col][byte_offset]
        uint32_t col_bits = 6;  // 64 byte cache line
        uint32_t bank_bits = 4; // 16 banks
        uint32_t row_bits = 15; // 32K rows

        col = (address >> col_bits) & ((1 << col_bits) - 1);
        bank = static_cast<uint8_t>((address >> (col_bits + col_bits)) & ((1 << bank_bits) - 1));
        row = (address >> (col_bits + col_bits + bank_bits)) & ((1 << row_bits) - 1);

        // Ensure bank is in range
        bank = bank % config_.num_banks;
    }

    /// Get bank ID for an address
    uint8_t get_bank_for_address(uint64_t address) const {
        uint8_t bank;
        uint32_t row, col;
        decode_address(address, bank, row, col);
        return bank;
    }

    // ========== Request Interface ==========

    /// Submit a read request
    /// Returns request ID, or nullopt if queue is full
    std::optional<uint64_t> submit_read(uint64_t address, uint32_t size,
                                         std::function<void()> callback = nullptr);

    /// Submit a write request
    std::optional<uint64_t> submit_write(uint64_t address, const void* data, uint32_t size,
                                          std::function<void()> callback = nullptr);

    /// Check if queue can accept more requests
    bool can_accept() const {
        return pending_queue_.size() < config_.queue_depth;
    }

    /// Check if there are pending requests
    bool has_pending() const {
        return !pending_queue_.empty() || !active_requests_.empty();
    }

    // ========== Cycle Processing ==========

    /// Advance one cycle
    void tick();

    /// Set current cycle (for external clock management)
    void set_cycle(uint64_t cycle) { current_cycle_ = cycle; }
    uint64_t get_cycle() const { return current_cycle_; }

    /// Process until all pending requests complete
    void drain();

    // ========== Bank State Query ==========

    BankState get_bank_state(uint8_t bank_id) const {
        return bank_id < banks_.size() ? banks_[bank_id].state : BankState::IDLE;
    }

    uint32_t get_open_row(uint8_t bank_id) const {
        return bank_id < banks_.size() ? banks_[bank_id].open_row : 0;
    }

    bool is_row_open(uint8_t bank_id, uint32_t row) const {
        if (bank_id >= banks_.size()) return false;
        return banks_[bank_id].state == BankState::ACTIVE &&
               banks_[bank_id].open_row == row;
    }

    // ========== Reset ==========

    void reset();

private:
    /// Schedule next request from pending queue
    void schedule_request();

    /// Update bank states and complete requests
    void update_banks();

    /// Handle periodic refresh
    void handle_refresh();

    /// Calculate completion time for a request
    uint64_t calculate_completion(const MemoryRequest& req);

    /// Trace a memory controller event
    void trace_event(dataflow::TileEventType type, uint8_t bank_id,
                     uint32_t row = 0, uint64_t address = 0, uint32_t bytes = 0);
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
