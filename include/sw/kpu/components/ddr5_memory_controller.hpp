// ============================================================================
// include/sw/kpu/components/ddr5_memory_controller.hpp
// DDR5 Memory Controller with formal invariant checking
//
// Implements cycle-accurate DDR5 memory controller
// Implements IMemoryController interface for multi-fidelity simulation
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>
#include <sw/trace/trace_entry.hpp>
#include <sw/kpu/components/memory/memory_controller_interface.hpp>

// Forward declarations for trace integration
namespace sw::trace {
    class ResourceTracker;
}

namespace sw::kpu::ddr5 {

// ============================================================================
// Timing Parameters (DDR5-4800 @ 2400 MHz)
// ============================================================================

struct TimingParams {
    // Core timing (in CK cycles)
    uint32_t tRCD = 16;      // Row address to column address delay
    uint32_t tRP = 16;       // Row precharge time
    uint32_t tRAS = 32;      // Row active time (minimum)
    uint32_t tRC = 48;       // Row cycle time (tRAS + tRP)
    uint32_t tCL = 16;       // CAS latency (read)
    uint32_t tWL = 8;        // CAS write latency
    uint32_t tWR = 24;       // Write recovery time
    uint32_t tRTP = 8;       // Read to precharge

    // Bank group timing
    uint32_t tRRD_L = 8;     // ACT to ACT (same bank group)
    uint32_t tRRD_S = 4;     // ACT to ACT (different bank group)
    uint32_t tCCD_L = 8;     // CAS to CAS (same bank group)
    uint32_t tCCD_S = 4;     // CAS to CAS (different bank group)

    // Turnaround timing
    uint32_t tWTR_L = 12;    // Write to read (same bank group)
    uint32_t tWTR_S = 4;     // Write to read (different bank group)
    uint32_t tRTW = 16;      // Read to write (bus turnaround)

    // Burst timing (BL16)
    uint32_t tBurst = 8;     // Burst length 16 (8 CK cycles for DDR)

    // Refresh timing (per-bank and all-bank)
    uint32_t tRFCpb = 130;   // Per-bank refresh cycle time (fine-grained refresh)
    uint32_t tRFCab = 295;   // All-bank refresh cycle time
    uint32_t tREFI = 3900;   // Refresh interval (7.8us at 2400MHz)

    // Activate window
    uint32_t tFAW = 32;      // Four activate window
};

// ============================================================================
// Burst Length Configuration
// ============================================================================

enum class BurstLength : uint8_t {
    BL8 = 8,
    BL16 = 16   // DDR5 default
};

// ============================================================================
// Bank State Machine
// ============================================================================

enum class BankState : uint8_t {
    IDLE,           // Precharged, no row open
    ACTIVATING,     // Row being opened (tRCD in progress)
    ACTIVE,         // Row open, ready for R/W
    READING,        // Read burst in progress
    WRITING,        // Write burst in progress
    PRECHARGING,    // Row being closed (tRP in progress)
    REFRESHING      // Per-bank refresh in progress
};

constexpr std::string_view bank_state_name(BankState state) {
    switch (state) {
        case BankState::IDLE:        return "IDLE";
        case BankState::ACTIVATING:  return "ACTIVATING";
        case BankState::ACTIVE:      return "ACTIVE";
        case BankState::READING:     return "READING";
        case BankState::WRITING:     return "WRITING";
        case BankState::PRECHARGING: return "PRECHARGING";
        case BankState::REFRESHING:  return "REFRESHING";
        default:                     return "UNKNOWN";
    }
}

struct Bank {
    BankState state = BankState::IDLE;
    uint32_t open_row = 0;           // Valid when ACTIVE
    uint64_t state_until = 0;        // Cycle when current state completes
    uint64_t last_activate = 0;      // For tRAS, tRC tracking
    uint64_t last_read_cmd = 0;      // For tRTP tracking
    uint64_t last_write_cmd = 0;     // For tWR tracking
    uint64_t last_refresh = 0;       // For per-bank refresh tracking
    uint8_t bank_group = 0;          // Which bank group (0-3)

    // Track burst completion for data bus
    uint64_t burst_end = 0;          // When current burst ends

    // INV-002: Track which request opened this page for PRECHARGE tracing
    uint64_t page_opener_request_id = 0;
};

// ============================================================================
// Bank Group State
// ============================================================================

struct BankGroup {
    uint64_t last_activate = 0;      // For tRRD_L tracking
    uint64_t last_cas = 0;           // For tCCD_L tracking
    uint64_t last_write = 0;         // For tWTR_L tracking
};

// ============================================================================
// Data Bus State
// ============================================================================

enum class DataBusState : uint8_t {
    IDLE,
    READ_BURST,
    WRITE_BURST,
    TURNAROUND
};

constexpr std::string_view data_bus_state_name(DataBusState state) {
    switch (state) {
        case DataBusState::IDLE:        return "IDLE";
        case DataBusState::READ_BURST:  return "READ_BURST";
        case DataBusState::WRITE_BURST: return "WRITE_BURST";
        case DataBusState::TURNAROUND:  return "TURNAROUND";
        default:                        return "UNKNOWN";
    }
}

// ============================================================================
// Command Bus State
// ============================================================================

enum class CommandBusState : uint8_t {
    IDLE,
    BUSY
};

// ============================================================================
// Channel State (DDR5 has two sub-channels per DIMM)
// ============================================================================

struct Channel {
    // Command bus
    CommandBusState cmd_bus_state = CommandBusState::IDLE;
    uint64_t cmd_bus_until = 0;

    // Data bus
    DataBusState data_bus_state = DataBusState::IDLE;
    uint64_t data_bus_until = 0;
    bool last_was_write = false;     // For R->W turnaround

    // Deferred tracing flags
    bool deferred_data_bus_idle_trace = false;
    bool deferred_cmd_bus_idle_trace = false;

    // tFAW tracking (circular buffer of last 4 activate times)
    std::array<uint64_t, 4> activate_window = {0, 0, 0, 0};
    uint8_t activate_index = 0;

    // Per-bank refresh tracking
    uint8_t next_refresh_bank = 0;   // Round-robin

    // Banks (32 banks: 4 bank groups × 8 banks per group)
    std::array<Bank, 32> banks;
    // Bank groups
    std::array<BankGroup, 4> bank_groups;
};

// ============================================================================
// Memory Request
// ============================================================================

enum class RequestType : uint8_t {
    READ,
    WRITE
};

struct MemoryRequest {
    uint64_t id = 0;
    RequestType type = RequestType::READ;
    uint64_t address = 0;
    uint32_t size = 0;
    uint8_t channel = 0;
    uint8_t bank = 0;
    uint32_t row = 0;
    uint32_t col = 0;

    uint64_t submit_cycle = 0;       // When request was submitted
    uint64_t issue_cycle = 0;        // When command was issued
    uint64_t complete_cycle = 0;     // When data transfer completes

    bool triggered_activate = false; // True if this request triggered a row activate
    bool triggered_conflict = false; // True if this request triggered a page conflict

    std::vector<uint8_t> data;
    std::function<void()> callback;
};

// ============================================================================
// Invariant Violation
// ============================================================================

struct InvariantViolation {
    uint64_t cycle;
    std::string invariant_id;
    std::string message;
    uint8_t channel;
    uint8_t bank;
};

// ============================================================================
// Statistics
// ============================================================================

struct Statistics {
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t page_hits = 0;
    uint64_t page_empty = 0;
    uint64_t page_conflicts = 0;
    uint64_t refreshes = 0;
    uint64_t total_latency = 0;
    uint64_t stall_cycles = 0;

    // Bank group utilization
    uint64_t same_bg_accesses = 0;
    uint64_t diff_bg_accesses = 0;

    // Turnaround stats
    uint64_t read_to_write_turnarounds = 0;
    uint64_t write_to_read_turnarounds = 0;

    // Per-channel stats
    uint64_t channel_a_accesses = 0;
    uint64_t channel_b_accesses = 0;

    // Calibration metrics
    uint64_t read_latency_total = 0;
    uint64_t write_latency_total = 0;
    uint64_t read_latency_min = UINT64_MAX;
    uint64_t read_latency_max = 0;
    uint64_t write_latency_min = UINT64_MAX;
    uint64_t write_latency_max = 0;

    // Per-scenario latency tracking
    uint64_t page_hit_latency_total = 0;
    uint64_t page_hit_count = 0;
    uint64_t page_empty_latency_total = 0;
    uint64_t page_empty_count = 0;
    uint64_t page_conflict_latency_total = 0;
    uint64_t page_conflict_count = 0;

    // Calibration helper methods
    double avg_latency() const {
        uint64_t total = reads + writes;
        return total > 0 ? static_cast<double>(total_latency) / total : 0.0;
    }

    double avg_read_latency() const {
        return reads > 0 ? static_cast<double>(read_latency_total) / reads : 0.0;
    }

    double avg_write_latency() const {
        return writes > 0 ? static_cast<double>(write_latency_total) / writes : 0.0;
    }

    double avg_page_hit_latency() const {
        return page_hit_count > 0 ? static_cast<double>(page_hit_latency_total) / page_hit_count : 0.0;
    }

    double avg_page_empty_latency() const {
        return page_empty_count > 0 ? static_cast<double>(page_empty_latency_total) / page_empty_count : 0.0;
    }

    double avg_page_conflict_latency() const {
        return page_conflict_count > 0 ? static_cast<double>(page_conflict_latency_total) / page_conflict_count : 0.0;
    }

    double page_hit_rate() const {
        uint64_t total = page_hits + page_empty + page_conflicts;
        return total > 0 ? static_cast<double>(page_hits) / total : 0.0;
    }

    double page_empty_rate() const {
        uint64_t total = page_hits + page_empty + page_conflicts;
        return total > 0 ? static_cast<double>(page_empty) / total : 0.0;
    }

    double page_conflict_rate() const {
        uint64_t total = page_hits + page_empty + page_conflicts;
        return total > 0 ? static_cast<double>(page_conflicts) / total : 0.0;
    }

    double page_hit_factor() const {
        double mean = avg_latency();
        return mean > 0 ? avg_page_hit_latency() / mean : 1.0;
    }

    double page_empty_factor() const {
        double mean = avg_latency();
        return mean > 0 ? avg_page_empty_latency() / mean : 1.0;
    }

    double page_conflict_factor() const {
        double mean = avg_latency();
        return mean > 0 ? avg_page_conflict_latency() / mean : 1.0;
    }

    double channel_balance() const {
        uint64_t total = channel_a_accesses + channel_b_accesses;
        return total > 0 ? static_cast<double>(channel_a_accesses) / total : 0.5;
    }
};

// ============================================================================
// DDR5 Memory Controller
// ============================================================================

/// Cycle-accurate DDR5 memory controller implementing IMemoryController
///
/// Provides full DDR5 protocol simulation including:
/// - Bank state machine (IDLE, ACTIVATING, ACTIVE, READING, WRITING, PRECHARGING, REFRESHING)
/// - 32 banks per channel (4 bank groups × 8 banks per group)
/// - Dual-channel operation (two independent 32-bit sub-channels)
/// - Per-bank refresh scheduling (fine-grained refresh)
/// - Bank group timing constraints (tRRD_L, tCCD_L, tWTR_L)
/// - tFAW (four-activate window) tracking
/// - Formal invariant checking
/// - Full tracing support for visualization
class DDR5MemoryController : public sw::kpu::IMemoryController {
public:
    /// DDR5-specific configuration
    struct Config {
        uint8_t num_channels = 2;        // Typically 2 (dual-channel)
        uint8_t banks_per_channel = 32;  // 4 bank groups × 8 banks
        uint8_t bank_groups = 4;         // Always 4
        uint8_t banks_per_group = 8;     // Always 8
        BurstLength burst_length = BurstLength::BL16;
        uint32_t queue_depth = 64;
        TimingParams timing;

        // Address mapping
        uint32_t row_bits = 17;          // 128K rows
        uint32_t col_bits = 10;          // 1K columns
        uint32_t bank_bits = 5;          // 32 banks
        uint32_t channel_bits = 1;       // 2 channels
    };

    explicit DDR5MemoryController(const Config& config);

    /// Construct from generic MemoryControllerConfig
    explicit DDR5MemoryController(const sw::kpu::MemoryControllerConfig& config);

    ~DDR5MemoryController() override = default;

    // ========================================================================
    // IMemoryController Interface - Request Interface
    // ========================================================================

    std::optional<uint64_t> submit_read(
        uint64_t address,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    std::optional<uint64_t> submit_write(
        uint64_t address,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override;
    bool has_pending() const override;
    size_t pending_count() const override;

    // ========================================================================
    // IMemoryController Interface - Simulation Interface
    // ========================================================================

    void tick() override;
    void drain() override;
    void reset() override;

    uint64_t current_cycle() const override { return current_cycle_; }
    void set_cycle(uint64_t cycle) override { current_cycle_ = cycle; }

    // ========================================================================
    // IMemoryController Interface - Configuration Queries
    // ========================================================================

    sw::kpu::SimulationFidelity fidelity() const override {
        return sw::kpu::SimulationFidelity::CYCLE_ACCURATE;
    }

    sw::kpu::MemoryTechnology technology() const override {
        return sw::kpu::MemoryTechnology::DDR5;
    }

    sw::kpu::VerificationLevel verification_level() const override {
        return check_invariants_ ? sw::kpu::VerificationLevel::INVARIANTS
                                 : sw::kpu::VerificationLevel::NONE;
    }

    const sw::kpu::MemoryControllerConfig& config() const override { return interface_config_; }

    // ========================================================================
    // IMemoryController Interface - Bank State Queries
    // ========================================================================

    IMemoryController::BankState get_bank_state(uint8_t channel, uint8_t bank) const override;
    bool is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const override;
    uint8_t num_channels() const override { return ddr5_config_.num_channels; }
    uint8_t banks_per_channel() const override { return ddr5_config_.banks_per_channel; }

    // ========================================================================
    // IMemoryController Interface - Statistics
    // ========================================================================

    const sw::kpu::MemoryControllerStatistics& stats() const override { return interface_stats_; }
    void reset_stats() override;

    // ========================================================================
    // IMemoryController Interface - Observability
    // ========================================================================

    void enable_tracing(bool enable) override { tracing_enabled_ = enable; }
    bool tracing_enabled() const override { return tracing_enabled_; }
    void set_resource_tracker(sw::trace::ResourceTracker* tracker) override {
        resource_tracker_ = tracker;
    }

    // ========================================================================
    // IMemoryController Interface - Invariant Checking
    // ========================================================================

    void enable_invariants(bool enable) override { check_invariants_ = enable; }
    bool invariants_enabled() const override { return check_invariants_; }
    const std::vector<IMemoryController::InvariantViolation>& violations() const override {
        return interface_violations_;
    }
    bool has_violations() const override { return !violations_.empty(); }
    void clear_violations() override { violations_.clear(); interface_violations_.clear(); }

    // ========================================================================
    // IMemoryController Interface - Refresh Control
    // ========================================================================

    void set_refresh_mode(RefreshMode mode) override;
    RefreshMode refresh_mode() const override { return refresh_mode_; }

    void set_refresh_interval(uint64_t cycles) override;
    uint64_t refresh_interval() const override { return refresh_interval_; }

    void set_deadline_enforcement(bool enforce) override { deadline_enforcement_ = enforce; }
    bool deadline_enforced() const override { return deadline_enforcement_; }

    uint64_t cycles_until_deadline(uint8_t channel, uint8_t bank) const override;
    bool refresh_pending(uint8_t channel, uint8_t bank) const override;
    uint32_t refresh_debt(uint8_t channel, uint8_t bank) const override;

    bool inject_refresh(uint8_t channel, int8_t bank = -1) override;

    // ========================================================================
    // DDR5-Specific API (not part of IMemoryController)
    // ========================================================================

    /// Get DDR5-specific bank state (more detailed than interface)
    ddr5::BankState get_ddr5_bank_state(uint8_t channel, uint8_t bank) const;

    /// Check if a bank is idle
    bool is_bank_idle(uint8_t channel, uint8_t bank) const;

    /// Get DDR5-specific statistics
    const Statistics& ddr5_stats() const { return stats_; }

    /// Get DDR5-specific violations
    const std::vector<ddr5::InvariantViolation>& ddr5_violations() const { return violations_; }

    /// Get resource tracker
    sw::trace::ResourceTracker* resource_tracker() const { return resource_tracker_; }

    /// Get trace entries
    const std::vector<sw::trace::TraceEntry>& trace_entries() const override { return trace_entries_; }

    /// Clear trace entries
    void clear_trace_entries() override { trace_entries_.clear(); }

    /// Get DDR5-specific configuration
    const Config& ddr5_config() const { return ddr5_config_; }

    /// Get timing parameters
    const TimingParams& timing() const { return ddr5_config_.timing; }

private:
    // Address decoding
    void decode_address(uint64_t address,
                        uint8_t& channel, uint8_t& bank,
                        uint32_t& row, uint32_t& col) const;

    // Command scheduling
    void schedule_requests();
    bool try_issue_request(MemoryRequest& req);

    // Bank operations
    bool can_activate(uint8_t channel, uint8_t bank, uint64_t cycle) const;
    bool can_read(uint8_t channel, uint8_t bank, uint64_t cycle) const;
    bool can_write(uint8_t channel, uint8_t bank, uint64_t cycle) const;
    bool can_precharge(uint8_t channel, uint8_t bank, uint64_t cycle) const;
    bool can_refresh(uint8_t channel, uint8_t bank) const;

    void do_activate(uint8_t channel, uint8_t bank, uint32_t row, uint64_t request_id);
    void do_read(uint8_t channel, uint8_t bank, MemoryRequest& req);
    void do_write(uint8_t channel, uint8_t bank, MemoryRequest& req);
    void do_precharge(uint8_t channel, uint8_t bank);
    void do_refresh(uint8_t channel, uint8_t bank);

    // State updates
    void update_bank_states();
    void update_bus_states();
    void finalize_bus_traces();
    void handle_refresh();
    void complete_requests();

    // Timing calculations
    uint64_t next_activate_time(uint8_t channel, uint8_t bank) const;
    uint64_t next_read_time(uint8_t channel, uint8_t bank) const;
    uint64_t next_write_time(uint8_t channel, uint8_t bank) const;
    uint32_t burst_cycles() const;

    // tFAW tracking
    bool check_tFAW(uint8_t channel) const;
    void record_activate(uint8_t channel);

    // Invariant checking
    void check_all_invariants();
    void check_bank_invariants(uint8_t channel, uint8_t bank);
    void check_timing_invariants(uint8_t channel);
    void check_bus_invariants(uint8_t channel);
    void report_violation(const std::string& id, const std::string& msg,
                          uint8_t channel = 0, uint8_t bank = 0);

    // Tracing helpers
    void trace_bank_state_change(uint8_t channel, uint8_t bank, ddr5::BankState new_state,
                                 const std::string& reason = "");
    void trace_bus_state_change(uint8_t channel, bool is_data_bus,
                                const std::string& state, const std::string& reason = "");
    void trace_command(uint8_t channel, uint8_t bank, const std::string& cmd,
                       uint64_t duration, uint64_t request_id = 0);

    // Synchronize interface stats from internal stats
    void sync_interface_stats();

    // Synchronize interface violations from internal violations
    void sync_interface_violations();

    // Configuration
    Config ddr5_config_;                              // DDR5-specific config
    sw::kpu::MemoryControllerConfig interface_config_; // Interface-compatible config

    // State
    std::array<Channel, 2> channels_;
    uint64_t current_cycle_ = 0;
    uint64_t next_request_id_ = 1;

    // Request queues
    std::queue<MemoryRequest> pending_queue_;
    std::vector<MemoryRequest> active_requests_;

    // Statistics
    Statistics stats_;
    mutable sw::kpu::MemoryControllerStatistics interface_stats_;

    // Invariant checking
    bool check_invariants_ = true;
    std::vector<ddr5::InvariantViolation> violations_;
    mutable std::vector<IMemoryController::InvariantViolation> interface_violations_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;
    std::vector<sw::trace::TraceEntry> trace_entries_;

    // Refresh control
    RefreshMode refresh_mode_ = RefreshMode::AUTOMATIC;
    uint64_t refresh_interval_ = 0;        // 0 = use tREFI
    bool deadline_enforcement_ = true;
    uint64_t last_interval_refresh_ = 0;   // For INTERVAL mode tracking
};

} // namespace sw::kpu::ddr5
