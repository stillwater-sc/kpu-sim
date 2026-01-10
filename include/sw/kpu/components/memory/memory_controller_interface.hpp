// ============================================================================
// include/sw/kpu/components/memory/memory_controller_interface.hpp
// Abstract interface for memory controllers at all fidelity levels
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>
#include <sw/trace/trace_entry.hpp>

// Forward declarations
namespace sw::trace {
    class ResourceTracker;
}

namespace sw::kpu {

// ============================================================================
// Memory Request Types
// ============================================================================

/// Type of memory request
enum class MemoryRequestType : uint8_t {
    READ,
    WRITE
};

constexpr std::string_view to_string(MemoryRequestType type) {
    switch (type) {
        case MemoryRequestType::READ:  return "READ";
        case MemoryRequestType::WRITE: return "WRITE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Memory Controller Statistics
// ============================================================================

/// Common statistics interface for all memory controllers
struct MemoryControllerStatistics {
    // Request counts
    uint64_t reads = 0;
    uint64_t writes = 0;

    // Page/row buffer statistics
    uint64_t page_hits = 0;
    uint64_t page_empty = 0;      // Access to closed bank
    uint64_t page_conflicts = 0;  // Different row open

    // Latency statistics
    uint64_t total_latency = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t max_latency = 0;

    // Utilization
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;
    uint64_t stall_cycles = 0;

    // Refresh (for cycle-accurate)
    uint64_t refreshes = 0;

    // Turnaround (for cycle-accurate)
    uint64_t read_to_write_turnarounds = 0;
    uint64_t write_to_read_turnarounds = 0;

    // Derived metrics
    double avg_latency() const {
        uint64_t total = reads + writes;
        return total > 0 ? static_cast<double>(total_latency) / total : 0.0;
    }

    double hit_rate() const {
        uint64_t total = page_hits + page_empty + page_conflicts;
        return total > 0 ? static_cast<double>(page_hits) / total : 0.0;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / total : 0.0;
    }

    uint64_t total_requests() const {
        return reads + writes;
    }

    /// Reset all statistics to zero
    void reset() {
        reads = writes = 0;
        page_hits = page_empty = page_conflicts = 0;
        total_latency = 0;
        min_latency = UINT64_MAX;
        max_latency = 0;
        busy_cycles = idle_cycles = stall_cycles = 0;
        refreshes = 0;
        read_to_write_turnarounds = write_to_read_turnarounds = 0;
    }
};

// ============================================================================
// Memory Controller Interface
// ============================================================================

/// Abstract interface for memory controllers at all fidelity levels
///
/// This interface provides a common API that is implemented by:
/// - BehavioralMemoryController (instant/fixed latency)
/// - TransactionalMemoryController (queue-based statistical timing)
/// - LPDDR5MemoryController (cycle-accurate LPDDR5)
/// - HBM3MemoryController (cycle-accurate HBM3)
/// - etc.
///
/// All implementations guarantee:
/// 1. Functional correctness (data is transferred correctly)
/// 2. Callback semantics (callbacks are invoked when operation completes)
/// 3. Statistics collection (if enabled)
/// 4. Tracing support (if enabled)
class IMemoryController {
public:
    virtual ~IMemoryController() = default;

    // ========================================================================
    // Request Interface
    // ========================================================================

    /// Submit a read request
    ///
    /// @param address Physical memory address
    /// @param size Number of bytes to read
    /// @param callback Optional callback invoked when read completes
    /// @return Request ID if accepted, nullopt if queue is full
    ///
    /// Behavior by fidelity:
    /// - BEHAVIORAL: Completes immediately, callback invoked before return
    /// - TRANSACTIONAL: Queued, callback invoked after statistical delay
    /// - CYCLE_ACCURATE: Queued, callback invoked after protocol timing
    virtual std::optional<uint64_t> submit_read(
        uint64_t address,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    /// Submit a write request
    ///
    /// @param address Physical memory address
    /// @param data Pointer to data to write
    /// @param size Number of bytes to write
    /// @param callback Optional callback invoked when write completes
    /// @return Request ID if accepted, nullopt if queue is full
    virtual std::optional<uint64_t> submit_write(
        uint64_t address,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    /// Check if queue can accept more requests
    virtual bool can_accept() const = 0;

    /// Check if there are pending requests
    virtual bool has_pending() const = 0;

    /// Get number of pending requests
    virtual size_t pending_count() const = 0;

    // ========================================================================
    // Simulation Interface
    // ========================================================================

    /// Advance simulation by one cycle
    ///
    /// For BEHAVIORAL: May be a no-op (instant completion)
    /// For TRANSACTIONAL: Updates queue state, may complete requests
    /// For CYCLE_ACCURATE: Full FSM advancement
    virtual void tick() = 0;

    /// Process until all pending requests complete
    ///
    /// Useful for draining the controller at end of simulation
    virtual void drain() = 0;

    /// Reset controller to initial state
    virtual void reset() = 0;

    /// Get current simulation cycle
    virtual uint64_t current_cycle() const = 0;

    /// Set current simulation cycle (for external clock management)
    virtual void set_cycle(uint64_t cycle) = 0;

    // ========================================================================
    // Configuration Queries
    // ========================================================================

    /// Get simulation fidelity level
    virtual SimulationFidelity fidelity() const = 0;

    /// Get memory technology
    virtual MemoryTechnology technology() const = 0;

    /// Get verification level
    virtual VerificationLevel verification_level() const = 0;

    /// Get full configuration
    virtual const MemoryControllerConfig& config() const = 0;

    // ========================================================================
    // Bank State Queries (primarily for CYCLE_ACCURATE)
    // ========================================================================

    /// Bank state enumeration (common across technologies)
    enum class BankState : uint8_t {
        IDLE,           // Precharged, no row open
        ACTIVATING,     // Row being opened
        ACTIVE,         // Row open, ready for R/W
        READING,        // Read burst in progress
        WRITING,        // Write burst in progress
        PRECHARGING,    // Row being closed
        REFRESHING      // Refresh in progress
    };

    /// Get state of a specific bank
    ///
    /// For BEHAVIORAL/TRANSACTIONAL: Always returns ACTIVE or IDLE
    /// For CYCLE_ACCURATE: Returns actual bank state
    virtual BankState get_bank_state(uint8_t channel, uint8_t bank) const = 0;

    /// Check if a specific row is open in a bank
    virtual bool is_row_open(uint8_t channel, uint8_t bank, uint32_t row) const = 0;

    /// Get number of channels
    virtual uint8_t num_channels() const = 0;

    /// Get number of banks per channel
    virtual uint8_t banks_per_channel() const = 0;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get current statistics
    virtual const MemoryControllerStatistics& stats() const = 0;

    /// Reset statistics
    virtual void reset_stats() = 0;

    // ========================================================================
    // Observability
    // ========================================================================

    /// Enable or disable tracing
    virtual void enable_tracing(bool enable) = 0;

    /// Check if tracing is enabled
    virtual bool tracing_enabled() const = 0;

    /// Set resource tracker for trace export
    virtual void set_resource_tracker(sw::trace::ResourceTracker* tracker) = 0;

    /// Get trace entries (for cycle-accurate controllers)
    /// Returns empty vector if tracing not supported or disabled
    virtual const std::vector<sw::trace::TraceEntry>& trace_entries() const {
        static const std::vector<sw::trace::TraceEntry> empty;
        return empty;
    }

    /// Clear trace entries
    virtual void clear_trace_entries() {}

    // ========================================================================
    // Invariant Checking (CYCLE_ACCURATE only)
    // ========================================================================

    /// Invariant violation record
    struct InvariantViolation {
        uint64_t cycle;
        std::string invariant_id;
        std::string message;
        uint8_t channel;
        uint8_t bank;
    };

    /// Enable or disable invariant checking
    virtual void enable_invariants(bool enable) = 0;

    /// Check if invariant checking is enabled
    virtual bool invariants_enabled() const = 0;

    /// Get list of invariant violations
    virtual const std::vector<InvariantViolation>& violations() const = 0;

    /// Check if any violations occurred
    virtual bool has_violations() const = 0;

    /// Clear violation list
    virtual void clear_violations() = 0;
};

// ============================================================================
// Memory Controller Factory
// ============================================================================

/// Create a memory controller based on configuration
///
/// The factory selects the appropriate implementation based on:
/// 1. Fidelity level (BEHAVIORAL, TRANSACTIONAL, CYCLE_ACCURATE)
/// 2. Memory technology (LPDDR5, HBM3, DDR5, etc.)
///
/// @param config Memory controller configuration
/// @return Unique pointer to memory controller implementation
std::unique_ptr<IMemoryController> create_memory_controller(
    const MemoryControllerConfig& config);

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert bank state to string
constexpr std::string_view to_string(IMemoryController::BankState state) {
    switch (state) {
        case IMemoryController::BankState::IDLE:        return "IDLE";
        case IMemoryController::BankState::ACTIVATING:  return "ACTIVATING";
        case IMemoryController::BankState::ACTIVE:      return "ACTIVE";
        case IMemoryController::BankState::READING:     return "READING";
        case IMemoryController::BankState::WRITING:     return "WRITING";
        case IMemoryController::BankState::PRECHARGING: return "PRECHARGING";
        case IMemoryController::BankState::REFRESHING:  return "REFRESHING";
        default: return "UNKNOWN";
    }
}

} // namespace sw::kpu
