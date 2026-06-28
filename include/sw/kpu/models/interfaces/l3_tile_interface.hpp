// ============================================================================
// include/sw/kpu/models/interfaces/memory/l3_tile_interface.hpp
// Abstract interface for L3 tiles at all fidelity levels
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

// Forward declarations
namespace sw::trace {
    class ResourceTracker;
}

namespace sw::kpu {

// ============================================================================
// L3 Tile Statistics
// ============================================================================

/// Common statistics interface for all L3 tile implementations
struct L3TileStatistics {
    // Access counts
    uint64_t reads = 0;
    uint64_t writes = 0;
    uint64_t block_reads = 0;
    uint64_t block_writes = 0;

    // Data volume
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;

    // Latency statistics
    uint64_t total_read_latency = 0;
    uint64_t total_write_latency = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t max_latency = 0;

    // Bank-level statistics
    uint64_t bank_conflicts = 0;
    uint64_t port_conflicts = 0;

    // Utilization
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;

    // Derived metrics
    double avg_read_latency() const {
        return reads > 0 ? static_cast<double>(total_read_latency) / static_cast<double>(reads) : 0.0;
    }

    double avg_write_latency() const {
        return writes > 0 ? static_cast<double>(total_write_latency) / static_cast<double>(writes) : 0.0;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    uint64_t total_accesses() const {
        return reads + writes + block_reads + block_writes;
    }

    void reset() {
        reads = writes = block_reads = block_writes = 0;
        bytes_read = bytes_written = 0;
        total_read_latency = total_write_latency = 0;
        min_latency = UINT64_MAX;
        max_latency = 0;
        bank_conflicts = port_conflicts = 0;
        busy_cycles = idle_cycles = 0;
    }
};

// ============================================================================
// L3 Tile Interface
// ============================================================================

/// Abstract interface for L3 tiles at all fidelity levels
///
/// This interface provides a common API that is implemented by:
/// - BehavioralL3Tile (instant access)
/// - TransactionalL3Tile (bank contention, access latency)
/// - CycleAccurateL3Tile (full port arbitration)
///
/// All implementations guarantee:
/// 1. Functional correctness (data is stored and retrieved correctly)
/// 2. Callback semantics (callbacks are invoked when operation completes)
/// 3. Statistics collection (if enabled)
/// 4. Tracing support (if enabled)
class IL3Tile {
public:
    virtual ~IL3Tile() = default;

    // ========================================================================
    // Basic Memory Operations
    // ========================================================================

    /// Read data from tile
    ///
    /// @param addr Address within tile
    /// @param data Buffer to store read data
    /// @param size Number of bytes to read
    /// @param callback Optional callback when read completes
    /// @return Request ID if accepted, nullopt if busy
    ///
    /// Behavior by fidelity:
    /// - BEHAVIORAL: Completes immediately
    /// - TRANSACTIONAL: Fixed latency with bank contention
    /// - CYCLE_ACCURATE: Full port and bank arbitration
    virtual std::optional<uint64_t> read(
        uint64_t addr,
        void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    /// Write data to tile
    virtual std::optional<uint64_t> write(
        uint64_t addr,
        const void* data,
        uint32_t size,
        std::function<void()> callback = nullptr) = 0;

    // ========================================================================
    // Block Operations (for matrix data)
    // ========================================================================

    /// Read a 2D block from tile
    ///
    /// @param base_addr Starting address
    /// @param data Buffer to store read data
    /// @param height Number of rows
    /// @param width Number of columns
    /// @param element_size Size of each element in bytes
    /// @param stride Row stride (0 = contiguous = width * element_size)
    /// @param callback Optional callback when read completes
    virtual std::optional<uint64_t> read_block(
        uint64_t base_addr,
        void* data,
        uint32_t height,
        uint32_t width,
        uint32_t element_size,
        uint32_t stride = 0,
        std::function<void()> callback = nullptr) = 0;

    /// Write a 2D block to tile
    virtual std::optional<uint64_t> write_block(
        uint64_t base_addr,
        const void* data,
        uint32_t height,
        uint32_t width,
        uint32_t element_size,
        uint32_t stride = 0,
        std::function<void()> callback = nullptr) = 0;

    // ========================================================================
    // Status Queries
    // ========================================================================

    /// Check if tile can accept more requests
    virtual bool can_accept() const = 0;

    /// Check if there are pending requests
    virtual bool has_pending() const = 0;

    /// Get number of pending requests
    virtual size_t pending_count() const = 0;

    /// Check if tile is ready (initialized)
    virtual bool is_ready() const = 0;

    // ========================================================================
    // Simulation Interface
    // ========================================================================

    /// Advance simulation by one cycle
    virtual void tick() = 0;

    /// Process until all pending requests complete
    virtual void drain() = 0;

    /// Reset tile to initial state
    virtual void reset() = 0;

    /// Get current simulation cycle
    virtual uint64_t current_cycle() const = 0;

    /// Set current simulation cycle
    virtual void set_cycle(uint64_t cycle) = 0;

    // ========================================================================
    // Configuration Queries
    // ========================================================================

    /// Get simulation fidelity level
    virtual SimulationFidelity fidelity() const = 0;

    /// Get full configuration
    virtual const L3TileConfig& config() const = 0;

    /// Get tile ID
    virtual uint32_t tile_id() const = 0;

    /// Get tile capacity in bytes
    virtual uint64_t capacity() const = 0;

    /// Get number of banks
    virtual uint8_t num_banks() const = 0;

    /// Get number of ports
    virtual uint8_t num_ports() const = 0;

    // ========================================================================
    // Bank State Queries (for TRANSACTIONAL and CYCLE_ACCURATE)
    // ========================================================================

    /// Bank state
    enum class BankState : uint8_t {
        IDLE,
        READING,
        WRITING
    };

    /// Get state of a specific bank
    virtual BankState get_bank_state(uint8_t bank) const = 0;

    /// Check if a specific bank is busy
    virtual bool is_bank_busy(uint8_t bank) const = 0;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get current statistics
    virtual const L3TileStatistics& stats() const = 0;

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

    // ========================================================================
    // Direct Memory Access (for testing/initialization)
    // ========================================================================

    /// Read data directly (bypasses timing model)
    virtual void read_direct(uint64_t addr, void* data, uint32_t size) const = 0;

    /// Write data directly (bypasses timing model)
    virtual void write_direct(uint64_t addr, const void* data, uint32_t size) = 0;
};

// ============================================================================
// L3 Tile Factory
// ============================================================================

/// Create an L3 tile based on configuration
///
/// @param config L3 tile configuration
/// @param tile_id Unique tile identifier
/// @return Unique pointer to L3 tile implementation
std::unique_ptr<IL3Tile> create_l3_tile(const L3TileConfig& config, uint32_t tile_id);

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert bank state to string
constexpr std::string_view to_string(IL3Tile::BankState state) {
    switch (state) {
        case IL3Tile::BankState::IDLE:    return "IDLE";
        case IL3Tile::BankState::READING: return "READING";
        case IL3Tile::BankState::WRITING: return "WRITING";
        default: return "UNKNOWN";
    }
}

} // namespace sw::kpu
