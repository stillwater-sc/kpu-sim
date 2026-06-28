// ============================================================================
// include/sw/kpu/models/interfaces/dma/dma_engine_interface.hpp
// Abstract interface for DMA engines at all fidelity levels
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
// DMA Request Types
// ============================================================================

/// Memory region type for DMA transfers
enum class DMAMemoryType : uint8_t {
    HOST_MEMORY,      // Host DDR
    DEVICE_MEMORY,    // Device main memory (HBM/GDDR)
    L3_TILE,          // L3 cache tile (on-chip)
    L2_BANK,          // L2 bank (on-chip)
    SCRATCHPAD        // Scratchpad memory
};

constexpr std::string_view to_string(DMAMemoryType type) {
    switch (type) {
        case DMAMemoryType::HOST_MEMORY:   return "HOST";
        case DMAMemoryType::DEVICE_MEMORY: return "DEVICE";
        case DMAMemoryType::L3_TILE:       return "L3";
        case DMAMemoryType::L2_BANK:       return "L2";
        case DMAMemoryType::SCRATCHPAD:    return "SPAD";
        default: return "UNKNOWN";
    }
}

/// DMA transfer descriptor
struct DMATransfer {
    // Source
    DMAMemoryType src_type = DMAMemoryType::HOST_MEMORY;
    uint32_t src_id = 0;        // Memory region ID (tile ID, bank ID, etc.)
    uint64_t src_addr = 0;      // Address within the region

    // Destination
    DMAMemoryType dst_type = DMAMemoryType::DEVICE_MEMORY;
    uint32_t dst_id = 0;
    uint64_t dst_addr = 0;

    // Transfer parameters
    uint32_t size = 0;          // Bytes to transfer
    uint32_t stride = 0;        // Stride for 2D transfers (0 = contiguous)
    uint32_t count = 1;         // Number of strided copies (1 = 1D transfer)

    // Priority and ordering
    uint8_t priority = 0;       // Higher = more urgent
    bool ordered = true;        // Must complete in order

    // User data
    uint64_t user_tag = 0;      // User-defined tag for identification
};

// ============================================================================
// DMA Engine Statistics
// ============================================================================

/// Common statistics interface for all DMA engines
struct DMAEngineStatistics {
    // Transfer counts
    uint64_t transfers_completed = 0;
    uint64_t transfers_pending = 0;
    uint64_t transfers_failed = 0;

    // Data volume
    uint64_t bytes_transferred = 0;
    uint64_t bytes_host_to_device = 0;
    uint64_t bytes_device_to_host = 0;
    uint64_t bytes_device_to_device = 0;

    // Latency statistics
    uint64_t total_latency_cycles = 0;
    uint64_t min_latency = UINT64_MAX;
    uint64_t max_latency = 0;

    // Utilization
    uint64_t busy_cycles = 0;
    uint64_t idle_cycles = 0;
    uint64_t stall_cycles = 0;  // Waiting for memory

    // Per-channel stats (optional)
    struct ChannelStats {
        uint64_t transfers = 0;
        uint64_t bytes = 0;
        uint64_t busy_cycles = 0;
    };
    std::vector<ChannelStats> channel_stats;

    // Derived metrics
    double avg_latency() const {
        return transfers_completed > 0
            ? static_cast<double>(total_latency_cycles) / static_cast<double>(transfers_completed)
            : 0.0;
    }

    double utilization() const {
        uint64_t total = busy_cycles + idle_cycles;
        return total > 0 ? static_cast<double>(busy_cycles) / static_cast<double>(total) : 0.0;
    }

    double bandwidth_gbps(double clock_ghz) const {
        uint64_t total_cycles = busy_cycles + idle_cycles;
        if (total_cycles == 0) return 0.0;
        double seconds = static_cast<double>(total_cycles) / (clock_ghz * 1e9);
        return (static_cast<double>(bytes_transferred) / 1e9) / seconds;
    }

    void reset() {
        transfers_completed = transfers_pending = transfers_failed = 0;
        bytes_transferred = bytes_host_to_device = bytes_device_to_host = 0;
        bytes_device_to_device = 0;
        total_latency_cycles = 0;
        min_latency = UINT64_MAX;
        max_latency = 0;
        busy_cycles = idle_cycles = stall_cycles = 0;
        for (auto& cs : channel_stats) {
            cs.transfers = cs.bytes = cs.busy_cycles = 0;
        }
    }
};

// ============================================================================
// DMA Engine Interface
// ============================================================================

/// Abstract interface for DMA engines at all fidelity levels
///
/// This interface provides a common API that is implemented by:
/// - BehavioralDMAEngine (instant/fixed latency)
/// - TransactionalDMAEngine (bandwidth-limited)
/// - CycleAccurateDMAEngine (full channel arbitration)
///
/// All implementations guarantee:
/// 1. Functional correctness (data is transferred correctly)
/// 2. Callback semantics (callbacks are invoked when transfer completes)
/// 3. Statistics collection (if enabled)
/// 4. Tracing support (if enabled)
class IDMAEngine {
public:
    virtual ~IDMAEngine() = default;

    // ========================================================================
    // Request Interface
    // ========================================================================

    /// Submit a DMA transfer
    ///
    /// @param transfer Transfer descriptor
    /// @param callback Optional callback invoked when transfer completes
    /// @return Transfer ID if accepted, nullopt if queue is full
    ///
    /// Behavior by fidelity:
    /// - BEHAVIORAL: Completes immediately, callback invoked before return
    /// - TRANSACTIONAL: Bandwidth-limited, callback after estimated delay
    /// - CYCLE_ACCURATE: Full channel arbitration, precise timing
    virtual std::optional<uint64_t> submit(
        const DMATransfer& transfer,
        std::function<void()> callback = nullptr) = 0;

    /// Convenience: submit a simple 1D transfer
    virtual std::optional<uint64_t> submit(
        DMAMemoryType src_type, uint32_t src_id, uint64_t src_addr,
        DMAMemoryType dst_type, uint32_t dst_id, uint64_t dst_addr,
        uint32_t size,
        std::function<void()> callback = nullptr);

    /// Check if engine can accept more transfers
    virtual bool can_accept() const = 0;

    /// Check if there are pending transfers
    virtual bool has_pending() const = 0;

    /// Get number of pending transfers
    virtual size_t pending_count() const = 0;

    /// Wait for a specific transfer to complete (by ID)
    virtual bool wait_for(uint64_t transfer_id) = 0;

    /// Cancel a pending transfer (if possible)
    virtual bool cancel(uint64_t transfer_id) = 0;

    // ========================================================================
    // Simulation Interface
    // ========================================================================

    /// Advance simulation by one cycle
    virtual void tick() = 0;

    /// Process until all pending transfers complete
    virtual void drain() = 0;

    /// Reset engine to initial state
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
    virtual const DMAEngineConfig& config() const = 0;

    /// Get number of DMA channels
    virtual uint32_t num_channels() const = 0;

    /// Get channel state
    enum class ChannelState : uint8_t {
        IDLE,           // Ready for new transfer
        BUSY,           // Transfer in progress
        STALLED,        // Waiting for memory
        ERROR           // Error state
    };

    virtual ChannelState get_channel_state(uint32_t channel) const = 0;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get current statistics
    virtual const DMAEngineStatistics& stats() const = 0;

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
    // Memory Interface (for data access)
    // ========================================================================

    /// Callback for memory read operations
    using ReadCallback = std::function<void(const void* data, uint32_t size)>;

    /// Callback for memory write operations
    using WriteCallback = std::function<void(void* buffer, uint32_t size)>;

    /// Set read callback for a memory type
    virtual void set_read_callback(DMAMemoryType type, ReadCallback callback) = 0;

    /// Set write callback for a memory type
    virtual void set_write_callback(DMAMemoryType type, WriteCallback callback) = 0;
};

// ============================================================================
// DMA Engine Factory
// ============================================================================

/// Create a DMA engine based on configuration
///
/// The factory selects the appropriate implementation based on fidelity level.
///
/// @param config DMA engine configuration
/// @return Unique pointer to DMA engine implementation
std::unique_ptr<IDMAEngine> create_dma_engine(const DMAEngineConfig& config);

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert channel state to string
constexpr std::string_view to_string(IDMAEngine::ChannelState state) {
    switch (state) {
        case IDMAEngine::ChannelState::IDLE:    return "IDLE";
        case IDMAEngine::ChannelState::BUSY:    return "BUSY";
        case IDMAEngine::ChannelState::STALLED: return "STALLED";
        case IDMAEngine::ChannelState::ERROR:   return "ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace sw::kpu
