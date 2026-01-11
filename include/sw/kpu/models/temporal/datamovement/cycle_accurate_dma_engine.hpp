// ============================================================================
// include/sw/kpu/models/temporal/datamovement/cycle_accurate_dma_engine.hpp
// Cycle-accurate DMA engine with memory and NoC integration
//
// This engine provides full cycle-accurate simulation with:
// - Per-channel state machines
// - Memory controller backpressure handling
// - NoC injection with congestion feedback
// - Accurate timing for DMA→Memory→NoC data path
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include <sw/kpu/models/interfaces/dma_engine_interface.hpp>
#include <sw/kpu/models/interfaces/memory_controller_interface.hpp>
#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>
#include <sw/kpu/models/interfaces/noc_interface.hpp>

namespace sw::kpu {

// ============================================================================
// DMA Channel State Machine
// ============================================================================

/// State of a DMA channel during transfer
enum class DMAChannelState : uint8_t {
    IDLE,                   // Ready for new transfer
    WAITING_MEMORY_READ,    // Issued memory read, awaiting response
    MEMORY_READ_COMPLETE,   // Data ready, preparing for NoC injection
    WAITING_NOC_INJECT,     // Retry NoC injection (got BUSY)
    NOC_INJECTING,          // Active NoC flit transfer in progress
    WAITING_MEMORY_WRITE,   // Issued memory write (for store operations)
    COMPLETE,               // Transfer done, invoking callback
    STALLED_MEMORY_FULL,    // Memory queue backpressure
    STALLED_NOC_FULL,       // NoC injection backpressure
    ERROR                   // Error state
};

/// Convert channel state to string
constexpr std::string_view to_string(DMAChannelState state) {
    switch (state) {
        case DMAChannelState::IDLE:                 return "IDLE";
        case DMAChannelState::WAITING_MEMORY_READ:  return "WAITING_MEMORY_READ";
        case DMAChannelState::MEMORY_READ_COMPLETE: return "MEMORY_READ_COMPLETE";
        case DMAChannelState::WAITING_NOC_INJECT:   return "WAITING_NOC_INJECT";
        case DMAChannelState::NOC_INJECTING:        return "NOC_INJECTING";
        case DMAChannelState::WAITING_MEMORY_WRITE: return "WAITING_MEMORY_WRITE";
        case DMAChannelState::COMPLETE:             return "COMPLETE";
        case DMAChannelState::STALLED_MEMORY_FULL:  return "STALLED_MEMORY_FULL";
        case DMAChannelState::STALLED_NOC_FULL:     return "STALLED_NOC_FULL";
        case DMAChannelState::ERROR:                return "ERROR";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// DMA Channel Structure
// ============================================================================

/// Per-channel state for cycle-accurate DMA
struct DMAChannel {
    uint32_t channel_id = 0;
    DMAChannelState state = DMAChannelState::IDLE;

    // Active transfer tracking
    uint64_t transfer_id = 0;
    DMATransfer request;
    uint64_t submit_cycle = 0;

    // Data buffer for staging
    std::vector<uint8_t> data_buffer;

    // Memory request tracking
    std::optional<uint64_t> memory_request_id;
    bool memory_complete = false;

    // NoC injection tracking
    uint8_t target_router_id = 0;
    TileDescriptor tile;
    bool noc_inject_started = false;
    uint16_t packet_seq = 0;

    // Timing constraints
    uint64_t last_memory_issue_cycle = 0;
    uint64_t last_noc_inject_cycle = 0;

    // Backpressure tracking
    uint32_t memory_retry_count = 0;
    uint32_t noc_retry_count = 0;
    uint32_t max_retries = 1000;  // Prevent infinite loops

    // Completion callback
    std::function<void()> callback;

    /// Reset channel to idle state
    void reset() {
        state = DMAChannelState::IDLE;
        transfer_id = 0;
        request = DMATransfer{};
        submit_cycle = 0;
        data_buffer.clear();
        memory_request_id.reset();
        memory_complete = false;
        target_router_id = 0;
        tile = TileDescriptor{};
        noc_inject_started = false;
        packet_seq = 0;
        memory_retry_count = 0;
        noc_retry_count = 0;
        callback = nullptr;
    }

    /// Check if channel is busy
    bool is_busy() const {
        return state != DMAChannelState::IDLE && state != DMAChannelState::ERROR;
    }
};

// ============================================================================
// Cycle-Accurate DMA Configuration
// ============================================================================

/// Extended configuration for cycle-accurate DMA engine
struct CycleAccurateDMAConfig : DMAEngineConfig {
    // Memory interface timing
    uint32_t memory_issue_interval = 1;        // Min cycles between memory requests
    uint32_t memory_response_buffer_depth = 8; // Max outstanding memory requests

    // NoC interface timing
    uint32_t noc_inject_interval = 1;          // Min cycles between NoC injections
    uint32_t noc_retry_delay = 4;              // Cycles before retry on BUSY

    // Channel arbitration
    enum class ArbitrationPolicy : uint8_t {
        ROUND_ROBIN,    // Simple round-robin
        PRIORITY,       // Higher priority channels first
        OLDEST_FIRST    // Oldest pending transfer first
    };
    ArbitrationPolicy arbitration = ArbitrationPolicy::ROUND_ROBIN;

    // Default constructor sets fidelity
    CycleAccurateDMAConfig() {
        fidelity = SimulationFidelity::CYCLE_ACCURATE;
    }
};

// ============================================================================
// Tile Mapper Function Type
// ============================================================================

/// Function type for converting DMA transfer to TileDescriptor
/// Parameters: source address, destination address, size, destination router ID
using TileMapperFunc = std::function<TileDescriptor(
    const DMATransfer& transfer,
    uint8_t& dst_router_id)>;

// ============================================================================
// Cycle-Accurate DMA Engine
// ============================================================================

/// Cycle-accurate DMA engine with memory and NoC integration
///
/// This is the highest fidelity DMA simulation, providing:
/// - Per-channel state machine with accurate timing
/// - Memory controller integration with backpressure
/// - NoC injection with congestion handling
/// - Full statistics and tracing
///
/// Use for:
/// - Detailed performance analysis
/// - Protocol verification
/// - Latency-critical optimizations
/// - Memory subsystem validation
class CycleAccurateDMAEngine : public IDMAEngine {
public:
    explicit CycleAccurateDMAEngine(const CycleAccurateDMAConfig& config);
    ~CycleAccurateDMAEngine() override = default;

    // Disable copying, allow moving
    CycleAccurateDMAEngine(const CycleAccurateDMAEngine&) = delete;
    CycleAccurateDMAEngine& operator=(const CycleAccurateDMAEngine&) = delete;
    CycleAccurateDMAEngine(CycleAccurateDMAEngine&&) = default;
    CycleAccurateDMAEngine& operator=(CycleAccurateDMAEngine&&) = default;

    // ========================================================================
    // External Bindings
    // ========================================================================

    /// Bind to a memory controller for data access
    /// @param controller Pointer to memory controller (must outlive this engine)
    void bind_memory_controller(IMemoryController* controller);

    /// Bind to NoC for tile injection
    /// @param noc Pointer to NoC (must outlive this engine)
    /// @param edge_router_id Router ID for DMA injection
    void bind_noc(noc::INoC* noc, uint8_t edge_router_id);

    /// Set tile mapper function for address-to-tile conversion
    /// @param mapper Function that converts DMATransfer to TileDescriptor
    void set_tile_mapper(TileMapperFunc mapper);

    /// Check if memory controller is bound
    bool has_memory_binding() const { return memory_controller_ != nullptr; }

    /// Check if NoC is bound
    bool has_noc_binding() const { return noc_ != nullptr; }

    /// Get bound router ID
    uint8_t bound_router_id() const { return bound_router_id_; }

    // ========================================================================
    // Request Interface (IDMAEngine)
    // ========================================================================

    std::optional<uint64_t> submit(
        const DMATransfer& transfer,
        std::function<void()> callback = nullptr) override;

    bool can_accept() const override;
    bool has_pending() const override { return pending_count_ > 0; }
    size_t pending_count() const override { return pending_count_; }

    bool wait_for(uint64_t transfer_id) override;
    bool cancel(uint64_t transfer_id) override;

    // ========================================================================
    // Simulation Interface (IDMAEngine)
    // ========================================================================

    void tick() override;
    void drain() override;
    void reset() override;

    uint64_t current_cycle() const override { return current_cycle_; }
    void set_cycle(uint64_t cycle) override { current_cycle_ = cycle; }

    // ========================================================================
    // Configuration Queries (IDMAEngine)
    // ========================================================================

    SimulationFidelity fidelity() const override {
        return SimulationFidelity::CYCLE_ACCURATE;
    }

    const DMAEngineConfig& config() const override { return config_; }
    uint32_t num_channels() const override { return config_.num_channels; }

    ChannelState get_channel_state(uint32_t channel) const override;

    /// Get detailed channel state (cycle-accurate specific)
    DMAChannelState get_detailed_channel_state(uint32_t channel) const;

    // ========================================================================
    // Statistics (IDMAEngine)
    // ========================================================================

    const DMAEngineStatistics& stats() const override { return stats_; }
    void reset_stats() override { stats_.reset(); }

    // ========================================================================
    // Observability (IDMAEngine)
    // ========================================================================

    void enable_tracing(bool enable) override { tracing_enabled_ = enable; }
    bool tracing_enabled() const override { return tracing_enabled_; }
    void set_resource_tracker(sw::trace::ResourceTracker* tracker) override {
        resource_tracker_ = tracker;
    }

    // ========================================================================
    // Memory Interface (IDMAEngine)
    // ========================================================================

    void set_read_callback(DMAMemoryType type, ReadCallback callback) override {
        read_callbacks_[static_cast<size_t>(type)] = callback;
    }

    void set_write_callback(DMAMemoryType type, WriteCallback callback) override {
        write_callbacks_[static_cast<size_t>(type)] = callback;
    }

private:
    // Configuration
    CycleAccurateDMAConfig config_;

    // State
    uint64_t current_cycle_ = 0;
    uint64_t next_transfer_id_ = 1;
    uint16_t next_packet_seq_ = 1;
    size_t pending_count_ = 0;

    // Channels
    std::vector<DMAChannel> channels_;
    uint32_t next_channel_rr_ = 0;  // Round-robin pointer

    // External bindings
    IMemoryController* memory_controller_ = nullptr;
    noc::INoC* noc_ = nullptr;
    uint8_t bound_router_id_ = 0;

    // Tile mapper
    TileMapperFunc tile_mapper_;

    // Pending transfer queue (when all channels are busy)
    struct PendingTransfer {
        uint64_t transfer_id;
        DMATransfer request;
        std::function<void()> callback;
    };
    std::deque<PendingTransfer> pending_queue_;

    // Transfer completion tracking
    std::map<uint64_t, uint32_t> transfer_to_channel_;

    // Statistics
    DMAEngineStatistics stats_;

    // Tracing
    bool tracing_enabled_ = false;
    sw::trace::ResourceTracker* resource_tracker_ = nullptr;

    // Memory callbacks (unused but required by interface)
    static constexpr size_t NUM_MEMORY_TYPES = 5;
    std::array<ReadCallback, NUM_MEMORY_TYPES> read_callbacks_;
    std::array<WriteCallback, NUM_MEMORY_TYPES> write_callbacks_;

    // ========================================================================
    // State Machine Methods
    // ========================================================================

    /// Process a single channel
    void process_channel(uint32_t ch_id);

    /// Try to assign a pending transfer to an idle channel
    void try_assign_pending_transfer(uint32_t ch_id);

    /// Issue memory read request
    void issue_memory_read(uint32_t ch_id);

    /// Process waiting for memory read completion
    void process_waiting_memory_read(uint32_t ch_id);

    /// Attempt NoC tile injection
    void attempt_noc_injection(uint32_t ch_id);

    /// Process NoC injection in progress
    void process_noc_injecting(uint32_t ch_id);

    /// Issue memory write request (for store operations)
    void issue_memory_write(uint32_t ch_id);

    /// Process waiting for memory write completion
    void process_waiting_memory_write(uint32_t ch_id);

    /// Complete transfer and invoke callback
    void complete_transfer(uint32_t ch_id);

    /// Handle transfer error
    void handle_error(uint32_t ch_id, const std::string& message);

    // ========================================================================
    // Arbitration
    // ========================================================================

    /// Select channel for new transfer assignment
    std::optional<uint32_t> select_idle_channel();

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Default tile mapper (if none provided)
    TileDescriptor default_tile_mapper(const DMATransfer& transfer,
                                        uint8_t& dst_router_id);

    /// Update statistics for completed transfer
    void update_stats(const DMAChannel& channel);

    /// Update per-channel statistics
    void update_channel_stats();

    /// Trace transfer event
    void trace_event(const std::string& event, uint32_t ch_id,
                     uint64_t transfer_id);
};

} // namespace sw::kpu
