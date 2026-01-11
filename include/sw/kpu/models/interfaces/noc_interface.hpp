// ============================================================================
// include/sw/kpu/models/interfaces/noc_interface.hpp
// Abstract NoC Interface for Runtime Selection
//
// This interface allows different NoC implementations to be used
// interchangeably, enabling:
// - Runtime selection between WormholeNoC and DataflowNoC
// - Easy testing with mock implementations
// - Gradual migration between implementations
// ============================================================================

#pragma once

#include <sw/kpu/models/temporal/datamovement/block_mover_isa.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace sw::kpu::noc {

// ============================================================================
// NoC Type Enumeration
// ============================================================================

enum class NoCType : uint8_t {
    WORMHOLE,     // Flit-level wormhole switching (current default)
    DATAFLOW,     // Unified dataflow model (new)
};

inline const char* to_string(NoCType type) {
    switch (type) {
        case NoCType::WORMHOLE: return "WORMHOLE";
        case NoCType::DATAFLOW: return "DATAFLOW";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Injection Result
// ============================================================================

enum class InjectResult : uint8_t {
    SUCCESS = 0,    // Tile injection started
    BUSY = 1,       // Injection buffer full, try again later
    INVALID = 2,    // Invalid parameters (bad router ID, etc.)
};

inline const char* to_string(InjectResult result) {
    switch (result) {
        case InjectResult::SUCCESS: return "SUCCESS";
        case InjectResult::BUSY: return "BUSY";
        case InjectResult::INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// NoC Statistics
// ============================================================================

struct NoCStats {
    uint64_t tiles_injected = 0;
    uint64_t tiles_delivered = 0;
    uint64_t total_flits = 0;
    uint64_t total_bytes = 0;
    uint64_t total_latency_cycles = 0;  // Sum of all tile latencies
    uint64_t max_latency_cycles = 0;

    double average_latency() const {
        return tiles_delivered > 0
            ? static_cast<double>(total_latency_cycles) / tiles_delivered
            : 0.0;
    }
};

// ============================================================================
// NoC Configuration
// ============================================================================

struct NoCConfig {
    uint8_t rows = 4;
    uint8_t cols = 4;
    size_t buffer_depth = 8;
    uint32_t flit_size_bytes = 64;
    uint32_t link_bandwidth_bytes_per_cycle = 64;

    // DMA edge configuration (for WormholeNoC compatibility)
    bool dma_on_north_edge = true;
    bool dma_on_south_edge = false;
    bool dma_on_east_edge = false;
    bool dma_on_west_edge = false;

    uint8_t num_routers() const { return rows * cols; }
    uint8_t router_id(uint8_t row, uint8_t col) const {
        return row * cols + col;
    }
};

// ============================================================================
// INoC - Abstract NoC Interface
// ============================================================================

class INoC {
public:
    virtual ~INoC() = default;

    // === Tile Injection ===

    /// Inject a tile from src_router to dst_router
    /// Returns SUCCESS if injection started, BUSY if buffer full
    virtual InjectResult inject_tile(uint8_t src_router, uint8_t dst_router,
                                     const TileDescriptor& tile, uint64_t cycle) = 0;

    /// Check if injection is possible at the given router
    virtual bool can_inject(uint8_t router_id) const = 0;

    // === Simulation ===

    /// Advance simulation by one cycle
    virtual void step(uint64_t cycle) = 0;

    /// Run until all in-flight tiles are delivered
    virtual void drain(uint64_t& cycle) = 0;

    /// Check if any tiles are still in flight
    virtual bool is_idle() const = 0;

    // === Delivery Callback ===

    /// Callback signature for tile delivery notification
    /// Parameters: tile descriptor, source router, destination router,
    ///             injection cycle, delivery cycle
    using DeliveryCallback = std::function<void(
        const TileDescriptor& tile,
        uint8_t src_router,
        uint8_t dst_router,
        uint64_t inject_cycle,
        uint64_t deliver_cycle
    )>;

    /// Set callback for when tiles are delivered to a specific router
    virtual void set_delivery_callback(uint8_t router_id, DeliveryCallback cb) = 0;

    /// Set a global callback for all deliveries (convenience)
    virtual void set_global_delivery_callback(DeliveryCallback cb) = 0;

    // === Configuration & Statistics ===

    virtual const NoCConfig& config() const = 0;
    virtual NoCStats stats() const = 0;
    virtual void reset_stats() = 0;

    // === Utility ===

    virtual uint8_t num_routers() const = 0;
    virtual NoCType type() const = 0;
    virtual std::string type_name() const { return to_string(type()); }
};

// ============================================================================
// Factory Function
// ============================================================================

/// Create a NoC instance of the specified type
std::unique_ptr<INoC> create_noc(NoCType type, const NoCConfig& config);

/// Create a NoC with default configuration
std::unique_ptr<INoC> create_noc(NoCType type);

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

using NoCPtr = std::unique_ptr<INoC>;

} // namespace sw::kpu::noc
