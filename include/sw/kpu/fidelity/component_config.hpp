// ============================================================================
// include/sw/kpu/fidelity/component_config.hpp
// Configuration structures for multi-fidelity simulation components
//
// See docs/SIMULATION_FIDELITY_FRAMEWORK.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <optional>
#include <sw/kpu/fidelity/simulation_fidelity.hpp>

namespace sw::kpu {

// ============================================================================
// Base Component Configuration
// ============================================================================

/// Base configuration shared by all simulator components
struct ComponentConfig {
    /// Simulation fidelity level
    SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL;

    /// Runtime verification level
    VerificationLevel verification = VerificationLevel::NONE;

    /// Enable tracing/observability
    bool enable_tracing = false;

    /// Enable statistics collection
    bool enable_statistics = true;

    /// Component instance ID (for multi-instance components)
    uint32_t instance_id = 0;

    ComponentConfig() = default;
    virtual ~ComponentConfig() = default;

    // Allow copying and moving
    ComponentConfig(const ComponentConfig&) = default;
    ComponentConfig& operator=(const ComponentConfig&) = default;
    ComponentConfig(ComponentConfig&&) = default;
    ComponentConfig& operator=(ComponentConfig&&) = default;
};

// ============================================================================
// Memory Controller Configuration
// ============================================================================

/// DRAM timing parameters (technology-specific defaults available)
struct MemoryTimingParams {
    // Core timing (cycles at memory clock)
    uint32_t tRCD = 14;      // Row-to-column delay
    uint32_t tRP = 14;       // Row precharge time
    uint32_t tRAS = 28;      // Row active time (minimum)
    uint32_t tRC = 42;       // Row cycle time (tRAS + tRP)
    uint32_t tCL = 14;       // CAS latency (read)
    uint32_t tWL = 8;        // CAS write latency
    uint32_t tWR = 24;       // Write recovery time
    uint32_t tRTP = 6;       // Read to precharge

    // Bank/bank group timing
    uint32_t tRRD_L = 6;     // ACT to ACT (same bank group)
    uint32_t tRRD_S = 4;     // ACT to ACT (different bank group)
    uint32_t tCCD_L = 6;     // CAS to CAS (same bank group)
    uint32_t tCCD_S = 4;     // CAS to CAS (different bank group)

    // Turnaround timing
    uint32_t tWTR_L = 10;    // Write to read (same bank group)
    uint32_t tWTR_S = 4;     // Write to read (different bank group)
    uint32_t tRTW = 14;      // Read to write (bus turnaround)

    // Burst timing
    uint32_t tBurst = 8;     // Burst length in cycles

    // Refresh timing
    uint32_t tRFC = 280;     // Refresh cycle time
    uint32_t tREFI = 3900;   // Refresh interval

    // Activate window
    uint32_t tFAW = 24;      // Four activate window

    // Behavioral model parameters
    uint32_t fixed_read_latency = 100;   // For BEHAVIORAL fidelity
    uint32_t fixed_write_latency = 100;  // For BEHAVIORAL fidelity

    // Transactional model parameters
    uint32_t mean_read_latency = 80;     // For TRANSACTIONAL fidelity
    uint32_t mean_write_latency = 90;
    uint32_t latency_variance = 20;      // Standard deviation

    // Transactional model: page scenario factors (calibrated from cycle-accurate)
    // These multiply the mean latency to get scenario-specific latency
    double page_hit_factor = 0.7;        // < 1.0: page hits are faster than average
    double page_empty_factor = 1.0;      // ~1.0: page empty is baseline
    double page_conflict_factor = 1.3;   // > 1.0: page conflicts are slower

    // Transactional model: per-scenario latencies (preferred, calibrated directly)
    // When non-zero, these are used instead of mean_latency * factor
    uint32_t page_hit_latency = 0;       // Latency for page hits (row already open)
    uint32_t page_empty_latency = 0;     // Latency for page empty (bank idle)
    uint32_t page_conflict_latency = 0;  // Latency for page conflicts (different row)
};

/// Memory controller configuration
struct MemoryControllerConfig : ComponentConfig {
    /// Memory technology standard
    MemoryTechnology technology = MemoryTechnology::IDEAL;

    /// Data rate in MT/s (megatransfers per second)
    uint32_t speed_mt_s = 6400;

    /// Total memory capacity in GB
    uint32_t capacity_gb = 1;

    /// Number of memory controllers
    uint8_t num_controllers = 1;

    /// Channels per controller
    uint8_t channels_per_controller = 1;

    /// Total number of channels (derived or set directly)
    uint8_t num_channels = 1;

    /// Banks per channel
    uint8_t banks_per_channel = 16;

    /// Bank groups per channel
    uint8_t bank_groups = 4;

    /// Request queue depth
    uint32_t queue_depth = 32;

    /// Address mapping
    uint32_t row_bits = 16;
    uint32_t col_bits = 10;
    uint32_t bank_bits = 4;
    uint32_t channel_bits = 1;

    /// Timing parameters (auto-populated from technology if not set)
    MemoryTimingParams timing;

    /// Get default timing for technology and speed grade
    static MemoryTimingParams get_default_timing(MemoryTechnology tech, uint32_t speed_mt_s);
};

// ============================================================================
// DMA Engine Configuration
// ============================================================================

/// DMA engine configuration
struct DMAEngineConfig : ComponentConfig {
    /// Number of DMA engines
    uint32_t num_engines = 1;

    /// Channels per engine
    uint32_t channels_per_engine = 8;

    /// Total number of DMA channels (derived or set directly)
    uint32_t num_channels = 8;

    /// Maximum burst size in bytes
    uint32_t max_burst_size = 256;

    /// Bandwidth in GB/s (for transactional model)
    uint32_t bandwidth_gbps = 100;

    /// Queue depth per channel
    uint32_t queue_depth_per_channel = 16;

    /// Behavioral model: fixed transfer latency per burst
    uint32_t fixed_latency_per_burst = 10;
};

// ============================================================================
// Storage Component Configurations
// ============================================================================

/// L3 tile configuration
struct L3TileConfig : ComponentConfig {
    /// Number of L3 tiles
    uint32_t num_tiles = 1;

    /// Capacity per tile in KB
    uint32_t capacity_kb = 256;

    /// Number of banks per tile
    uint8_t num_banks = 8;

    /// Number of ports per tile
    uint8_t num_ports = 4;

    /// Bank width in bytes
    uint32_t bank_width_bytes = 64;

    /// Access latency (for transactional/cycle-accurate)
    uint32_t access_latency_cycles = 4;
};

/// L2 bank configuration
struct L2BankConfig : ComponentConfig {
    /// Capacity in KB
    uint32_t capacity_kb = 64;

    /// Number of ports
    uint8_t num_ports = 2;

    /// Access latency
    uint32_t access_latency_cycles = 2;
};

/// L1 buffer configuration
struct L1BufferConfig : ComponentConfig {
    /// Capacity in KB
    uint32_t capacity_kb = 8;

    /// Double-buffering support
    bool double_buffered = true;

    /// Access latency
    uint32_t access_latency_cycles = 1;
};

// ============================================================================
// Data Movement Configurations
// ============================================================================

/// Block mover configuration
struct BlockMoverConfig : ComponentConfig {
    /// Maximum block dimensions
    uint32_t max_block_height = 256;
    uint32_t max_block_width = 256;

    /// Element sizes supported (bytes)
    uint8_t min_element_size = 1;
    uint8_t max_element_size = 8;

    /// Bandwidth in elements per cycle
    uint32_t elements_per_cycle = 64;
};

/// Streamer configuration
struct StreamerConfig : ComponentConfig {
    /// Maximum streaming dimensions
    uint32_t max_matrix_height = 4096;
    uint32_t max_matrix_width = 4096;

    /// Streaming bandwidth in elements per cycle
    uint32_t elements_per_cycle = 16;

    /// Prefetch depth
    uint32_t prefetch_depth = 4;
};

// ============================================================================
// Compute Configurations
// ============================================================================

/// Compute fabric configuration
struct ComputeFabricConfig : ComponentConfig {
    /// Compute technology
    ComputeTechnology technology = ComputeTechnology::IDEAL;

    /// Number of compute tiles
    uint32_t num_tiles = 1;

    /// Array dimensions (for systolic arrays)
    uint32_t array_rows = 16;
    uint32_t array_cols = 16;

    /// Throughput in MACs per cycle (for behavioral/transactional)
    uint32_t macs_per_cycle = 256;

    /// Pipeline depth (for cycle-accurate)
    uint32_t pipeline_depth = 4;
};

// ============================================================================
// Interconnect Configurations
// ============================================================================

/// NoC configuration
struct NoCConfig : ComponentConfig {
    /// Interconnect technology/topology
    InterconnectTechnology technology = InterconnectTechnology::IDEAL;

    /// Mesh dimensions (for MESH_2D, TORUS_2D)
    uint32_t mesh_rows = 4;
    uint32_t mesh_cols = 4;

    /// Link bandwidth in bytes per cycle
    uint32_t link_bandwidth = 64;

    /// Router latency in cycles
    uint32_t router_latency = 1;

    /// Virtual channels
    uint8_t num_virtual_channels = 4;
};

// ============================================================================
// Top-Level Simulator Configuration
// ============================================================================

/// Complete simulator configuration
struct SimulatorConfig {
    // === Global Defaults ===
    SimulationFidelity default_fidelity = SimulationFidelity::BEHAVIORAL;
    VerificationLevel default_verification = VerificationLevel::NONE;
    bool default_tracing = false;

    // === Component Counts ===
    uint32_t num_memory_controllers = 1;
    uint32_t num_dma_engines = 4;
    uint32_t num_l3_tiles = 4;
    uint32_t num_l2_banks = 16;
    uint32_t l2_capacity_kb = 64;  // L2 capacity per bank
    uint32_t num_l1_buffers = 0;  // Auto-computed if 0
    uint32_t num_block_movers = 4;
    uint32_t num_streamers = 4;
    uint32_t num_compute_tiles = 16;

    // === Per-Component Configurations (optional overrides) ===
    std::optional<MemoryControllerConfig> memory_controller;
    std::optional<DMAEngineConfig> dma_engine;
    std::optional<L3TileConfig> l3_tile;
    std::optional<L2BankConfig> l2_bank;
    std::optional<L1BufferConfig> l1_buffer;
    std::optional<BlockMoverConfig> block_mover;
    std::optional<StreamerConfig> streamer;
    std::optional<ComputeFabricConfig> compute_fabric;
    std::optional<NoCConfig> noc;

    // === Convenience Methods ===

    /// Set all components to the same fidelity level
    void set_fidelity(SimulationFidelity fidelity) {
        default_fidelity = fidelity;
        if (memory_controller) memory_controller->fidelity = fidelity;
        if (dma_engine) dma_engine->fidelity = fidelity;
        if (l3_tile) l3_tile->fidelity = fidelity;
        if (l2_bank) l2_bank->fidelity = fidelity;
        if (l1_buffer) l1_buffer->fidelity = fidelity;
        if (block_mover) block_mover->fidelity = fidelity;
        if (streamer) streamer->fidelity = fidelity;
        if (compute_fabric) compute_fabric->fidelity = fidelity;
        if (noc) noc->fidelity = fidelity;
    }

    /// Set fidelity for memory subsystem components
    void set_memory_fidelity(SimulationFidelity fidelity) {
        if (memory_controller) memory_controller->fidelity = fidelity;
        if (l3_tile) l3_tile->fidelity = fidelity;
        if (l2_bank) l2_bank->fidelity = fidelity;
        if (l1_buffer) l1_buffer->fidelity = fidelity;
    }

    /// Set fidelity for data movement components
    void set_data_movement_fidelity(SimulationFidelity fidelity) {
        if (dma_engine) dma_engine->fidelity = fidelity;
        if (block_mover) block_mover->fidelity = fidelity;
        if (streamer) streamer->fidelity = fidelity;
    }

    /// Set fidelity for compute components
    void set_compute_fidelity(SimulationFidelity fidelity) {
        if (compute_fabric) compute_fabric->fidelity = fidelity;
    }

    /// Set fidelity for interconnect
    void set_interconnect_fidelity(SimulationFidelity fidelity) {
        if (noc) noc->fidelity = fidelity;
    }

    /// Enable tracing for all components
    void enable_all_tracing(bool enable = true) {
        default_tracing = enable;
        if (memory_controller) memory_controller->enable_tracing = enable;
        if (dma_engine) dma_engine->enable_tracing = enable;
        if (l3_tile) l3_tile->enable_tracing = enable;
        if (l2_bank) l2_bank->enable_tracing = enable;
        if (l1_buffer) l1_buffer->enable_tracing = enable;
        if (block_mover) block_mover->enable_tracing = enable;
        if (streamer) streamer->enable_tracing = enable;
        if (compute_fabric) compute_fabric->enable_tracing = enable;
        if (noc) noc->enable_tracing = enable;
    }

    /// Get effective config for memory controller (with defaults applied)
    MemoryControllerConfig get_memory_controller_config() const {
        if (memory_controller) {
            return *memory_controller;
        }
        MemoryControllerConfig config;
        config.fidelity = default_fidelity;
        config.verification = default_verification;
        config.enable_tracing = default_tracing;
        return config;
    }

    /// Get effective config for DMA engine (with defaults applied)
    DMAEngineConfig get_dma_engine_config() const {
        if (dma_engine) {
            return *dma_engine;
        }
        DMAEngineConfig config;
        config.fidelity = default_fidelity;
        config.verification = default_verification;
        config.enable_tracing = default_tracing;
        return config;
    }

    // Similar getters for other component types...
};

} // namespace sw::kpu
