// ============================================================================
// include/sw/kpu/harness/harness_config.hpp
// Configuration structures for KPU component test harnesses
//
// This header defines configuration structures for:
// - Individual component harnesses (DMA, BlockMover, Streamer)
// - Pipeline integration harness
// - Statistics and tracing options
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
// ============================================================================

#pragma once

#include <cstdint>
#include <string>
#include <optional>

#include <sw/kpu/fidelity/simulation_fidelity.hpp>
#include <sw/kpu/fidelity/component_config.hpp>

namespace sw::kpu::harness {

// ============================================================================
// Size Literals
// ============================================================================

constexpr uint64_t operator""_KB(unsigned long long x) { return x * 1024ULL; }
constexpr uint64_t operator""_MB(unsigned long long x) { return x * 1024ULL * 1024ULL; }
constexpr uint64_t operator""_GB(unsigned long long x) { return x * 1024ULL * 1024ULL * 1024ULL; }

// ============================================================================
// Stall Reason Enumeration
// ============================================================================

/// Reasons for component stalls (for statistics and tracing)
enum class StallReason : uint8_t {
    NONE,               // No stall
    CREDIT_STALL,       // Waiting for downstream buffer credit
    DATA_STALL,         // Waiting for upstream data arrival
    MEMORY_STALL,       // Waiting for memory controller
    BANK_CONFLICT,      // Memory bank conflict
    RESOURCE_CONFLICT,  // Shared resource conflict
    SYNC_BARRIER,       // Waiting at synchronization barrier
    UNKNOWN
};

/// Convert stall reason to string
constexpr std::string_view to_string(StallReason reason) {
    switch (reason) {
        case StallReason::NONE:              return "NONE";
        case StallReason::CREDIT_STALL:      return "CREDIT_STALL";
        case StallReason::DATA_STALL:        return "DATA_STALL";
        case StallReason::MEMORY_STALL:      return "MEMORY_STALL";
        case StallReason::BANK_CONFLICT:     return "BANK_CONFLICT";
        case StallReason::RESOURCE_CONFLICT: return "RESOURCE_CONFLICT";
        case StallReason::SYNC_BARRIER:      return "SYNC_BARRIER";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Component Enumeration
// ============================================================================

/// Component types for statistics and tracing
enum class HarnessComponent : uint8_t {
    DMA_ENGINE,
    BLOCK_MOVER,
    STREAMER,
    COMPUTE_FABRIC,
    MEMORY_CONTROLLER,
    L3_BUFFER,
    L2_BANK,
    L1_BUFFER
};

/// Convert component to string
constexpr std::string_view to_string(HarnessComponent comp) {
    switch (comp) {
        case HarnessComponent::DMA_ENGINE:        return "DMA_ENGINE";
        case HarnessComponent::BLOCK_MOVER:       return "BLOCK_MOVER";
        case HarnessComponent::STREAMER:          return "STREAMER";
        case HarnessComponent::COMPUTE_FABRIC:    return "COMPUTE_FABRIC";
        case HarnessComponent::MEMORY_CONTROLLER: return "MEMORY_CONTROLLER";
        case HarnessComponent::L3_BUFFER:         return "L3_BUFFER";
        case HarnessComponent::L2_BANK:           return "L2_BANK";
        case HarnessComponent::L1_BUFFER:         return "L1_BUFFER";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Transform Enumeration
// ============================================================================

/// Data transformations supported by BlockMover
enum class Transform : uint8_t {
    IDENTITY,           // No transformation
    TRANSPOSE,          // Row/column swap
    SWIZZLE_4x4,        // 4x4 block swizzle for bank conflict avoidance
    SWIZZLE_8x8,        // 8x8 block swizzle
    PACK_FP16,          // Convert FP32 -> FP16
    UNPACK_FP16         // Convert FP16 -> FP32
};

/// Convert transform to string
constexpr std::string_view to_string(Transform xform) {
    switch (xform) {
        case Transform::IDENTITY:    return "IDENTITY";
        case Transform::TRANSPOSE:   return "TRANSPOSE";
        case Transform::SWIZZLE_4x4: return "SWIZZLE_4x4";
        case Transform::SWIZZLE_8x8: return "SWIZZLE_8x8";
        case Transform::PACK_FP16:   return "PACK_FP16";
        case Transform::UNPACK_FP16: return "UNPACK_FP16";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Base Harness Configuration
// ============================================================================

/// Base configuration shared by all harnesses
struct HarnessConfig {
    /// Simulation fidelity level
    SimulationFidelity fidelity = SimulationFidelity::BEHAVIORAL;

    /// Enable event tracing
    bool enable_tracing = true;

    /// Enable statistics collection
    bool enable_stats = true;

    /// Clock frequency for timing calculations (GHz)
    double clock_frequency_ghz = 1.0;

    /// Maximum simulation cycles before timeout
    uint64_t max_cycles = 1'000'000;

    /// Trace output path (empty = no file output)
    std::string trace_output_path;

    /// Stats output path (empty = no file output)
    std::string stats_output_path;

    /// Verification level for runtime checking
    VerificationLevel verification = VerificationLevel::ASSERTIONS;

    /// Default constructor with reasonable defaults
    HarnessConfig() = default;
    virtual ~HarnessConfig() = default;
};

// ============================================================================
// DMA Harness Configuration
// ============================================================================

/// Configuration for DMA engine testing harness
struct DMAHarnessConfig : HarnessConfig {
    /// Number of DMA engines to instantiate
    uint32_t num_dma_engines = 1;

    /// DRAM size for simulation (bytes)
    uint64_t dram_size = 4_GB;

    /// L3 total capacity (bytes)
    uint64_t l3_size = 8_MB;

    /// Number of L3 buffers (tiles)
    uint32_t l3_buffer_count = 8;

    /// Size of each L3 buffer (bytes)
    uint64_t l3_buffer_size = 256_KB;

    /// Default tile size (16x16 FP16 elements)
    uint32_t tile_size = 16 * 16 * 2;

    /// Bandwidth per DMA engine (GB/s)
    double bandwidth_gbps = 100.0;

    /// Queue depth per DMA engine
    uint32_t queue_depth = 16;

    /// Fixed latency per transfer (for behavioral mode)
    uint32_t fixed_latency_cycles = 10;

    DMAHarnessConfig() = default;
};

// ============================================================================
// BlockMover Harness Configuration
// ============================================================================

/// Configuration for BlockMover testing harness
struct BlockMoverHarnessConfig : HarnessConfig {
    /// Number of BlockMovers (typically 2: ingress/egress)
    uint32_t num_block_movers = 2;

    /// Number of L3 buffers
    uint32_t l3_buffers = 8;

    /// Number of L2 banks
    uint32_t l2_banks = 16;

    /// Size of each L2 bank (bytes)
    uint64_t l2_bank_size = 64_KB;

    /// Enable data transformations (transpose, etc.)
    bool enable_transforms = true;

    /// Bandwidth L3->L2 per BlockMover (GB/s)
    double l3_l2_bandwidth_gbps = 128.0;

    /// Elements transferred per cycle
    uint32_t elements_per_cycle = 64;

    BlockMoverHarnessConfig() = default;
};

// ============================================================================
// Streamer Harness Configuration
// ============================================================================

/// Configuration for Streamer testing harness
struct StreamerHarnessConfig : HarnessConfig {
    /// Number of Streamers (typically 2: West/North edges)
    uint32_t num_streamers = 2;

    /// Number of L2 banks
    uint32_t l2_banks = 16;

    /// L1 pipeline depth (number of buffers)
    uint32_t l1_depth = 4;

    /// Systolic array size
    uint32_t systolic_size = 16;

    /// Enable vector engine operations
    bool enable_vector_engine = true;

    /// Bandwidth L2->L1 per Streamer (GB/s)
    double l2_l1_bandwidth_gbps = 256.0;

    /// Elements streamed per cycle
    uint32_t elements_per_cycle = 16;

    StreamerHarnessConfig() = default;
};

// ============================================================================
// Pipeline Harness Configuration
// ============================================================================

/// Configuration for full data movement pipeline harness
struct PipelineHarnessConfig : HarnessConfig {
    /// DMA configuration
    DMAHarnessConfig dma_config;

    /// BlockMover configuration
    BlockMoverHarnessConfig block_mover_config;

    /// Streamer configuration
    StreamerHarnessConfig streamer_config;

    /// Compute tile size
    uint32_t compute_tile_size = 16;

    /// Enable double buffering throughout pipeline
    bool double_buffering = true;

    /// Peak compute throughput (GFLOPS)
    double peak_gflops = 1000.0;

    /// Apply fidelity to all sub-configs
    void set_fidelity(SimulationFidelity f) {
        fidelity = f;
        dma_config.fidelity = f;
        block_mover_config.fidelity = f;
        streamer_config.fidelity = f;
    }

    /// Apply tracing to all sub-configs
    void set_tracing(bool enable) {
        enable_tracing = enable;
        dma_config.enable_tracing = enable;
        block_mover_config.enable_tracing = enable;
        streamer_config.enable_tracing = enable;
    }

    PipelineHarnessConfig() = default;
};

// ============================================================================
// Tile Identifier
// ============================================================================

/// Unique identifier for a tile in the system
struct TileID {
    uint32_t matrix_id = 0;     // Which matrix this tile belongs to
    uint32_t tile_row = 0;      // Row index within matrix tile grid
    uint32_t tile_col = 0;      // Column index within matrix tile grid

    bool operator==(const TileID& other) const {
        return matrix_id == other.matrix_id &&
               tile_row == other.tile_row &&
               tile_col == other.tile_col;
    }

    bool operator<(const TileID& other) const {
        if (matrix_id != other.matrix_id) return matrix_id < other.matrix_id;
        if (tile_row != other.tile_row) return tile_row < other.tile_row;
        return tile_col < other.tile_col;
    }

    /// Create string representation
    std::string to_string() const {
        return "M" + std::to_string(matrix_id) +
               "[" + std::to_string(tile_row) + "," + std::to_string(tile_col) + "]";
    }
};

/// Hash function for TileID (for use in unordered containers)
struct TileIDHash {
    std::size_t operator()(const TileID& id) const {
        std::size_t h1 = std::hash<uint32_t>{}(id.matrix_id);
        std::size_t h2 = std::hash<uint32_t>{}(id.tile_row);
        std::size_t h3 = std::hash<uint32_t>{}(id.tile_col);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// ============================================================================
// Matrix Coordinate Types
// ============================================================================

/// Coordinate within a tile grid
struct TileCoord {
    uint32_t row = 0;
    uint32_t col = 0;

    bool operator==(const TileCoord& other) const {
        return row == other.row && col == other.col;
    }
};

/// Matrix identifier with dimensions
struct MatrixID {
    uint32_t id = 0;
    uint32_t rows = 0;       // Total rows in matrix
    uint32_t cols = 0;       // Total columns in matrix
    uint32_t tile_rows = 0;  // Number of tile rows
    uint32_t tile_cols = 0;  // Number of tile columns

    /// Calculate tile count
    uint32_t tile_count() const { return tile_rows * tile_cols; }

    /// Check if coordinate is valid
    bool is_valid_tile(const TileCoord& coord) const {
        return coord.row < tile_rows && coord.col < tile_cols;
    }
};

} // namespace sw::kpu::harness
