// patterns/dma/common/dma_configs.hpp
//
// Standard DMA configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/models/temporal/datamovement/cycle_accurate_dma_engine.hpp>
#include <sw/kpu/models/interfaces/memory_controller_interface.hpp>
#include <sw/kpu/fidelity/component_config.hpp>

namespace sw::kpu::patterns::dma {

// ============================================================================
// DMA Engine Configurations
// ============================================================================

/// Standard DMA configuration for pattern testing
/// 4 channels, moderate queue depth
inline CycleAccurateDMAConfig standard_dma_config() {
    CycleAccurateDMAConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.num_channels = 4;
    config.queue_depth_per_channel = 8;
    config.max_burst_size = 256;
    config.bandwidth_gbps = 100;
    config.memory_issue_interval = 1;
    config.noc_inject_interval = 1;
    config.noc_retry_delay = 4;
    return config;
}

/// High-throughput DMA configuration
/// 8 channels, deep queues for bandwidth testing
inline CycleAccurateDMAConfig high_throughput_dma_config() {
    CycleAccurateDMAConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.num_channels = 8;
    config.queue_depth_per_channel = 16;
    config.max_burst_size = 1024;
    config.bandwidth_gbps = 200;
    config.memory_issue_interval = 1;
    config.noc_inject_interval = 1;
    config.noc_retry_delay = 2;
    return config;
}

/// Minimal DMA configuration for simple tests
/// Single channel, shallow queue
inline CycleAccurateDMAConfig minimal_dma_config() {
    CycleAccurateDMAConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.num_channels = 1;
    config.queue_depth_per_channel = 4;
    config.max_burst_size = 64;
    config.bandwidth_gbps = 50;
    return config;
}

// ============================================================================
// Memory Controller Configurations
// ============================================================================

/// LPDDR5-6400 memory controller for mobile/embedded patterns
inline MemoryControllerConfig lpddr5_6400_config() {
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.technology = MemoryTechnology::LPDDR5;
    config.speed_mt_s = 6400;
    config.num_channels = 1;
    config.banks_per_channel = 8;
    config.bank_groups = 4;
    config.queue_depth = 32;
    config.row_bits = 16;
    config.col_bits = 10;
    return config;
}

/// DDR5-4800 memory controller for server patterns
inline MemoryControllerConfig ddr5_4800_config() {
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.technology = MemoryTechnology::DDR5;
    config.speed_mt_s = 4800;
    config.num_channels = 2;
    config.banks_per_channel = 32;
    config.bank_groups = 8;
    config.queue_depth = 64;
    config.row_bits = 17;
    config.col_bits = 10;
    return config;
}

/// HBM3 memory controller for high-bandwidth patterns
inline MemoryControllerConfig hbm3_config() {
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::CYCLE_ACCURATE;
    config.technology = MemoryTechnology::HBM3;
    config.speed_mt_s = 6400;
    config.num_channels = 8;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.queue_depth = 64;
    config.row_bits = 14;
    config.col_bits = 6;
    return config;
}

/// Behavioral memory controller for fast pattern iteration
inline MemoryControllerConfig behavioral_mc_config() {
    MemoryControllerConfig config;
    config.fidelity = SimulationFidelity::BEHAVIORAL;
    config.technology = MemoryTechnology::IDEAL;
    config.queue_depth = 256;
    return config;
}

// ============================================================================
// Common Constants
// ============================================================================

constexpr size_t CACHE_LINE_BYTES = 64;
constexpr size_t PAGE_SIZE_BYTES = 4096;
constexpr size_t PAGE_SIZE = 4096;  // Alias for convenience
constexpr size_t TILE_L3_SIZE_BYTES = 64 * 1024;  // 64KB L3 tile

// Element sizes
constexpr size_t SIZEOF_FLOAT = 4;
constexpr size_t SIZEOF_DOUBLE = 8;
constexpr size_t SIZEOF_HALF = 2;
constexpr size_t SIZEOF_BFLOAT16 = 2;
constexpr size_t SIZEOF_INT8 = 1;

// Common tile dimensions
constexpr size_t TILE_32 = 32;
constexpr size_t TILE_64 = 64;
constexpr size_t TILE_128 = 128;

} // namespace sw::kpu::patterns::dma
