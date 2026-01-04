// patterns/common/lpddr5_configs.hpp
//
// Standard LPDDR5 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/components/lpddr5_memory_controller.hpp>

namespace sw::kpu::patterns {

using namespace sw::kpu::lpddr5;

// ============================================================================
// Standard LPDDR5-6400 Configurations
// ============================================================================

/// Single-channel LPDDR5-6400 configuration
/// - 1 channel
/// - 16 banks (4 bank groups × 4 banks per group)
/// - 6400 MT/s (3200 MHz clock)
/// - Peak bandwidth: 12.8 GB/s
inline LPDDR5MemoryController::Config single_channel_config() {
    LPDDR5MemoryController::Config config;
    config.num_channels = 1;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    // Default LPDDR5-6400 timing parameters are already set
    return config;
}

/// Dual-channel LPDDR5-6400 configuration
/// - 2 channels
/// - 32 banks total (16 per channel)
/// - 6400 MT/s (3200 MHz clock)
/// - Peak bandwidth: 25.6 GB/s
inline LPDDR5MemoryController::Config dual_channel_config() {
    LPDDR5MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    return config;
}

/// Single-channel with BL32 (longer bursts, higher throughput per access)
inline LPDDR5MemoryController::Config single_channel_bl32_config() {
    LPDDR5MemoryController::Config config = single_channel_config();
    config.burst_length = BurstLength::BL32;
    return config;
}

// ============================================================================
// Address Generation Helpers
// ============================================================================

/// Generate address targeting specific bank and row (single channel)
/// Address format: [row | bank | col | byte_offset]
inline uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    addr |= static_cast<uint64_t>(row) << (6 + 10 + 4);   // row at top
    addr |= static_cast<uint64_t>(bank) << (6 + 10);      // bank
    addr |= static_cast<uint64_t>(col) << 6;              // column
    return addr;
}

/// Generate address for dual channel
/// Address format: [row | bank | col | channel | byte_offset]
inline uint64_t make_address_dual(uint8_t channel, uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    addr |= static_cast<uint64_t>(row) << (6 + 1 + 10 + 4);
    addr |= static_cast<uint64_t>(bank) << (6 + 1 + 10);
    addr |= static_cast<uint64_t>(col) << (6 + 1);
    addr |= static_cast<uint64_t>(channel) << 6;
    return addr;
}

/// Get bank group for a given bank (0-3)
inline uint8_t bank_group(uint8_t bank) {
    return bank / 4;
}

/// Check if two banks are in the same bank group
inline bool same_bank_group(uint8_t bank1, uint8_t bank2) {
    return bank_group(bank1) == bank_group(bank2);
}

// ============================================================================
// Expected Latency Calculations
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tCL + tBurst = 14 + 8 = 22 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 22;

/// Expected latency for page empty read (bank idle)
/// tRCD + tCL + tBurst = 14 + 14 + 8 = 36 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 36;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCD + tCL + tBurst = 14 + 14 + 14 + 8 = 50 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 50;

/// Expected latency for page hit write
/// tWL + tBurst = 8 + 8 = 16 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 16;

/// Expected latency for page empty write
/// tRCD + tWL + tBurst = 14 + 8 + 8 = 30 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 30;

/// Expected latency for page conflict write
/// tRP + tRCD + tWL + tBurst = 14 + 14 + 8 + 8 = 44 cycles
constexpr uint32_t PAGE_CONFLICT_WRITE_LATENCY = 44;

// ============================================================================
// Timing Constants (LPDDR5-6400 @ 3200 MHz)
// ============================================================================

constexpr uint32_t tRCD = 14;       // Row to column delay
constexpr uint32_t tRP = 14;        // Row precharge
constexpr uint32_t tRAS = 28;       // Row active time
constexpr uint32_t tRC = 42;        // Row cycle time
constexpr uint32_t tCL = 14;        // CAS read latency
constexpr uint32_t tWL = 8;         // CAS write latency
constexpr uint32_t tWR = 24;        // Write recovery
constexpr uint32_t tRTP = 6;        // Read to precharge
constexpr uint32_t tRRD_L = 6;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 4;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 6;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 4;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 10;     // Write to read (same BG)
constexpr uint32_t tWTR_S = 4;      // Write to read (diff BG)
constexpr uint32_t tRTW = 14;       // Read to write
constexpr uint32_t tFAW = 24;       // Four activate window
constexpr uint32_t tBurst_BL16 = 8; // BL16 burst cycles
constexpr uint32_t tBurst_BL32 = 16;// BL32 burst cycles

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL16 × 16-bit bus = 32 bytes)
constexpr uint32_t MIN_ACCESS_BYTES = 32;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32×32 int32)
constexpr uint32_t TILE_BYTES = 4096;

} // namespace sw::kpu::patterns
