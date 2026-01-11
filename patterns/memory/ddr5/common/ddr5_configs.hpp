// patterns/memory/ddr5/common/ddr5_configs.hpp
//
// Standard DDR5 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/models/temporal/memory/controllers/ddr5_controller.hpp>

namespace sw::kpu::patterns::ddr5 {

using namespace sw::kpu::ddr5;

// ============================================================================
// Standard DDR5-4800 Configurations
// ============================================================================

/// Single-channel DDR5-4800 configuration
/// - 1 channel
/// - 32 banks (4 bank groups x 8 banks per group)
/// - 4800 MT/s (2400 MHz clock)
/// - Peak bandwidth: 38.4 GB/s
inline DDR5MemoryController::Config single_channel_config() {
    DDR5MemoryController::Config config;
    config.num_channels = 1;
    config.banks_per_channel = 32;
    config.bank_groups = 4;
    config.banks_per_group = 8;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    // Default DDR5-4800 timing parameters are already set
    return config;
}

/// Dual-channel DDR5-4800 configuration
/// - 2 channels
/// - 64 banks total (32 per channel)
/// - 4800 MT/s (2400 MHz clock)
/// - Peak bandwidth: 76.8 GB/s
inline DDR5MemoryController::Config dual_channel_config() {
    DDR5MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 32;
    config.bank_groups = 4;
    config.banks_per_group = 8;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    return config;
}

/// DDR5-5600 configuration (higher speed grade)
inline DDR5MemoryController::Config ddr5_5600_config() {
    DDR5MemoryController::Config config = dual_channel_config();
    // DDR5-5600 timing (2800 MHz)
    config.timing.tRCD = 20;
    config.timing.tRP = 20;
    config.timing.tCL = 20;
    config.timing.tRAS = 35;
    config.timing.tRC = 55;
    return config;
}

/// DDR5-6400 configuration (higher speed grade)
inline DDR5MemoryController::Config ddr5_6400_config() {
    DDR5MemoryController::Config config = dual_channel_config();
    // DDR5-6400 timing (3200 MHz)
    config.timing.tRCD = 22;
    config.timing.tRP = 22;
    config.timing.tCL = 22;
    config.timing.tRAS = 38;
    config.timing.tRC = 60;
    return config;
}

// ============================================================================
// Address Generation Helpers
// ============================================================================

/// Generate address targeting specific bank and row (single channel)
/// Address format: [row | bank | col | byte_offset]
/// DDR5: 5 bank bits (32 banks), 17 row bits
inline uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    addr |= static_cast<uint64_t>(row) << (6 + 10 + 5);   // row at top (17 bits)
    addr |= static_cast<uint64_t>(bank) << (6 + 10);      // bank (5 bits)
    addr |= static_cast<uint64_t>(col) << 6;              // column (10 bits)
    return addr;
}

/// Generate address for dual channel
/// Address format: [row | bank | col | channel | byte_offset]
inline uint64_t make_address_dual(uint8_t channel, uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    addr |= static_cast<uint64_t>(row) << (6 + 1 + 10 + 5);
    addr |= static_cast<uint64_t>(bank) << (6 + 1 + 10);
    addr |= static_cast<uint64_t>(col) << (6 + 1);
    addr |= static_cast<uint64_t>(channel) << 6;
    return addr;
}

/// Get bank group for a given bank (0-3)
/// DDR5 has 4 bank groups with 8 banks each
inline uint8_t bank_group(uint8_t bank) {
    return bank / 8;
}

/// Check if two banks are in the same bank group
inline bool same_bank_group(uint8_t bank1, uint8_t bank2) {
    return bank_group(bank1) == bank_group(bank2);
}

// ============================================================================
// Expected Latency Calculations (DDR5-4800 @ 2400 MHz)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tCL + tBurst = 16 + 8 = 24 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 24;

/// Expected latency for page empty read (bank idle)
/// tRCD + tCL + tBurst = 16 + 16 + 8 = 40 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 40;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCD + tCL + tBurst = 16 + 16 + 16 + 8 = 56 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 56;

/// Expected latency for page hit write
/// tWL + tBurst = 8 + 8 = 16 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 16;

/// Expected latency for page empty write
/// tRCD + tWL + tBurst = 16 + 8 + 8 = 32 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 32;

/// Expected latency for page conflict write
/// tRP + tRCD + tWL + tBurst = 16 + 16 + 8 + 8 = 48 cycles
constexpr uint32_t PAGE_CONFLICT_WRITE_LATENCY = 48;

// ============================================================================
// Timing Constants (DDR5-4800 @ 2400 MHz)
// ============================================================================

constexpr uint32_t tRCD = 16;       // Row to column delay
constexpr uint32_t tRP = 16;        // Row precharge
constexpr uint32_t tRAS = 32;       // Row active time
constexpr uint32_t tRC = 48;        // Row cycle time
constexpr uint32_t tCL = 16;        // CAS read latency
constexpr uint32_t tWL = 8;         // CAS write latency
constexpr uint32_t tWR = 24;        // Write recovery
constexpr uint32_t tRTP = 8;        // Read to precharge
constexpr uint32_t tRRD_L = 8;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 4;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 8;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 4;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 12;     // Write to read (same BG)
constexpr uint32_t tWTR_S = 4;      // Write to read (diff BG)
constexpr uint32_t tRTW = 16;       // Read to write
constexpr uint32_t tFAW = 32;       // Four activate window
constexpr uint32_t tBurst = 8;      // BL16 burst cycles

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL16 x 64-bit bus = 64 bytes per channel)
constexpr uint32_t MIN_ACCESS_BYTES = 64;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32x32 int32)
constexpr uint32_t TILE_BYTES = 4096;

} // namespace sw::kpu::patterns::ddr5
