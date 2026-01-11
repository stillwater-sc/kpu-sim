// patterns/memory/gddr7/common/gddr7_configs.hpp
//
// Standard GDDR7 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/models/temporal/memory/controllers/gddr7_controller.hpp>

namespace sw::kpu::patterns::gddr7 {

using namespace sw::kpu::gddr7;

// ============================================================================
// Standard GDDR7-32000 Configurations
// ============================================================================

/// Single-channel GDDR7-32000 configuration
/// - 1 channel (16-bit with PAM3)
/// - 16 banks (4 bank groups x 4 banks per group)
/// - 32000 MT/s effective (PAM3 signaling at ~2.0 GHz CK)
/// - Peak bandwidth: ~64 GB/s per channel
inline GDDR7MemoryController::Config single_channel_config() {
    GDDR7MemoryController::Config config;
    config.num_channels = 1;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    // Default GDDR7-32000 timing parameters are already set
    return config;
}

/// Dual-channel GDDR7-32000 configuration
/// - 2 channels (dual 16-bit channels)
/// - 32 banks total (16 per channel)
/// - 32000 MT/s effective (PAM3 signaling)
/// - Peak bandwidth: ~128 GB/s
inline GDDR7MemoryController::Config dual_channel_config() {
    GDDR7MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;
    return config;
}

/// High-throughput BL32 configuration
inline GDDR7MemoryController::Config dual_channel_bl32_config() {
    GDDR7MemoryController::Config config = dual_channel_config();
    config.burst_length = BurstLength::BL32;
    return config;
}

// ============================================================================
// Address Generation Helpers
// ============================================================================

/// Generate address targeting specific bank and row (single channel)
/// Address format: [row | bank | col | byte_offset]
/// GDDR7: 4 bank bits (16 banks), 16 row bits
inline uint64_t make_address(uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    addr |= static_cast<uint64_t>(row) << (6 + 10 + 4);   // row at top (16 bits)
    addr |= static_cast<uint64_t>(bank) << (6 + 10);      // bank (4 bits)
    addr |= static_cast<uint64_t>(col) << 6;              // column (10 bits)
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
/// GDDR7 has 4 bank groups with 4 banks each
inline uint8_t bank_group(uint8_t bank) {
    return bank / 4;
}

/// Check if two banks are in the same bank group
inline bool same_bank_group(uint8_t bank1, uint8_t bank2) {
    return bank_group(bank1) == bank_group(bank2);
}

// ============================================================================
// Expected Latency Calculations (GDDR7-32000 @ 2.0 GHz CK)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 16 + 4 = 20 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 20;

/// Expected latency for page empty read (bank idle)
/// tRCDRD + tRL + tBurst = 14 + 16 + 4 = 34 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 34;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCDRD + tRL + tBurst = 14 + 14 + 16 + 4 = 48 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 48;

/// Expected latency for page hit write
/// tWL + tBurst = 6 + 4 = 10 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 10;

/// Expected latency for page empty write
/// tRCDWR + tWL + tBurst = 14 + 6 + 4 = 24 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 24;

/// Expected latency for page conflict write
/// tRP + tRCDWR + tWL + tBurst = 14 + 14 + 6 + 4 = 38 cycles
constexpr uint32_t PAGE_CONFLICT_WRITE_LATENCY = 38;

// ============================================================================
// Timing Constants (GDDR7-32000 @ 2.0 GHz CK with PAM3)
// ============================================================================

constexpr uint32_t tRCDRD = 14;     // Row to column delay (read)
constexpr uint32_t tRCDWR = 14;     // Row to column delay (write)
constexpr uint32_t tRP = 14;        // Row precharge
constexpr uint32_t tRAS = 24;       // Row active time
constexpr uint32_t tRC = 38;        // Row cycle time
constexpr uint32_t tRL = 16;        // CAS read latency
constexpr uint32_t tWL = 6;         // CAS write latency
constexpr uint32_t tWR = 14;        // Write recovery
constexpr uint32_t tRTP = 6;        // Read to precharge
constexpr uint32_t tRRD_L = 4;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 3;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 3;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 2;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 5;      // Write to read (same BG)
constexpr uint32_t tWTR_S = 3;      // Write to read (diff BG)
constexpr uint32_t tRTW = 12;       // Read to write
constexpr uint32_t tFAW = 12;       // Four activate window (tighter than GDDR6)
constexpr uint32_t tBurst = 4;      // BL16 burst cycles (with PAM3)

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL16 x 32-bit = 64 bytes with PAM3)
constexpr uint32_t MIN_ACCESS_BYTES = 64;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32x32 int32)
constexpr uint32_t TILE_BYTES = 4096;

} // namespace sw::kpu::patterns::gddr7
