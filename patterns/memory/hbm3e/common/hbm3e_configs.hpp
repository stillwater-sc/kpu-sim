// patterns/memory/hbm3e/common/hbm3e_configs.hpp
//
// HBM3E configurations for pattern testing (8.4-9.6 Gbps variants)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/components/hbm3_memory_controller.hpp>

namespace sw::kpu::patterns::hbm3e {

using namespace sw::kpu::hbm3;

// ============================================================================
// HBM3E Configurations
// ============================================================================

/// HBM3E-8400 configuration (8.4 Gbps per pin)
/// - 16 channels per stack (64-bit each)
/// - 2 pseudo-channels per channel (32-bit each)
/// - 16 banks per pseudo-channel
/// - 4.2 GHz CK (8.4 Gbps DDR)
/// - Peak bandwidth: 1.075 TB/s per stack
inline HBM3MemoryController::Config hbm3e_8400_config() {
    HBM3MemoryController::Config config;
    config.num_channels = 16;
    config.pseudo_channels_per_channel = 2;
    config.banks_per_pseudo_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL8;
    config.queue_depth = 64;

    // Address mapping
    config.row_bits = 14;
    config.col_bits = 5;
    config.bank_bits = 4;
    config.pc_bits = 1;
    config.channel_bits = 4;

    // HBM3E-8400 timing @ 4.2 GHz CK
    // Timing in cycles (higher clock = more cycles for same absolute time)
    // Scale factor from HBM3-5600: 4.2/2.8 = 1.5x
    config.timing.tRCD = 12;     // 8 * 1.5 = 12
    config.timing.tRP = 12;      // 8 * 1.5 = 12
    config.timing.tRAS = 24;     // 16 * 1.5 = 24
    config.timing.tRC = 36;      // 24 * 1.5 = 36
    config.timing.tRL = 12;      // 8 * 1.5 = 12
    config.timing.tWL = 6;       // 4 * 1.5 = 6
    config.timing.tWR = 18;      // 12 * 1.5 = 18
    config.timing.tRTP = 6;      // 4 * 1.5 = 6
    config.timing.tRRD_L = 6;    // 4 * 1.5 = 6
    config.timing.tRRD_S = 3;    // 2 * 1.5 = 3
    config.timing.tCCD_L = 6;    // 4 * 1.5 = 6
    config.timing.tCCD_S = 3;    // 2 * 1.5 = 3
    config.timing.tWTR_L = 9;    // 6 * 1.5 = 9
    config.timing.tWTR_S = 5;    // 3 * 1.5 ≈ 5
    config.timing.tRTW = 12;     // 8 * 1.5 = 12
    config.timing.tFAW = 24;     // 16 * 1.5 = 24
    config.timing.tRFCpb = 195;  // 130 * 1.5 = 195
    config.timing.tRFCab = 390;  // 260 * 1.5 = 390
    config.timing.tREFI = 2925;  // 1950 * 1.5 = 2925

    return config;
}

/// HBM3E-9600 configuration (9.6 Gbps per pin)
/// - 16 channels per stack (64-bit each)
/// - 2 pseudo-channels per channel (32-bit each)
/// - 16 banks per pseudo-channel
/// - 4.8 GHz CK (9.6 Gbps DDR)
/// - Peak bandwidth: 1.229 TB/s per stack
inline HBM3MemoryController::Config hbm3e_9600_config() {
    HBM3MemoryController::Config config;
    config.num_channels = 16;
    config.pseudo_channels_per_channel = 2;
    config.banks_per_pseudo_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL8;
    config.queue_depth = 64;

    // Address mapping
    config.row_bits = 14;
    config.col_bits = 5;
    config.bank_bits = 4;
    config.pc_bits = 1;
    config.channel_bits = 4;

    // HBM3E-9600 timing @ 4.8 GHz CK
    // Timing in cycles (higher clock = more cycles for same absolute time)
    // Scale factor from HBM3-5600: 4.8/2.8 = 1.71x
    config.timing.tRCD = 14;     // 8 * 1.71 ≈ 14
    config.timing.tRP = 14;      // 8 * 1.71 ≈ 14
    config.timing.tRAS = 27;     // 16 * 1.71 ≈ 27
    config.timing.tRC = 41;      // 24 * 1.71 ≈ 41
    config.timing.tRL = 14;      // 8 * 1.71 ≈ 14
    config.timing.tWL = 7;       // 4 * 1.71 ≈ 7
    config.timing.tWR = 21;      // 12 * 1.71 ≈ 21
    config.timing.tRTP = 7;      // 4 * 1.71 ≈ 7
    config.timing.tRRD_L = 7;    // 4 * 1.71 ≈ 7
    config.timing.tRRD_S = 3;    // 2 * 1.71 ≈ 3
    config.timing.tCCD_L = 7;    // 4 * 1.71 ≈ 7
    config.timing.tCCD_S = 3;    // 2 * 1.71 ≈ 3
    config.timing.tWTR_L = 10;   // 6 * 1.71 ≈ 10
    config.timing.tWTR_S = 5;    // 3 * 1.71 ≈ 5
    config.timing.tRTW = 14;     // 8 * 1.71 ≈ 14
    config.timing.tFAW = 27;     // 16 * 1.71 ≈ 27
    config.timing.tRFCpb = 222;  // 130 * 1.71 ≈ 222
    config.timing.tRFCab = 445;  // 260 * 1.71 ≈ 445
    config.timing.tREFI = 3334;  // 1950 * 1.71 ≈ 3334

    return config;
}

// ============================================================================
// Convenience Configurations
// ============================================================================

/// Single pseudo-channel HBM3E-8400 for simple tests
inline HBM3MemoryController::Config single_pc_config_8400() {
    auto config = hbm3e_8400_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single pseudo-channel HBM3E-9600 for simple tests
inline HBM3MemoryController::Config single_pc_config_9600() {
    auto config = hbm3e_9600_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single channel (2 PCs) HBM3E-8400
inline HBM3MemoryController::Config single_channel_config_8400() {
    auto config = hbm3e_8400_config();
    config.num_channels = 1;
    return config;
}

/// Single channel (2 PCs) HBM3E-9600
inline HBM3MemoryController::Config single_channel_config_9600() {
    auto config = hbm3e_9600_config();
    config.num_channels = 1;
    return config;
}

/// Eight channel HBM3E-8400
inline HBM3MemoryController::Config eight_channel_config_8400() {
    auto config = hbm3e_8400_config();
    config.num_channels = 8;
    return config;
}

/// Eight channel HBM3E-9600
inline HBM3MemoryController::Config eight_channel_config_9600() {
    auto config = hbm3e_9600_config();
    config.num_channels = 8;
    return config;
}

// ============================================================================
// Address Generation Helpers
// ============================================================================

/// Generate address targeting specific channel, pseudo-channel, bank and row
/// Address format: [row | bank | col | pseudo_channel | channel | byte_offset]
inline uint64_t make_address(uint8_t channel, uint8_t pc, uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    // Row at top (14 bits typical)
    addr |= static_cast<uint64_t>(row) << (5 + 4 + 1 + 5 + 4);
    // Bank (4 bits for 16 banks)
    addr |= static_cast<uint64_t>(bank) << (5 + 4 + 1 + 5);
    // Column (5 bits typical)
    addr |= static_cast<uint64_t>(col) << (5 + 4 + 1);
    // Pseudo-channel (1 bit)
    addr |= static_cast<uint64_t>(pc) << (5 + 4);
    // Channel (4 bits for 16 channels)
    addr |= static_cast<uint64_t>(channel) << 5;
    // Byte offset (5 bits for 32-byte alignment)
    return addr;
}

/// Generate address for single pseudo-channel operation
inline uint64_t make_address_single_pc(uint8_t bank, uint32_t row, uint32_t col = 0) {
    return make_address(0, 0, bank, row, col);
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
// Expected Latency Calculations (HBM3E-9600 @ 4.8 GHz)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 14 + 4 = 18 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY_9600 = 18;

/// Expected latency for page empty read (bank idle)
/// tRCD + tRL + tBurst = 14 + 14 + 4 = 32 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY_9600 = 32;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCD + tRL + tBurst = 14 + 14 + 14 + 4 = 46 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY_9600 = 46;

/// Expected latency for page hit write
/// tWL + tBurst = 7 + 4 = 11 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY_9600 = 11;

/// Expected latency for page empty write
/// tRCD + tWL + tBurst = 14 + 7 + 4 = 25 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY_9600 = 25;

// ============================================================================
// Timing Constants (HBM3E-9600 @ 4.8 GHz CK)
// ============================================================================

namespace timing_9600 {

constexpr uint32_t tRCD = 14;       // Row to column delay
constexpr uint32_t tRP = 14;        // Row precharge
constexpr uint32_t tRAS = 27;       // Row active time
constexpr uint32_t tRC = 41;        // Row cycle time
constexpr uint32_t tRL = 14;        // CAS read latency
constexpr uint32_t tWL = 7;         // CAS write latency
constexpr uint32_t tWR = 21;        // Write recovery
constexpr uint32_t tRTP = 7;        // Read to precharge
constexpr uint32_t tRRD_L = 7;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 3;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 7;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 3;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 10;     // Write to read (same BG)
constexpr uint32_t tWTR_S = 5;      // Write to read (diff BG)
constexpr uint32_t tRTW = 14;       // Read to write
constexpr uint32_t tFAW = 27;       // Four activate window
constexpr uint32_t tBurst = 4;      // BL8 burst cycles

} // namespace timing_9600

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL8 x 32-bit = 32 bytes)
constexpr uint32_t MIN_ACCESS_BYTES = 32;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32x32 int32)
constexpr uint32_t TILE_BYTES = 4096;

// ============================================================================
// Bandwidth Calculations
// ============================================================================

/// HBM3E-8400 peak bandwidth: 1.075 TB/s per stack
/// 16 channels x 64-bit x 8.4 Gbps / 8 = 1075.2 GB/s
constexpr double HBM3E_8400_BANDWIDTH = 1075.2;

/// HBM3E-9600 peak bandwidth: 1.229 TB/s per stack
/// 16 channels x 64-bit x 9.6 Gbps / 8 = 1228.8 GB/s
constexpr double HBM3E_9600_BANDWIDTH = 1228.8;

// ============================================================================
// Clock Frequencies
// ============================================================================

/// HBM3E-8400 clock frequency in GHz (for trace export)
constexpr double CLOCK_GHZ_8400 = 4.2;

/// HBM3E-9600 clock frequency in GHz (for trace export)
constexpr double CLOCK_GHZ_9600 = 4.8;

} // namespace sw::kpu::patterns::hbm3e
