// patterns/memory/hbm3/common/hbm3_configs.hpp
//
// Standard HBM3 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/models/temporal/memory/controllers/hbm3_controller.hpp>

namespace sw::kpu::patterns::hbm3 {

using namespace sw::kpu::hbm3;

// ============================================================================
// Standard HBM3 Configurations
// ============================================================================

/// HBM3 configuration (5.6 Gbps per pin)
/// - 16 channels per stack (64-bit each)
/// - 2 pseudo-channels per channel (32-bit each)
/// - 16 banks per pseudo-channel
/// - 2.8 GHz CK (5.6 Gbps DDR)
/// - Peak bandwidth: 716.8 GB/s per stack
inline HBM3MemoryController::Config hbm3_5600_config() {
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

    // HBM3-5600 timing @ 2.8 GHz CK
    config.timing.tRCD = 8;
    config.timing.tRP = 8;
    config.timing.tRAS = 16;
    config.timing.tRC = 24;
    config.timing.tRL = 8;
    config.timing.tWL = 4;
    config.timing.tWR = 12;
    config.timing.tRTP = 4;
    config.timing.tRRD_L = 4;
    config.timing.tRRD_S = 2;
    config.timing.tCCD_L = 4;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 6;
    config.timing.tWTR_S = 3;
    config.timing.tRTW = 8;
    config.timing.tFAW = 16;
    config.timing.tRFCpb = 130;
    config.timing.tRFCab = 260;
    config.timing.tREFI = 1950;

    return config;
}

/// HBM3E configuration (9.2 Gbps per pin)
/// - Higher data rate variant
/// - Peak bandwidth: 1.18 TB/s per stack
inline HBM3MemoryController::Config hbm3e_9200_config() {
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

    // HBM3E-9200 timing @ 4.6 GHz CK
    config.timing.tRCD = 13;
    config.timing.tRP = 13;
    config.timing.tRAS = 26;
    config.timing.tRC = 39;
    config.timing.tRL = 13;
    config.timing.tWL = 7;
    config.timing.tWR = 20;
    config.timing.tRTP = 7;
    config.timing.tRRD_L = 7;
    config.timing.tRRD_S = 3;
    config.timing.tCCD_L = 7;
    config.timing.tCCD_S = 3;
    config.timing.tWTR_L = 10;
    config.timing.tWTR_S = 5;
    config.timing.tRTW = 13;
    config.timing.tFAW = 26;
    config.timing.tRFCpb = 214;
    config.timing.tRFCab = 428;
    config.timing.tREFI = 3206;

    return config;
}

/// Single pseudo-channel configuration for simple tests
inline HBM3MemoryController::Config single_pc_config() {
    auto config = hbm3_5600_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single channel (2 PCs) configuration
inline HBM3MemoryController::Config single_channel_config() {
    auto config = hbm3_5600_config();
    config.num_channels = 1;
    return config;
}

/// Eight channel configuration
inline HBM3MemoryController::Config eight_channel_config() {
    auto config = hbm3_5600_config();
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
// Expected Latency Calculations (HBM3-5600)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 8 + 4 = 12 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 12;

/// Expected latency for page empty read (bank idle)
/// tRCD + tRL + tBurst = 8 + 8 + 4 = 20 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 20;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCD + tRL + tBurst = 8 + 8 + 8 + 4 = 28 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 28;

/// Expected latency for page hit write
/// tWL + tBurst = 4 + 4 = 8 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 8;

/// Expected latency for page empty write
/// tRCD + tWL + tBurst = 8 + 4 + 4 = 16 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 16;

// ============================================================================
// Timing Constants (HBM3-5600 @ 2.8 GHz CK)
// ============================================================================

constexpr uint32_t tRCD = 8;        // Row to column delay
constexpr uint32_t tRP = 8;         // Row precharge
constexpr uint32_t tRAS = 16;       // Row active time
constexpr uint32_t tRC = 24;        // Row cycle time
constexpr uint32_t tRL = 8;         // CAS read latency
constexpr uint32_t tWL = 4;         // CAS write latency
constexpr uint32_t tWR = 12;        // Write recovery
constexpr uint32_t tRTP = 4;        // Read to precharge
constexpr uint32_t tRRD_L = 4;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 2;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 4;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 2;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 6;      // Write to read (same BG)
constexpr uint32_t tWTR_S = 3;      // Write to read (diff BG)
constexpr uint32_t tRTW = 8;        // Read to write
constexpr uint32_t tFAW = 16;       // Four activate window
constexpr uint32_t tBurst = 4;      // BL8 burst cycles (DDR)

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

/// HBM3-5600 peak bandwidth: 716.8 GB/s per stack
/// 16 channels x 64-bit bus x 5.6 Gbps / 8 bits = 716.8 GB/s
constexpr double HBM3_5600_BANDWIDTH = 716.8;

/// HBM3E-9200 peak bandwidth: 1.18 TB/s per stack
constexpr double HBM3E_9200_BANDWIDTH = 1177.6;

} // namespace sw::kpu::patterns::hbm3
