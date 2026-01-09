// patterns/memory/hbm2/common/hbm2_configs.hpp
//
// Standard HBM2 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/components/hbm2_memory_controller.hpp>

namespace sw::kpu::patterns::hbm2 {

using namespace sw::kpu::hbm2;

// ============================================================================
// Standard HBM2 Configurations
// ============================================================================

/// HBM2 configuration (2.0 Gbps per pin)
/// - 8 channels per stack (128-bit each)
/// - 2 pseudo-channels per channel (64-bit each)
/// - 16 banks per pseudo-channel
/// - 1 GHz CK (2.0 Gbps DDR)
/// - Peak bandwidth: 256 GB/s per stack
inline HBM2MemoryController::Config hbm2_2000_config() {
    HBM2MemoryController::Config config;
    config.num_channels = 8;
    config.pseudo_channels_per_channel = 2;
    config.banks_per_pseudo_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL4;
    config.queue_depth = 64;

    // Address mapping
    config.row_bits = 14;
    config.col_bits = 6;
    config.bank_bits = 4;
    config.pc_bits = 1;
    config.channel_bits = 3;

    // HBM2-2000 timing @ 1.0 GHz CK
    config.timing.tRCDRD = 12;
    config.timing.tRCDWR = 6;
    config.timing.tRP = 14;
    config.timing.tRAS = 28;
    config.timing.tRC = 42;
    config.timing.tRL = 18;
    config.timing.tWL = 7;
    config.timing.tWR = 16;
    config.timing.tRTP = 6;
    config.timing.tRRD_L = 4;
    config.timing.tRRD_S = 3;
    config.timing.tCCD_L = 4;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 8;
    config.timing.tWTR_S = 4;
    config.timing.tRTW = 10;
    config.timing.tFAW = 16;
    config.timing.tRFCpb = 130;
    config.timing.tRFCab = 260;
    config.timing.tREFI = 3900;

    return config;
}

/// HBM2E configuration (3.2 Gbps per pin)
/// - Higher data rate variant
/// - Peak bandwidth: 409.6 GB/s per stack
inline HBM2MemoryController::Config hbm2e_3200_config() {
    HBM2MemoryController::Config config;
    config.num_channels = 8;
    config.pseudo_channels_per_channel = 2;
    config.banks_per_pseudo_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL4;
    config.queue_depth = 64;

    // Address mapping
    config.row_bits = 14;
    config.col_bits = 6;
    config.bank_bits = 4;
    config.pc_bits = 1;
    config.channel_bits = 3;

    // HBM2E-3200 timing @ 1.6 GHz CK
    config.timing.tRCDRD = 19;
    config.timing.tRCDWR = 10;
    config.timing.tRP = 22;
    config.timing.tRAS = 45;
    config.timing.tRC = 67;
    config.timing.tRL = 29;
    config.timing.tWL = 11;
    config.timing.tWR = 26;
    config.timing.tRTP = 10;
    config.timing.tRRD_L = 6;
    config.timing.tRRD_S = 5;
    config.timing.tCCD_L = 6;
    config.timing.tCCD_S = 3;
    config.timing.tWTR_L = 13;
    config.timing.tWTR_S = 6;
    config.timing.tRTW = 16;
    config.timing.tFAW = 26;
    config.timing.tRFCpb = 208;
    config.timing.tRFCab = 416;
    config.timing.tREFI = 6240;

    return config;
}

/// Single pseudo-channel configuration for simple tests
inline HBM2MemoryController::Config single_pc_config() {
    auto config = hbm2_2000_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single channel (2 PCs) configuration
inline HBM2MemoryController::Config single_channel_config() {
    auto config = hbm2_2000_config();
    config.num_channels = 1;
    return config;
}

/// Four channel configuration
inline HBM2MemoryController::Config four_channel_config() {
    auto config = hbm2_2000_config();
    config.num_channels = 4;
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
    addr |= static_cast<uint64_t>(row) << (5 + 3 + 1 + 6 + 4);
    // Bank (4 bits for 16 banks)
    addr |= static_cast<uint64_t>(bank) << (5 + 3 + 1 + 6);
    // Column (6 bits typical)
    addr |= static_cast<uint64_t>(col) << (5 + 3 + 1);
    // Pseudo-channel (1 bit)
    addr |= static_cast<uint64_t>(pc) << (5 + 3);
    // Channel (3 bits for 8 channels)
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
// Expected Latency Calculations (HBM2-2000)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 18 + 2 = 20 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 20;

/// Expected latency for page empty read (bank idle)
/// tRCDRD + tRL + tBurst = 12 + 18 + 2 = 32 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 32;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCDRD + tRL + tBurst = 14 + 12 + 18 + 2 = 46 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 46;

/// Expected latency for page hit write
/// tWL + tBurst = 7 + 2 = 9 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 9;

/// Expected latency for page empty write
/// tRCDWR + tWL + tBurst = 6 + 7 + 2 = 15 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 15;

// ============================================================================
// Timing Constants (HBM2-2000 @ 1.0 GHz CK)
// ============================================================================

constexpr uint32_t tRCDRD = 12;     // Row to column delay (read)
constexpr uint32_t tRCDWR = 6;      // Row to column delay (write)
constexpr uint32_t tRP = 14;        // Row precharge
constexpr uint32_t tRAS = 28;       // Row active time
constexpr uint32_t tRC = 42;        // Row cycle time
constexpr uint32_t tRL = 18;        // CAS read latency
constexpr uint32_t tWL = 7;         // CAS write latency
constexpr uint32_t tWR = 16;        // Write recovery
constexpr uint32_t tRTP = 6;        // Read to precharge
constexpr uint32_t tRRD_L = 4;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 3;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 4;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 2;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 8;      // Write to read (same BG)
constexpr uint32_t tWTR_S = 4;      // Write to read (diff BG)
constexpr uint32_t tRTW = 10;       // Read to write
constexpr uint32_t tFAW = 16;       // Four activate window
constexpr uint32_t tBurst = 2;      // BL4 burst cycles

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL4 x 64-bit = 32 bytes)
constexpr uint32_t MIN_ACCESS_BYTES = 32;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32x32 int32)
constexpr uint32_t TILE_BYTES = 4096;

// ============================================================================
// Bandwidth Calculations
// ============================================================================

/// HBM2-2000 peak bandwidth: 256 GB/s per stack
/// 8 channels x 128-bit bus x 2.0 Gbps / 8 bits = 256 GB/s
constexpr double HBM2_2000_BANDWIDTH = 256.0;

/// HBM2E-3200 peak bandwidth: 409.6 GB/s per stack
/// (See patterns/memory/hbm2e/common/hbm2e_configs.hpp for HBM2E configs)
constexpr double HBM2E_3200_BANDWIDTH = 409.6;

} // namespace sw::kpu::patterns::hbm2
