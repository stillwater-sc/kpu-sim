// patterns/memory/hbm2e/common/hbm2e_configs.hpp
//
// HBM2E configurations for pattern testing (3.2-3.6 Gbps variants)
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/components/hbm2_memory_controller.hpp>

namespace sw::kpu::patterns::hbm2e {

using namespace sw::kpu::hbm2;

// ============================================================================
// HBM2E Configurations
// ============================================================================

/// HBM2E-3200 configuration (3.2 Gbps per pin)
/// - 8 channels per stack (128-bit each)
/// - 2 pseudo-channels per channel (64-bit each)
/// - 16 banks per pseudo-channel
/// - 1.6 GHz CK (3.2 Gbps DDR)
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
    // Timing in cycles (higher clock = more cycles for same absolute time)
    config.timing.tRCDRD = 19;   // 12ns @ 1.6 GHz
    config.timing.tRCDWR = 10;   // 6ns @ 1.6 GHz
    config.timing.tRP = 22;      // 14ns @ 1.6 GHz
    config.timing.tRAS = 45;     // 28ns @ 1.6 GHz
    config.timing.tRC = 67;      // 42ns @ 1.6 GHz
    config.timing.tRL = 29;      // 18ns @ 1.6 GHz
    config.timing.tWL = 11;      // 7ns @ 1.6 GHz
    config.timing.tWR = 26;      // 16ns @ 1.6 GHz
    config.timing.tRTP = 10;     // 6ns @ 1.6 GHz
    config.timing.tRRD_L = 6;    // 4ns @ 1.6 GHz
    config.timing.tRRD_S = 5;    // 3ns @ 1.6 GHz
    config.timing.tCCD_L = 6;    // 4ns @ 1.6 GHz
    config.timing.tCCD_S = 3;    // 2ns @ 1.6 GHz
    config.timing.tWTR_L = 13;   // 8ns @ 1.6 GHz
    config.timing.tWTR_S = 6;    // 4ns @ 1.6 GHz
    config.timing.tRTW = 16;     // 10ns @ 1.6 GHz
    config.timing.tFAW = 26;     // 16ns @ 1.6 GHz
    config.timing.tRFCpb = 208;  // 130ns @ 1.6 GHz
    config.timing.tRFCab = 416;  // 260ns @ 1.6 GHz
    config.timing.tREFI = 6240;  // 3.9us @ 1.6 GHz

    return config;
}

/// HBM2E-3600 configuration (3.6 Gbps per pin)
/// - 8 channels per stack (128-bit each)
/// - 2 pseudo-channels per channel (64-bit each)
/// - 16 banks per pseudo-channel
/// - 1.8 GHz CK (3.6 Gbps DDR)
/// - Peak bandwidth: 460.8 GB/s per stack
inline HBM2MemoryController::Config hbm2e_3600_config() {
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

    // HBM2E-3600 timing @ 1.8 GHz CK
    // Timing in cycles (higher clock = more cycles for same absolute time)
    config.timing.tRCDRD = 22;   // 12ns @ 1.8 GHz
    config.timing.tRCDWR = 11;   // 6ns @ 1.8 GHz
    config.timing.tRP = 25;      // 14ns @ 1.8 GHz
    config.timing.tRAS = 50;     // 28ns @ 1.8 GHz
    config.timing.tRC = 76;      // 42ns @ 1.8 GHz
    config.timing.tRL = 32;      // 18ns @ 1.8 GHz
    config.timing.tWL = 13;      // 7ns @ 1.8 GHz
    config.timing.tWR = 29;      // 16ns @ 1.8 GHz
    config.timing.tRTP = 11;     // 6ns @ 1.8 GHz
    config.timing.tRRD_L = 7;    // 4ns @ 1.8 GHz
    config.timing.tRRD_S = 5;    // 3ns @ 1.8 GHz
    config.timing.tCCD_L = 7;    // 4ns @ 1.8 GHz
    config.timing.tCCD_S = 4;    // 2ns @ 1.8 GHz
    config.timing.tWTR_L = 14;   // 8ns @ 1.8 GHz
    config.timing.tWTR_S = 7;    // 4ns @ 1.8 GHz
    config.timing.tRTW = 18;     // 10ns @ 1.8 GHz
    config.timing.tFAW = 29;     // 16ns @ 1.8 GHz
    config.timing.tRFCpb = 234;  // 130ns @ 1.8 GHz
    config.timing.tRFCab = 468;  // 260ns @ 1.8 GHz
    config.timing.tREFI = 7020;  // 3.9us @ 1.8 GHz

    return config;
}

// ============================================================================
// Convenience Configurations
// ============================================================================

/// Single pseudo-channel HBM2E-3200 for simple tests
inline HBM2MemoryController::Config single_pc_config_3200() {
    auto config = hbm2e_3200_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single pseudo-channel HBM2E-3600 for simple tests
inline HBM2MemoryController::Config single_pc_config_3600() {
    auto config = hbm2e_3600_config();
    config.num_channels = 1;
    config.pseudo_channels_per_channel = 1;
    return config;
}

/// Single channel (2 PCs) HBM2E-3200
inline HBM2MemoryController::Config single_channel_config_3200() {
    auto config = hbm2e_3200_config();
    config.num_channels = 1;
    return config;
}

/// Single channel (2 PCs) HBM2E-3600
inline HBM2MemoryController::Config single_channel_config_3600() {
    auto config = hbm2e_3600_config();
    config.num_channels = 1;
    return config;
}

/// Four channel HBM2E-3200
inline HBM2MemoryController::Config four_channel_config_3200() {
    auto config = hbm2e_3200_config();
    config.num_channels = 4;
    return config;
}

/// Four channel HBM2E-3600
inline HBM2MemoryController::Config four_channel_config_3600() {
    auto config = hbm2e_3600_config();
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
// Expected Latency Calculations (HBM2E-3600 @ 1.8 GHz)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 32 + 2 = 34 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY_3600 = 34;

/// Expected latency for page empty read (bank idle)
/// tRCDRD + tRL + tBurst = 22 + 32 + 2 = 56 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY_3600 = 56;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCDRD + tRL + tBurst = 25 + 22 + 32 + 2 = 81 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY_3600 = 81;

/// Expected latency for page hit write
/// tWL + tBurst = 13 + 2 = 15 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY_3600 = 15;

/// Expected latency for page empty write
/// tRCDWR + tWL + tBurst = 11 + 13 + 2 = 26 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY_3600 = 26;

// ============================================================================
// Timing Constants (HBM2E-3600 @ 1.8 GHz CK)
// ============================================================================

namespace timing_3600 {

constexpr uint32_t tRCDRD = 22;     // Row to column delay (read)
constexpr uint32_t tRCDWR = 11;     // Row to column delay (write)
constexpr uint32_t tRP = 25;        // Row precharge
constexpr uint32_t tRAS = 50;       // Row active time
constexpr uint32_t tRC = 76;        // Row cycle time
constexpr uint32_t tRL = 32;        // CAS read latency
constexpr uint32_t tWL = 13;        // CAS write latency
constexpr uint32_t tWR = 29;        // Write recovery
constexpr uint32_t tRTP = 11;       // Read to precharge
constexpr uint32_t tRRD_L = 7;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 5;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 7;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 4;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 14;     // Write to read (same BG)
constexpr uint32_t tWTR_S = 7;      // Write to read (diff BG)
constexpr uint32_t tRTW = 18;       // Read to write
constexpr uint32_t tFAW = 29;       // Four activate window
constexpr uint32_t tBurst = 2;      // BL4 burst cycles

} // namespace timing_3600

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

/// HBM2E-3200 peak bandwidth: 409.6 GB/s per stack
/// 8 channels x 128-bit x 3.2 Gbps / 8 = 409.6 GB/s
constexpr double HBM2E_3200_BANDWIDTH = 409.6;

/// HBM2E-3600 peak bandwidth: 460.8 GB/s per stack
/// 8 channels x 128-bit x 3.6 Gbps / 8 = 460.8 GB/s
constexpr double HBM2E_3600_BANDWIDTH = 460.8;

// ============================================================================
// Clock Frequencies
// ============================================================================

/// HBM2E-3200 clock frequency in GHz (for trace export)
constexpr double CLOCK_GHZ_3200 = 1.6;

/// HBM2E-3600 clock frequency in GHz (for trace export)
constexpr double CLOCK_GHZ_3600 = 1.8;

} // namespace sw::kpu::patterns::hbm2e
