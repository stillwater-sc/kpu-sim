// patterns/memory/gddr6/common/gddr6_configs.hpp
//
// Standard GDDR6 configurations for pattern testing
//
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.

#pragma once

#include <sw/kpu/models/temporal/memory/controllers/gddr6_controller.hpp>

namespace sw::kpu::patterns::gddr6 {

using namespace sw::kpu::gddr6;

// ============================================================================
// Standard GDDR6 Configurations
// ============================================================================

/// Single-device GDDR6-16000 configuration (16 Gbps)
/// - 1 device (2 channels)
/// - 16 banks per channel (4 bank groups x 4 banks per group)
/// - 16 Gbps (2.0 GHz CK)
/// - Peak bandwidth: 64 GB/s per device (32 GB/s per channel)
inline GDDR6MemoryController::Config gddr6_16000_config() {
    GDDR6MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;

    // GDDR6-16000 timing @ 2.0 GHz CK
    config.timing.tRCDRD = 18;
    config.timing.tRCDWR = 18;
    config.timing.tRP = 18;
    config.timing.tRAS = 28;
    config.timing.tRC = 46;
    config.timing.tRL = 20;
    config.timing.tWL = 8;
    config.timing.tWR = 16;
    config.timing.tRTP = 8;
    config.timing.tRRD_L = 4;
    config.timing.tRRD_S = 4;
    config.timing.tCCD_L = 3;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 6;
    config.timing.tWTR_S = 4;
    config.timing.tRTW = 14;
    config.timing.tFAW = 16;
    config.timing.tRFCpb = 130;
    config.timing.tRFCab = 260;
    config.timing.tREFI = 3900;

    return config;
}

/// Single-device GDDR6-14000 configuration (14 Gbps)
/// - Lower frequency variant for power-efficient applications
inline GDDR6MemoryController::Config gddr6_14000_config() {
    GDDR6MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;

    // GDDR6-14000 timing @ 1.75 GHz CK
    config.timing.tRCDRD = 16;
    config.timing.tRCDWR = 16;
    config.timing.tRP = 16;
    config.timing.tRAS = 25;
    config.timing.tRC = 41;
    config.timing.tRL = 18;
    config.timing.tWL = 7;
    config.timing.tWR = 14;
    config.timing.tRTP = 7;
    config.timing.tRRD_L = 4;
    config.timing.tRRD_S = 4;
    config.timing.tCCD_L = 3;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 5;
    config.timing.tWTR_S = 4;
    config.timing.tRTW = 12;
    config.timing.tFAW = 14;
    config.timing.tRFCpb = 114;
    config.timing.tRFCab = 228;
    config.timing.tREFI = 3413;

    return config;
}

/// Single-device GDDR6-18000 configuration (18 Gbps)
/// - High-performance variant
inline GDDR6MemoryController::Config gddr6_18000_config() {
    GDDR6MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;

    // GDDR6-18000 timing @ 2.25 GHz CK
    config.timing.tRCDRD = 20;
    config.timing.tRCDWR = 20;
    config.timing.tRP = 20;
    config.timing.tRAS = 32;
    config.timing.tRC = 52;
    config.timing.tRL = 22;
    config.timing.tWL = 9;
    config.timing.tWR = 18;
    config.timing.tRTP = 9;
    config.timing.tRRD_L = 5;
    config.timing.tRRD_S = 4;
    config.timing.tCCD_L = 4;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 7;
    config.timing.tWTR_S = 4;
    config.timing.tRTW = 16;
    config.timing.tFAW = 18;
    config.timing.tRFCpb = 146;
    config.timing.tRFCab = 292;
    config.timing.tREFI = 4388;

    return config;
}

/// Single-device GDDR6-20000 configuration (20 Gbps)
/// - Ultra high-performance variant
inline GDDR6MemoryController::Config gddr6_20000_config() {
    GDDR6MemoryController::Config config;
    config.num_channels = 2;
    config.banks_per_channel = 16;
    config.bank_groups = 4;
    config.burst_length = BurstLength::BL16;
    config.queue_depth = 64;

    // GDDR6-20000 timing @ 2.5 GHz CK
    config.timing.tRCDRD = 22;
    config.timing.tRCDWR = 22;
    config.timing.tRP = 22;
    config.timing.tRAS = 36;
    config.timing.tRC = 58;
    config.timing.tRL = 24;
    config.timing.tWL = 10;
    config.timing.tWR = 20;
    config.timing.tRTP = 10;
    config.timing.tRRD_L = 5;
    config.timing.tRRD_S = 5;
    config.timing.tCCD_L = 4;
    config.timing.tCCD_S = 2;
    config.timing.tWTR_L = 8;
    config.timing.tWTR_S = 5;
    config.timing.tRTW = 18;
    config.timing.tFAW = 20;
    config.timing.tRFCpb = 162;
    config.timing.tRFCab = 324;
    config.timing.tREFI = 4875;

    return config;
}

// ============================================================================
// Address Generation Helpers
// ============================================================================

/// Generate address targeting specific channel, bank and row
/// Address format: [row | bank | col | channel | byte_offset]
inline uint64_t make_address(uint8_t channel, uint8_t bank, uint32_t row, uint32_t col = 0) {
    uint64_t addr = 0;
    // Row at top (16 bits typical)
    addr |= static_cast<uint64_t>(row) << (5 + 1 + 10 + 4);
    // Bank (4 bits for 16 banks)
    addr |= static_cast<uint64_t>(bank) << (5 + 1 + 10);
    // Column (10 bits typical)
    addr |= static_cast<uint64_t>(col) << (5 + 1);
    // Channel (1 bit)
    addr |= static_cast<uint64_t>(channel) << 5;
    // Byte offset (5 bits for 32-byte alignment)
    return addr;
}

/// Generate address for single-channel operation (channel 0)
inline uint64_t make_address_single(uint8_t bank, uint32_t row, uint32_t col = 0) {
    return make_address(0, bank, row, col);
}

/// Get bank group for a given bank (0-3)
inline uint8_t bank_group(uint8_t bank) {
    return bank / 4;
}

/// Check if two banks are in the same bank group
inline bool same_bank_group(uint8_t bank1, uint8_t bank2) {
    return bank_group(bank1) == bank_group(bank2);
}

/// Get bank index within its bank group (0-3)
inline uint8_t bank_in_group(uint8_t bank) {
    return bank % 4;
}

// ============================================================================
// Expected Latency Calculations (GDDR6-16000)
// ============================================================================

/// Expected latency for page hit read (row already open)
/// tRL + tBurst = 20 + 4 = 24 cycles
constexpr uint32_t PAGE_HIT_READ_LATENCY = 24;

/// Expected latency for page empty read (bank idle)
/// tRCDRD + tRL + tBurst = 18 + 20 + 4 = 42 cycles
constexpr uint32_t PAGE_EMPTY_READ_LATENCY = 42;

/// Expected latency for page conflict read (different row open)
/// tRP + tRCDRD + tRL + tBurst = 18 + 18 + 20 + 4 = 60 cycles
constexpr uint32_t PAGE_CONFLICT_READ_LATENCY = 60;

/// Expected latency for page hit write
/// tWL + tBurst = 8 + 4 = 12 cycles
constexpr uint32_t PAGE_HIT_WRITE_LATENCY = 12;

/// Expected latency for page empty write
/// tRCDWR + tWL + tBurst = 18 + 8 + 4 = 30 cycles
constexpr uint32_t PAGE_EMPTY_WRITE_LATENCY = 30;

/// Expected latency for page conflict write
/// tRP + tRCDWR + tWL + tBurst = 18 + 18 + 8 + 4 = 48 cycles
constexpr uint32_t PAGE_CONFLICT_WRITE_LATENCY = 48;

// ============================================================================
// Timing Constants (GDDR6-16000 @ 2.0 GHz CK)
// ============================================================================

constexpr uint32_t tRCDRD = 18;     // Row to column delay (read)
constexpr uint32_t tRCDWR = 18;     // Row to column delay (write)
constexpr uint32_t tRP = 18;        // Row precharge
constexpr uint32_t tRAS = 28;       // Row active time
constexpr uint32_t tRC = 46;        // Row cycle time
constexpr uint32_t tRL = 20;        // CAS read latency
constexpr uint32_t tWL = 8;         // CAS write latency
constexpr uint32_t tWR = 16;        // Write recovery
constexpr uint32_t tRTP = 8;        // Read to precharge
constexpr uint32_t tRRD_L = 4;      // ACT to ACT (same BG)
constexpr uint32_t tRRD_S = 4;      // ACT to ACT (diff BG)
constexpr uint32_t tCCD_L = 3;      // CAS to CAS (same BG)
constexpr uint32_t tCCD_S = 2;      // CAS to CAS (diff BG)
constexpr uint32_t tWTR_L = 6;      // Write to read (same BG)
constexpr uint32_t tWTR_S = 4;      // Write to read (diff BG)
constexpr uint32_t tRTW = 14;       // Read to write
constexpr uint32_t tFAW = 16;       // Four activate window
constexpr uint32_t tBurst = 4;      // BL16 burst cycles (BL16 / 4 WCK edges)

// ============================================================================
// Access Size Constants
// ============================================================================

/// Minimum access size (BL16 x 32-bit = 64 bytes)
constexpr uint32_t MIN_ACCESS_BYTES = 64;

/// Typical access size (64 bytes = cache line)
constexpr uint32_t CACHE_LINE_BYTES = 64;

/// Tile size (4KB = 32x32 int32)
constexpr uint32_t TILE_BYTES = 4096;

// ============================================================================
// Bandwidth Calculations
// ============================================================================

/// Calculate peak bandwidth in GB/s for given speed grade
/// @param data_rate_gbps Data rate in Gbps (e.g., 16 for GDDR6-16000)
/// @return Peak bandwidth in GB/s per device (both channels)
constexpr double peak_bandwidth_gbps(double data_rate_gbps) {
    // 32-bit bus (2 x 16-bit channels) * data rate / 8 bits per byte
    return data_rate_gbps * 32.0 / 8.0;
}

/// GDDR6-14000 peak bandwidth: 56 GB/s per device
constexpr double GDDR6_14000_BANDWIDTH = 56.0;

/// GDDR6-16000 peak bandwidth: 64 GB/s per device
constexpr double GDDR6_16000_BANDWIDTH = 64.0;

/// GDDR6-18000 peak bandwidth: 72 GB/s per device
constexpr double GDDR6_18000_BANDWIDTH = 72.0;

/// GDDR6-20000 peak bandwidth: 80 GB/s per device
constexpr double GDDR6_20000_BANDWIDTH = 80.0;

} // namespace sw::kpu::patterns::gddr6
