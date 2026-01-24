/**
 * @file event_hierarchy.hpp
 * @brief XUE Observation Architecture - Event Hierarchy
 * @version 0.4.0
 *
 * XUE provides a hierarchical event taxonomy for operational analysis,
 * modeled after the Intel Patent 6,023,759 Observation Architecture.
 *
 * The event hierarchy supports:
 *   - Aggregation (roll-up): Primitive events -> Operations -> Categories
 *   - Drill-down: Categories -> Operations -> Primitive events
 *   - Occupancy tracking for Operational Analysis (Little's Law)
 *
 * Event Metrics per Intel OA:
 *   - Occurrences: Number of times event occurred
 *   - Occupancy: Number of items in queue/buffer (tracked per cycle)
 *   - Latency: Time from arrival to completion
 *   - Cycles: Total clock cycles attributed to event
 *
 * XUE Metrics:
 *   - X (Throughput): Work completed per unit time
 *   - U (Utilization): Fraction of time resource is busy
 *   - E (Efficiency): Fraction of peak capability achieved
 *
 * Memory Hierarchy (all levels have reads/writes with sizes):
 *   - DRAM: reads, writes, activate, precharge, refresh
 *   - L3 (Tile Store): tile reads, tile writes
 *   - L2 (Tile Store): tile reads, tile writes
 *   - L1 (Stream Buffers): vectors read/written, elements to/from compute
 *
 * Data Movement:
 *   - DMA: reads from DRAM, writes to L3
 *   - BlockMover: reads from L3, writes to L2
 *   - Streamer: reads from L2, writes to L1; reads from L1, writes to L2
 *
 * Compute:
 *   - ALU primitives: MAC, MUL, ADD, DIV, EXP, TANH, etc.
 *   - Named operations: MATMUL, RELU, GELU, SOFTMAX (aggregate ALU ops)
 *
 * Synchronization:
 *   - Credit updates and stalls
 *   - Barriers and dependencies
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <array>

namespace sw::xue {

// ============================================================================
// Event Categories (Top-Level Hierarchy)
// ============================================================================

/**
 * @brief Top-level event categories
 */
enum class EventCategory : uint8_t {
    SYSTEM = 0,
    COMPUTE = 1,
    MEMORY = 2,
    DATA_MOVEMENT = 3,
    SYNCHRONIZATION = 4,
    _COUNT = 5
};

// ============================================================================
// Compute Event Hierarchy
// ============================================================================

/**
 * @brief Compute event subcategories
 *
 * ALU_PRIMITIVE: Individual arithmetic operations (MAC, MUL, ADD, etc.)
 * NAMED_OP: Aggregate operations built from primitives (MATMUL, RELU, etc.)
 */
enum class ComputeSubcategory : uint8_t {
    ALU_PRIMITIVE = 0,    // Primitive ALU operations
    NAMED_OP = 1,         // Named/aggregate operations
    REDUCTION = 2,        // Reduction operations
    SPECIAL = 3,          // Special function unit operations
    _COUNT = 4
};

// ============================================================================
// Memory Event Hierarchy
// ============================================================================

/**
 * @brief Memory event subcategories
 *
 * Each memory level tracks reads and writes with sizes.
 */
enum class MemorySubcategory : uint8_t {
    DRAM = 0,       // External memory
    L3 = 1,         // L3 tile store
    L2 = 2,         // L2 tile store
    L1 = 3,         // L1 stream buffers
    _COUNT = 4
};

// ============================================================================
// Data Movement Event Hierarchy
// ============================================================================

/**
 * @brief Data movement event subcategories
 */
enum class DataMovementSubcategory : uint8_t {
    DMA = 0,          // DMA engine (DRAM <-> L3)
    BLOCK_MOVER = 1,  // BlockMover (L3 <-> L2)
    STREAMER = 2,     // Streamer (L2 <-> L1)
    NOC = 3,          // Network on Chip
    _COUNT = 4
};

// ============================================================================
// Full Event Type Enumeration
// ============================================================================

/**
 * @brief Full enumeration of all leaf-level events
 *
 * Events are grouped by category and subcategory for efficient
 * hierarchical aggregation.
 *
 * Encoding: 0xCCSS where CC = category, SS = subcategory + event
 */
enum class EventType : uint16_t {
    // ========================================================================
    // SYSTEM EVENTS (0x0000 - 0x00FF)
    // ========================================================================
    SYSTEM_START = 0x0000,
    SYSTEM_END = 0x0001,
    KERNEL_START = 0x0002,
    KERNEL_END = 0x0003,
    TILE_ITERATION_START = 0x0004,
    TILE_ITERATION_END = 0x0005,

    // ========================================================================
    // COMPUTE EVENTS (0x0100 - 0x01FF)
    // ========================================================================

    // --- ALU Primitives (0x0100 - 0x010F) ---
    // These are the fundamental ALU operations that get aggregated
    // into named operations for analysis
    ALU_MAC = 0x0100,       // Multiply-accumulate
    ALU_MUL = 0x0101,       // Multiply
    ALU_ADD = 0x0102,       // Add
    ALU_SUB = 0x0103,       // Subtract
    ALU_DIV = 0x0104,       // Divide
    ALU_SQRT = 0x0105,      // Square root
    ALU_EXP = 0x0106,       // Exponential
    ALU_LOG = 0x0107,       // Logarithm
    ALU_TANH = 0x0108,      // Hyperbolic tangent
    ALU_SIGMOID = 0x0109,   // Sigmoid function
    ALU_RELU = 0x010A,      // ReLU (max(0,x))
    ALU_GELU = 0x010B,      // GELU activation
    ALU_CMP = 0x010C,       // Compare
    ALU_ABS = 0x010D,       // Absolute value
    ALU_NEG = 0x010E,       // Negate
    ALU_RECIP = 0x010F,     // Reciprocal

    // --- Named Operations (0x0110 - 0x011F) ---
    // These aggregate ALU primitives for high-level analysis
    OP_MATMUL = 0x0110,     // Matrix multiply (aggregates MACs)
    OP_GEMM = 0x0111,       // General matrix multiply (C = αAB + βC)
    OP_CONV2D = 0x0112,     // 2D convolution
    OP_BIAS_ADD = 0x0113,   // Bias addition
    OP_RELU = 0x0114,       // ReLU layer
    OP_GELU = 0x0115,       // GELU layer
    OP_TANH = 0x0116,       // Tanh layer
    OP_SIGMOID = 0x0117,    // Sigmoid layer
    OP_LAYERNORM = 0x0118,  // Layer normalization
    OP_BATCHNORM = 0x0119,  // Batch normalization

    // --- Reduction Operations (0x0120 - 0x012F) ---
    REDUCE_SUM = 0x0120,
    REDUCE_MAX = 0x0121,
    REDUCE_MEAN = 0x0122,
    REDUCE_SOFTMAX = 0x0123,
    REDUCE_ARGMAX = 0x0124,

    // --- Pooling and Convolution Operations (0x0128 - 0x012F) ---
    OP_POOL_MAX = 0x0128,    // Max pooling
    OP_POOL_AVG = 0x0129,    // Average pooling
    OP_IM2COL = 0x012A,      // Im2col transformation for convolution

    // --- Special Function Unit Operations (0x0130 - 0x013F) ---
    SFU_RSQRT = 0x0130,     // Reciprocal square root
    SFU_SIN = 0x0131,       // Sine
    SFU_COS = 0x0132,       // Cosine
    SFU_ATAN = 0x0133,      // Arctangent

    // ========================================================================
    // MEMORY EVENTS (0x0200 - 0x02FF)
    // All memory levels track reads and writes with sizes
    // ========================================================================

    // --- DRAM / External Memory (0x0200 - 0x020F) ---
    DRAM_READ = 0x0200,           // Read from DRAM (bytes)
    DRAM_WRITE = 0x0201,          // Write to DRAM (bytes)
    DRAM_ACTIVATE = 0x0202,       // Row activation
    DRAM_PRECHARGE = 0x0203,      // Row precharge
    DRAM_REFRESH = 0x0204,        // DRAM refresh cycle
    DRAM_PAGE_HIT = 0x0205,       // Row buffer hit
    DRAM_PAGE_MISS = 0x0206,      // Row buffer miss

    // --- L3 Tile Store (0x0210 - 0x021F) ---
    L3_TILE_READ = 0x0210,        // Read tile from L3 (bytes)
    L3_TILE_WRITE = 0x0211,       // Write tile to L3 (bytes)
    L3_BUFFER_FULL = 0x0212,      // L3 buffer full event
    L3_BUFFER_EMPTY = 0x0213,     // L3 buffer empty event

    // --- L2 Tile Store (0x0220 - 0x022F) ---
    L2_TILE_READ = 0x0220,        // Read tile from L2 (bytes)
    L2_TILE_WRITE = 0x0221,       // Write tile to L2 (bytes)
    L2_BUFFER_FULL = 0x0222,      // L2 buffer full event
    L2_BUFFER_EMPTY = 0x0223,     // L2 buffer empty event

    // --- L1 Stream Buffers (0x0230 - 0x023F) ---
    // L1 operates at vector and element granularity
    L1_VECTOR_READ = 0x0230,      // Read vector from L1 buffer (bytes)
    L1_VECTOR_WRITE = 0x0231,     // Write vector to L1 buffer (bytes)
    L1_ELEMENT_TO_COMPUTE = 0x0232,   // Element sent to compute (count)
    L1_ELEMENT_FROM_COMPUTE = 0x0233, // Element received from compute (count)
    L1_BUFFER_FULL = 0x0234,      // L1 buffer full event
    L1_BUFFER_EMPTY = 0x0235,     // L1 buffer empty event

    // ========================================================================
    // DATA MOVEMENT EVENTS (0x0300 - 0x03FF)
    // Each mover tracks reads and writes from its perspective
    // ========================================================================

    // --- DMA Engine (0x0300 - 0x030F) ---
    // DMA reads from DRAM, writes to L3
    DMA_READ_DRAM = 0x0300,       // DMA read from DRAM (bytes)
    DMA_WRITE_L3 = 0x0301,        // DMA write to L3 (bytes)
    DMA_READ_L3 = 0x0302,         // DMA read from L3 for writeback (bytes)
    DMA_WRITE_DRAM = 0x0303,      // DMA write to DRAM (bytes)
    DMA_TRANSFER_START = 0x0304,
    DMA_TRANSFER_COMPLETE = 0x0305,
    DMA_CHANNEL_BUSY = 0x0306,

    // --- BlockMover (0x0310 - 0x031F) ---
    // BlockMover reads from L3, writes to L2
    BM_READ_L3 = 0x0310,          // BlockMover read from L3 (bytes)
    BM_WRITE_L2 = 0x0311,         // BlockMover write to L2 (bytes)
    BM_READ_L2 = 0x0312,          // BlockMover read from L2 for writeback (bytes)
    BM_WRITE_L3 = 0x0313,         // BlockMover write to L3 (bytes)
    BM_TAG_MATCH = 0x0314,        // Tag CAM match

    // --- Streamer (0x0320 - 0x032F) ---
    // Streamer reads from L2, writes to stream buffers (L1)
    // Also reads from stream buffers, writes to L2
    STR_READ_L2 = 0x0320,         // Streamer read from L2 (bytes)
    STR_WRITE_BUFFER = 0x0321,    // Streamer write to stream buffer (bytes)
    STR_READ_BUFFER = 0x0322,     // Streamer read from stream buffer (bytes)
    STR_WRITE_L2 = 0x0323,        // Streamer write to L2 (bytes)
    STR_VECTOR_FILL = 0x0324,     // Vector fill operation
    STR_VECTOR_DRAIN = 0x0325,    // Vector drain operation

    // --- NoC (0x0330 - 0x033F) ---
    NOC_PACKET_SEND = 0x0330,
    NOC_PACKET_RECV = 0x0331,
    NOC_HOP = 0x0332,
    NOC_CONGESTION = 0x0333,

    // ========================================================================
    // SYNCHRONIZATION EVENTS (0x0400 - 0x04FF)
    // ========================================================================

    // --- Credit Events (0x0400 - 0x040F) ---
    // Track both credit updates and credit stalls
    CREDIT_L3_UPDATE = 0x0400,    // L3 credit updated
    CREDIT_L3_STALL = 0x0401,     // Stalled waiting for L3 credit
    CREDIT_L2_UPDATE = 0x0402,    // L2 credit updated
    CREDIT_L2_STALL = 0x0403,     // Stalled waiting for L2 credit
    CREDIT_L1_UPDATE = 0x0404,    // L1 credit updated
    CREDIT_L1_STALL = 0x0405,     // Stalled waiting for L1 credit
    CREDIT_DMA_UPDATE = 0x0406,   // DMA credit updated
    CREDIT_DMA_STALL = 0x0407,    // Stalled waiting for DMA credit

    // --- Barrier Events (0x0410 - 0x041F) ---
    BARRIER_ENTER = 0x0410,
    BARRIER_EXIT = 0x0411,
    BARRIER_WAIT_CYCLES = 0x0412,

    // --- Dependency Events (0x0420 - 0x042F) ---
    DEP_DATA_WAIT = 0x0420,       // Waiting for data dependency
    DEP_DATA_READY = 0x0421,      // Data dependency satisfied
    DEP_CONTROL_WAIT = 0x0422,    // Waiting for control dependency
    DEP_CONTROL_READY = 0x0423,   // Control dependency satisfied

    // Sentinel
    _COUNT = 0x0500
};

// ============================================================================
// Helper Functions for Event Type Classification
// ============================================================================

/**
 * @brief Get the category for an event type
 */
constexpr EventCategory get_category(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val < 0x0100) return EventCategory::SYSTEM;
    if (val < 0x0200) return EventCategory::COMPUTE;
    if (val < 0x0300) return EventCategory::MEMORY;
    if (val < 0x0400) return EventCategory::DATA_MOVEMENT;
    return EventCategory::SYNCHRONIZATION;
}

/**
 * @brief Get the compute subcategory for a compute event
 */
constexpr ComputeSubcategory get_compute_subcategory(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val >= 0x0100 && val < 0x0110) return ComputeSubcategory::ALU_PRIMITIVE;
    if (val >= 0x0110 && val < 0x0120) return ComputeSubcategory::NAMED_OP;
    if (val >= 0x0120 && val < 0x0130) return ComputeSubcategory::REDUCTION;
    return ComputeSubcategory::SPECIAL;
}

/**
 * @brief Get the memory subcategory for a memory event
 */
constexpr MemorySubcategory get_memory_subcategory(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val >= 0x0200 && val < 0x0210) return MemorySubcategory::DRAM;
    if (val >= 0x0210 && val < 0x0220) return MemorySubcategory::L3;
    if (val >= 0x0220 && val < 0x0230) return MemorySubcategory::L2;
    return MemorySubcategory::L1;
}

/**
 * @brief Get the data movement subcategory
 */
constexpr DataMovementSubcategory get_data_movement_subcategory(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val >= 0x0300 && val < 0x0310) return DataMovementSubcategory::DMA;
    if (val >= 0x0310 && val < 0x0320) return DataMovementSubcategory::BLOCK_MOVER;
    if (val >= 0x0320 && val < 0x0330) return DataMovementSubcategory::STREAMER;
    return DataMovementSubcategory::NOC;
}

/**
 * @brief Check if event is an ALU primitive
 */
constexpr bool is_alu_primitive(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    return val >= 0x0100 && val < 0x0110;
}

/**
 * @brief Check if event is a named operation
 */
constexpr bool is_named_operation(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    return val >= 0x0110 && val < 0x0120;
}

/**
 * @brief Check if event involves memory transfer
 */
constexpr bool is_memory_transfer(EventType type) {
    switch (type) {
        case EventType::DRAM_READ:
        case EventType::DRAM_WRITE:
        case EventType::L3_TILE_READ:
        case EventType::L3_TILE_WRITE:
        case EventType::L2_TILE_READ:
        case EventType::L2_TILE_WRITE:
        case EventType::L1_VECTOR_READ:
        case EventType::L1_VECTOR_WRITE:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if event is a stall/wait event (for utilization calculation)
 */
constexpr bool is_stall_event(EventType type) {
    switch (type) {
        case EventType::CREDIT_L3_STALL:
        case EventType::CREDIT_L2_STALL:
        case EventType::CREDIT_L1_STALL:
        case EventType::CREDIT_DMA_STALL:
        case EventType::BARRIER_WAIT_CYCLES:
        case EventType::DEP_DATA_WAIT:
        case EventType::DEP_CONTROL_WAIT:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Event Type String Conversion
// ============================================================================

/**
 * @brief Convert event type to string
 */
constexpr std::string_view to_string(EventType type) {
    switch (type) {
        // System events
        case EventType::SYSTEM_START: return "SYSTEM_START";
        case EventType::SYSTEM_END: return "SYSTEM_END";
        case EventType::KERNEL_START: return "KERNEL_START";
        case EventType::KERNEL_END: return "KERNEL_END";
        case EventType::TILE_ITERATION_START: return "TILE_ITERATION_START";
        case EventType::TILE_ITERATION_END: return "TILE_ITERATION_END";

        // ALU primitives
        case EventType::ALU_MAC: return "ALU_MAC";
        case EventType::ALU_MUL: return "ALU_MUL";
        case EventType::ALU_ADD: return "ALU_ADD";
        case EventType::ALU_SUB: return "ALU_SUB";
        case EventType::ALU_DIV: return "ALU_DIV";
        case EventType::ALU_SQRT: return "ALU_SQRT";
        case EventType::ALU_EXP: return "ALU_EXP";
        case EventType::ALU_LOG: return "ALU_LOG";
        case EventType::ALU_TANH: return "ALU_TANH";
        case EventType::ALU_SIGMOID: return "ALU_SIGMOID";
        case EventType::ALU_RELU: return "ALU_RELU";
        case EventType::ALU_GELU: return "ALU_GELU";
        case EventType::ALU_CMP: return "ALU_CMP";
        case EventType::ALU_ABS: return "ALU_ABS";
        case EventType::ALU_NEG: return "ALU_NEG";
        case EventType::ALU_RECIP: return "ALU_RECIP";

        // Named operations
        case EventType::OP_MATMUL: return "OP_MATMUL";
        case EventType::OP_GEMM: return "OP_GEMM";
        case EventType::OP_CONV2D: return "OP_CONV2D";
        case EventType::OP_BIAS_ADD: return "OP_BIAS_ADD";
        case EventType::OP_RELU: return "OP_RELU";
        case EventType::OP_GELU: return "OP_GELU";
        case EventType::OP_TANH: return "OP_TANH";
        case EventType::OP_SIGMOID: return "OP_SIGMOID";
        case EventType::OP_LAYERNORM: return "OP_LAYERNORM";
        case EventType::OP_BATCHNORM: return "OP_BATCHNORM";

        // Reduction operations
        case EventType::REDUCE_SUM: return "REDUCE_SUM";
        case EventType::REDUCE_MAX: return "REDUCE_MAX";
        case EventType::REDUCE_MEAN: return "REDUCE_MEAN";
        case EventType::REDUCE_SOFTMAX: return "REDUCE_SOFTMAX";
        case EventType::REDUCE_ARGMAX: return "REDUCE_ARGMAX";

        // Pooling and convolution
        case EventType::OP_POOL_MAX: return "OP_POOL_MAX";
        case EventType::OP_POOL_AVG: return "OP_POOL_AVG";
        case EventType::OP_IM2COL: return "OP_IM2COL";

        // Special function unit
        case EventType::SFU_RSQRT: return "SFU_RSQRT";
        case EventType::SFU_SIN: return "SFU_SIN";
        case EventType::SFU_COS: return "SFU_COS";
        case EventType::SFU_ATAN: return "SFU_ATAN";

        // DRAM events
        case EventType::DRAM_READ: return "DRAM_READ";
        case EventType::DRAM_WRITE: return "DRAM_WRITE";
        case EventType::DRAM_ACTIVATE: return "DRAM_ACTIVATE";
        case EventType::DRAM_PRECHARGE: return "DRAM_PRECHARGE";
        case EventType::DRAM_REFRESH: return "DRAM_REFRESH";
        case EventType::DRAM_PAGE_HIT: return "DRAM_PAGE_HIT";
        case EventType::DRAM_PAGE_MISS: return "DRAM_PAGE_MISS";

        // L3 events
        case EventType::L3_TILE_READ: return "L3_TILE_READ";
        case EventType::L3_TILE_WRITE: return "L3_TILE_WRITE";
        case EventType::L3_BUFFER_FULL: return "L3_BUFFER_FULL";
        case EventType::L3_BUFFER_EMPTY: return "L3_BUFFER_EMPTY";

        // L2 events
        case EventType::L2_TILE_READ: return "L2_TILE_READ";
        case EventType::L2_TILE_WRITE: return "L2_TILE_WRITE";
        case EventType::L2_BUFFER_FULL: return "L2_BUFFER_FULL";
        case EventType::L2_BUFFER_EMPTY: return "L2_BUFFER_EMPTY";

        // L1 events
        case EventType::L1_VECTOR_READ: return "L1_VECTOR_READ";
        case EventType::L1_VECTOR_WRITE: return "L1_VECTOR_WRITE";
        case EventType::L1_ELEMENT_TO_COMPUTE: return "L1_ELEMENT_TO_COMPUTE";
        case EventType::L1_ELEMENT_FROM_COMPUTE: return "L1_ELEMENT_FROM_COMPUTE";
        case EventType::L1_BUFFER_FULL: return "L1_BUFFER_FULL";
        case EventType::L1_BUFFER_EMPTY: return "L1_BUFFER_EMPTY";

        // DMA events
        case EventType::DMA_READ_DRAM: return "DMA_READ_DRAM";
        case EventType::DMA_WRITE_L3: return "DMA_WRITE_L3";
        case EventType::DMA_READ_L3: return "DMA_READ_L3";
        case EventType::DMA_WRITE_DRAM: return "DMA_WRITE_DRAM";
        case EventType::DMA_TRANSFER_START: return "DMA_TRANSFER_START";
        case EventType::DMA_TRANSFER_COMPLETE: return "DMA_TRANSFER_COMPLETE";
        case EventType::DMA_CHANNEL_BUSY: return "DMA_CHANNEL_BUSY";

        // BlockMover events
        case EventType::BM_READ_L3: return "BM_READ_L3";
        case EventType::BM_WRITE_L2: return "BM_WRITE_L2";
        case EventType::BM_READ_L2: return "BM_READ_L2";
        case EventType::BM_WRITE_L3: return "BM_WRITE_L3";
        case EventType::BM_TAG_MATCH: return "BM_TAG_MATCH";

        // Streamer events
        case EventType::STR_READ_L2: return "STR_READ_L2";
        case EventType::STR_WRITE_BUFFER: return "STR_WRITE_BUFFER";
        case EventType::STR_READ_BUFFER: return "STR_READ_BUFFER";
        case EventType::STR_WRITE_L2: return "STR_WRITE_L2";
        case EventType::STR_VECTOR_FILL: return "STR_VECTOR_FILL";
        case EventType::STR_VECTOR_DRAIN: return "STR_VECTOR_DRAIN";

        // NoC events
        case EventType::NOC_PACKET_SEND: return "NOC_PACKET_SEND";
        case EventType::NOC_PACKET_RECV: return "NOC_PACKET_RECV";
        case EventType::NOC_HOP: return "NOC_HOP";
        case EventType::NOC_CONGESTION: return "NOC_CONGESTION";

        // Credit events
        case EventType::CREDIT_L3_UPDATE: return "CREDIT_L3_UPDATE";
        case EventType::CREDIT_L3_STALL: return "CREDIT_L3_STALL";
        case EventType::CREDIT_L2_UPDATE: return "CREDIT_L2_UPDATE";
        case EventType::CREDIT_L2_STALL: return "CREDIT_L2_STALL";
        case EventType::CREDIT_L1_UPDATE: return "CREDIT_L1_UPDATE";
        case EventType::CREDIT_L1_STALL: return "CREDIT_L1_STALL";
        case EventType::CREDIT_DMA_UPDATE: return "CREDIT_DMA_UPDATE";
        case EventType::CREDIT_DMA_STALL: return "CREDIT_DMA_STALL";

        // Barrier events
        case EventType::BARRIER_ENTER: return "BARRIER_ENTER";
        case EventType::BARRIER_EXIT: return "BARRIER_EXIT";
        case EventType::BARRIER_WAIT_CYCLES: return "BARRIER_WAIT_CYCLES";

        // Dependency events
        case EventType::DEP_DATA_WAIT: return "DEP_DATA_WAIT";
        case EventType::DEP_DATA_READY: return "DEP_DATA_READY";
        case EventType::DEP_CONTROL_WAIT: return "DEP_CONTROL_WAIT";
        case EventType::DEP_CONTROL_READY: return "DEP_CONTROL_READY";

        default: return "UNKNOWN";
    }
}

constexpr std::string_view to_string(EventCategory cat) {
    switch (cat) {
        case EventCategory::SYSTEM: return "SYSTEM";
        case EventCategory::COMPUTE: return "COMPUTE";
        case EventCategory::MEMORY: return "MEMORY";
        case EventCategory::DATA_MOVEMENT: return "DATA_MOVEMENT";
        case EventCategory::SYNCHRONIZATION: return "SYNCHRONIZATION";
        default: return "UNKNOWN";
    }
}

constexpr std::string_view to_string(ComputeSubcategory sub) {
    switch (sub) {
        case ComputeSubcategory::ALU_PRIMITIVE: return "ALU_PRIMITIVE";
        case ComputeSubcategory::NAMED_OP: return "NAMED_OP";
        case ComputeSubcategory::REDUCTION: return "REDUCTION";
        case ComputeSubcategory::SPECIAL: return "SPECIAL";
        default: return "UNKNOWN";
    }
}

constexpr std::string_view to_string(MemorySubcategory sub) {
    switch (sub) {
        case MemorySubcategory::DRAM: return "DRAM";
        case MemorySubcategory::L3: return "L3";
        case MemorySubcategory::L2: return "L2";
        case MemorySubcategory::L1: return "L1";
        default: return "UNKNOWN";
    }
}

constexpr std::string_view to_string(DataMovementSubcategory sub) {
    switch (sub) {
        case DataMovementSubcategory::DMA: return "DMA";
        case DataMovementSubcategory::BLOCK_MOVER: return "BLOCK_MOVER";
        case DataMovementSubcategory::STREAMER: return "STREAMER";
        case DataMovementSubcategory::NOC: return "NOC";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Event Metadata
// ============================================================================

/**
 * @brief Event metadata for operational analysis
 *
 * Each event occurrence can carry additional metadata for
 * aggregation and analysis.
 */
struct EventMetadata {
    uint64_t bytes;          // Bytes transferred (memory/data movement)
    uint64_t flops;          // FLOPs performed (compute)
    uint64_t elements;       // Number of elements (for L1 element transfers)
    uint32_t tile_m;         // Tile dimension M
    uint32_t tile_n;         // Tile dimension N
    uint32_t tile_k;         // Tile dimension K
    uint16_t source_id;      // Source resource ID
    uint16_t dest_id;        // Destination resource ID

    EventMetadata() : bytes(0), flops(0), elements(0),
                      tile_m(0), tile_n(0), tile_k(0),
                      source_id(0), dest_id(0) {}

    static EventMetadata compute(uint64_t flops, uint32_t m = 0, uint32_t n = 0, uint32_t k = 0) {
        EventMetadata meta;
        meta.flops = flops;
        meta.tile_m = m;
        meta.tile_n = n;
        meta.tile_k = k;
        return meta;
    }

    static EventMetadata memory(uint64_t bytes, uint16_t src = 0, uint16_t dst = 0) {
        EventMetadata meta;
        meta.bytes = bytes;
        meta.source_id = src;
        meta.dest_id = dst;
        return meta;
    }

    static EventMetadata element_transfer(uint64_t elements) {
        EventMetadata meta;
        meta.elements = elements;
        return meta;
    }
};

// ============================================================================
// KPU Hardware Constants
// ============================================================================

/**
 * @brief Standard hardware parameters for the KPU
 */
struct KPUConstants {
    static constexpr uint32_t SYSTOLIC_SIZE = 16;        // 16x16 systolic array
    static constexpr uint32_t VECTOR_WIDTH = 64;         // 64-element vectors
    static constexpr uint64_t TILE_FLOPS = 16 * 16 * 2;  // 2*M*N*K for M=N=K=16
    static constexpr uint32_t L1_BUFFER_SIZE = 8192;     // 8KB per buffer
    static constexpr uint32_t L2_TILE_SIZE = 65536;      // 64KB tiles
    static constexpr uint32_t L3_TILE_SIZE = 262144;     // 256KB tiles
};

} // namespace sw::xue
