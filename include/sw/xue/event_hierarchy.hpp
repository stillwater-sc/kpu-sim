/**
 * @file event_hierarchy.hpp
 * @brief XUE (eXecution Usage Events) Event Hierarchy
 * @version 0.3.3
 *
 * XUE provides a hierarchical event taxonomy for operational analysis.
 * Events are organized in a tree structure that mirrors the KPU architecture:
 *
 *   System
 *   ├── Compute
 *   │   ├── Matmul
 *   │   │   ├── MATMUL_16x16        (tile-level)
 *   │   │   ├── MATMUL_ACCUMULATE
 *   │   │   └── MATMUL_WRITEBACK
 *   │   ├── Elementwise
 *   │   │   ├── ADD, MUL, DIV
 *   │   │   └── ACTIVATION (relu, sigmoid, tanh)
 *   │   └── Reduction
 *   │       ├── SUM, MAX, MEAN
 *   │       └── SOFTMAX
 *   ├── Memory
 *   │   ├── External (DRAM)
 *   │   │   ├── DRAM_READ, DRAM_WRITE
 *   │   │   └── DRAM_ACTIVATE, DRAM_PRECHARGE
 *   │   ├── L3
 *   │   │   ├── L3_TILE_PUSH, L3_TILE_POP
 *   │   │   └── L3_CREDIT_RETURN
 *   │   ├── L2
 *   │   │   ├── L2_TILE_PUSH, L2_TILE_POP
 *   │   │   └── L2_CREDIT_RETURN
 *   │   └── L1
 *   │       ├── L1_STREAM_FEED
 *   │       └── L1_CREDIT_RETURN
 *   └── DataMovement
 *       ├── DMA
 *       │   ├── DMA_TRANSFER_START
 *       │   └── DMA_TRANSFER_COMPLETE
 *       ├── BlockMover
 *       │   ├── BM_PUSH_L3_L2
 *       │   └── BM_CREDIT_WAIT
 *       └── Streamer
 *           ├── STR_FEED_L2_L1
 *           └── STR_CREDIT_WAIT
 *
 * Each event carries:
 *   - Event type (from hierarchy)
 *   - Count (number of occurrences)
 *   - Payload size (bytes, for memory/data movement)
 *   - Metadata (operation-specific details)
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <array>

namespace sw::xue {

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

/**
 * @brief Compute event subcategories
 */
enum class ComputeSubcategory : uint8_t {
    MATMUL = 0,
    ELEMENTWISE = 1,
    REDUCTION = 2,
    SPECIAL = 3,
    _COUNT = 4
};

/**
 * @brief Memory event subcategories
 */
enum class MemorySubcategory : uint8_t {
    EXTERNAL = 0,    // DRAM
    L3 = 1,
    L2 = 2,
    L1 = 3,
    _COUNT = 4
};

/**
 * @brief Data movement event subcategories
 */
enum class DataMovementSubcategory : uint8_t {
    DMA = 0,
    BLOCK_MOVER = 1,
    STREAMER = 2,
    NOC = 3,
    _COUNT = 4
};

/**
 * @brief Full enumeration of all leaf-level events
 *
 * Events are grouped by category and subcategory for efficient
 * hierarchical aggregation.
 */
enum class EventType : uint16_t {
    // === SYSTEM EVENTS (0x0000 - 0x00FF) ===
    SYSTEM_START = 0x0000,
    SYSTEM_END = 0x0001,
    KERNEL_START = 0x0002,
    KERNEL_END = 0x0003,
    TILE_ITERATION_START = 0x0004,
    TILE_ITERATION_END = 0x0005,

    // === COMPUTE EVENTS (0x0100 - 0x01FF) ===
    // Matmul (0x0100 - 0x010F)
    MATMUL_16x16 = 0x0100,           // Single systolic tile operation
    MATMUL_ACCUMULATE = 0x0101,      // Accumulate partial product
    MATMUL_WRITEBACK = 0x0102,       // Write result tile
    MATMUL_DRAIN = 0x0103,           // Drain systolic array

    // Elementwise (0x0110 - 0x011F)
    ELEM_ADD = 0x0110,
    ELEM_MUL = 0x0111,
    ELEM_DIV = 0x0112,
    ELEM_SUB = 0x0113,
    ELEM_RELU = 0x0114,
    ELEM_SIGMOID = 0x0115,
    ELEM_TANH = 0x0116,
    ELEM_GELU = 0x0117,
    ELEM_BIAS = 0x0118,

    // Reduction (0x0120 - 0x012F)
    REDUCE_SUM = 0x0120,
    REDUCE_MAX = 0x0121,
    REDUCE_MEAN = 0x0122,
    REDUCE_SOFTMAX = 0x0123,
    REDUCE_LAYERNORM = 0x0124,

    // Special (0x0130 - 0x013F)
    CONV_IM2COL = 0x0130,
    POOL_MAX = 0x0131,
    POOL_AVG = 0x0132,

    // === MEMORY EVENTS (0x0200 - 0x02FF) ===
    // External/DRAM (0x0200 - 0x020F)
    DRAM_READ = 0x0200,
    DRAM_WRITE = 0x0201,
    DRAM_ACTIVATE = 0x0202,
    DRAM_PRECHARGE = 0x0203,
    DRAM_REFRESH = 0x0204,

    // L3 Buffer (0x0210 - 0x021F)
    L3_TILE_PUSH = 0x0210,           // Tile arrives at L3
    L3_TILE_POP = 0x0211,            // Tile consumed from L3
    L3_CREDIT_RETURN = 0x0212,       // Credit returned upstream
    L3_CREDIT_WAIT = 0x0213,         // Waiting for credit

    // L2 Buffer (0x0220 - 0x022F)
    L2_TILE_PUSH = 0x0220,
    L2_TILE_POP = 0x0221,
    L2_CREDIT_RETURN = 0x0222,
    L2_CREDIT_WAIT = 0x0223,

    // L1 Stream (0x0230 - 0x023F)
    L1_STREAM_FEED = 0x0230,         // Data fed to compute
    L1_CREDIT_RETURN = 0x0231,
    L1_CREDIT_WAIT = 0x0232,

    // === DATA MOVEMENT EVENTS (0x0300 - 0x03FF) ===
    // DMA (0x0300 - 0x030F)
    DMA_TRANSFER_START = 0x0300,
    DMA_TRANSFER_COMPLETE = 0x0301,
    DMA_CREDIT_WAIT = 0x0302,
    DMA_CHANNEL_BUSY = 0x0303,

    // BlockMover (0x0310 - 0x031F)
    BM_PUSH_L3_L2 = 0x0310,          // L3 -> L2 transfer
    BM_CREDIT_WAIT = 0x0311,
    BM_TAG_MATCH = 0x0312,           // Tag CAM match

    // Streamer (0x0320 - 0x032F)
    STR_FEED_L2_L1 = 0x0320,         // L2 -> L1 transfer
    STR_CREDIT_WAIT = 0x0321,
    STR_TAG_MATCH = 0x0322,

    // NoC (0x0330 - 0x033F)
    NOC_PACKET_SEND = 0x0330,
    NOC_PACKET_RECV = 0x0331,
    NOC_HOP = 0x0332,

    // === SYNCHRONIZATION EVENTS (0x0400 - 0x04FF) ===
    SYNC_BARRIER = 0x0400,
    SYNC_CREDIT_STALL = 0x0401,
    SYNC_DATA_DEPENDENCY = 0x0402,

    // Sentinel
    _COUNT = 0x0500
};

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
    if (val < 0x0110) return ComputeSubcategory::MATMUL;
    if (val < 0x0120) return ComputeSubcategory::ELEMENTWISE;
    if (val < 0x0130) return ComputeSubcategory::REDUCTION;
    return ComputeSubcategory::SPECIAL;
}

/**
 * @brief Get the memory subcategory for a memory event
 */
constexpr MemorySubcategory get_memory_subcategory(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val < 0x0210) return MemorySubcategory::EXTERNAL;
    if (val < 0x0220) return MemorySubcategory::L3;
    if (val < 0x0230) return MemorySubcategory::L2;
    return MemorySubcategory::L1;
}

/**
 * @brief Get the data movement subcategory
 */
constexpr DataMovementSubcategory get_data_movement_subcategory(EventType type) {
    uint16_t val = static_cast<uint16_t>(type);
    if (val < 0x0310) return DataMovementSubcategory::DMA;
    if (val < 0x0320) return DataMovementSubcategory::BLOCK_MOVER;
    if (val < 0x0330) return DataMovementSubcategory::STREAMER;
    return DataMovementSubcategory::NOC;
}

/**
 * @brief Convert event type to string
 */
constexpr std::string_view to_string(EventType type) {
    switch (type) {
        case EventType::SYSTEM_START: return "SYSTEM_START";
        case EventType::SYSTEM_END: return "SYSTEM_END";
        case EventType::KERNEL_START: return "KERNEL_START";
        case EventType::KERNEL_END: return "KERNEL_END";
        case EventType::TILE_ITERATION_START: return "TILE_ITERATION_START";
        case EventType::TILE_ITERATION_END: return "TILE_ITERATION_END";

        case EventType::MATMUL_16x16: return "MATMUL_16x16";
        case EventType::MATMUL_ACCUMULATE: return "MATMUL_ACCUMULATE";
        case EventType::MATMUL_WRITEBACK: return "MATMUL_WRITEBACK";
        case EventType::MATMUL_DRAIN: return "MATMUL_DRAIN";

        case EventType::ELEM_ADD: return "ELEM_ADD";
        case EventType::ELEM_MUL: return "ELEM_MUL";
        case EventType::ELEM_DIV: return "ELEM_DIV";
        case EventType::ELEM_SUB: return "ELEM_SUB";
        case EventType::ELEM_RELU: return "ELEM_RELU";
        case EventType::ELEM_SIGMOID: return "ELEM_SIGMOID";
        case EventType::ELEM_TANH: return "ELEM_TANH";
        case EventType::ELEM_GELU: return "ELEM_GELU";
        case EventType::ELEM_BIAS: return "ELEM_BIAS";

        case EventType::REDUCE_SUM: return "REDUCE_SUM";
        case EventType::REDUCE_MAX: return "REDUCE_MAX";
        case EventType::REDUCE_MEAN: return "REDUCE_MEAN";
        case EventType::REDUCE_SOFTMAX: return "REDUCE_SOFTMAX";
        case EventType::REDUCE_LAYERNORM: return "REDUCE_LAYERNORM";

        case EventType::CONV_IM2COL: return "CONV_IM2COL";
        case EventType::POOL_MAX: return "POOL_MAX";
        case EventType::POOL_AVG: return "POOL_AVG";

        case EventType::DRAM_READ: return "DRAM_READ";
        case EventType::DRAM_WRITE: return "DRAM_WRITE";
        case EventType::DRAM_ACTIVATE: return "DRAM_ACTIVATE";
        case EventType::DRAM_PRECHARGE: return "DRAM_PRECHARGE";
        case EventType::DRAM_REFRESH: return "DRAM_REFRESH";

        case EventType::L3_TILE_PUSH: return "L3_TILE_PUSH";
        case EventType::L3_TILE_POP: return "L3_TILE_POP";
        case EventType::L3_CREDIT_RETURN: return "L3_CREDIT_RETURN";
        case EventType::L3_CREDIT_WAIT: return "L3_CREDIT_WAIT";

        case EventType::L2_TILE_PUSH: return "L2_TILE_PUSH";
        case EventType::L2_TILE_POP: return "L2_TILE_POP";
        case EventType::L2_CREDIT_RETURN: return "L2_CREDIT_RETURN";
        case EventType::L2_CREDIT_WAIT: return "L2_CREDIT_WAIT";

        case EventType::L1_STREAM_FEED: return "L1_STREAM_FEED";
        case EventType::L1_CREDIT_RETURN: return "L1_CREDIT_RETURN";
        case EventType::L1_CREDIT_WAIT: return "L1_CREDIT_WAIT";

        case EventType::DMA_TRANSFER_START: return "DMA_TRANSFER_START";
        case EventType::DMA_TRANSFER_COMPLETE: return "DMA_TRANSFER_COMPLETE";
        case EventType::DMA_CREDIT_WAIT: return "DMA_CREDIT_WAIT";
        case EventType::DMA_CHANNEL_BUSY: return "DMA_CHANNEL_BUSY";

        case EventType::BM_PUSH_L3_L2: return "BM_PUSH_L3_L2";
        case EventType::BM_CREDIT_WAIT: return "BM_CREDIT_WAIT";
        case EventType::BM_TAG_MATCH: return "BM_TAG_MATCH";

        case EventType::STR_FEED_L2_L1: return "STR_FEED_L2_L1";
        case EventType::STR_CREDIT_WAIT: return "STR_CREDIT_WAIT";
        case EventType::STR_TAG_MATCH: return "STR_TAG_MATCH";

        case EventType::NOC_PACKET_SEND: return "NOC_PACKET_SEND";
        case EventType::NOC_PACKET_RECV: return "NOC_PACKET_RECV";
        case EventType::NOC_HOP: return "NOC_HOP";

        case EventType::SYNC_BARRIER: return "SYNC_BARRIER";
        case EventType::SYNC_CREDIT_STALL: return "SYNC_CREDIT_STALL";
        case EventType::SYNC_DATA_DEPENDENCY: return "SYNC_DATA_DEPENDENCY";

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
        case ComputeSubcategory::MATMUL: return "MATMUL";
        case ComputeSubcategory::ELEMENTWISE: return "ELEMENTWISE";
        case ComputeSubcategory::REDUCTION: return "REDUCTION";
        case ComputeSubcategory::SPECIAL: return "SPECIAL";
        default: return "UNKNOWN";
    }
}

constexpr std::string_view to_string(MemorySubcategory sub) {
    switch (sub) {
        case MemorySubcategory::EXTERNAL: return "EXTERNAL";
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

/**
 * @brief Event metadata for operational analysis
 *
 * Each event occurrence can carry additional metadata for
 * aggregation and analysis.
 */
struct EventMetadata {
    uint64_t bytes;          // Bytes transferred (memory/data movement)
    uint64_t flops;          // FLOPs performed (compute)
    uint32_t tile_m;         // Tile dimension M
    uint32_t tile_n;         // Tile dimension N
    uint32_t tile_k;         // Tile dimension K
    uint16_t source_id;      // Source resource ID
    uint16_t dest_id;        // Destination resource ID

    EventMetadata() : bytes(0), flops(0), tile_m(0), tile_n(0), tile_k(0),
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
};

/**
 * @brief Standard tile sizes for the KPU
 */
struct KPUConstants {
    static constexpr uint32_t SYSTOLIC_SIZE = 16;        // 16x16 systolic array
    static constexpr uint64_t TILE_FLOPS = 16 * 16 * 2;  // 2*M*N*K for M=N=K=16
    static constexpr uint32_t L1_BUFFER_SIZE = 8192;     // 8KB per buffer
    static constexpr uint32_t L2_TILE_SIZE = 65536;      // 64KB tiles
    static constexpr uint32_t L3_TILE_SIZE = 262144;     // 256KB tiles
};

} // namespace sw::xue
