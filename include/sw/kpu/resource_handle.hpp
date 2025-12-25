#pragma once
// Resource Handle - Core types for identifying hardware resources
// Separated to avoid circular dependencies between resource_api.hpp and resource_stats.hpp

#include <cstdint>
#include <cstddef>
#include <string>

#include <sw/concepts.hpp>

namespace sw::kpu {

/**
 * @brief Types of hardware resources in the KPU
 *
 * The KPU has a hierarchical memory system and various compute/data movement resources.
 * This enum identifies all addressable resource types.
 */
enum class ResourceType : uint8_t {
    // Memory resources (hierarchical)
    HOST_MEMORY = 0,        // Host system memory (NUMA regions)
    EXTERNAL_MEMORY = 1,    // KPU-local memory (GDDR6/HBM banks)
    L3_TILE = 2,            // L3 distributed cache tiles
    L2_BANK = 3,            // L2 cache banks
    L1_BUFFER = 4,          // L1 streaming buffers (compute fabric)
    PAGE_BUFFER = 5,        // Page buffers to coalesce tile requests (memory controller)

    // Compute resources
    COMPUTE_TILE = 6,       // Compute tiles (systolic arrays)

    // Data movement resources
    DMA_ENGINE = 7,         // DMA engines for external transfers
    BLOCK_MOVER = 8,        // Block movers for L3-L2 transfers
    STREAMER = 9,           // Streamers for L2-L1 transfers

    // Sentinel
    COUNT = 10
};

/**
 * @brief Get string name of a resource type
 * @param type The resource type
 * @return String representation
 */
inline std::string resource_type_name(ResourceType type) {
    switch (type) {
        case ResourceType::HOST_MEMORY: return "host_memory";
        case ResourceType::EXTERNAL_MEMORY: return "external_memory";
        case ResourceType::L3_TILE: return "l3_tile";
        case ResourceType::L2_BANK: return "l2_bank";
        case ResourceType::L1_BUFFER: return "l1_buffer";
        case ResourceType::PAGE_BUFFER: return "page_buffer";
        case ResourceType::COMPUTE_TILE: return "compute_tile";
        case ResourceType::DMA_ENGINE: return "dma_engine";
        case ResourceType::BLOCK_MOVER: return "block_mover";
        case ResourceType::STREAMER: return "streamer";
        default: return "unknown";
    }
}

/**
 * @brief Check if a resource type is a memory resource
 * @param type The resource type
 * @return true if memory resource
 */
constexpr bool is_memory_resource(ResourceType type) {
    switch (type) {
        case ResourceType::HOST_MEMORY:
        case ResourceType::EXTERNAL_MEMORY:
        case ResourceType::L3_TILE:
        case ResourceType::L2_BANK:
        case ResourceType::L1_BUFFER:
        case ResourceType::PAGE_BUFFER:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a resource type is a compute resource
 * @param type The resource type
 * @return true if compute resource
 */
constexpr bool is_compute_resource(ResourceType type) {
    return type == ResourceType::COMPUTE_TILE;
}

/**
 * @brief Check if a resource type is a data movement resource
 * @param type The resource type
 * @return true if data movement resource
 */
constexpr bool is_data_movement_resource(ResourceType type) {
    switch (type) {
        case ResourceType::DMA_ENGINE:
        case ResourceType::BLOCK_MOVER:
        case ResourceType::STREAMER:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Handle to a hardware resource
 *
 * ResourceHandle provides a unified way to identify and access any hardware
 * resource in the KPU. Handles are lightweight value types that can be
 * stored, passed, and compared efficiently.
 */
struct ResourceHandle {
    ResourceType type;          // Type of resource
    size_t id;                  // Resource index within its type
    Address base_address;       // Base address in unified address space (for memory resources)
    Size capacity;              // Capacity in bytes (for memory resources)

    ResourceHandle()
        : type(ResourceType::COUNT), id(0), base_address(0), capacity(0) {}

    ResourceHandle(ResourceType t, size_t i, Address base = 0, Size cap = 0)
        : type(t), id(i), base_address(base), capacity(cap) {}

    // Check if handle is valid
    bool is_valid() const { return type != ResourceType::COUNT; }

    // Check if handle refers to a memory resource
    bool is_memory() const { return is_memory_resource(type); }

    // Check if handle refers to a compute resource
    bool is_compute() const { return is_compute_resource(type); }

    // Check if handle refers to a data movement resource
    bool is_data_movement() const { return is_data_movement_resource(type); }

    // Equality comparison
    bool operator==(const ResourceHandle& other) const {
        return type == other.type && id == other.id;
    }

    bool operator!=(const ResourceHandle& other) const {
        return !(*this == other);
    }

    // String representation for debugging
    std::string to_string() const {
        return resource_type_name(type) + "[" + std::to_string(id) + "]";
    }
};

} // namespace sw::kpu
