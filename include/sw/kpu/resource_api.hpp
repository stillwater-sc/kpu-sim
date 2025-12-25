#pragma once
// Resource API for KPU simulator
// Provides unified access to all addressable hardware resources

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <optional>

#include <sw/concepts.hpp>
#include <sw/kpu/data_types.hpp>

namespace sw::kpu {

// Forward declarations
class KPUSimulator;

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

/**
 * @brief Information about a memory allocation
 */
struct AllocationInfo {
    Address address;            // Allocated address
    Size size;                  // Allocated size in bytes
    Size alignment;             // Alignment used
    ResourceHandle resource;    // Resource where allocated
    std::string label;          // Optional label for debugging

    AllocationInfo()
        : address(0), size(0), alignment(0), resource(), label() {}

    AllocationInfo(Address addr, Size sz, Size align, ResourceHandle res, const std::string& lbl = "")
        : address(addr), size(sz), alignment(align), resource(res), label(lbl) {}
};

/**
 * @brief Resource Manager for unified access to all KPU resources
 *
 * ResourceManager provides a single interface to:
 * - Query available resources and their properties
 * - Allocate and deallocate memory in any memory resource
 * - Read and write data to memory resources
 * - Check resource busy status
 *
 * The ResourceManager holds a reference to the underlying KPUSimulator
 * and delegates operations to the appropriate components.
 */
class ResourceManager {
public:
    /**
     * @brief Construct a ResourceManager attached to a KPUSimulator
     * @param simulator Reference to the KPUSimulator instance
     */
    explicit ResourceManager(KPUSimulator& simulator);

    ~ResourceManager() = default;

    // Non-copyable, movable
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
    ResourceManager(ResourceManager&&) = default;
    ResourceManager& operator=(ResourceManager&&) = default;

    // =========================================
    // Resource Discovery
    // =========================================

    /**
     * @brief Get count of resources of a given type
     * @param type The resource type
     * @return Number of resources of that type
     */
    size_t get_resource_count(ResourceType type) const;

    /**
     * @brief Get handle to a specific resource
     * @param type The resource type
     * @param id The resource index
     * @return ResourceHandle for the resource
     * @throws std::out_of_range if id is invalid
     */
    ResourceHandle get_resource(ResourceType type, size_t id) const;

    /**
     * @brief Get all resources of a given type
     * @param type The resource type
     * @return Vector of ResourceHandles
     */
    std::vector<ResourceHandle> get_all_resources(ResourceType type) const;

    /**
     * @brief Get all memory resources
     * @return Vector of ResourceHandles for all memory resources
     */
    std::vector<ResourceHandle> get_memory_resources() const;

    /**
     * @brief Get all compute resources
     * @return Vector of ResourceHandles for all compute resources
     */
    std::vector<ResourceHandle> get_compute_resources() const;

    /**
     * @brief Get all data movement resources
     * @return Vector of ResourceHandles for all data movement resources
     */
    std::vector<ResourceHandle> get_data_movement_resources() const;

    // =========================================
    // Memory Allocation
    // =========================================

    /**
     * @brief Allocate memory in a specific resource
     * @param resource Handle to the memory resource
     * @param size Size in bytes to allocate
     * @param alignment Required alignment (must be power of 2, default 64)
     * @param label Optional label for debugging
     * @return Allocated address, or 0 if allocation failed
     * @throws std::invalid_argument if resource is not a memory resource
     */
    Address allocate(ResourceHandle resource, Size size, Size alignment = 64,
                     const std::string& label = "");

    /**
     * @brief Allocate memory in the first available resource of a given type
     * @param type The memory resource type
     * @param size Size in bytes to allocate
     * @param alignment Required alignment (must be power of 2, default 64)
     * @param label Optional label for debugging
     * @return Allocated address, or 0 if allocation failed
     * @throws std::invalid_argument if type is not a memory resource type
     */
    Address allocate(ResourceType type, Size size, Size alignment = 64,
                     const std::string& label = "");

    /**
     * @brief Deallocate memory
     * @param address The address to deallocate
     * @return true if deallocation succeeded
     */
    bool deallocate(Address address);

    /**
     * @brief Get information about an allocation
     * @param address The allocated address
     * @return AllocationInfo, or nullopt if address is not allocated
     */
    std::optional<AllocationInfo> get_allocation_info(Address address) const;

    /**
     * @brief Get all current allocations
     * @return Vector of AllocationInfo
     */
    std::vector<AllocationInfo> get_all_allocations() const;

    /**
     * @brief Get total allocated bytes in a resource
     * @param resource The resource handle
     * @return Total allocated bytes
     */
    Size get_allocated_bytes(ResourceHandle resource) const;

    /**
     * @brief Get available (unallocated) bytes in a resource
     * @param resource The resource handle
     * @return Available bytes
     */
    Size get_available_bytes(ResourceHandle resource) const;

    // =========================================
    // Memory Operations
    // =========================================

    /**
     * @brief Write data to memory
     * @param address Destination address
     * @param data Pointer to source data
     * @param size Size in bytes to write
     * @throws std::out_of_range if address range is invalid
     */
    void write(Address address, const void* data, Size size);

    /**
     * @brief Read data from memory
     * @param address Source address
     * @param data Pointer to destination buffer
     * @param size Size in bytes to read
     * @throws std::out_of_range if address range is invalid
     */
    void read(Address address, void* data, Size size);

    /**
     * @brief Copy data between memory locations
     * @param src_address Source address
     * @param dst_address Destination address
     * @param size Size in bytes to copy
     * @throws std::out_of_range if address ranges are invalid
     */
    void copy(Address src_address, Address dst_address, Size size);

    /**
     * @brief Fill memory with a pattern
     * @param address Start address
     * @param value Byte value to fill with
     * @param size Size in bytes to fill
     * @throws std::out_of_range if address range is invalid
     */
    void memset(Address address, uint8_t value, Size size);

    // =========================================
    // Resource Status
    // =========================================

    /**
     * @brief Check if a resource is currently busy
     * @param resource The resource handle
     * @return true if resource is busy (executing an operation)
     */
    bool is_busy(ResourceHandle resource) const;

    /**
     * @brief Check if a resource is ready (not busy)
     * @param resource The resource handle
     * @return true if resource is ready
     */
    bool is_ready(ResourceHandle resource) const {
        return !is_busy(resource);
    }

    /**
     * @brief Wait until a resource becomes ready
     *
     * This runs the simulator until the resource is no longer busy.
     * @param resource The resource handle
     */
    void wait_ready(ResourceHandle resource);

    // =========================================
    // Address Space Queries
    // =========================================

    /**
     * @brief Find which resource contains a given address
     * @param address The address to look up
     * @return ResourceHandle for the containing resource, or invalid handle if not found
     */
    ResourceHandle find_resource_for_address(Address address) const;

    /**
     * @brief Check if an address is valid (within any resource)
     * @param address The address to check
     * @return true if address is valid
     */
    bool is_valid_address(Address address) const;

    /**
     * @brief Check if an address range is valid and within a single resource
     * @param address Start address
     * @param size Size of range
     * @return true if entire range is valid
     */
    bool is_valid_range(Address address, Size size) const;

private:
    KPUSimulator& simulator_;

    // Allocation tracking per resource
    struct ResourceAllocator {
        Address next_free;          // Next free address (bump allocator)
        Size total_allocated;       // Total bytes allocated
        std::vector<AllocationInfo> allocations;  // All allocations in this resource
    };

    // Map from (type, id) to allocator state
    std::unordered_map<size_t, ResourceAllocator> allocators_;

    // Helper to get allocator key
    static size_t allocator_key(ResourceType type, size_t id) {
        return (static_cast<size_t>(type) << 32) | id;
    }

    // Get or create allocator for a resource
    ResourceAllocator& get_allocator(ResourceHandle resource);
    const ResourceAllocator* find_allocator(ResourceHandle resource) const;

    // Validate resource handle
    void validate_resource(ResourceHandle resource) const;
    void validate_memory_resource(ResourceHandle resource) const;
};

// =========================================
// Compute Operation Parameters
// =========================================

/**
 * @brief Parameters for matrix multiplication operation
 */
struct MatMulParams {
    Size M;                 // Number of rows in A and C
    Size N;                 // Number of columns in B and C
    Size K;                 // Number of columns in A / rows in B
    Address A_addr;         // Address of matrix A
    Address B_addr;         // Address of matrix B
    Address C_addr;         // Address of result matrix C
    DataType dtype;         // Data type for A, B, C
    DataType acc_dtype;     // Accumulator data type (optional, defaults based on dtype)

    MatMulParams()
        : M(0), N(0), K(0), A_addr(0), B_addr(0), C_addr(0),
          dtype(DataType::FLOAT32), acc_dtype(DataType::FLOAT32) {}

    MatMulParams(Size m, Size n, Size k, Address a, Address b, Address c,
                 DataType dt = DataType::FLOAT32)
        : M(m), N(n), K(k), A_addr(a), B_addr(b), C_addr(c),
          dtype(dt), acc_dtype(accumulator_type(dt)) {}
};

/**
 * @brief Elementwise operation types
 */
enum class ElementwiseOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    MAX = 4,
    MIN = 5,
    RELU = 6,
    GELU = 7,
    SIGMOID = 8,
    TANH = 9
};

/**
 * @brief Parameters for elementwise operation
 */
struct ElementwiseParams {
    ElementwiseOp op;       // Operation type
    Address A_addr;         // First input address
    Address B_addr;         // Second input address (ignored for unary ops)
    Address C_addr;         // Output address
    Size num_elements;      // Number of elements
    DataType dtype;         // Data type

    ElementwiseParams()
        : op(ElementwiseOp::ADD), A_addr(0), B_addr(0), C_addr(0),
          num_elements(0), dtype(DataType::FLOAT32) {}
};

} // namespace sw::kpu
