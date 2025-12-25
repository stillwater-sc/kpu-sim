// Resource Manager implementation for KPU simulator
// Provides unified access to all addressable hardware resources

#include <sw/kpu/resource_api.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/kpu/allocator.hpp>

#include <stdexcept>
#include <sstream>

namespace sw::kpu {

// =========================================
// ResourceManager Implementation
// =========================================

ResourceManager::ResourceManager(KPUSimulator& simulator)
    : simulator_(simulator) {
    // Initialize allocators for each memory resource
    // We create allocators lazily on first allocation
}

// =========================================
// Resource Discovery
// =========================================

size_t ResourceManager::get_resource_count(ResourceType type) const {
    switch (type) {
        case ResourceType::HOST_MEMORY:
            return simulator_.get_host_memory_region_count();
        case ResourceType::EXTERNAL_MEMORY:
            return simulator_.get_memory_bank_count();
        case ResourceType::L3_TILE:
            return simulator_.get_l3_tile_count();
        case ResourceType::L2_BANK:
            return simulator_.get_l2_bank_count();
        case ResourceType::L1_BUFFER:
            return simulator_.get_l1_buffer_count();
        case ResourceType::PAGE_BUFFER:
            return simulator_.get_scratchpad_count();
        case ResourceType::COMPUTE_TILE:
            return simulator_.get_compute_tile_count();
        case ResourceType::DMA_ENGINE:
            return simulator_.get_dma_engine_count();
        case ResourceType::BLOCK_MOVER:
            return simulator_.get_block_mover_count();
        case ResourceType::STREAMER:
            return simulator_.get_streamer_count();
        default:
            return 0;
    }
}

ResourceHandle ResourceManager::get_resource(ResourceType type, size_t id) const {
    size_t count = get_resource_count(type);
    if (id >= count) {
        throw std::out_of_range("Resource ID " + std::to_string(id) +
                                " out of range for " + resource_type_name(type) +
                                " (count=" + std::to_string(count) + ")");
    }

    ResourceHandle handle;
    handle.type = type;
    handle.id = id;

    // Set base address and capacity for memory resources
    switch (type) {
        case ResourceType::HOST_MEMORY:
            handle.base_address = simulator_.get_host_memory_region_base(id);
            handle.capacity = simulator_.get_host_memory_region_capacity(id);
            break;
        case ResourceType::EXTERNAL_MEMORY:
            handle.base_address = simulator_.get_external_bank_base(id);
            handle.capacity = simulator_.get_memory_bank_capacity(id);
            break;
        case ResourceType::L3_TILE:
            handle.base_address = simulator_.get_l3_tile_base(id);
            handle.capacity = simulator_.get_l3_tile_capacity(id);
            break;
        case ResourceType::L2_BANK:
            handle.base_address = simulator_.get_l2_bank_base(id);
            handle.capacity = simulator_.get_l2_bank_capacity(id);
            break;
        case ResourceType::L1_BUFFER:
            handle.base_address = simulator_.get_l1_buffer_base(id);
            handle.capacity = simulator_.get_l1_buffer_capacity(id);
            break;
        case ResourceType::PAGE_BUFFER:
            handle.base_address = simulator_.get_scratchpad_base(id);
            handle.capacity = simulator_.get_scratchpad_capacity(id);
            break;
        default:
            // Non-memory resources don't have base addresses
            handle.base_address = 0;
            handle.capacity = 0;
            break;
    }

    return handle;
}

std::vector<ResourceHandle> ResourceManager::get_all_resources(ResourceType type) const {
    std::vector<ResourceHandle> handles;
    size_t count = get_resource_count(type);
    handles.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        handles.push_back(get_resource(type, i));
    }
    return handles;
}

std::vector<ResourceHandle> ResourceManager::get_memory_resources() const {
    std::vector<ResourceHandle> handles;

    auto add_type = [&](ResourceType type) {
        auto resources = get_all_resources(type);
        handles.insert(handles.end(), resources.begin(), resources.end());
    };

    add_type(ResourceType::HOST_MEMORY);
    add_type(ResourceType::EXTERNAL_MEMORY);
    add_type(ResourceType::L3_TILE);
    add_type(ResourceType::L2_BANK);
    add_type(ResourceType::L1_BUFFER);
    add_type(ResourceType::PAGE_BUFFER);

    return handles;
}

std::vector<ResourceHandle> ResourceManager::get_compute_resources() const {
    return get_all_resources(ResourceType::COMPUTE_TILE);
}

std::vector<ResourceHandle> ResourceManager::get_data_movement_resources() const {
    std::vector<ResourceHandle> handles;

    auto add_type = [&](ResourceType type) {
        auto resources = get_all_resources(type);
        handles.insert(handles.end(), resources.begin(), resources.end());
    };

    add_type(ResourceType::DMA_ENGINE);
    add_type(ResourceType::BLOCK_MOVER);
    add_type(ResourceType::STREAMER);

    return handles;
}

// =========================================
// Memory Allocation
// =========================================

ResourceManager::ResourceAllocator& ResourceManager::get_allocator(ResourceHandle resource) {
    validate_memory_resource(resource);

    size_t key = allocator_key(resource.type, resource.id);
    auto it = allocators_.find(key);
    if (it == allocators_.end()) {
        // Create new allocator for this resource
        ResourceAllocator alloc;
        alloc.next_free = resource.base_address;
        alloc.total_allocated = 0;
        auto [inserted, success] = allocators_.emplace(key, std::move(alloc));
        return inserted->second;
    }
    return it->second;
}

const ResourceManager::ResourceAllocator* ResourceManager::find_allocator(ResourceHandle resource) const {
    size_t key = allocator_key(resource.type, resource.id);
    auto it = allocators_.find(key);
    if (it == allocators_.end()) {
        return nullptr;
    }
    return &it->second;
}

Address ResourceManager::allocate(ResourceHandle resource, Size size, Size alignment,
                                   const std::string& label) {
    validate_memory_resource(resource);

    if (size == 0) {
        return 0;
    }

    // Validate alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }

    auto& allocator = get_allocator(resource);

    // Calculate aligned address
    Address aligned = (allocator.next_free + alignment - 1) & ~(alignment - 1);

    // Check if allocation fits
    if (aligned + size > resource.base_address + resource.capacity) {
        return 0;  // Out of memory
    }

    // Record allocation
    AllocationInfo info(aligned, size, alignment, resource, label);
    allocator.allocations.push_back(info);

    // Update allocator state
    allocator.next_free = aligned + size;
    allocator.total_allocated += size;

    return aligned;
}

Address ResourceManager::allocate(ResourceType type, Size size, Size alignment,
                                   const std::string& label) {
    if (!is_memory_resource(type)) {
        throw std::invalid_argument("Cannot allocate in non-memory resource type: " +
                                     resource_type_name(type));
    }

    // Try each resource of this type until allocation succeeds
    size_t count = get_resource_count(type);
    for (size_t i = 0; i < count; ++i) {
        ResourceHandle resource = get_resource(type, i);
        Address addr = allocate(resource, size, alignment, label);
        if (addr != 0) {
            return addr;
        }
    }

    return 0;  // All resources exhausted
}

bool ResourceManager::deallocate(Address address) {
    // Find which resource contains this address
    ResourceHandle resource = find_resource_for_address(address);
    if (!resource.is_valid()) {
        return false;
    }

    const ResourceAllocator* alloc = find_allocator(resource);
    if (!alloc) {
        return false;
    }

    // Note: With a bump allocator, we can't truly deallocate individual allocations.
    // We just mark it as deallocated for tracking purposes.
    // A more sophisticated allocator would be needed for real deallocation.
    // For now, we return false to indicate we can't actually free the memory.
    return false;
}

std::optional<AllocationInfo> ResourceManager::get_allocation_info(Address address) const {
    // Search all allocators for this address
    for (const auto& [key, allocator] : allocators_) {
        for (const auto& alloc : allocator.allocations) {
            if (alloc.address == address) {
                return alloc;
            }
        }
    }
    return std::nullopt;
}

std::vector<AllocationInfo> ResourceManager::get_all_allocations() const {
    std::vector<AllocationInfo> all;
    for (const auto& [key, allocator] : allocators_) {
        all.insert(all.end(), allocator.allocations.begin(), allocator.allocations.end());
    }
    return all;
}

Size ResourceManager::get_allocated_bytes(ResourceHandle resource) const {
    const ResourceAllocator* alloc = find_allocator(resource);
    if (!alloc) {
        return 0;
    }
    return alloc->total_allocated;
}

Size ResourceManager::get_available_bytes(ResourceHandle resource) const {
    validate_memory_resource(resource);
    Size allocated = get_allocated_bytes(resource);
    return resource.capacity > allocated ? resource.capacity - allocated : 0;
}

// =========================================
// Memory Operations
// =========================================

void ResourceManager::write(Address address, const void* data, Size size) {
    ResourceHandle resource = find_resource_for_address(address);
    if (!resource.is_valid()) {
        throw std::out_of_range("Invalid address for write: " + std::to_string(address));
    }

    // Calculate offset within resource
    Address offset = address - resource.base_address;

    // Delegate to simulator
    switch (resource.type) {
        case ResourceType::HOST_MEMORY:
            simulator_.write_host_memory(resource.id, offset, data, size);
            break;
        case ResourceType::EXTERNAL_MEMORY:
            simulator_.write_memory_bank(resource.id, offset, data, size);
            break;
        case ResourceType::L3_TILE:
            simulator_.write_l3_tile(resource.id, offset, data, size);
            break;
        case ResourceType::L2_BANK:
            simulator_.write_l2_bank(resource.id, offset, data, size);
            break;
        case ResourceType::L1_BUFFER:
            simulator_.write_l1_buffer(resource.id, offset, data, size);
            break;
        case ResourceType::PAGE_BUFFER:
            simulator_.write_scratchpad(resource.id, offset, data, size);
            break;
        default:
            throw std::invalid_argument("Cannot write to non-memory resource");
    }
}

void ResourceManager::read(Address address, void* data, Size size) {
    ResourceHandle resource = find_resource_for_address(address);
    if (!resource.is_valid()) {
        throw std::out_of_range("Invalid address for read: " + std::to_string(address));
    }

    // Calculate offset within resource
    Address offset = address - resource.base_address;

    // Delegate to simulator
    switch (resource.type) {
        case ResourceType::HOST_MEMORY:
            simulator_.read_host_memory(resource.id, offset, data, size);
            break;
        case ResourceType::EXTERNAL_MEMORY:
            simulator_.read_memory_bank(resource.id, offset, data, size);
            break;
        case ResourceType::L3_TILE:
            simulator_.read_l3_tile(resource.id, offset, data, size);
            break;
        case ResourceType::L2_BANK:
            simulator_.read_l2_bank(resource.id, offset, data, size);
            break;
        case ResourceType::L1_BUFFER:
            simulator_.read_l1_buffer(resource.id, offset, data, size);
            break;
        case ResourceType::PAGE_BUFFER:
            simulator_.read_scratchpad(resource.id, offset, data, size);
            break;
        default:
            throw std::invalid_argument("Cannot read from non-memory resource");
    }
}

void ResourceManager::copy(Address src_address, Address dst_address, Size size) {
    // Read from source
    std::vector<uint8_t> buffer(size);
    read(src_address, buffer.data(), size);

    // Write to destination
    write(dst_address, buffer.data(), size);
}

void ResourceManager::memset(Address address, uint8_t value, Size size) {
    std::vector<uint8_t> buffer(size, value);
    write(address, buffer.data(), size);
}

// =========================================
// Resource Status
// =========================================

bool ResourceManager::is_busy(ResourceHandle resource) const {
    validate_resource(resource);

    switch (resource.type) {
        case ResourceType::DMA_ENGINE:
            return simulator_.is_dma_busy(resource.id);
        case ResourceType::BLOCK_MOVER:
            return simulator_.is_block_mover_busy(resource.id);
        case ResourceType::STREAMER:
            return simulator_.is_streamer_busy(resource.id);
        case ResourceType::COMPUTE_TILE:
            return simulator_.is_compute_busy(resource.id);
        case ResourceType::HOST_MEMORY:
            return !simulator_.is_host_memory_region_ready(resource.id);
        case ResourceType::EXTERNAL_MEMORY:
            return !simulator_.is_memory_bank_ready(resource.id);
        case ResourceType::L3_TILE:
            return !simulator_.is_l3_tile_ready(resource.id);
        case ResourceType::L2_BANK:
            return !simulator_.is_l2_bank_ready(resource.id);
        case ResourceType::L1_BUFFER:
            return !simulator_.is_l1_buffer_ready(resource.id);
        case ResourceType::PAGE_BUFFER:
            return !simulator_.is_scratchpad_ready(resource.id);
        default:
            return false;
    }
}

void ResourceManager::wait_ready(ResourceHandle resource) {
    while (is_busy(resource)) {
        simulator_.step();
    }
}

// =========================================
// Address Space Queries
// =========================================

ResourceHandle ResourceManager::find_resource_for_address(Address address) const {
    // Check each memory resource type
    auto check_type = [&](ResourceType type) -> std::optional<ResourceHandle> {
        size_t count = get_resource_count(type);
        for (size_t i = 0; i < count; ++i) {
            ResourceHandle handle = get_resource(type, i);
            if (address >= handle.base_address &&
                address < handle.base_address + handle.capacity) {
                return handle;
            }
        }
        return std::nullopt;
    };

    // Check in order of the memory hierarchy
    if (auto h = check_type(ResourceType::HOST_MEMORY)) return *h;
    if (auto h = check_type(ResourceType::EXTERNAL_MEMORY)) return *h;
    if (auto h = check_type(ResourceType::L3_TILE)) return *h;
    if (auto h = check_type(ResourceType::L2_BANK)) return *h;
    if (auto h = check_type(ResourceType::L1_BUFFER)) return *h;
    if (auto h = check_type(ResourceType::PAGE_BUFFER)) return *h;

    // Address not found in any resource
    return ResourceHandle();
}

bool ResourceManager::is_valid_address(Address address) const {
    return find_resource_for_address(address).is_valid();
}

bool ResourceManager::is_valid_range(Address address, Size size) const {
    if (size == 0) return true;

    ResourceHandle resource = find_resource_for_address(address);
    if (!resource.is_valid()) return false;

    // Check if entire range fits within the same resource
    Address end_address = address + size - 1;
    return end_address >= address &&  // No overflow
           end_address < resource.base_address + resource.capacity;
}

// =========================================
// Validation Helpers
// =========================================

void ResourceManager::validate_resource(ResourceHandle resource) const {
    if (!resource.is_valid()) {
        throw std::invalid_argument("Invalid resource handle");
    }
    if (resource.id >= get_resource_count(resource.type)) {
        throw std::out_of_range("Resource ID out of range: " + resource.to_string());
    }
}

void ResourceManager::validate_memory_resource(ResourceHandle resource) const {
    validate_resource(resource);
    if (!resource.is_memory()) {
        throw std::invalid_argument("Expected memory resource, got: " + resource.to_string());
    }
}

} // namespace sw::kpu
