#include <sw/driver/memory_manager.hpp>
#include <memory>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace sw::driver {

// ============================================================================
// MemoryManager Implementation
// ============================================================================

void* MemoryManager::allocate(size_t size) {
    if (size == 0) {
        return nullptr;  // or could return a unique non-null pointer
    }
    
    try {
		void* ptr = NULL; // std::aligned_alloc(alignof(std::max_align_t), size);  aligned_alloc is not supported in MSVC
        if (ptr) {
            update_statistics(size, true);
            // In a real implementation, you'd track this allocation
            // For now, we'll use a simple map (not thread-safe)
            static std::unordered_map<void*, size_t> allocations;
            allocations[ptr] = size;
        }
        return ptr;
    } catch (...) {
        return nullptr;
    }
}

void MemoryManager::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;  // Safe to deallocate nullptr
    }
    
    // Check if this is a valid allocation
    static std::unordered_map<void*, size_t> allocations;
    auto it = allocations.find(ptr);
    if (it == allocations.end()) {
        throw std::invalid_argument("Attempting to deallocate invalid pointer");
    }
    
    size_t size = it->second;
    allocations.erase(it);
    
    std::free(ptr);
    update_statistics(size, false);
}

bool MemoryManager::is_valid_address(void* ptr) const {
    if (ptr == nullptr) {
        return false;
    }
    
    // In a real implementation, check against tracked allocations
    static std::unordered_map<void*, size_t> allocations;
    return allocations.find(ptr) != allocations.end();
}

size_t MemoryManager::get_allocation_size(void* ptr) const {
    if (!is_valid_address(ptr)) {
        throw std::invalid_argument("Invalid pointer passed to get_allocation_size");
    }
    
    static std::unordered_map<void*, size_t> allocations;
    return allocations.at(ptr);
}

void MemoryManager::update_statistics(size_t size, bool allocating) {
    if (allocating) {
        ++allocation_count_;
        allocated_bytes_ += size;
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);
    } else {
        --allocation_count_;
        allocated_bytes_ -= size;
    }
}

// ============================================================================
// MemoryPool Implementation
// ============================================================================

MemoryPool::MemoryPool(size_t pool_size, size_t block_size)
    : pool_size_(pool_size)
    , block_size_(block_size)
    , total_blocks_(pool_size / block_size) {
    
    if (block_size == 0) {
        throw std::invalid_argument("Block size cannot be zero");
    }
    
    if (pool_size < block_size) {
        throw std::invalid_argument("Pool size must be at least block size");
    }
    
    // Allocate the pool memory
    pool_memory_ = NULL; //  std::aligned_alloc(block_size, pool_size);
    if (!pool_memory_) {
        throw std::bad_alloc();
    }
    
    initialize_free_list();
}

MemoryPool::~MemoryPool() {
    if (pool_memory_) {
        std::free(pool_memory_);
    }
}

void* MemoryPool::allocate() {
    if (free_list_head_ == nullptr) {
        return nullptr;  // Pool is full
    }
    
    // Remove from free list
    void* result = free_list_head_;
    free_list_head_ = *static_cast<void**>(free_list_head_);
    ++used_blocks_;
    
    return result;
}

void MemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return;  // Safe to deallocate nullptr
    }
    
    if (!is_from_pool(ptr)) {
        throw std::invalid_argument("Pointer is not from this pool");
    }
    
    if (!is_valid_pool_address(ptr)) {
        throw std::invalid_argument("Pointer is not properly aligned within pool");
    }
    
    // Add back to free list
    *static_cast<void**>(ptr) = free_list_head_;
    free_list_head_ = ptr;
    --used_blocks_;
}

bool MemoryPool::is_from_pool(void* ptr) const {
    if (ptr == nullptr || pool_memory_ == nullptr) {
        return false;
    }
    
    auto pool_start = reinterpret_cast<uintptr_t>(pool_memory_);
    auto pool_end = pool_start + pool_size_;
    auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
    
    return ptr_addr >= pool_start && ptr_addr < pool_end;
}

void MemoryPool::initialize_free_list() {
    free_list_head_ = pool_memory_;
    
    auto current = static_cast<char*>(pool_memory_);
    for (size_t i = 0; i < total_blocks_ - 1; ++i) {
        auto next = current + block_size_;
        *reinterpret_cast<void**>(current) = next;
        current = next;
    }
    
    // Last block points to nullptr
    *reinterpret_cast<void**>(current) = nullptr;
}

bool MemoryPool::is_valid_pool_address(void* ptr) const {
    if (!is_from_pool(ptr)) {
        return false;
    }
    
    auto pool_start = reinterpret_cast<uintptr_t>(pool_memory_);
    auto ptr_addr = reinterpret_cast<uintptr_t>(ptr);
    
    // Check if pointer is aligned to block boundary
    return (ptr_addr - pool_start) % block_size_ == 0;
}

// ============================================================================
// AlignedMemoryManager Implementation  
// ============================================================================

void* AlignedMemoryManager::allocate_aligned(size_t size, size_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument("Alignment cannot be zero");
    }
    
    if (!is_power_of_two(alignment)) {
        throw std::invalid_argument("Alignment must be a power of two");
    }
    
    if (size == 0) {
        return nullptr;
    }
    
    // Allocate extra space for alignment and store original pointer
    size_t total_size = size + alignment - 1 + sizeof(void*);
    void* original_ptr = std::malloc(total_size);
    
    if (!original_ptr) {
        return nullptr;
    }
    
    // Calculate aligned address
    auto raw_addr = reinterpret_cast<uintptr_t>(original_ptr) + sizeof(void*);
    auto aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    
    // Store original pointer just before the aligned address
    auto storage_addr = aligned_addr - sizeof(void*);
    *reinterpret_cast<void**>(storage_addr) = original_ptr;
    
    // Track the allocation
    allocations_.push_back({original_ptr, aligned_ptr, size, alignment});
    
    return aligned_ptr;
}

void AlignedMemoryManager::deallocate_aligned(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    
    // Find the allocation record
    auto it = std::find_if(allocations_.begin(), allocations_.end(),
        [ptr](const AlignedAllocation& alloc) {
            return alloc.aligned_ptr == ptr;
        });
    
    if (it == allocations_.end()) {
        throw std::invalid_argument("Pointer was not allocated by this manager");
    }
    
    // Free the original allocation
    std::free(it->original_ptr);
    
    // Remove from tracking
    allocations_.erase(it);
}

bool AlignedMemoryManager::is_valid_aligned_address(void* ptr) const {
    return std::any_of(allocations_.begin(), allocations_.end(),
        [ptr](const AlignedAllocation& alloc) {
            return alloc.aligned_ptr == ptr;
        });
}

std::pair<size_t, size_t> AlignedMemoryManager::get_allocation_info(void* ptr) const {
    auto it = std::find_if(allocations_.begin(), allocations_.end(),
        [ptr](const AlignedAllocation& alloc) {
            return alloc.aligned_ptr == ptr;
        });
    
    if (it == allocations_.end()) {
        throw std::invalid_argument("Pointer was not allocated by this manager");
    }
    
    return {it->size, it->alignment};
}

bool AlignedMemoryManager::is_power_of_two(size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void* AlignedMemoryManager::align_pointer(void* ptr, size_t alignment) noexcept {
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    auto aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

} // namespace sw::kpu