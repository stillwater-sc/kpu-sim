#pragma once
// Memory allocator implementations for KPU simulator
// Provides various allocation strategies for different use cases

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <algorithm>

#include <sw/concepts.hpp>

namespace sw::kpu {

/**
 * @brief Simple bump allocator for fast sequential allocation
 *
 * This allocator provides O(1) allocation by simply bumping a pointer.
 * Deallocation is not supported individually - all allocations are freed
 * together via reset(). This is suitable for:
 * - Per-kernel temporary allocations
 * - Scratch space that is freed after each operation
 * - Fast allocation when fragmentation is not a concern
 */
class BumpAllocator {
public:
    /**
     * @brief Construct a bump allocator for a memory region
     * @param base_address Base address of the region
     * @param capacity Total capacity in bytes
     */
    BumpAllocator(Address base_address, Size capacity)
        : base_address_(base_address)
        , capacity_(capacity)
        , next_free_(base_address)
        , peak_usage_(0)
        , allocation_count_(0) {}

    /**
     * @brief Allocate memory with alignment
     * @param size Size in bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @return Allocated address, or 0 if allocation failed
     */
    Address allocate(Size size, Size alignment = 64) {
        if (size == 0) return 0;

        // Validate alignment is power of 2
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be a power of 2");
        }

        // Calculate aligned address
        Address aligned = align_up(next_free_, alignment);

        // Check if allocation fits
        if (aligned + size > base_address_ + capacity_) {
            return 0;  // Out of memory
        }

        // Bump the pointer
        next_free_ = aligned + size;
        allocation_count_++;

        // Track peak usage
        Size current_usage = next_free_ - base_address_;
        if (current_usage > peak_usage_) {
            peak_usage_ = current_usage;
        }

        return aligned;
    }

    /**
     * @brief Reset the allocator (free all allocations)
     *
     * This is the only way to reclaim memory from a bump allocator.
     */
    void reset() {
        next_free_ = base_address_;
        allocation_count_ = 0;
    }

    /**
     * @brief Get current allocation offset from base
     */
    Size get_used_bytes() const {
        return next_free_ - base_address_;
    }

    /**
     * @brief Get remaining available bytes
     */
    Size get_available_bytes() const {
        return capacity_ - get_used_bytes();
    }

    /**
     * @brief Get peak memory usage
     */
    Size get_peak_usage() const {
        return peak_usage_;
    }

    /**
     * @brief Get number of allocations
     */
    size_t get_allocation_count() const {
        return allocation_count_;
    }

    /**
     * @brief Get base address
     */
    Address get_base_address() const {
        return base_address_;
    }

    /**
     * @brief Get capacity
     */
    Size get_capacity() const {
        return capacity_;
    }

    /**
     * @brief Check if an address is within this allocator's range
     */
    bool contains(Address addr) const {
        return addr >= base_address_ && addr < base_address_ + capacity_;
    }

private:
    Address base_address_;
    Size capacity_;
    Address next_free_;
    Size peak_usage_;
    size_t allocation_count_;

    static Address align_up(Address addr, Size alignment) {
        return (addr + alignment - 1) & ~(alignment - 1);
    }
};

/**
 * @brief Tracking allocator that supports individual deallocation
 *
 * This allocator tracks all allocations and supports freeing individual
 * allocations. It uses a bump allocator internally but maintains a free
 * list to reuse deallocated memory.
 */
class TrackingAllocator {
public:
    struct Allocation {
        Address address;
        Size size;
        Size alignment;
        std::string label;
        bool is_free;
    };

    /**
     * @brief Construct a tracking allocator for a memory region
     * @param base_address Base address of the region
     * @param capacity Total capacity in bytes
     */
    TrackingAllocator(Address base_address, Size capacity)
        : base_address_(base_address)
        , capacity_(capacity)
        , next_free_(base_address)
        , total_allocated_(0)
        , peak_allocated_(0) {}

    /**
     * @brief Allocate memory with alignment
     * @param size Size in bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @param label Optional label for debugging
     * @return Allocated address, or 0 if allocation failed
     */
    Address allocate(Size size, Size alignment = 64, const std::string& label = "") {
        if (size == 0) return 0;

        // Validate alignment
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be a power of 2");
        }

        // First, try to find a suitable free block
        for (auto& alloc : allocations_) {
            if (alloc.is_free && alloc.size >= size) {
                Address aligned = align_up(alloc.address, alignment);
                if (aligned + size <= alloc.address + alloc.size) {
                    // Reuse this block
                    alloc.is_free = false;
                    alloc.alignment = alignment;
                    alloc.label = label;
                    total_allocated_ += size;
                    update_peak();
                    return aligned;
                }
            }
        }

        // No suitable free block, allocate from the end
        Address aligned = align_up(next_free_, alignment);
        if (aligned + size > base_address_ + capacity_) {
            return 0;  // Out of memory
        }

        // Record allocation
        Allocation alloc;
        alloc.address = aligned;
        alloc.size = size;
        alloc.alignment = alignment;
        alloc.label = label;
        alloc.is_free = false;
        allocations_.push_back(alloc);

        // Update state
        next_free_ = aligned + size;
        total_allocated_ += size;
        update_peak();

        // Add to address map for fast lookup
        address_to_index_[aligned] = allocations_.size() - 1;

        return aligned;
    }

    /**
     * @brief Deallocate memory
     * @param address The address to deallocate
     * @return true if deallocation succeeded
     */
    bool deallocate(Address address) {
        auto it = address_to_index_.find(address);
        if (it == address_to_index_.end()) {
            return false;
        }

        size_t index = it->second;
        if (index >= allocations_.size() || allocations_[index].is_free) {
            return false;
        }

        allocations_[index].is_free = true;
        total_allocated_ -= allocations_[index].size;
        return true;
    }

    /**
     * @brief Get information about an allocation
     * @param address The allocated address
     * @return Optional allocation info
     */
    std::optional<Allocation> get_allocation(Address address) const {
        auto it = address_to_index_.find(address);
        if (it == address_to_index_.end()) {
            return std::nullopt;
        }
        size_t index = it->second;
        if (index >= allocations_.size() || allocations_[index].is_free) {
            return std::nullopt;
        }
        return allocations_[index];
    }

    /**
     * @brief Get all active allocations
     */
    std::vector<Allocation> get_all_allocations() const {
        std::vector<Allocation> active;
        for (const auto& alloc : allocations_) {
            if (!alloc.is_free) {
                active.push_back(alloc);
            }
        }
        return active;
    }

    /**
     * @brief Reset the allocator (free all allocations)
     */
    void reset() {
        allocations_.clear();
        address_to_index_.clear();
        next_free_ = base_address_;
        total_allocated_ = 0;
    }

    /**
     * @brief Get total allocated bytes
     */
    Size get_allocated_bytes() const {
        return total_allocated_;
    }

    /**
     * @brief Get available bytes (may be fragmented)
     */
    Size get_available_bytes() const {
        return capacity_ - (next_free_ - base_address_);
    }

    /**
     * @brief Get peak allocated bytes
     */
    Size get_peak_allocated() const {
        return peak_allocated_;
    }

    /**
     * @brief Get base address
     */
    Address get_base_address() const {
        return base_address_;
    }

    /**
     * @brief Get capacity
     */
    Size get_capacity() const {
        return capacity_;
    }

    /**
     * @brief Check if an address is within this allocator's range
     */
    bool contains(Address addr) const {
        return addr >= base_address_ && addr < base_address_ + capacity_;
    }

    /**
     * @brief Check if an address is currently allocated
     */
    bool is_allocated(Address addr) const {
        auto it = address_to_index_.find(addr);
        if (it == address_to_index_.end()) {
            return false;
        }
        return !allocations_[it->second].is_free;
    }

private:
    Address base_address_;
    Size capacity_;
    Address next_free_;
    Size total_allocated_;
    Size peak_allocated_;

    std::vector<Allocation> allocations_;
    std::unordered_map<Address, size_t> address_to_index_;

    static Address align_up(Address addr, Size alignment) {
        return (addr + alignment - 1) & ~(alignment - 1);
    }

    void update_peak() {
        if (total_allocated_ > peak_allocated_) {
            peak_allocated_ = total_allocated_;
        }
    }
};

/**
 * @brief Pool allocator for fixed-size blocks
 *
 * Efficient for allocating many objects of the same size.
 * Uses a free list for O(1) allocation and deallocation.
 */
class PoolAllocator {
public:
    /**
     * @brief Construct a pool allocator
     * @param base_address Base address of the pool
     * @param block_size Size of each block
     * @param num_blocks Number of blocks in the pool
     */
    PoolAllocator(Address base_address, Size block_size, size_t num_blocks)
        : base_address_(base_address)
        , block_size_(block_size)
        , num_blocks_(num_blocks)
        , allocated_count_(0) {
        // Initialize free list
        for (size_t i = 0; i < num_blocks; ++i) {
            free_list_.push_back(base_address + i * block_size);
        }
    }

    /**
     * @brief Allocate a block
     * @return Block address, or 0 if pool is exhausted
     */
    Address allocate() {
        if (free_list_.empty()) {
            return 0;
        }
        Address addr = free_list_.back();
        free_list_.pop_back();
        allocated_count_++;
        return addr;
    }

    /**
     * @brief Deallocate a block
     * @param address The block address
     * @return true if deallocation succeeded
     */
    bool deallocate(Address address) {
        // Validate address is within pool and aligned
        if (address < base_address_ ||
            address >= base_address_ + num_blocks_ * block_size_) {
            return false;
        }
        if ((address - base_address_) % block_size_ != 0) {
            return false;
        }

        free_list_.push_back(address);
        allocated_count_--;
        return true;
    }

    /**
     * @brief Get number of free blocks
     */
    size_t get_free_count() const {
        return free_list_.size();
    }

    /**
     * @brief Get number of allocated blocks
     */
    size_t get_allocated_count() const {
        return allocated_count_;
    }

    /**
     * @brief Get block size
     */
    Size get_block_size() const {
        return block_size_;
    }

    /**
     * @brief Get total capacity in bytes
     */
    Size get_capacity() const {
        return num_blocks_ * block_size_;
    }

    /**
     * @brief Reset the pool (free all blocks)
     */
    void reset() {
        free_list_.clear();
        for (size_t i = 0; i < num_blocks_; ++i) {
            free_list_.push_back(base_address_ + i * block_size_);
        }
        allocated_count_ = 0;
    }

private:
    Address base_address_;
    Size block_size_;
    size_t num_blocks_;
    size_t allocated_count_;
    std::vector<Address> free_list_;
};

} // namespace sw::kpu
