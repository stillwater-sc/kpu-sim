#pragma once
// Memory allocator implementations for KPU simulator
// Provides various allocation strategies for different use cases

#include <cstdint>
#include <cstddef>
#include <iterator>
#include <map>
#include <string>
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
 * The region [base, base + capacity) is split in two: the *used span*
 * [base, next_free) and the *tail* [next_free, base + capacity), which has never
 * been handed out (or has been given back). The used span is tiled by blocks, each
 * free or allocated.
 *
 * Invariants (is_consistent() checks them):
 *   I1  the blocks tile the used span exactly: no gaps, no overlaps;
 *   I2  no two adjacent blocks are both free (free neighbours are coalesced);
 *   I3  the last block, if any, is allocated (a free last block returns to the tail);
 *   I4  allocated_bytes == the sum of the allocated blocks' sizes, and
 *       peak_allocated >= allocated_bytes.
 *
 * An allocated block starts at the address allocate() returned and is exactly the
 * size requested, so deallocate(address) always finds it, and freeing it returns
 * exactly what allocating it took. Alignment padding stays a free block of its own.
 *
 * allocate() is first fit over the free blocks, in address order, then the tail.
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
     * @return Allocated address, or std::nullopt if size is 0 or no space fits.
     *         Unlike allocate(), an allocation at address 0 is distinguishable.
     * @throws std::invalid_argument if alignment is not a power of 2
     */
    std::optional<Address> try_allocate(Size size, Size alignment = 64,
                                        const std::string& label = "") {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be a power of 2");
        }
        if (size == 0) return std::nullopt;

        // First fit over the free blocks, in address order
        for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
            if (!it->second.is_free) continue;
            const Address start = it->first;
            const Size block_size = it->second.size;
            const Address aligned = align_up(start, alignment);
            if (aligned < start || aligned - start > block_size ||
                size > block_size - (aligned - start)) {
                continue;
            }
            const Address end = start + block_size;
            // Split: [start, aligned) stays free, [aligned, aligned + size) is taken,
            // [aligned + size, end) stays free. The block was free and coalesced, so
            // its neighbours are allocated, and I2 holds for the pieces.
            if (aligned > start) {
                it->second.size = aligned - start;
            } else {
                blocks_.erase(it);
            }
            if (aligned + size < end) {
                blocks_[aligned + size] = Block{end - (aligned + size), 0, std::string(), true};
            }
            blocks_[aligned] = Block{size, alignment, label, false};
            record_allocation(size);
            return aligned;
        }

        // The tail
        const Address limit = base_address_ + capacity_;
        const Address aligned = align_up(next_free_, alignment);
        if (aligned < next_free_ || aligned > limit || size > limit - aligned) {
            return std::nullopt;  // Out of memory
        }
        if (aligned > next_free_) {
            // The padding: the last block is allocated (I3), so it can't coalesce
            blocks_[next_free_] = Block{aligned - next_free_, 0, std::string(), true};
        }
        blocks_[aligned] = Block{size, alignment, label, false};
        next_free_ = aligned + size;
        record_allocation(size);
        return aligned;
    }

    /**
     * @brief Allocate memory with alignment
     * @return Allocated address, or 0 if allocation failed. A region based at 0 can
     *         hand out address 0: use try_allocate() there.
     */
    Address allocate(Size size, Size alignment = 64, const std::string& label = "") {
        return try_allocate(size, alignment, label).value_or(0);
    }

    /**
     * @brief Deallocate memory
     * @param address An address allocate() returned and not yet freed
     * @return true if it was allocated; false for any other address, or a double free
     */
    bool deallocate(Address address) {
        auto it = blocks_.find(address);
        if (it == blocks_.end() || it->second.is_free) {
            return false;
        }

        total_allocated_ -= it->second.size;
        it->second.is_free = true;
        it->second.alignment = 0;
        it->second.label.clear();

        // Coalesce with the next block (I2)
        auto next = std::next(it);
        if (next != blocks_.end() && next->second.is_free) {
            it->second.size += next->second.size;
            blocks_.erase(next);
        }
        // Coalesce with the previous block (I2)
        if (it != blocks_.begin()) {
            auto prev = std::prev(it);
            if (prev->second.is_free) {
                prev->second.size += it->second.size;
                blocks_.erase(it);
                it = prev;
            }
        }
        // A free last block returns to the tail (I3)
        if (std::next(it) == blocks_.end()) {
            next_free_ = it->first;
            blocks_.erase(it);
        }
        return true;
    }

    /**
     * @brief Get information about an allocation
     * @param address The allocated address
     * @return Optional allocation info
     */
    std::optional<Allocation> get_allocation(Address address) const {
        auto it = blocks_.find(address);
        if (it == blocks_.end() || it->second.is_free) {
            return std::nullopt;
        }
        return to_allocation(it->first, it->second);
    }

    /**
     * @brief Get all active allocations, in address order
     */
    std::vector<Allocation> get_all_allocations() const {
        std::vector<Allocation> active;
        for (const auto& [address, block] : blocks_) {
            if (!block.is_free) {
                active.push_back(to_allocation(address, block));
            }
        }
        return active;
    }

    /**
     * @brief Reset the allocator (free all allocations). The peak is kept.
     */
    void reset() {
        blocks_.clear();
        next_free_ = base_address_;
        total_allocated_ = 0;
    }

    /**
     * @brief Reset the peak to the bytes allocated now
     */
    void reset_peak() {
        peak_allocated_ = total_allocated_;
    }

    /**
     * @brief Get total allocated bytes
     */
    Size get_allocated_bytes() const {
        return total_allocated_;
    }

    /**
     * @brief Get available bytes: every byte not allocated (it may be fragmented,
     *        so one allocation of this size need not fit)
     */
    Size get_available_bytes() const {
        return capacity_ - total_allocated_;
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
        return addr >= base_address_ && addr - base_address_ < capacity_;
    }

    /**
     * @brief Check if an address is currently allocated
     */
    bool is_allocated(Address addr) const {
        auto it = blocks_.find(addr);
        return it != blocks_.end() && !it->second.is_free;
    }

    /**
     * @brief Check invariants I1-I4 (for tests and debugging)
     */
    bool is_consistent() const {
        Address expected = base_address_;
        Size allocated = 0;
        bool prev_free = false;
        for (const auto& [address, block] : blocks_) {
            if (address != expected || block.size == 0) return false;   // I1
            if (block.is_free && prev_free) return false;               // I2
            if (!block.is_free) allocated += block.size;
            prev_free = block.is_free;
            expected = address + block.size;
        }
        if (expected != next_free_) return false;                       // I1
        if (prev_free) return false;                                    // I3
        if (next_free_ - base_address_ > capacity_) return false;
        return allocated == total_allocated_ && peak_allocated_ >= total_allocated_;  // I4
    }

private:
    struct Block {
        Size size;
        Size alignment;     // 0 for a free block
        std::string label;
        bool is_free;
    };

    Address base_address_;
    Size capacity_;
    Address next_free_;
    Size total_allocated_;
    Size peak_allocated_;

    std::map<Address, Block> blocks_;  // the used span's tiling, by start address

    static Address align_up(Address addr, Size alignment) {
        return (addr + alignment - 1) & ~(alignment - 1);
    }

    static Allocation to_allocation(Address address, const Block& block) {
        return Allocation{address, block.size, block.alignment, block.label, block.is_free};
    }

    void record_allocation(Size size) {
        total_allocated_ += size;
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
