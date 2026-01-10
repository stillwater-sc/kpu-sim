// ============================================================================
// include/sw/runtime/behavioral/memory_model.hpp
// Behavioral memory model for functional simulation
//
// This model tracks actual data values across the KPU memory hierarchy:
// - Host Memory: Graph structures, weights, inputs, outputs
// - L3 Tiles: On-chip memory receiving DMA transfers
// - L2 Banks: Working memory for compute tiles
// - L1 Buffers: Streaming buffers attached to compute tile ports
//
// See docs/plans/BEHAVIORAL_EXECUTION_MODEL.md for design documentation
// ============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw::runtime::behavioral {

// ============================================================================
// Type Definitions
// ============================================================================

using Address = uint64_t;
using Size = uint64_t;

/// Memory region type in the KPU hierarchy
enum class MemoryRegionType : uint8_t {
    HOST,       // Host CPU memory (accessible via DMA)
    L3_TILE,    // L3 on-chip memory tile
    L2_BANK,    // L2 on-chip memory bank
    L1_BUFFER   // L1 streaming buffer
};

/// Convert region type to string
constexpr const char* to_string(MemoryRegionType type) {
    switch (type) {
        case MemoryRegionType::HOST:      return "HOST";
        case MemoryRegionType::L3_TILE:   return "L3_TILE";
        case MemoryRegionType::L2_BANK:   return "L2_BANK";
        case MemoryRegionType::L1_BUFFER: return "L1_BUFFER";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Memory Region
// ============================================================================

/// A contiguous region of memory with actual data storage
class MemoryRegion {
public:
    MemoryRegion(MemoryRegionType type, uint32_t id, Size capacity_bytes)
        : type_(type)
        , id_(id)
        , capacity_(capacity_bytes)
        , data_(capacity_bytes, 0)
    {}

    // Accessors
    MemoryRegionType type() const { return type_; }
    uint32_t id() const { return id_; }
    Size capacity() const { return capacity_; }
    Size used() const { return used_; }
    Size available() const { return capacity_ - used_; }

    /// Write data to this region at the given offset
    void write(Size offset, const void* data, Size size) {
        if (offset + size > capacity_) {
            throw std::out_of_range(
                "Write exceeds region capacity: offset=" + std::to_string(offset) +
                " size=" + std::to_string(size) +
                " capacity=" + std::to_string(capacity_)
            );
        }
        std::memcpy(data_.data() + offset, data, size);
    }

    /// Read data from this region at the given offset
    void read(Size offset, void* data, Size size) const {
        if (offset + size > capacity_) {
            throw std::out_of_range(
                "Read exceeds region capacity: offset=" + std::to_string(offset) +
                " size=" + std::to_string(size) +
                " capacity=" + std::to_string(capacity_)
            );
        }
        std::memcpy(data, data_.data() + offset, size);
    }

    /// Get raw pointer to data at offset (for efficient access)
    const uint8_t* data_ptr(Size offset = 0) const {
        if (offset > capacity_) {
            throw std::out_of_range("Offset exceeds capacity");
        }
        return data_.data() + offset;
    }

    uint8_t* data_ptr(Size offset = 0) {
        if (offset > capacity_) {
            throw std::out_of_range("Offset exceeds capacity");
        }
        return data_.data() + offset;
    }

    /// Zero-fill the entire region
    void clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }

    /// Set used bytes (for allocation tracking)
    void set_used(Size bytes) {
        if (bytes > capacity_) {
            throw std::out_of_range("Used bytes exceeds capacity");
        }
        used_ = bytes;
    }

private:
    MemoryRegionType type_;
    uint32_t id_;
    Size capacity_;
    Size used_ = 0;
    std::vector<uint8_t> data_;
};

// ============================================================================
// Allocation Handle
// ============================================================================

/// Handle to an allocation within a memory region
struct Allocation {
    MemoryRegionType region_type;
    uint32_t region_id;
    Address base_address;   // Global address (encoded with region info)
    Size size_bytes;
    std::string name;       // Optional name for debugging

    bool is_valid() const { return size_bytes > 0; }
};

// ============================================================================
// Data Location Tracking
// ============================================================================

/// Tracks where a named tensor currently resides
struct DataLocation {
    std::string tensor_name;
    std::vector<Allocation> allocations;  // May exist in multiple regions

    bool exists_in(MemoryRegionType type) const {
        for (const auto& alloc : allocations) {
            if (alloc.region_type == type) return true;
        }
        return false;
    }

    std::optional<Allocation> find_in(MemoryRegionType type) const {
        for (const auto& alloc : allocations) {
            if (alloc.region_type == type) return alloc;
        }
        return std::nullopt;
    }
};

// ============================================================================
// Address Encoding
// ============================================================================

/// Encode region type and ID into address space
/// Address format: [region_type:8][region_id:24][offset:32]
class AddressEncoder {
public:
    static constexpr Address OFFSET_BITS = 32;
    static constexpr Address REGION_ID_BITS = 24;
    static constexpr Address REGION_TYPE_BITS = 8;

    static constexpr Address OFFSET_MASK = (1ULL << OFFSET_BITS) - 1;
    static constexpr Address REGION_ID_MASK = (1ULL << REGION_ID_BITS) - 1;
    static constexpr Address REGION_TYPE_MASK = (1ULL << REGION_TYPE_BITS) - 1;

    static Address encode(MemoryRegionType type, uint32_t region_id, Size offset) {
        Address addr = 0;
        addr |= (static_cast<Address>(type) & REGION_TYPE_MASK) << (OFFSET_BITS + REGION_ID_BITS);
        addr |= (static_cast<Address>(region_id) & REGION_ID_MASK) << OFFSET_BITS;
        addr |= (offset & OFFSET_MASK);
        return addr;
    }

    static MemoryRegionType decode_type(Address addr) {
        return static_cast<MemoryRegionType>(
            (addr >> (OFFSET_BITS + REGION_ID_BITS)) & REGION_TYPE_MASK
        );
    }

    static uint32_t decode_region_id(Address addr) {
        return static_cast<uint32_t>(
            (addr >> OFFSET_BITS) & REGION_ID_MASK
        );
    }

    static Size decode_offset(Address addr) {
        return addr & OFFSET_MASK;
    }
};

// ============================================================================
// Behavioral Memory Model
// ============================================================================

/// Configuration for the memory model
struct MemoryModelConfig {
    // Host memory
    Size host_capacity_bytes = 1ULL << 30;  // 1 GB default

    // L3 tiles
    uint32_t l3_tile_count = 4;
    Size l3_tile_capacity_bytes = 128 * 1024;  // 128 KB per tile

    // L2 banks
    uint32_t l2_bank_count = 8;
    Size l2_bank_capacity_bytes = 64 * 1024;  // 64 KB per bank

    // L1 buffers
    uint32_t l1_buffer_count = 16;
    Size l1_buffer_capacity_bytes = 16 * 1024;  // 16 KB per buffer
};

/// Unified behavioral memory model for the KPU
///
/// This model:
/// - Stores actual data values (not just timing)
/// - Tracks allocations across all memory regions
/// - Supports data movement between regions (DMA, BlockMover, Streamer)
/// - Provides named tensor tracking for debugging
class BehavioralMemoryModel {
public:
    explicit BehavioralMemoryModel(const MemoryModelConfig& config = {});

    // ========================================================================
    // Configuration
    // ========================================================================

    const MemoryModelConfig& config() const { return config_; }

    // ========================================================================
    // Region Access
    // ========================================================================

    /// Get a memory region by type and ID
    MemoryRegion* get_region(MemoryRegionType type, uint32_t id);
    const MemoryRegion* get_region(MemoryRegionType type, uint32_t id) const;

    /// Get region from encoded address
    MemoryRegion* get_region(Address addr);
    const MemoryRegion* get_region(Address addr) const;

    // ========================================================================
    // Allocation
    // ========================================================================

    /// Allocate memory in host region
    Allocation allocate_host(Size bytes, const std::string& name = "");

    /// Allocate memory in specific L3 tile
    Allocation allocate_l3(uint32_t tile_id, Size bytes, const std::string& name = "");

    /// Allocate memory in specific L2 bank
    Allocation allocate_l2(uint32_t bank_id, Size bytes, const std::string& name = "");

    /// Allocate memory in specific L1 buffer
    Allocation allocate_l1(uint32_t buffer_id, Size bytes, const std::string& name = "");

    /// Free an allocation
    void free(const Allocation& alloc);

    /// Get allocation by name
    std::optional<Allocation> find_allocation(const std::string& name) const;

    // ========================================================================
    // Data Operations
    // ========================================================================

    /// Write data to an address
    void write(Address addr, const void* data, Size bytes);

    /// Read data from an address
    void read(Address addr, void* data, Size bytes) const;

    /// Copy data between addresses (may be different regions)
    void copy(Address dst, Address src, Size bytes);

    /// Zero-fill a region
    void zero(Address addr, Size bytes);

    // ========================================================================
    // Typed Data Operations (convenience)
    // ========================================================================

    /// Write a vector of floats
    void write_floats(Address addr, const std::vector<float>& data) {
        write(addr, data.data(), data.size() * sizeof(float));
    }

    /// Read a vector of floats
    std::vector<float> read_floats(Address addr, Size count) const {
        std::vector<float> data(count);
        read(addr, data.data(), count * sizeof(float));
        return data;
    }

    /// Get pointer to data (for direct compute access)
    template<typename T>
    const T* get_ptr(Address addr) const {
        auto* region = get_region(addr);
        if (!region) {
            throw std::runtime_error("Invalid address: no region found");
        }
        Size offset = AddressEncoder::decode_offset(addr);
        return reinterpret_cast<const T*>(region->data_ptr(offset));
    }

    template<typename T>
    T* get_ptr(Address addr) {
        auto* region = get_region(addr);
        if (!region) {
            throw std::runtime_error("Invalid address: no region found");
        }
        Size offset = AddressEncoder::decode_offset(addr);
        return reinterpret_cast<T*>(region->data_ptr(offset));
    }

    // ========================================================================
    // Data Location Tracking
    // ========================================================================

    /// Register a named tensor at an allocation
    void register_tensor(const std::string& name, const Allocation& alloc);

    /// Get location of a named tensor
    std::optional<DataLocation> locate_tensor(const std::string& name) const;

    /// Update tensor location (after data movement)
    void update_tensor_location(const std::string& name, const Allocation& new_alloc);

    /// Remove tensor tracking
    void unregister_tensor(const std::string& name);

    // ========================================================================
    // Validation
    // ========================================================================

    /// Check if an address range is valid
    bool is_valid(Address addr, Size bytes) const;

    /// Check if data at address matches expected values
    bool verify(Address addr, const void* expected, Size bytes) const;

    // ========================================================================
    // Statistics
    // ========================================================================

    struct Stats {
        Size total_host_allocated = 0;
        Size total_l3_allocated = 0;
        Size total_l2_allocated = 0;
        Size total_l1_allocated = 0;
        uint64_t write_count = 0;
        uint64_t read_count = 0;
        uint64_t copy_count = 0;
        Size bytes_written = 0;
        Size bytes_read = 0;
        Size bytes_copied = 0;
    };

    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

    // ========================================================================
    // Debug
    // ========================================================================

    /// Print memory map for debugging
    void dump_memory_map() const;

    /// Print contents of an address range (hex dump)
    void dump_hex(Address addr, Size bytes) const;

private:
    MemoryModelConfig config_;

    // Memory regions indexed by type and ID
    std::unique_ptr<MemoryRegion> host_region_;
    std::vector<std::unique_ptr<MemoryRegion>> l3_tiles_;
    std::vector<std::unique_ptr<MemoryRegion>> l2_banks_;
    std::vector<std::unique_ptr<MemoryRegion>> l1_buffers_;

    // Allocation tracking per region
    struct RegionAllocator {
        Size next_offset = 0;
        std::vector<Allocation> allocations;
    };
    RegionAllocator host_allocator_;
    std::vector<RegionAllocator> l3_allocators_;
    std::vector<RegionAllocator> l2_allocators_;
    std::vector<RegionAllocator> l1_allocators_;

    // Named tensor tracking
    std::unordered_map<std::string, DataLocation> tensor_locations_;

    // Named allocation lookup
    std::unordered_map<std::string, Allocation> named_allocations_;

    // Statistics
    mutable Stats stats_;

    // Helper to get allocator for a region
    RegionAllocator* get_allocator(MemoryRegionType type, uint32_t id);
};

} // namespace sw::runtime::behavioral
