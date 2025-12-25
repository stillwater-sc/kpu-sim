#pragma once

#include <vector>
#include <optional>
#include <string>
#include <stdexcept>
#include <cstdint>

#include <sw/concepts.hpp>

// Import address types from sw::kpu namespace
using sw::kpu::Address;
using sw::kpu::Size;

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::memory {

/**
 * @brief Memory types in the KPU hierarchy
 *
 * Used internally by AddressDecoder for routing DMA operations.
 * Applications should use addresses directly, not these types.
 *
 * Architecture notes:
 * - L3 → L2 → L1 → Compute: Standard cache hierarchy for compute datapath
 * - PAGE_BUFFER: Memory controller page buffers for internal/external memory aggregation/disaggregation
 *   (separate from cache hierarchy, used for row/column batching)
 */
enum class MemoryType {
    HOST_MEMORY,      // Host DDR (CPU-side)
    EXTERNAL,         // KPU external memory banks (GDDR6/HBM)
    L3_TILE,          // L3 cache tiles
    L2_BANK,          // L2 cache banks
    L1,               // L1 streaming buffers (compute fabric, fed by Streamers)
    PAGE_BUFFER       // Page buffers (memory controller, for internal/external memory efficiency)
};

/**
 * @brief Address decoder for unified address space
 *
 * Maps physical addresses to memory hierarchy components (type, ID, offset).
 * This follows industry-standard DMA design where:
 * - DMA commands use pure addresses
 * - Memory controller/interconnect handles routing
 * - Applications are decoupled from physical memory topology
 *
 * Example usage:
 * @code
 * AddressDecoder decoder;
 *
 * // Configure memory map
 * decoder.add_region(0x0000'0000, 512_MB, MemoryType::EXTERNAL, 0);
 * decoder.add_region(0x2000'0000, 512_MB, MemoryType::EXTERNAL, 1);
 * decoder.add_region(0x8000'0000, 128_KB, MemoryType::L3_TILE, 0);
 * decoder.add_region(0xFFFF'0000, 64_KB,  MemoryType::PAGE_BUFFER, 0);
 *
 * // Decode address to routing info
 * auto route = decoder.decode(0x0000'1000);
 * // route = {MemoryType::EXTERNAL, id=0, offset=0x1000}
 * @endcode
 */
class KPU_API AddressDecoder {
public:
    /**
     * @brief Routing information decoded from an address
     */
    struct RoutingInfo {
        MemoryType type;    ///< Memory type (EXTERNAL, L3, etc.)
        size_t id;          ///< Component ID (bank/tile/scratchpad index)
        Address offset;     ///< Offset within the component
        Size region_size;   ///< Size of the memory region

        RoutingInfo() = default;
        RoutingInfo(MemoryType t, size_t i, Address o, Size s)
            : type(t), id(i), offset(o), region_size(s) {}
    };

    /**
     * @brief Memory region configuration
     */
    struct Region {
        Address base;       ///< Base address of region
        Size size;          ///< Size of region in bytes
        MemoryType type;    ///< Memory type
        size_t id;          ///< Component ID
        std::string name;   ///< Optional name for debugging

        Region() = default;
        Region(Address b, Size s, MemoryType t, size_t i, const std::string& n = "")
            : base(b), size(s), type(t), id(i), name(n) {}

        Address end() const { return base + size; }
        bool contains(Address addr) const { return addr >= base && addr < end(); }
    };

private:
    std::vector<Region> regions_;

public:
    AddressDecoder() = default;
    ~AddressDecoder() = default;

    /**
     * @brief Add a memory region to the address map
     *
     * @param base Base address of the region
     * @param size Size of the region in bytes
     * @param type Memory type (EXTERNAL, L3_TILE, etc.)
     * @param id Component ID (bank/tile index)
     * @param name Optional name for debugging
     *
     * @throws std::invalid_argument if region overlaps with existing region
     */
    void add_region(Address base, Size size, MemoryType type, size_t id,
                   const std::string& name = "");

    /**
     * @brief Decode an address to routing information
     *
     * @param addr Physical address to decode
     * @return RoutingInfo containing type, ID, and offset
     * @throws std::out_of_range if address is not mapped
     */
    RoutingInfo decode(Address addr) const;

    /**
     * @brief Check if an address is valid (mapped to a region)
     *
     * @param addr Address to check
     * @return true if address is in a mapped region
     */
    bool is_valid(Address addr) const;

    /**
     * @brief Check if an address range is valid and doesn't cross region boundaries
     *
     * @param addr Starting address
     * @param size Transfer size
     * @return true if entire range is within a single region
     */
    bool is_valid_range(Address addr, Size size) const;

    /**
     * @brief Get all configured regions
     */
    const std::vector<Region>& get_regions() const { return regions_; }

    /**
     * @brief Clear all regions
     */
    void clear() { regions_.clear(); }

    /**
     * @brief Get region containing an address
     *
     * @param addr Address to find
     * @return Optional Region if found
     */
    std::optional<Region> find_region(Address addr) const;

    /**
     * @brief Get total mapped address space
     */
    Size get_total_mapped_size() const;

    /**
     * @brief Print memory map for debugging
     */
    std::string to_string() const;
};

} // namespace sw::memory

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
