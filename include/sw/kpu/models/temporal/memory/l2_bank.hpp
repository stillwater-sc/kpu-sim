#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

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

#include <sw/concepts.hpp>

namespace sw::kpu {

// L2 Bank - L2 cache bank for intermediate data storage
class KPU_API L2Bank {
private:
    std::vector<std::uint8_t> memory_model;
    Size capacity;
    size_t bank_id;

public:
    explicit L2Bank(size_t bank_id, Size capacity_kb = 64);
    ~L2Bank() = default;

    // Memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);

    // Cache line operations for streaming to L1
    void read_cache_line(Address addr, void* data, Size cache_line_size = 64);
    void write_cache_line(Address addr, const void* data, Size cache_line_size = 64);

    // 2D block operations
    void read_block(Address base_addr, void* data,
                   Size block_height, Size block_width, Size element_size,
                   Size stride = 0);

    void write_block(Address base_addr, const void* data,
                    Size block_height, Size block_width, Size element_size,
                    Size stride = 0);

    // Status and configuration
    Size get_capacity() const { return capacity; }
    size_t get_bank_id() const { return bank_id; }
    bool is_ready() const { return true; } // Simplified for now
    void reset();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif