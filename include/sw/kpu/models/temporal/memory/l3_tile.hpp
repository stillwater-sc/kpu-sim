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

// L3 Tile - Distributed L3 cache tile in checkerboard pattern
class KPU_API L3Tile {
private:
    std::vector<std::uint8_t> memory_model;
    Size capacity;
    size_t tile_id;

public:
    explicit L3Tile(size_t tile_id, Size capacity_kb = 128);
    ~L3Tile() = default;

    // Memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);

    // 2D block operations for matrix data
    void read_block(Address base_addr, void* data,
                   Size block_height, Size block_width, Size element_size,
                   Size stride = 0); // stride = 0 means contiguous

    void write_block(Address base_addr, const void* data,
                    Size block_height, Size block_width, Size element_size,
                    Size stride = 0);

    // Status and configuration
    Size get_capacity() const { return capacity; }
    size_t get_tile_id() const { return tile_id; }
    bool is_ready() const { return true; } // Simplified for now
    void reset();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif