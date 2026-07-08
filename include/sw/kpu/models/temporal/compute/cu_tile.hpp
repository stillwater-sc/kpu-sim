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

// Compute Tile - compute tile in checkerboard pattern
class KPU_API CUTile {
private:
    size_t tile_id;
    Size rows, cols;

public:
    explicit CUTile(size_t tile_id, Size rows = 16, Size cols = 16)
        : tile_id(tile_id), rows(rows), cols(cols) {}
    ~CUTile() = default;

    // Status and configuration
    size_t get_tile_id() const { return tile_id; }
    Size get_rows() const { return rows; }
    Size get_cols() const { return cols; }
    bool is_ready() const { return true; } // Simplified for now
    void reset() {} // No dynamic state yet
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
