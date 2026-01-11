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

/**
 * @brief L1 Streaming Buffer - High-bandwidth buffers for compute fabric
 *
 * L1 buffers are part of the compute fabric and feed data directly to
 * systolic arrays. They are fed by Streamers from L2 cache banks.
 *
 * Architecture:
 * - Part of compute fabric (NOT memory controller)
 * - Linear byte addressing (part of unified address space)
 * - Optimized for streaming data to/from compute engines
 * - Typically 32-64 KB per compute tile
 */
class KPU_API L1Buffer {
private:
    std::vector<std::uint8_t> memory_model;
    Size capacity;
    size_t buffer_id;

public:
    explicit L1Buffer(size_t buffer_id, Size capacity_kb = 32);
    ~L1Buffer() = default;

    // Memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);

    // Streaming operations for systolic arrays
    void read_stream(Address addr, void* data, Size size);
    void write_stream(Address addr, const void* data, Size size);

    // 2D matrix operations optimized for compute fabric
    void read_matrix_block(Address base_addr, void* data,
                          Size block_height, Size block_width, Size element_size,
                          Size stride = 0);

    void write_matrix_block(Address base_addr, const void* data,
                           Size block_height, Size block_width, Size element_size,
                           Size stride = 0);

    // Status and configuration
    Size get_capacity() const { return capacity; }
    size_t get_buffer_id() const { return buffer_id; }
    bool is_ready() const { return true; } // Simplified for now
    void reset();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
