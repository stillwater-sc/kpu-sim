#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <iomanip>

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

// Software-managed scratchpad memory
class KPU_API Scratchpad {
private:
    std::vector<std::uint8_t> memory_model;  // Dynamically managed resource
    Size capacity;
    
public:
    explicit Scratchpad(Size capacity_kb = 512);
    ~Scratchpad() = default;
    
    // Core memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);
    bool is_ready() const { return true; } // Always ready - on-chip
    
    // Configuration and status
    Size get_capacity() const { return capacity; }
    void reset();
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif