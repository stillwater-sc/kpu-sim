#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <optional>

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
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/page_buffer.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::memory {
    class AddressDecoder;
}

namespace sw::kpu {

// Forward declarations

class KPU_API MemoryController {
public:

};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
