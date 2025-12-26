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
#include <sw/trace/trace_logger.hpp>

namespace sw::memory {
    class AddressDecoder;
}

namespace sw::kpu {

// Forward declarations
class L3Tile;

// DMA Engine for data movement between memory hierarchies
class KPU_API DMAEngine {
public:
    enum class MemoryType {
        HOST_MEMORY,      // Host DDR
        KPU_MEMORY,       // KPU main memory banks (GDDR6/HBM)
        L3_TILE           // L3 cache tiles (on-chip cache hierarchy)
        // Note: L2 banks accessed via BlockMover, L1 buffers via Streamers
    };

    struct Transfer {
        MemoryType src_type;
        size_t src_id;
        Address src_addr;
        MemoryType dst_type;
        size_t dst_id;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;

        // Cycle-based timing
        trace::CycleCount start_cycle;
        trace::CycleCount end_cycle;
        uint64_t transaction_id;  // For trace correlation
    };

private:
    std::vector<Transfer> transfer_queue;  // Dynamically managed resource
    bool is_active;
    size_t engine_id;  // For debugging/identification

    // Multi-cycle timing state (like BlockMover)
    trace::CycleCount cycles_remaining;  // Cycles left for current transfer
    std::vector<uint8_t> transfer_buffer;  // Buffer for current transfer data

    // Tracing support
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;
    double clock_freq_ghz_;  // Clock frequency for bandwidth calculations
    double bandwidth_gb_s_;  // Theoretical bandwidth in GB/s

    // Current cycle (for timing)
    trace::CycleCount current_cycle_;

    // Address decoder for address-based API (optional)
    sw::memory::AddressDecoder* address_decoder_;

public:
    explicit DMAEngine(size_t engine_id = 0, double clock_freq_ghz = 1.0, double bandwidth_gb_s = 100.0);
    ~DMAEngine() = default;

    // Enable/disable tracing
    void enable_tracing(bool enabled = true, trace::TraceLogger* logger = nullptr) {
        tracing_enabled_ = enabled;
        if (logger) trace_logger_ = logger;
    }

    // Set current cycle (called by system clock/orchestrator)
    void set_current_cycle(trace::CycleCount cycle) {
        current_cycle_ = cycle;
    }

    trace::CycleCount get_current_cycle() const {
        return current_cycle_;
    }

    // Set address decoder for address-based API
    void set_address_decoder(sw::memory::AddressDecoder* decoder) {
        address_decoder_ = decoder;
    }

    sw::memory::AddressDecoder* get_address_decoder() const {
        return address_decoder_;
    }

    // ===========================================
    // Address-Based API (Recommended - Industry Standard)
    // ===========================================

    /**
     * @brief Enqueue a DMA transfer using pure addresses (recommended)
     *
     * This is the industry-standard DMA API that uses pure physical addresses.
     * The address decoder automatically routes transfers based on address ranges,
     * following the design of Intel IOAT, ARM PL330, AMD SDMA, and other
     * commercial DMA controllers.
     *
     * Benefits:
     * - Compatible with virtual memory systems
     * - Hardware topology independent
     * - Portable across different KPU configurations
     * - Enables dynamic memory management
     *
     * @param src_addr Source physical address
     * @param dst_addr Destination physical address
     * @param size Transfer size in bytes
     * @param callback Optional completion callback
     *
     * @throws std::runtime_error if address decoder is not configured
     * @throws std::out_of_range if addresses are not mapped
     *
     * Example:
     * @code
     * // Configure decoder once during initialization
     * decoder.add_region(0x0000'0000, 512_MB, KPU_MEMORY, 0);
     * decoder.add_region(0xFFFF'0000, 64_KB, SCRATCHPAD, 0);
     * dma.set_address_decoder(&decoder);
     *
     * // Use addresses directly - routing is automatic
     * dma.enqueue_transfer(0x0000'1000, 0xFFFF'0000, 4096);
     * @endcode
     */
    void enqueue_transfer(Address src_addr, Address dst_addr, Size size,
                         std::function<void()> callback = nullptr);

    // Process transfers with memory hierarchy access
    bool process_transfers(std::vector<ExternalMemory>& host_memory_regions,
                          std::vector<ExternalMemory>& memory_banks,
                          std::vector<L3Tile>& l3_tiles);

    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();

    // Status and identification
    size_t get_engine_id() const { return engine_id; }
    size_t get_queue_size() const { return transfer_queue.size(); }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
