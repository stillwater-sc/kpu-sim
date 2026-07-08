/**
 * @file kpu_c_api.cpp
 * @brief Legacy KPU Simulator C API implementations
 *
 * Provides C-callable implementations for the legacy simulator API.
 * Note: The legacy low-level memory and compute operations are deprecated.
 * Use the Runtime API (kpu_c_runtime.h) for memory and compute operations.
 */

#include <sw/kpu/kpu_c_api.h>
#include <sw/kpu/kpu_simulator.hpp>
#include <exception>
#include <cstring>

namespace {
    inline KPUError handle_exception_legacy() {
        try {
            throw;
        } catch (const std::out_of_range&) {
            return KPU_ERROR_OUT_OF_BOUNDS;
        } catch (const std::invalid_argument&) {
            return KPU_ERROR_INVALID_DIMENSIONS;
        } catch (...) {
            return KPU_ERROR_UNKNOWN;
        }
    }
}

extern "C" {

KPUHandle kpu_create(uint64_t main_memory_size, uint64_t scratchpad_size) {
    try {
        // Create simulator with default config
        // The main_memory_size and scratchpad_size parameters are kept for
        // backward compatibility but are now handled via the Config struct
        sw::kpu::KPUSimulator::Config config;

        // Set up minimal configuration for basic usage
        // Use reasonable defaults that match the old API behavior
        config.memory_bank_count = 1;
        config.memory_bank_capacity_mb = (main_memory_size > 0) ?
            (main_memory_size / (1024 * 1024)) : 1024;  // Default 1GB
        config.memory_bandwidth_gbps = 100;

        config.l3_layer.tile_groups = { {"l3",
            {(scratchpad_size > 0) ? (scratchpad_size / 1024) : 1024},  // Default 1MB
            1} };

        config.compute_tile_count = 1;
        config.dma_engine_count = 1;

        auto* simulator = new sw::kpu::KPUSimulator(config);
        return reinterpret_cast<KPUHandle>(simulator);
    } catch (...) {
        return nullptr;
    }
}

void kpu_destroy(KPUHandle handle) {
    if (handle) {
        delete reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
    }
}

KPUError kpu_main_memory_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;

    try {
        auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
        // Use memory bank 0 as "main memory" for legacy compatibility
        simulator->read_memory_bank(0, addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception_legacy();
    }
}

KPUError kpu_main_memory_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;

    try {
        auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
        simulator->write_memory_bank(0, addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception_legacy();
    }
}

KPUError kpu_scratchpad_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;

    try {
        auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
        // Use L3 tile 0 as "scratchpad" for legacy compatibility
        simulator->read_l3_tile(0, addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception_legacy();
    }
}

KPUError kpu_scratchpad_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;

    try {
        auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
        simulator->write_l3_tile(0, addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception_legacy();
    }
}

uint64_t kpu_main_memory_size(KPUHandle handle) {
    if (!handle) return 0;
    auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
    return simulator->get_memory_bank_capacity(0);
}

uint64_t kpu_scratchpad_size(KPUHandle handle) {
    if (!handle) return 0;
    auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
    return simulator->get_l3_tile_capacity(0);
}

KPUError kpu_dma_transfer_sync(KPUHandle handle,
                              uint64_t src_addr, uint64_t dst_addr, uint64_t size,
                              bool /*src_is_main_memory*/, bool /*dst_is_main_memory*/) {
    if (!handle) return KPU_ERROR_NULL_POINTER;

    try {
        auto* simulator = reinterpret_cast<sw::kpu::KPUSimulator*>(handle);
        // Use the new address-based DMA API
        // The addresses are global addresses in the unified address space
        simulator->start_dma_transfer(0, src_addr, dst_addr, size, nullptr);

        // Wait for completion (synchronous)
        while (simulator->is_dma_busy(0)) {
            simulator->step();
        }
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception_legacy();
    }
}

KPUError kpu_matmul_f32(KPUHandle handle,
                       uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                       KPUMatrixDim dim_A, KPUMatrixDim dim_B) {
    // Legacy compute operations are not supported in the new API
    // Users should use the Runtime API with kernels instead
    (void)handle; (void)addr_A; (void)addr_B; (void)addr_C;
    (void)dim_A; (void)dim_B;
    return KPU_ERROR_INVALID_ARGUMENT;
}

KPUError kpu_matmul_accumulate_f32(KPUHandle handle,
                                  uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                                  KPUMatrixDim dim_A, KPUMatrixDim dim_B) {
    // Legacy compute operations are not supported in the new API
    // Users should use the Runtime API with kernels instead
    (void)handle; (void)addr_A; (void)addr_B; (void)addr_C;
    (void)dim_A; (void)dim_B;
    return KPU_ERROR_INVALID_ARGUMENT;
}

} // extern "C"
