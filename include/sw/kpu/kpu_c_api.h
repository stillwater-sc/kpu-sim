/**
 * @file kpu_c_api.h
 * @brief KPU Simulator C API - Master Include Header
 *
 * This header includes all KPU C API components:
 * - kpu_c_types.h    - Common types, enums, and structures
 * - kpu_c_kernel.h   - Kernel creation and metadata
 * - kpu_c_runtime.h  - Runtime memory and execution
 * - kpu_c_executor.h - High-level tensor execution
 *
 * For new code, prefer including the specific headers you need.
 * This header also maintains backward compatibility with the original
 * simulator-only C API.
 */
#pragma once

/* Include new modular C API headers */
#include "kpu_c_types.h"
#include "kpu_c_kernel.h"
#include "kpu_c_runtime.h"
#include "kpu_c_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Legacy Simulator API
 *
 * These functions provide direct access to the KPU simulator for low-level
 * memory and compute operations. For most use cases, prefer the Runtime API.
 * ============================================================================ */

/**
 * @brief Create KPU simulator with specified memory sizes
 * @param main_memory_size Size of main memory (0 = default)
 * @param scratchpad_size Size of scratchpad (0 = default)
 * @return Simulator handle, or NULL on failure
 */
KPU_C_API KPUHandle kpu_create(uint64_t main_memory_size, uint64_t scratchpad_size);

/**
 * @brief Destroy KPU simulator
 * @param handle Simulator handle
 */
KPU_C_API void kpu_destroy(KPUHandle handle);

/* Memory operations */
KPU_C_API KPUError kpu_main_memory_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size);
KPU_C_API KPUError kpu_main_memory_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size);
KPU_C_API KPUError kpu_scratchpad_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size);
KPU_C_API KPUError kpu_scratchpad_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size);

KPU_C_API uint64_t kpu_main_memory_size(KPUHandle handle);
KPU_C_API uint64_t kpu_scratchpad_size(KPUHandle handle);

/* DMA operations */
KPU_C_API KPUError kpu_dma_transfer_sync(KPUHandle handle,
                              uint64_t src_addr, uint64_t dst_addr, uint64_t size,
                              bool src_is_main_memory, bool dst_is_main_memory);

/* Legacy compute operations (uses KPUMatrixDim from kpu_c_types.h) */
KPU_C_API KPUError kpu_matmul_f32(KPUHandle handle,
                       uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                       KPUMatrixDim dim_A, KPUMatrixDim dim_B);

KPU_C_API KPUError kpu_matmul_accumulate_f32(KPUHandle handle,
                                  uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                                  KPUMatrixDim dim_A, KPUMatrixDim dim_B);

#ifdef __cplusplus
}
#endif
