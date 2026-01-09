/**
 * @file kpu_c_executor.h
 * @brief C API for KPU GraphExecutor (high-level tensor execution API)
 *
 * This header provides a simplified execution API that handles:
 * - Automatic memory allocation for tensors
 * - Input data staging
 * - Output data retrieval
 * - Memory cleanup
 *
 * The GraphExecutor provides a higher-level abstraction than the Runtime API,
 * managing tensor bindings automatically based on kernel arguments.
 */
#pragma once

#include "kpu_c_types.h"
#include "kpu_c_runtime.h"
#include "kpu_c_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Executor Lifecycle
 * ============================================================================ */

/**
 * @brief Create a GraphExecutor
 *
 * @param runtime Runtime handle (must outlive executor)
 * @return Executor handle, or NULL on failure
 */
KPU_C_API KPUExecutorHandle kpu_executor_create(KPURuntimeHandle runtime);

/**
 * @brief Destroy an executor and free its resources
 *
 * @param executor Executor handle (safe to call with NULL)
 */
KPU_C_API void kpu_executor_destroy(KPUExecutorHandle executor);

/* ============================================================================
 * Kernel Setup
 * ============================================================================ */

/**
 * @brief Set the kernel to execute
 *
 * Allocates device memory for all kernel arguments.
 * The executor makes a copy of the kernel, so the original can be destroyed.
 *
 * @param executor Executor handle
 * @param kernel Kernel handle
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_set_kernel(KPUExecutorHandle executor, KPUKernelHandle kernel);

/**
 * @brief Create and set a matmul kernel
 *
 * Convenience method that creates a matmul kernel and sets it up.
 *
 * @param executor Executor handle
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param dtype Data type
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_create_matmul(KPUExecutorHandle executor,
                                     KPUSize M, KPUSize N, KPUSize K,
                                     KPUDataType dtype);

/**
 * @brief Create and set an MLP kernel
 *
 * Convenience method that creates an MLP kernel and sets it up.
 *
 * @param executor Executor handle
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param activation Activation function
 * @param has_bias Whether to include bias
 * @param dtype Data type
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_create_mlp(KPUExecutorHandle executor,
                                  KPUSize M, KPUSize N, KPUSize K,
                                  KPUActivationType activation,
                                  bool has_bias,
                                  KPUDataType dtype);

/**
 * @brief Check if executor has a kernel set
 *
 * @param executor Executor handle
 * @return true if kernel is set
 */
KPU_C_API bool kpu_executor_has_kernel(KPUExecutorHandle executor);

/* ============================================================================
 * Input/Output Binding
 * ============================================================================ */

/**
 * @brief Set input tensor data with shape validation
 *
 * Copies data from host to device. The shape must match the kernel argument.
 *
 * @param executor Executor handle
 * @param name Tensor name (e.g., "A", "B", "bias")
 * @param data Host data pointer
 * @param shape Array of dimensions
 * @param ndims Number of dimensions
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_set_input(KPUExecutorHandle executor,
                                 const char* name,
                                 const void* data,
                                 const KPUSize* shape,
                                 uint32_t ndims);

/**
 * @brief Set input tensor data (raw bytes, no shape check)
 *
 * @param executor Executor handle
 * @param name Tensor name
 * @param data Host data pointer
 * @param size_bytes Number of bytes to copy
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_set_input_raw(KPUExecutorHandle executor,
                                     const char* name,
                                     const void* data,
                                     KPUSize size_bytes);

/**
 * @brief Get output tensor data
 *
 * Copies data from device to host.
 *
 * @param executor Executor handle
 * @param name Tensor name (e.g., "C")
 * @param data Host destination pointer
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_get_output(KPUExecutorHandle executor,
                                  const char* name,
                                  void* data);

/**
 * @brief Get output tensor data with explicit size
 *
 * @param executor Executor handle
 * @param name Tensor name
 * @param data Host destination pointer
 * @param size_bytes Number of bytes to copy
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_get_output_sized(KPUExecutorHandle executor,
                                        const char* name,
                                        void* data,
                                        KPUSize size_bytes);

/**
 * @brief Get tensor binding info
 *
 * @param executor Executor handle
 * @param name Tensor name
 * @param info Output: tensor information
 * @return KPU_SUCCESS or KPU_ERROR_BINDING_NOT_FOUND
 */
KPU_C_API KPUError kpu_executor_get_binding_info(KPUExecutorHandle executor,
                                        const char* name,
                                        KPUTensorInfo* info);

/* ============================================================================
 * Execution
 * ============================================================================ */

/**
 * @brief Execute the kernel
 *
 * @param executor Executor handle
 * @param result Output: execution result with timing (can be NULL)
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_executor_execute(KPUExecutorHandle executor,
                               KPUExecutionResult* result);

/**
 * @brief Get the last execution time in milliseconds
 *
 * @param executor Executor handle
 * @return Execution time in ms, or 0.0 if never executed
 */
KPU_C_API double kpu_executor_get_last_time_ms(KPUExecutorHandle executor);

/**
 * @brief Get the last execution cycles
 *
 * @param executor Executor handle
 * @return Execution cycles, or 0 if never executed
 */
KPU_C_API KPUCycle kpu_executor_get_last_cycles(KPUExecutorHandle executor);

/* ============================================================================
 * Cleanup
 * ============================================================================ */

/**
 * @brief Free all allocated device memory
 *
 * After calling this, the executor can be reused with a new kernel.
 *
 * @param executor Executor handle
 */
KPU_C_API void kpu_executor_release(KPUExecutorHandle executor);

#ifdef __cplusplus
}
#endif
