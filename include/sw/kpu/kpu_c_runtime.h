/**
 * @file kpu_c_runtime.h
 * @brief C API for KPU runtime (CUDA-like memory and execution API)
 *
 * This header provides functions to:
 * - Create and destroy runtime instances
 * - Allocate and free device memory
 * - Copy data between host and device
 * - Launch kernels
 * - Manage streams and events for async execution
 */
#pragma once

#include "kpu_c_types.h"
#include "kpu_c_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Runtime Lifecycle
 * ============================================================================ */

/**
 * @brief Create a KPU runtime attached to a simulator
 *
 * @param simulator Handle to an existing simulator (must outlive runtime)
 * @param config Optional runtime configuration (NULL for defaults)
 * @return Runtime handle, or NULL on failure
 */
KPU_C_API KPURuntimeHandle kpu_runtime_create(KPUHandle simulator,
                                     const KPURuntimeConfig* config);

/**
 * @brief Destroy a runtime and free its resources
 *
 * @param runtime Runtime handle (safe to call with NULL)
 */
KPU_C_API void kpu_runtime_destroy(KPURuntimeHandle runtime);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Allocate device memory
 *
 * @param runtime Runtime handle
 * @param size Size in bytes
 * @param alignment Alignment requirement (use 0 for default 64 bytes)
 * @return Device address, or 0 if allocation failed
 */
KPU_C_API KPUAddress kpu_runtime_malloc(KPURuntimeHandle runtime,
                               KPUSize size,
                               KPUSize alignment);

/**
 * @brief Free device memory
 *
 * @param runtime Runtime handle
 * @param ptr Device address to free (safe to call with 0)
 */
KPU_C_API void kpu_runtime_free(KPURuntimeHandle runtime, KPUAddress ptr);

/**
 * @brief Copy data from host to device
 *
 * @param runtime Runtime handle
 * @param dst Device destination address
 * @param src Host source pointer
 * @param size Number of bytes to copy
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_memcpy_h2d(KPURuntimeHandle runtime,
                                 KPUAddress dst,
                                 const void* src,
                                 KPUSize size);

/**
 * @brief Copy data from device to host
 *
 * @param runtime Runtime handle
 * @param dst Host destination pointer
 * @param src Device source address
 * @param size Number of bytes to copy
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_memcpy_d2h(KPURuntimeHandle runtime,
                                 void* dst,
                                 KPUAddress src,
                                 KPUSize size);

/**
 * @brief Copy data within device memory
 *
 * @param runtime Runtime handle
 * @param dst Device destination address
 * @param src Device source address
 * @param size Number of bytes to copy
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_memcpy_d2d(KPURuntimeHandle runtime,
                                 KPUAddress dst,
                                 KPUAddress src,
                                 KPUSize size);

/**
 * @brief Set device memory to a value
 *
 * @param runtime Runtime handle
 * @param ptr Device address
 * @param value Byte value to set
 * @param size Number of bytes
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_memset(KPURuntimeHandle runtime,
                             KPUAddress ptr,
                             uint8_t value,
                             KPUSize size);

/* ============================================================================
 * Kernel Execution
 * ============================================================================ */

/**
 * @brief Launch a kernel synchronously
 *
 * Arguments must be provided in the order specified by kernel:
 * - For matmul: {A, B, C}
 * - For MLP with bias: {A, B, bias, C}
 *
 * @param runtime Runtime handle
 * @param kernel Kernel to execute
 * @param args Array of device addresses for kernel arguments
 * @param num_args Number of arguments
 * @param result Output: launch result with cycle count (can be NULL)
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_launch(KPURuntimeHandle runtime,
                             KPUKernelHandle kernel,
                             const KPUAddress* args,
                             uint32_t num_args,
                             KPULaunchResult* result);

/**
 * @brief Wait for all operations to complete
 *
 * Blocks until all kernels and memory operations finish.
 *
 * @param runtime Runtime handle
 */
KPU_C_API void kpu_runtime_synchronize(KPURuntimeHandle runtime);

/* ============================================================================
 * Stream Management
 * ============================================================================ */

/**
 * @brief Create a new execution stream
 *
 * @param runtime Runtime handle
 * @return Stream handle, or NULL on failure
 */
KPU_C_API KPUStreamHandle kpu_runtime_create_stream(KPURuntimeHandle runtime);

/**
 * @brief Destroy an execution stream
 *
 * @param runtime Runtime handle
 * @param stream Stream to destroy (safe to call with NULL)
 */
KPU_C_API void kpu_runtime_destroy_stream(KPURuntimeHandle runtime, KPUStreamHandle stream);

/**
 * @brief Wait for all operations on a stream to complete
 *
 * @param runtime Runtime handle
 * @param stream Stream to synchronize
 */
KPU_C_API void kpu_runtime_stream_synchronize(KPURuntimeHandle runtime, KPUStreamHandle stream);

/**
 * @brief Get the default stream (stream 0)
 *
 * @param runtime Runtime handle
 * @return Default stream handle
 */
KPU_C_API KPUStreamHandle kpu_runtime_default_stream(KPURuntimeHandle runtime);

/**
 * @brief Launch a kernel asynchronously on a stream
 *
 * @param runtime Runtime handle
 * @param kernel Kernel to execute
 * @param args Array of device addresses
 * @param num_args Number of arguments
 * @param stream Stream for async execution
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_launch_async(KPURuntimeHandle runtime,
                                   KPUKernelHandle kernel,
                                   const KPUAddress* args,
                                   uint32_t num_args,
                                   KPUStreamHandle stream);

/* ============================================================================
 * Event Management
 * ============================================================================ */

/**
 * @brief Create a new event
 *
 * @param runtime Runtime handle
 * @return Event handle, or NULL on failure
 */
KPU_C_API KPUEventHandle kpu_runtime_create_event(KPURuntimeHandle runtime);

/**
 * @brief Destroy an event
 *
 * @param runtime Runtime handle
 * @param event Event to destroy (safe to call with NULL)
 */
KPU_C_API void kpu_runtime_destroy_event(KPURuntimeHandle runtime, KPUEventHandle event);

/**
 * @brief Record an event on a stream
 *
 * The event captures the completion of all prior operations on the stream.
 *
 * @param runtime Runtime handle
 * @param event Event to record
 * @param stream Stream where event is recorded
 * @return KPU_SUCCESS or error code
 */
KPU_C_API KPUError kpu_runtime_record_event(KPURuntimeHandle runtime,
                                   KPUEventHandle event,
                                   KPUStreamHandle stream);

/**
 * @brief Wait for an event to complete
 *
 * @param runtime Runtime handle
 * @param event Event to wait for
 */
KPU_C_API void kpu_runtime_wait_event(KPURuntimeHandle runtime, KPUEventHandle event);

/**
 * @brief Calculate elapsed time between two events
 *
 * @param runtime Runtime handle
 * @param start Start event
 * @param end End event
 * @return Elapsed time in milliseconds
 */
KPU_C_API float kpu_runtime_elapsed_time(KPURuntimeHandle runtime,
                                KPUEventHandle start,
                                KPUEventHandle end);

/* ============================================================================
 * Device Information
 * ============================================================================ */

/**
 * @brief Get total device memory
 *
 * @param runtime Runtime handle
 * @return Total memory in bytes
 */
KPU_C_API KPUSize kpu_runtime_get_total_memory(KPURuntimeHandle runtime);

/**
 * @brief Get available device memory
 *
 * @param runtime Runtime handle
 * @return Free memory in bytes
 */
KPU_C_API KPUSize kpu_runtime_get_free_memory(KPURuntimeHandle runtime);

/**
 * @brief Get total cycles executed
 *
 * @param runtime Runtime handle
 * @return Total simulated cycles
 */
KPU_C_API KPUCycle kpu_runtime_get_total_cycles(KPURuntimeHandle runtime);

/**
 * @brief Get total kernels launched
 *
 * @param runtime Runtime handle
 * @return Number of kernel launches
 */
KPU_C_API uint64_t kpu_runtime_get_launch_count(KPURuntimeHandle runtime);

/**
 * @brief Print runtime statistics
 *
 * @param runtime Runtime handle
 */
KPU_C_API void kpu_runtime_print_stats(KPURuntimeHandle runtime);

#ifdef __cplusplus
}
#endif
