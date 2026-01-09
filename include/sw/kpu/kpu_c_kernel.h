/**
 * @file kpu_c_kernel.h
 * @brief C API for KPU kernel creation and metadata access
 *
 * This header provides functions to:
 * - Create kernels using factory methods (matmul, MLP)
 * - Query kernel metadata (dimensions, data types, statistics)
 * - Destroy kernels when no longer needed
 */
#pragma once

#include "kpu_c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Kernel Factory Methods
 * ============================================================================ */

/**
 * @brief Create a matrix multiplication kernel
 *
 * Creates a kernel for computing C = A * B where:
 * - A is M x K
 * - B is K x N
 * - C is M x N
 *
 * Uses automatic tile optimization and output-stationary dataflow.
 *
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param dtype Data type (use KPU_DTYPE_FLOAT32 for default)
 * @return Kernel handle, or NULL on failure
 */
KPU_C_API KPUKernelHandle kpu_kernel_create_matmul(KPUSize M, KPUSize N, KPUSize K,
                                          KPUDataType dtype);

/**
 * @brief Create a fused MLP kernel (matmul + bias + activation)
 *
 * Creates a kernel for computing C = activation(A * B + bias) where:
 * - A is M x K (input)
 * - B is K x N (weights)
 * - bias is N (optional)
 * - C is M x N (output)
 *
 * The Vector Engine applies bias and activation inline during
 * the output drain phase, avoiding extra memory passes.
 *
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 * @param activation Activation function type
 * @param has_bias Whether to apply bias addition
 * @param dtype Data type
 * @return Kernel handle, or NULL on failure
 */
KPU_C_API KPUKernelHandle kpu_kernel_create_mlp(KPUSize M, KPUSize N, KPUSize K,
                                       KPUActivationType activation,
                                       bool has_bias,
                                       KPUDataType dtype);

/**
 * @brief Destroy a kernel and free its resources
 *
 * @param kernel Kernel handle (safe to call with NULL)
 */
KPU_C_API void kpu_kernel_destroy(KPUKernelHandle kernel);

/* ============================================================================
 * Kernel Validity
 * ============================================================================ */

/**
 * @brief Check if kernel is valid (has instructions)
 *
 * @param kernel Kernel handle
 * @return true if kernel is valid, false otherwise
 */
KPU_C_API bool kpu_kernel_is_valid(KPUKernelHandle kernel);

/**
 * @brief Validate kernel for execution
 *
 * Performs comprehensive validation and returns error details.
 *
 * @param kernel Kernel handle
 * @param error_buffer Buffer to write error message (can be NULL)
 * @param buffer_size Size of error buffer
 * @return true if valid, false if invalid
 */
KPU_C_API bool kpu_kernel_validate(KPUKernelHandle kernel,
                          char* error_buffer,
                          size_t buffer_size);

/* ============================================================================
 * Kernel Metadata Accessors
 * ============================================================================ */

/**
 * @brief Get kernel operation type
 *
 * @param kernel Kernel handle
 * @return Operation type, or KPU_OP_CUSTOM if invalid
 */
KPU_C_API KPUKernelOpType kpu_kernel_get_op_type(KPUKernelHandle kernel);

/**
 * @brief Get kernel data type
 *
 * @param kernel Kernel handle
 * @return Data type, or KPU_DTYPE_FLOAT32 if invalid
 */
KPU_C_API KPUDataType kpu_kernel_get_dtype(KPUKernelHandle kernel);

/**
 * @brief Get kernel name
 *
 * @param kernel Kernel handle
 * @param buffer Buffer to write name
 * @param buffer_size Size of buffer
 * @return Length of name (may be truncated if buffer too small)
 */
KPU_C_API size_t kpu_kernel_get_name(KPUKernelHandle kernel,
                            char* buffer,
                            size_t buffer_size);

/* ============================================================================
 * Matrix Dimension Accessors
 * ============================================================================ */

/**
 * @brief Get M dimension (rows of A and C)
 * @param kernel Kernel handle
 * @return M dimension, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_M(KPUKernelHandle kernel);

/**
 * @brief Get N dimension (columns of B and C)
 * @param kernel Kernel handle
 * @return N dimension, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_N(KPUKernelHandle kernel);

/**
 * @brief Get K dimension (columns of A, rows of B)
 * @param kernel Kernel handle
 * @return K dimension, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_K(KPUKernelHandle kernel);

/**
 * @brief Get Ti tile size (M-dimension)
 * @param kernel Kernel handle
 * @return Ti tile size, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_Ti(KPUKernelHandle kernel);

/**
 * @brief Get Tj tile size (N-dimension)
 * @param kernel Kernel handle
 * @return Tj tile size, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_Tj(KPUKernelHandle kernel);

/**
 * @brief Get Tk tile size (K-dimension)
 * @param kernel Kernel handle
 * @return Tk tile size, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_Tk(KPUKernelHandle kernel);

/* ============================================================================
 * MLP-Specific Accessors
 * ============================================================================ */

/**
 * @brief Get activation function type
 *
 * @param kernel Kernel handle
 * @return Activation type, or KPU_ACTIVATION_NONE if not MLP
 */
KPU_C_API KPUActivationType kpu_kernel_get_activation(KPUKernelHandle kernel);

/**
 * @brief Check if kernel uses bias
 *
 * @param kernel Kernel handle
 * @return true if kernel uses bias, false otherwise
 */
KPU_C_API bool kpu_kernel_has_bias(KPUKernelHandle kernel);

/* ============================================================================
 * Statistics and Performance
 * ============================================================================ */

/**
 * @brief Get total FLOPs for this kernel
 *
 * @param kernel Kernel handle
 * @return Total floating point operations, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_total_flops(KPUKernelHandle kernel);

/**
 * @brief Get arithmetic intensity (FLOPs per byte from DRAM)
 *
 * @param kernel Kernel handle
 * @return Arithmetic intensity, or 0.0 if invalid
 */
KPU_C_API double kpu_kernel_get_arithmetic_intensity(KPUKernelHandle kernel);

/**
 * @brief Get instruction count
 *
 * @param kernel Kernel handle
 * @return Number of instructions, or 0 if invalid
 */
KPU_C_API size_t kpu_kernel_get_instruction_count(KPUKernelHandle kernel);

/**
 * @brief Get total input size in bytes
 *
 * @param kernel Kernel handle
 * @return Total input bytes, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_total_input_bytes(KPUKernelHandle kernel);

/**
 * @brief Get total output size in bytes
 *
 * @param kernel Kernel handle
 * @return Total output bytes, or 0 if invalid
 */
KPU_C_API KPUSize kpu_kernel_get_total_output_bytes(KPUKernelHandle kernel);

/**
 * @brief Get human-readable summary string
 *
 * @param kernel Kernel handle
 * @param buffer Buffer to write summary
 * @param buffer_size Size of buffer
 * @return Length of summary (may be truncated if buffer too small)
 */
KPU_C_API size_t kpu_kernel_summary(KPUKernelHandle kernel,
                           char* buffer,
                           size_t buffer_size);

#ifdef __cplusplus
}
#endif
