/**
 * @file kpu_c_types.h
 * @brief Common types, enums, and structures for the KPU C API
 *
 * This header defines all the shared types used across the KPU C API:
 * - Type aliases (KPUAddress, KPUSize, KPUCycle)
 * - Opaque handle types
 * - Error codes
 * - Data type and activation enums
 * - Configuration and result structures
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * Symbol Visibility for C API
 * ============================================================================ */

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef BUILDING_KPU_C_API
        #define KPU_C_API __declspec(dllexport)
    #else
        #define KPU_C_API __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define KPU_C_API __attribute__((visibility("default")))
#else
    #define KPU_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Aliases (matching C++ types in sw::kpu namespace)
 * ============================================================================ */

/** Memory address in unified address space */
typedef uint64_t KPUAddress;

/** Size type for memory sizes and array dimensions */
typedef size_t KPUSize;

/** Simulation cycle counter */
typedef uint64_t KPUCycle;

/* ============================================================================
 * Opaque Handle Types
 *
 * These are void pointers to allow the C++ implementation to cast to the
 * appropriate internal types without exposing C++ types to C code.
 * ============================================================================ */

/** Handle to KPU simulator instance */
typedef void* KPUHandle;

/** Handle to KPU runtime instance */
typedef void* KPURuntimeHandle;

/** Handle to compiled kernel */
typedef void* KPUKernelHandle;

/** Handle to graph executor */
typedef void* KPUExecutorHandle;

/** Handle to execution stream */
typedef void* KPUStreamHandle;

/** Handle to synchronization event */
typedef void* KPUEventHandle;

/* ============================================================================
 * Error Codes
 * ============================================================================ */

/**
 * @brief Error codes returned by KPU C API functions
 */
typedef enum {
    KPU_SUCCESS = 0,                    /**< Operation completed successfully */
    KPU_ERROR_INVALID_HANDLE = -1,      /**< Invalid or null handle */
    KPU_ERROR_OUT_OF_BOUNDS = -2,       /**< Memory access out of bounds */
    KPU_ERROR_INVALID_DIMENSIONS = -3,  /**< Invalid matrix/tensor dimensions */
    KPU_ERROR_NULL_POINTER = -4,        /**< Null pointer argument */
    KPU_ERROR_UNKNOWN = -5,             /**< Unknown/unspecified error */
    KPU_ERROR_OUT_OF_MEMORY = -6,       /**< Memory allocation failed */
    KPU_ERROR_INVALID_ARGUMENT = -7,    /**< Invalid argument value */
    KPU_ERROR_KERNEL_INVALID = -8,      /**< Kernel is invalid or not compiled */
    KPU_ERROR_LAUNCH_FAILED = -9,       /**< Kernel launch failed */
    KPU_ERROR_STREAM_NOT_FOUND = -10,   /**< Stream does not exist */
    KPU_ERROR_EVENT_NOT_FOUND = -11,    /**< Event does not exist */
    KPU_ERROR_BINDING_NOT_FOUND = -12,  /**< Tensor binding not found */
    KPU_ERROR_COMPILATION_FAILED = -13, /**< Kernel compilation failed */
    KPU_ERROR_SHAPE_MISMATCH = -14      /**< Tensor shape mismatch */
} KPUError;

/* ============================================================================
 * Data Type Enum
 * ============================================================================ */

/**
 * @brief Numeric data types supported by KPU compute fabric
 *
 * Values match the C++ enum class DataType for ABI compatibility.
 */
typedef enum {
    KPU_DTYPE_FLOAT32 = 0,   /**< 4 bytes, IEEE 754 single precision */
    KPU_DTYPE_FLOAT16 = 1,   /**< 2 bytes, IEEE 754 half precision */
    KPU_DTYPE_BFLOAT16 = 2,  /**< 2 bytes, brain float (bf16) */
    KPU_DTYPE_INT32 = 3,     /**< 4 bytes, signed 32-bit integer */
    KPU_DTYPE_INT8 = 4,      /**< 1 byte, signed 8-bit integer */
    KPU_DTYPE_UINT8 = 5,     /**< 1 byte, unsigned 8-bit integer */
    KPU_DTYPE_INT4 = 6       /**< 0.5 bytes (packed), signed 4-bit integer */
} KPUDataType;

/* ============================================================================
 * Activation Type Enum
 * ============================================================================ */

/**
 * @brief Activation function types for neural network operations
 *
 * Values match the C++ enum class ActivationType for ABI compatibility.
 */
typedef enum {
    KPU_ACTIVATION_NONE = 0,        /**< Pass-through (no activation) */
    KPU_ACTIVATION_RELU = 1,        /**< max(0, x) */
    KPU_ACTIVATION_GELU = 2,        /**< x * 0.5 * (1 + erf(x/sqrt(2))) */
    KPU_ACTIVATION_SIGMOID = 3,     /**< 1 / (1 + exp(-x)) */
    KPU_ACTIVATION_TANH = 4,        /**< (exp(x) - exp(-x)) / (exp(x) + exp(-x)) */
    KPU_ACTIVATION_SILU = 5,        /**< x * sigmoid(x), aka Swish */
    KPU_ACTIVATION_SOFTPLUS = 6,    /**< ln(1 + exp(x)) */
    KPU_ACTIVATION_LEAKY_RELU = 7   /**< max(alpha*x, x) */
} KPUActivationType;

/* ============================================================================
 * Kernel Operation Type Enum
 * ============================================================================ */

/**
 * @brief Types of kernel operations
 *
 * Values match the C++ enum class KernelOpType for ABI compatibility.
 */
typedef enum {
    KPU_OP_MATMUL = 0,       /**< Matrix multiplication C = A x B */
    KPU_OP_BATCH_MATMUL = 1, /**< Batched matrix multiplication */
    KPU_OP_CONV2D = 2,       /**< 2D convolution */
    KPU_OP_ELEMENTWISE = 3,  /**< Elementwise operations */
    KPU_OP_MLP = 4,          /**< Fused matmul + bias + activation */
    KPU_OP_CUSTOM = 255      /**< Custom/user-defined kernel */
} KPUKernelOpType;

/* ============================================================================
 * Memory Copy Direction Enum
 * ============================================================================ */

/**
 * @brief Direction for memory copy operations
 */
typedef enum {
    KPU_MEMCPY_HOST_TO_DEVICE = 0,   /**< Copy from host to device memory */
    KPU_MEMCPY_DEVICE_TO_HOST = 1,   /**< Copy from device to host memory */
    KPU_MEMCPY_DEVICE_TO_DEVICE = 2  /**< Copy within device memory */
} KPUMemcpyKind;

/* ============================================================================
 * Configuration Structures
 * ============================================================================ */

/**
 * @brief Configuration for creating a KPU simulator
 */
typedef struct {
    uint32_t memory_bank_count;       /**< Number of external memory banks */
    uint32_t memory_bank_capacity_mb; /**< Capacity per bank in MB */
    uint32_t memory_bandwidth_gbps;   /**< Memory bandwidth in GB/s */
    uint32_t l3_tile_count;           /**< Number of L3 cache tiles */
    uint32_t l3_tile_capacity_kb;     /**< Capacity per L3 tile in KB */
    uint32_t l2_bank_count;           /**< Number of L2 banks */
    uint32_t l2_bank_capacity_kb;     /**< Capacity per L2 bank in KB */
    uint32_t l1_buffer_count;         /**< Number of L1 buffers */
    uint32_t l1_buffer_capacity_kb;   /**< Capacity per L1 buffer in KB */
    uint32_t page_buffer_count;       /**< Number of page buffers */
    uint32_t page_buffer_capacity_kb; /**< Capacity per page buffer in KB */
    uint32_t compute_tile_count;      /**< Number of compute tiles */
    uint32_t dma_engine_count;        /**< Number of DMA engines */
    uint32_t block_mover_count;       /**< Number of block movers */
    uint32_t streamer_count;          /**< Number of streamers */
    uint32_t processor_array_rows;    /**< Rows in processor array */
    uint32_t processor_array_cols;    /**< Columns in processor array */
    bool use_systolic_array_mode;     /**< Enable systolic array execution */
} KPUSimulatorConfig;

/**
 * @brief Configuration for creating a KPU runtime
 */
typedef struct {
    double clock_ghz;  /**< Clock frequency in GHz (for timing calculations) */
    bool verbose;      /**< Enable verbose logging */
} KPURuntimeConfig;

/* ============================================================================
 * Result Structures
 * ============================================================================ */

/**
 * @brief Result from synchronous kernel launch
 */
typedef struct {
    bool success;        /**< Whether the launch succeeded */
    KPUCycle cycles;     /**< Number of cycles taken */
    char error[256];     /**< Error message if failed */
} KPULaunchResult;

/**
 * @brief Result from GraphExecutor execution
 */
typedef struct {
    bool success;        /**< Whether execution succeeded */
    KPUCycle cycles;     /**< Number of cycles taken */
    double time_ms;      /**< Execution time in milliseconds */
    char error[256];     /**< Error message if failed */
} KPUExecutionResult;

/**
 * @brief Information about a tensor binding in GraphExecutor
 */
typedef struct {
    char name[64];              /**< Tensor name (e.g., "A", "B", "C") */
    KPUSize shape[8];           /**< Shape dimensions */
    uint32_t ndims;             /**< Number of dimensions */
    KPUDataType dtype;          /**< Data type */
    KPUAddress device_address;  /**< Address in device memory */
    KPUSize size_bytes;         /**< Total size in bytes */
} KPUTensorInfo;

/**
 * @brief Matrix dimensions (legacy compatibility)
 */
typedef struct {
    uint32_t rows;
    uint32_t cols;
} KPUMatrixDim;

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Get human-readable error message
 * @param error The error code
 * @return Static string describing the error
 */
KPU_C_API const char* kpu_error_string(KPUError error);

/**
 * @brief Get name of a data type
 * @param dtype The data type
 * @return Static string with type name
 */
KPU_C_API const char* kpu_dtype_name(KPUDataType dtype);

/**
 * @brief Get size of a data type in bytes
 * @param dtype The data type
 * @return Size in bytes
 */
KPU_C_API KPUSize kpu_dtype_size(KPUDataType dtype);

/**
 * @brief Get name of an activation function
 * @param activation The activation type
 * @return Static string with activation name
 */
KPU_C_API const char* kpu_activation_name(KPUActivationType activation);

/**
 * @brief Get name of a kernel operation type
 * @param op_type The operation type
 * @return Static string with operation name
 */
KPU_C_API const char* kpu_op_type_name(KPUKernelOpType op_type);

/**
 * @brief Initialize simulator config with default values
 * @param config Pointer to config structure to initialize
 */
KPU_C_API void kpu_simulator_config_default(KPUSimulatorConfig* config);

/**
 * @brief Initialize runtime config with default values
 * @param config Pointer to config structure to initialize
 */
KPU_C_API void kpu_runtime_config_default(KPURuntimeConfig* config);

#ifdef __cplusplus
}
#endif
