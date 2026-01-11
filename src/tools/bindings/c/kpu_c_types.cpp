/**
 * @file kpu_c_types.cpp
 * @brief Implementation of utility functions for KPU C API types
 */

#include <sw/kpu/kpu_c_types.h>

extern "C" {

const char* kpu_error_string(KPUError error) {
    switch (error) {
        case KPU_SUCCESS:
            return "Success";
        case KPU_ERROR_INVALID_HANDLE:
            return "Invalid handle";
        case KPU_ERROR_OUT_OF_BOUNDS:
            return "Memory access out of bounds";
        case KPU_ERROR_INVALID_DIMENSIONS:
            return "Invalid matrix/tensor dimensions";
        case KPU_ERROR_NULL_POINTER:
            return "Null pointer";
        case KPU_ERROR_UNKNOWN:
            return "Unknown error";
        case KPU_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case KPU_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case KPU_ERROR_KERNEL_INVALID:
            return "Kernel is invalid or not compiled";
        case KPU_ERROR_LAUNCH_FAILED:
            return "Kernel launch failed";
        case KPU_ERROR_STREAM_NOT_FOUND:
            return "Stream not found";
        case KPU_ERROR_EVENT_NOT_FOUND:
            return "Event not found";
        case KPU_ERROR_BINDING_NOT_FOUND:
            return "Tensor binding not found";
        case KPU_ERROR_COMPILATION_FAILED:
            return "Kernel compilation failed";
        case KPU_ERROR_SHAPE_MISMATCH:
            return "Tensor shape mismatch";
        default:
            return "Invalid error code";
    }
}

const char* kpu_dtype_name(KPUDataType dtype) {
    switch (dtype) {
        case KPU_DTYPE_FLOAT32:
            return "float32";
        case KPU_DTYPE_FLOAT16:
            return "float16";
        case KPU_DTYPE_BFLOAT16:
            return "bfloat16";
        case KPU_DTYPE_INT32:
            return "int32";
        case KPU_DTYPE_INT8:
            return "int8";
        case KPU_DTYPE_UINT8:
            return "uint8";
        case KPU_DTYPE_INT4:
            return "int4";
        default:
            return "unknown";
    }
}

KPUSize kpu_dtype_size(KPUDataType dtype) {
    switch (dtype) {
        case KPU_DTYPE_FLOAT32:
            return 4;
        case KPU_DTYPE_FLOAT16:
            return 2;
        case KPU_DTYPE_BFLOAT16:
            return 2;
        case KPU_DTYPE_INT32:
            return 4;
        case KPU_DTYPE_INT8:
            return 1;
        case KPU_DTYPE_UINT8:
            return 1;
        case KPU_DTYPE_INT4:
            return 1;  // Minimum addressable unit (2 elements packed)
        default:
            return 0;
    }
}

const char* kpu_activation_name(KPUActivationType activation) {
    switch (activation) {
        case KPU_ACTIVATION_NONE:
            return "none";
        case KPU_ACTIVATION_RELU:
            return "relu";
        case KPU_ACTIVATION_GELU:
            return "gelu";
        case KPU_ACTIVATION_SIGMOID:
            return "sigmoid";
        case KPU_ACTIVATION_TANH:
            return "tanh";
        case KPU_ACTIVATION_SILU:
            return "silu";
        case KPU_ACTIVATION_SOFTPLUS:
            return "softplus";
        case KPU_ACTIVATION_LEAKY_RELU:
            return "leaky_relu";
        default:
            return "unknown";
    }
}

const char* kpu_op_type_name(KPUKernelOpType op_type) {
    switch (op_type) {
        case KPU_OP_MATMUL:
            return "matmul";
        case KPU_OP_BATCH_MATMUL:
            return "batch_matmul";
        case KPU_OP_CONV2D:
            return "conv2d";
        case KPU_OP_ELEMENTWISE:
            return "elementwise";
        case KPU_OP_MLP:
            return "mlp";
        case KPU_OP_CUSTOM:
            return "custom";
        default:
            return "unknown";
    }
}

void kpu_simulator_config_default(KPUSimulatorConfig* config) {
    if (!config) return;

    config->memory_bank_count = 4;
    config->memory_bank_capacity_mb = 256;
    config->memory_bandwidth_gbps = 128;
    config->l3_tile_count = 8;
    config->l3_tile_capacity_kb = 512;
    config->l2_bank_count = 16;
    config->l2_bank_capacity_kb = 64;
    config->l1_buffer_count = 4;
    config->l1_buffer_capacity_kb = 64;
    config->page_buffer_count = 2;
    config->page_buffer_capacity_kb = 64;
    config->compute_tile_count = 4;
    config->dma_engine_count = 4;
    config->block_mover_count = 8;
    config->streamer_count = 16;
    config->processor_array_rows = 16;
    config->processor_array_cols = 16;
    config->use_systolic_array_mode = true;
}

void kpu_runtime_config_default(KPURuntimeConfig* config) {
    if (!config) return;

    config->clock_ghz = 1.0;
    config->verbose = false;
}

} // extern "C"
