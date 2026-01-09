/**
 * @file kpu_c_kernel.cpp
 * @brief Implementation of C API kernel bindings
 */

#include <sw/kpu/kpu_c_kernel.h>
#include <sw/kpu/kernel.hpp>

#include <cstring>
#include <string>

namespace {

// Cast helper
inline sw::kpu::Kernel* to_kernel(KPUKernelHandle h) {
    return reinterpret_cast<sw::kpu::Kernel*>(h);
}

inline KPUKernelHandle from_kernel(sw::kpu::Kernel* k) {
    return reinterpret_cast<KPUKernelHandle>(k);
}

// Convert C enum to C++ enum
inline sw::kpu::DataType to_cpp_dtype(KPUDataType dtype) {
    return static_cast<sw::kpu::DataType>(dtype);
}

inline sw::kpu::ActivationType to_cpp_activation(KPUActivationType activation) {
    return static_cast<sw::kpu::ActivationType>(activation);
}

// Convert C++ enum to C enum
inline KPUDataType from_cpp_dtype(sw::kpu::DataType dtype) {
    return static_cast<KPUDataType>(dtype);
}

inline KPUActivationType from_cpp_activation(sw::kpu::ActivationType activation) {
    return static_cast<KPUActivationType>(activation);
}

inline KPUKernelOpType from_cpp_op_type(sw::kpu::KernelOpType op_type) {
    return static_cast<KPUKernelOpType>(op_type);
}

} // anonymous namespace

extern "C" {

/* ============================================================================
 * Kernel Factory Methods
 * ============================================================================ */

KPUKernelHandle kpu_kernel_create_matmul(KPUSize M, KPUSize N, KPUSize K,
                                          KPUDataType dtype) {
    try {
        auto kernel = sw::kpu::Kernel::create_matmul(M, N, K, to_cpp_dtype(dtype));
        return from_kernel(new sw::kpu::Kernel(std::move(kernel)));
    } catch (...) {
        return nullptr;
    }
}

KPUKernelHandle kpu_kernel_create_mlp(KPUSize M, KPUSize N, KPUSize K,
                                       KPUActivationType activation,
                                       bool has_bias,
                                       KPUDataType dtype) {
    try {
        auto kernel = sw::kpu::Kernel::create_mlp(
            M, N, K,
            to_cpp_activation(activation),
            has_bias,
            to_cpp_dtype(dtype)
        );
        return from_kernel(new sw::kpu::Kernel(std::move(kernel)));
    } catch (...) {
        return nullptr;
    }
}

void kpu_kernel_destroy(KPUKernelHandle kernel) {
    if (kernel) {
        delete to_kernel(kernel);
    }
}

/* ============================================================================
 * Kernel Validity
 * ============================================================================ */

bool kpu_kernel_is_valid(KPUKernelHandle kernel) {
    if (!kernel) return false;
    return to_kernel(kernel)->is_valid();
}

bool kpu_kernel_validate(KPUKernelHandle kernel,
                          char* error_buffer,
                          size_t buffer_size) {
    if (!kernel) {
        if (error_buffer && buffer_size > 0) {
            std::strncpy(error_buffer, "Null kernel handle", buffer_size - 1);
            error_buffer[buffer_size - 1] = '\0';
        }
        return false;
    }

    std::string error;
    bool valid = to_kernel(kernel)->validate(error);

    if (error_buffer && buffer_size > 0) {
        std::strncpy(error_buffer, error.c_str(), buffer_size - 1);
        error_buffer[buffer_size - 1] = '\0';
    }

    return valid;
}

/* ============================================================================
 * Kernel Metadata Accessors
 * ============================================================================ */

KPUKernelOpType kpu_kernel_get_op_type(KPUKernelHandle kernel) {
    if (!kernel) return KPU_OP_CUSTOM;
    return from_cpp_op_type(to_kernel(kernel)->op_type());
}

KPUDataType kpu_kernel_get_dtype(KPUKernelHandle kernel) {
    if (!kernel) return KPU_DTYPE_FLOAT32;
    return from_cpp_dtype(to_kernel(kernel)->dtype());
}

size_t kpu_kernel_get_name(KPUKernelHandle kernel,
                            char* buffer,
                            size_t buffer_size) {
    if (!kernel || !buffer || buffer_size == 0) return 0;

    const std::string& name = to_kernel(kernel)->name();
    size_t len = name.length();

    if (len >= buffer_size) {
        std::strncpy(buffer, name.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    } else {
        std::strcpy(buffer, name.c_str());
    }

    return len;
}

/* ============================================================================
 * Matrix Dimension Accessors
 * ============================================================================ */

KPUSize kpu_kernel_get_M(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->M();
}

KPUSize kpu_kernel_get_N(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->N();
}

KPUSize kpu_kernel_get_K(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->K();
}

KPUSize kpu_kernel_get_Ti(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->Ti();
}

KPUSize kpu_kernel_get_Tj(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->Tj();
}

KPUSize kpu_kernel_get_Tk(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->Tk();
}

/* ============================================================================
 * MLP-Specific Accessors
 * ============================================================================ */

KPUActivationType kpu_kernel_get_activation(KPUKernelHandle kernel) {
    if (!kernel) return KPU_ACTIVATION_NONE;
    return from_cpp_activation(to_kernel(kernel)->activation());
}

bool kpu_kernel_has_bias(KPUKernelHandle kernel) {
    if (!kernel) return false;
    return to_kernel(kernel)->has_bias();
}

/* ============================================================================
 * Statistics and Performance
 * ============================================================================ */

KPUSize kpu_kernel_get_total_flops(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->total_flops();
}

double kpu_kernel_get_arithmetic_intensity(KPUKernelHandle kernel) {
    if (!kernel) return 0.0;
    return to_kernel(kernel)->arithmetic_intensity();
}

size_t kpu_kernel_get_instruction_count(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->instruction_count();
}

KPUSize kpu_kernel_get_total_input_bytes(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->total_input_bytes();
}

KPUSize kpu_kernel_get_total_output_bytes(KPUKernelHandle kernel) {
    if (!kernel) return 0;
    return to_kernel(kernel)->total_output_bytes();
}

size_t kpu_kernel_summary(KPUKernelHandle kernel,
                           char* buffer,
                           size_t buffer_size) {
    if (!kernel || !buffer || buffer_size == 0) return 0;

    std::string summary = to_kernel(kernel)->summary();
    size_t len = summary.length();

    if (len >= buffer_size) {
        std::strncpy(buffer, summary.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    } else {
        std::strcpy(buffer, summary.c_str());
    }

    return len;
}

} // extern "C"
