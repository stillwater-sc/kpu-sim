/**
 * @file kpu_c_executor.cpp
 * @brief Implementation of C API executor bindings
 */

#include <sw/kpu/kpu_c_executor.h>
#include <sw/runtime/executor.hpp>
#include <sw/kpu/kernel.hpp>

#include <cstring>
#include <exception>
#include <algorithm>

namespace {

// Cast helpers
inline sw::runtime::GraphExecutor* to_executor(KPUExecutorHandle h) {
    return reinterpret_cast<sw::runtime::GraphExecutor*>(h);
}

inline KPUExecutorHandle from_executor(sw::runtime::GraphExecutor* e) {
    return reinterpret_cast<KPUExecutorHandle>(e);
}

inline sw::runtime::KPURuntime* to_runtime(KPURuntimeHandle h) {
    return reinterpret_cast<sw::runtime::KPURuntime*>(h);
}

inline sw::kpu::Kernel* to_kernel(KPUKernelHandle h) {
    return reinterpret_cast<sw::kpu::Kernel*>(h);
}

// Enum conversions
inline sw::kpu::DataType to_cpp_dtype(KPUDataType dtype) {
    return static_cast<sw::kpu::DataType>(dtype);
}

inline sw::kpu::ActivationType to_cpp_activation(KPUActivationType activation) {
    return static_cast<sw::kpu::ActivationType>(activation);
}

inline KPUDataType from_cpp_dtype(sw::kpu::DataType dtype) {
    return static_cast<KPUDataType>(dtype);
}

// Exception to error code translation
KPUError translate_exception() {
    try {
        throw;
    } catch (const std::out_of_range&) {
        return KPU_ERROR_OUT_OF_BOUNDS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        return KPU_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return KPU_ERROR_UNKNOWN;
    }
}

} // anonymous namespace

extern "C" {

/* ============================================================================
 * Executor Lifecycle
 * ============================================================================ */

KPUExecutorHandle kpu_executor_create(KPURuntimeHandle runtime) {
    if (!runtime) return nullptr;

    try {
        auto* exec = new sw::runtime::GraphExecutor(to_runtime(runtime));
        return from_executor(exec);
    } catch (...) {
        return nullptr;
    }
}

void kpu_executor_destroy(KPUExecutorHandle executor) {
    if (executor) {
        delete to_executor(executor);
    }
}

/* ============================================================================
 * Kernel Setup
 * ============================================================================ */

KPUError kpu_executor_set_kernel(KPUExecutorHandle executor, KPUKernelHandle kernel) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!kernel) return KPU_ERROR_KERNEL_INVALID;

    try {
        to_executor(executor)->set_kernel(*to_kernel(kernel));
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_create_matmul(KPUExecutorHandle executor,
                                     KPUSize M, KPUSize N, KPUSize K,
                                     KPUDataType dtype) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;

    try {
        to_executor(executor)->create_matmul(M, N, K, to_cpp_dtype(dtype));
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_create_mlp(KPUExecutorHandle executor,
                                  KPUSize M, KPUSize N, KPUSize K,
                                  KPUActivationType activation,
                                  bool has_bias,
                                  KPUDataType dtype) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;

    try {
        to_executor(executor)->create_mlp(
            M, N, K,
            to_cpp_activation(activation),
            has_bias,
            to_cpp_dtype(dtype)
        );
        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

bool kpu_executor_has_kernel(KPUExecutorHandle executor) {
    if (!executor) return false;
    return to_executor(executor)->has_kernel();
}

/* ============================================================================
 * Input/Output Binding
 * ============================================================================ */

KPUError kpu_executor_set_input(KPUExecutorHandle executor,
                                 const char* name,
                                 const void* data,
                                 const KPUSize* shape,
                                 uint32_t ndims) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!name || !data) return KPU_ERROR_NULL_POINTER;
    if (!shape && ndims > 0) return KPU_ERROR_NULL_POINTER;

    try {
        std::vector<sw::kpu::Size> shape_vec(shape, shape + ndims);
        to_executor(executor)->set_input(name, data, shape_vec);
        return KPU_SUCCESS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_BINDING_NOT_FOUND;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_set_input_raw(KPUExecutorHandle executor,
                                     const char* name,
                                     const void* data,
                                     KPUSize size_bytes) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!name || !data) return KPU_ERROR_NULL_POINTER;

    try {
        to_executor(executor)->set_input(name, data, size_bytes);
        return KPU_SUCCESS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_BINDING_NOT_FOUND;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_get_output(KPUExecutorHandle executor,
                                  const char* name,
                                  void* data) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!name || !data) return KPU_ERROR_NULL_POINTER;

    try {
        to_executor(executor)->get_output(name, data);
        return KPU_SUCCESS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_BINDING_NOT_FOUND;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_get_output_sized(KPUExecutorHandle executor,
                                        const char* name,
                                        void* data,
                                        KPUSize size_bytes) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!name || !data) return KPU_ERROR_NULL_POINTER;

    try {
        to_executor(executor)->get_output(name, data, size_bytes);
        return KPU_SUCCESS;
    } catch (const std::invalid_argument&) {
        return KPU_ERROR_BINDING_NOT_FOUND;
    } catch (...) {
        return translate_exception();
    }
}

KPUError kpu_executor_get_binding_info(KPUExecutorHandle executor,
                                        const char* name,
                                        KPUTensorInfo* info) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;
    if (!name || !info) return KPU_ERROR_NULL_POINTER;

    try {
        const sw::runtime::TensorBinding* binding =
            to_executor(executor)->get_binding(name);

        if (!binding) {
            return KPU_ERROR_BINDING_NOT_FOUND;
        }

        // Fill info structure
        std::strncpy(info->name, binding->name.c_str(), sizeof(info->name) - 1);
        info->name[sizeof(info->name) - 1] = '\0';

        info->ndims = static_cast<uint32_t>(
            std::min(binding->shape.size(), static_cast<size_t>(8)));

        for (uint32_t i = 0; i < info->ndims; ++i) {
            info->shape[i] = binding->shape[i];
        }

        info->dtype = from_cpp_dtype(binding->dtype);
        info->device_address = binding->device_address;
        info->size_bytes = binding->size_bytes;

        return KPU_SUCCESS;
    } catch (...) {
        return translate_exception();
    }
}

/* ============================================================================
 * Execution
 * ============================================================================ */

KPUError kpu_executor_execute(KPUExecutorHandle executor,
                               KPUExecutionResult* result) {
    if (!executor) return KPU_ERROR_INVALID_HANDLE;

    try {
        sw::runtime::ExecutionResult cpp_result = to_executor(executor)->execute();

        if (result) {
            result->success = cpp_result.success;
            result->cycles = cpp_result.cycles;
            result->time_ms = cpp_result.time_ms;

            if (!cpp_result.error.empty()) {
                std::strncpy(result->error, cpp_result.error.c_str(),
                            sizeof(result->error) - 1);
                result->error[sizeof(result->error) - 1] = '\0';
            } else {
                result->error[0] = '\0';
            }
        }

        return cpp_result.success ? KPU_SUCCESS : KPU_ERROR_LAUNCH_FAILED;
    } catch (...) {
        return translate_exception();
    }
}

double kpu_executor_get_last_time_ms(KPUExecutorHandle executor) {
    if (!executor) return 0.0;

    try {
        return to_executor(executor)->get_last_execution_time_ms();
    } catch (...) {
        return 0.0;
    }
}

KPUCycle kpu_executor_get_last_cycles(KPUExecutorHandle executor) {
    if (!executor) return 0;

    try {
        return to_executor(executor)->get_last_execution_cycles();
    } catch (...) {
        return 0;
    }
}

/* ============================================================================
 * Cleanup
 * ============================================================================ */

void kpu_executor_release(KPUExecutorHandle executor) {
    if (!executor) return;

    try {
        to_executor(executor)->release();
    } catch (...) {
        // Silently ignore errors
    }
}

} // extern "C"
